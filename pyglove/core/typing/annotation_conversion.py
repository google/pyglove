# Copyright 2023 The PyGlove Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversion from annotations to PyGlove value specs."""

import builtins
import collections
import inspect
import types
import typing

from pyglove.core import coding
from pyglove.core import utils
from pyglove.core.typing import annotated
from pyglove.core.typing import class_schema
from pyglove.core.typing import inspect as pg_inspect
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


_NoneType = type(None)

# Annotated is suppored after 3.9
_Annotated = getattr(typing, 'Annotated', None)  # pylint: disable=invalid-name

# UnionType is supported after 3.10.
_UnionType = getattr(types, 'UnionType', None)  # pylint: disable=invalid-name


def annotation_from_str(
    annotation_str: str,
    parent_module: typing.Optional[types.ModuleType] = None,
    ) -> typing.Any:
  """Parses annotations from str.

  BNF for PyType annotations:

  ```
  <maybe_union>      ::= <type> | <type> "|" <maybe_union>
  <type>             ::= <literal_type> | <non_literal_type>

  <literal_type>     ::= "Literal"<literal_params>
  <literal_params>   ::= "["<python_values>"]" (parsed by `pg.coding.evaluate`)

  <non_literal_type> ::= <type_id> | <type_id>"["<type_arg>"]"
  <type_arg>       ::= <maybe_type_list> | <maybe_type_list>","<maybe_type_list>
  <maybe_type_list>  ::= "["<type_arg>"]" | <maybe_union>
  <type_id>          ::= 'aAz_.1-9'
  ```

  Args:
    annotation_str: String form of type annotations. E.g. "list[str]"
    parent_module: The module where the annotation was defined.

  Returns:
    Object form of the annotation.

  Raises:
    SyntaxError: If the annotation string is invalid.
  """
  s = annotation_str
  context = dict(pos=0)

  def _eof() -> bool:
    return context['pos'] == len(s)

  def _pos() -> int:
    return context['pos']

  def _next(n: int = 1, offset: int = 0) -> str:
    if _eof():
      return '<EOF>'
    return s[_pos() + offset:_pos() + offset + n]

  def _advance(n: int) -> None:
    context['pos'] += n

  def _error_illustration() -> str:
    return f'{s}\n{" " * _pos()}' + '^'

  def _match(ch) -> bool:
    if _next(len(ch)) == ch:
      _advance(len(ch))
      return True
    return False

  def _skip_whitespaces() -> None:
    while _next() in ' \t':
      _advance(1)

  def _maybe_union():
    t = _type()
    while not _eof():
      _skip_whitespaces()
      if _match('|'):
        t = t | _type()
      else:
        break
    return t

  def _type():
    type_id = _type_id()
    t = _resolve(type_id)
    if t is typing.Literal:
      return t[_literal_params()]
    elif _match('['):
      arg = _type_arg()
      if not _match(']'):
        raise SyntaxError(
            f'Expected "]" at position {_pos()}.\n\n' + _error_illustration()
        )
      return t[arg]
    return t

  def _literal_params():
    if not _match('['):
      raise SyntaxError(
          f'Expected "[" at position {_pos()}.\n\n' + _error_illustration()
      )
    arg_start = _pos()
    in_str = False
    escape_mode = False
    num_open_bracket = 1

    while num_open_bracket > 0:
      ch = _next()
      if _eof():
        raise SyntaxError(
            f'Unexpected end of annotation at position {_pos()}.\n\n'
            + _error_illustration()
        )
      if ch == '\\':
        escape_mode = not escape_mode
      else:
        escape_mode = False

      if ch == "'" and not escape_mode:
        in_str = not in_str
      elif not in_str:
        if ch == '[':
          num_open_bracket += 1
        elif ch == ']':
          num_open_bracket -= 1
      _advance(1)

    arg_str = s[arg_start:_pos() - 1]
    return coding.evaluate(
        '(' + arg_str + ')', permission=coding.CodePermission.BASIC
    )

  def _type_arg():
    t_args = []
    t_args.append(_maybe_type_list())
    while _match(','):
      t_args.append(_maybe_type_list())
    return tuple(t_args) if len(t_args) > 1 else t_args[0]

  def _maybe_type_list():
    if _match('['):
      ret = _type_arg()
      if not _match(']'):
        raise SyntaxError(
            f'Expected "]" at position {_pos()}.\n\n' + _error_illustration()
        )
      return list(ret) if isinstance(ret, tuple) else [ret]
    return _maybe_union()

  def _type_id() -> str:
    _skip_whitespaces()
    if _match('...'):
      return '...'
    start = _pos()
    while not _eof():
      c = _next()
      if c.isalnum() or c in '_.':
        _advance(1)
      else:
        break
    t_id = s[start:_pos()]
    if not all(x.isidentifier() for x in t_id.split('.')):
      raise SyntaxError(
          f'Expected type identifier, got {t_id!r} at position {start}.\n\n'
          + _error_illustration()
      )
    return t_id

  def _resolve(type_id: str):

    def _as_forward_ref() -> typing.ForwardRef:
      return typing.ForwardRef(type_id, False, parent_module)  # pytype: disable=not-callable

    def _resolve_name(name: str, parent_obj: typing.Any):
      if name == 'None':
        return None, True
      if parent_obj is not None and hasattr(parent_obj, name):
        return getattr(parent_obj, name), False
      if hasattr(builtins, name):
        return getattr(builtins, name), True
      if type_id == '...':
        return ..., True
      return utils.MISSING_VALUE, False

    names = type_id.split('.')
    if len(names) == 1:
      reference, is_builtin = _resolve_name(names[0], parent_module)
      if is_builtin:
        return reference
      if not is_builtin and (
          # When reference is not found, we should treat it as a forward
          # reference.
          reference == utils.MISSING_VALUE
          # When module is being reloaded, we should treat all non-builtin
          # references as forward references.
          or getattr(parent_module, '__reloading__', False)
      ):
        return _as_forward_ref()
      return reference

    root_obj, _ = _resolve_name(names[0], parent_module)
    # When root object is not found, we should treat it as a forward reference.
    if root_obj == utils.MISSING_VALUE:
      return _as_forward_ref()

    parent_obj = root_obj
    # When root object is a module, we should treat reference to its children
    # as non-forward references.
    if inspect.ismodule(root_obj):
      for name in names[1:]:
        parent_obj, _ = _resolve_name(name, parent_obj)
        if parent_obj == utils.MISSING_VALUE:
          raise TypeError(f'{type_id!r} does not exist.')
      return parent_obj
    # When root object is non-module variable of current module, and when the
    # module is being reloaded, we should treat reference to its children as
    # forward references.
    elif getattr(parent_module, '__reloading__', False):
      return _as_forward_ref()
    # When root object is non-module variable of current module, we should treat
    # unresolved reference to its children as forward references.
    else:
      for name in names[1:]:
        parent_obj, _ = _resolve_name(name, parent_obj)
        if parent_obj == utils.MISSING_VALUE:
          return _as_forward_ref()
      return parent_obj

  root = _maybe_union()
  if _pos() != len(s):
    raise SyntaxError(
        'Unexpected end of annotation.\n\n' + _error_illustration()
    )
  return root


def _field_from_annotation(
    key: typing.Union[str, class_schema.KeySpec],
    annotation: typing.Any,
    description: typing.Optional[str] = None,
    metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
    auto_typing: bool = True,
    parent_module: typing.Optional[types.ModuleType] = None
) -> class_schema.Field:
  """Creates a field from Python annotation."""
  if isinstance(annotation, annotated.Annotated):
    field_spec = (
        key, annotation.value_spec, annotation.docstring, annotation.metadata)
  elif _Annotated and typing.get_origin(annotation) is _Annotated:
    type_args = typing.get_args(annotation)
    assert len(type_args) > 1, (annotation, type_args)
    field_spec = tuple([key] + list(type_args))
  else:
    field_spec = (key, annotation, description, metadata or {})
  return class_schema.create_field(
      field_spec,
      auto_typing=auto_typing,
      accept_value_as_annotation=False,
      parent_module=parent_module
  )


def _value_spec_from_default_value(
    value: typing.Any,
    set_default: bool = True) -> typing.Optional[class_schema.ValueSpec]:
  """Creates a value spec from a default value."""
  if isinstance(value, (bool, int, float, str, dict)):
    value_spec = _value_spec_from_type_annotation(type(value), False)
  elif isinstance(value, list):
    value_spec = vs.List(
        _value_spec_from_default_value(value[0], False) if value else vs.Any())
  elif isinstance(value, tuple):
    value_spec = vs.Tuple(
        [_value_spec_from_default_value(elem, False) for elem in value])
  elif inspect.isfunction(value) or isinstance(value, utils.Functor):
    value_spec = vs.Callable()
  elif not isinstance(value, type):
    value_spec = vs.Object(type(value))
  else:
    value_spec = None

  if value_spec and set_default:
    value_spec.set_default(value)
  return value_spec


def _value_spec_from_type_annotation(
    annotation: typing.Any,
    accept_value_as_annotation: bool,
    parent_module: typing.Optional[types.ModuleType] = None
) -> class_schema.ValueSpec:
  """Creates a value spec from type annotation."""
  if isinstance(annotation, str) and not accept_value_as_annotation:
    annotation = annotation_from_str(annotation, parent_module)

  if annotation is None:
    return vs.Object(type(None))
  elif annotation is bool:
    return vs.Bool()
  elif annotation is int:
    return vs.Int()
  elif annotation is float:
    return vs.Float()
  elif annotation is str:
    return vs.Str()
  elif annotation is typing.Any or annotation is vs.Any:
    return vs.Any().annotate(annotation)

  origin = typing.get_origin(annotation) or annotation
  args = list(typing.get_args(annotation))

  def _sub_value_spec_from_annotation(
      annotation: typing.Any) -> class_schema.ValueSpec:
    return _value_spec_from_type_annotation(
        annotation, accept_value_as_annotation, parent_module)

  # Handling list.
  if origin in (list, typing.List):
    return vs.List(
        _sub_value_spec_from_annotation(args[0])) if args else vs.List(vs.Any())
  # Handling tuple.
  elif origin in (tuple, typing.Tuple):
    if not args:
      return vs.Tuple(vs.Any())
    else:
      if args[-1] is ...:
        if len(args) != 2:
          raise TypeError(
              f'Tuple with ellipsis should have exact 2 type arguments. '
              f'Encountered: {annotation}.')
        return vs.Tuple(_sub_value_spec_from_annotation(args[0]))
      return vs.Tuple([_sub_value_spec_from_annotation(arg) for arg in args])
  # Handling sequence.
  elif origin in (collections.abc.Sequence,):
    elem = _sub_value_spec_from_annotation(args[0]) if args else vs.Any()
    return vs.Union([vs.List(elem), vs.Tuple(elem)])
  # Handling literals.
  elif origin is typing.Literal:
    return vs.Enum(utils.MISSING_VALUE, args)
  # Handling dict.
  elif origin in (dict, typing.Dict, collections.abc.Mapping):
    if not args:
      return vs.Dict()
    assert len(args) == 2, (annotation, args)
    if args[0] not in (str, typing.Text):
      raise TypeError(
          'Dict type field with non-string key is not supported.')
    elem_value_spec = _sub_value_spec_from_annotation(args[1])
    return vs.Dict([(ks.StrKey(), elem_value_spec)])
  elif origin is collections.abc.Callable:
    arg_specs = []
    return_spec = None
    if args:
      assert len(args) == 2, (annotation, args)

      # NOTE: Various ways of expressing the input args of `Callable` will
      # be normalized into a list (see examples below). Therefore, we only
      # check against list here.
      #
      #   Callable[int, Any] => Callable[[int], Any]
      #   Callable[(int, int), Any] => Callable[[int, int], Any]
      if isinstance(args[0], list):
        arg_specs = [_sub_value_spec_from_annotation(arg) for arg in args[0]]
      return_spec = _sub_value_spec_from_annotation(args[1])
    return vs.Callable(arg_specs, returns=return_spec)
  # Handling type
  elif origin is type or (annotation in (typing.Type, type)):
    if not args:
      return vs.Type(typing.Any)
    assert len(args) == 1, (annotation, args)
    return vs.Type(args[0])
  # Handling union.
  elif origin is typing.Union or (_UnionType and origin is _UnionType):
    optional = _NoneType in args
    if optional:
      args.remove(_NoneType)
    if len(args) == 1:
      spec = _sub_value_spec_from_annotation(args[0])
    else:
      spec = vs.Union([_sub_value_spec_from_annotation(x) for x in args])
    if optional:
      spec = spec.noneable(use_none_as_default=False)
    return spec
  elif origin is typing.Final:
    return _value_spec_from_type_annotation(
        args[0],
        accept_value_as_annotation=False
    ).freeze(vs._FROZEN_VALUE_PLACEHOLDER)  # pylint: disable=protected-access
  elif isinstance(annotation, typing.ForwardRef):
    annotation = annotation.__forward_arg__
    if parent_module is not None:
      annotation = class_schema.ForwardRef(parent_module, annotation)
    return vs.Object(annotation)
  elif isinstance(annotation, class_schema.ForwardRef):
    return vs.Object(annotation)
  # Handling class.
  elif (
      inspect.isclass(annotation)
      or pg_inspect.is_generic(annotation)
  ):
    return vs.Object(annotation)

  if accept_value_as_annotation:
    spec = _value_spec_from_default_value(annotation)
    if spec is not None:
      return spec
  raise TypeError(
      f'Cannot convert {annotation!r} to `pg.typing.ValueSpec` '
      f'with auto typing.')


def _any_spec_with_annotation(annotation: typing.Any) -> vs.Any:
  """Creates an ``Any`` value spec with annotation."""
  value_spec = vs.Any()
  if annotation != inspect.Parameter.empty:
    value_spec.annotate(annotation)
  return value_spec


def _value_spec_from_annotation(
    annotation: typing.Any,
    auto_typing: bool = False,
    accept_value_as_annotation: bool = False,
    parent_module: typing.Optional[types.ModuleType] = None
    ) -> class_schema.ValueSpec:
  """Creates a value spec from annotation."""
  if isinstance(annotation, class_schema.ValueSpec):
    return annotation
  elif annotation == inspect.Parameter.empty:
    return vs.Any()

  if annotation is None:
    if accept_value_as_annotation:
      return vs.Any().noneable()
    else:
      return vs.Object(type(None))

  if auto_typing:
    return _value_spec_from_type_annotation(
        annotation, accept_value_as_annotation, parent_module
    )
  else:
    value_spec = None
    if accept_value_as_annotation:
      # Accept default values is applicable only when auto typing is off.
      value_spec = _value_spec_from_default_value(annotation)
    return value_spec or _any_spec_with_annotation(annotation)


class_schema.Field.from_annotation = _field_from_annotation
class_schema.ValueSpec.from_annotation = _value_spec_from_annotation
