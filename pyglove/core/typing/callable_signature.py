# Copyright 2022 The PyGlove Authors
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
"""Callable signatures based on PyGlove typing."""

import dataclasses
import enum
import inspect
import sys
import types
import typing
from typing import Any, Callable, Dict, List, Optional, Union

from pyglove.core import coding
from pyglove.core import utils
from pyglove.core.typing import class_schema
from pyglove.core.typing import key_specs as ks


@dataclasses.dataclass
class Argument:
  """Definition for a callable argument."""

  class Kind(enum.Enum):
    """Arugment kind."""
    POSITIONAL_OR_KEYWORD = 1
    VAR_POSITIONAL = 2
    KEYWORD_ONLY = 3
    VAR_KEYWORD = 4

    @classmethod
    def from_parameter(cls, parameter: inspect.Parameter) -> 'Argument.Kind':
      """Returns Argument.Kind from inspect.Parameter."""
      if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        return Argument.Kind.POSITIONAL_OR_KEYWORD
      elif parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
        return Argument.Kind.POSITIONAL_OR_KEYWORD
      elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
        return Argument.Kind.VAR_POSITIONAL
      elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
        return Argument.Kind.KEYWORD_ONLY
      else:
        assert parameter.kind == inspect.Parameter.VAR_KEYWORD, parameter.kind
        return Argument.Kind.VAR_KEYWORD

  name: str
  kind: Kind
  value_spec: class_schema.ValueSpec
  description: Optional[str] = None

  def __post_init__(self):
    if self.kind == Argument.Kind.VAR_POSITIONAL:
      if not isinstance(self.value_spec, class_schema.ValueSpec.ListType):
        raise TypeError(
            f'Variable positional argument {self.name!r} should have a value '
            f'of `pg.typing.List` type. Encountered: {self.value_spec!r}.'
        )
      self.value_spec.set_default([])

    if (self.kind == Argument.Kind.VAR_KEYWORD
        and not isinstance(self.value_spec, class_schema.ValueSpec.DictType)):
      raise TypeError(
          f'Variable keyword argument {self.name!r} should have a value of '
          f'`pg.typing.Dict` type. Encountered: {self.value_spec!r}.'
      )

  @classmethod
  def from_annotation(
      cls,
      name: str,
      kind: Kind,
      annotation: Any = inspect.Parameter.empty,
      auto_typing: bool = False,
      parent_module: Optional[types.ModuleType] = None) -> 'Argument':
    """Creates an argument from annotation."""
    return cls(
        name,
        kind,
        class_schema.ValueSpec.from_annotation(
            annotation, auto_typing=auto_typing, parent_module=parent_module
        )
    )

  @classmethod
  def from_parameter(
      cls,
      param: inspect.Parameter,
      description: Optional[str] = None,
      auto_typing: bool = True,
      parent_module: Optional[types.ModuleType] = None,
  ) -> 'Argument':
    """Creates an argument from inspect.Parameter."""
    value_spec = class_schema.ValueSpec.from_annotation(
        param.annotation, auto_typing=auto_typing, parent_module=parent_module
    )
    if param.default != inspect.Parameter.empty:
      value_spec.set_default(param.default)

    # pytype: disable=wrong-arg-count
    # pytype: disable=not-instantiable
    if param.kind == inspect.Parameter.VAR_POSITIONAL:
      value_spec = class_schema.ValueSpec.ListType(value_spec, default=[])
    elif param.kind == inspect.Parameter.VAR_KEYWORD:
      value_spec = class_schema.ValueSpec.DictType(value_spec)
    # pytype: enable=wrong-arg-count
    # pytype: enable=not-instantiable
    return cls(
        param.name,
        Argument.Kind.from_parameter(param),
        value_spec,
        description=description
    )

  def to_field(self) -> class_schema.Field:
    """Converts current argument to a pg.typing.Field object."""
    if self.kind == Argument.Kind.VAR_KEYWORD:
      key = ks.StrKey()
      value = self.value_spec.schema.dynamic_field.value  # pytype: disable=attribute-error
    else:
      key = ks.ConstStrKey(self.name)
      value = self.value_spec
    return class_schema.Field(key, value, description=self.description)


class CallableType(enum.Enum):
  """Enum for Callable type."""
  # Regular function or lambdas without a subject bound.
  FUNCTION = 1

  # Function that is bound with subject. Like class methods or instance methods.
  METHOD = 2


class Signature(utils.Formattable):
  """PY3 function signature."""

  def __init__(self,
               callable_type: CallableType,
               name: str,
               module_name: str,
               args: Optional[List[Argument]] = None,
               kwonlyargs: Optional[List[Argument]] = None,
               varargs: Optional[Argument] = None,
               varkw: Optional[Argument] = None,
               return_value: Optional[class_schema.ValueSpec] = None,
               qualname: Optional[str] = None,
               description: Optional[str] = None):
    """Constructor.

    Args:
      callable_type: Type of callable.
      name: Function name.
      module_name: Module name.
      args: Specification for positional arguments
      kwonlyargs: Specification for keyword only arguments (PY3).
      varargs: Specification for wildcard list argument, e.g, 'args' is the name
        for `*args`.
      varkw: Specification for wildcard keyword argument, e.g, 'kwargs' is the
        name for `**kwargs`.
      return_value: Optional value spec for return value.
      qualname: Optional qualified name.
      description: Optional description of the signature.
    """
    args = args or []
    self.callable_type = callable_type
    self.name = name
    self.module_name = module_name
    self.args = args or []
    self.kwonlyargs = kwonlyargs or []
    self.varargs = varargs
    self.varkw = varkw
    self.return_value = return_value
    self.qualname = qualname or name
    self.description = description

  @property
  def named_args(self) -> List[Argument]:
    """Returns all named arguments according to their declaration order."""
    return self.args + self.kwonlyargs

  @property
  def arg_names(self):
    """Returns names of all arguments according to their declaration order."""
    return [arg.name for arg in self.named_args]

  def get_value_spec(self, name: str) -> Optional[class_schema.ValueSpec]:
    """Returns Value spec for an argument name.

    Args:
      name: Argument name.

    Returns:
      ValueSpec for the requested argument. If name is not found, value spec of
      wildcard keyword argument will be used. None will be returned if name
      does not exist in signature and wildcard keyword is not accepted.
    """
    for arg in self.named_args:
      if arg.name == name:
        return arg.value_spec
    if self.varkw is not None:
      return self.varkw.value_spec.schema.dynamic_field.value   # pytype: disable=attribute-error
    return None

  @property
  def id(self) -> str:
    """Returns ID of the function."""
    return f'{self.module_name}.{self.qualname}'

  @property
  def has_varargs(self) -> bool:
    """Returns whether wildcard positional argument is present."""
    return self.varargs is not None

  @property
  def has_varkw(self) -> bool:
    """Returns whether wildcard keyword argument is present."""
    return self.varkw is not None

  @property
  def has_wildcard_args(self) -> bool:
    """Returns whether any wildcard arguments are present."""
    return self.has_varargs or self.has_varkw

  def __ne__(self, other: Any) -> bool:
    """Not equals."""
    return not self.__eq__(other)

  def __eq__(self, other: Any) -> bool:
    """Equals."""
    if not isinstance(other, self.__class__):
      return False
    if self is other:
      return True
    return (self.callable_type == other.callable_type and
            self.name == other.name and
            self.qualname == other.qualname and
            self.module_name == other.module_name and
            self.args == other.args and self.kwonlyargs == other.kwonlyargs and
            self.varargs == other.varargs and self.varkw == other.varkw and
            self.return_value == other.return_value)

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs,
  ) -> str:
    """Format current object."""
    return utils.kvlist_str(
        [
            ('', self.id, ''),
            ('args', self.args, []),
            ('kwonlyargs', self.kwonlyargs, []),
            ('returns', self.return_value, None),
            ('varargs', self.varargs, None),
            ('varkw', self.varkw, None),
            ('description', self.description, None),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        **kwargs,
    )

  def annotate(
      self,
      args: Union[
          Dict[class_schema.FieldKeyDef, class_schema.FieldValueDef],
          List[class_schema.FieldDef],
          None,
      ] = None,
      return_value: Union[class_schema.ValueSpec, Any, None] = None,
  ) -> 'Signature':
    """Annotate arguments with extra typing."""
    if return_value is not None:
      return_value = class_schema.ValueSpec.from_annotation(
          return_value, auto_typing=True
      )
      if utils.MISSING_VALUE != return_value.default:
        raise ValueError('return value spec should not have default value.')
      self.return_value = return_value

    if not args:
      return self

    schema = class_schema.create_schema(args, allow_nonconst_keys=True)  # pylint: disable=redefined-outer-name

    arg_fields: Dict[str, class_schema.Field] = dict()
    varargs_field = None
    kwarg_field = None
    existing_names = set(self.arg_names)
    extra_arg_names = []

    for arg_name, field in schema.fields.items():
      if isinstance(arg_name, ks.StrKey):
        if kwarg_field is not None:
          raise KeyError(
              f'{self.id}: multiple StrKey found in override args.'
          )
        kwarg_field = field
      else:
        assert isinstance(arg_name, (str, ks.ConstStrKey))
        if self.varargs and self.varargs.name == arg_name:
          varargs_field = field

        elif self.varkw and self.varkw.name == arg_name:
          if kwarg_field is not None:
            raise KeyError(
                f'{self.id}: multiple StrKey found in '
                f'symbolic arguments declaration.')
          kwarg_field = field
        elif arg_name not in existing_names:
          if self.has_varkw:
            extra_arg_names.append(arg_name)
          else:
            raise KeyError(
                f'{self.id}: found extra symbolic argument {arg_name.text!r}.')
        arg_fields[arg_name.text] = field

    def update_arg(arg: Argument, field: class_schema.Field):
      """Updates an argument with override field."""
      if arg.value_spec.has_default and (
          not field.value.has_default
          # Loose the default as user may mark it as noneable.
          or field.value.default is None
      ):
        field.value.set_default(
            arg.value_spec.default, root_path=utils.KeyPath(arg.name)
        )
      if arg.value_spec.default != field.value.default:
        if field.value.is_noneable and not arg.value_spec.has_default:
          # Special handling noneable which always comes with a default.
          field.value.set_default(utils.MISSING_VALUE)
        elif not (
            # Special handling Dict type which always has default.
            isinstance(field.value, class_schema.ValueSpec.DictType)
            and not arg.value_spec.has_default
        ):
          raise ValueError(
              f'The annotated default value ({field.default_value}) of '
              f'symbolic argument {arg.name!r} is not equal to the default '
              f'value ({arg.value_spec.default}) specified from the function '
              'signature.'
          )
      arg.value_spec = field.value
      if field.description:
        arg.description = field.description

    # Named arguments.
    for arg in self.named_args:
      field = arg_fields.get(arg.name)
      if field is not None:
        update_arg(arg, field)

    # Add extra arguments.
    for arg_name in extra_arg_names:
      field = arg_fields.get(arg_name)
      assert field is not None
      self.kwonlyargs.append(
          Argument(
              arg_name.text,
              Argument.Kind.KEYWORD_ONLY,
              field.value,
              field.description
          )
      )

    # Update varargs.
    if varargs_field is not None:
      assert self.varargs is not None
      if not isinstance(varargs_field.value, class_schema.ValueSpec.ListType):
        raise ValueError(
            f'Variable positional argument {self.varargs.name!r} should have a '
            'value of `pg.typing.List` type. '
            f'Encountered: {varargs_field.value!r}.'
        )
      update_arg(self.varargs, varargs_field)

    # Update kwarg.
    if kwarg_field is not None:
      assert self.varkw is not None
      value_spec = class_schema.ValueSpec.DictType(kwarg_field.value)
      self.varkw.value_spec = value_spec
      if kwarg_field.description:
        self.varkw.description = kwarg_field.description

    return self

  def fields(
      self,
      remove_self: bool = True,
      include_return: bool = False,
  ) -> List[class_schema.Field]:
    """Returns the fields of this signature."""
    fields = [
        arg.to_field() for i, arg in enumerate(self.args)
        if not remove_self or i > 0 or arg.name != 'self'
    ]
    if self.varargs:
      fields.append(self.varargs.to_field())
    fields.extend([arg.to_field() for arg in self.kwonlyargs])
    if self.varkw:
      fields.append(self.varkw.to_field())
    if include_return and self.return_value:
      fields.append(
          class_schema.Field('return', self.return_value, 'Return value.')
      )
    return fields

  def to_schema(
      self,
      remove_self: bool = True,
      include_return: bool = False,
  ) -> class_schema.Schema:
    """Returns the schema of this signature."""
    init_arg_list = [arg.name for arg in self.args]
    if init_arg_list and init_arg_list[0] == 'self':
      init_arg_list.pop(0)

    if self.varargs:
      init_arg_list.append(f'*{self.varargs.name}')

    return class_schema.Schema(
        self.fields(remove_self=remove_self, include_return=include_return),
        name=f'{self.module_name}.{self.qualname}',
        description=self.description,
        allow_nonconst_keys=True,
        metadata=dict(
            init_arg_list=init_arg_list,
            varargs_name=self.varargs.name if self.varargs else None,
            varkw_name=self.varkw.name if self.varkw else None,
            returns=self.return_value,
        ),
    )

  @classmethod
  def from_schema(
      cls,
      schema: class_schema.Schema,    # pylint: disable=redefined-outer-name
      module_name: str,
      name: str,
      qualname: Optional[str] = None,
      is_method: bool = True) -> 'Signature':
    """Creates a signature from a schema object.

    Args:
      schema: A `pg.typing.Schema` object associated with a `pg.Object`.
      module_name: Module name for the signature.
      name: Function or method name of the signature.
      qualname: Qualname of the signature.
      is_method: If True, `self` will be added in the signature as the first
        argument.

    Returns:
      A signature object from the schema.
    """
    arg_names = list(schema.metadata.get('init_arg_list', []))
    if arg_names and arg_names[-1].startswith('*'):
      vararg_name = arg_names[-1][1:]
      arg_names.pop(-1)
    else:
      vararg_name = None

    def get_arg_spec(arg_name):
      field = schema.get_field(arg_name)
      if not field:
        raise ValueError(f'Argument {arg_name!r} is not a symbolic field.')
      return field.value

    args = []
    if is_method:
      args.append(
          Argument.from_annotation('self', Argument.Kind.POSITIONAL_OR_KEYWORD)
      )

    # Prepare positional arguments.
    args.extend([
        Argument(n, Argument.Kind.POSITIONAL_OR_KEYWORD, get_arg_spec(n))
        for n in arg_names
    ])

    # Prepare varargs.
    varargs = None
    if vararg_name:
      varargs = Argument(
          vararg_name,
          Argument.Kind.VAR_POSITIONAL,
          get_arg_spec(vararg_name)
      )  # pytype: disable=attribute-error

    # Prepare keyword-only arguments.
    existing_names = set(arg_names)
    if vararg_name:
      existing_names.add(vararg_name)

    kwonlyargs = []
    varkw = None
    for key, field in schema.fields.items():
      if key not in existing_names and not field.frozen:
        if key.is_const:
          kwonlyargs.append(
              Argument(str(key), Argument.Kind.KEYWORD_ONLY, field.value)
          )
        else:
          # pytype: disable=not-instantiable
          # pytype: disable=wrong-arg-count
          varkw = Argument(
              schema.metadata.get('varkw_name', None) or 'kwargs',
              Argument.Kind.VAR_KEYWORD,
              class_schema.ValueSpec.DictType(field.value)
          )
          # pytype: enable=wrong-arg-count
          # pytype: enable=not-instantiable

    return Signature(
        callable_type=CallableType.FUNCTION,
        name=name,
        module_name=module_name,
        qualname=qualname,
        description=schema.description,
        args=args,
        kwonlyargs=kwonlyargs,
        varargs=varargs,
        varkw=varkw,
        return_value=schema.metadata.get('returns', None)
    )

  @classmethod
  def from_callable(
      cls,
      callable_object: Callable[..., Any],
      auto_typing: bool = False,
      auto_doc: bool = False,
  ) -> 'Signature':
    """Creates Signature from a callable object."""
    callable_object = typing.cast(object, callable_object)
    if not callable(callable_object):
      raise TypeError(f'{callable_object!r} is not callable.')

    if isinstance(callable_object, utils.Functor):
      assert callable_object.__signature__ is not None
      return callable_object.__signature__

    func = callable_object
    docstr = None
    if inspect.isclass(func):
      callable_type = CallableType.METHOD
      try:
        sig = inspect.signature(func)
      except ValueError:
        sig = inspect.signature(func.__init__)

      if auto_doc:
        description = None
        args_doc = {}
        if func.__doc__:
          cls_doc = utils.DocStr.parse(func.__doc__)
          description = cls_doc.short_description
          args_doc.update(cls_doc.args)

        if func.__init__.__doc__:
          init_doc = utils.DocStr.parse(func.__init__.__doc__)
          args_doc.update(init_doc.args)
        docstr = utils.DocStr(
            utils.DocStrStyle.GOOGLE,
            short_description=description,
            long_description=None,
            examples=[],
            args=args_doc,
            returns=None,
            raises=[],
            blank_after_short_description=True,
        )
    else:
      if not inspect.isroutine(func):
        if not inspect.isroutine(callable_object.__call__):
          raise TypeError(f'{callable_object!r}.__call__ is not a method.')
        func = callable_object.__call__
      callable_type = (
          CallableType.METHOD if inspect.ismethod(func)
          else CallableType.FUNCTION
      )
      if auto_doc:
        docstr = utils.docstr(func)
      sig = inspect.signature(func)

    module_name = getattr(func, '__module__', None)
    return cls.from_signature(
        sig=sig,
        name=func.__name__,
        qualname=func.__qualname__,
        callable_type=callable_type,
        module_name=module_name or 'wrapper',
        auto_typing=auto_typing,
        docstr=docstr,
        parent_module=sys.modules[module_name] if module_name else None
    )

  @classmethod
  def from_signature(
      cls,
      sig: inspect.Signature,
      name: str,
      callable_type: CallableType,
      module_name: Optional[str] = None,
      qualname: Optional[str] = None,
      auto_typing: bool = False,
      docstr: Union[str, utils.DocStr, None] = None,
      parent_module: Optional[types.ModuleType] = None,
  ) -> 'Signature':
    """Returns PyGlove signature from Python signature.

    Args:
      sig: Python signature.
      name: Name of the entity (class name or function/method name).
      callable_type: the type of this callable.
      module_name: Module name of the entity.
      qualname: (Optional) qualified name of the entity.
      auto_typing: If True, automatically convert argument annotations 
        to PyGlove ValueSpec objects. Otherwise use pg.typing.Any()
        with annotations.
      docstr: (Optional) DocStr for this entity.
      parent_module: (Optional) Parent module from where the signature is
        derived. This is useful to infer classes with forward declarations.

    Returns:
      A PyGlove Signature object.
    """
    args = []
    kwonly_args = []
    varargs = None
    varkw = None

    if isinstance(docstr, str):
      docstr = utils.DocStr.parse(docstr)

    def make_arg_spec(param: inspect.Parameter) -> Argument:
      """Makes argument spec from inspect.Parameter."""
      docstr_arg = docstr.parameter(param) if docstr else None
      return Argument.from_parameter(
          param,
          description=docstr_arg.description if docstr_arg else None,
          auto_typing=auto_typing,
          parent_module=parent_module,
      )

    for param in sig.parameters.values():
      arg_spec = make_arg_spec(param)
      if (param.kind == inspect.Parameter.POSITIONAL_ONLY
          or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
        args.append(arg_spec)
      elif param.kind == inspect.Parameter.KEYWORD_ONLY:
        kwonly_args.append(arg_spec)
      elif param.kind == inspect.Parameter.VAR_POSITIONAL:
        varargs = arg_spec
      else:
        assert param.kind == inspect.Parameter.VAR_KEYWORD, param.kind
        varkw = arg_spec

    return_value = None
    if sig.return_annotation is not inspect.Parameter.empty:
      return_value = class_schema.ValueSpec.from_annotation(
          sig.return_annotation,
          auto_typing=auto_typing,
          parent_module=parent_module
      )

    return cls(
        callable_type=callable_type,
        name=name,
        module_name=module_name,
        qualname=qualname,
        description=docstr.short_description if docstr else None,
        args=args,
        kwonlyargs=kwonly_args,
        varargs=varargs,
        varkw=varkw,
        return_value=return_value,
    )

  def make_function(
      self,
      body: List[str],
      exec_globals: Optional[Dict[str, Any]] = None,
      exec_locals: Optional[Dict[str, Any]] = None):
    """Makes a function with current signature."""
    if exec_globals is None:
      exec_globals = {}
    if exec_locals is None:
      exec_locals = {}

    args = []
    def _append_arg(
        arg_name: str,
        arg_spec: class_schema.ValueSpec,
        force_missing_as_default: bool = False,
        arg_prefix: str = ''):
      s = [f'{arg_prefix}{arg_name}']
      if arg_spec.annotation != utils.MISSING_VALUE:
        s.append(f': _annotation_{arg_name}')
        exec_locals[f'_annotation_{arg_name}'] = arg_spec.annotation
      if not arg_prefix and (force_missing_as_default or arg_spec.has_default):
        s.append(f' = _default_{arg_name}')
        exec_locals[f'_default_{arg_name}'] = arg_spec.default
      args.append(''.join(s))

    has_previous_default = False
    # Build positional arguments.
    for arg in self.args:
      _append_arg(arg.name, arg.value_spec, has_previous_default)
      if arg.value_spec.has_default:
        has_previous_default = True

    # Build variable positional arguments.
    if self.varargs:
      assert isinstance(
          self.varargs.value_spec,
          class_schema.ValueSpec.ListType
      ), self.varargs
      _append_arg(
          self.varargs.name,
          getattr(self.varargs.value_spec, 'element'),
          arg_prefix='*',
      )
    elif self.kwonlyargs:
      args.append('*')

    # Build keyword-only arguments.
    for arg in self.kwonlyargs:
      _append_arg(arg.name, arg.value_spec)

    # Build variable keyword arguments.
    if self.varkw:
      assert isinstance(
          self.varkw.value_spec,
          class_schema.ValueSpec.DictType
      ), self.varkw
      _append_arg(
          self.varkw.name,
          self.varkw.value_spec.schema.dynamic_field.value,   # pytype: disable=attribute-error
          arg_prefix='**'
      )

    # Generate function.
    fn = coding.make_function(
        self.name,
        args=args,
        body=body,
        exec_globals=exec_globals,
        exec_locals=exec_locals,
        return_type=getattr(
            self.return_value, 'annotation', coding.NO_TYPE_ANNOTATION
        ),
    )
    fn.__module__ = self.module_name
    fn.__name__ = self.name
    fn.__qualname__ = self.qualname
    return fn


def signature(
    func: Callable[..., Any],
    auto_typing: bool = True,
    auto_doc: bool = True,
) -> Signature:  # pylint:disable=g-bare-generic
  """Gets signature from a python callable."""
  return Signature.from_callable(func, auto_typing, auto_doc)


def schema(
    cls_or_fn: Callable[..., Any],
    args: Union[
        List[Union[class_schema.Field, class_schema.FieldDef]],
        Dict[class_schema.FieldKeyDef, class_schema.FieldValueDef],
        None
    ] = None,
    returns: Optional[class_schema.ValueSpec] = None,
    *,
    auto_typing: bool = True,
    auto_doc: bool = True,
    remove_self: bool = True,
    include_return: bool = False,
) -> class_schema.Schema:
  """Returns the schema from the signature of a class or a function.

  Args:
    cls_or_fn: A class or a function.
    args: (Optional) additional annotations for arguments.
    returns: (Optional) additional annotation for return value.
    auto_typing: If True, enable type inference from annotations.
    auto_doc: If True, extract schema/field description form docstrs.
    remove_self: If True, remove the first `self` argument if it appears in the
      signature.
    include_return: If True, include the return value spec in the schema with
      key 'return_value'.

  Returns:
    A pg.typing.Schema object.
  """
  s = getattr(cls_or_fn, '__schema__', None)
  if isinstance(s, class_schema.Schema):
    return s
  return signature(
      cls_or_fn, auto_typing=auto_typing, auto_doc=auto_doc
  ).annotate(
      args, return_value=returns
  ).to_schema(
      remove_self=remove_self,
      include_return=include_return,
  )
