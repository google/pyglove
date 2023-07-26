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

import collections
import inspect
import types
import typing

from pyglove.core import object_utils
from pyglove.core.typing import class_schema
from pyglove.core.typing import generic
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


_NoneType = type(None)

# Annotated is suppored after 3.9
_Annotated = getattr(typing, 'Annotated', None)  # pylint: disable=invalid-name

# UnionType is supported after 3.10.
_UnionType = getattr(types, 'UnionType', None)  # pylint: disable=invalid-name


def _field_from_annotation(
    key: typing.Union[str, class_schema.KeySpec],
    annotation: typing.Any,
    description: typing.Optional[str] = None,
    metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
    auto_typing=True,
) -> class_schema.Field:
  """Creates a field from Python annotation."""
  if _Annotated and typing.get_origin(annotation) is _Annotated:
    type_args = typing.get_args(annotation)
    assert len(type_args) > 1, (annotation, type_args)
    field_spec = tuple([key] + list(type_args))
  else:
    field_spec = (key, annotation, description, metadata or {})
  return class_schema.create_field(
      field_spec,
      auto_typing=auto_typing,
      accept_value_as_annotation=False)


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
  elif inspect.isfunction(value) or isinstance(value, object_utils.Functor):
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
    accept_value_as_annotation: bool) -> class_schema.ValueSpec:
  """Creates a value spec from type annotation."""
  if annotation is bool:
    return vs.Bool()
  elif annotation is int:
    return vs.Int()
  elif annotation is float:
    return vs.Float()
  elif annotation is str:
    return vs.Str()
  elif annotation is typing.Any:
    return vs.Any().annotate(annotation)

  origin = typing.get_origin(annotation) or annotation
  args = list(typing.get_args(annotation))

  # Handling list.
  if origin in (list, typing.List):
    return _list_from_annotation(annotation, args)
  # Handling tuple.
  elif origin in (tuple, typing.Tuple):
    return _tuple_from_annotation(annotation, args)
  # Handle sequence.
  elif origin in (collections.abc.Sequence,):
    return _sequence_from_annotation(annotation, args)
  # Handling dict.
  elif origin in (dict, typing.Dict, collections.abc.Mapping):
    return _dict_from_annotation(annotation, args)
  # Handling callable.
  elif origin is collections.abc.Callable:
    return _callable_from_annotation(annotation, args)
  # Handling type
  elif origin is type or (annotation in (typing.Type, type)):
    return _type_from_annotation(annotation, args)
  # Handling union.
  elif origin is typing.Union or (_UnionType and origin is _UnionType):
    return _union_from_annotation(args)
  # Handling class.
  elif (
      inspect.isclass(annotation)
      or generic.is_generic(annotation)
      or (isinstance(annotation, str) and not accept_value_as_annotation)
  ):
    return vs.Object(annotation)

  if accept_value_as_annotation:
    spec = _value_spec_from_default_value(annotation)
    if spec is not None:
      return spec
  raise TypeError(
      f'Cannot convert {annotation!r} to `pg.typing.ValueSpec` '
      f'with auto typing.')


def _list_from_annotation(
    annotation: typing.Any,
    type_args: typing.Sequence[typing.Any]
    ) -> vs.List:
  """Makes list spec from annotation."""
  if type_args:
    assert len(type_args) == 1, (annotation, type_args)
    elem = _value_spec_from_annotation(type_args[0], True)
  else:
    elem = vs.Any()
  return vs.List(elem)


def _tuple_from_annotation(
    annotation: typing.Any,
    type_args: typing.Sequence[typing.Any]
    ) -> vs.Tuple:
  """Makes tuple spec from annotation."""
  if not type_args:
    return vs.Tuple(vs.Any())
  else:
    if type_args[-1] is ...:
      if len(type_args) != 2:
        raise TypeError(
            f'Tuple with ellipsis should have exact 2 type arguments. '
            f'Encountered: {annotation}.')
      return vs.Tuple(_value_spec_from_type_annotation(type_args[0], False))
    return vs.Tuple([_value_spec_from_type_annotation(type_arg, False)
                     for type_arg in type_args])


def _sequence_from_annotation(
    annotation: typing.Any,
    type_args: typing.Sequence[typing.Any]
    ) -> vs.Union:
  """Makes sequence spec from annotation."""
  if type_args:
    assert len(type_args) == 1, (annotation, type_args)
    elem = _value_spec_from_annotation(type_args[0], True)
  else:
    elem = vs.Any()
  return vs.Union([vs.List(elem), vs.Tuple(elem)])


def _dict_from_annotation(
    annotation: typing.Any,
    type_args: typing.Sequence[typing.Any]
    ) -> vs.Dict:
  """Makes dict spec from annotation."""
  if not type_args:
    return vs.Dict(non_symbolic=True)

  assert len(type_args) == 2, (annotation, type_args)
  if type_args[0] not in (str, typing.Text):
    return vs.Dict(non_symbolic=True)

  elem_value_spec = _value_spec_from_type_annotation(
      type_args[1], accept_value_as_annotation=False)

  return vs.Dict([(ks.StrKey(), elem_value_spec)])


def _callable_from_annotation(
    annotation: typing.Any,
    type_args: typing.Sequence[typing.Any]
    ) -> vs.Callable:
  """Makes callable spec from annotation."""
  arg_specs = []
  return_spec = None
  if type_args:
    assert len(type_args) == 2, (annotation, type_args)

    # NOTE: Various ways of expressing the input args of `Callable` will
    # be normalized into a list (see examples below). Therefore, we only
    # check against list here.
    #
    #   Callable[int, Any] => Callable[[int], Any]
    #   Callable[(int, int), Any] => Callable[[int, int], Any]
    if isinstance(type_args[0], list):
      arg_specs = [
          _value_spec_from_type_annotation(
              arg, accept_value_as_annotation=False)
          for arg in type_args[0]
      ]
    return_spec = _value_spec_from_type_annotation(
        type_args[1], accept_value_as_annotation=False)

  return vs.Callable(arg_specs, returns=return_spec)


def _type_from_annotation(
    annotation: typing.Any,
    type_args: typing.Sequence[typing.Any]
    ) -> vs.Type:
  """Makes type spec from annotation."""
  if not type_args:
    return vs.Type(typing.Any)
  assert len(type_args) == 1, (annotation, type_args)
  return vs.Type(type_args[0])


def _union_from_annotation(
    type_args: typing.List[typing.Any]
    ) -> class_schema.ValueSpec:
  """Makes union spec from annotation."""
  optional = _NoneType in type_args
  if optional:
    type_args.remove(_NoneType)
  if len(type_args) == 1:
    spec = _value_spec_from_annotation(type_args[0], True)
  else:
    spec = vs.Union([
        _value_spec_from_annotation(x, True) for x in set(type_args)])
  if optional:
    spec = spec.noneable()
  return spec


def _any_spec_with_annotation(annotation: typing.Any) -> vs.Any:
  """Creates an ``Any`` value spec with annotation."""
  value_spec = vs.Any()
  if annotation != inspect.Parameter.empty:
    value_spec.annotate(annotation)
  return value_spec


def _value_spec_from_annotation(
    annotation: typing.Any,
    auto_typing=False,
    accept_value_as_annotation=False
    ) -> class_schema.ValueSpec:
  """Creates a value spec from annotation."""
  if isinstance(annotation, class_schema.ValueSpec):
    return annotation
  elif annotation == inspect.Parameter.empty:
    return vs.Any()
  elif annotation is None:
    if accept_value_as_annotation:
      return vs.Any().noneable()
    else:
      return vs.Any().freeze(None)

  if auto_typing:
    return _value_spec_from_type_annotation(
        annotation, accept_value_as_annotation)
  else:
    value_spec = None
    if accept_value_as_annotation:
      # Accept default values is applicable only when auto typing is off.
      value_spec = _value_spec_from_default_value(annotation)
    return value_spec or _any_spec_with_annotation(annotation)


class_schema.Field.from_annotation = _field_from_annotation
class_schema.ValueSpec.from_annotation = _value_spec_from_annotation
