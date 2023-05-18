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
    return vs.List(
        _value_spec_from_annotation(
            args[0], True)) if args else vs.List(vs.Any())
  # Handling tuple.
  elif origin in (tuple, typing.Tuple):
    if args:
      return vs.Tuple([_value_spec_from_annotation(arg, True) for arg in args])
    else:
      return vs.Tuple(vs.Any())
  # Handle sequence.
  elif origin in (collections.abc.Sequence,):
    elem = _value_spec_from_annotation(args[0], True) if args else vs.Any()
    return vs.Union([vs.List(elem), vs.Tuple(elem)])
  # Handling dict.
  elif origin in (dict, typing.Dict, collections.abc.Mapping):
    # TODO(daiyip): Handle dict schema according to the dict args.
    return vs.Dict()
  # Handling union.
  elif origin is typing.Union or (_UnionType and origin is _UnionType):
    optional = _NoneType in args
    if optional:
      args.remove(_NoneType)
    if len(args) == 1:
      spec = _value_spec_from_annotation(args[0], True)
    else:
      spec = vs.Union([_value_spec_from_annotation(x, True) for x in set(args)])
    if optional:
      spec = spec.noneable()
    return spec
  # Handling class.
  elif (inspect.isclass(annotation)
        or (isinstance(annotation, str) and not accept_value_as_annotation)):
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
