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
"""Automatic type conversions."""

import calendar
import datetime
from typing import Any, Callable, Optional, Tuple, Type, Union

from pyglove.core import utils
from pyglove.core.typing import inspect as pg_inspect


class _TypeConverterRegistry:
  """Type converter registry."""

  _JSON_VALUE_TYPES = frozenset(
      [int, float, bool, type(None), list, tuple, dict, str]
  )

  def __init__(self):
    """Constructor."""
    self._converter_list = []

  def register(
      self,
      src: Union[Type[Any], Tuple[Type[Any], ...]],
      dest: Union[Type[Any], Tuple[Type[Any], ...]],
      convert_fn: Callable[[Any], Any]) -> None:  # pyformat: disable pylint: disable=line-too-long
    """Register a converter from src type to dest type."""
    if (
        not isinstance(src, (tuple, type))
        and not pg_inspect.is_generic(src)
        or not isinstance(dest, (tuple, type))
        and not pg_inspect.is_generic(dest)
    ):
      raise TypeError('Argument \'src\' and \'dest\' must be a type or '
                      'tuple of types.')
    if isinstance(dest, tuple):
      json_value_convertible = any(d in self._JSON_VALUE_TYPES for d in dest)
    else:
      json_value_convertible = dest in self._JSON_VALUE_TYPES
    self._converter_list.append((src, dest, convert_fn, json_value_convertible))

  def get_converter(
      self, src: Type[Any], dest: Type[Any]) -> Optional[Callable[[Any], Any]]:
    """Get converter from source type to destination type."""
    if pg_inspect.is_protocol(dest):
      return None

    # TODO(daiyip): Right now we don't see the need of a large number of
    # converters, thus its affordable to iterate the list.
    # We may consider more efficient way to do lookup in future.
    # NOTE(daiyip): We do reverse lookup since usually subclass converter
    # is register after base class.
    for src_type, dest_type, converter, _ in reversed(self._converter_list):
      if pg_inspect.is_subclass(src, src_type):
        dest_types = dest_type if isinstance(dest_type, tuple) else (dest_type,)
        for dest_type in dest_types:
          if pg_inspect.is_subclass(dest_type, dest):
            return converter
    return None

  def get_json_value_converter(
      self, src: Type[Any]) -> Optional[Callable[[Any], Any]]:
    """Get converter from source type to a JSON simple type."""
    for src_type, _, converter, json_value_convertible in reversed(
        self._converter_list):
      if pg_inspect.is_subclass(src, src_type) and json_value_convertible:
        return converter
    return None


_TYPE_CONVERTER_REGISTRY = _TypeConverterRegistry()


def get_converter(
    src: Type[Any], dest: Union[Type[Any], Tuple[Type[Any], ...]]
) -> Optional[Callable[[Any], Any]]:
  """Get converter from source type to destination type."""
  dest_types = dest if isinstance(dest, tuple) else (dest,)
  for dest in dest_types:
    converter = _TYPE_CONVERTER_REGISTRY.get_converter(src, dest)
    if converter is not None:
      return converter
  return None


def get_json_value_converter(src: Type[Any]) -> Optional[Callable[[Any], Any]]:
  """Get converter from source type to a JSON simple type."""
  return _TYPE_CONVERTER_REGISTRY.get_json_value_converter(src)


def register_converter(
    src_type: Union[Type[Any], Tuple[Type[Any], ...]],
    dest_type: Union[Type[Any], Tuple[Type[Any], ...]],
    convert_fn: Callable[[Any], Any]) -> None:
  """Register converter from source type to destination type.

  Examples::

    # Add converter from int to float.
    pg.typing.register_converter(int, float, float)

    assert pg.typing.Float().apply(1) is 1.0

    # Add converter from a dict to class A.
    def from_dict(d):
      return A(**d)

    assert isinstance(pg.typing.Object(A).apply({'x': 1, 'y': 2}), A)

  Args:
      src_type: Source value type.
      dest_type: Target value type.
      convert_fn: Function that performs the conversion, in signature
        (src_type) -> dest_type.
  """
  _TYPE_CONVERTER_REGISTRY.register(src_type, dest_type, convert_fn)


def _register_builtin_converters():
  """Register built-in converters."""
  # int => float.
  register_converter(int, float, float)

  # int <=> datetime.datetime.
  register_converter(int, datetime.datetime, datetime.datetime.utcfromtimestamp)
  register_converter(datetime.datetime, int,
                     lambda x: calendar.timegm(x.timetuple()))

  # string <=> KeyPath.
  register_converter(str, utils.KeyPath, utils.KeyPath.parse)
  register_converter(utils.KeyPath, str, lambda x: x.path)


_register_builtin_converters()
utils.JSONConvertible.TYPE_CONVERTER = get_json_value_converter
