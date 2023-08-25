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

from pyglove.core import object_utils
from pyglove.core.typing import inspect as pg_inspect


class _TypeConverterRegistry:
  """Type converter registry."""

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
    self._converter_list.append((src, dest, convert_fn))

  def get_converter(
      self, src: Type[Any], dest: Type[Any]) -> Optional[Callable[[Any], Any]]:
    """Get converter from source type to destination type."""
    # TODO(daiyip): Right now we don't see the need of a large number of
    # converters, thus its affordable to iterate the list.
    # We may consider more efficient way to do lookup in future.
    # NOTE(daiyip): We do reverse lookup since usually subclass converter
    # is register after base class.
    for src_type, dest_type, converter in reversed(self._converter_list):
      if pg_inspect.is_subclass(src, src_type):
        dest_types = dest_type if isinstance(dest_type, tuple) else (dest_type,)
        for dest_type in dest_types:
          if pg_inspect.is_subclass(dest_type, dest):
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
  register_converter(str, object_utils.KeyPath,
                     object_utils.KeyPath.parse)
  register_converter(object_utils.KeyPath, str, lambda x: x.path)


_register_builtin_converters()
