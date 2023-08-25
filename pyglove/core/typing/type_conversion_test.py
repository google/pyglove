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
"""Tests for pyglove.core.typing.type_conversion."""

import calendar
import datetime
import typing
import unittest

from pyglove.core import object_utils
from pyglove.core.typing import annotation_conversion  # pylint: disable=unused-import
from pyglove.core.typing import type_conversion
from pyglove.core.typing import value_specs as vs


class TypeConversionTest(unittest.TestCase):
  """Tests for type conversion during assignment."""

  def test_register_and_get_converter(self):

    class A:

      def __init__(self, x):
        self.x = x

    class B(A):

      def __init__(self, x, y):
        super().__init__(x)
        self.y = y

    a_converter = lambda a: a.x
    type_conversion.register_converter(A, (str, int), a_converter)

    self.assertIs(type_conversion.get_converter(A, str), a_converter)
    self.assertIs(type_conversion.get_converter(A, int), a_converter)
    self.assertIs(type_conversion.get_json_value_converter(A), a_converter)

    self.assertIsNone(type_conversion.get_converter(A, (float, bool)))
    self.assertIs(type_conversion.get_converter(A, (float, int)), a_converter)

    # B is a subclass of A, so A's converter applies.
    self.assertIs(type_conversion.get_converter(B, str), a_converter)
    self.assertIs(type_conversion.get_converter(B, int), a_converter)
    self.assertIs(type_conversion.get_json_value_converter(B), a_converter)

    b_converter = lambda b: b.y
    type_conversion.register_converter(B, (str, int), b_converter)

    # `b_converter` takes precedence over `a_converter` since it's an absolute
    # match.
    self.assertIs(type_conversion.get_converter(B, str), b_converter)
    self.assertIs(type_conversion.get_converter(B, int), b_converter)
    self.assertIs(type_conversion.get_json_value_converter(B), b_converter)

    self.assertIsNone(type_conversion.get_converter(B, (float, bool)))
    self.assertIs(type_conversion.get_converter(B, (float, int)), b_converter)

    # Test generics.
    T = typing.TypeVar('T')

    class G(typing.Generic[T]):

      def __init__(self, x: T):
        super().__init__()
        self.x = x

    class G1(G[int]):
      pass

    class G2(G[str]):
      pass

    type_conversion.register_converter(int, G1, G1)
    type_conversion.register_converter(str, G[str], G2)
    self.assertIs(type_conversion.get_converter(int, G[int]), G1)
    self.assertIs(type_conversion.get_converter(int, G1), G1)
    self.assertIs(type_conversion.get_converter(str, G[str]), G2)
    self.assertIsNone(type_conversion.get_converter(str, G2))

  def test_user_conversion(self):

    class A:

      def __init__(self, x):
        self.x = x

    class B:
      pass

    type_conversion.register_converter((int, str), A, A)
    type_conversion.register_converter(A, int, lambda a: a.x)

    # NOTE(daiyip): Consider places that accepts B also accepts A.
    type_conversion.register_converter(A, B, lambda x: x)

    with self.assertRaisesRegex(
        TypeError,
        'Argument \'src\' and \'dest\' must be a type or tuple of types.'):
      type_conversion.register_converter(0, 1, lambda x: x)

    self.assertEqual(
        vs.Union([vs.Object(B), vs.Float()]).apply(A(1)).x, 1)
    self.assertEqual(vs.Object(A).apply(1).x, 1)
    self.assertEqual(vs.Object(A).apply('foo').x, 'foo')
    self.assertEqual(type_conversion.get_json_value_converter(A)(A(1)), 1)
    self.assertIsNone(type_conversion.get_json_value_converter(B))


class BuiltInConversionsTest(unittest.TestCase):
  """Tests for built-in conversions."""

  def test_int_to_float(self):
    self.assertEqual(vs.Float().apply(1), 1.0)

  def test_datetime_to_int(self):
    """Test built-in converter between int and datetime.datetime."""
    timestamp = calendar.timegm(datetime.datetime.now().timetuple())
    now = datetime.datetime.utcfromtimestamp(timestamp)
    self.assertEqual(vs.Object(datetime.datetime).apply(timestamp), now)
    self.assertEqual(vs.Int().apply(now), timestamp)
    self.assertEqual(
        type_conversion.get_json_value_converter(datetime.datetime)(now),
        timestamp)

  def test_keypath_to_str(self):
    """Test built-in converter between string and KeyPath."""
    self.assertEqual(
        vs.Object(object_utils.KeyPath).apply('a.b.c').keys,
        ['a', 'b', 'c'])
    self.assertEqual(
        vs.Union([vs.Object(object_utils.KeyPath), vs.Int()]).apply(
            'a.b.c').keys,
        ['a', 'b', 'c'])
    self.assertEqual(
        vs.Str().apply(object_utils.KeyPath.parse('a.b.c')), 'a.b.c')
    self.assertEqual(
        type_conversion.get_json_value_converter(object_utils.KeyPath)(
            object_utils.KeyPath.parse('a.b.c')), 'a.b.c')


if __name__ == '__main__':
  unittest.main()
