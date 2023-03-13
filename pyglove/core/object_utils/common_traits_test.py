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
"""Tests for pyglove.object_utils.common_traits."""

import abc
import unittest
from pyglove.core.object_utils import common_traits


class JSONConvertibleTest(unittest.TestCase):
  """Tests for JSONConvertible type registry."""

  def test_registry(self):

    class A(common_traits.JSONConvertible):

      @abc.abstractmethod
      def value(self):
        pass

    class B(A):

      def __init__(self, x):
        super().__init__()
        self.x = x

      def to_json(self):
        return A.to_json_dict({
            'x': self.x
        })

      def value(self):
        return self.x

    typename = lambda cls: f'{cls.__module__}.{cls.__name__}'

    # A is abstract.
    self.assertFalse(common_traits.JSONConvertible.is_registered(typename(A)))
    self.assertTrue(common_traits.JSONConvertible.is_registered(typename(B)))
    self.assertIs(
        common_traits.JSONConvertible.class_from_typename(typename(B)), B)
    self.assertIn(
        (typename(B), B),
        list(common_traits.JSONConvertible.registered_types()))

    class C(B):
      auto_register = False

    # Auto-register is off.
    self.assertFalse(common_traits.JSONConvertible.is_registered(typename(C)))

    with self.assertRaisesRegex(
        KeyError, 'Type .* has already been registered with class .*'):
      common_traits.JSONConvertible.register(typename(B), C)

    common_traits.JSONConvertible.register(
        typename(B), C, override_existing=True)
    self.assertIn(
        (typename(B), C),
        list(common_traits.JSONConvertible.registered_types()))

  def test_json_conversion(self):

    class X(common_traits.JSONConvertible):

      def __init__(self, x):
        self.x = x

      def to_json(self):
        return X.to_json_dict(dict(x=self.x))

      def __eq__(self, other):
        return isinstance(other, X) and self.x == other.x

      def __ne__(self, other):
        return not self.__eq__(other)

    typename = lambda cls: f'{cls.__module__}.{cls.__name__}'
    json_value = common_traits.to_json([(X(1), 2), {'y': X(3)}])
    self.assertEqual(json_value, [
        ['__tuple__', {'_type': typename(X), 'x': 1}, 2],
        {'y': {'_type': typename(X), 'x': 3}}
    ])
    self.assertEqual(common_traits.from_json(json_value),
                     [(X(1), 2), {'y': X(3)}])

    # Test bad cases.
    with self.assertRaisesRegex(
        ValueError, 'Tuple should have at least one element besides .*'):
      common_traits.from_json(['__tuple__'])

    with self.assertRaisesRegex(
        TypeError, 'Type name .* is not registered'):
      common_traits.from_json({'_type': '__main__.ABC'})


if __name__ == '__main__':
  unittest.main()
