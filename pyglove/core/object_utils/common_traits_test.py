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

import unittest
from pyglove.core.object_utils import common_traits


class JSONConvertibleTest(unittest.TestCase):
  """Tests for JSONConvertible type registry."""

  def test_registry(self):

    class A(common_traits.JSONConvertible):

      def __init__(self, x):
        self.x = x

      @classmethod
      def from_json(cls, json_dict):
        return A(x=json_dict.pop('x'))

      def to_json(self):
        return {
            '_type': 'A',
            'x': self.x
        }

    common_traits.JSONConvertible.register('A', A)
    self.assertTrue(common_traits.JSONConvertible.is_registered('A'))
    self.assertIs(common_traits.JSONConvertible.class_from_typename('A'), A)
    self.assertIn(
        ('A', A),
        list(common_traits.JSONConvertible.registered_types()))

    class B(A):
      pass

    with self.assertRaisesRegex(
        NotImplementedError, 'Subclass should override this method'):
      _ = common_traits.JSONConvertible.from_json(1)

    with self.assertRaisesRegex(
        KeyError, 'Type .* has already been registered with class .*'):
      common_traits.JSONConvertible.register('A', B)

    common_traits.JSONConvertible.register('A', B, override_existing=True)
    self.assertIn(
        ('A', B),
        list(common_traits.JSONConvertible.registered_types()))


if __name__ == '__main__':
  unittest.main()
