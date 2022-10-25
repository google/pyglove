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
"""Tests for pyglove.core.typing.key_specs."""

import unittest
from pyglove.core.typing import key_specs as ks


class ConstStrKeyTest(unittest.TestCase):
  """Tests for `ConstStrKey`."""

  def test_basics(self):
    key = ks.ConstStrKey('a')
    self.assertEqual(key, key)
    self.assertEqual(key, 'a')
    self.assertEqual(key.text, 'a')
    self.assertNotEqual(key, 'b')
    self.assertIn(key, {'a': 1})
    self.assertEqual(str(key), 'a')
    self.assertEqual(repr(key), 'a')
    self.assertTrue(key.match('a'))
    self.assertFalse(key.match('b'))

  def test_extend(self):
    self.assertEqual(ks.ConstStrKey('a').extend(ks.ConstStrKey('a')).text, 'a')
    with self.assertRaisesRegex(KeyError,
                                '.* cannot extend .* for keys are different.'):
      ks.ConstStrKey('a').extend(ks.ConstStrKey('b'))

  def test_bad_cases(self):
    with self.assertRaisesRegex(KeyError, '\'.\' cannot be used in key.'):
      ks.ConstStrKey('a.b')


class StrKeyTest(unittest.TestCase):
  """Tests for `StrKey`."""

  def test_basics(self):
    key = ks.StrKey()
    self.assertEqual(key, key)
    self.assertEqual(key, ks.StrKey())
    self.assertTrue(key.match('a'))
    self.assertTrue(key.match('abc'))
    self.assertFalse(key.match(1))

  def test_match_with_regex(self):
    key = ks.StrKey('a.*')
    self.assertTrue(key.match('a1'))
    self.assertTrue(key.match('a'))
    self.assertFalse(key.match('b'))
    self.assertFalse(key.match({}))

  def test_extend(self):
    self.assertIsNone(ks.StrKey().extend(ks.StrKey()).regex)
    self.assertEqual(
        ks.StrKey('a.*').extend(ks.StrKey('a.*')).regex.pattern, 'a.*')

    with self.assertRaisesRegex(
        KeyError, '.* cannot extend .* for keys are different.'):
      ks.StrKey('a.*').extend(ks.StrKey(regex='.*'))


class ListKeyTest(unittest.TestCase):
  """Tests for `ListKey`."""

  def test_basics(self):
    self.assertEqual(ks.ListKey(), ks.ListKey())
    self.assertEqual(ks.ListKey(), ks.ListKey(min_value=0))
    self.assertEqual(
        ks.ListKey(min_value=0, max_value=10),
        ks.ListKey(max_value=10))
    self.assertEqual(
        ks.ListKey(max_value=10), ks.ListKey(max_value=10))
    self.assertNotEqual(
        ks.ListKey(min_value=10), ks.ListKey(min_value=5))
    self.assertNotEqual(
        ks.ListKey(max_value=10), ks.ListKey(max_value=5))
    self.assertNotEqual(
        ks.ListKey(min_value=5), ks.ListKey(max_value=5))

  def test_match_with_unbounded_key(self):
    key = ks.ListKey()
    self.assertEqual(key, key)
    self.assertEqual(key.min_value, 0)
    self.assertIsNone(key.max_value)

    self.assertTrue(key.match(1))
    self.assertTrue(key.match(10000))
    self.assertFalse(key.match('a'))

  def test_match_with_bounded_key(self):
    key = ks.ListKey(min_value=2, max_value=10)
    self.assertTrue(key.match(2))
    self.assertFalse(key.match(0))
    self.assertFalse(key.match(10000))
    self.assertFalse(key.match('a'))

  def test_extend(self):
    key = ks.ListKey()
    self.assertIsNone(key.extend(ks.ListKey()).max_value)
    self.assertEqual(key.extend(ks.ListKey(max_value=10)).max_value, 10)
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      key.extend(ks.TupleKey(1))

    key = ks.ListKey(min_value=2, max_value=10)
    self.assertEqual(key.extend(ks.ListKey()).min_value, 2)
    self.assertEqual(key.extend(ks.ListKey()).max_value, 10)

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      key.extend(ks.StrKey())

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: min_value is smaller.'):
      key.extend(ks.ListKey(min_value=3))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: max_value is greater.'):
      key.extend(ks.ListKey(max_value=5))


class TupleKeyTest(unittest.TestCase):
  """Tests for `TupleKey`."""

  def test_basics(self):
    key = ks.TupleKey(0)
    self.assertEqual(key, key)
    self.assertEqual(ks.TupleKey(0), ks.TupleKey(0))
    self.assertNotEqual(ks.TupleKey(0), ks.TupleKey(1))

  def test_match_with_bounded_key(self):
    key = ks.TupleKey(0)
    self.assertTrue(key.match(0))
    self.assertFalse(key.match(1))
    self.assertFalse(key.match('a'))

  def test_match_with_unbounded_key(self):
    key = ks.TupleKey()
    self.assertTrue(key.match(0))
    self.assertTrue(key.match(1))
    self.assertFalse(key.match('a'))

  def test_extend(self):
    self.assertEqual(ks.TupleKey().extend(ks.TupleKey(1)).index, 1)
    self.assertEqual(ks.TupleKey(0).extend(ks.TupleKey(0)).index, 0)

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      ks.TupleKey().extend(ks.ListKey(10))

    with self.assertRaisesRegex(
        KeyError, '.* cannot extend .*: unmatched index.'):
      ks.TupleKey(2).extend(ks.TupleKey(1))


if __name__ == '__main__':
  unittest.main()
