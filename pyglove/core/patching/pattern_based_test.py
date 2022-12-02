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
"""Tests for pattern-based patching."""

import unittest
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.patching import pattern_based


class PatternBasedPatchingTest(unittest.TestCase):
  """Pattern-based patching test."""

  def test_patch_on_key(self):
    d = symbolic.Dict(a=1, b={'a': 2, 'b': 1})
    pattern_based.patch_on_key(d, 'a', 3)
    self.assertEqual(d, {'a': 3, 'b': {'a': 3, 'b': 1}})

    pattern_based.patch_on_key(d, 'a', value_fn=lambda v: v + 1)
    self.assertEqual(d, {'a': 4, 'b': {'a': 4, 'b': 1}})

    with self.assertRaisesRegex(
        ValueError, 'Either `value` or `value_fn` should be specified'):
      pattern_based.patch_on_key(d, 'a', value=1, value_fn=lambda v: v + 1)

  def test_patch_on_path(self):
    d = symbolic.Dict(a=1, b={'a': 2, 'b': 1})
    pattern_based.patch_on_path(d, '.+b', 3)
    self.assertEqual(d, {'a': 1, 'b': {'a': 2, 'b': 3}})

    pattern_based.patch_on_path(d, '.*a', value_fn=lambda v: v + 1)
    self.assertEqual(d, {'a': 2, 'b': {'a': 3, 'b': 3}})

  def test_patch_on_value(self):
    d = symbolic.Dict(a=1, b={'a': 2, 'b': 1})
    pattern_based.patch_on_value(d, 1, 3)
    self.assertEqual(d, {'a': 3, 'b': {'a': 2, 'b': 3}})

    pattern_based.patch_on_value(d, 2, value_fn=lambda v: v * 2)
    self.assertEqual(d, {'a': 3, 'b': {'a': 4, 'b': 3}})

  def test_patch_on_type(self):
    d = symbolic.Dict(a='abc', b={'a': 2, 'b': 'def'})
    pattern_based.patch_on_type(d, str, 'foo')
    self.assertEqual(d, {'a': 'foo', 'b': {'a': 2, 'b': 'foo'}})

    pattern_based.patch_on_type(d, int, value_fn=lambda v: v * 2)
    self.assertEqual(d, {'a': 'foo', 'b': {'a': 4, 'b': 'foo'}})

  def test_patch_on_member(self):

    @symbolic.members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int()),
    ])
    class A(symbolic.Object):
      pass

    d = symbolic.Dict(a=A(x=1, y=2), x=1)
    pattern_based.patch_on_member(d, A, 'x', 2)
    self.assertEqual(d, {'a': A(x=2, y=2), 'x': 1})

    pattern_based.patch_on_member(d, A, 'y', value_fn=lambda v: v * 2)
    self.assertEqual(d, {'a': A(x=2, y=4), 'x': 1})


if __name__ == '__main__':
  unittest.main()
