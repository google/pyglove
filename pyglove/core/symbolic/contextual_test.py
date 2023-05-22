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
"""Tests for pyglove.symbolic.Contextual."""

import dataclasses
import unittest

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.contextual import Contextual


class ContextualTest(unittest.TestCase):
  """Tests for `pg.symbolic.Contextual`."""

  def test_str(self):
    self.assertEqual(str(Contextual()), 'CONTEXTUAL')
    self.assertEqual(str(Contextual(lambda k, p: 1)), 'CONTEXTUAL')

  def test_repr(self):
    self.assertEqual(repr(Contextual()), 'CONTEXTUAL')
    self.assertEqual(repr(Contextual(lambda k, p: 1)), 'CONTEXTUAL')

  def test_eq(self):
    self.assertEqual(Contextual(), Contextual())
    getter = lambda k, p: 1
    self.assertEqual(Contextual(getter), Contextual(getter))

    self.assertNotEqual(Contextual(), 1)
    self.assertNotEqual(Contextual(getter), Contextual())

  def test_value_from(self):
    @dataclasses.dataclass
    class A:
      x: int = 1
      y: int = 2

    self.assertEqual(Contextual().value_from('x', A()), 1)
    self.assertEqual(Contextual(lambda k, p: p.y).value_from('x', A()), 2)

  def test_custom_typing(self):
    v = Contextual()
    self.assertIs(pg_typing.Int().apply(v), v)
    self.assertIs(pg_typing.Str().apply(v), v)


if __name__ == '__main__':
  unittest.main()
