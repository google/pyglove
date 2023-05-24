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
"""Tests for pyglove.symbolic.base."""

import copy
import dataclasses
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic.dict import Dict


class FieldUpdateTest(unittest.TestCase):
  """Tests for `pg.symbolic.FieldUpdate`."""

  def test_basics(self):
    x = Dict(x=1)
    f = pg_typing.Field('x', pg_typing.Int())
    update = base.FieldUpdate(object_utils.KeyPath('x'), x, f, 1, 2)
    self.assertEqual(update.path, 'x')
    self.assertIs(update.target, x)
    self.assertIs(update.field, f)
    self.assertEqual(update.old_value, 1)
    self.assertEqual(update.new_value, 2)

  def test_format(self):
    self.assertEqual(
        base.FieldUpdate(
            object_utils.KeyPath('x'), Dict(x=1), None, 1, 2
        ).format(compact=True),
        'FieldUpdate(parent_path=, path=x, old_value=1, new_value=2)',
    )

    self.assertEqual(
        base.FieldUpdate(
            object_utils.KeyPath('a'), Dict(x=Dict(a=1)).x, None, 1, 2
        ).format(compact=True),
        'FieldUpdate(parent_path=x, path=a, old_value=1, new_value=2)',
    )

  def test_eq_ne(self):
    x = Dict()
    f = pg_typing.Field('x', pg_typing.Int())
    self.assertEqual(
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2),
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2),
    )

    # Targets are not the same instance.
    self.assertNotEqual(
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2),
        base.FieldUpdate(object_utils.KeyPath('a'), Dict(), f, 1, 2),
    )

    # Fields are not the same instance.
    self.assertNotEqual(
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2),
        base.FieldUpdate(object_utils.KeyPath('b'), x, copy.copy(f), 1, 2),
    )

    self.assertNotEqual(
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2),
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 0, 2),
    )

    self.assertNotEqual(
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2),
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 1),
    )

    self.assertNotEqual(
        base.FieldUpdate(object_utils.KeyPath('a'), x, f, 1, 2), Dict()
    )


class ContextualValueTest(unittest.TestCase):
  """Tests for `pg.symbolic.ContextualValue`."""

  def test_str(self):
    self.assertEqual(str(base.ContextualValue()), 'ContextualValue()')

  def test_repr(self):
    self.assertEqual(repr(base.ContextualValue()), 'ContextualValue()')

  def test_eq(self):
    self.assertEqual(base.ContextualValue(), base.ContextualValue())
    self.assertNotEqual(base.ContextualValue(), 1)

  def test_call(self):
    @dataclasses.dataclass
    class A:
      x: int = 1
      y: int = 2

    self.assertEqual(
        base.ContextualValue().get(base.GetAttributeContext('x', A(), Dict())),
        1,
    )
    self.assertEqual(
        base.ContextualValue().get(base.GetAttributeContext(0, [0, 1], Dict())),
        0,
    )
    self.assertEqual(
        base.ContextualValue().get(base.GetAttributeContext(0, Dict(), Dict())),
        pg_typing.MISSING_VALUE,
    )

  def test_custom_typing(self):
    v = base.ContextualValue()
    self.assertIs(pg_typing.Int().apply(v), v)
    self.assertIs(pg_typing.Str().apply(v), v)

  def test_to_json(self):
    self.assertEqual(
        base.to_json(base.ContextualValue()),
        {'_type': f'{base.ContextualValue.__module__}.ContextualValue'},
    )

  def test_from_json(self):
    self.assertEqual(
        base.from_json(base.ContextualValue().to_json()), base.ContextualValue()
    )


if __name__ == '__main__':
  unittest.main()
