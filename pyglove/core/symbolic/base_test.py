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
import inspect
from typing import Any
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.inferred import ValueFromParentChain
from pyglove.core.symbolic.object import Object


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


class HtmlFormattableTest(unittest.TestCase):

  def assert_html(self, actual, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = actual.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual.strip(), expected)

  def test_to_html(self):

    class Foo(Object):
      x: int
      y: Any = 'foo'
      z: pg_typing.Int().freeze(1)

    # Disable tooltip.
    self.assert_html(
        Foo(x=1, y='foo').to_html(enable_tooltip=False).body_content,
        """
        <details class="pyglove Foo" open>
        <summary>
        <div class="summary_title t_Foo">Foo(...)</div>

        </summary>
        <div><table><tr><td><span class="object_key k_str v_int">x</span>
        </td><td><span class="simple_value v_int">1</span>
        </td></tr><tr><td><span class="object_key k_str v_str">y</span>
        </td><td><span class="simple_value v_str">&#x27;foo&#x27;</span>
        </td></tr></table></div>
        </details>
        """
    )
    # Hide frozen and default values.
    self.assert_html(
        Foo(x=1, y='foo').to_html(
            enable_tooltip=False,
            collapse_level=0,
            hide_frozen=True,
            hide_default_values=True
        ).body_content,
        """
        <details class="pyglove Foo">
        <summary>
        <div class="summary_title t_Foo">Foo(...)</div>

        </summary>
        <div><table><tr><td><span class="object_key k_str v_int">x</span>
        </td><td><span class="simple_value v_int">1</span>
        </td></tr></table></div>
        </details>
        """
    )
    # Use inferred values.
    x = Dict(x=Dict(y=ValueFromParentChain()), y=2)
    self.assert_html(
        x.x.to_html(enable_tooltip=False, use_inferred=False).body_content,
        """
        <details class="pyglove Dict" open>
        <summary>
        <div class="summary_title t_Dict">Dict(...)</div>

        </summary>
        <div><table><tr><td><span class="object_key k_str v_ValueFromParentChain">y</span>
        </td><td><details class="pyglove ValueFromParentChain">
        <summary>
        <div class="summary_title t_ValueFromParentChain">ValueFromParentChain(...)</div>

        </summary>
        <span class="empty_container"></span>
        </details>
        </td></tr></table></div>
        </details>
        """
    )
    self.assert_html(
        x.x.to_html(enable_tooltip=False, use_inferred=True).body_content,
        """
        <details class="pyglove Dict" open>
        <summary>
        <div class="summary_title t_Dict">Dict(...)</div>

        </summary>
        <div><table><tr><td><span class="object_key k_str v_int">y</span>
        </td><td><span class="simple_value v_int">2</span>
        </td></tr></table></div>
        </details>
        """
    )


if __name__ == '__main__':
  unittest.main()
