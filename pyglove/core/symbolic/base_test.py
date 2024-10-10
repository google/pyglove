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


class HtmlTreeViewExtensionTest(unittest.TestCase):

  def assert_content(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual.strip(), expected)

  def test_to_html(self):

    class Foo(Object):
      x: int
      y: Any = 'foo'
      z: pg_typing.Int().freeze(1)

    # Disable tooltip.
    self.assert_content(
        Foo(x=1, y='foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False
        ),
        """
        <details open class="pyglove foo"><summary><div class="summary_title">Foo(...)</div></summary><div class="complex_value foo"><table><tr><td><span class="object_key str">x</span></td><td><div><span class="simple_value int">1</span></div></td></tr><tr><td><span class="object_key str">y</span></td><td><div><span class="simple_value str">&#x27;foo&#x27;</span></div></td></tr></table></div></details>
        """
    )
    # Hide frozen and default values.
    self.assert_content(
        Foo(x=1, y='foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=0,
            hide_frozen=True,
            hide_default_values=True
        ),
        """
        <details class="pyglove foo"><summary><div class="summary_title">Foo(...)</div></summary><div class="complex_value foo"><table><tr><td><span class="object_key str">x</span></td><td><div><span class="simple_value int">1</span></div></td></tr></table></div></details>
        """
    )
    # Use inferred values.
    x = Dict(x=Dict(y=ValueFromParentChain()), y=2)
    self.assert_content(
        x.x.to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            use_inferred=False
        ),
        """
        <details open class="pyglove dict"><summary><div class="summary_title">Dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key str">y</span></td><td><div><details class="pyglove value-from-parent-chain"><summary><div class="summary_title">ValueFromParentChain(...)</div></summary><div class="complex_value value-from-parent-chain"><span class="empty_container"></span></div></details></div></td></tr></table></div></details>
        """
    )
    self.assert_content(
        x.x.to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            use_inferred=True
        ),
        """
        <details open class="pyglove dict"><summary><div class="summary_title">Dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key str">y</span></td><td><div><span class="simple_value int">2</span></div></td></tr></table></div></details>
        """
    )


if __name__ == '__main__':
  unittest.main()
