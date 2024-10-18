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
from pyglove.core import views
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
        <details open class="pyglove foo"><summary><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">y</div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;foo&#x27;</span></details></div></details>
        """
    )
    # Hide frozen and default values.
    self.assert_content(
        Foo(x=1, y='foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=0,
            key_style='label',
            extra_flags=dict(
                hide_frozen=True,
                hide_default_values=True
            )
        ),
        """
        <details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><table><tr><td><span class="object-key str">x</span></td><td><span class="simple-value int">1</span></td></tr></table></div></details>
        """
    )
    self.assert_content(
        Foo(x=1, y='foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=0,
            extra_flags=dict(
                hide_frozen=True,
                hide_default_values=True
            )
        ),
        """
        <details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details></div></details>
        """
    )
    # Use inferred values.
    x = Dict(x=Dict(y=ValueFromParentChain()), y=2)
    self.assert_content(
        x.x.to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            key_style='label',
            extra_flags=dict(
                use_inferred=False
            )
        ),
        """
        <details open class="pyglove dict"><summary><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">y</span></td><td><details class="pyglove value-from-parent-chain"><summary><div class="summary-title">ValueFromParentChain(...)</div></summary><div class="complex-value value-from-parent-chain"><span class="empty-container"></span></div></details></td></tr></table></div></details>
        """
    )
    self.assert_content(
        x.x.to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            key_style='label',
            extra_flags=dict(
                use_inferred=True
            )
        ),
        """
        <details open class="pyglove dict"><summary><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">y</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details>
        """
    )
    # Test collapse level.
    v = Foo(1, Foo(2, Foo(3, Foo(4))))
    with views.view_options(key_style='label'):
      self.assertEqual(
          v.to_html(collapse_level=0).content.count('open'), 0
      )
      self.assertEqual(
          v.to_html(collapse_level=1).content.count('open'), 1
      )
      self.assertEqual(
          v.to_html(collapse_level=2).content.count('open'), 2
      )
      self.assertEqual(
          v.to_html(collapse_level=None).content.count('open'), 4
      )


if __name__ == '__main__':
  unittest.main()
