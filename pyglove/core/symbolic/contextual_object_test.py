# Copyright 2025 The PyGlove Authors
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
import inspect
from typing import Any
import unittest
import weakref

from pyglove.core import utils as pg_utils
from pyglove.core.symbolic import contextual_object

contextual_override = pg_utils.contextual_override
ContextualOverride = pg_utils.ContextualOverride
get_contextual_override = pg_utils.get_contextual_override
contextual_value = pg_utils.contextual_value
all_contextual_values = pg_utils.all_contextual_values

ContextualObject = contextual_object.ContextualObject
contextual_attribute = contextual_object.contextual_attribute


class ContextualObjectTest(unittest.TestCase):
  """Tests for ContextualObject."""

  def test_override(self):
    class A(ContextualObject):
      x: int

    a = A(x=1)
    with a.override(x=2, y=1):
      self.assertEqual(a.x, 2)

      # `y`` is not an attribute of `A`.
      with self.assertRaises(AttributeError):
        _ = a.y

  def test_context(self):
    class A(ContextualObject):
      x: int
      y: int = contextual_attribute()
      z: int = contextual_attribute(default=-1)

    with self.assertRaisesRegex(TypeError, '.* missing 1 required argument'):
      _ = A()

    a = A(x=1)
    with self.assertRaisesRegex(
        AttributeError, 'p'
    ):
      _ = a.p

    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'
    ):
      _ = a.y

    with contextual_override(y=1):
      self.assertEqual(a.y, 1)

    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'
    ):
      _ = a.y

    # Use contextual default if it's not provided.
    self.assertEqual(a.z, -1)

    a1 = A(x=1, y=2)
    self.assertEqual(a1.x, 1)
    self.assertEqual(a1.y, 2)
    self.assertEqual(a1.z, -1)

    with contextual_override(x=3, y=3, z=3) as parent_override:
      self.assertEqual(
          parent_override,
          dict(
              x=ContextualOverride(3, cascade=False, override_attrs=False),
              y=ContextualOverride(3, cascade=False, override_attrs=False),
              z=ContextualOverride(3, cascade=False, override_attrs=False),
          ),
      )
      self.assertEqual(
          get_contextual_override('y'),
          ContextualOverride(3, cascade=False, override_attrs=False),
      )
      self.assertEqual(contextual_value('x'), 3)
      self.assertIsNone(contextual_value('f', None))
      with self.assertRaisesRegex(KeyError, '.* does not exist'):
        contextual_value('f')

      self.assertEqual(all_contextual_values(), dict(x=3, y=3, z=3))

      # Member attributes take precedence over `contextual_override`.
      self.assertEqual(a1.x, 1)
      self.assertEqual(a1.y, 2)

      # Override attributes take precedence over member attribute.
      with a1.override(y=3):
        self.assertEqual(a1.y, 3)
        with a1.override(y=4):
          self.assertEqual(a1.y, 4)
        self.assertEqual(a1.y, 3)
      self.assertEqual(a1.y, 2)

      # `contextual_override` takes precedence over contextual default.
      self.assertEqual(a1.z, 3)

      # Test nested contextual override with override_attrs=True (default).
      with contextual_override(
          y=4, z=4, override_attrs=True) as nested_override:
        self.assertEqual(
            nested_override,
            dict(
                x=ContextualOverride(3, cascade=False, override_attrs=False),
                y=ContextualOverride(4, cascade=False, override_attrs=True),
                z=ContextualOverride(4, cascade=False, override_attrs=True),
            ),
        )

        # Member attribute is not overriden as current scope does not override
        # `x``.
        self.assertEqual(a1.x, 1)

        # Member attribute is overriden.
        self.assertEqual(a1.y, 4)

        # `ContextualObject.override` takes precedence over
        # `contextual_override(override_attrs=True)`.
        with a1.override(y=3):
          self.assertEqual(a1.y, 3)
        self.assertEqual(a1.y, 4)

        # Member default is overriden.
        self.assertEqual(a1.z, 4)

      self.assertEqual(a1.y, 2)
      self.assertEqual(a1.z, 3)

    self.assertEqual(a1.y, 2)
    self.assertEqual(a1.z, -1)

  def test_context_cascade(self):
    class A(ContextualObject):
      x: int
      y: int = contextual_attribute()
      z: int = contextual_attribute(default=-1)

    a = A(1, 2)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, -1)

    with contextual_override(x=3, y=3, z=3, cascade=True):
      self.assertEqual(a.x, 1)
      self.assertEqual(a.y, 2)
      self.assertEqual(a.z, 3)

      # Outter `pg.contextual_override` takes precedence
      # over inner `pg.contextual_override` when cascade=True.
      with contextual_override(y=4, z=4, cascade=True):
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertEqual(a.z, 3)

      with contextual_override(y=4, z=4, override_attrs=True):
        self.assertEqual(a.x, 1)
        self.assertEqual(a.y, 2)
        self.assertEqual(a.z, 3)

    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, -1)

    with contextual_override(x=3, y=3, z=3, cascade=True, override_attrs=True):
      self.assertEqual(a.x, 3)
      self.assertEqual(a.y, 3)
      self.assertEqual(a.z, 3)

      with contextual_override(y=4, z=4, override_attrs=True):
        self.assertEqual(a.x, 3)
        self.assertEqual(a.y, 3)
        self.assertEqual(a.z, 3)

    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, -1)

  def test_sym_inferred(self):
    class A(ContextualObject):
      x: int = 1
      y: int = contextual_attribute()

    a = A()
    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'):
      _ = a.sym_inferred('y')
    self.assertIsNone(a.sym_inferred('y', default=None))

    with self.assertRaises(AttributeError):
      _ = a.sym_inferred('z')
    self.assertIsNone(a.sym_inferred('z', default=None))

  def test_weak_ref(self):
    class A(ContextualObject):
      x: int = 1

    a = A()
    self.assertIsNotNone(weakref.ref(a))


class ContextualAttributeTest(unittest.TestCase):
  """Tests for Component."""

  def test_contextualibute_access(self):

    class A(ContextualObject):
      x: int
      y: int = contextual_attribute()

    # Not okay: `A.x` is required.
    with self.assertRaisesRegex(TypeError, 'missing 1 required argument'):
      _ = A()

    # Okay: `A.y` is contextual.
    a = A(1)

    # `a.y` is not yet available from the context.
    with self.assertRaises(AttributeError):
      _ = a.y

    class B(ContextualObject):
      # Attributes with annotation will be treated as symbolic fields.
      p: int
      q: A = A(2)
      z: int = contextual_attribute()

    class C(ContextualObject):
      a: int
      b: B = B(2)
      y: int = 1
      z = 2

    c = C(1)
    b = c.b
    a = b.q

    # Test symbolic attributes declared from C.
    self.assertTrue(c.sym_hasattr('a'))
    self.assertTrue(c.sym_hasattr('b'))
    self.assertTrue(c.sym_hasattr('y'))
    self.assertFalse(c.sym_hasattr('z'))

    # Contextual access to c.y from a.
    self.assertEqual(a.y, 1)
    self.assertEqual(b.z, 2)

    # 'y' is not defined as an attribute in 'B'.
    with self.assertRaises(AttributeError):
      _ = b.y

    c.rebind(y=2)
    self.assertEqual(c.y, 2)
    self.assertEqual(a.y, 2)

    c.z = 3
    self.assertEqual(c.z, 3)
    self.assertEqual(b.z, 3)

  def test_to_html(self):
    class A(ContextualObject):
      x: int = 1
      y: int = contextual_attribute()

    def assert_content(html, expected):
      expected = inspect.cleandoc(expected).strip()
      actual = html.content.strip()
      if actual != expected:
        print(actual)
      self.assertEqual(actual.strip(), expected)

    self.assertIn(
        inspect.cleandoc(
            """
            .contextual-attribute {
              color: purple;
            }
            .unavailable-contextual {
              color: gray;
              font-style: italic;
            }
            """
        ),
        A().to_html().style_section,
    )

    assert_content(
        A().to_html(enable_summary_tooltip=False),
        """
        <details open class="pyglove a"><summary><div class="summary-title">A(...)</div></summary><div class="complex-value a"><details open class="pyglove int"><summary><div class="summary-name">x<span class="tooltip">x</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove contextual-attribute"><summary><div class="summary-name">y<span class="tooltip">y</span></div><div class="summary-title">ContextualAttribute(...)</div></summary><div class="unavailable-contextual">(not available)</div></details></div></details>
        """
    )

    class B(ContextualObject):
      z: Any
      y: int = 2

    b = B(A())
    assert_content(
        b.z.to_html(enable_summary_tooltip=False),
        """
        <details open class="pyglove a"><summary><div class="summary-title">A(...)</div></summary><div class="complex-value a"><details open class="pyglove int"><summary><div class="summary-name">x<span class="tooltip">x</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove contextual-attribute"><summary><div class="summary-name">y<span class="tooltip">y</span></div><div class="summary-title">ContextualAttribute(...)</div></summary><span class="simple-value int">2</span></details></div></details>
        """
    )


if __name__ == '__main__':
  unittest.main()
