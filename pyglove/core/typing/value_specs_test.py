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
"""Tests for pyglove.core.typing.value_specs."""

import contextlib
import sys
import typing
import unittest

from pyglove.core import object_utils
from pyglove.core.typing import annotation_conversion   # pylint: disable=unused-import
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import typed_missing
from pyglove.core.typing import value_specs as vs


class BoolTest(unittest.TestCase):
  """Tests for `Bool`."""

  def test_value_type(self):
    self.assertEqual(vs.Bool().value_type, bool)

  def test_forward_refs(self):
    self.assertEqual(vs.Bool().forward_refs, set())

  def test_type_resolved(self):
    self.assertTrue(vs.Bool().type_resolved)

  def test_default(self):
    self.assertEqual(vs.Bool().default, typed_missing.MISSING_VALUE)
    self.assertEqual(vs.Bool(True).default, True)

  def test_noneable(self):
    self.assertFalse(vs.Bool().is_noneable)
    self.assertTrue(vs.Bool().noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Bool()), 'Bool()')
    self.assertEqual(str(vs.Bool(True)), 'Bool(default=True)')
    self.assertEqual(str(vs.Bool(True).freeze()),
                     'Bool(default=True, frozen=True)')
    self.assertEqual(
        str(vs.Bool().noneable()), 'Bool(default=None, noneable=True)')
    self.assertEqual(
        str(vs.Bool(True).noneable()), 'Bool(default=True, noneable=True)')

  def test_annotation(self):
    self.assertEqual(vs.Bool().annotation, bool)
    self.assertEqual(vs.Bool().noneable().annotation, typing.Optional[bool])

  def test_eq(self):
    v = vs.Bool()
    self.assertEqual(v, v)

    self.assertEqual(vs.Bool(), vs.Bool())
    self.assertEqual(vs.Bool(True), vs.Bool(True))
    self.assertEqual(vs.Bool().noneable(), vs.Bool().noneable())
    self.assertNotEqual(vs.Bool(True), vs.Int())
    self.assertNotEqual(vs.Bool(True), vs.Bool())
    self.assertNotEqual(vs.Bool().noneable(), vs.Bool())

  def test_apply(self):
    self.assertTrue(vs.Bool().apply(True))
    self.assertIsNone(vs.Bool().noneable().apply(None))
    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'bool\'> but encountered '
        '<(type|class) \'int\'>.'):
      vs.Bool().apply(1)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Bool().apply(None)

  def test_is_compatible(self):
    v = vs.Bool()
    self.assertTrue(v.is_compatible(v))
    self.assertTrue(vs.Bool().is_compatible(vs.Bool()))
    self.assertTrue(vs.Bool().noneable().is_compatible(vs.Bool()))
    self.assertFalse(vs.Bool().is_compatible(vs.Bool().noneable()))
    self.assertFalse(vs.Bool().is_compatible(vs.Int()))

  def test_extend(self):
    # Child may change default value.
    self.assertEqual(vs.Bool(False).extend(vs.Bool(True)).default, False)

    # Child may make a parent default value not specified.
    self.assertTrue(vs.Bool().extend(vs.Bool(True)).default)

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(vs.Bool().extend(vs.Bool().noneable()).is_noneable)

    # Child cannot extend a base with different type.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Bool().extend(vs.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Bool().noneable().extend(vs.Bool())

  def test_freeze(self):
    self.assertFalse(vs.Bool().frozen)

    v = vs.Bool().freeze(True)
    self.assertTrue(v.frozen)
    self.assertTrue(v.default)
    self.assertTrue(v.apply(True))
    self.assertTrue(v.apply(typed_missing.MISSING_VALUE))
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      v.apply(False)

    v = vs.Bool(default=True).freeze()
    self.assertTrue(v.frozen)
    self.assertTrue(v.default)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Bool().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Bool().freeze()


class StrTest(unittest.TestCase):
  """Tests for `Str`."""

  def test_value_type(self):
    self.assertEqual(vs.Str().value_type, str)

  def test_forward_refs(self):
    self.assertEqual(vs.Str().forward_refs, set())

  def test_type_resolved(self):
    self.assertTrue(vs.Str().type_resolved)

  def test_default(self):
    self.assertEqual(vs.Str().default, typed_missing.MISSING_VALUE)
    self.assertEqual(vs.Str('abc').default, 'abc')

  def test_noneable(self):
    self.assertFalse(vs.Str().is_noneable)
    self.assertTrue(vs.Str().noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Str()), 'Str()')
    self.assertEqual(
        str(vs.Str().noneable()), 'Str(default=None, noneable=True)')
    self.assertEqual(str(vs.Str('a')), 'Str(default=\'a\')')
    self.assertEqual(str(vs.Str('a').freeze()),
                     'Str(default=\'a\', frozen=True)')
    self.assertEqual(str(vs.Str(regex='.*')), 'Str(regex=\'.*\')')

  def test_annotation(self):
    self.assertEqual(vs.Str().annotation, str)
    self.assertEqual(vs.Str().noneable().annotation, typing.Optional[str])

  def test_eq(self):
    v = vs.Str()
    self.assertEqual(v, v)
    self.assertEqual(vs.Str(), vs.Str())
    self.assertEqual(vs.Str().noneable(), vs.Str().noneable())
    self.assertEqual(vs.Str('a'), vs.Str('a'))
    self.assertEqual(vs.Str('a', '.*'), vs.Str('a', '.*'))
    self.assertEqual(vs.Str(regex='a.*'), vs.Str(regex='a.*'))
    self.assertNotEqual(vs.Str(), vs.Bool())
    self.assertNotEqual(vs.Str(), vs.Str().noneable())
    self.assertNotEqual(vs.Str('a'), vs.Str())
    self.assertNotEqual(vs.Str('a'), vs.Str('b'))
    self.assertNotEqual(vs.Str(), vs.Str(regex='.*'))
    self.assertNotEqual(vs.Str(regex='a'), vs.Str(regex='.*'))

  def test_apply(self):
    self.assertEqual(vs.Str().apply('a'), 'a')
    self.assertEqual(vs.Str(regex='a.*').apply('a1'), 'a1')
    with self.assertRaisesRegex(
        TypeError, 'Expect .*str.* but encountered <(type|class) \'dict\'>.'):
      vs.Str().apply({})

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Str().apply(None)

    with self.assertRaisesRegex(
        ValueError, 'String \'b\' does not match regular expression \'a.*\'.'):
      vs.Str(regex='a.*').apply('b')

  def test_is_compatible(self):
    self.assertTrue(vs.Str().is_compatible(vs.Str()))
    self.assertTrue(vs.Str().noneable().is_compatible(vs.Str()))
    self.assertTrue(vs.Str(regex='.*').is_compatible(vs.Str(regex='.*')))

    # This is a false-positive, but we don't have a good way to check the
    # compatibility of two regular expressions.
    self.assertTrue(vs.Str(regex='abc.*').is_compatible(vs.Str(regex='xyz.*')))
    self.assertFalse(vs.Str().is_compatible(vs.Str().noneable()))
    self.assertFalse(vs.Str().is_compatible(vs.Int()))

  def test_extend(self):
    # Child may make a parent default value not specified.
    self.assertEqual(
        vs.Str().extend(vs.Str('foo')).default,
        typed_missing.MISSING_VALUE)

    # Child without regular expression remain unchanged.
    self.assertEqual(
        vs.Str(regex='a.*').extend(vs.Str(regex='.*')).regex.pattern, 'a.*')

    # Child with regular expression remain unchanged.
    self.assertEqual(
        vs.Str(regex='a.*').extend(vs.Str(regex='.*')).regex.pattern, 'a.*')

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(vs.Str().extend(vs.Str().noneable()).is_noneable)

    # Child cannot extend a base of different type.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Str().extend(vs.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Str().noneable().extend(vs.Str())

  def test_freeze(self):
    self.assertFalse(vs.Str().frozen)

    v = vs.Str().freeze('foo')
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 'foo')
    self.assertEqual(v.apply('foo'), 'foo')
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), 'foo')
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      v.apply('bar')

    v = vs.Str(default='foo').freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 'foo')

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Str().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Str().freeze()


class IntTest(unittest.TestCase):
  """Tests for `Int`."""

  def test_value_type(self):
    self.assertEqual(vs.Int().value_type, int)

  def test_forward_refs(self):
    self.assertEqual(vs.Int().forward_refs, set())

  def test_type_resolved(self):
    self.assertTrue(vs.Int().type_resolved)

  def test_default(self):
    self.assertEqual(vs.Int().default, typed_missing.MISSING_VALUE)
    self.assertEqual(vs.Int(1).default, 1)

  def test_noneable(self):
    self.assertFalse(vs.Int().is_noneable)
    self.assertTrue(vs.Int().noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Int()), 'Int()')
    self.assertEqual(str(vs.Int(1)), 'Int(default=1)')
    self.assertEqual(str(vs.Int(1).freeze()),
                     'Int(default=1, frozen=True)')
    self.assertEqual(
        str(vs.Int().noneable()), 'Int(default=None, noneable=True)')
    self.assertEqual(
        str(vs.Int(1).noneable()), 'Int(default=1, noneable=True)')
    self.assertEqual(
        str(vs.Int(min_value=0, max_value=1)), 'Int(min=0, max=1)')

  def test_annotation(self):
    self.assertEqual(vs.Int().annotation, int)
    self.assertEqual(vs.Int().noneable().annotation, typing.Optional[int])

  def test_eq(self):
    v = vs.Int()
    self.assertEqual(v, v)
    self.assertEqual(vs.Int(), vs.Int())
    self.assertEqual(vs.Int().noneable(), vs.Int().noneable())
    self.assertEqual(vs.Int(1), vs.Int(1))
    self.assertEqual(vs.Int(min_value=0, max_value=1),
                     vs.Int(min_value=0, max_value=1))
    self.assertNotEqual(vs.Int(), vs.Bool())
    self.assertNotEqual(vs.Int(), vs.Int().noneable())
    self.assertNotEqual(vs.Int(1), vs.Int())
    self.assertNotEqual(vs.Int(1), vs.Int(2))
    self.assertNotEqual(vs.Int(1, min_value=1), vs.Int(1))
    self.assertNotEqual(vs.Int(1, max_value=1), vs.Int(1))
    self.assertNotEqual(vs.Int(min_value=0), vs.Int())
    self.assertNotEqual(vs.Int(max_value=0), vs.Int())
    self.assertNotEqual(vs.Int(min_value=0), vs.Int(min_value=1))
    self.assertNotEqual(vs.Int(max_value=0), vs.Int(max_value=1))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError, '"max_value" must be equal or greater than "min_value".'):
      vs.Int(min_value=1, max_value=0)

  def test_apply(self):
    self.assertEqual(vs.Int().apply(1), 1)
    self.assertEqual(vs.Int(min_value=1, max_value=1).apply(1), 1)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Int().apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'int\'> but encountered '
        '<(type|class) \'float\'>.'):
      vs.Int().apply(1.0)

    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range \\(min=0, max=None\\)'):
      vs.Int(min_value=0).apply(-1)

    with self.assertRaisesRegex(
        ValueError, 'Value 1 is out of range \\(min=None, max=0\\)'):
      vs.Int(max_value=0).apply(1)

  def test_is_compatible(self):
    self.assertTrue(vs.Int().is_compatible(vs.Int()))
    self.assertTrue(vs.Int().noneable().is_compatible(vs.Int()))
    self.assertTrue(vs.Int().is_compatible(vs.Int(min_value=1)))
    self.assertTrue(vs.Int().is_compatible(vs.Int(max_value=1)))
    self.assertTrue(
        vs.Int(min_value=1, max_value=10).is_compatible(
            vs.Int(min_value=2, max_value=10)))
    self.assertFalse(vs.Int().is_compatible(vs.Int().noneable()))
    self.assertFalse(vs.Int().is_compatible(vs.Bool()))
    self.assertFalse(vs.Int(min_value=1).is_compatible(vs.Int()))
    self.assertFalse(
        vs.Int(min_value=2, max_value=5).is_compatible(
            vs.Int(min_value=2, max_value=10)))

  def test_extend(self):
    # Child without constraints will inheirt constraints.
    self.assertEqual(vs.Int().extend(vs.Int(min_value=0, max_value=1)),
                     vs.Int(min_value=0, max_value=1))

    # Child extends base with constraints will intersect valid range.
    self.assertEqual(
        vs.Int(min_value=2, max_value=5).extend(
            vs.Int(min_value=2, max_value=6)
        ),
        vs.Int(min_value=2, max_value=5),
    )

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(vs.Int().extend(vs.Int().noneable()).is_noneable)

    # Child may extend a union that has the same type.
    self.assertEqual(
        vs.Int().extend(vs.Union([vs.Int(min_value=1), vs.Bool()])),
        vs.Int(min_value=1),
    )

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      vs.Int().extend(vs.Bool())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Int().noneable().extend(vs.Int())

    # Child with wider range cannot extend a base with narrower range.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: min_value is smaller'):
      vs.Int(min_value=0).extend(vs.Int(min_value=1))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is larger'):
      vs.Int(max_value=1).extend(vs.Int(max_value=0))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      vs.Int(max_value=1).extend(vs.Int(min_value=5))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      vs.Int(min_value=1).extend(vs.Int(max_value=0))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: no compatible type found in Union.'):
      vs.Int().extend(vs.Union([vs.Bool(), vs.Str()]))

  def test_freeze(self):
    self.assertFalse(vs.Int().frozen)

    v = vs.Int().freeze(1)
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 1)
    self.assertEqual(v.apply(1), 1)
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), 1)
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      v.apply(2)

    v = vs.Int(default=1).freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 1)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Int().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Int().freeze()


class FloatTest(unittest.TestCase):
  """Tests for `Float`."""

  def test_value_type(self):
    self.assertEqual(vs.Float().value_type, float)

  def test_forward_refs(self):
    self.assertEqual(vs.Float().forward_refs, set())

  def test_type_resolved(self):
    self.assertTrue(vs.Float().type_resolved)

  def test_default(self):
    self.assertEqual(vs.Float().default, typed_missing.MISSING_VALUE)
    self.assertEqual(vs.Float(1.0).default, 1.0)

  def test_noneable(self):
    self.assertFalse(vs.Float().is_noneable)
    self.assertTrue(vs.Float().noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Float()), 'Float()')
    self.assertEqual(
        str(vs.Float().noneable()), 'Float(default=None, noneable=True)')
    self.assertEqual(
        str(vs.Float(1.0).freeze()), 'Float(default=1.0, frozen=True)')
    self.assertEqual(
        str(vs.Float(1.0).noneable()), 'Float(default=1.0, noneable=True)')
    self.assertEqual(
        str(vs.Float(default=1., min_value=0., max_value=1.).noneable()),
        'Float(default=1.0, min=0.0, max=1.0, noneable=True)')

  def test_annotation(self):
    self.assertEqual(vs.Float().annotation, float)
    self.assertEqual(vs.Float().noneable().annotation,
                     typing.Optional[float])

  def test_eq(self):
    f = vs.Float()
    self.assertEqual(f, f)
    self.assertEqual(vs.Float(), vs.Float())
    self.assertEqual(vs.Float().noneable(), vs.Float().noneable())
    self.assertEqual(vs.Float(1.), vs.Float(1.))
    self.assertEqual(
        vs.Float(min_value=0., max_value=1.),
        vs.Float(min_value=0., max_value=1.))
    self.assertNotEqual(vs.Float(), vs.Int())
    self.assertNotEqual(vs.Float(), vs.Float().noneable())
    self.assertNotEqual(vs.Float(1.), vs.Float())
    self.assertNotEqual(vs.Float(1.), vs.Float(2.))
    self.assertNotEqual(vs.Float(1., min_value=1.), vs.Float(1.))
    self.assertNotEqual(vs.Float(1., max_value=1.), vs.Float(1.))
    self.assertNotEqual(vs.Float(min_value=0.), vs.Float())
    self.assertNotEqual(vs.Float(max_value=0.), vs.Float())
    self.assertNotEqual(vs.Float(min_value=0.), vs.Float(min_value=1.))
    self.assertNotEqual(vs.Float(max_value=0.), vs.Float(max_value=1.))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError, '"max_value" must be equal or greater than "min_value".'):
      vs.Float(min_value=1., max_value=0.)

  def test_apply(self):
    self.assertEqual(vs.Float().apply(1.), 1.)
    self.assertEqual(vs.Float(min_value=1., max_value=1.).apply(1.), 1.)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Float().apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'float\'> but encountered '
        '<(type|class) \'int\'>.'):
      vs.Float().apply(1)

    with self.assertRaisesRegex(
        ValueError, 'Value -1.0 is out of range \\(min=0.0, max=None\\).'):
      vs.Float(min_value=0.).apply(-1.)

    with self.assertRaisesRegex(
        ValueError, 'Value 1.0 is out of range \\(min=None, max=0.0\\).'):
      vs.Float(max_value=0.).apply(1.)

  def test_is_compatible(self):
    self.assertTrue(vs.Float().is_compatible(vs.Float()))
    self.assertTrue(vs.Float().noneable().is_compatible(vs.Float()))
    self.assertTrue(vs.Float().is_compatible(vs.Float(min_value=1.)))
    self.assertTrue(vs.Float().is_compatible(vs.Float(max_value=1.)))
    self.assertTrue(
        vs.Float(min_value=1., max_value=10.).is_compatible(
            vs.Float(min_value=2., max_value=10.)))
    self.assertFalse(vs.Float().is_compatible(vs.Float().noneable()))
    self.assertFalse(vs.Float().is_compatible(vs.Bool()))
    self.assertFalse(vs.Float(min_value=1).is_compatible(vs.Float()))
    self.assertFalse(
        vs.Float(min_value=2, max_value=5).is_compatible(
            vs.Float(min_value=2, max_value=10)))

  def test_extend(self):
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        vs.Float().extend(vs.Float(min_value=0., max_value=1.)),
        vs.Float(min_value=0., max_value=1.))

    # Child extends base with constraints will intersect valid range.
    self.assertEqual(
        vs.Float(min_value=3.).extend(
            vs.Float(min_value=2., max_value=6.)),
        vs.Float(min_value=3., max_value=6.))

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(vs.Float().extend(vs.Float().noneable()).is_noneable)

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Float().extend(vs.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Float().noneable().extend(vs.Float())

    # Child with wider range cannot extend a base with narrower range.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: min_value is smaller'):
      vs.Float(min_value=0.).extend(vs.Float(min_value=1.))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: max_value is larger'):
      vs.Float(max_value=1.).extend(vs.Float(max_value=0.))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      vs.Float(max_value=1.).extend(vs.Float(min_value=5.))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      vs.Float(min_value=1.).extend(vs.Float(max_value=0.))

  def test_freeze(self):
    self.assertFalse(vs.Float().frozen)

    v = vs.Float().freeze(1.0)
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 1.0)
    self.assertEqual(v.apply(1.0), 1.0)
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), 1.0)
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      v.apply(2.0)

    v = vs.Float(default=1.0).freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 1.0)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Float().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Float().freeze()


class EnumTest(unittest.TestCase):
  """Tests for `Enum`."""

  def test_value_type(self):
    self.assertEqual(vs.Enum('a', ['a', None]).value_type, str)
    self.assertEqual(vs.Enum(1, [1, None]).value_type, int)

    class A:
      pass

    class B(A):
      pass

    class C:
      pass

    a = A()
    b1 = B()
    b2 = B()
    c = C()
    self.assertEqual(vs.Enum(a, [a, b1, None]).value_type, A)
    self.assertEqual(vs.Enum(b1, [b1, b2]).value_type, B)
    self.assertIsNone(vs.Enum(a, [a, b1, c]).value_type)

  def test_forward_refs(self):
    self.assertEqual(vs.Enum('a', ['a', None]).forward_refs, set())

  def test_type_resolved(self):
    self.assertTrue(vs.Enum('a', ['a', None]).type_resolved)

  def test_default(self):
    self.assertEqual(vs.Enum('a', ['a', 'b']).default, 'a')
    self.assertEqual(vs.Enum('a', ['a']).noneable().default, 'a')
    self.assertIsNone(vs.Enum(None, [None, 'a']).default)

  def test_noneable(self):
    self.assertFalse(vs.Enum('a', ['a', 'b']).is_noneable)
    self.assertTrue(vs.Enum('a', ['a', 'b', None]).is_noneable)
    self.assertEqual(
        vs.Enum('a', ['a', 'b']).noneable(),
        vs.Enum('a', ['a', 'b', None]))

  def test_str(self):
    self.assertEqual(
        str(vs.Enum('a', ['a', 'b', 'c'])),
        'Enum(default=\'a\', values=[\'a\', \'b\', \'c\'])')

    self.assertEqual(
        str(vs.Enum('a', ['a', 'b', 'c']).freeze()),
        'Enum(default=\'a\', values=[\'a\', \'b\', \'c\'], frozen=True)')

  def test_annotation(self):
    self.assertEqual(vs.Enum('a', ['a', 'b']).annotation, str)
    self.assertEqual(vs.Enum('a', ['a', None]).annotation, typing.Optional[str])
    self.assertEqual(vs.Enum(1, [1, 2]).annotation, int)
    self.assertEqual(vs.Enum(1, [1, None]).annotation, typing.Optional[int])
    self.assertEqual(vs.Enum(1, [1, 'foo']).annotation, typing.Any)

  def test_eq(self):
    e = vs.Enum('a', ['a', 'b'])
    self.assertEqual(e, e)
    self.assertEqual(vs.Enum('a', ['a']), vs.Enum('a', ['a']))
    self.assertNotEqual(vs.Enum('a', ['a']), vs.Int())
    self.assertNotEqual(vs.Enum('a', ['a']), vs.Enum('a', ['a', 'b']))
    self.assertNotEqual(
        vs.Enum('a', ['a', 'b']), vs.Enum('b', ['a', 'b']))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError, 'Values for Enum should be a non-empty list.'):
      vs.Enum(None, [])

    with self.assertRaisesRegex(
        ValueError, 'Enum default value \'a\' is not in candidate list.'):
      vs.Enum('a', ['b'])

  def test_apply(self):
    self.assertEqual(vs.Enum('a', ['a']).apply('a'), 'a')
    self.assertIsNone(vs.Enum('a', ['a', None]).apply(None))

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered <(type|class) \'int\'>'):
      vs.Enum('a', ['a']).apply(1)

    with self.assertRaisesRegex(
        ValueError, 'Value \'b\' is not in candidate list'):
      vs.Enum('a', ['a']).apply('b')

  def test_is_compatible(self):
    self.assertTrue(
        vs.Enum(0, [0, 1]).is_compatible(vs.Enum(0, [0, 1])))
    self.assertTrue(vs.Enum(0, [0, 1]).is_compatible(vs.Enum(0, [0])))
    self.assertFalse(vs.Enum(0, [0]).is_compatible(vs.Enum(0, [0, 1])))
    self.assertFalse(vs.Enum(0, [0]).is_compatible(vs.Int()))

  def test_extend(self):
    self.assertEqual(
        vs.Enum('a', ['a']).extend(vs.Enum('b', ['a', 'b'])),
        vs.Enum('a', ['a']))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: values in base should be super set.'):
      vs.Enum('a', ['a', 'b']).extend(vs.Enum('a', ['a']))

  def test_freeze(self):
    self.assertFalse(vs.Enum('a', ['a', 'b']).frozen)

    v = vs.Enum('a', ['a', 'b', 'c']).freeze('b')
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 'b')
    self.assertEqual(v.apply('b'), 'b')
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), 'b')
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      v.apply('a')

    v = vs.Enum('a', ['a', 'b', 'c']).freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 'a')

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Enum('c', ['a', 'b', 'c']).extend(v)


class ListTest(unittest.TestCase):
  """Tests for `List`."""

  def test_value_type(self):
    self.assertEqual(vs.List(vs.Int()).value_type, list)

  def test_forward_refs(self):
    self.assertEqual(vs.List(vs.Int()).forward_refs, set())
    self.assertEqual(
        vs.List(vs.Object('A')).forward_refs, set([forward_ref('A')])
    )

  def test_type_resolved(self):
    self.assertTrue(vs.List(vs.Int()).type_resolved)
    self.assertFalse(vs.List(vs.Object('A')).type_resolved)

    class A:
      pass

    with simulate_forward_declaration(A):
      self.assertTrue(vs.List(vs.Object('A')).type_resolved)

  def test_default(self):
    self.assertEqual(
        vs.List(vs.Int()).default, typed_missing.MISSING_VALUE)
    self.assertIsNone(vs.List(vs.Int()).noneable().default)
    self.assertEqual(vs.List(vs.Int(), []).default, [])

  def test_min_max_size(self):
    self.assertEqual(vs.List(vs.Int()).min_size, 0)
    self.assertIsNone(vs.List(vs.Int()).max_size)

    v = vs.List(vs.Int(), size=2)
    self.assertEqual(v.min_size, 2)
    self.assertEqual(v.max_size, 2)

  def test_noneable(self):
    self.assertFalse(vs.List(vs.Int()).is_noneable)
    self.assertTrue(vs.List(vs.Int()).noneable().is_noneable)

  def test_str(self):
    self.assertEqual(
        repr(vs.List(vs.Int(min_value=0))), 'List(Int(min=0))')
    self.assertEqual(
        repr(vs.List(vs.Int(min_value=0)).freeze([1])),
        'List(Int(min=0), default=[1], frozen=True)')
    self.assertEqual(
        repr(vs.List(vs.Str(), default=[], max_size=5).noneable()),
        'List(Str(), max_size=5, default=[], noneable=True)')

  def test_annotation(self):
    self.assertEqual(vs.List(vs.Int()).annotation, typing.List[int])
    self.assertEqual(
        vs.List(vs.Any()).annotation, typing.List[typing.Any])
    self.assertEqual(
        vs.List(vs.Int()).noneable().annotation,
        typing.Optional[typing.List[int]])
    self.assertEqual(
        vs.List(vs.Int().noneable()).annotation,
        typing.List[typing.Optional[int]])
    self.assertEqual(
        vs.List(vs.Object(BoolTest)).annotation, typing.List[BoolTest]
    )
    # Unresolved forward declaration will use string as annotation.
    self.assertEqual(vs.List(vs.Object('A')).annotation, typing.List['A'])

  def test_eq(self):
    self.assertEqual(vs.List(vs.Int()), vs.List(vs.Int()))
    self.assertEqual(
        vs.List(vs.Int(), []), vs.List(vs.Int(), []))
    self.assertEqual(
        vs.List(vs.Int(), [], max_size=10),
        vs.List(vs.Int(), [], max_size=10))

    self.assertNotEqual(vs.List(vs.Int()), vs.Int())
    self.assertNotEqual(
        vs.List(vs.Int()),
        vs.List(vs.Int()).noneable())
    self.assertNotEqual(vs.List(vs.Int()), vs.List(vs.Str()))
    self.assertNotEqual(
        vs.List(vs.Int(min_value=0)), vs.List(vs.Int()))
    self.assertNotEqual(
        vs.List(vs.Int(), []), vs.List(vs.Int()))
    self.assertNotEqual(
        vs.List(vs.Int(), max_size=10), vs.List(vs.Int()))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError, 'List element spec should be an ValueSpec object.'):
      vs.List(1)

    with self.assertRaisesRegex(ValueError,
                                '"min_size" of List must be no less than 0.'):
      vs.List(vs.Int(), min_size=-1)

    with self.assertRaisesRegex(
        ValueError, '"max_size" of List must be no less than "min_size".'):
      vs.List(vs.Int(), min_size=10, max_size=5)

    with self.assertRaisesRegex(
        ValueError,
        'Either "size" or "min_size"/"max_size" pair can be specified.'):
      vs.List(vs.Int(), size=5, min_size=1)

  def test_apply(self):
    self.assertEqual(vs.List(vs.Int()).apply([]), [])
    self.assertEqual(vs.List(vs.Int()).apply([1]), [1])
    self.assertEqual(vs.List(vs.Int().noneable()).apply([1, None]), [1, None])
    # Automatic conversion: str -> KeyPath is a registered conversion.
    # See 'type_conversion.py'.
    l = vs.List(vs.Object(object_utils.KeyPath)).apply(['a.b.c'])
    self.assertIsInstance(l[0], object_utils.KeyPath)
    self.assertEqual(l, [object_utils.KeyPath.parse('a.b.c')])
    self.assertEqual(
        vs.List(vs.Int()).apply(
            typed_missing.MISSING_VALUE, allow_partial=True),
        typed_missing.MISSING_VALUE)
    self.assertEqual(
        vs.List(vs.Dict([('a', vs.Str())])).apply([{}], allow_partial=True),
        [{'a': typed_missing.MissingValue(vs.Str())}],
    )

    l = vs.List(vs.Object('A'))

    # Element spec `vs.Object('A')` is not resolved, thus we do not perform type
    # checking on list element.
    self.assertEqual(l.apply([1]), [1])

    class A:
      pass

    with simulate_forward_declaration(A):
      # Now 'A' is resolved, thus type check will occur.
      with self.assertRaisesRegex(TypeError, 'Expect .* but encountered'):
        l.apply([1])

      # But objects of `A` are acceptable.
      a = A()
      self.assertEqual(l.apply([a]), [a])

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.List(vs.Int()).apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'list\'> but encountered '
        '<(type|class) \'int\'>.'):
      vs.List(vs.Int()).apply(1)

    with self.assertRaisesRegex(
        ValueError, 'Value 0 is out of range \\(min=1, max=None\\)'):
      vs.List(vs.Int(min_value=1)).apply([0])

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is less than min size \\(1\\).'):
      vs.List(vs.Int(), min_size=1).apply([])

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is greater than max size \\(1\\).'):
      vs.List(vs.Int(), max_size=1).apply([0, 1])

  def test_apply_with_user_validator(self):
    def _sum_greater_than_zero(value):
      if sum(value) <= 0:
        raise ValueError('Sum expected to be larger than zero')

    self.assertEqual(
        vs.List(vs.Int(), user_validator=_sum_greater_than_zero).apply([0, 1]),
        [0, 1],
    )

    with self.assertRaisesRegex(
        ValueError, 'Sum expected to be larger than zero \\(path=\\[0\\]\\)'):
      vs.List(vs.List(vs.Int(), user_validator=_sum_greater_than_zero)).apply(
          [[-1]]
      )

  def test_is_compatible(self):
    self.assertTrue(
        vs.List(vs.Int()).is_compatible(vs.List(vs.Int())))

    self.assertTrue(
        vs.List(vs.Int()).noneable().is_compatible(vs.List(vs.Int())))

    self.assertTrue(
        vs.List(vs.Int()).is_compatible(vs.List(vs.Int(min_value=1))))

    self.assertTrue(
        vs.List(vs.Int().noneable()).is_compatible(vs.List(vs.Int())))

    self.assertTrue(
        vs.List(vs.Int(), min_size=10).is_compatible(
            vs.List(vs.Int(), min_size=5)))

    self.assertTrue(
        vs.List(vs.Int()).is_compatible(vs.List(vs.Int(), max_size=10)))

    self.assertFalse(vs.List(vs.Int()).is_compatible(vs.Int()))

    self.assertFalse(vs.List(vs.Int()).is_compatible(vs.List(vs.Str())))

    self.assertFalse(
        vs.List(vs.Int()).is_compatible(vs.List(vs.Int().noneable())))

    self.assertFalse(
        vs.List(vs.Int(min_value=1)).is_compatible(vs.List(vs.Int())))

    self.assertFalse(
        vs.List(vs.Int()).is_compatible(vs.List(vs.Int()).noneable()))

    self.assertFalse(
        vs.List(vs.Int(), max_size=10).is_compatible(vs.List(vs.Int())))

    self.assertFalse(
        vs.List(vs.Int(), max_size=10).is_compatible(
            vs.List(vs.Int(), max_size=11)))

  def test_extend(self):
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        vs.List(vs.Int()).extend(vs.List(
            vs.Int(min_value=0))).element.value, vs.Int(min_value=0))

    self.assertEqual(
        vs.List(vs.Int()).extend(vs.List(vs.Int(), max_size=10)),
        vs.List(vs.Int(), max_size=10))

    self.assertFalse(
        vs.List(vs.Int()).extend(vs.List(vs.Int()).noneable()).is_noneable)

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.List(vs.Int()).extend(vs.Int())

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.List(vs.Int()).extend(vs.List(vs.Str()))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.List(vs.Int()).noneable().extend(vs.List(vs.Int()))

    # Child with smaller min_size cannot extend base with larger min_size.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: min_value is smaller.'):
      vs.List(vs.Int(), min_size=0).extend(vs.List(vs.Int(), min_size=1))

    # Child with larger max_size cannot extend base with smaller max_size.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: max_value is greater.'):
      vs.List(vs.Int(), max_size=10).extend(vs.List(vs.Int(), max_size=1))

  def test_freeze(self):
    self.assertFalse(vs.List(vs.Int()).frozen)

    v = vs.List(vs.Int()).freeze([1])
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, [1])
    self.assertEqual(v.apply([1]), [1])
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), [1])
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply([2])

    v = vs.List(vs.Int(), default=[1]).freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, [1])

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.List(vs.Int()).extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.List(vs.Int()).freeze()


class TupleTest(unittest.TestCase):
  """Tests for `Tuple`."""

  def test_value_type(self):
    self.assertEqual(vs.Tuple([vs.Int()]).value_type, tuple)
    self.assertEqual(vs.Tuple(vs.Int()).value_type, tuple)

  def test_forward_refs(self):
    self.assertEqual(vs.Tuple(vs.Int()).forward_refs, set())
    self.assertEqual(
        vs.Tuple(vs.Object('A')).forward_refs, set([forward_ref('A')])
    )
    self.assertEqual(
        vs.Tuple([vs.Object('A'), vs.Int(), vs.Object('B')]).forward_refs,
        set([forward_ref('A'), forward_ref('B')]),
    )

  def test_type_resolved(self):
    self.assertTrue(vs.Tuple(vs.Int()).type_resolved)
    self.assertFalse(vs.Tuple(vs.Object('A')).type_resolved)
    self.assertFalse(vs.Tuple([vs.Int(), vs.Object('A')]).type_resolved)

    class A:
      pass

    with simulate_forward_declaration(A):
      self.assertTrue(vs.Tuple(vs.Object('A')).type_resolved)
      self.assertTrue(vs.Tuple([vs.Int(), vs.Object('A')]).type_resolved)

  def test_default(self):
    self.assertEqual(
        vs.Tuple(vs.Int()).default, typed_missing.MISSING_VALUE)
    self.assertIsNone(vs.Tuple([vs.Int()]).noneable().default)
    self.assertEqual(vs.Tuple([vs.Int()], (1,)).default, (1,))

  def test_noneable(self):
    self.assertFalse(vs.Tuple([vs.Int()]).is_noneable)
    self.assertTrue(vs.Tuple([vs.Int()]).noneable().is_noneable)

  def test_fixed_length(self):
    self.assertFalse(vs.Tuple(vs.Int()).fixed_length)
    self.assertTrue(vs.Tuple(vs.Int(), size=2).fixed_length)
    self.assertTrue(vs.Tuple([vs.Int(), vs.Str()]).fixed_length)

  def test_min_size(self):
    self.assertEqual(vs.Tuple(vs.Int()).min_size, 0)
    self.assertEqual(vs.Tuple(vs.Int(), min_size=1).min_size, 1)
    self.assertEqual(vs.Tuple(vs.Int(), size=2).min_size, 2)
    self.assertEqual(vs.Tuple([vs.Int()]).min_size, 1)

  def test_max_size(self):
    self.assertIsNone(vs.Tuple(vs.Int()).max_size)
    self.assertEqual(vs.Tuple(vs.Int(), max_size=1).max_size, 1)
    self.assertEqual(vs.Tuple(vs.Int(), size=2).max_size, 2)
    self.assertEqual(vs.Tuple([vs.Int()]).max_size, 1)

  def test_len(self):
    self.assertEqual(len(vs.Tuple(vs.Int())), 0)
    self.assertEqual(len(vs.Tuple(vs.Int(), size=2)), 2)
    self.assertEqual(len(vs.Tuple([vs.Int(), vs.Str()])), 2)

  def test_str(self):
    self.assertEqual(repr(vs.Tuple(vs.Int())), 'Tuple(Int())')

    self.assertEqual(
        repr(vs.Tuple(vs.Int(), min_size=2, max_size=3)),
        'Tuple(Int(), min_size=2, max_size=3)')

    self.assertEqual(
        repr(vs.Tuple([vs.Int(), vs.Bool()])),
        'Tuple([Int(), Bool()])')
    self.assertEqual(
        repr(vs.Tuple([vs.Int(), vs.Bool()], default=(1, True)).noneable()),
        'Tuple([Int(), Bool()], default=(1, True), noneable=True)')
    self.assertEqual(
        repr(vs.Tuple([vs.Int(), vs.Bool()], default=(1, True)).freeze()),
        'Tuple([Int(), Bool()], default=(1, True), frozen=True)')

  def test_annotation(self):
    self.assertEqual(
        vs.Tuple(vs.Int()).annotation, typing.Tuple[int, ...])
    self.assertEqual(
        vs.Tuple(vs.Int()).annotation, typing.Tuple[int, ...])
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Str()]).annotation, typing.Tuple[int, str])
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Any()]).annotation,
        typing.Tuple[int, typing.Any])
    self.assertEqual(
        vs.Tuple(vs.Object('BoolTest')).annotation, typing.Tuple[BoolTest, ...]
    )
    self.assertEqual(
        vs.Tuple(vs.Object('A')).annotation, typing.Tuple['A', ...]
    )
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Object('BoolTest')]).annotation,
        typing.Tuple[int, BoolTest],
    )
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Object('A')]).annotation, typing.Tuple[int, 'A']
    )

  def test_eq(self):
    v = vs.Tuple([vs.Int(), vs.Int()])
    self.assertEqual(v, v)

    self.assertEqual(vs.Tuple(vs.Int()), vs.Tuple(vs.Int()))
    self.assertEqual(vs.Tuple([vs.Int()]), vs.Tuple([vs.Int()]))
    self.assertEqual(vs.Tuple(vs.Int(), size=2), vs.Tuple([vs.Int(), vs.Int()]))

    self.assertEqual(
        vs.Tuple([vs.Int()], (1,)), vs.Tuple([vs.Int()], (1,)))
    self.assertEqual(vs.Tuple([vs.Int(), vs.Bool()]),
                     vs.Tuple([vs.Int(), vs.Bool()]))

    self.assertNotEqual(vs.Tuple([vs.Int()]), vs.Int())
    self.assertNotEqual(vs.Tuple([vs.Int()]), vs.Tuple(vs.Int()))
    self.assertNotEqual(vs.Tuple(vs.Int()), vs.Tuple(vs.Int(), max_size=2))

    self.assertNotEqual(vs.Tuple([vs.Int()]), vs.Tuple([vs.Bool()]))
    self.assertNotEqual(vs.Tuple([vs.Int()]), vs.Tuple([vs.Int()]).noneable())
    self.assertNotEqual(vs.Tuple([vs.Int(min_value=0)]), vs.Tuple([vs.Int()]))
    self.assertNotEqual(vs.Tuple([vs.Int()], (1,)), vs.Tuple([vs.Int()]))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError, 'Argument \'element_values\' must be a non-empty list'):
      vs.Tuple(1)

    with self.assertRaisesRegex(
        ValueError, 'Argument \'element_values\' must be a non-empty list'):
      vs.Tuple([])

    with self.assertRaisesRegex(
        ValueError, 'Items in \'element_values\' must be ValueSpec objects.'):
      vs.Tuple([1])

    with self.assertRaisesRegex(
        ValueError,
        'Either "size" or "min_size"/"max_size" pair can be specified'):
      vs.Tuple(vs.Int(), size=2, min_size=1)

    with self.assertRaisesRegex(
        ValueError, '"min_size" of List must be no less than 0.'):
      vs.Tuple(vs.Int(), min_size=-1)

    with self.assertRaisesRegex(
        ValueError, '"max_size" of List must be no less than "min_size"'):
      vs.Tuple(vs.Int(), min_size=2, max_size=1)

    with self.assertRaisesRegex(
        ValueError, '"size", "min_size" and "max_size" are not applicable'):
      vs.Tuple([vs.Int(), vs.Str()], size=2)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'tuple\'> but encountered '
        '<(type|class) \'int\'>.'):
      vs.Tuple([vs.Int()], default=1)

  def test_apply(self):
    self.assertEqual(vs.Tuple(vs.Int()).apply(tuple()), tuple())
    self.assertEqual(vs.Tuple(vs.Int()).apply((1, 1, 1)), (1, 1, 1))
    self.assertEqual(vs.Tuple(vs.Int(), min_size=1).apply((1, 1, 1)), (1, 1, 1))
    self.assertEqual(vs.Tuple(vs.Int(), max_size=2).apply((1, 1)), (1, 1))

    self.assertEqual(vs.Tuple([vs.Int()]).apply((1,)), (1,))
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Bool()]).apply((1, True)), (1, True))
    self.assertIsNone(vs.Tuple([vs.Int()]).noneable().apply(None))
    self.assertEqual(vs.Tuple([vs.Int().noneable()]).apply((None,)), (None,))
    self.assertEqual(
        vs.Tuple([vs.Int()]).apply(
            typed_missing.MISSING_VALUE, allow_partial=True),
        typed_missing.MISSING_VALUE)
    self.assertEqual(
        vs.Tuple([vs.Int()]).apply(
            (typed_missing.MISSING_VALUE,), allow_partial=True),
        (typed_missing.MISSING_VALUE,))
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Dict([('a', vs.Str())])]).apply(
            (1, {}), allow_partial=True),
        (1, {
            'a': typed_missing.MissingValue(vs.Str())
        }))

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Tuple([vs.Int()]).apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'tuple\'> but encountered '
        '<(type|class) \'int\'>.'):
      vs.Tuple([vs.Int()]).apply(1)

    with self.assertRaisesRegex(
        TypeError,
        'Expect <(type|class) \'int\'> but encountered <(type|class) \'str\'>'):
      vs.Tuple([vs.Int()]).apply(('abc',))

    with self.assertRaisesRegex(
        TypeError,
        'Expect <(type|class) \'int\'> but encountered <(type|class) \'str\'>'):
      vs.Tuple(vs.Int()).apply(('abc',))

    with self.assertRaisesRegex(
        ValueError, 'Length of tuple .* is less than min size'):
      vs.Tuple(vs.Int(), min_size=2).apply((1,))

    with self.assertRaisesRegex(
        ValueError, 'Length of tuple .* is greater than max size'):
      vs.Tuple(vs.Int(), max_size=2).apply((1, 1, 1))

    with self.assertRaisesRegex(
        ValueError,
        'Length of input tuple .* does not match the length of spec.'):
      vs.Tuple([vs.Int()]).apply((1, 1))

  def test_apply_with_user_validator(self):
    def _sum_greater_than_zero(value):
      if sum(list(value)) <= 0:
        raise ValueError('Sum expected to be larger than zero')

    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Int()],
                 user_validator=_sum_greater_than_zero).apply((0, 1)),
        (0, 1))

    with self.assertRaisesRegex(
        ValueError, 'Sum expected to be larger than zero \\(path=\\[0\\]\\)'):
      vs.Tuple(
          [vs.Tuple([vs.Int()], user_validator=_sum_greater_than_zero)]
      ).apply(((-1,),))

  def test_is_compatible(self):
    self.assertTrue(vs.Tuple(vs.Int()).is_compatible(vs.Tuple(vs.Int())))
    self.assertTrue(vs.Tuple(vs.Int()).is_compatible(vs.Tuple([vs.Int()])))
    self.assertTrue(
        vs.Tuple(vs.Int()).is_compatible(
            vs.Tuple(vs.Int(), min_size=2, max_size=4)))

    self.assertTrue(
        vs.Tuple(vs.Int(), min_size=1).is_compatible(
            vs.Tuple(vs.Int(), min_size=2)))

    self.assertTrue(
        vs.Tuple(vs.Int(), max_size=5).is_compatible(
            vs.Tuple(vs.Int(), max_size=4)))

    self.assertTrue(
        vs.Tuple([vs.Int()]).is_compatible(vs.Tuple([vs.Int()])))

    self.assertTrue(
        vs.Tuple([vs.Int()]).noneable().is_compatible(vs.Tuple([vs.Int()])))

    self.assertTrue(
        vs.Tuple([vs.Int()]).is_compatible(vs.Tuple([vs.Int(min_value=1)])))

    self.assertTrue(
        vs.Tuple([vs.Int().noneable()]).is_compatible(vs.Tuple([vs.Int()])))

    self.assertFalse(vs.Tuple(vs.Int()).is_compatible(vs.Int()))
    self.assertFalse(vs.Tuple([vs.Int()]).is_compatible(vs.Int()))

    self.assertFalse(
        vs.Tuple(vs.Int(), min_size=2).is_compatible(
            vs.Tuple(vs.Int(), min_size=1)))

    self.assertFalse(
        vs.Tuple(vs.Int(), max_size=2).is_compatible(
            vs.Tuple(vs.Int(), max_size=3)))

    self.assertFalse(
        vs.Tuple(vs.Int()).is_compatible(
            vs.Tuple([vs.Int(), vs.Str()])))

    self.assertFalse(
        vs.Tuple(vs.Int(), min_size=3).is_compatible(
            vs.Tuple([vs.Int(), vs.Int()])))

    self.assertFalse(
        vs.Tuple(vs.Int(), max_size=2).is_compatible(
            vs.Tuple([vs.Int(), vs.Int(), vs.Int()])))

    self.assertFalse(
        vs.Tuple([vs.Int()]).is_compatible(
            vs.Tuple([vs.Int(), vs.Int()])))

    self.assertFalse(vs.Tuple([vs.Int()]).is_compatible(vs.Tuple([vs.Str()])))
    self.assertFalse(vs.Tuple([vs.Int()]).is_compatible(vs.List(vs.Int())))
    self.assertFalse(vs.Tuple([vs.Int()]).is_compatible(vs.Tuple(vs.Int())))

    self.assertFalse(
        vs.Tuple([vs.Int()]).is_compatible(vs.Tuple([vs.Int().noneable()])))

    self.assertFalse(
        vs.Tuple([vs.Int(min_value=1)]).is_compatible(vs.Tuple([vs.Int()])))

    self.assertFalse(
        vs.Tuple([vs.Int()]).is_compatible(vs.Tuple([vs.Int()]).noneable()))

  def test_extend(self):
    # Variable length tuple extend variable length tuple:
    self.assertEqual(
        vs.Tuple(vs.Int(), max_size=5).extend(vs.Tuple(vs.Int(), min_size=2)),
        vs.Tuple(vs.Int(), min_size=2, max_size=5))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has smaller min size'):
      vs.Tuple(vs.Int(), min_size=2).extend(vs.Tuple(vs.Int(), min_size=5))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has greater max size'):
      vs.Tuple(vs.Int(), max_size=5).extend(vs.Tuple(vs.Int(), max_size=3))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type'):
      vs.Tuple(vs.Int()).extend(vs.Tuple(vs.Str()))

    # Variable length tuple extend fixed length tuple.
    with self.assertRaisesRegex(
        TypeError,
        '.* cannot extend .*: a variable length tuple cannot extend a fixed '
        'length tuple'):
      vs.Tuple(vs.Int()).extend(vs.Tuple([vs.Int()]))

    # Fixed length tuple extend variable length tuple.
    self.assertEqual(
        vs.Tuple([vs.Int(), vs.Int()]).extend(vs.Tuple(vs.Int(min_value=0))),
        vs.Tuple([vs.Int(min_value=0), vs.Int(min_value=0)]))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has less elements than required'):
      vs.Tuple([vs.Int()]).extend(vs.Tuple(vs.Int(), min_size=2))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has more elements than required'):
      vs.Tuple([vs.Int(), vs.Int()]).extend(vs.Tuple(vs.Int(), max_size=1))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type'):
      vs.Tuple([vs.Int(), vs.Str()]).extend(vs.Tuple(vs.Int()))

    # Fixed length tuple extend fixed length tuple.
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        vs.Tuple([vs.Int()]).extend(
            vs.Tuple([vs.Int(min_value=0)])).elements[0].value,
        vs.Int(min_value=0))

    self.assertFalse(
        vs.Tuple([vs.Int()]).extend(
            vs.Tuple([vs.Int()]).noneable()).is_noneable)

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Tuple([vs.Int()]).extend(vs.Int())

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Tuple([vs.Int()]).extend(vs.Tuple([vs.Str()]))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: unmatched number of elements.'):
      vs.Tuple([vs.Int()]).extend(vs.Tuple([vs.Int(), vs.Int()]))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Tuple([vs.Int()]).noneable().extend(vs.Tuple([vs.Int()]))

    # Child with larger max_size cannot extend base with smaller max_size.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: max_value is larger.'):
      vs.Tuple([vs.Int(max_value=100)]).extend(vs.Tuple([vs.Int(max_value=0)]))

  def test_freeze(self):
    self.assertFalse(vs.Tuple(vs.Int()).frozen)

    v = vs.Tuple(vs.Int()).freeze((1,))
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, (1,))
    self.assertEqual(v.apply((1,)), (1,))
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), (1,))
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply((1, 1))

    v = vs.Tuple(vs.Int(), default=(1,)).freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, (1,))

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Tuple(vs.Int()).extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Tuple(vs.Int()).freeze()


class DictTest(unittest.TestCase):
  """Tests for `Dict`."""

  def test_value_type(self):
    self.assertIs(vs.Dict().value_type, dict)

  def test_forward_refs(self):
    self.assertEqual(vs.Dict().forward_refs, set())
    self.assertEqual(
        vs.Dict([
            ('x', vs.Object('A')),
            ('y', vs.Dict([
                ('z', vs.Object('B'))
            ]))
        ]).forward_refs,
        set((forward_ref('A'), forward_ref('B'))))

  def test_type_resolved(self):
    self.assertTrue(vs.Dict().type_resolved)
    v = vs.Dict([
        ('x', vs.Object('A')),
        ('y', vs.Dict([
            ('z', vs.Object('B'))
        ]))
    ])
    self.assertFalse(v.type_resolved)

    class A:
      pass

    class B:
      pass

    with simulate_forward_declaration(A):
      self.assertFalse(v.type_resolved)

    with simulate_forward_declaration(A, B):
      self.assertTrue(v.type_resolved)

  def test_default(self):
    self.assertEqual(
        vs.Dict([('a', vs.Int(), 'field 1')]).default,
        {'a': typed_missing.MISSING_VALUE})

    self.assertEqual(
        vs.Dict([
            ('a', vs.Int(1), 'field 1'),
            ('b', vs.Dict([
                ('c', 'foo', 'field 2.1')
            ]), 'field 2')
        ]).default,
        dict(a=1, b={'c': 'foo'}))

    self.assertEqual(
        vs.Dict([
            ('a', vs.Int(1), 'field 1'),
            ('b', vs.Dict([
                ('c', vs.Object('A'), 'field 2.1')
            ]), 'field 2')]).default,
        dict(a=1, b={'c': typed_missing.MISSING_VALUE}))

    self.assertIsNone(vs.Dict([('a', vs.Int(), 'field 1')]).noneable().default)

  def test_noneable(self):
    self.assertFalse(vs.Dict().is_noneable)
    self.assertTrue(vs.Dict().noneable().is_noneable)

  def test_repr(self):
    self.assertEqual(repr(vs.Dict()), 'Dict()')
    self.assertEqual(
        repr(
            vs.Dict([
                ('b', 1, 'field 1'),
                ('a', vs.Str(), 'field 2'),
            ]).noneable()),
        'Dict({b=Int(default=1), a=Str()}, noneable=True)')

    self.assertEqual(
        repr(
            vs.Dict([
                ('b', 1, 'field 1'),
                ('a', vs.Str('abc'), 'field 2'),
            ]).freeze()),
        'Dict({b=Int(default=1), a=Str(default=\'abc\')}, frozen=True)')

  def test_annotation(self):
    self.assertEqual(vs.Dict().annotation, typing.Dict[str, typing.Any])
    self.assertEqual(vs.Dict().noneable().annotation,
                     typing.Optional[typing.Dict[str, typing.Any]])

  def test_eq(self):
    d = vs.Dict([('a', vs.Int())])
    self.assertEqual(d, d)
    self.assertEqual(d.schema, class_schema.create_schema([('a', vs.Int())]))
    self.assertEqual(vs.Dict(), vs.Dict())
    self.assertEqual(vs.Dict().noneable(), vs.Dict().noneable())
    self.assertEqual(
        vs.Dict([('a', 1, 'field 1')]), vs.Dict([('a', 1, 'field 1')]))
    self.assertNotEqual(vs.Dict(), vs.Dict().noneable())
    self.assertNotEqual(vs.Dict(), vs.Dict([('a', 1, 'field 1')]))
    self.assertNotEqual(vs.Dict(), vs.Dict([('a', 1, 'field 1')]))
    self.assertNotEqual(
        vs.Dict([('a', vs.Int(), 'field 1')]), vs.Dict([('a', 1, 'field 1')]))

  def test_init(self):
    self.assertEqual(
        vs.Dict(class_schema.create_schema([('a', vs.Int())])),
        vs.Dict([('a', vs.Int())]))

    with self.assertRaisesRegex(
        TypeError, 'Schema definition should be a list of schema.Field or '
        'a list of tuples of \\(key, value, description, metadata\\).'):
      vs.Dict({})

    with self.assertRaisesRegex(
        TypeError, 'The 1st element of field definition should be of '
        '<(type|class) \'str\'>'):
      vs.Dict([(1, 1, 'field 1')])

    with self.assertRaisesRegex(
        TypeError, 'Description \\(the 3rd element\\) of field definition '
        'should be text type.'):
      vs.Dict([('key', 1, 1)])

    with self.assertRaisesRegex(
        TypeError, 'Metadata \\(the 4th element\\) of field definition '
        'should be a dict of objects.'):
      vs.Dict([('key', 1, 'field 1', 123)])

  def test_apply(self):
    self.assertEqual(vs.Dict().apply({'a': 1}), {'a': 1})
    self.assertEqual(
        list(vs.Dict().apply({'b': 1, 'a': 2}).keys()), ['b', 'a'])

    self.assertEqual(
        vs.Dict([
            ('a', vs.Int(), 'field 1'),
            ('b', vs.Bool().noneable(), 'field 2'),
        ]).apply({'a': 1}),
        {
            'a': 1,
            'b': None
        })

    # Tests with forward declaration.
    v = vs.Dict([
        ('a', vs.Int(), 'field 1'),
        ('b', vs.Object('B'), 'field 2')
    ])

    # Before forward reference 'B' can be resolved, `b` can accept anything.
    self.assertEqual(v.apply(dict(a=1, b=2)), dict(a=1, b=2))

    # After forward reference 'B' can be resolved, `b` can accept only objects
    # of `B`.
    class B:
      pass

    b = B()
    with simulate_forward_declaration(B):
      self.assertEqual(v.apply(dict(a=1, b=b)), dict(a=1, b=b))
      with self.assertRaisesRegex(TypeError, 'Expect .* but encountered'):
        v.apply(dict(a=1, b=2))

    # Tests for partial apply.
    self.assertEqual(
        vs.Dict([
            ('a', vs.Int(), 'field 1'),
            ('b', vs.Bool().noneable(), 'field 2'),
        ]).apply({'b': True}, allow_partial=True),
        {
            'a': typed_missing.MISSING_VALUE,
            'b': True
        })

    self.assertEqual(
        vs.Dict([
            ('a', 1, 'field a'),
            ('b', vs.Str(), 'field b'),
            ('c', vs.Dict([
                ('d', True, 'field d'),
                ('e', vs.Float(), 'field f'),
            ]), 'field c')]).apply({}, allow_partial=True),
        {
            'a': 1,
            'b': typed_missing.MissingValue(vs.Str()),
            'c': {
                'd': True,
                'e': typed_missing.MissingValue(vs.Float()),
            }
        })

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Dict().apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'dict\'> but encountered '
        '<(type|class) \'int\'>.'):
      vs.Dict().apply(1)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'int\'> but encountered '
        '<(type|class) \'str\'>.'):
      vs.Dict([('a', 1, 'field 1')]).apply({'a': 'foo'})

    with self.assertRaisesRegex(
        KeyError,
        'Keys \\[\'b\'\\] are not allowed in Schema. \\(parent=\'\'\\)'):
      vs.Dict([('a', 1, 'field 1')]).apply({'b': 1})

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified. \\(Path=\'a\'.*\\)'):
      vs.Dict([
          ('a', vs.Int(), 'field 1'),
          ('b', vs.Bool().noneable(), 'field 2'),
      ]).apply({'b': True})

  def test_apply_with_user_validator(self):
    def _sum_greater_than_zero(value):
      if sum(value.values()) <= 0:
        raise ValueError('Sum of values expected to be larger than zero')

    self.assertEqual(
        vs.Dict(user_validator=_sum_greater_than_zero).apply({
            'a': 1,
            'b': 2,
        }),
        {
            'a': 1,
            'b': 2
        })

    with self.assertRaisesRegex(
        ValueError,
        'Sum of values expected to be larger than zero \\(path=x\\)'):
      vs.Dict([
          ('x', vs.Dict(user_validator=_sum_greater_than_zero))
      ]).apply({'x': {'a': -1}})

  def test_is_compatible(self):
    self.assertTrue(vs.Dict().is_compatible(vs.Dict()))
    self.assertTrue(vs.Dict().noneable().is_compatible(vs.Dict()))
    self.assertTrue(vs.Dict().is_compatible(vs.Dict([('a', vs.Int())])))

    self.assertTrue(
        vs.Dict([
            ('a', vs.Int())
        ]).is_compatible(vs.Dict([('a', vs.Int(min_value=1))])))

    self.assertFalse(vs.Dict().is_compatible(vs.Int()))
    self.assertFalse(vs.Dict().is_compatible(vs.Dict().noneable()))
    self.assertFalse(vs.Dict([('a', vs.Int())]).is_compatible(vs.Dict()))

    self.assertFalse(
        vs.Dict([('a', vs.Int(min_value=1))]).is_compatible(
            vs.Dict([('a', vs.Int())])))

    self.assertFalse(
        vs.Dict([('a', vs.Int())]).is_compatible(
            vs.Dict([('a', vs.Int()), ('b', vs.Int())])))

  def test_extend(self):
    self.assertFalse(vs.Dict().extend(vs.Dict().noneable()).is_noneable)
    self.assertEqual(
        vs.Dict().extend(vs.Dict([('a', 1, 'field 1')])).schema,
        vs.Dict([('a', 1, 'field 1')]).schema)

    self.assertEqual(
        vs.Dict([('b', vs.Str(), 'field 2')]).extend(
            vs.Dict([('a', 1, 'field 1')])),
        vs.Dict([('a', 1, 'field 1'), ('b', vs.Str(), 'field 2')]))

    self.assertEqual(
        vs.Dict([('a', 1)]).extend(vs.Dict([('a', vs.Int(), 'field 1')])),
        vs.Dict([('a', 1, 'field 1')]))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Dict().extend(vs.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Dict().noneable().extend(vs.Dict())

    # Child extends base dict with incompatible values.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Dict([('a', 1, 'field 1')]).extend(
          vs.Dict([('a', vs.Str(), 'field 1')]))

  def test_freeze(self):
    # Test frozen value for dict.
    v = vs.Dict([
        ('x', vs.Int(default=1)),
        ('y', vs.Bool(default=True))
    ])
    v.freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), dict(x=1, y=True))


class ObjectTest(unittest.TestCase):
  """Tests for `Object`."""

  def setUp(self):
    super().setUp()

    class A:

      def __call__(self, a):
        pass

    class B(A):

      def __init__(self, value=0):
        self.value = value

    class C(A):
      pass

    class D(C, object_utils.MaybePartial):

      def missing_values(self):
        return {'SOME_KEY': 'SOME_VALUE'}

    # pylint: disable=invalid-name
    self.A = A
    self.B = B
    self.C = C
    self.D = D
    # pylint: enable=invalid-name

  def test_value_type(self):
    self.assertEqual(vs.Object(self.A).value_type, self.A)
    v = vs.Object('A')

    with simulate_forward_declaration(self.A):
      self.assertIs(v.value_type, self.A)

    with self.assertRaisesRegex(TypeError, "'A' does not exist in module .*"):
      _ = vs.Object('A').value_type

  def test_geneeric_type(self):
    class G(typing.Generic[typing.TypeVar('X'), typing.TypeVar('Y')]):
      pass

    class G1(G[int, str]):
      pass

    o = G1()

    v = vs.Object(G[int, str])
    self.assertIs(v.value_type, G[int, str])
    self.assertIs(v.apply(o), o)

    self.assertIs(vs.Object(G).apply(o), o)
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      vs.Object(G[str, int]).apply(o)

  def test_forward_refs(self):
    self.assertEqual(vs.Object(self.A).forward_refs, set())
    self.assertEqual(vs.Object('Foo').forward_refs, set([forward_ref('Foo')]))

  def test_default(self):
    self.assertEqual(vs.Object(self.A).default, typed_missing.MISSING_VALUE)
    a = self.A()
    self.assertEqual(vs.Object(self.A, a).default, a)

  def test_noneable(self):
    self.assertFalse(vs.Object(self.A).is_noneable)
    self.assertTrue(vs.Object(self.A).noneable().is_noneable)
    self.assertTrue(vs.Object('Foo').noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Object(self.A)), 'Object(A)')
    self.assertEqual(str(vs.Object('Foo')), 'Object(Foo)')
    self.assertEqual(
        str(vs.Object(self.A).noneable()),
        'Object(A, default=None, noneable=True)')
    self.assertEqual(
        str(vs.Object(self.A).noneable().freeze()),
        'Object(A, default=None, noneable=True, frozen=True)')

  def test_annotation(self):
    self.assertEqual(vs.Object(self.A).annotation, self.A)
    with simulate_forward_declaration(self.A):
      self.assertEqual(vs.Object('A').annotation, self.A)
    self.assertEqual(vs.Object('Foo').annotation, 'Foo')
    self.assertEqual(
        vs.Object(self.A).noneable().annotation, typing.Optional[self.A])

  def test_eq(self):
    o = vs.Object(self.A)
    self.assertEqual(o, o)
    self.assertIsNone(o.schema)
    self.assertEqual(vs.Object(self.A), vs.Object(self.A))

    class A:
      pass

    o = vs.Object('A')
    # The local class 'A' is not a module level class.
    self.assertNotEqual(o, vs.Object(A))
    with simulate_forward_declaration(self.A):
      self.assertEqual(o, vs.Object(self.A))

    self.assertEqual(vs.Object('A'), vs.Object('A'))
    self.assertNotEqual(vs.Object('A'), vs.Object('B'))
    self.assertEqual(
        vs.Object(self.A).noneable(), vs.Object(self.A).noneable())
    self.assertNotEqual(vs.Object(self.A).noneable(), vs.Object(self.A))
    self.assertNotEqual(vs.Object(self.A), vs.Object(self.B))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        TypeError, '"cls" for Object spec cannot be None.'):
      vs.Object(None)

    with self.assertRaisesRegex(
        TypeError, '"cls" for Object spec should be a type.'):
      vs.Object(1)

    with self.assertRaisesRegex(
        TypeError, '<(type|class) \'object\'> is too general for Object spec.'):
      vs.Object(object)

  def test_apply(self):
    a = self.A()
    self.assertEqual(vs.Object(self.A).apply(a), a)
    self.assertIsNone(vs.Object(self.A).noneable().apply(None))

    b = self.B()
    self.assertEqual(vs.Object(self.A).apply(b), b)

    d = self.D()
    self.assertEqual(vs.Object(self.C).apply(d, allow_partial=True), d)

    v = vs.Object('A')
    # 'A' is not resolved, so it could accept any object type.
    self.assertIs(v.apply(1), 1)
    with simulate_forward_declaration(self.A, self.B):
      self.assertIs(v.apply(a), a)
      self.assertIs(v.apply(b), b)
      with self.assertRaisesRegex(TypeError, 'Expect .* but encountered'):
        # 'A' is resolved, so it could not accept 1 any more.
        _ = v.apply(1)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Object(self.A).apply(None)

    with self.assertRaisesRegex(
        TypeError,
        'Expect <class .*A\'> but encountered <(type|class) \'int\'>.'):
      vs.Object(self.A).apply(1)

    with self.assertRaisesRegex(
        TypeError, 'Expect <class .*B\'> but encountered <class .*C\'>.'):
      vs.Object(self.B).apply(self.C())

    with self.assertRaisesRegex(ValueError, 'Object .* is not fully bound.'):
      vs.Object(self.C).apply(self.D())

  def test_apply_with_user_validator(self):
    def _value_is_zero(b):
      if b.value != 0:
        raise ValueError('Value should be zero')

    b = self.B()
    self.assertEqual(
        vs.Object(self.B, user_validator=_value_is_zero).apply(b), b)

    with self.assertRaisesRegex(ValueError, 'Value should be zero \\(path=\\)'):
      vs.Object(self.B, user_validator=_value_is_zero).apply(self.B(1))

  def test_is_compatible(self):
    self.assertTrue(vs.Object(self.A).is_compatible(vs.Object(self.A)))
    self.assertTrue(
        vs.Object(self.A).noneable().is_compatible(vs.Object(self.A)))

    # Before a forward declaration can be resolved, `is_compatible` always
    # returns True.
    self.assertTrue(vs.Object('A').is_compatible(vs.Object('B')))
    with simulate_forward_declaration(self.A, self.B):
      self.assertTrue(vs.Object('A').is_compatible(vs.Object('B')))
      self.assertFalse(vs.Object('B').is_compatible(vs.Object('A')))

    self.assertTrue(vs.Object(self.A).is_compatible(vs.Object(self.B)))
    self.assertTrue(vs.Object('Foo').is_compatible(vs.Object('Bar')))

    self.assertFalse(vs.Object(self.A).is_compatible(vs.Int()))
    self.assertFalse(
        vs.Object(self.A).is_compatible(vs.Object(self.A).noneable()))
    self.assertFalse(vs.Object(self.B).is_compatible(vs.Object(self.A)))
    self.assertFalse(vs.Object(self.B).is_compatible(vs.Object(self.C)))

  def test_extend(self):
    # Before a forward declaration can be resolved, `extend` will succeed
    self.assertEqual(
        vs.Object('A').extend(vs.Object('B')), vs.Object('A'))

    # When forward declaration is resolved, `extend` will follow class
    # relationships.
    with simulate_forward_declaration(self.A, self.B):
      with self.assertRaisesRegex(
          TypeError, '.* cannot extend .*: incompatible class.'):
        vs.Object('A').extend(vs.Object('B'))

    self.assertEqual(
        vs.Object(self.B).extend(vs.Object(self.A)),
        vs.Object(self.B))
    self.assertEqual(
        vs.Object(self.A).extend(vs.Callable([vs.Any()])),
        vs.Object(self.A))
    self.assertEqual(
        vs.Object(self.A).extend(vs.Callable(kw=[('a', vs.Any())])),
        vs.Object(self.A))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible class.'):
      vs.Object(self.A).extend(vs.Object(self.B))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Object(self.A).extend(vs.Callable([vs.Any(), vs.Any(), vs.Any()]))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Object(self.A).extend(vs.Callable(kw=[('b', vs.Any())]))

  def test_freeze(self):
    self.assertFalse(vs.Object(self.A).frozen)

    a = self.A()
    v = vs.Object(self.A).freeze(a)
    self.assertTrue(v.frozen)
    self.assertIs(v.default, a)
    self.assertIs(v.apply(a), a)
    self.assertIs(v.apply(typed_missing.MISSING_VALUE), a)
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply(self.B())

    b = self.B()
    v = vs.Object(self.A, default=b).freeze()
    self.assertTrue(v.frozen)
    self.assertIs(v.default, b)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Object(self.A).extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Object(self.A).freeze()


class CallableTest(unittest.TestCase):
  """Tests for `Callable`."""

  def test_value_type(self):
    self.assertIsNone(vs.Callable().value_type)
    self.assertEqual(vs.Functor().annotation, object_utils.Functor)

  def test_forward_refs(self):
    self.assertEqual(vs.Callable().forward_refs, set())
    self.assertEqual(
        vs.Callable([vs.Int(), vs.Object('A')],
                    kw=[('x', vs.Object('B'))],
                    returns=vs.Object('C')).forward_refs,
        set([forward_ref('A'), forward_ref('B'), forward_ref('C')]))

  def test_type_resolved(self):
    self.assertTrue(vs.Callable().type_resolved)
    self.assertTrue(vs.Callable([vs.Object('BoolTest')]).type_resolved)
    self.assertTrue(
        vs.Callable(kw=[('x', vs.Object('BoolTest'))]).type_resolved)
    self.assertTrue(vs.Callable(returns=vs.Object('BoolTest')).type_resolved)

    self.assertFalse(vs.Callable([vs.Object('A')]).type_resolved)
    self.assertFalse(
        vs.Callable(kw=[('x', vs.Object('A'))]).type_resolved)
    self.assertFalse(vs.Callable(returns=vs.Object('A')).type_resolved)

    class A:
      pass

    with simulate_forward_declaration(A):
      self.assertTrue(vs.Callable([vs.Object('A')]).type_resolved)
      self.assertTrue(
          vs.Callable(kw=[('x', vs.Object('A'))]).type_resolved)
      self.assertTrue(vs.Callable(returns=vs.Object('A')).type_resolved)

  def test_default(self):
    self.assertEqual(vs.Callable().default, typed_missing.MISSING_VALUE)
    func = lambda x: x
    self.assertIs(vs.Callable(default=func).default, func)

  def test_noneable(self):
    self.assertFalse(vs.Callable().is_noneable)
    self.assertTrue(vs.Callable().noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Callable()), 'Callable()')
    self.assertEqual(
        str(
            vs.Callable(
                args=[vs.Int(), vs.Int()],
                kw=[('a', vs.Str().noneable())],
                returns=vs.Int())),
        'Callable(args=[Int(), Int()], kw=[(\'a\', '
        'Str(default=None, noneable=True))], returns=Int())')
    self.assertEqual(
        str(
            vs.Callable(
                args=[vs.Int(), vs.Int()],
                kw=[('a', vs.Str().noneable())],
                returns=vs.Int()).noneable().freeze()),
        'Callable(args=[Int(), Int()], kw=[(\'a\', '
        'Str(default=None, noneable=True))], returns=Int(), default=None, '
        'noneable=True, frozen=True)')

  def test_annotation(self):
    self.assertEqual(vs.Callable().annotation, typing.Callable[[], None])
    self.assertEqual(
        vs.Callable([vs.Int(), vs.Bool()], returns=vs.Int()).annotation,
        typing.Callable[[int, bool], int])
    self.assertEqual(
        vs.Callable(kw=[('x', vs.Int())], returns=vs.Int()).annotation,
        typing.Callable[..., int])

  def test_eq(self):
    func = vs.Callable()
    self.assertEqual(func, func)
    self.assertEqual(vs.Callable(), vs.Callable())
    self.assertEqual(vs.Callable().noneable(), vs.Callable().noneable())
    self.assertEqual(
        vs.Callable(
            args=[vs.Str()], kw=[('a', vs.Int())], returns=vs.Any()).noneable(),
        vs.Callable(
            args=[vs.Str()], kw=[('a', vs.Int())], returns=vs.Any()).noneable())
    self.assertNotEqual(vs.Callable().noneable(), vs.Callable())
    self.assertNotEqual(vs.Callable(args=[vs.Int()]), vs.Callable())
    self.assertNotEqual(
        vs.Callable(kw=[('b', vs.Int())]),
        vs.Callable(kw=[('a', vs.Int())]))
    self.assertNotEqual(vs.Callable(returns=vs.Int()), vs.Callable())
    self.assertNotEqual(vs.Functor(), vs.Callable())
    self.assertNotEqual(vs.Functor(), vs.Callable())

  def test_bad_init(self):
    with self.assertRaisesRegex(
        TypeError, '\'args\' should be a list of ValueSpec objects.'):
      vs.Callable(1)

    with self.assertRaisesRegex(
        TypeError, '\'args\' should be a list of ValueSpec objects.'):
      vs.Callable([1])

    with self.assertRaisesRegex(
        TypeError, '\'kw\' should be a list of \\(name, value_spec\\) tuples'):
      vs.Callable(kw='a')

    with self.assertRaisesRegex(
        TypeError, '\'kw\' should be a list of \\(name, value_spec\\) tuples'):
      vs.Callable(kw=['a'])

    with self.assertRaisesRegex(
        TypeError, '\'kw\' should be a list of \\(name, value_spec\\) tuples'):
      vs.Callable(kw=[('a', 1)])

    with self.assertRaisesRegex(
        TypeError, '\'returns\' should be a ValueSpec object'):
      vs.Callable(returns=1)

  def test_apply_on_functions(self):
    self.assertIsNone(vs.Callable().noneable().apply(None))
    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Callable().apply(None)

    with self.assertRaisesRegex(TypeError, 'Value is not callable'):
      vs.Callable().apply(1)

    # Apply on function without wildcard arguments.
    f = lambda x: x
    self.assertEqual(vs.Callable().apply(f), f)
    self.assertEqual(vs.Callable([vs.Int()]).apply(f), f)
    self.assertEqual(vs.Callable(kw=[('x', vs.Int())]).apply(f), f)
    with self.assertRaisesRegex(
        TypeError, '.* only take 1 positional arguments, while 2 is required'):
      vs.Callable([vs.Int(), vs.Int()]).apply(f)
    with self.assertRaisesRegex(
        TypeError, 'Keyword argument \'y\' does not exist in .*'):
      vs.Callable(kw=[('y', vs.Int())]).apply(f)

    # Apply on function with wildcard positional args.
    f = lambda *args: sum(args)
    self.assertEqual(vs.Callable().apply(f), f)
    self.assertEqual(vs.Callable([vs.Int()]).apply(f), f)
    with self.assertRaisesRegex(TypeError,
                                'Keyword argument \'y\' does not exist in .*'):
      vs.Callable(kw=[('y', vs.Int())]).apply(f)

    with self.assertRaisesRegex(TypeError, 'Expect .*Functor'):
      vs.Functor().apply(f)

    # Apply on function with wildcard keyword args.
    f = lambda **kwargs: sum(kwargs.values())
    self.assertEqual(vs.Callable().apply(f), f)
    self.assertEqual(
        vs.Callable(kw=[('a', vs.Int())]).apply(f), f)
    with self.assertRaisesRegex(
        TypeError, '.* only take 0 positional arguments, while 1 is required'):
      vs.Callable([vs.Int()]).apply(f)

  def test_apply_on_callable_object(self):

    class CallableObject:

      def __call__(self, x, y):
        pass

    f = CallableObject()
    self.assertEqual(vs.Callable().apply(f), f)
    self.assertEqual(vs.Callable([vs.Int()]).apply(f), f)
    self.assertEqual(vs.Callable(kw=[('x', vs.Int())]).apply(f), f)

    with self.assertRaisesRegex(
        TypeError, '.* only take 2 positional arguments, while 3 is required'):
      vs.Callable([vs.Int(), vs.Int(), vs.Int()]).apply(f)

    with self.assertRaisesRegex(
        TypeError, 'Keyword argument \'z\' does not exist'):
      vs.Callable(kw=[('z', vs.Int())]).apply(f)

    with self.assertRaisesRegex(TypeError, 'Expect .*Functor'):
      vs.Functor().apply(f)

    self.assertEqual(vs.Callable().apply(CallableObject), CallableObject)

  def test_apply_with_user_validator(self):

    class CallableObject:
      def __init__(self, value):
        self.value = value

      def __call__(self):
        return self.value + 1

    def _value_is_one(func):
      if func.value != 1:
        raise ValueError('Value should be one')

    f = CallableObject(1)
    self.assertIs(vs.Callable(user_validator=_value_is_one).apply(f), f)
    with self.assertRaisesRegex(ValueError, 'Value should be one \\(path=\\)'):
      vs.Callable(user_validator=_value_is_one).apply(CallableObject(0))

  def test_apply_on_functor(self):

    class FunctorWithRegularArgs(object_utils.Functor):

      signature = callable_signature.Signature(
          callable_type=callable_signature.CallableType.FUNCTION,
          name='foo',
          module_name='__main__',
          args=[
              callable_signature.Argument('a', vs.Int()),
              callable_signature.Argument('b', vs.Str())
          ])

      def __init__(self, value):
        self.value = value

      def __call__(self, a, b):
        del a, b

    f = FunctorWithRegularArgs(1)
    self.assertEqual(vs.Callable().apply(f), f)
    self.assertEqual(vs.Callable([vs.Int()]).apply(f), f)
    self.assertEqual(
        vs.Callable([vs.Int(), vs.Str()]).apply(f), f)
    self.assertEqual(
        vs.Callable(kw=[('a', vs.Int())]).apply(f), f)
    self.assertEqual(vs.Functor().apply(f), f)
    self.assertEqual(vs.Functor([vs.Int()]).apply(f), f)
    self.assertEqual(vs.Functor(returns=vs.Any()).apply(f), f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of positional argument 0 is not compatible'):
      vs.Callable([vs.Str()]).apply(f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of keyword argument \'b\' is not compatible'):
      vs.Callable(kw=[('b', vs.Int())]).apply(f)

    with self.assertRaisesRegex(
        TypeError, '.* only take 2 positional arguments, while 3 is required'):
      vs.Callable([vs.Int(), vs.Str(), vs.Int()]).apply(f)

    with self.assertRaisesRegex(TypeError,
                                'Keyword argument \'c\' does not exist'):
      vs.Callable(kw=[('c', vs.Int())]).apply(f)

  def test_apply_on_functor_with_varargs(self):

    class FunctorWithVarArgs(object_utils.Functor):

      signature = callable_signature.Signature(
          callable_type=callable_signature.CallableType.FUNCTION,
          name='foo',
          module_name='__main__',
          args=[
              callable_signature.Argument('a', vs.Int()),
              callable_signature.Argument('b', vs.Str())
          ],
          varargs=callable_signature.Argument('args', vs.Int()),
          varkw=callable_signature.Argument('kwargs', vs.Int()),
          return_value=vs.Object(ValueError))

      def __init__(self, value):
        self.value = value

      def __call__(self, a, b, *args, **kwargs):
        del a, b, args, kwargs

    f = FunctorWithVarArgs(1)
    self.assertEqual(vs.Callable().apply(f), f)
    self.assertEqual(vs.Callable([vs.Int()]).apply(f), f)
    self.assertEqual(vs.Callable([vs.Int(), vs.Str(), vs.Int()]).apply(f), f)
    self.assertEqual(vs.Callable(kw=[('a', vs.Int())]).apply(f), f)
    self.assertEqual(vs.Callable(kw=[('c', vs.Int())]).apply(f), f)
    self.assertEqual(vs.Functor().apply(f), f)
    self.assertEqual(vs.Functor([vs.Int()]).apply(f), f)
    self.assertEqual(vs.Functor(returns=vs.Any()).apply(f), f)
    self.assertEqual(vs.Functor(returns=vs.Object(Exception)).apply(f), f)
    self.assertEqual(vs.Functor(returns=vs.Object(ValueError)).apply(f), f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of positional argument 0 is not compatible'):
      vs.Callable([vs.Str()]).apply(f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of keyword argument \'b\' is not compatible'):
      vs.Callable(kw=[('b', vs.Int())]).apply(f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of positional argument 2 is not compatible '
        'with the value spec of \\*args'):
      vs.Callable([vs.Int(), vs.Str(), vs.Str()]).apply(f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of keyword argument \'c\' is not compatible '
        'with the value spec of \\*\\*kwargs'):
      vs.Callable(kw=[('c', vs.Str())]).apply(f)

    with self.assertRaisesRegex(
        TypeError, 'Value spec for return value is not compatible'):
      vs.Callable(returns=vs.Object(KeyError)).apply(f)

  def test_is_compatible(self):

    class CallableObject:

      def __call__(self, x, y):
        return x + y

    self.assertTrue(vs.Callable().noneable().is_compatible(vs.Callable()))
    self.assertTrue(vs.Callable().is_compatible(vs.Callable([vs.Int()])))
    self.assertTrue(vs.Callable().is_compatible(
        vs.Callable(kw=[('a', vs.Int())])))
    self.assertTrue(vs.Callable().is_compatible(
        vs.Functor(kw=[('a', vs.Int())])))

    self.assertTrue(vs.Callable().is_compatible(vs.Object(CallableObject)))
    self.assertTrue(
        vs.Callable(kw=[('x', vs.Any())]).is_compatible(
            vs.Object(CallableObject)))
    self.assertTrue(
        vs.Callable(kw=[('x', vs.Any()), ('y', vs.Any())]).is_compatible(
            vs.Object(CallableObject)))

    self.assertFalse(
        vs.Callable([vs.Int()]).is_compatible(
            vs.Callable(kw=[('a', vs.Int())])))
    self.assertFalse(vs.Callable().is_compatible(vs.Callable().noneable()))
    self.assertFalse(vs.Callable([vs.Int()]).is_compatible(vs.Callable()))
    self.assertFalse(
        vs.Callable([vs.Int(min_value=0)]).is_compatible(
            vs.Callable([vs.Int(max_value=-1)])))
    self.assertFalse(
        vs.Callable(kw=[('a', vs.Int())]).is_compatible(vs.Callable()))
    self.assertFalse(
        vs.Callable(kw=[('a', vs.Int())]).is_compatible(
            vs.Callable(kw=[('b', vs.Int())])))
    self.assertFalse(
        vs.Callable(kw=[('a', vs.Int())]).is_compatible(vs.Object(Exception)))
    self.assertFalse(
        vs.Callable(kw=[('a', vs.Int())]).is_compatible(
            vs.Callable(kw=[('a', vs.Str())])))
    self.assertFalse(
        vs.Callable(returns=vs.Int()).is_compatible(
            vs.Callable(returns=vs.Str())))

  def test_extend(self):
    self.assertEqual(
        vs.Callable().extend(vs.Callable().noneable()), vs.Callable())
    self.assertEqual(
        vs.Callable(kw=[('a', vs.Str())]).extend(
            vs.Callable([vs.Int()], returns=vs.Any())),
        vs.Callable([vs.Int()], kw=[('a', vs.Str())], returns=vs.Any()))

  def test_freeze(self):
    self.assertFalse(vs.Callable().frozen)

    f = lambda: 1
    v = vs.Callable().freeze(f)
    self.assertTrue(v.frozen)
    self.assertIs(v.default, f)
    self.assertIs(v.apply(f), f)
    self.assertIs(v.apply(typed_missing.MISSING_VALUE), f)
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply(lambda: 2)

    v = vs.Callable().freeze(f)
    self.assertTrue(v.frozen)
    self.assertIs(v.default, f)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Callable().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Callable().freeze()


class TypeTest(unittest.TestCase):
  """Tests for `Type`."""

  def test_init(self):
    with self.assertRaisesRegex(TypeError, '.* is not a type'):
      _ = vs.Type(1)

  def test_value_type(self):
    self.assertEqual(vs.Type(Exception).value_type, type)
    self.assertEqual(vs.Type('A').value_type, type)

  def test_forward_refs(self):
    self.assertEqual(vs.Type(Exception).forward_refs, set())
    self.assertEqual(vs.Type('A').forward_refs, set([forward_ref('A')]))

  def test_type_resolved(self):
    self.assertTrue(vs.Type(Exception).type_resolved)
    self.assertTrue(vs.Type('BoolTest').type_resolved)
    self.assertFalse(vs.Type('A').type_resolved)

  def test_type(self):
    self.assertIs(vs.Type(int).type, int)
    self.assertIs(vs.Type(Exception).type, Exception)
    self.assertIs(vs.Type('BoolTest').type, BoolTest)

    class G(typing.Generic[typing.TypeVar('T1'), typing.TypeVar('T2')]):
      pass

    class G1(G[int, str]):
      pass

    self.assertIs(vs.Type(G).type, G)
    self.assertIs(vs.Type(G[int, str]).type, G[int, str])
    self.assertIs(vs.Type(G[int, str]).apply(G1), G1)

    with self.assertRaisesRegex(TypeError, '.* does not exist'):
      _ = vs.Type('A').type

    class A:
      pass

    with simulate_forward_declaration(A):
      self.assertIs(vs.Type('A').type, A)

  def test_default(self):
    self.assertEqual(vs.Type(Exception).default, typed_missing.MISSING_VALUE)
    self.assertEqual(vs.Type(Exception, default=ValueError).default, ValueError)

  def test_noneable(self):
    self.assertFalse(vs.Type(Exception).is_noneable)
    self.assertTrue(vs.Type(Exception).noneable().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Type(Exception)), 'Type(<class \'Exception\'>)')
    self.assertEqual(
        str(vs.Type(Exception).noneable()),
        'Type(<class \'Exception\'>, default=None, noneable=True)')
    self.assertEqual(
        str(vs.Type(Exception).noneable().freeze()),
        'Type(<class \'Exception\'>, default=None, noneable=True, frozen=True)')

  def test_annotation(self):
    self.assertEqual(vs.Type(Exception).annotation, typing.Type[Exception])
    self.assertEqual(vs.Type('BoolTest').annotation, typing.Type[BoolTest])
    self.assertEqual(vs.Type('A').annotation, typing.Type['A'])
    self.assertEqual(
        vs.Type(Exception).noneable().annotation,
        typing.Optional[typing.Type[Exception]])

  def test_eq(self):
    t = vs.Type(Exception)
    self.assertEqual(t, t)
    self.assertEqual(vs.Type(Exception), vs.Type(Exception))
    self.assertEqual(vs.Type('A'), vs.Type('A'))

    class A:
      pass

    self.assertNotEqual(vs.Type('A'), vs.Type(A))
    with simulate_forward_declaration(A):
      self.assertEqual(vs.Type('A'), vs.Type(A))

    self.assertEqual(
        vs.Type(Exception).noneable(),
        vs.Type(Exception).noneable())
    self.assertEqual(
        vs.Type(Exception, default=ValueError),
        vs.Type(Exception, default=ValueError))
    self.assertNotEqual(vs.Type(Exception), vs.Type(int))
    self.assertNotEqual(
        vs.Type(Exception),
        vs.Type(Exception).noneable())
    self.assertNotEqual(
        vs.Type(Exception), vs.Type(Exception, default=ValueError))

  def test_apply(self):
    self.assertEqual(vs.Type(Exception).apply(Exception), Exception)
    self.assertEqual(vs.Type(Exception).apply(ValueError), ValueError)
    self.assertIsNone(vs.Type(Exception).noneable().apply(None))

    v = vs.Type('A')

    # Before 'A' can be resolved, v could accept any type.
    self.assertIs(v.apply(int), int)

    class A:
      pass

    with simulate_forward_declaration(A):
      self.assertIs(v.apply(A), A)

      # After 'A' can be resolved, v could only accept 'A'.
      with self.assertRaisesRegex(ValueError, '.* is not a subclass of .*'):
        _ = v.apply(int)

    with self.assertRaisesRegex(ValueError, '.* is not a subclass of .*'):
      vs.Type(Exception).apply(int)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Type(Exception).apply(None)

  def test_is_compatible(self):
    self.assertTrue(vs.Type(Exception).is_compatible(vs.Type(Exception)))
    self.assertTrue(vs.Type(Exception).is_compatible(vs.Type(ValueError)))

    # Before `A` can be accepted, it's always considered compatible.
    self.assertTrue(vs.Type('A').is_compatible(vs.Type(Exception)))
    self.assertTrue(vs.Type(Exception).is_compatible(vs.Type('A')))

    # After `A` can be accepted, it can only accept subclasses of `A`.
    class A:
      pass

    class B(A):
      pass

    with simulate_forward_declaration(A, B):
      self.assertTrue(vs.Type('A').is_compatible(vs.Type('A')))
      self.assertTrue(vs.Type('A').is_compatible(vs.Type('B')))
      self.assertFalse(vs.Type('B').is_compatible(vs.Type('A')))
      self.assertFalse(vs.Type(Exception).is_compatible(vs.Type('A')))
      self.assertFalse(vs.Type('A').is_compatible(vs.Type(Exception)))

    self.assertTrue(
        vs.Type(Exception).noneable().is_compatible(vs.Type(ValueError)))
    self.assertFalse(
        vs.Type(Exception).is_compatible(vs.Type(ValueError).noneable()))
    self.assertFalse(vs.Type(Exception).is_compatible(vs.Type(int)))

  def test_extend(self):
    # Child may make a parent default value not specified.
    self.assertEqual(
        vs.Type(Exception).extend(
            vs.Type(Exception, default=ValueError)).default,
        typed_missing.MISSING_VALUE)

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(
        vs.Type(Exception).extend(vs.Type(Exception).noneable()).is_noneable)

    # Child cannot extend a base of different type.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Type(Exception).extend(vs.Type(ValueError))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Type(Exception).noneable().extend(vs.Type(Exception))

  def test_freeze(self):
    self.assertFalse(vs.Type(Exception).frozen)

    e = ValueError
    v = vs.Type(Exception).freeze(e)
    self.assertTrue(v.frozen)
    self.assertIs(v.default, e)
    self.assertIs(v.apply(e), e)
    self.assertIs(v.apply(typed_missing.MISSING_VALUE), e)
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply(KeyError)

    v = vs.Type(Exception, default=e).freeze()
    self.assertTrue(v.frozen)
    self.assertIs(v.default, e)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Type(Exception).extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Type(Exception).freeze()


class UnionTest(unittest.TestCase):
  """Tests for `Union`."""

  def setUp(self):
    super().setUp()

    class A:

      def __call__(self):
        pass

    class B(A):
      pass

    # pylint: disable=invalid-name
    self.A = A
    self.B = B
    # pylint: enable=invalid-name

  def test_value_type(self):
    self.assertEqual(
        set(vs.Union([vs.Int(), vs.Bool()]).value_type),
        set([int, bool]))
    v = vs.Union([vs.Object('A'), vs.Object('B')])
    with simulate_forward_declaration(self.A, self.B):
      self.assertEqual(set(v.value_type), set([self.A, self.B]))

    v = vs.Union(
        [vs.Int(), vs.Object('A'), vs.Union([vs.Int(), vs.Float()]).noneable()]
    )
    with simulate_forward_declaration(self.A):
      self.assertEqual(set(v.value_type), set((int, self.A, float)))

  def test_forward_refs(self):
    self.assertEqual(
        vs.Union([vs.Int(), vs.Object(self.A)]).forward_refs, set()
    )
    self.assertEqual(
        vs.Union([vs.Int(), vs.Object('Foo')]).forward_refs,
        set([forward_ref('Foo')]),
    )
    self.assertEqual(
        vs.Union([
            vs.Int(),
            vs.Object('Bar'),
            vs.Union([
                vs.Float(),
                vs.Object('Foo'),
            ]).noneable(),
        ]).forward_refs,
        set([forward_ref('Bar'), forward_ref('Foo')]),
    )

  def test_type_resolved(self):
    self.assertTrue(vs.Union([vs.Int(), vs.Float()]).type_resolved)
    self.assertFalse(vs.Union([vs.Object('A'), vs.Float()]).type_resolved)

    class A:
      pass

    with simulate_forward_declaration(A):
      self.assertTrue(vs.Union([vs.Object('A'), vs.Float()]).type_resolved)

  def test_default(self):
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()]).default,
        typed_missing.MISSING_VALUE)
    self.assertIsNone(
        vs.Union([vs.Int(), vs.Bool()]).noneable().default)
    self.assertEqual(vs.Union([vs.Int(), vs.Bool()], 1).default, 1)

  def test_noneable(self):
    self.assertFalse(vs.Union([vs.Int(), vs.Bool()]).is_noneable)
    self.assertTrue(
        vs.Union([vs.Int(), vs.Bool()]).noneable().is_noneable)
    self.assertFalse(
        vs.Union([vs.Int(), vs.Bool()]).candidates[0].is_noneable)
    self.assertTrue(
        vs.Union([vs.Int(), vs.Bool()]).noneable().candidates[0].is_noneable
    )
    self.assertTrue(
        vs.Union([vs.Int().noneable(), vs.Bool()]).is_noneable)

  def test_str(self):
    self.assertEqual(
        repr(vs.Union([vs.Int(), vs.Bool()])), 'Union([Int(), Bool()])')
    self.assertEqual(
        repr(vs.Union([vs.Int(), vs.Bool()], default=1).freeze()),
        'Union([Int(), Bool()], default=1, frozen=True)')
    self.assertEqual(
        repr(vs.Union([vs.Int(), vs.Bool()], default=1).noneable()),
        'Union([Int(default=None, noneable=True), '
        'Bool(default=None, noneable=True)], default=1, noneable=True)')

  def test_annotation(self):
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()]).annotation, typing.Union[int, bool])
    self.assertEqual(
        vs.Union([vs.Int(), vs.Union([vs.Bool(), vs.Str()])]).annotation,
        typing.Union[int, typing.Union[bool, str]])

  def test_eq(self):
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()]),
        vs.Union([vs.Int(), vs.Bool()]))
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()]),
        vs.Union([vs.Bool(), vs.Int()]))
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()], 1),
        vs.Union([vs.Bool(), vs.Int()], 1))

    self.assertNotEqual(
        vs.Union([vs.Int(), vs.Bool()]), vs.Int())
    self.assertNotEqual(
        vs.Union([vs.Int(), vs.Bool()]),
        vs.Union([vs.Int(), vs.Str()]))
    self.assertNotEqual(
        vs.Union([vs.Int(), vs.Bool(), vs.Str()]),
        vs.Union([vs.Int(), vs.Str()]))
    self.assertNotEqual(
        vs.Union([vs.Int(), vs.Bool()]),
        vs.Union([vs.Int(), vs.Bool()]).noneable())
    self.assertNotEqual(
        vs.Union([vs.Int(min_value=0), vs.Bool()]),
        vs.Union([vs.Int(), vs.Bool()]))
    self.assertNotEqual(
        vs.Union([vs.Int(), vs.Bool()], 1),
        vs.Union([vs.Int(), vs.Bool()]))
    self.assertNotEqual(
        vs.Union([vs.Int(), vs.Enum('abc', [True, 'abc'])]),
        vs.Union([vs.Int(), vs.Any(), vs.Bool()]))

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'candidates\' must be a list of at least 2 elements'):
      vs.Union(1)

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'candidates\' must be a list of at least 2 elements'):
      vs.Union([vs.Int()])

    with self.assertRaisesRegex(
        ValueError, 'Items in \'candidates\' must be ValueSpec objects.'):
      vs.Union([1, 2])

    with self.assertRaisesRegex(
        ValueError, 'Found 2 value specs of the same type '):
      vs.Union([vs.Int(min_value=1), vs.Int(max_value=2)])

  def test_get_candidate(self):
    self.assertEqual(
        vs.Union([vs.Int(), vs.Float()]).get_candidate(vs.Float()), vs.Float())

    self.assertIsNone(
        vs.Union([vs.Int(), vs.Float()]).get_candidate(vs.Float(min_value=1)))

    self.assertEqual(
        vs.Union([vs.Object(self.A), vs.Object(self.B)]).get_candidate(
            vs.Object(self.B)),
        vs.Object(self.B))

    self.assertEqual(
        vs.Union([vs.Callable(), vs.Object(self.B)]).get_candidate(
            vs.Object(self.A)),
        vs.Object(self.B))

    self.assertEqual(
        vs.Union([vs.Callable(), vs.Int()]).get_candidate(vs.Callable()),
        vs.Callable())

    self.assertEqual(
        vs.Union([vs.Union([vs.Float(), vs.Int()]), vs.Str()]).get_candidate(
            vs.Int()),
        vs.Int())

    self.assertEqual(
        vs.Union([vs.Callable(), vs.Int()]).get_candidate(vs.Any()),
        vs.Callable())

  def test_apply(self):
    self.assertEqual(vs.Union([vs.Int(), vs.Str()]).apply(1), 1)
    self.assertEqual(
        vs.Union([vs.Int(), vs.Str()]).apply('abc'), 'abc')
    self.assertIsNone(
        vs.Union([vs.Int(), vs.Str()]).noneable().apply(None))
    self.assertEqual(
        vs.Union([vs.Int(), vs.Str()]).apply(
            typed_missing.MISSING_VALUE, allow_partial=True),
        typed_missing.MISSING_VALUE)

    v = vs.Union([vs.Int(), vs.Object('A')])

    # Before `A` could be resolved, `v` can accept anything.
    self.assertEqual(v.apply('foo'), 'foo')

    # After `A` is resolved, `v` can only accept subclasses of A.
    class A:
      pass

    class B(A):
      pass

    b = B()
    with simulate_forward_declaration(A):
      self.assertIs(v.apply(b), b)
      with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
        _ = v.apply('foo')

    # Bad cases.
    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      vs.Union([vs.Int(), vs.Str()]).apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect \\(.*\\) but encountered <(type|class) \'list\'>.'):
      vs.Union([vs.Int(), vs.Str()]).apply([])

  def test_is_compatible(self):
    self.assertTrue(
        vs.Union([vs.Int(), vs.Bool()]).is_compatible(
            vs.Union([vs.Bool(), vs.Int()])))

    self.assertTrue(
        vs.Union([vs.Int(), vs.Bool(), vs.Str()]).is_compatible(
            vs.Union([vs.Int(), vs.Str()])))

    self.assertTrue(
        vs.Union([vs.Object(self.A), vs.Bool()]).is_compatible(
            vs.Union([vs.Object(self.B), vs.Bool()])))

    self.assertTrue(
        vs.Union([vs.Int(), vs.Bool()]).noneable().is_compatible(
            vs.Int(min_value=1).noneable()))

    self.assertFalse(vs.Union([vs.Int(), vs.Bool()]).is_compatible(vs.Str()))

    self.assertFalse(
        vs.Union([vs.Int(min_value=1), vs.Bool()]).is_compatible(vs.Int()))

    self.assertFalse(
        vs.Union([vs.Int(min_value=1), vs.Bool()]).is_compatible(
            vs.Union([vs.Int(), vs.Bool()])))

    self.assertFalse(
        vs.Union([vs.Int(), vs.Bool()]).is_compatible(vs.Int().noneable()))

    self.assertFalse(
        vs.Union([vs.Object(self.B), vs.Bool()]).is_compatible(
            vs.Object(self.A)))

  def test_extend(self):
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()]).extend(
            vs.Union([vs.Bool(), vs.Int(min_value=0)])).candidates[0],
        vs.Int(min_value=0))

    # Narrow constraint.
    self.assertEqual(
        vs.Union([vs.Int(), vs.Bool()]).extend(
            vs.Union([vs.Bool(), vs.Str(), vs.Int(min_value=0)])),
        vs.Union([vs.Int(min_value=0), vs.Bool()]))

    self.assertFalse(
        vs.Union([vs.Int(), vs.Str()]).extend(
            vs.Union([vs.Int(), vs.Str()]).noneable()).is_noneable)

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type'):
      vs.Union([vs.Int(), vs.Bool()]).extend(vs.Int())

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible value spec .*'):
      vs.Union([vs.Int(), vs.Bool()]).extend(
          vs.Union([vs.Str(), vs.Bool()]))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      vs.Union([vs.Int(), vs.Bool()]).noneable().extend(
          vs.Union([vs.Int(), vs.Bool()]))

    # Test enum of different values cannot be extended.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: values in base should be super set.'):
      vs.Union([vs.Enum(1, [1, 2]), vs.Int()]).extend(
          vs.Union([vs.Enum('a', ['a', 'b']), vs.Int()]))

    # Child with larger max_size cannot extend base with smaller max_size.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: max_value is larger.'):
      vs.Union([vs.Int(max_value=100), vs.Bool()]).extend(
          vs.Union([vs.Int(max_value=0), vs.Bool()]))

    # Test vs.Object
    vs.Union([vs.Object(int), vs.Object(float)])
    with self.assertRaisesRegex(
        ValueError, 'Found 2 value specs of the same type'):
      vs.Union([vs.Object(int), vs.Object(int)])

  def test_freeze(self):
    self.assertFalse(vs.Union([vs.Int(), vs.Str()]).frozen)

    v = vs.Union([vs.Int(), vs.Str()]).freeze(1)
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 1)
    self.assertEqual(v.apply(1), 1)
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), 1)
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply(2)

    v = vs.Union([vs.Int(), vs.Str()], default='foo').freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 'foo')

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Str().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Union([vs.Int(), vs.Str()]).freeze()


class AnyTest(unittest.TestCase):
  """Tests for `Any`."""

  def test_value_type(self):
    self.assertEqual(vs.Any().value_type, object)

  def test_default(self):
    self.assertEqual(vs.Any().default, typed_missing.MISSING_VALUE)
    self.assertIsNone(vs.Any().noneable().default)
    self.assertEqual(vs.Any(True).default, True)

  def test_noneable(self):
    self.assertTrue(vs.Any().is_noneable)

  def test_str(self):
    self.assertEqual(str(vs.Any()), 'Any()')
    self.assertEqual(str(vs.Any(1)), 'Any(default=1)')
    self.assertEqual(str(vs.Any(1).freeze()), 'Any(default=1, frozen=True)')

  def test_annotation(self):
    self.assertEqual(vs.Any().annotation, typed_missing.MISSING_VALUE)
    self.assertEqual(
        vs.Any().noneable().annotation, typed_missing.MISSING_VALUE
    )
    self.assertEqual(vs.Any(annotation=int).noneable().annotation, int)

  def test_eq(self):
    self.assertEqual(vs.Any(), vs.Any())
    self.assertEqual(vs.Any(True), vs.Any(True))
    self.assertNotEqual(vs.Any(), vs.Int())
    self.assertNotEqual(vs.Any(True), vs.Any())

  def test_apply(self):
    self.assertEqual(vs.Any().apply(True), True)
    self.assertEqual(vs.Any().apply(1), 1)
    self.assertIsNone(vs.Any().apply(None))

  def test_apply_with_user_validator(self):
    def _value_is_none(value):
      if value is not None:
        raise ValueError('Value should be None.')

    self.assertIsNone(vs.Any(user_validator=_value_is_none).apply(None))

    with self.assertRaisesRegex(
        ValueError, 'Value should be None. \\(path=\\)'):
      vs.Any(user_validator=_value_is_none).apply(1)

  def test_is_compatible(self):
    self.assertTrue(vs.Any().is_compatible(vs.Int()))
    self.assertTrue(vs.Any().is_compatible(vs.Int().noneable()))

  def test_extend(self):
    # Child may change default value.
    self.assertEqual(vs.Any(False).extend(vs.Any(True)).default, False)

    # Child may make a parent default value not specified.
    self.assertTrue(vs.Any().extend(vs.Any(True)).default)

    # Child cannot extend a base with different type.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type.'):
      vs.Any().extend(vs.Int())

  def test_freeze(self):
    self.assertFalse(vs.Any().frozen)

    v = vs.Any().freeze(1)
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 1)
    self.assertEqual(v.apply(1), 1)
    self.assertEqual(v.apply(typed_missing.MISSING_VALUE), 1)
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      v.apply(2)

    v = vs.Any(default='foo').freeze()
    self.assertTrue(v.frozen)
    self.assertEqual(v.default, 'foo')

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      vs.Any().extend(v)

    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value.'):
      vs.Any().freeze()


@contextlib.contextmanager
def simulate_forward_declaration(*module_level_symbols):
  try:
    for symbol in module_level_symbols:
      setattr(sys.modules[__name__], symbol.__name__, symbol)
    yield
  finally:
    for symbol in module_level_symbols:
      delattr(sys.modules[__name__], symbol.__name__)


def forward_ref(name):
  return class_schema.ForwardRef(sys.modules[__name__], name)


if __name__ == '__main__':
  unittest.main()
