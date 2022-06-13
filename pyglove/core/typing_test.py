# Copyright 2019 The PyGlove Authors
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
"""Tests pyglove.schema."""

import calendar
import copy
import datetime
import inspect
import typing
import unittest
from pyglove.core import object_utils
from pyglove.core import typing as schema


class FieldTest(unittest.TestCase):
  """Test for schema.Field class."""

  def testBasics(self):
    """Test basic methods of Field."""
    f = schema.Field('a', schema.Int(max_value=10), 'a field')
    self.assertIsInstance(f.key, schema.ConstStrKey)
    self.assertEqual(f.key, 'a')
    self.assertEqual(f.value, schema.Int(max_value=10))
    self.assertFalse(f.frozen)
    self.assertIs(f.annotation, int)
    self.assertEqual(f.default_value, object_utils.MISSING_VALUE)
    self.assertEqual(f.description, 'a field')
    self.assertIsInstance(f.metadata, dict)
    self.assertEqual(len(f.metadata), 0)

    # Cover the self comparison in __eq__.
    self.assertEqual(f, f)

    # Test field with metadata.
    f = schema.Field('a', schema.Bool(), 'a field', {'meta1': 'foo'})
    self.assertEqual(f.metadata['meta1'], 'foo')

    with self.assertRaisesRegex(ValueError, 'metadata must be a dict.'):
      schema.Field('a', schema.Bool(), 'a field', 'abc')

  def testExtend(self):
    """Test for Field.extend."""
    # Extend value spec and description.
    self.assertEqual(
        schema.Field('a', schema.Int(1)).extend(
            schema.Field('a', schema.Int(min_value=0), 'field a')),
        schema.Field('a', schema.Int(min_value=0, default=1), 'field a'))

    # Overrided description will not inherit.
    self.assertEqual(
        schema.Field('a', schema.Int(1), 'overrided field a').extend(
            schema.Field('a', schema.Int(min_value=0), 'field a')),
        schema.Field('a', schema.Int(min_value=0, default=1),
                     'overrided field a'))

    # Extend metadata.
    self.assertEqual(
        schema.Field('a', schema.Int(), None, {
            'b': 1
        }).extend(schema.Field('a', schema.Int(), 'field a', {'a': 2})),
        schema.Field('a', schema.Int(), 'field a', {
            'a': 2,
            'b': 1
        }))

    # Extend with base field with different key is not allowed.
    with self.assertRaisesRegex(KeyError,
                                '.* cannot extend .* for keys are different.'):
      schema.Field('a', schema.Int()).extend(schema.Field('b', schema.Int()))

  def testApply(self):
    """Test for Field.validate."""
    self.assertEqual(schema.Field('a', schema.Int()).apply(1), 1)

    self.assertEqual(
        schema.Field('a',
                     schema.Dict([('b', schema.Int()),
                                  ('c', schema.Bool())])).apply({
                                      'b': 1,
                                      'c': True
                                  }), {
                                      'b': 1,
                                      'c': True
                                  })

    self.assertEqual(
        schema.Field('a',
                     schema.Dict([('b', schema.Int()), ('c', schema.Bool())
                                 ])).apply({
                                     'b': 1,
                                 }, allow_partial=True), {
                                     'b': 1,
                                     'c': object_utils.MISSING_VALUE
                                 })

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'int\'> but encountered '
        '<(type|class) \'float\'>'):
      schema.Field('a', schema.Int()).apply(1.0)

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified. \\(Path=\'c\'.*\\)'):
      schema.Field('a', schema.Dict([('b', schema.Int()), ('c', schema.Bool())
                                    ])).apply({
                                        'b': 1,
                                    }, allow_partial=False)

  def testFormat(self):
    """Test for Field.format."""
    self.assertEqual(
        schema.Field('a', schema.Dict([('b', schema.Int())]),
                     'this is a very long field.', {
                         'm1': 1,
                         'm2': 2,
                         'm3': 3,
                         'm4': 4,
                         'm5': 5
                     }).format(compact=True, verbose=False),
        'Field(key=a, value=Dict({b=Int()}), '
        'description=\'this is a very long ...\', '
        'metadata={...})')

    self.assertEqual(
        schema.Field('a', schema.Dict([('b', schema.Int())]),
                     'this is a very long field.', {
                         'm1': 1,
                         'm2': 2,
                         'm3': 3,
                         'm4': 4,
                         'm5': 5
                     }).format(compact=True, verbose=True),
        'Field(key=a, value=Dict({b=Int()}), '
        'description=\'this is a very long field.\', '
        'metadata={\'m1\': 1, \'m2\': 2, \'m3\': 3, \'m4\': 4, \'m5\': 5})')

    self.assertEqual(
        schema.Field('a', schema.Dict([('b', schema.Int())]),
                     'field a').format(compact=False, verbose=False),
        'Field(key=a, value=Dict({\n'
        '    b = Int()\n'
        '  }), description=\'field a\')')


class KeySpecTest(unittest.TestCase):
  """Tests for KeySpec implementations."""

  def testConstStrKey(self):
    """Tests for ConstStrKey."""
    key = schema.ConstStrKey('a')
    self.assertEqual(key, key)
    self.assertEqual(key, 'a')
    self.assertEqual(key.text, 'a')
    self.assertNotEqual(key, 'b')
    self.assertIn(key, {'a': 1})
    self.assertEqual(str(key), 'a')
    self.assertEqual(repr(key), 'a')
    self.assertTrue(key.match('a'))
    self.assertFalse(key.match('b'))

    with self.assertRaisesRegex(KeyError, '\'.\' cannot be used in key.'):
      schema.ConstStrKey('a.b')

    # Test extends.
    self.assertEqual(key.extend(schema.ConstStrKey('a')).text, 'a')
    with self.assertRaisesRegex(KeyError,
                                '.* cannot extend .* for keys are different.'):
      key.extend(schema.ConstStrKey('b'))

  def testStrKey(self):
    """Tests for StrKey."""
    key = schema.StrKey()
    self.assertEqual(key, key)
    self.assertEqual(key, schema.StrKey())
    self.assertTrue(key.match('a'))
    self.assertTrue(key.match('abc'))
    self.assertFalse(key.match(1))

    regex_key = schema.StrKey('a.*')
    self.assertTrue(regex_key.match('a1'))
    self.assertTrue(regex_key.match('a'))
    self.assertFalse(regex_key.match('b'))
    self.assertFalse(regex_key.match({}))

    # Test extends.
    self.assertIsNone(key.extend(schema.StrKey()).regex)
    self.assertEqual(
        regex_key.extend(schema.StrKey('a.*')).regex.pattern, 'a.*')

    with self.assertRaisesRegex(KeyError,
                                '.* cannot extend .* for keys are different.'):
      key.extend(schema.StrKey(regex='.*'))

  def testListKey(self):
    """Tests for ListKey."""
    self.assertEqual(schema.ListKey(), schema.ListKey())
    self.assertEqual(schema.ListKey(), schema.ListKey(min_value=0))
    self.assertEqual(
        schema.ListKey(min_value=0, max_value=10), schema.ListKey(max_value=10))
    self.assertEqual(schema.ListKey(max_value=10), schema.ListKey(max_value=10))
    self.assertNotEqual(
        schema.ListKey(min_value=10), schema.ListKey(min_value=5))
    self.assertNotEqual(
        schema.ListKey(max_value=10), schema.ListKey(max_value=5))
    self.assertNotEqual(
        schema.ListKey(min_value=5), schema.ListKey(max_value=5))

    unbounded_key = schema.ListKey()
    self.assertEqual(unbounded_key, unbounded_key)
    self.assertEqual(unbounded_key.min_value, 0)
    self.assertIsNone(unbounded_key.max_value)

    self.assertTrue(unbounded_key.match(1))
    self.assertTrue(unbounded_key.match(10000))
    self.assertFalse(unbounded_key.match('a'))

    self.assertIsNone(unbounded_key.extend(schema.ListKey()).max_value)
    self.assertEqual(
        unbounded_key.extend(schema.ListKey(max_value=10)).max_value, 10)
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      unbounded_key.extend(schema.TupleKey(1))

    bounded_key = schema.ListKey(min_value=2, max_value=10)
    self.assertTrue(bounded_key.match(2))
    self.assertFalse(bounded_key.match(0))
    self.assertFalse(bounded_key.match(10000))
    self.assertFalse(bounded_key.match('a'))

    self.assertEqual(bounded_key.extend(schema.ListKey()).min_value, 2)
    self.assertEqual(bounded_key.extend(schema.ListKey()).max_value, 10)
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      bounded_key.extend(schema.StrKey())

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: min_value is smaller.'):
      bounded_key.extend(schema.ListKey(min_value=3))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is greater.'):
      bounded_key.extend(schema.ListKey(max_value=5))

  def testTupleKey(self):
    """Tests for TupleKey."""
    tuple_key = schema.TupleKey(0)

    self.assertEqual(tuple_key, tuple_key)
    self.assertEqual(schema.TupleKey(0), schema.TupleKey(0))
    self.assertNotEqual(schema.TupleKey(0), schema.TupleKey(1))

    self.assertTrue(tuple_key.match(0))
    self.assertFalse(tuple_key.match(1))
    self.assertFalse(tuple_key.match('a'))

    self.assertEqual(tuple_key.extend(schema.TupleKey(0)).index, 0)
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      tuple_key.extend(schema.ListKey(10))

    with self.assertRaisesRegex(KeyError,
                                '.* cannot extend .*: unmatched index.'):
      tuple_key.extend(schema.TupleKey(1))


class ValueSpecTest(unittest.TestCase):
  """Tests for ValueSpec implementations."""

  def testBool(self):
    """Tests for Bool."""

    # Test __str__.
    self.assertEqual(str(schema.Bool()), 'Bool()')
    self.assertEqual(str(schema.Bool(True)), 'Bool(default=True)')
    self.assertEqual(str(schema.Bool(True).freeze()),
                     'Bool(default=True, frozen=True)')
    self.assertEqual(
        str(schema.Bool().noneable()), 'Bool(default=None, noneable=True)')
    self.assertEqual(
        str(schema.Bool(True).noneable()), 'Bool(default=True, noneable=True)')

    # Test value_type
    self.assertEqual(schema.Bool().value_type, bool)

    # Test default.
    self.assertEqual(schema.Bool().default, object_utils.MISSING_VALUE)
    self.assertEqual(schema.Bool(True).default, True)

    # Test annotation.
    self.assertEqual(schema.Bool().annotation, bool)
    self.assertEqual(schema.Bool().noneable().annotation, typing.Optional[bool])

    # Test is_noneable
    self.assertFalse(schema.Bool().is_noneable)
    self.assertTrue(schema.Bool().noneable().is_noneable)

    # Test comparison.
    b = schema.Bool()
    self.assertEqual(b, b)
    self.assertEqual(schema.Bool(), schema.Bool())
    self.assertEqual(schema.Bool(True), schema.Bool(True))
    self.assertEqual(schema.Bool().noneable(), schema.Bool().noneable())
    self.assertNotEqual(schema.Bool(True), schema.Int())
    self.assertNotEqual(schema.Bool(True), schema.Bool())
    self.assertNotEqual(schema.Bool().noneable(), schema.Bool())

    # Test apply.
    self.assertTrue(schema.Bool().apply(True))
    self.assertIsNone(schema.Bool().noneable().apply(None))

    # Test is_compatible.
    self.assertTrue(b.is_compatible(b))
    self.assertTrue(schema.Bool().is_compatible(schema.Bool()))
    self.assertTrue(schema.Bool().noneable().is_compatible(schema.Bool()))
    self.assertFalse(schema.Bool().is_compatible(schema.Bool().noneable()))
    self.assertFalse(schema.Bool().is_compatible(schema.Int()))

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'bool\'> but encountered '
        '<(type|class) \'int\'>.'):
      schema.Bool().apply(1)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Bool().apply(None)

    # Test extends.
    # Child may change default value.
    self.assertEqual(
        schema.Bool(False).extend(schema.Bool(True)).default, False)

    # Child may make a parent default value not specified.
    self.assertTrue(schema.Bool().extend(schema.Bool(True)).default)

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(schema.Bool().extend(schema.Bool().noneable()).is_noneable)

    # Child cannot extend a base with different type.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Bool().extend(schema.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Bool().noneable().extend(schema.Bool())

  def testStr(self):
    """Tests for Str."""

    # Test __str__.
    self.assertEqual(str(schema.Str()), 'Str()')
    self.assertEqual(
        str(schema.Str().noneable()), 'Str(default=None, noneable=True)')
    self.assertEqual(str(schema.Str('a')), 'Str(default=\'a\')')
    self.assertEqual(str(schema.Str('a').freeze()),
                     'Str(default=\'a\', frozen=True)')
    self.assertEqual(str(schema.Str(regex='.*')), 'Str(regex=\'.*\')')

    # Test value_type
    self.assertEqual(schema.Str().value_type, str)

    # Test annotation.
    self.assertEqual(schema.Str().annotation, typing.Text)
    self.assertEqual(schema.Str().noneable().annotation,
                     typing.Optional[typing.Text])

    # Test default.
    self.assertEqual(schema.Str().default, object_utils.MISSING_VALUE)
    self.assertEqual(schema.Str('abc').default, 'abc')

    # Test is_noneable
    self.assertFalse(schema.Str().is_noneable)
    self.assertTrue(schema.Str().noneable().is_noneable)

    # Test comparison.
    s = schema.Str()
    self.assertEqual(s, s)
    self.assertEqual(schema.Str(), schema.Str())
    self.assertEqual(schema.Str().noneable(), schema.Str().noneable())
    self.assertEqual(schema.Str('a'), schema.Str('a'))
    self.assertEqual(schema.Str('a', '.*'), schema.Str('a', '.*'))
    self.assertEqual(schema.Str(regex='a.*'), schema.Str(regex='a.*'))
    self.assertNotEqual(schema.Str(), schema.Bool())
    self.assertNotEqual(schema.Str(), schema.Str().noneable())
    self.assertNotEqual(schema.Str('a'), schema.Str())
    self.assertNotEqual(schema.Str('a'), schema.Str('b'))
    self.assertNotEqual(schema.Str(), schema.Str(regex='.*'))
    self.assertNotEqual(schema.Str(regex='a'), schema.Str(regex='.*'))

    # Test apply.
    self.assertEqual(schema.Str().apply('a'), 'a')
    self.assertEqual(schema.Str(regex='a.*').apply('a1'), 'a1')
    with self.assertRaisesRegex(
        TypeError, 'Expect .*str.* but encountered <(type|class) \'dict\'>.'):
      schema.Str().apply({})

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Str().apply(None)

    with self.assertRaisesRegex(
        ValueError, 'String \'b\' does not match regular expression \'a.*\'.'):
      schema.Str(regex='a.*').apply('b')

    # Test is_compatible.
    self.assertTrue(schema.Str().is_compatible(schema.Str()))
    self.assertTrue(schema.Str().noneable().is_compatible(schema.Str()))
    self.assertTrue(
        schema.Str(regex='.*').is_compatible(schema.Str(regex='.*')))
    # This is a false-positive, but we don't have a good way to check the
    # compatibility of two regular expressions.
    self.assertTrue(
        schema.Str(regex='abc.*').is_compatible(schema.Str(regex='xyz.*')))
    self.assertFalse(schema.Str().is_compatible(schema.Str().noneable()))
    self.assertFalse(schema.Str().is_compatible(schema.Int()))

    # Test extends.
    # Child may make a parent default value not specified.
    self.assertEqual(schema.Str().extend(schema.Str('foo')).default,
                     object_utils.MISSING_VALUE)

    # Child without regular expression remain unchanged.
    self.assertEqual(
        schema.Str(regex='a.*').extend(schema.Str(regex='.*')).regex.pattern,
        'a.*')

    # Child with regular expression remain unchanged.
    self.assertEqual(
        schema.Str(regex='a.*').extend(schema.Str(regex='.*')).regex.pattern,
        'a.*')

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(schema.Str().extend(schema.Str().noneable()).is_noneable)

    # Child cannot extend a base of different type.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Str().extend(schema.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Str().noneable().extend(schema.Str())

  def testInt(self):
    """Tests for Int."""

    # Test __str__.
    self.assertEqual(str(schema.Int()), 'Int()')
    self.assertEqual(str(schema.Int(1)), 'Int(default=1)')
    self.assertEqual(str(schema.Int(1).freeze()),
                     'Int(default=1, frozen=True)')
    self.assertEqual(
        str(schema.Int().noneable()), 'Int(default=None, noneable=True)')
    self.assertEqual(
        str(schema.Int(1).noneable()), 'Int(default=1, noneable=True)')
    self.assertEqual(
        str(schema.Int(min_value=0, max_value=1)), 'Int(min=0, max=1)')

    # Test value_type
    self.assertEqual(schema.Int().value_type, int)

    # Test annotation.
    self.assertEqual(schema.Int().annotation, int)
    self.assertEqual(schema.Int().noneable().annotation, typing.Optional[int])

    # Test default.
    self.assertEqual(schema.Int().default, object_utils.MISSING_VALUE)
    self.assertEqual(schema.Int(1).default, 1)

    # Test is_noneable
    self.assertFalse(schema.Int().is_noneable)
    self.assertTrue(schema.Int().noneable().is_noneable)

    # Test comparison.
    i = schema.Int()
    self.assertEqual(i, i)
    self.assertEqual(schema.Int(), schema.Int())
    self.assertEqual(schema.Int().noneable(), schema.Int().noneable())
    self.assertEqual(schema.Int(1), schema.Int(1))
    self.assertEqual(
        schema.Int(min_value=0, max_value=1),
        schema.Int(min_value=0, max_value=1))
    self.assertNotEqual(schema.Int(), schema.Bool())
    self.assertNotEqual(schema.Int(), schema.Int().noneable())
    self.assertNotEqual(schema.Int(1), schema.Int())
    self.assertNotEqual(schema.Int(1), schema.Int(2))
    self.assertNotEqual(schema.Int(1, min_value=1), schema.Int(1))
    self.assertNotEqual(schema.Int(1, max_value=1), schema.Int(1))
    self.assertNotEqual(schema.Int(min_value=0), schema.Int())
    self.assertNotEqual(schema.Int(max_value=0), schema.Int())
    self.assertNotEqual(schema.Int(min_value=0), schema.Int(min_value=1))
    self.assertNotEqual(schema.Int(max_value=0), schema.Int(max_value=1))

    # Test bad __init__.
    with self.assertRaisesRegex(
        ValueError, '"max_value" must be equal or greater than "min_value".'):
      schema.Int(min_value=1, max_value=0)

    # Test apply
    self.assertEqual(schema.Int().apply(1), 1)
    self.assertEqual(schema.Int(min_value=1, max_value=1).apply(1), 1)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Int().apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'int\'> but encountered '
        '<(type|class) \'float\'>.'):
      schema.Int().apply(1.0)

    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range \\(min=0, max=None\\)'):
      schema.Int(min_value=0).apply(-1)

    with self.assertRaisesRegex(
        ValueError, 'Value 1 is out of range \\(min=None, max=0\\)'):
      schema.Int(max_value=0).apply(1)

    # Test is_compatible:
    self.assertTrue(schema.Int().is_compatible(schema.Int()))
    self.assertTrue(schema.Int().noneable().is_compatible(schema.Int()))
    self.assertTrue(schema.Int().is_compatible(schema.Int(min_value=1)))
    self.assertTrue(schema.Int().is_compatible(schema.Int(max_value=1)))
    self.assertTrue(
        schema.Int(min_value=1, max_value=10).is_compatible(
            schema.Int(min_value=2, max_value=10)))
    self.assertFalse(schema.Int().is_compatible(schema.Int().noneable()))
    self.assertFalse(schema.Int().is_compatible(schema.Bool()))
    self.assertFalse(schema.Int(min_value=1).is_compatible(schema.Int()))
    self.assertFalse(
        schema.Int(min_value=2, max_value=5).is_compatible(
            schema.Int(min_value=2, max_value=10)))

    # Test extends.
    # Child without constraints will inheirt constraints.
    self.assertEqual(schema.Int().extend(schema.Int(min_value=0, max_value=1)),
                     schema.Int(min_value=0, max_value=1))

    # Child extends base with constraints will intersect valid range.
    self.assertEqual(
        schema.Int(min_value=2,
                   max_value=5).extend(schema.Int(min_value=2, max_value=6)),
        schema.Int(min_value=2, max_value=5))

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(schema.Int().extend(schema.Int().noneable()).is_noneable)

    # Child may extend a union that has the same type.
    self.assertEqual(
        schema.Int().extend(
            schema.Union([schema.Int(min_value=1),
                          schema.Bool()])), schema.Int(min_value=1))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Int().extend(schema.Bool())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Int().noneable().extend(schema.Int())

    # Child with wider range cannot extend a base with narrower range.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: min_value is smaller'):
      schema.Int(min_value=0).extend(schema.Int(min_value=1))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is larger'):
      schema.Int(max_value=1).extend(schema.Int(max_value=0))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      schema.Int(max_value=1).extend(schema.Int(min_value=5))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      schema.Int(min_value=1).extend(schema.Int(max_value=0))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: no compatible type found in Union.'):
      schema.Int().extend(schema.Union([schema.Bool(), schema.Str()]))

  def testFloat(self):
    """Tests for Float."""

    # Test __str__.
    self.assertEqual(str(schema.Float()), 'Float()')
    self.assertEqual(
        str(schema.Float().noneable()), 'Float(default=None, noneable=True)')
    self.assertEqual(
        str(schema.Float(1.0).freeze()), 'Float(default=1.0, frozen=True)')
    self.assertEqual(
        str(schema.Float(1.0).noneable()), 'Float(default=1.0, noneable=True)')
    self.assertEqual(
        str(schema.Float(default=1., min_value=0., max_value=1.).noneable()),
        'Float(default=1.0, min=0.0, max=1.0, noneable=True)')

    # Test value_type
    self.assertEqual(schema.Float().value_type, float)

    # Test annotation.
    self.assertEqual(schema.Float().annotation, float)
    self.assertEqual(schema.Float().noneable().annotation,
                     typing.Optional[float])

    # Test default.
    self.assertEqual(schema.Float().default, object_utils.MISSING_VALUE)
    self.assertEqual(schema.Float(1.0).default, 1.0)

    # Test is_noneable
    self.assertFalse(schema.Float().is_noneable)
    self.assertTrue(schema.Float().noneable().is_noneable)

    # Test comparison.
    f = schema.Float()
    self.assertEqual(f, f)
    self.assertEqual(schema.Float(), schema.Float())
    self.assertEqual(schema.Float().noneable(), schema.Float().noneable())
    self.assertEqual(schema.Float(1.), schema.Float(1.))
    self.assertEqual(
        schema.Float(min_value=0., max_value=1.),
        schema.Float(min_value=0., max_value=1.))
    self.assertNotEqual(schema.Float(), schema.Int())
    self.assertNotEqual(schema.Float(), schema.Float().noneable())
    self.assertNotEqual(schema.Float(1.), schema.Float())
    self.assertNotEqual(schema.Float(1.), schema.Float(2.))
    self.assertNotEqual(schema.Float(1., min_value=1.), schema.Float(1.))
    self.assertNotEqual(schema.Float(1., max_value=1.), schema.Float(1.))
    self.assertNotEqual(schema.Float(min_value=0.), schema.Float())
    self.assertNotEqual(schema.Float(max_value=0.), schema.Float())
    self.assertNotEqual(schema.Float(min_value=0.), schema.Float(min_value=1.))
    self.assertNotEqual(schema.Float(max_value=0.), schema.Float(max_value=1.))

    # Test bad __init__.
    with self.assertRaisesRegex(
        ValueError, '"max_value" must be equal or greater than "min_value".'):
      schema.Float(min_value=1., max_value=0.)

    # Test apply
    self.assertEqual(schema.Float().apply(1.), 1.)
    self.assertEqual(schema.Float(min_value=1., max_value=1.).apply(1.), 1.)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Float().apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'float\'> but encountered '
        '<(type|class) \'int\'>.'):
      schema.Float().apply(1)

    with self.assertRaisesRegex(
        ValueError, 'Value -1.0 is out of range \\(min=0.0, max=None\\).'):
      schema.Float(min_value=0.).apply(-1.)

    with self.assertRaisesRegex(
        ValueError, 'Value 1.0 is out of range \\(min=None, max=0.0\\).'):
      schema.Float(max_value=0.).apply(1.)

    # Test is_compatible:
    self.assertTrue(schema.Float().is_compatible(schema.Float()))
    self.assertTrue(schema.Float().noneable().is_compatible(schema.Float()))
    self.assertTrue(schema.Float().is_compatible(schema.Float(min_value=1.)))
    self.assertTrue(schema.Float().is_compatible(schema.Float(max_value=1.)))
    self.assertTrue(
        schema.Float(min_value=1., max_value=10.).is_compatible(
            schema.Float(min_value=2., max_value=10.)))
    self.assertFalse(schema.Float().is_compatible(schema.Float().noneable()))
    self.assertFalse(schema.Float().is_compatible(schema.Bool()))
    self.assertFalse(schema.Float(min_value=1).is_compatible(schema.Float()))
    self.assertFalse(
        schema.Float(min_value=2, max_value=5).is_compatible(
            schema.Float(min_value=2, max_value=10)))

    # Test extends.
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        schema.Float().extend(schema.Float(min_value=0., max_value=1.)),
        schema.Float(min_value=0., max_value=1.))

    # Child extends base with constraints will intersect valid range.
    self.assertEqual(
        schema.Float(min_value=3.).extend(
            schema.Float(min_value=2., max_value=6.)),
        schema.Float(min_value=3., max_value=6.))

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(schema.Float().extend(
        schema.Float().noneable()).is_noneable)

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Float().extend(schema.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Float().noneable().extend(schema.Float())

    # Child with wider range cannot extend a base with narrower range.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: min_value is smaller'):
      schema.Float(min_value=0.).extend(schema.Float(min_value=1.))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is larger'):
      schema.Float(max_value=1.).extend(schema.Float(max_value=0.))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      schema.Float(max_value=1.).extend(schema.Float(min_value=5.))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: '
        'min_value .* is greater than max_value .* after extension'):
      schema.Float(min_value=1.).extend(schema.Float(max_value=0.))

  def testEnum(self):
    """Tests for Enum."""

    # Test __str__.
    self.assertEqual(
        str(schema.Enum('a', ['a', 'b', 'c'])),
        'Enum(default=\'a\', values=[\'a\', \'b\', \'c\'])')

    self.assertEqual(
        str(schema.Enum('a', ['a', 'b', 'c']).freeze()),
        'Enum(default=\'a\', values=[\'a\', \'b\', \'c\'], frozen=True)')

    # Test value_type
    self.assertEqual(schema.Enum('a', ['a', None]).value_type, str)
    self.assertEqual(schema.Enum(1, [1, None]).value_type, int)

    # Test annotation.
    self.assertEqual(schema.Enum('a', ['a', 'b']).annotation, typing.Text)
    self.assertEqual(
        schema.Enum('a', ['a', None]).annotation, typing.Optional[typing.Text])
    self.assertEqual(schema.Enum(1, [1, 2]).annotation, int)
    self.assertEqual(schema.Enum(1, [1, None]).annotation, typing.Optional[int])
    self.assertEqual(schema.Enum(1, [1, 'foo']).annotation, typing.Any)

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
    self.assertEqual(schema.Enum(a, [a, b1, None]).value_type, A)
    self.assertEqual(schema.Enum(b1, [b1, b2]).value_type, B)
    self.assertIsNone(schema.Enum(a, [a, b1, c]).value_type)

    # Test default.
    self.assertEqual(schema.Enum('a', ['a', 'b']).default, 'a')
    self.assertEqual(schema.Enum('a', ['a']).noneable().default, 'a')
    self.assertIsNone(schema.Enum(None, [None, 'a']).default)

    # Test is_noneable
    self.assertFalse(schema.Enum('a', ['a', 'b']).is_noneable)
    self.assertTrue(schema.Enum('a', ['a', 'b', None]).is_noneable)
    self.assertEqual(
        schema.Enum('a', ['a', 'b']).noneable(),
        schema.Enum('a', ['a', 'b', None]))

    # Test comparison.
    e = schema.Enum('a', ['a', 'b'])
    self.assertEqual(e, e)
    self.assertEqual(schema.Enum('a', ['a']), schema.Enum('a', ['a']))
    self.assertNotEqual(schema.Enum('a', ['a']), schema.Int())
    self.assertNotEqual(schema.Enum('a', ['a']), schema.Enum('a', ['a', 'b']))
    self.assertNotEqual(
        schema.Enum('a', ['a', 'b']), schema.Enum('b', ['a', 'b']))

    # Test bad __init__.
    with self.assertRaisesRegex(ValueError,
                                'Values for Enum should be a non-empty list.'):
      schema.Enum(None, [])

    with self.assertRaisesRegex(
        ValueError, 'Enum default value \'a\' is not in candidate list.'):
      schema.Enum('a', ['b'])

    # Test apply.
    self.assertEqual(schema.Enum('a', ['a']).apply('a'), 'a')
    self.assertIsNone(schema.Enum('a', ['a', None]).apply(None))

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered <(type|class) \'int\'>'):
      schema.Enum('a', ['a']).apply(1)

    with self.assertRaisesRegex(ValueError,
                                'Value \'b\' is not in candidate list'):
      schema.Enum('a', ['a']).apply('b')

    # Test is_compatible:
    self.assertTrue(
        schema.Enum(0, [0, 1]).is_compatible(schema.Enum(0, [0, 1])))
    self.assertTrue(schema.Enum(0, [0, 1]).is_compatible(schema.Enum(0, [0])))
    self.assertFalse(schema.Enum(0, [0]).is_compatible(schema.Enum(0, [0, 1])))
    self.assertFalse(schema.Enum(0, [0]).is_compatible(schema.Int()))

    # Test extends.
    self.assertEqual(
        schema.Enum('a', ['a']).extend(schema.Enum('b', ['a', 'b'])),
        schema.Enum('a', ['a']))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: values in base should be super set.'):
      schema.Enum('a', ['a', 'b']).extend(schema.Enum('a', ['a']))

  def testList(self):
    """Tests for List."""
    # Test min_size and max_size
    self.assertEqual(schema.List(schema.Int()).min_size, 0)
    self.assertIsNone(schema.List(schema.Int()).max_size)

    list_spec = schema.List(schema.Int(), size=2)
    self.assertEqual(list_spec.min_size, 2)
    self.assertEqual(list_spec.max_size, 2)

    # Test __str__.
    self.assertEqual(
        repr(schema.List(schema.Int(min_value=0))), 'List(Int(min=0))')
    self.assertEqual(
        repr(schema.List(schema.Int(min_value=0)).freeze([1])),
        'List(Int(min=0), default=[1], frozen=True)')
    self.assertEqual(
        repr(schema.List(schema.Str(), default=[], max_size=5).noneable()),
        'List(Str(), max_size=5, default=[], noneable=True)')

    # Test value_type
    self.assertEqual(schema.List(schema.Int()).value_type, list)

    # Test annotation.
    self.assertEqual(schema.List(schema.Int()).annotation, typing.List[int])
    self.assertEqual(
        schema.List(schema.Any()).annotation, typing.List[typing.Any])
    self.assertEqual(
        schema.List(schema.Int()).noneable().annotation,
        typing.Optional[typing.List[int]])
    self.assertEqual(
        schema.List(schema.Int().noneable()).annotation,
        typing.List[typing.Optional[int]])

    # Test default.
    self.assertEqual(
        schema.List(schema.Int()).default, object_utils.MISSING_VALUE)
    self.assertIsNone(schema.List(schema.Int()).noneable().default)
    self.assertEqual(schema.List(schema.Int(), []).default, [])

    # Test is_noneable
    self.assertFalse(schema.List(schema.Int()).is_noneable)
    self.assertTrue(schema.List(schema.Int()).noneable().is_noneable)

    # Test comparison.
    self.assertEqual(schema.List(schema.Int()), schema.List(schema.Int()))
    self.assertEqual(
        schema.List(schema.Int(), []), schema.List(schema.Int(), []))
    self.assertEqual(
        schema.List(schema.Int(), [], max_size=10),
        schema.List(schema.Int(), [], max_size=10))

    self.assertNotEqual(schema.List(schema.Int()), schema.Int())
    self.assertNotEqual(
        schema.List(schema.Int()),
        schema.List(schema.Int()).noneable())
    self.assertNotEqual(schema.List(schema.Int()), schema.List(schema.Str()))
    self.assertNotEqual(
        schema.List(schema.Int(min_value=0)), schema.List(schema.Int()))
    self.assertNotEqual(
        schema.List(schema.Int(), []), schema.List(schema.Int()))
    self.assertNotEqual(
        schema.List(schema.Int(), max_size=10), schema.List(schema.Int()))

    # Test bad __init__.
    with self.assertRaisesRegex(
        ValueError, 'List element spec should be an ValueSpec object.'):
      schema.List(1)

    with self.assertRaisesRegex(ValueError,
                                '"min_size" of List must be no less than 0.'):
      schema.List(schema.Int(), min_size=-1)

    with self.assertRaisesRegex(
        ValueError, '"max_size" of List must be no less than "min_size".'):
      schema.List(schema.Int(), min_size=10, max_size=5)

    with self.assertRaisesRegex(
        ValueError,
        'Either "size" or "min_size"/"max_size" pair can be specified.'):
      schema.List(schema.Int(), size=5, min_size=1)

    # Test apply.
    self.assertEqual(schema.List(schema.Int()).apply([]), [])
    self.assertEqual(schema.List(schema.Int()).apply([1]), [1])
    self.assertEqual(
        schema.List(schema.Int().noneable()).apply([1, None]), [1, None])
    self.assertEqual(
        schema.List(schema.Int()).apply(
            object_utils.MISSING_VALUE, allow_partial=True),
        object_utils.MISSING_VALUE)
    self.assertEqual(
        schema.List(schema.Dict([('a', schema.Str())
                                ])).apply([{}], allow_partial=True), [{
                                    'a': schema.MissingValue(schema.Str())
                                }])

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.List(schema.Int()).apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'list\'> but encountered '
        '<(type|class) \'int\'>.'):
      schema.List(schema.Int()).apply(1)

    with self.assertRaisesRegex(
        ValueError, 'Value 0 is out of range \\(min=1, max=None\\)'):
      schema.List(schema.Int(min_value=1)).apply([0])

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is less than min size \\(1\\).'):
      schema.List(schema.Int(), min_size=1).apply([])

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is greater than max size \\(1\\).'):
      schema.List(schema.Int(), max_size=1).apply([0, 1])

    # Test custom validator.
    def _sum_greater_than_zero(value):
      if sum(value) <= 0:
        raise ValueError('Sum expected to be larger than zero')

    self.assertEqual(
        schema.List(schema.Int(),
                    user_validator=_sum_greater_than_zero).apply([0, 1]),
        [0, 1])

    with self.assertRaisesRegex(
        ValueError, 'Sum expected to be larger than zero \\(path=\\[0\\]\\)'):
      schema.List(
          schema.List(schema.Int(),
                      user_validator=_sum_greater_than_zero)).apply([[-1]])

    # Test is_compatible:
    self.assertTrue(
        schema.List(schema.Int()).is_compatible(schema.List(schema.Int())))

    self.assertTrue(
        schema.List(schema.Int()).noneable().is_compatible(
            schema.List(schema.Int())))

    self.assertTrue(
        schema.List(schema.Int()).is_compatible(
            schema.List(schema.Int(min_value=1))))

    self.assertTrue(
        schema.List(schema.Int().noneable()).is_compatible(
            schema.List(schema.Int())))

    self.assertTrue(
        schema.List(schema.Int(), min_size=10).is_compatible(
            schema.List(schema.Int(), min_size=5)))

    self.assertTrue(
        schema.List(schema.Int()).is_compatible(
            schema.List(schema.Int(), max_size=10)))

    self.assertFalse(schema.List(schema.Int()).is_compatible(schema.Int()))

    self.assertFalse(
        schema.List(schema.Int()).is_compatible(schema.List(schema.Str())))

    self.assertFalse(
        schema.List(schema.Int()).is_compatible(
            schema.List(schema.Int().noneable())))

    self.assertFalse(
        schema.List(schema.Int(min_value=1)).is_compatible(
            schema.List(schema.Int())))

    self.assertFalse(
        schema.List(schema.Int()).is_compatible(
            schema.List(schema.Int()).noneable()))

    self.assertFalse(
        schema.List(schema.Int(),
                    max_size=10).is_compatible(schema.List(schema.Int())))

    self.assertFalse(
        schema.List(schema.Int(), max_size=10).is_compatible(
            schema.List(schema.Int(), max_size=11)))

    # Test extends.
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        schema.List(schema.Int()).extend(schema.List(
            schema.Int(min_value=0))).element.value, schema.Int(min_value=0))

    self.assertEqual(
        schema.List(schema.Int()).extend(
            schema.List(schema.Int(), max_size=10)),
        schema.List(schema.Int(), max_size=10))

    self.assertFalse(
        schema.List(schema.Int()).extend(schema.List(
            schema.Int()).noneable()).is_noneable)

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.List(schema.Int()).extend(schema.Int())

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.List(schema.Int()).extend(schema.List(schema.Str()))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.List(schema.Int()).noneable().extend(schema.List(schema.Int()))

    # Child with smaller min_size cannot extend base with larger min_size.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: min_value is smaller.'):
      schema.List(
          schema.Int(),
          min_size=0).extend(schema.List(schema.Int(), min_size=1))

    # Child with larger max_size cannot extend base with smaller max_size.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is greater.'):
      schema.List(
          schema.Int(),
          max_size=10).extend(schema.List(schema.Int(), max_size=1))

  def testTuple(self):
    """Tests for Tuple."""

    # Test __str__.
    self.assertEqual(repr(schema.Tuple(schema.Int())), 'Tuple(Int())')

    self.assertEqual(
        repr(schema.Tuple(schema.Int(), min_size=2, max_size=3)),
        'Tuple(Int(), min_size=2, max_size=3)')

    self.assertEqual(
        repr(schema.Tuple([schema.Int(), schema.Bool()])),
        'Tuple([Int(), Bool()])')
    self.assertEqual(
        repr(
            schema.Tuple([schema.Int(), schema.Bool()],
                         default=(1, True)).noneable()),
        'Tuple([Int(), Bool()], default=(1, True), noneable=True)')
    self.assertEqual(
        repr(
            schema.Tuple([schema.Int(), schema.Bool()],
                         default=(1, True)).freeze()),
        'Tuple([Int(), Bool()], default=(1, True), frozen=True)')

    # Test value_type
    self.assertEqual(schema.Tuple([schema.Int()]).value_type, tuple)
    self.assertEqual(schema.Tuple(schema.Int()).value_type, tuple)

    # Test annotation.
    self.assertEqual(
        schema.Tuple(schema.Int()).annotation, typing.Tuple[int, ...])
    self.assertEqual(
        schema.Tuple(schema.Int()).annotation, typing.Tuple[int, ...])
    self.assertEqual(
        schema.Tuple([schema.Int(), schema.Str()]).annotation,
        typing.Tuple[int, typing.Text])
    self.assertEqual(
        schema.Tuple([schema.Int(), schema.Any()]).annotation,
        typing.Tuple[int, typing.Any])

    # Test default.
    self.assertEqual(
        schema.Tuple(schema.Int()).default, object_utils.MISSING_VALUE)
    self.assertIsNone(schema.Tuple([schema.Int()]).noneable().default)
    self.assertEqual(schema.Tuple([schema.Int()], (1,)).default, (1,))

    # Test is_noneable
    self.assertFalse(schema.Tuple([schema.Int()]).is_noneable)
    self.assertTrue(schema.Tuple([schema.Int()]).noneable().is_noneable)

    # Test fixed_length.
    self.assertFalse(schema.Tuple(schema.Int()).fixed_length)
    self.assertTrue(schema.Tuple(schema.Int(), size=2).fixed_length)
    self.assertTrue(schema.Tuple([schema.Int(), schema.Str()]).fixed_length)

    # Test min_size
    self.assertEqual(schema.Tuple(schema.Int()).min_size, 0)
    self.assertEqual(schema.Tuple(schema.Int(), min_size=1).min_size, 1)
    self.assertEqual(schema.Tuple(schema.Int(), size=2).min_size, 2)
    self.assertEqual(schema.Tuple([schema.Int()]).min_size, 1)

    # Test max_size
    self.assertIsNone(schema.Tuple(schema.Int()).max_size)
    self.assertEqual(schema.Tuple(schema.Int(), max_size=1).max_size, 1)
    self.assertEqual(schema.Tuple(schema.Int(), size=2).max_size, 2)
    self.assertEqual(schema.Tuple([schema.Int()]).max_size, 1)

    # Test __len__.
    self.assertEqual(len(schema.Tuple(schema.Int())), 0)
    self.assertEqual(len(schema.Tuple(schema.Int(), size=2)), 2)
    self.assertEqual(len(schema.Tuple([schema.Int(), schema.Str()])), 2)

    # Test comparison.
    t = schema.Tuple([schema.Int(), schema.Int()])
    self.assertEqual(t, t)

    self.assertEqual(schema.Tuple(schema.Int()), schema.Tuple(schema.Int()))
    self.assertEqual(schema.Tuple([schema.Int()]), schema.Tuple([schema.Int()]))
    self.assertEqual(
        schema.Tuple(schema.Int(), size=2),
        schema.Tuple([schema.Int(), schema.Int()]))

    self.assertEqual(
        schema.Tuple([schema.Int()], (1,)), schema.Tuple([schema.Int()], (1,)))
    self.assertEqual(
        schema.Tuple([schema.Int(), schema.Bool()]),
        schema.Tuple([schema.Int(), schema.Bool()]))

    self.assertNotEqual(schema.Tuple([schema.Int()]), schema.Int())
    self.assertNotEqual(
        schema.Tuple([schema.Int()]), schema.Tuple(schema.Int()))
    self.assertNotEqual(
        schema.Tuple(schema.Int()), schema.Tuple(schema.Int(), max_size=2))

    self.assertNotEqual(
        schema.Tuple([schema.Int()]), schema.Tuple([schema.Bool()]))
    self.assertNotEqual(
        schema.Tuple([schema.Int()]),
        schema.Tuple([schema.Int()]).noneable())
    self.assertNotEqual(
        schema.Tuple([schema.Int(min_value=0)]), schema.Tuple([schema.Int()]))
    self.assertNotEqual(
        schema.Tuple([schema.Int()], (1,)), schema.Tuple([schema.Int()]))

    # Test bad __init__.
    with self.assertRaisesRegex(
        ValueError, 'Argument \'element_values\' must be a non-empty list'):
      schema.Tuple(1)

    with self.assertRaisesRegex(
        ValueError, 'Argument \'element_values\' must be a non-empty list'):
      schema.Tuple([])

    with self.assertRaisesRegex(
        ValueError, 'Items in \'element_values\' must be ValueSpec objects.'):
      schema.Tuple([1])

    with self.assertRaisesRegex(
        ValueError,
        'Either "size" or "min_size"/"max_size" pair can be specified'):
      schema.Tuple(schema.Int(), size=2, min_size=1)

    with self.assertRaisesRegex(ValueError,
                                '"min_size" of List must be no less than 0.'):
      schema.Tuple(schema.Int(), min_size=-1)

    with self.assertRaisesRegex(
        ValueError, '"max_size" of List must be no less than "min_size"'):
      schema.Tuple(schema.Int(), min_size=2, max_size=1)

    with self.assertRaisesRegex(
        ValueError, '"size", "min_size" and "max_size" are not applicable'):
      schema.Tuple([schema.Int(), schema.Str()], size=2)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'tuple\'> but encountered '
        '<(type|class) \'int\'>.'):
      schema.Tuple([schema.Int()], default=1)

    # Test apply.
    self.assertEqual(schema.Tuple(schema.Int()).apply(tuple()), tuple())
    self.assertEqual(schema.Tuple(schema.Int()).apply((1, 1, 1)), (1, 1, 1))
    self.assertEqual(
        schema.Tuple(schema.Int(), min_size=1).apply((1, 1, 1)), (1, 1, 1))
    self.assertEqual(
        schema.Tuple(schema.Int(), max_size=2).apply((1, 1)), (1, 1))

    self.assertEqual(schema.Tuple([schema.Int()]).apply((1,)), (1,))
    self.assertEqual(
        schema.Tuple([schema.Int(), schema.Bool()]).apply((1, True)), (1, True))
    self.assertIsNone(schema.Tuple([schema.Int()]).noneable().apply(None))
    self.assertEqual(
        schema.Tuple([schema.Int().noneable()]).apply((None,)), (None,))
    self.assertEqual(
        schema.Tuple([schema.Int()
                     ]).apply(object_utils.MISSING_VALUE, allow_partial=True),
        object_utils.MISSING_VALUE)
    self.assertEqual(
        schema.Tuple([schema.Int()]).apply((object_utils.MISSING_VALUE,),
                                           allow_partial=True),
        (object_utils.MISSING_VALUE,))
    self.assertEqual(
        schema.Tuple([schema.Int(),
                      schema.Dict([('a', schema.Str())])]).apply(
                          (1, {}), allow_partial=True), (1, {
                              'a': schema.MissingValue(schema.Str())
                          }))

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Tuple([schema.Int()]).apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'tuple\'> but encountered '
        '<(type|class) \'int\'>.'):
      schema.Tuple([schema.Int()]).apply(1)

    with self.assertRaisesRegex(
        TypeError,
        'Expect <(type|class) \'int\'> but encountered <(type|class) \'str\'>'):
      schema.Tuple([schema.Int()]).apply(('abc',))

    with self.assertRaisesRegex(
        TypeError,
        'Expect <(type|class) \'int\'> but encountered <(type|class) \'str\'>'):
      schema.Tuple(schema.Int()).apply(('abc',))

    with self.assertRaisesRegex(ValueError,
                                'Length of tuple .* is less than min size'):
      schema.Tuple(schema.Int(), min_size=2).apply((1,))

    with self.assertRaisesRegex(ValueError,
                                'Length of tuple .* is greater than max size'):
      schema.Tuple(schema.Int(), max_size=2).apply((1, 1, 1))

    with self.assertRaisesRegex(
        ValueError,
        'Length of input tuple .* does not match the length of spec.'):
      schema.Tuple([schema.Int()]).apply((1, 1))

    # Test user validation.
    def _sum_greater_than_zero(value):
      if sum(list(value)) <= 0:
        raise ValueError('Sum expected to be larger than zero')

    self.assertEqual(
        schema.Tuple([schema.Int(), schema.Int()],
                     user_validator=_sum_greater_than_zero).apply((0, 1)),
        (0, 1))

    with self.assertRaisesRegex(
        ValueError, 'Sum expected to be larger than zero \\(path=\\[0\\]\\)'):
      schema.Tuple(
          [schema.Tuple([schema.Int()],
                        user_validator=_sum_greater_than_zero)]).apply(((-1,),))

    # Test is_compatible:
    self.assertTrue(
        schema.Tuple(schema.Int()).is_compatible(schema.Tuple(schema.Int())))

    self.assertTrue(
        schema.Tuple(schema.Int()).is_compatible(schema.Tuple([schema.Int()])))

    self.assertTrue(
        schema.Tuple(schema.Int()).is_compatible(
            schema.Tuple(schema.Int(), min_size=2, max_size=4)))

    self.assertTrue(
        schema.Tuple(schema.Int(), min_size=1).is_compatible(
            schema.Tuple(schema.Int(), min_size=2)))

    self.assertTrue(
        schema.Tuple(schema.Int(), max_size=5).is_compatible(
            schema.Tuple(schema.Int(), max_size=4)))

    self.assertTrue(
        schema.Tuple([schema.Int()
                     ]).is_compatible(schema.Tuple([schema.Int()])))

    self.assertTrue(
        schema.Tuple([schema.Int()
                     ]).noneable().is_compatible(schema.Tuple([schema.Int()])))

    self.assertTrue(
        schema.Tuple([schema.Int()
                     ]).is_compatible(schema.Tuple([schema.Int(min_value=1)])))

    self.assertTrue(
        schema.Tuple([schema.Int().noneable()
                     ]).is_compatible(schema.Tuple([schema.Int()])))

    self.assertFalse(schema.Tuple(schema.Int()).is_compatible(schema.Int()))

    self.assertFalse(schema.Tuple([schema.Int()]).is_compatible(schema.Int()))

    self.assertFalse(
        schema.Tuple(schema.Int(), min_size=2).is_compatible(
            schema.Tuple(schema.Int(), min_size=1)))

    self.assertFalse(
        schema.Tuple(schema.Int(), max_size=2).is_compatible(
            schema.Tuple(schema.Int(), max_size=3)))

    self.assertFalse(
        schema.Tuple(schema.Int()).is_compatible(
            schema.Tuple([schema.Int(), schema.Str()])))

    self.assertFalse(
        schema.Tuple(schema.Int(), min_size=3).is_compatible(
            schema.Tuple([schema.Int(), schema.Int()])))

    self.assertFalse(
        schema.Tuple(schema.Int(), max_size=2).is_compatible(
            schema.Tuple([schema.Int(),
                          schema.Int(),
                          schema.Int()])))

    self.assertFalse(
        schema.Tuple([schema.Int()]).is_compatible(
            schema.Tuple([schema.Int(), schema.Int()])))

    self.assertFalse(
        schema.Tuple([schema.Int()
                     ]).is_compatible(schema.Tuple([schema.Str()])))

    self.assertFalse(
        schema.Tuple([schema.Int()]).is_compatible(schema.List(schema.Int())))

    self.assertFalse(
        schema.Tuple([schema.Int()]).is_compatible(schema.Tuple(schema.Int())))

    self.assertFalse(
        schema.Tuple([schema.Int()
                     ]).is_compatible(schema.Tuple([schema.Int().noneable()])))

    self.assertFalse(
        schema.Tuple([schema.Int(min_value=1)
                     ]).is_compatible(schema.Tuple([schema.Int()])))

    self.assertFalse(
        schema.Tuple([schema.Int()
                     ]).is_compatible(schema.Tuple([schema.Int()]).noneable()))

    # Test extends.
    # Variable length tuple extend variable length tuple:
    self.assertEqual(
        schema.Tuple(schema.Int(),
                     max_size=5).extend(schema.Tuple(schema.Int(), min_size=2)),
        schema.Tuple(schema.Int(), min_size=2, max_size=5))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has smaller min size'):
      schema.Tuple(
          schema.Int(),
          min_size=2).extend(schema.Tuple(schema.Int(), min_size=5))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has greater max size'):
      schema.Tuple(
          schema.Int(),
          max_size=5).extend(schema.Tuple(schema.Int(), max_size=3))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type'):
      schema.Tuple(schema.Int()).extend(schema.Tuple(schema.Str()))

    # Variable length tuple extend fixed length tuple.
    with self.assertRaisesRegex(
        TypeError,
        '.* cannot extend .*: a variable length tuple cannot extend a fixed '
        'length tuple'):
      schema.Tuple(schema.Int()).extend(schema.Tuple([schema.Int()]))

    # Fixed length tuple extend variable length tuple.
    self.assertEqual(
        schema.Tuple([schema.Int(), schema.Int()
                     ]).extend(schema.Tuple(schema.Int(min_value=0))),
        schema.Tuple([schema.Int(min_value=0),
                      schema.Int(min_value=0)]))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has less elements than required'):
      schema.Tuple([schema.Int()
                   ]).extend(schema.Tuple(schema.Int(), min_size=2))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .* as it has more elements than required'):
      schema.Tuple([schema.Int(), schema.Int()
                   ]).extend(schema.Tuple(schema.Int(), max_size=1))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type'):
      schema.Tuple([schema.Int(),
                    schema.Str()]).extend(schema.Tuple(schema.Int()))

    # Fixed length tuple extend fixed length tuple.
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        schema.Tuple([schema.Int()
                     ]).extend(schema.Tuple([schema.Int(min_value=0)
                                            ])).elements[0].value,
        schema.Int(min_value=0))

    self.assertFalse(
        schema.Tuple([schema.Int()
                     ]).extend(schema.Tuple([schema.Int()
                                            ]).noneable()).is_noneable)

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Tuple([schema.Int()]).extend(schema.Int())

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Tuple([schema.Int()]).extend(schema.Tuple([schema.Str()]))

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: unmatched number of elements.'):
      schema.Tuple([schema.Int()
                   ]).extend(schema.Tuple([schema.Int(),
                                           schema.Int()]))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Tuple([schema.Int()
                   ]).noneable().extend(schema.Tuple([schema.Int()]))

    # Child with larger max_size cannot extend base with smaller max_size.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is larger.'):
      schema.Tuple([schema.Int(max_value=100)
                   ]).extend(schema.Tuple([schema.Int(max_value=0)]))

  def testDict(self):
    """Tests for Dict."""
    # Test __repr__.
    self.assertEqual(repr(schema.Dict()), 'Dict()')

    self.assertEqual(
        repr(
            schema.Dict([
                ('b', 1, 'field 1'),
                ('a', schema.Str(), 'field 2'),
            ]).noneable()), 'Dict({b=Int(default=1), a=Str()}, noneable=True)')

    self.assertEqual(
        repr(
            schema.Dict([
                ('b', 1, 'field 1'),
                ('a', schema.Str('abc'), 'field 2'),
            ]).freeze()),
        'Dict({b=Int(default=1), a=Str(default=\'abc\')}, frozen=True)')

    # Test value_type
    self.assertEqual(schema.Dict().value_type, dict)

    # Test annotation.
    self.assertEqual(schema.Dict().annotation, typing.Dict[typing.Text,
                                                           typing.Any])
    self.assertEqual(schema.Dict().noneable().annotation,
                     typing.Optional[typing.Dict[typing.Text, typing.Any]])

    # Test default.
    self.assertEqual(
        schema.Dict([('a', schema.Int(), 'field 1')]).default,
        {'a': object_utils.MISSING_VALUE})

    self.assertEqual(
        schema.Dict([('a', schema.Int(1), 'field 1'),
                     ('b', schema.Dict([('c', 'foo', 'field 2.1')]), 'field 2')
                    ]).default,
        dict(a=1, b={'c': 'foo'}))

    self.assertEqual(
        schema.Dict([('a', schema.Int(1), 'field 1'),
                     ('b', schema.Dict([('c', schema.Str(), 'field 2.1')]),
                      'field 2')]).default,
        dict(a=1, b={'c': object_utils.MISSING_VALUE}))

    self.assertIsNone(
        schema.Dict([('a', schema.Int(), 'field 1')]).noneable().default)

    # Test noneable.
    self.assertFalse(schema.Dict().is_noneable)
    self.assertTrue(schema.Dict().noneable().is_noneable)

    # Test comparison.
    d = schema.Dict([('a', schema.Int())])
    self.assertEqual(d, d)
    self.assertEqual(d.schema, schema.create_schema([('a', schema.Int())]))
    self.assertEqual(schema.Dict(), schema.Dict())
    self.assertEqual(schema.Dict().noneable(), schema.Dict().noneable())
    self.assertEqual(
        schema.Dict([('a', 1, 'field 1')]), schema.Dict([('a', 1, 'field 1')]))
    self.assertNotEqual(schema.Dict(), schema.Dict().noneable())
    self.assertNotEqual(schema.Dict(), schema.Dict([('a', 1, 'field 1')]))
    self.assertNotEqual(schema.Dict(), schema.Dict([('a', 1, 'field 1')]))
    self.assertNotEqual(
        schema.Dict([('a', schema.Int(), 'field 1')]),
        schema.Dict([('a', 1, 'field 1')]))

    # Test __init__.
    self.assertEqual(
        schema.Dict(schema.create_schema([('a', schema.Int())])),
        schema.Dict([('a', schema.Int())]))

    with self.assertRaisesRegex(
        TypeError, 'Schema definition should be a list of schema.Field or '
        'a list of tuples of \\(key, value, description, metadata\\).'):
      schema.Dict({})

    with self.assertRaisesRegex(
        TypeError, 'The 1st element of field definition should be of '
        '<(type|class) \'str\'>'):
      schema.Dict([(1, 1, 'field 1')])

    with self.assertRaisesRegex(
        TypeError,
        'Only primitive types \\(bool, int, float, str\\) are supported '):
      schema.Dict([('key', lambda x: x, 'field 1')])

    with self.assertRaisesRegex(
        TypeError, 'Description \\(the 3rd element\\) of field definition '
        'should be text type.'):
      schema.Dict([('key', 1, 1)])

    with self.assertRaisesRegex(
        TypeError, 'Metadata \\(the 4th element\\) of field definition '
        'should be a dict of objects.'):
      schema.Dict([('key', 1, 'field 1', 123)])

    # Test apply.
    self.assertEqual(schema.Dict().apply({'a': 1}), {'a': 1})
    self.assertEqual(
        list(schema.Dict().apply({'b': 1, 'a': 2}).keys()), ['b', 'a'])

    self.assertEqual(
        schema.Dict([
            ('a', schema.Int(), 'field 1'),
            ('b', schema.Bool().noneable(), 'field 2'),
        ]).apply({'a': 1}), {
            'a': 1,
            'b': None
        })

    self.assertEqual(
        schema.Dict([
            ('a', schema.Int(), 'field 1'),
            ('b', schema.Bool().noneable(), 'field 2'),
        ]).apply({'b': True}, allow_partial=True), {
            'a': object_utils.MISSING_VALUE,
            'b': True
        })

    self.assertEqual(
        schema.Dict([('a', 1, 'field a'), ('b', schema.Str(), 'field b'),
                     ('c',
                      schema.Dict([
                          ('d', True, 'field d'),
                          ('e', schema.Float(), 'field f'),
                      ]), 'field c')]).apply({}, allow_partial=True), {
                          'a': 1,
                          'b': schema.MissingValue(schema.Str()),
                          'c': {
                              'd': True,
                              'e': schema.MissingValue(schema.Float()),
                          }
                      })

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Dict().apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'dict\'> but encountered '
        '<(type|class) \'int\'>.'):
      schema.Dict().apply(1)

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'int\'> but encountered '
        '<(type|class) \'str\'>.'):
      schema.Dict([('a', 1, 'field 1')]).apply({'a': 'foo'})

    with self.assertRaisesRegex(
        KeyError,
        'Keys \\[\'b\'\\] are not allowed in Schema. \\(parent=\'\'\\)'):
      schema.Dict([('a', 1, 'field 1')]).apply({'b': 1})

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified. \\(Path=\'a\'.*\\)'):
      schema.Dict([
          ('a', schema.Int(), 'field 1'),
          ('b', schema.Bool().noneable(), 'field 2'),
      ]).apply({'b': True})

    # Test user validator.
    def _sum_greater_than_zero(value):
      if sum(value.values()) <= 0:
        raise ValueError('Sum of values expected to be larger than zero')

    self.assertEqual(
        schema.Dict(user_validator=_sum_greater_than_zero).apply({
            'a': 1,
            'b': 2,
        }), {
            'a': 1,
            'b': 2
        })

    with self.assertRaisesRegex(
        ValueError,
        'Sum of values expected to be larger than zero \\(path=x\\)'):
      schema.Dict([('x', schema.Dict(user_validator=_sum_greater_than_zero))
                  ]).apply({'x': {
                      'a': -1
                  }})

    # Test is_compatible:
    self.assertTrue(schema.Dict().is_compatible(schema.Dict()))

    self.assertTrue(schema.Dict().noneable().is_compatible(schema.Dict()))

    self.assertTrue(schema.Dict().is_compatible(
        schema.Dict([('a', schema.Int())])))

    self.assertTrue(
        schema.Dict([
            ('a', schema.Int())
        ]).is_compatible(schema.Dict([('a', schema.Int(min_value=1))])))

    self.assertFalse(schema.Dict().is_compatible(schema.Int()))

    self.assertFalse(schema.Dict().is_compatible(schema.Dict().noneable()))

    self.assertFalse(
        schema.Dict([('a', schema.Int())]).is_compatible(schema.Dict()))

    self.assertFalse(
        schema.Dict([('a', schema.Int(min_value=1))
                    ]).is_compatible(schema.Dict([('a', schema.Int())])))

    self.assertFalse(
        schema.Dict([('a', schema.Int())]).is_compatible(
            schema.Dict([('a', schema.Int()), ('b', schema.Int())])))

    # Test extends.
    self.assertFalse(schema.Dict().extend(schema.Dict().noneable()).is_noneable)

    self.assertEqual(
        schema.Dict().extend(schema.Dict([('a', 1, 'field 1')])).schema,
        schema.Dict([('a', 1, 'field 1')]).schema)

    self.assertEqual(
        schema.Dict([('b', schema.Str(), 'field 2')
                    ]).extend(schema.Dict([('a', 1, 'field 1')])),
        schema.Dict([('a', 1, 'field 1'), ('b', schema.Str(), 'field 2')]))

    self.assertEqual(
        schema.Dict([('a', 1)
                    ]).extend(schema.Dict([('a', schema.Int(), 'field 1')])),
        schema.Dict([('a', 1, 'field 1')]))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Dict().extend(schema.Int())

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Dict().noneable().extend(schema.Dict())

    # Child extends base dict with incompatible values.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Dict([('a', 1, 'field 1')
                  ]).extend(schema.Dict([('a', schema.Str(), 'field 1')]))

  def testObject(self):
    """Tests for Object."""

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

    # Test __str__.
    self.assertEqual(str(schema.Object(A)), 'Object(A)')
    self.assertEqual(
        str(schema.Object(A).noneable()),
        'Object(A, default=None, noneable=True)')
    self.assertEqual(
        str(schema.Object(A).noneable().freeze()),
        'Object(A, default=None, noneable=True, frozen=True)')

    # Test value_type
    self.assertEqual(schema.Object(A).value_type, A)

    # Test annotation.
    self.assertEqual(schema.Object(A).annotation, A)
    self.assertEqual(schema.Object(A).noneable().annotation, typing.Optional[A])

    # Test default.
    self.assertEqual(schema.Object(A).default, object_utils.MISSING_VALUE)
    a = A()
    self.assertEqual(schema.Object(A, a).default, a)

    # Test is_noneable
    self.assertFalse(schema.Object(A).is_noneable)
    self.assertTrue(schema.Object(A).noneable().is_noneable)

    # Test comparison.
    o = schema.Object(A)
    self.assertEqual(o, o)
    self.assertIsNone(o.schema)
    self.assertEqual(schema.Object(A), schema.Object(A))
    self.assertEqual(schema.Object(A).noneable(), schema.Object(A).noneable())
    self.assertNotEqual(schema.Object(A).noneable(), schema.Object(A))
    self.assertNotEqual(schema.Object(A), schema.Object(B))

    # Test bad __init__.
    with self.assertRaisesRegex(TypeError,
                                '"cls" for Object spec cannot be None.'):
      schema.Object(None)

    with self.assertRaisesRegex(TypeError,
                                '"cls" for Object spec should be a type.'):
      schema.Object(1)

    with self.assertRaisesRegex(
        TypeError, '<(type|class) \'object\'> is too general for Object spec.'):
      schema.Object(object)

    # Test apply.
    self.assertEqual(schema.Object(A).apply(a), a)
    self.assertIsNone(schema.Object(A).noneable().apply(None))
    b = B()
    self.assertEqual(schema.Object(A).apply(b), b)

    d = D()
    self.assertEqual(schema.Object(C).apply(d, allow_partial=True), d)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Object(A).apply(None)

    with self.assertRaisesRegex(
        TypeError,
        'Expect <class .*A\'> but encountered <(type|class) \'int\'>.'):
      schema.Object(A).apply(1)

    with self.assertRaisesRegex(
        TypeError, 'Expect <class .*B\'> but encountered <class .*C\'>.'):
      schema.Object(B).apply(C())

    with self.assertRaisesRegex(ValueError, 'Object .* is not fully bound.'):
      schema.Object(C).apply(D())

    # Test user validation.
    def _value_is_zero(b):
      if b.value != 0:
        raise ValueError('Value should be zero')

    b = B()
    self.assertEqual(
        schema.Object(B, user_validator=_value_is_zero).apply(b), b)

    with self.assertRaisesRegex(ValueError, 'Value should be zero \\(path=\\)'):
      schema.Object(B, user_validator=_value_is_zero).apply(B(1))

    # Test is_compatible.
    self.assertTrue(schema.Object(A).is_compatible(schema.Object(A)))
    self.assertTrue(schema.Object(A).noneable().is_compatible(schema.Object(A)))
    self.assertTrue(schema.Object(A).is_compatible(schema.Object(B)))

    self.assertFalse(schema.Object(A).is_compatible(schema.Int()))
    self.assertFalse(
        schema.Object(A).is_compatible(schema.Object(A).noneable()))
    self.assertFalse(schema.Object(B).is_compatible(schema.Object(A)))
    self.assertFalse(schema.Object(B).is_compatible(schema.Object(C)))

    # Test extend.
    self.assertEqual(
        schema.Object(B).extend(schema.Object(A)), schema.Object(B))
    self.assertEqual(
        schema.Object(A).extend(schema.Callable([schema.Any()])),
        schema.Object(A))
    self.assertEqual(
        schema.Object(A).extend(schema.Callable(kw=[('a', schema.Any())])),
        schema.Object(A))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible class.'):
      schema.Object(A).extend(schema.Object(B))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Object(A).extend(
          schema.Callable([schema.Any(),
                           schema.Any(),
                           schema.Any()]))

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Object(A).extend(schema.Callable(kw=[('b', schema.Any())]))

  def testCallable(self):
    """Tests for Callable."""
    # Test __str__.
    self.assertEqual(str(schema.Callable()), 'Callable()')
    self.assertEqual(
        str(
            schema.Callable(
                args=[schema.Int(), schema.Int()],
                kw=[('a', schema.Str().noneable())],
                returns=schema.Int())),
        'Callable(args=[Int(), Int()], kw=[(\'a\', '
        'Str(default=None, noneable=True))], returns=Int())')
    self.assertEqual(
        str(
            schema.Callable(
                args=[schema.Int(), schema.Int()],
                kw=[('a', schema.Str().noneable())],
                returns=schema.Int()).noneable().freeze()),
        'Callable(args=[Int(), Int()], kw=[(\'a\', '
        'Str(default=None, noneable=True))], returns=Int(), default=None, '
        'noneable=True, frozen=True)')

    # Test value_type
    self.assertIsNone(schema.Callable().value_type)

    # Test annotation.
    self.assertEqual(schema.Callable().annotation, typing.Callable[[], None])
    self.assertEqual(
        schema.Callable([schema.Int(), schema.Bool()],
                        returns=schema.Int()).annotation,
        typing.Callable[[int, bool], int])
    self.assertEqual(
        schema.Callable(kw=[('x', schema.Int())],
                        returns=schema.Int()).annotation, typing.Callable[...,
                                                                          int])

    # Test default.
    self.assertEqual(schema.Callable().default, object_utils.MISSING_VALUE)
    func = lambda x: x
    self.assertIs(schema.Callable(default=func).default, func)

    # Test is_noneable
    self.assertFalse(schema.Callable().is_noneable)
    self.assertTrue(schema.Callable().noneable().is_noneable)

    # Test comparison.
    func = schema.Callable()
    self.assertEqual(func, func)
    self.assertEqual(schema.Callable(), schema.Callable())
    self.assertEqual(schema.Callable().noneable(), schema.Callable().noneable())
    self.assertEqual(
        schema.Callable(
            args=[schema.Str()], kw=[('a', schema.Int())],
            returns=schema.Any()).noneable(),
        schema.Callable(
            args=[schema.Str()], kw=[('a', schema.Int())],
            returns=schema.Any()).noneable())
    self.assertNotEqual(schema.Callable().noneable(), schema.Callable())
    self.assertNotEqual(schema.Callable(args=[schema.Int()]), schema.Callable())
    self.assertNotEqual(
        schema.Callable(kw=[('b', schema.Int())]),
        schema.Callable(kw=[('a', schema.Int())]))
    self.assertNotEqual(
        schema.Callable(returns=schema.Int()), schema.Callable())
    self.assertNotEqual(schema.Functor(), schema.Callable())
    self.assertNotEqual(schema.Functor(), schema.Callable())

    # Test bad __init__.
    with self.assertRaisesRegex(
        TypeError, '\'args\' should be a list of ValueSpec objects.'):
      schema.Callable(1)

    with self.assertRaisesRegex(
        TypeError, '\'args\' should be a list of ValueSpec objects.'):
      schema.Callable([1])

    with self.assertRaisesRegex(
        TypeError, '\'kw\' should be a list of \\(name, value_spec\\) tuples'):
      schema.Callable(kw='a')

    with self.assertRaisesRegex(
        TypeError, '\'kw\' should be a list of \\(name, value_spec\\) tuples'):
      schema.Callable(kw=['a'])

    with self.assertRaisesRegex(
        TypeError, '\'kw\' should be a list of \\(name, value_spec\\) tuples'):
      schema.Callable(kw=[('a', 1)])

    with self.assertRaisesRegex(TypeError,
                                '\'returns\' should be a ValueSpec object'):
      schema.Callable(returns=1)

    # Test apply.
    self.assertIsNone(schema.Callable().noneable().apply(None))
    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Callable().apply(None)

    with self.assertRaisesRegex(TypeError, 'Value is not callable'):
      schema.Callable().apply(1)

    # Apply on function without wildcard arguments.
    func1 = lambda x: x
    self.assertEqual(schema.Callable().apply(func1), func1)
    self.assertEqual(schema.Callable([schema.Int()]).apply(func1), func1)
    self.assertEqual(
        schema.Callable(kw=[('x', schema.Int())]).apply(func1), func1)
    with self.assertRaisesRegex(
        TypeError, '.* only take 1 positional arguments, while 2 is required'):
      schema.Callable([schema.Int(), schema.Int()]).apply(func1)
    with self.assertRaisesRegex(TypeError,
                                'Keyword argument \'y\' does not exist in .*'):
      schema.Callable(kw=[('y', schema.Int())]).apply(func1)

    # Apply on function with wildcard positional args.
    func2 = lambda *args: sum(args)
    self.assertEqual(schema.Callable().apply(func2), func2)
    self.assertEqual(schema.Callable([schema.Int()]).apply(func2), func2)
    with self.assertRaisesRegex(TypeError,
                                'Keyword argument \'y\' does not exist in .*'):
      schema.Callable(kw=[('y', schema.Int())]).apply(func2)

    with self.assertRaisesRegex(TypeError, 'Expect .*Functor'):
      schema.Functor().apply(func2)

    # Apply on function with wildcard keyword args.
    func3 = lambda **kwargs: sum(kwargs.values())
    self.assertEqual(schema.Callable().apply(func3), func3)
    self.assertEqual(
        schema.Callable(kw=[('a', schema.Int())]).apply(func3), func3)
    with self.assertRaisesRegex(
        TypeError, '.* only take 0 positional arguments, while 1 is required'):
      schema.Callable([schema.Int()]).apply(func3)

    class Functor1(object_utils.Functor):

      signature = schema.Signature(
          callable_type=schema.CallableType.FUNCTION,
          name='foo',
          module_name='__main__',
          args=[
              schema.Argument('a', schema.Int()),
              schema.Argument('b', schema.Str())
          ])

      def __init__(self, value):
        self.value = value

      def __call__(self, a, b):
        del a, b

    func4 = Functor1(1)
    self.assertEqual(schema.Callable().apply(func4), func4)
    self.assertEqual(schema.Callable([schema.Int()]).apply(func4), func4)
    self.assertEqual(
        schema.Callable([schema.Int(), schema.Str()]).apply(func4), func4)
    self.assertEqual(
        schema.Callable(kw=[('a', schema.Int())]).apply(func4), func4)
    self.assertEqual(schema.Functor().apply(func4), func4)
    self.assertEqual(schema.Functor([schema.Int()]).apply(func4), func4)
    self.assertEqual(schema.Functor(returns=schema.Any()).apply(func4), func4)

    self.assertEqual(schema.Functor().annotation, object_utils.Functor)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of positional argument 0 is not compatible'):
      schema.Callable([schema.Str()]).apply(func4)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of keyword argument \'b\' is not compatible'):
      schema.Callable(kw=[('b', schema.Int())]).apply(func4)

    with self.assertRaisesRegex(
        TypeError, '.* only take 2 positional arguments, while 3 is required'):
      schema.Callable([schema.Int(), schema.Str(), schema.Int()]).apply(func4)

    with self.assertRaisesRegex(TypeError,
                                'Keyword argument \'c\' does not exist'):
      schema.Callable(kw=[('c', schema.Int())]).apply(func4)

    class Functor2(object_utils.Functor):

      signature = schema.Signature(
          callable_type=schema.CallableType.FUNCTION,
          name='foo',
          module_name='__main__',
          args=[
              schema.Argument('a', schema.Int()),
              schema.Argument('b', schema.Str())
          ],
          varargs=schema.Argument('args', schema.Int()),
          varkw=schema.Argument('kwargs', schema.Int()),
          return_value=schema.Object(ValueError))

      def __init__(self, value):
        self.value = value

      def __call__(self, a, b, *args, **kwargs):
        del a, b, args, kwargs

    func5 = Functor2(1)
    self.assertEqual(schema.Callable().apply(func5), func5)
    self.assertEqual(schema.Callable([schema.Int()]).apply(func5), func5)
    self.assertEqual(
        schema.Callable([schema.Int(), schema.Str(),
                         schema.Int()]).apply(func5), func5)
    self.assertEqual(
        schema.Callable(kw=[('a', schema.Int())]).apply(func5), func5)
    self.assertEqual(
        schema.Callable(kw=[('c', schema.Int())]).apply(func5), func5)
    self.assertEqual(schema.Functor().apply(func5), func5)
    self.assertEqual(schema.Functor([schema.Int()]).apply(func5), func5)
    self.assertEqual(schema.Functor(returns=schema.Any()).apply(func5), func5)
    self.assertEqual(
        schema.Functor(returns=schema.Object(Exception)).apply(func5), func5)
    self.assertEqual(
        schema.Functor(returns=schema.Object(ValueError)).apply(func5), func5)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of positional argument 0 is not compatible'):
      schema.Callable([schema.Str()]).apply(func5)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of keyword argument \'b\' is not compatible'):
      schema.Callable(kw=[('b', schema.Int())]).apply(func5)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of positional argument 2 is not compatible '
        'with the value spec of \\*args'):
      schema.Callable([schema.Int(), schema.Str(), schema.Str()]).apply(func5)

    with self.assertRaisesRegex(
        TypeError, 'Value spec of keyword argument \'c\' is not compatible '
        'with the value spec of \\*\\*kwargs'):
      schema.Callable(kw=[('c', schema.Str())]).apply(func5)

    with self.assertRaisesRegex(
        TypeError, 'Value spec for return value is not compatible'):
      schema.Callable(returns=schema.Object(KeyError)).apply(func5)

    class CallableObject:

      def __call__(self, x, y):
        pass

    func6 = CallableObject()
    self.assertEqual(schema.Callable().apply(func6), func6)
    self.assertEqual(schema.Callable([schema.Int()]).apply(func6), func6)
    self.assertEqual(
        schema.Callable(kw=[('x', schema.Int())]).apply(func6), func6)

    with self.assertRaisesRegex(
        TypeError, '.* only take 2 positional arguments, while 3 is required'):
      schema.Callable([schema.Int(), schema.Int(), schema.Int()]).apply(func6)

    with self.assertRaisesRegex(TypeError,
                                'Keyword argument \'z\' does not exist'):
      schema.Callable(kw=[('z', schema.Int())]).apply(func6)

    with self.assertRaisesRegex(TypeError, 'Expect .*Functor'):
      schema.Functor().apply(func6)

    self.assertEqual(schema.Callable().apply(CallableObject), CallableObject)

    # Test user validator.
    def _value_is_one(func):
      if func.value != 1:
        raise ValueError('Value should be one')

    self.assertEqual(
        schema.Callable(user_validator=_value_is_one).apply(func4), func4)
    with self.assertRaisesRegex(ValueError, 'Value should be one \\(path=\\)'):
      schema.Callable(user_validator=_value_is_one).apply(Functor1(0))

    # Test is_compatible.
    self.assertTrue(schema.Callable().noneable().is_compatible(
        schema.Callable()))
    self.assertTrue(schema.Callable().is_compatible(
        schema.Callable([schema.Int()])))
    self.assertTrue(schema.Callable().is_compatible(
        schema.Callable(kw=[('a', schema.Int())])))
    self.assertTrue(schema.Callable().is_compatible(
        schema.Callable(kw=[('a', schema.Int())])))
    self.assertTrue(schema.Callable().is_compatible(
        schema.Functor(kw=[('a', schema.Int())])))

    self.assertTrue(schema.Callable().is_compatible(
        schema.Object(CallableObject)))
    self.assertTrue(
        schema.Callable(kw=[('x', schema.Any())]).is_compatible(
            schema.Object(CallableObject)))
    self.assertTrue(
        schema.Callable(kw=[('x',
                             schema.Any()), ('y', schema.Any())]).is_compatible(
                                 schema.Object(CallableObject)))

    self.assertFalse(
        schema.Callable([schema.Int()]).is_compatible(
            schema.Callable(kw=[('a', schema.Int())])))
    self.assertFalse(schema.Callable().is_compatible(
        schema.Callable().noneable()))
    self.assertFalse(
        schema.Callable([schema.Int()]).is_compatible(schema.Callable()))
    self.assertFalse(
        schema.Callable([schema.Int(min_value=0)]).is_compatible(
            schema.Callable([schema.Int(max_value=-1)])))
    self.assertFalse(
        schema.Callable(kw=[('a',
                             schema.Int())]).is_compatible(schema.Callable()))
    self.assertFalse(
        schema.Callable(kw=[('a', schema.Int())]).is_compatible(
            schema.Object(Exception)))
    self.assertFalse(
        schema.Callable(kw=[('a', schema.Int())]).is_compatible(
            schema.Callable(kw=[('a', schema.Str())])))
    self.assertFalse(
        schema.Callable(returns=schema.Int()).is_compatible(
            schema.Callable(returns=schema.Str())))

    # Test extend.
    self.assertEqual(schema.Callable().extend(schema.Callable().noneable()),
                     schema.Callable())
    self.assertEqual(
        schema.Callable(kw=[('a', schema.Str())]).extend(
            schema.Callable([schema.Int()], returns=schema.Any())),
        schema.Callable([schema.Int()],
                        kw=[('a', schema.Str())],
                        returns=schema.Any()))

  def testType(self):
    """Tests for Type."""
    # Test __str__.
    self.assertEqual(
        str(schema.Type(Exception)), 'Type(<class \'Exception\'>)')
    self.assertEqual(
        str(schema.Type(Exception).noneable()),
        'Type(<class \'Exception\'>, default=None, noneable=True)')
    self.assertEqual(
        str(schema.Type(Exception).noneable().freeze()),
        'Type(<class \'Exception\'>, default=None, noneable=True, frozen=True)')

    # Test value_type
    self.assertEqual(schema.Type(Exception).value_type, type)

    # Test annotation.
    self.assertEqual(schema.Type(Exception).annotation, typing.Type[Exception])
    self.assertEqual(
        schema.Type(Exception).noneable().annotation,
        typing.Optional[typing.Type[Exception]])

    # Test default.
    self.assertEqual(schema.Type(Exception).default, object_utils.MISSING_VALUE)
    self.assertEqual(
        schema.Type(Exception, default=ValueError).default, ValueError)

    # Test is_noneable
    self.assertFalse(schema.Type(Exception).is_noneable)
    self.assertTrue(schema.Type(Exception).noneable().is_noneable)

    # Test comparison.
    t = schema.Type(Exception)
    self.assertEqual(t, t)
    self.assertEqual(schema.Type(Exception), schema.Type(Exception))
    self.assertEqual(
        schema.Type(Exception).noneable(),
        schema.Type(Exception).noneable())
    self.assertEqual(
        schema.Type(Exception, default=ValueError),
        schema.Type(Exception, default=ValueError))
    self.assertNotEqual(schema.Type(Exception), schema.Type(int))
    self.assertNotEqual(
        schema.Type(Exception),
        schema.Type(Exception).noneable())
    self.assertNotEqual(
        schema.Type(Exception), schema.Type(Exception, default=ValueError))

    # Test apply.
    self.assertEqual(schema.Type(Exception).apply(Exception), Exception)
    self.assertEqual(schema.Type(Exception).apply(ValueError), ValueError)
    self.assertIsNone(schema.Type(Exception).noneable().apply(None))

    with self.assertRaisesRegex(ValueError, '.* is not a subclass of .*'):
      schema.Type(Exception).apply(int)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Type(Exception).apply(None)

    # Test is_compatible.
    self.assertTrue(
        schema.Type(Exception).is_compatible(schema.Type(Exception)))
    self.assertTrue(
        schema.Type(Exception).is_compatible(schema.Type(ValueError)))
    self.assertTrue(
        schema.Type(Exception).noneable().is_compatible(
            schema.Type(ValueError)))
    self.assertFalse(
        schema.Type(Exception).is_compatible(
            schema.Type(ValueError).noneable()))
    self.assertFalse(schema.Type(Exception).is_compatible(schema.Type(int)))

    # Test extends.
    # Child may make a parent default value not specified.
    self.assertEqual(
        schema.Type(Exception).extend(
            schema.Type(Exception, default=ValueError)).default,
        object_utils.MISSING_VALUE)

    # Child may extend a noneable base into non-noneable.
    self.assertFalse(
        schema.Type(Exception).extend(
            schema.Type(Exception).noneable()).is_noneable)

    # Child cannot extend a base of different type.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Type(Exception).extend(schema.Type(ValueError))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Type(Exception).noneable().extend(schema.Type(Exception))

  def testUnion(self):
    """Tests for Union."""
    # Test __str__.
    self.assertEqual(
        repr(schema.Union([schema.Int(), schema.Bool()])),
        'Union([Int(), Bool()])')
    self.assertEqual(
        repr(schema.Union([schema.Int(), schema.Bool()], default=1).freeze()),
        'Union([Int(), Bool()], default=1, frozen=True)')
    self.assertEqual(
        repr(schema.Union([schema.Int(), schema.Bool()], default=1).noneable()),
        'Union([Int(default=None, noneable=True), '
        'Bool(default=None, noneable=True)], default=1, noneable=True)')

    # Test value_type
    self.assertEqual(
        set(schema.Union([schema.Int(), schema.Bool()]).value_type),
        set([int, bool]))

    # Test annotation.
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()]).annotation,
        typing.Union[int, bool])
    self.assertEqual(
        schema.Union(
            [schema.Int(),
             schema.Union([schema.Bool(), schema.Str()])]).annotation,
        typing.Union[int, typing.Union[bool, typing.Text]])

    # schema.Callable.value_type is None, thus union of it with another spec
    # also produce None.
    self.assertIsNone(
        schema.Union([schema.Callable(), schema.Int()]).value_type)

    # Test default.
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()]).default,
        object_utils.MISSING_VALUE)
    self.assertIsNone(
        schema.Union([schema.Int(), schema.Bool()]).noneable().default)
    self.assertEqual(schema.Union([schema.Int(), schema.Bool()], 1).default, 1)

    # Test is_noneable
    self.assertFalse(schema.Union([schema.Int(), schema.Bool()]).is_noneable)
    self.assertTrue(
        schema.Union([schema.Int(), schema.Bool()]).noneable().is_noneable)
    self.assertFalse(
        schema.Union([schema.Int(), schema.Bool()]).candidates[0].is_noneable)
    self.assertTrue(
        schema.Union([schema.Int(),
                      schema.Bool()]).noneable().candidates[0].is_noneable)
    self.assertTrue(
        schema.Union([schema.Int().noneable(), schema.Bool()]).is_noneable)

    # Test comparison.
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()]),
        schema.Union([schema.Int(), schema.Bool()]))
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()]),
        schema.Union([schema.Bool(), schema.Int()]))
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()], 1),
        schema.Union([schema.Bool(), schema.Int()], 1))

    self.assertNotEqual(
        schema.Union([schema.Int(), schema.Bool()]), schema.Int())
    self.assertNotEqual(
        schema.Union([schema.Int(), schema.Bool()]),
        schema.Union([schema.Int(), schema.Str()]))
    self.assertNotEqual(
        schema.Union([schema.Int(), schema.Bool()]),
        schema.Union([schema.Int(), schema.Bool()]).noneable())
    self.assertNotEqual(
        schema.Union([schema.Int(min_value=0),
                      schema.Bool()]),
        schema.Union([schema.Int(), schema.Bool()]))
    self.assertNotEqual(
        schema.Union([schema.Int(), schema.Bool()], 1),
        schema.Union([schema.Int(), schema.Bool()]))
    self.assertNotEqual(
        schema.Union([schema.Int(), schema.Bool()]),
        schema.Union([schema.Int(), schema.Str(),
                      schema.Bool()]))

    # Test bad __init__.
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'candidates\' must be a list of at least 2 elements'):
      schema.Union(1)

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'candidates\' must be a list of at least 2 elements'):
      schema.Union([schema.Int()])

    with self.assertRaisesRegex(
        ValueError, 'Items in \'candidates\' must be ValueSpec objects.'):
      schema.Union([1, 2])

    with self.assertRaisesRegex(ValueError,
                                'Found 2 value specs of the same type '):
      schema.Union([schema.Int(min_value=1), schema.Int(max_value=2)])

    # Test get_candidate.
    self.assertEqual(
        schema.Union([schema.Int(),
                      schema.Float()]).get_candidate(schema.Float()),
        schema.Float())

    self.assertIsNone(
        schema.Union([schema.Int(),
                      schema.Float()]).get_candidate(schema.Float(min_value=1)))

    class A:

      def __call__(self):
        pass

    class B(A):
      pass

    self.assertEqual(
        schema.Union([schema.Object(A),
                      schema.Object(B)]).get_candidate(schema.Object(B)),
        schema.Object(B))

    self.assertEqual(
        schema.Union([schema.Callable(),
                      schema.Object(B)]).get_candidate(schema.Object(A)),
        schema.Object(B))

    self.assertEqual(
        schema.Union([schema.Callable(),
                      schema.Int()]).get_candidate(schema.Callable()),
        schema.Callable())

    self.assertEqual(
        schema.Union(
            [schema.Union([schema.Float(), schema.Int()]),
             schema.Str()]).get_candidate(schema.Int()), schema.Int())

    # Test apply.
    self.assertEqual(schema.Union([schema.Int(), schema.Str()]).apply(1), 1)
    self.assertEqual(
        schema.Union([schema.Int(), schema.Str()]).apply('abc'), 'abc')
    self.assertIsNone(
        schema.Union([schema.Int(), schema.Str()]).noneable().apply(None))
    self.assertEqual(
        schema.Union([schema.Int(), schema.Str()
                     ]).apply(object_utils.MISSING_VALUE, allow_partial=True),
        object_utils.MISSING_VALUE)

    with self.assertRaisesRegex(ValueError, 'Value cannot be None'):
      schema.Union([schema.Int(), schema.Str()]).apply(None)

    with self.assertRaisesRegex(
        TypeError, 'Expect \\(.*\\) but encountered <(type|class) \'list\'>.'):
      schema.Union([schema.Int(), schema.Str()]).apply([])

    # Test is_compatible.
    self.assertTrue(
        schema.Union([schema.Int(), schema.Bool()]).is_compatible(
            schema.Union([schema.Bool(), schema.Int()])))

    self.assertTrue(
        schema.Union([schema.Int(), schema.Bool(),
                      schema.Str()]).is_compatible(
                          schema.Union([schema.Int(),
                                        schema.Str()])))

    self.assertTrue(
        schema.Union([schema.Object(A), schema.Bool()]).is_compatible(
            schema.Union([schema.Object(B), schema.Bool()])))

    self.assertTrue(
        schema.Union([schema.Int(), schema.Bool()]).noneable().is_compatible(
            schema.Int(min_value=1).noneable()))

    self.assertFalse(
        schema.Union([schema.Int(), schema.Bool()]).is_compatible(schema.Str()))

    self.assertFalse(
        schema.Union([schema.Int(min_value=1),
                      schema.Bool()]).is_compatible(schema.Int()))

    self.assertFalse(
        schema.Union([schema.Int(min_value=1),
                      schema.Bool()]).is_compatible(
                          schema.Union([schema.Int(),
                                        schema.Bool()])))

    self.assertFalse(
        schema.Union([schema.Int(),
                      schema.Bool()]).is_compatible(schema.Int().noneable()))

    self.assertFalse(
        schema.Union([schema.Object(B),
                      schema.Bool()]).is_compatible(schema.Object(A)))

    # Test extends.
    # Child without constraints will inheirt constraints.
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()]).extend(
            schema.Union([schema.Bool(),
                          schema.Int(min_value=0)])).candidates[0],
        schema.Int(min_value=0))

    # Narrow constraint.
    self.assertEqual(
        schema.Union([schema.Int(), schema.Bool()]).extend(
            schema.Union([schema.Bool(),
                          schema.Str(),
                          schema.Int(min_value=0)])),
        schema.Union([schema.Int(min_value=0),
                      schema.Bool()]))

    self.assertFalse(
        schema.Union([schema.Int(), schema.Str()]).extend(
            schema.Union([schema.Int(), schema.Str()]).noneable()).is_noneable)

    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type'):
      schema.Union([schema.Int(), schema.Bool()]).extend(schema.Int())

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible value spec .*'):
      schema.Union([schema.Int(), schema.Bool()
                   ]).extend(schema.Union([schema.Str(),
                                           schema.Bool()]))

    # Child cannot extend a non-noneable base to noneable.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):
      schema.Union([schema.Int(), schema.Bool()]).noneable().extend(
          schema.Union([schema.Int(), schema.Bool()]))

    # Test enum of different values cannot be extended.
    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: values in base should be super set.'):
      schema.Union([
          schema.Enum(1, [1, 2]), schema.Int()
      ]).extend(schema.Union([schema.Enum('a', ['a', 'b']),
                              schema.Int()]))

    # Child with larger max_size cannot extend base with smaller max_size.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: max_value is larger.'):
      schema.Union([schema.Int(max_value=100),
                    schema.Bool()]).extend(
                        schema.Union([schema.Int(max_value=0),
                                      schema.Bool()]))
    # Test schema.Object
    schema.Union([schema.Object(int), schema.Object(float)])
    with self.assertRaisesRegex(ValueError,
                                'Found 2 value specs of the same type'):
      schema.Union([schema.Object(int), schema.Object(int)])

  def testAny(self):
    """Tests for Any."""
    # Test __str__.
    self.assertEqual(str(schema.Any()), 'Any()')
    self.assertEqual(str(schema.Any(1)), 'Any(default=1)')
    self.assertEqual(str(schema.Any(1).freeze()), 'Any(default=1, frozen=True)')

    # Test value_type
    self.assertEqual(schema.Any().value_type, object)

    # Test annotation.
    self.assertEqual(schema.Any().annotation, schema.MISSING_VALUE)
    self.assertEqual(schema.Any().noneable().annotation, schema.MISSING_VALUE)
    self.assertEqual(schema.Any(annotation=int).noneable().annotation, int)

    # Test default.
    self.assertEqual(schema.Any().default, object_utils.MISSING_VALUE)
    self.assertIsNone(schema.Any().noneable().default)
    self.assertEqual(schema.Any(True).default, True)

    # Test is_noneable
    self.assertTrue(schema.Any().is_noneable)

    # Test comparison.
    self.assertEqual(schema.Any(), schema.Any())
    self.assertEqual(schema.Any(True), schema.Any(True))
    self.assertNotEqual(schema.Any(), schema.Int())
    self.assertNotEqual(schema.Any(True), schema.Any())

    # Test apply.
    self.assertEqual(schema.Any().apply(True), True)
    self.assertEqual(schema.Any().apply(1), 1)
    self.assertIsNone(schema.Any().apply(None))

    # Test user validator.
    def _value_is_none(value):
      if value is not None:
        raise ValueError('Value should be None.')

    self.assertIsNone(schema.Any(user_validator=_value_is_none).apply(None))

    with self.assertRaisesRegex(ValueError,
                                'Value should be None. \\(path=\\)'):
      schema.Any(user_validator=_value_is_none).apply(1)

    # Test is_compatible.
    self.assertTrue(schema.Any().is_compatible(schema.Int()))
    self.assertTrue(schema.Any().is_compatible(schema.Int().noneable()))

    # Test extends.
    # Child may change default value.
    self.assertEqual(schema.Any(False).extend(schema.Any(True)).default, False)

    # Child may make a parent default value not specified.
    self.assertTrue(schema.Any().extend(schema.Any(True)).default)

    # Child cannot extend a base with different type.
    with self.assertRaisesRegex(TypeError,
                                '.* cannot extend .*: incompatible type.'):
      schema.Any().extend(schema.Int())

  def testFrozenValueSpec(self):
    """Test frozen value spec."""
    vs = schema.Int()
    self.assertFalse(vs.frozen)

    vs = schema.Int().freeze(1)
    self.assertTrue(vs.frozen)
    self.assertEqual(vs.default, 1)
    self.assertEqual(vs.apply(1), 1)
    self.assertEqual(vs.apply(schema.MISSING_VALUE), 1)
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      vs.apply(2)

    vs = schema.Int(default=2).freeze()
    self.assertTrue(vs.frozen)
    self.assertEqual(vs.apply(schema.MISSING_VALUE), 2)
    self.assertEqual(vs.apply(2), 2)
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      vs.apply(1)

    with self.assertRaisesRegex(
        TypeError, 'Cannot extend a frozen value spec.'):
      schema.Int(min_value=1).extend(vs)

    # Test freeze a value spec without a default value.
    with self.assertRaisesRegex(
        ValueError, 'Cannot freeze .* without a default value'):
      schema.Int().freeze()

    # Test frozen value for dict.
    vs = schema.Dict([
        ('x', schema.Int(default=1)),
        ('y', schema.Bool(default=True))
    ])
    vs.freeze()
    self.assertTrue(vs.frozen)
    self.assertEqual(vs.apply(schema.MISSING_VALUE), dict(x=1, y=True))

  def testConverter(self):
    """Test converter."""
    # Test built-in converter between int and datetime.datetime.
    timestamp = calendar.timegm(datetime.datetime.now().timetuple())
    now = datetime.datetime.utcfromtimestamp(timestamp)
    self.assertEqual(schema.Object(datetime.datetime).apply(timestamp), now)
    self.assertEqual(schema.Int().apply(now), timestamp)
    self.assertEqual(
        schema.get_json_value_converter(datetime.datetime)(now), timestamp)

    # Test built-in converter between string and KeyPath.
    self.assertEqual(
        schema.Object(object_utils.KeyPath).apply('a.b.c').keys,
        ['a', 'b', 'c'])
    self.assertEqual(
        schema.Union([schema.Object(object_utils.KeyPath),
                      schema.Int()]).apply('a.b.c').keys, ['a', 'b', 'c'])
    self.assertEqual(
        schema.Union([schema.Object(object_utils.KeyPath),
                      schema.Int()]).apply('a.b.c').keys, ['a', 'b', 'c'])
    self.assertEqual(schema.Str().apply(object_utils.KeyPath.parse('a.b.c')),
                     'a.b.c')
    self.assertEqual(
        schema.get_json_value_converter(object_utils.KeyPath)(
            object_utils.KeyPath.parse('a.b.c')), 'a.b.c')

    # Test custom converter.
    class A:

      def __init__(self, x):
        self.x = x

    class B:
      pass

    schema.register_converter((int, str), A, A)
    schema.register_converter(A, int, lambda a: a.x)

    # NOTE(daiyip): Consider places that accepts B also accepts A.
    schema.register_converter(A, B, lambda x: x)

    with self.assertRaisesRegex(
        TypeError,
        'Argument \'src\' and \'dest\' must be a type or tuple of types.'):
      schema.register_converter(0, 1, lambda x: x)

    self.assertEqual(
        schema.Union([schema.Object(B), schema.Float()]).apply(A(1)).x, 1)
    self.assertEqual(schema.Object(A).apply(1).x, 1)
    self.assertEqual(schema.Object(A).apply('foo').x, 'foo')
    self.assertEqual(schema.get_json_value_converter(A)(A(1)), 1)
    self.assertIsNone(schema.get_json_value_converter(B))


class MissingValueTest(unittest.TestCase):
  """Tests for MissingValue class."""

  def testEqual(self):
    """Test MissingValue.__eq__."""
    self.assertEqual(
        schema.MissingValue(schema.Int()), object_utils.MISSING_VALUE)

    self.assertEqual(object_utils.MISSING_VALUE,
                     schema.MissingValue(schema.Int()))

    self.assertEqual(
        schema.MissingValue(schema.Int()), schema.MissingValue(schema.Int()))

    self.assertNotEqual(
        schema.MissingValue(schema.Int()),
        schema.MissingValue(schema.Int(max_value=1)))

    self.assertNotEqual(
        schema.MissingValue(schema.Int()), schema.MissingValue(schema.Str()))

    m = schema.MissingValue(schema.Int())
    self.assertEqual(m, m)

  def testHash(self):
    """Test MissingValue.__hash__."""
    self.assertEqual(
        hash(schema.MissingValue(schema.Int())),
        hash(schema.MissingValue(schema.Float())))

    self.assertEqual(
        hash(schema.MissingValue(schema.Int())), hash(schema.MISSING_VALUE))

    self.assertNotEqual(hash(schema.MissingValue(schema.Int())), hash(1))

  def testFormat(self):
    """Test MissingValue.format."""
    self.assertEqual(
        schema.MissingValue(schema.Int()).format(compact=True), 'MISSING_VALUE')

    self.assertEqual(
        schema.MissingValue(schema.Int()).format(compact=False),
        'MISSING_VALUE(Int())')


class SimpleObject:
  """Simple object for testing."""
  pass


class SchemaTest(unittest.TestCase):
  """Tests for Schema class."""

  def _create_test_schema(self, init_arg_list=None):
    return schema.Schema([
        schema.Field('a', schema.Int(1), 'Field a.'),
        schema.Field('b', schema.Bool().noneable(), 'Field b.'),
        schema.Field('c', schema.Dict([
            schema.Field('d', schema.List(
                schema.Enum(0, [0, 1, None]), default=[0, 1]), 'Field d.'),
            schema.Field('e', schema.List(
                schema.Dict([
                    (schema.StrKey(regex='foo.*'), schema.Str(),
                     'Mapped values.')
                ])
            ), 'Field e.'),
            schema.Field('f', schema.Object(SimpleObject), 'Field f.')
        ]).noneable(), 'Field c.'),
    ], metadata={'init_arg_list': init_arg_list or []})

  def testBasics(self):
    s = schema.Schema([schema.Field('a', schema.Int())], 'schema1',
                      [schema.create_schema([('b', schema.Bool())])])

    # Test Schema.fields.
    self.assertEqual(
        s.fields,
        {
            schema.ConstStrKey('b'): schema.Field(
                schema.ConstStrKey('b'), schema.Bool()),
            schema.ConstStrKey('a'): schema.Field(
                schema.ConstStrKey('a'), schema.Int()),
        })

    # Test Schema.__getitem__ and Schema.__contains__.
    self.assertEqual(s['a'], schema.Field('a', schema.Int()))
    with self.assertRaises(KeyError):
      _ = s['foo']
    self.assertEqual(s.get('a'), schema.Field('a', schema.Int()))
    self.assertIsNone(s.get('foo'))
    self.assertIn('a', s)
    self.assertNotIn('foo', s)

    # Test Schema.keys and Schema.values
    self.assertEqual(
        list(s.keys()), [schema.ConstStrKey('b'), schema.ConstStrKey('a')])
    self.assertEqual(
        list(s.values()),
        [schema.Field(schema.ConstStrKey('b'), schema.Bool()),
         schema.Field(schema.ConstStrKey('a'), schema.Int())])

    # Test Schema.allow_nonconst_keys
    self.assertFalse(s.allow_nonconst_keys)

    # Test Schema.name
    self.assertEqual(s.name, 'schema1')

    # Test Schema.set_name
    s.set_name('schema2')
    self.assertEqual(s.name, 'schema2')

    # Test equal.
    self.assertEqual(s, s)
    self.assertNotEqual(s, schema.create_schema([]))

    with self.assertRaisesRegex(TypeError,
                                'Argument \'fields\' must be a list.'):
      schema.Schema({'a': schema.Int()})

    with self.assertRaisesRegex(ValueError,
                                'NonConstKey is not allowed in schema'):
      schema.Schema([schema.Field(schema.StrKey(), schema.Int())],
                    allow_nonconst_keys=False)

  def testMetadata(self):
    """Tests for Schema.metadata."""
    self.assertEqual(schema.Schema([]).metadata, {})
    self.assertEqual(
        schema.Schema([], metadata={
            'meta1': 1
        }).metadata, {'meta1': 1})

  def testEqual(self):
    """Tests for Schema.__eq__."""
    self.assertEqual(self._create_test_schema(), self._create_test_schema())

  def testFormat(self):
    """Tests for Schema.format."""
    self.assertEqual(
        self._create_test_schema().format(compact=True),
        'Schema(a=Int(default=1), b=Bool(default=None, noneable=True), '
        'c=Dict({d=List('
        'Enum(default=0, values=[0, 1, None]), default=[0, 1]), '
        'e=List('
        'Dict({StrKey(regex=\'foo.*\')=Str()})), '
        'f=Object(SimpleObject)}, noneable=True))')

    self.assertEqual(
        inspect.cleandoc(self._create_test_schema().format(
            compact=False, verbose=False)),
        inspect.cleandoc("""Schema(
          a = Int(default=1),
          b = Bool(default=None, noneable=True),
          c = Dict({
            d = List(Enum(default=0, values=[0, 1, None]), default=[0, 1]),
            e = List(Dict({
              StrKey(regex=\'foo.*\') = Str()
            })),
            f = Object(SimpleObject)
          }, noneable=True)
        )"""))

    self.assertEqual(
        inspect.cleandoc(
            # Equal to schema.format(compact=False, verbose=True)
            str(self._create_test_schema())),
        inspect.cleandoc("""Schema(
            # Field a.
            a = Int(default=1),

            # Field b.
            b = Bool(default=None, noneable=True),

            # Field c.
            c = Dict({
              # Field d.
              d = List(Enum(default=0, values=[0, 1, None]), default=[0, 1]),

              # Field e.
              e = List(Dict({
                # Mapped values.
                StrKey(regex=\'foo.*\') = Str()
              })),

              # Field f.
              f = Object(SimpleObject)
            }, noneable=True)
          )"""))

  def testExtend(self):
    """Tests for Schema.extend."""

    # Disjoint inheirtance with preserved field order.
    self.assertEqual(
        list(
            schema.create_schema([
                ('a', schema.Int()), ('c', schema.Dict([('d', schema.Str())]))
            ]).extend(
                schema.create_schema([('b', schema.Bool().noneable()),
                                      ('c',
                                       schema.Dict([
                                           ('e', schema.Object(SimpleObject)),
                                       ]))])).values()),
        list(
            schema.create_schema([
                # Order matters!
                ('b', schema.Bool().noneable()),
                ('c',
                 schema.Dict([
                     ('e', schema.Object(SimpleObject)),
                     ('d', schema.Str()),
                 ])),
                ('a', schema.Int()),
            ]).values()))

    # Override field with default value.
    self.assertEqual(
        list(
            schema.create_schema([
                ('b', True),
            ]).extend(schema.create_schema([
                ('b', schema.Bool().noneable()),
            ])).values()),
        list(schema.create_schema([
            ('b', schema.Bool(True)),
        ]).values()))

    # Extend a base schema with non-const keys while current schema does not
    # allow const keys.
    with self.assertRaisesRegex(
        ValueError,
        'Non-const key .* is not allowed to be added to the schema'):
      schema.create_schema(
          [('b', schema.Int())], allow_nonconst_keys=False).extend(
              schema.create_schema([(schema.StrKey(), schema.Int())],
                                   allow_nonconst_keys=True))

    # Override field with different type.
    with self.assertRaisesRegex(
        TypeError, 'Int\\(default=1\\) cannot extend Bool\\(.*\\): '
        'incompatible type. \\(path=b\\)'):
      schema.create_schema([
          ('b', 1),
      ]).extend(schema.create_schema([
          ('b', schema.Bool().noneable()),
      ]))

    # Override metadata.
    self.assertEqual(
        schema.create_schema([], metadata={
            'meta2': 'bar',
            'meta3': 'new'
        }).extend(
            schema.create_schema([], metadata={
                'meta1': 1,
                'meta2': 'foo'
            })).metadata, {
                'meta1': 1,
                'meta2': 'bar',
                'meta3': 'new'
            })

  def testIsCompatible(self):
    """Tests for Schema.is_compatible."""
    with self.assertRaisesRegex(TypeError,
                                'Argument \'other\' should be a Schema object'):
      schema.create_schema([]).is_compatible(1)

    self.assertFalse(
        schema.create_schema([]).is_compatible(
            schema.create_schema([('a', schema.Int())])))

    self.assertFalse(
        schema.create_schema([('a', schema.Int())
                             ]).is_compatible(schema.create_schema([])))

    self.assertFalse(
        schema.create_schema([
            ('a', schema.Int())
        ]).is_compatible(schema.create_schema([('a', schema.Str())])))

    self.assertTrue(
        schema.create_schema([
            ('a', schema.Any())
        ]).is_compatible(schema.create_schema([('a', schema.Str())])))

  def testGetField(self):
    """Tests for Schema.get_field."""
    s = schema.create_schema([('a', schema.Int()),
                              (schema.StrKey('foo.*'), schema.Int()),
                              (schema.StrKey('f.*'), schema.Bool())],
                             allow_nonconst_keys=True)
    self.assertEqual(s.get_field('a'), schema.Field('a', schema.Int()))
    self.assertIsNone(s.get_field('b'))
    self.assertEqual(
        s.get_field('foo1'), schema.Field(schema.StrKey('foo.*'), schema.Int()))
    self.assertEqual(
        s.get_field('far'), schema.Field(schema.StrKey('f.*'), schema.Bool()))

  def testResolve(self):
    """Tests for Schema.resolve."""
    s = schema.create_schema([
        ('a', schema.Int()),
        ('b', schema.Int()),
        (schema.StrKey('foo.*'), schema.Int()),
    ],
                             allow_nonconst_keys=True)
    matched, unmatched = s.resolve(['a', 'b', 'c', 'foo1', 'foo2', 'd'])
    self.assertEqual(matched, {
        'a': ['a'],
        'b': ['b'],
        schema.StrKey('foo.*'): ['foo1', 'foo2'],
    })
    self.assertEqual(unmatched, ['c', 'd'])

  def testValidate(self):
    """Tests for Schema.validate."""
    # Validate fully specified fields.
    self._create_test_schema().validate({
        'a': 1,
        'b': False,
        'c': {
            'd': [None],
            'e': [{
                'foo1': 'bar'
            }],
            'f': SimpleObject()
        }
    })

    # Validate partially specified fields.
    # This should pass since field 'c' is noneable.
    self._create_test_schema().validate({})

    # Missing required field 'c.e'.
    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified. \\(Path=\'c\\.e\'.*\\)'):
      self._create_test_schema().validate({'c': {}})

    # Wrong key in field 'c.e[0]'
    with self.assertRaisesRegex(
        KeyError, 'Keys \\[\'bar\'\\] are not allowed in Schema. \\(parent='
        '\'c\\.e\\[0\\]\'\\)'):
      self._create_test_schema().validate({'c': {'e': [{'bar': 'foo'}]}})

    # Wrong type in field 'c.f'
    with self.assertRaisesRegex(
        TypeError, 'Expect <class \'pyglove.core.typing_test.SimpleObject\'> but encountered '
        '<(type|class) \'int\'>: 1. \\(path=c\\.f\\)'):
      self._create_test_schema().validate(
          {'c': {
              'e': [{
                  'foo': 'bar'
              }],
              'f': 1
          }})

  def testApply(self):
    """Tests for Schema.apply."""
    # Use default.
    self.assertEqual(self._create_test_schema().apply({}), {
        'a': 1,
        'b': None,
        'c': None,
    })

    # Apply using complex object.
    so = SimpleObject()
    self.assertEqual(
        self._create_test_schema().apply({
            'a': 2,
            'b': True,
            'c': {
                'e': [{
                    'foo': 'bar'
                }],
                'f': so
            }
        }), {
            'a': 2,
            'b': True,
            'c': {
                'd': [0, 1],
                'e': [{
                    'foo': 'bar'
                }],
                'f': so
            }
        })

    # Raises when input doesn't match schema.
    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified. \\(Path=\'c\\.e\'.*\\)'):
      self._create_test_schema().apply({'c': {}})

    self.assertEqual(
        self._create_test_schema().apply({'c': {}}, allow_partial=True),
        {
            'a': 1,
            'b': None,
            'c': {
                'd': [0, 1],
                # NOTE: we can use object_utils.MISSING_VALUE as general missing
                # value identifier.
                'e': object_utils.MISSING_VALUE,
                # Or we can use missing value for specific value spec.
                'f': schema.MissingValue(schema.Object(SimpleObject)),
            }
        })

  def testApplyWithCustomTyping(self):
    """Tests for Schema.apply with value placeholder."""

    class NumberType(schema.CustomTyping):

      def __init__(self, value):
        self.value = value

      def custom_apply(self, root_path, value_spec, **kwargs):
        # Pass through value to Enum.apply.
        if isinstance(value_spec, schema.Enum):
          return (True, self.value)

        if not isinstance(value_spec, schema.Number):
          raise ValueError(f'NumberType can only apply to numbers. '
                           f'(path=\'{root_path}\')')
        return (False, self)

      def __eq__(self, other):
        return isinstance(other, NumberType) and self.value == other.value

      def __ne__(self, other):
        return not self.__eq__(other)

    s = schema.create_schema([('a', schema.Int(1)),
                              ('b', schema.Bool().noneable()),
                              ('c',
                               schema.Dict([
                                   ('d', schema.Int()),
                                   ('e', schema.Enum(0, [0, 1])),
                               ]).noneable())])

    self.assertEqual(
        s.apply({
            'a': NumberType(1),
            'c': {
                'd': NumberType(2),
                'e': NumberType(0),
            }
        }), {
            'a': NumberType(1),
            'b': None,
            'c': {
                'd': NumberType(2),
                'e': 0
            }
        })

    with self.assertRaisesRegex(ValueError,
                                'NumberType can only apply to numbers.'):
      s.apply({
          'b': NumberType(1),
      })

  def testApplyCustomTransform(self):
    """Tests for Schema.apply with custom transform_fn."""
    transformed = []

    def _transform_fn(path, field, value):
      transformed.append((path, field.key, copy.deepcopy(value)))
      if isinstance(field.value, schema.Int):
        return value + 1
      return value

    self.assertEqual(
        self._create_test_schema().apply(
            {
                'b': False,
                'c': {
                    'd': [0],
                    'e': [{
                        'foo': 'bar'
                    }],
                },
            },
            allow_partial=True,
            child_transform=_transform_fn),
        {
            'a': 2,
            'b': False,
            'c': {
                # d is Enum type.
                'd': [0],
                'e': [{
                    'foo': 'bar'
                }],
                'f': object_utils.MISSING_VALUE,
            },
        })

    self.assertEqual(transformed, [
        ('a', 'a', 1),
        ('b', 'b', False),
        ('c.d[0]', schema.ListKey(), 0),
        ('c.d', 'd', [0]),
        ('c.e[0].foo', schema.StrKey('foo.*'), 'bar'),
        ('c.e[0]', schema.ListKey(), {
            'foo': 'bar'
        }),
        ('c.e', 'e', [{
            'foo': 'bar'
        }]),
        ('c.f', 'f', object_utils.MISSING_VALUE),
        ('c', 'c', {
            'd': [0],
            'e': [{
                'foo': 'bar'
            }],
            'f': object_utils.MISSING_VALUE
        }),
    ])

  def testDefaultValues(self):
    """Test Schema.default_values method."""
    self.assertEqual(self._create_test_schema().apply({}, allow_partial=False),
                     {
                         'a': 1,
                         'b': None,
                         'c': None,
                     })

    self.assertEqual(
        self._create_test_schema().apply({'c': {}}, allow_partial=True),
        {
            'a': 1,
            'b': None,
            'c': {
                'd': [0, 1],
                'e': object_utils.MISSING_VALUE,
                'f': object_utils.MISSING_VALUE,
            }
        })


class CreateSchemaTest(unittest.TestCase):
  """Tests for schema.create_schema."""

  def testBasics(self):
    """Test basic cases."""
    s = schema.create_schema([('a', 1), ('b', 'foo', 'field b'),
                              ('c', True, 'field c', {
                                  'user_data': 1
                              }), ('d', 1.0), ('e', schema.Enum(0, [0, 1]))],
                             'schema1',
                             metadata={'user_data': 2})

    self.assertEqual(s.name, 'schema1')
    self.assertEqual(s['a'], schema.Field('a', schema.Int(1)))
    self.assertEqual(s['b'],
                     schema.Field('b', schema.Str('foo'), 'field b'))
    self.assertEqual(
        s['c'],
        schema.Field('c', schema.Bool(True), 'field c', {'user_data': 1}))
    self.assertEqual(s['d'], schema.Field('d', schema.Float(1.0)))
    self.assertEqual(s['e'], schema.Field('e', schema.Enum(0, [0, 1])))
    self.assertEqual(s.metadata, {'user_data': 2})

  def testBadInput(self):
    """Test bad input."""
    with self.assertRaisesRegex(TypeError,
                                'Metadata of schema should be a dict.'):
      schema.create_schema([], metadata=1)

    with self.assertRaisesRegex(
        TypeError, 'Field definition should be tuples with 2 to 4 elements.'):
      schema.create_schema(['a'])

    with self.assertRaisesRegex(
        TypeError, 'Field definition should be tuples with 2 to 4 elements.'):
      schema.create_schema([('a',)])


class EnsureValueSpecTest(unittest.TestCase):
  """Tests for schema.ensure_value_spec."""

  def testBasics(self):
    """Tests basic cases."""
    self.assertEqual(
        schema.ensure_value_spec(schema.Int(min_value=1), schema.Int()),
        schema.Int(min_value=1))

    self.assertEqual(
        schema.ensure_value_spec(schema.Int(min_value=1), schema.Number(int)),
        schema.Int(min_value=1))

    with self.assertRaisesRegex(
        TypeError, 'Source spec .* is not compatible with destination spec'):
      schema.ensure_value_spec(schema.Int(min_value=1), schema.Bool())

  def testUnion(self):
    """Test Union as source type."""
    self.assertEqual(
        schema.ensure_value_spec(
            schema.Union([schema.Int(), schema.Str(regex='a.*')]),
            schema.Str()), schema.Str(regex='a.*'))

    with self.assertRaisesRegex(
        TypeError, 'Source spec .* is not compatible with destination spec'):
      schema.ensure_value_spec(
          schema.Union([schema.Int(), schema.Str()]), schema.Bool())

  def testAny(self):
    """Test Any as source type."""
    self.assertIsNone(schema.ensure_value_spec(schema.Any(), schema.Int()))


class SignatureTest(unittest.TestCase):
  """Tests for `schema.Signature`."""

  def testBasics(self):
    """Test basics of schema.Signature."""

    def foo(a, b: int = 1):
      del a, b

    signature = schema.get_signature(foo)
    self.assertEqual(signature.module_name, 'pyglove.core.typing_test')
    self.assertEqual(signature.name, 'foo')
    self.assertEqual(signature.id,
                     'pyglove.core.typing_test.SignatureTest.testBasics.<locals>.foo')
    self.assertEqual(
        str(signature),
        'Signature(\'pyglove.core.typing_test.SignatureTest.testBasics.<locals>.foo\', '
        'args=[(\'a\', Any()), '
        '(\'b\', Any(default=1, annotation=<class \'int\'>))])')

    self.assertEqual(signature.named_args, [
        schema.Argument('a', schema.Any()),
        schema.Argument('b',
                        schema.Any(default=1).annotate(int)),
    ])
    self.assertEqual(signature.arg_names, ['a', 'b'])

    # Test __eq__ and __ne__.
    def assert_not_equal(signature, field_name, modified_value):
      other = copy.copy(signature)
      setattr(other, field_name, modified_value)
      self.assertNotEqual(signature, other)

    assert_not_equal(signature, 'name', 'bar')
    assert_not_equal(signature, 'module_name', 'other_module')
    assert_not_equal(signature, 'args',
                     [signature.args[0],
                      schema.Argument('b', schema.Int())])
    assert_not_equal(
        signature, 'kwonlyargs',
        list(signature.kwonlyargs) + [schema.Argument('x', schema.Any())])
    assert_not_equal(signature, 'varargs',
                     schema.Argument('args', schema.Any()))
    assert_not_equal(signature, 'varkw',
                     schema.Argument('kwargs', schema.Any()))
    self.assertNotEqual(signature, 1)
    self.assertEqual(signature, signature)
    self.assertEqual(signature, copy.deepcopy(signature))

    with self.assertRaisesRegex(TypeError, '.* is not callable'):
      schema.get_signature(1)

  def testFunction(self):
    """Test for schema.get_signature on regular functions."""

    def foo(a, b: int = 1, **kwargs):
      del a, b, kwargs

    signature = schema.get_signature(foo)
    self.assertEqual(signature.callable_type, schema.CallableType.FUNCTION)
    self.assertEqual(signature.args, [
        schema.Argument('a', schema.Any()),
        schema.Argument('b',
                        schema.Any(default=1).annotate(int)),
    ])
    self.assertEqual(signature.kwonlyargs, [])
    self.assertIsNone(signature.varargs)
    self.assertEqual(signature.varkw, schema.Argument('kwargs', schema.Any()))
    self.assertFalse(signature.has_varargs)
    self.assertTrue(signature.has_varkw)
    self.assertTrue(signature.has_wildcard_args)
    self.assertEqual(
        signature.get_value_spec('b'), schema.Any(default=1, annotation=int))
    # NOTE: 'x' matches **kwargs
    self.assertEqual(signature.get_value_spec('x'), schema.Any())

  def testLambda(self):
    """Test for schema.get_signature on lambda function."""
    signature = schema.get_signature(lambda x: x)
    self.assertEqual(signature.callable_type, schema.CallableType.FUNCTION)
    self.assertEqual(signature.args, [schema.Argument('x', schema.Any())])
    self.assertEqual(signature.kwonlyargs, [])
    self.assertIsNone(signature.varargs)
    self.assertIsNone(signature.varkw)
    self.assertFalse(signature.has_varargs)
    self.assertFalse(signature.has_varkw)
    self.assertFalse(signature.has_wildcard_args)
    self.assertIsNone(signature.get_value_spec('y'))

  def testMethod(self):
    """Test for schema.get_signature on class methods."""

    class A:

      @classmethod
      def foo(cls, x: int = 1):
        return x

      def bar(self, y: int, *args, z=1):
        del args
        return y + z

      def __call__(self, z: int, **kwargs):
        del kwargs
        return z

    # Test class static method.
    signature = schema.get_signature(A.foo)
    self.assertEqual(signature.callable_type, schema.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [schema.Argument('x',
                         schema.Any(default=1).annotate(int))])
    self.assertEqual(signature.kwonlyargs, [])

    # Test instance method.
    signature = schema.get_signature(A().bar)
    self.assertEqual(signature.callable_type, schema.CallableType.METHOD)
    self.assertEqual(signature.args,
                     [schema.Argument('y',
                                      schema.Any().annotate(int))])
    self.assertEqual(signature.kwonlyargs,
                     [schema.Argument('z', schema.Any(default=1))])
    self.assertEqual(signature.varargs, schema.Argument('args', schema.Any()))
    self.assertTrue(signature.has_varargs)
    self.assertFalse(signature.has_varkw)

    # Test unbound instance method
    signature = schema.get_signature(A.bar)
    self.assertEqual(signature.callable_type, schema.CallableType.FUNCTION)
    self.assertEqual(signature.args, [
        schema.Argument('self', schema.Any()),
        schema.Argument('y',
                        schema.Any().annotate(int))
    ])
    self.assertEqual(signature.kwonlyargs,
                     [schema.Argument('z', schema.Any(default=1))])
    self.assertEqual(signature.varargs, schema.Argument('args', schema.Any()))
    self.assertTrue(signature.has_varargs)
    self.assertFalse(signature.has_varkw)

    # Test object as callable.
    signature = schema.get_signature(A())
    self.assertEqual(signature.callable_type, schema.CallableType.METHOD)
    self.assertEqual(signature.args,
                     [schema.Argument('z',
                                      schema.Any().annotate(int))])
    self.assertEqual(signature.kwonlyargs, [])
    self.assertFalse(signature.has_varargs)
    self.assertTrue(signature.has_varkw)
    self.assertEqual(signature.varkw, schema.Argument('kwargs', schema.Any()))

  def testGetArgFields(self):
    """Test get_arg_fields."""
    signature = schema.get_signature(lambda a, *args, b=1, **kwargs: 1)
    arg_fields = schema.get_arg_fields(signature)
    self.assertEqual(arg_fields, [
        schema.Field('a', schema.Any(), 'Argument \'a\'.'),
        schema.Field('args', schema.List(schema.Any(), default=[]),
                     'Wildcard positional arguments.'),
        schema.Field('b',
                     schema.Any().set_default(1), 'Argument \'b\'.'),
        schema.Field(schema.StrKey(), schema.Any(),
                     'Wildcard keyword arguments.')
    ])

    # Full specification.
    signature = schema.get_signature(lambda a, *args, b='foo', **kwargs: 1)
    arg_fields = schema.get_arg_fields(signature, [
        ('b', schema.Str()),
        ('args', schema.List(schema.Int())),
        ('a', schema.Int()),
        ('c', schema.Str()),
    ])
    self.assertEqual(arg_fields, [
        schema.Field('a', schema.Int()),
        schema.Field('args', schema.List(schema.Int(), default=[])),
        schema.Field('b', schema.Str(default='foo')),
        schema.Field('c', schema.Str()),
        schema.Field(schema.StrKey(), schema.Any(),
                     'Wildcard keyword arguments.')
    ])

    # Partial specification.
    signature = schema.get_signature(lambda a, b='foo': 1)
    arg_fields = schema.get_arg_fields(signature, [
        ('b', schema.Str()),
    ])
    self.assertEqual(arg_fields, [
        schema.Field('a', schema.Any(), 'Argument \'a\'.'),
        schema.Field('b', schema.Str(default='foo')),
    ])

    # Special cases for Dict
    signature = schema.get_signature(lambda a: 1)
    arg_fields = schema.get_arg_fields(signature, [
        ('a', schema.Dict([
            ('x', schema.Int())
        ])),
    ])
    self.assertEqual(arg_fields, [
        schema.Field('a', schema.Dict([('x', schema.Int())]))
    ])

    # Bad specifications.
    with self.assertRaisesRegex(
        KeyError, 'multiple StrKey found in symbolic arguments declaration.'):
      schema.get_arg_fields(
          schema.get_signature(lambda a: 1), [(schema.StrKey(), schema.Int()),
                                              (schema.StrKey(), schema.Int())])

    with self.assertRaisesRegex(
        KeyError, 'multiple symbolic fields found for argument \'a\'.'):
      schema.get_arg_fields(
          schema.get_signature(lambda a: 1), [('a', schema.Int()),
                                              ('a', schema.Int())])

    with self.assertRaisesRegex(KeyError,
                                'found extra symbolic argument \'b\'.'):
      schema.get_arg_fields(
          schema.get_signature(lambda a: 1), [('b', schema.Int())])

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*.'):
      schema.get_arg_fields(
          schema.get_signature(lambda a=1: 1), [('a', schema.Str())])

    with self.assertRaisesRegex(
        ValueError,
        'the default value .* of symbolic argument \'a\' does not equal '
        'to the default value .* specified at function signature'):
      schema.get_arg_fields(
          schema.get_signature(lambda a=1: 1), [('a', schema.Int(default=2))])

    with self.assertRaisesRegex(
        ValueError,
        'the default value .* of symbolic argument \'a\' does not equal '
        'to the default value .* specified at function signature'):
      schema.get_arg_fields(
          schema.get_signature(lambda a={}: 1),
          [('a', schema.Dict([('x', schema.Int())]))])

    with self.assertRaisesRegex(
        ValueError,
        '.*the value spec for positional wildcard argument .* must be a '
        '`pg.typing.List` instance'):
      schema.get_arg_fields(
          schema.get_signature(lambda a, *args: 1), [('args', schema.Int())])

  def testMakeFunction(self):
    """Test Signature.make_function."""

    def func1(x, y=1):
      pass

    def func2(x=1, *, y):
      pass

    def func3(x=1, *y):
      pass

    def func4(*y):
      pass

    def func5(*, x=1, y):
      pass

    def func6(x=1, *, y, **z):
      pass

    for func in [func1, func2, func3, func4, func5, func6]:
      new_func = schema.get_signature(func).make_function(['pass'])
      old_signature = inspect.signature(func)
      new_signature = inspect.signature(new_func)
      self.assertEqual(old_signature, new_signature)

  def testSchemaGetSignature(self):
    """Test Schema.get_signature."""
    def _create_schema(init_arg_list):
      return schema.Schema([
          schema.Field('x', schema.Int(), 'x'),
          schema.Field('y', schema.Int(), 'y'),
          schema.Field('z', schema.List(schema.Int()), 'z'),
          schema.Field(schema.StrKey(), schema.Str(), 'kwargs'),
      ], metadata=dict(init_arg_list=init_arg_list), allow_nonconst_keys=True)

    s1 = _create_schema(['x', 'y', 'z'])
    self.assertEqual(s1.get_signature('__main__', 'foo'), schema.Signature(
        callable_type=schema.CallableType.FUNCTION,
        module_name='__main__',
        name='foo',
        args=[
            schema.Argument('self', schema.Any()),
            schema.Argument('x', schema.Int()),
            schema.Argument('y', schema.Int()),
            schema.Argument('z', schema.List(schema.Int())),
        ],
        varkw=schema.Argument('kwargs', schema.Str())))

    s2 = _create_schema(['x', '*z'])
    self.assertEqual(
        s2.get_signature('__main__', 'foo', is_method=False),
        schema.Signature(
            callable_type=schema.CallableType.FUNCTION,
            module_name='__main__',
            name='foo',
            args=[
                schema.Argument('x', schema.Int()),
            ],
            kwonlyargs=[
                schema.Argument('y', schema.Int()),
            ],
            varargs=schema.Argument('z', schema.Int()),
            varkw=schema.Argument('kwargs', schema.Str())))

    s3 = _create_schema([])
    self.assertEqual(s3.get_signature('__main__', 'foo'), schema.Signature(
        callable_type=schema.CallableType.FUNCTION,
        module_name='__main__',
        name='foo',
        args=[
            schema.Argument('self', schema.Any()),
        ],
        kwonlyargs=[
            schema.Argument('x', schema.Int()),
            schema.Argument('y', schema.Int()),
            schema.Argument('z', schema.List(schema.Int())),
        ],
        varkw=schema.Argument('kwargs', schema.Str())))

    s4 = _create_schema(['*x'])
    with self.assertRaisesRegex(
        ValueError,
        'Variable positional argument \'x\' should have a value of '
        '`pg.typing.List` type'):
      _ = s4.get_signature('__main__', 'foo')

    s5 = schema.Schema([], metadata=dict(init_arg_list=['a']))
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'a\' is not a symbolic field.'):
      _ = s5.get_signature('__main__', 'foo')


class CallWithOptionalKeywordArgsTest(unittest.TestCase):
  """Tests for typing.CallWithOptionalKeywordArgs."""

  def testFunction(self):
    """Test call with function."""

    def foo(a, b):
      return a + b

    f = schema.CallableWithOptionalKeywordArgs(foo, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)

    def bar(a, **kwargs):
      return sum([a] + list(kwargs.values()))

    f = schema.CallableWithOptionalKeywordArgs(bar, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 6)

  def testMethod(self):
    """Test call with method."""

    class A:

      def __call__(self, a, b):
        return a + b

    f = schema.CallableWithOptionalKeywordArgs(A(), ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)

    class B:

      def __call__(self, a, **kwargs):
        return sum([a] + list(kwargs.values()))

    f = schema.CallableWithOptionalKeywordArgs(B(), ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 6)

  def testClassMethod(self):
    """Test call with class method."""

    class A:

      @classmethod
      def foo(cls, a, b):
        return a + b

    f = schema.CallableWithOptionalKeywordArgs(A.foo, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)


if __name__ == '__main__':
  unittest.main()
