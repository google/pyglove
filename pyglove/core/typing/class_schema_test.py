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
"""Tests for pyglove.core.typing.class_schema."""

import copy
import inspect
import unittest

from pyglove.core.typing import class_schema
from pyglove.core.typing import custom_typing
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import typed_missing
from pyglove.core.typing import value_specs as vs
from pyglove.core.typing.class_schema import Field
from pyglove.core.typing.class_schema import Schema


class FieldTest(unittest.TestCase):
  """Test for `Field` class."""

  def test_basics(self):
    f = Field('a', vs.Int(max_value=10), 'a field')
    self.assertIsInstance(f.key, ks.ConstStrKey)
    self.assertEqual(f.key, 'a')
    self.assertEqual(f.value, vs.Int(max_value=10))
    self.assertFalse(f.frozen)
    self.assertIs(f.annotation, int)
    self.assertEqual(f.default_value, typed_missing.MISSING_VALUE)
    self.assertEqual(f.description, 'a field')
    self.assertIsInstance(f.metadata, dict)
    self.assertEqual(len(f.metadata), 0)

    # Cover the self comparison in __eq__.
    self.assertEqual(f, f)

    # Test field with metadata.
    f = Field('a', vs.Bool(), 'a field', {'meta1': 'foo'})
    self.assertEqual(f.metadata['meta1'], 'foo')

    with self.assertRaisesRegex(ValueError, 'metadata must be a dict.'):
      Field('a', vs.Bool(), 'a field', 'abc')

  def test_extend(self):
    # Extend value spec and description.
    self.assertEqual(
        Field('a', vs.Int(1)).extend(
            Field('a', vs.Int(min_value=0), 'field a')),
        Field('a', vs.Int(min_value=0, default=1), 'field a'))

    # Overrided description will not inherit.
    self.assertEqual(
        Field('a', vs.Int(1), 'overrided field a').extend(
            Field('a', vs.Int(min_value=0), 'field a')),
        Field('a', vs.Int(min_value=0, default=1), 'overrided field a'))

    # Extend metadata.
    self.assertEqual(
        Field('a', vs.Int(), None, {
            'b': 1
        }).extend(Field('a', vs.Int(), 'field a', {'a': 2})),
        Field('a', vs.Int(), 'field a', {
            'a': 2,
            'b': 1
        }))

    # Extend with base field with different key is not allowed.
    with self.assertRaisesRegex(
        KeyError, '.* cannot extend .* for keys are different.'):
      Field('a', vs.Int()).extend(Field('b', vs.Int()))

  def test_apply(self):
    self.assertEqual(Field('a', vs.Int()).apply(1), 1)
    self.assertEqual(
        Field('a', vs.Dict([
            ('b', vs.Int()),
            ('c', vs.Bool())
        ])).apply({
            'b': 1,
            'c': True
        }), {
            'b': 1,
            'c': True
        })

    self.assertEqual(
        Field('a', vs.Dict([
            ('b', vs.Int()),
            ('c', vs.Bool())
        ])).apply({
            'b': 1,
        }, allow_partial=True), {
            'b': 1,
            'c': typed_missing.MISSING_VALUE
        })

    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'int\'> but encountered '
        '<(type|class) \'float\'>'):
      Field('a', vs.Int()).apply(1.0)

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified. \\(Path=\'c\'.*\\)'):
      Field('a', vs.Dict([
          ('b', vs.Int()),
          ('c', vs.Bool())
      ])).apply({
          'b': 1,
      }, allow_partial=False)

  def test_format(self):
    self.assertEqual(
        Field('a', vs.Dict([
            ('b', vs.Int())
        ]), 'this is a very long field.', {
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
        Field('a', vs.Dict([
            ('b', vs.Int())
        ]), 'this is a very long field.', {
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
        Field('a', vs.Dict([
            ('b', vs.Int())
        ]), 'field a').format(compact=False, verbose=False),
        'Field(key=a, value=Dict({\n'
        '    b = Int()\n'
        '  }), description=\'field a\')')


class SimpleObject:
  """Simple object for testing."""


class SchemaTest(unittest.TestCase):
  """Tests for Schema class."""

  def _create_test_schema(self, init_arg_list=None):
    return Schema([
        Field('a', vs.Int(1), 'Field a.'),
        Field('b', vs.Bool().noneable(), 'Field b.'),
        Field('c', vs.Dict([
            Field('d', vs.List(
                vs.Enum(0, [0, 1, None]), default=[0, 1]), 'Field d.'),
            Field('e', vs.List(
                vs.Dict([
                    (ks.StrKey(regex='foo.*'), vs.Str(),
                     'Mapped values.')
                ])
            ), 'Field e.'),
            Field('f', vs.Object(SimpleObject), 'Field f.')
        ]).noneable(), 'Field c.'),
    ], metadata={'init_arg_list': init_arg_list or []})

  def test_basics(self):
    s = Schema([Field('a', vs.Int())], 'schema1',
               [class_schema.create_schema([('b', vs.Bool())])])

    # Test Schema.fields.
    self.assertEqual(
        s.fields,
        {
            ks.ConstStrKey('b'): Field(ks.ConstStrKey('b'), vs.Bool()),
            ks.ConstStrKey('a'): Field(ks.ConstStrKey('a'), vs.Int()),
        })

    # Test Schema.__getitem__ and Schema.__contains__.
    self.assertEqual(s['a'], Field('a', vs.Int()))
    with self.assertRaises(KeyError):
      _ = s['foo']
    self.assertEqual(s.get('a'), Field('a', vs.Int()))
    self.assertIsNone(s.get('foo'))
    self.assertIn('a', s)
    self.assertNotIn('foo', s)

    # Test Schema.keys and Schema.values
    self.assertEqual(
        list(s.keys()), [ks.ConstStrKey('b'), ks.ConstStrKey('a')])
    self.assertEqual(
        list(s.values()),
        [Field(ks.ConstStrKey('b'), vs.Bool()),
         Field(ks.ConstStrKey('a'), vs.Int())])

    # Test Schema.allow_nonconst_keys
    self.assertFalse(s.allow_nonconst_keys)

    # Test Schema.name
    self.assertEqual(s.name, 'schema1')

    # Test Schema.set_name
    s.set_name('schema2')
    self.assertEqual(s.name, 'schema2')

    # Test equal.
    self.assertEqual(s, s)
    self.assertNotEqual(s, class_schema.create_schema([]))

    with self.assertRaisesRegex(
        TypeError, 'Argument \'fields\' must be a list.'):
      Schema({'a': vs.Int()})

    with self.assertRaisesRegex(
        ValueError, 'NonConstKey is not allowed in schema'):
      Schema([Field(ks.StrKey(), vs.Int())], allow_nonconst_keys=False)

  def test_metadata(self):
    self.assertEqual(Schema([]).metadata, {})
    self.assertEqual(
        Schema([], metadata={
            'meta1': 1
        }).metadata, {'meta1': 1})

  def test_eq(self):
    self.assertEqual(self._create_test_schema(), self._create_test_schema())

  def test_format(self):
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

  def test_extend(self):
    """Tests for Schema.extend."""

    # Disjoint inheirtance with preserved field order.
    self.assertEqual(
        list(
            class_schema.create_schema([
                ('a', vs.Int()), ('c', vs.Dict([('d', vs.Str())]))
            ]).extend(
                class_schema.create_schema([
                    ('b', vs.Bool().noneable()),
                    ('c', vs.Dict([
                        ('e', vs.Object(SimpleObject)),
                    ]))])).values()),
        list(
            class_schema.create_schema([
                # Order matters!
                ('b', vs.Bool().noneable()),
                ('c', vs.Dict([
                    ('e', vs.Object(SimpleObject)),
                    ('d', vs.Str()),
                ])),
                ('a', vs.Int()),
            ]).values()))

    # Override field with default value.
    self.assertEqual(
        list(
            class_schema.create_schema([
                ('b', True),
            ]).extend(class_schema.create_schema([
                ('b', vs.Bool().noneable()),
            ])).values()),
        list(class_schema.create_schema([
            ('b', vs.Bool(True)),
        ]).values()))

    # Extend a base schema with non-const keys while current schema does not
    # allow const keys.
    with self.assertRaisesRegex(
        ValueError,
        'Non-const key .* is not allowed to be added to the schema'):
      class_schema.create_schema(
          [('b', vs.Int())],
          allow_nonconst_keys=False
          ).extend(
              class_schema.create_schema([
                  (ks.StrKey(), vs.Int())
              ], allow_nonconst_keys=True))

    # Override field with different type.
    with self.assertRaisesRegex(
        TypeError, 'Int\\(default=1\\) cannot extend Bool\\(.*\\): '
        'incompatible type. \\(path=b\\)'):
      class_schema.create_schema([
          ('b', 1),
      ]).extend(class_schema.create_schema([
          ('b', vs.Bool().noneable()),
      ]))

    # Override metadata.
    self.assertEqual(
        class_schema.create_schema([], metadata={
            'meta2': 'bar',
            'meta3': 'new'
        }).extend(
            class_schema.create_schema([], metadata={
                'meta1': 1,
                'meta2': 'foo'
            })).metadata, {
                'meta1': 1,
                'meta2': 'bar',
                'meta3': 'new'
            })

  def test_is_compatible(self):
    with self.assertRaisesRegex(
        TypeError, 'Argument \'other\' should be a Schema object'):
      class_schema.create_schema([]).is_compatible(1)

    self.assertFalse(
        class_schema.create_schema([]).is_compatible(
            class_schema.create_schema([('a', vs.Int())])))

    self.assertFalse(
        class_schema.create_schema([
            ('a', vs.Int())
        ]).is_compatible(class_schema.create_schema([])))

    self.assertFalse(
        class_schema.create_schema([
            ('a', vs.Int())
        ]).is_compatible(class_schema.create_schema([('a', vs.Str())])))

    self.assertTrue(
        class_schema.create_schema([
            ('a', vs.Any())
        ]).is_compatible(class_schema.create_schema([('a', vs.Str())])))

  def test_get_field(self):
    s = class_schema.create_schema([
        ('a', vs.Int()),
        (ks.StrKey('foo.*'), vs.Int()),
        (ks.StrKey('f.*'), vs.Bool())
    ], allow_nonconst_keys=True)
    self.assertEqual(s.get_field('a'), Field('a', vs.Int()))
    self.assertIsNone(s.get_field('b'))
    self.assertEqual(
        s.get_field('foo1'), Field(ks.StrKey('foo.*'), vs.Int()))
    self.assertEqual(
        s.get_field('far'), Field(ks.StrKey('f.*'), vs.Bool()))

  def test_resolve(self):
    s = class_schema.create_schema([
        ('a', vs.Int()),
        ('b', vs.Int()),
        (ks.StrKey('foo.*'), vs.Int()),
    ], allow_nonconst_keys=True)
    matched, unmatched = s.resolve(['a', 'b', 'c', 'foo1', 'foo2', 'd'])
    self.assertEqual(matched, {
        'a': ['a'],
        'b': ['b'],
        ks.StrKey('foo.*'): ['foo1', 'foo2'],
    })
    self.assertEqual(unmatched, ['c', 'd'])

  def test_dynamic_field(self):
    # No dynamic field.
    s = class_schema.create_schema([
        ('b', vs.Bool())
    ], allow_nonconst_keys=True)
    self.assertIsNone(s.dynamic_field)

    # Self-declared dynamic field.
    s = class_schema.create_schema([
        ('a', vs.Int()),
        (ks.StrKey(), vs.Str())
    ], allow_nonconst_keys=True)
    self.assertEqual(s.dynamic_field, Field(ks.StrKey(), vs.Str()))

    # Inherited dynamic field.
    s = class_schema.create_schema([
        ('b', vs.Bool())
    ], base_schema_list=[s], allow_nonconst_keys=True)
    self.assertEqual(s.dynamic_field, Field(ks.StrKey(), vs.Str()))

  def test_Validate(self):
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
        TypeError, 'Expect <class \'pyglove.core.typing.class_schema_test.SimpleObject\'> but encountered '
        '<(type|class) \'int\'>: 1. \\(path=c\\.f\\)'):
      self._create_test_schema().validate(
          {'c': {
              'e': [{
                  'foo': 'bar'
              }],
              'f': 1
          }})

  def test_apply(self):
    """Tests for Schema.apply."""
    # Use default.
    self.assertEqual(
        self._create_test_schema().apply({}), {
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
                # NOTE: we can use typed_missing.MISSING_VALUE as general
                # missing value identifier.
                'e': typed_missing.MISSING_VALUE,
                # Or we can use missing value for specific value spec.
                'f': typed_missing.MissingValue(vs.Object(SimpleObject)),
            }
        })

  def test_apply_with_custom_typing(self):

    class NumberType(custom_typing.CustomTyping):

      def __init__(self, value):
        self.value = value

      def custom_apply(self, root_path, value_spec, **kwargs):
        # Pass through value to Enum.apply.
        if isinstance(value_spec, vs.Enum):
          return (True, self.value)
        if not isinstance(value_spec, vs.Number):
          raise ValueError(
              f'NumberType can only apply to numbers. (path=\'{root_path}\')')
        return (False, self)

      def __eq__(self, other):
        return isinstance(other, NumberType) and self.value == other.value

      def __ne__(self, other):
        return not self.__eq__(other)

    s = class_schema.create_schema([
        ('a', vs.Int(1)),
        ('b', vs.Bool().noneable()),
        ('c', vs.Dict([
            ('d', vs.Int()),
            ('e', vs.Enum(0, [0, 1])),
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

    with self.assertRaisesRegex(
        ValueError, 'NumberType can only apply to numbers.'):
      s.apply({
          'b': NumberType(1),
      })

  def test_apply_custom_transform(self):
    """Tests for Schema.apply with custom transform_fn."""
    transformed = []

    def _transform_fn(path, field, value):
      transformed.append((path, field.key, copy.deepcopy(value)))
      if isinstance(field.value, vs.Int):
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
                'f': typed_missing.MISSING_VALUE,
            },
        })

    self.assertEqual(transformed, [
        ('a', 'a', 1),
        ('b', 'b', False),
        ('c.d[0]', ks.ListKey(), 0),
        ('c.d', 'd', [0]),
        ('c.e[0].foo', ks.StrKey('foo.*'), 'bar'),
        ('c.e[0]', ks.ListKey(), {
            'foo': 'bar'
        }),
        ('c.e', 'e', [{
            'foo': 'bar'
        }]),
        ('c.f', 'f', typed_missing.MISSING_VALUE),
        ('c', 'c', {
            'd': [0],
            'e': [{
                'foo': 'bar'
            }],
            'f': typed_missing.MISSING_VALUE
        }),
    ])

  def test_default_values(self):
    self.assertEqual(
        self._create_test_schema().apply({}, allow_partial=False),
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
                'e': typed_missing.MISSING_VALUE,
                'f': typed_missing.MISSING_VALUE,
            }
        })


class CreateSchemaTest(unittest.TestCase):
  """Tests for class_schema.create_schema."""

  def test_basics(self):
    s = class_schema.create_schema([
        ('a', 1),
        ('b', 'foo', 'field b'),
        ('c', True, 'field c', {'user_data': 1}),
        ('d', 1.0), ('e', vs.Enum(0, [0, 1]))
    ], 'schema1', metadata={'user_data': 2})

    self.assertEqual(s.name, 'schema1')
    self.assertEqual(s['a'], Field('a', vs.Int(1)))
    self.assertEqual(s['b'],
                     Field('b', vs.Str('foo'), 'field b'))
    self.assertEqual(
        s['c'],
        Field('c', vs.Bool(True), 'field c', {'user_data': 1}))
    self.assertEqual(s['d'], Field('d', vs.Float(1.0)))
    self.assertEqual(s['e'], Field('e', vs.Enum(0, [0, 1])))
    self.assertEqual(s.metadata, {'user_data': 2})

  def test_bad_cases(self):
    with self.assertRaisesRegex(
        TypeError, 'Metadata of schema should be a dict.'):
      class_schema.create_schema([], metadata=1)

    with self.assertRaisesRegex(
        TypeError, 'Field definition should be tuples with 2 to 4 elements.'):
      class_schema.create_schema(['a'])

    with self.assertRaisesRegex(
        TypeError, 'Field definition should be tuples with 2 to 4 elements.'):
      class_schema.create_schema([('a',)])


if __name__ == '__main__':
  unittest.main()
