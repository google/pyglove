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
import copy
import inspect
import sys
from typing import Optional, Union, List
import unittest

from pyglove.core import utils
from pyglove.core.typing import annotation_conversion  # pylint: disable=unused-import
from pyglove.core.typing import class_schema
from pyglove.core.typing import custom_typing
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import typed_missing
from pyglove.core.typing import value_specs as vs
from pyglove.core.typing.class_schema import Field
from pyglove.core.typing.class_schema import Schema


class ForwardRefTest(unittest.TestCase):
  """Test for `ForwardRef` class."""

  class A:
    pass

  def setUp(self):
    super().setUp()
    self._module = sys.modules[__name__]

  def test_basics(self):
    r = class_schema.ForwardRef(self._module, 'ForwardRefTest.A')
    self.assertIs(r.module, self._module)
    self.assertEqual(r.name, 'A')
    self.assertEqual(r.qualname, 'ForwardRefTest.A')
    self.assertEqual(r.type_id, f'{self._module.__name__}.ForwardRefTest.A')

  def test_resolved(self):
    self.assertTrue(
        class_schema.ForwardRef(self._module, 'ForwardRefTest.A').resolved
    )
    self.assertFalse(class_schema.ForwardRef(self._module, 'Foo').resolved)

  def test_as_annotation(self):
    self.assertEqual(
        class_schema.ForwardRef(
            self._module, 'ForwardRefTest.A').as_annotation(),
        ForwardRefTest.A,
    )
    self.assertEqual(
        class_schema.ForwardRef(self._module, 'Foo').as_annotation(), 'Foo'
    )

  def test_cls(self):
    self.assertIs(
        class_schema.ForwardRef(self._module, 'ForwardRefTest.A').cls,
        ForwardRefTest.A
    )

    with self.assertRaisesRegex(TypeError, '.* does not exist in module'):
      _ = class_schema.ForwardRef(self._module, 'Foo').cls

    with self.assertRaisesRegex(TypeError, '.* is not a class'):
      _ = class_schema.ForwardRef(self._module, 'unittest').cls

  def test_repr(self):
    self.assertEqual(
        repr(class_schema.ForwardRef(self._module, 'FieldTest')),
        f'ForwardRef(module=\'{self._module.__name__}\', name=\'FieldTest\')',
    )

  def test_eq_ne(self):
    ref = class_schema.ForwardRef(self._module, 'FieldTest')
    self.assertEqual(ref, ref)
    self.assertEqual(ref, class_schema.ForwardRef(self._module, 'FieldTest'))
    self.assertEqual(ref, FieldTest)
    self.assertEqual(FieldTest, ref)

    self.assertNotEqual(int, ref)
    self.assertNotEqual(ref, class_schema.ForwardRef(unittest, 'FieldTest'))
    self.assertNotEqual(ref, class_schema.ForwardRef(self._module, 'Foo'))

  def test_copy(self):
    self.assertEqual(
        copy.copy(class_schema.ForwardRef(self._module, 'FieldTest')),
        class_schema.ForwardRef(self._module, 'FieldTest'),
    )
    self.assertEqual(
        copy.deepcopy(class_schema.ForwardRef(self._module, 'FieldTest')),
        class_schema.ForwardRef(self._module, 'FieldTest'),
    )


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
    self.assertIsNone(f.origin)

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

  def test_repr(self):
    self.assertEqual(
        repr(
            Field(
                'a', vs.Dict([('b', vs.Int())]), 'this is a very long field.',
                {'m1': 1, 'm2': 2, 'm3': 3, 'm4': 4, 'm5': 5}
            )
        ),
        (
            'Field(key=a, value=Dict(fields=[Field(key=b, '
            'value=Int())]), description=\'this is a very long field.\', '
            'metadata={\'m1\': 1, \'m2\': 2, \'m3\': 3, \'m4\': 4, \'m5\': 5})'
        )
    )

  def test_json_conversion(self):
    def assert_json_conversion(f):
      self.assertEqual(utils.from_json(f.to_json()), f)

    assert_json_conversion(Field('a', vs.Int()))
    assert_json_conversion(Field('a', vs.Int(), 'description'))
    assert_json_conversion(Field('a', vs.Int(), 'description', {'a': 1}))


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

  def test_create_schema_dict_input(self):
    self.assertEqual(
        class_schema.create_schema({
            'a': (1, 'Field a.'),
            'b': (vs.Bool() | None, 'Field b.'),
            'c': (vs.Dict[{
                'd': (vs.List[
                    vs.Enum[0, 1, None].set_default(0)].set_default([0, 1]),
                      'Field d.'),
                'e': (vs.List[vs.Dict({
                    ks.StrKey(regex='foo.*'): (vs.Str(), 'Mapped values.')
                })], 'Field e.'),
                'f': (SimpleObject, 'Field f.')
            }] | None, 'Field c.')
        }, metadata={'init_arg_list': []}),
        self._create_test_schema())

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

    # Test Schema.description.
    self.assertIsNone(s.description)

    # Test Schema.set_description.
    s.set_description('schema1')
    self.assertEqual(s.description, 'schema1')

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

  def test_repr(self):
    """Tests for Schema.format."""
    self.assertEqual(
        repr(self._create_test_schema()),
        (
            "Schema(fields=[Field(key=a, value=Int(default=1), description="
            "'Field a.'), Field(key=b, value=Bool(default=None, noneable="
            "True), description='Field b.'), Field(key=c, value=Dict("
            "fields=[Field(key=d, value=List(Enum(default=0, values=[0, 1, "
            "None]), default=[0, 1]), description='Field d.'), Field(key=e, "
            "value=List(Dict(fields=[Field(key=StrKey(regex='foo.*'), "
            "value=Str(), description='Mapped values.')])), description='Field"
            " e.'), Field(key=f, value=Object(SimpleObject), description="
            "'Field f.')], noneable=True), description='Field c.')], "
            "allow_nonconst_keys=False, metadata={'init_arg_list': []})"
        )
    )

  def test_str(self):
    self.maxDiff = None
    self.assertEqual(
        str(self._create_test_schema()),
        inspect.cleandoc("""
        Schema(
          fields=[
            Field(
              key=a,
              value=Int(
                default=1
              ),
              description='Field a.'
            ),
            Field(
              key=b,
              value=Bool(
                default=None,
                noneable=True
              ),
              description='Field b.'
            ),
            Field(
              key=c,
              value=Dict(
                fields=[
                  Field(
                    key=d,
                    value=List(
                      Enum(
                        default=0,
                        values=[0, 1, None]
                      ),
                      default=[0, 1]
                    ),
                    description='Field d.'
                  ),
                  Field(
                    key=e,
                    value=List(
                      Dict(
                        fields=[
                          Field(
                            key=StrKey(regex='foo.*'),
                            value=Str(),
                            description='Mapped values.'
                          )
                        ]
                      )
                    ),
                    description='Field e.'
                  ),
                  Field(
                    key=f,
                    value=Object(
                      SimpleObject
                    ),
                    description='Field f.'
                  )
                ],
                noneable=True
              ),
              description='Field c.'
            )
          ],
          allow_nonconst_keys=False,
          metadata={
            'init_arg_list': []
          }
        )""")
    )

  def test_merge(self):
    """Tests for Schema.merge."""
    self.assertEqual(
        Schema.merge([
            class_schema.create_schema([
                ('a', vs.Int()),
                ('b', vs.Bool().noneable()),
            ]),
            class_schema.create_schema([
                ('a', vs.Str()),
                ('c', vs.Float()),
            ]),
        ]),
        class_schema.create_schema([
            ('a', vs.Int()),
            ('b', vs.Bool().noneable()),
            ('c', vs.Float()),
        ]),
    )
    self.assertEqual(
        Schema.merge([
            class_schema.create_schema([
                ('a', vs.Int()),
                (ks.StrKey(), vs.Str().noneable()),
            ], allow_nonconst_keys=True),
            class_schema.create_schema([
                ('c', vs.Float()),
            ]),
        ]),
        class_schema.create_schema([
            ('a', vs.Int()),
            ('c', vs.Float()),
            (ks.StrKey(), vs.Str().noneable()),
        ], allow_nonconst_keys=True),
    )

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

  def test_validate(self):
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

  def test_json_conversion(self):
    schema = self._create_test_schema()
    schema.set_description('Foo')
    schema.set_name('Bar')
    schema_copy = utils.from_json(schema.to_json())

    # This compares fields only
    self.assertEqual(schema_copy, schema)

    self.assertEqual(schema_copy.name, schema.name)
    self.assertEqual(schema_copy.description, schema.description)
    self.assertEqual(schema.metadata, schema_copy.metadata)
    self.assertEqual(
        schema.allow_nonconst_keys, schema_copy.allow_nonconst_keys
    )


class CreateSchemaTest(unittest.TestCase):
  """Tests for class_schema.create_schema."""

  def test_basics(self):
    s = class_schema.create_schema(
        [
            ('a', 1),
            ('b', 'foo', 'field b'),
            ('c', True, 'field c', {'user_data': 1}),
            ('d', 1.0),
            ('e', vs.Enum(0, [0, 1])),
            ('f', int),
            ('f1', Optional[int]),
            ('g', float),
            ('g1', Optional[float]),
            ('h', bool),
            ('h1', Optional[bool]),
            ('i', str),
            ('i1', Optional[str]),
            ('j', Union[int, float, bool]),
            ('k', list),
            ('l', list[int]),
            ('L', List[int]),
            ('l1', Optional[list[int]]),
            ('l2', ['black', 'white']),
            ('m', {'brand': 'google'}),
            ('n', None),
        ],
        'schema1',
        metadata={'user_data': 2},
    )

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
    self.assertEqual(s['f'], Field('f', vs.Int()))
    self.assertEqual(
        s['f1'], Field('f1', vs.Int().noneable(use_none_as_default=False))
    )
    self.assertEqual(s['g'], Field('g', vs.Float()))
    self.assertEqual(
        s['g1'], Field('g1', vs.Float().noneable(use_none_as_default=False))
    )
    self.assertEqual(s['h'], Field('h', vs.Bool()))
    self.assertEqual(
        s['h1'], Field('h1', vs.Bool().noneable(use_none_as_default=False))
    )
    self.assertEqual(s['i'], Field('i', vs.Str()))
    self.assertEqual(
        s['i1'], Field('i1', vs.Str().noneable(use_none_as_default=False))
    )
    self.assertEqual(
        s['j'], Field('j', vs.Union([vs.Int(), vs.Float(), vs.Bool()]))
    )
    self.assertEqual(s['k'], Field('k', vs.List(vs.Any())))
    self.assertEqual(s['l'], Field('l', vs.List(vs.Int())))
    self.assertEqual(s['L'], Field('L', vs.List(vs.Int())))
    self.assertEqual(
        s['l1'],
        Field('l1', vs.List(vs.Int()).noneable(use_none_as_default=False))
    )
    self.assertEqual(
        s['l2'],
        Field('l2', vs.List(vs.Str()).set_default(['black', 'white'])),
    )
    self.assertEqual(
        s['m'], Field('m', vs.Dict().set_default({'brand': 'google'}))
    )
    self.assertEqual(s['n'], Field('n', vs.Any().noneable()))

  def test_bad_cases(self):
    with self.assertRaisesRegex(
        TypeError, 'Metadata of schema should be a dict.'
    ):
      class_schema.create_schema([], metadata=1)

    with self.assertRaisesRegex(
        TypeError, 'Schema definition should be a dict.*a list.'
    ):
      class_schema.create_schema(1, metadata=1)

    with self.assertRaisesRegex(
        TypeError, 'Field definition should be tuples with 2 to 4 elements.'
    ):
      class_schema.create_schema(['a'])

    with self.assertRaisesRegex(
        TypeError, 'Field definition should be tuples with 2 to 4 elements.'
    ):
      class_schema.create_schema([('a',)])


if __name__ == '__main__':
  unittest.main()
