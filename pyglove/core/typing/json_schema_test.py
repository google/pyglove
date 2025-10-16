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
"""Tests for JSON schema conversion."""

from typing import Annotated, Literal, Optional
import unittest
from pyglove.core.symbolic import object as pg_object
from pyglove.core.typing import annotation_conversion  # pylint: disable=unused-import
from pyglove.core.typing import json_schema
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


class Foo(pg_object.Object):
  x: int
  y: 'Bar'


class Bar(pg_object.Object):
  z: Optional[Foo]


class ToJsonSchemaTest(unittest.TestCase):

  maxDiff = None

  def assert_json_schema(
      self, t, expected_json_schema,
      *,
      include_type_name: bool = True,
      include_subclasses: bool = False,
      inline_nested_refs: bool = False):
    actual_schema = json_schema.to_json_schema(
        t,
        include_type_name=include_type_name,
        include_subclasses=include_subclasses,
        inline_nested_refs=inline_nested_refs,
    )
    self.assertEqual(actual_schema, expected_json_schema)

  def test_bool(self):
    self.assert_json_schema(bool, {
        'type': 'boolean',
    })
    self.assert_json_schema(vs.Bool(default=True), {
        'type': 'boolean',
        'default': True,
    })

  def test_int(self):
    self.assert_json_schema(int, {
        'type': 'integer',
    })
    self.assert_json_schema(vs.Int(min_value=0), {
        'type': 'integer',
        'minimum': 0,
    })
    self.assert_json_schema(vs.Int(max_value=1, default=0), {
        'type': 'integer',
        'maximum': 1,
        'default': 0,
    })

  def test_float(self):
    self.assert_json_schema(float, {
        'type': 'number',
    })
    self.assert_json_schema(vs.Float(min_value=0.0), {
        'type': 'number',
        'minimum': 0.0,
    })
    self.assert_json_schema(vs.Float(max_value=1.0, default=0.0), {
        'type': 'number',
        'maximum': 1.0,
        'default': 0.0,
    })

  def test_str(self):
    self.assert_json_schema(str, {
        'type': 'string',
    })
    self.assert_json_schema(vs.Str(regex='a.*', default='a1'), {
        'type': 'string',
        'pattern': 'a.*',
        'default': 'a1',
    })

  def test_enum(self):
    self.assert_json_schema(Literal['a', 1], {
        'enum': ['a', 1]
    })
    self.assert_json_schema(vs.Enum(1, ['a', 1]), {
        'enum': ['a', 1],
        'default': 1,
    })
    self.assert_json_schema(Literal['a', 1, None], {
        'anyOf': [
            {'enum': ['a', 1]},
            {'type': 'null'}
        ]
    })
    self.assert_json_schema(vs.Enum(None, ['a', 1, None]), {
        'anyOf': [
            {'enum': ['a', 1]},
            {'type': 'null'}
        ],
        'default': None
    })
    with self.assertRaisesRegex(
        ValueError, 'Enum candidate .* is not supported'
    ):
      json_schema.to_json_schema(Literal['a', 1, None, ValueError()])

  def test_list(self):
    self.assert_json_schema(vs.List[int], {
        'type': 'array',
        'items': {
            'type': 'integer',
        }
    })
    self.assert_json_schema(vs.List(int, default=[1, 2]), {
        'type': 'array',
        'items': {
            'type': 'integer',
        },
        'default': [1, 2],
    })

  def test_dict(self):
    self.assert_json_schema(vs.Dict(), {
        'type': 'object', 'additionalProperties': True
    })
    self.assert_json_schema(vs.Dict({'a': int}), {
        'type': 'object',
        'properties': {'a': {
            'type': 'integer',
        }},
        'required': ['a'],
        'additionalProperties': False,
    })
    self.assert_json_schema(vs.Dict([(ks.StrKey(), int)]), {
        'type': 'object',
        'additionalProperties': {'type': 'integer'},
    })
    self.assert_json_schema(vs.Dict([(ks.StrKey(), vs.Any())]), {
        'type': 'object',
        'additionalProperties': True,
    })

  def test_union(self):
    self.assert_json_schema(vs.Union([int, vs.Union([str, int]).noneable()]), {
        'anyOf': [
            {'type': 'integer'},
            {'type': 'string'},
            # TODO(daiyip): Remove duplicates for nested Union in future.
            {'type': 'integer'},
            {'type': 'null'},
        ]
    })
    self.assert_json_schema(vs.Union([int, vs.Union([str, int]).noneable()]), {
        'anyOf': [
            {'type': 'integer'},
            {'type': 'string'},
            # TODO(daiyip): Remove duplicates for nested Union in future.
            {'type': 'integer'},
            {'type': 'null'},
        ]
    })

  def test_any(self):
    self.assert_json_schema(vs.Any(), {
        'anyOf': [
            {'type': 'boolean'},
            {'type': 'number'},
            {'type': 'string'},
            {'type': 'array'},
            {'type': 'object', 'additionalProperties': True},
            {'type': 'null'},
        ]
    })
    self.assert_json_schema(vs.Any(default=1), {
        'anyOf': [
            {'type': 'boolean'},
            {'type': 'number'},
            {'type': 'string'},
            {'type': 'array'},
            {'type': 'object', 'additionalProperties': True},
            {'type': 'null'},
        ],
        'default': 1,
    })

  def test_object(self):
    class A:
      def __init__(self, x: int, y: str):
        pass

    self.assert_json_schema(vs.Object(A), {
        'type': 'object',
        'properties': {
            'x': {
                'type': 'integer',
            },
            'y': {
                'type': 'string',
            },
        },
        'required': ['x', 'y'],
        'title': 'A',
        'additionalProperties': False,
    }, include_type_name=False)

    class B(pg_object.Object):
      x: int
      y: str

    self.assert_json_schema(vs.Object(B), {
        'type': 'object',
        'properties': {
            '_type': {
                'const': B.__type_name__,
            },
            'x': {
                'type': 'integer',
            },
            'y': {
                'type': 'string',
            },
        },
        'required': ['_type', 'x', 'y'],
        'title': 'B',
        'additionalProperties': False,
    }, include_type_name=True)

    self.assert_json_schema(vs.Object(B, default=B(x=1, y='a')), {
        'type': 'object',
        'properties': {
            'x': {
                'type': 'integer',
            },
            'y': {
                'type': 'string',
            },
        },
        'required': ['x', 'y'],
        'title': 'B',
        'additionalProperties': False,
        'default': {
            'x': 1,
            'y': 'a',
        },
    }, include_type_name=False)

  def test_pg_object(self):

    class A(pg_object.Object):
      x: int
      y: str

    self.assert_json_schema(vs.Object(A), {
        'type': 'object',
        'properties': {
            'x': {
                'type': 'integer',
            },
            'y': {
                'type': 'string',
            },
        },
        'required': ['x', 'y'],
        'title': 'A',
        'additionalProperties': False,
    }, include_type_name=False)

    self.assert_json_schema(vs.Object(A), {
        'type': 'object',
        'properties': {
            '_type': {
                'const': A.__type_name__,
            },
            'x': {
                'type': 'integer',
            },
            'y': {
                'type': 'string',
            },
        },
        'required': ['_type', 'x', 'y'],
        'title': 'A',
        'additionalProperties': False,
    }, include_type_name=True)

  def test_pg_object_nested(self):

    class A(pg_object.Object):
      x: Annotated[int, 'field x']
      y: str

    class B(pg_object.Object):
      z: A = A(x=1, y='a')

    self.assert_json_schema(vs.Object(B), {
        '$defs': {
            'A': {
                'type': 'object',
                'properties': {
                    '_type': {
                        'const': A.__type_name__,
                    },
                    'x': {
                        'type': 'integer',
                        'description': 'field x',
                    },
                    'y': {
                        'type': 'string',
                    },
                },
                'required': ['_type', 'x', 'y'],
                'title': 'A',
                'additionalProperties': False,
            }
        },
        'type': 'object',
        'properties': {
            '_type': {
                'const': B.__type_name__,
            },
            'z': {
                '$ref': '#/$defs/A',
                'default': {
                    '_type': A.__type_name__,
                    'x': 1,
                    'y': 'a',
                },
            },
        },
        'required': ['_type'],
        'title': 'B',
        'additionalProperties': False,
    }, include_type_name=True)

    self.assert_json_schema(vs.Object(B), {
        'type': 'object',
        'properties': {
            '_type': {
                'const': B.__type_name__,
            },
            'z': {
                'type': 'object',
                'properties': {
                    '_type': {
                        'const': A.__type_name__,
                    },
                    'x': {
                        'type': 'integer',
                        'description': 'field x',
                    },
                    'y': {
                        'type': 'string',
                    },
                },
                'required': ['_type', 'x', 'y'],
                'title': 'A',
                'additionalProperties': False,
            },
        },
        'required': ['_type'],
        'title': 'B',
        'additionalProperties': False,
    }, include_type_name=True, inline_nested_refs=True)

  def test_pg_object_with_subclasses(self):
    class A(pg_object.Object):
      x: int
      y: str

    class B(A):
      z: int

    self.assert_json_schema(
        vs.Object(A).noneable(),
        {
            'anyOf': [
                {
                    'additionalProperties': False,
                    'properties': {
                        '_type': {
                            'const': A.__type_name__,
                        },
                        'x': {
                            'type': 'integer',
                        },
                        'y': {
                            'type': 'string',
                        },
                    },
                    'required': ['_type', 'x', 'y'],
                    'title': 'A',
                    'type': 'object',
                },
                {
                    'additionalProperties': False,
                    'properties': {
                        '_type': {
                            'const': B.__type_name__,
                        },
                        'x': {
                            'type': 'integer',
                        },
                        'y': {
                            'type': 'string',
                        },
                        'z': {
                            'type': 'integer',
                        },
                    },
                    'required': ['_type', 'x', 'y', 'z'],
                    'title': 'B',
                    'type': 'object',
                },
                {
                    'type': 'null',
                }
            ],
            'default': None,
        },
        include_type_name=True,
        include_subclasses=True,
    )

  def test_pg_object_with_recursive_refs(self):
    self.assert_json_schema(
        vs.Object(Foo),
        {
            '$defs': {
                'Foo': {
                    'additionalProperties': False,
                    'properties': {
                        '_type': {
                            'const': Foo.__type_name__,
                        },
                        'x': {
                            'type': 'integer',
                        },
                        'y': {
                            '$ref': '#/$defs/Bar',
                        },
                    },
                    'required': ['_type', 'x', 'y'],
                    'title': 'Foo',
                    'type': 'object',
                },
                'Bar': {
                    'additionalProperties': False,
                    'properties': {
                        '_type': {
                            'const': Bar.__type_name__,
                        },
                        'z': {
                            'anyOf': [
                                {'$ref': '#/$defs/Foo'},
                                {'type': 'null'},
                            ]
                        },
                    },
                    'required': ['_type', 'z'],
                    'title': 'Bar',
                    'type': 'object',
                }
            },
            '$ref': '#/$defs/Foo',
        }
    )

  def test_unsupported_value_spec(self):
    with self.assertRaisesRegex(
        TypeError, 'Value spec .* cannot be converted to JSON schema'
    ):
      json_schema.to_json_schema(vs.Callable())

  def test_schema(self):
    class A(pg_object.Object):
      x: int
      y: str

    self.assert_json_schema(
        A.__schema__,
        {
            'type': 'object',
            'properties': {
                '_type': {
                    'const': A.__type_name__,
                },
                'x': {
                    'type': 'integer',
                },
                'y': {
                    'type': 'string',
                },
            },
            'required': ['_type', 'x', 'y'],
            'title': 'A',
            'additionalProperties': False,
        },
        include_type_name=True,
    )

  def test_value_spec_to_json_schema(self):
    self.assertEqual(
        vs.Int().to_json_schema(),
        {
            'type': 'integer',
        }
    )

  def test_schema_to_json_schema(self):
    class A(pg_object.Object):
      x: int
      y: str

    self.assertEqual(
        A.__schema__.to_json_schema(),
        {
            'type': 'object',
            'properties': {
                '_type': {
                    'const': A.__type_name__,
                },
                'x': {
                    'type': 'integer',
                },
                'y': {
                    'type': 'string',
                },
            },
            'required': ['_type', 'x', 'y'],
            'title': 'A',
            'additionalProperties': False,
        }
    )


class FromJsonSchemaTest(unittest.TestCase):

  def assert_value_spec(self, input_json_schema, expected_value_spec):
    value_spec = vs.ValueSpec.from_json_schema(input_json_schema)
    self.assertEqual(value_spec, expected_value_spec)

  def test_bool(self):
    self.assert_value_spec(
        {
            'type': 'boolean',
        },
        vs.Bool(),
    )
    self.assert_value_spec(
        {
            'type': 'boolean',
            'default': True
        },
        vs.Bool(default=True),
    )

  def test_int(self):
    self.assert_value_spec(
        {
            'type': 'integer',
        },
        vs.Int(),
    )
    self.assert_value_spec(
        {
            'type': 'integer',
            'minimum': 0,
        },
        vs.Int(min_value=0),
    )
    self.assert_value_spec(
        {
            'type': 'integer',
            'maximum': 1,
            'default': 0,
        },
        vs.Int(max_value=1, default=0),
    )

  def test_number(self):
    self.assert_value_spec(
        {
            'type': 'number',
        },
        vs.Float(),
    )
    self.assert_value_spec(
        {
            'type': 'number',
            'minimum': 0.0,
        },
        vs.Float(min_value=0.0),
    )
    self.assert_value_spec(
        {
            'type': 'number',
            'maximum': 1.0,
            'default': 0.0,
        },
        vs.Float(max_value=1.0, default=0.0),
    )

  def test_str(self):
    self.assert_value_spec(
        {
            'type': 'string',
        },
        vs.Str(),
    )
    self.assert_value_spec(
        {
            'type': 'string',
            'pattern': 'a.*',
            'default': 'a1',
        },
        vs.Str(regex='a.*', default='a1'),
    )

  def test_enum(self):
    self.assert_value_spec(
        {
            'enum': ['a', 'b', 'c'],
            'default': 'b',
        },
        vs.Enum('b', ['a', 'b', 'c']),
    )
    with self.assertRaisesRegex(
        ValueError, 'Enum candidate .* is not supported'
    ):
      vs.ValueSpec.from_json_schema({'enum': [{'x': 1}, {'y': 'abc'}]})

  def test_null(self):
    self.assert_value_spec(
        {
            'type': 'null',
        },
        vs.Any().freeze(None),
    )

  def test_any_of(self):
    self.assert_value_spec(
        {
            'anyOf': [
                {'type': 'integer'},
            ],
        },
        vs.Int(),
    )
    self.assert_value_spec(
        {
            'anyOf': [
                {'type': 'integer'},
                {'type': 'string'},
            ],
        },
        vs.Union([vs.Int(), vs.Str()]),
    )
    self.assert_value_spec(
        {
            'anyOf': [
                {'type': 'integer'},
                {'type': 'string'},
                {'type': 'null'},
            ],
        },
        vs.Union([vs.Int(), vs.Str()]).noneable(),
    )

  def test_list(self):
    self.assert_value_spec(
        {
            'type': 'array',
        },
        vs.List(vs.Any()),
    )
    self.assert_value_spec(
        {
            'type': 'array',
            'items': {
                'type': 'integer',
            },
        },
        vs.List(vs.Int()),
    )
    self.assert_value_spec(
        {
            'type': 'array',
            'items': {
                'type': 'integer',
            },
            'default': [1, 2],
        },
        vs.List(vs.Int(), default=[1, 2]),
    )

  def test_dict(self):
    self.assert_value_spec(
        {
            'type': 'object',
        },
        vs.Dict(),
    )
    self.assert_value_spec(
        {
            'type': 'object',
            'properties': {
                'a': {
                    'type': 'integer',
                },
            },
            'required': ['a'],
            'additionalProperties': False,
        },
        vs.Dict({'a': vs.Int()}),
    )
    self.assert_value_spec(
        {
            'type': 'object',
            'additionalProperties': {'type': 'integer'},
        },
        vs.Dict([(ks.StrKey(), vs.Int())]),
    )
    self.assert_value_spec(
        {
            'type': 'object',
            'additionalProperties': True,
        },
        vs.Dict([(ks.StrKey(), vs.Any())]),
    )

  def _cls_value_spec(self, input_json_schema):
    def schema_to_class(name, schema):
      class _Class(pg_object.Object):
        pass
      cls = _Class
      cls.__name__ = name
      cls.__doc__ = schema.description
      cls.apply_schema(schema)
      return cls
    return vs.ValueSpec.from_json_schema(
        input_json_schema, class_fn=schema_to_class
    )

  def test_simple_object(self):
    cls_spec = self._cls_value_spec(
        {
            'type': 'object',
            'title': 'A',
            'description': 'Class A',
            'properties': {
                'x': {
                    'type': 'integer',
                    'description': 'field x',
                },
                'y': {
                    'type': 'string',
                },
            },
            'required': ['x'],
            'additionalProperties': False,
        },
    )
    self.assertIsNone(cls_spec.cls(x=1).y)
    self.assertEqual(cls_spec.cls.__name__, 'A')
    self.assertEqual(cls_spec.cls.__doc__, 'Class A')
    self.assertEqual(
        cls_spec.cls.__schema__['x'], vs.Field('x', vs.Int(), 'field x')
    )
    self.assertEqual(
        cls_spec.cls.__schema__['y'], vs.Field('y', vs.Str().noneable())
    )

  def test_nested_object(self):
    cls_spec = self._cls_value_spec(
        {
            'type': 'object',
            'title': 'A',
            'description': 'Class A',
            'properties': {
                'x': {
                    'type': 'integer',
                    'description': 'field x',
                },
                'y': {
                    'type': 'object',
                    'title': 'B',
                    'description': 'Class B',
                    'properties': {
                        'z': {
                            'type': 'string',
                        },
                    },
                    'required': ['z'],
                    'additionalProperties': False,
                },
            },
            'required': ['x'],
            'additionalProperties': False,
        },
    )
    self.assertIsNone(cls_spec.cls(x=1).y)
    self.assertEqual(cls_spec.cls.__name__, 'A')
    self.assertEqual(cls_spec.cls.__doc__, 'Class A')
    self.assertEqual(
        cls_spec.cls.__schema__['x'], vs.Field('x', vs.Int(), 'field x')
    )
    b_cls = cls_spec.cls.__schema__['y'].value.cls
    self.assertEqual(b_cls.__schema__['z'], vs.Field('z', vs.Str()))

  def test_simple_object_with_def(self):
    cls_spec = self._cls_value_spec(
        {
            '$defs': {
                'A': {
                    'type': 'object',
                    'title': 'A',
                    'description': 'Class A',
                    'properties': {
                        'x': {
                            'type': 'integer',
                            'description': 'field x',
                            'default': 1,
                        },
                        'y': {
                            'type': 'string',
                        },
                    },
                    'required': ['x'],
                    'additionalProperties': False,
                }
            },
            '$ref': '#/$defs/A',
        }
    )
    self.assertEqual(cls_spec.cls(y='a').x, 1)
    self.assertEqual(cls_spec.cls.__name__, 'A')
    self.assertEqual(cls_spec.cls.__doc__, 'Class A')

  def test_complex_object_with_def(self):
    cls_spec = self._cls_value_spec(
        {
            '$defs': {
                'B': {
                    'type': 'object',
                    'title': 'B',
                    'description': 'Class B',
                    'properties': {
                        'z': {
                            'type': 'string',
                        },
                    },
                    'required': ['z'],
                    'additionalProperties': False,
                },
                'A': {
                    'type': 'object',
                    'title': 'A',
                    'description': 'Class A',
                    'properties': {
                        'x': {
                            'type': 'integer',
                            'description': 'field x',
                            'default': 1,
                        },
                        'y': {
                            '$ref': '#/$defs/B',
                        }
                    },
                    'required': ['x'],
                    'additionalProperties': False,
                },
            },
            '$ref': '#/$defs/A',
        }
    )
    self.assertIsNone(cls_spec.cls(x=1).y)
    self.assertEqual(cls_spec.cls.__name__, 'A')
    self.assertEqual(cls_spec.cls.__doc__, 'Class A')
    self.assertEqual(
        cls_spec.cls.__schema__['x'],
        vs.Field('x', vs.Int(default=1), 'field x')
    )
    b_cls = cls_spec.cls.__schema__['y'].value.cls
    self.assertEqual(b_cls.__name__, 'B')
    self.assertEqual(b_cls.__doc__, 'Class B')
    self.assertEqual(b_cls.__schema__['z'], vs.Field('z', vs.Str()))

    with self.assertRaisesRegex(
        ValueError, 'Reference .* not defined'
    ):
      self._cls_value_spec(
          {
              '$defs': {
                  'A': {
                      'type': 'object',
                      'title': 'A',
                      'description': 'Class A',
                      'properties': {
                          'x': {
                              '$ref': '#/$defs/B',
                          }
                      },
                      'required': ['x'],
                      'additionalProperties': False,
                  },
                  # B should go before A.
                  'B': {
                      'type': 'object',
                      'title': 'B',
                      'description': 'Class B',
                      'properties': {
                          'z': {
                              'type': 'string',
                          },
                      },
                      'required': ['z'],
                      'additionalProperties': False,
                  },
              },
              '$ref': '#/$defs/A',
          }
      )

  def test_unsupported_json_schema(self):
    with self.assertRaisesRegex(
        ValueError, 'Unsupported type .* in JSON schema'
    ):
      vs.ValueSpec.from_json_schema({'type': 'oneOf'})

  def test_schema_from_json_schema(self):
    schema = vs.Schema.from_json_schema(
        {
            'type': 'object',
            'title': 'A',
            'description': 'Class A',
            'properties': {
                'x': {
                    'type': 'integer',
                },
            },
            'required': ['x'],
            'additionalProperties': False,
        },
    )
    self.assertEqual(schema.description, 'Class A')
    self.assertEqual(list(schema.fields.keys()), ['x'])
    self.assertEqual(schema.fields['x'].value, vs.Int())

    with self.assertRaisesRegex(
        ValueError, 'JSON schema is not an object type'
    ):
      vs.Schema.from_json_schema({'type': 'integer'})

if __name__ == '__main__':
  unittest.main()
