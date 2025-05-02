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


class JsonSchemaTest(unittest.TestCase):

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

  def test_int(self):
    self.assert_json_schema(int, {
        'type': 'integer',
    })
    self.assert_json_schema(vs.Int(min_value=0), {
        'type': 'integer',
        'minimum': 0,
    })
    self.assert_json_schema(vs.Int(max_value=1), {
        'type': 'integer',
        'maximum': 1,
    })

  def test_float(self):
    self.assert_json_schema(float, {
        'type': 'number',
    })
    self.assert_json_schema(vs.Float(min_value=0.0), {
        'type': 'number',
        'minimum': 0.0,
    })
    self.assert_json_schema(vs.Float(max_value=1.0), {
        'type': 'number',
        'maximum': 1.0,
    })

  def test_str(self):
    self.assert_json_schema(str, {
        'type': 'string',
    })
    self.assert_json_schema(vs.Str(regex='a.*'), {
        'type': 'string',
        'pattern': 'a.*',
    })

  def test_enum(self):
    self.assert_json_schema(Literal['a', 1], {
        'enum': ['a', 1]
    })
    self.assert_json_schema(Literal['a', 1, None], {
        'anyOf': [
            {'enum': ['a', 1]},
            {'type': 'null'}
        ]
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
    }, include_type_name=True)

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

  def test_pg_object_nessted(self):

    class A(pg_object.Object):
      x: Annotated[int, 'field x']
      y: str

    class B(pg_object.Object):
      z: A

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
                '$ref': '#/$defs/A'
            },
        },
        'required': ['_type', 'z'],
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
        'required': ['_type', 'z'],
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

if __name__ == '__main__':
  unittest.main()
