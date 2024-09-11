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
"""Tests for pyglove.boilerplate_class."""

import unittest

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.base import from_json_str as pg_from_json_str
from pyglove.core.symbolic.boilerplate import boilerplate_class as pg_boilerplate_class
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.object import members as pg_members
from pyglove.core.symbolic.object import Object


@pg_members([
    ('a', pg_typing.Int()),
    ('b', pg_typing.Union([pg_typing.Int(), pg_typing.Str()])),
    ('c', pg_typing.Dict([
        ('d', pg_typing.List(pg_typing.Dict([
            ('e', pg_typing.Float()),
            ('f', pg_typing.Bool())
        ])))
    ]))
])
class A(Object):
  pass


template_object = A.partial(b='foo', c={'d': [{'e': 1.0, 'f': True}]})


# pylint: disable=invalid-name

B = pg_boilerplate_class('B', template_object)
C = pg_boilerplate_class('C', template_object, init_arg_list=['a', 'c', 'b'])

# pylint: enable=invalid-name


class BoilerplateClassTest(unittest.TestCase):
  """Tests for `pg.boilerplate_class`."""

  def test_basics(self):
    self.assertTrue(issubclass(B, A))
    self.assertEqual(B.__type_name__, 'pyglove.core.symbolic.boilerplate_test.B')

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'value\' must be an instance of .*Object subclass'):
      pg_boilerplate_class('A', 1)

    with self.assertRaisesRegex(
        TypeError, 'Unsupported keyword arguments'):
      pg_boilerplate_class('A', template_object, unsupported_keyword=1)

    with self.assertRaisesRegex(
        TypeError, '.* from `init_arg_list` is not defined'):
      pg_boilerplate_class('A', template_object, init_arg_list=['x', 'y'])

  def test_init_arg_list(self):
    self.assertEqual(B.init_arg_list, ['a'])
    self.assertEqual(C.init_arg_list, ['a', 'c', 'b'])

  def test_schema(self):
    # Boilerplate class' schema should carry the default value and be frozen.
    self.assertEqual(
        list(B.__schema__.fields.values()),
        list(pg_typing.create_schema(
            [
                ('a', pg_typing.Int()),
                (
                    'b',
                    pg_typing.Union(
                        [pg_typing.Int(), pg_typing.Str()], default='foo'
                    ).freeze(),
                ),
                (
                    'c',
                    pg_typing.Dict([(
                        'd',
                        pg_typing.List(
                            pg_typing.Dict([
                                ('e', pg_typing.Float()),
                                ('f', pg_typing.Bool()),
                            ]),
                            default=List([Dict(e=1.0, f=True)]),
                        ).freeze(),
                    )]).freeze(),
                ),
            ],
        ).fields.values())
    )

    # Original class' schema should remain unchanged.
    self.assertEqual(
        A.__schema__,
        pg_typing.create_schema([
            ('a', pg_typing.Int()),
            ('b', pg_typing.Union([pg_typing.Int(), pg_typing.Str()])),
            (
                'c',
                pg_typing.Dict([(
                    'd',
                    pg_typing.List(
                        pg_typing.Dict([
                            ('e', pg_typing.Float()),
                            ('f', pg_typing.Bool()),
                        ])
                    ),
                )]),
            ),
        ]),
    )

  def test_init(self):
    b = B(0)
    self.assertEqual(
        b,
        B.partial(
            a=0, b='foo', c={'d': [{
                'e': 1.0,
                'f': True
            }]}))
    self.assertEqual(C.init_arg_list, ['a', 'c', 'b'])

  def test_do_not_modify_original_object(self):
    b = B(a=1)
    with self.assertRaisesRegex(ValueError, 'Frozen field is not assignable.'):
      b.rebind(b=1)

    b.rebind({'c.d[0].f': False})
    self.assertFalse(b.c.d[0].f)

    # Default value of the boilerplate class remain unchanged.
    self.assertEqual(
        B.__schema__['c'].default_value,
        Dict.partial(
            {
                'd': [{
                    'e': 1.0,
                    'f': True,
                }]
            },
            value_spec=B.__schema__['c'].value,
        ),
    )

    # Original object remain unchanged.
    self.assertTrue(template_object.c.d[0].f)

  def test_serialization(self):
    b = B(a=1)
    self.assertEqual(b.to_json(), {'_type': B.__type_name__, 'a': 1})
    self.assertEqual(pg_from_json_str(b.to_json_str()), b)


if __name__ == '__main__':
  unittest.main()
