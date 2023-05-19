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
"""Tests for pyglove.core.typing.class_schema_utils."""

import unittest

from pyglove.core import object_utils
from pyglove.core.typing import annotation_conversion   # pylint: disable=unused-import
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import class_schema_utils
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


class GetArgFieldsTest(unittest.TestCase):
  """Tests for `get_arg_fields`."""

  def test_basics(self):
    signature = callable_signature.get_signature(
        lambda a, *args, b=1, **kwargs: 1)
    arg_fields = class_schema_utils.get_arg_fields(signature)
    self.assertEqual(arg_fields, [
        class_schema.Field('a', vs.Any()),
        class_schema.Field(
            'args', vs.List(vs.Any(), default=[])),
        class_schema.Field(
            'b', vs.Any().set_default(1)),
        class_schema.Field(ks.StrKey(), vs.Any())
    ])

  def test_full_typing(self):
    signature = callable_signature.get_signature(
        lambda a, *args, b='foo', **kwargs: 1)
    arg_fields = class_schema_utils.get_arg_fields(signature, [
        ('b', vs.Str()),
        ('args', vs.List(vs.Int())),
        ('a', vs.Int()),
        ('c', vs.Str()),
    ])
    self.assertEqual(arg_fields, [
        class_schema.Field('a', vs.Int()),
        class_schema.Field('args', vs.List(vs.Int(), default=[])),
        class_schema.Field('b', vs.Str(default='foo')),
        class_schema.Field('c', vs.Str()),
        class_schema.Field(ks.StrKey(), vs.Any())
    ])

  def test_partial_typing(self):
    signature = callable_signature.get_signature(lambda a, b='foo': 1)
    arg_fields = class_schema_utils.get_arg_fields(signature, [
        ('b', vs.Str()),
    ])
    self.assertEqual(arg_fields, [
        class_schema.Field('a', vs.Any()),
        class_schema.Field('b', vs.Str(default='foo')),
    ])

    # Special cases for Dict
    signature = callable_signature.get_signature(lambda a: 1)
    arg_fields = class_schema_utils.get_arg_fields(signature, [
        ('a', vs.Dict([
            ('x', vs.Int())
        ])),
    ])
    self.assertEqual(arg_fields, [
        class_schema.Field('a', vs.Dict([('x', vs.Int())]))
    ])

  def test_use_docstr_as_description(self):
    signature = callable_signature.get_signature(
        lambda a, *args, b='foo', **kwargs: 1)
    arg_fields = class_schema_utils.get_arg_fields(
        signature,
        [
            ('a', vs.Int()),
            ('b', vs.Str(default='foo'), 'Original description for b'),
        ],
        {
            'a': object_utils.DocStrArgument(name='a', description='An int'),
            'b': object_utils.DocStrArgument(name='b', description='A str'),
            '*args': object_utils.DocStrArgument(name='*args',
                                                 description='Args'),
            '**kwargs': object_utils.DocStrArgument(name='**kwargs',
                                                    description='Kwargs'),
        })
    self.assertEqual(arg_fields, [
        class_schema.Field('a', vs.Int(), 'An int'),
        class_schema.Field(
            'args', vs.List(vs.Any(), default=[]), 'Args'),
        class_schema.Field('b', vs.Str(default='foo'),
                           'Original description for b'),
        class_schema.Field(ks.StrKey(), vs.Any(), 'Kwargs')
    ])

  def test_bad_typing(self):
    with self.assertRaisesRegex(
        KeyError, 'multiple StrKey found in symbolic arguments declaration.'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a: 1),
          [(ks.StrKey(), vs.Int()), (ks.StrKey(), vs.Int())])

    with self.assertRaisesRegex(
        KeyError, 'multiple symbolic fields found for argument \'a\'.'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a: 1),
          [('a', vs.Int()), ('a', vs.Int())])

    with self.assertRaisesRegex(
        KeyError, 'found extra symbolic argument \'b\'.'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a: 1),
          [('b', vs.Int())])

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*.'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a=1: 1),
          [('a', vs.Str())])

    with self.assertRaisesRegex(
        ValueError,
        'the default value .* of symbolic argument \'a\' does not equal '
        'to the default value .* specified at function signature'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a=1: 1),
          [('a', vs.Int(default=2))])

    with self.assertRaisesRegex(
        ValueError,
        'the default value .* of symbolic argument \'a\' does not equal '
        'to the default value .* specified at function signature'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a={}: 1),
          [('a', vs.Dict([('x', vs.Int())]))])

    with self.assertRaisesRegex(
        ValueError,
        '.*the value spec for positional wildcard argument .* must be a '
        '`pg.typing.List` instance'):
      class_schema_utils.get_arg_fields(
          callable_signature.get_signature(lambda a, *args: 1),
          [('args', vs.Int())])


class EnsureValueSpecTest(unittest.TestCase):
  """Tests for `ensure_value_spec`."""

  def test_basics(self):
    self.assertEqual(
        class_schema_utils.ensure_value_spec(
            vs.Int(min_value=1), vs.Int()),
        vs.Int(min_value=1))

    self.assertEqual(
        class_schema_utils.ensure_value_spec(
            vs.Int(min_value=1), vs.Number(int)),
        vs.Int(min_value=1))

    with self.assertRaisesRegex(
        TypeError, 'Source spec .* is not compatible with destination spec'):
      class_schema_utils.ensure_value_spec(
          vs.Int(min_value=1), vs.Bool())

  def test_union(self):
    self.assertEqual(
        class_schema_utils.ensure_value_spec(
            vs.Union([vs.Int(), vs.Str(regex='a.*')]),
            vs.Str()), vs.Str(regex='a.*'))

    with self.assertRaisesRegex(
        TypeError, 'Source spec .* is not compatible with destination spec'):
      class_schema_utils.ensure_value_spec(
          vs.Union([vs.Int(), vs.Str()]),
          vs.Bool())

  def test_any(self):
    self.assertIsNone(
        class_schema_utils.ensure_value_spec(vs.Any(), vs.Int()))


if __name__ == '__main__':
  unittest.main()
