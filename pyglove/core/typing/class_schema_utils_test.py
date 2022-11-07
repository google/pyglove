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
        class_schema.Field('a', vs.Any(), 'Argument \'a\'.'),
        class_schema.Field(
            'args', vs.List(vs.Any(), default=[]),
            'Wildcard positional arguments.'),
        class_schema.Field(
            'b', vs.Any().set_default(1), 'Argument \'b\'.'),
        class_schema.Field(ks.StrKey(), vs.Any(), 'Wildcard keyword arguments.')
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
        class_schema.Field(
            'args', vs.List(vs.Int(), default=[])),
        class_schema.Field('b', vs.Str(default='foo')),
        class_schema.Field('c', vs.Str()),
        class_schema.Field(ks.StrKey(), vs.Any(), 'Wildcard keyword arguments.')
    ])

  def test_partial_typing(self):
    signature = callable_signature.get_signature(lambda a, b='foo': 1)
    arg_fields = class_schema_utils.get_arg_fields(signature, [
        ('b', vs.Str()),
    ])
    self.assertEqual(arg_fields, [
        class_schema.Field('a', vs.Any(), 'Argument \'a\'.'),
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


class GetInitSignatureTest(unittest.TestCase):
  """Tests for `get_init_signature`."""

  def _get_signature(self, init_arg_list, is_method: bool = True):
    s = class_schema.Schema([
        class_schema.Field('x', vs.Int(), 'x'),
        class_schema.Field('y', vs.Int(), 'y'),
        class_schema.Field('z', vs.List(vs.Int()), 'z'),
        class_schema.Field(ks.StrKey(), vs.Str(), 'kwargs'),
    ], metadata=dict(init_arg_list=init_arg_list), allow_nonconst_keys=True)
    return class_schema_utils.get_init_signature(
        s, '__main__', 'foo', is_method=is_method)

  def test_classmethod_with_regular_args(self):
    self.assertEqual(
        self._get_signature(['x', 'y', 'z']),
        callable_signature.Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='__main__',
            name='foo',
            args=[
                callable_signature.Argument('self', vs.Any()),
                callable_signature.Argument('x', vs.Int()),
                callable_signature.Argument('y', vs.Int()),
                callable_signature.Argument('z', vs.List(vs.Int())),
            ],
            varkw=callable_signature.Argument('kwargs', vs.Str())))

  def test_function_with_varargs(self):
    self.assertEqual(
        self._get_signature(['x', '*z'], is_method=False),
        callable_signature.Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='__main__',
            name='foo',
            args=[
                callable_signature.Argument('x', vs.Int()),
            ],
            kwonlyargs=[
                callable_signature.Argument('y', vs.Int()),
            ],
            varargs=callable_signature.Argument('z', vs.Int()),
            varkw=callable_signature.Argument('kwargs', vs.Str())))

  def test_classmethod_with_kwonly_args(self):
    self.assertEqual(
        self._get_signature([]),
        callable_signature.Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='__main__',
            name='foo',
            args=[
                callable_signature.Argument('self', vs.Any()),
            ],
            kwonlyargs=[
                callable_signature.Argument('x', vs.Int()),
                callable_signature.Argument('y', vs.Int()),
                callable_signature.Argument(
                    'z', vs.List(vs.Int())),
            ],
            varkw=callable_signature.Argument('kwargs', vs.Str())))

  def test_bad_cases(self):
    with self.assertRaisesRegex(
        ValueError,
        'Variable positional argument \'x\' should have a value of '
        '`pg.typing.List` type'):
      _ = self._get_signature(['*x'])

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'a\' is not a symbolic field.'):
      _ = class_schema_utils.get_init_signature(
          class_schema.Schema([], metadata=dict(init_arg_list=['a'])),
          '__main__', 'foo')


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
