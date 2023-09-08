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
"""Tests for pyglove.core.typing.callable_signature."""

import copy
import inspect
import unittest

from pyglove.core.typing import annotation_conversion   # pylint: disable=unused-import
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


class SignatureTest(unittest.TestCase):
  """Tests for `Signature` class."""

  def test_basics(self):
    """Test basics of `Signature` class."""

    def foo(a, b: int = 1):
      del a, b

    signature = callable_signature.get_signature(foo)
    self.assertEqual(signature.module_name, 'pyglove.core.typing.callable_signature_test')
    self.assertEqual(signature.name, 'foo')
    self.assertEqual(signature.id,
                     'pyglove.core.typing.callable_signature_test.SignatureTest.test_basics.<locals>.foo')
    self.assertEqual(
        str(signature),
        'Signature(\'pyglove.core.typing.callable_signature_test.SignatureTest.test_basics.<locals>.foo\', '
        'args=[\n'
        '  Argument(name=\'a\', value_spec=Any()),\n'
        '  Argument(name=\'b\', value_spec=Any('
        'default=1, annotation=<class \'int\'>))\n])')

    self.assertEqual(signature.named_args, [
        callable_signature.Argument('a', vs.Any()),
        callable_signature.Argument(
            'b', vs.Any(default=1).annotate(int)),
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
                      callable_signature.Argument('b', vs.Int())])
    assert_not_equal(
        signature, 'kwonlyargs',
        list(signature.kwonlyargs) + [
            callable_signature.Argument('x', vs.Any())])
    assert_not_equal(signature, 'varargs',
                     callable_signature.Argument('args', vs.Any()))
    assert_not_equal(signature, 'varkw',
                     callable_signature.Argument('kwargs', vs.Any()))
    self.assertNotEqual(signature, 1)
    self.assertEqual(signature, signature)
    self.assertEqual(signature, copy.deepcopy(signature))

    with self.assertRaisesRegex(TypeError, '.* is not callable'):
      callable_signature.get_signature(1)

  def test_function(self):
    """Tests `get_signature` on regular functions."""

    def foo(a, b: int = 1, **kwargs):
      del a, b, kwargs

    signature = callable_signature.get_signature(foo)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.FUNCTION)
    self.assertEqual(signature.args, [
        callable_signature.Argument('a', vs.Any()),
        callable_signature.Argument(
            'b', vs.Any(default=1).annotate(int)),
    ])
    self.assertEqual(signature.kwonlyargs, [])
    self.assertIsNone(signature.varargs)
    self.assertEqual(signature.varkw,
                     callable_signature.Argument('kwargs', vs.Any()))
    self.assertFalse(signature.has_varargs)
    self.assertTrue(signature.has_varkw)
    self.assertTrue(signature.has_wildcard_args)
    self.assertEqual(
        signature.get_value_spec('b'),
        vs.Any(default=1, annotation=int))
    # NOTE: 'x' matches **kwargs
    self.assertEqual(signature.get_value_spec('x'), vs.Any())

  def test_lambda(self):
    """Tests `get_signature` on lambda function."""
    signature = callable_signature.get_signature(lambda x: x)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.FUNCTION)
    self.assertEqual(
        signature.args, [callable_signature.Argument('x', vs.Any())])
    self.assertEqual(signature.kwonlyargs, [])
    self.assertIsNone(signature.varargs)
    self.assertIsNone(signature.varkw)
    self.assertFalse(signature.has_varargs)
    self.assertFalse(signature.has_varkw)
    self.assertFalse(signature.has_wildcard_args)
    self.assertIsNone(signature.get_value_spec('y'))

  def test_method(self):
    """Tests get_signature on class methods."""

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
    signature = callable_signature.get_signature(A.foo)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [callable_signature.Argument(
            'x', vs.Any(default=1).annotate(int))])
    self.assertEqual(signature.kwonlyargs, [])

    # Test instance method.
    signature = callable_signature.get_signature(A().bar)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [callable_signature.Argument('y', vs.Any().annotate(int))])
    self.assertEqual(
        signature.kwonlyargs,
        [callable_signature.Argument('z', vs.Any(default=1))])
    self.assertEqual(
        signature.varargs,
        callable_signature.Argument('args', vs.Any()))
    self.assertTrue(signature.has_varargs)
    self.assertFalse(signature.has_varkw)

    # Test unbound instance method
    signature = callable_signature.get_signature(A.bar)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.FUNCTION)
    self.assertEqual(signature.args, [
        callable_signature.Argument('self', vs.Any()),
        callable_signature.Argument('y', vs.Any().annotate(int))
    ])
    self.assertEqual(
        signature.kwonlyargs,
        [callable_signature.Argument('z', vs.Any(default=1))])
    self.assertEqual(
        signature.varargs,
        callable_signature.Argument('args', vs.Any()))
    self.assertTrue(signature.has_varargs)
    self.assertFalse(signature.has_varkw)

    # Test object as callable.
    signature = callable_signature.get_signature(A())
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [callable_signature.Argument('z', vs.Any().annotate(int))])
    self.assertEqual(signature.kwonlyargs, [])
    self.assertFalse(signature.has_varargs)
    self.assertTrue(signature.has_varkw)
    self.assertEqual(
        signature.varkw,
        callable_signature.Argument('kwargs', vs.Any()))

  def test_make_function(self):
    """Tests `Signature.make_function`."""

    def func1(x, y=1) -> int:
      del x, y

    def func2(x=1, *, y):
      del x, y

    def func3(x=1, *y):    # pylint: disable=keyword-arg-before-vararg
      del x, y

    def func4(*y):
      del y

    def func5(*, x=1, y):
      del x, y

    def func6(x=1, *, y, **z) -> str:
      del x, y, z

    for func in [func1, func2, func3, func4, func5, func6]:
      new_func = callable_signature.get_signature(func).make_function(['pass'])
      old_signature = inspect.signature(func)
      new_signature = inspect.signature(new_func)
      self.assertEqual(old_signature, new_signature)


class FromSchemaTest(unittest.TestCase):
  """Tests for `Signature.from_schema`."""

  def _get_signature(self, init_arg_list, is_method: bool = True):
    s = class_schema.Schema([
        class_schema.Field('x', vs.Int(), 'x'),
        class_schema.Field('y', vs.Int(), 'y'),
        class_schema.Field('z', vs.List(vs.Int()), 'z'),
        class_schema.Field(ks.StrKey(), vs.Str(), 'kwargs'),
    ], metadata=dict(init_arg_list=init_arg_list), allow_nonconst_keys=True)
    return callable_signature.Signature.from_schema(
        s, 'bar', 'foo', is_method=is_method)

  def test_classmethod_with_regular_args(self):
    self.assertEqual(
        self._get_signature(['x', 'y', 'z']),
        callable_signature.Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='bar',
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
            module_name='bar',
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
            module_name='bar',
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
      _ = callable_signature.Signature.from_schema(
          class_schema.Schema([], metadata=dict(init_arg_list=['a'])),
          '__main__', 'foo')


if __name__ == '__main__':
  unittest.main()
