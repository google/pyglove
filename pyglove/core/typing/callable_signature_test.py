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
import dataclasses
import inspect
from typing import List
import unittest

from pyglove.core import object_utils
from pyglove.core.typing import annotation_conversion   # pylint: disable=unused-import
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs

Argument = callable_signature.Argument
Signature = callable_signature.Signature


class ArgumentTest(unittest.TestCase):
  """Tests for `Argument` class."""

  def test_kind(self):

    class Foo:
      def bar(self, x, *args, y, **kwargs):
        del x, args, y, kwargs

    sig = inspect.signature(Foo.bar)
    self.assertEqual(
        Argument.Kind.from_parameter(sig.parameters['self']),
        Argument.Kind.POSITIONAL_OR_KEYWORD
    )
    self.assertEqual(
        Argument.Kind.from_parameter(sig.parameters['x']),
        Argument.Kind.POSITIONAL_OR_KEYWORD
    )
    self.assertEqual(
        Argument.Kind.from_parameter(sig.parameters['args']),
        Argument.Kind.VAR_POSITIONAL
    )
    self.assertEqual(
        Argument.Kind.from_parameter(sig.parameters['y']),
        Argument.Kind.KEYWORD_ONLY
    )
    self.assertEqual(
        Argument.Kind.from_parameter(sig.parameters['kwargs']),
        Argument.Kind.VAR_KEYWORD
    )

  def test_init(self):
    self.assertEqual(
        Argument(
            'x', Argument.Kind.VAR_POSITIONAL, vs.List(vs.Int())).value_spec,
        vs.List(vs.Int(), default=[])
    )
    with self.assertRaisesRegex(
        TypeError,
        'Variable positional argument .* should have a value of .*List'
    ):
      Argument('x', Argument.Kind.VAR_POSITIONAL, vs.Int())

    with self.assertRaisesRegex(
        TypeError,
        'Variable keyword argument .* should have a value of .*Dict'
    ):
      Argument('x', Argument.Kind.VAR_KEYWORD, vs.Int())

  def test_from_parameter(self):

    def bar(x: int, *args, y: str, **kwargs):
      del x, args, y, kwargs

    sig = inspect.signature(bar)
    self.assertEqual(
        Argument.from_parameter(sig.parameters['x'], 'arg x', auto_typing=True),
        Argument(
            'x', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int(),
            description='arg x'
        )
    )
    self.assertEqual(
        Argument.from_parameter(sig.parameters['args'], 'varargs'),
        Argument(
            'args',
            Argument.Kind.VAR_POSITIONAL,
            vs.List(vs.Any(), default=[]),
            'varargs'
        )
    )
    self.assertEqual(
        Argument.from_parameter(sig.parameters['y'], 'arg y', auto_typing=True),
        Argument(
            'y', Argument.Kind.KEYWORD_ONLY, vs.Str(), description='arg y'
        )
    )
    self.assertEqual(
        Argument.from_parameter(sig.parameters['kwargs'], 'kwargs'),
        Argument(
            'kwargs',
            Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Any()),
            'kwargs'
        )
    )


class SignatureTest(unittest.TestCase):
  """Tests for `Signature` class."""

  def test_basics(self):
    """Test basics of `Signature` class."""

    def foo(a, *, b: int = 1):
      del a, b

    signature = callable_signature.signature(
        foo, auto_typing=False, auto_doc=False
    )
    self.assertEqual(signature.module_name, 'pyglove.core.typing.callable_signature_test')
    self.assertEqual(signature.name, 'foo')
    self.assertEqual(
        signature.id, 'pyglove.core.typing.callable_signature_test.SignatureTest.test_basics.<locals>.foo'
    )
    self.assertEqual(
        str(signature),
        inspect.cleandoc("""
        Signature(
          'pyglove.core.typing.callable_signature_test.SignatureTest.test_basics.<locals>.foo',
          args=[
            Argument(name='a', kind=<Kind.POSITIONAL_OR_KEYWORD: 1>, value_spec=Any(), description=None)
          ],
          kwonlyargs=[
            Argument(name='b', kind=<Kind.KEYWORD_ONLY: 3>, value_spec=Any(default=1, annotation=<class 'int'>), description=None)
          ]
        )
        """)
    )

    self.assertEqual(signature.named_args, [
        Argument('a', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any()),
        Argument(
            'b', Argument.Kind.KEYWORD_ONLY, vs.Any(default=1).annotate(int)
        ),
    ])
    self.assertEqual(signature.arg_names, ['a', 'b'])

    # Test __eq__ and __ne__.
    def assert_not_equal(signature, field_name, modified_value):
      other = copy.copy(signature)
      setattr(other, field_name, modified_value)
      self.assertNotEqual(signature, other)

    assert_not_equal(signature, 'name', 'bar')
    assert_not_equal(signature, 'module_name', 'other_module')
    assert_not_equal(
        signature, 'args',
        [signature.args[0],
         Argument('b', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int())]
    )
    assert_not_equal(
        signature, 'kwonlyargs',
        list(signature.kwonlyargs) + [
            Argument('x', Argument.Kind.KEYWORD_ONLY, vs.Any())]
    )
    assert_not_equal(
        signature, 'varargs',
        Argument('args', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any())
    )
    assert_not_equal(
        signature, 'varkw',
        Argument(
            'kwargs',
            Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Any())
        )
    )
    self.assertNotEqual(signature, 1)
    self.assertEqual(signature, signature)
    self.assertEqual(signature, copy.deepcopy(signature))

    with self.assertRaisesRegex(TypeError, '.* is not callable'):
      callable_signature.signature(1)

  def test_annotate(self):
    def foo(a, *args, b=1, **kwargs):
      del a, b, args, kwargs
      return 1

    signature = callable_signature.signature(foo).annotate(
        dict(
            a=int,
            b=(int, 'Field b'),
            c=(vs.Bool(default=True), 'Field c', dict(meta=1)),
            args=(List[int], 'Varargs'),
            kwargs=(str, 'Kwargs'),
        ),
        return_value=int,
    )
    self.assertEqual(signature.args, [
        Argument('a', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int()),
    ])
    self.assertEqual(signature.kwonlyargs, [
        Argument(
            'b', Argument.Kind.KEYWORD_ONLY, vs.Int(default=1), 'Field b'
        ),
        Argument(
            'c', Argument.Kind.KEYWORD_ONLY, vs.Bool(default=True), 'Field c'
        ),
    ])
    self.assertEqual(
        signature.varargs,
        Argument(
            'args',
            Argument.Kind.VAR_POSITIONAL,
            vs.List(vs.Int(), default=[]),
            'Varargs'
        )
    )
    self.assertEqual(
        signature.varkw,
        Argument(
            'kwargs', Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Str()), 'Kwargs'
        )
    )
    self.assertEqual(signature.return_value, vs.Int())

    # Customize the typing of kwargs.
    signature = callable_signature.signature(foo).annotate({ks.StrKey(): int})

    self.assertEqual(
        signature.varkw,
        Argument(
            'kwargs', Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Int())
        )
    )

    # Special handling noneable type specification.
    signature = callable_signature.signature(
        foo).annotate({'a': vs.Int().noneable()})

    self.assertEqual(
        signature.args,
        [Argument(
            'a', Argument.Kind.POSITIONAL_OR_KEYWORD,
            # No default value shall be specified.
            vs.Int(is_noneable=True),
        )]
    )

    # Special handling dict type specification.
    signature = callable_signature.signature(
        foo).annotate({'a': vs.Dict([('x', vs.Int())])})

    self.assertEqual(
        signature.args,
        [Argument(
            'a', Argument.Kind.POSITIONAL_OR_KEYWORD,
            vs.Dict([('x', vs.Int())])
        )]
    )

    # Bad override.
    signature = callable_signature.signature(foo)

    with self.assertRaisesRegex(
        ValueError, 'return value spec should not have default value'
    ):
      signature.annotate(return_value=vs.Int(default=1))

    with self.assertRaisesRegex(ValueError, '.*pg.typing.List'):
      signature.annotate(dict(args=int))

    with self.assertRaisesRegex(KeyError, '.*multiple StrKey'):
      signature.annotate([(ks.StrKey(), int), ('kwargs', str)])

    with self.assertRaisesRegex(KeyError, '.*multiple StrKey'):
      signature.annotate([('kwargs', str), (ks.StrKey(), int)])

    with self.assertRaisesRegex(ValueError, 'The annotated default value'):
      signature.annotate([('b', vs.Int(default=2))])

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered'):
      signature.annotate([('b', vs.Str())])

    signature = callable_signature.signature(lambda a: 1)
    with self.assertRaisesRegex(KeyError, '.*found extra symbolic argument'):
      signature.annotate([('b', vs.Int())])

  def test_to_schema(self):

    class Foo:
      def foo(self, a: int, *args, b: str = 'x', **kwargs) -> str:
        """Function foo.

        Args:
          a: An int.
          *args: Varargs.
          b: A str.
          **kwargs: Kwargs.

        Returns:
          A str.
        """
        del a, args, kwargs
        return b

    schema = Signature.from_callable(
        Foo.foo, auto_typing=True, auto_doc=True
    ).to_schema()
    self.assertEqual(
        schema,
        class_schema.Schema(
            [
                class_schema.Field('a', vs.Int(), 'An int.'),
                class_schema.Field(
                    'args', vs.List(vs.Any(), default=[]), 'Varargs.'
                ),
                class_schema.Field('b', vs.Str(default='x'), 'A str.'),
                class_schema.Field(ks.StrKey(), vs.Any(), 'Kwargs.'),
            ],
            allow_nonconst_keys=True,
        )
    )
    self.assertEqual(
        schema.name, f'{Foo.foo.__module__}.{Foo.foo.__qualname__}'
    )
    self.assertEqual(schema.description, 'Function foo.')
    self.assertTrue(schema.allow_nonconst_keys)
    self.assertEqual(
        schema.metadata,
        dict(
            init_arg_list=['a', '*args'],
            varargs_name='args',
            varkw_name='kwargs',
            returns=vs.Str(),
        )
    )

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
      new_func = callable_signature.signature(func).make_function(['pass'])
      old_signature = inspect.signature(func)
      new_signature = inspect.signature(new_func)
      self.assertEqual(old_signature, new_signature)


class FromCallableTest(unittest.TestCase):
  """Tests for `Signature.from_callable`."""

  def test_function(self):
    """Tests `from_callable` on regular functions."""

    def foo(a, *, b: int = 1, **kwargs):
      del a, b, kwargs

    signature = Signature.from_callable(foo)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.FUNCTION)
    self.assertEqual(signature.args, [
        Argument('a', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any()),
    ])
    self.assertEqual(signature.kwonlyargs, [
        Argument(
            'b', Argument.Kind.KEYWORD_ONLY, vs.Any(default=1).annotate(int)
        ),
    ])
    self.assertIsNone(signature.varargs)
    self.assertEqual(
        signature.varkw,
        Argument(
            'kwargs',
            Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Any())
        )
    )
    self.assertFalse(signature.has_varargs)
    self.assertTrue(signature.has_varkw)
    self.assertTrue(signature.has_wildcard_args)
    self.assertEqual(
        signature.get_value_spec('b'),
        vs.Any(default=1, annotation=int))
    # NOTE: 'x' matches **kwargs
    self.assertEqual(signature.get_value_spec('x'), vs.Any())

  def test_lambda(self):
    """Tests `from_callable` on lambda function."""
    signature = Signature.from_callable(lambda x: x)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.FUNCTION)
    self.assertEqual(
        signature.args,
        [Argument('x', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any())]
    )
    self.assertEqual(signature.kwonlyargs, [])
    self.assertIsNone(signature.varargs)
    self.assertIsNone(signature.varkw)
    self.assertFalse(signature.has_varargs)
    self.assertFalse(signature.has_varkw)
    self.assertFalse(signature.has_wildcard_args)
    self.assertIsNone(signature.get_value_spec('y'))

  def test_method(self):
    """Tests `from_callable` on class methods."""

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
    signature = Signature.from_callable(A.foo)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [Argument(
            'x',
            Argument.Kind.POSITIONAL_OR_KEYWORD,
            vs.Any(default=1).annotate(int)
        )]
    )
    self.assertEqual(signature.kwonlyargs, [])

    # Test instance method.
    signature = Signature.from_callable(A().bar)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [Argument(
            'y', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any().annotate(int)
        )]
    )
    self.assertEqual(
        signature.kwonlyargs,
        [Argument('z', Argument.Kind.KEYWORD_ONLY, vs.Any(default=1))]
    )
    self.assertEqual(
        signature.varargs,
        Argument(
            'args',
            Argument.Kind.VAR_POSITIONAL,
            vs.List(vs.Any(), default=[])
        )
    )
    self.assertTrue(signature.has_varargs)
    self.assertFalse(signature.has_varkw)

    # Test unbound instance method
    signature = Signature.from_callable(A.bar)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.FUNCTION)
    self.assertEqual(signature.args, [
        Argument('self', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any()),
        Argument(
            'y', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any().annotate(int)
        )
    ])
    self.assertEqual(
        signature.kwonlyargs,
        [Argument('z', Argument.Kind.KEYWORD_ONLY, vs.Any(default=1))]
    )
    self.assertEqual(
        signature.varargs,
        Argument(
            'args',
            Argument.Kind.VAR_POSITIONAL,
            vs.List(vs.Any(), default=[])
        )
    )
    self.assertTrue(signature.has_varargs)
    self.assertFalse(signature.has_varkw)

    # Test object as callable.
    signature = Signature.from_callable(A())
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD)
    self.assertEqual(
        signature.args,
        [Argument(
            'z', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any().annotate(int)
        )]
    )
    self.assertEqual(signature.kwonlyargs, [])
    self.assertFalse(signature.has_varargs)
    self.assertTrue(signature.has_varkw)
    self.assertEqual(
        signature.varkw,
        Argument(
            'kwargs', Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Any())
        )
    )

  def test_class(self):
    """Tests `from_callable` on classes."""

    class A:
      def __init__(self, x: int, *, y: str, **kwargs):
        """Constructor.

        Args:
          x: An int.
          y: A str.
          **kwargs: Kwargs.
        """

    signature = Signature.from_callable(A, auto_typing=True, auto_doc=True)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD
    )
    self.assertEqual(signature.name, A.__name__)
    self.assertEqual(signature.module_name, A.__module__)
    self.assertEqual(signature.qualname, A.__qualname__)
    self.assertIsNone(signature.description)

    self.assertEqual(
        signature.args,
        [Argument(
            'x', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int(),
            description='An int.'
        )]
    )
    self.assertEqual(
        signature.kwonlyargs,
        [Argument(
            'y', Argument.Kind.KEYWORD_ONLY, vs.Str(),
            description='A str.'
        )]
    )
    self.assertEqual(
        signature.varkw,
        Argument(
            'kwargs',
            Argument.Kind.VAR_KEYWORD,
            vs.Dict(vs.Any()),
            description='Kwargs.'
        )
    )

    @dataclasses.dataclass
    class B:
      """Class B.

      Params:
        x: An int.
        y: A str.
      """
      x: int
      y: str

    signature = Signature.from_callable(B, auto_typing=True, auto_doc=True)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD
    )
    self.assertEqual(signature.name, B.__name__)
    self.assertEqual(signature.module_name, B.__module__)
    self.assertEqual(signature.qualname, B.__qualname__)
    self.assertEqual(signature.description, 'Class B.')
    self.assertEqual(
        signature.args,
        [
            Argument(
                'x', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int(),
                description='An int.'
            ),
            Argument(
                'y', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Str(),
                description='A str.'
            )
        ]
    )

    # Signature for builtin classes
    signature = callable_signature.signature(bytes)
    self.assertEqual(
        signature.callable_type, callable_signature.CallableType.METHOD
    )
    self.assertEqual(signature.name, bytes.__name__)
    self.assertEqual(signature.module_name, bytes.__module__)
    self.assertEqual(signature.qualname, bytes.__qualname__)
    self.assertIsNotNone(signature.varargs)
    self.assertIsNotNone(signature.varkw)

  def test_signature_with_forward_declarations(self):
    signature = callable_signature.signature(object_utils.KeyPath)
    self.assertIs(signature.get_value_spec('parent').cls, object_utils.KeyPath)


class FromSchemaTest(unittest.TestCase):
  """Tests for `Signature.from_schema`."""

  def _signature(self, init_arg_list, is_method: bool = True):
    s = class_schema.Schema([
        class_schema.Field('x', vs.Int(), 'x'),
        class_schema.Field('y', vs.Int(), 'y'),
        # Frozen fields will be ignored.
        class_schema.Field('v', vs.Bool().freeze(True), 'v'),
        class_schema.Field('z', vs.List(vs.Int()), 'z'),
        class_schema.Field(ks.StrKey(), vs.Str(), 'kwargs'),
    ], metadata=dict(init_arg_list=init_arg_list), allow_nonconst_keys=True)
    return Signature.from_schema(
        s, 'bar', 'foo', is_method=is_method)

  def test_classmethod_with_regular_args(self):
    self.assertEqual(
        self._signature(['x', 'y', 'z']),
        Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='bar',
            name='foo',
            args=[
                Argument('self', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any()),
                Argument('x', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int()),
                Argument('y', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int()),
                Argument(
                    'z', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.List(vs.Int())
                ),
            ],
            varkw=Argument(
                'kwargs',
                Argument.Kind.VAR_KEYWORD,
                vs.Dict(vs.Str())
            )
        )
    )

  def test_function_with_varargs(self):
    self.assertEqual(
        self._signature(['x', '*z'], is_method=False),
        Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='bar',
            name='foo',
            args=[
                Argument('x', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Int()),
            ],
            kwonlyargs=[
                Argument('y', Argument.Kind.KEYWORD_ONLY, vs.Int()),
            ],
            varargs=Argument(
                'z', Argument.Kind.VAR_POSITIONAL, vs.List(vs.Int())
            ),
            varkw=Argument(
                'kwargs',
                Argument.Kind.VAR_KEYWORD,
                vs.Dict(vs.Str())
            )
        )
    )

  def test_classmethod_with_kwonly_args(self):
    self.assertEqual(
        self._signature([]),
        Signature(
            callable_type=callable_signature.CallableType.FUNCTION,
            module_name='bar',
            name='foo',
            args=[
                Argument('self', Argument.Kind.POSITIONAL_OR_KEYWORD, vs.Any()),
            ],
            kwonlyargs=[
                Argument('x', Argument.Kind.KEYWORD_ONLY, vs.Int()),
                Argument('y', Argument.Kind.KEYWORD_ONLY, vs.Int()),
                Argument(
                    'z', Argument.Kind.KEYWORD_ONLY, vs.List(vs.Int())
                ),
            ],
            varkw=Argument(
                'kwargs',
                Argument.Kind.VAR_KEYWORD,
                vs.Dict(vs.Str())
            )
        )
    )

  def test_bad_cases(self):
    with self.assertRaisesRegex(
        TypeError,
        'Variable positional argument \'x\' should have a value of '
        '`pg.typing.List` type'):
      _ = self._signature(['*x'])

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'a\' is not a symbolic field.'):
      _ = Signature.from_schema(
          class_schema.Schema([], metadata=dict(init_arg_list=['a'])),
          '__main__', 'foo')

    class Foo:
      __call__ = 1

    with self.assertRaisesRegex(
        TypeError, '.*__call__ is not a method'):
      Signature.from_callable(Foo())


class GetSchemaTest(unittest.TestCase):
  """Tests for `schema`."""

  def test_function_schema(self):
    def foo(x: int, *args, y: str, **kwargs) -> float:
      """A function.

      Args:
        x: Input 1.
        *args: Variable positional args.
        y: Input 2.
        **kwargs: Variable keyword args.

      Returns:
        The result.
      """
      del x, y, args, kwargs

    schema = callable_signature.schema(foo, auto_typing=True, auto_doc=True)
    self.assertEqual(schema.name, f'{foo.__module__}.{foo.__qualname__}')
    self.assertEqual(schema.description, 'A function.')
    self.assertEqual(
        list(schema.fields.values()),
        [
            class_schema.Field('x', vs.Int(), description='Input 1.'),
            class_schema.Field(
                'args',
                vs.List(vs.Any(), default=[]),
                description='Variable positional args.',
            ),
            class_schema.Field('y', vs.Str(), description='Input 2.'),
            class_schema.Field(
                ks.StrKey(),
                vs.Any(),
                description='Variable keyword args.',
            ),
        ],
    )

  def test_schema_on_symbolic_classes(self):

    class A:
      pass

    setattr(A, '__schema__', class_schema.create_schema([]))
    self.assertIs(callable_signature.schema(A), A.__schema__)

  def test_class_init_schema(self):
    class A:

      def __init__(self, x: int, *args, y: str, **kwargs) -> float:
        """Constructor.

        Args:
          x: Input 1.
          *args: Variable positional args.
          y: Input 2.
          **kwargs: Variable keyword args.

        Returns:
          The result.
        """
        del x, y, args, kwargs

    schema = callable_signature.schema(
        A.__init__, auto_typing=True, auto_doc=True, remove_self=True
    )
    self.assertEqual(schema.name, f'{A.__module__}.{A.__init__.__qualname__}')
    self.assertEqual(schema.description, 'Constructor.')
    self.assertEqual(
        list(schema.fields.values()),
        [
            class_schema.Field('x', vs.Int(), description='Input 1.'),
            class_schema.Field(
                'args',
                vs.List(vs.Any(), default=[]),
                description='Variable positional args.',
            ),
            class_schema.Field('y', vs.Str(), description='Input 2.'),
            class_schema.Field(
                ks.StrKey(),
                vs.Any(),
                description='Variable keyword args.',
            ),
        ],
    )


if __name__ == '__main__':
  unittest.main()
