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
"""Tests for pyglove.Functor."""

import inspect
import io
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.base import from_json_str as pg_from_json_str
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.functor import as_functor as pg_as_functor
from pyglove.core.symbolic.functor import Functor
from pyglove.core.symbolic.functor import functor as pg_functor
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.object import members as pg_members
from pyglove.core.symbolic.object import Object


MISSING_VALUE = object_utils.MISSING_VALUE


class FunctorTest(unittest.TestCase):
  """Tests for `pg.Functor`."""

  def test_basics(self):
    @pg_functor
    def f(a, *args, b, c=0, **kwargs):
      return a + sum(args)  + b + c + sum(kwargs.values())

    # Basic properties of a symbolic function (functor).
    x = f(1, 2, 3, b=4, c=5, d=6)
    self.assertIsInstance(x, Functor)
    self.assertEqual(x(), 1 + 2 + 3 + 4 + 5 + 6)

    self.assertEqual(x.a, 1)
    self.assertEqual(x.args, [2, 3])
    self.assertEqual(x.b, 4)
    self.assertEqual(x.c, 5)
    self.assertEqual(x.d, 6)    # kwargs.

    self.assertEqual(x.specified_args, {'a', 'args', 'b', 'c', 'd'})
    self.assertEqual(x.bound_args, {'a', 'args', 'b', 'c', 'd'})
    self.assertEqual(x.unbound_args, set())
    self.assertEqual(x.non_default_args, {'a', 'args', 'b', 'c', 'd'})
    self.assertEqual(x.default_args, set())

    # Partial binding.
    x = f.partial(b=2)
    self.assertEqual(x(1), 1 + 2 + 0)

    # Rebinding.
    x.rebind(b=3)
    self.assertEqual(x(1), 1 + 3 + 0)

    # Incremental binding.
    x.a = 1
    self.assertEqual(x(), 1 + 3 + 0)

    # Temporarily override binding at call time.
    self.assertEqual(x(a=2, override_args=True), 2 + 3 + 0)
    self.assertEqual(x.a, 1)
    self.assertEqual(x(), 1 + 3 + 0)

    # Inspection.
    self.assertEqual(repr(x), 'f(a=1, args=[], b=3, c=0)')
    self.assertEqual(
        x.format(compact=True, hide_default_values=True), 'f(a=1, b=3)')

    # Comparison.
    self.assertEqual(x, f(1, b=3))
    self.assertTrue(x.sym_lt(f(1, b=4)))

    # Serialization.
    self.assertEqual(pg_from_json_str(x.to_json_str()), x)

  def test_automatic_typing(self):
    @pg_functor()
    def f(a, b, *args, c=0, **kwargs):
      return a + b + c + sum(args) + sum(kwargs.values())

    self.assertEqual(
        list(f.schema.values()), [
            pg_typing.Field('a', pg_typing.Any(), 'Argument \'a\'.'),
            pg_typing.Field('b', pg_typing.Any(), 'Argument \'b\'.'),
            pg_typing.Field('args', pg_typing.List(pg_typing.Any(), default=[]),
                            'Wildcard positional arguments.'),
            pg_typing.Field('c', pg_typing.Any(default=0), 'Argument \'c\'.'),
            pg_typing.Field(pg_typing.StrKey(), pg_typing.Any(),
                            'Wildcard keyword arguments.'),
        ])
    self.assertEqual(f.signature.args, [
        pg_typing.Argument('a', pg_typing.Any()),
        pg_typing.Argument('b', pg_typing.Any())
    ])
    self.assertEqual(
        f.signature.varargs,
        pg_typing.Argument('args', pg_typing.List(pg_typing.Any(), default=[])))
    self.assertEqual(
        f.signature.varkw, pg_typing.Argument('kwargs', pg_typing.Any()))
    self.assertEqual(
        f.signature.kwonlyargs,
        [pg_typing.Argument('c', pg_typing.Any(default=0))])
    self.assertIsNone(f.signature.return_value, None)
    self.assertTrue(f.signature.has_varargs)
    self.assertIsInstance(f.partial(), Functor)
    self.assertEqual(f.partial()(1, 2), 3)
    self.assertEqual(f.partial(b=1)(1), 2)
    self.assertEqual(f.partial(b=1)(a=1), 2)

    self.assertEqual(f(1, 2, 3, 4)(), 10)
    self.assertEqual(f(1, 2, 3, 4, c=5)(), 15)
    self.assertEqual(f(1, 2, 3, 4, c=5, x=5)(), 20)
    self.assertEqual(f(1, 2, 3, 4)(2, 3, 4, override_args=True), 9)
    self.assertEqual(f(1, 2, 3, 4)(c=1), 11)
    self.assertEqual(f(1, 2, 3, 4)(x=2), 12)
    self.assertEqual(f.partial(b=1)(2, c=4), 7)

  def test_full_typing(self):
    @pg_functor([
        ('a', pg_typing.Int()),
        ('b', pg_typing.Int()),
    ], returns=pg_typing.Int())
    def f(a=1, b=2):
      return a + b

    self.assertEqual(f.signature.args, [
        pg_typing.Argument('a', pg_typing.Int(default=1)),
        pg_typing.Argument('b', pg_typing.Int(default=2)),
    ])
    self.assertEqual(
        list(f.schema.values()), [
            pg_typing.Field('a', pg_typing.Int(default=1)),
            pg_typing.Field('b', pg_typing.Int(default=2)),
        ])
    self.assertEqual(f.signature.return_value, pg_typing.Int())
    self.assertFalse(f.signature.has_varargs)
    self.assertFalse(f.signature.has_varkw)
    self.assertEqual(f.partial()(), 3)
    self.assertEqual(f.partial(a=2)(b=2), 4)
    self.assertEqual(f.partial(a=3, b=2)(), 5)
    self.assertEqual(f.partial(1, 2)(), 3)

    # Override default value.
    self.assertEqual(f.partial(a=2)(), 4)
    self.assertEqual(f.partial(a=2, override_args=True)(a=3), 5)
    self.assertEqual(f.partial(a=1, b=1, override_args=True)(a=2, b=2), 4)
    self.assertEqual(f.partial(2, 4, override_args=True)(1), 5)

  def test_partial_typing(self):
    @pg_functor([
        ('c', pg_typing.Int()),
        ('a', pg_typing.Int()),
    ])
    def f(a, b=1, c=1):
      return a + b + c

    self.assertEqual(f.signature.args, [
        pg_typing.Argument('a', pg_typing.Int()),
        pg_typing.Argument('b', pg_typing.Any(default=1)),
        pg_typing.Argument('c', pg_typing.Int(default=1)),
    ])
    self.assertFalse(f.signature.has_varargs)
    self.assertFalse(f.signature.has_varkw)
    self.assertEqual(
        list(f.schema.values()), [
            pg_typing.Field('a', pg_typing.Int()),
            pg_typing.Field('b', pg_typing.Any(default=1), 'Argument \'b\'.'),
            pg_typing.Field('c', pg_typing.Int(default=1)),
        ])
    self.assertEqual(f.partial()(a=2), 4)
    # Override 'a' with 2, provide 'b' with 2, use 'c' from default value 1.
    self.assertEqual(f.partial()(2, 2), 5)

    # 'a' is not provided.
    with self.assertRaisesRegex(
        TypeError, 'missing 1 required positional argument'):
      f.partial()()

  def test_runtime_type_check(self):
    @pg_functor([
        ('a', pg_typing.Int(min_value=0)),
        ('b', pg_typing.Int(max_value=10)),
        ('args', pg_typing.List(pg_typing.Int())),
        (pg_typing.StrKey(), pg_typing.Int(max_value=5))
    ], returns=pg_typing.Int(min_value=0))
    def f(a, *args, b, **kwargs):
      return a + b + sum(args) + sum(kwargs.values())

    self.assertEqual(f(1, 2, b=3)(c=4), 10)

    # Validate during pre-binding.
    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range .*min=0'):
      f(-1, b=1)

    with self.assertRaisesRegex(
        TypeError, 'Expect .*int.* but encountered .*float'):
      f(1, 0.1, b=1)

    with self.assertRaisesRegex(
        ValueError, 'Value 11 is out of range .*max=10'):
      f(1, b=11)

    with self.assertRaisesRegex(
        ValueError, 'Value 10 is out of range .*max=5'):
      f(1, b=1, c=10)

    # Validate during late binding.
    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range .*min=0'):
      f.partial(b=2)(-1)

    with self.assertRaisesRegex(
        TypeError, 'Expect .*int.* but encountered .*float'):
      f(1, b=1)(1, 0.1, override_args=True)

    with self.assertRaisesRegex(
        ValueError, 'Value 11 is out of range .*max=10'):
      f(1)(b=11)   # pylint: disable=missing-kwoa

    with self.assertRaisesRegex(
        ValueError, 'Value 10 is out of range .*max=5'):
      f(1, b=1)(c=10)

    with self.assertRaisesRegex(
        ValueError, 'Value 6 is out of range .*max=5'):
      f(1, b=1)(c=6)

    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range .*min=0'):
      f(0, b=-1)()

  def test_symbolization_on_nested_containers(self):
    @pg_functor
    def f(a, b):
      del a, b

    x = f([], {})
    self.assertIsInstance(x.a, List)
    self.assertIsInstance(x.b, Dict)

  def test_implicit_copy_during_assignment(self):
    @pg_functor
    def f(a, b):
      del a, b

    class X:
      pass

    # There is no impliit copy when assigning a root symbolic object to
    # another tree.
    x = X()
    f1 = f(x, f({}, []))
    f2 = f(f1, 1)
    self.assertIs(f2.a, f1)

    # There is an implicit copy when assigning a symbolic object with
    # a parent to another tree.
    sd = Dict(f=f1)
    self.assertEqual(f1, sd.f)
    self.assertIsNot(f1, sd.f)
    self.assertIsNot(f1.b.a, sd.f.b.a)
    self.assertIsNot(f1.b.b, sd.f.b.b)
    # Non-symbolic member is copied by reference.
    self.assertIs(f1.a, sd.f.a)

  def test_as_functor(self):
    f = pg_as_functor(lambda x: x)
    self.assertIsInstance(f, Functor)
    self.assertEqual(
        f.signature.args, [pg_typing.Argument('x', pg_typing.Any())])
    self.assertIsNone(f.signature.return_value)
    self.assertEqual(f(1), 1)

  def test_bad_definition(self):
    # `pg.functor` decorator is not applicable to class.
    with self.assertRaisesRegex(TypeError, '.* is not a method.'):

      @pg_functor([('a', pg_typing.Int())])
      class A:
        pass

      del A

    with self.assertRaisesRegex(
        TypeError, '.* got an unexpected keyword argument'):
      @pg_functor(unsupported_keyword=1)
      def f1(x):   # pylint: disable=unused-variable
        return x

    # `pg.functor` decorator found extra symbolic argument.
    with self.assertRaisesRegex(
        KeyError, '.* found extra symbolic argument \'a\'.'):
      @pg_functor([('a', pg_typing.Int())])
      def f2():  # pylint: disable=unused-variable
        pass

    # `pg.functor` decorator found extra symbolic argument.
    with self.assertRaisesRegex(
        ValueError,
        'the value spec for positional wildcard argument .*'
        'must be a `pg.typing.List` instance'):
      @pg_functor([('args', pg_typing.Int())])
      def f3(*args):  # pylint: disable=unused-variable
        del args

    # `pg.functor` decorator has multiple StrKey.
    with self.assertRaisesRegex(
        KeyError,
        '.* multiple StrKey found in symbolic arguments declaration.'):
      @pg_functor([
          ('a', pg_typing.Int()),
          (pg_typing.StrKey(), pg_typing.Any()),
          (pg_typing.StrKey(), pg_typing.Any()),
      ])  # pylint: disable=unused-variable
      def f4(a, **kwargs):
        del a, kwargs

    with self.assertRaisesRegex(
        KeyError, '.* multiple symbolic fields found for argument.'):
      @pg_functor([
          ('a', pg_typing.Int()),
          ('a', pg_typing.Str()),
      ])  # pylint: disable=unused-variable
      def f5(a):
        del a

    with self.assertRaisesRegex(
        ValueError,
        '.* the default value .* of symbolic argument .* does not equal '
        'to the default value .* specified at function signature'):
      @pg_functor([
          ('a', pg_typing.Int(default=2)),
      ])  # pylint: disable=unused-variable
      def f6(a=1):
        del a

    with self.assertRaisesRegex(
        ValueError, 'return value spec should not have default value'):
      @pg_functor([
          ('a', pg_typing.Int()),
      ], returns=pg_typing.Any(default=None))
      def f7(a=1):  # pylint: disable=unused-variable
        del a

  def test_bad_call(self):

    @pg_functor([
        ('a', pg_typing.Int()),
    ])
    def f(a=1):
      del a

    with self.assertRaisesRegex(
        TypeError, '.* got multiple values for keyword argument \'a\''):
      f(1, a=1)  # pylint: disable=redundant-keyword-arg

    with self.assertRaisesRegex(
        TypeError, '.* takes 1 positional argument but 2 were given'):
      f(1, 2)  # pylint: disable=too-many-function-args

    with self.assertRaisesRegex(
        TypeError, '.* takes 1 positional argument but 2 were given'):
      f.partial()(1, 2)  # pylint: disable=too-many-function-args

  def test_inspect_signature(self):
    @pg_functor([
        ('a', pg_typing.Int())
    ])
    def f(a, *b, c=1, **d):
      del a, b, c, d

    signature = inspect.signature(f.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'a', 'b', 'c', 'd'])
    self.assertEqual(
        signature.parameters['self'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['self'].annotation,
        inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['self'].default,
        inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['a'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['a'].annotation, int)
    self.assertEqual(
        signature.parameters['a'].default,
        inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['b'].kind,
        inspect.Parameter.VAR_POSITIONAL)
    self.assertEqual(
        signature.parameters['b'].annotation,
        inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['c'].kind,
        inspect.Parameter.KEYWORD_ONLY)
    self.assertEqual(
        signature.parameters['c'].annotation,
        inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['c'].default, 1)

    self.assertEqual(
        signature.parameters['d'].kind,
        inspect.Parameter.VAR_KEYWORD)
    self.assertEqual(
        signature.parameters['d'].annotation,
        inspect.Signature.empty)

  def test_inspect(self):
    @pg_functor
    def f(x):
      del x

    s = io.StringIO()
    x = f([f(1), f(2)])
    x.inspect(file=s, compact=True)
    self.assertEqual(s.getvalue(), 'f(x=[0: f(x=1), 1: f(x=2)])\n')

    s = io.StringIO()
    x.inspect(where=lambda v: v == 1, file=s)
    self.assertEqual(s.getvalue(), '{\n  \'x[0].x\': 1\n}\n')

  def test_sym_init_args(self):

    @pg_functor
    def f(x, y=True):
      del x, y

    x = f(1)
    self.assertEqual(x.sym_init_args, {
        'x': 1,
        'y': True
    })
    x = f.partial()
    self.assertEqual(x.sym_init_args, {
        'x': MISSING_VALUE,
        'y': True
    })

  def test_setattr(self):
    @pg_functor
    def f(a, b=1):
      return a + b

    x = f(0)
    self.assertEqual(x.specified_args, set(['a']))
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x.non_default_args, set(['a']))
    self.assertEqual(x.default_args, set(['b']))
    self.assertEqual(x(), 1)                          # pylint: disable=not-callable

    x.b = 2
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x(), 2)                          # pylint: disable=not-callable

    x.a = 1
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x(), 3)                          # pylint: disable=not-callable

    # Reset x.b to default value 1.
    x.b = MISSING_VALUE
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x(), 2)                          # pylint: disable=not-callable

  def test_getattr(self):
    @pg_functor
    def f(a, b=True):
      del a, b

    x = f(1)
    self.assertEqual(x.a, 1)
    self.assertTrue(x.b)      # Use default value from the field definition.

  def test_rebind(self):
    @pg_functor
    def f(a, b=1):
      return a + b

    x = f(0)
    self.assertEqual(x.specified_args, set(['a']))
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x.unbound_args, set())
    self.assertEqual(x.non_default_args, set(['a']))
    self.assertEqual(x.default_args, set(['b']))
    self.assertEqual(x(), 1)                          # pylint: disable=not-callable

    x.rebind(b=2)
    self.assertEqual(x.specified_args, set(['a', 'b']))
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x.unbound_args, set())
    self.assertEqual(x.non_default_args, set(['a', 'b']))
    self.assertEqual(x.default_args, set())
    self.assertEqual(x(), 2)                          # pylint: disable=not-callable

    x.rebind(a=1)
    self.assertEqual(x.specified_args, set(['a', 'b']))
    self.assertEqual(x.bound_args, set(['a', 'b']))
    self.assertEqual(x.unbound_args, set())
    self.assertEqual(x.non_default_args, set(['a', 'b']))
    self.assertEqual(x.default_args, set())
    self.assertEqual(x(), 3)                          # pylint: disable=not-callable

    # Unspecify x.x.
    x.rebind(a=MISSING_VALUE)
    self.assertEqual(x.specified_args, set(['b']))
    self.assertEqual(x.bound_args, set(['b']))
    self.assertEqual(x.unbound_args, set(['a']))
    self.assertEqual(x.non_default_args, set(['b']))
    self.assertEqual(x.default_args, set())
    with self.assertRaisesRegex(
        TypeError, 'missing 1 required positional argument'):
      _ = x()                                          # pylint: disable=not-callable

    # Reset x.b to default value 1.
    x.rebind(b=MISSING_VALUE)
    # self.assertEqual(x.specified_args, set(['a']))
    self.assertEqual(x.bound_args, set(['b']))
    self.assertEqual(x.unbound_args, set(['a']))
    self.assertEqual(x.non_default_args, set())
    self.assertEqual(x.default_args, set(['b']))

    x.rebind(a=1)
    self.assertEqual(x(), 2)                          # pylint: disable=not-callable

  def test_partial(self):
    @pg_functor
    def f(a, b):
      del a, b

    # An incompletely bound functor is not considered partial.
    self.assertFalse(f.partial().is_partial)

    @pg_members([('x', pg_typing.Int())])
    class A(Object):
      pass

    # A bound argument with partial value will make the functor partial.
    x = f(a=A.partial(), b=1)
    self.assertTrue(x.is_partial)
    self.assertEqual(x.missing_values(), {'a.x': MISSING_VALUE})

  def test_clone(self):

    @pg_functor
    def f(a, b=1):
      return a + b

    x = f(0, override_args=True, ignore_extra_args=True).seal(True)
    # Make sure `bound_args` is correctly set while `clone` constructs
    # a new object with non-args arguments.
    # (e.g. sealed, etc.)
    y = x.clone()
    self.assertTrue(y.is_sealed)
    self.assertEqual(y.bound_args, x.bound_args)
    self.assertEqual(y.default_args, x.default_args)
    self.assertEqual(y.non_default_args, x.non_default_args)
    self.assertEqual(y._override_args, x._override_args)
    self.assertEqual(y._ignore_extra_args, x._ignore_extra_args)

    y = x.clone(deep=True)
    self.assertTrue(y.is_sealed)
    self.assertEqual(y.bound_args, x.bound_args)
    self.assertEqual(y.default_args, x.default_args)
    self.assertEqual(y.non_default_args, x.non_default_args)
    self.assertEqual(y._override_args, x._override_args)
    self.assertEqual(y._ignore_extra_args, x._ignore_extra_args)

  def test_override_args(self):
    @pg_functor()
    def f(x, *args, a=1, **kwargs):
      return x + sum(args) + a + sum(kwargs.values())

    self.assertEqual(f.partial()(1), 2)
    # Override default value is not treated as override.
    self.assertEqual(f.partial()(1, a=2), 3)
    self.assertEqual(f(1, override_args=True)(2), 3)
    self.assertEqual(f(1, override_args=True)(a=2), 3)
    with self.assertRaisesRegex(
        TypeError,
        '.* got new value for argument \'x\' from position 0'):
      f(0)(1)

    with self.assertRaisesRegex(
        TypeError,
        '.* got new value for argument \'a\' from keyword argument'):
      f(0, a=1)(a=2)

  def test_ignore_extra_args(self):
    """Test functor with/without `ignore_extra_args` flag."""
    # Test ignore extra args.
    @pg_functor()
    def f(a=1):
      return a

    self.assertEqual(f.partial(ignore_extra_args=True)(1, c=0), 1)
    self.assertEqual(f.partial()(1, c=0, ignore_extra_args=True), 1)
    self.assertEqual(f.partial()(1, 2, ignore_extra_args=True), 1)
    with self.assertRaisesRegex(
        TypeError, '.* got an unexpected keyword argument \'c\''):
      f.partial()(1, c=0)

  def test_specified_args(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    self.assertEqual(f(0).specified_args, set(['a']))
    self.assertEqual(f(0, 1).specified_args, set(['a', 'args']))
    self.assertEqual(f(0, 1, b=0).specified_args, set(['a', 'args', 'b']))
    self.assertEqual(f(0, 1, d=0).specified_args, set(['a', 'args', 'd']))

    x = f(0, 1, b=2, c=1, d=1)
    self.assertEqual(x.specified_args, set(['a', 'args', 'b', 'c', 'd']))

    # Delete an attribute removes the bound argument.
    del x.a
    self.assertEqual(x.specified_args, set(['args', 'b', 'c', 'd']))

  @unittest.skip('need to track the original value of FieldUpdate.new_value.')
  def test_specified_args_with_removing_args_with_default(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    x = f(0, 1, b=0, c=0, d=0)

    # Assign an attribute to MISSING_VALUE removes the bound argument.
    x.d = MISSING_VALUE
    self.assertEqual(x.specified_args, set(['a', 'args', 'c', 'd']))

    # Rebind an attribute to MISSING_VALUE also removes the bound argument.
    x.rebind(c=MISSING_VALUE)
    self.assertEqual(x.specified_args, set(['a', 'args', 'd']))

  def test_non_default_args(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    self.assertEqual(f(0).non_default_args, set(['a']))
    self.assertEqual(f(0, 1).non_default_args, set(['a', 'args']))
    self.assertEqual(f(0, b=0, c=1).non_default_args, set(['a']))
    self.assertEqual(f(0, b=1, c=0).non_default_args, set(['a', 'b', 'c']))
    self.assertEqual(f(0, d=2).non_default_args, set(['a', 'd']))

    x = f(0)
    self.assertEqual(x.non_default_args, set(['a']))
    x.b = 1
    self.assertEqual(x.non_default_args, set(['a', 'b']))
    x.rebind(c=0)
    self.assertEqual(x.non_default_args, set(['a', 'b', 'c']))

  def test_default_args(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    self.assertEqual(f(0).default_args, set(['args', 'b', 'c']))
    self.assertEqual(f(0, 1).default_args, set(['b', 'c']))
    self.assertEqual(f(0, b=0, c=1).default_args, set(['args', 'b', 'c']))
    self.assertEqual(f(0, b=1, c=0).default_args, set(['args']))
    self.assertEqual(f(0, b=1, c=0, d=2).default_args, set(['args']))

    x = f(0)
    self.assertEqual(x.default_args, set(['args', 'b', 'c']))
    x.args = [1]
    self.assertEqual(x.default_args, set(['b', 'c']))
    x.b = 1
    self.assertEqual(x.default_args, set(['c']))
    x.rebind(c=0)
    self.assertEqual(x.default_args, set())

  def test_bound_args(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    self.assertEqual(f.partial().bound_args, set(['args', 'b', 'c']))
    self.assertEqual(f(1).bound_args, set(['a', 'args', 'b', 'c']))
    self.assertEqual(f(1, 2).bound_args, set(['a', 'args', 'b', 'c']))
    self.assertEqual(f(1, d=2).bound_args, set(['a', 'args', 'b', 'c', 'd']))

    x = f(1, 1, b=1)
    self.assertEqual(x.bound_args, set(['a', 'args', 'b', 'c']))

    x.a = MISSING_VALUE
    self.assertEqual(x.bound_args, set(['args', 'b', 'c']))

    # Setting args to MISSING_VALUE will make it [], thus still considered
    # bound.
    x.args = MISSING_VALUE
    self.assertEqual(x.bound_args, set(['args', 'b', 'c']))

    # Restore x.b to its default value.
    x.rebind(b=MISSING_VALUE)
    self.assertEqual(x.b, 0)
    self.assertEqual(x.bound_args, set(['args', 'b', 'c']))

  def test_unbound_args(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    self.assertEqual(f.partial().unbound_args, set(['a']))
    self.assertEqual(f(1).unbound_args, set())
    self.assertEqual(f(1, 2).unbound_args, set())
    self.assertEqual(f(1, d=2).unbound_args, set())

    x = f(1, 1, b=1)
    self.assertEqual(x.unbound_args, set())

    x.a = MISSING_VALUE
    self.assertEqual(x.unbound_args, set(['a']))

    # Restore x.b to its default value.
    x.rebind(b=MISSING_VALUE)
    self.assertEqual(x.b, 0)
    self.assertEqual(x.unbound_args, set(['a']))

  def test_is_fully_bound(self):
    @pg_functor
    def f(a, *args, b=0, c=1, **kwargs):
      del a, args, b, c, kwargs

    self.assertFalse(f.partial().is_fully_bound)
    self.assertTrue(f(1).is_fully_bound)
    self.assertTrue(f(1, 0).is_fully_bound)
    self.assertTrue(f(1, 0, b=MISSING_VALUE).is_fully_bound)
    self.assertTrue(f(1, 0, c=MISSING_VALUE).is_fully_bound)
    self.assertTrue(f(1, 0, d=MISSING_VALUE).is_fully_bound)

    x = f(1, 2, 3, b=4, c=5, d=6)

    x.d = MISSING_VALUE
    self.assertTrue(x.is_fully_bound)

    del x.c
    self.assertTrue(x.is_fully_bound)

    x.rebind(args=MISSING_VALUE)
    self.assertEqual(x.args, [])
    self.assertTrue(x.is_fully_bound)

    x.a = MISSING_VALUE
    self.assertFalse(x.is_fully_bound)

  def test_custom_base_class(self):

    class MyFunctor(Functor):
      pass

    @pg_functor(base_class=MyFunctor)
    def my_fun():
      return 0

    self.assertIsInstance(my_fun(), MyFunctor)

  def test_non_default_values(self):
    @pg_functor([
        ('c', pg_typing.Dict([
            ('p', pg_typing.Int(default=1)),
            ('q', pg_typing.Bool(default=True))
        ]))
    ])
    def f(a, b=1, c=Dict(p=1, q=True)):
      del a, b, c

    x = f(1, 1, dict(p=False))
    self.assertEqual(x.non_default_values(), {'a': 1, 'c.p': False})
    self.assertEqual(
        x.non_default_values(flatten=False), {'a': 1, 'c': {'p': False}})

  def test_missing_values(self):
    @pg_functor
    def f(x, y, z=1):
      del x, y, z

    @pg_members([
        ('x', pg_typing.Int())
    ])
    class A(Object):
      pass

    x = f.partial(1)
    # Unbound argument `y` is not considered missing.
    self.assertEqual(x.missing_values(), {})

    # Bound argument `x` with partial object is considered missing.
    x = f.partial(x=A.partial())
    self.assertEqual(x.missing_values(), {'x.x': MISSING_VALUE})


if __name__ == '__main__':
  unittest.main()
