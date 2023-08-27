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
"""Tests for pyglove.compounding."""

import abc
import dataclasses
import sys
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.compounding import compound as pg_compound
from pyglove.core.symbolic.compounding import compound_class as pg_compound_class
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.inferred import ValueFromParentChain
from pyglove.core.symbolic.object import Object


class BasicTest(unittest.TestCase):

  def test_basics(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: int

    @pg_compound
    def foo(x) -> Foo:
      return Foo(x + 1, x + 1)

    self.assertIsInstance(foo(1), Foo)

    # The compound attribute takes precedence over the decomposed one.
    self.assertEqual(foo(1).x, 1)

    # However, we can use the decomposed to get the real object's value.
    self.assertEqual(foo(1).decomposed.x, 2)

    self.assertEqual(foo(1).y, 2)
    self.assertEqual(foo(1), foo(1))
    self.assertNotEqual(foo(1), foo(1).decomposed)
    self.assertEqual(foo(1).decomposed, foo(1).decomposed)

    # Bad cases.
    with self.assertRaisesRegex(
        TypeError, 'Decorator `compound` is only applicable to functions.'
    ):
      pg_compound(Foo)(Foo)

    with self.assertRaisesRegex(
        ValueError,
        'Cannot inference the base class from return value annotation',
    ):

      @pg_compound
      def bar(unused_x):
        pass

  def test_lazy_build(self):
    count = dict(x=0)

    class Foo:
      def __init__(self, y):
        self.y = y
        self.z = y
        count['x'] += 1

    @pg_compound
    def foo(x) -> Foo:
      return Foo(x)

    f = foo(1)
    self.assertEqual(count['x'], 0)
    _ = f.y
    self.assertEqual(count['x'], 1)
    _ = f.z
    self.assertEqual(count['x'], 1)

    @pg_compound(lazy_build=False)
    def bar(x) -> Foo:
      return Foo(x)

    b = bar(1)
    self.assertEqual(count['x'], 2)
    _ = b.y
    self.assertEqual(count['x'], 2)
    _ = b.z
    self.assertEqual(count['x'], 2)


class UserClassTest(unittest.TestCase):

  def test_non_symbolic_user_class(self):
    class A:

      def __init__(self, x):
        self.x = x

      def value(self):
        return self.x

    def factory(y) -> A:
      return A(y + 1)

    CompoundA = pg_compound_class(factory)  # pylint: disable=invalid-name
    ca = CompoundA(1)

    self.assertIsInstance(ca, A)
    self.assertEqual(ca.y, 1)
    self.assertEqual(ca.x, 2)
    self.assertEqual(ca.value(), 2)
    self.assertIs(type(ca.decomposed), A)

  def test_symbolic_user_class(self):
    class A(Object):
      x: int = 1

      def value(self):
        return self.x

    def factory(y) -> A:
      return A(y + 1)

    CompoundA = pg_compound_class(factory)  # pylint: disable=invalid-name
    ca = CompoundA(1)

    self.assertIsInstance(ca, A)
    self.assertEqual(ca.y, 1)
    self.assertEqual(ca.x, 2)
    self.assertEqual(ca.value(), 2)
    self.assertIs(type(ca.decomposed), A)

  def test_user_class_with_side_effect_init(self):
    class A(Object):
      x: int

      @object_utils.explicit_method_override
      def __init__(self, x):
        super().__init__(x=x)
        assert type(self) is A  # pylint: disable=unidiomatic-typecheck

      def value(self):
        return self.x

    def factory(y) -> A:
      return A(y + 1)

    CompoundA = pg_compound_class(factory)  # pylint: disable=invalid-name
    ca = CompoundA(1)

    self.assertIsInstance(ca, A)
    self.assertEqual(ca.y, 1)
    self.assertEqual(ca.x, 2)
    self.assertEqual(ca.value(), 2)
    self.assertIs(type(ca.decomposed), A)

  def test_user_class_with_side_effect_init_subclass(self):
    class A:

      def __init_subclass__(cls):
        super().__init_subclass__()
        assert type(cls) is A  # pylint: disable=unidiomatic-typecheck

      def foo(self):
        return 'bar'

    CompoundA = pg_compound_class(lambda: A(), A)  # pylint: disable=invalid-name, unnecessary-lambda
    self.assertTrue(issubclass(CompoundA, A))
    self.assertEqual(CompoundA().foo(), 'bar')

  def test_on_bound_side_effect_free(self):
    class Foo(Object):
      x: int

      def _on_bound(self):
        # Side effect.
        super()._on_bound()
        assert type(self) is Foo  # pylint: disable=unidiomatic-typecheck

      def hello(self):
        return self.x

    @pg_compound(Foo)
    def foo(x):
      return Foo(x)

    # This does not trigger assertion.
    self.assertEqual(foo(1).hello(), 1)

  @unittest.skipIf(
      sys.version_info < (3, 10),
      'This feature is only supported on Python 3.10 and above.',
  )
  def test_use_abstract_base(self):
    class Foo(metaclass=abc.ABCMeta):

      @abc.abstractmethod
      def foo(self, x):
        pass

      @property
      @abc.abstractmethod
      def bar(self):
        pass

    class Bar(Foo):

      def foo(self, x):
        return x

      @property
      def bar(self):
        return 1

    @pg_compound(Foo)
    def bar():
      return Bar()

    b = bar()
    self.assertEqual(b.bar, 1)
    self.assertEqual(b.foo(1), 1)

  def test_inferred_value_access(self):
    class Foo(Object):
      x: int = ValueFromParentChain()

    @pg_compound(Foo)
    def foo():
      return Foo()

    f = foo()
    d = Dict(x=1, y=f)
    self.assertEqual(f.x, 1)
    d.y = None
    self.assertIsNone(f.sym_parent)

    with self.assertRaisesRegex(
        AttributeError, '.* is not found under its context'
    ):
      _ = f.x

    _ = Dict(x=2, y=f)
    self.assertEqual(f.x, 2)


class TypingTest(unittest.TestCase):

  def test_no_typing(self):
    @dataclasses.dataclass
    class Foo:
      x: int

    @pg_compound(Foo)
    def foo(y):
      return Foo(y)

    self.assertEqual(
        foo.__schema__.get_field('y'), pg_typing.Field('y', pg_typing.Any())
    )

  def test_auto_typing(self):
    @dataclasses.dataclass
    class Foo:
      x: int

    @pg_compound
    def foo(y: int) -> Foo:
      return Foo(y + 1)

    self.assertEqual(
        foo.__schema__.get_field('y'), pg_typing.Field('y', pg_typing.Int())
    )

  def test_auto_typing_with_docstr(self):
    @dataclasses.dataclass
    class Foo:
      x: int

    @pg_compound
    def foo(y: int) -> Foo:
      """Compound foo.

      Args:
        y: field y.

      Returns:
        the compound foo.
      """
      return Foo(y + 1)

    self.assertEqual(
        foo.__schema__.get_field('y'),
        pg_typing.Field('y', pg_typing.Int(), 'field y.'),
    )

  def test_manual_typing(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: int

    @pg_compound(Foo, [('v', pg_typing.Int(), 'field v.')])
    def foo(v):
      return Foo(v + 1, v + 2)

    self.assertEqual(
        foo.__schema__.get_field('v'),
        pg_typing.Field('v', pg_typing.Int(), 'field v.'),
    )


class SymbolicTraitsTest(unittest.TestCase):

  def test_rebind(self):
    @dataclasses.dataclass
    class A:
      x: int
      y: int

    @pg_compound
    def a_with_equal_pairs(v: int) -> A:
      return A(v, v)

    a = a_with_equal_pairs(1)
    self.assertEqual(a.v, 1)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 1)

    a.rebind(v=2)
    self.assertEqual(a.v, 2)
    self.assertEqual(a.x, 2)
    self.assertEqual(a.y, 2)

  def test_runtime_typing(self):
    @dataclasses.dataclass
    class Foo:
      x: int
      y: int

    @pg_compound(Foo, [('v', pg_typing.Int(min_value=0), 'field v.')])
    def foo(v):
      return Foo(v + 1, v + 2)

    _ = foo(1)
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      foo(1.0)

    _ = foo(1)
    with self.assertRaisesRegex(ValueError, 'Value .* is out of range'):
      foo(-1)


if __name__ == '__main__':
  unittest.main()
