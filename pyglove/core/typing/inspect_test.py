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
"""Tests for generic type utility."""

from typing import Any, Generic, Protocol, TypeVar
import unittest

from pyglove.core.typing import callable_signature
from pyglove.core.typing import inspect


XType = TypeVar('XType')
YType = TypeVar('YType')


class Str(str):
  pass


class A(Generic[XType, YType]):
  pass


class B(Generic[XType]):
  pass


class B1(B, Generic[YType]):
  pass


class B2(B1[int]):
  pass


class C(A[str, int], B[Str]):
  pass


class D(C):
  pass


class AA:
  pass


class AA1(AA):
  class BB1:
    class CC1:
      pass


class E(Protocol):

  def __call__(self, x: int) -> int:
    pass


class E1(E):
  pass


class E2(E, Protocol):

  def bar(self, x: int) -> int:
    pass


class F(Protocol[XType]):
  x: XType


class InspectTest(unittest.TestCase):

  def test_issubclass(self):
    # Any.
    self.assertTrue(inspect.is_subclass(int, Any))
    self.assertFalse(inspect.is_subclass(Any, int))

    # Non-generic vs. non-inspect.
    self.assertTrue(inspect.is_subclass(int, object))
    self.assertTrue(inspect.is_subclass(int, int))
    self.assertTrue(inspect.is_subclass(Str, str))
    self.assertFalse(inspect.is_subclass(str, Str))
    self.assertTrue(inspect.is_subclass(B1, B))
    self.assertFalse(inspect.is_subclass(B, B1))
    self.assertTrue(inspect.is_subclass(B1, Generic))
    self.assertTrue(inspect.is_subclass(C, C))
    self.assertTrue(inspect.is_subclass(D, C))
    self.assertFalse(inspect.is_subclass(C, D))

    # Non-generic vs. inspect.
    self.assertTrue(inspect.is_subclass(C, A[str, int]))
    self.assertFalse(inspect.is_subclass(C, A[int, int]))
    self.assertFalse(inspect.is_subclass(Str, A[int, int]))

    self.assertTrue(inspect.is_subclass(C, B[Str]))
    self.assertTrue(inspect.is_subclass(C, B[str]))
    self.assertFalse(inspect.is_subclass(C, B[int]))
    self.assertTrue(inspect.is_subclass(D, B[Str]))

    # B1 is a subclass of B without type args.
    self.assertFalse(inspect.is_subclass(B1, B[Any]))

    # Non-generic vs. generic type.
    self.assertTrue(inspect.is_subclass(C, A[str, int]))
    self.assertTrue(inspect.is_subclass(C, A[Any, Any]))
    self.assertTrue(inspect.is_subclass(C, B[Str]))
    self.assertTrue(inspect.is_subclass(C, B[str]))

    # Generic vs. non-generic type:
    self.assertTrue(inspect.is_subclass(B[Str], B))
    self.assertTrue(inspect.is_subclass(B1, B))
    self.assertTrue(inspect.is_subclass(B1[Any], B))

    # Generic type vs. generic type.
    self.assertTrue(inspect.is_subclass(B[Str], B[str]))
    self.assertTrue(inspect.is_subclass(B[str], B[str]))
    self.assertFalse(inspect.is_subclass(B1[Str], B[str]))
    self.assertFalse(inspect.is_subclass(B1[Str], A[str, int]))

    # Protocol check.
    self.assertTrue(inspect.is_subclass(E, E))
    self.assertTrue(inspect.is_subclass(E, Protocol))
    # We do not really check protocol comformance at runtime for performance
    # reasons. So all user classes are considered as subclasses of a Protocol
    # class.
    self.assertTrue(inspect.is_subclass(A, E))
    self.assertFalse(inspect.is_subclass(str, E))
    self.assertFalse(inspect.is_subclass(1, E))

    # Test tuple cases.
    self.assertTrue(inspect.is_subclass(int, (str, int)))
    self.assertTrue(inspect.is_subclass(C, (int, A[str, int])))

  def test_isinstance(self):
    self.assertTrue(inspect.is_instance('abc', str))
    self.assertTrue(inspect.is_instance('abc', Any))
    self.assertTrue(inspect.is_instance('abc', (int, str)))

    self.assertTrue(inspect.is_instance(D(), Any))
    self.assertTrue(inspect.is_instance(D(), A[str, int]))
    self.assertTrue(inspect.is_instance(D(), B[str]))

  def test_is_protocol(self):
    self.assertFalse(inspect.is_protocol(1))
    self.assertFalse(inspect.is_protocol(str))
    self.assertFalse(inspect.is_protocol(Any))
    self.assertFalse(inspect.is_protocol(A))
    self.assertFalse(inspect.is_protocol(A[str, int]))
    self.assertFalse(inspect.is_protocol(Protocol))
    self.assertFalse(inspect.is_protocol(Protocol[XType]))
    self.assertTrue(inspect.is_protocol(E))
    # subclasses of a Protocol class will downgrade if it's not explicitly
    # inherited from Protocol.
    # See https://typing.readthedocs.io/en/latest/spec/protocol.html
    self.assertFalse(inspect.is_protocol(E1))
    self.assertTrue(inspect.is_protocol(E2))
    self.assertTrue(inspect.is_protocol(F))

  def test_is_generic(self):
    self.assertFalse(inspect.is_generic(1))
    self.assertFalse(inspect.is_generic(str))
    self.assertFalse(inspect.is_generic(Any))
    self.assertFalse(inspect.is_generic(A))
    self.assertTrue(inspect.is_generic(A[str, int]))

  def test_has_generic_bases(self):
    self.assertFalse(inspect.has_generic_bases(str))
    self.assertFalse(inspect.has_generic_bases(Any))
    self.assertTrue(inspect.has_generic_bases(A))
    self.assertTrue(inspect.has_generic_bases(C))

  def test_get_type(self):
    self.assertIs(inspect.get_type(str), str)
    self.assertIs(inspect.get_type(A), A)
    self.assertIs(inspect.get_type(A[str, int]), A)
    with self.assertRaisesRegex(TypeError, '.* is not a type.'):
      inspect.get_type(1)

  def test_get_type_args(self):
    self.assertEqual(inspect.get_type_args(str), ())
    self.assertEqual(inspect.get_type_args(A), ())
    self.assertEqual(inspect.get_type_args(A[str, int]), (str, int))
    self.assertEqual(inspect.get_type_args(B1, A), ())
    self.assertEqual(inspect.get_type_args(B1[str], A), ())
    self.assertEqual(inspect.get_type_args(C), ())
    self.assertEqual(inspect.get_type_args(C, A), (str, int))
    self.assertEqual(inspect.get_type_args(C, B), (Str,))

  def test_outer_class(self):
    class Foo:
      pass

    with self.assertRaisesRegex(ValueError, '.* locally defined class'):
      inspect.get_outer_class(Foo)

    self.assertIsNone(inspect.get_outer_class(AA))
    self.assertIs(inspect.get_outer_class(AA1.BB1), AA1)
    self.assertIs(inspect.get_outer_class(AA1.BB1, AA), AA1)
    self.assertIs(inspect.get_outer_class(AA1.BB1, A), None)
    self.assertIs(inspect.get_outer_class(AA1.BB1.CC1), AA1.BB1)
    self.assertIsNone(
        inspect.get_outer_class(AA1.BB1.CC1, base_cls=AA, immediate=True)
    )
    self.assertIs(inspect.get_outer_class(AA1.BB1.CC1, AA), AA1)
    self.assertIs(
        inspect.get_outer_class(callable_signature.Argument.Kind),
        callable_signature.Argument
    )

    class Bar:
      pass

    Bar.__qualname__ = 'NonExist.Bar'
    self.assertIsNone(inspect.get_outer_class(Bar))

  def test_callable_eq(self):
    def foo(unused_x):
      pass

    def bar(unused_x):
      pass

    def baz(x):
      return x + 1

    # Non-callables.
    self.assertTrue(inspect.callable_eq(1, 1))
    self.assertFalse(inspect.callable_eq(1, 2))

    # Noneables.
    self.assertTrue(inspect.callable_eq(None, None))
    self.assertFalse(inspect.callable_eq(None, print))
    self.assertFalse(inspect.callable_eq(print, None))

    # Builtins.
    self.assertTrue(inspect.callable_eq(print, print))

    # User defined functions.
    self.assertTrue(inspect.callable_eq(foo, foo))
    self.assertTrue(inspect.callable_eq(foo, bar))
    self.assertFalse(inspect.callable_eq(foo, baz))

    # Lambda function.
    self.assertTrue(inspect.callable_eq(lambda x: x, lambda x: x))
    self.assertTrue(inspect.callable_eq(lambda x: x, lambda y: y))
    self.assertFalse(inspect.callable_eq(lambda x: x, lambda x, y: x + y))

    # Class methods.
    class A:  # pylint: disable=redefined-outer-name

      @staticmethod
      def static_method():
        pass

      @classmethod
      def class_method(cls):
        pass

      def instance_method(self):
        pass

    class B(A):  # pylint: disable=redefined-outer-name
      pass

    class C(B):  # pylint: disable=redefined-outer-name

      @staticmethod
      def static_method():
        return 1

      @classmethod
      def class_method(cls):
        return cls.__name__

      def instance_method(self):
        return self.__class__.__name__

    a1, a2 = A(), A()
    b, c = B(), C()

    # Static method.
    self.assertTrue(inspect.callable_eq(A.static_method, A.static_method))
    self.assertTrue(inspect.callable_eq(a1.static_method, a2.static_method))
    self.assertTrue(inspect.callable_eq(A.static_method, B.static_method))
    self.assertTrue(inspect.callable_eq(a1.static_method, b.static_method))
    self.assertFalse(inspect.callable_eq(A.static_method, C.static_method))
    self.assertFalse(inspect.callable_eq(b.static_method, c.static_method))

    # Class method.
    self.assertTrue(inspect.callable_eq(A.class_method, A.class_method))
    self.assertTrue(inspect.callable_eq(a1.class_method, a2.class_method))
    self.assertFalse(inspect.callable_eq(A.class_method, B.class_method))
    self.assertFalse(inspect.callable_eq(a1.class_method, b.class_method))
    self.assertFalse(inspect.callable_eq(A.class_method, C.class_method))
    self.assertFalse(inspect.callable_eq(a1.class_method, c.class_method))

    # Instance method.
    self.assertTrue(inspect.callable_eq(A.instance_method, A.instance_method))
    self.assertTrue(inspect.callable_eq(a1.instance_method, a1.instance_method))
    self.assertFalse(
        inspect.callable_eq(a1.instance_method, a2.instance_method)
    )
    self.assertTrue(inspect.callable_eq(A.instance_method, B.instance_method))
    self.assertFalse(inspect.callable_eq(a1.instance_method, b.instance_method))
    self.assertFalse(inspect.callable_eq(A.instance_method, C.instance_method))
    self.assertFalse(inspect.callable_eq(a1.instance_method, c.instance_method))


if __name__ == '__main__':
  unittest.main()
