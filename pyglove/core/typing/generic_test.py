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

from typing import Any, Generic, TypeVar
import unittest

from pyglove.core.typing import generic

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


class GenericTest(unittest.TestCase):

  def test_issubclass(self):
    # Any.
    self.assertTrue(generic.is_subclass(int, Any))
    self.assertFalse(generic.is_subclass(Any, int))

    # Non-generic vs. non-generic.
    self.assertTrue(generic.is_subclass(int, object))
    self.assertTrue(generic.is_subclass(int, int))
    self.assertTrue(generic.is_subclass(Str, str))
    self.assertFalse(generic.is_subclass(str, Str))
    self.assertTrue(generic.is_subclass(B1, B))
    self.assertFalse(generic.is_subclass(B, B1))
    self.assertTrue(generic.is_subclass(B1, Generic))
    self.assertTrue(generic.is_subclass(C, C))
    self.assertTrue(generic.is_subclass(D, C))
    self.assertFalse(generic.is_subclass(C, D))

    # Non-generic vs. generic.
    self.assertTrue(generic.is_subclass(C, A[str, int]))
    self.assertFalse(generic.is_subclass(C, A[int, int]))
    self.assertFalse(generic.is_subclass(Str, A[int, int]))

    self.assertTrue(generic.is_subclass(C, B[Str]))
    self.assertTrue(generic.is_subclass(C, B[str]))
    self.assertFalse(generic.is_subclass(C, B[int]))
    self.assertTrue(generic.is_subclass(D, B[Str]))

    # B1 is a subclass of B without type args.
    self.assertFalse(generic.is_subclass(B1, B[Any]))

    # Non-generic vs. generic type.
    self.assertTrue(generic.is_subclass(C, A[str, int]))
    self.assertTrue(generic.is_subclass(C, A[Any, Any]))
    self.assertTrue(generic.is_subclass(C, B[Str]))
    self.assertTrue(generic.is_subclass(C, B[str]))

    # Generic vs. non-generic type:
    self.assertTrue(generic.is_subclass(B[Str], B))
    self.assertTrue(generic.is_subclass(B1, B))
    self.assertTrue(generic.is_subclass(B1[Any], B))

    # Generic type vs. generic type.
    self.assertTrue(generic.is_subclass(B[Str], B[str]))
    self.assertTrue(generic.is_subclass(B[str], B[str]))
    self.assertFalse(generic.is_subclass(B1[Str], B[str]))
    self.assertFalse(generic.is_subclass(B1[Str], A[str, int]))

    # Test tuple cases.
    self.assertTrue(generic.is_subclass(int, (str, int)))
    self.assertTrue(generic.is_subclass(C, (int, A[str, int])))

  def test_isinstance(self):
    self.assertTrue(generic.is_instance('abc', str))
    self.assertTrue(generic.is_instance('abc', Any))
    self.assertTrue(generic.is_instance('abc', (int, str)))

    self.assertTrue(generic.is_instance(D(), Any))
    self.assertTrue(generic.is_instance(D(), A[str, int]))
    self.assertTrue(generic.is_instance(D(), B[str]))

  def test_is_generic(self):
    self.assertFalse(generic.is_generic(str))
    self.assertFalse(generic.is_generic(Any))
    self.assertFalse(generic.is_generic(A))
    self.assertTrue(generic.is_generic(A[str, int]))

  def test_has_generic_bases(self):
    self.assertFalse(generic.has_generic_bases(str))
    self.assertFalse(generic.has_generic_bases(Any))
    self.assertTrue(generic.has_generic_bases(A))
    self.assertTrue(generic.has_generic_bases(C))

  def test_get_type(self):
    self.assertIs(generic.get_type(str), str)
    self.assertIs(generic.get_type(A), A)
    self.assertIs(generic.get_type(A[str, int]), A)
    with self.assertRaisesRegex(TypeError, '.* is not a type.'):
      generic.get_type(Any)

  def test_get_type_args(self):
    self.assertEqual(generic.get_type_args(str), ())
    self.assertEqual(generic.get_type_args(A), ())
    self.assertEqual(generic.get_type_args(A[str, int]), (str, int))
    self.assertEqual(generic.get_type_args(B1, A), ())
    self.assertEqual(generic.get_type_args(B1[str], A), ())
    self.assertEqual(generic.get_type_args(C), ())
    self.assertEqual(generic.get_type_args(C, A), (str, int))
    self.assertEqual(generic.get_type_args(C, B), (Str,))


if __name__ == '__main__':
  unittest.main()
