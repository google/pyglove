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
"""Tests for pg.Ref."""

import copy
import inspect
import pickle
from typing import Any
import unittest

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import ref
from pyglove.core.symbolic.base import contains
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.object import Object


class A(Object):
  x: int


class RefTest(unittest.TestCase):

  def test_basics(self):

    a = A(1)
    r = ref.Ref(a)
    self.assertIs(r.value, a)
    self.assertIs(r.infer(), a)
    self.assertEqual(ref.Ref(a), r)
    b = copy.copy(a)
    self.assertNotEqual(ref.Ref(b), r)

  def test_new(self):
    self.assertIsInstance(ref.Ref(1), int)
    self.assertIsInstance(ref.Ref(1.0), float)
    self.assertIsInstance(ref.Ref(True), bool)
    self.assertIsInstance(ref.Ref('abc'), str)

    r = ref.Ref(A(1))
    self.assertIsInstance(r, ref.Ref)
    self.assertIsInstance(r.infer(), A)
    self.assertEqual(ref.Ref(r), r)

  def test_type_check(self):

    class B:
      pass

    r = ref.Ref(A(1))
    self.assertIs(pg_typing.Object(A).apply(r), r)
    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'
    ):
      pg_typing.Object(A).apply(ref.Ref(B()))

  def test_partial_check(self):
    r = ref.Ref(A.partial())
    self.assertIs(pg_typing.Object(A).apply(r, allow_partial=True), r)
    with self.assertRaisesRegex(
        ValueError, '.* is not fully bound'
    ):
      pg_typing.Object(A).apply(r)

  def test_clone_basics(self):
    r = ref.Ref(A(1))
    self.assertIs(r.clone().infer(), r.infer())
    self.assertIs(r.clone(deep=True).infer(), r.infer())

    self.assertIs(copy.copy(r).infer(), r.infer())
    self.assertIs(copy.deepcopy(r).infer(), r.infer())

  def test_self_reference(self):

    class B(Object):
      x: Any
      y: Any

    b = B.partial(x=1)
    with self.assertRaisesRegex(
        NotImplementedError, 'Self-referential object is not supported.'
    ):
      b.rebind(y=ref.Ref(b))

  def test_repr(self):
    self.assertEqual(repr(ref.Ref(A(1))), 'Ref(A(x=1))')

  def test_str(self):
    self.assertEqual(
        str(ref.Ref(A(1))),
        inspect.cleandoc("""
        Ref(
          value = A(
            x = 1
          )
        )
        """))

  def test_to_json(self):
    with self.assertRaisesRegex(
        TypeError, '.* cannot be serialized at the moment'):
      ref.Ref(A(1)).to_json()

  def test_pickle(self):
    with self.assertRaisesRegex(
        TypeError, '.* cannot be pickled at the moment'):
      pickle.dumps(ref.Ref(A(1)))

  def test_maybe_ref(self):
    a = ref.maybe_ref(A(1))
    self.assertIsInstance(a, A)
    self.assertNotIsInstance(a, ref.Ref)

    b = Dict(x=a)
    x = ref.maybe_ref(b.x)
    self.assertIsInstance(x, ref.Ref)
    self.assertIs(x.value, a)

  def test_deref(self):
    class Foo(Object):
      x: int
      y: Any

    # Deref on non-ref value.
    a = Foo(1, 2)
    self.assertIs(ref.deref(a), a)
    self.assertEqual(ref.deref(a), Foo(1, 2))

    # Deref top-level Ref.
    a = ref.Ref(Foo(1, 2))
    self.assertIs(ref.deref(a), a.value)

    a = ref.Ref(Foo(1, ref.Ref(Foo(2, ref.Ref(Foo(3, 4))))))
    self.assertTrue(contains(a, type=ref.Ref))
    a_prime = ref.deref(a)
    self.assertIs(a_prime, a.value)
    self.assertIsInstance(a_prime.sym_getattr('y'), ref.Ref)
    self.assertIsInstance(a_prime.y.sym_getattr('y'), ref.Ref)

    # Deref entire tree.
    a = ref.Ref(Foo(1, ref.Ref(Foo(2, ref.Ref(Foo(3, 4))))))
    self.assertTrue(contains(a, type=ref.Ref))
    a_prime = ref.deref(a, recursive=True)
    self.assertIs(a_prime, a.value)
    self.assertFalse(contains(a_prime, type=ref.Ref))


if __name__ == '__main__':
  unittest.main()
