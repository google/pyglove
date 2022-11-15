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
"""Tests for pyglove.detouring.class_detour."""

import threading
import unittest
from pyglove.core.detouring import class_detour


class ClassDetourTest(unittest.TestCase):
  """Tests for `pg.detour` on classes."""

  def test_basics(self):

    class A:
      pass

    class B:
      pass

    class C:
      pass

    def foo(unused_cls, unused_value):
      return A()

    with class_detour.detour([(A, B), (B, A), (C, foo)]) as mappings:
      self.assertEqual(mappings, dict([
          (A, B),
          (B, A),
          (C, foo)
      ]))
      self.assertIsInstance(A(), B)
      self.assertIsInstance(B(), A)
      self.assertIsInstance(C(1), B)

    with self.assertRaisesRegex(TypeError, 'Detour source .* is not a class'):
      with class_detour.detour([(foo, A)]):
        pass

    with self.assertRaisesRegex(
        TypeError, 'Detour destination .* is not a class or a function'):
      with class_detour.detour([(A, 1)]):
        pass

  def test_nested_detour(self):

    class A:
      pass

    class B:
      pass

    class C:
      pass

    with class_detour.detour([(A, B), (C, A)]) as mappings1:
      self.assertEqual(mappings1, dict([
          (A, B),
          (C, A)
      ]))
      self.assertEqual(class_detour.current_mappings(), mappings1)
      self.assertIsInstance(A(), B)
      self.assertIsInstance(B(), B)
      self.assertIsInstance(C(), A)
      with class_detour.detour([(A, C), (B, C)]) as mappings2:
        self.assertEqual(mappings2, dict([
            (A, B),
            (B, A),
            (C, A),
        ]))
        self.assertEqual(class_detour.current_mappings(), mappings2)
        self.assertIsInstance(A(), B)
        self.assertIsInstance(B(), A)
        self.assertIsInstance(C(), A)

  def test_detour_with_subclassing(self):

    class A:

      def __init__(self, value):
        self.value = value

    class B:

      def __init__(self, value):
        self.value = value

    class A1(A):

      def __init__(self, value):
        super().__init__(value)
        self.another_value = value ** 2

    # Detouring parent class to subclass.
    with class_detour.detour([(A, A1)]):
      a = A(2)
      self.assertIsInstance(a, A1)
      self.assertEqual(a.another_value, 4)

    # Detouring parent class does not detour subclass.
    with class_detour.detour([(A, B)]):
      self.assertIsInstance(A(1), B)
      self.assertIsInstance(A1(1), A1)

    # Detouring subclass does not detour parent class.
    with class_detour.detour([(A1, B)]):
      self.assertIsInstance(A1(1), B)
      self.assertIsInstance(A(1), A)

  def test_custom_new(self):

    class A:
      pass

    class B(A):

      def __new__(cls):
        del cls
        return 1

    with class_detour.detour([(A, B)]):
      self.assertEqual(A(), 1)

  def test_custom_new_with_multi_inheritance(self):
    invoked_classes = []

    class A:

      def __new__(cls, *args, **kwargs):
        invoked_classes.append(A)
        return super(A, cls).__new__(cls, *args, **kwargs)

    class A1(A):
      pass

    class B:

      def __new__(cls, *args, **kwargs):
        invoked_classes.append(B)
        return super(B, cls).__new__(cls, *args, **kwargs)

    class B1(B):
      pass

    class C(A1, B1):
      pass

    class C1(C):
      pass

    with class_detour.detour([(C, C1)]):
      invoked_classes[:] = []
      self.assertIsInstance(C(), C1)
      self.assertEqual(invoked_classes, [A, B])

    with class_detour.detour([(A, B), (B, A)]):
      invoked_classes[:] = []
      C()
      self.assertEqual(invoked_classes, [A, B])

  def test_thread_safety(self):

    class A:
      pass

    class B:
      pass

    def t1_main():
      with class_detour.detour([(A, B)]):
        for _ in range(10000):
          self.assertIsInstance(A(), B)

    def t2_main():
      for _ in range(10000):
        self.assertIsInstance(A(), A)

    t1 = threading.Thread(target=t1_main)
    t2 = threading.Thread(target=t2_main)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

  def test_undetoured_new(self):

    class A:

      def __init__(self, x):
        self.x = x

    class B:

      def __init__(self, x):
        self.x = x ** 2

    class C:

      def __new__(cls, x):
        if x > 0:
          return A(x)
        return object.__new__(C)

      def __init__(self, x):
        self.x = x

    a = class_detour.undetoured_new(A, 2)
    self.assertIsInstance(a, A)
    self.assertEqual(a.x, 2)

    with class_detour.detour([(A, B)]):
      a = class_detour.undetoured_new(A, 2)
      self.assertIsInstance(a, A)
      self.assertEqual(a.x, 2)

    with class_detour.detour([(C, B)]):
      c1 = class_detour.undetoured_new(C, 0)
      self.assertIsInstance(c1, C)
      self.assertEqual(c1.x, 0)

      c2 = class_detour.undetoured_new(C, 2)
      self.assertIsInstance(c2, A)
      self.assertEqual(c2.x, 2)


if __name__ == '__main__':
  unittest.main()
