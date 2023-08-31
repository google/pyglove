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
"""Tests for pyglove.object_utils.thread_local."""

import threading
import time
import unittest

from pyglove.core.object_utils import thread_local


class ThreadLocalTest(unittest.TestCase):
  """Tests for `pg.symbolic.thread_local`."""

  def assert_thread_func(self, funcs, period_in_second=1):
    has_errors = [True] * len(funcs)
    def repeat_for_period(func, i):
      def _fn():
        begin = time.time()
        while True:
          func()
          if time.time() - begin > period_in_second:
            break
        has_errors[i] = False
      return _fn

    threads = [threading.Thread(target=repeat_for_period(f, i))
               for i, f in enumerate(funcs)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    self.assertFalse(any(has_error for has_error in has_errors))

  def test_set_get_has_delete(self):
    k, v = 'x', 1
    self.assertFalse(thread_local.thread_local_has(k))
    thread_local.thread_local_set(k, v)
    self.assertTrue(thread_local.thread_local_has(k))
    self.assertEqual(thread_local.thread_local_get(k), v)
    thread_local.thread_local_del(k)
    self.assertFalse(thread_local.thread_local_has(k))

    self.assertFalse(thread_local.thread_local_has('y'))
    with self.assertRaisesRegex(
        ValueError, 'Key .* does not exist in thread-local storage'):
      thread_local.thread_local_get('abc')

    # Test thread locality.
    def thread_fun(i):
      def _fn():
        self.assertFalse(thread_local.thread_local_has('x'))
        thread_local.thread_local_set('x', i)
        self.assertTrue(thread_local.thread_local_has('x'))
        self.assertEqual(thread_local.thread_local_get('x'), i)
        thread_local.thread_local_del('x')
        self.assertFalse(thread_local.thread_local_has('x'))
      return _fn
    self.assert_thread_func([thread_fun(i) for i in range(5)], 2)

  def test_thread_local_value_scope(self):
    thread_local.thread_local_set('y', 2)
    with thread_local.thread_local_value_scope('y', 1, None):
      self.assertEqual(thread_local.thread_local_get('y'), 1)
    self.assertEqual(thread_local.thread_local_get('y'), 2)

    # Test thread locality.
    def thread_fun(i):
      def _fn():
        with thread_local.thread_local_value_scope('y', i, None):
          self.assertEqual(thread_local.thread_local_get('y'), i)
        self.assertFalse(thread_local.thread_local_has('y'))
      return _fn
    self.assert_thread_func([thread_fun(i) for i in range(5)], 2)

  def test_thread_local_increment_decrement(self):
    k = 'z'
    self.assertEqual(thread_local.thread_local_increment(k, 5), 6)
    self.assertEqual(thread_local.thread_local_increment(k), 7)
    self.assertEqual(thread_local.thread_local_decrement(k), 6)
    thread_local.thread_local_del(k)
    self.assertEqual(thread_local.thread_local_increment(k), 1)
    thread_local.thread_local_del(k)

    # Test thread locality.
    def thread_fun(_):
      def _fn():
        self.assertEqual(thread_local.thread_local_increment(k), 1)
        self.assertEqual(thread_local.thread_local_increment(k), 2)
        self.assertEqual(thread_local.thread_local_increment(k), 3)
        thread_local.thread_local_del(k)
      return _fn
    self.assert_thread_func([thread_fun(i) for i in range(5)], 2)

  def test_thread_local_push_pop(self):
    k = 'p'
    self.assertFalse(thread_local.thread_local_has(k))
    thread_local.thread_local_push(k, 1)
    self.assertEqual(thread_local.thread_local_get(k), [1])
    thread_local.thread_local_push(k, 2)
    self.assertEqual(thread_local.thread_local_get(k), [1, 2])
    self.assertEqual(thread_local.thread_local_pop(k), 2)
    self.assertEqual(thread_local.thread_local_get(k), [1])
    self.assertEqual(thread_local.thread_local_pop(k), 1)
    with self.assertRaisesRegex(IndexError, 'pop from empty list'):
      thread_local.thread_local_pop(k)
    self.assertEqual(thread_local.thread_local_pop(k, -1), -1)
    with self.assertRaisesRegex(
        ValueError, 'Key .* does not exist in thread-local storage'):
      thread_local.thread_local_pop('unknown_key')
    self.assertEqual(
        thread_local.thread_local_pop('unknown_key', 0), 0)

    thread_local.thread_local_set('q', 1)
    with self.assertRaisesRegex(
        TypeError, 'Key .* from thread-local storage is not a list'):
      thread_local.thread_local_pop('q')

    # Test thread locality.
    def thread_fun(i):
      def _fn():
        thread_local.thread_local_push(k, i)
        thread_local.thread_local_push(k, i + 1)
        self.assertEqual(thread_local.thread_local_pop(k), i + 1)
        self.assertEqual(thread_local.thread_local_pop(k), i)
        self.assertEqual(thread_local.thread_local_get(k), [])
      return _fn
    self.assert_thread_func([thread_fun(i) for i in range(5)], 2)


if __name__ == '__main__':
  unittest.main()
