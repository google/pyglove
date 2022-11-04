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

  def test_set_get(self):
    k, v = 'x', 1
    thread_local.set_value(k, v)
    self.assertEqual(thread_local.get_value(k), v)
    self.assertIsNone(thread_local.get_value('y', None))
    with self.assertRaisesRegex(
        ValueError, 'Key .* does not exist in thread-local storage'):
      thread_local.get_value('abc')

    # Test thread locality.
    def thread_fun(i):
      def _fn():
        thread_local.set_value('x', i)
        self.assertEqual(thread_local.get_value('x'), i)
      return _fn
    self.assert_thread_func([thread_fun(i) for i in range(5)], 2)

  def test_value_scope(self):
    with thread_local.value_scope('y', 1, None):
      self.assertEqual(thread_local.get_value('y'), 1)
    self.assertIsNone(thread_local.get_value('y'))

    # Test thread locality.
    def thread_fun(i):
      def _fn():
        with thread_local.value_scope('y', i, None):
          self.assertEqual(thread_local.get_value('y'), i)
        self.assertIsNone(thread_local.get_value('y'))
      return _fn
    self.assert_thread_func([thread_fun(i) for i in range(5)], 2)


if __name__ == '__main__':
  unittest.main()
