# Copyright 2024 The PyGlove Authors
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

import time
import unittest

from pyglove.core.object_utils import profiling


class TimeItTest(unittest.TestCase):
  """Tests for `pg.symbolic.thread_local`."""

  def test_basics(self):
    tc = profiling.TimeIt('node')
    self.assertFalse(tc.has_started)
    self.assertEqual(tc.elapse, 0)

    tc.start()
    self.assertTrue(tc.has_started)
    self.assertGreater(tc.elapse, 0)

    self.assertTrue(tc.end())
    self.assertTrue(tc.has_ended)
    elapse1 = tc.elapse
    self.assertFalse(tc.end())
    self.assertEqual(tc.elapse, elapse1)

  def test_timeit(self):
    with profiling.timeit('node') as t:
      self.assertEqual(t.name, 'node')
      self.assertIsNotNone(t.start_time)
      self.assertTrue(t.has_started)
      self.assertIsNone(t.end_time)
      self.assertFalse(t.has_ended)
      time.sleep(0.5)
      elapse1 = t.elapse
      self.assertEqual(t.children, [])
      with profiling.timeit('child') as t1:
        time.sleep(0.5)
        with profiling.timeit('grandchild') as t2:
          time.sleep(0.5)

        self.assertEqual(t.children, [t1])
        self.assertEqual(t1.children, [t2])

        r = t.status()
        self.assertTrue(r['node'].has_started)
        self.assertGreater(r['node'].elapse, 0)
        self.assertFalse(r['node'].has_ended)
        self.assertFalse(r['node'].has_error)
        self.assertFalse(r['node.child'].has_ended)
        self.assertTrue(r['node.child.grandchild'].has_ended)

        with self.assertRaisesRegex(ValueError, '.* already exists'):
          with profiling.timeit('grandchild'):
            pass

    elapse2 = t.elapse
    self.assertTrue(t.has_ended)
    self.assertGreater(elapse2, elapse1)
    time.sleep(0.5)
    self.assertEqual(elapse2, t.elapse)

    statuss = t.status()
    self.assertEqual(
        list(statuss.keys()),
        ['node', 'node.child', 'node.child.grandchild']
    )
    self.assertEqual(
        sorted([v.elapse for v in statuss.values()], reverse=True),
        [v.elapse for v in statuss.values()],
    )
    self.assertTrue(all(v.has_ended for v in statuss.values()))
    self.assertFalse(any(v.has_error for v in statuss.values()))

  def test_timeit_with_error(self):
    with self.assertRaises(ValueError):
      with profiling.timeit('node') as t:
        with profiling.timeit('child') as t1:
          with profiling.timeit('grandchild') as t2:
            raise ValueError('error')

    r = t.status()
    self.assertTrue(r['node'].has_error)
    self.assertTrue(t.has_error)
    self.assertIsInstance(t.error, ValueError)
    self.assertIsInstance(r['node'].error, ValueError)
    self.assertTrue(r['node.child'].has_error)
    self.assertTrue(t1.has_error)
    self.assertTrue(r['node.child.grandchild'].has_error)
    self.assertTrue(t2.has_error)

  def test_timeit_summary(self):
    summary = profiling.TimeIt.StatusSummary()
    for i in range(10):
      with profiling.timeit('node') as t:
        time.sleep(0.1)
        with profiling.timeit('child'):
          time.sleep(0.1)
          try:
            with profiling.timeit('grandchild'):
              time.sleep(0.1)
              if i < 2:
                raise ValueError('error')
          except ValueError:
            pass
      summary.aggregate(t)
    self.assertEqual(
        list(summary.breakdown.keys()),
        ['node', 'node.child', 'node.child.grandchild']
    )
    self.assertEqual(
        [x.num_started for x in summary.breakdown.values()],
        [10, 10, 10]
    )
    self.assertEqual(
        [x.num_ended for x in summary.breakdown.values()],
        [10, 10, 10]
    )
    self.assertEqual(
        [x.num_failed for x in summary.breakdown.values()],
        [0, 0, 2]
    )


if __name__ == '__main__':
  unittest.main()
