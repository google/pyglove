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

from pyglove.core.symbolic import error_info  # pylint: disable=unused-import
from pyglove.core.utils import json_conversion
from pyglove.core.utils import timing


class TimeItTest(unittest.TestCase):
  """Tests for `pg.symbolic.thread_local`."""

  def test_basics(self):
    tc = timing.TimeIt('node')
    self.assertFalse(tc.has_started)
    self.assertEqual(tc.elapse, 0)

    tc.start()
    self.assertTrue(tc.has_started)
    self.assertGreaterEqual(tc.elapse, 0)

    self.assertTrue(tc.end())
    self.assertTrue(tc.has_ended)
    elapse1 = tc.elapse
    self.assertFalse(tc.end())
    self.assertEqual(tc.elapse, elapse1)

  def test_timeit(self):
    with timing.timeit('node') as t:
      self.assertEqual(t.name, 'node')
      self.assertIsNotNone(t.start_time)
      self.assertTrue(t.has_started)
      self.assertIsNone(t.end_time)
      self.assertFalse(t.has_ended)
      time.sleep(0.5)
      elapse1 = t.elapse
      self.assertEqual(t.children, [])
      with timing.timeit('child') as t1:
        time.sleep(0.5)
        with timing.timeit('grandchild') as t2:
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

        with timing.timeit('grandchild') as t3:
          time.sleep(0.5)
        self.assertEqual(t1.children, [t2, t3])

    elapse2 = t.elapse
    self.assertTrue(t.has_ended)
    self.assertGreater(elapse2, elapse1)
    time.sleep(0.5)
    self.assertEqual(elapse2, t.elapse)

    status = t.status()
    self.assertEqual(
        list(status.keys()),
        ['node', 'node.child', 'node.child.grandchild']
    )
    self.assertEqual(
        status['node.child.grandchild'].elapse, t2.elapse + t3.elapse
    )
    self.assertEqual(
        sorted([v.elapse for v in status.values()], reverse=True),
        [v.elapse for v in status.values()],
    )
    self.assertTrue(all(v.has_ended for v in status.values()))
    self.assertFalse(any(v.has_error for v in status.values()))
    status = t.status()
    json_dict = json_conversion.to_json(status)
    status2 = json_conversion.from_json(json_dict)
    self.assertIsNot(status2, status)
    self.assertEqual(status2, status)

  def test_timeit_with_error(self):
    with self.assertRaises(ValueError):
      with timing.timeit('node') as t:
        with timing.timeit('child') as t1:
          with timing.timeit('grandchild') as t2:
            raise ValueError('error')

    r = t.status()
    self.assertTrue(r['node'].has_error)
    self.assertTrue(t.has_error)
    self.assertTrue(t.error.tag.startswith('ValueError'))
    self.assertTrue(r['node'].error.tag.startswith('ValueError'))
    self.assertTrue(r['node.child'].has_error)
    self.assertTrue(t1.has_error)
    self.assertTrue(r['node.child.grandchild'].has_error)
    self.assertTrue(t2.has_error)

  def test_timeit_summary(self):
    summary = timing.TimeIt.StatusSummary()
    self.assertFalse(summary)
    for i in range(10):
      with timing.timeit() as t:
        time.sleep(0.1)
        with timing.timeit('child'):
          time.sleep(0.1)
          try:
            with timing.timeit('grandchild'):
              time.sleep(0.1)
              if i < 2:
                raise ValueError('error')
          except ValueError:
            pass
      summary.aggregate(t.status())
    self.assertTrue(summary)
    self.assertEqual(
        list(summary.breakdown.keys()),
        ['', 'child', 'child.grandchild']
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
    self.assertEqual(
        summary.breakdown['child.grandchild'].error_tags,
        {'ValueError': 2},
    )
    # Test serialization.
    json_dict = summary.to_json()
    summary2 = timing.TimeIt.StatusSummary.from_json(json_dict)
    self.assertIsNot(summary2, summary)
    self.assertEqual(summary2.breakdown, summary.breakdown)


if __name__ == '__main__':
  unittest.main()
