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
"""Tests for pyglove.core.tuning.protocols."""

import time
import unittest

from pyglove.core import geno
from pyglove.core.tuning.protocols import Measurement
from pyglove.core.tuning.protocols import Trial


class TrialTest(unittest.TestCase):
  """Test for Trial class."""

  def test_get_reward_for_feedback(self):
    t = Trial(
        id=0, dna=geno.DNA(0),
        status='PENDING',
        created_time=int(time.time()))
    self.assertIsNone(t.get_reward_for_feedback())

    t = Trial(
        id=0, dna=geno.DNA(0),
        status='COMPLETED',
        infeasible=True,
        created_time=int(time.time()))
    self.assertIsNone(t.get_reward_for_feedback())

    t = Trial(
        id=0, dna=geno.DNA(0),
        status='COMPLETED',
        infeasible=False,
        final_measurement=Measurement(
            step=1, elapse_secs=0.1, reward=1.0, metrics=dict(
                accuracy=0.9, latency=750.0)),
        created_time=int(time.time()))
    self.assertEqual(t.get_reward_for_feedback(), 1.0)
    self.assertEqual(t.get_reward_for_feedback(['accuracy', 'latency']),
                     (0.9, 750.0))
    with self.assertRaisesRegex(
        ValueError, 'Metric \'foo\' does not exist'):
      t.get_reward_for_feedback(['foo'])


if __name__ == '__main__':
  unittest.main()
