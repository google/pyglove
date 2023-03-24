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
"""Tests for step_wise."""

import unittest
import pyglove.core as pg
from pyglove.ext.early_stopping import step_wise


def get_stopping_steps(policy,
                       num_trials,
                       num_steps,
                       simulate_process_abort=False):
  """Gets early stopping steps based on policy."""
  trials = [pg.tuning.Trial(i + 1, dna=pg.DNA(i), created_time=0)
            for i in range(num_trials)]
  assert not any(policy.should_stop_early(t) for t in trials)
  stopping_trial_steps = []
  for step in range(1, num_steps + 1):
    if simulate_process_abort and step == num_steps - 1:
      return trials, stopping_trial_steps
    for i in range(num_trials):
      if trials[i].status == 'COMPLETED':
        continue
      trial_id = trials[i].id
      reward = float(1 / trial_id * step)
      trials[i].measurements.append(pg.tuning.Measurement(
          step=step, elapse_secs=float(step),
          reward=reward,
          metrics={'loss': trial_id * step / 10}))
      should_stop = policy.should_stop_early(trials[i])

      # Make another call to should_stop_early to test that a measurement
      # will not be inserted to the histogram twice.
      assert should_stop == policy.should_stop_early(trials[i])

      if should_stop:
        trials[i].status = 'COMPLETED'
        trials[i].infeasible = True
        stopping_trial_steps.append((trial_id, step))
  return trials, stopping_trial_steps


class EarlyStopByValueTest(unittest.TestCase):
  """Tests for early_stop_by_value."""

  def test_bad_gating_rules(self):
    with self.assertRaisesRegex(
        ValueError, 'Invalid definition in `step_values`'):
      _ = step_wise.early_stop_by_value([1])()
    with self.assertRaisesRegex(
        ValueError, 'Invalid definition in `step_values`'):
      _ = step_wise.early_stop_by_value([
          (1, 2, 3)
      ])()
    with self.assertRaisesRegex(
        ValueError, 'Invalid definition in `step_values`'):
      _ = step_wise.early_stop_by_value([
          (1, 'abc')
      ])()

  def test_maximize_the_reward(self):
    policy = step_wise.early_stop_by_value([
        # Gate 1 at step 1, stop when reward < 0.6.
        (1, 0.5),

        # Gate 2 at step 3, stop when reward < 2.5.
        (3, 2.5),
    ])()

    # STEP 1
    # Trial 1, step 1, reward 1.00: should stop=False
    # Trial 2, step 1, reward 0.50: should stop=False
    # Trial 3, step 1, reward 0.33: should stop=True
    # Trial 4, step 1, reward 0.25: should stop=True
    # STEP 2
    # Trial 1, step 2, reward 2.00: should stop=False
    # Trial 2, step 2, reward 1.00: should stop=False
    # STEP 3
    # Trial 1, step 3, reward 3.00: should stop=False
    # Trial 2, step 3, reward 1.50: should stop=True
    # STEP 4
    # Trial 1, step 4, reward 4.00: should stop=False
    # STEP 5
    # Trial 1, step 5, reward 5.00: should stop=False
    self.assertEqual(
        get_stopping_steps(policy, 4, 5)[1],
        [(3, 1), (4, 1), (2, 3)])

  def test_minimize_a_metric(self):
    policy = step_wise.early_stop_by_value([
        # Unorder the gating steps to make sure it still work.
        # Gate 1 at step 3, stop when loss > 0.8.
        (3, 0.8),

        # Gate 2 at step 1, stop when loss > 0.3.
        (1, 0.3),
    ], maximize=False, metric='loss')()

    # STEP 1
    # Trial 1, step 1, loss 0.10: should stop=False
    # Trial 2, step 1, loss 0.20: should stop=False
    # Trial 3, step 1, loss 0.30: should stop=False
    # Trial 4, step 1, loss 0.40: should stop=True
    # STEP 2
    # Trial 1, step 2, loss 0.20: should stop=False
    # Trial 2, step 2, loss 0.40: should stop=False
    # Trial 3, step 2, loss 0.60: should stop=False
    # STEP 3
    # Trial 1, step 3, loss 0.30: should stop=False
    # Trial 2, step 3, loss 0.60: should stop=False
    # Trial 3, step 3, loss 0.90: should stop=True
    # STEP 4
    # Trial 1, step 4, loss 0.40: should stop=False
    # Trial 2, step 4, loss 0.80: should stop=False
    # STEP 5
    # Trial 1, step 5, loss 0.50: should stop=False
    # Trial 2, step 5, loss 1.00: should stop=False
    self.assertEqual(
        get_stopping_steps(policy, 4, 5)[1],
        [(4, 1), (3, 3)])

  def test_callable_metrics(self):
    policy = step_wise.early_stop_by_value([
        # Gate 1 at step 1, stop when loss > 0.3.
        (1, 0.3),

        # Gate 2 at step 3, stop when loss > 0.8.
        (3, 0.8),
    ], maximize=False, metric=lambda m: m.metrics['loss'])()

    self.assertEqual(
        get_stopping_steps(policy, 4, 5)[1],
        [(4, 1), (3, 3)])


class EarlyStopByRankTest(unittest.TestCase):
  """Tests for early_stop_by_rank."""

  def test_bad_gating_rules(self):
    with self.assertRaisesRegex(
        ValueError, 'Invalid definition in `step_ranks`'):
      _ = step_wise.early_stop_by_rank([
          (1, 2)
      ])()
    with self.assertRaisesRegex(
        ValueError, 'Invalid definition in `step_ranks`'):
      _ = step_wise.early_stop_by_rank([
          (1, 2, None)
      ])()
    with self.assertRaisesRegex(
        ValueError, 'Invalid definition in `step_ranks`'):
      _ = step_wise.early_stop_by_rank([
          (0.2, 2, 1)
      ])()
    with self.assertRaisesRegex(
        ValueError, r'Rank must be within range \[0.0, 1.0\]'):
      _ = step_wise.early_stop_by_rank([
          (1, 2.0, 1)
      ])()

  def test_maximize_the_reward(self):
    policy = step_wise.early_stop_by_rank([
        # Gate 1 at step 1, stop when reward < top 50% and len(hist) >= 2.
        (1, 0.5, 2),

        # Gate 2 at step 2, stop when reward < top 100.
        (2, 100, 0),

        # Gate 3 at step 3, stop when reward < top 1.
        (3, 1, 0),
    ])()

    # STEP 1
    # Trial 1, step 1, reward 1.00: should stop=False
    # Trial 2, step 1, reward 0.50: should stop=False
    # Trial 3, step 1, reward 0.33: should stop=True
    # Trial 4, step 1, reward 0.25: should stop=True
    # STEP 2
    # Trial 1, step 2, reward 2.00: should stop=False
    # Trial 2, step 2, reward 1.00: should stop=False
    # STEP 3
    # Trial 1, step 3, reward 3.00: should stop=False
    # Trial 2, step 3, reward 1.50: should stop=True
    # STEP 4
    # Trial 1, step 4, reward 4.00: should stop=False
    # STEP 5
    # Trial 1, step 5, reward 5.00: should stop=False
    self.assertEqual(
        get_stopping_steps(policy, 4, 5)[1],
        [(3, 1), (4, 1), (2, 3)])

  def test_minimize_a_metric(self):
    policy = step_wise.early_stop_by_rank([
        # Gate 1 at step 1, stop when loss rank < top % 80 and len(hist) >= 3.
        (1, 0.8, 3),

        # Gate 2 at step 3, stop when loss rank < top 1.
        (3, 1, 0),
    ], maximize=False, metric='loss')()

    # STEP 1
    # Trial 1, step 1, loss 0.10: should stop=False
    # Trial 2, step 1, loss 0.20: should stop=False
    # Trial 3, step 1, loss 0.30: should stop=False
    # Trial 4, step 1, loss 0.40: should stop=True
    # STEP 2
    # Trial 1, step 2, loss 0.20: should stop=False
    # Trial 2, step 2, loss 0.40: should stop=False
    # Trial 3, step 2, loss 0.60: should stop=False
    # STEP 3
    # Trial 1, step 3, loss 0.30: should stop=False
    # Trial 2, step 3, loss 0.60: should stop=True
    # Trial 3, step 3, loss 0.90: should stop=True
    # STEP 4
    # Trial 1, step 4, loss 0.40: should stop=False
    # STEP 5
    # Trial 1, step 5, loss 0.50: should stop=False
    self.assertEqual(
        get_stopping_steps(policy, 4, 5)[1],
        [(4, 1), (2, 3), (3, 3)])

  def test_callable_metric(self):
    policy = step_wise.early_stop_by_rank([
        # Gate 1 at step 1, stop when loss rank < top % 80 and len(hist) >= 3.
        (1, 0.8, 3),

        # Gate 2 at step 3, stop when loss rank < top 1.
        (3, 1, 0),
    ], maximize=False, metric=lambda m: m.metrics['loss'])()
    self.assertEqual(
        get_stopping_steps(policy, 4, 5)[1],
        [(4, 1), (2, 3), (3, 3)])

  def test_recover(self):
    policy = step_wise.early_stop_by_rank([
        # Gate 1 at step 1, stop when reward < top 50% and len(hist) >= 2.
        (1, 0.5, 2),

        # Gate 2 at step 3, stop when reward < top 1.
        (3, 1, 0),
    ])
    p1 = policy()
    trials = get_stopping_steps(p1, 4, 5, simulate_process_abort=True)[0]
    p2 = policy()
    p2.recover(trials)
    self.assertEqual(p1._gate_history, p2._gate_history)
    self.assertEqual(p1._trial_gate_decision, p2._trial_gate_decision)


if __name__ == '__main__':
  unittest.main()
