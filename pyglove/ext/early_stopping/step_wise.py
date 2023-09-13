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
"""Step-wise early stopping policies."""

import numbers
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import pyglove.core as pg
from pyglove.ext.early_stopping import base


def step_wise_stopping_predicate_spec():
  """Returns the value spec for step-wise predicate."""
  return pg.typing.Callable(
      [pg.typing.Object(pg.tuning.Measurement),  # Measurement of current trial.
       pg.typing.List(                           # Historical measurements.
           pg.typing.Object(pg.tuning.Measurement))],
      returns=pg.typing.Bool())                  # Should stop or not.


@pg.members([
    ('plan', pg.typing.List(
        pg.typing.Tuple([
            pg.typing.Int(min_value=0),           # Gating step.
            step_wise_stopping_predicate_spec(),  # Step-wise predicate.
        ]), min_size=1),
     ('A list of tuple (step, step-wise stopping predicate) as the step-wise '
      'stopping plan.'))
])
class StepWise(base.EarlyStopingPolicyBase):
  """Step-wise early stopping policy."""

  def _on_bound(self):
    super()._on_bound()
    self.rebind(
        plan=sorted(self.plan, key=lambda x: x[0]), skip_notification=True)
    self._gate_history: List[List[pg.tuning.Measurement]] = [
        [] for _ in range(len(self.plan))]
    self._trial_gate_decision: Dict[int, Tuple[int, bool]] = {}

  def should_stop_early(self, trial: pg.tuning.Trial) -> bool:
    """Returns True if a trial should be stopped early."""
    if not trial.measurements:
      return False

    should_stop = False
    gate_index = self._get_gate_index(trial)
    if gate_index >= 0:
      if trial.id in self._trial_gate_decision:
        decision_gate_index, decision = self._trial_gate_decision[trial.id]
        if decision_gate_index == gate_index:
          return decision
      gate_predicate = self.plan[gate_index][1]
      gate_history = self._gate_history[gate_index]
      m = trial.measurements[-1]
      if gate_predicate(m, gate_history):
        should_stop = True
      gate_history.append(m)
      self._trial_gate_decision[trial.id] = (gate_index, should_stop)
    return should_stop

  def _get_gate_index(self, trial: pg.tuning.Trial) -> int:
    """Gets the index of gate for a trial."""
    step = trial.measurements[-1].step
    index = -1
    for i, (gating_step, _) in enumerate(self.plan):
      if gating_step <= step:
        index = i
    return index

  def recover(self, history: Iterable[pg.tuning.Trial]):
    """Recovers the policy state based on history."""
    for t in history:
      prev_m = None
      next_gate = 0
      for i, m in enumerate(t.measurements):
        if (next_gate != len(self.plan)
            and (prev_m is None or prev_m.step < self.plan[next_gate][0])
            and m.step >= self.plan[next_gate][0]):
          stopping_decision = False
          gate_history = self._gate_history[next_gate]
          if i == len(t.measurements) - 1:
            if t.status == 'COMPLETED':
              stopping_decision = t.infeasible
            elif t.status == 'PENDING':
              # For the last measurement of a pending trial, we use gate
              # predicate to determine the stopping decision.
              gate_predicate = self.plan[next_gate][1]
              stopping_decision = gate_predicate(m, gate_history)
          gate_history.append(m)
          self._trial_gate_decision[t.id] = (next_gate, stopping_decision)
          next_gate += 1
        prev_m = m


@pg.symbolize
def early_stop_by_value(
    step_values: List[Tuple[
        int,          # Gating step.
        float]],      # Value threshold.
    metric: Union[str, Callable[[pg.tuning.Measurement], float]] = 'reward',
    maximize: bool = True):
  """Step-wise early stopping policy based on the value of reward/metric.

  Example::

    policy = early_stop_by_value([
      # Stop at step 1 if trial reward is less than 0.2.
      (1, 0.2),

      # Stop at step 2 if trial reward is less than 0.8.
      (2, 0.8),
    ])()

  Args:
    step_values: A list of tuple (gating step, value threshold).
        gating step - At which step this rule will be triggered.
        value threshold - A float number indicating the threshold value for
          early stopping.
    metric: Based on which metric the value should be compared against.
      Use str for metric name or a callable object that takes a measurement
      object at a given step as input and returns a float value.
    maximize: If True, reward or metric value below the threshold will be
      stopped, otherwise trials with values above the threshold will be stopped.

  Returns:
    A `StepWise` early stopping policy.
  """
  assert isinstance(step_values, list), step_values
  for v in step_values:
    if (not isinstance(v, tuple)
        or len(v) != 2
        or not isinstance(v[0], int)
        or not isinstance(v[1], numbers.Number)):
      raise ValueError(
          f'Invalid definition in `step_values`: {v}. '
          f'Expect a tuple of 2 elements: '
          f'(step: int, threshold: float).')
  def _cmp(x, y) -> bool:
    return x < y if maximize else x > y

  def _value(m: pg.tuning.Measurement) -> float:
    if isinstance(metric, str):
      return m.reward if metric == 'reward' else m.metrics[metric]
    assert callable(metric), metric
    return metric(m)

  def _make_predicate(threshold: float):
    def _predicate(m: pg.tuning.Measurement, unused_history):
      v = _value(m)
      ret = _cmp(v, threshold)
      return ret
    return _predicate

  return StepWise([
      (step, _make_predicate(threshold))
      for step, threshold in step_values])


@pg.symbolize
def early_stop_by_rank(
    step_ranks: List[Tuple[
        int,            # Gating step.
        Union[float,    # Rank percentage in (0.0, 1.0).
              int],     # Absolute rank, below which will be stopped.
        int]],          # Min histogram size at the step to trigger stopping.
    metric: Union[str, Callable[[pg.tuning.Measurement], float]] = 'reward',
    maximize: bool = True) -> StepWise:
  """Step-wise early stopping policy based on the rank of reward/metric.

  Example::

    policy = early_stop_by_rank([
      # Stop at step 1 if accuracy is less than top 80% previous trials at
      # this step, enabled when there are at least 5 previous trials reported
      # at this step.
      (1, 0.8, 5),

      # Stop at step 2 if accuracy is less than top 20% previous trials at
      # this step, enabled when there are at least 10 previous trials reported
      # at this step.
      (2, 0.2, 10),

      # Stop at step 3 if accuracy is less than the 3rd best trial at this step,
      # enabled when there are at least 3 previous trials reported at this step.
      (3, 3, 3)
    ], metric='accuracy')()

  Args:
    step_ranks: A list of tuple (gating step, rank threshold, trigger histogram
      size).
        gating step - At which step this rule will be triggered.
        rank threshold - A float number in range (0, 1) indicating the rank
          percentage or an integer (> 0) indicating the absolute rank as the
          threshold for early stopping.
        trigger historgram size - The minimal number of historical trials
          repoted at current step for this rule to trigger.
    metric: Based on which metric the rank will be computed.
      Use str for metric name or a callable object that takes a measurement
      object at a given step as input and returns a float value.
    maximize: If True, reward or metric value below the threshold will be
      stopped, otherwise trials with values above the threshold will be stopped.

  Returns:
    A `StepWise` early stopping policy.
  """
  assert isinstance(step_ranks, list), step_ranks
  for v in step_ranks:
    if (not isinstance(v, tuple)
        or len(v) != 3
        or not isinstance(v[0], int)
        or not isinstance(v[1], (int, float))
        or not isinstance(v[2], int)):
      raise ValueError(
          f'Invalid definition in `step_ranks`: {v}. '
          f'Expect a tuple of 3 elements: '
          f'(step: int, rank: Union[float, int], trigger_size: int).')
    if isinstance(v[1], float) and (v[1] > 1 or v[1] < 0):
      raise ValueError(
          f'Rank must be within range [0.0, 1.0] when it is percentage '
          f'(float). Encountered: {v[1]} in {v}.')

  def _cmp(x, y) -> bool:
    if y is None:
      return False
    return x < y if maximize else x > y

  def _value(m: pg.tuning.Measurement) -> float:
    if isinstance(metric, str):
      return m.reward if metric == 'reward' else m.metrics[metric]
    assert callable(metric), metric
    return metric(m)

  def _value_by_rank(
      h: List[pg.tuning.Measurement],
      threshold: Union[int, float]) -> Optional[float]:
    if isinstance(threshold, float):
      assert 0.0 <= threshold <= 1.0, threshold
      k = int((len(h) - 1) * threshold)
    else:
      assert isinstance(threshold, int)
      k = threshold - 1
      if k < -len(h) or k >= len(h):
        return None
    return sorted([_value(r) for r in h], reverse=maximize)[k]

  def _make_predicate(rank: Union[int, float], trigger_size: int):
    def _predicate(
        m: pg.tuning.Measurement,
        history: List[pg.tuning.Measurement]):
      if not history or len(history) < trigger_size:
        return False
      return _cmp(_value(m), _value_by_rank(history, rank))
    return _predicate
  return StepWise([(step, _make_predicate(rank, trigger_size))
                   for step, rank, trigger_size in step_ranks])
