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
"""Step-based scalars used as evolution hyper-parameter values."""


from typing import Any
import pyglove.core as pg
from pyglove.generators.scalars import base


#
# Scheduled values that can be designed in multiple phases.
#


@pg.members([
    ('phases', pg.typing.List(
        pg.typing.Tuple([
            pg.typing.Union([pg.typing.Int(min_value=0),
                             pg.typing.Float(min_value=0.)]),
            base.scalar_spec(pg.typing.Any())
        ]), min_size=1),
     ('All the phases in the schedule. Each item in the list is a tuple of '
      '`(length of phase, scheduled value)`. The length of phase can be an '
      'integer representing number of steps used for that phase, or a float as '
      'the proportion of that phase if `total_steps` is specified. All items '
      'in the list should use the same type (integer or float) for the length '
      'of phase. When a proportion is used, their sum does not have to sum up '
      'to 1.')),
    ('total_steps', pg.typing.Int(min_value=1).noneable(),
     ('Total number of steps for the schedule. If None, the length of each '
      'phase must be an integer.'))
])
class StepWise(base.Scalar):
  """A step-wise schedule that is specified via multiple phases."""

  def _on_bound(self):
    super()._on_bound()

    last_step = 0
    phase_ending_steps = []
    if self.total_steps is None:
      for phase_len, phase_value in self.phases:
        if isinstance(phase_len, float):
          raise ValueError(
              f'`total_steps` must be specified when float is used as the '
              f'value for phase length. '
              f'Encountered: ({phase_len}, {phase_value}).')
        last_step += phase_len
        phase_ending_steps.append(last_step - 1)
    else:
      proportion_sum = 0.
      for proportion, phase_value in self.phases:
        if isinstance(proportion, int):
          raise ValueError(
              f'The phase length should be a float as a proportion of the '
              f'entire schedule when `total_steps` is specified. '
              f'Encountered: ({proportion}, {phase_value}).')
        proportion_sum += proportion

      if proportion_sum == 0:
        raise ValueError(
            f'The sum of all proportions must be greater than 0. '
            f'Encountered: {self.phases!r}')

      for proportion, _ in self.phases:
        phase_len = int(proportion / proportion_sum * self.total_steps)
        last_step += phase_len
        phase_ending_steps.append(last_step - 1)
    # Phase ending step is the step AFTER which the next phase will start.
    self._phase_ending_steps = phase_ending_steps
    self._phases = [base.make_scalar(p) for l, p in self.phases]
    self._current_phase = 0
    self._last_value = None

  def call(self, step: int) -> Any:
    if self._current_phase < len(self.phases):
      if self._current_phase > 0:
        phase_step = step - (
            self._phase_ending_steps[self._current_phase - 1] + 1)
      else:
        phase_step = step
      self._last_value = self._phases[self._current_phase](phase_step)
      if step == self._phase_ending_steps[self._current_phase]:
        self._current_phase += 1
    return self._last_value
