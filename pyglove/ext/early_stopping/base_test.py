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

from typing import Iterable
import unittest
import pyglove.core as pg
from pyglove.ext.early_stopping import base


@pg.members([
    ('decision', pg.typing.Bool())
])
class ConstantPolicy(base.EarlyStopingPolicyBase):

  def _on_bound(self):
    super()._on_bound()
    self.requested_trials = []
    self.recovered = False

  def should_stop_early(self, trial: pg.tuning.Trial) -> bool:
    self.requested_trials.append(trial)
    return self.decision

  def recover(self, history: Iterable[pg.tuning.Trial]) -> None:
    self.recovered = True


class EarlyStoppingPolicyComposabilityTest(unittest.TestCase):
  """Test the composability of early stopping policies."""

  def test_logical_and(self):
    t = pg.tuning.Trial(id=1, dna=pg.DNA(1), created_time=0)
    x = ConstantPolicy(True)
    self.assertTrue(x.should_stop_early(t))

    y = ConstantPolicy(False)
    self.assertFalse(y.should_stop_early(t))

    self.assertTrue((x & x).should_stop_early(t))
    self.assertFalse((x & y).should_stop_early(t))
    self.assertFalse((y & x).should_stop_early(t))
    self.assertFalse((y & y).should_stop_early(t))

  def test_logical_or(self):
    t = pg.tuning.Trial(id=1, dna=pg.DNA(1), created_time=0)
    self.assertTrue(
        (ConstantPolicy(True) | ConstantPolicy(True)).should_stop_early(t))
    self.assertTrue(
        (ConstantPolicy(True) | ConstantPolicy(False)).should_stop_early(t))
    self.assertTrue(
        (ConstantPolicy(False) | ConstantPolicy(True)).should_stop_early(t))
    self.assertFalse(
        (ConstantPolicy(False) | ConstantPolicy(False)).should_stop_early(t))

  def test_logical_not(self):
    t = pg.tuning.Trial(id=1, dna=pg.DNA(1), created_time=0)
    self.assertFalse((~ConstantPolicy(True)).should_stop_early(t))  # pylint: disable=invalid-unary-operand-type
    self.assertTrue((-ConstantPolicy(False)).should_stop_early(t))  # pylint: disable=invalid-unary-operand-type

  def test_recorver(self):
    t = pg.tuning.Trial(id=1, dna=pg.DNA(1), created_time=0)
    x = ConstantPolicy(True)
    y = ConstantPolicy(False)
    z = x & y
    z.recover([t])
    self.assertTrue(x.recovered)
    self.assertTrue(y.recovered)


if __name__ == '__main__':
  unittest.main()
