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
"""Base early stopping policy that is friendly to composition."""

from typing import Iterable
import pyglove.core as pg


class EarlyStopingPolicyBase(pg.tuning.EarlyStoppingPolicy):
  """An early stopping policy base class that supports composition."""

  def __and__(self, other) -> pg.tuning.EarlyStoppingPolicy:
    """Operator &."""
    return And(self, other)

  def __or__(self, other) -> pg.tuning.EarlyStoppingPolicy:
    """Operator |."""
    return Or(self, other)

  def __neg__(self) -> pg.tuning.EarlyStoppingPolicy:
    """Operator -."""
    return Not(self)

  def __invert__(self) -> pg.tuning.EarlyStoppingPolicy:
    return Not(self)


@pg.members([
    ('children',
     pg.typing.List(pg.typing.Object(pg.tuning.EarlyStoppingPolicy))),
], init_arg_list=['*children'])
class Composite(EarlyStopingPolicyBase):
  """Base class for composite early stopping policies."""

  def recover(self, history: Iterable[pg.tuning.Trial]):
    for child in self.children:
      child.recover(history)


class And(Composite):
  """Logical AND as a composite early stopping policy."""

  def should_stop_early(self, trial: pg.tuning.Trial) -> bool:
    for child in self.children:
      if not child.should_stop_early(trial):
        return False
    return True


class Or(Composite):
  """Logical OR as a composite early stopping policy."""

  def should_stop_early(self, trial: pg.tuning.Trial) -> bool:
    for child in self.children:
      if child.should_stop_early(trial):
        return True
    return False


class Not(Composite):
  """Logical OR as a composite early stopping policy."""

  def should_stop_early(self, trial: pg.tuning.Trial) -> bool:
    assert len(self.children) == 1, self.children
    return not self.children[0].should_stop_early(trial)
