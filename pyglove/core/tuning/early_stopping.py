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
"""Interface for early stopping policies."""

import abc
from typing import Iterable, Optional

from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core.tuning.protocols import Trial


class EarlyStoppingPolicy(symbolic.Object):
  """Interface for early stopping policy."""

  def setup(self, dna_spec: geno.DNASpec) -> None:
    """Setup states of an early stopping policy based on dna_spec.

    Args:
      dna_spec: DNASpec for DNA to propose.

    Raises:
      RuntimeError: if dna_spec is not supported.
    """
    self._dna_spec = dna_spec

  @property
  def dna_spec(self) -> Optional[geno.DNASpec]:
    return getattr(self, '_dna_spec', None)

  @abc.abstractmethod
  def should_stop_early(self, trial: Trial) -> bool:
    """Should stop the input trial early based on its measurements."""

  def recover(self, history: Iterable[Trial]) -> None:
    """Recover states by replaying the trial history.

    Subclass can override.

    NOTE: `recover` will always be called before the first `should_stop_early`
    is called. It could be called multiple times if there are multiple source
    of history, e.g: trials from a previous study and existing trials from
    current study.

    The default behavior is to replay `should_stop_early` on all intermediate
    measurements on all trials.

    Args:
      history: An iterable object of trials.
    """
    for trial in history:
      if trial.status in ['COMPLETED', 'PENDING', 'STOPPING']:
        self.should_stop_early(trial)
