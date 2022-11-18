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
"""Sweeping DNA generator."""

from typing import Any

from pyglove.core.geno.base import DNA
from pyglove.core.geno.dna_generator import DNAGenerator


class Sweeping(DNAGenerator):
  """Sweeping (Grid Search) DNA generator."""

  def _setup(self):
    """Setup DNA spec."""
    self._last_proposed_dna = None

  def _propose(self) -> DNA:
    """Propose a random DNA."""
    next_dna = self.dna_spec.next_dna(self._last_proposed_dna)
    if next_dna is None:
      raise StopIteration()
    self._last_proposed_dna = next_dna
    return next_dna

  def _replay(self, trial_id: int, dna: DNA, reward: Any) -> None:
    """Replay the history to recover the last proposed DNA."""
    del trial_id, reward
    self._last_proposed_dna = dna
