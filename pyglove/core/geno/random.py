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
"""Random DNA generator."""

import random
import types
from typing import Any, Optional, Union

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.geno.base import DNA
from pyglove.core.geno.base import DNASpec
from pyglove.core.geno.dna_generator import DNAGenerator


@symbolic.members([
    ('seed', pg_typing.Int().noneable(), 'Random seed.')
])
class Random(DNAGenerator):
  """Random DNA generator."""

  def _setup(self):
    """Setup DNA spec."""
    if self.seed is None:
      self._random = random
    else:
      self._random = random.Random(self.seed)

  def _propose(self) -> DNA:
    """Propose a random DNA."""
    return random_dna(self._dna_spec, self._random)

  def _replay(self, trial_id: int, dna: DNA, reward: Any) -> None:
    """Replay the history to recover the last proposed DNA."""
    # NOTE(daiyip): If the seed is fixed, we want to reproduce the same
    # sequence of random examples, we can do this simply by repeating previous
    # generation process.
    if self.seed is not None:
      random_dna(self._dna_spec, self._random)


def random_dna(
    dna_spec: DNASpec,
    random_generator: Union[None, types.ModuleType, random.Random] = None,
    attach_spec: bool = True,
    previous_dna: Optional[DNA] = None
    ) -> DNA:
  """Generates a random DNA from a DNASpec.

  Example::

    spec = pg.geno.space([
        pg.geno.oneof([
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant()
        ]),
        pg.geno.floatv(0.1, 0.2)
    ])

    print(pg.random_dna(spec))
    # DNA([2, 0.1123])

  Args:
    dna_spec: a DNASpec object.
    random_generator: a Python random generator.
    attach_spec: If True, attach the DNASpec to generated DNA.
    previous_dna: An optional DNA representing previous DNA. This field might
        be useful for generating stateful random DNAs.

  Returns:
    A DNA object.
  """
  return dna_spec.random_dna(
      random_generator or random, attach_spec, previous_dna)

