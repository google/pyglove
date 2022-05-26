# Copyright 2019 The PyGlove Authors
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
"""Generic Hill-Climbing Algorithm."""

from typing import Optional

import pyglove.core as pg
from pyglove.generators.evolution import base
from pyglove.generators.evolution import mutators
from pyglove.generators.evolution import selectors


def hill_climb(mutator=mutators.Uniform(),
               batch_size: int = 1,
               init_population_size: int = 1,
               seed: Optional[int] = None) -> base.Evolution:
  """Hill-Climbing algorithm, with an extra batched setting.

  Batched setting was shown to be effective in
  https://arxiv.org/pdf/1911.06317.pdf and https://arxiv.org/pdf/2003.01239.pdf,
  especially in noisy objective settings.

  Args:
    mutator: Mutator to use.
    batch_size: Number of mutations of the current best.
    init_population_size: Initial population size (randomly generated).
    seed: Random seed.

  Returns:
    An `Evolution` object.
  """
  return base.Evolution(
      selectors.Top(1) >> (mutator * batch_size),   # pytype: disable=unsupported-operands
      population_init=(pg.geno.Random(seed), init_population_size),
      population_update=selectors.Top(1))

