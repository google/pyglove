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
"""Regularized Evolution: https://arxiv.org/abs/1802.01548."""

from typing import Optional

import pyglove.core as pg
from pyglove.generators.evolution import base
from pyglove.generators.evolution import mutators
from pyglove.generators.evolution import selectors


def regularized_evolution(
    mutator=mutators.Uniform(),
    population_size: int = 100,
    tournament_size: int = 10,
    seed: Optional[int] = None):
  """Regularized Evolution algorithm.

  https://www.aaai.org/ojs/index.php/AAAI/article/view/4405.

  Args:
    mutator: Mutator to use.
    population_size: Population size. Must be larger than tournament size.
    tournament_size: Tournament size.
    seed: Random seed. If None, the current system time is used as seed.

  Returns:
    An `Evolution` object.
  """
  if tournament_size < 2:
    raise ValueError(
        f'`tournament_size` must be no less than 2. '
        f'Encountered: {tournament_size}')
  if population_size < tournament_size:
    raise ValueError(
        f'The value of `population_size` ({population_size}) must be no '
        f'less than the value of `tournament_size` ({tournament_size}).')
  return base.Evolution(
      selectors.Random(
          tournament_size, seed=seed) >> selectors.Top(1) >> mutator,
      population_init=(pg.geno.Random(seed=seed), population_size),
      population_update=selectors.Last(population_size))
