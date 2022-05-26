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
"""NEAT Algorithm: http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf."""

from typing import List, Optional, Tuple

import pyglove.core as pg
from pyglove.generators.evolution import base
from pyglove.generators.evolution import mutators
from pyglove.generators.evolution import selectors


# We disable implicit str concat as it is commonly used class schema docstr.
# pylint: disable=implicit-str-concat


_USERDATA_KEY_SPECIES = 'species'


def neat(
    mutator=mutators.Uniform(),
    population_size: int = 100,
    disjoint_coefficient: float = 1.0,
    matching_coefficient: float = 3.0,
    compatibility_threshold: float = 0.4,
    remaining_ratio: float = 0.6,
    seed: Optional[int] = None) -> base.Evolution:
  """NEAT Algorithm.

  See http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf for the
  original paper.

  NOTE(daiyip): PyGlove supports search spaces that do not change during
  exploration. Therefore we do not grow the program (e.g. Neural Architecture
  in the paper) from a minimal program. Also, crossover can take place on any
  two individuals due to a fixed program structure, which will be implemented
  later. This algorithm illustrates how speciation is expressed in the
  compositional evolution framework.

  Args:
    mutator: Mutator to use.
    population_size: Population size for each generation.
    disjoint_coefficient: Coefficient for DNA disjointness. Used for distance
      computations.
    matching_coefficient: Coefficient for DNA matching. Used for distance
      computations.
    compatibility_threshold: Threshold for max distances between two DNA in
      the same species.
    remaining_ratio: Ratio of best individuals in a species to remain.
    seed: Random seed. If None, use the current system time.

  Returns:
    An `Evolution` object that represents the NEAT algorithm.
  """
  # pylint: disable=no-value-for-parameter
  return base.Evolution(
      reproduction=(
          # Get the living species from the population.
          base.GlobalStateGetter('living_species')
          # Select species proportional to their scaled average fitness.
          >> selectors.Proportional(
              population_size, scaled_average_fitness()).for_each(
                  # Randomly select 1 member from each species's top performers.
                  ((lambda x: x.members)
                   >> selectors.Top(remaining_ratio)
                   >> selectors.Random(1, seed=seed))).flatten()
          >> mutator),
      population_init=(pg.geno.Random(seed=seed), population_size),
      population_update=(
          # Keep only the individuals from the latest generation in the
          # population.
          selectors.Top(1, cluster=True, key=base.get_generation_id)
          # Speciate new individuals that are recently added to the population.
          >> speciate(
              distance=compatibility_distance(
                  disjoint_coefficient=disjoint_coefficient,
                  matching_coefficient=matching_coefficient),
              distance_threshold=compatibility_threshold))
      )
  # pylint: enable=no-value-for-parameter


class Species:
  """Represents a species of Gnomes."""

  def __init__(self):
    self._representative = None
    self._members = []

  @property
  def representative(self) -> pg.DNA:
    """Returns the representative of current species."""
    return self._representative

  @property
  def members(self) -> List[pg.DNA]:
    """Returns the living members of current species."""
    return self._members

  def __len__(self):
    """Returns the number of living members."""
    return len(self._members)

  def add(self, member: pg.DNA) -> None:
    """Adds a member to the species."""
    self._members.append(member)
    if self._representative is None:
      self._representative = member

  def clear(self) -> None:
    """Remove all the members from this species."""
    self._members.clear()


@pg.functor([
    ('distance', pg.typing.Callable(
        [pg.typing.Object(pg.DNA), pg.typing.Object(pg.DNA)],
        returns=pg.typing.Float()),
     'A callable object to compute the distance of two DNA.'),
    ('distance_threshold', pg.typing.Float(min_value=0.),
     'Threshold for max distances between two DNA in the same species.')
])
def speciate(
    dna_list: List[pg.DNA],
    global_state: pg.geno.AttributeDict,
    distance,
    distance_threshold) -> List[pg.DNA]:
  """Speciate new DNA in the population and update the living species."""
  if 'living_species' not in global_state:
    global_state.living_species = []

  # Clear species members.
  for species in global_state.living_species:
    species.clear()

  # Speciate each DNA.
  for dna in dna_list:
    # DNA is already associated with a species, skip it.
    parent_species = dna.userdata.get(_USERDATA_KEY_SPECIES)
    if parent_species is None:
      for species in global_state.living_species:
        dist = distance(species.representative, dna)
        if dist <= distance_threshold:
          species.add(dna)
          parent_species = species
          break

      # Create a new species.
      if parent_species is None:
        parent_species = Species()
        global_state.living_species.append(parent_species)

      dna.set_userdata(_USERDATA_KEY_SPECIES, parent_species)
    # Add current living DNA to its species.
    parent_species.add(dna)

  # Update living species.
  global_state.living_species = [s for s in global_state.living_species if s]
  return dna_list


@pg.functor([
    ('disjoint_coefficient', pg.typing.Float(min_value=0.0),
     'Coefficient for DNA disjointness.'),
    ('matching_coefficient', pg.typing.Float(min_value=0.0),
     'Coefficient for DNA matching.')
])
def compatibility_distance(
    left: pg.DNA,
    right: pg.DNA,
    disjoint_coefficient,
    matching_coefficient) -> float:
  """"Computes compatibility distance with representative."""
  n, w, d = _compute_diff(left, right)
  return (disjoint_coefficient * float(d) / float(n)
          + matching_coefficient * float(w) / float(n))


@pg.functor()
def scaled_average_fitness(inputs: List[Species], step: int) -> List[float]:
  """Returns scaled average fitness of each species."""
  del step
  global_min = None
  examples = []
  for species in inputs:
    for dna in species.members:
      fitness = base.get_fitness(dna)
      if global_min is None or global_min > fitness:
        global_min = fitness
      examples.append(id(dna))
  assert global_min is not None
  return [sum([(base.get_fitness(d) - global_min) for d in s.members])
          / float(len(s)) for s in inputs]


def _compute_diff(left: pg.DNA, right: pg.DNA) -> Tuple[int, int, int]:
  """Compute different positions in two DNAs.

  Args:
    left: the first DNA to compare.
    right: the right DNA to compare.

  Returns:
    A tuple of (N, W, D). 'N' is the total number of components in the larger
    DNA, 'W' is the number of matching genes with different values, and 'D' is
    the number of disjoint
    genes. PyGlove DNAs have no notion of 'E' (i.e. excess genes from the
    original paper), so we exclude them.
  """
  if left.value == right.value:
    assert len(left.children) == len(right.children)
    n = 0 if left.value is None else 1
    w = 0
    d = 0
    for c1, c2 in zip(left.children, right.children):
      cn, cw, cd = _compute_diff(c1, c2)
      n += cn
      w += cw
      d += cd
    return (n, w, d)
  else:
    nl = len(left.to_numbers())
    nr = len(right.to_numbers())
    n = max(nl, nr)
    return (n, 1, n - 1)
