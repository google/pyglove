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
"""NSGA-II algorithm: https://ieeexplore.ieee.org/document/996017."""

from typing import List, Optional, Tuple

import pyglove.core as pg
from pyglove.generators.evolution import base
from pyglove.generators.evolution import mutators
from pyglove.generators.evolution import selectors


def nsga2(mutator=mutators.Uniform(),
          population_size: int = 100,
          seed: Optional[int] = None):
  """NSGA-II: A multi-objective evolutionary search algorithm.

  For reference and for citations, please use:
  https://ieeexplore.ieee.org/document/996017

  Args:
    mutator: Mutator to use.
    population_size: Population size, which will be used as both batch size for
      proposing new individuals and tourament size for finding the elites among
      recently added populations.
    seed: Random seed for initializing the population. If None, the system time
      will be used.

  Returns:
    An `Evolution` object.
  """
  # pylint: disable=no-value-for-parameter
  return base.Evolution(
      # Reproduction is to simply select the next elite and mutate it.
      next_elite() >> mutator,
      population_init=(pg.geno.Random(seed=seed), population_size * 2),
      population_update=(
          # Take previous elites and unprocessed population as input to compute
          # the new elites.
          base.GlobalStateGetter('elites', []) + base.Identity()
          # Apply non-dominated sorting, which returns a list of frontiers.
          # Each frontier is a list of DNA.
          # NOTE(daiyip): We converts the functor into a Lambda operation here
          # in order to call `for_each`.
          >> base.Lambda(nondominated_sort()).for_each(
              # Apply crowding distance sort on each frontier, and flatten
              # their output into a single list.
              crowding_distance_sort()).flatten()
          # Choose the first N (N=population_size) individuals as elites.
          # Reset the next cursor.
          >> (selectors.First(population_size)
              .as_global_state('elites')
              .set_global_state('elite_cursor', 0))
          # We update the elites only when unprocessed population reaches the
          # population limit.
          ).if_true(lambda x: len(x) >= population_size),
      multi_objective=True)
  # pylint: enable=no-value-for-parameter


@pg.functor()
def next_elite(dna_list: List[pg.DNA], global_state: pg.Dict):
  """Returns the next elite."""
  del dna_list
  elite = global_state.elites[global_state.elite_cursor]
  global_state.elite_cursor = (
      global_state.elite_cursor + 1) % len(global_state.elites)
  return [elite]


@pg.functor()
def nondominated_sort(inputs: List[pg.DNA]) -> List[List[pg.DNA]]:
  """Algorithm fast-non-dominated-sort implementation using topological sort.

  Please see section III A in the original paper.
  Args:
    inputs: A list of DNA that need to be sorted.

  Returns:
    A list of sorted frontier (a list of DNA).
  """
  dependency_graph = [[] for i in range(len(inputs))]
  indegree = [0] * len(inputs)
  queue = []
  for i in range(len(inputs)):
    for j in range(len(inputs)):
      if dominates(base.get_fitness(inputs[i]),
                   base.get_fitness(inputs[j])):
        dependency_graph[i].append(j)
      elif dominates(base.get_fitness(inputs[j]),
                     base.get_fitness(inputs[i])):
        indegree[i] += 1
    if indegree[i] == 0:
      queue.append(i)

  # Topological sort.
  result = []
  while queue:
    l = len(queue)
    frontier = []
    for i in range(l):
      parent = queue.pop(0)
      frontier.append(inputs[parent])
      for child in dependency_graph[parent]:
        indegree[child] -= 1
        if indegree[child] == 0:
          queue.append(child)
    result.append(frontier)
  return result


@pg.functor()
def crowding_distance_sort(frontier: List[pg.DNA]) -> List[pg.DNA]:
  """Algorithm crowding-distance-assignment implementation.

  Check section III B in the original paper.

  Args:
    frontier: A list of Individual that need to be sorted.

  Returns:
    sorted list of the original list.
  """
  if len(frontier) <= 1:
    return frontier

  individual_num = len(frontier)
  objective_num = len(base.get_fitness(frontier[0]))
  distances = [0.0] * individual_num

  # dist is an index array, dist[i][j] represents the individual index in the
  # frontier array. This array is also used for per objective sorting.
  dist = [list(range(individual_num)) for i in range(objective_num)]
  for i in range(objective_num):
    # pylint: disable=cell-var-from-loop
    dist[i] = sorted(dist[i],
                     key=lambda idx: base.get_fitness(frontier[idx])[i])
    max_value = base.get_fitness(frontier[dist[i][individual_num - 1]])[i]
    min_value = base.get_fitness(frontier[dist[i][0]])[i]
    for j in range(individual_num):
      if j == 0 or j == individual_num - 1:
        distances[dist[i][j]] = objective_num
      elif max_value > min_value:
        distances[dist[i][j]] += (
            (base.get_fitness(frontier[dist[i][j + 1]])[i] -
             base.get_fitness(frontier[dist[i][j - 1]])[i])
            / (max_value - min_value))
  # Sorts frontier in decreasing order wrt crowding distances.
  idx_arr = list(range(individual_num))
  idx_arr = sorted(idx_arr, key=lambda idx: distances[idx], reverse=True)
  return [frontier[idx_arr[i]] for i in range(individual_num)]


def dominates(ind1: Tuple[float, ...], ind2: Tuple[float, ...]) -> bool:
  """Returns True if ind1 pareto dominates ind2."""
  if len(ind1) != len(ind2):
    raise ValueError(
        f'The length of {ind1!r} is not equal to the length '
        f'of individual objectives {ind2!r}.')
  is_dominating = False
  for i in range(len(ind1)):
    if ind1[i] < ind2[i]:
      return False
    elif ind1[i] > ind2[i]:
      is_dominating = True
  return is_dominating

