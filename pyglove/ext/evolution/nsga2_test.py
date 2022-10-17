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
"""Tests for evolutionary algorithms."""

import random
import time
import unittest

import pyglove.core as pg
from pyglove.ext.evolution import base
from pyglove.ext.evolution import nsga2_lib as nsga2


def get_trivial_search_space():
  """Trivial search space.

  Each point in the space is a value in [0, 1].

  Returns:
    A tunable value.
  """
  return pg.floatv(0.0, 1.0)


def trivial_reward(example):
  """Reward for the trivial search space.

  The reward (i.e. fitness) is the value itself. The goal of the search,
  therefore, is to find the value 1.

  Args:
    example: a materialized value.

  Returns:
    The corresponding reward.
  """
  return example


def simple_two_objective_reward(example):
  """Reward for the trivial search space.

  The reward (i.e. fitness) is a 2-element list. The goal of the search,
  therefore, is to find the pareto frontier in simple_two_objective_pareto
  function.

  Args:
    example: a materialized value.

  Returns:
    A 2-element list.
  """
  num = int(example * 10) % 9 + 1  # Maps to [1 ... 9].

  return [num, 10 - num]


def simple_two_objective_pareto():
  """The pareto frontier of simple_two_objective_reward function."""
  return [[9, 1], [8, 2], [7, 3], [6, 4], [5, 5], [4, 6], [3, 7], [2, 8],
          [1, 9]]


def multi_frontier_two_objective_reward(example):
  """Reward for the trivial search space.

  The reward (i.e. fitness) is a 2-element list. The goal of the search,
  therefore, is to find the pareto frontier in
  multi_frontier_two_objective_pareto function.

  Args:
    example: a materialized value.

  Returns:
    A 2-element list.
  """
  int_val = int(example * 10)
  if int_val >= 0 and int_val < 3:
    return [int_val, 10 - int_val]  # Maps to [0, 10], [1, 9], [2, 8]
  elif int_val >= 3 and int_val < 7:
    return [int_val * 10, 100 - int_val * 10
           ]  # Maps to [30, 70], [40, 60], [50, 50], [60, 40]
  else:
    return [int_val, 10 - int_val]  # Maps to [7, 3], [8, 2], [9, 1], [10, 0]


def multi_frontier_two_objective_pareto():
  """The pareto frontier of multi_frontier_two_objective_reward function."""
  return [[30, 70], [40, 60], [50, 50], [60, 40]]


def get_trivial_hash(search_space, algo):
  hashed_value = 0
  for example, feedback in pg.iter(search_space, 30, algo):
    hashed_value ^= int(example * 1000000)
    feedback([example])
  return hashed_value


@pg.members([
    ('seed', pg.typing.Int().noneable()),
])
class TrivialMutator(base.Mutator):
  """Mutator for trivial search space.

  Mutations can only change the value by a small amount.
  """

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)

  def mutate(self, dna, step):
    del step
    value = dna.value + self._random.uniform(-0.01, 0.01)
    if value < 0.0:
      value = 0.0
    if value > 1.0:
      value = 1.0
    return pg.DNA(value)


class Nsga2Test(unittest.TestCase):

  def _contains(self, outgoing_population, pareto_frontier):
    """Returns True if outgoing_population contains pareto_frontier."""
    # Record the number of occurrence of each element in
    # expected_pareto_frontier.
    cache = [False] * len(pareto_frontier)
    for individual in outgoing_population:
      if pareto_frontier.count(base.get_fitness(individual)):
        cache[pareto_frontier.index(base.get_fitness(individual))] = True
      if False not in cache:
        return True
    return False

  def testDominatesFunctionWorksAsIntended(self):
    self.assertTrue(nsga2.dominates(tuple((1.0, 2.0)), tuple((1.0, 1.0))))
    self.assertFalse(nsga2.dominates(tuple((1.0, 2.0)), tuple((2.0, 1.0))))
    with self.assertRaises(ValueError):
      nsga2.dominates(tuple((1.0, 2.0, 3.0)), tuple((1.0, 2.0)))

  def testFastNondominatedSort(self):
    ind1 = base.set_fitness(pg.DNA(None), tuple((1.0, 1.0)))
    ind2 = base.set_fitness(pg.DNA(None), tuple((1.0, 2.0)))
    ind3 = base.set_fitness(pg.DNA(None), tuple((3.0, 4.0)))
    ind4 = base.set_fitness(pg.DNA(None), tuple((4.0, 3.0)))

    self.assertEqual(
        nsga2.nondominated_sort()([ind1, ind2]), [[ind2], [ind1]])
    self.assertEqual(
        nsga2.nondominated_sort()([ind3, ind4]), [[ind3, ind4]])
    self.assertEqual(
        nsga2.nondominated_sort()([ind1, ind2, ind3, ind4]),
        [[ind3, ind4], [ind2], [ind1]])

  def testCrowdingDistanceSort(self):
    ind1 = base.set_fitness(pg.DNA(None), tuple((100.0, 1.0)))
    ind2 = base.set_fitness(pg.DNA(None), tuple((50.0, 10.0)))
    ind3 = base.set_fitness(pg.DNA(None), tuple((30.0, 30.0)))
    ind4 = base.set_fitness(pg.DNA(None), tuple((10.0, 50.0)))
    ind5 = base.set_fitness(pg.DNA(None), tuple((1.0, 100.0)))

    self.assertEqual(
        nsga2.crowding_distance_sort()([ind1, ind2, ind3, ind4, ind5]),
        [ind1, ind5, ind2, ind4, ind3])

  def testIntegrationWithSingleObjective(self):
    """Integration test for the NSGA2 with 1 objective.

    Carries out a full search in the trivial search space.
    """
    # Set up search space.
    search_space = get_trivial_search_space()

    # Search algorithm.
    algo = nsga2.nsga2(TrivialMutator(), 20)
    self.assertTrue(algo.multi_objective)

    # Search.
    best_reward = None
    iters = 0
    start_time = time.time()
    while True:
      for example, feedback in pg.iter(search_space, 100, algo):
        reward = trivial_reward(example)
        feedback([reward])
        if best_reward is None or reward > best_reward:
          best_reward = reward
        iters += 1
        if reward >= 1.0:
          break
      if reward >= 1.0:
        break
      if time.time() - start_time > 300.0:
        self.fail('Took too long to find a solution.')

  def testIntegrationWithSimpleTwoObjective(self):
    """Integration test for the NSGA2 with 2 objectives.

    Carries out a full search in the trivial search space.
    """
    # Set up search space.
    search_space = get_trivial_search_space()

    # Set up expected pareto frontier.
    expected_pareto_frontier = simple_two_objective_pareto()

    # Search algorithm.
    algo = nsga2.nsga2(TrivialMutator(), 20)

    # Search.
    iters = 0
    start_time = time.time()
    while True:
      for example, feedback in pg.iter(search_space, 100, algo):
        reward = simple_two_objective_reward(example)
        feedback(reward)
        iters += 1
        if self._contains(
            algo.global_state.get('elites', []), expected_pareto_frontier):
          break
      if self._contains(
          algo.global_state.get('elites', []), expected_pareto_frontier):
        break
      if time.time() - start_time > 300.0:
        self.fail('Took too long to find a solution.')

  def testIntegrationWithLayeredTwoObjective(self):
    """Integration test for the NSGA2 with 2 objectives.

    Carries out a full search in the trivial search space.
    """
    # Set up search space.
    search_space = get_trivial_search_space()

    # Set up expected pareto frontier.
    expected_pareto_frontier = multi_frontier_two_objective_pareto()

    # Search algorithm.
    algo = nsga2.nsga2(TrivialMutator(), 20)

    # Search.
    iters = 0
    start_time = time.time()
    while True:
      for example, feedback in pg.iter(search_space, 100, algo):
        reward = multi_frontier_two_objective_reward(example)
        feedback(reward)
        iters += 1
        if self._contains(
            algo.global_state.get('elites', []), expected_pareto_frontier):
          break
      if self._contains(
          algo.global_state.get('elites', []), expected_pareto_frontier):
        break
      if time.time() - start_time > 300.0:
        self.fail('Took too long to find a solution.')

  def testPermanence(self):
    """Permanence test for the NSGA2."""
    search_space = get_trivial_search_space()
    algo = nsga2.nsga2(TrivialMutator(seed=1), 10, seed=1)

    # If a CL causes the following assert to fail, it means that the CL is
    # causing a difference in the behavior of the evolutionary algorithms. If
    # this is expected (e.g. a change in the random number generator), then
    # simply update the hash to the new value.
    self.assertEqual(get_trivial_hash(search_space, algo), 264567)


if __name__ == '__main__':
  unittest.main()
