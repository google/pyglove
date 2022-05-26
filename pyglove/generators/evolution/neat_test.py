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
"""Tests for neat."""

import random
import time
import unittest

import pyglove.core as pg
from pyglove.generators.evolution import base
from pyglove.generators.evolution import mutators
from pyglove.generators.evolution import neat_lib as neat


def get_trivial_search_space():
  """Trivial search space.

  Each point in the space is a value in [-1, 1].

  Returns:
    A tunable value.
  """
  return pg.float_value(-1.0, 1.0)


def get_hierarchical_search_space():
  """Sets up hierarchical search space to check NEAT can handle complex search spaces."""
  return pg.oneof([
      pg.oneof([0.1, 0.4, pg.oneof([0.3, 4.0, 6.0])]),
      pg.manyof(3, list(range(5)))
  ])


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
    if value < -1.0:
      value = -1.0
    if value > 1.0:
      value = 1.0
    return pg.DNA(value)


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


def get_trivial_hash(search_space, algo):
  hashed_value = 0
  for example, feedback in pg.iter(search_space, 30, algo):
    hashed_value ^= int(example * 1000000)
    feedback(example)
  return hashed_value


class NeatTest(unittest.TestCase):

  def testUtilityComputeDiff(self):
    dna_data_pairs = [((0, 0), (0, 1)), ((0, 0), (1, [0, 1, 2])), ((0, 0), 2),
                      ((1, [0, 1, 2]), (1, [0, 2, 1])), ((1, [0, 1, 2]), 2)]
    corresponding_outputs = [(2, 1, 0), (4, 1, 3), (2, 1, 1), (4, 2, 0),
                             (4, 1, 3)]
    for i, dna_data_pair in enumerate(dna_data_pairs):
      left = pg.DNA(dna_data_pair[0])
      right = pg.DNA(dna_data_pair[1])

      self.assertEqual(
          neat._compute_diff(left, right), corresponding_outputs[i])

  def testIntegration(self):
    # Set up search space.
    search_space = get_trivial_search_space()

    # Search algorithm.
    algo = neat.neat(mutator=TrivialMutator(), population_size=10)

    # Search.
    best_reward = None
    iters = 0
    start_time = time.time()

    while True:
      for example, feedback in pg.iter(search_space, 100, algo):
        reward = trivial_reward(example)
        feedback(reward)
        if best_reward is None or reward > best_reward:
          best_reward = reward
        iters += 1
        if reward >= 1.0:
          break
      if reward >= 1.0:
        break
      if time.time() - start_time > 300.0:
        self.fail('Took too long to find a solution.')

  def testHierarchicalSS(self):
    """Checks if the NEAT algorithm can loop over a complex search space."""
    search_space = get_hierarchical_search_space()
    dna_spec = pg.template(search_space).dna_spec()
    algo = neat.neat(mutator=mutators.Uniform(), population_size=10)
    algo.setup(dna_spec)

    for _ in range(100):
      dna = algo.propose()
      reward = random.uniform(-100.0, 100.0)
      algo.feedback(dna, reward)

  def testPermanence(self):
    """Permanence test for the NEAT generator."""
    search_space = get_trivial_search_space()
    algo = neat.neat(mutator=TrivialMutator(seed=1), population_size=10, seed=1)

    # If a CL causes the following assert to fail, it means that the CL is
    # causing a difference in the behavior of the evolutionary algorithms. If
    # this is expected (e.g. a change in the random number generator), then
    # simply update the hash to the new value.
    self.assertEqual(get_trivial_hash(search_space, algo), 561661)


if __name__ == '__main__':
  unittest.main()
