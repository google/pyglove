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
"""Tests for hill climbing algorithm."""

import random
import time
import unittest

import pyglove.core as pg
from pyglove.ext.evolution import base
from pyglove.ext.evolution import hill_climb_lib as hill_climb


def get_trivial_search_space():
  """Trivial search space.

  Each point in the space is a value in [0, 1].

  Returns:
    A tunable value.
  """
  return pg.float_value(0.0, 1.0)


class TrivialMutator(base.Mutator):
  """Mutator for trivial search space.

  Mutations can only change the value by a small amount.
  """

  def mutate(self, dna, step):
    del step
    dna.value = dna.value + random.uniform(-0.01, 0.01)
    if dna.value < 0.0:
      dna.value = 0.0
    if dna.value > 1.0:
      dna.value = 1.0
    return dna


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
  for example, _ in pg.iter(search_space, 30, algo):
    hashed_value ^= int(example * 1000000)
  return hashed_value


class HillClimbingTest(unittest.TestCase):

  def testIntegration(self):
    """Integration test for the HillClimb generator.

    Carries out a full search in the trivial search space.
    """
    # Set up search space.
    search_space = get_trivial_search_space()

    # Search algorithm.
    algo = hill_climb.hill_climb(mutator=TrivialMutator(), batch_size=1)

    # Search.
    best_reward = None
    iters = 0
    start_time = time.time()
    while True:
      for example, feedback in pg.iter(search_space, 500, algo):
        reward = trivial_reward(example)
        feedback(reward)
        if best_reward is None or reward > best_reward:
          best_reward = reward
        iters += 1
        if reward >= 1.0:
          break
      if reward >= 1.0:
        break
      if time.time() - start_time > 600.0:
        self.fail('Took too long to find a solution.')

  def testPermanence(self):
    """Permanence test for the HillClimb generator."""
    search_space = get_trivial_search_space()
    algo = hill_climb.hill_climb(mutator=TrivialMutator(), batch_size=1, seed=0)

    # If a CL causes the following assert to fail, it means that the CL is
    # causing a difference in the behavior of the evolutionary algorithms. If
    # this is expected (e.g. a change in the random number generator), then
    # simply update the hash to the new value.
    self.assertEqual(get_trivial_hash(search_space, algo), 789108)


if __name__ == '__main__':
  unittest.main()
