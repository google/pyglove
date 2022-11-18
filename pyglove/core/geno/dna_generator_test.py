# Copyright 2022 The PyGlove Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain algo copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pyglove.geno.DNA."""

import unittest

from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.dna_generator import dna_generator
from pyglove.core.geno.dna_generator import DNAGenerator
from pyglove.core.geno.space import constant


class DNAGeneratorTest(unittest.TestCase):
  """Tests for DNAGenerator base."""

  def _dna_spec(self):
    return oneof([constant(), constant()])

  def test_generator_without_feedback(self):

    # Test for DNAGenerator that does not need feedbacks.
    class DummyGenerator(DNAGenerator):

      def _propose(self):
        return DNA(0)

    algo = DummyGenerator()
    algo.setup(self._dna_spec)
    self.assertFalse(algo.needs_feedback)
    self.assertFalse(algo.multi_objective)
    self.assertEqual(algo.num_proposals, 0)
    dna = algo.propose()
    self.assertEqual(dna, DNA(0))
    self.assertEqual(algo.num_proposals, 1)
    self.assertEqual(algo.num_feedbacks, 0)
    algo.feedback(dna, 0)
    self.assertEqual(algo.num_feedbacks, 1)

  def test_single_objective_optimizer(self):

    class DummySingleObjectiveOptimizer(DNAGenerator):

      def _setup(self):
        self.max_reward = None

      def _propose(self):
        return DNA(1)

      def _feedback(self, dna, reward):
        if self.max_reward is None or reward > self.max_reward:
          self.max_reward = reward

    algo = DummySingleObjectiveOptimizer()
    algo.setup(self._dna_spec)
    self.assertTrue(algo.needs_feedback)
    self.assertFalse(algo.multi_objective)
    self.assertEqual(algo.num_proposals, 0)
    dna = algo.propose()
    self.assertEqual(dna, DNA(1))
    self.assertEqual(algo.num_proposals, 1)
    self.assertEqual(algo.num_feedbacks, 0)
    algo.feedback(dna, 1.2)
    self.assertEqual(algo.num_feedbacks, 1)
    self.assertEqual(algo.max_reward, 1.2)

    with self.assertRaisesRegex(
        ValueError,
        '.* is single objective, but the reward .*'
        'contains multiple objectives'):
      algo.feedback(dna, (0, 1))

  def test_multi_objective_optimizer(self):

    class DummyMultiObjectiveOptimizer(DNAGenerator):

      @property
      def multi_objective(self):
        return True

      def _setup(self):
        self.rewards = []

      def _propose(self):
        return DNA(1)

      def _feedback(self, dna, reward):
        self.rewards.append(reward)

    algo = DummyMultiObjectiveOptimizer()
    algo.setup(self._dna_spec)
    self.assertTrue(algo.needs_feedback)
    self.assertTrue(algo.multi_objective)
    self.assertEqual(algo.num_proposals, 0)
    dna = algo.propose()
    self.assertEqual(dna, DNA(1))
    self.assertEqual(algo.num_proposals, 1)
    self.assertEqual(algo.num_feedbacks, 0)
    algo.feedback(dna, (0.9, 1.2))
    self.assertEqual(algo.num_feedbacks, 1)
    self.assertEqual(algo.rewards, [(0.9, 1.2)])
    algo.feedback(dna, 1.)
    self.assertEqual(algo.num_feedbacks, 2)
    self.assertEqual(algo.rewards, [(0.9, 1.2), (1.,)])


class DNAGeneratorDecoratorTest(unittest.TestCase):

  def test_dna_generator_decorator(self):

    @dna_generator
    def first_ten(dna_spec):
      dna = None
      for _ in range(10):
        dna = dna_spec.next_dna(dna)
        if dna is None:
          break
        yield dna

    algo = first_ten()    # pylint: disable=no-value-for-parameter
    dna_spec = oneof([constant()] * 8)
    algo.setup(dna_spec)

    dna_list = []
    with self.assertRaises(StopIteration):
      while True:
        dna_list.append(algo.propose())
    self.assertEqual(
        dna_list,
        [DNA(i) for i in range(8)])

    # Test error preservation from early proposal.
    @dna_generator
    def bad_generator(unused_spec):
      if True:  # pylint: disable=using-constant-test
        raise ValueError('bad initializer')
      yield DNA(0)

    algo = bad_generator.partial()
    algo.setup(None)

    with self.assertRaisesRegex(
        ValueError, 'bad initializer'):
      algo.propose()

    with self.assertRaisesRegex(
        ValueError, 'Error happened earlier: bad initializer'):
      algo.propose()


if __name__ == '__main__':
  unittest.main()
