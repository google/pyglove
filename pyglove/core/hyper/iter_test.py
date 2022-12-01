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
"""Tests for pyglove.hyper.Float."""

import unittest

from pyglove.core import geno
from pyglove.core.hyper.categorical import oneof
from pyglove.core.hyper.dynamic_evaluation import trace as pg_trace
from pyglove.core.hyper.iter import iterate as pg_iterate
from pyglove.core.hyper.iter import random_sample as pg_random_sample


class IterateTest(unittest.TestCase):
  """Tests for pg.iter."""

  def test_iter_with_default_algorithm(self):
    v = oneof(range(100))
    examples = list(pg_iterate(v))
    self.assertEqual(examples, list(range(100)))

    examples = list(pg_iterate(v, 10))
    self.assertEqual(examples, list(range(10)))

  def test_iter_with_custom_algorithm(self):

    class ConstantAlgorithm(geno.DNAGenerator):

      def _on_bound(self):
        self._rewards = []

      def _propose(self):
        if len(self._rewards) == 100:
          raise StopIteration()
        return geno.DNA(0)

      def _feedback(self, dna, reward):
        self._rewards.append(reward)

      @property
      def rewards(self):
        return self._rewards

    algo = ConstantAlgorithm()
    examples = []
    for i, (x, feedback) in enumerate(pg_iterate(oneof([1, 3]), 5, algo)):
      examples.append(x)
      feedback(float(i))
      self.assertEqual(feedback.dna, geno.DNA(0))
    self.assertEqual(len(examples), 5)
    self.assertEqual(examples, [1] * 5)
    self.assertEqual(algo.rewards, [float(i) for i in range(5)])

    for x, feedback in pg_iterate(oneof([1, 3]), algorithm=algo):
      examples.append(x)
      feedback(0.)
    self.assertEqual(len(examples), 100)

  def test_iter_with_dynamic_evaluation(self):
    def foo():
      return oneof([1, 3])
    examples = []
    for x in pg_iterate(pg_trace(foo)):
      with x():
        examples.append(foo())
    self.assertEqual(examples, [1, 3])

  def test_iter_with_continuation(self):

    class ConstantAlgorithm3(geno.DNAGenerator):

      def setup(self, dna_spec):
        super().setup(dna_spec)
        self.num_trials = 0

      def propose(self):
        self.num_trials += 1
        return geno.DNA(0)

    algo = ConstantAlgorithm3()
    for unused_x in pg_iterate(oneof([1, 3]), 10, algo):
      pass
    for unused_x in pg_iterate(oneof([1, 3]), 10, algo):
      pass
    self.assertEqual(algo.num_trials, 20)

  def test_iter_with_forced_feedback(self):

    class ConstantAlgorithm2(geno.DNAGenerator):

      def propose(self):
        return geno.DNA(0)

    algo = ConstantAlgorithm2()
    examples = []
    for x, feedback in pg_iterate(
        oneof([1, 3]), 10, algorithm=algo, force_feedback=True):
      examples.append(x)
      # No op.
      feedback(0.)
    self.assertEqual(len(examples), 10)

  def test_bad_iter(self):
    with self.assertRaisesRegex(
        ValueError, '\'hyper_value\' is a constant value'):
      next(pg_iterate('foo'))

    algo = geno.Random()
    next(pg_iterate(oneof([1, 2]), 1, algo))
    with self.assertRaisesRegex(
        ValueError, '.* has been set up with a different DNASpec'):
      next(pg_iterate(oneof([2, 3]), 10, algo))


class RandomSampleTest(unittest.TestCase):
  """Tests for pg.random_sample."""

  def test_random_sample(self):
    self.assertEqual(
        list(pg_random_sample(oneof([0, 1]), 3, seed=123)), [0, 1, 0])


if __name__ == '__main__':
  unittest.main()
