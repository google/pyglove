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
"""Tests for selectors."""

import unittest

import pyglove.core as pg
from pyglove.ext.evolution import base
from pyglove.ext.evolution import selectors


class SelectorsTest(unittest.TestCase):

  def test_random_selector(self):
    inputs = [pg.DNA(i) for i in range(10)]

    # Test n is an integer.
    self.assertEqual(
        selectors.Random(2, seed=1)(inputs, 0), [pg.DNA(2), pg.DNA(1)])

    # Test n is Float.
    self.assertEqual(len(selectors.Random(0.2, seed=1)(inputs)), 2)

    # Test n is None.
    self.assertEqual(len(selectors.Random(seed=1)(inputs)), 10)

    # Test random with replacement.
    self.assertEqual(
        selectors.Random(5, replacement=True, seed=1)(inputs),
        [pg.DNA(2), pg.DNA(9), pg.DNA(1), pg.DNA(4), pg.DNA(1)])

    # Test random without replacement.
    self.assertEqual(
        selectors.Random(seed=1)(inputs),
        [pg.DNA(2), pg.DNA(1), pg.DNA(4), pg.DNA(0), pg.DNA(3),
         pg.DNA(5), pg.DNA(7), pg.DNA(9), pg.DNA(8), pg.DNA(6)])

    # Test scheduled N.
    selector = selectors.Random(lambda step: 1 if step == 0 else 2, seed=1)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(2)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(9), pg.DNA(1)])

  def test_sample_selector(self):
    inputs = [pg.DNA(i) for i in range(5)]
    weights = lambda x: [0.13, 0.23, 0., 0., 0.26]

    # Test n is an integer.
    self.assertEqual(
        selectors.Sample(10, weights, seed=1)(inputs, 0),
        [pg.DNA(0), pg.DNA(4), pg.DNA(4), pg.DNA(1), pg.DNA(1),
         pg.DNA(1), pg.DNA(4), pg.DNA(4), pg.DNA(0), pg.DNA(0)])

    # Test n is Float.
    self.assertEqual(len(selectors.Sample(0.5, weights, seed=1)(inputs)), 3)

    # Test n is None.
    self.assertEqual(len(selectors.Sample(weights=weights, seed=1)(inputs)), 5)

    # Test scheduled N.
    selector = selectors.Sample(
        lambda step: 1 if step == 0 else 2, weights, seed=1)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(0)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(4), pg.DNA(4)])

  def test_proportional_selector(self):
    inputs = [pg.DNA(i) for i in range(5)]
    weights = lambda x: [0.13, 0.23, 0., 0., 0.26]

    # Test n is an integer.
    self.assertEqual(
        selectors.Proportional(10, weights)(inputs, 0),
        [pg.DNA(0), pg.DNA(0), pg.DNA(1), pg.DNA(1), pg.DNA(1),
         pg.DNA(1), pg.DNA(4), pg.DNA(4), pg.DNA(4), pg.DNA(4)])

    # Test n is Float.
    self.assertEqual(len(selectors.Proportional(0.5, weights)(inputs)), 3)

    # Test n is None.
    self.assertEqual(
        len(selectors.Proportional(weights=weights)(inputs)), 5)

    # Test scheduled N.
    selector = selectors.Proportional(
        lambda step: 1 if step == 0 else 2, weights)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(4)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(1), pg.DNA(4)])

    # Test `_partition` method.
    self.assertEqual(
        selector._partition([0.11, 0.26, 0.0, 0.0, 1.2], 5),
        [0, 1, 0, 0, 4])
    self.assertEqual(
        selector._partition([0.01, 0.0, 0.0, 0.0, 1.2], 5),
        [0, 0, 0, 0, 5])
    # `extra` == 1.
    self.assertEqual(selector._partition([0.11, 0.11, 0.11], 3), [1, 1, 1])
    # `extra` > 1.
    self.assertEqual(selector._partition([0.314, 0.288, 0.052], 5), [3, 2, 0])
    # `extra` < 1.
    self.assertEqual(selector._partition([0.109, 0.506, 0.297], 5), [0, 3, 2])

  def test_top_selector(self):
    inputs = [base.set_fitness(pg.DNA(i), float(i) if i % 2 else -float(i))
              for i in range(10)]

    # Test n is an integer.
    self.assertEqual(selectors.Top(2)(inputs), [pg.DNA(9), pg.DNA(7)])

    # Test n is a float number.
    self.assertEqual(selectors.Top(0.1)(inputs), [pg.DNA(9)])

    # Test n is None
    self.assertEqual(
        selectors.Top(None)(inputs),
        [
            pg.DNA(9), pg.DNA(7), pg.DNA(5), pg.DNA(3), pg.DNA(1),
            pg.DNA(0), pg.DNA(2), pg.DNA(4), pg.DNA(6), pg.DNA(8),
        ])

    # Test scheduled N.
    selector = selectors.Top(lambda step: 1 if step == 0 else 2)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(9)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(9), pg.DNA(7)])

    # Test custom key.
    selector = selectors.Top(2, key=lambda dna: dna.value)
    self.assertEqual(selector(inputs), [pg.DNA(9), pg.DNA(8)])

    # Test cluster
    inputs = [
        base.set_fitness(pg.DNA(i), float(int(i / 2))) for i in range(10)]
    selector = selectors.Top(1, cluster=True)
    self.assertEqual(selector(inputs), [pg.DNA(8), pg.DNA(9)])

    # Test custom type.
    self.assertEqual(selectors.Top(1)([1, 0, 2]), [2])

  def test_bottom_selector(self):
    inputs = [base.set_fitness(pg.DNA(i), float(i) if i % 2 else -float(i))
              for i in range(10)]

    # Test n is an integer.
    self.assertEqual(selectors.Bottom(2)(inputs), [pg.DNA(8), pg.DNA(6)])

    # Test n is a float number.
    self.assertEqual(selectors.Bottom(0.1)(inputs), [pg.DNA(8)])

    # Test n is None
    self.assertEqual(
        selectors.Bottom(None)(inputs),
        [
            pg.DNA(8), pg.DNA(6), pg.DNA(4), pg.DNA(2), pg.DNA(0),
            pg.DNA(1), pg.DNA(3), pg.DNA(5), pg.DNA(7), pg.DNA(9),
        ])

    # Test scheduled N.
    selector = selectors.Bottom(lambda step: 1 if step == 0 else 2)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(8)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(8), pg.DNA(6)])

    # Test custom key.
    selector = selectors.Bottom(2, key=lambda dna: dna.value)
    self.assertEqual(selector(inputs), [pg.DNA(0), pg.DNA(1)])

    # Test cluster
    inputs = [
        base.set_fitness(pg.DNA(i), float(int(i / 2))) for i in range(10)]
    selector = selectors.Bottom(1, cluster=True)
    self.assertEqual(selector(inputs), [pg.DNA(0), pg.DNA(1)])

    # Test custom type.
    self.assertEqual(selectors.Bottom(1)([1, 0, 2]), [0])

  def test_first_selector(self):
    inputs = [pg.DNA(i) for i in range(10)]

    # Test n is an integer.
    self.assertEqual(selectors.First(2)(inputs), [pg.DNA(0), pg.DNA(1)])

    # Test n is a float number.
    self.assertEqual(selectors.First(0.001)(inputs), [pg.DNA(0)])

    # Test n is None
    self.assertEqual(selectors.First(None)(inputs), inputs)

    # Test scheduled N.
    selector = selectors.First(lambda step: 1 if step == 0 else 2)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(0)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(0), pg.DNA(1)])

  def test_last_selector(self):
    inputs = [pg.DNA(i) for i in range(10)]

    # Test n is an integer.
    self.assertEqual(selectors.Last(2)(inputs), [pg.DNA(8), pg.DNA(9)])

    # Test n is a float number.
    self.assertEqual(selectors.Last(0.001)(inputs), [pg.DNA(9)])

    # Test n is None
    self.assertEqual(selectors.Last(None)(inputs), inputs)

    # Test scheduled N.
    selector = selectors.Last(lambda step: 1 if step == 0 else 2)
    self.assertEqual(selector(inputs, step=0), [pg.DNA(9)])
    self.assertEqual(selector(inputs, step=1), [pg.DNA(8), pg.DNA(9)])


if __name__ == '__main__':
  unittest.main()
