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
"""Tests for pyglove.geno.Random and pyglove.random_dna."""

import random
import unittest

from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import manyof
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.custom import custom
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.random import Random
from pyglove.core.geno.random import random_dna
from pyglove.core.geno.space import constant
from pyglove.core.geno.space import Space


class RandomTest(unittest.TestCase):
  """Test the `pg.geno.Random`."""

  def _dna_spec(self):
    return Space([
        # Single choice.
        oneof([
            constant(),
            constant(),
            constant()
        ]),
        # Multi-choice.
        manyof(3, [
            manyof(4, [
                constant(),
                constant(),
                constant()
            ], distinct=False, sorted=True),
            manyof(3, [
                constant(),
                constant(),
                constant()
            ], distinct=False, sorted=False),
            manyof(3, [
                constant(),
                constant(),
                constant()
            ], distinct=True, sorted=True)
        ], distinct=True, sorted=False),
        floatv(0.0, 1.0)
    ])

  def test_propose(self):
    algo = Random(seed=123)
    algo.setup(self._dna_spec())
    result = algo.propose()
    expected = DNA([
        0,                       # Single choice.
        [                        # Distinct, unsorted multi-choices.
            (1, [1, 0, 0]),      # Non-distinct, unsorted multi-choices.
            (0, [1, 1, 2, 2]),   # Non-distinct, sorted multi-choices.
            (2, [0, 1, 2])       # Distinct, sorted multi-choices.
        ],
        0.1350574593038607])
    self.assertEqual(result, expected)

  def test_recover(self):
    # Test recover scenario when random seed is provided.
    # The recovered algorithm should produce the same sequence of random
    # examples after recovery.
    algo1 = Random(seed=1)
    algo1.setup(self._dna_spec())
    dna_list = [algo1.propose() for _ in range(10)]
    algo2 = algo1.clone(deep=True)
    algo2.setup(self._dna_spec())
    algo2.recover([(dna, 0.) for dna in dna_list])
    self.assertEqual(algo1._random.getstate(), algo2._random.getstate())
    self.assertEqual(algo1.propose(), algo2.propose())


class RandomDNATest(unittest.TestCase):
  """Tests for `pg.random_dna`."""

  def test_oneof(self):
    spec = oneof([constant(), constant(), constant()])
    self.assertEqual(random_dna(spec, random.Random(1)), DNA(0))

  def test_manyof(self):
    spec = manyof(2, [constant(), constant(), constant()])
    self.assertEqual(random_dna(spec, random.Random(1)), DNA([0, 2]))

  def test_floatv(self):
    spec = floatv(min_value=0.0, max_value=1.0)
    self.assertEqual(
        random_dna(spec, random.Random(1)), DNA(0.13436424411240122))

  def test_custom_decision_point(self):
    with self.assertRaisesRegex(
        NotImplementedError, '`random_dna` is not supported'):
      _ = random_dna(custom())

  def test_custom_decision_with_random_dna_fn(self):
    def custom_random_dna_fn(random_generator, previous_dna):
      del random_generator
      if previous_dna is None:
        return DNA('abc')
      return DNA(previous_dna.value + 'x')
    spec = custom(random_dna_fn=custom_random_dna_fn)
    self.assertEqual(random_dna(spec, random.Random(1)), DNA('abc'))
    self.assertEqual(
        random_dna(spec, random.Random(1), previous_dna=DNA('abc')),
        DNA('abcx'))

  def test_complex(self):
    def custom_random_dna_fn(random_generator, previous_dna):
      del random_generator
      if previous_dna is None:
        return DNA('abc')
      return DNA(previous_dna.value + 'x')

    spec = Space([
        oneof([
            oneof([
                custom(random_dna_fn=custom_random_dna_fn),
                constant()
            ]),
            floatv(0.1, 1.0),
            constant()
        ]),
        manyof(2, [
            constant(),
            constant(),
            constant()
        ])
    ])
    dna = random_dna(
        spec, random.Random(1),
        previous_dna=DNA([(0, 0, 'xyz'), [0, 1]]))
    self.assertEqual(dna, DNA([(0, 0, 'xyzx'), [1, 0]]))


if __name__ == '__main__':
  unittest.main()
