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
"""Tests for pyglove.geno.Sweeping."""

import unittest

from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import manyof
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.space import constant
from pyglove.core.geno.space import Space
from pyglove.core.geno.sweeping import Sweeping


class SweepingTest(unittest.TestCase):
  """Test the `pg.geno.Sweeping`."""

  def _dna_spec(self):
    return Space([
        # Single choice.
        oneof([
            manyof(2, [constant(), constant(), constant()]),
            oneof([constant(), constant()])
        ]),
        manyof(2, [constant(), constant(), constant()], sorted=True),
    ])

  def test_propose(self):
    algo = Sweeping()
    algo.setup(self._dna_spec())
    results = []
    while True:
      try:
        results.append(algo.propose())
      except StopIteration:
        break

    self.assertEqual(results, [
        DNA([(0, [0, 1]), [0, 1]]),
        DNA([(0, [0, 1]), [0, 2]]),
        DNA([(0, [0, 1]), [1, 2]]),
        DNA([(0, [0, 2]), [0, 1]]),
        DNA([(0, [0, 2]), [0, 2]]),
        DNA([(0, [0, 2]), [1, 2]]),
        DNA([(0, [1, 0]), [0, 1]]),
        DNA([(0, [1, 0]), [0, 2]]),
        DNA([(0, [1, 0]), [1, 2]]),
        DNA([(0, [1, 2]), [0, 1]]),
        DNA([(0, [1, 2]), [0, 2]]),
        DNA([(0, [1, 2]), [1, 2]]),
        DNA([(0, [2, 0]), [0, 1]]),
        DNA([(0, [2, 0]), [0, 2]]),
        DNA([(0, [2, 0]), [1, 2]]),
        DNA([(0, [2, 1]), [0, 1]]),
        DNA([(0, [2, 1]), [0, 2]]),
        DNA([(0, [2, 1]), [1, 2]]),
        DNA([(1, 0), [0, 1]]),
        DNA([(1, 0), [0, 2]]),
        DNA([(1, 0), [1, 2]]),
        DNA([(1, 1), [0, 1]]),
        DNA([(1, 1), [0, 2]]),
        DNA([(1, 1), [1, 2]])
    ])

  def test_recover(self):
    algo1 = Sweeping()
    algo1.setup(self._dna_spec())
    dna_list = [algo1.propose() for _ in range(10)]
    algo2 = algo1.clone(deep=True)
    algo2.setup(self._dna_spec())
    algo2.recover([(dna, 0.) for dna in dna_list])
    self.assertEqual(algo1.propose(), algo2.propose())


if __name__ == '__main__':
  unittest.main()
