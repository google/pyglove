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
"""Tests for pyglove.geno.Deduping."""

import unittest

from pyglove.core import symbolic
from pyglove.core import typing
from pyglove.core.geno.base import DNA
from pyglove.core.geno.deduping import Deduping
from pyglove.core.geno.dna_generator import dna_generator
from pyglove.core.geno.dna_generator import DNAGenerator
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.random import Random


@dna_generator
def dummy_generator(unused_dna_spec):
  yield DNA([0, 0])
  yield DNA([0, 0])
  yield DNA([1, 1])
  yield DNA([2, 2])
  yield DNA([1, 1])
  yield DNA([3, 0])


@symbolic.members([
    ('num_dups', typing.Int(min_value=1))
])
class DuplicatesGenerator(DNAGenerator):

  def _propose(self):
    return DNA(self.num_feedbacks // self.num_dups)

  def _feedback(self, dna, reward):
    pass


class DedupingTest(unittest.TestCase):
  """Tests for `pg.geno.Deduping`."""

  def test_default_hash_fn(self):
    dedup = Deduping(dummy_generator.partial())
    dedup.setup(None)
    self.assertEqual(
        list(iter(dedup)),
        [DNA([0, 0]), DNA([1, 1]), DNA([2, 2]),
         DNA([3, 0])])

  def test_custom_hash_fn(self):
    dedup = Deduping(
        dummy_generator.partial(),
        hash_fn=lambda x: x.children[0].value - x.children[1].value)
    dedup.setup(None)
    self.assertEqual(
        list(iter(dedup)),
        [DNA([0, 0]), DNA([3, 0])])

  def test_max_duplicates(self):
    dedup = Deduping(
        dummy_generator.partial(),
        hash_fn=lambda x: x.children[0].value - x.children[1].value,
        max_duplicates=2)
    dedup.setup(None)
    self.assertEqual(
        list(iter(dedup)),
        [DNA([0, 0]), DNA([0, 0]), DNA([3, 0])])

  def test_feedback(self):
    dedup = Deduping(DuplicatesGenerator(2))
    dedup.setup(None)
    it = iter(dedup)
    x1, f1 = next(it)
    self.assertEqual(x1, DNA(0))

    # f1 is not yet called, so the hash is not in cache yet, the generator
    # will return the same `DNA(0)`.
    x2, f2 = next(it)
    self.assertEqual(x2, DNA(0))
    f1(0)
    f2(0)

    # Both f1, f2 are called, so the next proposal will be
    # DNA(num_feedbacks / 2), which is DNA(1)
    x3, f3 = next(it)
    self.assertEqual(x3, DNA(1))

    self.assertEqual(dedup.generator.num_proposals, 3)
    self.assertEqual(dedup.generator.num_feedbacks, 2)

    # Once f3 is called, the next proposal will still be DNA(3 // 2) == DNA(1)
    # Since all subsequent call will return the same DNA, which is already
    # duplicated with x3, so StopIteration will be raised.
    f3(0)
    with self.assertRaises(StopIteration):
      _ = next(it)

    # The inner generator of dedup should have made another proposals for
    # `max_proposal_attempts` times.
    self.assertEqual(dedup.generator.num_proposals,
                     3 + dedup.max_proposal_attempts)
    self.assertEqual(dedup.generator.num_feedbacks, 3)

  def test_auto_reward(self):
    dedup = Deduping(
        DuplicatesGenerator(4), hash_fn=lambda x: x.value, auto_reward_fn=sum)
    dedup.setup(None)

    for i, (x, f) in enumerate(dedup):
      if i == 13:
        break
      # NOTE(daiyi): This logic will be taken care of by `pg.sample`.
      reward = x.value if 'reward' not in x.metadata else x.metadata['reward']
      f(reward)
    self.assertEqual(
        dedup._cache,
        {
            0: [0, 0, 0, 0],
            1: [1, 1, 2, 4],
            2: [2, 2, 4, 8],
            3: [3]
        })

  def test_recover(self):
    dedup = Deduping(Random(seed=1))
    dna_spec = floatv(0.1, 0.5)
    dedup.setup(dna_spec)

    history = []
    for i, x in enumerate(dedup):
      history.append((x, None))
      if i == 10:
        break

    dedup2 = dedup.clone(deep=True)
    dedup2.setup(dna_spec)
    dedup2.recover(history)
    self.assertEqual(
        dedup2._cache, dedup._cache)
    self.assertEqual(
        dedup2.generator._random.getstate(),
        dedup.generator._random.getstate())


if __name__ == '__main__':
  unittest.main()
