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
"""Deduping DNA generator."""

from typing import Any, Tuple, Union

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.geno.base import DNA
from pyglove.core.geno.dna_generator import DNAGenerator


@symbolic.members([
    ('generator', pg_typing.Object(DNAGenerator),
     'Inner generator, whose proposal will be deduped.'),
    ('hash_fn', pg_typing.Callable(
        [pg_typing.Object(DNA)], returns=pg_typing.Int()).noneable(),
     'Hashing function. If None, the hash will be based on DNA values.'),
    ('auto_reward_fn', pg_typing.Callable(
        [pg_typing.List(
            pg_typing.Union([pg_typing.Float(),
                             pg_typing.Tuple(pg_typing.Float())]))],
        returns=pg_typing.Union(
            [pg_typing.Float(),
             pg_typing.Tuple(pg_typing.Float())])).noneable(),
     ('If None, duplicated proposals above the `max_duplicates` limit will be '
      'dropped. Otherwise, its reward will be automatically computed from the '
      'rewards of previous duplicates, without the client to evaluate it.')),
    ('max_duplicates', pg_typing.Int(min_value=1, default=1),
     'Max number of duplicates allowed per entry.'),
    ('max_proposal_attempts', pg_typing.Int(min_value=1, default=100),
     'Max number of attempts if duplicated entries are encountered.')
])
class Deduping(DNAGenerator):
  """Deduping generator.

  A deduping generator can be applied on another generator to dedup its
  proposed DNAs.

  **Hash function**

  By default, the hash function is the symbolic hash of the DNA, which returns
  the same hash when the decisions from the DNA are the same.

  For example::

    pg.geno.Deduping(pg.geno.Random())

  will only generate unique DNAs. When `hash_fn` is specified, it allows the
  user to compute the hash for a DNA.

  For example::

    pg.geno.Deduping(pg.geno.Random(),
                     hash_fn=lambda dna: sum(dna.to_numbers()))

  will dedup based on the sum of all decision values.

  **Number of duplicates**

  An optional `max_duplicates` can be provided by the user to allow a few
  duplicates.

  For example::

    pg.geno.Deduping(pg.geno.Random(), max_duplicates=5)

  Note: for inner DNAGenerators that requires user feedback, duplication
  accounting is based on DNAs that are fed back to the DNAGenerator, not
  proposed ones.

  **Automatic reward computation**

  Automatic reward computation will be enabled when `auto_reward_fn` is
  provided AND when the inner generator takes feedback. It allows users to
  compute the reward for new duplicates (which exceed the `max_duplicates`
  limit) by aggregating rewards from previous duplicates. Such DNAs will be
  fed back to the DNAGenerator without client's evaluation (supported by
  `pg.sample` through the 'reward' metadata).

  For example::

    pg.geno.Deduping(pg.evolution.regularized_evolution(),
                     auto_reward_fn=lambda rs: sum(rs) / len(rs))
  """

  @property
  def needs_feedback(self) -> bool:
    return self.generator.needs_feedback

  def _setup(self):
    self.generator.setup(self.dna_spec)
    self._hash_fn = self.hash_fn or symbolic.hash
    self._cache = {}
    self._enables_auto_reward = (
        self.needs_feedback and self.auto_reward_fn is not None)

  def _propose(self):
    """Proposes a deduped DNA."""
    attempts = 0
    while attempts < self.max_proposal_attempts:
      dna = self.generator.propose()
      hash_key = self._hash_fn(dna)
      history = self._cache.get(hash_key, [])
      dna.set_metadata('dedup_key', hash_key)
      if len(history) < self.max_duplicates:
        break
      elif self._enables_auto_reward:
        reward = self.auto_reward_fn(history)
        dna.set_metadata('reward', reward)
        break
      attempts += 1
    if attempts == self.max_proposal_attempts:
      raise StopIteration()
    if not self.needs_feedback:
      self._add_dna_to_cache(dna, None)
    return dna

  def _feedback(self, dna: DNA, reward: Union[float, Tuple[float]]) -> None:
    self.generator.feedback(dna, reward)
    self._add_dna_to_cache(dna, reward)

  def _replay(self, trial_id: int, dna: DNA, reward: Any) -> None:
    self.generator._replay(trial_id, dna, reward)  # pylint: disable=protected-access
    self._add_dna_to_cache(dna, reward)

  def _add_dna_to_cache(
      self, dna: DNA, reward: Union[None, float, Tuple[float]]) -> None:
    hash_key = dna.metadata.get('dedup_key', None)
    assert hash_key is not None, dna
    if hash_key not in self._cache:
      self._cache[hash_key] = []
    self._cache[hash_key].append(reward)
