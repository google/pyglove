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
"""Solving the One-Max problem with custom encoding.

Reference: https://tracer.lcc.uma.es/problems/onemax/onemax.html#SE91
For more details, see:
https://colab.research.google.com/github/google/pyglove/blob/main/docs/notebooks/evolution/onemax.ipynb
"""

import random
import pyglove as pg


def one_max(search_space, search_algorithm, num_trials=200):
  """Solves the One-Max program through running a search."""
  best_sequence, best_reward = None, None
  for sequence, feedback in pg.sample(
      search_space, search_algorithm,
      num_examples=num_trials):
    reward = sum(sequence)
    if best_reward is None or best_reward < reward:
      best_sequence, best_reward = sequence, reward
    feedback(reward)
  print(f'Best sequence: {list(best_sequence)} (sum={best_reward})')


def one_max_with_builtin_primitive(n: int):
  """Solves One-Max problem using built-in hyper primitive."""
  search_space = pg.List([pg.oneof([0, 1])] * n)
  search_algorithm = pg.evolution.regularized_evolution(
      population_size=20, tournament_size=10)
  one_max(search_space, search_algorithm)


def one_max_with_custom_primitive(n: int):
  """Sovles One-Max problem using user-defined hyper primitive."""

  class BitString(pg.hyper.CustomHyper):
    """Custom hyper primitive that represents a bit string of size n."""

    def custom_decode(self, dna: pg.DNA):
      assert isinstance(dna.value, str)
      bitstr = dna.value
      return [int(x) for x in bitstr]

  class MutateOneBit(pg.evolution.Mutator):

    def mutate(self, dna: pg.DNA):
      bitstr = dna.value
      index = random.randint(0, len(dna.value) - 1)
      new_bitstr = (
          bitstr[:index]
          + ('0' if bitstr[index] == '1' else '1')
          + bitstr[index + 1:])
      return pg.DNA(new_bitstr)

  def init_population(population_size):
    @pg.geno.dna_generator
    def initializer(dna_spec):
      del dna_spec
      for _ in range(population_size):
        bits = [str(random.randint(0, 1)) for _ in range(n)]
        yield pg.DNA(''.join(bits))
    return initializer()  # pylint: disable=no-value-for-parameter

  search_space = BitString()
  search_algorithm = pg.evolution.Evolution(
      (pg.evolution.selectors.Random(10)
       >> pg.evolution.selectors.Top(1)
       >> MutateOneBit()),
      population_init=init_population(10),
      population_update=pg.evolution.selectors.Last(20))
  one_max(search_space, search_algorithm)


def main():
  one_max_with_builtin_primitive(10)
  one_max_with_custom_primitive(10)


if __name__ == '__main__':
  main()
