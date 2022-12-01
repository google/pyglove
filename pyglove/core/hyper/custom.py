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
"""Custom hyper primitives."""

import abc
import random
import types
from typing import Any, Callable, Optional, Tuple, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.hyper import base


class CustomHyper(base.HyperPrimitive):
  """User-defined hyper primitive.

  User-defined hyper primitive is useful when users want to have full control
  on the semantics and genome encoding of the search space. For example, the
  decision points are of variable length, which is not yet supported by
  built-in hyper primitives.

  To use user-defined hyper primitive is simple, the user should:


  1) Subclass `CustomHyper` and implements the
     :meth:`pyglove.hyper.CustomHyper.custom_decode` method.
     It's optional to implement the
     :meth:`pyglove.hyper.CustomHyper.custom_encode` method, which is only
     necessary when the user want to encoder a material object into a DNA.
  2) Introduce a DNAGenerator that can generate DNA for the
     :class:`pyglove.geno.CustomDecisionPoint`.

  For example, the following code tries to find an optimal sub-sequence of an
  integer sequence by their sums::

    import random

    class IntSequence(pg.hyper.CustomHyper):

      def custom_decode(self, dna):
        return [int(v) for v in dna.value.split(',') if v != '']

    class SubSequence(pg.evolution.Mutator):

      def mutate(self, dna):
        genome = dna.value
        items = genome.split(',')
        start = random.randint(0, len(items))
        end = random.randint(start, len(items))
        new_genome = ','.join(items[start:end])
        return pg.DNA(new_genome, spec=dna.spec)

    @pg.geno.dna_generator
    def initial_population():
      yield pg.DNA('12,-34,56,-2,100,98', spec=dna_spec)

    algo = pg.evolution.Evolution(
        (pg.evolution.selectors.Random(10)
         >> pg.evolution.selectors.Top(1)
         >> SubSequence()),
        population_init=initial_population(),
        population_update=pg.evolution.selectors.Last(20))

    best_reward, best_example = None, None
    for int_seq, feedback in pg.sample(IntSequence(), algo, num_examples=100):
      reward = sum(int_seq)
      if best_reward is None or best_reward < reward:
        best_reward, best_example = reward, int_seq
      feedback(reward)

    print(best_reward, best_example)

  Please note that user-defined hyper value can be used together with PyGlove's
  built-in hyper primitives, for example::

    pg.oneof([IntSequence(), None])

  Therefore it's also a mechanism to extend PyGlove's search space definitions.
  """

  def _decode(self):
    if not isinstance(self.dna.value, str):
      raise ValueError(
          f'{self.__class__} expects string type DNA. '
          f'Encountered {self.dna!r}.')
    return self.custom_decode(self.dna)

  @abc.abstractmethod
  def custom_decode(self, dna: geno.DNA) -> Any:
    """Decode a DNA whose value is a string of user-defined genome."""

  def encode(self, value: Any) -> geno.DNA:
    """Encode value into DNA with user-defined genome."""
    return self.custom_encode(value)

  def custom_encode(self, value: Any) -> geno.DNA:
    """Encode value to user defined genome."""
    raise NotImplementedError(
        f'\'custom_encode\' is not supported by {self.__class__.__name__!r}.')

  def dna_spec(
      self, location: Optional[object_utils.KeyPath] = None) -> geno.DNASpec:
    """Always returns CustomDecisionPoint for CustomHyper."""
    return geno.CustomDecisionPoint(
        hyper_type=self.__class__.__name__,
        next_dna_fn=self.next_dna,
        random_dna_fn=self.random_dna,
        hints=self.hints, name=self.name, location=location)

  def first_dna(self) -> geno.DNA:
    """Returns the first DNA of current sub-space.

    Returns:
      A string-valued DNA.
    """
    if self.next_dna.__code__ is CustomHyper.next_dna.__code__:
      raise NotImplementedError(
          f'{self.__class__!r} must implement method `next_dna` to be used in '
          f'dynamic evaluation mode.')
    return self.next_dna(None)

  def next_dna(self, dna: Optional[geno.DNA] = None) -> Optional[geno.DNA]:
    """Subclasses should override this method to support pg.Sweeping."""
    raise NotImplementedError(
        f'`next_dna` is not implemented in f{self.__class__!r}')

  def random_dna(
      self,
      random_generator: Union[types.ModuleType, random.Random, None] = None,
      previous_dna: Optional[geno.DNA] = None) -> geno.DNA:
    """Subclasses should override this method to support pg.random_dna."""
    raise NotImplementedError(
        f'`random_dna` is not implemented in {self.__class__!r}')

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
      ) -> Tuple[bool, 'CustomHyper']:
    """Validate candidates during value_spec binding time."""
    del path, value_spec, allow_partial, child_transform
    # Allow custom hyper to be assigned to any type.
    return (False, self)
