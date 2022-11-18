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
"""Base for DNA generators."""

from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

from pyglove.core import symbolic
from pyglove.core.geno.base import DNA
from pyglove.core.geno.base import DNASpec


class DNAGenerator(symbolic.Object):
  """Base class for DNA generator.

  A DNA generator is an object that produces :class:`pyglove.DNA`, and
  optionally takes feedback from the caller to improve its future proposals.

  To implement a DNA generator, the user must implement the `_propose` method,
  and can optionally override the `_setup`, `_feedback` and `_replay` methods.

   * Making proposals (Required): This method defines what to return as the
     next DNA from the generator, users MUST override the `_propose` method to
     implement this logic. `_propose` can raise `StopIteration` when no more
     DNA can be produced.

   * Custom setup (Optional): Usually a DNAGenerator subclass has its internal
     state, which can be initialized when the search space definition is
     attached to the DNAGenerator. To do so, the user can override the `_setup`
     method, in which we can access the search space definition (DNASpec object)
     via `self.dna_spec`.

   * Taking feedback (Optional): A DNAGenerator may take feedbacks from the
     caller on the fitness of proposed DNA to improve future proposals. The
     fitness is measured by a reward (a float number as the measure of a single
     objective, or a tuple of float numbers as the measure for multiple
     objectives). The user should override the `_feedback` method to implement
     such logics. If the reward is for multiple objectives. The user should
     override the `multi_objective` property to return True.

   * State recovery (Optional): DNAGenerator was designed with distributed
     computing in mind, in which a process can be preempted or killed
     unexpectedly. Therefore, a DNAGenerator should be able to recover its
     state from historical proposals and rewards. The `recover` method was
     introduced for such purpose, whose default implementation is to replay the
     history through the `_feedback` method. If the user has a custom replay
     logic other than `_feedback`, they should override the `_replay` method.
     In some use cases, the user may want to implement their own checkpointing
     logic. In such cases, the user can override the `recover` method as a
     no-op. As aside note, the `recover` method will be called by the tuning
     backend (see `tuning.py`) after `setup` but before `propose`.

  See also:

   * :class:`pyglove.geno.Sweeping`
   * :class:`pyglove.geno.Random`
   * :func:`pyglove.geno.dna_generator`
  """

  def setup(self, dna_spec: DNASpec) -> None:
    """Setup DNA spec."""
    self._dna_spec = dna_spec
    self._num_proposals = 0
    self._num_feedbacks = 0
    self._setup()

  def _setup(self) -> None:
    """Subclass should override this for adding additional setup logics."""

  @property
  def multi_objective(self) -> bool:
    """If True, current DNA generator supports multi-objective optimization."""
    return False

  @property
  def needs_feedback(self) -> bool:
    """Returns True if the DNAGenerator needs feedback."""
    return self._feedback.__code__ is not DNAGenerator._feedback.__code__  # pytype: disable=attribute-error

  @property
  def dna_spec(self) -> Optional[DNASpec]:
    return getattr(self, '_dna_spec', None)

  @property
  def num_proposals(self):
    """Get number of proposals that are already produced."""
    return self._num_proposals

  @property
  def num_feedbacks(self):
    """Get number of proposals whose feedback are provided."""
    return self._num_feedbacks

  def propose(self) -> DNA:
    """Propose a DNA to evaluate."""
    dna = self._propose()
    self._num_proposals += 1
    return dna

  def _propose(self) -> DNA:
    """Actual propose method which should be implemented by the child class."""
    raise NotImplementedError()

  def feedback(self, dna: DNA, reward: Union[float, Tuple[float]]) -> None:
    """Feedback a completed trial to the algorithm.

    Args:
      dna: a DNA object.
      reward: reward for the DNA. It is a float if `self.multi_objective`
        returns False, otherwise it's a tuple of floats.
    """
    if self.needs_feedback:
      if self.multi_objective and isinstance(reward, float):
        reward = (reward,)
      elif not self.multi_objective and isinstance(reward, tuple):
        if len(reward) != 1:
          raise ValueError(
              f'{self!r} is single objective, but the reward {reward!r} '
              f'contains multiple objectives.')
        reward = reward[0]
      self._feedback(dna, reward)
    self._num_feedbacks += 1

  def _feedback(self, dna: DNA, reward: Union[float, Tuple[float]]) -> None:
    """Actual feedback method which should be implemented by the child class.

    The default implementation is no-op.

    Args:
      dna: a DNA object.
      reward: reward for the DNA. It is a float if `self.multi_objective`
        returns False, otherwise it's a tuple of floats.
    """

  def recover(
      self,
      history: Iterable[Tuple[DNA, Union[None, float, Tuple[float]]]]
      ) -> None:
    """Recover states by replaying the proposal history.

    NOTE: `recover` will always be called before first `propose` and could be
    called multiple times if there are multiple source of history, e.g: trials
    from a previous study and existing trials from current study.

    Args:
      history: An iterable object that consists of historically proposed DNA
        with its reward. The reward will be None if it is not yet provided
        (via feedback).
    """
    for i, (dna, reward) in enumerate(history):
      self._replay(i, dna, reward)
      self._num_proposals += 1
      if reward is not None:
        self._num_feedbacks += 1

  def _replay(
      self,
      trial_id: int,
      dna: DNA,
      reward: Union[None, float, Tuple[float]]):
    """Replay a single DNA from the history for state recovery.

    The default implementation to call `DNAGenerator._feedback`. Subclasses that
    have states and can be recovered from replaying the history should override
    this method. See class `Sweeping` as an example.

    Args:
      trial_id: A zero-based integer as the trial ID for the DNA.
      dna: A historically proposed DNA.
      reward: The reward for the DNA. If None, the reward is not yet fed back
        to the optimizer.
    """
    del trial_id
    if reward is not None:
      self._feedback(dna, reward)

  def __iter__(self) -> Iterator[
      Union[DNA,
            Tuple[DNA, Callable[[Union[float, Tuple[float]]], None]]]]:
    """Iterates DNA generated from current DNAGenerator.

    NOTE(daiyip): `setup` needs to be called first before a DNAGenerator can
    be iterated.

    Yields:
      A tuple of (DNA, feedback) if current DNAGenerator requires feedback,
        otherwise DNA.
    """
    while True:
      try:
        dna = self.propose()
        if self.needs_feedback:
          feedback = lambda r: self.feedback(dna, r)
          yield (dna, feedback)
        else:
          yield dna
      except StopIteration:
        break


def dna_generator(func: Callable[[DNASpec], Iterator[DNA]]):
  """Decorator that converts a generation function to a DNAGenerator class.

  Example::

    # A DNA generator that reads DNA from file.

    def from_file(filepath):
      @pg.geno.dna_generator
      def file_based_dna_generator(dna_spec):
        dna_list = pg.load(filepath)
        for dna in dna_list:
          dna.use_spec(dna_spec)
          yield dna

      return file_based_dna_generator

  See also: :class:`pyglove.DNAGenerator`

  Args:
    func: the generation function in signature:
      `(DNASpec) -> Iterator[DNA]`

  Returns:
    A DNAGenerator class.
  """

  class SimpleDNAGenerator(DNAGenerator):
    """Simple DNA generator."""

    def _setup(self):
      self._iterator = func(self.dna_spec)
      self._error = None

    def _propose(self) -> DNA:
      if self._error is not None:
        raise ValueError(
            f'Error happened earlier: {self._error}') from self._error
      try:
        return next(self._iterator)
      except Exception as e:
        if not isinstance(e, StopIteration):
          self._error = e
        raise

  return SimpleDNAGenerator
