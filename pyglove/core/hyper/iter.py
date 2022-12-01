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
"""Iterating hyper values."""

from typing import Any, Callable, Optional, Tuple, Union

from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core.hyper import base
from pyglove.core.hyper import dynamic_evaluation
from pyglove.core.hyper import object_template


def iterate(hyper_value: Any,
            num_examples: Optional[int] = None,
            algorithm: Optional[geno.DNAGenerator] = None,
            where: Optional[Callable[[base.HyperPrimitive], bool]] = None,
            force_feedback: bool = False):
  """Iterate a hyper value based on an algorithm.

  Example::

    hyper_dict = pg.Dict(x=pg.oneof([1, 2, 3]), y=pg.oneof(['a', 'b']))

    # Get all examples from the hyper_dict.
    assert list(pg.iter(hyper_dict)) == [
        pg.Dict(x=1, y='a'),
        pg.Dict(x=1, y='b'),
        pg.Dict(x=2, y='a'),
        pg.Dict(x=2, y='b'),
        pg.Dict(x=3, y='a'),
        pg.Dict(x=3, y='b'),
    ]

    # Get the first two examples.
    assert list(pg.iter(hyper_dict, 2)) == [
        pg.Dict(x=1, y='a'),
        pg.Dict(x=1, y='b'),
    ]

    # Random sample examples, which is equivalent to `pg.random_sample`.
    list(pg.iter(hyper_dict, 2, pg.geno.Random()))

    # Iterate examples with feedback loop.
    for d, feedback in pg.iter(
        hyper_dict, 10,
        pg.evolution.regularized_evolution(pg.evolution.mutators.Uniform())):
      feedback(d.x)

    # Only materialize selected parts.
    assert list(
        pg.iter(hyper_dict, where=lambda x: len(x.candidates) == 2)) == [
            pg.Dict(x=pg.oneof([1, 2, 3]), y='a'),
            pg.Dict(x=pg.oneof([1, 2, 3]), y='b'),
        ]

  ``pg.iter`` distinguishes from `pg.sample` in that it's designed
  for simple in-process iteration, which is handy for quickly generating
  examples from algorithms without maintaining trail states. On the contrary,
  `pg.sample` is designed for distributed sampling, with parallel workers and
  failover handling.

  Args:
    hyper_value: A hyper value that represents a space of instances.
    num_examples: An optional integer as the max number of examples to
        propose. If None, propose will return an iterator of infinite examples.
    algorithm: An optional DNA generator. If None, Sweeping will be used, which
        iterates examples in order.
    where: Function to filter hyper primitives. If None, all hyper primitives
      from `value` will be included in the encoding/decoding process. Otherwise
      only the hyper primitives on which 'where' returns True will be included.
      `where` can be useful to partition a search space into separate
      optimization processes. Please see 'Template' docstr for details.
    force_feedback: If True, always return the Feedback object together
      with the example, this is useful when the user want to pass different
      DNAGenerators to `pg.iter` and want to handle them uniformly.

  Yields:
    A tuple of (example, feedback_fn) if the algorithm needs a feedback or
    `force_feedback` is True, otherwise the example.

  Raises:
    ValueError: when `hyper_value` is a constant value.
  """
  if isinstance(hyper_value, dynamic_evaluation.DynamicEvaluationContext):
    dynamic_evaluation_context = hyper_value
    spec = hyper_value.dna_spec
    t = None
  else:
    t = object_template.template(hyper_value, where)
    if t.is_constant:
      raise ValueError(
          f'\'hyper_value\' is a constant value: {hyper_value!r}.')
    dynamic_evaluation_context = None
    spec = t.dna_spec()

  if algorithm is None:
    algorithm = geno.Sweeping()

  # NOTE(daiyip): algorithm can continue if it's already set up with the same
  # DNASpec, or we will setup the algorithm with the DNASpec from the template.
  if algorithm.dna_spec is None:
    algorithm.setup(spec)
  elif symbolic.ne(spec, algorithm.dna_spec):
    raise ValueError(
        f'{algorithm!r} has been set up with a different DNASpec. '
        f'Existing: {algorithm.dna_spec!r}, New: {spec!r}.')

  count = 0
  while num_examples is None or count < num_examples:
    try:
      count += 1
      dna = algorithm.propose()
      if t is not None:
        example = t.decode(dna)
      else:
        assert dynamic_evaluation_context is not None
        example = lambda: dynamic_evaluation_context.apply(dna)
      if force_feedback or algorithm.needs_feedback:
        yield example, Feedback(algorithm, dna)
      else:
        yield example
    except StopIteration:
      return


class Feedback:
  """Feedback object."""

  def __init__(self, algorithm: geno.DNAGenerator, dna: geno.DNA):
    """Creates a feedback object."""
    self._algorithm = algorithm
    self._dna = dna

  def __call__(self, reward: Union[float, Tuple[float, ...]]):
    """Call to feedback reward."""
    self._algorithm.feedback(self._dna, reward)

  @property
  def dna(self) -> geno.DNA:
    """Returns DNA."""
    return self._dna


def random_sample(
    value: Any,
    num_examples: Optional[int] = None,
    where: Optional[Callable[[base.HyperPrimitive], bool]] = None,
    seed: Optional[int] = None):
  """Returns an iterator of random sampled examples.

  Example::

    hyper_dict = pg.Dict(x=pg.oneof(range(3)), y=pg.floatv(0.0, 1.0))

    # Generate one random example from the hyper_dict.
    d = next(pg.random_sample(hyper_dict))

    # Generate 5 random examples with random seed.
    ds = list(pg.random_sample(hyper_dict, 5, seed=1))

    # Generate 3 random examples of `x` with `y` intact.
    ds = list(pg.random_sample(hyper_dict, 3,
        where=lambda x: isinstance(x, pg.hyper.OneOf)))


  Args:
    value: A (maybe) hyper value.
    num_examples: An optional integer as number of examples to propose. If None,
      propose will return an iterator that iterates forever.
    where: Function to filter hyper primitives. If None, all hyper primitives in
      `value` will be included in the encoding/decoding process. Otherwise only
      the hyper primitives on which 'where' returns True will be included.
      `where` can be useful to partition a search space into separate
      optimization processes. Please see 'Template' docstr for details.
    seed: An optional integer as random seed.

  Returns:
    Iterator of random examples.
  """
  return iterate(
      value, num_examples, geno.Random(seed), where=where)
