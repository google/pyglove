# Copyright 2021 The PyGlove Authors
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
"""Common selectors for evolutionary algorithms."""

import math
import random
from typing import Any, List, Union

import pyglove.core as pg
from pyglove.generators.evolution import base
from pyglove.generators.evolution import scalars


# We disable implicit str concat as it is commonly used class schema docstr.
# pylint: disable=implicit-str-concat


def num_output_spec():
  """Returns the value spec for the number of elements in the output."""
  return scalars.scalar_spec(pg.typing.Union([
      pg.typing.Int(min_value=0),                      # Output count.
      pg.typing.Float(min_value=0.0, max_value=1.0)    # Output proportion.
  ])).noneable()


def compute_num_output(
    n: Union[int, float, None], num_inputs: int, step: int) -> int:
  """Returns the number of output."""
  n = scalars.scalar_value(n, step)
  if isinstance(n, float):
    n = math.ceil(n * num_inputs)
  elif n is None:
    n = num_inputs
  return n


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.'),
    ('replacement', pg.typing.Bool(False),
     'If True, the output examples are generated randomly with replacement. '
     'Otherwise they will be generated randomly without replacement.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed for sampling the input.')
])
class Random(base.Selector):
  """Random N selector."""

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)

  def select(self, inputs: List[Any], step: int) -> List[Any]:
    n = compute_num_output(self.n, len(inputs), step)
    if self.replacement:
      return [self._random.choice(inputs) for _ in range(n)]
    else:
      n = min(n, len(inputs))
      return self._random.sample(inputs, n)


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.'),
    ('weights', base.operation_spec(
        pg.typing.Any(), pg.typing.Float(min_value=0.0)),
     'A callable object that takes a list of items as input and returns a list '
     'of float numbers as the weights for each item. Optional keyword '
     'arguments \'global_state\' and \'step\' can be accepted.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed for sampling the input.')
])
class Sample(base.Selector):
  """Sample N items from a weighting function."""

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)
    self._weights = base.make_operation_compatible(self.weights)

  def select(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, inputs: List[Any], global_state: pg.Dict, step: int) -> List[Any]:
    n = compute_num_output(self.n, len(inputs), step)
    weights = self._weights(inputs, global_state=global_state, step=step)
    return self._random.choices(inputs, weights=weights, k=n)


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.'),
    ('weights', base.operation_spec(
        pg.typing.Any(), pg.typing.Float(min_value=0.0)),
     'A callable object that takes a list of items as input and returns a list '
     'of float numbers as the weights for each item. Optional keyword '
     'arguments \'global_state\' and \'step\' can be accepted.'),
])
class Proportional(base.Selector):
  """Select N items proportional to the input weights."""

  def _on_bound(self):
    super()._on_bound()
    self._weights = base.make_operation_compatible(self.weights)

  def select(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, inputs: List[Any], global_state: pg.Dict, step: int) -> List[Any]:
    n = compute_num_output(self.n, len(inputs), step)
    weights = self._weights(inputs, global_state=global_state, step=step)
    partition_info = self._partition(weights, n)
    output = []
    for item, num_occurences in zip(inputs, partition_info):
      output.extend([item] * num_occurences)
    return output

  def _partition(self, weights: List[float], n: int) -> List[int]:
    """Partition N into a list of integers roughly proportional to `weights`."""
    denominator = sum(weights)
    allocation = [int(n * w / denominator + 0.5) for w in weights]
    extra = n - sum(allocation)
    if extra != 0:
      # NOTE(daiyip): The allocation is computed by rounding up the weights
      # multiplying N, which can be less or larger than N. Therefore, we need
      # to make adjustment when such mismatch happends.
      # The strategy here is: always allocate extra items (when `extra` > 0) to
      # the slots that have larger weights, or reduce allocations
      # (when `extra` < 0) from the slots that have smaller weights.
      # To do so, we sort all the slots whose weights are not zero, and adjust
      # their allocation in a round-robin fashion.
      candidates_to_adjust = sorted(
          [i for i, w in enumerate(weights) if w > 0.],
          key=lambda x: weights[x], reverse=extra > 0)

      delta = 1 if extra > 0 else -1
      next_candidate = 0
      while extra != 0:
        item_index = candidates_to_adjust[
            next_candidate % len(candidates_to_adjust)]
        while delta == -1 and allocation[item_index] == 0:
          next_candidate = (next_candidate + 1) % len(candidates_to_adjust)
          item_index = candidates_to_adjust[next_candidate]
        allocation[item_index] += delta
        extra -= delta
        next_candidate += 1
    return allocation


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.'),
    ('key', pg.typing.Callable(
        [pg.typing.Any()], returns=pg.typing.Any()).noneable(),
     'A callable object as the key argument for sorting the input list. '
     'If None and when the input element type is DNA, the fitness will be '
     'used as `key`.'),
    ('cluster', pg.typing.Bool(default=False),
     'If True, returns bottom N clusters. Otherwise returns N individuals. '
     'Individuals that produces the same key form a cluster.')
])
class Top(base.Selector):
  """Top N selector."""

  def select(self, inputs: List[Any], step: int) -> List[Any]:
    key = self.key
    if key is None and inputs and isinstance(inputs[0], pg.DNA):
      key = base.get_fitness
    n = compute_num_output(self.n, len(inputs), step)
    if self.cluster:
      keys = [key(x) for x in inputs]
      sorted_keys = set(sorted(set(keys), reverse=True)[:n])
      selected = [(k, x) for k, x in zip(keys, inputs)
                  if k in sorted_keys]
      return [x for k, x in sorted(
          selected, key=lambda x: x[0], reverse=True)]
    else:
      return sorted(inputs, key=key, reverse=True)[:n]


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.'),
    ('key', pg.typing.Callable(
        [pg.typing.Any()], returns=pg.typing.Any()).noneable(),
     'A callable object as the key argument for sorting the input list. '
     'If None and when the input element type is DNA, the fitness will be '
     'used as `key`.'),
    ('cluster', pg.typing.Bool(default=False),
     'If True, returns bottom N clusters. Otherwise returns N individuals. '
     'Individuals that produces the same key form a cluster.')
])
class Bottom(base.Selector):
  """Bottom N selector."""

  def select(self, inputs: List[Any], step: int) -> List[Any]:
    key = self.key
    if key is None and inputs and isinstance(inputs[0], pg.DNA):
      key = base.get_fitness
    n = compute_num_output(self.n, len(inputs), step)
    if self.cluster:
      keys = [key(x) for x in inputs]
      sorted_keys = set(sorted(set(keys))[:n])
      selected = [(k, x) for k, x in zip(keys, inputs)
                  if k in sorted_keys]
      return [x for k, x in sorted(selected, key=lambda x: x[0])]
    else:
      return sorted(inputs, key=key)[:n]


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.'),
])
class First(base.Selector):
  """First N selector."""

  def select(self, inputs: List[Any], step: int) -> List[Any]:
    return inputs[:compute_num_output(self.n, len(inputs), step)]


@pg.members([
    ('n', num_output_spec(),
     'Number of items to select, which can be an integer, a float value or '
     'None. It also can be a callable object that returns a value based on a '
     'step (an integer). If its value or returned value is an integer, the '
     'selector outputs `n` examples from the input. If float, which must be '
     'within range [0., 1.], the selector outputs `math.ceil(n * len(inputs))` '
     'examples. If None, the selector outputs the same number of examples as '
     'the input.')
])
class Last(base.Selector):
  """Last N selector."""

  def select(self, inputs: List[Any], step: int) -> List[Any]:
    n = compute_num_output(self.n, len(inputs), step)
    return inputs[max(0, len(inputs) - n):]
