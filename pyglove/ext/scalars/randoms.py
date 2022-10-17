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
"""Step-based scalars used as evolution hyper-parameter values."""

import abc
import random
from typing import Union
import pyglove.core as pg
from pyglove.ext.scalars import base


@pg.members([
    ('seed', pg.typing.Int().noneable(), 'Random seed.')
])
class RandomScalar(base.Scalar):
  """Base class for random operation for computing scheduled value."""

  def _on_bound(self):
    self._random = random if self.seed is None else random.Random(self.seed)

  def call(self, step: int):
    return self.next_value()

  @abc.abstractmethod
  def next_value(self) -> Union[int, float]:
    """Return next value.."""


@pg.members([
    ('low', pg.typing.Union([
        pg.typing.Int(), pg.typing.Float()], default=0.0),
     'Lower bound (inclusive).'),
    ('high', pg.typing.Union([
        pg.typing.Int(), pg.typing.Float()], default=1.0),
     'Higher bound (inclusive).'),
], init_arg_list=['low', 'high', 'seed'])
class Uniform(RandomScalar):
  """Generate a random number in uniform distribution."""

  def _on_bound(self):
    super()._on_bound()
    if self.high < self.low:
      raise ValueError(
          f'`low` must be less or equal than `high`. '
          f'Encountered: low={self.low}, high={self.high}.')

  def next_value(self) -> Union[int, float]:
    if isinstance(self.low, int) and isinstance(self.high, int):
      return self._random.randint(self.low, self.high)
    else:
      return self._random.uniform(self.low, self.high)


@pg.members([
    ('low', pg.typing.Union([
        pg.typing.Int(), pg.typing.Float()], default=0.0),
     'Lower bound of the examples.'),
    ('high', pg.typing.Union([
        pg.typing.Int(), pg.typing.Float()], default=1.0),
     'Higher bound of the examples.'),
    ('mode', pg.typing.Union([
        pg.typing.Int(), pg.typing.Float()]).noneable(),
     ('A pivot point to weight possible output closer to `low` or '
      '`high`. If None, it is set to the middle of `low` and `high`. '
      'leading to a symmetric distribution.'))
], init_arg_list=['low', 'high', 'mode', 'seed'])
class Triangular(RandomScalar):
  """Generate a random float number in a triangular distribution."""

  def _on_bound(self):
    super()._on_bound()
    self._mode = self.mode
    if self.mode is None:
      self._mode = (self.low + self.high) / 2

  def next_value(self) -> Union[int, float]:
    result = self._random.triangular(self.low, self.high, self._mode)
    if isinstance(self.low, int) and isinstance(self.high, int):
      result = int(result)
    return result


@pg.members([
    ('mean', pg.typing.Float(), 'Mean of samples.'),
    ('std', pg.typing.Float(), 'Standard deviation of samples.')
], init_arg_list=['mean', 'std', 'seed'])
class Gaussian(RandomScalar):
  """Generate a random float number in gaussian distribution."""

  def next_value(self) -> float:
    return self._random.gauss(self.mean, self.std)


@pg.members([
    ('mean', pg.typing.Float(), 'Mean of samples.'),
    ('std', pg.typing.Float(), 'Standard deviation of samples.')
], init_arg_list=['mean', 'std', 'seed'])
class Normal(RandomScalar):
  """Generate a random float number in normal distribution."""

  def next_value(self) -> float:
    return self._random.normalvariate(self.mean, self.std)


@pg.members([
    ('mean', pg.typing.Float(), 'Mean of samples.'),
    ('std', pg.typing.Float(), 'Standard deviation of samples.')
], init_arg_list=['mean', 'std', 'seed'])
class LogNormal(RandomScalar):
  """Generate a random float number in log normal distribution."""

  def next_value(self) -> float:
    return self._random.lognormvariate(self.mean, self.std)
