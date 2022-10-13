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

import math
import pyglove.core as pg
from pyglove.generators.scalars import base


class SquareRoot(base.UnaryOp):
  """The square root scalar."""

  def operate(self, x: float) -> float:
    return math.sqrt(x)

sqrt = SquareRoot   # pylint: disable=invalid-name


@pg.members([])
class Exp(base.UnaryOp):
  """More accurate version for math.e ** x."""

  def operate(self, x: float) -> float:
    return math.exp(x)

exp = Exp   # pylint: disable=invalid-name


@pg.members([
    ('x', base.scalar_spec(pg.typing.Union([
        pg.typing.Int(min_value=2),
        pg.typing.Float(min_value=0.0)]))),
    ('base', base.scalar_spec(pg.typing.Union([
        pg.typing.Int(min_value=2),
        pg.typing.Float(min_value=0.0)])).set_default(math.e),
     'Base of the log function.'),
])
class Log(base.Scalar):
  """A log scheduled float."""

  def _on_bound(self):
    super()._on_bound()
    self._x = base.make_scalar(self.x)
    self._base = base.make_scalar(self.base)

  def call(self, step: int) -> float:
    return math.log(self._x(step), self._base(step))

log = Log    # pylint: disable=invalid-name


@pg.members([])
class Cosine(base.UnaryOp):
  """Cosine that works for scalars."""

  def operate(self, x: float) -> float:
    return math.cos(x)

cos = Cosine  # pylint: disable=invalid-name


@pg.members([])
class Sine(base.UnaryOp):
  """Sine that works for scalars."""

  def operate(self, x: float) -> float:
    return math.sin(x)

sin = Sine  # pylint: disable=invalid-name


#
#  Helper function for create popular scalar scheddule.
#


def linear(total_steps: int, start: float = 1.0, end: float = 0.0):
  """Returns a linear scalar from start to end."""
  return start + base.STEP * ((end - start) / total_steps)


def cosine_decay(total_steps: int, start: float = 1.0, end: float = 0.0):
  """Returns a cosine decayed scalar from start to end."""
  return 0.5 * (start - end) * (
      1 + cos(math.pi * base.STEP / total_steps)) + end


def exponential_decay(
    decay_rate: float, decay_interval: int,
    start: float = 1.0, staircase: bool = True):
  """Returns a scalar that exponentially decays from start to end."""
  exponent = base.STEP / float(decay_interval)
  if staircase:
    exponent = exponent.floor()
  return start * (decay_rate ** exponent)


def cyclic(cycle: int, initial_radiant: float = 0.0,
           high: float = 1.0, low: float = 0.0):
  """Returns a cyclic scalar using sin/cos."""
  return 0.5 * (high - low) * (
      1 + cos(initial_radiant + math.pi * 2 * base.STEP / cycle)) + low
