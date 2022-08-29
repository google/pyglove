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
"""Step-based scalars used as evolution hyper-parameter values."""

import abc
import math
import random
from typing import Any, Union

import pyglove.core as pg


# We disable implicit str concat as it is commonly used class schema docstr.
# pylint: disable=implicit-str-concat


def scalar_spec(value_spec: pg.typing.ValueSpec) -> pg.typing.ValueSpec:
  """Returns the value spec for a schedule scalar.

  Args:
    value_spec: a value spec for the schedule-based scalar type.

  Returns:
    A value spec for either the value itself or a callable that produces such
      value based on a step (integer).
  """
  return pg.typing.Union([
      value_spec,
      pg.typing.Callable([pg.typing.Int()], returns=value_spec)
  ])


def scalar_value(value: Any, step: int) -> Any:
  """Returns a scheduled value based on a step."""
  if callable(value):
    return value(step)
  return value


def make_scalar(value: Any) -> 'Scalar':
  """Make a scalar from a value."""
  if isinstance(value, Scalar):
    return value
  elif callable(value):
    return Lambda(value)    # pytype: disable=bad-return-type
  else:
    return Constant(value)  # pytype: disable=bad-return-type


class Scalar(pg.Object):
  """Interface for step-based scalar."""

  def __call__(self, step: int) -> Any:
    """Returns the value of the scalar at a given step."""
    return self.call(step)

  @abc.abstractmethod
  def call(self, step: int) -> Any:
    """Implementation. Subclass should override this method."""

  def __add__(self, x):
    """Operator +."""
    return Addition(self, x)

  def __radd__(self, x):
    """Right-hand operator +."""
    return Addition(x, self)

  def __sub__(self, x):
    """Operator -."""
    return Substraction(self, x)

  def __rsub__(self, x):
    """Right hand operator -."""
    return Substraction(x, self)

  def __mul__(self, x):
    """Operator *."""
    return Multiplication(self, x)

  def __rmul__(self, x):
    """Right-hand operator *."""
    return Multiplication(x, self)

  def __truediv__(self, x):
    """Operator /."""
    return Division(self, x)

  def __rtruediv__(self, x):
    """Right-hand operator /."""
    return Division(x, self)

  def __floordiv__(self, x):
    """Operator //."""
    return Floor(Division(self, x))

  def __rfloordiv__(self, x):
    """Right-hand operator //."""
    return Floor(Division(x, self))

  def __mod__(self, x):
    """Operator %."""
    return Mod(self, x)

  def __rmod__(self, x):
    """Right-hand operator %."""
    return Mod(x, self)

  def __pow__(self, x):
    """Operator **."""
    return Power(self, x)

  def __rpow__(self, x):
    """Right-hand operator **."""
    return Power(x, self)

  def __neg__(self):
    """Returns the negation of current scalar."""
    return Negation(self)

  def __abs__(self):
    """Returns the absolute of current scalar."""
    return Absolute(self)

  def floor(self):
    """Returns the floor of current scalar."""
    return Floor(self)

  def ceil(self):
    """Returns the ceiling of current scalar."""
    return Ceiling(self)


#
# Conversion from functions and constant values.
#


@pg.members([
    ('fn', scalar_spec(pg.typing.Any()),
     'A callable that takes an integer as input and output a value.')
], init_arg_list=['fn'])
class Lambda(Scalar):
  """Lambda operation."""

  def call(self, step: int) -> Any:
    return self.fn(step)


@pg.members([
    ('value', pg.typing.Any(), 'Value for the constant.')
])
class Constant(Scalar):
  """A constant number."""

  def call(self, step: int) -> Any:
    del step
    return self.value


class _Step(Scalar):
  """Scalar for current step."""

  def call(self, step: int) -> int:
    return step


STEP = _Step()


#
# Scalar with random values.
#


@pg.members([
    ('seed', pg.typing.Int().noneable(), 'Random seed.')
])
class RandomScalar(Scalar):
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
     'A pivot point to weight possible output closer to `low` or '
     '`high`. If None, it is set to the middle of `low` and `high`. '
     'leading to a symmetric distribution.')
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


#
# Unary numeric operations.
#


@pg.members([
    ('x', scalar_spec(
        pg.typing.Union([pg.typing.Int(), pg.typing.Float()])),
     'The inner scheduled value.')
])
class UnaryOp(Scalar):
  """Unary scalar operators."""

  def _on_bound(self):
    super()._on_bound()
    self._x = make_scalar(self.x)

  def call(self, step: int) -> Any:
    return self.operate(self._x(step))

  @abc.abstractmethod
  def operate(self, x: Union[int, float]) -> Union[int, float]:
    """Implementation of the operation on a computed value."""


class Negation(UnaryOp):
  """Negation operator."""

  def operate(self, x: Union[int, float]) -> Union[int, float]:
    return -x


class Absolute(UnaryOp):
  """Absolute operator."""

  def operate(self, x: Union[int, float]) -> Union[int, float]:
    return abs(x)


class Floor(UnaryOp):
  """Floor operator."""

  def operate(self, x: Union[int, float]) -> int:
    return math.floor(x)


class Ceiling(UnaryOp):
  """Ceil operator."""

  def operate(self, x: Union[int, float]) -> int:
    return math.ceil(x)


#
# Binary numeric operations.
#


@pg.members([
    ('x', scalar_spec(
        pg.typing.Union([pg.typing.Int(), pg.typing.Float()])),
     'Left hand scheduled value.'),
    ('y', scalar_spec(
        pg.typing.Union([pg.typing.Int(), pg.typing.Float()])),
     'Right hand scheduled value.')
])
class BinaryOp(Scalar):
  """Binary operation for computing scheduled value."""

  def _on_bound(self):
    super()._on_bound()
    self._x = make_scalar(self.x)
    self._y = make_scalar(self.y)

  def call(self, step: int) -> Union[int, float]:
    return self.operate(self._x(step), self._y(step))

  @abc.abstractmethod
  def operate(
      self, x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    """Implementation of the operation on two computed value."""


class Addition(BinaryOp):
  """Add operation."""

  def operate(
      self, x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x + y


class Substraction(BinaryOp):
  """Substract operation."""

  def operate(
      self, x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x - y


class Multiplication(BinaryOp):
  """Multiply operation."""

  def operate(
      self, x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x * y


class Division(BinaryOp):
  """Divide operation."""

  def operate(
      self, x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x / y


class Mod(BinaryOp):
  """Mod operation."""

  def operate(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, x: int, y: int) -> int:
    return x % y


class Power(BinaryOp):
  """Power operation."""

  def operate(
      self, x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return math.pow(x, y)


#
# Frequently used math formulas used as scalars.
#


@pg.members([])
class SquareRoot(UnaryOp):
  """The square root scalar."""

  def operate(self, x: float) -> float:
    return math.sqrt(x)

sqrt = SquareRoot   # pylint: disable=invalid-name


@pg.members([])
class Exp(UnaryOp):
  """More accurate version for math.e ** x."""

  def operate(self, x: float) -> float:
    return math.exp(x)

exp = Exp   # pylint: disable=invalid-name


@pg.members([
    ('x', scalar_spec(pg.typing.Union([
        pg.typing.Int(min_value=2),
        pg.typing.Float(min_value=0.0)]))),
    ('base', scalar_spec(pg.typing.Union([
        pg.typing.Int(min_value=2),
        pg.typing.Float(min_value=0.0)])).set_default(math.e),
     'Base of the log function.'),
])
class Log(Scalar):
  """A log scheduled float."""

  def _on_bound(self):
    super()._on_bound()
    self._x = make_scalar(self.x)
    self._base = make_scalar(self.base)

  def call(self, step: int) -> float:
    return math.log(self._x(step), self._base(step))

log = Log    # pylint: disable=invalid-name


@pg.members([])
class Cosine(UnaryOp):
  """Cosine that works for scalars."""

  def operate(self, x: float) -> float:
    return math.cos(x)

cos = Cosine  # pylint: disable=invalid-name


@pg.members([])
class Sine(UnaryOp):
  """Sine that works for scalars."""

  def operate(self, x: float) -> float:
    return math.sin(x)

sin = Sine  # pylint: disable=invalid-name


#
# Scheduled values that can be designed in multiple phases.
#


@pg.members([
    ('phases', pg.typing.List(
        pg.typing.Tuple([
            pg.typing.Union([pg.typing.Int(min_value=0),
                             pg.typing.Float(min_value=0.)]),
            scalar_spec(pg.typing.Any())
        ]), min_size=1),
     'All the phases in the schedule. Each item in the list is a tuple of '
     '`(length of phase, scheduled value)`. The length of phase can be an '
     'integer representing number of steps used for that phase, or a float as '
     'the proportion of that phase if `total_steps` is specified. All items '
     'in the list should use the same type (integer or float) for the length '
     'of phase. When a proportion is used, their sum does not have to sum up '
     'to 1.'),
    ('total_steps', pg.typing.Int(min_value=1).noneable(),
     'Total number of steps for the schedule. If None, the length of each '
     'phase must be an integer.')
])
class StepWise(Scalar):
  """A step-wise schedule that is specified via multiple phases."""

  def _on_bound(self):
    super()._on_bound()

    last_step = 0
    phase_ending_steps = []
    if self.total_steps is None:
      for phase_len, phase_value in self.phases:
        if isinstance(phase_len, float):
          raise ValueError(
              f'`total_steps` must be specified when float is used as the '
              f'value for phase length. '
              f'Encountered: ({phase_len}, {phase_value}).')
        last_step += phase_len
        phase_ending_steps.append(last_step - 1)
    else:
      proportion_sum = 0.
      for proportion, phase_value in self.phases:
        if isinstance(proportion, int):
          raise ValueError(
              f'The phase length should be a float as a proportion of the '
              f'entire schedule when `total_steps` is specified. '
              f'Encountered: ({proportion}, {phase_value}).')
        proportion_sum += proportion

      if proportion_sum == 0:
        raise ValueError(
            f'The sum of all proportions must be greater than 0. '
            f'Encountered: {self.phases!r}')

      for proportion, _ in self.phases:
        phase_len = int(proportion / proportion_sum * self.total_steps)
        last_step += phase_len
        phase_ending_steps.append(last_step - 1)
    # Phase ending step is the step AFTER which the next phase will start.
    self._phase_ending_steps = phase_ending_steps
    self._phases = [make_scalar(p) for l, p in self.phases]
    self._current_phase = 0
    self._last_value = None

  def call(self, step: int) -> Any:
    if self._current_phase < len(self.phases):
      if self._current_phase > 0:
        phase_step = step - (
            self._phase_ending_steps[self._current_phase - 1] + 1)
      else:
        phase_step = step
      self._last_value = self._phases[self._current_phase](phase_step)
      if step == self._phase_ending_steps[self._current_phase]:
        self._current_phase += 1
    return self._last_value


#
#  Helper function for create popular scalar scheddule.
#


def linear(total_steps: int, start: float = 1.0, end: float = 0.0):
  """Returns a linear scalar from start to end."""
  return start + STEP * ((end - start) / total_steps)


def cosine_decay(total_steps: int, start: float = 1.0, end: float = 0.0):
  """Returns a cosine decayed scalar from start to end."""
  return 0.5 * (start - end) * (
      1 + cos(math.pi * STEP / total_steps)) + end


def exponential_decay(
    decay_rate: float, decay_interval: int,
    start: float = 1.0, staircase: bool = True):
  """Returns a scalar that exponentially decays from start to end."""
  exponent = STEP / float(decay_interval)
  if staircase:
    exponent = exponent.floor()
  return start * (decay_rate ** exponent)


def cyclic(cycle: int, initial_radiant: float = 0.0,
           high: float = 1.0, low: float = 0.0):
  """Returns a cyclic scalar using sin/cos."""
  return 0.5 * (high - low) * (
      1 + cos(initial_radiant + math.pi * 2 * STEP / cycle)) + low
