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
"""Base symbolic scalars."""

import abc
import math
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
