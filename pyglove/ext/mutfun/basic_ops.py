# Copyright 2023 The PyGlove Authors
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
"""Basic operators on mutable instructions."""

from typing import Any, Callable, Dict, Optional, Type
import pyglove.core as pg
from pyglove.ext.mutfun import base


class Operator(base.Instruction):
  """Base class for operators."""

  # The order of operator, larger values indicate higher orders.
  # For example, multiply has larger ORDER value than addition.
  ORDER: Optional[int] = None

  # Operator str, which will be used to create Python program representation.
  OPERATOR_STR: Optional[str] = None

  def maybe_parenthesize(self, child: Any) -> str:
    """Maybe add parenthesis to a child expression."""
    if isinstance(child, Operator):
      assert self.ORDER is not None, self.__class__
      assert child.ORDER is not None, child.__class__
      if self.ORDER >= child.ORDER:
        return '(' + base.python_repr(child) + ')'
    return base.python_repr(child)


@pg.members([
    ('x', pg.typing.Any())
])
class UnaryOperator(Operator):
  """Base class for unary math operations."""

  OPERATOR_FN: Optional[Callable[[Type[Any], Any], Any]] = None

  def python_repr(self, block_indent: int = 0) -> str:
    assert self.OPERATOR_STR is not None
    return base.indent(
        self.OPERATOR_STR + self.maybe_parenthesize(self.x), block_indent)

  def evaluate(self, context: Dict[str, Any]) -> Any:
    assert self.OPERATOR_FN is not None, self.__class__
    return self.OPERATOR_FN(base.evaluate(self.x, context))  # pylint: disable=not-callable


class Negate(UnaryOperator):
  """Negates the operrand."""
  ORDER = 7
  OPERATOR_STR = '-'
  OPERATOR_FN = lambda cls, x: -x


@pg.members([
    ('x', pg.typing.Any()),
    ('y', pg.typing.Any()),
])
class BinaryOperator(Operator):
  """Base class for binary math operations."""

  OPERATOR_FN: Optional[Callable[[Any], Any]] = None

  def python_repr(self, block_indent: int = 0) -> str:
    assert self.OPERATOR_STR is not None
    return base.indent(
        '%s %s %s' % (self.maybe_parenthesize(self.x),
                      self.OPERATOR_STR,
                      self.maybe_parenthesize(self.y)), block_indent)

  def evaluate(self, context: Dict[str, Any]) -> Any:
    assert self.OPERATOR_FN is not None, self.__class__
    return self.OPERATOR_FN(                # pylint: disable=not-callable
        base.evaluate(self.x, context),
        base.evaluate(self.y, context))


class Add(BinaryOperator):
  """Add operator."""
  ORDER = 3
  OPERATOR_STR = '+'
  OPERATOR_FN = lambda cls, x, y: x + y


class Substract(BinaryOperator):
  """Substract operator."""
  ORDER = 3
  OPERATOR_STR = '-'
  OPERATOR_FN = lambda cls, x, y: x - y


class Multiply(BinaryOperator):
  """Multiply operator."""
  ORDER = 4
  OPERATOR_STR = '*'
  OPERATOR_FN = lambda cls, x, y: x * y


class Divide(BinaryOperator):
  """Divide operator."""
  ORDER = 4
  OPERATOR_STR = '/'
  OPERATOR_FN = lambda cls, x, y: x / y


class FloorDivide(BinaryOperator):
  """Floor divide operator."""
  ORDER = 4
  OPERATOR_STR = '//'
  OPERATOR_FN = lambda cls, x, y: x // y


class Mod(BinaryOperator):
  """Mod operator."""
  ORDER = 4
  OPERATOR_STR = '%'
  OPERATOR_FN = lambda cls, x, y: x % y


class Power(BinaryOperator):
  """Power operator."""
  ORDER = 6
  OPERATOR_STR = '**'
  OPERATOR_FN = lambda cls, x, y: x ** y


class GreaterThan(BinaryOperator):
  """Greater than operator."""

  ORDER = 2
  OPERATOR_STR = '>'
  OPERATOR_FN = lambda cls, x, y: x > y


class Equals(BinaryOperator):
  """Equals operation."""

  ORDER = 3
  OPERATOR_STR = '=='
  OPERATOR_FN = lambda cls, x, y: x == y


class NotEquals(BinaryOperator):
  """Not Equals operator."""

  ORDER = 3
  OPERATOR_STR = '!='
  OPERATOR_FN = lambda cls, x, y: x != y


class LessThan(BinaryOperator):
  """Less than operator."""

  ORDER = 2
  OPERATOR_STR = '<'
  OPERATOR_FN = lambda cls, x, y: x < y


# NOTE(daiyip): This enables users to apply common operators to symbolic
# instructions, while allowing subclasses to override the behaviors.

# pylint: disable=unnecessary-lambda

base.Instruction.__neg__ = lambda self: Negate(self)
base.Instruction.__radd__ = lambda self, x: Add(x, self)
base.Instruction.__add__ = lambda self, y: Add(self, y)
base.Instruction.__rsub__ = lambda self, x: Substract(x, self)
base.Instruction.__sub__ = lambda self, y: Substract(self, y)
base.Instruction.__rmul__ = lambda self, x: Multiply(x, self)
base.Instruction.__mul__ = lambda self, y: Multiply(self, y)
base.Instruction.__rtruediv__ = lambda self, x: Divide(x, self)
base.Instruction.__truediv__ = lambda self, y: Divide(self, y)
base.Instruction.__rfloordiv__ = lambda self, x: FloorDivide(x, self)
base.Instruction.__floordiv__ = lambda self, y: FloorDivide(self, y)
base.Instruction.__rmod__ = lambda self, x: Mod(x, self)
base.Instruction.__mod__ = lambda self, y: Mod(self, y)
base.Instruction.__rpow__ = lambda self, x: Power(x, self)
base.Instruction.__pow__ = lambda self, y: Power(self, y)
base.Instruction.__gt__ = lambda self, y: GreaterThan(self, y)
base.Instruction.__lt__ = lambda self, y: LessThan(self, y)
# pylint: enable=unnecessary-lambda
