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
"""Tests for symbolic_program/base.py."""

import unittest
import pyglove.core as pg
from pyglove.ext.mutfun import base
from pyglove.ext.mutfun import basic_ops


class NegateTest(unittest.TestCase):
  """Tests for Negate."""

  def test_python_repr(self):
    y = basic_ops.Negate(base.Var('x'))
    self.assertEqual(base.python_repr(y), '-x')

    z = basic_ops.Negate(base.Var('x') + base.Var('y'))
    self.assertEqual(base.python_repr(z), '-(x + y)')

  def test_evaluate(self):
    x = basic_ops.Negate(base.Var('x'))
    variables = dict(x=1)
    self.assertEqual(x.evaluate(variables), -1)
    self.assertEqual(variables, dict(x=1))

  def test_operator_overload(self):
    self.assertTrue(pg.eq(-base.Var('x'), basic_ops.Negate(base.Var('x'))))


class AddTest(unittest.TestCase):
  """Tests for Add."""

  def test_python_repr(self):
    y = basic_ops.Add(base.Var('x'), 1)
    self.assertEqual(base.python_repr(y), 'x + 1')

    z = basic_ops.Add(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x + (y + 1)')

    z = basic_ops.Add(base.Var('x'), basic_ops.Multiply(base.Var('y'), 2))
    self.assertEqual(base.python_repr(z), 'x + y * 2')

  def test_evaluate(self):
    x = basic_ops.Add(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=1, y=2)
    self.assertEqual(x.evaluate(variables), 1 + 2 + 1)

  def test_operator_overload(self):
    self.assertTrue(pg.eq(1 + base.Var('x'), basic_ops.Add(1, base.Var('x'))))
    self.assertTrue(pg.eq(base.Var('x') + 1, basic_ops.Add(base.Var('x'), 1)))


class SubstractTest(unittest.TestCase):
  """Tests for Substract."""

  def test_python_repr(self):
    y = basic_ops.Substract(base.Var('x'), 1)
    self.assertEqual(base.python_repr(y), 'x - 1')

    z = basic_ops.Substract(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x - (y + 1)')

    z = basic_ops.Substract(base.Var('x'), basic_ops.Multiply(base.Var('y'), 2))
    self.assertEqual(base.python_repr(z), 'x - y * 2')

  def test_evaluate(self):
    x = basic_ops.Substract(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=1, y=2)
    self.assertEqual(x.evaluate(variables), 1 - (2 + 1))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(1 - base.Var('x'), basic_ops.Substract(1, base.Var('x'))))
    self.assertTrue(
        pg.eq(base.Var('x') - 1, basic_ops.Substract(base.Var('x'), 1)))


class MultiplyTest(unittest.TestCase):
  """Tests for Multiply."""

  def test_python_repr(self):
    y = basic_ops.Multiply(base.Var('x'), 2)
    self.assertEqual(base.python_repr(y), 'x * 2')

    z = basic_ops.Multiply(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x * (y + 1)')

    z = basic_ops.Multiply(base.Var('x') ** 2, -base.Var('y'))
    self.assertEqual(base.python_repr(z), 'x ** 2 * -y')

  def test_evaluate(self):
    x = basic_ops.Multiply(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=2, y=3)
    self.assertEqual(x.evaluate(variables), 2 * (3 + 1))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(2 * base.Var('x'), basic_ops.Multiply(2, base.Var('x'))))
    self.assertTrue(
        pg.eq(base.Var('x') * 2, basic_ops.Multiply(base.Var('x'), 2)))


class DivideTest(unittest.TestCase):
  """Tests for Divide."""

  def test_python_repr(self):
    y = basic_ops.Divide(base.Var('x'), 2)
    self.assertEqual(base.python_repr(y), 'x / 2')

    z = basic_ops.Divide(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x / (y + 1)')

    z = basic_ops.Divide(base.Var('x'), basic_ops.Multiply(base.Var('y'), 2))
    self.assertEqual(base.python_repr(z), 'x / (y * 2)')

    z = basic_ops.Divide(base.Var('x'), 2 ** -base.Var('y'))
    self.assertEqual(base.python_repr(z), 'x / 2 ** -y')

  def test_evaluate(self):
    x = basic_ops.Divide(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=2, y=3)
    self.assertEqual(x.evaluate(variables), 2 / (3 + 1))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(2 / base.Var('x'), basic_ops.Divide(2, base.Var('x'))))
    self.assertTrue(
        pg.eq(base.Var('x') / 2, basic_ops.Divide(base.Var('x'), 2)))


class FloorDivideTest(unittest.TestCase):
  """Tests for FloorDivide."""

  def test_python_repr(self):
    y = basic_ops.FloorDivide(base.Var('x'), 2)
    self.assertEqual(base.python_repr(y), 'x // 2')

    z = basic_ops.FloorDivide(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x // (y + 1)')

    z = basic_ops.FloorDivide(base.Var('x'),
                              basic_ops.Multiply(base.Var('y'), 2))
    self.assertEqual(base.python_repr(z), 'x // (y * 2)')

    z = basic_ops.FloorDivide(base.Var('x'), 2 ** -base.Var('y'))
    self.assertEqual(base.python_repr(z), 'x // 2 ** -y')

  def test_evaluate(self):
    x = basic_ops.FloorDivide(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=2, y=3)
    self.assertEqual(x.evaluate(variables), 2 // (3 + 1))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(2 // base.Var('x'), basic_ops.FloorDivide(2, base.Var('x'))))
    self.assertTrue(
        pg.eq(base.Var('x') // 2, basic_ops.FloorDivide(base.Var('x'), 2)))


class ModTest(unittest.TestCase):
  """Tests for Mod."""

  def test_python_repr(self):
    y = basic_ops.Mod(base.Var('x'), 2)
    self.assertEqual(base.python_repr(y), 'x % 2')

    z = basic_ops.Mod(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x % (y + 1)')

    z = basic_ops.Mod(base.Var('x'), basic_ops.Multiply(base.Var('y'), 2))
    self.assertEqual(base.python_repr(z), 'x % (y * 2)')

    z = basic_ops.Mod(base.Var('x'), 2 ** -base.Var('y'))
    self.assertEqual(base.python_repr(z), 'x % 2 ** -y')

  def test_evaluate(self):
    x = basic_ops.Mod(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=2, y=3)
    self.assertEqual(x.evaluate(variables), 2 % (3 + 1))

  def test_operator_overload(self):
    self.assertTrue(pg.eq(2 % base.Var('x'), basic_ops.Mod(2, base.Var('x'))))
    self.assertTrue(pg.eq(base.Var('x') % 2, basic_ops.Mod(base.Var('x'), 2)))


class PowerTest(unittest.TestCase):
  """Tests for Power."""

  def test_python_repr(self):
    y = basic_ops.Power(base.Var('x'), 2)
    self.assertEqual(base.python_repr(y), 'x ** 2')

    z = basic_ops.Power(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    self.assertEqual(base.python_repr(z), 'x ** (y + 1)')

    z = basic_ops.Power(base.Var('x'), basic_ops.Multiply(base.Var('y'), 2))
    self.assertEqual(base.python_repr(z), 'x ** (y * 2)')

    z = basic_ops.Power(base.Var('x'), 2 ** -base.Var('y'))
    self.assertEqual(base.python_repr(z), 'x ** (2 ** -y)')

  def test_evaluate(self):
    x = basic_ops.Power(base.Var('x'), basic_ops.Add(base.Var('y'), 1))
    variables = dict(x=2, y=3)
    self.assertEqual(x.evaluate(variables), 2 ** (3 + 1))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(2 ** base.Var('x'), basic_ops.Power(2, base.Var('x'))))
    self.assertTrue(
        pg.eq(base.Var('x') ** 2, basic_ops.Power(base.Var('x'), 2)))


class GreaterThanTest(unittest.TestCase):
  """Tests for Greater Than."""

  def test_python_repr(self):
    y = basic_ops.GreaterThan(base.Var('x'), 1)
    self.assertEqual(base.python_repr(y), 'x > 1')
    y = basic_ops.GreaterThan(1, base.Var('x'))
    self.assertEqual(base.python_repr(y), '1 > x')

    z = basic_ops.GreaterThan(
        base.Var('x'), basic_ops.Add(base.Var('x'), base.Var('y'))
    )
    self.assertEqual(base.python_repr(z), 'x > x + y')

    z = basic_ops.GreaterThan(
        base.Var('x'), basic_ops.LessThan(base.Var('x'), base.Var('y'))
    )
    self.assertEqual(base.python_repr(z), 'x > (x < y)')

  def test_evaluate(self):
    x = basic_ops.GreaterThan(base.Var('x'), 1)
    variables = dict(x=3)
    self.assertTrue(x.evaluate(variables))

    z = basic_ops.GreaterThan(
        base.Var('x'), basic_ops.Add(base.Var('x'), base.Var('y'))
    )
    variables = dict(x=1, y=2)
    self.assertFalse(z.evaluate(variables))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(base.Var('x') > 1, basic_ops.GreaterThan(base.Var('x'), 1))
    )
    self.assertTrue(
        pg.eq(
            base.Var('x') > base.Var('y'),
            basic_ops.GreaterThan(base.Var('x'), base.Var('y')),
        )
    )


class LessThanTest(unittest.TestCase):
  """Tests for Less Than."""

  def test_python_repr(self):
    y = basic_ops.LessThan(
        base.Var('x'), basic_ops.Divide(base.Var('a'), base.Var('b'))
    )
    self.assertEqual(base.python_repr(y), 'x < a / b')

  def test_evaluate(self):
    x = basic_ops.LessThan(base.Var('x'), 1)
    variables = dict(x=3)
    self.assertFalse(x.evaluate(variables))

    z = basic_ops.LessThan(
        base.Var('x'), basic_ops.Add(base.Var('x'), base.Var('y'))
    )
    variables = dict(x=1, y=2)
    self.assertTrue(z.evaluate(variables))

  def test_operator_overload(self):
    self.assertTrue(
        pg.eq(base.Var('x') < 1, basic_ops.LessThan(base.Var('x'), 1))
    )
    self.assertTrue(
        pg.eq(
            base.Var('x') < base.Var('y'),
            basic_ops.LessThan(base.Var('x'), base.Var('y')),
        )
    )


class EqualsTest(unittest.TestCase):
  """Tests for Equals."""

  def test_python_repr(self):
    y = basic_ops.Equals(
        base.Var('x'), basic_ops.Divide(base.Var('a'), base.Var('b'))
    )
    self.assertEqual(base.python_repr(y), 'x == a / b')

  def test_evaluate(self):
    x = basic_ops.Equals(base.Var('x'), 1)
    variables = dict(x=3)
    self.assertFalse(x.evaluate(variables))

    z = basic_ops.Equals(
        base.Var('x'), basic_ops.Multiply(base.Var('x'), base.Var('y'))
    )
    variables = dict(x=1, y=2)
    print(base.python_repr(z))
    self.assertFalse(z.evaluate(variables))


class NotEqualsTest(unittest.TestCase):
  """Tests for NotEquals."""

  def test_python_repr(self):
    y = basic_ops.NotEquals(
        base.Var('x'), basic_ops.Add(base.Var('a'), base.Var('b'))
    )
    self.assertEqual(base.python_repr(y), 'x != (a + b)')

  def test_evaluate(self):
    x = basic_ops.NotEquals(base.Var('x'), 1)
    variables = dict(x=3)
    self.assertTrue(x.evaluate(variables))

    z = basic_ops.Equals(
        base.Var('x'), basic_ops.Divide(base.Var('x'), base.Var('y'))
    )
    variables = dict(x=1, y=3)
    self.assertFalse(z.evaluate(variables))


if __name__ == '__main__':
  unittest.main()
