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
"""Tests for base scalars."""

import unittest
from pyglove.ext.scalars import base as scalars


class BasicScalarTest(unittest.TestCase):
  """Test basic scalars."""

  def testMakeScalar(self):
    sv = scalars.make_scalar(scalars.Constant(1))
    self.assertIsInstance(sv, scalars.Scalar)
    self.assertEqual(sv(0), 1)
    self.assertEqual(sv(10), 1)

    sv = scalars.make_scalar(1)
    self.assertIsInstance(sv, scalars.Scalar)
    self.assertIsInstance(sv(0), int)
    self.assertEqual(sv(0), 1)
    self.assertEqual(sv(10), 1)

    sv = scalars.make_scalar(lambda step: step)
    self.assertIsInstance(sv, scalars.Scalar)
    self.assertEqual(sv(1), 1)
    self.assertEqual(sv(10), 10)

  def testCurrentStep(self):
    sv = scalars.STEP * 2
    self.assertEqual(sv(0), 0)
    self.assertEqual(sv(10), 20)


class UnaryOpTest(unittest.TestCase):
  """Tests for unary scalar operators."""

  def testNegation(self):
    sv = -scalars.STEP
    self.assertEqual(sv(1), -1)
    self.assertEqual(sv(2), -2)

  def testFloor(self):
    sv = scalars.Constant(1.6).floor()
    self.assertEqual(sv(0), 1)

  def testCeil(self):
    sv = scalars.Constant(1.6).ceil()
    self.assertEqual(sv(0), 2)

  def testAbs(self):
    sv = abs(scalars.Constant(-1))
    self.assertEqual(sv(0), 1)


class BinaryOpTest(unittest.TestCase):
  """Tests for binary scalar operators."""

  def testAdd(self):
    sv = scalars.Constant(1) + 2
    self.assertEqual(sv(0), 3)

    sv = 2 + scalars.Constant(1)
    self.assertEqual(sv(0), 3)

  def testSubstract(self):
    sv = scalars.Constant(1) - 2
    self.assertEqual(sv(0), -1)

    sv = 2 - scalars.Constant(1)
    self.assertEqual(sv(0), 1)

  def testMultiply(self):
    sv = scalars.Constant(1) * 2
    self.assertEqual(sv(0), 2)

    sv = 2 * scalars.Constant(1)
    self.assertEqual(sv(0), 2)

  def testDivide(self):
    sv = scalars.Constant(1) / 2
    self.assertEqual(sv(0), 0.5)

    sv = 2 / scalars.Constant(1)
    self.assertEqual(sv(0), 2)

  def testFloorDivide(self):
    sv = scalars.Constant(1) // 2
    self.assertEqual(sv(0), 0)

    sv = 2 // scalars.Constant(1)
    self.assertEqual(sv(0), 2)

  def testMod(self):
    sv = scalars.Constant(2) % 3
    self.assertEqual(sv(0), 2)

    sv = 3 % scalars.Constant(2)
    self.assertEqual(sv(0), 1)

  def testPower(self):
    sv = scalars.Constant(2) ** 3
    self.assertEqual(sv(0), 8)

    sv = 3 ** scalars.Constant(2)
    self.assertEqual(sv(0), 9)


if __name__ == '__main__':
  unittest.main()
