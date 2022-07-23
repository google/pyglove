# Copyright 2019 The PyGlove Authors
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
"""Tests for scalars."""

import math
import unittest
from pyglove.generators.evolution import scalars


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


class RandomScalarTest(unittest.TestCase):
  """Test scheduled random numbers."""

  def testUniform(self):
    sv = scalars.Uniform(seed=1)
    self.assertEqual(sv(0), 0.13436424411240122)
    self.assertEqual(sv(0), 0.8474337369372327)

    sv = scalars.Uniform(1, 10, seed=1)
    self.assertEqual(sv(0), 3)
    self.assertEqual(sv(0), 10)

    with self.assertRaisesRegex(
        ValueError,
        '`low` must be less or equal than `high`.'):
      scalars.Uniform(10, 1, seed=1)

  def testTriangular(self):
    sv = scalars.Triangular(0.0, 1.0, 0.9, seed=1)
    self.assertEqual(sv(0), 0.34774677525630787)
    self.assertEqual(sv(0), 0.8733214547023962)

    sv = scalars.Triangular(10, 20, seed=1)
    self.assertEqual(sv(0), 12)
    self.assertEqual(sv(0), 17)

  def testGaussian(self):
    sv = scalars.Gaussian(1.0, 0.2, seed=1)
    self.assertEqual(sv(0), 1.2576369506310927)
    self.assertEqual(sv(0), 1.2898891217399542)

  def testNormal(self):
    sv = scalars.Normal(1.0, 0.2, seed=1)
    self.assertEqual(sv(0), 1.1214911715287412)
    self.assertEqual(sv(0), 0.997154910897843)

  def testLogNormal(self):
    sv = scalars.LogNormal(1.0, 0.2, seed=1)
    self.assertEqual(sv(0), 3.0694278358084994)
    self.assertEqual(sv(0), 2.710559065635824)


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


class MathScalarTest(unittest.TestCase):

  def assertIsClose(self, x, y):
    assert abs(x - y) < 1e-15, (x, y)

  def testSqrt(self):
    sv = scalars.sqrt(scalars.STEP)
    self.assertEqual(sv(0), 0)
    self.assertEqual(sv(4), 2)

  def testExp(self):
    sv = scalars.exp(scalars.STEP)
    self.assertEqual(sv(0), 1)
    self.assertEqual(sv(1), math.e)

  def testLog(self):
    sv = scalars.log(scalars.STEP, 2)
    self.assertEqual(sv(1), 0)
    self.assertEqual(sv(4), 2)

    sv = scalars.log(16, scalars.STEP)
    self.assertEqual(sv(2), 4)
    self.assertEqual(sv(4), 2)

  def testCos(self):
    sv = scalars.cos(scalars.STEP * math.pi / 4)
    self.assertIsClose(sv(0), 1)
    self.assertIsClose(sv(1), math.sqrt(2) / 2)
    self.assertIsClose(sv(2), 0)

  def testSin(self):
    sv = scalars.sin(scalars.STEP * math.pi / 4)
    self.assertIsClose(sv(0), 0)
    self.assertIsClose(sv(1), math.sqrt(2) / 2)
    self.assertIsClose(sv(2), 1)


class StepWiseScalarTest(unittest.TestCase):
  """Test step-wise scalar schedule."""

  def testStepsAsPhaseLength(self):
    sv = scalars.StepWise([
        (2, 1),
        (2, scalars.STEP),
        (3, scalars.STEP ** 2)
    ])
    self.assertEqual([sv(i) for i in range(10)], [
        # For each phase, scalars.STEP
        # is evaluated to 0 when phase starts.
        1, 1,         # Phase 1
        0, 1,         # Phase 2
        0, 1, 4,      # Phase 3
        4, 4, 4       # Use the last value for the rest.
    ])

  def testProportionAsPhaseLength(self):
    sv = scalars.StepWise([
        (0.2, 1),
        (0.2, scalars.STEP),
        (0.3, scalars.STEP ** 2)
    ], total_steps=8)
    self.assertEqual([sv(i) for i in range(10)], [
        # For each phase, scalars.STEP
        # is evaluated to 0 when phase starts.
        1, 1,         # Phase 1
        0, 1,         # Phase 2
        0, 1, 4,      # Phase 3
        4, 4, 4       # Use the last value for the rest.
    ])

  def testBadSpecification(self):
    with self.assertRaisesRegex(
        ValueError,
        '`total_steps` must be specified when float is used as the value'):
      _ = scalars.StepWise([
          (0.2, 1),
          (0.2, scalars.STEP),
          (0.3, scalars.STEP ** 2)])

    with self.assertRaisesRegex(
        ValueError,
        'The sum of all proportions must be greater than 0'):
      _ = scalars.StepWise([
          (0.0, 1),
          (0.0, scalars.STEP),
          (0.0, scalars.STEP ** 2)
      ], total_steps=10)

    with self.assertRaisesRegex(
        ValueError,
        'The phase length should be a float as a proportion of the'):
      _ = scalars.StepWise([
          (1, 1),
          (2, scalars.STEP),
          (3, scalars.STEP ** 2)
      ], total_steps=10)


class HelperFunctionsTest(unittest.TestCase):
  """Test helper functions for popular scalar schedule."""

  def assertIsClose(self, x, y):
    assert abs(x - y) < 1e-15, (x, y)

  def testLinear(self):
    sv = scalars.linear(10, 1, 6)
    self.assertEqual(sv(0), 1)
    self.assertEqual(sv(1), 1.5)
    self.assertEqual(sv(2), 2)
    self.assertEqual(sv(9), 5.5)
    self.assertEqual(sv(10), 6)

    sv = scalars.linear(10, 6, 1)
    self.assertEqual(sv(0), 6)
    self.assertEqual(sv(1), 5.5)
    self.assertEqual(sv(2), 5)
    self.assertEqual(sv(9), 1.5)
    self.assertEqual(sv(10), 1)

  def testCosineDecay(self):
    sv = scalars.cosine_decay(10, end=0.1)
    self.assertIsClose(sv(0), 1.0)
    self.assertIsClose(sv(6), 0.41094235253127376)
    self.assertIsClose(sv(10), 0.1)

  def testExponentialDecay(self):
    sv = scalars.exponential_decay(0.2, decay_interval=2) * 5
    self.assertIsClose(sv(0), 5.0)
    self.assertIsClose(sv(1), 5.0)
    self.assertIsClose(sv(2), 1.0)
    self.assertIsClose(sv(3), 1.0)
    self.assertIsClose(sv(4), 0.2)
    self.assertIsClose(sv(5), 0.2)
    self.assertIsClose(sv(6), 0.04)
    self.assertIsClose(sv(7), 0.04)

    sv = scalars.exponential_decay(0.2, decay_interval=2, staircase=False) * 5
    self.assertIsClose(sv(0), 5.0)
    self.assertIsClose(sv(1), 2.23606797749979)
    self.assertIsClose(sv(2), 1.0)
    self.assertIsClose(sv(3), 0.447213595499958)
    self.assertIsClose(sv(4), 0.2)
    self.assertIsClose(sv(5), 0.0894427190999916)
    self.assertIsClose(sv(6), 0.04)
    self.assertIsClose(sv(7), 0.01788854381999832)

  def testCyclic(self):
    sv = scalars.cyclic(10) * 2
    self.assertIsClose(sv(0), 2.0)
    self.assertIsClose(sv(2), 1.3090169943749475)
    self.assertIsClose(sv(5), 0.0)
    self.assertIsClose(sv(7), 0.6909830056250524)
    self.assertIsClose(sv(10), 2.0)

    sv = scalars.cyclic(10, math.pi) * 2
    self.assertIsClose(sv(0), 0.0)
    self.assertIsClose(sv(2), 0.6909830056250524)
    self.assertIsClose(sv(5), 2.0)
    self.assertIsClose(sv(7), 1.3090169943749477)
    self.assertIsClose(sv(10), 0.0)

if __name__ == '__main__':
  unittest.main()
