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
"""Tests for math scalars."""

import math
import unittest
from pyglove.generators.scalars import base
from pyglove.generators.scalars import maths as scalars


class MathScalarsTest(unittest.TestCase):
  """Math scalars tests."""

  def assertIsClose(self, x, y):
    assert abs(x - y) < 1e-15, (x, y)

  def testSqrt(self):
    sv = scalars.sqrt(base.STEP)
    self.assertEqual(sv(0), 0)
    self.assertEqual(sv(4), 2)

  def testExp(self):
    sv = scalars.exp(base.STEP)
    self.assertEqual(sv(0), 1)
    self.assertEqual(sv(1), math.e)

  def testLog(self):
    sv = scalars.log(base.STEP, 2)
    self.assertEqual(sv(1), 0)
    self.assertEqual(sv(4), 2)

    sv = scalars.log(16, base.STEP)
    self.assertEqual(sv(2), 4)
    self.assertEqual(sv(4), 2)

  def testCos(self):
    sv = scalars.cos(base.STEP * math.pi / 4)
    self.assertIsClose(sv(0), 1)
    self.assertIsClose(sv(1), math.sqrt(2) / 2)
    self.assertIsClose(sv(2), 0)

  def testSin(self):
    sv = scalars.sin(base.STEP * math.pi / 4)
    self.assertIsClose(sv(0), 0)
    self.assertIsClose(sv(1), math.sqrt(2) / 2)
    self.assertIsClose(sv(2), 1)


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
