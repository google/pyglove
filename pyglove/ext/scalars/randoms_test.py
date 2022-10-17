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
"""Tests for random scalars."""

import unittest
from pyglove.ext.scalars import randoms as scalars


class RandomScalarsTest(unittest.TestCase):
  """Random scalars tests."""

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


if __name__ == '__main__':
  unittest.main()
