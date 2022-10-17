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
"""Test for evolution decision point filters."""

import unittest
import pyglove.core as pg
from pyglove.ext.evolution import where


class DecisionPointFiltersTest(unittest.TestCase):
  """Tests for DecisionPointFilter subclasses."""

  def testLambda(self):
    w = where.Lambda(lambda xs: xs[:1])
    output = w([pg.geno.Float(0., float(i + 1)) for i in range(5)])
    self.assertEqual(len(output), 1)

  def testAll(self):
    w = where.ALL
    inputs = [pg.geno.Float(0., float(i + 1)) for i in range(5)]
    output = w(inputs)
    self.assertIs(inputs, output)

  def testAny(self):
    w = where.Any(seed=1)
    inputs = [pg.geno.Float(0., float(i + 1)) for i in range(5)]
    output = w(inputs)
    self.assertEqual(len(output), 1)
    self.assertEqual(output[0].max_value, 2.0)

    w = where.Any(4, seed=2)
    output = w(inputs)
    self.assertEqual(len(output), 4)
    # Check output is sorted.
    self.assertEqual([v.max_value for v in output], [1., 2., 4., 5.])

    w = where.Any(10, seed=1)
    output = w(inputs)
    self.assertIs(inputs, output)

  def testAutomaticConversion(self):

    @pg.members([
        ('where', where.where_spec())
    ])
    class MyFilter(pg.Object):
      pass

    w = MyFilter(lambda xs: xs[:2])
    self.assertIsInstance(w.where, where.Lambda)


if __name__ == '__main__':
  unittest.main()
