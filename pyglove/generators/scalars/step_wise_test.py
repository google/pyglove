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
"""Tests for step-wise scalars."""

import unittest
from pyglove.generators.scalars import base
from pyglove.generators.scalars import step_wise as scalars


class StepWiseScalarTest(unittest.TestCase):
  """Test step-wise scalar schedule."""

  def testStepsAsPhaseLength(self):
    sv = scalars.StepWise([
        (2, 1),
        (2, base.STEP),
        (3, base.STEP ** 2)
    ])
    self.assertEqual([sv(i) for i in range(10)], [
        # For each phase, base.STEP
        # is evaluated to 0 when phase starts.
        1, 1,         # Phase 1
        0, 1,         # Phase 2
        0, 1, 4,      # Phase 3
        4, 4, 4       # Use the last value for the rest.
    ])

  def testProportionAsPhaseLength(self):
    sv = scalars.StepWise([
        (0.2, 1),
        (0.2, base.STEP),
        (0.3, base.STEP ** 2)
    ], total_steps=8)
    self.assertEqual([sv(i) for i in range(10)], [
        # For each phase, base.STEP
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
          (0.2, base.STEP),
          (0.3, base.STEP ** 2)])

    with self.assertRaisesRegex(
        ValueError,
        'The sum of all proportions must be greater than 0'):
      _ = scalars.StepWise([
          (0.0, 1),
          (0.0, base.STEP),
          (0.0, base.STEP ** 2)
      ], total_steps=10)

    with self.assertRaisesRegex(
        ValueError,
        'The phase length should be a float as a proportion of the'):
      _ = scalars.StepWise([
          (1, 1),
          (2, base.STEP),
          (3, base.STEP ** 2)
      ], total_steps=10)


if __name__ == '__main__':
  unittest.main()
