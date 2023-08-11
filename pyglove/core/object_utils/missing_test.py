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
"""Tests for pyglove.object_utils.missing."""

import unittest
from pyglove.core.object_utils import json_conversion
from pyglove.core.object_utils import missing


class MissingValueTest(unittest.TestCase):
  """Tests for class MissingValue."""

  def test_basics(self):
    self.assertEqual(missing.MissingValue(),
                     missing.MissingValue())
    self.assertNotEqual(missing.MissingValue(), 1)
    self.assertNotEqual(missing.MissingValue(), {})

    self.assertEqual(str(missing.MissingValue()), 'MISSING_VALUE')
    self.assertEqual(repr(missing.MissingValue()), 'MISSING_VALUE')

  def test_to_json(self):
    json = json_conversion.to_json(missing.MissingValue())
    self.assertEqual(json_conversion.from_json(json), missing.MissingValue())


if __name__ == '__main__':
  unittest.main()
