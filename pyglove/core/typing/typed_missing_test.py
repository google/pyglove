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
import unittest

from pyglove.core import utils
from pyglove.core.typing import typed_missing
from pyglove.core.typing import value_specs


class MissingValueTest(unittest.TestCase):
  """Tests for typed MissingValue class."""

  def test_eq(self):
    self.assertEqual(
        typed_missing.MissingValue(value_specs.Int()), utils.MISSING_VALUE
    )

    self.assertEqual(
        utils.MISSING_VALUE, typed_missing.MissingValue(value_specs.Int())
    )

    self.assertEqual(
        typed_missing.MissingValue(value_specs.Int()),
        typed_missing.MissingValue(value_specs.Int()))

    self.assertNotEqual(
        typed_missing.MissingValue(value_specs.Int()),
        typed_missing.MissingValue(value_specs.Int(max_value=1)))

    self.assertNotEqual(
        typed_missing.MissingValue(value_specs.Int()),
        typed_missing.MissingValue(value_specs.Str()))

    m = typed_missing.MissingValue(value_specs.Int())
    self.assertEqual(m, m)

  def test_hash(self):
    self.assertEqual(
        hash(typed_missing.MissingValue(value_specs.Int())),
        hash(typed_missing.MissingValue(value_specs.Float())))

    self.assertEqual(
        hash(typed_missing.MissingValue(value_specs.Int())),
        hash(utils.MISSING_VALUE),
    )

    self.assertNotEqual(
        hash(typed_missing.MissingValue(value_specs.Int())),
        hash(1))

  def test_format(self):
    """Test MissingValue.format."""
    self.assertEqual(
        typed_missing.MissingValue(value_specs.Int()).format(compact=True),
        'MISSING_VALUE')

    self.assertEqual(
        typed_missing.MissingValue(value_specs.Int()).format(compact=False),
        'MISSING_VALUE(Int())')


if __name__ == '__main__':
  unittest.main()
