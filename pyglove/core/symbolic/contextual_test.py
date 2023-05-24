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
"""Tests for pyglove.symbolic.ContextualGetter."""

import unittest

from pyglove.core.symbolic import base
from pyglove.core.symbolic import contextual
from pyglove.core.symbolic.dict import Dict


class ContextualGetterTest(unittest.TestCase):
  """Tests for `pg.symbolic.ContextualGetter`."""

  def test_basics(self):
    @contextual.contextual_getter
    def static_value(context, v):
      del context
      return v

    getter = static_value(v=1)  # pylint: disable=no-value-for-parameter
    self.assertIsInstance(getter, base.ContextualValue)
    self.assertEqual(
        getter.get(base.GetAttributeContext('x', Dict(), Dict())), 1
    )
    self.assertEqual(base.from_json(base.to_json(getter)), getter)


if __name__ == '__main__':
  unittest.main()
