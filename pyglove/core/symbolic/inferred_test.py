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
"""Tests for pyglove.symbolic.inferred."""

import unittest

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import inferred
from pyglove.core.symbolic.dict import Dict


class ValueFromParentChain(unittest.TestCase):
  """Tests for `pg.symbolic.ValueFromParentChain`."""

  def test_inference(self):
    v = Dict(y=1, x=Dict(x=1, y=inferred.ValueFromParentChain()))
    self.assertEqual(v.x.y, 1)

    v.rebind(y=2)
    self.assertEqual(v.x.y, 2)

  def test_custom_typing(self):
    v = inferred.ValueFromParentChain()
    self.assertIs(pg_typing.Int().apply(v), v)
    self.assertIs(pg_typing.Str().apply(v), v)


if __name__ == '__main__':
  unittest.main()
