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
"""Tests for pyglove.core.typing.callable_ext."""

import unittest
from pyglove.core.typing import annotation_conversion   # pylint: disable=unused-import
from pyglove.core.typing import callable_ext


class CallWithOptionalKeywordArgsTest(unittest.TestCase):
  """Tests for typing.CallWithOptionalKeywordArgs."""

  def test_function(self):
    """Test call with function."""

    def foo(a, b):
      return a + b

    f = callable_ext.CallableWithOptionalKeywordArgs(foo, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)

    def bar(a, **kwargs):
      return sum([a] + list(kwargs.values()))

    f = callable_ext.CallableWithOptionalKeywordArgs(bar, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 6)

  def testMethod(self):
    """Test call with method."""

    class A:

      def __call__(self, a, b):
        return a + b

    f = callable_ext.CallableWithOptionalKeywordArgs(A(), ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)

    class B:

      def __call__(self, a, **kwargs):
        return sum([a] + list(kwargs.values()))

    f = callable_ext.CallableWithOptionalKeywordArgs(B(), ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 6)

  def test_classmethod(self):
    """Test call with class method."""

    class A:

      @classmethod
      def foo(cls, a, b):
        return a + b

    f = callable_ext.CallableWithOptionalKeywordArgs(A.foo, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)


if __name__ == '__main__':
  unittest.main()
