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
from pyglove.core.utils import common_traits


class ExplicitlyOverrideTest(unittest.TestCase):

  def test_explicitly_override(self):
    class A:

      @common_traits.explicit_method_override
      def __init__(self, x, y):
        pass

      def bar(self):
        pass

    common_traits.ensure_explicit_method_override(A.__init__)
    with self.assertRaisesRegex(TypeError, '.* is a PyGlove managed method'):
      common_traits.ensure_explicit_method_override(A.bar)


if __name__ == '__main__':
  unittest.main()
