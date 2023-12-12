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
"""Tests for pyglove.object_utils.common_traits."""

import unittest
from pyglove.core.object_utils import common_traits


class Foo(common_traits.Formattable):

  def format(self, compact: bool = False, verbose: bool = True, **kwargs):
    return f'{self.__class__.__name__}(compact={compact}, verbose={verbose})'


class FormattableTest(unittest.TestCase):

  def test_formattable(self):
    foo = Foo()
    self.assertEqual(repr(foo), 'Foo(compact=True, verbose=True)')
    self.assertEqual(str(foo), 'Foo(compact=False, verbose=True)')

  def test_formattable_with_custom_format(self):
    class Bar(Foo):
      __str_format_kwargs__ = {'compact': False, 'verbose': False}
      __repr_format_kwargs__ = {'compact': True, 'verbose': False}

    bar = Bar()
    self.assertEqual(repr(bar), 'Bar(compact=True, verbose=False)')
    self.assertEqual(str(bar), 'Bar(compact=False, verbose=False)')

  def test_formattable_with_context_managers(self):
    foo = Foo()
    with common_traits.str_format(verbose=False):
      with common_traits.repr_format(compact=False):
        self.assertEqual(repr(foo), 'Foo(compact=False, verbose=True)')
        self.assertEqual(str(foo), 'Foo(compact=False, verbose=False)')


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
