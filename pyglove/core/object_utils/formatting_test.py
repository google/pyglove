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
"""Tests for pyglove.object_utils.formatting."""
import inspect
import unittest

from pyglove.core.object_utils import common_traits
from pyglove.core.object_utils import formatting


class StringHelperTest(unittest.TestCase):
  """Tests for string helper methods in formatting."""

  def test_kvlist_str(self):
    self.assertEqual(
        formatting.kvlist_str([
            ('', 'foo', None),
            ('a', 1, None),
            ('b', 'str', (None, 'str')),
            ('c', True, False),
        ]), 'foo, a=1, c=True')

  def test_quote_if_str(self):
    self.assertEqual(formatting.quote_if_str(1), 1)
    self.assertEqual(formatting.quote_if_str('foo'), '\'foo\'')
    self.assertEqual(formatting.quote_if_str('foo\'s\na'), '"foo\'s\\na"')

  def test_message_on_path(self):
    self.assertEqual(formatting.message_on_path('hi.', None), 'hi.')
    self.assertEqual(
        formatting.message_on_path('hi.', formatting.KeyPath()),
        'hi. (path=)')

  def test_comma_delimited_str(self):
    self.assertEqual(
        formatting.comma_delimited_str([1, 2, 'abc']), '1, 2, \'abc\'')

  def test_auto_plural(self):
    self.assertEqual(formatting.auto_plural(2, 'number'), 'numbers')
    self.assertEqual(formatting.auto_plural(2, 'was', 'were'), 'were')


class FormatTest(unittest.TestCase):
  """Tests for formatting.format."""

  def test_formattable(self):

    class A(common_traits.Formattable):

      def format(self, compact=True, **kwargs):
        if compact:
          return 'A()'
        else:
          return 'A(...)'

    self.assertEqual(str(A()), 'A(...)')
    self.assertEqual(repr(A()), 'A()')

  def test_simple_types(self):
    self.assertEqual(formatting.format(True, compact=True), 'True')
    self.assertEqual(formatting.format(1, compact=True), '1')
    self.assertEqual(formatting.format(1.0, compact=True), '1.0')
    self.assertEqual(formatting.format('foo', compact=True), '\'foo\'')
    self.assertEqual(
        formatting.format('foo\'s\na', compact=True), '"foo\'s\\na"')

    # Compact=False has no impact on simple types.
    self.assertEqual(formatting.format(True, compact=False), 'True')
    self.assertEqual(formatting.format(1, compact=False), '1')
    self.assertEqual(formatting.format(1.0, compact=False), '1.0')
    self.assertEqual(formatting.format('foo', compact=False), '\'foo\'')
    self.assertEqual(
        formatting.format('foo\'s\na', compact=False), '"foo\'s\\na"')

    # Verbose has no impact on simple types.
    self.assertEqual(formatting.format(True, verbose=True), 'True')
    self.assertEqual(formatting.format(1, verbose=True), '1')
    self.assertEqual(formatting.format(1.0, verbose=True), '1.0')
    self.assertEqual(formatting.format('foo', verbose=True), '\'foo\'')
    self.assertEqual(
        formatting.format('foo\'s\na', verbose=True), '"foo\'s\\na"')

    # Root indent has no impact on simple types.
    self.assertEqual(formatting.format(True, root_indent=4), 'True')
    self.assertEqual(formatting.format(1, root_indent=4), '1')
    self.assertEqual(formatting.format(1.0, root_indent=4), '1.0')
    self.assertEqual(formatting.format('foo', root_indent=4), '\'foo\'')
    self.assertEqual(
        formatting.format('foo\'s\na', root_indent=4), '"foo\'s\\na"')

  def test_complex_types(self):

    class CustomFormattable(common_traits.Formattable):
      """Custom formattable."""

      def format(self, custom_param=None, **kwargs):
        return f'CustomFormattable({custom_param})'

    class A:
      pass

    self.assertEqual(
        formatting.format(
            {
                'a': CustomFormattable(),
                'b': {
                    'c': [1, 2, 3],
                    'd': ['foo', 'bar\na', 3, 4, 5]
                }
            },
            compact=True,
            custom_param='foo'),
        "{'a': CustomFormattable(foo), 'b': {'c': [1, 2, 3], "
        "'d': ['foo', 'bar\\na', 3, 4, 5]}}")

    self.assertEqual(
        formatting.format(
            {
                'a': A(),
                'b': {
                    'c': [1, 2, 3],
                    'd': ['foo', 'bar\na', 3, 4, 5]
                }
            },
            compact=False,
            list_wrap_threshold=15,
            strip_object_id=True),
        inspect.cleandoc("""{
          'a': A(...),
          'b': {
            'c': [1, 2, 3],
            'd': [
              'foo',
              'bar\\na',
              3,
              4,
              5
            ]
          }
        }"""))

  def test_include_exclude_keys(self):
    """Test format with excluded keys."""

    class A:
      pass

    class B(common_traits.Formattable):
      """Custom formattable."""

      def format(
          self, custom_param=None,
          include_keys=None, exclude_keys=None, **kwargs):
        exclude_keys = exclude_keys or set()
        kv = dict(a=1, b=2, c=3)
        def _should_include(k):
          if include_keys:
            return k in include_keys
          return k not in exclude_keys
        kv_pairs = [(k, v, None) for k, v in kv.items() if _should_include(k)]
        return f'B({formatting.kvlist_str(kv_pairs, compact=True)})'

    self.assertEqual(
        formatting.format(B(), compact=False, include_keys=set(['a', 'c'])),
        'B(a=1, c=3)')
    self.assertEqual(
        formatting.format(B(), compact=False, exclude_keys=set(['a', 'c'])),
        'B(b=2)')
    self.assertEqual(
        formatting.format(
            {
                'a': A(),
                'b': B(),
                'c': {
                    'd': [1, 2, 3],
                }
            },
            compact=False,
            list_wrap_threshold=15,
            strip_object_id=True,
            # 'a' should be removed, but 'b.a', 'c.d' should be kept as they are
            # not at the top level.
            exclude_keys=set(['a', 'd'])),
        inspect.cleandoc("""{
          'b': B(a=1, b=2, c=3),
          'c': {
            'd': [1, 2, 3]
          }
        }"""))

  def test_markdown(self):
    self.assertEqual(
        formatting.format([1], compact=True, markdown=True), '`[1]`'
    )
    self.assertEqual(
        formatting.format(
            [1, 2, 3], list_wrap_threshold=5, compact=False, markdown=True
        ),
        inspect.cleandoc("""
            ```
            [
              1,
              2,
              3
            ]
            ```
            """),
    )


if __name__ == '__main__':
  unittest.main()
