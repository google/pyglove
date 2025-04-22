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
import inspect
import io
import unittest

from pyglove.core.utils import formatting


class Foo(formatting.Formattable):

  def format(
      self, compact: bool = False, verbose: bool = True, **kwargs):
    return f'{self.__class__.__name__}(compact={compact}, verbose={verbose})'


class Bar(formatting.Formattable):

  def __init__(self, foo: Foo):
    self._foo = foo

  def format(
      self, compact: bool = False, verbose: bool = True,
      root_indent: int = 0, **kwargs):
    foo_str = self._foo.format(
        compact=compact, verbose=verbose, root_indent=root_indent + 1)
    return f'{self.__class__.__name__}(foo={foo_str})'


class FormattableTest(unittest.TestCase):

  def test_formattable(self):
    foo = Foo()
    self.assertEqual(repr(foo), 'Foo(compact=True, verbose=True)')
    self.assertEqual(str(foo), 'Foo(compact=False, verbose=True)')

  def test_formattable_with_custom_format(self):
    class Baz(Foo):
      __str_format_kwargs__ = {'compact': False, 'verbose': False}
      __repr_format_kwargs__ = {'compact': True, 'verbose': False}

    bar = Baz()
    self.assertEqual(repr(bar), 'Baz(compact=True, verbose=False)')
    self.assertEqual(str(bar), 'Baz(compact=False, verbose=False)')

  def test_formattable_with_context_managers(self):
    foo = Foo()
    with formatting.str_format(verbose=False):
      with formatting.repr_format(compact=False):
        self.assertEqual(repr(foo), 'Foo(compact=False, verbose=True)')
        self.assertEqual(str(foo), 'Foo(compact=False, verbose=False)')


class StringHelperTest(unittest.TestCase):
  """Tests for string helper methods in formatting."""

  def test_raw_text(self):
    raw = formatting.RawText('abc')
    self.assertEqual(formatting.format(raw), 'abc')
    self.assertEqual(formatting.format(raw, compact=True), 'abc')
    self.assertEqual(raw, formatting.RawText('abc'))
    self.assertEqual(raw, 'abc')
    self.assertNotEqual(raw, formatting.RawText('abcd'))
    self.assertNotEqual(raw, 'abcd')

  def test_camel_to_snake(self):
    self.assertEqual(formatting.camel_to_snake('foo'), 'foo')
    self.assertEqual(formatting.camel_to_snake('Foo'), 'foo')
    self.assertEqual(formatting.camel_to_snake('FooBar'), 'foo_bar')
    self.assertEqual(formatting.camel_to_snake('AI'), 'ai')
    self.assertEqual(formatting.camel_to_snake('AIMessage'), 'ai_message')
    self.assertEqual(formatting.camel_to_snake('ABCMeta'), 'abc_meta')
    self.assertEqual(formatting.camel_to_snake('ABC123Meta'), 'abc123_meta')

  def test_special_format_support(self):

    class NewLine:
      def _repr_html_(self):
        return '<hr>'

      def __str__(self):
        return 'NewLine()'

      def __repr__(self):
        return 'NewLine()'

    v = NewLine()
    self.assertEqual(formatting.str_ext(v), 'NewLine()')
    def _method(attr_name: str):
      def fn(v, root_indent):
        del root_indent
        f = getattr(v, attr_name, None)
        return f() if f is not None else None
      return fn

    self.assertEqual(
        formatting.str_ext(v, custom_format=_method('_repr_html_')),
        '<hr>'
    )
    self.assertEqual(
        formatting.str_ext(v, custom_format=_method('_repr_xml_')),
        'NewLine()'
    )
    self.assertEqual(formatting.repr_ext(v), 'NewLine()')
    self.assertEqual(
        formatting.repr_ext(v, custom_format=_method('_repr_html_')),
        '<hr>'
    )
    self.assertEqual(
        formatting.repr_ext(v, custom_format=_method('_repr_xml_')),
        'NewLine()'
    )

  def test_kvlist_str(self):
    self.assertEqual(
        formatting.kvlist_str([
            ('', 'foo', None),
            ('a', 1, None),
            ('b', 'str', (None, 'str')),
            ('c', [1, 2, 3], False),
        ], label='Foo'),
        'Foo(\'foo\', a=1, c=[1, 2, 3])'
    )

    self.assertEqual(
        formatting.kvlist_str([
            ('', 'foo', None),
            ('a', 1, None),
            ('b', 'str', (None, 'str')),
            ('c', True, False),
        ]),
        '\'foo\', a=1, c=True'
    )

    self.assertEqual(
        formatting.kvlist_str([
            ('', 'foo', None),
            ('a', 1, None),
            ('b', 'str', (None, 'str')),
            ('c', True, False),
        ], label='Foo', compact=False),
        'Foo(\n  \'foo\',\n  a=1,\n  c=True\n)'
    )

    self.assertEqual(
        formatting.kvlist_str([
            ('', 'foo', None),
            ('a', 1, None),
            ('b', 'str', (None, 'str')),
            ('c', dict(x=1), False),
        ], compact=False),
        '\'foo\',\na=1,\nc={\n  \'x\': 1\n}'
    )

    self.assertEqual(
        formatting.kvlist_str([
            ('', 'foo', 'foo')
        ], label='Foo', compact=False),
        'Foo()'
    )

    class Foo:    # pylint: disable=redefined-outer-name
      def _repr_xml_(self):
        return '<foo/>'

      def __str__(self):
        return 'Foo()'

    self.assertEqual(
        formatting.kvlist_str([
            ('', Foo(), None)
        ], compact=False),
        'Foo()'
    )
    def _custom_format(v, root_indent):
      del root_indent
      f = getattr(v, '_repr_xml_', None)
      return f() if f is not None else None

    self.assertEqual(
        formatting.kvlist_str([
            ('', Foo(), None)
        ], compact=False, custom_format=_custom_format),
        '<foo/>'
    )
    self.assertEqual(
        formatting.kvlist_str([
            ('', (Foo(), 1), None)
        ], compact=True, custom_format=_custom_format),
        '(<foo/>, 1)'
    )

  def test_quote_if_str(self):
    self.assertEqual(formatting.quote_if_str(1), 1)
    self.assertEqual(formatting.quote_if_str('foo'), '\'foo\'')
    self.assertEqual(formatting.quote_if_str('foo\'s\na'), '"foo\'s\\na"')

  def test_comma_delimited_str(self):
    self.assertEqual(
        formatting.comma_delimited_str([1, 2, 'abc']), '1, 2, \'abc\'')

  def test_auto_plural(self):
    self.assertEqual(formatting.auto_plural(2, 'number'), 'numbers')
    self.assertEqual(formatting.auto_plural(2, 'was', 'were'), 'were')


class FormatTest(unittest.TestCase):
  """Tests for formatting.format."""

  def test_formattable(self):

    class A(formatting.Formattable):

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

    class CustomFormattable(formatting.Formattable):
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

    class B(formatting.Formattable):
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

  def test_custom_format(self):

    class A:

      def _repr_xml_(self):
        return '<a/>'

      def __repr__(self):
        return 'A()'

      def __str__(self):
        return 'AA()'

    self.assertEqual(formatting.format(A), str(A))
    self.assertEqual(formatting.format(A()), 'AA()')
    def _custom_format(v, root_indent):
      del root_indent
      f = getattr(v, '_repr_xml_', None)
      return f() if f else None

    self.assertEqual(
        formatting.format(A(), custom_format=_custom_format), '<a/>'
    )
    self.assertEqual(
        formatting.format(A(), compact=True),
        'A()'
    )
    self.assertEqual(
        formatting.format(A(), compact=True, custom_format=_custom_format),
        '<a/>'
    )
    self.assertEqual(
        formatting.format([A()], compact=True, custom_format=_custom_format),
        '[<a/>]'
    )
    self.assertEqual(
        formatting.format((A(), 1), compact=True, custom_format=_custom_format),
        '(<a/>, 1)'
    )
    self.assertEqual(
        formatting.format(
            dict(x=A()), compact=True, custom_format=_custom_format
        ),
        '{\'x\': <a/>}'
    )

  def test_recursion(self):
    # Recursive dict.
    x = dict(x=1)
    x['y'] = x
    self.assertEqual(
        formatting.format(x, compact=True),
        f'{{\'x\': 1, \'y\': <recursive dict at 0x{id(x):x}>}}'
    )
    self.assertEqual(
        formatting.format(x, compact=False),
        f'{{\n  \'x\': 1,\n  \'y\': <recursive dict at 0x{id(x):x}>\n}}'
    )
    # Non-recursive dict.
    y = dict(x=1)
    a = dict(x=y, y=y)
    self.assertEqual(
        formatting.format(a, compact=True),
        '{\'x\': {\'x\': 1}, \'y\': {\'x\': 1}}'
    )

  def test_markdown(self):

    class A(formatting.Formattable):
      def __init__(self, x):
        self.x = x

      def format(self, *args, **kwargs):
        del args, kwargs
        return 'A(' + formatting.format(self.x) + ')'

    with formatting.str_format(markdown=True):
      self.assertEqual(str(A(1)), '`A(1)`')
      self.assertEqual(str(A([A(1)])), '`A([A(1)])`')
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

  def test_max_len(self):
    self.assertEqual(
        formatting.format('foo', max_str_len=2), '\'fo...\''
    )
    self.assertEqual(
        formatting.format(b'bar', max_bytes_len=2), 'b\'ba...\''
    )

  def test_printv(self):
    with io.StringIO() as f:
      formatting.printv(1, 2, 3, file=f)
      self.assertEqual(f.getvalue(), '1 2 3\n')


if __name__ == '__main__':
  unittest.main()
