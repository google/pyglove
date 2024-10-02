# Copyright 2024 The Langfun Authors
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
"""Tests for HTML utilities.."""

import dataclasses
import inspect
from typing import Any
import unittest

from pyglove.core.object_utils import html_formatting

KeyPath = html_formatting.KeyPath
Html = html_formatting.Html
HtmlComponent = html_formatting.HtmlComponent
HtmlView = html_formatting.HtmlView
HtmlFormattable = html_formatting.HtmlFormattable
DefaultView = html_formatting._DefaultHtmlView

# pylint: disable=line-too-long


class HtmlTest(unittest.TestCase):

  def assert_html(self, actual, expected, newline: bool = False):
    expected = inspect.cleandoc(expected)
    if newline:
      expected += '\n'
    self.assertEqual(actual, expected)

  def test_basics(self):
    html = Html(
        '<h1>foo</h1>',
        styles=['h1 {color: red;}'],
        scripts=['function myFun1(p1, p2) { return p1 * p2; }']
    )
    self.assert_html(html.body_content, '<h1>foo</h1>')
    self.assert_html(
        html.style_section,
        """
        <style>
        h1 {color: red;}
        </style>
        """,
        newline=True
    )
    self.assert_html(
        html.script_section,
        """
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        """,
        newline=True
    )

    # Adding the same style.
    html.add_style('h1 {color: red;}')
    self.assertEqual(html.styles, ['h1 {color: red;}'])

    html.include_style('./style1.css')
    self.assertEqual(html.style_files, ['./style1.css'])
    self.assert_html(
        html.style_section,
        """
        <link rel="stylesheet" href="./style1.css">
        <style>
        h1 {color: red;}
        </style>
        """,
        newline=True
    )

    html.add_script('function myFun1(p1, p2) { return p1 * p2; }')
    self.assertEqual(
        html.scripts,
        ['function myFun1(p1, p2) { return p1 * p2; }']
    )
    html.include_script('./script1.js')
    self.assertEqual(html.script_files, ['./script1.js'])
    self.assert_html(
        html.script_section,
        """
        <script src="./script1.js"></script>
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        """,
        newline=True
    )

    html.add('<h2>bar</h2>')
    html.add_style('h2 {color: blue;}')
    html.add_script('function myFun2(p1, p2) { return p1 + p2; }')
    self.assert_html(
        html._repr_html_(),
        """
        <html>
        <head>
        <link rel="stylesheet" href="./style1.css">
        <style>
        h1 {color: red;}
        h2 {color: blue;}
        </style>
        <script src="./script1.js"></script>
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        function myFun2(p1, p2) { return p1 + p2; }
        </script>
        </head>
        <body>
        <h1>foo</h1><h2>bar</h2>
        </body>
        </html>
        """,
        newline=True
    )
    self.assertEqual(
        repr(html),
        """Html(body_content='<h1>foo</h1><h2>bar</h2>', style_files=['./style1.css'], styles=['h1 {color: red;}', 'h2 {color: blue;}'], script_files=['./script1.js'], scripts=['function myFun1(p1, p2) { return p1 * p2; }', 'function myFun2(p1, p2) { return p1 + p2; }'])"""
    )
    self.assert_html(
        str(html),
        """
        <html>
        <head>
        <link rel="stylesheet" href="./style1.css">
        <style>
        h1 {color: red;}
        h2 {color: blue;}
        </style>
        <script src="./script1.js"></script>
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        function myFun2(p1, p2) { return p1 + p2; }
        </script>
        </head>
        <body>
        <h1>foo</h1><h2>bar</h2>
        </body>
        </html>
        """,
        newline=True
    )

  def test_eq(self):
    def make_test_html(style_file):
      return Html(
          '<h1>foo</h1>',
          styles=['h1 {color: red;}'],
          style_files=[f'./{style_file}.css'],
          scripts=['function myFun1(p1, p2) { return p1 * p2; }'],
          script_files=['./script1.js'],
      )
    html = make_test_html('style1')
    self.assertEqual(html, make_test_html('style1'))
    self.assertNotEqual(html, make_test_html('style2'))
    self.assertEqual(hash(html), hash(make_test_html('style1')))

  def test_add(self):
    html1 = Html()
    html1.add_script('function myFun1(p1, p2) { return p1 * p2; }')
    html1.add_style('div.a { color: red; }')
    html1.add_style('div.b { color: red; }')

    html2 = Html()
    html2.add_style('div.a { color: red; }')
    html2.add_style('div.b { color: green; }')
    html2.add_style('div.c { color: blue; }')
    html1.add_script('function myFun1(p1, p2) { return p1 * p2; }')
    html2.add('<div class="c">bar</div>')
    html2.add_script('console.log("bar");', local=True)

    html1.add('<div class="a">foo</div>')
    html1.add_script('console.log("foo");', local=True)
    html1.add(html2)

    self.assert_html(
        html1._repr_html_(),
        """
        <html>
        <head>
        <style>
        div.a { color: red; }
        div.b { color: red; }
        div.b { color: green; }
        div.c { color: blue; }
        </style>
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        </head>
        <body>
        <div class="a">foo</div><script>
        console.log("foo");
        </script><div class="c">bar</div><script>
        console.log("bar");
        </script>
        </body>
        </html>
        """,
        newline=True
    )
    self.assert_html(
        html1.html_str(content_only=True),
        """
        <div class="a">foo</div><script>
        console.log("foo");
        </script><div class="c">bar</div><script>
        console.log("bar");
        </script>
        """
    )

  def test_from_value(self):
    html = Html(
        'hi', styles=['h1 {color: red;}'], scripts=['console.log("hi");']
    )
    self.assertIs(Html.from_value(html), html)
    self.assertIsNone(Html.from_value(None))
    self.assertEqual(
        Html.from_value('abc'), Html('abc')
    )
    html2 = Html.from_value(html, copy=True)
    self.assertIsNot(html2, html)
    self.assertEqual(html2, html)

    with self.assertRaises(TypeError):
      Html.from_value(1)

  def test_add_and_radd(self):
    s1 = Html('<h1>foo</h1>', styles=['h1 {color: red;}'])
    s2 = Html('<h2>bar</h2>', styles=['h2 {color: blue;}'])
    s3 = s1 + s2
    self.assertIsNot(s3, s1)
    self.assertIsNot(s3, s2)
    self.assertEqual(
        s3,
        Html(
            '<h1>foo</h1><h2>bar</h2>',
            styles=['h1 {color: red;}', 'h2 {color: blue;}'],
        )
    )
    s4 = s1 + '<h3>baz</h3>'
    self.assertEqual(
        s4,
        Html(
            '<h1>foo</h1><h3>baz</h3>',
            styles=['h1 {color: red;}'],
        )
    )
    s5 = '<h3>baz</h3>' + s1
    self.assertIsNot(s5, s1)
    self.assertEqual(
        s5,
        Html(
            '<h3>baz</h3><h1>foo</h1>',
            styles=['h1 {color: red;}'],
        )
    )


class HtmlComponentTest(unittest.TestCase):

  def assert_html(self, actual, expected, newline: bool = False):
    expected = inspect.cleandoc(expected)
    if newline:
      expected += '\n'
    self.assertEqual(actual, expected)

  def test_nesting(self):

    @dataclasses.dataclass
    class Foo(HtmlComponent):
      text: str

      HTML = '<span class="foo">{{text}}</span>'
      STYLES = [
          """
          .foo {color: red;}
          """
      ]
      STYLE_FILES = [
          './foo.css'
      ]
      SCRIPTS = [
          """
          function myFun1(p1, p2) { return p1 * p2; }
          """
      ]
      SCRIPT_FILES = [
          './foo.js'
      ]

    self.assert_html(
        str(Foo(text='hello')),
        """
        <html>
        <head>
        <link rel="stylesheet" href="./foo.css">
        <style>
        .foo {color: red;}
        </style>
        <script src="./foo.js"></script>
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        </head>
        <body>
        <span class="foo">hello</span>
        </body>
        </html>
        """,
        newline=True
    )

    @dataclasses.dataclass
    class Bar(HtmlComponent):
      foo: Foo

      HTML = '<div class="bar">{{foo}}</div>'
      STYLES = [
          """
          .bar {color: green;}
          """
      ]
      STYLE_FILES = [
          './bar.css'
      ]
      SCRIPTS = [
          """
          function myFun2(p1, p2) { return p1 + p2; }
          """
      ]
      SCRIPT_FILES = [
          './bar.js'
      ]

    self.assert_html(
        str(Bar(foo=Foo(text='hello'))),
        """
        <html>
        <head>
        <link rel="stylesheet" href="./bar.css">
        <link rel="stylesheet" href="./foo.css">
        <style>
        .bar {color: green;}
        .foo {color: red;}
        </style>
        <script src="./bar.js"></script>
        <script src="./foo.js"></script>
        <script>
        function myFun2(p1, p2) { return p1 + p2; }
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        </head>
        <body>
        <div class="bar"><span class="foo">hello</span></div>
        </body>
        </html>
        """,
        newline=True
    )

    @dataclasses.dataclass
    class Baz(HtmlComponent):
      x: Any

      HTML = '<div class="baz">{{x}} {{y}}</div>'
      STYLES = [
          """
          .baz {color: blue;}
          """
      ]
      STYLE_FILES = [
          './baz.css'
      ]
      SCRIPTS = [
          """
          function myFun3(p1, p2) { return p1 + p2; }
          """
      ]
      SCRIPT_FILES = [
          './baz.js'
      ]

      @property
      def y(self):
        return Html(
            'Yes!',
            styles=['.y {color: red;}'],
            script_files=['./y.js'],
        ).add(
            Foo(text='No')
        )

    self.assert_html(
        str(Baz(Baz(x=1))),
        """
        <html>
        <head>
        <link rel="stylesheet" href="./baz.css">
        <link rel="stylesheet" href="./foo.css">
        <style>
        .baz {color: blue;}
        .y {color: red;}
        .foo {color: red;}
        </style>
        <script src="./baz.js"></script>
        <script src="./y.js"></script>
        <script src="./foo.js"></script>
        <script>
        function myFun3(p1, p2) { return p1 + p2; }
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        </head>
        <body>
        <div class="baz"><div class="baz">1 Yes!<span class="foo">No</span></div> Yes!<span class="foo">No</span></div>
        </body>
        </html>
        """,
        newline=True
    )
    self.assert_html(
        str(Baz(Baz(x=Bar(Foo('hello'))))),
        """
        <html>
        <head>
        <link rel="stylesheet" href="./baz.css">
        <link rel="stylesheet" href="./bar.css">
        <link rel="stylesheet" href="./foo.css">
        <style>
        .baz {color: blue;}
        .bar {color: green;}
        .foo {color: red;}
        .y {color: red;}
        </style>
        <script src="./baz.js"></script>
        <script src="./bar.js"></script>
        <script src="./foo.js"></script>
        <script src="./y.js"></script>
        <script>
        function myFun3(p1, p2) { return p1 + p2; }
        function myFun2(p1, p2) { return p1 + p2; }
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        </head>
        <body>
        <div class="baz"><div class="baz"><div class="bar"><span class="foo">hello</span></div> Yes!<span class="foo">No</span></div> Yes!<span class="foo">No</span></div>
        </body>
        </html>
        """,
        newline=True
    )

  def test_bad_usages(self):

    @dataclasses.dataclass
    class Foo(HtmlComponent):
      HTML = '<span class="foo">{{text}}</span>'

    with self.assertRaisesRegex(
        ValueError, 'Missing variable \'text\''
    ):
      Foo()

    @dataclasses.dataclass
    class Bar(HtmlComponent):
      HTML = '<hr>'

    with self.assertRaisesRegex(
        ValueError, 'Adding content through `HtmlComponent.add` not supported'
    ):
      Bar().add('hi')

    class Baz(HtmlComponent):
      HTML = '{% iff %}'

    with self.assertRaisesRegex(
        ValueError, 'Bad template string'
    ):
      Baz()


class HtmlViewTest(unittest.TestCase):

  def test_basics(self):

    class TestView(HtmlView):

      VIEW_TYPE = 'test'

      def render(self, value: Any, **kwargs):
        return Html('foo')

      def render_summary(self, value: Any, **kwargs):
        return Html('summary')

      def render_tooltip(self, value: Any, **kwargs):
        return Html('tooltip')

      def render_key(self, value: Any, **kwargs):
        return Html('key')

      def render_content(self, value: Any, **kwargs):
        return Html('key')

    self.assertIsInstance(HtmlView.get('test'), TestView)
    self.assertEqual(
        html_formatting.to_html_str(1, view='test'),
        '<html>\n<head>\n</head>\n<body>\nfoo\n</body>\n</html>\n'
    )

  def test_unregistered(self):
    class UnregisteredView(HtmlView):  # pylint: disable=unused-variable

      def render(self, value: Any, **kwargs):
        return Html('bar')

      def render_summary(self, value: Any, **kwargs):
        return Html('summary')

      def render_tooltip(self, value: Any, **kwargs):
        return Html('tooltip')

      def render_key(self, value: Any, **kwargs):
        return Html('key')

      def render_content(self, value: Any, **kwargs):
        return Html('key')

    with self.assertRaisesRegex(
        ValueError, 'Unknown view type'
    ):
      HtmlView.get('not_registered')

    self.assertEqual(
        html_formatting.to_html_str(1, view=UnregisteredView()),
        '<html>\n<head>\n</head>\n<body>\nbar\n</body>\n</html>\n'
    )

  def test_tooltip_setting(self):
    setting = HtmlView.TooltipSetting()
    self.assertTrue(setting.enable_tooltip)

    setting = HtmlView.TooltipSetting.from_kwargs(dict(enable_tooltip=False))
    self.assertFalse(setting.enable_tooltip)

  def test_summary_setting(self):
    self.maxDiff = None
    setting = HtmlView.SummarySetting()
    self.assertEqual(
        setting,
        HtmlView.SummarySetting(
            enable_summary=None,
            max_str_len=40,
            tooltip=HtmlView.TooltipSetting()
        )
    )
    setting = HtmlView.SummarySetting.from_kwargs(
        dict(enable_summary=False,
             max_summary_len_for_str=10,
             enable_tooltip=False)
    )
    self.assertEqual(
        setting,
        HtmlView.SummarySetting(
            enable_summary=False,
            max_str_len=10,
            tooltip=HtmlView.TooltipSetting(enable_tooltip=False)
        )
    )

  def test_key_setting(self):
    setting = HtmlView.KeySetting()
    self.assertEqual(
        setting,
        HtmlView.KeySetting(
            tooltip=HtmlView.TooltipSetting()
        )
    )
    setting = HtmlView.KeySetting.from_kwargs(
        dict(enable_tooltip=False)
    )
    self.assertEqual(
        setting,
        HtmlView.KeySetting(
            tooltip=HtmlView.TooltipSetting(enable_tooltip=False)
        )
    )

  def test_content_setting(self):
    child_key = HtmlView.ContentSetting.ChildKey.from_kwargs(
        dict(
            special_keys=[1, 2],
            include_keys=['a', 'b'],
            exclude_keys={'c', 'd'},
            enable_tooltip=False
        )
    )
    self.assertEqual(
        child_key,
        HtmlView.ContentSetting.ChildKey(
            special_keys=[1, 2],
            include_keys={'a', 'b'},
            exclude_keys={'c', 'd'},
            setting=HtmlView.KeySetting(
                tooltip=HtmlView.TooltipSetting(enable_tooltip=False)
            ),
        )
    )

    child_value = HtmlView.ContentSetting.ChildValue.from_kwargs(
        dict(hide_frozen=False,
             hide_default_values=True,
             use_inferred=False)
    )
    self.assertEqual(
        child_value,
        HtmlView.ContentSetting.ChildValue(
            hide_frozen=False,
            hide_default_values=True,
            use_inferred=False,
        )
    )

    collapsing = HtmlView.ContentSetting.Collapsing.from_kwargs(
        dict(collapse_level=2, uncollapse=['a[0]', KeyPath.parse('b.c')]),
        root_path=KeyPath()
    )
    self.assertEqual(
        collapsing,
        HtmlView.ContentSetting.Collapsing(
            level=2, unless={
                KeyPath(),
                KeyPath('a'),
                KeyPath.parse('a[0]'),
                KeyPath.parse('b'),
                KeyPath.parse('b.c')}
        )
    )
    setting = HtmlView.ContentSetting()
    self.assertEqual(
        setting,
        HtmlView.ContentSetting(
            child_key=HtmlView.ContentSetting.ChildKey(),
            child_value=HtmlView.ContentSetting.ChildValue(),
            collapsing=HtmlView.ContentSetting.Collapsing(),
        )
    )

    setting = HtmlView.ContentSetting.from_kwargs(
        dict(
            collapse_level=2,
            uncollapse=['a[0]', KeyPath.parse('b.c')],
            special_keys=[1, 2],
            include_keys=['a', 'b'],
            exclude_keys={'c', 'd'},
            hide_frozen=False,
            hide_default_values=True,
            use_inferred=False,
            enable_tooltip=False
        ),
        root_path=KeyPath()
    )
    self.assertEqual(
        setting,
        HtmlView.ContentSetting(
            child_key=child_key,
            child_value=child_value,
            collapsing=collapsing,
        )
    )

  def test_view_setting(self):
    self.assertEqual(
        HtmlView.ViewSetting(),
        HtmlView.ViewSetting(
            summary=HtmlView.SummarySetting(),
            content=HtmlView.ContentSetting(),
        )
    )
    self.assertEqual(
        HtmlView.ViewSetting().from_kwargs(
            dict(
                enable_summary=False,
                max_summary_len_for_str=10,
                enable_tooltip=False,
                special_keys=[1, 2],
                include_keys=['a', 'b'],
                exclude_keys={'c', 'd'},
                hide_frozen=False,
                hide_default_values=True,
                use_inferred=False,
                collapse_level=2,
                uncollapse=['a[0]', KeyPath.parse('b.c')],
            ),
            root_path=KeyPath()
        ),
        HtmlView.ViewSetting(
            summary=HtmlView.SummarySetting(
                enable_summary=False,
                max_str_len=10,
                tooltip=HtmlView.TooltipSetting(enable_tooltip=False)
            ),
            content=HtmlView.ContentSetting(
                child_key=HtmlView.ContentSetting.ChildKey(
                    special_keys=[1, 2],
                    include_keys={'a', 'b'},
                    exclude_keys={'c', 'd'},
                    setting=HtmlView.KeySetting(
                        tooltip=HtmlView.TooltipSetting(enable_tooltip=False)
                    ),
                ),
                child_value=HtmlView.ContentSetting.ChildValue(
                    hide_frozen=False,
                    hide_default_values=True,
                    use_inferred=False,
                ),
                collapsing=HtmlView.ContentSetting.Collapsing(
                    level=2, unless={
                        KeyPath(),
                        KeyPath('a'),
                        KeyPath.parse('a[0]'),
                        KeyPath.parse('b'),
                        KeyPath.parse('b.c')}
                ),
            ),
        )
    )


class DefaultHtmlViewTest(unittest.TestCase):

  maxDiff = None

  def assert_pattern(self, pattern, value, **kwargs):
    html = html_formatting.to_html_str(value, **kwargs)
    if isinstance(pattern, str):
      pattern = [pattern]
    for p in pattern:
      self.assertRegex(html, p)

  def _assert_equal(self, actual, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = actual.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(expected, actual)

  def assert_html(self, value, expected, **kwargs):
    self._assert_equal(
        html_formatting.to_html_str(value, **kwargs),
        expected,
    )

  def assert_body(self, value, expected, **kwargs):
    self._assert_equal(
        html_formatting.to_html(value, **kwargs).body_content,
        expected,
    )

  def assert_style(self, value, expected, **kwargs):
    self._assert_equal(
        html_formatting.to_html(value, **kwargs).style_section,
        expected
    )

  def test_tooltip(self):
    self.assert_html(
        DefaultView.Tooltip(
            value='hello',
            root_path=KeyPath(),
            setting=DefaultView.TooltipSetting(
                enable_tooltip=True
            ),
            current_view=DefaultView()
        ),
        """
        <html>
        <head>
        <style>
        span.tooltip {
          visibility: hidden;
          white-space: pre-wrap;
          font-weight: normal;
          background-color: #808080;
          color: #fff;
          padding: 6px;
          border-radius: 6px;
          position: absolute;
          z-index: 1;
        }
        </style>
        </head>
        <body>
        <span class="tooltip">&#x27;hello&#x27;</span>
        </body>
        </html>
        """
    )
    self.assertIsNone(
        DefaultView.Tooltip(
            value='hello',
            root_path=KeyPath(),
            setting=DefaultView.TooltipSetting(
                enable_tooltip=False
            ),
            current_view=DefaultView()
        ),
    )

  def test_summary(self):
    # Case 1: No summary for primitive types.
    self.assertIsNone(
        DefaultView.Summary(
            value=1,
            name=None,
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(),
            current_view=DefaultView(),
        )
    )

    # Case 2: No summary for short strings.
    self.assertIsNone(
        DefaultView.Summary(
            value='abcd',
            name=None,
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(),
            current_view=DefaultView(),
        )
    )

    # Case 3: Have summary for long strings.
    self.assert_body(
        DefaultView.Summary(
            value='abcdefgh',
            name=None,
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(
                max_str_len=6
            ),
            current_view=DefaultView(),
        ),
        """
        <summary>
        <div class="summary_title t_str">&#x27;abcdef...&#x27;</div>
        <span class="tooltip">&#x27;abcdefgh&#x27;</span>
        </summary>
        """
    )

    # Case 4: Have summary with custom title.
    self.assert_body(
        DefaultView.Summary(
            value=1,
            name=None,
            title='Custom title',
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(
                max_str_len=6
            ),
            current_view=DefaultView(),
        ),
        """
        <summary>
        <div class="summary_title t_int">Custom title</div>
        <span class="tooltip">1</span>
        </summary>
        """
    )

    # Case 5: Have summary with name.
    self.assert_body(
        DefaultView.Summary(
            value=1,
            name='count',
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(
                max_str_len=6
            ),
            current_view=DefaultView(),
        ),
        """
        <summary>
        <div class="summary_name">count</div><div class="summary_title t_int">int(...)</div>
        <span class="tooltip">1</span>
        </summary>
        """
    )

    # Case 6: Force enable summary.
    self.assert_body(
        DefaultView.Summary(
            value=1,
            name=None,
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(
                enable_summary=True,
            ),
            current_view=DefaultView(),
        ),
        """
        <summary>
        <div class="summary_title t_int">int(...)</div>
        <span class="tooltip">1</span>
        </summary>
        """
    )

    # Case 7: Force disable summary.
    self.assertIsNone(
        DefaultView.Summary(
            value=1,
            name=None,
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(
                enable_summary=False,
            ),
            current_view=DefaultView(),
        ),
    )

    # Case 8: Disable tooltip.
    self.assert_body(
        DefaultView.Summary(
            value=dict(x=1),
            name=None,
            title=None,
            title_class=None,
            root_path=KeyPath(),
            setting=DefaultView.SummarySetting(
                tooltip=DefaultView.TooltipSetting(
                    enable_tooltip=False
                )
            ),
            current_view=DefaultView(),
        ),
        """
        <summary>
        <div class="summary_title t_dict">dict(...)</div>

        </summary>
        """
    )

  def test_key(self):
    # Case 1: String key.
    self.assert_body(
        DefaultView.Key(
            key='foo',
            value=1,
            root_path=KeyPath(),
            setting=DefaultView.KeySetting(),
            current_view=DefaultView(),
        ),
        """
        <span class="object_key k_str v_int">foo</span>
        <span class="tooltip"></span>
        """
    )
    # Case 2: Int key.
    self.assert_body(
        DefaultView.Key(
            key=1,
            value=1,
            root_path=KeyPath(),
            setting=DefaultView.KeySetting(),
            current_view=DefaultView(),
        ),
        """
        <span class="object_key k_int v_int">1</span>
        <span class="tooltip"></span>
        """
    )
    # Case 3: No tooltip.
    self.assert_body(
        DefaultView.Key(
            key=1,
            value=1,
            root_path=KeyPath(),
            setting=DefaultView.KeySetting(
                tooltip=DefaultView.TooltipSetting(enable_tooltip=False)
            ),
            current_view=DefaultView(),
        ),
        """
        <span class="object_key k_int v_int">1</span>
        """,
        newline=True
    )

  def test_content(self):
    # Case 1: Simple type.
    self.assert_body(
        DefaultView.Content(
            value=1,
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(),
            current_view=DefaultView(),
        ),
        """
        <span class="simple_value v_int">1</span>
        """
    )

    # Case 2: Short string (use repr()).
    self.assert_body(
        DefaultView.Content(
            value='ab\ncd',
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(),
            current_view=DefaultView(),
        ),
        """
        <span class="simple_value v_str">&#x27;ab\\ncd&#x27;</span>
        """
    )

    # Case 3: Long string (use str).
    self.assert_body(
        DefaultView.Content(
            value='ab\ncd' * 5,
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(
                summary=DefaultView.SummarySetting(max_str_len=2)
            ),
            current_view=DefaultView(),
        ),
        """
        <span class="simple_value v_str">ab
        cdab
        cdab
        cdab
        cdab
        cd</span>
        """
    )

    # Case 4: Lists.
    self.assert_body(
        DefaultView.Content(
            value=[1, 2, 3],
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(),
            current_view=DefaultView(),
        ),
        """
        <div><table><tr><td><span class="object_key k_int v_int">0</span>
        <span class="tooltip">[0]</span></td><td><span class="simple_value v_int">1</span>
        </td></tr><tr><td><span class="object_key k_int v_int">1</span>
        <span class="tooltip">[1]</span></td><td><span class="simple_value v_int">2</span>
        </td></tr><tr><td><span class="object_key k_int v_int">2</span>
        <span class="tooltip">[2]</span></td><td><span class="simple_value v_int">3</span>
        </td></tr></table></div>
        """
    )
    # Case 5: Dict.
    self.assert_body(
        DefaultView.Content(
            value=dict(x=1, y='abc'),
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(),
            current_view=DefaultView(),
        ),
        """
        <div><table><tr><td><span class="object_key k_str v_int">x</span>
        <span class="tooltip">x</span></td><td><span class="simple_value v_int">1</span>
        </td></tr><tr><td><span class="object_key k_str v_str">y</span>
        <span class="tooltip">y</span></td><td><span class="simple_value v_str">&#x27;abc&#x27;</span>
        </td></tr></table></div>
        """
    )

    # Case 6: Other types.

    class Foo():

      def __str__(self):
        return 'Foo()'

    self.assert_body(
        DefaultView.Content(
            value=Foo(),
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(),
            current_view=DefaultView(),
        ),
        """
        <span class="simple_value v_Foo">Foo()</span>
        """
    )

    # Case 7: Special keys.
    self.assert_body(
        DefaultView.Content(
            value=dict(x=1, y='abc'),
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(
                content=DefaultView.ContentSetting(
                    child_key=DefaultView.ContentSetting.ChildKey(
                        special_keys=['y']
                    )
                )
            ),
            current_view=DefaultView(),
        ),
        """
        <details class="pyglove">
        <summary>
        <div class="summary_name">y</div><div class="summary_title t_str">&#x27;abc&#x27;</div>
        <span class="tooltip">&#x27;abc&#x27;</span>
        </summary>
        <span class="simple_value v_str">&#x27;abc&#x27;</span>
        </details>
        <div><table><tr><td><span class="object_key k_str v_int">x</span>
        <span class="tooltip">x</span></td><td><span class="simple_value v_int">1</span>
        </td></tr></table></div>
        """
    )

    # Case 9: Key filters.
    self.assert_body(
        DefaultView.Content(
            value=dict(x=1, y='abc', z='def'),
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(
                content=DefaultView.ContentSetting(
                    child_key=DefaultView.ContentSetting.ChildKey(
                        include_keys=['x', 'y'],
                        exclude_keys={'y'},
                    )
                )
            ),
            current_view=DefaultView(),
        ),
        """
        <div><table><tr><td><span class="object_key k_str v_int">x</span>
        <span class="tooltip">x</span></td><td><span class="simple_value v_int">1</span>
        </td></tr></table></div>
        """
    )

    # Case 10: Collapsing.
    self.assertEqual(DefaultView.Content(
        value=dict(a=dict(b=dict(), c=dict(d=dict()))),
        name=None,
        root_path=KeyPath(),
        setting=DefaultView.ViewSetting(
            content=DefaultView.ContentSetting(
                collapsing=DefaultView.ContentSetting.Collapsing(
                    level=0
                )
            )
        ),
        current_view=DefaultView(),
    ).body_content.count(' open'), 0)

    self.assertEqual(DefaultView.Content(
        value=dict(a=dict(b=dict(), c=dict(d=dict()))),
        name=None,
        root_path=KeyPath(),
        setting=DefaultView.ViewSetting(
            content=DefaultView.ContentSetting(
                collapsing=DefaultView.ContentSetting.Collapsing(
                    level=4
                )
            )
        ),
        current_view=DefaultView(),
    ).body_content.count(' open'), 4)

    self.assertEqual(
        DefaultView.Content(
            value=dict(a=dict(b=dict(), c=dict(d=dict()))),
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(
                content=DefaultView.ContentSetting(
                    collapsing=DefaultView.ContentSetting.Collapsing.from_kwargs(
                        dict(collapse_level=0, uncollapse=['a.c']),
                        KeyPath()
                    )
                )
            ),
            current_view=DefaultView(),
        ).body_content.count(' open'),
        2   # a and a.c (Not counting the root dict which only has content.)
    )

  def test_object_view(self):

    class Foo():

      def __str__(self):
        return 'Foo()'

    self.assert_html(
        DefaultView.ObjectView(
            value=dict(x=1, y=['abc'], z=dict(a=Foo())),
            name=None,
            root_path=KeyPath(),
            setting=DefaultView.ViewSetting(),
            current_view=DefaultView(),
        ),
        """
        <html>
        <head>
        <style>
        details.pyglove {
          border: 1px solid #aaa;
          border-radius: 4px;
          padding: 0.5em 0.5em 0;
          margin: 0.5em 0;
        }
        details.pyglove[open] {
          padding: 0.5em 0.5em 0.5em;
        }
        details.pyglove summary {
          font-weight: bold;
          margin: -0.5em -0.5em 0;
          padding: 0.5em;
        }
        .summary_name {
          display: inline;
          padding: 0 5px;
        }
        .summary_title {
          display: inline;
        }
        .summary_name + div.summary_title {
          display: inline;
          color: #aaa;
        }
        .summary_title.t_Ref::before {
          content: 'ref: ';
          color: #aaa;
        }
        .summary_title:hover + span.tooltip {
          visibility: visible;
        }
        .t_str {
          color: darkred;
          font-style: italic;
        }
        .empty_container::before {
            content: '(empty)';
            font-style: italic;
            margin-left: 0.5em;
            color: #aaa;
        }
        span.tooltip {
          visibility: hidden;
          white-space: pre-wrap;
          font-weight: normal;
          background-color: #808080;
          color: #fff;
          padding: 6px;
          border-radius: 6px;
          position: absolute;
          z-index: 1;
        }
        span.object_key {
          margin-right: 0.25em;
        }
        .k_str{
          color: white;
          background-color: #ccc;
          border-radius: 0.2em;
          padding: 0.3em;
        }
        .k_int{
            color: #aaa;
        }
        .k_int::before{
            content: '[';
        }
        .k_int::after{
            content: ']';
        }
        span.object_key:hover + span.tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        .simple_value {
          color: blue;
          display: inline-block;
          white-space: pre-wrap;
          padding: 0.2em;
        }
        .simple_value.v_str {
          color: darkred;
          font-style: italic;
        }
        .simple_value.v_int, .simple_value.v_float {
          color: darkblue;
        }
        </style>
        <script>
        </script>
        </head>
        <body>
        <details class="pyglove" open>
        <summary>
        <div class="summary_title t_dict">dict(...)</div>
        <span class="tooltip">{
          &#x27;x&#x27;: 1,
          &#x27;y&#x27;: [&#x27;abc&#x27;],
          &#x27;z&#x27;: {
            &#x27;a&#x27;: Foo()
          }
        }</span>
        </summary>
        <div><table><tr><td><span class="object_key k_str v_int">x</span>
        <span class="tooltip">x</span></td><td><span class="simple_value v_int">1</span>
        </td></tr><tr><td><span class="object_key k_str v_list">y</span>
        <span class="tooltip">y</span></td><td><details class="pyglove">
        <summary>
        <div class="summary_title t_list">list(...)</div>
        <span class="tooltip">[&#x27;abc&#x27;]</span>
        </summary>
        <div><table><tr><td><span class="object_key k_int v_str">0</span>
        <span class="tooltip">y[0]</span></td><td><span class="simple_value v_str">&#x27;abc&#x27;</span>
        </td></tr></table></div>
        </details>
        </td></tr><tr><td><span class="object_key k_str v_dict">z</span>
        <span class="tooltip">z</span></td><td><details class="pyglove">
        <summary>
        <div class="summary_title t_dict">dict(...)</div>
        <span class="tooltip">{
          &#x27;a&#x27;: Foo()
        }</span>
        </summary>
        <div><table><tr><td><span class="object_key k_str v_Foo">a</span>
        <span class="tooltip">z.a</span></td><td><details class="pyglove">
        <summary>
        <div class="summary_title t_Foo">Foo(...)</div>
        <span class="tooltip">Foo()</span>
        </summary>
        <span class="simple_value v_Foo">Foo()</span>
        </details>
        </td></tr></table></div>
        </details>
        </td></tr></table></div>
        </details>

        </body>
        </html>
        """
    )

# pylint: enable=line-too-long

if __name__ == '__main__':
  unittest.main()
