# Copyright 2024 The PyGlove Authors
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
import unittest

from pyglove.core.views.html import base

Html = base.Html

# pylint: disable=line-too-long


class TestCase(unittest.TestCase):

  def assert_html(self, actual, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = actual.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual.strip(), expected)


class SharedPartTest(TestCase):

  def test_styles(self):
    self.assert_html(Html.Styles().content, '')
    styles = Html.Styles('h1 {color: red;}', 'h2 {color: blue;}')
    self.assert_html(
        styles.content,
        """
        <style>
        h1 {color: red;}
        h2 {color: blue;}
        </style>
        """
    )
    self.assertTrue(styles.add('h3 {color: green;}'))
    self.assertFalse(styles.add('h1 {color: red;}'))
    self.assertEqual(
        styles.parts,
        {
            'h1 {color: red;}': 2,
            'h2 {color: blue;}': 1,
            'h3 {color: green;}': 1,
        },
    )
    self.assert_html(
        styles.content,
        """
        <style>
        h1 {color: red;}
        h2 {color: blue;}
        h3 {color: green;}
        </style>
        """
    )
    styles2 = Html.Styles('h1 {color: red;}', 'h4 {color: yellow;}')
    styles.add(styles2)
    self.assert_html(
        styles.content,
        """
        <style>
        h1 {color: red;}
        h2 {color: blue;}
        h3 {color: green;}
        h4 {color: yellow;}
        </style>
        """
    )
    self.assertEqual(
        styles.parts,
        {
            'h1 {color: red;}': 3,
            'h2 {color: blue;}': 1,
            'h3 {color: green;}': 1,
            'h4 {color: yellow;}': 1,
        },
    )
    self.assertTrue(styles)
    self.assertFalse(Html.Styles())
    self.assertIn('h1 {color: red;}', styles)
    self.assertNotIn('h3 {color: red;}', styles)
    self.assertEqual(
        list(styles),
        [
            'h1 {color: red;}',
            'h2 {color: blue;}',
            'h3 {color: green;}',
            'h4 {color: yellow;}',
        ]
    )
    self.assertEqual(
        styles,
        Html.Styles(
            'h1 {color: red;}',
            'h2 {color: blue;}',
            'h3 {color: green;}',
            'h4 {color: yellow;}'
        )
    )
    self.assertNotEqual(
        styles,
        Html.Styles(
            'h1 {color: red;}',
        )
    )
    self.assertNotEqual(styles, Html.Scripts())
    self.assertEqual(
        repr(styles),
        """Styles(parts={'h1 {color: red;}': 3, 'h2 {color: blue;}': 1, 'h3 {color: green;}': 1, 'h4 {color: yellow;}': 1})"""
    )
    self.assert_html(
        str(styles),
        """
        <style>
        h1 {color: red;}
        h2 {color: blue;}
        h3 {color: green;}
        h4 {color: yellow;}
        </style>
        """
    )

  def test_style_files(self):
    self.assert_html(Html.StyleFiles().content, '')
    style_files = Html.StyleFiles('./a.css', 'https://x/y.css')
    self.assert_html(
        style_files.content,
        """
        <link rel="stylesheet" href="./a.css">
        <link rel="stylesheet" href="https://x/y.css">
        """
    )
    self.assertTrue(style_files.add('./b.css'))
    self.assertFalse(style_files.add('./a.css'))
    self.assertEqual(
        style_files.parts,
        {
            './a.css': 2,
            'https://x/y.css': 1,
            './b.css': 1,
        },
    )
    self.assert_html(
        style_files.content,
        """
        <link rel="stylesheet" href="./a.css">
        <link rel="stylesheet" href="https://x/y.css">
        <link rel="stylesheet" href="./b.css">
        """
    )
    style_files2 = Html.StyleFiles('./a.css', './c.css')
    style_files.add(style_files2)
    self.assert_html(
        style_files.content,
        """
        <link rel="stylesheet" href="./a.css">
        <link rel="stylesheet" href="https://x/y.css">
        <link rel="stylesheet" href="./b.css">
        <link rel="stylesheet" href="./c.css">
        """
    )
    self.assertEqual(
        style_files.parts,
        {
            './a.css': 3,
            'https://x/y.css': 1,
            './b.css': 1,
            './c.css': 1,
        },
    )
    self.assertTrue(style_files)
    self.assertFalse(Html.StyleFiles())
    self.assertIn('./a.css', style_files)
    self.assertNotIn('./d.css}', style_files)
    self.assertEqual(
        list(style_files),
        [
            './a.css',
            'https://x/y.css',
            './b.css',
            './c.css',
        ]
    )
    self.assertEqual(
        style_files,
        Html.StyleFiles(
            './a.css',
            'https://x/y.css',
            './b.css',
            './c.css',
        )
    )
    self.assertNotEqual(
        style_files,
        Html.StyleFiles(
            './a.css',
        )
    )
    self.assertNotEqual(style_files, Html.Scripts())
    self.assertEqual(
        repr(style_files),
        """StyleFiles(parts={'./a.css': 3, 'https://x/y.css': 1, './b.css': 1, './c.css': 1})"""
    )
    self.assert_html(
        str(style_files),
        """
        <link rel="stylesheet" href="./a.css">
        <link rel="stylesheet" href="https://x/y.css">
        <link rel="stylesheet" href="./b.css">
        <link rel="stylesheet" href="./c.css">
        """
    )

  def test_scripts(self):
    self.assert_html(Html.Scripts().content, '')
    scripts = Html.Scripts(
        'function myFun1(p1, p2) { return p1 * p2; }',
        'console.log("hi");'
    )
    self.assert_html(
        scripts.content,
        """
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        console.log("hi");
        </script>
        """
    )
    self.assertTrue(scripts.add('function myFun2(p1, p2) { return p1 * p2; }'))
    self.assertFalse(scripts.add('console.log("hi");'))
    self.assertEqual(
        scripts.parts,
        {
            'function myFun1(p1, p2) { return p1 * p2; }': 1,
            'function myFun2(p1, p2) { return p1 * p2; }': 1,
            'console.log("hi");': 2,
        },
    )
    self.assert_html(
        scripts.content,
        """
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        console.log("hi");
        function myFun2(p1, p2) { return p1 * p2; }
        </script>
        """
    )
    scripts2 = Html.Scripts(
        'function myFun3(p1, p2) { return p1 * p2; }',
        'console.log("hi");'
    )
    self.assertTrue(scripts.add(scripts2))
    self.assert_html(
        scripts.content,
        """
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        console.log("hi");
        function myFun2(p1, p2) { return p1 * p2; }
        function myFun3(p1, p2) { return p1 * p2; }
        </script>
        """
    )
    self.assertEqual(
        scripts.parts,
        {
            'function myFun1(p1, p2) { return p1 * p2; }': 1,
            'console.log("hi");': 3,
            'function myFun2(p1, p2) { return p1 * p2; }': 1,
            'function myFun3(p1, p2) { return p1 * p2; }': 1,
        },
    )
    self.assertTrue(scripts)
    self.assertFalse(Html.Scripts())
    self.assertIn('function myFun1(p1, p2) { return p1 * p2; }', scripts)
    self.assertNotIn('function myFun4(p1, p2) { return p1 * p2; }', scripts)
    self.assertEqual(
        list(scripts),
        [
            'function myFun1(p1, p2) { return p1 * p2; }',
            'console.log("hi");',
            'function myFun2(p1, p2) { return p1 * p2; }',
            'function myFun3(p1, p2) { return p1 * p2; }',
        ]
    )
    self.assertEqual(
        scripts,
        Html.Scripts(
            'function myFun1(p1, p2) { return p1 * p2; }',
            'console.log("hi");',
            'function myFun2(p1, p2) { return p1 * p2; }',
            'function myFun3(p1, p2) { return p1 * p2; }',
        )
    )
    self.assertNotEqual(
        scripts,
        Html.Scripts(
            'function myFun1(p1, p2) { return p1 * p2; }'
        )
    )
    self.assertNotEqual(scripts, Html.ScriptFiles())
    self.assertEqual(
        repr(scripts),
        """Scripts(parts={'function myFun1(p1, p2) { return p1 * p2; }': 1, 'console.log("hi");': 3, 'function myFun2(p1, p2) { return p1 * p2; }': 1, 'function myFun3(p1, p2) { return p1 * p2; }': 1})"""
    )
    self.assert_html(
        str(scripts),
        """
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        console.log("hi");
        function myFun2(p1, p2) { return p1 * p2; }
        function myFun3(p1, p2) { return p1 * p2; }
        </script>
        """
    )

  def test_script_files(self):
    self.assert_html(Html.ScriptFiles().content, '')
    script_files = Html.ScriptFiles('./a.js', 'https://x/y.js')
    self.assert_html(
        script_files.content,
        """
        <script src="./a.js"></script>
        <script src="https://x/y.js"></script>
        """
    )
    self.assertTrue(script_files.add('./b.js'))
    self.assertFalse(script_files.add('./a.js'))
    self.assertEqual(
        script_files.parts,
        {
            './a.js': 2,
            'https://x/y.js': 1,
            './b.js': 1,
        },
    )
    self.assert_html(
        script_files.content,
        """
        <script src="./a.js"></script>
        <script src="https://x/y.js"></script>
        <script src="./b.js"></script>
        """
    )
    script_files2 = Html.ScriptFiles('./a.js', './c.js')
    script_files.add(script_files2)
    self.assert_html(
        script_files.content,
        """
        <script src="./a.js"></script>
        <script src="https://x/y.js"></script>
        <script src="./b.js"></script>
        <script src="./c.js"></script>
        """
    )
    self.assertEqual(
        script_files.parts,
        {
            './a.js': 3,
            'https://x/y.js': 1,
            './b.js': 1,
            './c.js': 1,
        },
    )
    self.assertTrue(script_files)
    self.assertFalse(Html.StyleFiles())
    self.assertIn('./a.js', script_files)
    self.assertNotIn('./d.js}', script_files)
    self.assertEqual(
        list(script_files),
        [
            './a.js',
            'https://x/y.js',
            './b.js',
            './c.js',
        ]
    )
    self.assertEqual(
        script_files,
        Html.ScriptFiles(
            './a.js',
            'https://x/y.js',
            './b.js',
            './c.js',
        )
    )
    self.assertNotEqual(
        script_files,
        Html.StyleFiles(
            './a.js',
        )
    )
    self.assertNotEqual(script_files, Html.Scripts())
    self.assertEqual(
        repr(script_files),
        """ScriptFiles(parts={'./a.js': 3, 'https://x/y.js': 1, './b.js': 1, './c.js': 1})"""
    )
    self.assert_html(
        str(script_files),
        """
        <script src="./a.js"></script>
        <script src="https://x/y.js"></script>
        <script src="./b.js"></script>
        <script src="./c.js"></script>
        """
    )


class HtmlTest(TestCase):

  class Foo(base.HtmlConvertible):
    def to_html(self, **kwargs):
      return base.Html('<h1>foo</h1>')

  def test_content_init(self):
    html = Html()
    self.assertEqual(html, Html())

    html = Html('abc')
    self.assertEqual(html, Html('abc'))

    html = Html(None)
    self.assertEqual(html, Html())

    html = Html(lambda: 'abc')
    self.assertEqual(html, Html('abc'))

    html = Html(Html('abc'))
    self.assertEqual(html, Html('abc'))

    html = Html(
        'abc',
        lambda: 'def',
        None,
        Html('ghi')
    )
    self.assertEqual(html, Html('abcdefghi'))

    html = Html(HtmlTest.Foo())
    self.assertEqual(html, Html('<h1>foo</h1>'))

  def test_basics(self):
    html = Html(
        '<h1>foo</h1>',
        styles=['h1 {color: red;}'],
        scripts=['function myFun1(p1, p2) { return p1 * p2; }']
    )
    self.assert_html(html.content, '<h1>foo</h1>')
    self.assert_html(
        html.style_section,
        """
        <style>
        h1 {color: red;}
        </style>
        """,
    )
    self.assert_html(
        html.script_section,
        """
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        """,
    )

    # Adding the same style.
    html.styles.add('h1 {color: red;}')
    self.assertEqual(list(html.styles), ['h1 {color: red;}'])

    html.add_style_file('./style1.css')
    self.assertEqual(list(html.style_files), ['./style1.css'])
    self.assert_html(
        html.style_section,
        """
        <link rel="stylesheet" href="./style1.css">
        <style>
        h1 {color: red;}
        </style>
        """,
    )

    html.scripts.add('function myFun1(p1, p2) { return p1 * p2; }')
    self.assertEqual(
        list(html.scripts),
        ['function myFun1(p1, p2) { return p1 * p2; }']
    )
    html.add_script_file('./script1.js')
    self.assertEqual(list(html.script_files), ['./script1.js'])
    self.assert_html(
        html.script_section,
        """
        <script src="./script1.js"></script>
        <script>
        function myFun1(p1, p2) { return p1 * p2; }
        </script>
        """,
    )

    html.write('<h2>bar</h2>')
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
    )
    self.assertEqual(
        repr(html),
        """Html(content='<h1>foo</h1><h2>bar</h2>', style_files=StyleFiles(parts={'./style1.css': 1}), styles=Styles(parts={'h1 {color: red;}': 2, 'h2 {color: blue;}': 1}), script_files=ScriptFiles(parts={'./script1.js': 1}), scripts=Scripts(parts={'function myFun1(p1, p2) { return p1 * p2; }': 2, 'function myFun2(p1, p2) { return p1 + p2; }': 1}))"""
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

  def test_write(self):
    html1 = Html()
    html1.scripts.add('function myFun1(p1, p2) { return p1 * p2; }')
    html1.styles.add('div.a { color: red; }')
    html1.styles.add('div.b { color: red; }')

    html2 = Html()
    html2.styles.add('div.a { color: red; }')
    html2.styles.add('div.b { color: green; }')
    html2.styles.add('div.c { color: blue; }')
    html1.scripts.add('function myFun1(p1, p2) { return p1 * p2; }')
    html2.write('<div class="c">bar</div>')
    html2.write('\n<script>\nconsole.log("bar");\n</script>')

    html1.write(HtmlTest.Foo())
    html1.write('\n<script>\nconsole.log("foo");\n</script>\n')
    html1.write(html2)

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
        <h1>foo</h1>
        <script>
        console.log("foo");
        </script>
        <div class="c">bar</div>
        <script>
        console.log("bar");
        </script>
        </body>
        </html>
        """,
    )
    self.assert_html(
        html1.to_str(content_only=True),
        """
        <h1>foo</h1>
        <script>
        console.log("foo");
        </script>
        <div class="c">bar</div>
        <script>
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
    html3 = Html.from_value(HtmlTest.Foo())
    self.assertEqual(html3, Html('<h1>foo</h1>'))

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

    s6 = Html('<hr>')
    self.assertIs(s6 + None, s6)
    self.assertIs(s6, s6 + None)

    self.assertEqual(
        Html('<hr>') + (lambda: '<div>bar</div>'),
        Html('<hr><div>bar</div>')
    )
    self.assertEqual(
        (lambda: '<div>bar</div>') + Html('<hr>'),
        Html('<div>bar</div><hr>')
    )

  def test_escape(self):
    self.assertIsNone(Html.escape(None))
    self.assertEqual(Html.escape('foo'), 'foo')
    self.assertEqual(Html.escape('foo"bar'), 'foo&quot;bar')
    self.assertEqual(Html.escape(Html('foo"bar')), Html('foo&quot;bar'))
    self.assertEqual(Html.escape(HtmlTest.Foo()), Html('&lt;h1&gt;foo&lt;/h1&gt;'))
    self.assertEqual(Html.escape(lambda: 'foo"bar'), 'foo&quot;bar')
    self.assertEqual(Html.escape('"x=y"', javascript_str=True), '\\"x=y\\"')
    self.assertEqual(Html.escape('x\n"', javascript_str=True), 'x\\n\\"')
    self.assertEqual(
        Html.escape(HtmlTest.Foo(), javascript_str=True), Html('<h1>foo</h1>')
    )

  def test_concate(self):
    self.assertIsNone(Html.concate(None))
    self.assertIsNone(Html.concate([None, [None, [None, None]]]))
    self.assertEqual(Html.concate('a'), 'a')
    self.assertEqual(Html.concate(['a']), 'a')
    self.assertEqual(Html.concate(['a', 'a']), 'a')
    self.assertEqual(Html.concate(['a', 'a'], dedup=False), 'a a')
    self.assertEqual(Html.concate(['a', None, 'b']), 'a b')
    self.assertEqual(
        Html.concate(['a', 'b', [None, 'c', [None, 'd']]]), 'a b c d')

  def test_element(self):
    # Empty element.
    self.assertEqual(Html.element('div').content, '<div></div>')
    # CSS class as list.
    self.assertEqual(
        Html.element('div', css_classes=['a', 'b', None]).content,
        '<div class="a b"></div>',
    )
    self.assertEqual(
        Html.element('div', css_classes=[None, None]).content,
        '<div></div>',
    )
    # Style as string.
    self.assertEqual(
        Html.element('div', style='color:red;').content,
        '<div style="color:red;"></div>',
    )
    # Style as dictionary.
    self.assertEqual(
        Html.element(
            'div',
            styles=dict(
                color='red', background_color='blue', width=None,
            )
        ).content,
        '<div style="color:red;background-color:blue;"></div>',
    )
    self.assertEqual(
        Html.element(
            'div',
            styles=dict(
                color=None,
            )
        ).content,
        '<div></div>',
    )
    # Properties as kwargs
    self.assertEqual(
        Html.element(
            'details',
            options='open',
            css_classes='my_class',
            id='my_id',
            custom_property='1'
        ).content,
        (
            '<details open class="my_class" id="my_id" custom-property="1">'
            '</details>'
        )
    )
    self.assertEqual(
        Html.element(
            'details',
            options=[None],
            css_classes='my_class',
            id='my_id',
            custom_property='1'
        ).content,
        (
            '<details class="my_class" id="my_id" custom-property="1">'
            '</details>'
        )
    )
    # Child.
    self.assertEqual(
        Html.element(
            'div',
            css_classes='my_class',
            inner_html='<h1>foo</h1>'
        ).content,
        '<div class="my_class"><h1>foo</h1></div>',
    )
    # Children.
    self.assert_html(
        str(Html.element(
            'div',
            [
                '<hr>',
                lambda: '<div>bar</div>',
                None,
                Html.element(
                    'div',
                    css_classes='my_class',
                ).add_style('div.my_class { color: red; }')
            ]
        )),
        """
        <html>
        <head>
        <style>
        div.my_class { color: red; }
        </style>
        </head>
        <body>
        <div><hr><div>bar</div><div class="my_class"></div></div>
        </body>
        </html>
        """
    )


# pylint: enable=line-too-long


if __name__ == '__main__':
  unittest.main()
