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

import inspect
import unittest

from pyglove.core.views.html import base
from pyglove.core.views.html import tree_view

Html = base.Html


class TestCase(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._view = tree_view.HtmlTreeView()

  def assert_style(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.style_section.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual.strip(), expected)

  def assert_content(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual.strip(), expected)

  def assert_count(self, html, part, expected):
    content = html.content.strip()
    actual = content.count(part)
    if actual != expected:
      print(content)
    self.assertEqual(actual, expected)


class TooltipTest(TestCase):

  def test_style(self):
    self.assert_style(
        self._view.tooltip(
            'This <hello><world></world></hello>.',
            name='name',
        ),
        """
        <style>
        /* Tooltip styles. */
        span.tooltip {
          visibility: hidden;
          white-space: pre-wrap;
          font-weight: normal;
          background-color: #484848;
          color: #fff;
          padding: 10px;
          border-radius: 6px;
          position: absolute;
          z-index: 1;
        }
        </style>
        """
    )

  def test_render(self):
    self.assert_content(
        self._view.tooltip(
            'This <hello><world></world></hello>.',
            name='name',
        ),
        """
        <span class="tooltip">&#x27;This &lt;hello&gt;&lt;world&gt;&lt;/world&gt;&lt;/hello&gt;.&#x27;</span>
        """
    )
    self.assert_content(
        self._view.tooltip(
            1,
            content=Html.element('div', ['hello']),
            name='name',
        ),
        """
        <span class="tooltip"><div>hello</div></span>
        """
    )


class SummaryTest(TestCase):

  def test_style(self):
    self.assert_style(
        self._view.summary(1, enable_summary=True),
        """
        <style>
        /* Tooltip styles. */
        span.tooltip {
          visibility: hidden;
          white-space: pre-wrap;
          font-weight: normal;
          background-color: #484848;
          color: #fff;
          padding: 10px;
          border-radius: 6px;
          position: absolute;
          z-index: 1;
        }
        /* Summary styles. */
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
        .summary_title:hover + span.tooltip {
          visibility: visible;
        }
        /* Type-specific styles. */
        .pyglove.str .summary_title {
          color: darkred;
          font-style: italic;
        }
        </style>
        """
    )

  def test_enable_summary(self):
    self.assertIsNone(
        self._view.summary(1, enable_summary=None),
    )
    self.assertIsNone(
        self._view.summary('foo', enable_summary=False),
    )
    self.assertIsNone(
        self._view.summary('foo', enable_summary=None),
    )
    self.assert_content(
        self._view.summary('foo', enable_summary=True),
        """
        <summary><div class="summary_title">&#x27;foo&#x27;</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            'foo', enable_summary=None, max_summary_len_for_str=1
        ),
        """
        <summary><div class="summary_title">&#x27;f...&#x27;</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )

  def test_max_summary_len_for_str(self):
    self.assert_content(
        self._view.summary(
            'abcdefg',
            max_summary_len_for_str=5
        ),
        """
        <summary><div class="summary_title">&#x27;abcde...&#x27;</div><span class="tooltip">&#x27;abcdefg&#x27;</span></summary>
        """
    )

  def test_render_with_name(self):
    self.assert_content(
        self._view.summary('foo', name='x'),
        """
        <summary><div class="summary_name">x</div><div class="summary_title">&#x27;foo&#x27;</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )

  def test_enable_summary_tooltip(self):
    self.assert_content(
        self._view.summary(
            'foo', enable_summary=True, enable_summary_tooltip=True
        ),
        """
        <summary><div class="summary_title">&#x27;foo&#x27;</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            'foo', enable_summary=True, enable_summary_tooltip=False
        ),
        """
        <summary><div class="summary_title">&#x27;foo&#x27;</div></summary>
        """
    )


class ContentTest(TestCase):

  def test_style(self):
    self.assert_style(
        self._view.content(dict(x=1)),
        """
        <style>
        /* Tooltip styles. */
        span.tooltip {
          visibility: hidden;
          white-space: pre-wrap;
          font-weight: normal;
          background-color: #484848;
          color: #fff;
          padding: 10px;
          border-radius: 6px;
          position: absolute;
          z-index: 1;
        }
        /* Object key styles. */
        .object_key {
          margin-right: 0.25em;
        }
        .object_key:hover + .tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        .complex_value .object_key{
          color: gray;
          border: 1px solid lightgray;
          background-color: ButtonFace;
          border-radius: 0.2em;
          padding: 0.3em;
        }
        .complex_value.list .object_key{
          border: 0;
          color: lightgray;
          background-color: transparent;
          border-radius: 0;
          padding: 0;
        }
        .complex_value.list .object_key::before{
          content: '[';
        }
        .complex_value.list .object_key::after{
          content: ']';
        }
        /* Simple value styles. */
        .simple_value {
          color: blue;
          display: inline-block;
          white-space: pre-wrap;
          padding: 0.2em;
          margin-top: 0.15em;
        }
        .simple_value.str {
          color: darkred;
          font-style: italic;
        }
        .simple_value.int, .simple_value.float {
          color: darkblue;
        }
        /* Complex value styles. */
        span.empty_container::before {
            content: '(empty)';
            font-style: italic;
            margin-left: 0.5em;
            color: #aaa;
        }
        </style>
        """
    )

  def test_simple_types(self):
    self.assert_content(
        self._view.content(1),
        """
        <span class="simple_value int">1</span>
        """
    )
    self.assert_content(
        self._view.content(1.5),
        """
        <span class="simple_value float">1.5</span>
        """
    )
    self.assert_content(
        self._view.content(True),
        """
        <span class="simple_value bool">True</span>
        """
    )
    self.assert_content(
        self._view.content(None),
        """
        <span class="simple_value none-type">None</span>
        """
    )
    self.assert_content(
        self._view.content('<foo>'),
        """ 
        <span class="simple_value str">&#x27;&lt;foo&gt;&#x27;</span>
        """
    )
    self.assert_content(
        self._view.content(
            '<hello><world> \nto everyone.', max_summary_len_for_str=10
        ),
        """ 
        <span class="simple_value str">&lt;hello&gt;&lt;world&gt; 
        to everyone.</span>
        """
    )

  def test_list(self):
    self.assert_content(
        self._view.content([1, 2, 'abc']),
        """
        <div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[1]</span></td><td><span class="simple_value int">2</span></td></tr><tr><td><span class="object_key">2</span><span class="tooltip">[2]</span></td><td><span class="simple_value str">&#x27;abc&#x27;</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content([]),
        """
        <div class="complex_value list"><span class="empty_container"></span></div>
        """
    )

  def test_tuple(self):
    self.assert_content(
        self._view.content((1, True)),
        """
        <div class="complex_value tuple"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[1]</span></td><td><span class="simple_value bool">True</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content(()),
        """
        <div class="complex_value tuple"><span class="empty_container"></span></div>
        """
    )

  def test_dict(self):
    self.assert_content(
        self._view.content(dict(x=1, y='foo')),
        """
        <div class="complex_value dict"><table><tr><td><span class="object_key">x</span><span class="tooltip">x</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">y</span><span class="tooltip">y</span></td><td><span class="simple_value str">&#x27;foo&#x27;</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content({}),
        """
        <div class="complex_value dict"><span class="empty_container"></span></div>
        """
    )

  def test_custom_types(self):

    class Foo:
      def __str__(self):
        return '<Foo></Foo>'

    self.assert_content(
        self._view.content(Foo()),
        """
        <span class="simple_value foo">&lt;Foo&gt;&lt;/Foo&gt;</span>
        """
    )

  def test_nesting(self):

    class Foo:
      def __str__(self):
        return '<Foo></Foo>'

    self.assert_content(
        self._view.content(
            [
                dict(
                    x=[(1, 2)],
                    y=['b', Foo()],
                ),
                1,
                [1, dict(xx=1, yy='a')]
            ]
        ),
        """
        <div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex_value dict"><table><tr><td><span class="object_key">x</span><span class="tooltip">[0].x</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary_title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex_value tuple"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple_value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">y</span><span class="tooltip">[0].y</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple_value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary_title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple_value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[1]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex_value dict"><table><tr><td><span class="object_key">xx</span><span class="tooltip">[2][1].xx</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">yy</span><span class="tooltip">[2][1].yy</span></td><td><span class="simple_value str">&#x27;a&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )

  def test_enable_disable_tooltip(self):

    class Foo:
      def __str__(self):
        return '<Foo></Foo>'

    value = [
        dict(
            x=[(1, 2)],
            y=['b', Foo()],
        ),
        1,
        [1, dict(xx=1, yy='a')]
    ]
    self.assert_content(
        self._view.content(value, enable_summary_tooltip=False),
        """
        <div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">x</span><span class="tooltip">[0].x</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary_title">tuple(...)</div></summary><div class="complex_value tuple"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple_value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">y</span><span class="tooltip">[0].y</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple_value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary_title">Foo(...)</div></summary><span class="simple_value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[1]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">xx</span><span class="tooltip">[2][1].xx</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">yy</span><span class="tooltip">[2][1].yy</span></td><td><span class="simple_value str">&#x27;a&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content(value, enable_key_tooltip=False),
        """
        <div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex_value dict"><table><tr><td><span class="object_key">x</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><details class="pyglove tuple"><summary><div class="summary_title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex_value tuple"><table><tr><td><span class="object_key">0</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span></td><td><span class="simple_value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">y</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><span class="simple_value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object_key">1</span></td><td><details class="pyglove foo"><summary><div class="summary_title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple_value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">1</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">2</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex_value dict"><table><tr><td><span class="object_key">xx</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">yy</span></td><td><span class="simple_value str">&#x27;a&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content(
            value, enable_summary_tooltip=False, enable_key_tooltip=False
        ),
        """
        <div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">x</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><details class="pyglove tuple"><summary><div class="summary_title">tuple(...)</div></summary><div class="complex_value tuple"><table><tr><td><span class="object_key">0</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span></td><td><span class="simple_value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">y</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><span class="simple_value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object_key">1</span></td><td><details class="pyglove foo"><summary><div class="summary_title">Foo(...)</div></summary><span class="simple_value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">1</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">2</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">xx</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">yy</span></td><td><span class="simple_value str">&#x27;a&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )

  def test_include_exclude_immediate_child_keys(self):
    self.assert_count(
        self._view.content(
            dict(a='x', b=dict(e='y'), c='z', d='w', e='v', f='u'),
            include_keys=['a', 'b', 'c', 'd'],
            exclude_keys=['c', 'e'],
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        'object_key',
        3 + 1  # 'a', 'b', 'd' at the first level and 'e' at the second level.
    )

  def test_special_keys(self):
    self.assert_content(
        self._view.content(
            dict(a='x', b=dict(e='y'), c='z', d='w', e='v', f='u'),
            special_keys=['c', 'b'],
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <div class="complex_value dict"><details class="pyglove str special_value"><summary><div class="summary_name">c</div><div class="summary_title">&#x27;z&#x27;</div></summary><span class="simple_value str">&#x27;z&#x27;</span></details><details class="pyglove dict special_value"><summary><div class="summary_name">b</div><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">e</span></td><td><span class="simple_value str">&#x27;y&#x27;</span></td></tr></table></div></details><table><tr><td><span class="object_key">a</span></td><td><span class="simple_value str">&#x27;x&#x27;</span></td></tr><tr><td><span class="object_key">d</span></td><td><span class="simple_value str">&#x27;w&#x27;</span></td></tr><tr><td><span class="object_key">e</span></td><td><span class="simple_value str">&#x27;v&#x27;</span></td></tr><tr><td><span class="object_key">f</span></td><td><span class="simple_value str">&#x27;u&#x27;</span></td></tr></table></div>
        """
    )
    self.assert_count(
        self._view.content(
            dict(a='x', b=dict(e='y'), c='z', d='w', e='v', f='u'),
            special_keys=['b', 'd'],
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        'special_value',
        2
    )

  def test_filter(self):
    self.assert_content(
        self._view.content(
            dict(a='x', b=dict(e='y'), c='z', d='w', e='v', f='u'),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            filter=(lambda k, v, p: len(k) > 1 or isinstance(v, dict))
        ),
        """
        <div class="complex_value dict"><table><tr><td><span class="object_key">b</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">e</span></td><td><span class="simple_value str">&#x27;y&#x27;</span></td></tr></table></div></details></td></tr></table></div>
        """
    )

  def test_highlight_lowlight(self):
    self.assert_content(
        self._view.content(
            dict(a=1, b=dict(e='y'), c=2, d='w', e=3, f='u'),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            highlight_keys=(lambda k, v, p: isinstance(v, int)),
            lowlight_keys=(lambda k, v, p: isinstance(v, str)),
        ),
        """
        <div class="complex_value dict"><table><tr><td><span class="object_key">a</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">b</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key">e</span></td><td><span class="simple_value str">&#x27;y&#x27;</span></td></tr></table></div></details></td></tr><tr><td><span class="object_key">c</span></td><td><span class="simple_value int">2</span></td></tr><tr><td><span class="object_key">d</span></td><td><span class="simple_value str">&#x27;w&#x27;</span></td></tr><tr><td><span class="object_key">e</span></td><td><span class="simple_value int">3</span></td></tr><tr><td><span class="object_key">f</span></td><td><span class="simple_value str">&#x27;u&#x27;</span></td></tr></table></div>
        """
    )

  def test_collapse_level(self):
    x = dict(
        a=dict(
            b=dict(
                c=dict(
                    d=dict()
                )
            )
        ),
        aa=dict(
            bb=dict(
                cc=dict(
                    dd=dict()
                )
            )
        )
    )
    self.assert_count(
        self._view.content(x, collapse_level=0),
        'open',
        0
    )
    # There is no summary section, so there is no root.
    self.assert_count(
        self._view.content(x, collapse_level=1),
        'open',
        0
    )
    self.assert_count(
        self._view.content(x, collapse_level=2),
        'open',
        2
    )
    self.assert_count(
        self._view.content(x, collapse_level=3),
        'open',
        4
    )
    self.assert_count(
        self._view.content(x, collapse_level=4),
        'open',
        6
    )
    self.assert_count(
        self._view.content(x, collapse_level=None),
        'open',
        8
    )

  def test_uncollapse(self):
    x = dict(
        a=dict(
            b=dict(
                c=dict(
                    d=dict()
                )
            )
        ),
        aa=dict(
            bb=dict(
                cc=dict(
                    dd=dict()
                )
            )
        )
    )
    self.assert_count(
        self._view.content(x, uncollapse=['a.b.c']),
        'open',
        3
    )
    self.assert_count(
        self._view.content(x, uncollapse=['aa.bb.cc.dd', 'a.b']),
        'open',
        6
    )
    # Use both collapse_level and uncollapse.
    self.assert_count(
        self._view.content(x, collapse_level=2, uncollapse=['aa.bb.cc']),
        'open',
        4
    )
    self.assert_count(
        self._view.content(
            x, collapse_level=0, uncollapse=lambda k, v, p: len(k) < 3
        ),
        'open',
        4
    )


class RenderTest(TestCase):

  def test_render(self):
    class Foo:
      def __str__(self):
        return '<Foo></Foo>'

    self.assert_content(
        self._view.render(
            [
                dict(
                    x=[(1, 2)],
                    y=['b', Foo()],
                ),
                1,
                [1, dict(xx=1, yy='a')]
            ],
        ),
        """
        <details open class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[
          {
            &#x27;x&#x27;: [(1, 2)],
            &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
          },
          1,
          [1, {
              &#x27;xx&#x27;: 1,
              &#x27;yy&#x27;: &#x27;a&#x27;
            }]
        ]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex_value dict"><table><tr><td><span class="object_key">x</span><span class="tooltip">[0].x</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary_title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex_value tuple"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple_value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">y</span><span class="tooltip">[0].y</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple_value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary_title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple_value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[1]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary_title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex_value list"><table><tr><td><span class="object_key">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary_title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex_value dict"><table><tr><td><span class="object_key">xx</span><span class="tooltip">[2][1].xx</span></td><td><span class="simple_value int">1</span></td></tr><tr><td><span class="object_key">yy</span><span class="tooltip">[2][1].yy</span></td><td><span class="simple_value str">&#x27;a&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details>
        """
    )


class ExtensionTest(TestCase):

  def test_no_overrides(self):

    class Foo(tree_view.HtmlTreeView.Extension):

      def __str__(self):
        return 'Foo()'

    self.assert_content(
        self._view.summary(
            Foo(),
            enable_summary=True,
            max_summary_len_for_str=10,
            enable_tooltip=True,
        ),
        """
        <summary><div class="summary_title">Foo(...)</div><span class="tooltip">Foo()</span></summary>
        """
    )
    self.assert_content(
        self._view.content(
            Foo(),
        ),
        """
        <span class="simple_value foo">Foo()</span>
        """
    )

  def test_overrides(self):

    class Foo(tree_view.HtmlTreeView.Extension):

      def _html_tree_view_summary(
          self,
          *,
          view: tree_view.HtmlTreeView,
          **kwargs
      ):
        kwargs.pop('title', None)
        return view.summary(
            self,
            title='MyFoo',
            title_class='t_MyFoo',
            **kwargs
        )

      def _html_tree_view_content(
          self,
          **kwargs
      ):
        del kwargs
        return 'Content of MyFoo'

      def _html_tree_view_tooltip(
          self,
          *,
          view: tree_view.HtmlTreeView,
          **kwargs
      ):
        del kwargs
        return view.tooltip(
            value=self,
            content='Tooltip <MyFoo>'
        )

    self.assert_content(
        self._view.render(
            Foo(),
        ),
        """
        <details open class="pyglove foo"><summary><div class="summary_title">MyFoo</div><span class="tooltip">Tooltip <MyFoo></span></summary>Content of MyFoo</details>
        """
    )
    self.assert_content(
        self._view.render(
            Foo(),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove foo"><summary><div class="summary_title">MyFoo</div></summary>Content of MyFoo</details>
        """
    )
    self.assert_content(
        self._view.render(
            Foo(),
            enable_summary=False,
        ),
        """
        Content of MyFoo
        """
    )


if __name__ == '__main__':
  unittest.main()
