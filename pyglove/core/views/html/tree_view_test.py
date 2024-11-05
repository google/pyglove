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
from typing import Any
import unittest

from pyglove.core.views.html import base
from pyglove.core.views.html import tree_view

Html = base.Html
KeyPath = tree_view.KeyPath
KeyPathSet = tree_view.KeyPathSet


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
        span.tooltip:hover {
          visibility: visible;
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
            css_classes=['int'],
            name='name',
        ),
        """
        <span class="tooltip int"><div>hello</div></span>
        """
    )


class SummaryTest(TestCase):

  def test_style(self):
    self.assert_style(
        self._view.summary(
            1, enable_summary=True,
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
        span.tooltip:hover {
          visibility: visible;
        }
        /* Summary styles. */
        details.pyglove summary {
          font-weight: bold;
          margin: -0.5em -0.5em 0;
          padding: 0.5em;
        }
        .summary-name {
          display: inline;
          padding: 3px 5px 3px 5px;
          margin: 0 5px;
          border-radius: 3px;
        }
        .summary-title {
          display: inline;
        }
        .summary-name + div.summary-title {
          display: inline;
          color: #aaa;
        }
        .summary-title:hover + span.tooltip {
          visibility: visible;
        }
        .summary-name:hover > span.tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        </style>
        """
    )

  def test_summary_title(self):
    self.assert_content(
        self._view.summary(
            1, enable_summary=True, enable_summary_tooltip=False
        ),
        """
        <summary><div class="summary-title">int</div></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            int, enable_summary=True, enable_summary_tooltip=False
        ),
        """
        <summary><div class="summary-title">type</div></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            [0, 1], enable_summary=True, enable_summary_tooltip=False
        ),
        """
        <summary><div class="summary-title">list(...)</div></summary>
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
        <summary><div class="summary-title">str</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            'foo', enable_summary=None, max_summary_len_for_str=1
        ),
        """
        <summary><div class="summary-title">str</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )
    self.assertIsNone(
        self._view.summary(
            'foo', enable_summary=None, enable_summary_for_str=False
        ),
    )

  def test_css_styles(self):
    self.assert_content(
        self._view.summary(
            'foo', name='x', css_classes=['bar'], enable_summary=True
        ),
        """
        <summary><div class="summary-name bar">x<span class="tooltip bar"></span></div><div class="summary-title bar">str</div><span class="tooltip bar">&#x27;foo&#x27;</span></summary>
        """
    )

  def test_max_summary_len_for_str(self):
    self.assert_content(
        self._view.summary(
            'abcdefg',
            max_summary_len_for_str=5
        ),
        """
        <summary><div class="summary-title">str</div><span class="tooltip">&#x27;abcdefg&#x27;</span></summary>
        """
    )

  def test_name(self):
    self.assert_content(
        self._view.summary('foo', name='x'),
        """
        <summary><div class="summary-name">x<span class="tooltip"></span></div><div class="summary-title">str</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )

  def test_enable_key_tooltip(self):
    self.assert_content(
        self._view.summary(
            'foo',
            name='x',
            root_path=KeyPath.parse('y.a.x'),
            enable_summary_tooltip=False,
            enable_key_tooltip=True
        ),
        """
        <summary><div class="summary-name">x<span class="tooltip">y.a.x</span></div><div class="summary-title">str</div></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            'foo',
            name='x',
            root_path=KeyPath.parse('y.a.x'),
            enable_summary_tooltip=False,
            enable_key_tooltip=False
        ),
        """
        <summary><div class="summary-name">x</div><div class="summary-title">str</div></summary>
        """
    )

  def test_summary_color(self):
    self.assert_content(
        self._view.summary(
            'foo',
            name='x',
            summary_color=('white', 'red'),
            enable_summary_tooltip=False
        ),
        """
        <summary><div class="summary-name" style="color:white;background-color:red;">x<span class="tooltip"></span></div><div class="summary-title">str</div></summary>
        """
    )

  def test_enable_summary_tooltip(self):
    self.assert_content(
        self._view.summary(
            'foo', enable_summary=True, enable_summary_tooltip=True
        ),
        """
        <summary><div class="summary-title">str</div><span class="tooltip">&#x27;foo&#x27;</span></summary>
        """
    )
    self.assert_content(
        self._view.summary(
            'foo', enable_summary=True, enable_summary_tooltip=False
        ),
        """
        <summary><div class="summary-title">str</div></summary>
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
        span.tooltip:hover {
          visibility: visible;
        }
        /* Summary styles. */
        details.pyglove summary {
          font-weight: bold;
          margin: -0.5em -0.5em 0;
          padding: 0.5em;
        }
        .summary-name {
          display: inline;
          padding: 3px 5px 3px 5px;
          margin: 0 5px;
          border-radius: 3px;
        }
        .summary-title {
          display: inline;
        }
        .summary-name + div.summary-title {
          display: inline;
          color: #aaa;
        }
        .summary-title:hover + span.tooltip {
          visibility: visible;
        }
        .summary-name:hover > span.tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        /* Simple value styles. */
        .simple-value {
          color: blue;
          display: inline-block;
          white-space: pre-wrap;
          padding: 0.2em;
          margin-top: 0.15em;
        }
        .simple-value.str {
          color: darkred;
          font-style: italic;
        }
        .simple-value.int, .simple-value.float {
          color: darkblue;
        }
        /* Value details styles. */
        details.pyglove {
          border: 1px solid #aaa;
          border-radius: 4px;
          padding: 0.5em 0.5em 0;
          margin: 0.25em 0;
        }
        details.pyglove[open] {
          padding: 0.5em 0.5em 0.5em;
        }
        .highlight {
          background-color: Mark;
        }
        .lowlight {
          opacity: 0.2;
        }
        /* Complex value styles. */
        span.empty-container::before {
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
        <span class="simple-value int">1</span>
        """
    )
    self.assert_content(
        self._view.content(1.5),
        """
        <span class="simple-value float">1.5</span>
        """
    )
    self.assert_content(
        self._view.content(True),
        """
        <span class="simple-value bool">True</span>
        """
    )
    self.assert_content(
        self._view.content(None),
        """
        <span class="simple-value none-type">None</span>
        """
    )
    self.assert_content(
        self._view.content('<foo>'),
        """
        <span class="simple-value str">&#x27;&lt;foo&gt;&#x27;</span>
        """
    )
    self.assert_content(
        self._view.content(
            '<hello><world> \nto everyone.', max_summary_len_for_str=10
        ),
        """
        <span class="simple-value str">&lt;hello&gt;&lt;world&gt; 
        to everyone.</span>
        """
    )

  def test_list(self):
    self.assert_content(
        self._view.content([1, 2, 'abc']),
        """
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><span class="simple-value str">&#x27;abc&#x27;</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content([1, 2, 'abc'], key_style='label'),
        """
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><span class="simple-value str">&#x27;abc&#x27;</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content([]),
        """
        <div class="complex-value list"><span class="empty-container"></span></div>
        """
    )

  def test_tuple(self):
    self.assert_content(
        self._view.content((1, True)),
        """
        <div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value bool">True</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content((1, True), key_style='label'),
        """
        <div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value bool">True</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content(()),
        """
        <div class="complex-value tuple"><span class="empty-container"></span></div>
        """
    )

  def test_dict(self):
    self.assert_content(
        self._view.content(dict(x=1, y='foo')),
        """
        <div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">x<span class="tooltip">x</span></div><div class="summary-title">int</div><span class="tooltip">1</span></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">y<span class="tooltip">y</span></div><div class="summary-title">str</div><span class="tooltip">&#x27;foo&#x27;</span></summary><span class="simple-value str">&#x27;foo&#x27;</span></details></div>
        """
    )
    self.assert_content(
        self._view.content(dict(x=1, y='foo'), key_style='label'),
        """
        <div class="complex-value dict"><table><tr><td><span class="object-key str">x</span><span class="tooltip">x</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">y</span></td><td><span class="simple-value str">&#x27;foo&#x27;</span></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content({}),
        """
        <div class="complex-value dict"><span class="empty-container"></span></div>
        """
    )

  def test_custom_types(self):

    class Foo:
      def __str__(self):
        return '<Foo></Foo>'

    self.assert_content(
        self._view.content(Foo()),
        """
        <span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span>
        """
    )

  def test_custom_key_value_render(self):
    def render_key(view, key, **kwargs):
      del view, kwargs
      return Html.element('span', [f'custom {key}'])

    def render_value(view, *, value, **kwargs):
      del view, kwargs
      return Html.element('span', [f'custom {value}'])

    self.assert_content(
        self._view.complex_value(
            dict(x=1, y='foo'),
            parent=None,
            root_path=KeyPath(),
            render_key_fn=render_key,
            render_value_fn=render_value,
        ),
        """
        <div class="complex-value none-type"><span>custom 1</span><span>custom foo</span></div>
        """
    )
    self.assert_content(
        self._view.complex_value(
            dict(x=1, y='foo'),
            parent=None,
            key_style='label',
            root_path=KeyPath(),
            render_key_fn=render_key,
            render_value_fn=render_value,
        ),
        """
        <div class="complex-value none-type"><table><tr><td><span>custom x</span></td><td><span>custom 1</span></td></tr><tr><td><span>custom y</span></td><td><span>custom foo</span></td></tr></table></div>
        """
    )

  def test_key_style_with_nesting(self):

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
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">x<span class="tooltip">[0].x</span></div><div class="summary-title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details><details class="pyglove list"><summary><div class="summary-name">y<span class="tooltip">[0].y</span></div><div class="summary-title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></div></details></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">xx<span class="tooltip">[2][1].xx</span></div><div class="summary-title">int</div><span class="tooltip">1</span></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">yy<span class="tooltip">[2][1].yy</span></div><div class="summary-title">str</div><span class="tooltip">&#x27;a&#x27;</span></summary><span class="simple-value str">&#x27;a&#x27;</span></details></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )

    self.assert_content(
        self._view.content(
            [
                dict(
                    x=[(1, 2)],
                    y=['b', Foo()],
                ),
                1,
                [1, dict(xx=1, yy='a')]
            ],
            key_style='label',
        ),
        """
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">x</span><span class="tooltip">[0].x</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">[0].y</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">xx</span><span class="tooltip">[2][1].xx</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">yy</span><span class="tooltip">[2][1].yy</span></td><td><span class="simple-value str">&#x27;a&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )

    def _key_style(k, v, p):
      del v, p
      if k and k.key in ('x', 'xx'):
        return 'label'
      return 'summary'

    self.assert_content(
        self._view.content(
            [
                dict(
                    x=[(1, 2)],
                    y=['b', Foo()],
                ),
                1,
                [1, dict(xx=1, yy='a')]
            ],
            key_style=_key_style,
        ),
        """
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">y<span class="tooltip">[0].y</span></div><div class="summary-title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details><table><tr><td><span class="object-key str">x</span><span class="tooltip">[0].x</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex-value dict"><details open class="pyglove str"><summary><div class="summary-name">yy<span class="tooltip">[2][1].yy</span></div><div class="summary-title">str</div><span class="tooltip">&#x27;a&#x27;</span></summary><span class="simple-value str">&#x27;a&#x27;</span></details><table><tr><td><span class="object-key str">xx</span><span class="tooltip">[2][1].xx</span></td><td><span class="simple-value int">1</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )

  def test_key_color(self):
    self.assert_content(
        self._view.content(
            dict(
                x=[(1, 2)],
                y=['b', dict(y=1)],
            ),
            key_color=('white', 'red'),
            enable_summary_tooltip=False,
        ),
        """
        <div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">x<span class="tooltip">x</span></div><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int" style="color:white;background-color:red;">0</span><span class="tooltip">x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int" style="color:white;background-color:red;">0</span><span class="tooltip">x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int" style="color:white;background-color:red;">1</span><span class="tooltip">x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details><details class="pyglove list"><summary><div class="summary-name">y<span class="tooltip">y</span></div><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int" style="color:white;background-color:red;">0</span><span class="tooltip">y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int" style="color:white;background-color:red;">1</span><span class="tooltip">y[1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">y<span class="tooltip">y[1].y</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details></div></details></td></tr></table></div></details></div>
        """
    )
    self.assert_content(
        self._view.content(
            dict(
                x=[(1, 2)],
                y=['b', dict(y=1)],
            ),
            key_style='label',
            enable_summary_tooltip=False,
            key_color=(
                lambda k, v, p: ('white', 'red')
                if k.key == 'y' else (None, None)
            ),
        ),
        """
        <div class="complex-value dict"><table><tr><td><span class="object-key str">x</span><span class="tooltip">x</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">y</span><span class="tooltip">y</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">y[1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">y</span><span class="tooltip">y[1].y</span></td><td><span class="simple-value int">1</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div>
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
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">x<span class="tooltip">[0].x</span></div><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details><details class="pyglove list"><summary><div class="summary-name">y<span class="tooltip">[0].y</span></div><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></div></details></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">xx<span class="tooltip">[2][1].xx</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">yy<span class="tooltip">[2][1].yy</span></div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;a&#x27;</span></details></div></details></td></tr></table></div></details></td></tr></table></div>
        """
    )
    self.assert_content(
        self._view.content(value, enable_key_tooltip=False),
        """
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">x</div><div class="summary-title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details><details class="pyglove list"><summary><div class="summary-name">y</div><div class="summary-title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></div></details></td></tr><tr><td><span class="object-key int">1</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">xx</div><div class="summary-title">int</div><span class="tooltip">1</span></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">yy</div><div class="summary-title">str</div><span class="tooltip">&#x27;a&#x27;</span></summary><span class="simple-value str">&#x27;a&#x27;</span></details></div></details></td></tr></table></div></details></td></tr></table></div>
      """
    )
    self.assert_content(
        self._view.content(
            value, enable_summary_tooltip=False, enable_key_tooltip=False
        ),
        """
        <div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">x</div><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details><details class="pyglove list"><summary><div class="summary-name">y</div><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></div></details></td></tr><tr><td><span class="object-key int">1</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">xx</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">yy</div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;a&#x27;</span></details></div></details></td></tr></table></div></details></td></tr></table></div>
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
            key_style='label',
        ),
        'object-key',
        3 + 1  # 'a', 'b', 'd' at the first level and 'e' at the second level.
    )

  def test_include_nodes_in_subtree(self):
    self.assert_count(
        self._view.content(
            dict(a='x', b=dict(e='y'), c='z', d='w', e='v', f='u'),
            include_keys=(lambda k, v, p: k.key == 'e' or isinstance(v, dict)),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            key_style='label',
        ),
        'object-key',
        3   # 'b', 'b.e', 'e'.
    )

  def test_exclude_nodes_in_subtree(self):
    self.assert_count(
        self._view.content(
            dict(a='x', b=dict(e='y'), c='z', d='w', e='v', f='u'),
            exclude_keys=(lambda k, v, p: k.key == 'e'),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            key_style='label',
        ),
        'object-key',
        5   # 'a', 'b', 'c', 'd', 'f'.
    )

  def test_highlight_lowlight(self):
    self.assert_content(
        self._view.content(
            dict(a=1, b=dict(e='y'), c=2, d='w', e=3, f='u'),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            highlight=(lambda k, v, p: isinstance(v, int)),
            lowlight=(lambda k, v, p: isinstance(v, str)),
        ),
        """
        <div class="complex-value dict"><div class="highlight"><details open class="pyglove int"><summary><div class="summary-name">a</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details></div><details class="pyglove dict"><summary><div class="summary-name">b</div><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><div class="lowlight"><details open class="pyglove str"><summary><div class="summary-name">e</div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;y&#x27;</span></details></div></div></details><div class="highlight"><details open class="pyglove int"><summary><div class="summary-name">c</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div><div class="lowlight"><details open class="pyglove str"><summary><div class="summary-name">d</div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;w&#x27;</span></details></div><div class="highlight"><details open class="pyglove int"><summary><div class="summary-name">e</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details></div><div class="lowlight"><details open class="pyglove str"><summary><div class="summary-name">f</div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;u&#x27;</span></details></div></div>
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

  def test_render_html_convertible(self):
    class Foo(base.HtmlConvertible):
      def to_html(self, **kwargs):
        return base.Html('<span>foo</span>')
    self.assert_content(base.to_html(Foo()), '<span>foo</span>')

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
        <details open class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[
          {
            &#x27;x&#x27;: [(1, 2)],
            &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
          },
          1,
          [1, {
              &#x27;xx&#x27;: 1,
              &#x27;yy&#x27;: &#x27;a&#x27;
            }]
        ]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: [(1, 2)],
          &#x27;y&#x27;: [&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]
        }</span></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">x<span class="tooltip">[0].x</span></div><div class="summary-title">list(...)</div><span class="tooltip">[(1, 2)]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0]</span></td><td><details class="pyglove tuple"><summary><div class="summary-title">tuple(...)</div><span class="tooltip">(1, 2)</span></summary><div class="complex-value tuple"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].x[0][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].x[0][1]</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details><details class="pyglove list"><summary><div class="summary-name">y<span class="tooltip">[0].y</span></div><div class="summary-title">list(...)</div><span class="tooltip">[&#x27;b&#x27;, &lt;Foo&gt;&lt;/Foo&gt;]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[0].y[0]</span></td><td><span class="simple-value str">&#x27;b&#x27;</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[0].y[1]</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div><span class="tooltip">&lt;Foo&gt;&lt;/Foo&gt;</span></summary><span class="simple-value foo">&lt;Foo&gt;&lt;/Foo&gt;</span></details></td></tr></table></div></details></div></details></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><details class="pyglove list"><summary><div class="summary-title">list(...)</div><span class="tooltip">[1, {
            &#x27;xx&#x27;: 1,
            &#x27;yy&#x27;: &#x27;a&#x27;
          }]</span></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">[2][0]</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[2][1]</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;xx&#x27;: 1,
          &#x27;yy&#x27;: &#x27;a&#x27;
        }</span></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">xx<span class="tooltip">[2][1].xx</span></div><div class="summary-title">int</div><span class="tooltip">1</span></summary><span class="simple-value int">1</span></details><details open class="pyglove str"><summary><div class="summary-name">yy<span class="tooltip">[2][1].yy</span></div><div class="summary-title">str</div><span class="tooltip">&#x27;a&#x27;</span></summary><span class="simple-value str">&#x27;a&#x27;</span></details></div></details></td></tr></table></div></details></td></tr></table></div></details>
        """
    )

  def test_debug(self):
    self.assertIn(
        inspect.cleandoc(
            """
            .debug-info-trigger {
              display: inline-flex;
              cursor: pointer;
              font-size: 0.6em;
              background-color: red;
              color: white;
              padding: 5px;
              border-radius: 3px;
              margin: 5px 0 5px 0;
            }
            .debug-info-trigger:hover + span.tooltip {
              visibility: visible;
            }
            """
        ),
        self._view.render(dict(x=dict(y=1)), debug=True).style_section,
    )
    self.assert_content(
        self._view.render(dict(x=1), debug=True),
        """
        <details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div><span class="tooltip">{
          &#x27;x&#x27;: 1
        }</span></summary><div><span class="debug-info-trigger">DEBUG</span><span class="tooltip debug-info">{
          &#x27;css_classes&#x27;: None,
          &#x27;collapse_level&#x27;: 1,
          &#x27;uncollapse&#x27;: KeyPathSet(),
          &#x27;extra_flags&#x27;: {},
          &#x27;child_config&#x27;: {},
          &#x27;key_style&#x27;: &#x27;summary&#x27;,
          &#x27;key_color&#x27;: None,
          &#x27;include_keys&#x27;: None,
          &#x27;exclude_keys&#x27;: None,
          &#x27;summary_color&#x27;: None,
          &#x27;enable_summary&#x27;: None,
          &#x27;enable_summary_for_str&#x27;: True,
          &#x27;max_summary_len_for_str&#x27;: 80,
          &#x27;enable_summary_tooltip&#x27;: True,
          &#x27;enable_key_tooltip&#x27;: True
        }</span></div><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">x<span class="tooltip">x</span></div><div class="summary-title">int</div><span class="tooltip">1</span></summary><div><span class="debug-info-trigger">DEBUG</span><span class="tooltip debug-info">{
          &#x27;css_classes&#x27;: None,
          &#x27;collapse_level&#x27;: 0,
          &#x27;uncollapse&#x27;: KeyPathSet(),
          &#x27;extra_flags&#x27;: {},
          &#x27;child_config&#x27;: {},
          &#x27;key_style&#x27;: &#x27;summary&#x27;,
          &#x27;key_color&#x27;: None,
          &#x27;include_keys&#x27;: None,
          &#x27;exclude_keys&#x27;: None,
          &#x27;summary_color&#x27;: None,
          &#x27;enable_summary&#x27;: None,
          &#x27;enable_summary_for_str&#x27;: True,
          &#x27;max_summary_len_for_str&#x27;: 80,
          &#x27;enable_summary_tooltip&#x27;: True,
          &#x27;enable_key_tooltip&#x27;: True
        }</span></div><span class="simple-value int">1</span></details></div></details>
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
            enable_summary_tooltip=True,
        ),
        """
        <summary><div class="summary-title">Foo(...)</div><span class="tooltip">Foo()</span></summary>
        """
    )
    self.assert_content(
        self._view.content(
            Foo(),
        ),
        """
        <span class="simple-value foo">Foo()</span>
        """
    )

  def test_config_override(self):

    class Foo(tree_view.HtmlTreeView.Extension):

      def __str__(self):
        return 'Foo()'

      @classmethod
      def _html_tree_view_css_styles(cls) -> list[str]:
        return [
            """
            /* Foo style */
            """
        ]

      @classmethod
      def _html_tree_view_config(cls) -> dict[str, Any]:
        return dict(
            css_classes=['foo', 'bar'],
            include_keys=['x', 'y', 'z', 'w'],
            exclude_keys=['z'],
            key_style='label',
            key_color=('white', 'red'),
            collapse_level=2,
            uncollapse=KeyPathSet(['x.a'], include_intermediate=True),
            child_config={
                'x': dict(
                    summary_color=('white', 'blue'),
                    key_style='label',
                    key_color=('white', 'blue'),
                    collapse_level=1,
                ),
                'y': dict(
                    uncollapse=KeyPathSet(['e']),
                )
            }
        )

      def _html_tree_view_content(
          self,
          *,
          view: tree_view.HtmlTreeView,
          parent: Any,
          root_path: KeyPath,
          **kwargs) -> base.Html:
        return view.complex_value(
            kv=dict(
                x=dict(a=dict(foo=2)),
                y=dict(b=dict(bar=3), e=dict(srt=6)),
                z=dict(c=dict(baz=4)),
                w=dict(d=dict(qux=5)),
            ),
            parent=self,
            root_path=root_path,
            **kwargs
        )

    self.assertIn(
        '/* Foo style */',
        self._view.render(
            Foo(),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=0,
        ).styles.content,
    )
    self.assert_content(
        self._view.render(
            [Foo()],
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=0,
        ),
        """
        <details class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details open class="pyglove foo bar"><summary><div class="summary-title foo bar">Foo(...)</div></summary><div class="complex-value foo"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">x</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">a</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">foo</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">y</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">b</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">bar</span></td><td><span class="simple-value int">3</span></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">e</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">srt</span></td><td><span class="simple-value int">6</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">w</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">d</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">qux</span></td><td><span class="simple-value int">5</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details>
        """
    )
    self.assert_content(
        self._view.render(
            [Foo()],
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=1,
        ),
        """
        <details open class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details open class="pyglove foo bar"><summary><div class="summary-title foo bar">Foo(...)</div></summary><div class="complex-value foo"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">x</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">a</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">foo</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">y</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">b</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">bar</span></td><td><span class="simple-value int">3</span></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">e</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">srt</span></td><td><span class="simple-value int">6</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">w</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">d</span></td><td><details class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">qux</span></td><td><span class="simple-value int">5</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details>
        """
    )
    self.assert_content(
        self._view.render(
            [Foo()],
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            collapse_level=None,
        ),
        """
        <details open class="pyglove list"><summary><div class="summary-title">list(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details open class="pyglove foo bar"><summary><div class="summary-title foo bar">Foo(...)</div></summary><div class="complex-value foo"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">x</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">a</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:blue;">foo</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">y</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">b</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">bar</span></td><td><span class="simple-value int">3</span></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">e</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">srt</span></td><td><span class="simple-value int">6</span></td></tr></table></div></details></td></tr></table></div></details></td></tr><tr><td><span class="object-key str" style="color:white;background-color:red;">w</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">d</span></td><td><details open class="pyglove dict"><summary><div class="summary-title">dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str" style="color:white;background-color:red;">qux</span></td><td><span class="simple-value int">5</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details>
        """
    )

  def test_behavior_overrides(self):

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
            **kwargs
        )

      def _html_tree_view_content(
          self,
          **kwargs
      ):
        del kwargs
        return 'Content of MyFoo'

    self.assert_content(
        self._view.render(
            Foo(),
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove foo"><summary><div class="summary-title">MyFoo</div></summary>Content of MyFoo</details>
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


class OverrideKwargsTest(TestCase):

  def test_override_collapse_level(self):
    def assert_collapse_level(
        call_kwargs, overriden_kwargs, expected_collapse_level
    ):
      self.assertEqual(
          tree_view.HtmlTreeView.get_kwargs(
              call_kwargs,
              overriden_kwargs,
              KeyPath(''),
          ).get('collapse_level', -1),
          expected_collapse_level,
      )
    assert_collapse_level(
        dict(),
        dict(),
        -1,
    )
    assert_collapse_level(
        dict(collapse_level=1),
        dict(),
        1,
    )
    assert_collapse_level(
        dict(collapse_level=1),
        dict(collapse_level=None),
        None,
    )
    assert_collapse_level(
        dict(collapse_level=1),
        dict(collapse_level=2),
        2,
    )
    assert_collapse_level(
        dict(collapse_level=2),
        dict(collapse_level=1),
        2,
    )

  def test_override_uncollapse(self):
    def assert_uncollapse(
        call_kwargs,
        overriden_kwargs,
        expected_uncollapse,
        root_path=KeyPath(),
    ):
      self.assertEqual(
          tree_view.HtmlTreeView.get_kwargs(
              call_kwargs,
              overriden_kwargs,
              root_path,
          ).get('uncollapse', None),
          expected_uncollapse
      )
    assert_uncollapse(
        dict(uncollapse=None),
        dict(),
        None,
    )
    assert_uncollapse(
        dict(),
        dict(uncollapse=None),
        KeyPathSet(),
    )
    assert_uncollapse(
        dict(uncollapse=KeyPathSet(['a.b'])),
        dict(),
        KeyPathSet(['a.b']),
    )
    assert_uncollapse(
        dict(uncollapse=KeyPathSet(['a.b.c'])),
        dict(uncollapse=KeyPathSet(['b'])),
        KeyPathSet(['a.b.c', 'a.b']),
        root_path=KeyPath('a'),
    )
    assert_uncollapse(
        dict(uncollapse=[]),
        dict(uncollapse=KeyPathSet(['b'])),
        KeyPathSet(['a.b']),
        root_path=KeyPath('a'),
    )
    assert_uncollapse(
        dict(uncollapse=KeyPathSet(['a.b.c'])),
        dict(uncollapse=KeyPathSet(['b'])),
        KeyPathSet(['a.b.c', 'a.b']),
        root_path=KeyPath('a'),
    )

  def test_override_kwargs(self):
    def assert_child_config(
        call_kwargs,
        overriden_kwargs,
        expected_child_config,
        root_path=KeyPath(''),
    ):
      self.maxDiff = None
      actual = tree_view.HtmlTreeView.get_kwargs(
          call_kwargs,
          overriden_kwargs,
          root_path,
      )
      if actual != expected_child_config:
        print(actual)
      self.assertEqual(actual, expected_child_config)

    assert_child_config(
        dict(
            child_config=dict(
                x=dict(
                    enable_summary=True
                ),
            )
        ),
        dict(
            child_config=dict(
                x=dict(
                    enable_summary_tooltip=False
                ),
                y=dict(
                    collapse_level=None
                ),
            )
        ),
        dict(
            child_config=dict(
                x=dict(
                    enable_summary=True,
                    enable_summary_tooltip=False,
                ),
                y=dict(
                    collapse_level=None
                ),
            ),
        ),
    )

  def test_get_child_kwargs(self):
    def assert_child_kwargs(
        call_kwargs,
        child_key,
        root_path,
        expected_child_kwargs,
    ):
      self.maxDiff = None
      actual = tree_view.HtmlTreeView.get_child_kwargs(
          call_kwargs,
          call_kwargs.pop('child_config', {}),
          child_key,
          root_path,
      )
      if actual != expected_child_kwargs:
        print(actual)
      self.assertEqual(actual, expected_child_kwargs)

    assert_child_kwargs(
        call_kwargs=dict(
            enable_summary=False,
            collapse_level=1,
            uncollapse=KeyPathSet(['a.y']),
            child_config=dict(
                x=dict(
                    enable_summary=True,
                    uncollapse=KeyPathSet(['b']),
                    child_config=dict(
                        y=dict(
                            collapse_level=2
                        ),
                    ),
                ),
            )
        ),
        child_key='x',
        root_path=KeyPath(['a']),
        expected_child_kwargs=dict(
            enable_summary=True,
            collapse_level=1,
            uncollapse=KeyPathSet(['a.y', 'a.x.b']),
            child_config=dict(
                y=dict(
                    collapse_level=2
                ),
            ),
        ),
    )


class HelperTest(TestCase):

  def test_get_collapse_level(self):
    self.assertIsNone(
        tree_view.HtmlTreeView.get_collapse_level(
            None,
            None
        ),
    )
    self.assertIsNone(
        tree_view.HtmlTreeView.get_collapse_level(
            1,
            None
        ),
    )
    self.assertIsNone(
        tree_view.HtmlTreeView.get_collapse_level(
            None,
            1
        ),
    )
    self.assertEqual(
        tree_view.HtmlTreeView.get_collapse_level(
            2,
            3
        ),
        3
    )
    self.assertEqual(
        tree_view.HtmlTreeView.get_collapse_level(
            (3, -1),
            0
        ),
        2
    )
    self.assertEqual(
        tree_view.HtmlTreeView.get_collapse_level(
            0,
            (2, 1)
        ),
        3
    )

  def test_get_color(self):
    self.assertEqual(
        tree_view.HtmlTreeView.get_color(None, KeyPath('a.b.c'), 1, None),
        (None, None)
    )
    self.assertEqual(
        tree_view.HtmlTreeView.get_color(
            ('blue', None), KeyPath('a.b.c'), 1, None
        ),
        ('blue', None)
    )
    self.assertEqual(
        tree_view.HtmlTreeView.get_color(
            lambda k, v, p: ('white', 'black'), KeyPath('a.b.c'), 1, None),
        ('white', 'black')
    )

  def test_merge_uncollapse(self):
    self.assertEqual(
        tree_view.HtmlTreeView.merge_uncollapse(
            None,
            None,
            child_path=KeyPath.parse('a.b'),
        ),
        KeyPathSet()
    )
    self.assertEqual(
        tree_view.HtmlTreeView.merge_uncollapse(
            KeyPathSet(['a.b']),
            None,
            child_path=KeyPath.parse('x'),
        ),
        KeyPathSet(['a.b'])
    )
    path_fn = lambda k, v, p: True
    self.assertIs(
        tree_view.HtmlTreeView.merge_uncollapse(
            path_fn,
            None,
            child_path=KeyPath.parse('x'),
        ),
        path_fn
    )
    self.assertEqual(
        tree_view.HtmlTreeView.merge_uncollapse(
            KeyPathSet(),
            KeyPathSet(['d']),
            child_path=KeyPath.parse('a.b'),
        ),
        KeyPathSet(['a.b.d']),
    )
    self.assertEqual(
        tree_view.HtmlTreeView.merge_uncollapse(
            KeyPathSet(['a.b.c']),
            KeyPathSet(['d']),
            child_path=KeyPath.parse('a.b'),
        ),
        KeyPathSet(['a.b.c', 'a.b.d']),
    )

  def test_get_passthrough_kwargs(self):
    key_filter = (lambda k, v, p: k == 'foo')
    self.assertEqual(
        tree_view.HtmlTreeView.get_passthrough_kwargs(
            name='foo',
            value=1,
            root_path=KeyPath(),
            enable_summary=False,
            include_keys=key_filter,
        ),
        {
            'enable_summary': False,
            'include_keys': key_filter,
        },
    )

    self.assertEqual(
        tree_view.HtmlTreeView.get_passthrough_kwargs(
            name='foo',
            value=1,
            root_path=KeyPath(),
            enable_summary=False,
            include_keys=['a'],
            extra_flags=dict(x=1),
            exclude_keys='a',
            remove=['extra_flags']
        ),
        {
            'enable_summary': False,
        },
    )


if __name__ == '__main__':
  unittest.main()
