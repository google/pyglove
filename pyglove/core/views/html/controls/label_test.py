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

from pyglove.core import symbolic   # pylint: disable=unused-import
from pyglove.core.views.html.controls import label as label_lib


class TestCase(unittest.TestCase):

  def assert_html_content(self, control, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = control.to_html().content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)


class LabelTest(TestCase):

  def test_text_only(self):
    label = label_lib.Label('foo')
    self.assertIsNone(label.tooltip)
    self.assertIsNone(label.link)
    self.assert_html_content(label, '<a class="label">foo</a>')
    with self.assertRaisesRegex(ValueError, 'Non-interactive .*'):
      label.update('bar')

  def test_with_link(self):
    label = label_lib.Label('foo', link='http://google.com', id='my-foo')
    self.assertEqual(label.element_id(), 'my-foo')
    self.assertIsNone(label.tooltip)
    self.assertEqual(label.link, 'http://google.com')
    self.assert_html_content(
        label, '<a class="label" id="my-foo" href="http://google.com">foo</a>'
    )

  def test_with_tooltip(self):
    label = label_lib.Label('foo', tooltip='bar')
    self.assertEqual(label.tooltip.for_element, '.label')
    self.assertIsNone(label.link)
    self.assert_html_content(
        label,
        (
            '<div class="label-container"><a class="label">foo</a>'
            '<span class="tooltip">bar</span></div>'
        )
    )

  def test_update(self):
    label = label_lib.Label('foo', 'bar', 'http://google.com', interactive=True)
    self.assertIn('id="control-', label.to_html_str(content_only=True))
    with label.track_scripts() as scripts:
      label.update(
          'bar',
          tooltip='baz',
          link='http://www.yahoo.com',
          styles=dict(color='red')
      )
    self.assertEqual(label.text, 'bar')
    self.assertEqual(label.tooltip.content, 'baz')
    self.assertEqual(label.link, 'http://www.yahoo.com')
    self.assertEqual(label.styles, dict(color='red'))
    self.assertEqual(
        scripts,
        [
            inspect.cleandoc(
                f"""
                elem = document.getElementById("{label.element_id()}");
                elem.textContent = "bar";
                """
            ),
            inspect.cleandoc(
                f"""
                elem = document.getElementById("{label.element_id()}");
                elem.style = "color:red;";
                """
            ),
            inspect.cleandoc(
                f"""
                elem = document.getElementById("{label.element_id()}");
                elem.href = "http://www.yahoo.com";
                """
            ),
            inspect.cleandoc(
                f"""
                elem = document.getElementById("{label.tooltip.element_id()}");
                elem.textContent = "baz";
                """
            ),
            inspect.cleandoc(
                f"""
                elem = document.getElementById("{label.tooltip.element_id()}");
                elem.classList.remove("html-content");
                """
            ),
        ]
    )

  def test_badge(self):
    badge = label_lib.Badge('foo')
    self.assert_html_content(badge, '<a class="label badge">foo</a>')


class LabelGroupTest(TestCase):

  def test_basic(self):
    group = label_lib.LabelGroup(['foo', 'bar'], name='baz')
    self.assert_html_content(
        group,
        (
            '<div class="label-group"><a class="label group-name">baz</a>'
            '<a class="label group-value">foo</a>'
            '<a class="label group-value">bar</a></div>'
        )
    )

  def test_interactive(self):
    group = label_lib.LabelGroup(['foo', 'bar'], name='baz', interactive=True)
    self.assertTrue(group.name.interactive)
    self.assertTrue(group.labels[0].interactive)
    self.assertTrue(group.labels[1].interactive)


if __name__ == '__main__':
  unittest.main()
