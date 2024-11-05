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
from pyglove.core.views.html import base
from pyglove.core.views.html.controls import tooltip as tooltip_lib


class TooltipTest(unittest.TestCase):

  def assert_html_content(self, control, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = control.to_html().content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_basic(self):
    tooltip = tooltip_lib.Tooltip('foo')
    with self.assertRaisesRegex(
        ValueError, 'CSS selector `for_element` is required'
    ):
      tooltip.to_html()

    tooltip = tooltip_lib.Tooltip('foo', for_element='.bar')
    self.assertEqual(tooltip.for_element, '.bar')
    self.assert_html_content(
        tooltip,
        '<span class="tooltip">foo</span>'
    )
    self.assertIn(
        inspect.cleandoc(
            """
            .bar:hover + .tooltip {
              visibility: visible;
            }
            """
        ),
        tooltip.to_html().style_section,
    )

  def test_update(self):
    tooltip = tooltip_lib.Tooltip('foo', for_element='.bar', interactive=True)
    self.assertIn('id="control-', tooltip.to_html_str(content_only=True))
    with tooltip.track_scripts() as scripts:
      tooltip.update('normal text')
      self.assertEqual(tooltip.content, 'normal text')
      self.assertEqual(
          scripts,
          [
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{tooltip.element_id()}");
                  elem.textContent = "normal text";
                  """
              ),
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{tooltip.element_id()}");
                  elem.classList.remove("html-content");
                  """
              ),
          ]
      )
    with tooltip.track_scripts() as scripts:
      tooltip.update(base.Html('<b>bold text</b>'))
      self.assertEqual(
          scripts,
          [
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{tooltip.element_id()}");
                  elem.innerHTML = "<b>bold text</b>";
                  """
              ),
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{tooltip.element_id()}");
                  elem.classList.add("html-content");
                  """
              ),
          ]
      )

if __name__ == '__main__':
  unittest.main()
