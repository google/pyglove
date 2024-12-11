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
from pyglove.core.views.html.controls import progress_bar


class ProgressBarTest(unittest.TestCase):

  def assert_html_content(self, control, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = control.to_html().content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_basic(self):
    bar = progress_bar.ProgressBar(
        subprogresses=[
            progress_bar.SubProgress('foo'),
            progress_bar.SubProgress('bar', 20),
        ],
        total=None,
    )
    self.assert_html_content(
        bar,
        (
            '<div class="progress-bar"><div class="shade">'
            f'<div class="sub-progress foo" id="{bar["foo"].element_id()}">'
            '</div><div class="sub-progress bar" '
            f'id="{bar["bar"].element_id()}"></div></div>'
            '<div class="label-container"><span class="label progress-label"'
            f' id="{bar._progress_label.element_id()}">n/a</span><span class='
            f'"tooltip" id="{bar._progress_label.tooltip.element_id()}">'
            'Not started</span></div></div>'
        )
    )
    self.assertEqual(bar['foo'], progress_bar.SubProgress('foo'))
    self.assertEqual(bar['bar'], progress_bar.SubProgress('bar', 20))
    with self.assertRaisesRegex(KeyError, 'Sub progress bar .* not found'):
      _ = bar['baz']
    self.assertIsNone(bar['foo'].total)
    self.assertIsNone(bar['foo'].width)

    bar.update(total=100)
    self.assertEqual(bar.total, 100)
    self.assertEqual(bar['foo'].total, 100)
    self.assertEqual(bar['foo'].width, '0%')
    self.assertEqual(bar['bar'].width, '20%')
    with bar.track_scripts() as scripts:
      bar['foo'].increment()
      self.assertEqual(
          scripts,
          [
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{bar['foo'].element_id()}");
                  elem.style = "width:1%;";
                  """
              ),
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{bar._progress_label.element_id()}");
                  elem.textContent = " 21.0% (21/100)";
                  """
              ),
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{bar._progress_label.tooltip.element_id()}");
                  elem.textContent = "foo: 1.0% (1/100)\\nbar: 20.0% (20/100)";
                  """
              ),
              inspect.cleandoc(
                  f"""
                  elem = document.getElementById("{bar._progress_label.tooltip.element_id()}");
                  elem.classList.remove("html-content");
                  """
              ),
          ]
      )


if __name__ == '__main__':
  unittest.main()
