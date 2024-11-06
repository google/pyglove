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
from pyglove.core.views.html.controls import tab as tab_lib


class TabControlTest(unittest.TestCase):

  def assert_html_content(self, control, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = control.to_html().content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_basic(self):
    tab = tab_lib.TabControl([
        tab_lib.Tab('foo', base.Html('<h1>foo</h1>')),
        tab_lib.Tab('bar', base.Html('<h1>bar</h1>')),
    ])
    self.assert_html_content(
        tab,
        (
            f'<div class="tab-control top" id="{tab.element_id()}">'
            '<div class="tab-button-group"><button class="tab-button selected" '
            f'''onclick="openTab(event, '{tab.element_id()}', '''
            f''''{tab.element_id(str(0))}')">'''
            '<a class="label">foo</a></button><button class="tab-button" '
            f'''onclick="openTab(event, '{tab.element_id()}', '''
            f''''{tab.element_id(str(1))}')">'''
            '<a class="label">bar</a></button></div>'
            '<div class="tab-content" style="display:block;" '
            f'id="{tab.element_id(str(0))}"><h1>foo</h1></div>'
            '<div class="tab-content" style="display:none;" '
            f'id="{tab.element_id(str(1))}"><h1>bar</h1></div></div>'
        )
    )


if __name__ == '__main__':
  unittest.main()
