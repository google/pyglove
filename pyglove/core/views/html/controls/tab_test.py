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
        tab_lib.Tab('foo', base.Html('<h1>foo</h1>'), css_classes=['foo']),
        tab_lib.Tab('bar', base.Html('<h1>bar</h1>')),
    ])
    elem_id = tab.element_id()
    self.assert_html_content(
        tab,
        f"""<table class="tab-control"><tr><td><div class="tab-button-group top" id="{elem_id}-button-group"><button class="tab-button selected foo" onclick="openTab(event, '{elem_id}', '{elem_id}-0')"><a class="label">foo</a></button><button class="tab-button" onclick="openTab(event, '{elem_id}', '{elem_id}-1')"><a class="label">bar</a></button></div></td></tr><tr><td><div class="tab-content-group top" id="{elem_id}-content-group"><div class="tab-content selected foo" id="{elem_id}-0"><h1>foo</h1></div><div class="tab-content" id="{elem_id}-1"><h1>bar</h1></div></div></td></tr></table>"""
    )
    with tab.track_scripts() as scripts:
      tab.extend([
          tab_lib.Tab(
              'baz',
              base.Html(
                  '<h1 class="a">baz</h1>').add_style('.a { color: red; }')
              ),
          tab_lib.Tab('qux', base.Html('<h1>qux</h1>')),
      ])
      self.assertEqual(len(scripts), 6)


if __name__ == '__main__':
  unittest.main()
