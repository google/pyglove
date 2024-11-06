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
"""Tab control."""

from typing import Annotated, List, Literal, Union

from pyglove.core.symbolic import object as pg_object
# pylint: disable=g-importing-member
from pyglove.core.views.html.base import Html
from pyglove.core.views.html.base import HtmlConvertible
from pyglove.core.views.html.controls.base import HtmlControl
from pyglove.core.views.html.controls.label import Label
# pylint: enable=g-importing-member


class Tab(pg_object.Object):
  """A tab."""

  label: Annotated[
      Label,
      'The label of the tab.'
  ]

  content: Annotated[
      Union[Html, HtmlConvertible],
      'The content of the tab.'
  ]


@pg_object.use_init_args(
    ['tabs', 'selected', 'tab_position', 'id', 'css_classes', 'styles']
)
class TabControl(HtmlControl):
  """A tab control."""

  tabs: Annotated[
      List[Tab],
      'The tabs of the tab control.'
  ]

  selected: Annotated[
      int,
      'The index of the selected tab.'
  ] = 0

  tab_position: Annotated[
      Literal['top', 'left'],
      'The direction of the tab control.'
  ] = 'top'

  interactive = True

  def _to_html(self, **kwargs):
    return Html.element(
        'div',
        [
            Html.element(
                'div',
                [
                    Html.element(
                        'button',
                        [
                            tab.label
                        ],
                        css_classes=[
                            'tab-button',
                            'selected' if i == self.selected else None
                        ],
                        onclick=(
                            f"""openTab(event, '{self.element_id()}', '{self.element_id(str(i))}')"""
                        )
                    ) for i, tab in enumerate(self.tabs)
                ],
                css_classes=['tab-button-group'],
            ),
        ] + [
            Html.element(
                'div',
                [
                    tab.content
                ],
                css_classes=['tab-content'],
                styles=dict(
                    display='block' if i == self.selected else 'none'
                ),
                id=self.element_id(str(i))
            ) for i, tab in enumerate(self.tabs)
        ],
        css_classes=['tab-control', self.tab_position] + self.css_classes,
        id=self.element_id(),
        styles=self.styles,
    ).add_script(
        """
        function openTab(event, controlId, tabId) {
          const tabButtons = document.querySelectorAll('#' + controlId + '> .tab-button-group > .tab-button');
          for (let i = 0; i < tabButtons.length; i++) {
            tabButtons[i].classList.remove('selected');
          }
          const tabContents = document.querySelectorAll('#' + controlId + '> .tab-content');
          for (let i = 0; i < tabContents.length; i++) {
            tabContents[i].style.display = 'none';
          }
          const tabButton = event.currentTarget;
          tabButton.classList.add('selected');
          document.getElementById(tabId).style.display = "block";
        }
        """
    ).add_style(
        """
        .top .tab-button-group {
          overflow-x: hidden;
          border-bottom: 1px solid #EEE;
        }
        .left .tab-button-group {
          float: left;
          top: 0;
        }
        .tab-button {
          background-color: #EEE;
          border: 1px solid #EEE;
          outline: none;
          cursor: pointer;
          transition: 0.3s;
          padding: 10px 15px 10px 15px;
        }
        .top .tab-button {
          border-top-width: 2px;
        }
        .left .tab-button {
          display: block;
          width: 100%;
          border-left-width: 2px;
        }
        .top .tab-button.selected {
          border-bottom-color: #fff;
          border-top-color: #B721FF;
          background: #fff;
        }
        .left .tab-button.selected {
          border-right-color: #fff;
          border-left-color: #B721FF;
          background: #fff;
        }
        .top .tab-button:hover {
          border-top-color: orange;
        }
        .left .tab-button:hover {
          border-left-color: orange;
        }
        .tab-content {
          display: none;
          padding: 10px;
        }
        .left .tab-content {
          float: left;
        }
        """
    )
