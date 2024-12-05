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

from pyglove.core.symbolic import flags as pg_flags
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

  css_classes: Annotated[
      List[str],
      'The CSS classes of the tab.'
  ] = []


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

  def append(self, tab: Tab) -> None:
    with pg_flags.notify_on_change(False):
      self.tabs.append(tab)

    def class_list(elem_class: str) -> str:
      classes = [elem_class] + tab.css_classes
      if len(self.tabs) == 1:
        classes.append('selected')
      return ', '.join(f'"{v}"' for v in classes)

    # pytype: disable=attribute-error
    self._run_javascript(
        f"""
        const tabButtonGroups = document.querySelectorAll('#{self.element_id()}' + '> .tab-button-group');
        console.assert(tabButtonGroups.length === 1);
        const tabButton = document.createElement('button');
        tabButton.classList.add({class_list('tab-button')});
        tabButton.innerHTML = "{Html.escape(tab.label, javascript_str=True).to_str(content_only=True)}";
        tabButton.onclick = function() {{
            openTab(event, '{self.element_id()}', '{self.element_id(str(len(self.tabs) - 1))}');
        }};
        tabButtonGroups[0].appendChild(tabButton);

        const tabContentGroups = document.querySelectorAll('#{self.element_id()}' + '> .tab-content-group');
        console.assert(tabContentGroups.length === 1);
        const tabContent = document.createElement('div');
        tabContent.id = '{self.element_id(str(len(self.tabs) - 1))}',
        tabContent.classList.add({class_list('tab-content')});
        tabContent.innerHTML = "{Html.escape(tab.content, javascript_str=True).to_str(content_only=True)}";
        tabContentGroups[0].appendChild(tabContent)
        """
    )
    # pytype: enable=attribute-error

  def extend(self, tabs: List[Tab]) -> None:
    for tab in tabs:
      self.append(tab)

  def _to_html(self, **kwargs):
    def _tab_button(tab: Tab, i: int) -> Html:
      return Html.element(
          'button',
          [
              tab.label
          ],
          css_classes=[
              'tab-button',
              'selected' if i == self.selected else None
          ] + tab.css_classes,
          onclick=(
              f"""openTab(event, '{self.element_id()}', '{self.element_id(str(i))}')"""
          )
      )

    def _tab_content(tab: Tab, i: int) -> Html:
      return Html.element(
          'div',
          [
              tab.content
          ],
          css_classes=[
              'tab-content',
              'selected' if i == self.selected else None
          ] + tab.css_classes,
          id=self.element_id(str(i))
      )
    return Html.element(
        'div',
        [
            Html.element(
                'div',
                [_tab_button(tab, i) for i, tab in enumerate(self.tabs)],
                css_classes=['tab-button-group'],
            ),
        ] + [
            Html.element(
                'div',
                [_tab_content(tab, i) for i, tab in enumerate(self.tabs)],
                css_classes=['tab-content-group'],
            )
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
          const tabContents = document.querySelectorAll('#' + controlId + '> .tab-content-group > .tab-content');
          for (let i = 0; i < tabContents.length; i++) {
            tabContents[i].classList.remove('selected')
          }
          const tabButton = event.currentTarget;
          tabButton.classList.add('selected');
          document.getElementById(tabId).classList.add('selected');
        }
        """
    ).add_style(
        """
        .top .tab-button-group {
          overflow-x: hidden;
          border-bottom: 1px solid #EEE;
        }
        .left .tab-button-group {
          display: inline-flex;
          flex-direction: column;
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
        .tab-content-group {
            display: block;
        }
        .left .tab-content-group {
          display: inline-flex;
        }
        .tab-content {
          display: none;
          padding: 10px;
        }
        .tab-content.selected {
          display: block;
        }
        """
    )
