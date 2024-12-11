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

from typing import Annotated, List, Literal, Optional, Union

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

  name: Annotated[
      Optional[str],
      'An optional name that can be used to identify a tab under a tab control'
  ] = None


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

    self._insert_adjacent_html(
        f"""
        const elem = document.getElementById('{self.element_id()}-button-group');
        """,
        self._tab_button(tab, len(self.tabs) - 1),
        position='beforeend',
    )
    self._insert_adjacent_html(
        f"""
        const elem = document.getElementById('{self.element_id()}-content-group');
        """,
        self._tab_content(tab, len(self.tabs) - 1),
        position='beforeend',
    )

  def insert(self, index_or_name: Union[int, str], tab: Tab) -> None:
    """Inserts a tab before a tab identified by index or name."""
    index = self.indexof(index_or_name)
    if index == -1:
      raise ValueError(f'Tab not found: {index_or_name!r}')
    with pg_flags.notify_on_change(False):
      self.tabs.insert(index, tab)

    self._insert_adjacent_html(
        f"""
        const elem = document.querySelectorAll('#{self.element_id()}-button-group > .tab-button')[{index}];
        """,
        self._tab_button(tab, len(self.tabs) - 1),
        position='beforebegin',
    )
    self._insert_adjacent_html(
        f"""
        const elem = document.querySelectorAll('#{self.element_id()}-content-group > .tab-content')[{index}];
        """,
        self._tab_content(tab, len(self.tabs) - 1),
        position='beforebegin',
    )

  def indexof(self, index_or_name: Union[int, str]) -> int:
    if isinstance(index_or_name, int):
      index = index_or_name
      if index >= len(self.tabs):
        return len(self.tabs) - 1
      elif index < -len(self.tabs):
        return -1
      elif index < 0:
        index = index + len(self.tabs)
      return index
    else:
      name = index_or_name
      assert isinstance(name, str), name
      for i, tab in enumerate(self.tabs):
        if tab.name == name:
          return i
      return -1

  def extend(self, tabs: List[Tab]) -> None:
    for tab in tabs:
      self.append(tab)

  def select(
      self,
      index_or_name: Union[int, str, List[str]]) -> Union[int, str]:
    """Selects a tab identified by an index or name.

    Args:
      index_or_name: The index or name of the tab to select. If a list of names
        is provided, the first name in the list that is found will be selected.

    Returns:
      The index (if the index was provided) or name of the selected tab.
    """
    selected_name = index_or_name if isinstance(index_or_name, str) else None
    index = -1
    if isinstance(index_or_name, list):
      for name in index_or_name:
        index = self.indexof(name)
        if index != -1:
          selected_name = name
          break
    else:
      index = self.indexof(index_or_name)
    if index == -1:
      raise ValueError(f'Tab not found: {index_or_name!r}')
    self._sync_members(selected=index)
    self._run_javascript(
        f"""
        const tabButtons = document.querySelectorAll('#{self.element_id()}-button-group > .tab-button');
        tabButtons[{index}].click();
        """
    )
    return selected_name or index

  def _to_html(self, **kwargs):
    return Html.element(
        'table',
        [
            '<tr><td>',
            Html.element(
                'div',
                [self._tab_button(tab, i) for i, tab in enumerate(self.tabs)],
                css_classes=[
                    'tab-button-group',
                    self.tab_position
                ] + self.css_classes,
                id=self.element_id('button-group'),
            ),
            ('</td><td>' if self.tab_position == 'left'
             else '</td></tr><tr><td>'),
            Html.element(
                'div',
                [self._tab_content(tab, i) for i, tab in enumerate(self.tabs)],
                css_classes=[
                    'tab-content-group',
                    self.tab_position
                ] + self.css_classes,
                id=self.element_id('content-group'),
            ),
            '</td></tr>'
        ],
        css_classes=['tab-control'],
        styles=self.styles,
    ).add_script(
        """
        function openTab(event, controlId, tabId) {
          const tabButtons = document.querySelectorAll('#' + controlId + '-button-group > .tab-button');
          for (let i = 0; i < tabButtons.length; i++) {
            tabButtons[i].classList.remove('selected');
          }
          const tabContents = document.querySelectorAll('#' + controlId + '-content-group > .tab-content');
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
        .tab-control {
          border-spacing: 0px;
          border-collapse: collapse;
          margin-top: 10px;
        }
        .tab-control td {
          padding: 0px;
          margin: 0px;
          vertical-align: top;
        }
        .top.tab-button-group {
          border-left: 1px solid #DDD;
          border-top: 1px solid #DDD;
          border-right: 1px solid #DDD;
          border-radius: 5px 5px 0px 0px;
          padding: 0px 5px 0px 0px;
          margin-bottom: -2px;
        }
        .tab-button {
          background-color: transparent;
          border-radius: 5px;
          border: 0px;
          font-weight: bold;
          color: gray;
          outline: none;
          cursor: pointer;
          transition: 0.3s;
        }
        .tab-button:hover {
          background-color: #fff1dd;
        }
        .tab-button.selected {
          background-color: #f0ecf9;
          color: black;
        }
        .top.tab-content-group {
          border-left: 1px solid #DDD;
          border-right: 1px solid #DDD;
          border-bottom: 1px solid #DDD;
          border-radius: 0px 0px 5px 5px;
          margin: 0px;
          padding: 5px;
          height: 100%;
        }
        .top > .tab-button {
          margin: 5px 0px 5px 5px;
        }
        .tab-content {
          display: none;
        }
        .tab-content.selected {
          display: block;
        }
        .left.tab-button-group {
          display: inline-flex;
          flex-direction: column;
          border: 1px solid #DDD;
          border-radius: 5px;
          margin-right: 5px;
          padding: 0px 0px 5px 0px;
        }
        .left.tab-content-group {
          border: 0px
          margin: 0px;
          padding: 0px;
          height: 100%;
        }
        .left > .tab-button {
          text-align: left;
          margin: 5px 5px 0px 5px;
        }
        """
    )

  def _tab_button(self, tab: Tab, i: int) -> Html:
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

  def _tab_content(self, tab: Tab, i: int) -> Html:
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
