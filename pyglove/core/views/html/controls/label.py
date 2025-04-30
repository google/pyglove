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
"""Html label control."""

from typing import Annotated, Any, Dict, List, Optional, Union

from pyglove.core import typing as pg_typing
from pyglove.core import utils as pg_utils
from pyglove.core.symbolic import flags as pg_flags
from pyglove.core.symbolic import object as pg_object
# pylint: disable=g-importing-member
from pyglove.core.views.html.base import Html
from pyglove.core.views.html.controls.base import HtmlControl
from pyglove.core.views.html.controls.tooltip import Tooltip
# pylint: enable=g-importing-member


@pg_object.use_init_args(
    ['text', 'tooltip', 'link', 'id', 'css_classes', 'styles']
)
class Label(HtmlControl):
  """Html label."""

  text: Annotated[
      Union[str, Html],
      'The text or HTML content of the label.'
  ]

  tooltip: Annotated[
      Optional[Tooltip],
      '(Optional) The tooltip for the label.'
  ] = None

  link: Annotated[
      Optional[str],
      '(Optional) The URL for the link of the label.'
  ] = None

  target: Annotated[
      Optional[str],
      '(Optional) The target for the link of the label.'
  ] = None

  def _on_bound(self):
    super()._on_bound()
    if self.tooltip is not None:
      self.tooltip.rebind(
          for_element='.label',
          interactive=self.tooltip.interactive or self.interactive,
          notify_parents=False
      )
    self.add_style(
        """
        .label {
          display: inline-block;
          color: inherit;
          padding: 5px;
        }
        .label-container {
            display: inline-block;
        }
        """
    )

  def _to_html(self, **kwargs) -> Html:
    text_elem = Html.element(
        'a' if self.link is not None else 'span',
        [self.text],
        id=self.element_id(),
        href=self.link,
        css_classes=['label'] + self.css_classes,
        styles=self.styles,
        target=self.target,
    )
    if self.tooltip is None:
      return text_elem
    return Html.element(
        'div',
        [text_elem, self.tooltip],
        css_classes=['label-container'],
    )

  def update(
      self,
      text: Union[str, Html, None] = None,
      tooltip: Union[str, Html, None] = None,
      link: Optional[str] = None,
      styles: Optional[Dict[str, Any]] = None,
      add_class: Optional[List[str]] = None,
      remove_class: Optional[List[str]] = None,
  ) -> None:
    if text is not None:
      self._sync_members(text=self._update_content(text))
    if styles:
      self._sync_members(styles=self._update_style(styles))
    if link is not None:
      self._sync_members(link=self._update_property('href', link))
    if tooltip is not None:
      self.tooltip.update(content=tooltip)
    if add_class or remove_class:
      css_classes = list(self.css_classes)
      for x in add_class or []:
        self._add_css_class(x)
        css_classes.append(x)
      for x in remove_class or []:
        self._remove_css_class(x)
        if x in css_classes:
          css_classes.remove(x)
      self._sync_members(css_classes=css_classes)

# Register converter for automatic conversion.
pg_typing.register_converter(str, Label, Label)
pg_typing.register_converter(Html, Label, Label)


class Badge(Label):
  """A badge."""

  def _on_bound(self):
    super()._on_bound()
    with pg_flags.notify_on_change(False):
      self.css_classes.append('badge')
    self.add_style(
        """
        .badge {
          background-color: #EEE;
          border-radius: 5px;
          color: #777;
        }
        """
    )


@pg_object.use_init_args(
    ['labels', 'name', 'id', 'css_classes', 'styles']
)
class LabelGroup(HtmlControl):
  """Label group."""

  labels: Annotated[
      List[Label],
      'The labels in the group.'
  ]

  name: Annotated[
      Optional[Label],
      'The label for the name of the group.'
  ] = None

  @pg_utils.explicit_method_override
  def __init__(
      self,
      labels: List[Union[Label, str, Html, None, List[Any]]],
      name: Optional[Label] = None,
      id: Optional[str] = None,  # pylint: disable=redefined-builtin
      css_classes: Optional[List[str]] = None,
      styles: Optional[Dict[str, str]] = None,
      **kwargs,
  ) -> None:
    if labels:
      labels = [l for l in pg_utils.flatten(labels).values() if l is not None]
    super().__init__(
        labels=labels, name=name, id=id, css_classes=css_classes or [],
        styles=styles or {}, **kwargs
    )

  def _on_bound(self):
    super()._on_bound()
    with pg_flags.notify_on_change(False):
      if self.name is not None:
        self.name.rebind(
            interactive=self.interactive or self.name.interactive,
            raise_on_no_change=False
        )
        self.name.css_classes.append('group-name')
      for label in self.labels:
        label.css_classes.append('group-value')
        label.rebind(
            interactive=self.interactive or label.interactive,
            raise_on_no_change=False,
        )

  def _to_html(self, **kwargs) -> Html:
    return Html.element(
        'div',
        [self.name] + self.labels,
        id=self.id or None,
        css_classes=['label-group'] + self.css_classes,
        styles=self.styles,
    ).add_style(
        """
        .label-group {
          display: inline-flex;
          overflow: hidden;
          border-radius: 5px;
          border: 1px solid #DDD;
          padding: 0px;
          margin: 5px
        }
        .label-group .group-name {
          padding: 5px;
          color: white;
          background-color: dodgerblue;
        }
        .label-group .group-name > .text {
          color: white;
        }
        .label-group .group-value {
        }
        .label-group .badge {
          border-radius: 0px;
        }
        """
    )
