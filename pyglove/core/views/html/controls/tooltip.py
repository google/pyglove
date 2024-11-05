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
"""Html tooltip."""

from typing import Optional, Union
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import object as pg_object
# pylint: disable=g-importing-member
from pyglove.core.views.html.base import Html
from pyglove.core.views.html.controls.base import HtmlControl
# pylint: disable=g-importing-member


@pg_object.use_init_args(
    ['content', 'for_element', 'id', 'css_classes', 'styles']
)
class Tooltip(HtmlControl):
  """A tooltip control.
  
  Attributes:
    content: The content of the tooltip. It could be a string or a HTML object.
    id: The id of the tooltip.
    css_classes: The CSS classes for the tooltip.
    for_element: The CSS selector for the element to attach the tooltip to.
      e.g. '.my-element' or '#my-element'.
  """

  content: Union[str, Html]
  for_element: Optional[str] = None

  def _to_html(self, **kwargs):
    if self.for_element is None:
      raise ValueError(
          'CSS selector `for_element` is required for tooltip to display.'
      )
    content = self.content
    if isinstance(self.content, str):
      content = Html.escape(self.content)
    return Html.element(
        'span',
        [content],
        id=self.element_id(),
        css_classes=[
            'tooltip',
            'html-content' if isinstance(content, Html) else None,
        ] + self.css_classes,
        styles=self.styles,
    ).add_style(
        """
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
        .tooltip.html-content {
          white-space: inherit;
          background-color: white;
          color: inherit;
          box-shadow: rgba(0, 0, 0, 0.16) 0px 1px 4px;
        }
        """,
        f"""
        {self.for_element}:hover + .tooltip {{
          visibility: visible;
        }}
        """
    )

  def update(self, content: Union[str, Html]) -> None:
    self._sync_members(content=self._update_content(content))
    if isinstance(content, Html):
      self._add_css_class('html-content')
    else:
      self._remove_css_class('html-content')


# Register converter for automatic conversion.
pg_typing.register_converter(str, Tooltip, Tooltip)
pg_typing.register_converter(Html, Tooltip, Tooltip)
