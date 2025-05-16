# Copyright 2025 The PyGlove Authors
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
"""Symbolic class for capture error information."""

from typing import Annotated, List
from pyglove.core import utils
from pyglove.core.symbolic import dict as pg_dict  # pylint: disable=unused-import
from pyglove.core.symbolic import list as pg_list  # pylint: disable=unused-import
from pyglove.core.symbolic import object as pg_object
from pyglove.core.views.html import controls
from pyglove.core.views.html import tree_view


@pg_object.members(
    [],
    # For backwards compatibility.
    additional_keys=['pyglove.core.utils.error_utils.ErrorInfo']
)
class ErrorInfo(
    pg_object.Object,
    utils.ErrorInfo,
    tree_view.HtmlTreeView.Extension
):
  """Symbolic error info."""

  tag: Annotated[
      str,
      (
          'A path of the error types in the exception chain. For example, '
          '`ValueError.ZeroDivisionError` means the error is a '
          '`ZeroDivisionError` raised at the first place and then reraised as '
          'a `ValueError`.'
      )
  ]

  description: Annotated[
      str,
      'The description of the error.',
  ]

  stacktrace: Annotated[
      str,
      'The stacktrace of the error.',
  ]

  def _html_tree_view_summary(
      self,
      *,
      view: tree_view.HtmlTreeView,
      **kwargs
  ) -> tree_view.Html:
    kwargs.pop('title', None)
    kwargs.pop('enable_summary_tooltip', None)
    return view.summary(
        self,
        title=tree_view.Html.element(
            'div',
            [
                'ErrorInfo',
                controls.Label(self.tag, css_classes=['error_tag']),
            ],
            styles=dict(display='inline-block')
        ),
        enable_summary_tooltip=False,
        **kwargs
    )

  def _html_tree_view_content(self, **kwargs) -> tree_view.Html:
    del kwargs
    return controls.TabControl([
        controls.Tab(
            'description',
            tree_view.Html.element(
                'div',
                [
                    tree_view.Html.escape(utils.decolor(self.description))
                ],
                css_classes=['error_text']
            )
        ),
        controls.Tab(
            'stacktrace',
            tree_view.Html.element(
                'div',
                [
                    tree_view.Html.escape(utils.decolor(self.stacktrace))
                ],
                css_classes=['error_text']
            )
        )
    ]).to_html()

  @classmethod
  def _html_tree_view_css_styles(cls) -> List[str]:
    return super()._html_tree_view_css_styles() + [
        '''
        .error_tag {
          background-color: red;
          border-radius: 5px;
          padding: 5px;
          margin-left: 5px;
          color: white;
          font-size: small;
        }
        .error_text {
          display: block;
          white-space: pre-wrap;
          padding: 0.5em;
          margin-top: 0.15em;
          background-color: #f2f2f2;
        }
        '''
    ]

# Use the symbolic error info class as the default ErrorInfo implementation.
utils.ErrorInfo._IMPLEMENTATION = ErrorInfo  # pylint: disable=protected-access
