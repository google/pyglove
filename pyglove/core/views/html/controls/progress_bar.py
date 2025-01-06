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
"""Progress bar control."""

import functools
from typing import Annotated, List, Optional, Union

from pyglove.core import utils
from pyglove.core.symbolic import object as pg_object
# pylint: disable=g-importing-member
from pyglove.core.views.html.base import Html
from pyglove.core.views.html.controls.base import HtmlControl
from pyglove.core.views.html.controls.label import Label
# pylint: enable=g-importing-member


@pg_object.use_init_args(
    ['name', 'value', 'id', 'css_classes', 'styles']
)
class SubProgress(HtmlControl):
  """A sub progress bar control."""

  name: Annotated[
      str, 'The name of the sub progress bar.'
  ]

  value: Annotated[
      int, 'The value of the sub progress bar.'
  ] = 0

  interactive = True

  def _on_parent_change(self, *args, **kwargs):
    super()._on_parent_change(*args, **kwargs)  # pytype: disable=attribute-error
    self.__dict__.pop('parent', None)

  @functools.cached_property
  def parent(self) -> Optional['ProgressBar']:
    """Returns the parent progress bar."""
    return self.sym_ancestor(
        lambda x: isinstance(x, ProgressBar)
    )

  @property
  def total(self) -> Optional[int]:
    """Returns the total number of the sub progress bar."""
    assert self.parent is not None
    return self.parent.total

  @property
  def width(self) -> Optional[str]:
    """Returns the width of the sub progress bar."""
    if self.total is None:
      return None
    return f'{self.value / self.total:.0%}'

  def _to_html(self, **kwargs) -> Html:
    styles = self.styles.copy()
    styles.update(width=self.width)
    return Html.element(
        'div',
        [],
        id=self.element_id(),
        styles=styles,
        css_classes=['sub-progress', utils.camel_to_snake(self.name, '-')]
        + self.css_classes,
    )

  def increment(self, delta: int = 1):
    """Increments the value of the sub progress bar."""
    self.update(self.value + delta)

  def update(self, value: Optional[int] = None):
    if value is not None:
      self.rebind(
          value=value, skip_notification=True, raise_on_no_change=False
      )
    self._update_style(dict(width=self.width))
    self.parent.update()


@pg_object.use_init_args(
    ['subprogresses', 'total', 'id', 'css_classes', 'styles']
)
class ProgressBar(HtmlControl):
  """A progress bar control."""

  subprogresses: Annotated[
      List[SubProgress],
      'The sub progress bars of the progress bar.'
  ]

  total: Annotated[
      Optional[int],
      (
          'The total number of steps represented by the progress bar.'
          'If None, the progress steps are not determined.'
      )
  ] = None

  interactive = True

  def _on_bound(self):
    super()._on_bound()
    self._progress_label = Label(
        text=self._progress_text(),
        tooltip=self._progress_tooltip(),
        css_classes=['progress-label'],
        interactive=True,
    )

  def _progress_text(self) -> Union[str, Html]:
    completed = sum(s.value for s in self.subprogresses)
    if self.total is None:
      return 'n/a'
    complete_rate = completed / self.total
    return f'{complete_rate: .1%} ({completed}/{self.total})'

  def _progress_tooltip(self) -> str:
    if self.total is None:
      return 'Not started'
    assert self.total > 0
    return '\n'.join([
        f'{s.name}: {s.value / self.total:.1%} ({s.value}/{self.total})'
        for s in self.subprogresses
    ])

  def _to_html(self, **kwargs) -> Html:
    return Html.element(
        'div',
        [
            Html.element('div', self.subprogresses, css_classes=['shade']),
            self._progress_label,
        ],
        css_classes=['progress-bar']
    ).add_style(
        """
        .progress-bar {
            display: inline-block;
        }
        .progress-bar .shade {
          position: relative;
          background-color: #EEE;
          display: inline-flex;
          margin: 10px;
          width: 100px; 
          height: 5px;
          border-radius: 5px; 
          overflow: hidden;
        }
        .progress-bar .sub-progress {
          background-color: dodgerblue;
          height: 10px;
          width: 0%;
          border-radius: 0px;
        }
        """
    )

  def update(self, total: Optional[int] = None):
    if total is not None:
      assert self.total is None
      self._sync_members(total=total)
    self._progress_label.update(
        self._progress_text(), tooltip=self._progress_tooltip()
    )

  def __getitem__(self, name: str) -> SubProgress:
    for sub in self.subprogresses:
      if sub.name == name:
        return sub
    raise KeyError(f'Sub progress bar {name!r} not found.')
