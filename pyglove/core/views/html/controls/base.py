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
"""Base for HTML controls."""

import abc
import contextlib
import inspect
import sys
from typing import Annotated, Any, Dict, Iterator, List, Optional, Union

from pyglove.core import utils
from pyglove.core.symbolic import object as pg_object
from pyglove.core.views.html import base

Html = base.Html


try:
  _notebook = sys.modules['IPython'].display
except Exception:  # pylint: disable=broad-except
  _notebook = None


class HtmlControl(pg_object.Object):
  """Base class for HTML controls."""

  id: Annotated[
      Optional[str],
      'Optional ID of the root element.'
  ] = None

  css_classes: Annotated[
      List[str],
      'Optional CSS classes of the root element.'
  ] = []

  styles: Annotated[
      Dict[str, Any],
      'Ad-hoc styles (vs. class-based) of the root element.'
  ] = {}

  interactive: Annotated[
      bool,
      (
          'Whether the control is interactive. If False, optimizations will be '
          'applied to reduce the size of the HTML.'
      )
  ] = False

  def _on_bound(self):
    super()._on_bound()
    self._rendered = False
    self._css_styles = []
    self._dynamic_injected_css = set()
    self._scripts = []

  def add_style(self, *css: str) -> 'HtmlControl':
    """Adds CSS styles to the HTML."""
    self._css_styles.extend(css)
    return self

  def add_script(self, *scripts: str) -> 'HtmlControl':
    self._scripts.extend(scripts)
    return self

  def to_html(self, **kwargs) -> Html:
    """Returns the HTML representation of the control."""
    self._rendered = True
    self._dynamic_injected_css = set()
    html = self._to_html(**kwargs)
    return html.add_style(*self._css_styles).add_script(*self._scripts)

  @abc.abstractmethod
  def _to_html(self, **kwargs) -> Html:
    """Returns the HTML representation of the control."""

  @classmethod
  @contextlib.contextmanager
  def track_scripts(cls) -> Iterator[List[str]]:
    del cls
    all_tracked = utils.thread_local_get(_TLS_TRACKED_SCRIPTS, [])
    current = []
    all_tracked.append(current)
    utils.thread_local_set(_TLS_TRACKED_SCRIPTS, all_tracked)
    try:
      yield current
    finally:
      all_tracked.pop(-1)
      if not all_tracked:
        utils.thread_local_del(_TLS_TRACKED_SCRIPTS)

  def _sync_members(self, **fields) -> None:
    """Synchronizes displayed values to members."""
    self.rebind(fields, skip_notification=True, raise_on_no_change=False)

  def _run_javascript(self, code: str, debug: bool = False) -> None:
    """Runs the given JavaScript code."""
    if not self.interactive:
      raise ValueError(
          f'Non-interactive control {self} cannot be updated. '
          'Please set `interactive=True` in the constructor.'
      )
    if not self._rendered:
      return

    code = inspect.cleandoc(code)
    if debug:
      print('RUN JAVSCRIPT:\n', code)
    if _notebook is not None:
      _notebook.display(_notebook.Javascript(code))

    # Track script execution.
    all_tracked = utils.thread_local_get(_TLS_TRACKED_SCRIPTS, [])
    for tracked in all_tracked:
      tracked.append(code)

  def _add_css_rules(self, css: str) -> None:
    if not self._rendered or not css or css in self._dynamic_injected_css:
      return
    self._run_javascript(
        f"""
        const style = document.createElement('style');
        style.type = 'text/css';
        style.textContent = "{Html.escape(css, javascript_str=True)}";
        document.head.appendChild(style);
        """
    )
    self._dynamic_injected_css.add(css)

  def _apply_css_rules(self, html: Html) -> None:
    self._add_css_rules(html.styles.content)

  def _insert_adjacent_html(
      self,
      element_selector_js: str,
      html: Html,
      var_name: str = 'elem',
      position: str = 'beforeend'
  ):
    self._run_javascript(
        f"""
        {element_selector_js}
        {var_name}.insertAdjacentHTML(
            "{position}",
            "{Html.escape(html, javascript_str=True).to_str(content_only=True)}"
        );
        """,
    )
    self._apply_css_rules(html)

  def element_id(self, child: Optional[str] = None) -> Optional[str]:
    """Returns the element id of this control or a child."""
    if self.id is not None:
      return self.id
    elif not self.interactive:
      return None
    elif child is None:
      return f'control-{id(self)}'
    else:
      return f'control-{id(self)}-{child}'

  def _update_content(
      self,
      content: Union[str, Html],
      child: Optional[str] = None,
  ) -> None:
    """Updates the content of the control."""
    if isinstance(content, str):
      return self._update_text(content, child)
    else:
      return self._update_inner_html(content, child)

  def _update_text(
      self,
      content: str,
      child: Optional[str] = None
  ) -> str:
    """Updates the content of the control."""
    self._run_javascript(
        f"""
        elem = document.getElementById("{self.element_id(child)}");
        elem.textContent = "{Html.escape(content, javascript_str=True)}";
        """
    )
    return content

  def _update_inner_html(
      self,
      html: base.Html,
      child: Optional[str] = None,
  ) -> base.Html:
    """Updates the inner HTML of the control."""
    self._run_javascript(
        f"""
        elem = document.getElementById("{self.element_id(child)}");
        elem.innerHTML = "{Html.escape(html, javascript_str=True).to_str(content_only=True)}";
        """
    )
    self._add_css_rules(html.styles.content)
    return html

  def _update_style(
      self,
      styles: Dict[str, Any],
      *,
      child: Optional[str] = None,
      updates_only: bool = False,
  ) -> Dict[str, Any]:
    """Updates the style of the control."""
    updated_styles = {} if updates_only else dict(self.styles or {})
    updated_styles.update(styles)
    self._run_javascript(
        f"""
        elem = document.getElementById("{self.element_id(child)}");
        elem.style = "{Html.style_str(updated_styles)}";
        """
    )
    return updated_styles

  def _update_property(
      self,
      name: str,
      value: str,
      child: Optional[str] = None,
  ) -> str:
    """Updates a property of the control."""
    self._run_javascript(
        f"""
        elem = document.getElementById("{self.element_id(child)}");
        elem.{name} = "{value}";
        """
    )
    return value

  def _add_css_class(
      self,
      css_class: str,
      child: Optional[str] = None,
  ) -> str:
    """Adds a CSS class to the control."""
    self._run_javascript(
        f"""
        elem = document.getElementById("{self.element_id(child)}");
        elem.classList.add("{css_class}");
        """
    )
    return css_class

  def _remove_css_class(
      self,
      css_class: str,
      child: Optional[str] = None,
  ) -> str:
    """Removes a CSS class from the control."""
    self._run_javascript(
        f"""
        elem = document.getElementById("{self.element_id(child)}");
        elem.classList.remove("{css_class}");
        """
    )
    return css_class


_TLS_TRACKED_SCRIPTS = '__tracked_scripts__'
