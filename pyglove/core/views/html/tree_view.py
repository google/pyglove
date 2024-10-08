# Copyright 2024 The Langfun Authors
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
"""HTML Tree View (The default view for PyGlove objects)."""

import inspect
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Union

from pyglove.core import object_utils
from pyglove.core.views.html import base


KeyPath = object_utils.KeyPath
Html = base.Html
HtmlView = base.HtmlView


# pytype: disable=annotation-type-mismatch


class HtmlTreeView(HtmlView):
  """HTML Tree View."""

  VIEW_ID = 'html-tree-view'

  class Extension(HtmlView.Extension):
    """The base class for extensions for HtmlTreeView."""

    def _html_tree_view_render(
        self,
        *,
        view: 'HtmlView',
        name: Optional[str],
        parent: Any,
        root_path: KeyPath,
        title: Union[str, Html, None] = None,
        special_keys: Optional[Sequence[Union[int, str]]] = None,
        include_keys: Optional[Iterable[Union[int, str]]] = None,
        exclude_keys: Optional[Iterable[Union[int, str]]] = None,
        filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
        highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        enable_summary: Optional[bool] = HtmlView.PresetArgValue(None),
        max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
        enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
        enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
        collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
        uncollapse: Union[
            Iterable[Union[KeyPath, str]], base.NodeFilter, None
        ] = HtmlView.PresetArgValue(None),
        **kwargs,
    ) -> Html:
      """Returns the topmost HTML representation of the object.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        title: The title of the summary.
        special_keys: The special keys to display (at the immediate child
          level).
        include_keys: The keys to include (at the immediate child level).
        exclude_keys: The keys to exclude (at the immediate child level).
        filter: A function with signature (path, value, parent) -> include
          to determine whether to include a field (at all levels).
        highlight: A function with signature (path, value, parent) -> bool
          to determine whether to highlight a field.
        lowlight: A function with signature (path, value, parent) -> bool
          to determine whether to lowlight a field.
        enable_summary: Whether to enable the summary. If None, summary will
          be enabled for complex types or when string exceeds
          `max_summary_len_for_str`.
        max_summary_len_for_str: The maximum length of the string to display.
        enable_summary_tooltip: Whether to enable the tooltip for the summary.
        enable_key_tooltip: Whether to enable the tooltip for the key.
        collapse_level: The level to collapse the tree
        uncollapse: A set of key paths for the nodes to uncollapse. or a
          function with signature (path, value, parent) -> bool to determine
          whether to uncollapse a node.
        **kwargs: Additional keyword arguments passed from `pg.to_html`.

      Returns:
        The rendered HTML.
      """
      return view.render(
          self,
          name=name,
          parent=parent,
          root_path=root_path,
          title=title,
          special_keys=special_keys,
          include_keys=include_keys,
          exclude_keys=exclude_keys,
          filter=filter,
          highlight=highlight,
          lowlight=lowlight,
          enable_summary=enable_summary,
          max_summary_len_for_str=max_summary_len_for_str,
          enable_summary_tooltip=enable_summary_tooltip,
          enable_key_tooltip=enable_key_tooltip,
          collapse_level=collapse_level,
          uncollapse=uncollapse,
          **kwargs
      )

    def _html_tree_view_summary(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str],
        parent: Any,
        root_path: KeyPath,
        css_class: Optional[Sequence[str]] = None,
        title: Union[str, Html, None] = None,
        enable_summary: Optional[bool] = HtmlView.PresetArgValue(None),
        max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
        enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
        **kwargs,
    ) -> Optional[Html]:
      """Returns the HTML representation of the object.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        css_class: The CSS classes to add to the root element
        title: The title of the summary.
        enable_summary: Whether to enable the summary. If None, summary will
          be enabled for complex types or when string exceeds
          `max_summary_len_for_str`.
        max_summary_len_for_str: The maximum length of the string to display.
        enable_summary_tooltip: Whether to enable the tooltip for the summary.
        **kwargs: Additional keyword arguments passed from `pg.to_html`. These
          arguments may be handled by the user logic but not the general
          HtmlTreeView.

      Returns:
        An optional HTML object representing the summary of the object. If None,
        the summary will be hidden.
      """
      return view.summary(
          self,
          name=name,
          parent=parent,
          root_path=root_path,
          css_class=css_class,
          title=title,
          enable_summary=enable_summary,
          max_summary_len_for_str=max_summary_len_for_str,
          enable_summary_tooltip=enable_summary_tooltip,
          **kwargs,
      )

    def _html_tree_view_content(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str],
        parent: Any,
        root_path: KeyPath,
        special_keys: Optional[Sequence[Union[int, str]]] = None,
        include_keys: Optional[Iterable[Union[int, str]]] = None,
        exclude_keys: Optional[Iterable[Union[int, str]]] = None,
        filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
        highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
        enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
        enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
        collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
        uncollapse: Union[
            Iterable[Union[KeyPath, str]], base.NodeFilter, None
        ] = HtmlView.PresetArgValue(None),
        **kwargs,
        ) -> Html:
      """Returns the main content for the object.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        special_keys: The special keys to display (at the immediate child
          level).
        include_keys: The keys to include (at the immediate child level).
        exclude_keys: The keys to exclude (at the immediate child level).
        filter: A function with signature (path, value, parent) -> include
          to determine whether to include a field (at all levels).
        highlight: A function with signature (path, value, parent) -> bool
          to determine whether to highlight a field (at all levels).
        lowlight: A function with signature (path, value, parent) -> bool
          to determine whether to lowlight a field (at all levels).
        max_summary_len_for_str: The maximum length of the string to display.
        enable_summary_tooltip: Whether to enable the tooltip for the summary.
        enable_key_tooltip: Whether to enable the key tooltip.
        collapse_level: The level to collapse the tree
        uncollapse: The paths to uncollapse.
        **kwargs: Additional keyword arguments passed from `pg.to_html`. These
          arguments may be handled by the user logic but not the general
          HtmlTreeView.

      Returns:
        The rendered HTML as the main content of the object.
      """
      return view.content(
          self,
          name=name,
          parent=parent,
          root_path=root_path,
          special_keys=special_keys,
          include_keys=include_keys,
          exclude_keys=exclude_keys,
          filter=filter,
          highlight=highlight,
          lowlight=lowlight,
          max_summary_len_for_str=max_summary_len_for_str,
          enable_summary_tooltip=enable_summary_tooltip,
          enable_key_tooltip=enable_key_tooltip,
          collapse_level=collapse_level,
          uncollapse=uncollapse,
          **kwargs
      )

    def _html_tree_view_tooltip(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str],
        parent: Any,
        root_path: KeyPath,
        content: Union[str, Html, None] = None,
        **kwargs,
    ) -> Optional[Html]:
      """Returns the tooltip for the object.

      Args:
        view: The view to render the object.
        name: The referenced name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        content: Custom content to display in the tooltip.
        **kwargs: Additional keyword arguments passed from `pg.to_html`. These
          arguments may be handled by the user logic but not the general
          HtmlTreeView.

      Returns:
        An optional HTML object representing the tooltip of the object. If None,
        the tooltip will be hidden.
      """
      return view.tooltip(
          value=self, name=name, parent=parent, root_path=root_path,
          content=content, **kwargs
      )

  @HtmlView.extension_method('_html_tree_view_render')
  def render(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      title: Union[str, Html, None] = None,
      special_keys: Optional[Sequence[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
      highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      enable_summary: Optional[bool] = HtmlView.PresetArgValue(None),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
      collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
      uncollapse: Union[
          Iterable[Union[KeyPath, str]], base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
      **kwargs
  ) -> Html:
    """Renders the entire HTML tree view for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_class: The CSS classes to add to the root element
      title: The title of the summary.
      special_keys: The special keys to display (at the immediate child level).
      include_keys: The keys to include (at the immediate child level).
      exclude_keys: The keys to exclude (at the immediate child level).
      filter: A function with signature (path, value, parent) -> include
        to determine whether to include a field (at all levels).
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight a field.
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight a field.
      enable_summary: Whether to enable the summary. If None, summary will
        be enabled for complex types or when string exceeds
        `max_summary_len_for_str`.
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the tooltip for the summary.
      enable_key_tooltip: Whether to enable the key tooltip.
      collapse_level: The level to collapse the tree.
      uncollapse: The paths to uncollapse.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML.
    """
    root_path = root_path or KeyPath()
    uncollapse = self.normalize_uncollapse(uncollapse)
    summary = self.summary(
        value,
        name=name,
        parent=parent,
        root_path=root_path,
        title=title,
        enable_summary=enable_summary,
        enable_summary_tooltip=enable_summary_tooltip,
        max_summary_len_for_str=max_summary_len_for_str,
        **kwargs,
    )
    content = self.content(
        value,
        name=name,
        parent=parent,
        root_path=root_path,
        css_class=css_class if summary is None else None,
        filter=filter,
        special_keys=special_keys,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        enable_key_tooltip=enable_key_tooltip,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        **kwargs,
    )
    if summary is None:
      content = Html.from_value(content)
      assert content is not None
      return content

    collapse_view = self.should_collapse(
        value, parent=parent, root_path=root_path,
        collapse_level=collapse_level, uncollapse=uncollapse,
    )
    return Html.element(
        'details',
        [
            summary,
            content,
        ],
        options=[None if collapse_view else 'open'],
        css_class=[
            'pyglove',
            self.css_class_name(value),
        ] + (css_class or []),
    ).add_style(
        """
        /* Value details styles. */
        details.pyglove {
          border: 1px solid #aaa;
          border-radius: 4px;
          padding: 0.5em 0.5em 0;
          margin: 0.1em 0;
        }
        details.pyglove.special_value {
          margin-bottom: 0.75em;
        }
        details.pyglove[open] {
          padding: 0.5em 0.5em 0.5em;
        }
        .highlight {
          background-color: Mark;
        }
        .lowlight {
          opacity: 0.2;
        }
        """
    )

  def normalize_uncollapse(
      self,
      uncollapse: Union[
          Iterable[Union[KeyPath, str]], base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
  ) -> Union[None, Set[KeyPath], base.NodeFilter]:
    """Normalize the uncollapse argument."""
    if uncollapse is None:
      return None
    elif isinstance(uncollapse, set) or callable(uncollapse):
      return uncollapse
    else:
      expanded = set()
      for path in uncollapse:
        path = object_utils.KeyPath.from_value(path)
        expanded.add(path)
        while path:
          expanded.add(path.parent)
          path = path.parent
      return expanded

  def should_collapse(
      self,
      value: Any,
      root_path: KeyPath,
      parent: Any,
      collapse_level: Optional[int] = 0,
      uncollapse: Union[
          Set[Union[KeyPath, str]], base.NodeFilter, None
      ] = None,
  ) -> bool:
    """Returns whether the object should be collapsed."""
    if collapse_level is None or root_path.depth < collapse_level:
      return False
    if uncollapse is None:
      return True
    elif callable(uncollapse):
      return not uncollapse(root_path, value, parent)
    else:
      return root_path not in uncollapse

  def needs_summary(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      title: Union[str, Html, None] = None,
      enable_summary: Optional[bool] = HtmlView.PresetArgValue(True),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
  ) -> bool:
    """Returns whether the object needs a summary."""
    del parent
    if enable_summary is None:
      if name is not None or title is not None or not (
          isinstance(value, (int, float, bool, type(None)))
          or (
              isinstance(value, str)
              and len(value) <= max_summary_len_for_str
          )
      ):
        return True
    return enable_summary

  @HtmlView.extension_method('_html_tree_view_summary')
  def summary(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      title: Union[str, Html, None] = None,
      enable_summary: Optional[bool] = HtmlView.PresetArgValue(None),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      **kwargs
  ) -> Optional[Html]:
    """Renders a summary for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_class: The CSS classes to add to the root element
      title: The title of the summary.
      enable_summary: Whether to enable the summary. If None, summary will
        be enabled for complex types or when string exceeds
        `max_summary_len_for_str`.
      enable_summary_tooltip: Whether to enable the tooltip for the summary.
      max_summary_len_for_str: The maximum length of the string to display.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      An optional HTML object representing the summary of the value. If None,
      the summary will be hidden.
    """
    if not self.needs_summary(
        value,
        name=name,
        parent=parent,
        title=title,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary=enable_summary,
    ):
      return None

    def make_title(value: Any):
      if isinstance(value, str):
        if len(value) > max_summary_len_for_str:
          value = value[:max_summary_len_for_str] + '...'
        return Html.escape(repr(value))
      return f'{type(value).__name__}(...)'

    return Html.element(
        'summary',
        [
            # Summary name.
            lambda: Html.element(  # pylint: disable=g-long-ternary
                'div',
                [
                    name,
                ],
                css_class=['summary_name']
            ) if name else None,

            # Summary title
            Html.element(
                'div',
                [
                    title or make_title(value),
                ],
                css_class=['summary_title']
            ),

            # Tooltip.
            lambda: self.tooltip(   # pylint: disable=g-long-ternary
                value,
                name=name,
                parent=parent,
                root_path=root_path,
                **kwargs,
            ) if enable_summary_tooltip else None,
        ],
        css_class=css_class,
    ).add_style(
        """
        /* Summary styles. */
        details.pyglove summary {
          font-weight: bold;
          margin: -0.5em -0.5em 0;
          padding: 0.5em;
        }
        .summary_name {
          display: inline;
          padding: 0 5px;
        }
        .summary_title {
          display: inline;
        }
        .summary_name + div.summary_title {
          display: inline;
          color: #aaa;
        }
        .summary_title:hover + span.tooltip {
          visibility: visible;
        }
        /* Type-specific styles. */
        .pyglove.str .summary_title {
          color: darkred;
          font-style: italic;
        }
        """
    )

  # NOTE(daiyip)" `object_key`` does not have a corresponding extension
  # method in `HtmlTreeView.Extension`, because the rendering of the key is not
  # delegated to `HtmlTreeView.Extension`.
  def object_key(
      self,
      key: Union[str, int],
      *,
      name: Optional[str] = None,
      parent: Any,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      key_color: Optional[str] = None,
      enable_tooltip: bool = HtmlView.PresetArgValue(True),
      **kwargs
  ) -> Html:
    """Renders the key for the value.

    Args:
      key: The key of the value.
      name: The name of the value.
      parent: The parent value of the key.
      root_path: The root path of the value.
      css_class: The CSS classes to add to the root element.
      key_color: The color of the key.
      enable_tooltip: Whether to enable the tooltip.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML as the key of the value.
    """
    return (
        # Key span.
        Html.element(
            'span',
            [
                str(key),
            ],
            css_class=['object_key'] + (css_class or []),
            style=dict(
                color=key_color,
            )
        ) + (
            # Tooltip if enabled.
            lambda: self.tooltip(  # pylint: disable=g-long-ternary
                value=root_path,
                root_path=root_path,
                name=name,
                parent=parent,
                **kwargs
            ) if enable_tooltip else None
        )
    ).add_style(
        """
        /* Object key styles. */
        .object_key {
          margin-right: 0.25em;
        }
        .object_key:hover + .tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        .complex_value .object_key{
          color: gray;
          border: 1px solid lightgray;
          background-color: ButtonFace;
          border-radius: 0.2em;
          padding: 0.3em;
        }
        .complex_value.list .object_key{
          border: 0;
          color: lightgray;
          background-color: transparent;
          border-radius: 0;
          padding: 0;
        }
        .complex_value.list .object_key::before{
          content: '[';
        }
        .complex_value.list .object_key::after{
          content: ']';
        }
        """
    )

  @HtmlView.extension_method('_html_tree_view_content')
  def content(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      special_keys: Optional[Sequence[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
      highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
      collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
      uncollapse: Union[
          Iterable[Union[KeyPath, str]], base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
      **kwargs
  ) -> Html:
    """Renders the main content for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_class: Additional CSS classes for root element.
      special_keys: The special keys to display (at the immediate child level).
      include_keys: The keys to include (at the immediate child level).
      exclude_keys: The keys to exclude (at the immediate child level).
      filter: A function with signature (path, value, parent) -> include
        to determine whether to include a field (at all levels).
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight a field (at all levels).
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight a field (at all levels).
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      enable_key_tooltip: Whether to enable the key tooltip.
      collapse_level: The level of the tree to collapse.
      uncollapse: The keys to uncollapse.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML as the main content of the value.
    """
    if isinstance(value, (tuple, list)):
      items = {i: v for i, v in enumerate(value)}
    elif isinstance(value, dict):
      items = value
    else:
      return self.simple_value(
          value, name=name, parent=parent, root_path=root_path,
          css_class=css_class, max_summary_len_for_str=max_summary_len_for_str
      )
    return self.complex_value(
        items,
        name=name,
        parent=value,
        root_path=root_path or KeyPath(),
        css_class=css_class,
        special_keys=special_keys,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        filter=filter,
        highlight=highlight,
        lowlight=lowlight,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        enable_key_tooltip=enable_key_tooltip,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        **kwargs,
    )

  def simple_value(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
  ) -> Html:
    """Renders a simple value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_class: Additional CSS classes for root span.
      max_summary_len_for_str: The maximum length of the string to display.

    Returns:
      The rendered HTML as the simple value.
    """
    del name, parent, root_path
    def value_repr() -> str:
      if isinstance(value, str):
        if len(value) < max_summary_len_for_str:
          return repr(value)
      return object_utils.format(
          value,
          compact=False, verbose=False, hide_default_values=True,
          python_format=True, use_inferred=True,
          max_bytes_len=64, max_str_len=256,
      )
    return Html.element(
        'span',
        [
            Html.escape(value_repr),
        ],
        css_class=['simple_value', self.css_class_name(value)] + (
            css_class or []
        ),
    ).add_style(
        """
        /* Simple value styles. */
        .simple_value {
          color: blue;
          display: inline-block;
          white-space: pre-wrap;
          padding: 0.2em;
          margin-top: 0.15em;
        }
        .simple_value.str {
          color: darkred;
          font-style: italic;
        }
        .simple_value.int, .simple_value.float {
          color: darkblue;
        }
        """
    )

  def complex_value(
      self,
      kv: Dict[Union[int, str], Any],
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      special_keys: Optional[Sequence[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
      highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
      collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
      uncollapse: Union[
          Iterable[Union[KeyPath, str]], base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
      **kwargs,
  ) -> Html:
    """Renders a list of key-value pairs.

    Args:
      kv: The key-value pairs to render.
      name: The name of the value.
      parent: The parent value of the key-value pairs.
      root_path: The root path of the value.
      css_class: Additional CSS classes for root div.
      special_keys: The special keys to display (at the immediate child level).
      include_keys: The keys to include (at the immediate child level).
      exclude_keys: The keys to exclude (at the immediate child level).
      filter: A function with signature (path, value, parent) -> include
        to determine whether to include.
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight.
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight.
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      enable_key_tooltip: Whether to enable the key tooltip.
      collapse_level: The level of the tree to collapse.
      uncollapse: The keys to uncollapse.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML as the key-value pairs.
    """
    del name
    special_keys = special_keys or []
    include_keys = set(include_keys or [])
    exclude_keys = set(exclude_keys or [])
    uncollapse = self.normalize_uncollapse(uncollapse)

    s = Html()
    if kv:
      include_keys = include_keys or set(kv.keys())
      if filter is not None:
        include_keys -= set(
            k for k, v in kv.items()
            if not filter(root_path + k, v, parent)
        )
      if exclude_keys:
        include_keys -= exclude_keys

      if special_keys:
        for k in special_keys:
          if k in include_keys and k in kv:
            child_path = root_path + k
            v = kv[k]

            child_css_class = [
                'special_value',
                (
                    'highlight' if highlight
                    and highlight(child_path, v, parent) else None
                ),
                (
                    'lowlight' if lowlight
                    and lowlight(child_path, v, parent) else None
                )
            ]
            s.write(
                self.render(
                    value=v,
                    name=k,
                    parent=v,
                    root_path=child_path,
                    css_class=child_css_class,
                    filter=filter,
                    special_keys=None,
                    include_keys=None,
                    exclude_keys=None,
                    highlight=highlight,
                    lowlight=lowlight,
                    max_summary_len_for_str=max_summary_len_for_str,
                    enable_summary_tooltip=enable_summary_tooltip,
                    enable_key_tooltip=enable_key_tooltip,
                    collapse_level=collapse_level,
                    uncollapse=uncollapse,
                    **kwargs
                )
            )
            include_keys.remove(k)

      if include_keys:
        s.write('<table>')
        for k, v in kv.items():
          if k not in include_keys:
            continue
          child_path = root_path + k
          key = self.object_key(
              key=k, parent=v, root_path=child_path,
              enable_tooltip=enable_key_tooltip,
          )
          child_css_class = [
              (
                  'highlight' if highlight and highlight(child_path, v, parent)
                  else None
              ),
              (
                  'lowlight' if lowlight and lowlight(child_path, v, parent)
                  else None
              )
          ]
          value = self.render(
              value=v,
              name=None,
              parent=v,
              root_path=child_path,
              css_class=child_css_class,
              special_keys=None,
              include_keys=None,
              exclude_keys=None,
              filter=filter,
              highlight=highlight,
              lowlight=lowlight,
              max_summary_len_for_str=max_summary_len_for_str,
              enable_summary_tooltip=enable_summary_tooltip,
              enable_key_tooltip=enable_key_tooltip,
              collapse_level=collapse_level,
              uncollapse=uncollapse,
              **kwargs,
          )
          s.write(
              Html.element(
                  'tr',
                  [
                      '<td>', key, '</td>',
                      '<td>', value, '</td>',
                  ],
              )
          )
        s.write('</table>')
    else:
      s.write(Html.element('span', css_class=['empty_container']))
    return Html.element(
        'div',
        [s],
        css_class=['complex_value', self.css_class_name(parent)] + (
            css_class or []
        ),
    ).add_style(
        """
        /* Complex value styles. */
        span.empty_container::before {
            content: '(empty)';
            font-style: italic;
            margin-left: 0.5em;
            color: #aaa;
        }
        """
    )

  @HtmlView.extension_method('_html_tree_view_tooltip')
  def tooltip(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Optional[Sequence[str]] = None,
      content: Union[str, Html, None] = HtmlView.PresetArgValue(None),
      **kwargs
  ) -> Html:
    """Renders a tooltip for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent value of the key-value pairs.
      root_path: The root path of the value.
      css_class: Additional CSS classes for the tooltip span.
      content: The content of the tooltip. If None, the formatted value will be
        used as the content.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML as the tooltip of the value.
    """
    del name, parent
    if content is None:
      content = Html.escape(
          object_utils.format(
              value,
              root_path=root_path,
              compact=False,
              verbose=False,
              python_format=True,
              max_bytes_len=64,
              max_str_len=256,
              **kwargs
          )
      )
    return Html.element(
        'span',
        [content],
        css_class=['tooltip'] + (css_class or []),
    ).add_style(
        """
        /* Tooltip styles. */
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
        """
    )

  @staticmethod
  def css_class_name(value: Any) -> str:
    """Returns the CSS class name for the value."""
    value = value if inspect.isclass(value) else type(value)
    class_name = value.__name__
    return _REGEX_CAMEL_TO_SNAKE.sub('-', class_name).lower()


_REGEX_CAMEL_TO_SNAKE = re.compile(r'(?<!^)(?=[A-Z])')

# pytype: enable=annotation-type-mismatch
