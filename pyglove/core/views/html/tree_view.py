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
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

from pyglove.core import object_utils
from pyglove.core.views.html import base


KeyPath = object_utils.KeyPath
KeyPathSet = object_utils.KeyPathSet
Html = base.Html
HtmlView = base.HtmlView


# pytype: disable=annotation-type-mismatch


class HtmlTreeView(HtmlView):
  """HTML Tree View."""

  VIEW_ID = 'html-tree-view'

  class Extension(HtmlView.Extension):
    """The base class for extensions for HtmlTreeView."""

    #
    # Default extension-level rendering options overrides.
    #

    def _html_tree_view_special_keys(self) -> Sequence[str]:
      """Returns the special keys to display (at the immediate child level)."""
      return []

    def _html_tree_view_include_keys(self) -> Optional[Sequence[str]]:
      """Returns the keys to include (at the immediate child level)."""
      return None

    def _html_tree_view_exclude_keys(self) -> Sequence[str]:
      """Returns the keys to include (at the immediate child level)."""
      return []

    def _html_tree_view_uncollapse_level(self) -> Optional[int]:
      """Returns the level of the subtree to uncollapse.

      Returns:
        The level of subtree to uncollapse. If None, the subtree will be fully
        expanded. Please note that the uncollapsed subtree will show only when
        current node is uncollapsed.
      """
      return 1

    def _html_tree_view_uncollapse(self) -> KeyPathSet:
      """Returns the node paths (relative to current node) to uncollapse."""
      return KeyPathSet()

    #
    # Default behavior overrides.
    #

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
        collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
        uncollapse: Union[
            KeyPathSet, base.NodeFilter, None
        ] = HtmlView.PresetArgValue(None),
        filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
        highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        enable_summary: Optional[bool] = HtmlView.PresetArgValue(None),
        max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
        enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
        enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
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
        collapse_level: The level to collapse the tree (relative to this node).
        uncollapse: A key path set (relative to root_path) for the nodes to
          uncollapse. or a function with signature (path, value, parent) -> bool
          to filter nodes to uncollapse.
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
        collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
        uncollapse: Union[
            KeyPathSet, base.NodeFilter, None
        ] = HtmlView.PresetArgValue(None),
        filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
        highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
        max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
        enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
        enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
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
        collapse_level: The level to collapse the tree (relative to this node).
        uncollapse: A key path set (relative to root_path) for the nodes to
          uncollapse. or a function with signature (path, value, parent) -> bool
          to filter nodes to uncollapse.
        filter: A function with signature (path, value, parent) -> include
          to determine whether to include a field (at all levels).
        highlight: A function with signature (path, value, parent) -> bool
          to determine whether to highlight a field (at all levels).
        lowlight: A function with signature (path, value, parent) -> bool
          to determine whether to lowlight a field (at all levels).
        max_summary_len_for_str: The maximum length of the string to display.
        enable_summary_tooltip: Whether to enable the tooltip for the summary.
        enable_key_tooltip: Whether to enable the key tooltip.
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
      title: Union[str, Html, None] = None,
      special_keys: Optional[Sequence[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
      uncollapse: Union[
          KeyPathSet, base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
      filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
      highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      enable_summary: Optional[bool] = HtmlView.PresetArgValue(None),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
      **kwargs
  ) -> Html:
    """Renders the entire HTML tree view for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      title: The title of the summary.
      special_keys: The special keys to display (at the immediate child level).
      include_keys: The keys to include (at the immediate child level).
      exclude_keys: The keys to exclude (at the immediate child level).
      collapse_level: The level to collapse the tree (relative to this node).
      uncollapse: A key path set (relative to root_path) for the nodes to
        uncollapse. or a function with signature (path, value, parent) -> bool
        to filter nodes to uncollapse.
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
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML.
    """
    root_path = root_path or KeyPath()
    uncollapse = self.init_uncollapse(uncollapse)

    child_collapse_level = collapse_level
    if isinstance(value, HtmlTreeView.Extension):
      subtree_uncollapse_level = value._html_tree_view_uncollapse_level()    # pylint: disable=protected-access

      # If the extension has child levels to uncollapse, honor them above the
      # collapse level passed from the root. However, we can see the
      # uncollapsed extension subtree only when the extension's parent node is
      # uncollapsed.
      child_collapse_level = self.max_collapse_level(
          collapse_level, subtree_uncollapse_level, root_path
      )
      if not callable(uncollapse):
        extension_uncollapse = value._html_tree_view_uncollapse().copy()  # pylint: disable=protected-access
        if extension_uncollapse:
          extension_uncollapse.rebase(root_path)
          uncollapse = uncollapse.union(extension_uncollapse)

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
        filter=filter,
        special_keys=special_keys,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        collapse_level=child_collapse_level,
        uncollapse=uncollapse,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        enable_key_tooltip=enable_key_tooltip,
        **kwargs,
    )
    extension_style = (
        value._html_style() if isinstance(value, HtmlView.Extension) else []  # pylint: disable=protected-access
    )

    if summary is None:
      content = Html.from_value(content)
      assert content is not None
      content.add_style(*extension_style)
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
        ],
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
        """,
        *extension_style,
    )

  def init_uncollapse(
      self,
      uncollapse: Union[
          Iterable[Union[KeyPath, str]], base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
  ) -> Union[KeyPathSet, base.NodeFilter]:
    """Normalize the uncollapse argument."""
    if uncollapse is None:
      return KeyPathSet()
    elif callable(uncollapse):
      return uncollapse
    else:
      return KeyPathSet.from_value(uncollapse, include_intermediate=True)

  def should_collapse(
      self,
      value: Any,
      root_path: KeyPath,
      parent: Any,
      collapse_level: Optional[int] = 0,
      uncollapse: Union[KeyPathSet, base.NodeFilter] = None,
  ) -> bool:
    """Returns whether the object should be collapsed."""
    if collapse_level is None or root_path.depth < collapse_level:
      return False
    if callable(uncollapse):
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
      css_class: Union[str, Sequence[str], None] = None,
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
      css_class: Additional CSS classes to add to the HTML element.
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
            css_class=[
                'object_key',
                type(key).__name__,
                css_class,
            ],
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
        .object_key.str {
          color: gray;
          border: 1px solid lightgray;
          background-color: ButtonFace;
          border-radius: 0.2em;
          padding: 0.3em;
        }
        .object_key.int::before{
          content: '[';
        }
        .object_key.int::after{
          content: ']';
        }
        .object_key.int{
          border: 0;
          color: lightgray;
          background-color: transparent;
          border-radius: 0;
          padding: 0;
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
      special_keys: Optional[Sequence[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
      uncollapse: Union[
          KeyPathSet, base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
      filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
      highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
      **kwargs
  ) -> Html:
    """Renders the main content for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      special_keys: The special keys to display (at the immediate child level).
      include_keys: The keys to include (at the immediate child level).
      exclude_keys: The keys to exclude (at the immediate child level).
      collapse_level: The level to collapse the tree (relative to this node).
      uncollapse: A key path set (relative to root_path) for the nodes to
        uncollapse. or a function with signature (path, value, parent) -> bool
        to filter nodes to uncollapse.
      filter: A function with signature (path, value, parent) -> include
        to determine whether to include a field (at all levels).
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight a field (at all levels).
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight a field (at all levels).
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      enable_key_tooltip: Whether to enable the key tooltip.
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
          max_summary_len_for_str=max_summary_len_for_str
      )
    return self.complex_value(
        items,
        name=name,
        parent=value,
        root_path=root_path or KeyPath(),
        special_keys=special_keys,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        filter=filter,
        highlight=highlight,
        lowlight=lowlight,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        enable_key_tooltip=enable_key_tooltip,
        **kwargs,
    )

  def simple_value(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_class: Union[str, Sequence[str], None] = None,
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
  ) -> Html:
    """Renders a simple value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_class: Additional CSS classes to add to the HTML element.
      max_summary_len_for_str: The maximum length of the string to display.

    Returns:
      The rendered HTML as the simple value.
    """
    del name, parent, root_path
    def value_repr() -> str:
      if isinstance(value, str):
        if len(value) < max_summary_len_for_str:
          return repr(value)
        else:
          return value
      return object_utils.format(
          value,
          compact=False, verbose=False, hide_default_values=True,
          python_format=True, use_inferred=True,
          max_bytes_len=64,
      )
    return Html.element(
        'span',
        [
            Html.escape(value_repr),
        ],
        css_class=[
            'simple_value',
            self.css_class_name(value),
            css_class,
        ],
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
      parent: Any,
      root_path: KeyPath,
      name: Optional[str] = None,
      css_class: Union[str, Sequence[str], None] = None,
      render_key_fn: Optional[Callable[..., Html]] = None,
      render_value_fn: Optional[Callable[..., Html]] = None,
      special_keys: Optional[Sequence[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      collapse_level: Optional[int] = HtmlView.PresetArgValue(1),
      uncollapse: Union[
          KeyPathSet, base.NodeFilter, None
      ] = HtmlView.PresetArgValue(None),
      filter: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),  # pylint: disable=redefined-builtin
      highlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      lowlight: Optional[base.NodeFilter] = HtmlView.PresetArgValue(None),
      max_summary_len_for_str: int = HtmlView.PresetArgValue(40),
      enable_summary_tooltip: bool = HtmlView.PresetArgValue(True),
      enable_key_tooltip: bool = HtmlView.PresetArgValue(True),
      **kwargs,
  ) -> Html:
    """Renders a list of key-value pairs.

    Args:
      kv: The key-value pairs to render.
      parent: The parent value of the key-value pairs.
      root_path: The root path of the value.
      name: The name of the value.
      css_class: Additional CSS classes to add to the HTML element.
      render_key_fn: A function to render the key. The function has the
        same signature as `HtmlTreeView.object_key`.
        If None, `HtmlTreeView.object_key` will be used to render the key.
      render_value_fn: A function to render the value. The function has the
        same signature as `HtmlTreeView.render`.
        If None, `HtmlTreeView.render` will be used to render child value.
      special_keys: The special keys to display (at the immediate child level).
      include_keys: The keys to include (at the immediate child level).
      exclude_keys: The keys to exclude (at the immediate child level).
      collapse_level: The level to collapse the tree (relative to this node).
      uncollapse: A key path set (relative to root_path) for the nodes to
        uncollapse. or a function with signature (path, value, parent) -> bool
        to filter nodes to uncollapse.
      filter: A function with signature (path, value, parent) -> include
        to determine whether to include.
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight.
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight.
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      enable_key_tooltip: Whether to enable the key tooltip.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML as the key-value pairs.
    """
    del name
    root_path = root_path or KeyPath()
    uncollapse = self.init_uncollapse(uncollapse)

    if isinstance(parent, HtmlTreeView.Extension):
      special_keys = special_keys or parent._html_tree_view_special_keys()  # pylint: disable=protected-access
      include_keys = include_keys or parent._html_tree_view_include_keys()  # pylint: disable=protected-access
      exclude_keys = exclude_keys or parent._html_tree_view_exclude_keys()  # pylint: disable=protected-access

    special_keys = special_keys or []
    include_keys = set(include_keys or [])
    exclude_keys = set(exclude_keys or [])

    render_key_fn = render_key_fn or self.object_key
    render_value_fn = render_value_fn or self.render

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
            s.write(
                Html.element(
                    'div',
                    [
                        render_value_fn(
                            value=v,
                            name=k,
                            parent=parent,
                            root_path=child_path,
                            filter=filter,
                            special_keys=None,
                            include_keys=None,
                            exclude_keys=None,
                            collapse_level=collapse_level,
                            uncollapse=uncollapse,
                            highlight=highlight,
                            lowlight=lowlight,
                            max_summary_len_for_str=max_summary_len_for_str,
                            enable_summary_tooltip=enable_summary_tooltip,
                            enable_key_tooltip=enable_key_tooltip,
                            **kwargs
                        )
                    ],
                    css_class=[
                        'special_value',
                        (
                            'highlight' if highlight
                            and highlight(child_path, v, parent) else None
                        ),
                        (
                            'lowlight' if lowlight
                            and lowlight(child_path, v, parent) else None
                        )
                    ],
                )
            )
            include_keys.remove(k)

      if include_keys:
        s.write('<table>')
        for k, v in kv.items():
          if k not in include_keys:
            continue
          child_path = root_path + k
          key_cell = render_key_fn(
              key=k,
              parent=parent,
              root_path=child_path,
              enable_tooltip=enable_key_tooltip,
          )
          value_cell = Html.element(
              'div',
              [
                  render_value_fn(
                      value=v,
                      name=None,
                      parent=parent,
                      root_path=child_path,
                      special_keys=None,
                      include_keys=None,
                      exclude_keys=None,
                      collapse_level=collapse_level,
                      uncollapse=uncollapse,
                      filter=filter,
                      highlight=highlight,
                      lowlight=lowlight,
                      max_summary_len_for_str=max_summary_len_for_str,
                      enable_summary_tooltip=enable_summary_tooltip,
                      enable_key_tooltip=enable_key_tooltip,
                      **kwargs,
                  )
              ],
              css_class=[
                  (
                      'highlight' if highlight
                      and highlight(child_path, v, parent) else None
                  ),
                  (
                      'lowlight' if lowlight
                      and lowlight(child_path, v, parent) else None
                  )
              ],
          )
          s.write(
              Html.element(
                  'tr',
                  [
                      '<td>', key_cell, '</td>',
                      '<td>', value_cell, '</td>',
                  ],
              )
          )
        s.write('</table>')
    else:
      s.write(Html.element('span', css_class=['empty_container']))
    return Html.element(
        'div',
        [s],
        css_class=[
            'complex_value',
            self.css_class_name(parent),
            css_class,
        ]
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
      content: Union[str, Html, None] = HtmlView.PresetArgValue(None),
      **kwargs
  ) -> Html:
    """Renders a tooltip for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent value of the key-value pairs.
      root_path: The root path of the value.
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
        css_class=['tooltip', self.css_class_name(value)],
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
  def css_class_name(value: Any) -> Optional[str]:
    """Returns the CSS class name for the value."""
    if isinstance(value, HtmlTreeView.Extension):
      return Html.concate(value._html_element_class())  # pylint: disable=protected-access
    value = value if inspect.isclass(value) else type(value)
    return object_utils.camel_to_snake(value.__name__, '-')

  @staticmethod
  def max_collapse_level(
      original_level: int | None,
      subtree_uncollapse_level: int | None,
      root_path: KeyPath
  ) -> int | None:
    """Consolidates the collapse level."""
    if original_level is None or subtree_uncollapse_level is None:
      return None
    return max(original_level, root_path.depth + subtree_uncollapse_level)


_REGEX_CAMEL_TO_SNAKE = re.compile(r'(?<!^)(?=[A-Z])')

# pytype: enable=annotation-type-mismatch
