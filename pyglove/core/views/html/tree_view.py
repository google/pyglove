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
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

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

    def _html_tree_view_render(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str] = None,
        parent: Any = None,
        root_path: Optional[KeyPath] = None,
        **kwargs,
    ) -> Html:
      """The entrypoint of rendering the subtree represented by this extension.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        **kwargs: kwargs to pass to `view.render()` on this extension.

      Returns:
        The rendered HTML.
      """
      return self._html_tree_view(
          view=view,
          name=name,
          parent=parent,
          root_path=root_path,
          **view.get_kwargs(
              kwargs, self._html_tree_view_config(), root_path or KeyPath()
          )
      ).add_style(
          *self._html_tree_view_css_styles()
      )

    #
    # Users could override this methods to customize the styles and
    # rendering arguments for the subtree.
    #

    @classmethod
    def _html_tree_view_css_styles(cls) -> list[str]:
      """Returns the CSS styles for the subtree."""
      del cls
      return []

    @classmethod
    def _html_tree_view_config(cls) -> Dict[str, Any]:
      """Returns the config (rendering arguments) of current extension.

      Returns:
        A dictionary of rendering arguments for the subtree. These arguments
        will override the arguments passed to `view.render()`. See the
        `render()` method for the full list of arguments.
      """
      return {}

    #
    # Users could override the methods below to customize rendering
    # logics.
    #

    def _html_tree_view(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str] = None,
        parent: Any = None,
        root_path: Optional[KeyPath] = None,
        **kwargs,
    ) -> Html:
      """Returns the topmost HTML representation of this extension.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        **kwargs: kwargs to pass to the view. See `_html_tree_view_config` for
          the builtin arguments.

      Returns:
        The rendered HTML.
      """
      return view.render(
          self,
          name=name,
          parent=parent,
          root_path=root_path,
          **kwargs,
      )

    def _html_tree_view_summary(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str] = None,
        parent: Any = None,
        root_path: Optional[KeyPath] = None,
        **kwargs,
    ) -> Optional[Html]:
      """Returns the HTML summary for the object.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        **kwargs: kwargs to pass to the view. See `_html_tree_view_config` for
          the builtin arguments.

      Returns:
        An optional HTML object representing the summary of the object. If None,
        the content will be returned directly instead of having a <details>
        container.
      """
      return view.summary(
          self,
          name=name,
          parent=parent,
          root_path=root_path,
          **kwargs,
      )

    def _html_tree_view_content(
        self,
        *,
        view: 'HtmlTreeView',
        name: Optional[str] = None,
        parent: Any = None,
        root_path: Optional[KeyPath] = None,
        **kwargs,
        ) -> Html:
      """Returns the main content for the object.

      Args:
        view: The view to render the object.
        name: The name of the object.
        parent: The parent of the object.
        root_path: The key path of the object relative to the root.
        **kwargs: kwargs to pass to the view. See `_html_tree_view_config` for
          the builtin arguments.

      Returns:
        The rendered HTML as the main content of the object.
      """
      return view.content(
          self,
          name=name,
          parent=parent,
          root_path=root_path,
          **kwargs,
      )

  # NOTE(daiyip): update `get_kwargs()` and `get_passthrough_kwargs()` when new
  # arguments are added.
  @HtmlView.extension_method('_html_tree_view_render')
  def render(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_classes: Optional[Sequence[str]] = None,
      # Summary settings.
      title: Union[str, Html, None] = None,
      enable_summary: Optional[bool] = None,
      enable_summary_for_str: bool = True,
      max_summary_len_for_str: int = 80,
      enable_summary_tooltip: bool = True,
      summary_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = None,
      # Content settings.
      key_style: Union[
          Literal['label', 'summary'],
          Callable[[KeyPath, Any, Any], Literal['label', 'summary']]
      ] = 'summary',
      key_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = None,
      include_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = None,
      exclude_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = None,
      enable_key_tooltip: bool = True,
      # Collapse settings.
      collapse_level: Optional[int] = 1,
      uncollapse: Union[KeyPathSet, base.NodeFilter, None] = None,
      # Extension settings.
      child_config: Optional[Dict[str, Any]] = None,
      extra_flags: Optional[Dict[str, Any]] = None,
      # Tree operations.
      highlight: Optional[base.NodeFilter] = None,
      lowlight: Optional[base.NodeFilter] = None,
      debug: bool = False,
  ) -> Html:
    """Renders the entire HTML tree view for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_classes: CSS classes to add to the top-most element.
      title: The title of the summary. If None, the default title will be used,
        which is the type name of the value.
      enable_summary: Whether to enable the summary. If None, summary will
        be enabled for complex types or when string exceeds
        `max_summary_len_for_str`.
      enable_summary_for_str: Whether to enable the summary for strings.
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the tooltip for the summary.
      summary_color: The color used for the summary for displaying the referred
        field name of the object. It can be a tuple of (color, background-color)
        or a function that takes (root_path, value, parent) and returns the
        color tuple.
      key_style: The style of the key. If 'label', the key will be rendered as a
        label. If 'summary', it will be rendered as a summary in the <details>
        tag. If a function, it will be called with (root_path, value, parent)
        and return the style.
      key_color: The color for label-style keys under this extension. It can be
        a tuple of (color, background-color) or a function that takes
        (root_path, value, parent) and returns the color tuple.
      include_keys: A list of keys to include when displaying the sub-nodes of
        the object. If None, all keys will be displayed. If a function, it will
        be called with (root_path, value, parent) and return whether the key
        should be included.
      exclude_keys: A set of keys to exclude when displaying the sub-nodes of
        the object. If None, all keys will be displayed. If a function, it will
        be called with (root_path, value, parent) and return whether the key
        should be excluded.
      enable_key_tooltip: Whether to enable the tooltip for the object name.
      collapse_level: The level of collapsing. If 0, the object will be
        collapsed (without showing its sub-nodes). If 1, the immediate sub-nodes
        will be shown in collapsed form. If None, all sub-tree will be shown.
      uncollapse: Indivdual nodes to uncollapse. It can be a KeyPathSet or a
        function that takes (root_path, value, parent) and returns a KeyPathSet.
      child_config: The configs for the immediate child nodes of the object
        being rendered. It's a dictionary of (key, child-config) where the key
        is the name of the child node and the child-config is a dictionary of
        (key, value) to override the default configs for the child node.
      extra_flags: A dictionary of user-defined flags to control the rendering
        behavior.
      highlight: A function that takes (root_path, value, parent) and returns
        whether the node should be highlighted.
      lowlight: A function that takes (root_path, value, parent) and returns
        whether the node should be lowlighted.
      debug: Whether to show debug information for this rendering.

    Returns:
      The rendered HTML.
    """
    root_path = root_path or KeyPath()
    child_config = child_config or {}
    extra_flags = extra_flags or {}
    uncollapse = self.init_uncollapse(uncollapse)

    summary = self.summary(
        value,
        name=name,
        parent=parent,
        root_path=root_path,
        css_classes=css_classes,
        title=title,
        summary_color=summary_color,
        enable_summary=enable_summary,
        enable_summary_for_str=enable_summary_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        enable_key_tooltip=enable_key_tooltip,
        max_summary_len_for_str=max_summary_len_for_str,
        extra_flags=extra_flags,
    )

    if debug:
      debug_info = Html.element(
          'div',
          [
              Html.element(
                  'span', ['DEBUG'], css_classes=['debug-info-trigger']
              ),
              self.tooltip(
                  dict(
                      # Most error-prone settings.
                      css_classes=css_classes,
                      collapse_level=collapse_level,
                      uncollapse=uncollapse,
                      extra_flags=extra_flags,
                      child_config=child_config,
                      # Relative obvious settings.
                      key_style=key_style,
                      key_color=key_color,
                      include_keys=include_keys,
                      exclude_keys=exclude_keys,
                      # More obvious settings.
                      summary_color=summary_color,
                      enable_summary=enable_summary,
                      enable_summary_for_str=enable_summary_for_str,
                      max_summary_len_for_str=max_summary_len_for_str,
                      enable_summary_tooltip=enable_summary_tooltip,
                      enable_key_tooltip=enable_key_tooltip,
                  ),
                  name='debug_info',
                  parent=parent,
                  root_path=root_path,
                  css_classes=['debug-info'],
              ),
          ],
      ).add_style(
          """
          .debug-info-trigger {
            display: inline-flex;
            cursor: pointer;
            font-size: 0.6em;
            background-color: red;
            color: white;
            padding: 5px;
            border-radius: 3px;
            margin: 5px 0 5px 0;
          }
          .debug-info-trigger:hover + span.tooltip {
            visibility: visible;
          }
          """
      )
    else:
      debug_info = None

    content = self.content(
        value,
        name=name,
        parent=parent,
        root_path=root_path,
        css_classes=css_classes if summary is None else None,
        # Summary settings (child nodes).
        enable_summary=enable_summary,
        enable_summary_for_str=enable_summary_for_str,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        # Content settings.
        key_style=key_style,
        key_color=key_color,
        enable_key_tooltip=enable_key_tooltip,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        highlight=highlight,
        lowlight=lowlight,
        child_config=child_config,
        extra_flags=extra_flags,
        debug=debug,
    )

    if summary is None:
      content = Html.from_value(content)
      assert content is not None
      return debug_info + content

    collapse_details = self.should_collapse(
        value, name=name, parent=parent, root_path=root_path,
        collapse_level=collapse_level, uncollapse=uncollapse,
    )
    return Html.element(
        'details',
        [
            summary,
            debug_info,
            content,
        ],
        options=[None if collapse_details else 'open'],
        css_classes=[
            'pyglove',
            self.css_class_name(value),
            css_classes,
        ],
    ).add_style(
        """
        /* Value details styles. */
        details.pyglove {
          border: 1px solid #aaa;
          border-radius: 4px;
          padding: 0.5em 0.5em 0;
          margin: 0.25em 0;
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
    )

  def should_collapse(
      self,
      value: Any,
      name: Optional[str],
      root_path: KeyPath,
      parent: Any,
      collapse_level: Optional[int] = 1,
      uncollapse: Union[KeyPathSet, base.NodeFilter] = None,
  ) -> bool:
    """Returns True if the object should be collapsed.

    Args:
      value: The value to render.
      name: The referred field name of the value.
      root_path: The root path of the value.
      parent: The parent of the value.
      collapse_level: The level of collapsing. If 0, the object will be
        collapsed (without showing its sub-nodes). If 1, the immediate sub-nodes
        will be shown in collapsed form. If None, all sub-tree will be shown.
      uncollapse: Indivdual nodes to uncollapse. It can be a KeyPathSet or a
        function that takes (root_path, value, parent) and returns a KeyPathSet.

    Returns:
      True if the object should be collapsed.
    """
    if collapse_level is None or collapse_level > 0:
      return False
    if callable(uncollapse):
      return not uncollapse(root_path, value, parent)
    if root_path in uncollapse:
      return False
    # Always uncollapse simple types.
    if (name is not None
        and isinstance(value, (bool, int, float, str, type(None)))):
      return False
    return True

  def needs_summary(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      title: Union[str, Html, None] = None,
      enable_summary: Optional[bool] = None,
      enable_summary_for_str: bool = True,
      max_summary_len_for_str: int = 80,
  ) -> bool:
    """Returns True if the object needs a summary.

    Args:
      value: The value to render.
      name: The referred field name of the value.
      parent: The parent of the value.
      title: The title of the summary.
      enable_summary: Whether to enable the summary. If None, summary will
        be enabled for complex types or when string exceeds
        `max_summary_len_for_str`.
      enable_summary_for_str: Whether to enable the summary for strings.
      max_summary_len_for_str: The maximum length of the string to display.

    Returns:
      True if the object needs a summary.
    """
    del parent
    if isinstance(enable_summary, bool):
      return enable_summary
    assert enable_summary is None
    if not enable_summary_for_str and isinstance(value, str):
      return False
    if name is None and title is None and (
        isinstance(value, (int, float, bool, type(None)))
        or (isinstance(value, str) and len(value) <= max_summary_len_for_str)
    ):
      return False
    return True

  @HtmlView.extension_method('_html_tree_view_summary')
  def summary(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_classes: Optional[Sequence[str]] = None,
      title: Union[str, Html, None] = None,
      enable_summary: Optional[bool] = None,
      enable_summary_tooltip: bool = True,
      summary_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = None,
      max_summary_len_for_str: int = 80,
      enable_summary_for_str: bool = True,
      enable_key_tooltip: bool = True,
      summary_tooltip_fn: Optional[Callable[..., Html]] = None,
      key_tooltip_fn: Optional[Callable[..., Html]] = None,
      extra_flags: Optional[Dict[str, Any]] = None,
  ) -> Optional[Html]:
    """Renders the summary for an input value.

    Args:
      value: The value to render.
      name: The referred field name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_classes: The CSS classes to add to the HTML element.
      title: The title of the summary.
      enable_summary: Whether to enable the summary. If None, summary will
        be enabled for complex types or when string exceeds
        `max_summary_len_for_str`.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      summary_color: The color of the summary. If None, the summary will be
        rendered without a color. If a tuple, the first element is the text
        color and the second element is the background color. If a function,
        the function takes (root_path, value, parent) and returns a tuple of
        (text_color, background_color).
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_for_str: Whether to enable the summary for strings.
      enable_key_tooltip: Whether to enable the key tooltip.
      summary_tooltip_fn: The function to render the summary tooltip.
      key_tooltip_fn: The function to render the key tooltip.
      extra_flags: The extra flags to pass to the summary.

    Returns:
      An optional HTML object representing the summary of the value. If None,
      the summary will not be rendered.
    """
    del extra_flags
    root_path = root_path or KeyPath()
    if not self.needs_summary(
        value,
        name=name,
        parent=parent,
        title=title,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary=enable_summary,
        enable_summary_for_str=enable_summary_for_str,
    ):
      return None

    key_tooltip_fn = key_tooltip_fn or self.tooltip
    summary_tooltip_fn = summary_tooltip_fn or self.tooltip

    def make_title(value: Any):
      if inspect.isclass(value):
        return 'type'
      elif isinstance(value, (int, float, bool, str)):
        return type(value).__name__
      return f'{type(value).__name__}(...)'

    if name is not None:
      summary_color = self.get_color(
          summary_color, root_path + name, value, parent
      )
    else:
      summary_color = (None, None)

    return Html.element(
        'summary',
        [
            # Summary name.
            lambda: Html.element(  # pylint: disable=g-long-ternary
                'div',
                [
                    name,
                    key_tooltip_fn(   # pylint: disable=g-long-ternary
                        root_path,
                        name=name,
                        parent=parent,
                        root_path=root_path,
                        css_classes=css_classes,
                    ) if enable_key_tooltip else None,
                ],
                css_classes=['summary-name', css_classes],
                styles=dict(
                    color=summary_color[0],
                    background_color=summary_color[1],
                ),
            ) if name else None,

            # Summary title
            Html.element(
                'div',
                [
                    title or make_title(value),
                ],
                css_classes=['summary-title', css_classes],
            ),

            # Summary tooltip.
            lambda: summary_tooltip_fn(   # pylint: disable=g-long-ternary
                value,
                parent=parent,
                root_path=root_path,
                css_classes=css_classes,
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
        .summary-name {
          display: inline;
          padding: 3px 5px 3px 5px;
          margin: 0 5px;
          border-radius: 3px;
        }
        .summary-title {
          display: inline;
        }
        .summary-name + div.summary-title {
          display: inline;
          color: #aaa;
        }
        .summary-title:hover + span.tooltip {
          visibility: visible;
        }
        .summary-name:hover > span.tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        """
    )

  # NOTE(daiyip)" `object_key`` does not have a corresponding extension
  # method in `HtmlTreeView.Extension`, because the rendering of the key is not
  # delegated to `HtmlTreeView.Extension`.
  def object_key(
      self,
      root_path: KeyPath,
      *,
      value: Any,
      parent: Any,
      css_classes: Optional[Sequence[str]] = None,
      key_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = None,
      enable_key_tooltip: bool = True,
      key_tooltip_fn: Optional[Callable[..., Html]] = None,
      **kwargs,
  ) -> Html:
    """Renders a label-style key for the value.

    Args:
      root_path: The root path of the value.
      value: The value to render.
      parent: The parent of the value.
      css_classes: The CSS classes to add to the HTML element.
      key_color: The color of the key. If None, the key will be rendered
        without a color. If a tuple, the first element is the text color and
        the second element is the background color. If a function, the function
        takes (root_path, value, parent) and returns a tuple of (text_color,
        background_color).
      enable_key_tooltip: Whether to enable the tooltip.
      key_tooltip_fn: The function to render the key tooltip.
      **kwargs: Additional arguments passed by the user that will be ignored.

    Returns:
      The rendered HTML as the key of the value.
    """
    del kwargs
    key_tooltip_fn = key_tooltip_fn or self.tooltip
    key_color = self.get_color(key_color, root_path, value, parent)
    return (
        # Key span.
        Html.element(
            'span',
            [
                str(root_path.key),
            ],
            css_classes=[
                'object-key',
                type(root_path.key).__name__,
                css_classes,
            ],
            styles=dict(
                color=key_color[0],
                background_color=key_color[1],
            )
        ) + (
            # Tooltip if enabled.
            lambda: key_tooltip_fn(  # pylint: disable=g-long-ternary
                value=root_path,
                root_path=root_path,
                parent=parent,
            ) if enable_key_tooltip else None
        )
    ).add_style(
        """
        /* Object key styles. */
        .object-key {
          margin: 0.15em 0.3em 0.15em 0;
          display: block;
        }
        .object-key:hover + .tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        .object-key.str {
          color: gray;
          border: 1px solid lightgray;
          background-color: ButtonFace;
          border-radius: 0.2em;
          padding: 0.3em;
        }
        .object-key.int::before{
          content: '[';
        }
        .object-key.int::after{
          content: ']';
        }
        .object-key.int{
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
      css_classes: Optional[Sequence[str]] = None,
      # Summary settings (for child nodes).
      enable_summary: Optional[bool] = None,
      enable_summary_for_str: bool = True,
      max_summary_len_for_str: int = 80,
      enable_summary_tooltip: bool = True,
      # Content settings.
      key_style: Union[
          Literal['label', 'summary'],
          Callable[[KeyPath, Any, Any], Literal['label', 'summary']]
      ] = 'summary',
      key_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = None,
      include_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = None,
      exclude_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = None,
      enable_key_tooltip: bool = True,
      # Collapse settings.
      collapse_level: Optional[int] = 1,
      uncollapse: Union[KeyPathSet, base.NodeFilter, None] = None,
      # Other settings.
      highlight: Optional[base.NodeFilter] = None,
      lowlight: Optional[base.NodeFilter] = None,
      child_config: Optional[Dict[str, Any]] = None,
      extra_flags: Optional[Dict[str, Any]] = None,
      debug: bool = False,
  ) -> Html:
    """Renders the main content for the value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_classes: CSS classes to add to the HTML element.
      enable_summary: Whether to enable the summary.
      enable_summary_for_str: Whether to enable the summary for string.
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      key_style: The style of the key. It can be either 'label' or 'summary'.
        If it is a function, the function takes (root_path, value, parent) and
        returns either 'label' or 'summary'.
      key_color: The color of the key. If it is a tuple, the first element is
        the text color and the second element is the background color. If it is
        a function, the function takes (root_path, value, parent) and returns
        a tuple of (text_color, background_color).
      include_keys: The keys to include (at the immediate child level). If it is
        a function, the function takes (root_path, value, parent) and returns
        an iterable of keys to include.
      exclude_keys: The keys to exclude (at the immediate child level). If it is
        a function, the function takes (root_path, value, parent) and returns
        an iterable of keys to exclude.
      enable_key_tooltip: Whether to enable the key tooltip.
      collapse_level: The level to collapse the tree.
      uncollapse: A key path set (relative to root_path) for the nodes to
        uncollapse. or a function with signature (path, value, parent) -> bool
        to filter nodes to uncollapse.
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight.
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight.
      child_config: The configuration for rendering the child nodes.
      extra_flags: Extra flags to pass to the child render.
      debug: Whether to enable debug mode.

    Returns:
      The rendered HTML as the main content of the value.
    """
    root_path = root_path or KeyPath()

    if isinstance(value, (tuple, list)):
      items = {i: v for i, v in enumerate(value)}
    elif isinstance(value, dict):
      items = value
    else:
      return self.simple_value(
          value, name=name, parent=parent, root_path=root_path,
          css_classes=css_classes,
          max_summary_len_for_str=max_summary_len_for_str
      )
    return self.complex_value(
        items,
        name=name,
        parent=value,
        root_path=root_path,
        css_classes=css_classes,
        enable_summary=enable_summary,
        enable_summary_for_str=enable_summary_for_str,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        key_style=key_style,
        key_color=key_color,
        enable_key_tooltip=enable_key_tooltip,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        child_config=child_config,
        highlight=highlight,
        lowlight=lowlight,
        extra_flags=extra_flags,
        debug=debug,
    )

  def simple_value(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_classes: Optional[Sequence[str]] = None,
      max_summary_len_for_str: int = 80,
  ) -> Html:
    """Renders a simple value.

    Args:
      value: The value to render.
      name: The name of the value.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_classes: CSS classes to add to the HTML element.
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
        css_classes=[
            'simple-value',
            self.css_class_name(value),
            css_classes,
        ],
    ).add_style(
        """
        /* Simple value styles. */
        .simple-value {
          color: blue;
          display: inline-block;
          white-space: pre-wrap;
          padding: 0.2em;
          margin-top: 0.15em;
        }
        .simple-value.str {
          color: darkred;
          font-style: italic;
        }
        .simple-value.int, .simple-value.float {
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
      css_classes: Optional[Sequence[str]] = None,
      # Summary settings (for child nodes).
      enable_summary: Optional[bool] = None,
      enable_summary_for_str: bool = True,
      max_summary_len_for_str: int = 80,
      enable_summary_tooltip: bool = True,
      # Content settings.
      key_style: Union[
          Literal['label', 'summary'],
          Callable[..., Literal['label', 'summary']]
      ] = 'summary',
      key_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = None,
      include_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = None,
      exclude_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = None,
      enable_key_tooltip: bool = True,
      # Collapse settings.
      collapse_level: Optional[int] = 1,
      uncollapse: Union[KeyPathSet, base.NodeFilter, None] = None,
      # Other settings.
      child_config: Optional[Dict[str, Any]] = None,
      highlight: Optional[base.NodeFilter] = None,
      lowlight: Optional[base.NodeFilter] = None,
      # Custom render functions.
      render_key_fn: Optional[Callable[..., Html]] = None,
      render_value_fn: Optional[Callable[..., Html]] = None,
      extra_flags: Optional[Dict[str, Any]] = None,
      debug: bool = False,
  ) -> Html:
    """Renders a list of key-value pairs.

    Args:
      kv: The key-value pairs to render.
      parent: The parent of the value.
      root_path: The root path of the value.
      name: The name of the value.
      css_classes: CSS classes to add to the HTML element.
      enable_summary: Whether to enable the summary. If None, the default is
        to enable the summary for non-string and disable the summary for
        string.
      enable_summary_for_str: Whether to enable the summary for string.
      max_summary_len_for_str: The maximum length of the string to display.
      enable_summary_tooltip: Whether to enable the summary tooltip.
      key_style: The style of the key. It can be either 'label' or 'summary'.
        If it is a function, the function takes (root_path, value, parent) and
        returns either 'label' or 'summary'.
      key_color: The color of the key. If it is a tuple, the first element is
        the text color and the second element is the background color. If it is
        a function, the function takes (root_path, value, parent) and returns
        a tuple of (text_color, background_color).
      include_keys: The keys to include (at the immediate child level). If it is
        a function, the function takes (root_path, value, parent) and returns
        an iterable of keys to include.
      exclude_keys: The keys to exclude (at the immediate child level). If it is
        a function, the function takes (root_path, value, parent) and returns
        an iterable of keys to exclude.
      enable_key_tooltip: Whether to enable the key tooltip.
      collapse_level: The level to collapse the tree.
      uncollapse: A key path set (relative to root_path) for the nodes to
        uncollapse. or a function with signature (path, value, parent) -> bool
        to filter nodes to uncollapse.
      child_config: The configuration for rendering the child nodes.
      highlight: A function with signature (path, value, parent) -> bool
        to determine whether to highlight.
      lowlight: A function with signature (path, value, parent) -> bool
        to determine whether to lowlight.
      render_key_fn: A custom function to render the label-style key.
      render_value_fn: A custom function to render the child value.
      extra_flags: Extra flags to pass to the child render.
      debug: Whether to enable debug mode.

    Returns:
      The rendered HTML as the key-value pairs.
    """
    del name
    root_path = root_path or KeyPath()
    uncollapse = self.init_uncollapse(uncollapse)
    extra_flags = extra_flags or {}

    inherited_kwargs = dict(
        # For child summary.
        enable_summary=enable_summary,
        enable_summary_for_str=enable_summary_for_str,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        # For child content.
        enable_key_tooltip=enable_key_tooltip,
        key_style=key_style,
        key_color=key_color,
        include_keys=include_keys if callable(include_keys) else None,
        exclude_keys=exclude_keys if callable(exclude_keys) else None,
        collapse_level=None if collapse_level is None else (collapse_level - 1),
        uncollapse=uncollapse,
        highlight=highlight,
        lowlight=lowlight,
        extra_flags=extra_flags,
        debug=debug,
    )

    render_key_fn = render_key_fn or HtmlTreeView.object_key
    render_value_fn = render_value_fn or HtmlTreeView.render

    def render_child_key(child_path, value, parent, child_kwargs):
      render_child_key_fn = child_kwargs['extra_flags'].get(
          'render_key_fn', render_key_fn
      )
      return render_child_key_fn(
          self,
          child_path,
          value=value,
          parent=parent,
          **child_kwargs
      )

    def render_child_value(name, value, child_path, child_kwargs):
      render_child_value_fn = child_kwargs['extra_flags'].get(
          'render_value_fn', render_value_fn
      )
      child_html = render_child_value_fn(
          self,
          value=value, name=child_kwargs.pop('name', name),
          parent=parent,
          root_path=child_path,
          **child_kwargs
      )
      should_highlight = highlight and highlight(child_path, value, parent)
      should_lowlight = lowlight and lowlight(child_path, value, parent)
      if should_highlight or should_lowlight:
        return Html.element(
            'div', [child_html],
            css_classes=[
                'highlight' if should_highlight else None,
                'lowlight' if should_lowlight else None,
            ],
        )
      else:
        return child_html

    has_child = False
    s = Html()
    if kv:
      # Compute included keys.
      if callable(include_keys):
        include_keys = [
            k for k, v in kv.items() if include_keys(root_path + k, v, parent)
        ]
      elif include_keys is not None:
        include_keys = list(k for k in include_keys if k in kv)
      else:
        include_keys = list(kv.keys())

      # Filter with excluded keys.
      if callable(exclude_keys):
        include_keys = [
            k for k in include_keys if not exclude_keys(
                root_path + k, kv[k], parent
            )
        ]
      elif exclude_keys is not None:
        exclude_keys = set(exclude_keys)
        include_keys = [k for k in include_keys if k not in exclude_keys]

      # Figure out keys of different styles.
      label_keys = []
      summary_keys = []
      if isinstance(parent, (tuple, list)) or key_style == 'label':
        label_keys = include_keys
      elif key_style == 'summary':
        summary_keys = include_keys
      else:
        assert callable(key_style), key_style
        for k in include_keys:
          ks = key_style(root_path + k, kv[k], parent)
          if ks == 'summary':
            summary_keys.append(k)
          elif ks == 'label':
            label_keys.append(k)

      # Render child nodes with summary keys.
      if summary_keys:
        for k in summary_keys:
          child_path = root_path + k
          child_kwargs = self.get_child_kwargs(
              inherited_kwargs, child_config, k, root_path
          )
          s.write(render_child_value(k, kv[k], child_path, child_kwargs))
          has_child = True

      # Render child nodes with label keys.
      if label_keys:
        s.write('<table>')
        for k in label_keys:
          v = kv[k]
          child_path = root_path + k
          child_kwargs = self.get_child_kwargs(
              inherited_kwargs, child_config, k, root_path
          )
          key_cell = render_child_key(child_path, v, parent, child_kwargs)
          value_cell = render_child_value(None, v, child_path, child_kwargs)
          if value_cell is not None:
            s.write(
                Html.element(
                    'tr',
                    [
                        '<td>', key_cell, '</td>',
                        '<td>', value_cell, '</td>',
                    ],
                )
            )
            has_child = True
        s.write('</table>')

    if not has_child:
      s.write(Html.element('span', css_classes=['empty-container']))

    return Html.element(
        'div',
        [s],
        css_classes=[
            'complex-value',
            self.css_class_name(parent),
            css_classes,
        ]
    ).add_style(
        """
        /* Complex value styles. */
        span.empty-container::before {
            content: '(empty)';
            font-style: italic;
            margin-left: 0.5em;
            color: #aaa;
        }
        """
    )

  def tooltip(
      self,
      value: Any,
      *,
      parent: Any = None,
      root_path: Optional[KeyPath] = None,
      css_classes: Optional[Sequence[str]] = None,
      content: Union[str, Html, None] = None,
      **kwargs,
  ) -> Html:
    """Renders a tooltip for the value.

    Args:
      value: The value to render.
      parent: The parent of the value.
      root_path: The root path of the value.
      css_classes: CSS classes to add to the HTML element.
      content: The content to render. If None, the value will be rendered.
      **kwargs: Additional keyword arguments passed from the user that 
        will be ignored.

    Returns:
      The rendered HTML as the tooltip of the value.
    """
    del parent, kwargs
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
          )
      )
    return Html.element(
        'span',
        [content],
        css_classes=[
            'tooltip',
            css_classes,
        ],
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
    # if isinstance(value, HtmlTreeView.Extension):
    #   return Html.concate(value._html_element_class())  # pylint: disable=protected-access
    if inspect.isclass(value):
      class_name = f'{value.__name__}-class'
    else:
      class_name = type(value).__name__
    return object_utils.camel_to_snake(class_name, '-')

  @staticmethod
  def init_uncollapse(
      uncollapse: Union[Iterable[Union[KeyPath, str]], base.NodeFilter, None],
  ) -> Union[KeyPathSet, base.NodeFilter]:
    """Initializes the uncollapse argument."""
    if uncollapse is None:
      return KeyPathSet()
    elif callable(uncollapse):
      return uncollapse
    else:
      return KeyPathSet.from_value(uncollapse, include_intermediate=True)

  @staticmethod
  def get_child_kwargs(
      call_kwargs: Dict[str, Any],
      child_config: Dict[str, Any],
      child_key: Optional[str],
      root_path: KeyPath,
  ) -> Dict[str, Any]:
    """Enter the child config for a child key."""
    if not child_config:
      return call_kwargs

    child_kwargs = child_config.get(
        child_key, child_config.get('__default__', None)
    )
    if not child_kwargs:
      return call_kwargs

    return HtmlTreeView.get_kwargs(
        call_kwargs, child_kwargs, root_path + child_key,
    )

  # pytype: disable=annotation-type-mismatch
  @staticmethod
  def get_passthrough_kwargs(
      *,
      enable_summary: Optional[bool] = object_utils.MISSING_VALUE,
      enable_summary_for_str: bool = object_utils.MISSING_VALUE,
      max_summary_len_for_str: int = object_utils.MISSING_VALUE,
      enable_summary_tooltip: bool = object_utils.MISSING_VALUE,
      key_style: Union[
          Literal['label', 'summary'],
          Callable[[KeyPath, Any, Any], Literal['label', 'summary']]
      ] = object_utils.MISSING_VALUE,
      key_color: Union[
          Tuple[Optional[str], Optional[str]],
          Callable[[KeyPath, Any, Any], Tuple[Optional[str], Optional[str]]]
      ] = object_utils.MISSING_VALUE,
      include_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = object_utils.MISSING_VALUE,
      exclude_keys: Union[
          Iterable[Union[int, str]],
          Callable[[KeyPath, Any, Any], Iterable[Union[int, str]]],
          None
      ] = object_utils.MISSING_VALUE,
      enable_key_tooltip: bool = object_utils.MISSING_VALUE,
      uncollapse: Union[
          KeyPathSet, base.NodeFilter, None
      ] = object_utils.MISSING_VALUE,
      extra_flags: Optional[Dict[str, Any]] = object_utils.MISSING_VALUE,
      highlight: Optional[base.NodeFilter] = object_utils.MISSING_VALUE,
      lowlight: Optional[base.NodeFilter] = object_utils.MISSING_VALUE,
      debug: bool = object_utils.MISSING_VALUE,
      remove: Optional[Iterable[str]] = None,
      **kwargs,
  ):
  # pytype: enable=annotation-type-mismatch
    """Gets the rendering arguments to pass through to the child nodes."""
    del kwargs
    passthrough_kwargs = dict(
        enable_summary=enable_summary,
        enable_summary_for_str=enable_summary_for_str,
        max_summary_len_for_str=max_summary_len_for_str,
        enable_summary_tooltip=enable_summary_tooltip,
        enable_key_tooltip=enable_key_tooltip,
        key_style=key_style,
        key_color=key_color,
        include_keys=(
            include_keys if callable(include_keys)
            else object_utils.MISSING_VALUE
        ),
        exclude_keys=(
            exclude_keys if callable(exclude_keys)
            else object_utils.MISSING_VALUE
        ),
        uncollapse=uncollapse,
        highlight=highlight,
        lowlight=lowlight,
        extra_flags=extra_flags,
        debug=debug
    )
    # Filter out missing values.
    passthrough_kwargs = {
        k: v for k, v in passthrough_kwargs.items()
        if v is not object_utils.MISSING_VALUE
    }
    if remove:
      return {
          k: v for k, v in passthrough_kwargs.items()
          if k not in remove
      }
    return passthrough_kwargs

  @staticmethod
  def get_collapse_level(
      original_level: Union[None, int, Tuple[Optional[int], int]],
      overriden_level: Union[None, int, Tuple[Optional[int], int]],
      ) -> Optional[int]:
    """Gets the collapse level for a child node."""
    original_offset, overriden_offset = 0, 0
    if isinstance(original_level, tuple):
      original_level, original_offset = original_level
    if isinstance(overriden_level, tuple):
      overriden_level, overriden_offset = overriden_level

    if original_level is None:
      return original_level
    elif overriden_level is None:
      return overriden_level
    else:
      return max(
          original_level + original_offset,
          overriden_level + overriden_offset
      )

  @staticmethod
  def get_kwargs(
      call_kwargs: Dict[str, Any],
      overriden_kwargs: Dict[str, Any],
      root_path: Optional[KeyPath] = None,
  ) -> Dict[str, Any]:
    """Override render arguments."""
    # Select child config to override.
    if not overriden_kwargs:
      return call_kwargs

    call_kwargs = call_kwargs.copy()
    overriden_kwargs = overriden_kwargs.copy()

    # Override collapse_level.
    if 'collapse_level' in call_kwargs or 'collapse_level' in overriden_kwargs:
      call_kwargs['collapse_level'] = HtmlTreeView.get_collapse_level(
          call_kwargs.pop('collapse_level', 1),
          overriden_kwargs.pop('collapse_level', 0)
      )

    # Override uncollapse.
    if 'uncollapse' in call_kwargs or 'uncollapse' in overriden_kwargs:
      uncollapse = KeyPathSet.from_value(
          call_kwargs.pop('uncollapse', None) or []
      )
      child_uncollapse = KeyPathSet.from_value(
          overriden_kwargs.pop('uncollapse', None) or []
      )
      call_kwargs['uncollapse'] = HtmlTreeView.merge_uncollapse(
          uncollapse, child_uncollapse, root_path
      )

    # Deep hierarchy merge.
    return object_utils.merge_tree(call_kwargs, overriden_kwargs)

  @staticmethod
  def merge_uncollapse(
      uncollapse: Union[KeyPathSet, base.NodeFilter, None],
      child_uncollapse: Optional[KeyPathSet],
      child_path: Optional[KeyPath] = None,
  ) -> Union[KeyPathSet, base.NodeFilter]:
    """Merge uncollapse paths."""
    if not uncollapse and not child_uncollapse:
      return KeyPathSet()

    if callable(uncollapse) or not child_uncollapse:
      assert uncollapse is not None
      return uncollapse

    assert isinstance(uncollapse, KeyPathSet), uncollapse
    assert isinstance(child_uncollapse, KeyPathSet), child_uncollapse
    if child_path:
      child_uncollapse = child_uncollapse.copy()
      child_uncollapse.rebase(child_path)
    uncollapse.update(child_uncollapse)
    return uncollapse

  @staticmethod
  def get_color(
      color: Union[
          Tuple[str, str],
          Callable[
              [KeyPath, Any, Any],
              Tuple[Optional[str], Optional[str]]
          ],
          None
      ],
      root_path,
      value,
      parent
  ) -> Tuple[Optional[str], Optional[str]]:
    if callable(color):
      return color(root_path, value, parent)
    if color is None:
      return (None, None)
    assert isinstance(color, tuple) and len(color) == 2, color
    return color

# pytype: enable=annotation-type-mismatch
