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
"""HTML and the base HtmlView."""

import abc
import functools
import html as html_lib
import inspect
import typing
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.views import base

NestableStr = Union[
    str,
    Sequence[Union[str, None, Sequence[Optional[str]]]],
    None,
]

NodeFilter = base.NodeFilter
NodeColor = Callable[
    [
        object_utils.KeyPath,    # The path to the value.
        Any,        # Current value.
        Any,        # Parent value
    ],
    Optional[str]   # The color of the node.
]


class Html(base.Content):
  """HTML with consolidated CSS and Scripts.

  Example::

  .. code-block:: python

    def foo() -> pg.Html:
      s = pg.Html()
      s.add_style('div.foo { color: red; }')
      s.add_script('function myFoo() { console.log("foo");}')
      s.write('<div class="foo">Foo</div>')
      return s

    def bar() -> pg.Html:
      s = pg.Html()
      s.add_style('div.bar { color: green; }')
      s.add_script('function myBar() { console.log("bar");}')
      s.write('<div class="bar">')
      s.write(foo())
      s.write('</div>')
      return s

    html = bar.html_str()

    This will output::

     <html>
     <head>
     <style>
     div.bar { color: green; }
     div.foo { color: red; }
     </style>
     <script>
     function myBar() { console.log("bar");}
     function myFoo() { console.log("foo");}
     </script>
     </head>
     <body><div class="bar"><div class="foo">Foo</div></div></body></html>

  """

  WritableTypes = Union[    # pylint: disable=invalid-name
      str,
      'Html',
      'HtmlConvertible',
      Callable[[], Union[str, 'Html', 'HtmlConvertible', None]],
      None
  ]

  class Scripts(base.Content.SharedParts):
    """Shared script definitions in the HEAD section."""

    @functools.cached_property
    def content(self) -> str:
      if self.parts:
        code = '\n'.join([inspect.cleandoc(v) for v in self.parts.keys()])
        return f'<script>\n{code}\n</script>'
      return ''

  class ScriptFiles(base.Content.SharedParts):
    """Shared script files to link to in the HEAD section."""

    @functools.cached_property
    def content(self) -> str:
      return '\n'.join(
          [f'<script src="{url}"></script>' for url in self.parts.keys()]
      )

  class Styles(base.Content.SharedParts):
    """Shared style definitions in the HEAD section."""

    @functools.cached_property
    def content(self) -> str:
      if self.parts:
        styles = '\n'.join([inspect.cleandoc(v) for v in self.parts.keys()])
        return f'<style>\n{styles}\n</style>'
      return ''

  class StyleFiles(base.Content.SharedParts):
    """Shared style files to link to in the HEAD section."""

    @functools.cached_property
    def content(self) -> str:
      return '\n'.join(
          [
              f'<link rel="stylesheet" href="{url}">'
              for url in self.parts.keys()
          ]
      )

  def __init__(  # pylint: disable=useless-super-delegation
      self,
      *content: WritableTypes,
      style_files: Optional[Iterable[str]] = None,
      styles: Optional[Iterable[str]] = None,
      script_files: Optional[Iterable[str]] = None,
      scripts: Optional[Iterable[str]] = None,
  ) -> None:
    """Constructor.

    Args:
      *content: One or multiple body part (str, Html, lambda, None) of the HTML.
      style_files: URLs for external styles to include.
      styles: CSS styles to include.
      script_files: URLs for external scripts to include.
      scripts: JavaScript scripts to include.
    """
    super().__init__(
        *content,
        style_files=Html.StyleFiles(*(style_files or [])),
        styles=Html.Styles(*(styles or [])),
        script_files=Html.ScriptFiles(*(script_files or [])),
        scripts=Html.Scripts(*(scripts or [])),
    )

  def _repr_html_(self) -> str:
    return self.to_str()

  @property
  def styles(self) -> 'Html.Styles':
    """Returns the styles to include in the HTML."""
    return self._shared_parts['styles']

  @property
  def style_files(self) -> 'Html.StyleFiles':
    """Returns the style files to link to."""
    return self._shared_parts['style_files']

  @property
  def scripts(self) -> 'Html.Scripts':
    """Returns the scripts to include in the HTML."""
    return self._shared_parts['scripts']

  @property
  def script_files(self) -> 'Html.ScriptFiles':
    """Returns the script files to link to."""
    return self._shared_parts['script_files']

  @property
  def head_section(self) -> str:
    """Returns the head section."""
    return '\n'.join(
        [
            v for v in [
                '<head>', self.style_section, self.script_section, '</head>']
            if v
        ]
    )

  @property
  def style_section(self) -> str:
    """Returns the style section."""
    return '\n'.join(
        [
            v for v in [self.style_files.content, self.styles.content]
            if v
        ]
    )

  @property
  def script_section(self) -> str:
    """Returns the script section."""
    return '\n'.join(
        [
            v for v in [self.script_files.content, self.scripts.content]
            if v
        ]
    )

  @property
  def body_section(self) -> str:
    """Returns the body section."""
    return f'<body>\n{self.content}\n</body>'

  #
  # Methods for adding shared parts and writing HTML content.
  #

  def add_style(self, *css: str) -> 'Html':
    """Adds CSS styles to the HTML."""
    self.styles.add(*css)
    return self

  def add_script(self, *js: str) -> 'Html':
    """Adds JavaScript scripts to the HTML."""
    self.scripts.add(*js)
    return self

  def add_style_file(self, *url: str) -> 'Html':
    """Adds a style file to the HTML."""
    self.style_files.add(*url)
    return self

  def add_script_file(self, *url: str) -> 'Html':
    """Adds a script file to the HTML."""
    self.script_files.add(*url)
    return self

  def to_str(
      self,
      *,
      content_only: bool = False,
      **kwargs
  ) -> str:
    """Returns the HTML str.

    Args:
      content_only: If True, only the content will be returned.
      **kwargs: Additional keyword arguments passed from the user that 
        will be ignored.

    Returns:
      The generated HTML str.
    """
    if content_only:
      return self.content
    return '\n'.join(
        [
            v for v in ['<html>', self.head_section,
                        self.body_section, '</html>'] if v
        ]
    )

  @classmethod
  def from_value(
      cls,
      value: WritableTypes,
      copy: bool = False
  ) -> Union['Html', None]:
    return typing.cast(
        Html, super().from_value(value, copy=copy)
    )

  @classmethod
  def _to_content(
      cls, value: WritableTypes
  ) -> Union['Html', str, None]:
    if callable(value):
      value = value()
    if value is None:
      return None
    elif isinstance(value, HtmlConvertible):
      value = value.to_html()
    elif not isinstance(value, (str, cls)):
      raise TypeError(f'Not a writable value for `{cls.__name__}`: {value!r}')
    return value

  #
  # Helper methods for creating templated Html objects.
  #

  @classmethod
  def element(
      cls,
      tag: str,
      inner_html: Optional[List[WritableTypes]] = None,
      *,
      options: Union[str, Iterable[str], None] = None,
      css_classes: NestableStr = None,
      styles: Union[str, Dict[str, Any], None] = None,
      **properties
  ) -> 'Html':
    """Creates an HTML element.

    Args:
      tag: The HTML tag name.
      inner_html: The inner HTML of the element.
      options: Positional options that will be added to the element. E.g. 'open'
        for `<details open>`.
      css_classes: The CSS class name or a list of CSS class names.
      styles: A single CSS style string or a dictionary of CSS properties.
      **properties: Keyword arguments for HTML properties. For properties with
        underscore in the name, the underscore will be replaced by dash in the
        generated HTML. E.g. `background_color` will be converted to
        `background-color`.

    Returns:
      The opening tag of an HTML element.
    """
    s = cls()

    # Write the open tag.
    css_classes = cls.concate(css_classes)
    options = cls.concate(options)
    styles = cls.style_str(styles)
    s.write(
        f'<{tag}',
        f' {options}' if options else None,
        f' class="{css_classes}"' if css_classes else None,
        f' style="{styles}"' if styles else None,
    )
    for k, v in properties.items():
      if v is not None:
        s.write(f' {k.replace("_", "-")}="{v}"')
    s.write('>')

    # Write the inner HTML.
    if inner_html:
      for child in inner_html:
        s.write(child)

    # Write the closing tag.
    s.write(f'</{tag}>')
    return s

  @classmethod
  def escape(
      cls,
      s: WritableTypes,
      javascript_str: bool = False
  ) -> Union[str, 'Html', None]:
    """Escapes an HTML writable object."""
    if s is None:
      return None

    if callable(s):
      s = s()
    if isinstance(s, HtmlConvertible):
      s = s.to_html()

    def _escape(s: str) -> str:
      if javascript_str:
        return (
            s.replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\r', '\\r')
            .replace('\n', '\\n')
            .replace('\t', '\\t')
        )
      return html_lib.escape(s)

    if isinstance(s, str):
      return _escape(s)
    else:
      assert isinstance(s, Html), s
      return Html(_escape(s.content)).write(
          s, shared_parts_only=True
      )

  @classmethod
  def concate(
      cls, nestable_str: NestableStr, separator: str = ' ', dedup: bool = True
  ) -> Optional[str]:
    """Concates the string nodes in a nestable object."""
    flattened = object_utils.flatten(nestable_str)
    if isinstance(flattened, str):
      return flattened
    elif isinstance(flattened, dict):
      str_items = [v for v in flattened.values() if isinstance(v, str)]
      if dedup:
        str_items = list(dict.fromkeys(str_items).keys())
      if str_items:
        return separator.join(str_items)
    return None

  @classmethod
  def style_str(
      cls, style: Union[str, Dict[str, Any], None],
  ) -> Optional[str]:
    """Gets a string representing an inline CSS style.

    Args:
      style: A single CSS style string, or a dictionary for CSS properties.
        When dictionary form is used, underscore in the key name will be
        replaced by dash in the generated CSS style string.
        For example, `background_color` will be converted to `background-color`.

    Returns:
      A CSS style string or None if no CSS property is provided.
    """
    if not style:
      return None
    if isinstance(style, str):
      return style
    else:
      assert isinstance(style, dict), style
      return ''.join(
          [
              f'{k.replace("_", "-")}:{v};'
              for k, v in style.items() if v is not None
          ]
      ) or None


# Allow automatic conversion from str to Html.
pg_typing.register_converter(str, Html, convert_fn=Html.from_value)


class HtmlConvertible:
  """Base class for HTML convertible objects."""

  def to_html(self, **kwargs) -> Html:
    """Returns the HTML representation of the object."""
    return to_html(self, **kwargs)

  def to_html_str(self, *, content_only: bool = False, **kwargs) -> str:
    """Returns the HTML str of the object."""
    return self.to_html(**kwargs).to_str(content_only=content_only)

  def _repr_html_(self) -> str:
    return self.to_html_str()


class HtmlView(base.View):
  """Base class for HTML views."""

  class Extension(base.View.Extension, HtmlConvertible):
    """Base class for HtmlView extensions."""

  def render(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      root_path: Optional[object_utils.KeyPath] = None,
      **kwargs
  ) -> Html:
    """Renders the input value into an HTML object."""
    # For customized HtmlConvertible objects, call their `to_html()` method.
    if (isinstance(value, HtmlConvertible)
        and not isinstance(value, self.__class__.Extension)
        and value.__class__.to_html is not HtmlConvertible.to_html):
      return value.to_html(name=name, root_path=root_path, **kwargs)
    return self._render(value, name=name, root_path=root_path, **kwargs)

  @abc.abstractmethod
  def _render(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      root_path: Optional[object_utils.KeyPath] = None,
      **kwargs
  ) -> Html:
    """View's implementation of HTML rendering."""


def to_html(
    value: Any,
    *,
    name: Optional[str] = None,
    root_path: Optional[object_utils.KeyPath] = None,
    view_id: str = 'html-tree-view',
    **kwargs
) -> Html:
  """Returns the HTML representation of a value.

  Args:
    value: The value to render.
    name: The name of the value.
    root_path: The root path of the value.
    view_id: The ID of the view to render the value.
      See `pg.views.HtmlView.dir()` for all available HTML view IDs.
    **kwargs: Additional keyword arguments passed from `pg.to_html`, wich
        will be passed to the `HtmlView.render_xxx()` (thus
        `Extension._html_xxx()`) methods.

  Returns:
    The rendered HTML.
  """
  content = base.view(
      value,
      name=name,
      root_path=root_path,
      view_id=view_id,
      **kwargs
  )
  assert isinstance(content, Html), content
  return content


def to_html_str(
    value: Any,
    *,
    name: Optional[str] = None,
    root_path: Optional[object_utils.KeyPath] = None,
    view_id: str = 'html-tree-view',
    content_only: bool = False,
    **kwargs
) -> str:
  """Returns a HTML str for a value.

  Args:
    value: The value to render.
    name: The name of the value.
    root_path: The root path of the value.
    view_id: The ID of the view to render the value.
      See `pg.views.HtmlView.dir()` for all available HTML view IDs.
    content_only: If True, only the content will be returned.
    **kwargs: Additional keyword arguments passed from `pg.to_html`, wich
        will be passed to the `HtmlView.render_xxx()` (thus
        `Extension._html_xxx()`) methods.

  Returns:
    The rendered HTML str.
  """
  return to_html(
      value,
      name=name,
      root_path=root_path,
      view_id=view_id,
      **kwargs
  ).to_str(content_only=content_only)

