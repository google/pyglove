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
"""HTML conversion."""

import abc
import contextlib
import copy as copy_lib
import dataclasses
import functools
import html as html_lib
import inspect
import io
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Union

import jinja2
from jinja2 import meta as jinja2_meta

from pyglove.core.object_utils import formatting
from pyglove.core.object_utils import thread_local
from pyglove.core.object_utils import value_location


KeyPath = value_location.KeyPath


class Html(formatting.Formattable):
  """HTML with hierarchical generation with consolidated CSS and Scripts.

  This utility manages the structured creation and consumption of HTML, ensuring
  that CSS and JavaScript are included only once in the `<head>`, even when
  rendering nested HTML.

  1. **Using `Html.write` for nested Html objects**:

   Pass an `Html` object from a sub-component to the parent writer 
   to merge their CSS and JavaScript:

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

  class _SharedParts:
    """A type of shared parts of the HTML."""

    def __init__(self, *parts: Union[str, 'Html._SharedParts']) -> None:
      self._parts: Dict[str, int] = {}
      self._children: Dict[Html._SharedParts, int] = {}
      self.add(*parts)

    def add(
        self,
        *parts: Union[str, 'Html._SharedParts'],
    ) -> bool:
      """Adds one or multiple parts."""
      updated = False
      for part in parts:
        if isinstance(part, str):
          container = self._parts
        else:
          assert isinstance(part, Html._SharedParts), part  # pylint: disable=protected-access
          container = self._children

        if part not in container:
          container[part] = 1
          updated = True
        else:
          container[part] += 1
      if updated:
        self.__dict__.pop('parts', None)
      return updated

    def __bool__(self) -> bool:
      """Returns True if there is any part."""
      return bool(self._parts) or bool(self._children)

    @functools.cached_property
    def parts(self) -> Dict[str, int]:
      """Iterates all parts and their reference counts."""
      parts = {}
      for part, num_refs in self._parts.items():
        if part in parts:
          parts[part] += num_refs
        else:
          parts[part] = num_refs

      for child in self._children:
        for part, num_refs in child.parts.items():    # pytype: disable=attribute-error
          if part in parts:
            parts[part] += num_refs
          else:
            parts[part] = num_refs
      return parts

    def __contains__(self, part: str) -> bool:
      """Returns True if there is a part."""
      if part in self._parts:
        return True
      for child in self._children:
        if part in child:
          return True
      return False

    def __repr__(self) -> str:
      """Returns the representation."""
      return ('SharedParts('
              + formatting.format(self.parts, compact=True)
              + ')')

  def __init__(
      self,
      content: Optional[str] = None,
      *,
      style_files: Optional[Iterable[str]] = None,
      styles: Optional[Iterable[str]] = None,
      script_files: Optional[Iterable[str]] = None,
      scripts: Optional[Iterable[str]] = None,
  ):
    self._style_files = Html._SharedParts(*(style_files or []))
    self._styles = Html._SharedParts(*(styles or []))
    self._script_files = Html._SharedParts(*(script_files or []))
    self._scripts = Html._SharedParts(*(scripts or []))
    self._children = [content] if content else []

  def _repr_html_(self) -> str:
    return self.html_str()

  @property
  def styles(self) -> List[str]:
    """Returns the styles to include in the HTML."""
    return list(self._styles.parts.keys())

  @property
  def style_files(self) -> List[str]:
    """Returns the style files to link to."""
    return list(self._style_files.parts.keys())

  @property
  def scripts(self) -> List[str]:
    """Returns the scripts to include in the HTML."""
    return list(self._scripts.parts.keys())

  @property
  def script_files(self) -> List[str]:
    """Returns the script files to link to."""
    return list(self._script_files.parts.keys())

  @functools.cached_property
  def body_content(self) -> str:
    """Returns the content."""
    s = io.StringIO()
    for child in self._children:
      if isinstance(child, str):
        s.write(child)
      else:
        assert isinstance(child, Html), child
        s.write(child.body_content)
    return s.getvalue()

  @property
  def head_section(self) -> str:
    """Returns the head section."""
    s = io.StringIO()
    s.write('<head>\n')
    s.write(self.style_section)
    s.write(self.script_section)
    s.write('</head>\n')
    return s.getvalue()

  @functools.cached_property
  def style_section(self) -> str:
    """Returns the style section."""
    s = io.StringIO()
    for url in self._style_files.parts:
      s.write(f'<link rel="stylesheet" href="{url}">\n')

    if self._styles:
      s.write('<style>')
      for style in self._styles.parts:
        s.write('\n')
        s.write(inspect.cleandoc(style))
      s.write('\n</style>\n')
    return s.getvalue()

  @functools.cached_property
  def script_section(self) -> str:
    """Returns the script section."""
    s = io.StringIO()
    for url in self._script_files.parts:
      s.write(f'<script src="{url}"></script>\n')

    if self._scripts:
      s.write('<script>')
      for script in self._scripts.parts:
        s.write('\n')
        s.write(inspect.cleandoc(script))
      s.write('\n</script>\n')
    return s.getvalue()

  @property
  def body_section(self) -> str:
    """Returns the body section."""
    return '<body>\n' + self.body_content + '\n</body>\n'

  def add_style(self, *css: str) -> 'Html':
    """Adds styles to to the HTML.

    Args:
      *css: CSS styles to add. Each item is a CSS block.

    Returns:
      The current HTML.
    """
    if self._styles.add(*css):
      self.__dict__.pop('style_section', None)
    return self

  def include_style(self, *urls: str) -> 'Html':
    """Includes external styles in the HTML.

    Args:
      *urls: URLs for external styles to include.

    Returns:
      The current HTML.
    """
    if self._style_files.add(*urls):
      self.__dict__.pop('style_section', None)
    return self

  def add_script(self, *js: str, local: bool = False) -> 'Html':
    """Add scripts to the HTML.

    Args:
      *js: scripts to add.
      local: Whether to add the script on the spot of the content.

    Returns:
      The current HTML.
    """
    if not js:
      return self
    if local:
      self._children.append(
          '<script>\n' + '\n'.join(
              [inspect.cleandoc(v) for v in js]
          ) + '\n</script>'
      )
      self.__dict__.pop('body_content', None)
    elif self._scripts.add(*js):
      self.__dict__.pop('script_section', None)
    return self

  def include_script(
      self,
      *urls: str
  ) -> 'Html':
    """Includes external scripts in the HTML.

    Args:
      *urls: URLs for external scripts to include.

    Returns:
      The current HTML.
    """
    if self._script_files.add(*urls):
      self.__dict__.pop('script_section', None)
    return self

  def add(
      self,
      *parts: Union[str, 'Html'],
      shared_parts_only: bool = False
  ) -> 'Html':
    """Writes a HTML part to current HTML."""
    if not shared_parts_only and parts:
      self._children.extend(parts)
      self.__dict__.pop('body_content', None)

    invalidate_style_section = False
    invalidate_script_section = False

    for part in parts:
      # We need to access protected members of `Html` to collect shared parts.
      # pylint: disable=protected-access
      if isinstance(part, Html):
        self._styles.add(part._styles)
        self._style_files.add(part._style_files)

        if not invalidate_style_section and (
            part._styles or part._style_files):
          invalidate_style_section = True

        self._scripts.add(part._scripts)
        self._script_files.add(part._script_files)

        if not invalidate_script_section and (
            part._scripts or part._script_files):
          invalidate_script_section = True
      # pylint: enable=protected-access
    if invalidate_style_section:
      self.__dict__.pop('style_section', None)

    if invalidate_script_section:
      self.__dict__.pop('script_section', None)
    return self

  def __add__(self, other: Union[str, 'Html']) -> 'Html':
    """Concatenates two HTMLs."""
    s = copy_lib.deepcopy(self)
    s.add(other)
    return s

  def __radd__(self, other: Union[str, 'Html']) -> 'Html':
    """Right-hand concatenates two HTMLs."""
    s = Html.from_value(other, copy=True)
    assert s is not None
    s.add(self)
    return s

  def html_str(
      self,
      *,
      content_only: bool = False,
  ) -> str:
    """Returns the generated HTML.

    Args:
      content_only: If True, only the content will be returned.

    Returns:
      The generated HTML str.
    """
    if content_only:
      return self.body_content

    # We first access `body_section` compute content, which causes the styles/
    # scripts to be pulled during the computation of `content` when
    # `_trace_children_through_function_call`` is active.
    body_section = self.body_section

    s = io.StringIO()
    s.write('<html>\n')
    s.write(self.head_section)
    s.write(body_section)
    s.write('</html>\n')
    return s.getvalue()

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             trace_children: bool = False,
             content_only: bool = False,
             **kwargs) -> str:
    """Formats the HTML object."""
    if compact:
      return formatting.kvlist_str(
          [
              ('body_content', self.body_content, ''),
              ('style_files', self.style_files, []),
              ('styles', self.styles, []),
              ('script_files', self.script_files, []),
              ('scripts', self.scripts, []),
          ],
          label='Html',
          compact=True,
          bracket_type=formatting.BracketType.ROUND,
      )
    if trace_children:
      with self._trace_children_through_function_call():
        return self.html_str(content_only=content_only)
    return self.html_str(content_only=content_only)

  @contextlib.contextmanager
  def _trace_children_through_function_call(self) -> Iterator[None]:
    """Context manager for setting the current HTML as the parent."""
    parent: Optional[Html] = thread_local.thread_local_get(
        _TLS_KEY_PARENT_HTML_OBJECT, None
    )
    if parent is not None and parent is not self:
      # Reset referenced shared parts.
      parent.add(self, shared_parts_only=True)

    thread_local.thread_local_set(_TLS_KEY_PARENT_HTML_OBJECT, self)
    try:
      yield
    finally:
      if parent is None:
        thread_local.thread_local_del(_TLS_KEY_PARENT_HTML_OBJECT)
      else:
        thread_local.thread_local_set(_TLS_KEY_PARENT_HTML_OBJECT, parent)

  def _invalidate_children_from_function_call(self) -> None:
    """Subclass can override."""

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Html):
      return False
    return (
        self.body_content == other.body_content
        and self.style_files == other.style_files
        and self.styles == other.styles
        and self.script_files == other.script_files
        and self.scripts == other.scripts
    )

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.html_str())

  @classmethod
  def from_value(
      cls,
      html: Union[str, 'Html', None],
      copy: bool = False
  ) -> Union['Html', None]:
    """Returns an `Html` object from a value."""
    if html is None:
      return None
    if isinstance(html, Html):
      if copy:
        return copy_lib.deepcopy(html)
      return html
    if isinstance(html, str):
      return cls(content=html)
    raise TypeError(f'Unsupported value type: {type(html)}')


_TLS_KEY_PARENT_HTML_OBJECT = '__parent_html_object__'


_MISSING_VAR = (None,)


@dataclasses.dataclass
class HtmlComponent(Html):
  """Jinja2-based HTML component."""

  # Jinja2-based template string for render the component in HTML.
  HTML = None

  # Shared CSS styles for the component.
  STYLES = []

  # Shared CSS style files for the component.
  STYLE_FILES = []

  # Shared scripts for the component.
  SCRIPTS = []

  # Shared script files for the component.
  SCRIPT_FILES = []

  def __init_subclass__(cls):
    return dataclasses.dataclass(cls)

  def __post_init__(self):
    super().__init__(
        styles=self.STYLES,
        style_files=self.STYLE_FILES,
        scripts=self.SCRIPTS,
        script_files=self.SCRIPT_FILES,
    )
    referred_vars = dict()
    for var_name in self.var_names():
      v = getattr(self, var_name, _MISSING_VAR)
      if v is _MISSING_VAR:
        raise ValueError(f'Missing variable {var_name!r} for {self!r}.')
      referred_vars[var_name] = v
    self._referred_vars = referred_vars

  @classmethod
  @functools.cache
  def _template(cls) -> jinja2.Template:
    assert isinstance(cls.HTML, str), cls.HTML
    return jinja2.Template(cls.HTML)

  @classmethod
  @functools.cache
  def var_names(cls) -> Set[str]:
    if isinstance(cls.HTML, str) and not cls.HTML:
      raise TypeError(
          f'Class variable `HTML` must be a non-empty string for {cls!r}'
      )
    try:
      return jinja2_meta.find_undeclared_variables(
          jinja2.Environment().parse(cls.HTML)
      )
    except jinja2.TemplateSyntaxError as e:
      raise ValueError(f'Bad template string:\n\n{cls.HTML}') from e

  def add(
      self,
      *parts,
      shared_parts_only: bool = False
  ) -> Html:
    if not shared_parts_only:
      raise ValueError(
          'Adding content through `HtmlComponent.add` not supported. '
          'Use `HTML` to write content with child components instead.'
      )
    return super().add(*parts, shared_parts_only=shared_parts_only)

  @functools.cached_property
  def body_content(self) -> str:
    with self._trace_children_through_function_call():
      with formatting.str_format(trace_children=True, content_only=True):
        return self._template().render(**self._referred_vars)


# pylint: disable=unnecessary-lambda


class HtmlView(metaclass=abc.ABCMeta):
  """Base class for HTML views.

  The same value can be rendered in different HTML represenations. Each
  representation is a `HtmlView`. Users can register a view by subclassing this
  class and adding a class variable `VIEW_TYPE` to be the name of the view.
  Then the view can be created by calling `HtmlView.get(VIEW_TYPE)`.
  """

  VIEW_TYPE = None

  @dataclasses.dataclass
  class TooltipSetting:
    """Settings for the tooltip.

    Attributes:
      enable_tooltip: Whether to enable the tooltip.
    """
    enable_tooltip: bool = True

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> 'HtmlView.TooltipSetting':
      return cls(
          enable_tooltip=kwargs.get('enable_tooltip', True),
      )

  @dataclasses.dataclass
  class SummarySetting:
    """Settings for object summary.

    Attributes:
      max_str_len: The max length of a string value to be shown in the summary.
      enable_summary: If True, summary will be shown. If False, summary will be
        hidden. If None, summary will be shown if the value is not a primitive
        type.
      tooltip: Settings for the tooltip.
    """
    enable_summary: Optional[bool] = None
    max_str_len: int = 40
    tooltip: 'HtmlView.TooltipSetting' = dataclasses.field(
        default_factory=lambda: HtmlView.TooltipSetting()   # pytype: disable=name-error
    )

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> 'HtmlView.SummarySetting':
      return cls(
          enable_summary=kwargs.get('enable_summary', None),
          max_str_len=kwargs.get('max_summary_len_for_str', 40),
          tooltip=HtmlView.TooltipSetting.from_kwargs(kwargs),
      )

  @dataclasses.dataclass
  class KeySetting:
    """Settings for object key.

    Attributes:
      tooltip: Settings for the tooltip.
    """
    tooltip: 'HtmlView.TooltipSetting' = dataclasses.field(
        default_factory=lambda: HtmlView.TooltipSetting()   # pytype: disable=name-error
    )

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> 'HtmlView.KeySetting':
      return cls(
          tooltip=HtmlView.TooltipSetting.from_kwargs(kwargs),
      )

  @dataclasses.dataclass
  class ContentSetting:
    """Setting for rendering object content.

    Attributes:
      child_key: Settings for rendering child keys.
      child_value: Settings for rendering child values.
      collapsing: Settings for collapsing the object.
    """

    @dataclasses.dataclass
    class ChildKey:
      """Settings for object keys."""
      special_keys: List[Union[int, str]] = dataclasses.field(
          default_factory=list
      )

      exclude_keys: Set[Union[int, str]] = dataclasses.field(
          default_factory=set
      )

      include_keys: Optional[Set[Union[int, str]]] = None

      setting: 'HtmlView.KeySetting' = dataclasses.field(
          default_factory=lambda: HtmlView.KeySetting()   # pytype: disable=name-error
      )

      @classmethod
      def from_kwargs(
          cls, kwargs: Dict[str, Any]
      ) -> 'HtmlView.ContentSetting.ChildKey':
        include_keys = kwargs.get('include_keys', None)
        return cls(
            special_keys=list(kwargs.get('special_keys', [])),
            exclude_keys=set(kwargs.get('exclude_keys', [])),
            include_keys=set(
                include_keys) if include_keys is not None else None,
            setting=HtmlView.KeySetting.from_kwargs(kwargs),
        )

    @dataclasses.dataclass
    class ChildValue:
      """Settings for object values."""
      hide_frozen: bool = True
      hide_default_values: bool = False
      use_inferred: bool = True

      @classmethod
      def from_kwargs(
          cls, kwargs: Dict[str, Any]
      ) -> 'HtmlView.ContentSetting.ChildValue':
        return cls(
            hide_frozen=kwargs.get('hide_frozen', True),
            hide_default_values=kwargs.get('hide_default_values', False),
            use_inferred=kwargs.get('use_inferred', True),
        )

    @dataclasses.dataclass
    class Collapsing:
      """Settings for collapsing the object.

      Attributes:
        level: The level of the tree at which collapse starts.
        unless: Key paths relative to the root path to uncollapse.
      """

      level: Optional[int] = 1
      unless: Set[KeyPath] = dataclasses.field(default_factory=set)

      def should_collapse(self, value: Any, root_path: KeyPath) -> bool:
        del value
        if self.level is None:
          return False
        if root_path.depth < self.level:
          return False
        if root_path in self.unless:
          return False
        return True

      @classmethod
      def from_kwargs(
          cls,
          kwargs: Dict[str, Any],
          root_path: KeyPath
      ) -> 'HtmlView.ContentSetting.Collapsing':
        return cls(
            level=kwargs.get('collapse_level', 1),
            unless=cls.normalize_paths(
                kwargs.get('uncollapse', []), root_path
            ),
        )

      @classmethod
      def normalize_paths(
          cls,
          paths: Iterable[Union[KeyPath, str]],
          root_path: KeyPath,
      ) -> Set[KeyPath]:
        """Normalizes key paths."""
        normalized_paths = set()
        for path in paths:
          path = KeyPath.from_value(path)
          if not path.is_relative_to(root_path):
            raise ValueError(
                f'Uncollapse path {path!r} is not relative to {root_path!r}.'
            )
          normalized_paths.add(path)
          while True:
            path = path.parent
            normalized_paths.add(path)
            if not path:
              break
        return normalized_paths

    child_key: ChildKey = dataclasses.field(
        default_factory=lambda: HtmlView.ContentSetting.ChildKey()   # pytype: disable=name-error
    )
    child_value: ChildValue = dataclasses.field(
        default_factory=lambda: HtmlView.ContentSetting.ChildValue()   # pytype: disable=name-error
    )
    collapsing: Collapsing = dataclasses.field(
        default_factory=lambda: HtmlView.ContentSetting.Collapsing()   # pytype: disable=name-error
    )

    @classmethod
    def from_kwargs(
        cls, kwargs: Dict[str, Any], root_path: KeyPath
    ) -> 'HtmlView.ContentSetting':
      return cls(
          child_key=HtmlView.ContentSetting.ChildKey.from_kwargs(kwargs),
          child_value=HtmlView.ContentSetting.ChildValue.from_kwargs(kwargs),
          collapsing=HtmlView.ContentSetting.Collapsing.from_kwargs(
              kwargs, root_path
          ),
      )

  @dataclasses.dataclass
  class ViewSetting:
    """Settings for the object view.

    Attributes:
      summary: Settings for the summary.
      content: Settings for the content.
    """
    summary: 'HtmlView.SummarySetting' = dataclasses.field(
        default_factory=lambda: HtmlView.SummarySetting()   # pytype: disable=name-error
    )
    content: 'HtmlView.ContentSetting' = dataclasses.field(
        default_factory=lambda: HtmlView.ContentSetting()   # pytype: disable=name-error
    )

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Dict[str, Any],
        root_path: KeyPath
    ) -> 'HtmlView.ViewSetting':
      """Returns a setting from kwargs."""
      return cls(
          summary=HtmlView.SummarySetting.from_kwargs(kwargs),
          content=HtmlView.ContentSetting.from_kwargs(kwargs, root_path),
      )

  def __init_subclass__(cls):
    super().__init_subclass__()
    if cls.VIEW_TYPE is not None:
      _VIEW_REGISTRY[cls.VIEW_TYPE] = cls

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def user_kwargs(self, **kwargs):
    """Returns the custom kwargs passed by the user."""
    user_kwargs = self.kwargs.copy()
    user_kwargs.update(kwargs)
    return user_kwargs

  @staticmethod
  def get(name: str, **kwargs) -> 'HtmlView':
    if name not in _VIEW_REGISTRY:
      raise ValueError(f'Unknown view type: {name!r}')
    return _VIEW_REGISTRY[name](**kwargs)

  @abc.abstractmethod
  def render(
      self,
      value: Any,
      *,
      name: Optional[str],
      root_path: Optional[KeyPath],
      setting: ViewSetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Html:
    """Renders the entire HTML view for a value."""

  @abc.abstractmethod
  def render_summary(
      self,
      value: Any,
      *,
      name: Optional[str],
      title: Union[str, Html, None],
      title_class: Optional[str],
      root_path: Optional[KeyPath],
      setting: SummarySetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Optional[Html]:
    """Renders a summary for the value."""

  @abc.abstractmethod
  def render_content(
      self,
      value: Any,
      *,
      name: Optional[str],
      root_path: Optional[KeyPath],
      setting: ViewSetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Html:
    pass

  @abc.abstractmethod
  def render_key(
      self,
      key: Union[str, int],
      *,
      value: Any,
      root_path: Optional[KeyPath],
      setting: KeySetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Html:
    pass

  @abc.abstractmethod
  def render_tooltip(
      self,
      value: Any,
      *,
      name: Optional[str],
      root_path: Optional[KeyPath],
      setting: TooltipSetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Optional[Html]:
    """Renders a tooltip for the value."""


_VIEW_REGISTRY = {}


class HtmlFormattable:
  """Base class for objects that can be formatted as HTML.

  By subclassing this class, the object will be able to render HTML
  formatting. Users could override class variables `__css__` and `__js__` to add
  custom CSS and JavaScript. Users could also override `__html_post_render__`
  to add custom post-rendering logic. Three methods are provided for users to
  override the summary, content and tooltip of the object.
  """

  def _repr_html_(self) -> str:
    return str(self.to_html())

  def _html(
      self,
      *,
      name: Optional[str],
      root_path: KeyPath,
      view: HtmlView,
      setting: HtmlView.ViewSetting,
      **kwargs,
  ) -> Html:
    """Returns the topmost HTML representation of the object.

    Args:
      name: The name of the object.
      root_path: The key path of the object relative to the root.
      view: The view to render the object.
      setting: Settings for the object view.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML.
    """
    return view.render(
        self, name=name, root_path=root_path,
        setting=setting, **kwargs,
    )

  def _html_summary(
      self,
      *,
      name: Optional[str],
      root_path: KeyPath,
      view: HtmlView,
      setting: HtmlView.SummarySetting,
      **kwargs,
  ) -> Optional[Html]:
    """Returns the HTML representation of the object.

    Args:
      name: The name of the object.
      root_path: The key path of the object relative to the root.
      view: The view to render the object.
      setting: Settings for rendering the summary ofthe object.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      An optional HTML object representing the summary of the object. If None,
      the summary will be hidden.
    """
    return view.render_summary(
        self, name=name, title=None, title_class=None,
        root_path=root_path, setting=setting, **kwargs,
    )

  def _html_key(
      self,
      key: Union[str, int],
      *,
      value: Any,
      root_path: KeyPath,
      view: HtmlView,
      setting: HtmlView.KeySetting,
      **kwargs
  ) -> Html:
    """Returns the HTML representation of the object.

    Args:
      key: The key of the object.
      value: The value of the object.
      root_path: The key path of the object relative to the root.
      view: The view to render the object.
      setting: Settings for the key.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      A HTML object representing the key.
    """
    return view.render_key(
        key, value=value, root_path=root_path, setting=setting, **kwargs
    )

  def _html_content(
      self,
      *,
      name: Optional[str],
      root_path: KeyPath,
      view: HtmlView,
      setting: HtmlView.ViewSetting,
      **kwargs,
      ) -> Html:
    """Returns the main content for the object.

    Args:
      name: The name of the object.
      root_path: The key path of the object relative to the root.
      view: The view to render the object.
      setting: Settings for the content.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      The rendered HTML as the main content of the object.
    """
    return view.render_content(
        self, name=name, root_path=root_path, setting=setting, **kwargs,
    )

  def _html_tooltip(
      self,
      *,
      root_path: KeyPath,
      view: HtmlView,
      setting: HtmlView.TooltipSetting,
      **kwargs,
  ) -> Optional[Html]:
    """Returns the tooltip for the object.

    Args:
      root_path: The key path of the object relative to the root.
      view: The view to render the object.
      setting: Settings for the tooltip.
      **kwargs: Additional keyword arguments passed from `pg.to_html`.

    Returns:
      An optional HTML object representing the tooltip of the object. If None,
      the tooltip will be hidden.
    """
    return view.render_tooltip(
        self, root_path=root_path, setting=setting, **kwargs
    )

  def to_html(
      self,
      *,
      name: Optional[str] = None,
      root_path: Optional[KeyPath] = None,
      view: Union[str, HtmlView] = 'default',
      enable_tooltip: bool = True,
      enable_summary: Optional[bool] = None,
      max_summary_len_for_str: int = 80,
      special_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      hide_frozen: bool = True,
      hide_default_values: bool = False,
      use_inferred: bool = True,
      collapse_level: int = 1,
      uncollapse: Optional[Iterable[Union[KeyPath, str]]] = None,
      **kwargs
  ) -> Html:
    """Returns the HTML representation of the object.

    Args:
      name: The name of the object.
      root_path: The root path of the object.
      view: The view to render the object.
      enable_tooltip: If True, tooltip will be enabled.
      enable_summary: If True, summary will be enabled.
      max_summary_len_for_str: The maximum length of a string to be shown in
        summary.
      special_keys: The special keys to be shown in summary.
      exclude_keys: The keys to be excluded in summary.
      include_keys: The keys to be included in summary.
      hide_frozen: If True, frozen fields will be hidden in summary.
      hide_default_values: If True, default values will be hidden in summary.
      use_inferred: If True, inferred values will be used in summary.
      collapse_level: The collapse level of the object.
      uncollapse: The paths to be uncollapsed.
      **kwargs: Additional keyword arguments passed from `pg.to_html`, wich
        will be passed to the `HtmlView.render_xxx()` (thus
        `HtmlFormattable._html_xxx()`) methods.

    Returns:
      The rendered HTML.
    """
    return to_html(
        self,
        name=name,
        root_path=root_path,
        view=view,
        enable_tooltip=enable_tooltip,
        enable_summary=enable_summary,
        max_summary_len_for_str=max_summary_len_for_str,
        special_keys=special_keys,
        exclude_keys=exclude_keys,
        include_keys=include_keys,
        hide_frozen=hide_frozen,
        hide_default_values=hide_default_values,
        use_inferred=use_inferred,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        **kwargs
    )

  def to_html_str(
      self,
      *,
      content_only: bool = False,
      name: Optional[str] = None,
      root_path: Optional[KeyPath] = None,
      view: Union[str, HtmlView] = 'default',
      enable_tooltip: bool = True,
      enable_summary: Optional[bool] = None,
      max_summary_len_for_str: int = 80,
      special_keys: Optional[Iterable[Union[int, str]]] = None,
      exclude_keys: Optional[Iterable[Union[int, str]]] = None,
      include_keys: Optional[Iterable[Union[int, str]]] = None,
      hide_frozen: bool = True,
      hide_default_values: bool = False,
      use_inferred: bool = True,
      collapse_level: int = 1,
      uncollapse: Optional[Iterable[Union[KeyPath, str]]] = None,
      **kwargs
  ) -> str:
    """Returns the HTML str of the object.

    Args:
      content_only: If True, only the content will be returned.
      name: The name of the object.
      root_path: The root path of the object.
      view: The view to render the object.
      enable_tooltip: If True, tooltip will be enabled.
      enable_summary: If True, summary will be enabled. If None, summary will
        be enabled for complex types or when string exceeds
        `max_summary_len_for_str`.
      max_summary_len_for_str: The maximum length of a string to be shown in
        summary.
      special_keys: The special keys to be shown in summary.
      exclude_keys: The keys to be excluded in summary.
      include_keys: The keys to be included in summary.
      hide_frozen: If True, frozen fields will be hidden in summary.
      hide_default_values: If True, default values will be hidden in summary.
      use_inferred: If True, inferred values will be used in summary.
      collapse_level: The collapse level of the object.
      uncollapse: The paths to be uncollapsed.
      **kwargs: Additional keyword arguments passed from `pg.to_html`, wich
        will be passed to the `HtmlView.render_xxx()` (thus
        `HtmlFormattable._html_xxx()`) methods.

    Returns:
      The rendered HTML.
    """
    return to_html_str(
        self,
        content_only=content_only,
        name=name,
        root_path=root_path,
        view=view,
        enable_tooltip=enable_tooltip,
        enable_summary=enable_summary,
        max_summary_len_for_str=max_summary_len_for_str,
        special_keys=special_keys,
        exclude_keys=exclude_keys,
        include_keys=include_keys,
        hide_frozen=hide_frozen,
        hide_default_values=hide_default_values,
        use_inferred=use_inferred,
        collapse_level=collapse_level,
        uncollapse=uncollapse,
        **kwargs
    )


def to_html(
    value: Any,
    *,
    name: Optional[str] = None,
    root_path: Optional[KeyPath] = None,
    view: Union[str, HtmlView] = 'default',
    enable_tooltip: bool = True,
    enable_summary: Optional[bool] = None,
    max_summary_len_for_str: int = 80,
    special_keys: Optional[Iterable[Union[int, str]]] = None,
    exclude_keys: Optional[Iterable[Union[int, str]]] = None,
    include_keys: Optional[Iterable[Union[int, str]]] = None,
    hide_frozen: bool = True,
    hide_default_values: bool = False,
    use_inferred: bool = True,
    collapse_level: int = 1,
    uncollapse: Optional[Iterable[Union[KeyPath, str]]] = None,
    **kwargs
) -> Html:
  """Returns the HTML representation of a value.

  Args:
    value: The value to render.
    name: The name of the value.
    root_path: The root path of the value.
    view: The view to render the value.
    enable_tooltip: If True, tooltip will be enabled.
    enable_summary: If True, summary will be enabled. If None, summary will
      be enabled for complex types or when string exceeds
      `max_summary_len_for_str`.
    max_summary_len_for_str: The maximum length of a string to be shown in
      summary.
    special_keys: The special keys to be shown in summary.
    exclude_keys: The keys to be excluded in summary.
    include_keys: The keys to be included in summary.
    hide_frozen: If True, frozen fields will be hidden in summary.
    hide_default_values: If True, default values will be hidden in summary.
    use_inferred: If True, inferred values will be used in summary.
    collapse_level: The collapse level of the value.
    uncollapse: The paths to be uncollapsed.
    **kwargs: Additional keyword arguments passed from `pg.to_html`, wich
        will be passed to the `HtmlView.render_xxx()` (thus
        `HtmlFormattable._html_xxx()`) methods.

  Returns:
    The rendered HTML.
  """
  if isinstance(value, Html):
    return value

  if isinstance(view, str):
    view = HtmlView.get(view, **kwargs)

  root_path = root_path or KeyPath()
  setting = HtmlView.ViewSetting.from_kwargs(
      dict(
          enable_tooltip=enable_tooltip,
          enable_summary=enable_summary,
          max_summary_len_for_str=max_summary_len_for_str,
          special_keys=special_keys or [],
          exclude_keys=exclude_keys or set(),
          include_keys=include_keys,
          hide_frozen=hide_frozen,
          hide_default_values=hide_default_values,
          use_inferred=use_inferred,
          collapse_level=collapse_level,
          uncollapse=uncollapse or [],
      ),
      root_path
  )
  return view.render(
      value, name=name, root_path=root_path,
      setting=setting, prefer_user_override=True, **view.kwargs
  )


def to_html_str(
    value: Any,
    *,
    content_only: bool = False,
    name: Optional[str] = None,
    root_path: Optional[KeyPath] = None,
    view: Union[str, HtmlView] = 'default',
    enable_tooltip: bool = True,
    enable_summary: Optional[bool] = None,
    max_summary_len_for_str: int = 80,
    special_keys: Optional[Iterable[Union[int, str]]] = None,
    exclude_keys: Optional[Iterable[Union[int, str]]] = None,
    include_keys: Optional[Iterable[Union[int, str]]] = None,
    hide_frozen: bool = True,
    hide_default_values: bool = False,
    use_inferred: bool = True,
    collapse_level: int = 1,
    uncollapse: Optional[Iterable[Union[KeyPath, str]]] = None,
    **kwargs
) -> str:
  """Returns a HTML str for a value.

  Args:
    value: The value to render.
    content_only: If True, only the content will be returned.
    name: The name of the value.
    root_path: The root path of the value.
    view: The view to render the value.
    enable_tooltip: If True, tooltip will be enabled.
    enable_summary: If True, summary will be enabled. If None, summary will
      be enabled for complex types or when string exceeds
      `max_summary_len_for_str`.
    max_summary_len_for_str: The maximum length of a string to be shown in
      summary.
    special_keys: The special keys to be shown in summary.
    exclude_keys: The keys to be excluded in summary.
    include_keys: The keys to be included in summary.
    hide_frozen: If True, frozen fields will be hidden in summary.
    hide_default_values: If True, default values will be hidden in summary.
    use_inferred: If True, inferred values will be used in summary.
    collapse_level: The collapse level of the value.
    uncollapse: The paths to be uncollapsed.
    **kwargs: Additional keyword arguments passed from `pg.to_html`, wich
        will be passed to the `HtmlView.render_xxx()` (thus
        `HtmlFormattable._html_xxx()`) methods.

  Returns:
    The rendered HTML str.
  """
  return to_html(
      value,
      name=name,
      root_path=root_path,
      view=view,
      enable_tooltip=enable_tooltip,
      enable_summary=enable_summary,
      max_summary_len_for_str=max_summary_len_for_str,
      special_keys=special_keys,
      exclude_keys=exclude_keys,
      include_keys=include_keys,
      hide_frozen=hide_frozen,
      hide_default_values=hide_default_values,
      use_inferred=use_inferred,
      collapse_level=collapse_level,
      uncollapse=uncollapse,
      **kwargs
  ).html_str(content_only=content_only)


#
# Default HtmlView for PyGlove.
#


class _DefaultHtmlView(HtmlView):
  """Default HTML view for PyGlove objects."""

  VIEW_TYPE = 'default'

  @dataclasses.dataclass
  class Tooltip(HtmlComponent):
    """Tooltip.

    Attributes:
      value: The value of the object.
      root_path: The key path of the object relative to the root.
      setting: Settings for the tooltip.
    """
    value: Any
    root_path: KeyPath
    setting: HtmlView.TooltipSetting
    current_view: '_DefaultHtmlView'

    HTML = inspect.cleandoc(
        """
        <span class="tooltip">{{ text }}</span>
        """
    )
    STYLES = [
        """
        span.tooltip {
          visibility: hidden;
          white-space: pre-wrap;
          font-weight: normal;
          background-color: #808080;
          color: #fff;
          padding: 6px;
          border-radius: 6px;
          position: absolute;
          z-index: 1;
        }
        """
    ]

    def __new__(cls, *, setting: HtmlView.TooltipSetting, **kwargs):
      del kwargs
      if setting.enable_tooltip:
        return super().__new__(cls)
      return None

    @property
    def text(self) -> str:
      return html_lib.escape(
          formatting.format(
              self.value,
              compact=False, verbose=False, hide_default_values=True,
              python_format=True, use_inferred=True,
              max_bytes_len=64, max_str_len=256,
          )
      )

  @dataclasses.dataclass
  class Summary(HtmlComponent):
    """Summary.

    Attributes:
      value: The value of the object.
      name: The name of the object.
      root_path: The key path of the object relative to the root.
      setting: Settings for the summary.
    """
    value: Any
    name: Optional[str]
    title: Optional[str]
    title_class: Optional[str]
    root_path: KeyPath
    setting: HtmlView.SummarySetting
    current_view: '_DefaultHtmlView'

    HTML = inspect.cleandoc(
        """
        <summary>
        {% if name %}<div class="summary_name">{{ name }}</div>{% endif -%}
        <div class="summary_title {{summary_title_class}}">{{ summary_title }}</div>
        {{ tooltip if tooltip}}
        </summary>
        """
    )

    STYLES = [
        """
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
        .summary_title.t_Ref::before {
          content: 'ref: ';
          color: #aaa;
        }
        .summary_title:hover + span.tooltip {
          visibility: visible;
        }
        .t_str {
          color: darkred;
          font-style: italic;
        }
        .empty_container::before {
            content: '(empty)';
            font-style: italic;
            margin-left: 0.5em;
            color: #aaa;
        }
        """
    ]

    def __new__(
        cls,
        *,
        value: Any,
        name: Optional[str],
        title: Optional[str],
        setting: HtmlView.SummarySetting,
        **kwargs):
      del kwargs
      if cls.needs_summary(value, name, title, setting):
        return super().__new__(cls)
      return None

    @staticmethod
    def needs_summary(
        value: Any,
        name: Optional[str],
        title: Union[str, Html, None],
        setting: HtmlView.SummarySetting
    ) -> bool:
      """Returns whether the object should have a summary."""
      if setting.enable_summary is None:
        if name is not None or title is not None or not (
            isinstance(value, (int, float, bool, type(None)))
            or (isinstance(value, str) and len(value) <= setting.max_str_len)
        ):
          return True
      return setting.enable_summary

    @property
    def summary_title(self) -> Union[str, Html]:
      """Returns the summary title."""
      if self.title is not None:
        return self.title

      value = self.value
      if isinstance(value, str):
        if len(value) > self.setting.max_str_len:
          value = value[:self.setting.max_str_len] + '...'
        return html_lib.escape(repr(value))
      return f'{type(value).__name__}(...)'

    @property
    def summary_title_class(self) -> Optional[str]:
      if self.title_class is not None:
        return self.title_class
      return f't_{type(self.value).__name__}'

    @property
    def tooltip(self) -> Optional['_DefaultHtmlView.Tooltip']:
      """Returns the tooltip."""
      if not self.setting.tooltip.enable_tooltip:
        return None
      return self.current_view.render_tooltip(
          value=self.value,
          root_path=self.root_path,
          setting=self.setting.tooltip,
          prefer_user_override=True,
      )

  @dataclasses.dataclass
  class Key(HtmlComponent):
    """Object key.

    Attributes:
      key: The key of the object.
      value: The value of the object.
      root_path: The key path of the object relative to the root.
      setting: Settings for the key.
    """

    key: Union[int, str]
    value: Any
    root_path: KeyPath
    setting: HtmlView.KeySetting
    current_view: '_DefaultHtmlView'

    HTML = inspect.cleandoc(
        """
        <span class="object_key k_{{key_type}} v_{{value_type}}"
        {%- if key_color %} style="background-color: {{key_color}}"
        {%- endif %}>{{ key }}</span>
        {{ tooltip if tooltip-}}
        """
    )

    STYLES = [
        """
        span.object_key {
          margin-right: 0.25em;
        }
        .k_str{
          color: white;
          background-color: #ccc;
          border-radius: 0.2em;
          padding: 0.3em;
        }
        .k_int{
            color: #aaa;
        }
        .k_int::before{
            content: '[';
        }
        .k_int::after{
            content: ']';
        }
        span.object_key:hover + span.tooltip {
          visibility: visible;
          background-color: darkblue;
        }
        """
    ]

    @property
    def key_type(self) -> str:
      return type(self.key).__name__

    @property
    def value_type(self) -> str:
      return type(self.value).__name__

    @property
    def key_color(self) -> Optional[str]:
      return None

    @property
    def tooltip(self) -> Optional['_DefaultHtmlView.Tooltip']:
      return self.current_view.render_tooltip(
          value=self.root_path,
          root_path=self.root_path,
          setting=self.setting.tooltip,
          prefer_user_override=True
      )

  @dataclasses.dataclass
  class Content(HtmlComponent):
    """Object content.

    Attributes:
      value: The value of the object.
      root_path: The key path of the object relative to the root.
      setting: Settings for the content.
    """

    value: Any
    name: Optional[str]
    root_path: KeyPath
    setting: HtmlView.ViewSetting
    current_view: '_DefaultHtmlView'

    HTML = '{{value_html}}'

    @property
    def value_html(self) -> Html:
      """Returns the HTML for the object."""
      value = self.value
      if isinstance(value, (int, float, bool, str, type(None))):
        return self.simple_value(value)
      if isinstance(value, list):
        return self.tree_value({i: v for i, v in enumerate(value)})
      if isinstance(value, dict):
        return self.tree_value(value)
      return self.simple_value(value)

    def tree_value(self, kv: Dict[Union[int, str], Any]) -> Html:
      s = Html()
      if kv:
        ksetting = self.setting.content.child_key
        include_keys = set(ksetting.include_keys or kv.keys())
        if ksetting.exclude_keys:
          include_keys -= ksetting.exclude_keys

        if ksetting.special_keys:
          for k in ksetting.special_keys:
            if k in kv and k in include_keys:
              s.add(
                  self.current_view.render(
                      value=kv[k],
                      name=k,
                      root_path=self.root_path + k,
                      setting=self.setting,
                      prefer_user_override=True,
                  )
              )
              include_keys.remove(k)

        if include_keys:
          s.add('<div>')
          s.add('<table>')
          for k, v in kv.items():
            if k not in include_keys:
              continue
            child_path = self.root_path + k
            key = self.current_view.render_key(
                key=k, value=v, root_path=child_path,
                setting=self.setting.content.child_key.setting,
                prefer_user_override=True,
            )
            value = self.current_view.render(
                value=v,
                name=None,
                root_path=child_path,
                setting=self.setting,
                prefer_user_override=True,
            )
            s.add(
                '<tr><td>', key, '</td><td>', value, '</td></tr>'
            )
          s.add('</table>')
          s.add('</div>')
      else:
        s.add('<span class="empty_container"></span>')
      return s

    def simple_value(self, value: Any) -> Html:
      """Returns the HTML for a simple value."""
      vclass = f'v_{type(value).__name__}'
      if isinstance(value, str):
        if len(value) < self.setting.summary.max_str_len:
          value = repr(value)
      else:
        value = formatting.format(
            value,
            compact=False, verbose=False, hide_default_values=True,
            python_format=True, use_inferred=True,
            max_bytes_len=64, max_str_len=256,
        )
      value = html_lib.escape(value)
      x = Html(
          f'<span class="simple_value {vclass}">{value}</span>',
          styles=[
              """
              .simple_value {
                color: blue;
                display: inline-block;
                white-space: pre-wrap;
                padding: 0.2em;
              }
              .simple_value.v_str {
                color: darkred;
                font-style: italic;
              }
              .simple_value.v_int, .simple_value.v_float {
                color: darkblue;
              }
              """
          ]
      )
      return x

  @dataclasses.dataclass
  class ObjectView(HtmlComponent):
    """HTML representation for an object value.

    Attributes:
      value: The value of the object.
      name: The name of the object.
      root_path: The key path of the object relative to the root.
      setting: Settings for the object.
    """

    value: Any
    name: Optional[str]
    root_path: KeyPath
    setting: HtmlView.ViewSetting
    current_view: '_DefaultHtmlView'

    HTML = inspect.cleandoc(
        """
        {% if summary -%}
        <details class="pyglove{{ css_classes }}"{{ " open" if should_open }}>
        {{ summary }}
        {{ content }}
        </details>
        {% else -%}
        {{ content}}
        {% endif -%}
        """
    )

    STYLES = [
        """
        details.pyglove {
          border: 1px solid #aaa;
          border-radius: 4px;
          padding: 0.5em 0.5em 0;
          margin: 0.5em 0;
        }
        details.pyglove[open] {
          padding: 0.5em 0.5em 0.5em;
        }
        """
    ]

    @property
    def css_classes(self) -> str:
      if isinstance(self.value, HtmlFormattable):
        return f' {type(self.value).__name__}'
      return ''

    @property
    def summary(self) -> '_DefaultHtmlView.Summary':
      x = self.current_view.render_summary(
          value=self.value,
          name=self.name,
          title=None,
          title_class=None,
          root_path=self.root_path,
          setting=self.setting.summary,
          prefer_user_override=True,
      )
      return x

    @property
    def content(self) -> Optional['_DefaultHtmlView.Content']:
      return self.current_view.render_content(
          value=self.value,
          name=self.name,
          root_path=self.root_path,
          setting=self.setting,
          prefer_user_override=True,
      )

    @property
    def should_open(self) -> bool:
      return not self.setting.content.collapsing.should_collapse(
          self.value, self.root_path
      )

  def render(
      self,
      value: Any,
      *,
      name: Optional[str],
      root_path: Optional[KeyPath],
      setting: HtmlView.ViewSetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Html:
    """Renders the entire HTML view for a value."""
    if prefer_user_override and isinstance(value, HtmlFormattable):
      return value._html(  # pylint: disable=protected-access
          name=name, root_path=root_path, setting=setting, view=self,
          **self.user_kwargs(**kwargs)
      )
    return _DefaultHtmlView.ObjectView(
        value=value, name=name, root_path=root_path,
        setting=setting, current_view=self
    )

  def render_summary(
      self,
      value: Any,
      *,
      name: Optional[str],
      title: Union[str, Html, None],
      title_class: Optional[str],
      root_path: Optional[KeyPath],
      setting: HtmlView.SummarySetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Optional[Html]:
    """Renders a summary for the value."""
    if prefer_user_override and isinstance(value, HtmlFormattable):
      return value._html_summary(   # pylint: disable=protected-access
          name=name, root_path=root_path, setting=setting, view=self,
          **self.user_kwargs(**kwargs)
      )
    return _DefaultHtmlView.Summary(
        value=value, name=name, title=title, title_class=title_class,
        root_path=root_path, setting=setting, current_view=self
    )

  def render_content(
      self,
      value: Any,
      *,
      name: Optional[str],
      root_path: Optional[KeyPath],
      setting: HtmlView.ViewSetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Html:
    if prefer_user_override and isinstance(value, HtmlFormattable):
      return value._html_content(   # pylint: disable=protected-access
          name=name, root_path=root_path, setting=setting, view=self,
          **self.user_kwargs(**kwargs)
      )
    return _DefaultHtmlView.Content(
        value=value, name=name, root_path=root_path,
        setting=setting, current_view=self
    )

  def render_key(
      self,
      key: Union[str, int],
      *,
      value: Any,
      root_path: Optional[KeyPath],
      setting: HtmlView.KeySetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Html:
    if prefer_user_override and isinstance(value, HtmlFormattable):
      return value._html_key(   # pylint: disable=protected-access
          key=key, value=value, root_path=root_path, setting=setting,
          view=self, **self.user_kwargs(**kwargs)
      )
    return _DefaultHtmlView.Key(
        key=key, value=value, root_path=root_path,
        setting=setting, current_view=self
    )

  def render_tooltip(
      self,
      value: Any,
      *,
      root_path: Optional[KeyPath],
      setting: HtmlView.TooltipSetting,
      prefer_user_override: bool = False,
      **kwargs
  ) -> Optional[Html]:
    if prefer_user_override and isinstance(value, HtmlFormattable):
      return value._html_tooltip(   # pylint: disable=protected-access
          root_path=root_path, setting=setting, view=self,
          **self.user_kwargs(**kwargs)
      )
    return _DefaultHtmlView.Tooltip(
        value=value, root_path=root_path,
        setting=setting, current_view=self
    )

# pylint: enable=unnecessary-lambda
