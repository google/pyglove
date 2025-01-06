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
"""Interface of PyGlove's view system: Content, View, and Extension.

PyGlove introduces a flexible and extensible view system for rendering objects
in different formats, such as text or HTML. This system allows users to plug in
various predefined views or create their own custom views with minimal effort.
The two core concepts of this system are ``View`` and ``Extension``.

A ``View`` defines how objects are displayed, with specific classes like
``HtmlView`` and more specialized subclasses such as ``HtmlTreeView``. Each
concrete ``View`` class has a ``VIEW_ID`` that can be used to select the view
when calling functions like ``pg.view`` or ``pg.to_html``. Views provide default
rendering behavior for built-in Python types such as ``int``, ``str``, ``bool``,
``list``, and ``dict``. However, users can extend this behavior by subclassing
the ``View.Extension`` class to define custom rendering logic for user-defined
classes.

.. code-block::

     pg.to_html
         |
         | (calls)
         v
      pg.view
         |
         | (refers)
         v
       pg.View --------------------------------> pg.View.Extension
         ^                   (calls)                     ^
         | (extends)                                     |
    +-----------+                                        |
    |           |                                        |
   ...   pg.views.HtmlView   ---------------------> pg.views.HtmlView.Extension
                ^             (calls)                    ^
                | (extends)                              |  (extends)
    pg.views.HtmlTreeView  ------------------> pg.views.HtmlTreeView.Extension
                              (calls)                 ^
                                                      |
                                                  pg.Symbolic
                                                      ^
                                                      |
                                                   pg.Object

Each ``View`` class also defines an inner ``Extension`` class that manages
interactions with user-defined types. The ``Extension`` class typically
implements several methods chosen by the ``View`` developer, such as
``_html_tree_view_render``, ``_html_tree_view__summary``, etc., depending on
the specific renderin needs.
The ``View.extension_method(<extension_method_name>)`` decorator enables
flexible mapping between a ``View`` method and its corresponding ``Extension``
method. This allows users to define custom rendering logic within their
``Extension`` class and bind it to the view.

An example of introducing a new view is described below:

.. code-block:: python

  class MyView(pg.View):
    VIEW_ID = 'my_view'

    class Extension(pg.View.Extension):
      def _myview_render(self, **kwargs):
        return ...

      def _myview_part1(self, **kwargs):
        return ...

      def _myview_part2(self, **kwargs):
        return ...

    @pg.View.extension_method('_myview_render')
    def render(self, value: Any, **kwargs):
      part1 = self.render_part1(value.part1, **kwargs)
      part2 = self.render_part2(value.part2, **kwargs)
      return ...

    @pg.View.extension_method('_myview_part1')
    def render_part1(self, value: Any, **kwargs):
      return ...

    @pg.View.extension_method('_myview_part2')
    def render_part2(self, value: Any, **kwargs):
      return ...

The ``View.extension_method`` decorator also supports multiple views for a
single user class by allowing multi-inheritance of ``Extension`` classes.
For example, a class can support both ``View1`` and ``View2`` by inheriting
their respective ``Extension`` classes and implementing methods like
``_render_view1`` and ``_render_view2`` to handle different view-specific
rendering:

.. code-block:: python

  class MyObject(MyView1.Extension, MyView2.Extension):
    def _myview1_render(self, value, **kwargs):
      return ...

    def _myview2_render(self, value, **kwargs):
      return ...

For ``extension_method``-decorated ``View`` methods, they must include a
``value`` argument, which is the target object to be rendered. The view will
delegate rendering to the user logic only when the ``value`` is an instance of
the ``Extension`` class.

`pg.view` is the function to display an object with a specific view. It takes
a ``view_id`` argument to specify which view to use, and returns a ``Content``
object, which acts as the media to be displayed. For example::

.. code-block:: python

  pg.view([1, 2, MyObject(...)], view_id='my_view1')

"""
import abc
import collections
import contextlib
import copy as copy_lib
import functools
import inspect
import io
import os
import types
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Set, Type, Union

from pyglove.core import io as pg_io
from pyglove.core import typing as pg_typing
from pyglove.core import utils


# Type definition for the value filter function.
NodeFilter = Callable[
    [
        utils.KeyPath,  # The path to the value.
        Any,  # Current value.
        Any,  # Parent value
    ],
    bool,  # Whether to include the value.
]


_VIEW_ARGS_PRESET_NAME = 'pyglove_view_args'
_VIEW_REGISTRY = {}
_TLS_KEY_OPERAND_STACK_BY_METHOD = '__view_operand_stack__'
_TLS_KEY_VIEW_OPTIONS = '__view_options__'


class Content(utils.Formattable, metaclass=abc.ABCMeta):
  """Content: A type of media to be displayed in a view.

  For example, `pg.Html` is a `Content` type that represents HTML to be
  displayed in an HTML view.
  """

  WritableTypes = Union[    # pylint: disable=invalid-name
      str,
      'Content',
      Callable[[], Union[str, 'Content', None]],
      None
  ]

  class SharedParts(utils.Formattable):
    """A part of the content that should appear just once.

    For example, `pg.Html.Styles` is a `SharedParts` type that represents
    a group of CSS styles that should only be included once in the HEAD section.
    """

    __slots__ = ('_parts',)

    def __init__(
        self,
        *parts: Union[str, 'Content.SharedParts', None]
    ) -> None:
      self._parts: Dict[str, int] = collections.defaultdict(int)
      self.add(*parts)

    def add(self, *parts: Union[str, 'Content.SharedParts', None]) -> bool:
      """Adds one or multiple parts."""
      updated = False
      for part in parts:
        if part is None:
          continue
        if isinstance(part, Content.SharedParts):
          assert isinstance(part, self.__class__), (part, self.__class__)
          for p, count in part.parts.items():
            self._parts[p] += count
            updated = True
        else:
          assert isinstance(part, str), part
          updated |= (part not in self._parts)
          self._parts[part] += 1

      if updated:
        self.__dict__.pop('content', None)
      return updated

    @property
    def parts(self) -> Dict[str, int]:
      """Returns all parts and their reference counts."""
      return self._parts

    def __bool__(self) -> bool:
      """Returns True if there is any part."""
      return bool(self._parts)

    def __contains__(self, part: str) -> bool:
      """Returns True if the part is in the shared parts."""
      return part in self._parts

    def __iter__(self) -> Iterator[str]:
      """Iterates all parts."""
      return iter(self._parts.keys())

    def __copy__(self) -> 'Content.SharedParts':
      """Returns a copy of the shared parts."""
      return self.__class__(self)   # pytype: disable=not-instantiable

    def __eq__(self, other: Any):
      if not isinstance(other, self.__class__):
        return False
      return set(self._parts.keys()) == set(other.parts.keys())

    def __ne__(self, other: Any):
      return not self.__eq__(other)

    def format(
        self,
        compact: bool = False,
        verbose: bool = True,
        root_indent: int = 0,
        **kwargs
    ) -> str:
      if compact:
        return utils.kvlist_str(
            [
                ('parts', self._parts, {}),
            ],
            label=self.__class__.__name__,
            compact=compact,
            verbose=verbose,
            root_indent=root_indent,
            bracket_type=utils.BracketType.ROUND,
        )
      return self.content

    @property
    @abc.abstractmethod
    def content(self) -> str:
      """Returns the content string representing the the shared parts."""

  __slots__ = ('_content_stream', '_shared_parts',)

  def __init__(
      self,
      *content: WritableTypes,
      **shared_parts: 'Content.SharedParts'
  ):
    self._content_stream = io.StringIO()
    self._shared_parts = shared_parts

    for c in content:
      c = self._to_content(c)
      if c is None:
        continue
      elif isinstance(c, str):
        self._content_stream.write(c)
      else:
        self.write(c)

  @functools.cached_property
  def content(self) -> str:
    """Returns the content."""
    return self._content_stream.getvalue()

  @property
  def shared_parts(self) -> Dict[str, 'Content.SharedParts']:
    """Returns the shared parts."""
    return self._shared_parts

  def write(
      self, *parts: WritableTypes, shared_parts_only: bool = False
  ) -> 'Content':
    """Writes one or more parts to current Content.

    Args:
      *parts: The parts to be written. Each part can be a string, a Content
        object, a callable that returns one of the above, or None.
      shared_parts_only: If True, only write the shared parts.

    Returns:
      The current Content object for chaining.
    """
    content_updated = False
    for p in parts:
      p = self._to_content(p)
      if p is None:
        continue

      if not isinstance(p, (str, self.__class__)):
        raise TypeError(
            f'{p!r} ({type(p)}) cannot be writable. '
            f'Only str, None, {self.__class__.__name__} and callable object '
            'that returns one of them are supported.'
        )

      if isinstance(p, Content):
        current = self._shared_parts
        for k, v in p.shared_parts.items():
          # Since `p` is the same type of `self`, we expect they have the
          # same set of shared parts, which is determined at the __init__ time.
          current[k].add(v)
        p = p.content

      if not shared_parts_only:
        self._content_stream.write(p)
        content_updated = True

    if content_updated:
      self.__dict__.pop('content', None)
    return self

  def save(self, file: str, **kwargs):
    """Save content to a file."""
    pg_io.mkdirs(os.path.dirname(file), exist_ok=True)
    pg_io.writefile(file, self.to_str(**kwargs))

  def __add__(self, other: WritableTypes) -> 'Content':
    """Operator +: Concatenates two Content objects."""
    other = self._to_content(other)
    if not other:
      return self
    s = copy_lib.deepcopy(self)
    s.write(other)
    return s

  def __radd__(self, other: WritableTypes) -> 'Content':
    """Right-hand operator +: concatenates two Content objects."""
    s = self.from_value(other, copy=True)
    if s is None:
      return self
    s.write(self)
    return s

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             content_only: bool = False,
             **kwargs) -> str:
    """Formats the Content object."""
    del kwargs
    if compact:
      return utils.kvlist_str(
          [
              ('content', self.content, ''),
          ]
          + [(k, v, None) for k, v in self._shared_parts.items()],
          label=self.__class__.__name__,
          compact=compact,
          verbose=verbose,
          root_indent=root_indent,
          bracket_type=utils.BracketType.ROUND,
      )
    return self.to_str(content_only=content_only)

  @abc.abstractmethod
  def to_str(self, *, content_only: bool = False, **kwargs) -> str:
    """Returns the string representation of the content."""

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, self.__class__):
      return False
    return (
        self.content == other.content
        and self._shared_parts == other._shared_parts  # pylint: disable=protected-access
    )

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.to_str())

  @classmethod
  def from_value(
      cls,
      value: WritableTypes,
      copy: bool = False
  ) -> Union['Content', None]:
    """Returns a Content object or None from a writable type."""
    if value is None:
      return None

    if isinstance(value, Content):
      assert isinstance(value, cls), (value, cls)
      if copy:
        return copy_lib.deepcopy(value)
      return value
    return cls(value)   # pytype: disable=not-instantiable

  @classmethod
  def _to_content(cls, value: WritableTypes) -> Union['Content', str, None]:
    """Returns a Content object or None from a writable type."""
    if callable(value):
      value = value()
    if value is None:
      return None
    assert isinstance(value, (str, cls)), value
    return value


def view(
    value: Any,
    *,
    name: Optional[str] = None,
    root_path: Optional[utils.KeyPath] = None,
    view_id: str = 'html-tree-view',
    **kwargs,
) -> Content:
  """Views an object through generating content based on a specific view.

  Args:
    value: The value to view.
    name: The name of the value.
    root_path: The root path of the value.
    view_id: The ID of the view to use. See `pg.View.dir()` for all available
      view IDs.
    **kwargs: Additional keyword arguments passed to the view, wich
      will be used as the preset arguments for the View and Extension methods.

  Returns:
    The rendered `Content` object.
  """
  if isinstance(value, Content):
    return value

  with view_options(**kwargs) as options:
    view_object = View.create(view_id)
    return view_object.render(
        value, name=name, root_path=root_path or utils.KeyPath(), **options
    )


@contextlib.contextmanager
def view_options(**kwargs) -> Iterator[Dict[str, Any]]:
  """Context manager to inject rendering args to view.

  Example:

    with pg.view_options(enable_summary_tooltip=False):
      MyObject().to_html()

  Args:
    **kwargs: Keyword arguments for View.render method.

  Yields:
    The merged keyword arguments.
  """
  parent_options = utils.thread_local_peek(_TLS_KEY_VIEW_OPTIONS, {})
  # Deep merge the two dict.
  options = utils.merge([parent_options, kwargs])
  utils.thread_local_push(_TLS_KEY_VIEW_OPTIONS, options)
  try:
    yield options
  finally:
    utils.thread_local_pop(_TLS_KEY_VIEW_OPTIONS)


class View(metaclass=abc.ABCMeta):
  """Base class for views.

  A view defines an unique way/format of rendering an object (e.g. int, str,
  list, dict, user-defined object). For example, `pg.HtmlView` is a view that
  renders an object into HTML. `pg.HtmlTreeViews` is a concrete `pg.HtmlView`
  that renders an object into a HTML tree view.
  """

  # The ID of the view, which will be used as the value for the `view`
  # argument of `pg.view()`. Must be set for non-abstract subclasses.
  VIEW_ID = None

  class Extension:
    """Extension for the View class.

    View developers should always create a corresponding ``Extension`` class as
    an inner class of the ``View`` class. The ``Extension`` class defines custom
    rendering logic for user-defined types and provides methods that can be
    bound to the ``View`` methods via the ``@pg.View.extension_method``
    decorator.

    Example:

    .. code-block:: python

      class MyView(pg.View):
        VIEW_TYPE = 'my_view'

        class Extension(pg.View.Extension):

          def _my_view_render(self, value, *, view, **kwargs):
            return view.render(value, **kwargs)

          def _my_view_title(self, value, *, view, **kwargs):
            return view.render_title(value, **kwargs)

        @pg.View.extension_method('_my_view_title')
        def title(self, value: Any, **kwargs):
          return pg.Html('<h1>' + str(value) + '</h1>')

        @pg.View.extension_method('_my_view_render')
        def render(self, value: Any, **kwargs):
          return pg.Html(str(value))

    To use the ``Extension`` class, users can subclass it and override the
    extension methods in their own classes.

    For example:

    .. code-block:: python

      class MyObject(MyView.Extension):

        def _my_view_title(self, value, *, view, **kwargs):
          return pg.Html('Custom title for ' + str(value) + '</h1>')

        def _my_view_render(self, value, *, view, **kwargs):
          return self._my_view_title(value, **kwargs) + pg.Html(str(value))

    In this example, ``MyObject`` subclasses the ``Extension`` class of
    ``MyView`` and overrides the ``_my_view_title`` and ``_my_view_render``
    methods to provide custom view rendering for the object.
    """

    @classmethod
    @functools.cache
    def supported_view_classes(cls) -> Set[Type['View']]:
      """Returns all non-abstract View classes that the current class supports.

      A class can inherit from multiple ``View.Extension`` classes. For example:

      .. code-block:: python

        class MyObject(View1.Extension, View2.Extension):
          ...

      In this case, ``MyObject`` supports both ``View1`` and ``View2``.

      Returns:
        All non-abstract View classes that the current class supports.
      """
      supported_view_classes = set()
      view_class = pg_typing.get_outer_class(
          cls, base_cls=View, immediate=True
      )
      if view_class is not None and not inspect.isabstract(view_class):
        supported_view_classes.add(view_class)

      for base_cls in cls.__bases__:
        if issubclass(base_cls, View.Extension):
          supported_view_classes.update(base_cls.supported_view_classes())
      return supported_view_classes

  @classmethod
  def extension_method(cls, method_name: str) -> Any:
    """Decorator that dispatches a View method to a View.Extension method.

    A few things to note:
    1) The View method being decorated must have a `value` argument, based on
       which the Extension method will be dispatched.
    2) The View method's `value` argument will map to the Extension method's
       `self` argument.
    3) The Extension method can optionally have a `view` argument, which will
       be set to the current View class.

    Args:
      method_name: The name of the method in the Extension class to dispatch
        from current View method.

    Returns:
      A decorator that dispatches a View method to a View.Extension method.
    """

    def decorator(func):
      sig = pg_typing.signature(
          func, auto_typing=False, auto_doc=False
      )
      # We substract 1 to offset the `self` argument.
      try:
        extension_arg_index = sig.arg_names.index('value') - 1
      except ValueError as e:
        raise TypeError(
            f'View method {func.__name__!r} must have a `value` argument, '
            'which represents the target object to render.'
        ) from e
      if sig.varargs is not None:
        raise TypeError(
            f'View method must not have variable positional argument. '
            f'Found `*{sig.varargs.name}` in {func.__name__!r}'
        )

      def get_extension(args: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
        if 'value' in kwargs:
          return kwargs['value']
        if extension_arg_index < len(args):
          return args[extension_arg_index]
        raise ValueError(
            'No value is provided for the `value` argument '
            f'for {func.__name__!r}.'
        )

      def map_args(
          args: Sequence[Any], kwargs: Dict[str, Any]
      ) -> Dict[str, Any]:
        # `args` does not contain self, therefore we use less than.
        assert len(args) < len(sig.args), (args, sig.args)
        kwargs.update({
            sig.args[i].name: arg
            for i, arg in enumerate(args) if i != extension_arg_index
        })
        kwargs.pop('value', None)
        return kwargs

      # We use the original function signature to generate the view method.
      @functools.wraps(func)
      def _generated_view_fn(self, *args, **kwargs):
        # This allows a View method to consume the preset kwargs from the
        # parent call to `pg.view`, yet also customizes the preset kwargs
        # to the calls to other View/Extension methods within this context.
        return self._maybe_dispatch(   # pylint: disable=protected-access
            *args, **kwargs,
            extension=get_extension(args, kwargs),
            view_method=func,
            extension_method_name=method_name,
            arg_map_fn=map_args
        )
      return _generated_view_fn
    return decorator

  def __init_subclass__(cls):
    if inspect.isabstract(cls):
      return

    # Register the view type.
    if cls.VIEW_ID is None:
      raise ValueError(
          f'`VIEW_ID` must be set for non-abstract View subclass {cls!r}.'
      )
    if cls.VIEW_ID == cls.__base__.VIEW_ID:
      raise ValueError(
          f'The `VIEW_ID` {cls.VIEW_ID!r} is the same as the base class '
          f'{cls.__base__.__name__}. Please choose a different ID.'
      )

    _VIEW_REGISTRY[cls.VIEW_ID] = cls
    super().__init_subclass__()

  @classmethod
  def dir(cls) -> Dict[str, Type['View']]:
    """Returns all registered View classes with their view IDs."""
    return {
        view_id: view_cls
        for view_id, view_cls in _VIEW_REGISTRY.items()
        if issubclass(view_cls, cls)
    }

  @staticmethod
  def create(view_id: str, **kwargs) -> 'View':
    """Creates a View instance with the given view ID."""
    if view_id not in _VIEW_REGISTRY:
      raise ValueError(
          f'No view class found with VIEW_ID: {view_id!r}'
      )
    return _VIEW_REGISTRY[view_id](**kwargs)

  def __init__(self, **kwargs):
    del kwargs
    super().__init__()

  @abc.abstractmethod
  def render(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
      root_path: Optional[utils.KeyPath] = None,
      **kwargs,
  ) -> Content:
    """Renders the input value.

    Args:
      value: The value to render.
      name: (Optional) The referred name of the value from its container.
      root_path: (Optional) The path of `value` under its object tree.
      **kwargs: Additional keyword arguments passed from `pg.view` or wrapper
        functions (e.g. `pg.to_html`).

    Returns:
      The rendered content.
    """

  #
  # Implementation for routing calls to View methods to to Extension methods.
  #

  def _maybe_dispatch(
      self,
      *args,
      extension: Any,
      view_method: types.FunctionType,
      extension_method_name: str,
      arg_map_fn: Callable[[Sequence[Any], Dict[str, Any]], Dict[str, Any]],
      **kwargs
  ) -> Any:
    """Dispatches the call to View method to corresponding Extension method.

    The dispatching should take care of these three scenarios.

    Scenario 1:
      The user code within an Extension delegates the handling of value `v` back
      to a View method.

      In this case, we should dispatch the call on `v` to the Extension method
      first, and use the View method if we detects that the request is routed
      back to the View method based on the same value.

    Scenario 2:
      The Extension (user code) employes a View method to handle a value `v`'s
      child value `w`.

      In this case, we should allow the Extension method to handle `w` first.

    Scenario 3:
      A Extension (custom render) uses another method of the view to render the
      same value `v`:

    To address these 3 scenarios, we use a thread-local stack to track the
    value being rendered by each view method.

    Args:
      *args: Positional arguments passed to the render method.
      extension: Extension to render.
      view_method: The default render method to call.
      extension_method_name: The name of the method in the Extension class
        to call.
      arg_map_fn: A function to map the view method args to the extension method
        args.
      **kwargs: Keyword arguments passed to the render method.

    Returns:
      The rendered HTML.
    """
    # No Extension involved, call the view's default render method.
    if not isinstance(extension, self.Extension):
      return view_method(self, *args, **kwargs)

    # Identify whether an extension is calling the view's method
    # within its extension method, in such case, we should delegate the
    # rendering logic to the default render method to avoid infinite recursion.
    with self._track_rendering(extension, view_method) as being_rendered:
      if being_rendered is extension:
        return view_method(self, *args, **kwargs)

      # Call the extension's method.
      mapped = arg_map_fn(args, kwargs)
      return getattr(extension, extension_method_name)(
          view=self, **mapped
      )

  @contextlib.contextmanager
  def _track_rendering(
      self,
      value: Extension,
      view_method
  ) -> Iterator[Any]:
    """Context manager for tracking the value being rendered."""
    del self
    rendering_stack = utils.thread_local_get(
        _TLS_KEY_OPERAND_STACK_BY_METHOD, {}
    )
    callsite_value = rendering_stack.get(view_method, None)
    rendering_stack[view_method] = value
    utils.thread_local_set(_TLS_KEY_OPERAND_STACK_BY_METHOD, rendering_stack)
    try:
      yield callsite_value
    finally:
      if callsite_value is None:
        rendering_stack.pop(view_method)
        if not rendering_stack:
          utils.thread_local_del(_TLS_KEY_OPERAND_STACK_BY_METHOD)
      else:
        rendering_stack[view_method] = callsite_value
