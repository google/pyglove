# Copyright 2019 The PyGlove Authors
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
"""Class detour."""

import contextlib
import inspect
import threading
import types
from typing import Any, Dict, Sequence, Tuple, Type, Union


@contextlib.contextmanager
def detour(
    mappings: Sequence[Tuple[
        Type[Any],                               # Source class
        Union[Type[Any], types.FunctionType]     # Target class or function
    ]]):
  """Context manager for detouring object creation.

  At times, we want to replace an object of a class to an object of a different
  class. Usually, we do so by passing the object as a function argument using
  dependency injection. However, it's not always possible to expose those
  internal objects as parameters to the class, as we cannot predict what needs
  to be customized in future. Also, exposing too many arguments will hurt
  usability, it's big burden to figure out 20 arguments of a function for a user
  to get started.

  `pg.detour` provides another option for object replacement in Python, which
  creates a context in which some source classes can be detoured to specified
  destination classes or functions. For example, the code snippet below will
  detour instantation of class A to class B, and vice-versa::

    class A:
      pass

    class B:
      pass

    # Exchange class A and class B.
    with pg.detour([(A, B), (B, A)]):
      a = A()   # a is a B object.
      b = B()   # b is an A object.

  Detour destination can be a function, which allows users to intercept the
  arguments passed to the class constructor. For example::

    class Foo:
      def __init__(self, value):
        self.value = value

    class Bar:
      def __init__(self, value):
        self.value = value

    def detoured_foo(cls, value):
      # cls is the original class before detour.
      return Bar(value + 1)

    with pg.detour([(Foo, detoured_foo)]):
      f = Foo(1)   # f will be Bar(2).

  Detour can be nested. The outer scope mappings take precedence over the
  mappings from the inner loop, allowing users to change object creation
  behaviors from the outside. For example, the following code will detour
  class A to class C::

    with pg.detour([(A, C)]):
      with pg.detour([A, B]):
        a = A()   # a is a C object.

  Detour is transisive across the inner and outer scope. For example, the code
  below will detour class A to class C through B::

    with pg.detour([(B, C)]):
      a1 = A()     # a1 is an A object.
      with pg.detour([A, B]):
        a2 = A()    # a2 is a C object. (A -> B -> C)

  Detour is thread-sfe.

  Args:
    mappings: A sequence of tuple (src_cls, dest_cls_or_fn) as mappings for the
      detour - 'src_cls' is the source class to be detoured, while
      'dest_cls_or_fn' is the destination class or function. When it's a class,
      its `__init__` method should have the same signature as the `__init__` of
      the original class. When it's a function, it should accept a positional
      argument `cls`, for passing the original class that is being detoured,
      followed by all the arguments that the original class should accept. For
      example, a class with `__init__(self, x, *args, y, **kwargs)` can be
      detoured to a function with signature `(cls, x, *args, y, **kwargs)`.

  Yields:
    Resolved detour mappings.

  Raises:
    TypeError: If the first item in each mapping is not a class, or the second
      item in each mapping is neither a class nor a function.
  """
    # Placeholder for Google-internal usage instrumentation.

  for src, dest in mappings:
    if not inspect.isclass(src):
      raise TypeError(f'Detour source {src!r} is not a class.')
    if not inspect.isclass(dest) and not inspect.isfunction(dest):
      raise TypeError(
          f'Detour destination {dest!r} is not a class or a function.')

  try:
    yield _global_detour_context.enter_scope(mappings)
  finally:
    _global_detour_context.leave_scope()


def current_mappings() -> Dict[Type[Any], Union[Type[Any], types.FunctionType]]:
  """Returns detour mappings under current scope."""
  return _global_detour_context.current_mappings


def undetoured_new(cls, *args, **kwargs) -> Any:
  """Create a new instance of cls without detouring.

  If cls.__init__ creates sub-objects, creation of sub-objects
  maybe detoured based on current context. For example::

    class A:

      def __init__(self, x):
        if x < 0:
          self.child = A(x)
        else:
          self.x = x

    with pg.detour([A, B]):
      a = A(-1)
      assert isinstance(a, A)
      assert isinstance(a.child, B)

  Args:
    cls: The class whose instance will be created.
    *args: Positional arguments to be passed to class __init__ method.
    **kwargs: Keyword arguments to be passed to class __init__ method.

  Returns:
    A instance of `cls`.
  """
  new_method = _global_detour_context.get_original_new(cls)
  if new_method is object.__new__:
    instance = new_method(cls)
  else:
    instance = new_method(cls, *args, **kwargs)
  instance.__init__(*args, **kwargs)
  return instance


class _DetourContext:
  """Context that sets/gets detoured class mappings under current thread."""

  _DETOUR_STACK_KEY = 'detour_stack'
  _DETOUR_MAPPING_KEY = 'detour_map'
  _NEW_CALL_STACK = 'new_stack'

  def __init__(self):
    self._tls = threading.local()
    self._original_new = dict()

  @property
  def _detour_stack(self):
    detour_stack = getattr(self._tls, self._DETOUR_STACK_KEY, None)
    if detour_stack is None:
      detour_stack = []
      setattr(self._tls, self._DETOUR_STACK_KEY, detour_stack)
    return detour_stack

  @property
  def current_mappings(
      self) -> Dict[Type[Any], Union[Type[Any], types.FunctionType]]:
    if self._detour_stack:
      return self._detour_stack[-1]
    return dict()

  def enter_scope(
      self,
      mappings: Sequence[Tuple[
          Type[Any],                            # Source class
          Union[Type[Any], types.FunctionType]  # Target class or function
      ]]) -> Dict[Type[Any], Union[Type[Any], types.FunctionType]]:
    """Enter a new scope."""
    # Create a copy of current mapping so we can modify.
    cur_mappings = dict(self.current_mappings)

    # Compute transisive mappings between current scope and new scope.
    # 1) If a source cls exists in current mappings, use the existing
    #    destination. E.g:
    #    ```
    #    with pg.detour([(A, B)]):
    #      with pg.detour([(A, C]]):
    #        A()   # should produce B.
    #
    # 2) If a destination cls exists in current mapping, and source class does
    #    not exist in current mappings, inserts a mapping from the source class
    #    to the destination of the target class in current mapping. E.g:
    #    ```
    #    with pg.detour([(B, C)]):
    #      with pg.detour([(A, B)]):
    #        A()   # should produce C.
    #
    # 3) Otherwise insert the new mapping. E.g:
    #    ```
    #    with pg.detour([(A, B)]):
    #      with pg.detour([(C, D)]):
    #        C()   # should produce D.
    new_mappings = []
    for src, dest in mappings:
      if src not in cur_mappings:
        if dest in cur_mappings:
          new_mappings.append((src, cur_mappings[dest]))
        else:
          new_mappings.append((src, dest))

    for src, dest in new_mappings:
      if src not in self._original_new:
        self._original_new[src] = src.__new__
        setattr(src, '__new__', _maybe_detoured_new)
      cur_mappings[src] = dest
    self._detour_stack.append(cur_mappings)
    return cur_mappings

  def leave_scope(self):
    """Leave current detour scope."""
    assert self._detour_stack
    self._detour_stack.pop(-1)

  def get_destination(self, src_cls):
    return self.current_mappings.get(src_cls, None)

  def get_original_new(self, src_cls):
    """Returns the original new method of source cls."""
    if not _is_detoured_new(src_cls.__new__):
      orig_new = src_cls.__new__
    else:
      # NOTE(daiyip): there are usually 3 patterns in implementing __new__.
      # 1) call super.__new__ to return an instance.
      # 2) explicitly call object.__new__ to return an instance.
      # 3) return an instance from another class.
      #
      # The following code aims to support case #1 by mimicing the call
      # convention of super.__new__ without access to the super object.
      # We implement this by maintaining a call history of `__new__` method
      # returned by `get_original_new` for each top-most call to
      # `_maybe_detour_new`. Based on the history, we always return the next
      # __new__ along the inheritance hierarchy. For example, for code:
      #
      # ```
      #   class A:
      #     def __new__(cls, *args, **kwargs):
      #       return super(A, cls).__new__(cls, *args, **kwargs)
      #
      #   class B:
      #     def __new__(cls, *args, **kwargs):
      #       return super(A, cls).__new__(cls, *args, **kwargs)
      #
      #   class C(A, B):
      #     pass
      # ```
      # when we detour A and B to other classes, their `__new__` method will be
      # replaced with `_maybe_detoured_new`. As we create an object of C, it
      # will call `C.__new__`, which inherits the `_maybe_detoured_new` assigned
      # to `A.__new__`. `_maybe_detoured_new` calls `get_original_new` on class
      # C, which should return the original `A.__new__`. It then executes
      # `super(A, cls).__new__`, which triggers `_maybe_detoured_new` method
      # again assigned to `B.__new__`. In such case, we cannot differentiate the
      # first call to `_maybe_detoured_new` (C.__new__) from this call, since
      # both take class C as the cls argument. However, by assuming that nested
      # `_maybe_detoured_new` call should always reflect the `super.__new__`
      # call convention, we can store the call history for these invoked __new__
      # methods, and return the one that is one-step closer to `object.__new__`.
      # This may not work for the most complicated __new__ customization, but
      # should work well for most __new__ implementations.
      orig_new = self._original_new.get(src_cls, object.__new__)
      if orig_new is object.__new__ or orig_new in self._new_stack:
        for base in src_cls.__bases__:
          base_new = self.get_original_new(base)
          if base_new is not object.__new__ and base_new not in self._new_stack:
            orig_new = base_new
            break
    return orig_new

  @property
  def _new_stack(self):
    """Returns the stack of new methods in current thread."""
    stack = getattr(self._tls, self._NEW_CALL_STACK, None)
    if stack is None:
      stack = []
      setattr(self._tls, self._NEW_CALL_STACK, stack)
    return stack

  def call_new(self, new_method, cls, *args, **kwargs):
    """Call __new__ method with correctly handling super.__new__."""
    try:
      self._new_stack.append(new_method)
      if new_method is object.__new__:
        return object.__new__(cls)
      else:
        return new_method(cls, *args, **kwargs)
    finally:
      self._new_stack.pop(-1)

# Global detour context.
_global_detour_context = _DetourContext()


@staticmethod   # This decorator is required for Python2.
def _maybe_detoured_new(cls, *args, **kwargs):
  """A __new__ method to replace user class' __new__ for detour."""
  dest_cls_or_fn = _global_detour_context.get_destination(cls)
  if dest_cls_or_fn is None:
    # No detour in current thread.
    return _global_detour_context.call_new(
        _global_detour_context.get_original_new(cls),
        cls, *args, **kwargs)

  if inspect.isclass(dest_cls_or_fn):
    dest_cls = dest_cls_or_fn
    instance = _global_detour_context.call_new(
        _global_detour_context.get_original_new(dest_cls),
        dest_cls, *args, **kwargs)

    # NOTE(daiyip): when an overridden `__new__` returns an instance whose
    # class is not strictly the user class, `__init__` will not be called
    # by Python runtime. We can detect such case and invoke `__init__`
    # manually.
    if not isinstance(instance, cls):
      instance.__init__(*args, **kwargs)
    return instance
  else:
    # NOTE(daiyip): when function is used as detour destination, we handle it
    # specially to allow instances of the source class to be created in the
    # function. E.g.
    #
    # ```
    # def create_foo(value):
    #   return Foo(value + 1)
    #
    # with pg.detour([(Foo, create_foo)]):
    #    Foo(1)
    # ```
    try:
      _global_detour_context.current_mappings[cls] = cls
      return dest_cls_or_fn(cls, *args, **kwargs)
    finally:
      _global_detour_context.current_mappings[cls] = dest_cls_or_fn


def _is_detoured_new(method):
  """Returns if a method is detoured new."""
  return method is getattr(_maybe_detoured_new, '__func__')
