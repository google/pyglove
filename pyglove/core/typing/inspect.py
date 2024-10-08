# Copyright 2023 The PyGlove Authors
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
"""Utility module for inspecting generics types."""

import inspect
import sys
import typing
from typing import Any, Callable, Optional, Tuple, Type, Union


def is_instance(value: Any, target: Union[Type[Any], Tuple[Type[Any]]]) -> bool:
  """An isinstance extension that supports Any and generic types."""
  return is_subclass(type(value), target)


def is_subclass(
    src: Type[Any], target: Union[Type[Any], Tuple[Type[Any]]]
) -> bool:
  """An issubclass extension that supports Any and generic types."""

  def _is_subclass(src: Type[Any], target: Type[Any]) -> bool:
    if target is Any:
      return True
    elif src is Any:
      return False

    orig_target = typing.get_origin(target)
    orig_src = typing.get_origin(src)

    if orig_target is None:
      if orig_src is None:
        # Both soure and target is not a generic class.
        return issubclass(src, target)
      # Source class is generic but not the target class.
      return issubclass(orig_src, target)
    elif orig_src is None:
      # Target class is generic but not the source class.
      if not issubclass(src, orig_target):
        return False
    elif not issubclass(orig_src, orig_target):
      # Both are generic, but the source is not a subclass of the target.
      return False

    # Check type args.
    t_args = get_type_args(target)
    if not t_args:
      return True

    s_args = get_type_args(src, base=orig_target)
    if s_args:
      assert len(s_args) == len(t_args), (s_args, t_args)
      for s_arg, t_arg in zip(s_args, t_args):
        if not _is_subclass(s_arg, t_arg):
          return False
      return True
    else:
      # A class could inherit multiple generic types. However it does not
      # provide the type arguments for the target generic base. E.g.
      #
      # class A(Generic[X, Y]):
      # class B(A, Generic[X, Y]) :
      # B[int, int] is not a subclass of A[int, int].
      return False

  if isinstance(target, tuple):
    return any(_is_subclass(src, x) for x in target)
  return _is_subclass(src, target)


def is_generic(maybe_generic: Type[Any]) -> bool:
  """Returns True if a type is a generic class."""
  return typing.get_origin(maybe_generic) is not None


def has_generic_bases(maybe_generic: Type[Any]) -> bool:
  """Returns True if a type is a generic subclass."""
  return bool(getattr(maybe_generic, '__orig_bases__', None))


def get_type(maybe_type: Any) -> Type[Any]:
  """Gets the type of a maybe generic type."""
  if isinstance(maybe_type, type):
    return maybe_type
  origin = typing.get_origin(maybe_type)
  if origin is not None:
    return origin
  else:
    raise TypeError(f'{maybe_type!r} is not a type.')


def get_type_args(
    maybe_generic: Type[Any], base: Optional[Type[Any]] = None
) -> Tuple[Type[Any], ...]:
  """Gets generic type args conditioned on an optional base class."""
  if base is None:
    return typing.get_args(maybe_generic)
  else:
    orig_cls = typing.get_origin(maybe_generic)
    if orig_cls is not None:
      orig_bases = (maybe_generic,)
    else:
      orig_bases = getattr(maybe_generic, '__orig_bases__', ())
    for orig_base in orig_bases:
      if get_type(orig_base) is base:
        return typing.get_args(orig_base)
    return ()


def get_outer_class(
    cls: Type[Any],
    base_cls: Union[Type[Any], Tuple[Type[Any], ...], None] = None,
    immediate: bool = False,
) -> Optional[Type[Any]]:
  """Returns the outer class.

  Example::

    class A:
      pass

    class A1:
      class B:
        class C:
          ...

    pg.typing.outer_class(B) is A1
    pg.typing.outer_class(C) is B
    pg.typing.outer_class(C, base_cls=A) is None
    pg.typing.outer_class(C, base_cls=A1) is None

  Args:
    cls: The class to get the outer class for.
    base_cls: The base class of the outer class. If provided, an outer class
      that is not a subclass of `base_cls` will be returned as None.
    immediate: Whether to return the immediate outer class or a class in the
      nesting hierarchy that is a subclass of `base_cls`. Applicable when
      `base_cls` is not None.

  Returns:
    The outer class of `cls`. None if cannot find one or the outer class is
      not a subclass of `base_cls`.
  """
  if '<locals>' in cls.__qualname__:
    raise ValueError(
        'Cannot find the outer class for locally defined class '
        f'{cls.__qualname__!r}'
    )

  names = cls.__qualname__.split('.')
  if len(names) < 2:
    return None

  parent = sys.modules[cls.__module__]
  symbols = []
  for name in names[:-1]:
    symbol = getattr(parent, name, None)
    if symbol is None:
      return None
    assert inspect.isclass(symbol), symbol
    symbols.append(symbol)
    parent = symbol

  for symbol in reversed(symbols):
    if immediate:
      return symbol if not base_cls or issubclass(symbol, base_cls) else None
    if not base_cls or issubclass(symbol, base_cls):
      return symbol
  return None


def callable_eq(
    x: Optional[Callable[..., Any]], y: Optional[Callable[..., Any]]
) -> bool:
  """Returns True if two (maybe) callables are equal.

  For functions: `x` and `y` are considered equal when they are the same
    instance or have the same code (e.g. lambda x: x).

  For methods: `x` and `y` are considered equal when:
    static method: The same method from the same class hierarchy. E.g. subclass
      inherits a base class' static method.
    class method: The same method from the same class. Inherited class method
      are considered different class method.
    instance method: When `self` is not bound, the same method from the same
      class hierarchy (like static method). When `self` is bound, the same
      method on the same object.

  Args:
    x: An optional function or method object.
    y: An optinoal function or method object.

  Returns:
    Returns True if `x` and `y` are considered equal. Meaning that they are
      either the same instance or derived from the same code and have the same
      effect.
  """
  if x is y:
    return True
  if x is None or y is None:
    return False
  if inspect.isfunction(x) and inspect.isfunction(y):
    return _code_eq(x.__code__, y.__code__)
  elif inspect.ismethod(x) and inspect.ismethod(y):
    return _code_eq(x.__code__, y.__code__) and x.__self__ is y.__self__  # pytype: disable=attribute-error
  return x == y


def _code_eq(x, y) -> bool:
  """Returns True if two compiled byte code is the same."""
  return x.co_code == y.co_code
