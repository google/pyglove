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

import typing
from typing import Any, Optional, Tuple, Type, Union


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
