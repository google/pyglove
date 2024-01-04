# Copyright 2022 The PyGlove Authors
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
"""Common traits for Python objects.

This file defines interfaces for describing the common traits of a Python
object, for example, partiality (MaybePartial), formatting (Formattable),
functor (Functor).
"""

import abc
from typing import Any, ContextManager, Dict, Optional
from pyglove.core.object_utils import thread_local


_TLS_STR_FORMAT_KWARGS = '_str_format_kwargs'
_TLS_REPR_FORMAT_KWARGS = '_repr_format_kwargs'


def str_format(**kwargs) -> ContextManager[Dict[str, Any]]:
  """Context manager for setting the default format kwargs for __str__."""
  return thread_local.thread_local_arg_scope(_TLS_STR_FORMAT_KWARGS, **kwargs)


def repr_format(**kwargs) -> ContextManager[Dict[str, Any]]:
  """Context manager for setting the default format kwargs for __repr__."""
  return thread_local.thread_local_arg_scope(_TLS_REPR_FORMAT_KWARGS, **kwargs)


class Formattable(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be pretty-formatted.

  This interface overrides the default ``__repr__`` and ``__str__`` method, thus
  all ``Formattable`` objects can be printed nicely.

  All symbolic types implement this interface.
  """

  # Additional format keyword arguments for `__str__`.
  __str_format_kwargs__ = dict(compact=False, verbose=True)

  # Additional format keyword arguments for `__repr__`.
  __repr_format_kwargs__ = dict(compact=True)

  @abc.abstractmethod
  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> str:
    """Formats this object into a string representation.

    Args:
      compact: If True, this object will be formatted into a single line.
      verbose: If True, this object will be formatted with verbosity.
        Subclasses should define `verbosity` on their own.
      root_indent: The start indent level for this object if the output is a
        multi-line string.
      **kwargs: Subclass specific keyword arguments.

    Returns:
      A string of formatted object.
    """

  def __str__(self) -> str:
    """Returns the full (maybe multi-line) representation of this object."""
    kwargs = dict(self.__str_format_kwargs__)
    kwargs.update(thread_local.thread_local_kwargs(_TLS_STR_FORMAT_KWARGS))
    return self._maybe_quote(self.format(**kwargs), **kwargs)

  def __repr__(self) -> str:
    """Returns a single-line representation of this object."""
    kwargs = dict(self.__repr_format_kwargs__)
    kwargs.update(thread_local.thread_local_kwargs(_TLS_REPR_FORMAT_KWARGS))
    return self._maybe_quote(self.format(**kwargs), **kwargs)

  def _maybe_quote(
      self,
      s: str,
      *,
      compact: bool = False,
      root_indent: int = 0,
      markdown: bool = False,
      **kwargs
  ) -> str:
    """Maybe quote the formatted string with markdown."""
    del kwargs
    if not markdown or root_indent > 0:
      return s
    if compact:
      return f'`{s}`'
    else:
      return f'\n```\n{s}\n```\n'


class MaybePartial(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be partially constructed.

  A ``MaybePartial`` object is an object whose ``__init__`` method can accept
  ``pg.MISSING_VALUE`` as its argument values. All symbolic types (see
  :class:`pyglove.Symbolic`) implements this interface, as their symbolic
  attributes can be partially filled.

  Example::

    d = pg.Dict(x=pg.MISSING_VALUE, y=1)
    assert d.is_partial
    assert 'x' in d.missing_values()
  """

  @property
  def is_partial(self) -> bool:
    """Returns True if this object is partial. Otherwise False.

    An object is considered partial when any of its required fields is missing,
    or at least one member is partial. The subclass can override this method
    to provide a more efficient solution.
    """
    return len(self.missing_values()) > 0  # pylint: disable=g-explicit-length-test

  @abc.abstractmethod
  def missing_values(self, flatten: bool = True) -> Dict[str, Any]:  # pylint: disable=redefined-outer-name
    """Returns missing values from this object.

    Args:
      flatten: If True, convert nested structures into a flattened dict using
        key path (delimited by '.' and '[]') as key.

    Returns:
      A dict of key to MISSING_VALUE.
    """


class Functor(metaclass=abc.ABCMeta):
  """Interface for functor."""

  @abc.abstractmethod
  def __call__(self, *args, **kwargs) -> Any:
    """Calls the functor.

    Args:
      *args: Any positional arguments.
      **kwargs: Any keyword arguments.

    Returns:
      Any value.
    """


def explicit_method_override(method):
  """Decorator that marks a member method as explicitly overridden.

  In PyGlove, many methods are managed by the framework - for example -
  ``pg.Object.__init__``. It's easy for users to override these methods
  unconsciously. Therefore, we introduce this decorator to catch error at
  the first place when such overrides incidentally take place, while allowing
  advanced users to override them.

  Usage::

    class Foo(pg.Object):

      @pg.explicit_method_override
      def __init__(self, *args, **kwargs):
       ...

  Args:
    method: method to explicitly overriden.

  Returns:
    The original method with an explicit overriden stamp.
  """
  setattr(method, '__explicit_override__', True)
  return method


def ensure_explicit_method_override(
    method, error_message: Optional[str] = None) -> None:
  """Returns True if a method is explicitly overridden."""
  if not getattr(method, '__explicit_override__', False):
    if error_message is None:
      error_message = (
          f'{method} is a PyGlove managed method. If you do need to override '
          'it, please decorate the method with `@pg.explicit_method_override`.')
    raise TypeError(error_message)
