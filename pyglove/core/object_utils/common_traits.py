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
object, for example, partiality (MaybePartial), functor (Functor).
"""

import abc
from typing import Any, Dict, Optional, Union


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
  def missing_values(self, flatten: bool = True) -> Dict[Union[str, int], Any]:  # pylint: disable=redefined-outer-name
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
