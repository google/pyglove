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
"""Contextual value marker."""
from typing import Any, Callable, Optional, Tuple
import pyglove.core.typing as pg_typing

# The default contextual getter.
_DEFAULT_GETTER = lambda name, x: getattr(x, name, pg_typing.MISSING_VALUE)


class Contextual(pg_typing.CustomTyping):
  """Marker for values to be read from current field's symbolic parents.

  Example::

    class A(pg.Object):
      x: int
      y: int = pg.Contextual()

    # Not okay: `x` is not contextual and is not specified.
    A()

    # Okay: both `x` and `y` are specified.
    A(x=1, y=2)

    # Okay: `y` is contextual, hence optional.
    a = A(x=1)

    # Raises: `y` is neither specified during __init__
    # nor provided from the context.
    a.y
  """

  def __init__(self, getter: Optional[Callable[[str, Any], Any]] = None):
    """Constructor.

    Args:
      getter: An optional callable object to get the value of the request
        attribute name from a symbolic parent, with signature: (attribute_name,
        symbolic_parent) -> attribute_value If the getter returns
        ``pg.MISSING_VALUE` or a ``pg.Contextual`` object, the context will be
        moved unto the parent's parent. If None, the getter will be quering the
        attribute of the same name from the the parent.
    """
    super().__init__()
    self._getter = getter or _DEFAULT_GETTER

  def custom_apply(self, *args, **kwargs) -> Tuple[bool, Any]:
    # This is to make a ``Contextual`` object assignable
    # to any symbolic attribute.
    return (False, self)

  def value_from(self, name: str, parent) -> Any:
    """Returns the contextual attribute value from the parent object.

    Args:
      name: The name of request attribute.
      parent: Current context (symbolic parent).

    Returns:
      The value for the contextual attribute.
    """
    return self._getter(name, parent)

  def __repr__(self):
    return str(self)

  def __str__(self):
    return 'CONTEXTUAL'

  def __eq__(self, other):
    return isinstance(other, Contextual) and self._getter == other._getter

  def __ne__(self, other):
    return not self.__eq__(other)
