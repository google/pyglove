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
"""Common inferential values."""

from typing import Any, Tuple
from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic.object import Object


class InferredValue(Object, base.Inferential):
  """Base class for inferred values."""

  def custom_apply(self, *args, **kwargs: Any) -> Tuple[bool, Any]:
    # This is to make a ``InferredValue`` object assignable
    # to any symbolic attribute.
    return (False, self)


class ValueFromParentChain(InferredValue):
  """A value that could inferred from the parent chain.

  For example::

    class A(pg.Object):
      x: int
      y: int = pg.symbolic.ValueFromParentChain()

    # Not okay: `x` is not inferential and is not specified.
    A()

    # Okay: both `x` and `y` are specified.
    A(x=1, y=2)

    # Okay: `y` is inferential, hence optional.
    a = A(x=1)

    # Raises: `y` is neither specified during __init__
    # nor provided from the context.
    a.y

    d = pg.Dict(y=2, z=pg.Dict(a=a))

    # `a.y` now refers to `d.a` since `d` is in its symbolic parent chain,
    # aka. context.
    assert a.y == 2
  """

  def infer(self, **kwargs) -> Any:
    parent = self.sym_parent
    while True:
      v = self.value_from(parent, **kwargs)
      if v == pg_typing.MISSING_VALUE:
        if parent is None:
          raise AttributeError(
              object_utils.message_on_path(
                  (
                      f'`{self.inference_key}` is not found under its context '
                      '(along its symbolic parent chain).'
                  ),
                  self.sym_path,
              )
          )
        parent = parent.sym_parent
      else:
        return v

  @property
  def inference_key(self) -> str:
    """Returns the key for attribute inference from parents."""
    return self.sym_path.key

  def value_from(self, parent: base.Symbolic, **kwargs) -> Any:
    del kwargs
    if parent is self.sym_parent or parent is None:
      # NOTE(daiyip): The inferred value could not be read from the immediate
      # parent since its key points to current inferential value.
      # We should also return MISSING_VALUE when the traversal has gone beyond
      # of the symbolic tree root.
      return pg_typing.MISSING_VALUE

    # Use current key to lookup from the parent.
    key = self.inference_key
    if isinstance(key, int):
      return (
          parent[key]
          if isinstance(parent, (list, tuple))
          else pg_typing.MISSING_VALUE
      )
    return getattr(parent, key, pg_typing.MISSING_VALUE)
