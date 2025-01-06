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
"""Interfaces for pure symbolic objects."""

from typing import Any, Callable, Optional, Tuple
from pyglove.core import typing as pg_typing
from pyglove.core import utils


class PureSymbolic(pg_typing.CustomTyping):
  """Base class to classes whose objects are considered pure symbolic.

  Pure symbolic objects can be used for representing abstract concepts - for
  example, a search space of objects - which cannot be executed but soely
  representational.

  Having pure symbolic object is a key differentiator of symbolic OOP from
  regular OOP, which can be used to placehold values in an object as a
  high-level expression of ideas. Later, with symbolic manipulation, the
  pure symbolic objects are replaced with material values so the object
  can be evaluated. This effectively decouples the expression of ideas from
  the implementation of ideas. For example: ``pg.oneof(['a', 'b', 'c']`` will
  be manipulated into 'a', 'b' or 'c' based on the decision of a search
  algorithm, letting the program evolve itself.
  """

  def custom_apply(
      self,
      path: utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[
          Callable[[utils.KeyPath, pg_typing.Field, Any], Any]
      ] = None,
  ) -> Tuple[bool, Any]:
    """Custom apply on a value based on its original value spec.

    This implements ``pg.pg_typing.CustomTyping``, allowing a pure symbolic
    value to be assigned to any field. To customize this behavior, override
    this method in subclasses.

    Args:
      path: KeyPath of current object under its object tree.
      value_spec: Original value spec for this field.
      allow_partial: Whether allow partial object to be created.
      child_transform: Function to transform child node values into their final
        values. Transform function is called on leaf nodes first, then on their
        parents, recursively.

    Returns:
      A tuple (proceed_with_standard_apply, value_to_proceed).
        If proceed_with_standard_apply is set to False, value_to_proceed
        will be used as final value.

    Raises:
      Error when the value is not compatible with the value spec.
    """
    del path, value_spec, allow_partial, child_transform
    return (False, self)


class NonDeterministic(PureSymbolic):
  """Base class to mark a class whose objects are considered non-deterministic.

  A non-deterministic value represents a value that will be decided later.
  In PyGlove system, `pg.one_of`, `pg.sublist_of`, `pg.float_value` are
  non-deterministic values. Please search `NonDeterministic` subclasses for more
  details.
  """
