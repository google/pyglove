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
"""Typed value placeholders."""

from typing import Any
from pyglove.core import utils
from pyglove.core.typing import class_schema


# Non-typed missing value.
MISSING_VALUE = utils.MISSING_VALUE


class MissingValue(utils.MissingValue, utils.Formattable):
  """Class represents missing value **for a specific value spec**."""

  def __init__(self, value_spec: class_schema.ValueSpec):
    """Constructor."""
    self._value_spec = value_spec

  @property
  def value_spec(self) -> class_schema.ValueSpec:
    """Returns value spec of current missing value."""
    return self._value_spec

  def __eq__(self, other: Any) -> bool:
    """Operator ==.

    NOTE: `MissingValue(value_spec) and `utils.MissingValue` are
    considered equal, but `MissingValue(value_spec1)` and
    `MissingValue(value_spec2)` are considered different. That being said,
    the 'eq' operation is not transitive.

    However in practice this is not a problem, since user always compare
    against `schema.MISSING_VALUE` which is `utils.MissingValue`.
    Therefore the `__hash__` function returns the same value with
    `utils.MissingValue`.

    Args:
      other: the value to compare against.

    Returns:
      True if the other value is a general MissingValue or MissingValue of the
        same value spec.
    """
    if self is other:
      return True
    if isinstance(other, MissingValue):
      return self._value_spec == other.value_spec
    return MISSING_VALUE == other

  def __hash__(self) -> int:
    """Overridden hashing to make all MissingValue return the same value."""
    return hash(MISSING_VALUE)

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> str:
    """Format current object."""
    if compact:
      return 'MISSING_VALUE'
    else:
      spec_str = self._value_spec.format(
          compact=compact, verbose=verbose, root_indent=root_indent, **kwargs)
      return f'MISSING_VALUE({spec_str})'

  def __deepcopy__(self, memo):
    """Avoid deep copy by copying value_spec by reference."""
    return MissingValue(self.value_spec)
