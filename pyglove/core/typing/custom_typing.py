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
"""Interface for intercepting typing checking."""

import abc
from typing import Any, Callable, Optional, Tuple

from pyglove.core import object_utils
from pyglove.core.typing import class_schema


class CustomTyping(metaclass=abc.ABCMeta):
  """Interface of custom value type.

  Instances of subclasses of CustomTyping can be assigned to fields of
  any ValueSpec, and take over `apply` via `custom_apply` method.

  As a result, CustomTyping makes the schema system extensible without modifying
  existing value specs. For example, value generators can extend CustomTyping
  and be assignable to any fields.
  """

  @abc.abstractmethod
  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: class_schema.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, class_schema.Field, Any], Any]] = None
  ) -> Tuple[bool, Any]:
    """Custom apply on a value based on its original value spec.

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
