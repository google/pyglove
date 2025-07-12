# Copyright 2025 The PyGlove Authors
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
"""Python code permissions."""

import contextlib
import enum
from typing import Optional

from pyglove.core import utils


class CodePermissionMeta(enum.EnumMeta):

  @property
  def BASIC(cls) -> 'CodePermission':  # pylint: disable=invalid-name
    """Returns basic permissions."""
    return cls.ASSIGN | cls.CALL    # pytype: disable=attribute-error

  @property
  def ALL(cls) -> 'CodePermission':  # pylint: disable=invalid-name
    """Returns all permissions."""
    return (
        cls.BASIC | cls.CONDITION | cls.LOOP | cls.EXCEPTION |         # pytype: disable=attribute-error
        cls.CLASS_DEFINITION | cls.FUNCTION_DEFINITION | cls.IMPORT    # pytype: disable=attribute-error
    )


class CodePermission(enum.Flag, metaclass=CodePermissionMeta):
  """Permissions for code execution."""

  # Allows assignment.
  ASSIGN = enum.auto()

  # Allows conditions.
  CONDITION = enum.auto()

  # Allows loops.
  LOOP = enum.auto()

  # Call functions or methods.
  CALL = enum.auto()

  # Allows exception.
  EXCEPTION = enum.auto()

  # Allows class definitions.
  CLASS_DEFINITION = enum.auto()

  # Allows function definitions.
  FUNCTION_DEFINITION = enum.auto()

  # Allows import.
  IMPORT = enum.auto()


_TLS_CODE_RUN_PERMISSION = '__code_run_permission__'


@contextlib.contextmanager
def permission(perm: CodePermission):
  """Context manager for controling the permission for code execution.

  When the `permission` context manager is nested, the outtermost permission
  will be used. This design allows users to control permission at the top level.

  Args:
    perm: Code execution permission.

  Yields:
    Actual permission applied.
  """

  outter_perm = utils.thread_local_get(_TLS_CODE_RUN_PERMISSION, None)

  # Use the top-level permission as the actual permission
  if outter_perm is not None:
    perm = outter_perm

  utils.thread_local_set(_TLS_CODE_RUN_PERMISSION, perm)

  try:
    yield perm
  finally:
    if outter_perm is None:
      utils.thread_local_del(_TLS_CODE_RUN_PERMISSION)


def get_permission() -> Optional[CodePermission]:
  """Gets the current permission for code execution."""
  return utils.thread_local_get(_TLS_CODE_RUN_PERMISSION, None)
