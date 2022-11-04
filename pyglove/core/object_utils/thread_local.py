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
"""Thread-local utilities."""

import contextlib
import threading
from typing import Any, Iterator

from pyglove.core import object_utils


_thread_local_state = threading.local()


@contextlib.contextmanager
def value_scope(
    key: str,
    value_in_scope: Any,
    initial_value: Any) -> Iterator[None]:
  """Context manager to set a thread local state within the scope."""
  previous_value = getattr(_thread_local_state, key, initial_value)
  try:
    setattr(_thread_local_state, key, value_in_scope)
    yield
  finally:
    setattr(_thread_local_state, key, previous_value)


def set_value(key: str, value: Any) -> None:
  """Sets thread-local value by key."""
  setattr(_thread_local_state, key, value)


def get_value(
    key: str,
    default_value: Any = object_utils.MISSING_VALUE) -> Any:
  """Gets thread-local value."""
  value = getattr(_thread_local_state, key, default_value)
  if value == object_utils.MISSING_VALUE:
    raise ValueError(f'Key {key!r} does not exist in thread-local storage.')
  return value
