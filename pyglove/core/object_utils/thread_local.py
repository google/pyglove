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
from typing import Any, Callable, Iterator

from pyglove.core.object_utils.missing import MISSING_VALUE

_RAISE_IF_NOT_FOUND = (MISSING_VALUE,)
_thread_local_state = threading.local()


@contextlib.contextmanager
def thread_local_value_scope(
    key: str, value_in_scope: Any, initial_value: Any
) -> Iterator[None]:
  """Context manager to set a thread local state within the scope."""
  has_key = thread_local_has(key)
  previous_value = thread_local_get(key, initial_value)
  try:
    thread_local_set(key, value_in_scope)
    yield
  finally:
    if has_key:
      thread_local_set(key, previous_value)
    else:
      thread_local_del(key)


def thread_local_has(key: str) -> bool:
  """Deletes thread-local value by key."""
  return hasattr(_thread_local_state, key)


def thread_local_set(key: str, value: Any) -> None:
  """Sets thread-local value by key."""
  setattr(_thread_local_state, key, value)


def thread_local_get(
    key: str, default_value: Any = _RAISE_IF_NOT_FOUND) -> Any:
  """Gets thread-local value."""
  value = getattr(_thread_local_state, key, default_value)
  if value is _RAISE_IF_NOT_FOUND:
    raise ValueError(f'Key {key!r} does not exist in thread-local storage.')
  return value


def thread_local_del(key: str) -> None:
  """Deletes thread-local value by key."""
  delattr(_thread_local_state, key)


def thread_local_map(
    key: str,
    value_fn: Callable[[Any], Any],
    default_initial_value: Any = _RAISE_IF_NOT_FOUND) -> Any:
  """Map a thread-local value."""
  value = thread_local_get(key, MISSING_VALUE)
  if value == MISSING_VALUE:
    value = default_initial_value
    if value is _RAISE_IF_NOT_FOUND:
      raise ValueError(f'Key {key!r} does not exist in thread-local storage.')
    thread_local_set(key, value)

  new_value = value_fn(value)
  if value is not new_value:
    thread_local_set(key, new_value)
  return new_value


def thread_local_increment(key: str, default_initial_value: int = 0) -> int:
  """Increment an integer identified by key."""
  return thread_local_map(
      key,
      lambda x: x + 1,
      default_initial_value=default_initial_value
  )


def thread_local_decrement(
    key: str,
    default_initial_value: int = _RAISE_IF_NOT_FOUND  # pytype: disable=annotation-type-mismatch
    ) -> int:
  """Increment an integer identified by key."""
  return thread_local_map(
      key,
      lambda x: x - 1,
      default_initial_value=default_initial_value
  )


def thread_local_push(key: str, value: Any) -> None:
  """Pushes a value to a stack identified by key."""
  thread_local_map(
      key,
      lambda x: x.append(value) or x,
      default_initial_value=[]
  )


def thread_local_pop(key: str, default_value: Any = _RAISE_IF_NOT_FOUND) -> Any:
  """Pops a value from a stack identified by key."""
  stack = thread_local_get(key, MISSING_VALUE)
  if stack == MISSING_VALUE:
    if default_value is _RAISE_IF_NOT_FOUND:
      raise ValueError(f'Key {key!r} does not exist in thread-local storage.')
    return default_value

  if not isinstance(stack, list):
    raise TypeError(
        f'Key {key!r} from thread-local storage is not a list: {stack}')

  if not stack and default_value is not _RAISE_IF_NOT_FOUND:
    return default_value
  return stack.pop()
