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
"""Injecting and manipulating values through context managers."""

import contextlib
import dataclasses
import threading
from typing import Any, Callable, ContextManager, Iterator, Optional

from pyglove.core.utils import missing

RAISE_IF_HAS_ERROR = (missing.MISSING_VALUE,)
_TLS_KEY_CONTEXTUAL_OVERRIDES = '__contextual_overrides__'
_global_contextual_overrides = threading.local()


@dataclasses.dataclass(frozen=True)
class ContextualOverride:
  """Value marker for contextual override for an attribute."""

  # Overridden value.
  value: Any

  # If True, this override will apply to both current scope and nested scope,
  # meaning current `pg.contextual_override` will take precedence over all
  # nested `pg.contextual_override` on this attribute.
  cascade: bool = False

  # If True, this override will apply to attributes that already have values.
  override_attrs: bool = False


def contextual_override(
    *,
    cascade: bool = False,
    override_attrs: bool = False,
    **variables,
) -> ContextManager[dict[str, ContextualOverride]]:
  """Context manager to provide contextual values under a scope.

  Please be aware that contextual value override are per-thread. If you want
  to propagate the contextual value override to other threads, please obtain
  a wrapper function for a user function using
  `pg.with_contextual_override(func)`.

  Args:
    cascade: If True, this override will apply to both current scope and nested
      scope, meaning that this `pg.contextual_override` will take precedence
      over all nested `pg.contextual_override` on the overriden variables.
    override_attrs: If True, this override will apply to attributes that already
      have values. Otherwise overridden variables will only be used for
      contextual attributes whose values are not present.
    **variables: Key/values as override for contextual attributes.

  Returns:
    A dict of attribute names to their contextual overrides.
  """
  vs = {}
  for k, v in variables.items():
    if not isinstance(v, ContextualOverride):
      v = ContextualOverride(v, cascade, override_attrs)
    vs[k] = v
  return contextual_scope(_global_contextual_overrides, **vs)


def with_contextual_override(func: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps a user function with the access to the current contextual override.

  The wrapped function can be called from another thread.

  Args:
    func: The user function to be wrapped.

  Returns:
    A wrapper function that have the access to the current contextual override,
    which can be called from another thread.
  """
  with contextual_override() as current_context:
    pass

  def _func(*args, **kwargs) -> Any:
    with contextual_override(**current_context):
      return func(*args, **kwargs)

  return _func


def get_contextual_override(var_name: str) -> Optional[ContextualOverride]:
  """Returns the overriden contextual value in current scope."""
  return get_scoped_value(_global_contextual_overrides, var_name)


def contextual_value(var_name: str, default: Any = RAISE_IF_HAS_ERROR) -> Any:
  """Returns the value of a variable defined in `pg.contextual_override`."""
  override = get_contextual_override(var_name)
  if override is None:
    if default == RAISE_IF_HAS_ERROR:
      raise KeyError(f'{var_name!r} does not exist in current context.')
    return default
  return override.value


def all_contextual_values() -> dict[str, Any]:
  """Returns all values provided from `pg.contextual_override` in scope."""
  overrides = getattr(
      _global_contextual_overrides, _TLS_KEY_CONTEXTUAL_OVERRIDES, {}
  )
  return {k: v.value for k, v in overrides.items()}


@contextlib.contextmanager
def contextual_scope(
    tls: threading.local, **variables
) -> Iterator[dict[str, ContextualOverride]]:
  """Context manager to set variables within a scope."""
  previous_values = getattr(tls, _TLS_KEY_CONTEXTUAL_OVERRIDES, {})
  current_values = dict(previous_values)
  for k, v in variables.items():
    old_v = current_values.get(k, None)
    if old_v and old_v.cascade:
      v = old_v
    current_values[k] = v
  try:
    setattr(tls, _TLS_KEY_CONTEXTUAL_OVERRIDES, current_values)
    yield current_values
  finally:
    setattr(tls, _TLS_KEY_CONTEXTUAL_OVERRIDES, previous_values)


def get_scoped_value(
    tls: threading.local, var_name: str, default: Any = None
) -> ContextualOverride:
  """Gets the value for requested variable from current scope."""
  scoped_values = getattr(tls, _TLS_KEY_CONTEXTUAL_OVERRIDES, {})
  return scoped_values.get(var_name, default)

