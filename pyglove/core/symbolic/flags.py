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
"""Global, thread-local and scoped flags for handling symbolic objects."""

from typing import Any, Callable, ContextManager, Optional
from pyglove.core.object_utils import thread_local


#
# Global flags.
#

_ALLOW_EMPTY_FIELD_DESCRIPTION = True
_ALLOW_REPEATED_CLASS_REGISTRATION = True
_ORIGIN_STACKTRACE_LIMIT = 10

_LOAD_HANDLER = None
_SAVE_HANDLER = None


def allow_empty_field_description(allow: bool = True) -> None:
  """Allow empty field description, which is useful for testing purposes."""
  global _ALLOW_EMPTY_FIELD_DESCRIPTION
  _ALLOW_EMPTY_FIELD_DESCRIPTION = allow


def is_empty_field_description_allowed() -> bool:
  """Returns True if empty field description is allowed."""
  return _ALLOW_EMPTY_FIELD_DESCRIPTION


def allow_repeated_class_registration(allow: bool = True) -> None:
  """Allow repeated class registration, which is useful for testing purposes."""
  global _ALLOW_REPEATED_CLASS_REGISTRATION
  _ALLOW_REPEATED_CLASS_REGISTRATION = allow


def is_repeated_class_registration_allowed() -> bool:
  """Returns True if repeated class registration is allowed."""
  return _ALLOW_REPEATED_CLASS_REGISTRATION


def set_origin_stacktrace_limit(limit: int) -> None:
  """Set stack trace limit for origin tracking."""
  global _ORIGIN_STACKTRACE_LIMIT
  _ORIGIN_STACKTRACE_LIMIT = limit


def get_origin_stacktrace_limit() -> int:
  """Returns the limited depth of stacktrace for tracking."""
  return _ORIGIN_STACKTRACE_LIMIT


def set_load_handler(
    load_handler: Optional[Callable[..., Any]]) -> Optional[Callable[..., Any]]:
  """Sets global load handler.

  Args:
    load_handler: A callable object that takes arbitrary arguments and returns
      a value. `symbolic.load` method will pass through all arguments to this
      handler and return its return value.

  Returns:
    Previous global load handler.
  """
  if load_handler and not callable(load_handler):
    raise ValueError('`load_handler` must be callable.')
  global _LOAD_HANDLER
  old_handler = _LOAD_HANDLER
  _LOAD_HANDLER = load_handler
  return old_handler


def get_load_handler() -> Optional[Callable[..., Any]]:
  """Returns global load handler."""
  return _LOAD_HANDLER


def set_save_handler(
    save_handler: Optional[Callable[..., Any]]) -> Optional[Callable[..., Any]]:
  """Sets global save handler.

  Args:
    save_handler: A callable object that takes at least one argument as value to
      save. `symbolic.save` method will pass through all arguments to this
      handler and return its return value.

  Returns:
    Previous global save handler.
  """
  if save_handler and not callable(save_handler):
    raise ValueError('`save_handler` must be callable.')
  global _SAVE_HANDLER
  old_handler = _SAVE_HANDLER
  _SAVE_HANDLER = save_handler
  return old_handler


def get_save_handler() -> Optional[Callable[..., Any]]:
  """Returns global save handler."""
  return _SAVE_HANDLER


#
# Scoped flags.
#

_TLS_ENABLE_CHANGE_NOTIFICATION = '_enable_change_notification'
_TLS_ENABLE_TYPE_CHECK = '_enable_type_check'
_TLS_ENABLE_ORIGIN_TRACKING = '_enable_origin_tracking'
_TLS_ACCESSOR_WRITABLE = '_accessor_writable'
_TLS_ALLOW_PARTIAL = '_allow_partial'
_TLS_SEALED = '_sealed'
_TLS_AUTO_CALL_FUNCTORS = '_allow_auto_call_functors'


def notify_on_change(enabled: bool = True) -> ContextManager[None]:
  """Returns a context manager to enable or disable notification upon change.

  `notify_on_change` is thread-safe and can be nested. For example, in the
  following code, `_on_change` (thus `_on_bound`) method of `a` will be
  triggered due to the rebind in the inner `with` statement, and those of `b`
  will not be triggered as the outer `with` statement disables the
  notification::

    with pg.notify_on_change(False):
      with pg.notify_on_change(True):
        a.rebind(b=1)
      b.rebind(x=2)

  Args:
    enabled: If True, enable change notification in current scope.
      Otherwise, disable notification.

  Returns:
    A context manager for allowing/disallowing change notification in scope.
  """
  return thread_local.thread_local_value_scope(
      _TLS_ENABLE_CHANGE_NOTIFICATION, enabled, True
  )


def is_change_notification_enabled() -> bool:
  """Returns True if change notification is enabled."""
  return thread_local.thread_local_get(
      _TLS_ENABLE_CHANGE_NOTIFICATION, True
  )


def track_origin(enabled: bool = True) -> ContextManager[None]:
  """Returns a context manager to enable or disable origin tracking.

  `track_origin` is thread-safe and can be nested. For example::

    a = pg.Dict(x=1)
    with pg.track_origin(False):
      with pg.track_origin(True):
        # b's origin will be tracked, which can be accessed by `b.sym_origin`.
        b = a.clone()
      # c's origin will not be tracked, `c.sym_origin` returns None.
      c = a.clone()

  Args:
    enabled: If True, the origin of symbolic values will be tracked during
      object cloning and retuning from functors under current scope.

  Returns:
    A context manager for enable or disable origin tracking.
  """
  return thread_local.thread_local_value_scope(
      _TLS_ENABLE_ORIGIN_TRACKING, enabled, False
  )


def is_tracking_origin() -> bool:
  """Returns if origin of symbolic object are being tracked."""
  return thread_local.thread_local_get(_TLS_ENABLE_ORIGIN_TRACKING, False)


def enable_type_check(enabled: bool = True) -> ContextManager[None]:
  """Returns a context manager to enable or disable runtime type check.

  `enable_type_check` is thread-safe and can be nested. For example,
  in the following code, runtime type check with be `a` but not on `b`::

    with pg.enable_type_check(False):
      with pg.enable_type_check(True):
        a = pg.Dict(x=1, value_spec=pg.typing.Dict([('x', pg.typing.Int())]))
      b = pg.Dict(y=1, value_spec=pg.typing.Dict([('x', pg.typing.Int())]))


  Args:
    enabled: If True, enable runtime type check in current scope.
      Otherwise, disable runtime type check.

  Returns:
    A context manager for allowing/disallowing runtime type check.
  """
  return thread_local.thread_local_value_scope(
      _TLS_ENABLE_TYPE_CHECK, enabled, True
  )


def is_type_check_enabled() -> bool:
  """Returns True if runtme type check is enabled."""
  return thread_local.thread_local_get(_TLS_ENABLE_TYPE_CHECK, True)


def allow_writable_accessors(
    writable: Optional[bool] = True) -> ContextManager[None]:
  """Returns a context manager that makes accessor writable in scope.

  This function is thread-safe and can be nested. In the nested use case, the
  writable flag of immediate parent context is effective.

  Example::

    sd1 = pg.Dict()
    sd2 = pg.Dict(accessor_writable=False)
    with pg.allow_writable_accessors(False):
      sd1.a = 2  # NOT OK
      sd2.a = 2  # NOT OK
      with pg.allow_writable_accessors(True):
        sd1.a = 2   # OK
        sd2.a = 2  # OK
        with pg.allow_writable_accessors(None):
          sd1.a = 1  # OK
          sd2.a = 1  # NOT OK

  Args:
    writable: If True, allow write access with accessors (__setattr__,
      __setitem__) for all symbolic values in scope.
      If False, disallow write access via accessors for all symbolic values
      in scope, even if individual objects allow so.
      If None, honor object-level `accessor_writable` flag.

  Returns:
    A context manager that allows/disallows writable accessors of all
      symbolic values in scope. After leaving the scope, the
      `accessor_writable` flag of individual objects will remain intact.
  """
  return thread_local.thread_local_value_scope(
      _TLS_ACCESSOR_WRITABLE, writable, None
  )


def is_under_accessor_writable_scope() -> Optional[bool]:
  """Return True if symbolic values are treated as sealed in current context."""
  return thread_local.thread_local_get(_TLS_ACCESSOR_WRITABLE, None)


def as_sealed(sealed: Optional[bool] = True) -> ContextManager[None]:
  """Returns a context manager to treat symbolic values as sealed/unsealed.

  While the user can use `Symbolic.seal` to seal or unseal an individual object.
  This context manager is useful to create a readonly zone for operations on
  all existing symbolic objects.

  This function is thread-safe and can be nested. In the nested use case, the
  sealed flag of immediate parent context is effective.

  Example::

    sd1 = pg.Dict()
    sd2 = pg.Dict().seal()

    with pg.as_sealed(True):
      sd1.a = 2  # NOT OK
      sd2.a = 2  # NOT OK
      with pg.as_sealed(False):
        sd1.a = 2   # OK
        sd2.a = 2  # OK
        with pg.as_sealed(None):
          sd1.a = 1  # OK
          sd2.a = 1  # NOT OK

  Args:
    sealed: If True, treats all symbolic values as sealed in scope.
      If False, treats all as unsealed.
      If None, honor object-level `sealed` state.

  Returns:
    A context manager that treats all symbolic values as sealed/unsealed
      in scope. After leaving the scope, the sealed state of individual objects
      will remain intact.
  """
  return thread_local.thread_local_value_scope(_TLS_SEALED, sealed, None)


def is_under_sealed_scope() -> Optional[bool]:
  """Return True if symbolic values are treated as sealed in current context."""
  return thread_local.thread_local_get(_TLS_SEALED, None)


def allow_partial(allow: Optional[bool] = True) -> ContextManager[None]:
  """Returns a context manager that allows partial values in scope.

  This function is thread-safe and can be nested. In the nested use case, the
  allow flag of immediate parent context is effective.

  Example::

    @pg.members([
        ('x', pg.typing.Int()),
        ('y', pg.typing.Int())
    ])
    class A(pg.Object):
      pass

    with pg.allow_partial(True):
      a = A(x=1)  # Missing `y`, but OK
      with pg.allow_partial(False):
        a.rebind(x=pg.MISSING_VALUE)  # NOT OK
      a.rebind(x=pg.MISSING_VALUE)  # OK

  Args:
    allow: If True, allow partial symbolic values in scope.
      If False, do not allow partial symbolic values in scope even if
      individual objects allow so. If None, honor object-level
      `allow_partial` property.

  Returns:
    A context manager that allows/disallow partial symbolic values in scope.
      After leaving the scope, the `allow_partial` state of individual objects
      will remain intact.
  """
  return thread_local.thread_local_value_scope(_TLS_ALLOW_PARTIAL, allow, None)


def is_under_partial_scope() -> Optional[bool]:
  """Return True if partial value is allowed in current context."""
  return thread_local.thread_local_get(_TLS_ALLOW_PARTIAL, None)


def auto_call_functors(enabled: bool = True) -> ContextManager[None]:
  """Returns a context manager to enable or disable auto call for functors.

  `auto_call_functors` is thread-safe and can be nested. For example::

    @pg.symbolize
    def foo(x, y):
      return x + y

    with pg.auto_call_functors(True):
      a = foo(1, 2)
      assert a == 3
      with pg.auto_call_functors(False):
        b = foo(1, 2)
        assert isinstance(b, foo)

  Args:
    enabled: If True, enable auto call for functors.
      Otherwise, auto call will be disabled.

  Returns:
    A context manager for enabling/disabling auto call for functors.
  """
  return thread_local.thread_local_value_scope(
      _TLS_AUTO_CALL_FUNCTORS, enabled, False
  )


def should_call_functors_during_init() -> Optional[bool]:
  """Return True functors should be automatically called during __init__."""
  return thread_local.thread_local_get(_TLS_AUTO_CALL_FUNCTORS, None)
