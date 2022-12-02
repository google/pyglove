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
"""Pattern-based patching."""

import re
from typing import Any, Callable, Optional, Tuple, Type, Union
from pyglove.core import object_utils
from pyglove.core import symbolic


def patch_on_key(
    src: symbolic.Symbolic,
    regex: str,
    value: Any = None,
    value_fn: Optional[Callable[[Any], Any]] = None,
    skip_notification: Optional[bool] = None) -> Any:
  """Recursively patch values on matched keys (leaf-node names).

  Example::

    d = pg.Dict(a=0, b=2)
    print(pg.patching.patch_on_key(d, 'a', value=3))
    # {a=3, b=2}

    print(pg.patching.patch_on_key(d, '.', value=3))
    # {a=3, b=3}

    @pg.members([
      ('x', schema.Int())
    ])
    class A(pg.Object):

      def _on_init(self):
        super()._on_init()
        self._num_changes = 0

      def _on_change(self, updates):
        super()._on_change(updates)
        self._num_changes += 1

    a = A()
    pg.patching.patch_on_key(a, 'x', value=2)
    # a._num_changes is 1.

    pg.patching.patch_on_key(a, 'x', value=3)
    # a._num_changes is 2.

    pg.patching.patch_on_keys(a, 'x', value=4, skip_notification=True)
    # a._num_changes is still 2.

  Args:
    src: symbolic value to patch.
    regex: Regex for key name.
    value: New value for field that satisfy `condition`.
    value_fn: Callable object that produces new value based on old value.
      If not None, `value` must be None.
    skip_notification: If True, `on_change` event will not be triggered for this
      operation. If None, the behavior is decided by `pg.notify_on_rebind`.
      Please see `symbolic.Symbolic.rebind` for details.

  Returns:
    `src` after being patched.
  """
  regex = re.compile(regex)
  return _conditional_patch(
      src,
      lambda k, v, p: k and regex.match(str(k.key)),
      value,
      value_fn,
      skip_notification)


def patch_on_path(
    src: symbolic.Symbolic,
    regex: str,
    value: Any = None,
    value_fn: Optional[Callable[[Any], Any]] = None,
    skip_notification: Optional[bool] = None) -> Any:
  """Recursively patch values on matched paths.

  Example::

    d = pg.Dict(a={'x': 1}, b=2)
    print(pg.patching.patch_on_path(d, '.*x', value=3))
    # {a={x=1}, b=2}

  Args:
    src: symbolic value to patch.
    regex: Regex for key path.
    value: New value for field that satisfy `condition`.
    value_fn: Callable object that produces new value based on old value.
      If not None, `value` must be None.
    skip_notification: If True, `on_change` event will not be triggered for this
      operation. If None, the behavior is decided by `pg.notify_on_rebind`.
      Please see `symbolic.Symbolic.rebind` for details.

  Returns:
    `src` after being patched.
  """
  regex = re.compile(regex)
  return _conditional_patch(
      src, lambda k, v, p: regex.match(str(k)),
      value, value_fn, skip_notification)


def patch_on_value(
    src: symbolic.Symbolic,
    old_value: Any,
    value: Any = None,
    value_fn: Optional[Callable[[Any], Any]] = None,
    skip_notification: Optional[bool] = None) -> Any:
  """Recursively patch values on matched values.

  Example::

    d = pg.Dict(a={'x': 1}, b=1)
    print(pg.patching.patch_on_value(d, 1, value=3))
    # {a={x=3}, b=3}

  Args:
    src: symbolic value to patch.
    old_value: Old value to match.
    value: New value for field that satisfy `condition`.
    value_fn: Callable object that produces new value based on old value.
      If not None, `value` must be None.
    skip_notification: If True, `on_change` event will not be triggered for this
      operation. If None, the behavior is decided by `pg.notify_on_rebind`.
      Please see `symbolic.Symbolic.rebind` for details.

  Returns:
    `src` after being patched.
  """
  return _conditional_patch(
      src, lambda k, v, p: v == old_value,
      value, value_fn, skip_notification)


def patch_on_type(
    src: symbolic.Symbolic,
    value_type: Union[Type[Any], Tuple[Type[Any], ...]],
    value: Any = None,
    value_fn: Optional[Callable[[Any], Any]] = None,
    skip_notification: Optional[bool] = None) -> Any:
  """Recursively patch values on matched types.

  Example::

    d = pg.Dict(a={'x': 1}, b=2)
    print(pg.patching.patch_on_type(d, int, value_fn=lambda x: x * 2))
    # {a={x=2}, b=4}

  Args:
    src: symbolic value to patch.
    value_type: Value type to match.
    value: New value for field that satisfy `condition`.
    value_fn: Callable object that produces new value based on old value.
      If not None, `value` must be None.
    skip_notification: If True, `on_change` event will not be triggered for this
      operation. If None, the behavior is decided by `pg.notify_on_rebind`.
      Please see `symbolic.Symbolic.rebind` for details.

  Returns:
    `src` after being patched.
  """
  return _conditional_patch(
      src, lambda k, v, p: isinstance(v, value_type),
      value, value_fn, skip_notification)


def patch_on_member(
    src: symbolic.Symbolic,
    cls: Union[Type[Any], Tuple[Type[Any], ...]],
    name: str,
    value: Any = None,
    value_fn: Optional[Callable[[Any], Any]] = None,
    skip_notification: Optional[bool] = None) -> Any:
  """Recursively patch values that are the requested member of classes.

  Example::

    d = pg.Dict(a=A(x=1), b=2)
    print(pg.patching.patch_on_member(d, A, 'x', 2)
    # {a=A(x=2), b=4}

  Args:
    src: symbolic value to patch.
    cls: In which class the member belongs to.
    name: Member name.
    value: New value for field that satisfy `condition`.
    value_fn: Callable object that produces new value based on old value.
      If not None, `value` must be None.
    skip_notification: If True, `on_change` event will not be triggered for this
      operation. If None, the behavior is decided by `pg.notify_on_rebind`.
      Please see `symbolic.Symbolic.rebind` for details.

  Returns:
    `src` after being patched.
  """
  return _conditional_patch(
      src, lambda k, v, p: isinstance(p, cls) and k.key == name,
      value, value_fn, skip_notification)


def _conditional_patch(
    src: symbolic.Symbolic,
    condition: Callable[
        [object_utils.KeyPath, Any, symbolic.Symbolic], bool],
    value: Any = None,
    value_fn: Optional[Callable[[Any], Any]] = None,
    skip_notification: Optional[bool] = None) -> Any:
  """Recursive patch values on condition.

  Args:
    src: symbolic value to patch.
    condition: Callable object with signature (key_path, value, parent) which
      returns whether a field should be patched.
    value: New value for field that satisfy `condition`.
    value_fn: Callable object that produces new value based on old value.
      If not None, `value` must be None.
    skip_notification: If True, `on_change` event will not be triggered for this
      operation. If None, the behavior is decided by `pg.notify_on_rebind`.
      Please see `symbolic.Symbolic.rebind` for details.

  Returns:
    `src` after being patched.
  """
  if value_fn is not None and value is not None:
    raise ValueError(
        'Either `value` or `value_fn` should be specified.')
  def _fn(k, v, p):
    if condition(k, v, p):
      return value_fn(v) if value_fn else value
    return v
  return src.rebind(
      _fn, raise_on_no_change=False, skip_notification=skip_notification)
