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
"""Callable extensions."""

import contextlib
import functools
import inspect
import types
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from pyglove.core import utils
from pyglove.core.typing import callable_signature


_TLS_KEY_PRESET_KWARGS = '__preset_kwargs__'


class PresetArgValue(utils.Formattable):
  """Value placeholder for arguments whose value will be provided by presets.

  Example:

    def foo(x, y=pg.PresetArgValue(default=1))
      return x + y

    with pg.preset_args(y=2):
      print(foo(x=1))  # 3: y=2
    print(foo(x=1))  # 2: y=1
  """

  def __init__(self, default: Any = utils.MISSING_VALUE):
    self.default = default

  @property
  def has_default(self) -> bool:
    return self.default != utils.MISSING_VALUE

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, PresetArgValue) and (
        self.default == other.default
    )

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def format(self, *args, **kwargs):
    return utils.kvlist_str(
        [
            ('default', self.default, utils.MISSING_VALUE),
        ],
        label='PresetArgValue',
        *args,
        **kwargs,
    )

  @classmethod
  def inspect(
      cls, func: types.FunctionType) -> Dict[str, 'PresetArgValue']:
    """Gets the PresetArgValue specified in a function's signature."""
    assert inspect.isfunction(func), func
    sig = inspect.signature(func)
    preset_arg_markers = {}
    for p in sig.parameters.values():
      if isinstance(p.default, cls):
        preset_arg_markers[p.name] = p.default
    return preset_arg_markers

  @classmethod
  def resolve_args(
      cls,
      call_args: Tuple[Any, ...],
      call_kwargs: Dict[str, Any],
      positional_arg_names: Sequence[str],
      arg_defaults: Dict[str, Any],
      preset_kwargs: Dict[str, Any],
      include_all_preset_kwargs: bool = False,
  ) -> Tuple[Sequence[Any], Dict[str, Any]]:
    """Resolves calling arguments passed to a method with presets."""
    # Step 1: compute marked kwargs.
    resolved_kwargs = {}
    for arg_name, arg_default in arg_defaults.items():
      if not isinstance(arg_default, PresetArgValue):
        resolved_kwargs[arg_name] = arg_default
        continue
      if arg_name in preset_kwargs:
        resolved_kwargs[arg_name] = preset_kwargs[arg_name]
      elif arg_default.has_default:
        resolved_kwargs[arg_name] = arg_default.default
      else:
        raise ValueError(
            f'Argument {arg_name!r} is not present as a keyword argument '
            'from the caller.'
        )

    # Step 2: add preset kwargs
    if include_all_preset_kwargs:
      for k, v in preset_kwargs.items():
        if k not in resolved_kwargs:
          resolved_kwargs[k] = v

    # Step 3: merge call kwargs with resolved preset kwargs.
    resolved_kwargs.update(call_kwargs)

    # Step 3: remove resolved kwargs items as it's present in call args.
    for i in range(len(call_args)):
      if i >= len(positional_arg_names):
        break
      resolved_kwargs.pop(positional_arg_names[i], None)

    # Step 4: convert kwargs back to postional arguments if applicable
    resolved_args = call_args
    if len(positional_arg_names) > len(call_args):
      resolved_args = list(resolved_args)
      for arg_name in positional_arg_names[len(call_args):]:
        if arg_name not in resolved_kwargs:
          break
        arg_value = resolved_kwargs.pop(arg_name)
        resolved_args.append(arg_value)
    return resolved_args, resolved_kwargs


class _ArgPresets:
  """Preset argument collection."""

  def __init__(self, presets: Optional[Dict[str, Dict[str, Any]]] = None):
    self._presets: Dict[str, Dict[str, Any]] = presets or {}

  def derive(
      self,
      kwargs: Dict[str, Any],
      preset_name: str = 'global',
      inherit_preset: Union[str, bool] = False
  ) -> '_ArgPresets':
    """Derives new presets from current presets."""
    presets = self._presets.copy()   # Just do a shallow copy.
    if isinstance(inherit_preset, bool) and inherit_preset:
      inherit_preset = preset_name

    if inherit_preset and inherit_preset in presets:
      current_preset = presets[inherit_preset].copy()
      current_preset.update(kwargs)
    else:
      current_preset = kwargs
    presets[preset_name] = current_preset
    return _ArgPresets(presets)

  def get_preset(self, preset_name: str) -> Dict[str, Any]:
    return self._presets.get(preset_name, {})


@contextlib.contextmanager
def preset_args(
    kwargs: Dict[str, Any],
    *,
    preset_name: str = 'global',
    inherit_preset: Union[str, bool] = False
) -> Iterator[Dict[str, Any]]:
  """Context manager to enable calling with user kwargs.

  Args:
    kwargs: The preset kwargs to be used by preset-enabled functions within
      the context.
    preset_name: The name of the preset to specify kwargs.
      `enable_preset_args` allows users to pass a preset name, which will be
      used to identify the present to be used.
    inherit_preset: The name of the preset defined by the parent context to
      inherit kwargs from. Or a boolean to indicate whether to inherit a
      parent preset of the same name.

  Yields:
    Current preset kwargs.
  """

  parent_presets = utils.thread_local_peek(
      _TLS_KEY_PRESET_KWARGS, _ArgPresets()
  )
  current_preset = parent_presets.derive(kwargs, preset_name, inherit_preset)
  utils.thread_local_push(_TLS_KEY_PRESET_KWARGS, current_preset)
  try:
    yield current_preset
  finally:
    utils.thread_local_pop(_TLS_KEY_PRESET_KWARGS, None)


def enable_preset_args(
    include_all_preset_kwargs: bool = False,
    preset_name: str = 'global'
) -> Callable[[types.FunctionType], types.FunctionType]:
  """Decorator for functions that maybe use preset argument values.

  Usage::

    @pg.typing.enable_preset_args
    def foo(x, y=pg.typing.PresetArgValue(default=1)):
      return x + y

    with pg.typing.preset_args(y=2):
      print(foo(x=1))  # 3: y=2
    print(foo(x=1))  # 2: y=1

  Args:
    include_all_preset_kwargs: Whether to include all preset kwargs (even
      not makred as `PresetArgValue`) when callng the function.
    preset_name: The name of the preset to specify kwargs.

  Returns:
    A decorated function that could consume the preset argument values.
  """
  def decorator(func):
    sig = inspect.signature(func)
    positional_arg_names = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    arg_defaults = {}
    has_preset_value = False
    has_varkw = False
    for p in sig.parameters.values():
      if p.kind == inspect.Parameter.VAR_KEYWORD:
        has_varkw = True
        continue
      if p.kind == inspect.Parameter.VAR_POSITIONAL:
        continue
      if p.default == inspect.Parameter.empty:
        continue
      if isinstance(p.default, PresetArgValue):
        has_preset_value = True
      arg_defaults[p.name] = p.default

    if has_preset_value:
      @functools.wraps(func)
      def _func(*args, **kwargs):
        # Map positional arguments to keyword arguments.
        presets = utils.thread_local_peek(_TLS_KEY_PRESET_KWARGS, None)
        preset_kwargs = presets.get_preset(preset_name) if presets else {}
        args, kwargs = PresetArgValue.resolve_args(
            args, kwargs, positional_arg_names, arg_defaults, preset_kwargs,
            include_all_preset_kwargs=include_all_preset_kwargs and has_varkw
        )
        return func(*args, **kwargs)
      return _func
    return func
  return decorator


class CallableWithOptionalKeywordArgs:
  """Helper class for invoking callable objects with optional keyword args.

  Examples::

    f = pg.typing.CallableWithOptionalKeywordArgs(lambda x: x ** 2, 'y')
    # Returns 4. Keyword 'y' is ignored.
    f(2, y=3)
  """

  def __init__(self,
               func: Callable[..., Any],
               optional_keywords: List[str]):
    sig = callable_signature.signature(
        func, auto_typing=False, auto_doc=False
    )

    # Check for variable keyword arguments.
    if sig.has_varkw:
      absent_keywords = []
    else:
      all_keywords = set(sig.arg_names)
      absent_keywords = [k for k in optional_keywords if k not in all_keywords]
    self._absent_keywords = absent_keywords
    self._func = func

  def __call__(self, *args, **kwargs):
    """Delegate the call to function."""
    for k in self._absent_keywords:
      kwargs.pop(k, None)
    return self._func(*args, **kwargs)
