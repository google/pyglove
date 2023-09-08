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
"""Patcher: modular rule-based patching."""

import re
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing


class Patcher(symbolic.Functor):
  """Class that patches a symbolic value by returning a rebind dict.

  To accomondate reusable patching, we introduced the concept of patcher,
  which is a symbolic function that takes an input value with a list of
  optional arguments and produces a patching rule (see :func:`pyglove.patch`)
  or a tuple of (patching_rule, validation_rule). Validation rule is a callable
  object that validate the patched object's integrity if it's being patched
  by others later. A patcher can be created from a URI-like string, to
  better serve the command-line interface.

  A patcher can be created via ``pg.patcher`` decorator::

    @pg.patcher([
        ('lr', pg.typing.Float(min_value=0.0))
    ])
    def learning_rate(trainer, lr):
      return {
          'training.learning_rate': lr
      }

  OR:

    @pg.patcher([
        ('lr', pg.typing.Float(min_value=0.0))
    ])
    def learning_rate(trainer, lr):
      def _rebind_fn(k, v, p):
         if k and k.key == 'learning_rate' and isinstance(v, float):
           return lr
         return v
      return _rebind_fn

  OR a composition of rules.

    @pg.patcher([
        ('lr', pg.typing.Float(min_value=0.0)),
        ('weight_decay', pg.typing.Float(min_value=0.0))
    ])
    def complex(trainer, lr, weight_decay):
      # `change_lr` and `change_weight_decay` are instances of other patchers.
      return [
          change_lr(lr),
          change_weight_decay(weight_decay)
      ]

  After registration, a patcher object can be obtained from a URL-like string::

    patcher = pg.patching.from_uri('cosine_decay?lr=0.1')

  Then the user can use the patcher to patch an object::

    patched_trainer = patcher.patch(trainer)

  A list of patchers can be applied sequentially to accomondate combination of
  semantic groups. A patcher in the sequence can propose updates to the original
  value or generate a replacement to the original value. ``pg.patching.patch``
  is introduced for making it convenient to chain multiple patchers using
  URI-like strings::

    pg.patching.patch(trainer, [
        'cosine_decay?lr=0.1',
        'some_other_patcher_string',
    ])

  The user can lookup all registered patchers via::

    print(pg.patching.patcher_names)
  """

  def _on_bound(self):
    super()._on_bound()
    self._validator = None

  def patch(self, x: symbolic.Symbolic) -> Any:
    """Patches an input and return the input itself unless fully replaced."""
    # Placeholder for Google-internal usage instrumentation.

    if not isinstance(x, symbolic.Symbolic):
      raise TypeError(
          f'The 1st argument of {self.__class__.__name__!r} must be a '
          f'symbolic type. Encountered: {x!r}.')
    # The patching rule returned from the patcher body should be either
    # (patching_rule, validation_rule) or just patching_rule.
    # The patching rule can be a dict, a rebind function, another patcher
    # or a mixture of them as a list. The patching rule is then passed to
    # ``pg.patch`` which allows patchers to support composition of sub-patchers.
    patching_rule = self.__call__(x)

    # Set validator if applicable.
    validator = None
    if isinstance(patching_rule, tuple) and len(patching_rule) == 2:
      patching_rule, validator = patching_rule
    if validator is not None and not callable(validator):
      raise TypeError(
          f'The validator returned from patcher {self.__class__.__name__!r} '
          f'is not callable. Encountered: {validator!r}.')
    self._validator = validator
    return patch(x, patching_rule)

  def validate(self, x: symbolic.Symbolic) -> None:
    """Validates an input's integrity.

    This method will be called in :func:`pyglove.patch` when a chain of patchers
    have been applied, as to validate the patched object in chain still
    conforms to the patcher's plan.

    Args:
      x: The input after modification.
    """
    if self._validator is not None:
      self._validator(x)

  def __call__(
      self,
      x: symbolic.Symbolic
      ) -> Union[Dict[str, Any],
                 Tuple[Dict[str, Any], Callable[[Any], None]]]:
    """Override __call__ to get rebind dict."""
    return super().__call__(x, override_args=True)


_ALLOW_REPEATED_PATCHER_REGISTRATION = True


def allow_repeated_patcher_registration(allow: bool = True):
  """If True, allow registration with the same patch name."""
  global _ALLOW_REPEATED_PATCHER_REGISTRATION
  _ALLOW_REPEATED_PATCHER_REGISTRATION = allow


class _PatcherRegistry:
  """Patcher registry."""

  def __init__(self):
    self._registry = dict()

  def get(self, name) -> Type[Patcher]:
    """Returns patch class by name."""
    if name not in self._registry:
      raise KeyError(f'Patcher {name!r} is not registered.')
    return self._registry[name]

  def register(self, name: str, patcher_cls: Type[Patcher]):
    """Register a function with a scheme name."""
    if name in self._registry and not _ALLOW_REPEATED_PATCHER_REGISTRATION:
      raise KeyError(f'Patcher {name!r} already registered.')
    self._registry[name] = patcher_cls

  @property
  def names(self) -> List[str]:
    """Returns registered scheme names."""
    return list(self._registry.keys())


_PATCHER_REGISTRY = _PatcherRegistry()


def patcher(
    args: Optional[List[Tuple[str, pg_typing.ValueSpec]]] = None,
    name: Optional[str] = None) -> Any:
  """Decorate a function into a Patcher and register it.

  A patcher function is defined as:

     `<patcher_fun> := <fun_name>(<target>, [parameters])`

  The signature takes at least one argument as the patching target,
  with additional arguments as patching parameters to control the details of
  this patch.

  Example::

    @pg.patching.patcher([
      ('x': pg.typing.Int())
    ])
    def increment(v, x=1):
      return pg.symbolic.get_rebind_dict(
          lambda k, v, p: v + x if isinstance(v, int) else v)

    # This patcher can be called via:
    # pg.patching.apply(v, [increment(x=2)])
    # or pg.patching.apply(v, ['increment?x=2'])

  Args:
    args: A list of (arg_name, arg_value_spec) to schematize patcher arguments.
    name: String to be used as patcher name in URI. If None, function name will
      be used as patcher name.

  Returns:
    A decorator that converts a function into a Patcher subclass.
  """
  functor_decorator = symbolic.functor(args, base_class=Patcher)
  def _decorator(fn):
    """Returns decorated Patcher class."""
    cls = functor_decorator(fn)
    _PATCHER_REGISTRY.register(name or fn.__name__,
                               typing.cast(Type[Patcher], cls))
    arg_specs = cls.__signature__.args
    if len(arg_specs) < 1:
      raise TypeError(
          'Patcher function should have at least 1 argument '
          f'as patching target. (Patcher={cls.__type_name__!r})'
      )
    if not _is_patcher_target_spec(arg_specs[0].value_spec):
      raise TypeError(
          f'{arg_specs[0].value_spec!r} cannot be used for constraining '
          f'Patcher target. (Patcher={cls.__type_name__!r}, '
          f'Argument={arg_specs[0].name!r})\n'
          'Acceptable value spec types are: '
          'Any, Callable, Dict, Functor, List, Object.'
      )
    for arg_spec in arg_specs[1:]:
      if not _is_patcher_parameter_spec(arg_spec.value_spec):
        raise TypeError(
            f'{arg_spec.value_spec!r} cannot be used for constraining '
            f'Patcher argument. (Patcher={cls.__type_name__!r}, '
            f'Argument={arg_spec.name!r})\n'
            'Consider to treat it as string and parse yourself.'
        )
    return cls
  return _decorator


def _is_patcher_target_spec(value_spec):
  """Return True if value_spec can be used for patcher target."""
  return isinstance(
      value_spec, (pg_typing.Any, pg_typing.Object,
                   pg_typing.Dict, pg_typing.List, pg_typing.Callable))


def _is_patcher_parameter_spec(value_spec, leaf_only=False):
  """Return True if value_spec can be used for patcher parameters."""
  if isinstance(value_spec, (pg_typing.Any, pg_typing.Str, pg_typing.Bool,
                             pg_typing.Int, pg_typing.Float)):
    return True
  elif isinstance(value_spec, pg_typing.Enum):
    return value_spec.value_type == str
  elif isinstance(value_spec, pg_typing.List):
    return (not leaf_only  and _is_patcher_parameter_spec(
        value_spec.element.value, leaf_only=True))
  return False


def patcher_names():
  """Returns all registered patch names."""
  return _PATCHER_REGISTRY.names


PatchType = Union[
    Dict[str, Any], Callable, Patcher, str,               # pylint: disable = g-bare-generic
    List[Union[Dict[str, Any], Callable, Patcher, str]]]  # pylint: disable = g-bare-generic


def patch(value: symbolic.Symbolic, rule: PatchType) -> Any:
  """Apply patches to a symbolic value.

  Args:
    value: A symbolic value to patch.
    rule: A patching rule is one of the following:
      1) A dict of symbolic paths to the new values.
      2) A rebind function defined by signature (k, v) or (k, v, p).
         See :meth:`pyglove.Symbolic.rebind`.
      3) A :class:`pyglove.patching.Patcher` object.
      4) A URL-like string representing an instance of a register Patcher.
         The format is "<patcher_name>?<arg1>&<arg2>=<val2>".
      5) A list of the mixtures of above.

  Returns:
    Value after applying the patchers. If any patcher returned a new value
    (by returning a single-item dict that containing '' as key), the return
    value will be a different object other than `value`, otherwise `value`
    will be returned after applying the patches.

  Raises:
    ValueError: Error if the patch name and arguments cannot
      be parsed successfully.
  """
  patches = []
  rules = rule if isinstance(rule, list) else [rule]
  for p in rules:
    if isinstance(p, str):
      p = from_uri(p)
    if not isinstance(p, (Patcher, dict)) and not callable(p):
      raise TypeError(
          f'Patching rule {p!r} should be a dict of path to values, a rebind '
          f'function, a patcher (object or string), or a list of their '
          f'mixtures.')
    patches.append(p)

  # Apply patches in chain.
  for p in patches:
    if isinstance(p, Patcher):
      value = p.patch(value)
    elif isinstance(p, dict):
      if len(p) == 1 and '' in p:
        value = p['']
      else:
        value = value.rebind(p, raise_on_no_change=False)
    else:
      value = value.rebind(p, raise_on_no_change=False)

  # Validate patched values.
  for p in patches:
    if isinstance(p, Patcher):
      p.validate(value)
  return value


_ID_REGEX = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*')
_ARG_ASSIGN_REGEX = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)')


def from_uri(uri: str) -> Patcher:
  """Create a Patcher object from a URI-like string."""
  name, args, kwargs = parse_uri(uri)
  patcher_cls = typing.cast(Type[Any], _PATCHER_REGISTRY.get(name))
  args, kwargs = parse_args(patcher_cls.__signature__, args, kwargs)
  return patcher_cls(object_utils.MISSING_VALUE, *args, **kwargs)


def parse_uri(uri: str) -> Tuple[str, List[str], Dict[str, str]]:
  """Parse patcher name and arguments from a URI-like string."""
  pos = uri.find('?')
  args = []
  kwargs = dict()
  if pos >= 0:
    name = uri[:pos]
    args_str = uri[pos + 1:]
    for arg_str in args_str.split('&'):
      if '=' in arg_str:
        m = _ARG_ASSIGN_REGEX.match(arg_str)
        if m is None:
          raise ValueError(f'Invalid argument specification: {arg_str}.')
        assert len(m.groups()) == 2
        arg_name, arg_value = m.groups()
        assert _ID_REGEX.match(arg_name)
        kwargs[arg_name] = arg_value
      else:
        if kwargs:
          raise ValueError(
              f'Positional argument should be provided before keyword '
              f'arguments. Encountered: {uri!r}.')
        args.append(arg_str)
  else:
    name = uri

  if not _ID_REGEX.match(name):
    raise ValueError(f'{name!r} is not a valid Patcher name.')
  return name, args, kwargs    # pytype: disable=bad-return-type


def parse_args(signature: pg_typing.Signature,
               args: List[str],
               kwargs: Dict[str, str]) -> Tuple[List[Any], Dict[str, Any]]:
  """Parse patcher arguments based on its signature."""
  acceptable_arg_names = [arg.name for arg in signature.args[1:]]
  if len(signature.args) < len(args) + 1:
    raise KeyError(
        f'Too many positional arguments are provided. '
        f'(Patcher={signature.id})\n'
        f'Expected: {acceptable_arg_names!r}, Received: {args!r}.')
  kw = {arg.name: arg for arg in signature.args}
  parsed_args = []
  parsed_kwargs = {}
  for i, arg_str in enumerate(args):
    arg_spec = signature.args[i + 1]
    parsed_args.append(
        parse_arg(signature.id, arg_spec.name, arg_spec.value_spec, arg_str))
  for k, arg_str in kwargs.items():
    if k not in kw:
      raise KeyError(
          f'Unexpected argument {k!r}. (Patcher={signature.id!r})\n'
          f'Acceptable argument names: {acceptable_arg_names!r}.')
    arg_spec = kw[k]
    parsed_kwargs[k] = parse_arg(
        signature.id, arg_spec.name, arg_spec.value_spec, arg_str)
  return parsed_args, parsed_kwargs


_BOOL_TRUE_LITERAL_SET = frozenset(['True', 'true', 'yes', '1'])
_BOOL_FALSE_LITERAL_SET = frozenset(['False', 'false', 'no', '0'])
_NONE_LITERAL_SET = frozenset(['None', 'none'])


def parse_arg(patcher_id: str, arg_name: str,
              value_spec: pg_typing.ValueSpec, arg_str: str):
  """Parse patcher argument based on value spec."""
  def _value_error(msg):
    return ValueError(f'{msg} (Patcher={patcher_id!r}, Argument={arg_name!r})')

  if arg_str in _NONE_LITERAL_SET:
    # NOTE(daiyip): If string type value needs literal 'None' or 'none', they
    # can quote the string with "".
    arg = None
  elif isinstance(value_spec, (pg_typing.Any, pg_typing.Str)):
    if len(arg_str) > 1:
      begin_quote = (arg_str[0] == '"')
      end_quote = (arg_str[-1] == '"')
      if begin_quote and end_quote:
        arg_str = arg_str[1:-1]
      elif begin_quote or end_quote:
        raise _value_error(f'Unmatched quote for string value: {arg_str!r}.')
    arg = arg_str
  elif isinstance(value_spec, pg_typing.Bool):
    if (arg_str not in _BOOL_TRUE_LITERAL_SET
        and arg_str not in _BOOL_FALSE_LITERAL_SET):
      raise _value_error(f'Cannot convert {arg_str!r} to bool.')
    arg = arg_str in _BOOL_TRUE_LITERAL_SET
  elif isinstance(value_spec, pg_typing.Int):
    try:
      arg = int(arg_str)
    except ValueError:
      raise _value_error(f'Cannot convert {arg_str!r} to int.')  # pylint: disable=raise-missing-from
  elif isinstance(value_spec, pg_typing.Float):
    try:
      arg = float(arg_str)
    except ValueError:
      raise _value_error(f'Cannot convert {arg_str!r} to float.')  # pylint: disable=raise-missing-from
  elif isinstance(value_spec, pg_typing.Enum):
    if value_spec.value_type != str:
      raise _value_error(
          f'{value_spec!r} cannot be used for Patcher argument. '
          f'Only Enum of string type can be used.')
    arg = arg_str
  elif isinstance(value_spec, pg_typing.List):
    arg = parse_list(
        arg_str,
        lambda i, s: parse_arg(    # pylint: disable=g-long-lambda
            patcher_id, f'{arg_name}[{i}]', value_spec.element.value, s))
  else:
    raise _value_error(
        f'{value_spec!r} cannot be used for Patcher argument.\n'
        f'Consider to treat this argument as string and parse it yourself.')
  return value_spec.apply(
      arg, root_path=object_utils.KeyPath.parse(f'{patcher_id}.{arg_name}'))


def parse_list(string: str,
               convert_fn: Callable[[int, str], Any]) -> List[Any]:
  """Parse a (possibly empty) colon-separated list of values."""
  string = string.strip()
  if string:
    return [convert_fn(i, piece) for i, piece in enumerate(string.split(':'))]
  return []
