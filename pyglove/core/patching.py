# Copyright 2019 The PyGlove Authors
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

"""Systematic patching on symbolic values.

As :meth:`pyglove.Symbolic.rebind` provides a flexible programming
interface for modifying symbolic values, why bother to have this module?
Here are the  motivations:

  * Provide user friendly methods for addressing the most common patching
    patterns.

  * Provide a systematic solution for

    * Patch semantic groups.
    * Enable combination of these groups.
    * Provide an interface that patching can be invoked from the command line.
"""

import re
import typing
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Type, Union
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as schema


def patch_on_key(
    src: symbolic.Symbolic,
    regex: Text,
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
    regex: Text,
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
    name: Text,
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
      ) -> Union[Dict[Text, Any], Tuple[Dict[Text, Any], Callable[[], None]]]:
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

  def register(self, name: Text, patcher_cls: Type[Patcher]):
    """Register a function with a scheme name."""
    if name in self._registry and not _ALLOW_REPEATED_PATCHER_REGISTRATION:
      raise KeyError(f'Patcher {name!r} already registered.')
    self._registry[name] = patcher_cls

  @property
  def names(self) -> List[Text]:
    """Returns registered scheme names."""
    return list(self._registry.keys())


_PATCHER_REGISTRY = _PatcherRegistry()


def patcher(
    args: Optional[List[Tuple[Text, schema.ValueSpec]]] = None,
    name: Optional[Text] = None) -> Any:
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
    arg_specs = cls.signature.args
    if len(arg_specs) < 1:
      raise TypeError(
          f'Patcher function should have at least 1 argument '
          f'as patching target. (Patcher={cls.type_name!r})')
    if not _is_patcher_target_spec(arg_specs[0].value_spec):
      raise TypeError(
          f'{arg_specs[0].value_spec!r} cannot be used for constraining '
          f'Patcher target. (Patcher={cls.type_name!r}, '
          f'Argument={arg_specs[0].name!r})\n'
          f'Acceptable value spec types are: '
          f'Any, Callable, Dict, Functor, List, Object.')
    for arg_spec in arg_specs[1:]:
      if not _is_patcher_parameter_spec(arg_spec.value_spec):
        raise TypeError(
            f'{arg_spec.value_spec!r} cannot be used for constraining '
            f'Patcher argument. (Patcher={cls.type_name!r}, '
            f'Argument={arg_spec.name!r})\n'
            f'Consider to treat it as string and parse yourself.')
    return cls
  return _decorator


def _is_patcher_target_spec(value_spec):
  """Return True if value_spec can be used for patcher target."""
  return isinstance(
      value_spec, (schema.Any, schema.Object,
                   schema.Dict, schema.List, schema.Callable))


def _is_patcher_parameter_spec(value_spec, leaf_only=False):
  """Return True if value_spec can be used for patcher parameters."""
  if isinstance(value_spec, (schema.Any, schema.Str, schema.Bool,
                             schema.Int, schema.Float)):
    return True
  elif isinstance(value_spec, schema.Enum):
    return value_spec.value_type == str
  elif isinstance(value_spec, schema.List):
    return (not leaf_only  and _is_patcher_parameter_spec(
        value_spec.element.value, leaf_only=True))
  return False


def patcher_names():
  """Returns all registered patch names."""
  return _PATCHER_REGISTRY.names


PatchType = Union[
    Dict[Text, Any], Callable, Patcher, Text,               # pylint: disable = g-bare-generic
    List[Union[Dict[Text, Any], Callable, Patcher, Text]]]  # pylint: disable = g-bare-generic


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


def from_uri(uri: Text) -> Patcher:
  """Create a Patcher object from a URI-like string."""
  name, args, kwargs = parse_uri(uri)
  patcher_cls = typing.cast(Type[Any], _PATCHER_REGISTRY.get(name))
  args, kwargs = parse_args(patcher_cls.signature, args, kwargs)
  return patcher_cls(object_utils.MISSING_VALUE, *args, **kwargs)


def parse_uri(uri: Text) -> Tuple[Text, List[Text], Dict[Text, Text]]:
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


def parse_args(signature: schema.Signature,
               args: List[Text],
               kwargs: Dict[Text, Text]) -> Tuple[List[Any], Dict[Text, Any]]:
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


def parse_arg(patcher_id: Text, arg_name: Text,
              value_spec: schema.ValueSpec, arg_str: Text):
  """Parse patcher argument based on value spec."""
  def _value_error(msg):
    return ValueError(f'{msg} (Patcher={patcher_id!r}, Argument={arg_name!r})')

  if arg_str in _NONE_LITERAL_SET:
    # NOTE(daiyip): If string type value needs literal 'None' or 'none', they
    # can quote the string with "".
    arg = None
  elif isinstance(value_spec, (schema.Any, schema.Str)):
    if len(arg_str) > 1:
      begin_quote = (arg_str[0] == '"')
      end_quote = (arg_str[-1] == '"')
      if begin_quote and end_quote:
        arg_str = arg_str[1:-1]
      elif begin_quote or end_quote:
        raise _value_error(f'Unmatched quote for string value: {arg_str!r}.')
    arg = arg_str
  elif isinstance(value_spec, schema.Bool):
    if (arg_str not in _BOOL_TRUE_LITERAL_SET
        and arg_str not in _BOOL_FALSE_LITERAL_SET):
      raise _value_error(f'Cannot convert {arg_str!r} to bool.')
    arg = arg_str in _BOOL_TRUE_LITERAL_SET
  elif isinstance(value_spec, schema.Int):
    try:
      arg = int(arg_str)
    except ValueError:
      raise _value_error(f'Cannot convert {arg_str!r} to int.')  # pylint: disable=raise-missing-from
  elif isinstance(value_spec, schema.Float):
    try:
      arg = float(arg_str)
    except ValueError:
      raise _value_error(f'Cannot convert {arg_str!r} to float.')  # pylint: disable=raise-missing-from
  elif isinstance(value_spec, schema.Enum):
    if value_spec.value_type != str:
      raise _value_error(
          f'{value_spec!r} cannot be used for Patcher argument. '
          f'Only Enum of string type can be used.')
    arg = arg_str
  elif isinstance(value_spec, schema.List):
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


def parse_list(string: Text,
               convert_fn: Callable[[int, Text], Any]) -> List[Any]:
  """Parse a (possibly empty) colon-separated list of values."""
  string = string.strip()
  if string:
    return [convert_fn(i, piece) for i, piece in enumerate(string.split(':'))]
  return []


@symbolic.functor()
def object_factory(
    value_type: Type[symbolic.Symbolic],
    base_value: Union[symbolic.Symbolic,
                      Callable[[], symbolic.Symbolic],
                      Text],
    patches: Optional[PatchType] = None,
    params_override: Optional[Union[Dict[Text, Any], Text]] = None) -> Any:
  """A factory to create symbolic object from a base value and patches.

  Args:
    value_type: Type of return value.
    base_value: An instance of `value_type`,
      or a callable object that produces an instance of `value_type`,
      or a string as the path to the serialized value.
    patches: Optional patching rules. See :func:`patch` for details.
    params_override: A rebind dict (or a JSON string as serialized rebind dict)
      as an additional patch to the value,

  Returns:
    Value after applying `patchers` and `params_override` based on `base_value`.
  """
  # Step 1: Load base value.
  if not isinstance(base_value, value_type) and callable(base_value):
    value = base_value()
  elif isinstance(base_value, str):
    value = symbolic.load(base_value)
  else:
    value = base_value

  if not isinstance(value, value_type):
    raise TypeError(
        f'{base_value!r} is neither an instance of {value_type!r}, '
        f'nor a factory or a path of JSON file that produces an '
        f'instance of {value_type!r}.')

  # Step 2: Patch with patchers if available.
  if patches is not None:
    value = patch(value, patches)

  # Step 3: Patch with additional parameter override dict if available.
  if params_override:
    value = value.rebind(
        object_utils.flatten(from_maybe_serialized(params_override, dict)),
        raise_on_no_change=False)
  return value


def from_maybe_serialized(
    source: Union[Any, Text],
    value_type: Optional[Type[Any]] = None) -> Any:
  """Load value from maybe serialized form (e.g. JSON file or JSON string).

  Args:
    source: Source of value. It can be value (non-string type) itself, or a
      filepath, or a JSON string from where the value will be loaded.
    value_type: An optional type to constrain the value.

  Returns:
    Value from source.
  """
  if isinstance(source, str):
    if source.endswith('.json'):
      value = symbolic.load(source)
    else:
      value = symbolic.from_json_str(source)
  else:
    value = source
  if value_type is not None and not isinstance(value, value_type):
    raise TypeError(
        f'Loaded value {value!r} is not an instance of {value_type!r}.')
  return value
