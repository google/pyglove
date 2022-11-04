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
"""Symbolic function (Functor)."""

import abc
import functools
import inspect

import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags
from pyglove.core.symbolic import object as pg_object
from pyglove.core.symbolic import schema_utils


class Functor(pg_object.Object, object_utils.Functor):
  """Symbolic functions (Functors).

  A symbolic function is a symbolic class with a ``__call__`` method, whose
  arguments can be bound partially, incrementally bound by attribute
  assignment, or provided at call time.

  Another useful trait is that a symbolic function is serializable, when
  its definition is imported by the target program and its arguments are also
  serializable. Therefore, it is very handy to move a symbolic function
  around in distributed scenarios.

  Symbolic functions can be created from regular function via
  :func:`pyglove.functor`::

    # Create a functor class using @pg.functor decorator.
    @pg.functor([
      ('a', pg.typing.Int(), 'Argument a'),
      # No field specification for 'b', which will be treated as any type.
    ])
    def sum(a, b=1, *args, **kwargs):
      return a + b + sum(args + kwargs.values())

    sum(1)()           # returns 2: prebind a=1, invoke with b=1 (default)
    sum(a=1)()         # returns 2: same as above.
    sum()(1)           # returns 2: bind a=1 at call time, b=1(default)

    sum(b=2)(1)        # returns 3: prebind b=2, invoke with a=1.
    sum(b=2)()         # wrong: `a` is not provided.

    sum(1)(2)                      # wrong: 'a' is provided multiple times.
    sum(1)(2, override_args=True)   # ok: override `a` value with 2.

    sum()(1, 2, 3, 4)  # returns 10: a=1, b=2, *args=[3, 4]
    sum(c=4)(1, 2, 3)  # returns 10: a=1, b=2, *args=[3], **kwargs={'c': 4}
  """

  # Allow assignment on symbolic attributes.
  allow_symbolic_assignment = True

  # Signature of this function.
  signature: pg_typing.Signature

  def __init__(
      self,
      *args,
      root_path: Optional[object_utils.KeyPath] = None,
      override_args: bool = False,
      ignore_extra_args: bool = False,
      **kwargs):
    """Constructor.

    Args:
      *args: prebound positional arguments.
      root_path: The symbolic path for current object.
      override_args: If True, allows arguments provided during `__call__` to
        override existing bound arguments.
      ignore_extra_args: If True, unsupported arguments can be passed in
        during `__call__` without using them. Otherwise, calling with
        unsupported arguments will raise error.
      **kwargs: prebound keyword arguments.

    Raises:
      KeyError: constructor got unexpected arguments.
    """
    # NOTE(daiyip): Since Functor is usually late bound (until call time),
    # we pass `allow_partial=True` during functor construction.
    _ = kwargs.pop('allow_partial', None)

    varargs = None
    if len(args) > len(self.signature.args):
      if self.signature.varargs:
        varargs = list(args[len(self.signature.args):])
        args = args[:len(self.signature.args)]
      else:
        arg_phrase = object_utils.auto_plural(
            len(self.signature.args), 'argument')
        was_phrase = object_utils.auto_plural(len(args), 'was', 'were')
        raise TypeError(
            f'{self.signature.id}() takes {len(self.signature.args)} '
            f'positional {arg_phrase} but {len(args)} {was_phrase} given.')

    bound_kwargs = dict()
    for i, v in enumerate(args):
      if pg_typing.MISSING_VALUE != v:
        bound_kwargs[self.signature.args[i].name] = v

    if varargs is not None:
      bound_kwargs[self.signature.varargs.name] = varargs

    for k, v in kwargs.items():
      if pg_typing.MISSING_VALUE != v:
        if k in bound_kwargs:
          raise TypeError(
              f'{self.signature.id}() got multiple values for keyword '
              f'argument {k!r}.')
        bound_kwargs[k] = v

    default_args = set()
    non_default_args = set(bound_kwargs)

    for arg_spec in self.signature.named_args:
      if not arg_spec.value_spec.has_default:
        continue
      arg_name = arg_spec.name
      if arg_name not in non_default_args:
        default_args.add(arg_name)
      elif bound_kwargs[arg_name] == arg_spec.value_spec.default:
        default_args.add(arg_name)
        non_default_args.discard(arg_name)

    if self.signature.varargs and not varargs:
      default_args.add(self.signature.varargs.name)

    super().__init__(allow_partial=True,
                     root_path=root_path,
                     **bound_kwargs)

    self._non_default_args = non_default_args
    self._default_args = default_args
    self._specified_args = set(bound_kwargs)
    self._override_args = override_args
    self._ignore_extra_args = ignore_extra_args

  def _sym_clone(self, deep: bool, memo: Any = None) -> 'Functor':
    """Override to copy bound args."""
    other = super()._sym_clone(deep, memo)
    # pylint: disable=protected-access
    other._non_default_args = set(self._non_default_args)
    other._default_args = self._default_args
    other._specified_args = self._specified_args
    other._override_args = self._override_args
    other._ignore_extra_args = self._ignore_extra_args
    # pylint: enable=protected-access
    return typing.cast(Functor, other)

  def _on_change(
      self, field_updates: Dict[object_utils.KeyPath, base.FieldUpdate]):
    """Custom handling field change to update bound args."""
    for relative_path, update in field_updates.items():
      assert relative_path
      if len(relative_path) != 1:
        continue
      arg_name = str(relative_path)
      if update.field.default_value == update.new_value:
        if update.field.value.has_default:
          self._default_args.add(arg_name)
        self._non_default_args.discard(arg_name)
      else:
        self._default_args.discard(arg_name)
        self._non_default_args.add(arg_name)

      if update.new_value == pg_typing.MISSING_VALUE:
        self._specified_args.discard(arg_name)
      else:
        self._specified_args.add(arg_name)

  def __delattr__(self, name: str) -> None:
    """Discard a previously bound argument and reset to its default value."""
    del self._sym_attributes[name]
    if self.signature.get_value_spec(name).has_default:
      self._default_args.add(name)
    self._specified_args.discard(name)
    self._non_default_args.discard(name)

  def _sym_missing(self) -> Dict[str, Any]:
    """Returns missing values for Functor.

    Semantically unbound arguments are not missing, thus we only return partial
    bound arguments in `sym_missing`. As a result, a functor is partial only
    when any of its bound arguments is partial.

    Returns:
      A dict of missing key (or path) to missing value.
    """
    missing = dict()
    for k, v in self._sym_attributes.items():
      if pg_typing.MISSING_VALUE != v and isinstance(v, base.Symbolic):
        missing_child = v.sym_missing(flatten=False)
        if missing_child:
          missing[k] = missing_child
    return missing

  @property
  def specified_args(self) -> Set[str]:
    """Returns user specified argument names."""
    return self._specified_args

  @property
  def non_default_args(self) -> Set[str]:
    """Returns the names of bound arguments whose values are not the default."""
    return self._non_default_args

  @property
  def default_args(self) -> Set[str]:
    """Returns the names of bound argument whose values are the default."""
    return self._default_args

  @property
  def bound_args(self) -> Set[str]:
    """Returns bound argument names."""
    return self._non_default_args | self._default_args

  @property
  def unbound_args(self) -> Set[str]:
    """Returns unbound argument names."""
    return set([name for name in self._sym_attributes.keys()
                if name not in self.bound_args])

  @property
  def is_fully_bound(self) -> bool:
    """Returns if all arguments of functor is bound."""
    return (len(self._non_default_args) + len(self._default_args)
            == len(self._sym_attributes))

  @abc.abstractmethod
  def _call(self, *args, **kwargs) -> Callable:  # pylint: disable=g-bare-generic
    """Actual function logic. Subclasses should implement this method."""

  def __call__(self, *args, **kwargs) -> Any:
    """Call with late bound arguments.

    Args:
      *args: list arguments.
      **kwargs: keyword arguments.

    Returns:
      Any.

    Raises:
      TypeError: got multiple values for arguments or extra argument name.
    """
    override_args = kwargs.pop('override_args', self._override_args)
    ignore_extra_args = kwargs.pop('ignore_extra_args', self._ignore_extra_args)

    if len(args) > len(self.signature.args) and not self.signature.has_varargs:
      if ignore_extra_args:
        args = args[:len(self.signature.args)]
      else:
        arg_phrase = object_utils.auto_plural(
            len(self.signature.args), 'argument')
        was_phrase = object_utils.auto_plural(len(args), 'was', 'were')
        raise TypeError(
            f'{self.signature.id}() takes {len(self.signature.args)} '
            f'positional {arg_phrase} but {len(args)} {was_phrase} given.')

    keyword_args = {
        k: v for k, v in self._sym_attributes.items()
        if k in self._specified_args
    }
    assert len(keyword_args) == len(self._specified_args)

    # Work out varargs when positional arguments are provided.
    varargs = None
    if self.signature.has_varargs:
      varargs = list(args[len(self.signature.args):])
      if flags.is_type_check_enabled():
        varargs = self.signature.varargs.value_spec.apply(
            varargs, root_path=self.sym_path + self.signature.varargs.name)
      args = args[:len(self.signature.args)]

    # Convert positional arguments to keyword arguments so we can map them back
    # later.
    for i in range(len(args)):
      arg_spec = self.signature.args[i]
      arg_name = arg_spec.name
      if arg_name in self._specified_args:
        if not override_args:
          raise TypeError(
              f'{self.signature.id}() got new value for argument {arg_name!r} '
              f'from position {i}, but \'override_args\' is set to False. '
              f'Old value: {keyword_args[arg_name]!r}, new value: {args[i]!r}.')
      arg_value = args[i]
      if flags.is_type_check_enabled():
        arg_value = arg_spec.value_spec.apply(
            arg_value, root_path=self.sym_path + arg_name)
      keyword_args[arg_name] = arg_value

    for arg_name, arg_value in kwargs.items():
      if arg_name in self._specified_args:
        if not override_args:
          raise TypeError(
              f'{self.signature.id}() got new value for argument {arg_name!r} '
              f'from keyword argument, while \'override_args\' is set to '
              f'False. Old value: {keyword_args[arg_name]!r}, '
              f'new value: {arg_value!r}.')
      arg_spec = self.signature.get_value_spec(arg_name)
      if arg_spec and flags.is_type_check_enabled():
        arg_value = arg_spec.apply(
            arg_value, root_path=self.sym_path + arg_name)
        keyword_args[arg_name] = arg_value
      elif not ignore_extra_args:
        raise TypeError(
            f'{self.signature.id}() got an unexpected '
            f'keyword argument {arg_name!r}.')

    # Use positional arguments if possible. This allows us to handle varargs
    # with simplicity.
    list_args = []
    missing_required_arg_names = []
    for arg in self.signature.args:
      if arg.name in keyword_args:
        list_args.append(keyword_args[arg.name])
        del keyword_args[arg.name]
      elif arg.value_spec.default != pg_typing.MISSING_VALUE:
        list_args.append(arg.value_spec.default)
      else:
        missing_required_arg_names.append(arg.name)

    if missing_required_arg_names:
      arg_phrase = object_utils.auto_plural(
          len(missing_required_arg_names), 'argument')
      args_str = object_utils.comma_delimited_str(missing_required_arg_names)
      raise TypeError(
          f'{self.signature.id}() missing {len(missing_required_arg_names)} '
          f'required positional {arg_phrase}: {args_str}.')

    if self.signature.has_varargs:
      prebound_varargs = keyword_args.pop(self.signature.varargs.name, None)
      varargs = varargs or prebound_varargs
      if varargs:
        list_args.extend(varargs)

    return_value = self._call(*list_args, **keyword_args)
    if self.signature.return_value and flags.is_type_check_enabled():
      return_value = self.signature.return_value.apply(
          return_value, root_path=self.sym_path + 'returns')
    if flags.is_tracking_origin() and isinstance(return_value, base.Symbolic):
      return_value.sym_setorigin(self, 'return')
    return return_value


def functor(
    args: Optional[List[Union[
        Tuple[Union[str, pg_typing.KeySpec], pg_typing.ValueSpec, str],
        Tuple[Union[str, pg_typing.KeySpec], pg_typing.ValueSpec, str, Any]]]
    ] = None,    # pylint: disable=bad-continuation
    returns: Optional[pg_typing.ValueSpec] = None,
    base_class: Optional[Type[Functor]] = None,
    **kwargs):
  """Function/Decorator for creating symbolic function from regular function.

  Example::

    # Create a symbolic function without specifying the
    # validation rules for arguments.
    @pg.functor
    def foo(x, y):
      return x + y

    f = foo(1, 2)
    assert f() == 3

    # Create a symbolic function with specifying the
    # the validation rules for argument 'a', 'args', and 'kwargs'.
    @pg.functor([
      ('a', pg.typing.Int()),
      ('b', pg.typing.Float()),
      ('args', pg.List(pg.typing.Int())),
      (pg.typing.StrKey(), pg.typing.Int())
    ])
    def bar(a, b, c, *args, **kwargs):
      return a * b / c + sum(args) + sum(kwargs.values())

  See :class:`pyglove.Functor` for more details on symbolic function.

  Args:
    args: A list of tuples that defines the schema for function arguments.
      Please see `functor_class` for detailed explanation of `args`.
    returns: Optional value spec for return value.
    base_class: Optional base class derived from `symbolic.Functor`. If None,
      returning functor will inherit from `symbolic.Functor`.
    **kwargs: Keyword arguments for infrequently used options:
      Acceptable keywords are:

      * `serialization_key`: An optional string to be used as the serialization
        key for the class during `sym_jsonify`. If None, `cls.type_name` will
        be used. This is introduced for scenarios when we want to relocate a
        class, before the downstream can recognize the new location, we need
        the class to serialize it using previous key.
      * `additional_keys`: An optional list of strings as additional keys to
        deserialize an object of the registered class. This can be useful
        when we need to relocate or rename the registered class while being
        able to load existing serialized JSON values.

  Returns:
    A function that converts a regular function into a symbolic function.
  """
  if inspect.isfunction(args):
    assert returns is None
    assert base_class is None
    return functor_class(
        typing.cast(Callable[..., Any], args),
        add_to_registry=True, **kwargs)
  return lambda fn: functor_class(  # pylint: disable=g-long-lambda
      fn, args, returns, base_class, add_to_registry=True, **kwargs)


def functor_class(
    func: Callable,  # pylint: disable=g-bare-generic
    args: Optional[List[Union[
        Tuple[Tuple[str, pg_typing.KeySpec], pg_typing.ValueSpec, str],
        Tuple[Tuple[str, pg_typing.KeySpec], pg_typing.ValueSpec, str, Any]]]
    ] = None,   # pylint: disable=bad-continuation
    returns: Optional[pg_typing.ValueSpec] = None,
    base_class: Optional[Type['Functor']] = None,
    serialization_key: Optional[str] = None,
    additional_keys: Optional[List[str]] = None,
    add_to_registry: bool = False,
) -> Type[Functor]:
  """Returns a functor class from a function.

  Args:
    func: Function to be wrapped into a functor.
    args: Symbolic args specification. `args` is a list of tuples, each
      describes an argument from the input
      function. Each tuple is the format of:  (<argumment-name>, <value-spec>,
      [description], [metadata-objects]).  `argument-name` - a `str` or
      `pg_typing.StrKey` object. When `pg_typing.StrKey` is used, it
      describes the wildcard keyword argument. `value-spec` - a
      `pg_typing.ValueSpec` object or equivalent, e.g. primitive values which
      will be converted to ValueSpec implementation according to its type and
      used as its default value. `description` - a string to describe the
      agument. `metadata-objects` - an optional list of any type, which can be
      used to generate code according to the schema.
      There are notable rules in filling the `args`: 1) When `args` is None or
      arguments from the function signature are missing from it,
      `schema.Field` for these fields will be automatically generated and
      inserted into `args`.  That being said, every arguments in input
      function will have a `schema.Field` counterpart in
      `Functor.schema.fields` sorted by the declaration order of each argument
      in the function signature ( other than the order in `args`).  2) Default
      argument values are specified along with function definition as regular
      python functions, instead of being set at `schema.Field` level. But
      validation rules can be set using `args` and apply to argument values.
      For example::

          @pg.functor([('c', pg.typing.Int(min_value=0), 'Arg c')])
          def foo(a, b, c=1, **kwargs):
            return a + b + c + sum(kwargs.values())
            assert foo.schema.fields() == [
                pg.typing.Field('a', pg.Any(), 'Argument a'.),
                pg.typing.Field('b', pg.Any(), 'Argument b'.),
                pg.typing.Field('c', pg.typing.Int(), 'Arg c.),
                pg.typing.Filed(
                    pg.typing.StrKey(), pg.Any(), 'Other arguments.')
            ]
            # Prebind a=1, b=2, with default value c=1.
            assert foo(1, 2)() == 4

    returns: Optional schema specification for the return value.
    base_class: Optional base class (derived from `symbolic.Functor`).
      If None, returned type will inherit from `Functor` directly.
    serialization_key: An optional string to be used as the serialization key
      for the class during `sym_jsonify`. If None, `cls.type_name` will be used.
      This is introduced for scenarios when we want to relocate a class, before
      the downstream can recognize the new location, we need the class to
      serialize it using previous key.
    additional_keys: An optional list of strings as additional keys to
      deserialize an object of the registered class. This can be useful
      when we need to relocate or rename the registered class while being able
      to load existing serialized JSON values.
    add_to_registry: If True, the newly created functor class will be added to
      the registry for deserialization.
  Returns:
    `symbolic.Functor` subclass that wraps input function.

  Raises:
    KeyError: names of symbolic arguments are not compatible with
      function signature.
    TypeError: types of symbolic arguments are not compatible with
      function signature.
    ValueError: default values of symbolic arguments are not compatible
      with  function signature.
  """
  signature = pg_typing.get_signature(func)
  arg_fields = pg_typing.get_arg_fields(signature, args)
  if returns is not None and pg_typing.MISSING_VALUE != returns.default:
    raise ValueError('return value spec should not have default value.')

  base_class = base_class or Functor

  class _Functor(base_class):
    """Functor wrapper for input function."""

    # Disable auto register so we can use function module and name
    # for registration later.
    auto_register = False

    def _call(self, *args, **kwargs):
      return func(*args, **kwargs)

  cls = _Functor
  cls.__name__ = signature.name
  cls.__qualname__ = signature.qualname
  cls.__module__ = signature.module_name
  cls.__doc__ = func.__doc__

  # Enable automatic registration for subclass.
  cls.auto_register = True  # pylint: disable=protected-access

  # Generate init_arg_list from signature.
  init_arg_list = [arg.name for arg in signature.args]
  if signature.varargs:
    init_arg_list.append(f'*{signature.varargs.name}')
  schema_utils.update_schema(
      cls,
      arg_fields,
      init_arg_list=init_arg_list,
      serialization_key=serialization_key,
      additional_keys=additional_keys,
      add_to_registry=add_to_registry)

  # Update signature with symbolic value specs.
  def _value_spec_by_name(name: str):
    field = cls.schema.get_field(name)
    assert field is not None
    return field.value
  varkw_field = cls.schema.dynamic_field
  assert signature.has_varkw == (varkw_field is not None), varkw_field
  signature = pg_typing.Signature(
      callable_type=signature.callable_type,
      name=signature.name,
      module_name=signature.module_name,
      qualname=signature.qualname,
      args=[
          pg_typing.Argument(arg.name, _value_spec_by_name(arg.name))
          for arg in signature.args
      ],
      kwonlyargs=[
          pg_typing.Argument(arg.name, _value_spec_by_name(arg.name))
          for arg in signature.kwonlyargs
      ],
      varargs=(
          pg_typing.Argument(signature.varargs.name,
                             _value_spec_by_name(signature.varargs.name))
          if signature.varargs else None),
      varkw=(pg_typing.Argument(signature.varkw.name, varkw_field.value)
             if signature.has_varkw else None),
      return_value=returns or signature.return_value)
  setattr(cls, 'signature', signature)

  # Update signature for the __init__ method.
  varargs = None
  if signature.varargs:
    # For variable positional arguments, PyType uses the element type as
    # anntoation. Therefore we need to use the element type to generate
    # the right annotation.
    varargs_spec = signature.varargs.value_spec
    assert isinstance(varargs_spec, pg_typing.List), varargs_spec
    varargs = pg_typing.Argument(signature.varargs.name, varargs_spec.element)

  init_signature = pg_typing.Signature(
      callable_type=pg_typing.CallableType.FUNCTION,
      name='__init__',
      module_name=signature.module_name,
      qualname=f'{signature.name}.__init__',
      args=[
          pg_typing.Argument('self', pg_typing.Any())
      ] + signature.args,
      kwonlyargs=signature.kwonlyargs,
      varargs=varargs,
      varkw=signature.varkw)
  pseudo_init = init_signature.make_function(['pass'])

  @functools.wraps(pseudo_init)
  def _init(self, *args, **kwargs):
    Functor.__init__(self, *args, **kwargs)
  setattr(cls, '__init__', _init)
  return cls


def as_functor(
    func: Callable,  # pylint: disable=g-bare-generic
    ignore_extra_args: bool = False) -> Functor:
  """Make a functor object from a regular python function.

  NOTE(daiyip): This method is designed to create on-the-go functor object,
  usually for lambdas. To create a reusable functor class, please use
  `functor_class` method.

  Args:
    func: A regular python function.
    ignore_extra_args: If True, extra argument which is not acceptable by `func`
      will be ignored.

  Returns:
    Functor object from input function.
  """
  return functor_class(func)(ignore_extra_args=ignore_extra_args)  # pytype: disable=not-instantiable

