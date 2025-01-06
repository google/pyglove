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
import contextlib
import functools
import inspect
import threading
import types
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pyglove.core import typing as pg_typing
from pyglove.core import utils
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags
from pyglove.core.symbolic import object as pg_object


class Functor(pg_object.Object, utils.Functor):
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

  Or created by subclassing ``pg.Functor``::

    class Sum(pg.Functor):
      a: int
      b: int = 1

      def _call(self) -> int:
        return self.a + self.b

  Usage on subclassed functors is the same as functors created from functions.
  """

  # Allow assignment on symbolic attributes.
  allow_symbolic_assignment = True

  # Key for storing override members during call.
  _TLS_OVERRIDE_MEMBERS_KEY = '__override_members__'

  #
  # Customizable class traits.
  #

  @classmethod
  @property
  def is_subclassed_functor(cls) -> bool:
    """Returns True if this class is a subclassed Functor."""
    return cls.auto_schema

  @classmethod
  def _update_signatures_based_on_schema(cls):
    # Update the return value of subclassed functors.
    if cls.is_subclassed_functor:  # pylint: disable=using-constant-test
      private_call_signature = pg_typing.Signature.from_callable(
          cls._call, auto_typing=True, auto_doc=True,
      )
      if (
          len(private_call_signature.args) > 1
          or private_call_signature.kwonlyargs
      ):
        raise TypeError(
            '`_call` of a subclassed Functor should take no argument. '
            f'Encountered: {cls._call}.'
        )
      cls.__schema__.metadata['returns'] = private_call_signature.return_value

    # Update __init_ signature.
    init_signature = pg_typing.Signature.from_schema(
        cls.__schema__,
        name='__init__',
        module_name=cls.__module__,
        qualname=cls.__qualname__,
    )

    pseudo_init = init_signature.make_function(['pass'])

    # Save the original `Functor.__init__` before overriding it.
    if not hasattr(cls, '__orig_init__'):
      setattr(cls, '__orig_init__', cls.__init__)

    @utils.explicit_method_override
    @functools.wraps(pseudo_init)
    def _init(self, *args, **kwargs):
      self.__class__.__orig_init__(self, *args, **kwargs)

    setattr(cls, '__init__', _init)

    # Update __call__ signature.
    call_signature = pg_typing.Signature.from_schema(
        cls.__schema__,
        name='__call__',
        module_name=cls.__module__,
        qualname=cls.__qualname__,
        is_method=False,
    )
    setattr(cls, '__signature__', call_signature)

  def __new__(cls, *args, **kwargs):
    instance = object.__new__(cls)
    if flags.should_call_functors_during_init():
      instance.__init__(*args, **kwargs)
      return instance()
    return instance

  @utils.explicit_method_override
  def __init__(
      self,
      *args,
      root_path: Optional[utils.KeyPath] = None,
      override_args: bool = False,
      ignore_extra_args: bool = False,
      **kwargs,
  ):
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
    signature = self.__signature__
    if len(args) > len(signature.args):
      if signature.varargs:
        varargs = list(args[len(signature.args) :])
        args = args[: len(signature.args)]
      else:
        arg_phrase = utils.auto_plural(len(signature.args), 'argument')
        was_phrase = utils.auto_plural(len(args), 'was', 'were')
        raise TypeError(
            f'{signature.id}() takes {len(signature.args)} '
            f'positional {arg_phrase} but {len(args)} {was_phrase} given.'
        )

    bound_kwargs = dict()
    for i, v in enumerate(args):
      if pg_typing.MISSING_VALUE != v:
        bound_kwargs[signature.args[i].name] = v

    if varargs is not None:
      bound_kwargs[signature.varargs.name] = varargs

    for k, v in kwargs.items():
      if pg_typing.MISSING_VALUE != v:
        if k in bound_kwargs:
          raise TypeError(
              f'{signature.id}() got multiple values for keyword '
              f'argument {k!r}.'
          )
        bound_kwargs[k] = v

    default_args = set()
    non_default_args = set(bound_kwargs)

    for arg_spec in signature.named_args:
      if not arg_spec.value_spec.has_default:
        continue
      arg_name = arg_spec.name
      if arg_name not in non_default_args:
        default_args.add(arg_name)
      elif bound_kwargs[arg_name] == arg_spec.value_spec.default:
        default_args.add(arg_name)
        non_default_args.discard(arg_name)

    if signature.varargs and not varargs:
      default_args.add(signature.varargs.name)

    super().__init__(allow_partial=True,
                     root_path=root_path,
                     **bound_kwargs)

    self._non_default_args = non_default_args
    self._default_args = default_args
    self._specified_args = set(bound_kwargs)
    self._override_args = override_args
    self._ignore_extra_args = ignore_extra_args

    # For subclassed Functor, we use thread-local storage for storing temporary
    # member overrides from the arguments during functor call.
    self._tls = threading.local() if self.is_subclassed_functor else None

  def _sym_inferred(self, key: str, **kwargs: Any) -> Any:
    """Overrides method to allow member overrides during call."""
    if self._tls is not None:
      overrides = getattr(self._tls, Functor._TLS_OVERRIDE_MEMBERS_KEY, {})
      v = overrides.get(key, pg_typing.MISSING_VALUE)
      if pg_typing.MISSING_VALUE != v:
        return overrides[key]
    return super()._sym_inferred(key, **kwargs)

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

  def _on_change(self, field_updates: Dict[utils.KeyPath, base.FieldUpdate]):
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
    if self.__signature__.get_value_spec(name).has_default:
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

  def _call(self, *args, **kwargs) -> Callable:  # pylint: disable=g-bare-generic
    """Actual function logic. Subclasses should implement this method."""
    raise NotImplementedError()

  # TODO(b/183649930): We pretend that _call is not abstract to avoid
  # [not-instantiable] errors from pytype.
  if not typing.TYPE_CHECKING:
    _call = abc.abstractmethod(_call)

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
    args, kwargs = self._parse_call_time_overrides(*args, **kwargs)
    signature = self.__signature__

    if self.is_subclassed_functor:
      for arg_spec, arg_value in zip(signature.args, args):
        kwargs[arg_spec.name] = arg_value

      # Temporarily override members with argument values from the call.
      with self._apply_call_time_overrides_to_members(**kwargs):
        return_value = self._call()
    else:
      return_value = self._call(*args, **kwargs)

    # Return value check.
    if (
        signature.return_value
        and flags.is_type_check_enabled()
        and pg_typing.MISSING_VALUE != return_value
    ):
      return_value = signature.return_value.apply(
          return_value, root_path=self.sym_path + 'returns'
      )
    if flags.is_tracking_origin() and isinstance(return_value, base.Symbolic):
      return_value.sym_setorigin(self, 'return')
    return return_value

  @contextlib.contextmanager
  def _apply_call_time_overrides_to_members(self, **kwargs):
    """Overrides member values within the scope."""
    assert self._tls is not None
    setattr(self._tls, Functor._TLS_OVERRIDE_MEMBERS_KEY, kwargs)
    try:
      yield
    finally:
      delattr(self._tls, Functor._TLS_OVERRIDE_MEMBERS_KEY)

  def _parse_call_time_overrides(
      self, *args, **kwargs
  ) -> Tuple[List[Any], Dict[str, Any]]:
    """Parses positional and keyword arguments from call-time overrides."""
    override_args = kwargs.pop('override_args', self._override_args)
    ignore_extra_args = kwargs.pop('ignore_extra_args', self._ignore_extra_args)

    signature = self.__signature__
    if len(args) > len(signature.args) and not signature.has_varargs:
      if ignore_extra_args:
        args = args[: len(signature.args)]
      else:
        arg_phrase = utils.auto_plural(len(signature.args), 'argument')
        was_phrase = utils.auto_plural(len(args), 'was', 'were')
        raise TypeError(
            f'{signature.id}() takes {len(signature.args)} '
            f'positional {arg_phrase} but {len(args)} {was_phrase} given.'
        )

    keyword_args = {
        k: v for k, v in self._sym_attributes.items()
        if k in self._specified_args
    }
    assert len(keyword_args) == len(self._specified_args)

    # Work out varargs when positional arguments are provided.
    varargs = None
    if signature.has_varargs:
      varargs = list(args[len(signature.args) :])
      if flags.is_type_check_enabled():
        varargs = [
            signature.varargs.value_spec.element.value.apply(
                v, root_path=self.sym_path + signature.varargs.name
            )
            for v in varargs
        ]
      args = args[: len(signature.args)]

    # Convert positional arguments to keyword arguments so we can map them back
    # later.
    for i in range(len(args)):
      arg_spec = signature.args[i]
      arg_name = arg_spec.name
      if arg_name in self._specified_args:
        if not override_args:
          raise TypeError(
              f'{signature.id}() got new value for argument {arg_name!r} '
              f"from position {i}, but 'override_args' is set to False. "
              f'Old value: {keyword_args[arg_name]!r}, new value: {args[i]!r}.'
          )
      arg_value = args[i]
      if flags.is_type_check_enabled():
        arg_value = arg_spec.value_spec.apply(
            arg_value, root_path=self.sym_path + arg_name)
      keyword_args[arg_name] = arg_value

    for arg_name, arg_value in kwargs.items():
      if arg_name in self._specified_args:
        if not override_args:
          raise TypeError(
              f'{signature.id}() got new value for argument {arg_name!r} '
              "from keyword argument, while 'override_args' is set to "
              f'False. Old value: {keyword_args[arg_name]!r}, '
              f'new value: {arg_value!r}.'
          )
      arg_spec = signature.get_value_spec(arg_name)
      if arg_spec and flags.is_type_check_enabled():
        arg_value = arg_spec.apply(
            arg_value, root_path=self.sym_path + arg_name)
        keyword_args[arg_name] = arg_value
      elif not ignore_extra_args:
        raise TypeError(
            f'{signature.id}() got an unexpected keyword argument {arg_name!r}.'
        )

    # Use positional arguments if possible. This allows us to handle varargs
    # with simplicity.
    list_args = []
    missing_required_arg_names = []
    for arg in signature.args:
      if arg.name in keyword_args:
        list_args.append(keyword_args[arg.name])
        del keyword_args[arg.name]
      elif arg.value_spec.default != pg_typing.MISSING_VALUE:
        list_args.append(arg.value_spec.default)
      else:
        missing_required_arg_names.append(arg.name)

    if missing_required_arg_names:
      arg_phrase = utils.auto_plural(
          len(missing_required_arg_names), 'argument'
      )
      args_str = utils.comma_delimited_str(missing_required_arg_names)
      raise TypeError(
          f'{signature.id}() missing {len(missing_required_arg_names)} '
          f'required positional {arg_phrase}: {args_str}.'
      )

    if signature.has_varargs:
      prebound_varargs = keyword_args.pop(signature.varargs.name, None)
      varargs = varargs or prebound_varargs
      if varargs:
        list_args.extend(varargs)
    return list_args, keyword_args


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
    **kwargs: Keyword arguments for infrequently used options: Acceptable
      keywords are:  * `serialization_key`: An optional string to be used as the
      serialization key for the class during `sym_jsonify`. If None,
      `cls.__type_name__` will be used. This is introduced for scenarios when we
      want to relocate a class, before the downstream can recognize the new
      location, we need the class to serialize it using previous key. *
      `additional_keys`: An optional list of strings as additional keys to
      deserialize an object of the registered class. This can be useful when we
      need to relocate or rename the registered class while being able to load
      existing serialized JSON values.

  Returns:
    A function that converts a regular function into a symbolic function.
  """
  if inspect.isfunction(args):
    assert returns is None
    return functor_class(
        typing.cast(Callable[..., Any], args),
        base_class=base_class,
        add_to_registry=True,
        **kwargs,
    )
  return lambda fn: functor_class(  # pylint: disable=g-long-lambda  # pytype: disable=wrong-arg-types
      fn, args, returns,
      base_class=base_class,
      add_to_registry=True,
      **kwargs)


def functor_class(
    func: types.FunctionType,
    args: Union[
        List[Union[pg_typing.Field, pg_typing.FieldDef]],
        Dict[pg_typing.FieldKeyDef, pg_typing.FieldValueDef],
        None
    ] = None,
    returns: Optional[pg_typing.ValueSpec] = None,
    base_class: Optional[Type[Functor]] = None,
    *,
    auto_doc: bool = False,
    auto_typing: bool = False,
    serialization_key: Optional[str] = None,
    additional_keys: Optional[List[str]] = None,
    add_to_registry: bool = False,
) -> Type[Functor]:
  """Returns a functor class from a function.

  Args:
    func: Function to be wrapped into a functor.
    args: Symbolic args specification. `args` is a list of tuples, each
      describes an argument from the input function. Each tuple is the format of
      (<argumment-name>, <value-spec>, [description], [metadata-objects]).
      `argument-name` - a `str` or `pg_typing.StrKey` object. When
      `pg_typing.StrKey` is used, it describes the wildcard keyword argument.
      `value-spec` - a `pg_typing.ValueSpec` object or equivalent, e.g.
      primitive values which will be converted to ValueSpec implementation
      according to its type and used as its default value. `description` - a
      string to describe the agument. `metadata-objects` - an optional list of
      any type, which can be used to generate code according to the schema.
      There are notable rules in filling the `args`: 1) When `args` is None or
      arguments from the function signature are missing from it, `schema.Field`
      for these fields will be automatically generated and inserted into `args`.
      That being said, every arguments in input function will have a
      `schema.Field` counterpart in `Functor.__schema__.fields` sorted by the
      declaration order of each argument in the function signature ( other than
      the order in `args`). 2) Default argument values are specified along with
      function definition as regular python functions, instead of being set at
      `schema.Field` level. But validation rules can be set using `args` and
      apply to argument values.
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
    base_class: Optional base class (derived from `symbolic.Functor`). If None,
      returned type will inherit from `Functor` directly.
    auto_doc: If True, the descriptions of argument fields will be inherited
      from funciton docstr if they are not explicitly specified through
      ``args``.
    auto_typing: If True, the value spec for constraining each argument will be
      inferred from its annotation. Otherwise the value specs for all arguments
      will be ``pg.typing.Any()``.
    serialization_key: An optional string to be used as the serialization key
      for the class during `sym_jsonify`. If None, `cls.__type_name__` will be
      used. This is introduced for scenarios when we want to relocate a class,
      before the downstream can recognize the new location, we need the class to
      serialize it using previous key.
    additional_keys: An optional list of strings as additional keys to
      deserialize an object of the registered class. This can be useful when we
      need to relocate or rename the registered class while being able to load
      existing serialized JSON values.
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
  if not inspect.isfunction(func):
    raise TypeError(f'{func!r} is not a function.')

  class _Functor(base_class or Functor):
    """Functor wrapper for input function."""

    # The schema for function-based Functor will be inferred from the function
    # signature. Therefore we do not infer the schema automatically during class
    # creation.
    auto_schema = False

    # Do not infer symbolic fields from annotations, since this functor is
    # created from function definition which does not have class-level
    # attributes.
    infer_symbolic_fields_from_annotations = True

    def _call(self, *args, **kwargs):
      return func(*args, **kwargs)

  cls = typing.cast(Type[Functor], _Functor)
  cls.__name__ = func.__name__
  cls.__qualname__ = func.__qualname__
  cls.__module__ = getattr(func, '__module__', 'wrapper')
  cls.__doc__ = func.__doc__

  # Enable automatic registration for subclass.
  cls.auto_register = True

  # Apply function schema.
  cls.apply_schema(
      pg_typing.schema(
          func, args, returns, auto_doc=auto_doc, auto_typing=auto_typing
      )
  )

  # Register functor class for deserialization if needed.
  if add_to_registry:
    cls.register_for_deserialization(serialization_key, additional_keys)
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
