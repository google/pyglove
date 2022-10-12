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
"""Symbolic Object Model.

This module enables Symbolic Object Model, which defines and implements the
symbolic interfaces for common Python types (e.g. symbolic class, symbolic
function and symbolic container types). Based on symbolic types, symbolic
objects can be created, which can be then inspected, manipulated symbolically.
"""

import abc
import contextlib
import copy
import enum
import functools
import inspect
import json
import re
import sys
import threading
import traceback
import typing

from pyglove.core import object_utils
from pyglove.core import typing as schema_lib


SymbolicT = typing.TypeVar('SymbolicT', bound='Symbolic')
SymbolicObjectT = typing.TypeVar('SymbolicObjectT', bound='Object')


_TYPE_NAME_KEY = '_type'
_ALLOW_EMPTY_FIELD_DESCRIPTION = True
_ALLOW_REPEATED_CLASS_REGISTRATION = True
_STACKTRACE_LIMIT = 10

# Thread-local states.
_thread_local_state = threading.local()
_TLS_ACCESSOR_WRITABLE = '_accessor_writable'
_TLS_ALLOW_PARTIAL = '_allow_partial'
_TLS_ENABLE_CHANGE_NOTIFICATION = '_enable_change_notification'
_TLS_ENABLE_ORIGIN_TRACKING = '_enable_origin_tracking'
_TLS_ENABLE_TYPE_CHECK = '_enable_type_check'
_TLS_SEALED = '_sealed'


def allow_empty_field_description(allow: bool = True):
  """Allow empty field description, which is useful for testing purposes."""
  global _ALLOW_EMPTY_FIELD_DESCRIPTION
  _ALLOW_EMPTY_FIELD_DESCRIPTION = allow


def allow_repeated_class_registration(allow: bool = True):
  """Allow repeated class registration, which is useful for testing purposes."""
  global _ALLOW_REPEATED_CLASS_REGISTRATION
  _ALLOW_REPEATED_CLASS_REGISTRATION = allow


def set_stacktrace_limit(limit: int):
  """Set stack trace limit for origin tracking."""
  global _STACKTRACE_LIMIT
  _STACKTRACE_LIMIT = limit


@contextlib.contextmanager
def _thread_local_state_scope(
    tls_key: typing.Text,
    value_in_scope: typing.Any,
    initial_value: typing.Any):
  """Context manager to set a thread local state within the scope."""
  previous_value = getattr(_thread_local_state, tls_key, initial_value)
  try:
    setattr(_thread_local_state, tls_key, value_in_scope)
    yield
  finally:
    setattr(_thread_local_state, tls_key, previous_value)


def notify_on_change(enabled: bool = True):
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
  return _thread_local_state_scope(
      _TLS_ENABLE_CHANGE_NOTIFICATION, enabled, True)


def _enabled_notification():
  """Returns True if change notification is enabled."""
  return getattr(_thread_local_state, _TLS_ENABLE_CHANGE_NOTIFICATION, True)


def enable_type_check(enabled: bool = True):
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
  return _thread_local_state_scope(
      _TLS_ENABLE_TYPE_CHECK, enabled, True)


def _enabled_type_check():
  """Returns True if runtme type check is enabled."""
  return getattr(_thread_local_state, _TLS_ENABLE_TYPE_CHECK, True)


def track_origin(enabled: bool = True):
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
  return _thread_local_state_scope(
      _TLS_ENABLE_ORIGIN_TRACKING, enabled, False)


def _is_tracking_origin() -> bool:
  """Returns if origin of symbolic object are being tracked."""
  return getattr(_thread_local_state, _TLS_ENABLE_ORIGIN_TRACKING, False)


def allow_writable_accessors(writable: typing.Optional[bool] = True):
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
  return _thread_local_state_scope(_TLS_ACCESSOR_WRITABLE, writable, None)


def _is_accessor_writable(value: 'Symbolic') -> bool:
  """Returns if a symbolic value is accessor writable."""
  writable_in_scope = getattr(_thread_local_state, _TLS_ACCESSOR_WRITABLE, None)
  if writable_in_scope is not None:
    return writable_in_scope
  return value.accessor_writable


def as_sealed(sealed: typing.Optional[bool] = True):
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
  return _thread_local_state_scope(_TLS_SEALED, sealed, None)


def _is_sealed(value: 'Symbolic') -> bool:
  """Returns if a symbolic value is sealed."""
  sealed_in_scope = getattr(_thread_local_state, _TLS_SEALED, None)
  if sealed_in_scope is not None:
    return sealed_in_scope
  return value.is_sealed


def allow_partial_values(allow: typing.Optional[bool] = True):
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
  return _thread_local_state_scope(_TLS_ALLOW_PARTIAL, allow, None)


def _allow_partial(value: 'Symbolic') -> bool:
  """Returns if a symbolic value is allowed to be partial."""
  allow_in_scope = getattr(_thread_local_state, _TLS_ALLOW_PARTIAL, None)
  if allow_in_scope is not None:
    return allow_in_scope
  return value.allow_partial


def members(
    fields: typing.List[typing.Union[
        schema_lib.Field, typing.Tuple[typing.Union[typing.Text,
                                                    schema_lib.KeySpec],
                                       schema_lib.ValueSpec, typing.Text],
        typing.Tuple[typing.Union[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text, typing.Any]]],
    metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None,
    init_arg_list: typing.Optional[typing.Sequence[typing.Text]] = None,
    **kwargs
) -> schema_lib.Decorator:
  """Function/Decorator for declaring symbolic fields for ``pg.Object``.

  Example::

    @pg.members([
      # Declare symbolic fields. Each field produces a symbolic attribute
      # for its object, which can be accessed by `self.<field_name>`.
      # Description is optional.
      ('x', pg.typing.Int(min_value=0, default=0), 'Description for `x`.'),
      ('y', pg.typing.Str(), 'Description for `y`.')
    ])
    class A(pg.Object):
      def sum(self):
        return self.x + self.y

    @pg.members([
      # Override field 'x' inherited from class A and make it more restrictive.
      ('x', pg.typing.Int(max_value=10, default=5)),
      # Add field 'z'.
      ('z', pg.typing.Bool().noneable())
    ])
    class B(A):
      pass

    @pg.members([
      # Declare dynamic fields: any keyword can be acceptable during `__init__`
      # and can be accessed using `self.<field_name>`.
      (pg.typing.StrKey(), pg.typing.Int())
    ])
    class D(B):
      pass

    @pg.members([
      # Declare dynamic fields: keywords started with 'foo' is acceptable.
      (pg.typing.StrKey('foo.*'), pg.typing.Int())
    ])
    class E(pg.Object):
      pass

  See :class:`pyglove.typing.ValueSpec` for supported value specifications.

  Args:
    fields: A list of pg.typing.Field or equivalent tuple representation as
      (<key>, <value-spec>, [description], [metadata-objects]). `key` should be
      a string. `value-spec` should be schema_lib.ValueSpec classes or
      equivalent, e.g. primitive values which will be converted to ValueSpec
      implementation according to its type and used as its default value.
      `description` is optional only when field overrides a field from its
      parent class. `metadata-objects` is an optional list of any type, which
      can be used to generate code according to the schema.
    metadata: Optional dict of user objects as class-level metadata which will
      be attached to class schema.
    init_arg_list: An optional sequence of strings as the positional argument
      list for `__init__`. This is helpful when symbolic attributes are
      inherited from base classes or the user want to change its order.
      If not provided, the `init_arg_list` will be automatically generated
      from symbolic attributes defined from ``pg.members`` in their declaration
      order, from the base classes to the subclass.
    **kwargs: Keyword arguments for infrequently used options.
      Acceptable keywords are:

      * `serialization_key`: An optional string to be used as the serialization
        key for the class during `sym_jsonify`. If None, `cls.type_name` will be
        used. This is introduced for scenarios when we want to relocate a class,
        before the downstream can recognize the new location, we need the class
        to serialize it using previous key.
      * `additional_keys`: An optional list of strings as additional keys to
        deserialize an object of the registered class. This can be useful
        when we need to relocate or rename the registered class while being able
        to load existing serialized JSON values.

  Returns:
    a decorator function that register the class or function with schema
      created from the fields.

  Raises:
    TypeError: Decorator cannot be applied on target class or keyword argument
      provided is not supported.
    KeyError: If type has already been registered in the registry.
    ValueError: schema cannot be created from fields.
  """
  serialization_key = kwargs.pop('serialization_key', None)
  additional_keys = kwargs.pop('additional_keys', None)
  if kwargs:
    raise TypeError(f'Unsupported keyword arguments: {list(kwargs.keys())!r}.')

  def _decorator(cls):
    """Decorator function that registers schema with an Object class."""
    update_schema(cls,
                  fields,
                  metadata=metadata,
                  init_arg_list=init_arg_list,
                  serialization_key=serialization_key,
                  additional_keys=additional_keys)
    return cls
  return typing.cast(schema_lib.Decorator, _decorator)


# Create an alias method for members.
schema = members


def update_schema(
    cls,
    fields: typing.List[typing.Union[
        schema_lib.Field,
        typing.Tuple[typing.Union[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text],
        typing.Tuple[typing.Union[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text, typing.Any]]],
    metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None,
    init_arg_list: typing.Optional[typing.Sequence[typing.Text]] = None,
    extend: bool = True,
    serialization_key: typing.Optional[typing.Text] = None,
    additional_keys: typing.Optional[typing.List[typing.Text]] = None,
    add_to_registry: bool = True) -> None:
  """Updates the schema for a ``pg.Object`` subclass.

  This function allows the user to update the symbolic fields associated
  with a symbolic class. It was intended to support meta-programming
  scenarios in which symbolic fileds are dynamically generated.

  Example::

    class A(pg.Object):
      pass

    # Add symbolic field 'x' to class A.
    pg.symbolic.update_schema(A, [
      ('x', schema.Int())
    ])

    # B inherits the symbolic field 'x' from A.
    class B(A):
      pass

    # Wipe out the symbolic field 'x' from B.
    pg.symbolic.update_schema(B, [], extend=False)

  See also: :func:`pyglove.members`, :func:`pyglove.functor` and
  :func:`pyglove.symbolize`.

  Args:
    cls: A symbolic Object subclass.
    fields: A list of `pg.typing.Field` or equivalent tuple representation as
      (<key>, <value-spec>, [description], [metadata-objects]). `key` should be
      a string. `value-spec` should be schema_lib.ValueSpec classes or
      equivalent, e.g. primitive values which will be converted to ValueSpec
      implementation according to its type and used as its default value.
      `description` is optional only when field overrides a field from its
      parent class. `metadata-objects` is an optional list of any type, which
      can be used to generate code according to the schema.
    metadata: Optional dict of user objects as class-level metadata which will
      be attached to class schema.
    init_arg_list: An optional sequence of strings as the positional argument
      list for `__init__`. This is helpful when symbolic attributes are
      inherited from base classes or the user want to change its order.
      If not provided, the `init_arg_list` will be automatically generated
      from symbolic attributes defined from ``pg.members`` in their declaration
      order, from the base classes to the subclass.
    extend: If True, extend existing schema using `fields`. Otherwise replace
      existing schema with a new schema created from `fields`.
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
  """
  metadata = metadata or {}
  init_arg_list = init_arg_list or metadata.pop('init_arg_list', None)
  cls_schema = _formalize_schema(schema_lib.create_schema(
      maybe_field_list=fields,
      name=cls.type_name,
      base_schema_list=[cls.schema] if extend else [],
      allow_nonconst_keys=True,
      metadata=metadata))

  setattr(cls, '__schema__', cls_schema)
  setattr(cls, '__sym_fields', schema_lib.Dict(cls_schema))
  setattr(cls, '__serialization_key__', serialization_key or cls.type_name)

  init_arg_list = init_arg_list or _auto_init_arg_list(cls)
  _validate_init_arg_list(init_arg_list, cls_schema)
  cls_schema.metadata['init_arg_list'] = init_arg_list

  if add_to_registry:
    _register_cls_for_deserialization(cls, serialization_key, additional_keys)

  cls._update_init_signature_based_on_schema()  # pylint: disable=protected-access
  cls._generate_sym_attributes_if_enabled()  # pylint: disable=protected-access


def _validate_init_arg_list(
    init_arg_list: typing.List[typing.Text],
    cls_schema: schema_lib.Schema) -> None:
  """Validate init arg list."""
  for i, arg in enumerate(init_arg_list):
    is_vararg = False
    if i == len(init_arg_list) - 1 and arg.startswith('*'):
      arg = arg[1:]
      is_vararg = True
    field = cls_schema.get_field(arg)
    if field is None:
      raise TypeError(
          f'Argument {arg!r} from `init_arg_list` is not defined as a '
          f'symbolic field. init_arg_list={init_arg_list!r}.')
    if is_vararg and not isinstance(field.value, schema_lib.List):
      raise TypeError(
          f'Variable positional argument {arg!r} should be declared with '
          f'`pg.typing.List(...)`. Encountered {field.value!r}.')


def _auto_init_arg_list(cls):
  """Generate the init_arg_list metadata from an pg.Object subclass."""
  # Inherit from the first non-empty base if they have the same signature.
  # This allows to bypass interface-only bases.
  init_arg_list = None
  for base in cls.__bases__:
    if issubclass(base, Object) and base.schema:
      if list(base.schema.keys()) == list(cls.schema.keys()):
        init_arg_list = base.init_arg_list
      else:
        break
  if init_arg_list is None:
    # Automatically generate from the field definitions in their
    # declaration order from base classes to subclasses.
    init_arg_list = [str(key) for key in cls.schema.fields.keys()
                     if isinstance(key, schema_lib.ConstStrKey)]
  return init_arg_list


def _register_cls_for_deserialization(
    cls,
    serialization_key: typing.Optional[typing.Text] = None,
    additional_keys: typing.Optional[typing.List[typing.Text]] = None):
  """Register a symbolic class for deserialization."""
  serialization_keys = []
  if serialization_key:
    serialization_keys.append(serialization_key)
  serialization_keys.append(cls.type_name)
  if additional_keys:
    serialization_keys.extend(additional_keys)

  # Register class with 'type' property.
  for key in serialization_keys:
    object_utils.JSONConvertible.register(
        key, cls, _ALLOW_REPEATED_CLASS_REGISTRATION)


def _formalize_schema(schema: schema_lib.Schema) -> schema_lib.Schema:  # pylint: disable=redefined-outer-name
  """Formalize schema by applying restrictions for symbolic Object."""

  def _formalize_field(path: object_utils.KeyPath, node: typing.Any) -> bool:
    """Formalize field."""
    if isinstance(node, schema_lib.Field):
      field = typing.cast(schema_lib.Field, node)
      if not _ALLOW_EMPTY_FIELD_DESCRIPTION and not field.description:
        raise ValueError(
            f'Field description must not be empty (path={path}).')

      field.value.set_default(
          field.apply(
              field.default_value,
              allow_partial=True,
              transform_fn=_symbolic_transform_fn(allow_partial=True)),
          use_default_apply=False)
      if isinstance(field.value, schema_lib.Dict):
        if field.value.schema is not None:
          field.value.schema.set_name(f'{schema.name}.{path.path}')
          object_utils.traverse(field.value.schema.fields, _formalize_field,
                                None, path)
      elif isinstance(field.value, schema_lib.List):
        _formalize_field(object_utils.KeyPath(0, path), field.value.element)
      elif isinstance(field.value, schema_lib.Tuple):
        for i, elem in enumerate(field.value.elements):
          _formalize_field(object_utils.KeyPath(i, path), elem)
      elif isinstance(field.value, schema_lib.Union):
        for i, c in enumerate(field.value.candidates):
          _formalize_field(
              object_utils.KeyPath(i, path),
              schema_lib.Field(field.key, c, 'Union sub-type.'))
    return True

  object_utils.traverse(schema.fields, _formalize_field)
  return schema


def _symbolic_transform_fn(allow_partial: bool):
  """Symbolic object transform function builder."""

  def _fn(path: object_utils.KeyPath, field: schema_lib.Field,
          value: typing.Any) -> typing.Any:
    """Transform schema-less List and Dict to symbolic."""
    if isinstance(value, Symbolic):
      return value
    if isinstance(value, dict):
      value_spec = schema_lib.ensure_value_spec(
          field.value, schema_lib.Dict(), path)
      value = Dict(
          value,
          value_spec=value_spec,
          allow_partial=allow_partial,
          root_path=path,
          # NOTE(daiyip): members are already checked and transformed
          # into final object, thus we simply pass through.
          # This prevents the Dict members from repeated validation
          # and transformation.
          pass_through=True)
    elif isinstance(value, list):
      value_spec = schema_lib.ensure_value_spec(
          field.value, schema_lib.List(schema_lib.Any()), path)
      value = List(
          value,
          value_spec=value_spec,
          allow_partial=allow_partial,
          root_path=path)
    return value

  return _fn


def boilerplate_class(
    cls_name: typing.Text,
    value: 'Object',
    init_arg_list: typing.Optional[typing.List[typing.Text]] = None,
    **kwargs) -> typing.Type['Object']:
  """Create a boilerplate class using a symbolic object.

  As the name indicates, a boilerplate class is a class that can be used
  as a boilerplate to create object.

  Implementation-wise it's a class that extends the type of input value, while
  setting the default values of its (inherited) schema using the value from
  input.

  An analogy to boilerplate class is prebound function.

    For example::

      # A regular function: correspond to a pg.Object subclass.
      def f(a, b, c)
        return a + b + c

      # A partially bound function: correspond to a boilerplate class created
      # from a partially bound object.
      def g(c):
        return f(1, 2, c)

      # A fully bound function: correspond to a boilerplate class created from
      # a fully bound object.
      def h():
        return f(1, 2, 3)

  Boilerplate class can be created with a value that is fully bound
  (like function `h` above), or partially bound (like function `g` above).
  Since boilerplate class extends the type of the input, we can rebind members
  of its instances as we modify the input.

  Here are a few examples::

    @pg.members([
      ('a', pg.typing.Str(), 'Field A.'),
      ('b', pg.typing.Int(), 'Field B.'),
    ])
    class A(pg.Object):
      pass

    A1 = pg.boilerplate_class('A1', A.partial(a='foo'))
    assert A1(b=1) == A(a='foo', b=1)

    A2 = pg.boilerplate_class('A2', A(a='bar', b=2))
    assert A2() == A(a='bar', b=2)

  Args:
    cls_name: Name of the boilerplate class.
    value: Value that is used as the default value of the boilerplate class.
    init_arg_list: An optional list of strings as __init__ positional arguments
      names.
    **kwargs: Keyword arguments for infrequently used options.
      Acceptable keywords are:

      * `serialization_key`: An optional string to be used as the serialization
        key for the class during `sym_jsonify`. If None, `cls.type_name` will
        be used. This is introduced for scenarios when we want to relocate a
        class, before the downstream can recognize the new location, we need
        the class to serialize it using previous key.
      * `additional_keys`: An optional list of strings as additional keys to
        deserialize an object of the registered class. This can be useful
        when we need to relocate or rename the registered class while being able
        to load existing serialized JSON values.

  Returns:
    A class which extends the input value's type, with its schema's default
      values set from the input value.

  Raises:
    TypeError: Keyword argumment provided is not supported.
  """
  if not isinstance(value, Object):
    raise ValueError('Argument \'value\' must be an instance of '
                     'symbolic.Object subclass.')

  serialization_key = kwargs.pop('serialization_key', None)
  additional_keys = kwargs.pop('additional_keys', None)
  if kwargs:
    raise TypeError(
        f'Unsupported keyword arguments: {list(kwargs.keys())!r}.')

  base_cls = value.__class__

  class _BoilerplateClass(base_cls):
    """Boilerplate class."""
    auto_register = False

    is_boilerplate = True

  caller_module = inspect.getmodule(inspect.stack()[1][0])
  cls_module = caller_module.__name__ if caller_module else '__main__'
  cls = _BoilerplateClass
  cls.__name__ = cls_name
  cls.__module__ = cls_module

  # Enable automatic registration for subclass.
  cls.auto_register = True  # pylint: disable=protected-access

  allow_partial = value.allow_partial
  def _freeze_field(path: object_utils.KeyPath,
                    field: schema_lib.Field,
                    value: typing.Any) -> typing.Any:
    # We do not do validation since Object is already in valid form.
    del path
    if not isinstance(field.key, schema_lib.ListKey):
      # Recursively freeze dict field.
      if isinstance(field.value, schema_lib.Dict) and field.value.schema:
        field.value.schema.apply(
            value, allow_partial=allow_partial, child_transform=_freeze_field)
        field.value.set_default(value)
        if all(f.frozen for f in field.value.schema.values()):
          field.value.freeze()
      else:
        if value != schema_lib.MISSING_VALUE:
          field.value.freeze(copy.deepcopy(value), apply_before_use=False)
        else:
          field.value.set_default(
              schema_lib.MISSING_VALUE, use_default_apply=False)
    return value

  with allow_writable_accessors():
    cls.schema.apply(
        value._sym_attributes,  # pylint: disable=protected-access
        allow_partial=allow_partial,
        child_transform=_freeze_field)

  _formalize_schema(cls.schema)
  if init_arg_list is not None:
    _validate_init_arg_list(init_arg_list, cls.schema)
    cls.schema.metadata['init_arg_list'] = init_arg_list
  setattr(cls, '__serialization_key__', serialization_key or cls.type_name)
  _register_cls_for_deserialization(cls, serialization_key, additional_keys)
  return cls


def functor(
    args: typing.Optional[typing.List[typing.Union[
        typing.Tuple[typing.Union[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text],
        typing.Tuple[typing.Union[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text, typing.Any]]]] = None,
    returns: typing.Optional[schema_lib.ValueSpec] = None,
    base_class: typing.Optional[typing.Type['Functor']] = None,
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

    # Create a symbolic function with specifiying the
    # the validation rules for argument 'a', 'args', and 'kwargs'.
    @pg.functor([
      ('a', pg.typing.Int()),
      ('b', pg.typing.Float()),
      ('args', pg.typing.List(pg.typing.Int())),
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
    A function that converts a regualr function into a symbolic function.
  """
  if inspect.isfunction(args):
    assert returns is None
    assert base_class is None
    return functor_class(
        typing.cast(typing.Callable[..., typing.Any], args),
        add_to_registry=True, **kwargs)
  return lambda fn: functor_class(  # pylint: disable=g-long-lambda
      fn, args, returns, base_class, add_to_registry=True, **kwargs)


def functor_class(
    func: typing.Callable,  # pylint: disable=g-bare-generic
    args: typing.Optional[typing.List[typing.Union[
        typing.Tuple[typing.Tuple[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text],
        typing.Tuple[typing.Tuple[typing.Text, schema_lib.KeySpec],
                     schema_lib.ValueSpec, typing.Text, typing.Any]]]] = None,
    returns: typing.Optional[schema_lib.ValueSpec] = None,
    base_class: typing.Optional[typing.Type['Functor']] = None,
    serialization_key: typing.Optional[typing.Text] = None,
    additional_keys: typing.Optional[typing.List[typing.Text]] = None,
    add_to_registry: bool = False,
) -> typing.Type['Functor']:
  """Returns a functor class from a function.

  Args:
    func: Function to be wrapped into a functor.
    args: Symbolic args specification. `args` is a list of tuples, each
      describes an argument from the input
      function. Each tuple is the format of:  (<argumment-name>, <value-spec>,
      [description], [metadata-objects]).  `argument-name` - a `str` or
      `schema_lib.StrKey` object. When `schema_lib.StrKey` is used, it
      describes the wildcard keyword argument. `value-spec` - a
      `schema_lib.ValueSpec` object or equivalent, e.g. primitive values which
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
            assert foo.arg_schema.values() == [
                pg.typing.Field('a', pg.typing.Any(), 'Argument a'.),
                pg.typing.Field('b', pg.typing.Any(), 'Argument b'.),
                pg.typing.Field('c', pg.typing.Int(), 'Arg c.),
                pg.typing.Filed(
                    pg.typing.StrKey(), pg.typing.Any(), 'Other arguments.')
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
  signature = schema_lib.get_signature(func)
  arg_fields = schema_lib.get_arg_fields(signature, args)
  if returns is not None and schema_lib.MISSING_VALUE != returns.default:
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
  update_schema(cls,
                arg_fields,
                init_arg_list=init_arg_list,
                serialization_key=serialization_key,
                additional_keys=additional_keys,
                add_to_registry=add_to_registry)

  # Update signature with symbolic value specs.
  value_spec_by_name = {field.key: field.value for field in arg_fields}
  signature = schema_lib.Signature(
      callable_type=signature.callable_type,
      name=signature.name,
      module_name=signature.module_name,
      qualname=signature.qualname,
      args=[
          schema_lib.Argument(arg.name, value_spec_by_name[arg.name])
          for arg in signature.args
      ],
      kwonlyargs=[
          schema_lib.Argument(arg.name, value_spec_by_name[arg.name])
          for arg in signature.kwonlyargs
      ],
      varargs=(
          schema_lib.Argument(signature.varargs.name,
                              value_spec_by_name[signature.varargs.name])
          if signature.varargs else None),
      varkw=(schema_lib.Argument(signature.varkw.name,
                                 value_spec_by_name[schema_lib.StrKey()])
             if signature.has_varkw else None),
      return_value=returns or signature.return_value)
  setattr(cls, 'signature', signature)

  # Update signature for the __init__ method.
  varargs = None
  if signature.varargs:
    # For variable positional arguments, PyType uses the element type as
    # anntoation. Therefore we need to use the element type to generate
    # the right annotation.
    varargs = schema_lib.Argument(
        signature.varargs.name, signature.varargs.value_spec.element)

  init_signature = schema_lib.Signature(
      callable_type=schema_lib.CallableType.FUNCTION,
      name='__init__',
      module_name=signature.module_name,
      qualname=f'{signature.name}.__init__',
      args=[
          schema_lib.Argument('self', schema_lib.Any())
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
    func: typing.Callable,  # pylint: disable=g-bare-generic
    ignore_extra_args: bool = False) -> 'Functor':
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


def from_json(json_value: typing.Any,
              *,
              allow_partial: bool = False,
              root_path: typing.Optional[object_utils.KeyPath] = None,
              **kwargs) -> typing.Any:
  """Deserializes a (maybe) symbolic value from JSON value.

  Example::

    @pg.members([
      ('x', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    a1 = A(1)
    json = a1.to_json()
    a2 = pg.from_json(json)
    assert pg.eq(a1, a2)

  Args:
    json_value: Input JSON value.
    allow_partial: Whether to allow elements of the list to be partial.
    root_path: KeyPath of loaded object in its object tree.
    **kwargs: Allow passing through keyword arguments to from_json of specific
      types.

  Returns:
    Deserialized value, which is
    * pg.Dict for dict.
    * pg.List for list.
    * (TODO:daiyip) symbolic.Tuple for tuple.
    * symbolic.Object for dict with '_type' property.
    * value itself.
  """
  if isinstance(json_value, Symbolic):
    return json_value

  kwargs.update({
      'allow_partial': allow_partial,
      'root_path': root_path,
  })
  if isinstance(json_value, list):
    if json_value and json_value[0] == _TUPLE_MARKER:
      if len(json_value) < 2:
        raise ValueError(
            object_utils.message_on_path(
                f'Tuple should have at least one element '
                f'besides \'{_TUPLE_MARKER}\'. Encountered: {json_value}',
                root_path))
      return tuple([
          from_json(v, allow_partial=allow_partial,
                    root_path=object_utils.KeyPath(i, root_path))
          for i, v in enumerate(json_value[1:])
      ])
    return List(json_value, **kwargs)
  elif isinstance(json_value, dict):
    if _TYPE_NAME_KEY not in json_value:
      return Dict.from_json(json_value, **kwargs)
    cls = object_utils.JSONConvertible.class_from_typename(
        json_value[_TYPE_NAME_KEY])
    if cls is None:
      raise TypeError(
          object_utils.message_on_path(
              f'Type name \'{json_value[_TYPE_NAME_KEY]}\' is not registered '
              f'with a symbolic.Object subclass.', root_path))
    del json_value[_TYPE_NAME_KEY]
    return cls.from_json(json_value, **kwargs)
  return json_value


def from_json_str(json_str: typing.Text,
                  *,
                  allow_partial: bool = False,
                  root_path: typing.Optional[object_utils.KeyPath] = None,
                  **kwargs) -> typing.Any:
  """Deserialize (maybe) symbolic object from JSON string.

  Example::

    @pg.members([
      ('x', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    a1 = A(1)
    json_str = a1.to_json_str()
    a2 = pg.from_json_str(json_str)
    assert pg.eq(a1, a2)

  Args:
    json_str: JSON string.
    allow_partial: If True, allow a partial symbolic object to be created.
      Otherwise error will be raised on partial value.
    root_path: The symbolic path used for the deserialized root object.
    **kwargs: Additional keyword arguments that will be passed to
      ``pg.from_json``.

  Returns:
    A deserialized value.
  """
  return from_json(
      json.loads(json_str),
      allow_partial=allow_partial,
      root_path=root_path,
      **kwargs)


def to_json(value: typing.Any, **kwargs) -> typing.Any:
  """Serializes a (maybe) symbolic value into a plain Python object.

  Example::

    @pg.members([
      ('x', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    a1 = A(1)
    json = a1.to_json()
    a2 = pg.from_json(json)
    assert pg.eq(a1, a2)

  Args:
    value: value to serialize. Applicable value types are:

      * Builtin python types: None, bool, int, float, string;
      * JSONConvertible types;
      * List types;
      * Dict types.

    **kwargs: Keyword arguments to pass to value.to_json if value is
      JSONConvertible.

  Returns:
    JSON value.
  """
  if isinstance(value, (type(None), bool, int, float, str)):
    return value
  if isinstance(value, Symbolic):
    return value.sym_jsonify(**kwargs)
  elif isinstance(value, object_utils.JSONConvertible):
    return value.to_json(**kwargs)
  elif isinstance(value, tuple):
    return [_TUPLE_MARKER] + to_json(list(value), **kwargs)
  elif isinstance(value, list):
    return [to_json(item, **kwargs) for item in value]
  elif isinstance(value, dict):
    return {k: to_json(v, **kwargs) for k, v in value.items()}
  else:
    converter = schema_lib.get_json_value_converter(type(value))
    if not converter:
      raise ValueError(f'Cannot convert complex type {value} to JSON.')
    return to_json(converter(value))


def to_json_str(value: typing.Any,
                *,
                json_indent=None,
                **kwargs) -> typing.Text:
  """Serializes a (maybe) symbolic value into a JSON string.

  Example::

    @pg.members([
      ('x', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    a1 = A(1)
    json_str = a1.to_json_str()
    a2 = pg.from_json_str(json_str)
    assert pg.eq(a1, a2)

  Args:
    value: Value to serialize.
    json_indent: The size of indentation for JSON format.
    **kwargs: Additional keyword arguments that are passed to ``pg.to_json``.

  Returns:
    A JSON string.
  """
  return json.dumps(to_json(value, **kwargs), indent=json_indent)


class WritePermissionError(Exception):
  """Exception raisen when write access to object fields is not allowed."""
  pass


class FieldUpdate(object_utils.Formattable):
  """Class that describes an update to a field in an object tree."""

  def __init__(self,
               path: object_utils.KeyPath,
               target: 'Symbolic',
               field: typing.Optional[schema_lib.Field],
               old_value: typing.Any,
               new_value: typing.Any):
    """Constructor.

    Args:
      path: KeyPath of the field that is updated.
      target: Parent of updated field.
      field: Specification of the updated field.
      old_value: Old value of the field.
      new_value: New value of the field.
    """
    self.path = path
    self.target = target
    self.field = field
    self.old_value = old_value
    self.new_value = new_value

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Formats this object."""
    details = object_utils.kvlist_str([
        ('parent_path', self.target.sym_path, None),
        ('path', self.path.path, None),
        ('old_value', object_utils.format(
            self.old_value, compact, verbose, root_indent + 1, **kwargs),
         object_utils.MISSING_VALUE),
        ('new_value',
         object_utils.format(
             self.new_value, compact, verbose, root_indent + 1, **kwargs),
         object_utils.MISSING_VALUE),
    ])
    return f'{self.__class__.__name__}({details})'

  def __eq__(self, other: typing.Any) -> bool:
    """Operator ==."""
    if not isinstance(other, self.__class__):
      return False
    return (self.path == other.path and self.target is other.target and
            self.field is other.field and self.old_value == other.old_value and
            self.new_value == other.new_value)

  def __ne__(self, other: typing.Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)


class Origin(object_utils.Formattable):
  """Class that represents the origin of a symbolic value.

  Origin is used for debugging the creation chain of a symbolic value, as
  well as keeping track of the factory or builder in creational design patterns.
  An `Origin` object records the source value, a string tag, and optional
  stack information on where a symbolic value is created.

  Built-in tags are '__init__', 'clone', 'deepclone' and 'return'.
  Users can pass custom tags to the `sym_setorigin` method of a symbolic value
  for tracking its source in their own scenarios.

  When origin tracking is enabled by calling `pg.track_origin(True)`, the
  `sym_setorigin` method of symbolic values will be automatically called during
  object creation, cloning or being returned from a functor. The stack
  information can be obtained by `origin.stack` or `origin.stacktrace`.
  """

  def __init__(self,
               source: typing.Any,
               tag: typing.Text,
               stacktrace: typing.Optional[bool] = None,
               stacklimit: typing.Optional[int] = None,
               stacktop: int = -1):
    """Constructor.

    Args:
      source: Source value for the origin.
      tag: A descriptive tag of the origin. Built-in tags are:
        '__init__', 'clone', 'deepclone', 'return'. Users can manually
        call `sym_setorigin` with custom tag value.
      stacktrace: If True, enable stack trace for the origin. If None, enable
        stack trace if `pg.tracek_origin()` is called. Otherwise stack trace is
        disabled.
      stacklimit: An optional integer to limit the stack depth. If None, it's
        determined by the value passed to `pg.set_stacktrace_limit`,
        which is 10 by default.
      stacktop: A negative integer to indicate the stack top among the stack
        frames that we want to present to user, by default it's 2-level up from
        the stack within current `sym_setorigin` call.
    """
    if not isinstance(tag, str):
      raise ValueError(f'`tag` must be a string. Encountered: {tag!r}.')

    self._source = source
    self._tag = tag
    self._stack = None
    self._stacktrace = None

    if stacktrace is None:
      stacktrace = _is_tracking_origin()

    if stacklimit is None:
      stacklimit = _STACKTRACE_LIMIT
    if stacktrace:
      self._stack = traceback.extract_stack(limit=stacklimit - stacktop)
      if stacktop < 0:
        self._stack = self._stack[:stacktop]

  @property
  def source(self) -> typing.Any:
    """Returns the source object."""
    return self._source

  @property
  def tag(self) -> typing.Text:
    """Returns tag."""
    return self._tag

  @property
  def stack(self) -> typing.Optional[typing.List[traceback.FrameSummary]]:
    """Returns the frame summary of original stack."""
    return self._stack

  @property
  def stacktrace(self) -> typing.Optional[typing.Text]:
    """Returns stack trace string."""
    if self._stack is None:
      return None
    if self._stacktrace is None:
      self._stacktrace = ''.join(traceback.format_list(self._stack))
    return self._stacktrace

  def chain(
      self, tag: typing.Optional[typing.Text] = None
      ) -> typing.List['Origin']:
    """Get the origin list from the neareast to the farthest filtered by tag."""
    origins = []
    o = self
    while o is not None:
      if tag is None or tag == o.tag:
        origins.append(o)
      if isinstance(o.source, Symbolic):
        o = o.source.sym_origin
      else:
        o = None
    return origins

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Formats this object."""
    if isinstance(self._source, (str, type(None))):
      source_str = object_utils.quote_if_str(self._source)
    else:
      source_info = object_utils.format(
          self._source, compact, verbose, root_indent + 1, **kwargs)
      source_str = f'{source_info} at 0x{id(self._source):8x}'
    details = object_utils.kvlist_str([
        ('tag', object_utils.quote_if_str(self._tag), None),
        ('source', source_str, None),
    ])
    return f'{self.__class__.__name__}({details})'

  def __eq__(self, other: typing.Any) -> bool:
    """Operator ==."""
    if not isinstance(other, self.__class__):
      return False
    return self._source is other.source and self._tag == other.tag

  def __ne__(self, other: typing.Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)


class PureSymbolic(schema_lib.CustomTyping):
  """Base class to classes whose objects are considered pure symbolic.

  Pure symbolic objects can be used for representing abstract concepts - for
  example, a search space of objects - which cannot be executed but soely
  representational.

  Having pure symbolic object is a key differentiator of symbolic OOP from
  regular OOP, which can be used to placehold values in an object as a
  high-level expression of ideas. Later, with symbolic manipulation, the
  pure symbolic objects are replaced with material values so the object
  can be evaluated. This effectively decouples the expression of ideas from
  the implementation of ideas. For example: ``pg.oneof(['a', 'b', 'c']`` will
  be manipulated into 'a', 'b' or 'c' based on the decision of a search
  algorithm, letting the program evolve itself.
  """

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: schema_lib.ValueSpec,
      allow_partial: bool,
      child_transform: typing.Optional[typing.Callable[
          [object_utils.KeyPath, schema_lib.Field, typing.Any],
          typing.Any]] = None
  ) -> typing.Tuple[bool, typing.Any]:
    """Custom apply on a value based on its original value spec.

    This implements ``pg.typing.CustomTyping``, allowing a pure symbolic
    value to be assigned to any field. To customize this behavior, override
    this method in subclasses.

    Args:
      path: KeyPath of current object under its object tree.
      value_spec: Original value spec for this field.
      allow_partial: Whether allow partial object to be created.
      child_transform: Function to transform child node values into their final
        values. Transform function is called on leaf nodes first, then on their
        parents, recursively.

    Returns:
      A tuple (proceed_with_standard_apply, value_to_proceed).
        If proceed_with_standard_apply is set to False, value_to_proceed
        will be used as final value.

    Raises:
      Error when the value is not compatible with the value spec.
    """
    del path, value_spec, allow_partial, child_transform
    return (False, self)


class NonDeterministic(PureSymbolic):
  """Base class to mark a class whose objects are considered non-deterministic.

  A non-deterministic value represents a value that will be decided later.
  In PyGlove system, `pg.one_of`, `pg.sublist_of`, `pg.float_value` are
  non-deterministic values. Please search `NonDeterministic` subclasses for more
  details.
  """


class Symbolic(object_utils.JSONConvertible, object_utils.MaybePartial,
               object_utils.Formattable):
  """Base for all symbolic types.

  Symbolic types are types that provide interfaces for symbolic programming,
  based on which symbolic objects can be created. In PyGlove, there are three
  categories of symbolic types:

    * Symbolic classes: Defined by :class:`pyglove.Object` subclasses,
      including symbolic classes created from :func:`pyglove.symbolize`, which
      inherit :class:`pyglove.ClassWrapper`, a subclass of ``pg.Object``.
    * Symbolic functions: Defined by :class:`pyglove.Functor`.
    * Symbolic container types: Defined by :class:`pyglove.List` and
      :class:`pyglove.Dict`.
  """

  def __init__(self,
               *,
               allow_partial: bool,
               accessor_writable: bool,
               sealed: bool,
               root_path: typing.Optional[object_utils.KeyPath],
               init_super: bool = True):
    """Constructor.

    Args:
      allow_partial: Whether to allow required fields to be MISSING_VALUE or
        partial.
      accessor_writable: Whether to allow write access via attributes. This flag
        is useful when we want to enforce update of fields using `rebind`
        method, which leads to better trackability and batched field update
        notification.
      sealed: Whether object is sealed that cannot be changed. This flag is
        useful when we don't want downstream to modify the object.
      root_path: KeyPath of current object in its context (object tree).
      init_super: If True, call super.__init__, otherwise short-circuit. This
        flag is useful when user want to explicitly implement `__init__` for
        multi-inheritance, which is needed to pass different arguments to
        different bases. Please see `symbolic_test.py#testMultiInheritance`
        for more details.
    """
    # NOTE(daiyip): we uses `self._set_raw_attr` here to avoid overridden
    # `__setattr__` from subclasses change the behavior unintentionally.
    self._set_raw_attr('_allow_partial', allow_partial)
    self._set_raw_attr('_accessor_writable', accessor_writable)
    self._set_raw_attr('_sealed', sealed)

    # NOTE(daiyip): parent is used for rebind call to notify their ancestors
    # for updates, not for external usage.
    self._set_raw_attr('_sym_parent', None)
    self._set_raw_attr('_sym_path', root_path or object_utils.KeyPath())
    self._set_raw_attr('_sym_puresymbolic', None)
    self._set_raw_attr('_sym_missing_values', None)
    self._set_raw_attr('_sym_nondefault_values', None)

    origin = Origin(None, '__init__') if _is_tracking_origin() else None
    self._set_raw_attr('_sym_origin', origin)

    # super.__init__ may enter into next base class's __init__ when
    # multi-inheritance is used. Since we have override `__setattr__` for
    # symbolic.Object, which depends on `_accessor_writable` and so on,
    # we want to call make `__setattr__` ready to call before entering
    # other base's `__init__`.
    if init_super:
      super().__init__()
    else:
      object.__init__(self)

  @classmethod
  def partial(cls, *args, **kwargs) -> 'Symbolic':
    """Class method that creates a partial object of current class.

    Args:
      *args: Positional arguments that are subclass specific.
      **kwargs: Keyword arguments that are subclass specific.

    Returns:
      A partial symbolic value.
    """
    raise NotImplementedError()

  #
  # Formal contract for symbolic operations.
  #
  # NOTE(daiyip): Since methods such as `__getattr__`, `keys` can be overriden
  # by subclasses of `pg.Object`, we introduces a set of methods in signature
  # `sym_<xxx>` as the contract to symbolically operate on a symbolic
  # value, which are less likely to clash with other names. These methods will
  # be used insided PyGlove library. Users can either use these methods or their
  # convenient version at their preferences.
  #

  @property
  def sym_partial(self) -> bool:
    """Returns True if current value is partial."""
    # NOTE(daiyip): allow_partial can rule out most deep comparisons.
    return self.allow_partial and bool(self.sym_missing(flatten=False))

  @property
  def sym_puresymbolic(self) -> bool:
    """Returns True if current value is or contains subnodes of PureSymbolic."""
    pure_symbolic = getattr(self, '_sym_puresymbolic')
    if pure_symbolic is None:
      pure_symbolic = isinstance(self, PureSymbolic)
      if not pure_symbolic:
        for v in self.sym_values():
          if is_pure_symbolic(v):
            pure_symbolic = True
            break
      self._set_raw_attr('_sym_puresymbolic', pure_symbolic)
    return pure_symbolic

  @property
  def sym_abstract(self) -> bool:
    """Returns True if current value is abstract (partial or pure symbolic)."""
    return self.sym_partial or self.sym_puresymbolic

  @property
  def sym_sealed(self) -> bool:
    """Returns True if current object is sealed."""
    return getattr(self, '_sealed')

  def sym_seal(self, is_seal: bool = True) -> 'Symbolic':
    """Seals or unseals current object from further modification."""
    return self._set_raw_attr('_sealed', is_seal)

  def sym_missing(
      self, flatten: bool = True) -> typing.Dict[typing.Text, typing.Any]:
    """Returns missing values."""
    missing = getattr(self, '_sym_missing_values')
    if missing is None:
      missing = self._sym_missing()
      self._set_raw_attr('_sym_missing_values', missing)
    if flatten:
      missing = object_utils.flatten(missing)
    return missing

  def sym_nondefault(
      self,
      flatten: bool = True
      ) -> typing.Dict[typing.Union[int, typing.Text], typing.Any]:
    """Returns missing values."""
    nondefault = getattr(self, '_sym_nondefault_values')
    if nondefault is None:
      nondefault = self._sym_nondefault()
      self._set_raw_attr('_sym_nondefault_values', nondefault)
    if flatten:
      nondefault = object_utils.flatten(nondefault)
    return nondefault

  @property
  def sym_field(self) -> typing.Optional[schema_lib.Field]:
    """Returns the symbolic field for current object."""
    if self.sym_parent is None:
      return None
    return self.sym_parent.sym_attr_field(self.sym_path.key)

  @abc.abstractmethod
  def sym_attr_field(
      self, key: typing.Union[typing.Text, int]
      ) -> typing.Optional[schema_lib.Field]:
    """Returns the field definition for a symbolic attribute."""

  def sym_has(
      self,
      path: typing.Union[object_utils.KeyPath, typing.Text, int]) -> bool:
    """Returns True if a path exists in the sub-tree.

    Args:
      path: A KeyPath object or equivalence.

    Returns:
      True if the path exists in current sub-tree, otherwise False.
    """
    return object_utils.KeyPath.from_value(path).exists(self)

  def sym_get(
      self,
      path: typing.Union[object_utils.KeyPath, typing.Text, int],
      default: typing.Any = object_utils.MISSING_VALUE) -> typing.Any:
    """Returns a sub-node by path.

    NOTE: there is no `sym_set`, use `sym_rebind`.

    Args:
      path: A KeyPath object or equivalence.
      default: Default value if path does not exists. If absent, `KeyError`
        will be thrown.

    Returns:
      Value of symbolic attribute specified by path if found, otherwise the
      default value if it's specified.

    Raises:
      KeyError if `path` does not exist and default value is `pg.MISSING_VALUE`.
    """
    path = object_utils.KeyPath.from_value(path)
    if default == object_utils.MISSING_VALUE:
      return path.query(self)
    else:
      return path.get(self, default)

  @abc.abstractmethod
  def sym_hasattr(self, key: typing.Union[typing.Text, int]) -> bool:
    """Returns if a symbolic attribute exists."""

  def sym_getattr(
      self,
      key: typing.Union[typing.Text, int],
      default: typing.Any = object_utils.MISSING_VALUE) -> typing.Any:
    """Gets a symbolic attribute.

    Args:
      key: Key of symbolic attribute.
      default: Default value if attribute does not exist. If absent,
        `AttributeError` will be thrown.

    Returns:
      Value of symbolic attribute if found, otherwise the default value
      if it's specified.

    Raises:
      AttributeError if `key` does not exist.
    """
    if not self.sym_hasattr(key):
      if default != object_utils.MISSING_VALUE:
        return default
      raise AttributeError(
          self._error_message(
              f'{self.__class__!r} object has no symbolic attribute {key!r}.'))
    return self._sym_getattr(key)

  @abc.abstractmethod
  def sym_keys(self) -> typing.Iterator[typing.Union[typing.Text, int]]:
    """Iterates the keys of symbolic attributes."""

  @abc.abstractmethod
  def sym_values(self) -> typing.Iterator[typing.Any]:
    """Iterates the values of symbolic attributes."""

  @abc.abstractmethod
  def sym_items(self) -> typing.Iterator[
      typing.Tuple[typing.Union[typing.Text, int], typing.Any]]:
    """Iterates the (key, value) pairs of symbolic attributes."""

  @property
  def sym_parent(self) -> 'Symbolic':
    """Returns the containing symbolic object."""
    return getattr(self, '_sym_parent')

  def sym_setparent(self, parent: 'Symbolic'):
    """Sets the parent of current node in the symbolic tree."""
    self._set_raw_attr('_sym_parent', parent)

  def sym_contains(
      self,
      value: typing.Any = None,
      type: typing.Union[    # pylint: disable=redefined-builtin
          None, typing.Type[typing.Any], typing.Tuple[typing.Type[typing.Any]]
      ]=None) -> bool:
    """Returns True if the object contains sub-nodes of given value or type."""
    return contains(self, value, type)

  @property
  def sym_path(self) -> object_utils.KeyPath:
    """Returns the path of current object from the root of its symbolic tree."""
    return getattr(self, '_sym_path')

  def sym_setpath(
      self,
      path: typing.Optional[typing.Union[typing.Text, object_utils.KeyPath]]
      ) -> None:
    """Sets the path of current node in its symbolic tree."""
    if self.sym_path != path:
      old_path = self.sym_path
      self._set_raw_attr('_sym_path', path)
      self._update_children_paths(old_path, path)

  def sym_rebind(
      self,
      path_value_pairs: typing.Optional[typing.Union[
          typing.Dict[
              typing.Union[object_utils.KeyPath, typing.Text, int],
              typing.Any],
          typing.Callable]] = None,  # pylint: disable=g-bare-generic
      raise_on_no_change: bool = True,
      skip_notification: typing.Optional[bool] = None,
      **kwargs) -> 'Symbolic':
    """Mutates the sub-nodes of current object. Please see `rebind`."""
    if path_value_pairs and kwargs:
      raise ValueError(
          self._error_message(
              'Either argument \'path_value_pairs\' or \'**kwargs\' '
              'shall be specified. Encountered both.'))

    if callable(path_value_pairs):
      path_value_pairs = get_rebind_dict(path_value_pairs, self)
    elif path_value_pairs is None:
      path_value_pairs = kwargs

    if not isinstance(path_value_pairs, dict):
      raise ValueError(
          self._error_message(
              f'Argument \'path_value_pairs\' should be a dict. '
              f'Encountered {path_value_pairs}'))

    path_value_pairs = {object_utils.KeyPath.from_value(k): v
                        for k, v in path_value_pairs.items()}
    updates = self._sym_rebind(path_value_pairs)
    if not updates and raise_on_no_change:
      raise ValueError(self._error_message('There are no values to rebind.'))
    if skip_notification is None:
      skip_notification = not _enabled_notification()
    if not skip_notification:
      self._notify_field_updates(updates)
    return self

  def _sym_rebind(
      self,
      path_value_pairs: typing.Dict[object_utils.KeyPath, typing.Any]
      ) -> typing.List[FieldUpdate]:
    """Subclass specific rebind implementation.

    Args:
      path_value_pairs: A dictionary of key path to new field value.

    Returns:
      A list of FieldUpdate from this rebind.

    Raises:
      WritePermissionError: If object is sealed.
      KeyError: If update location specified by key or key path is not aligned
        with the schema of the object tree.
      TypeError: If updated field value type does not conform to field spec.
      ValueError: If updated field value is not acceptable according to field
        spec.
    """
    return [self._set_item_of_current_tree(k, v)
            for k, v in path_value_pairs.items()]

  def sym_clone(self,
                deep: bool = False,
                memo: typing.Optional[typing.Any] = None,
                override: typing.Optional[typing.Dict[typing.Text,
                                                      typing.Any]] = None):
    """Clones current object symbolically."""
    assert deep or not memo
    new_value = self._sym_clone(deep, memo)
    if override:
      new_value.sym_rebind(override, raise_on_no_change=False)
    if _is_tracking_origin():
      new_value.sym_setorigin(self, 'deepclone' if deep else 'clone')
    return new_value

  @abc.abstractmethod
  def sym_jsonify(self,
                  *,
                  hide_default_values: bool = False,
                  **kwargs) -> object_utils.JSONValueType:
    """Converts representation of current object to a plain Python object."""

  def sym_ne(self, other: typing.Any) -> bool:
    """Returns if this object does not equal to another object symbolically."""
    return not self.sym_eq(other)

  @abc.abstractmethod
  def sym_eq(self, other: typing.Any) -> bool:
    """Returns if this object equals to another object symbolically."""

  @abc.abstractmethod
  def sym_hash(self) -> int:
    """Computes the symbolic hash of current object."""

  @property
  def sym_origin(self) -> typing.Optional[Origin]:
    """Returns the symbolic origin of current object."""
    return getattr(self, '_sym_origin')

  def sym_setorigin(
      self,
      source: typing.Any,
      tag: typing.Text,
      stacktrace: typing.Optional[bool] = None,
      stacklimit: typing.Optional[int] = None,
      stacktop: int = -1):
    """Sets the symbolic origin of current object.

    Args:
      source: Source value for current object.
      tag: A descriptive tag of the origin. Built-in tags are:
        `__init__`, `clone`, `deepclone`, `return`. Users can manually
        call `sym_setorigin` with custom tag value.
      stacktrace: If True, enable stack trace for the origin. If None, enable
        stack trace if `pg.tracek_origin()` is called. Otherwise stack trace is
        disabled.
      stacklimit: An optional integer to limit the stack depth. If None, it's
        determined by the value passed to `pg.set_stacktrace_limit`,
        which is 10 by default.
      stacktop: A negative or zero-value integer indicating the stack top among
        the stack frames that we want to present to user, by default it's
        1-level up from the stack within current `sym_setorigin` call.

    Example::

      def foo():
        return bar()

      def bar():
        s = MyObject()
        t = s.build()
        t.sym_setorigin(s, 'builder',
            stacktrace=True, stacklimit=5, stacktop=-1)

    This example sets the origin of `t` using `s` as its source with tag
    'builder'. We also record the callstack where the `sym_setorigin` is
    called, so users can call `t.sym_origin.stacktrace` to get the call stack
    later. The `stacktop` -1 indicates that we do not need the stack frame
    within ``sym_setorigin``, so users will see the stack top within the
    function `bar`. We also set the max number of stack frames to display to 5,
    not including the stack frame inside ``sym_setorigin``.
    """
    if self.sym_origin is not None:
      current_source = typing.cast(Origin, self.sym_origin).source
      if current_source is not None and current_source is not source:
        raise ValueError(
            f'Cannot set the origin with a different source value. '
            f'Origin source: {current_source!r}, New source: {source!r}.')
    # NOTE(daiyip): We decrement the stacktop by 1 as the physical stack top
    # is within Origin.
    self._set_raw_attr(
        '_sym_origin',
        Origin(source, tag, stacktrace, stacklimit, stacktop - 1))

  #
  # Methods for operating the control flags of symbolic behaviors.
  #

  @property
  def allow_partial(self) -> bool:
    """Returns True if partial binding is allowed."""
    return getattr(self, '_allow_partial')

  @property
  def accessor_writable(self) -> bool:
    """Returns True if mutation can be made by attribute assignment."""
    return getattr(self, '_accessor_writable')

  def set_accessor_writable(self, writable: bool = True) -> 'Symbolic':
    """Sets accessor writable."""
    return self._set_raw_attr('_accessor_writable', writable)

  #
  # Easier-to-access aliases of formal symbolic operations.
  #

  @property
  def is_partial(self) -> bool:
    """Alias for `sym_partial`."""
    return self.sym_partial

  @property
  def is_pure_symbolic(self) -> bool:
    """Alias for `sym_puresymbolic`."""
    return self.sym_puresymbolic

  @property
  def is_abstract(self) -> bool:
    """Alias for `sym_abstract`."""
    return self.sym_abstract

  @property
  def is_deterministic(self) -> bool:
    """Returns if current object is deterministic."""
    return is_deterministic(self)

  def missing_values(
      self, flatten: bool = True) -> typing.Dict[typing.Text, typing.Any]:
    """Alias for `sym_missing`."""
    return self.sym_missing(flatten)

  def non_default_values(
      self,
      flatten: bool = True
      ) -> typing.Dict[typing.Union[int, typing.Text], typing.Any]:
    """Alias for `sym_nondefault`."""
    return self.sym_nondefault(flatten)

  def seal(self, sealed: bool = True) -> 'Symbolic':
    """Alias for `sym_seal`."""
    return self.sym_seal(sealed)

  @property
  def is_sealed(self) -> bool:
    """Alias for `sym_sealed`."""
    return self.sym_sealed

  def rebind(
      self,
      path_value_pairs: typing.Optional[typing.Union[
          typing.Dict[
              typing.Union[object_utils.KeyPath, typing.Text, int],
              typing.Any],
          typing.Callable]] = None,  # pylint: disable=g-bare-generic
      raise_on_no_change: bool = True,
      skip_notification: typing.Optional[bool] = None,
      **kwargs) -> 'Symbolic':
    """Alias for `sym_rebind`.

    Alias for `sym_rebind`. `rebind` is the recommended way for mutating
    symbolic objects in PyGlove:

      * It allows mutations not only on immediate child nodes, but on the
        entire sub-tree.
      * It allows mutations by rules via passing a callable object as the
        value for `path_value_pairs`.
      * It batches the updates from multiple sub-nodes, which triggers the
        `_on_change` or `_on_bound` event once for recomputing the parent
        object's internal states.
      * It respects the "sealed" flag of the object or the `pg.seal`
        context manager to trigger permission error.

    Example::

      #
      # Rebind on pg.Object subclasses.
      #

      @pg.members([
        ('x', pg.typing.Dict([
          ('y', pg.typing.Int(default=0))
         ])),
        ('z', pg.typing.Int(default=1))
      ])
      class A(pg.Object):
        pass

      a = A()
      # Rebind using path-value pairs.
      a.rebind({
        'x.y': 1,
        'z': 0
      })

      # Rebind using **kwargs.
      a.rebind(x={y: 1}, z=0)

      # Rebind using rebinders.
      # Rebind based on path.
      a.rebind(lambda k, v: 1 if k == 'x.y' else v)
      # Rebind based on key.
      a.rebind(lambda k, v: 1 if k and k.key == 'y' else v)
      # Rebind based on value.
      a.rebind(lambda k, v: 0 if v == 1 else v)
      # Rebind baesd on value and parent.
      a.rebind(lambda k, v, p: (0 if isinstance(p, A) and isinstance(v, int)
                                else v))

      # Rebind on pg.Dict.
      #
      d = pg.Dict(value_spec=schema.Dict([
        ('a', pg.typing.Dict([
          ('b', pg.typing.Int()),
        ])),
        ('c', pg.typing.Float())
      ])

      # Rebind using **kwargs.
      d.rebind(a={b: 1}, c=1.0)

      # Rebind using key path to value dict.
      d.rebind({
        'a.b': 2,
        'c': 2.0
      })

      # NOT OKAY: **kwargs and dict/rebinder cannot be used at the same time.
      d.rebind({'a.b': 2}, c=2)

      # Rebind with rebinder by path (on subtree).
      d.rebind(lambda k, v: 1 if k.key == 'b' else v)

      # Rebind with rebinder by value (on subtree).
      d.rebind(lambda k, v: 0 if isinstance(v, int) else v)

      #
      # Rebind on pg.List.
      #
      l = pg.List([{
            'a': 'foo',
            'b': 0,
          }
        ],
        value_spec = pg.typing.List(schema.Dict([
            ('a', pg.typing.Str()),
            ('b', pg.typing.Int())
        ]), max_size=10))

      # Rebind using integer as list index: update semantics on list[0].
      l.rebind({
        0: {
          'a': 'bar',
          'b': 1
        }
      })

      # Rebind: trigger append semantics when index is larger than list length.
      l.rebind({
        999: {
          'a': 'fun',
          'b': 2
        }
      })

      # Rebind using key path.
      l.rebind({
        '[0].a': 'bar2'
        '[1].b': 3
      })

      # Rebind using function (rebinder).
      # Change all integers to 0 in sub-tree.
      l.rebind(lambda k, v: v if not isinstance(v, int) else 0)

    Args:
      path_value_pairs: A dictionary of key/or key path to new field value, or
        a function that generate updates based on the key path, value and
        parent of each node under current object. We use terminology 'rebinder'
        for this type of functions. The signature of a rebinder is:

            `(key_path: pg.KeyPath, value: Any)` or
            `(key_path: pg.KeyPath, value: Any, parent: pg.Symbolic)`

      raise_on_no_change: If True, raises ``ValueError`` when there are no
        values to change. This is useful when rebinder is used, which may or
        may not generate any updates.
      skip_notification: If True, there will be no ``_on_change`` event
        triggered from current `rebind`. If None, the default value will be
        inferred from the :func:`pyglove.notify_on_change` context manager.
        Use it only when you are certain that current rebind does not
        invalidate internal states of its object tree.
      **kwargs: For ``pg.Dict`` and ``pg.Object`` subclasses, user can use
        keyword arguments (in format of `<field_name>=<field_value>`) to
        directly modify immediate child nodes.

    Returns:
      Self.

    Raises:
      WritePermissionError: If object is sealed.
      KeyError: If update location specified by key or key path is not aligned
        with the schema of the object tree.
      TypeError: If updated field value type does not conform to field spec.
      ValueError: If updated field value is not acceptable according to field
        spec, or nothing is updated and `raise_on_no_change` is set to
        True.
    """
    return self.sym_rebind(
        path_value_pairs, raise_on_no_change, skip_notification, **kwargs)

  def _set_parent(self, parent: 'Symbolic'):
    """Alias for `sym_setparent` to backward compatibility."""
    self.sym_setparent(parent)

  def clone(
      self,
      deep: bool = False,
      memo: typing.Optional[typing.Any] = None,
      override: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None
  ) -> 'Symbolic':
    """Clones current object symbolically.

    Args:
      deep: If True, perform deep copy (equivalent to copy.deepcopy). Otherwise
        shallow copy (equivalent to copy.copy).
      memo: Memo object for deep clone.
      override: An optional dict of key path to new values to override cloned
        value.

    Returns:
      A copy of self.
    """
    return self.sym_clone(deep, memo, override)

  def to_json(self, **kwargs) -> object_utils.JSONValueType:
    """Alias for `sym_jsonify`."""
    return self.sym_jsonify(**kwargs)

  def to_json_str(
      self, json_indent: typing.Optional[int] = None, **kwargs) -> typing.Text:
    """Serializes current object into a JSON string."""
    return json.dumps(self.sym_jsonify(**kwargs), indent=json_indent)

  @classmethod
  def load(cls, *args, **kwargs) -> typing.Any:
    """Loads an instance of this type using the global load handler."""
    value = load(*args, **kwargs)
    if not isinstance(value, cls):
      raise TypeError(f'Value is not of type {cls!r}: {value!r}.')
    return value

  def save(self, *args, **kwargs) -> typing.Any:
    """Saves current object using the global save handler."""
    return save(self, *args, **kwargs)

  def inspect(
      self,
      path_regex: typing.Optional[typing.Text] = None,
      where: typing.Optional[typing.Union[typing.Callable[
          [typing.Any], bool], typing.Callable[[typing.Any, typing.Any],
                                               bool]]] = None,
      custom_selector: typing.Optional[typing.Union[
          typing.Callable[[object_utils.KeyPath, typing.Any], bool],
          typing.Callable[[object_utils.KeyPath, typing.Any, typing.Any],
                          bool]]] = None,
      file=sys.stdout,  # pylint: disable=redefined-builtin
      **kwargs) -> None:
    """Inspects current object by printing out selected values.

    Example::

      @pg.members([
          ('x', pg.typing.Int(0)),
          ('y', pg.typing.Str())
      ])
      class A(pg.Object):
        pass

      value = {
        'a1': A(x=0, y=0),
        'a2': [A(x=1, y=1), A(x=1, y=2)],
        'a3': {
          'p': A(x=2, y=1),
          'q': A(x=2, y=2)
        }
      }

      # Inspect without constraint,
      # which is equivalent as `print(value.format(hide_default_values=True))`
      # Shall print:
      # {
      #   a1 = A(y=0)
      #   a2 = [
      #     0: A(x=1, y=1)
      #     1: A(x=1, y=2)
      #   a3 = {
      #     p = A(x=2, y=1)
      #     q = A(x=2, y=2)
      #   }
      # }
      value.inspect(hide_default_values=True)

      # Inspect by path regex.
      # Shall print:
      # {'a3.p': A(x=2, y=1)}
      value.inspect(r'.*p')

      # Inspect by value.
      # Shall print:
      # {
      #    'a3.p.x': 2,
      #    'a3.q.x': 2,
      #    'a3.q.y': 2,
      # }
      value.inspect(where=lambda v: v==2)

      # Inspect by path, value and parent.
      # Shall print:
      # {
      #    'a2[1].y': 2
      # }
      value.inspect(
        r'.*y', where=lambda v, p: v > 1 and isinstance(p, A) and p.x == 1))

      # Inspect by custom_selector.
      # Shall print:
      # {
      #   'a2[0].x': 1,
      #   'a2[0].y': 1,
      #   'a3.q.x': 2,
      #   'a3.q.y': 2
      # }
      value.inspect(
        custom_selector=lambda k, v, p: (
          len(k) == 3 and isinstance(p, A) and p.x == v))

    Args:
      path_regex: Optional regex expression to constrain path.
      where: Optional callable to constrain value and parent when path matches
        `path_regex` or `path_regex` is not provided. The signature is:
        `(value) -> should_select`, or `(value, parent) -> should_select`.
      custom_selector: Optional callable object as custom selector. When
        `custom_selector` is provided, `path_regex` and `where` must be None.
        The signature of `custom_selector` is:
        `(key_path, value) -> should_select`
        or `(key_path, value, parent) -> should_select`.
      file: Output file stream. This can be any object with a `write(str)`
        method.
      **kwargs: Wildcard keyword arguments to pass to `format`.
    """
    if path_regex is None and where is None and custom_selector is None:
      v = self
    else:
      v = query(self, path_regex, where, False, custom_selector)
    object_utils.printv(v, file=file, **kwargs)

  def __str__(self) -> typing.Text:
    """Override Formattable.__str__ by setting verbose to False."""
    return self.format(compact=False, verbose=False)

  def __repr__(self) -> typing.Text:
    return self.format(compact=True)

  def __copy__(self) -> 'Symbolic':
    """Overridden shallow copy."""
    return self.sym_clone(deep=False)

  def __deepcopy__(self, memo) -> 'Symbolic':
    """Overridden deep copy."""
    return self.sym_clone(deep=True, memo=memo)

  #
  # Proteted methods to implement from subclasses
  #
  @abc.abstractmethod
  def _sym_missing(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns missing values."""

  @abc.abstractmethod
  def _sym_nondefault(self) -> typing.Dict[typing.Union[int, typing.Text],
                                           typing.Any]:
    """Returns non-default values."""

  @abc.abstractmethod
  def _sym_getattr(self, key: typing.Union[typing.Text, int]) -> typing.Any:
    """Get symbolic attribute by key."""

  @abc.abstractmethod
  def _sym_clone(self, deep: bool, memo=None) -> 'Symbolic':
    """Subclass specific clone implementation."""

  @abc.abstractmethod
  def _update_children_paths(
      self,
      old_path: object_utils.KeyPath,
      new_path: object_utils.KeyPath) -> None:
    """Update children paths according to root_path of current node."""

  @abc.abstractmethod
  def _set_item_without_permission_check(self, key: typing.Union[typing.Text,
                                                                 int],
                                         value: typing.Any) -> FieldUpdate:
    """Child should implement: set an item without permission check."""

  @abc.abstractmethod
  def _on_change(self, field_updates: typing.Dict[object_utils.KeyPath,
                                                  FieldUpdate]):
    """Event that is triggered when field values in the subtree are updated.

    This event will be called
      * On per-field basis when object is modified via attribute.
      * In batch when multiple fields are modified via `rebind` method.

    When a field in an object tree is updated, all ancestors' `_on_change` event
    will be triggered in order, from the nearest one to furthest one.

    Args:
      field_updates: Updates made to the subtree. Key path is relative to
        current object.
    """

  @property
  @abc.abstractmethod
  def _subscribes_field_updates(self) -> bool:
    """Returns True if current object subscribes field updates in `on_change`.

    NOTE(daiyip): When it returns False, we don't need to compute field updates
    for this object, but simply invoke onchange with empty fields.
    """

  #
  # Protected helper methods.
  #

  def _set_raw_attr(self,
                    name: typing.Text,
                    value: typing.Any) -> 'Symbolic':
    """Set raw property without trigger __setattr__."""
    # `object.__setattr__` adds a property to the instance without side effects.
    object.__setattr__(self, name, value)
    return self

  def _relocate_if_symbolic(self, key: typing.Union[typing.Text, int],
                            value: typing.Any) -> typing.Any:
    """Relocate if a symbolic value is to be inserted as member.

    NOTE(daiyip): when a symbolic value is inserted into the object tree,
    if it already has a parent, we need to make a shallow copy of this object
    to avoid multiple parents. Otherwise we need to set its parent and root_path
    according to current object.

    Args:
      key: Key used to insert the value.
      value: formal value to be inserted.

    Returns:
      Formalized value that is ready for insertion as members.
    """
    if isinstance(value, Symbolic):
      # NOTE(daiyip): make a copy of symbolic object if it belongs to another
      # object tree, this prevents it from having multiple parents. See
      # List._formalized_value for similar logic.
      root_path = object_utils.KeyPath(key, self.sym_path)
      if (value.sym_parent is not None and
          (value.sym_parent is not self
           or root_path != value.sym_path)):
        value = value.clone()
      value.sym_setpath(root_path)
      # NOTE(daiyip): Dict may set '_pass_through_parent' member to True when
      # it's used as the field container of an Object.
      if getattr(self, '_pass_through_parent', False):
        parent = self.sym_parent
      else:
        parent = self
      value.sym_setparent(parent)
      value.set_accessor_writable(self.accessor_writable)
    return value

  def _set_item_of_current_tree(
      self, path: object_utils.KeyPath, value: typing.Any) -> FieldUpdate:
    """Set a field of current tree by key path and return its parent."""
    assert isinstance(path, object_utils.KeyPath), path
    if not path:
      raise KeyError(
          self._error_message(
              f'Root key \'$\' cannot be used in '
              f'{self.__class__.__name__}.rebind. '
              f'Encountered {path!r}'))

    parent_node = path.parent.query(self)
    if not isinstance(parent_node, Symbolic):
      raise KeyError(
          f'Cannot rebind key {path.key!r}: {parent_node!r} is not a '
          f'symbolic type. (path=\'{path.parent}\')')

    if _is_sealed(parent_node):
      raise WritePermissionError(
          f'Cannot rebind key {path.key!r} of '
          f'sealed {parent_node.__class__.__name__}: {parent_node!r}. '
          f'(path=\'{path.parent}\')')
    return parent_node._set_item_without_permission_check(path.key, value)  # pylint: disable=protected-access

  def _notify_field_updates(self,
                            field_updates: typing.List[FieldUpdate]) -> None:
    """Notify field updates."""
    per_target_updates = dict()

    def _get_target_updates(
        target: 'Symbolic'
    ) -> typing.Dict[object_utils.KeyPath, FieldUpdate]:
      target_id = id(target)
      if target_id not in per_target_updates:
        per_target_updates[target_id] = (target, dict())
      return per_target_updates[target_id][1]

    for update in field_updates:
      target = update.target
      while target is not None:
        target_updates = _get_target_updates(target)
        if target._subscribes_field_updates:  # pylint: disable=protected-access
          relative_path = update.path - target.sym_path
          target_updates[relative_path] = update
        target = target.sym_parent

    # Trigger the notification bottom-up, thus the parent node will always
    # be notified after the child nodes.
    for target, updates in sorted(per_target_updates.values(),
                                  key=lambda x: x[0].sym_path,
                                  reverse=True):
      # Reset content-based cache for the object being notified.
      target._set_raw_attr('_sym_puresymbolic', None)       # pylint: disable=protected-access
      target._set_raw_attr('_sym_missing_values', None)     # pylint: disable=protected-access
      target._set_raw_attr('_sym_nondefault_values', None)  # pylint: disable=protected-access
      target._on_change(updates)   # pylint: disable=protected-access

  def _error_message(self, message: typing.Text) -> typing.Text:
    """Create error message to include path information."""
    return object_utils.message_on_path(message, self.sym_path)


class Dict(dict, Symbolic, schema_lib.CustomTyping):
  """Symbolic dict.

  ``pg.Dict`` implements a dict type whose instances are symbolically
  programmable, which is a subclass of the built-in Python ``dict`` and
  a subclass of :class:`pyglove.Symbolic`.

  ``pg.Dict`` provides the following features:

   * It a symbolic programmable dict with string keys.
   * It enables attribute access on dict keys.
   * It supports symbolic validation and value completitions based on schema.
   * It provides events to handle sub-nodes changes.

  ``pg.Dict`` can be used as a regular dict with string keys::

    # Construct a symbolic dict from key value pairs.
    d = pg.Dict(x=1, y=2)

  or::

    # Construct a symbolic dict from a mapping object.
    d = pg.Dict({'x': 1, 'y': 2})

  Besides regular items access using ``[]``, it allows attribute access
  to its keys::

    # Read access to key `x`.
    assert d.x == 1

    # Write access to key 'y'.
    d.y = 1

  ``pg.Dict`` supports symbolic validation when the ``value_spec`` argument
  is provided::

    d = pg.Dict(x=1, y=2, value_spec=pg.typing.Dict([
        ('x', pg.typing.Int(min_value=1)),
        ('y', pg.typing.Int(min_value=1)),
        (pg.typing.StrKey('foo.*'), pg.typing.Str())
    ])

    # Okay: all keys started with 'foo' is acceptable and are strings.
    d.foo1 = 'abc'

    # Raises: 'bar' is not acceptable as keys in the dict.
    d.bar = 'abc'

  Users can mutate the values contained in it::

    d = pg.Dict(x=pg.Dict(y=1), p=pg.List([0]))
    d.rebind({
      'x.y': 2,
      'p[0]': 1
    })

  It also allows the users to subscribe subtree updates::

    def on_change(updates):
      print(updates)

    d = pg.Dict(x=1, onchange_callaback=on_change)

    # `on_change` will be triggered on item insertion.
    d['y'] = {'z': 1}

    # `on_change` will be triggered on item removal.
    del d.x

    # `on_change` will also be triggered on subtree change.
    d.rebind({'y.z': 2})
  """

  @classmethod
  def partial(cls,
              dict_obj: typing.Optional[typing.Dict[typing.Text,
                                                    typing.Any]] = None,
              *,
              value_spec: typing.Optional[schema_lib.Dict] = None,
              onchange_callback: typing.Optional[typing.Callable[
                  [typing.Dict[object_utils.KeyPath,
                               FieldUpdate]], None]] = None,
              **kwargs) -> 'Dict':
    """Class method that creates a partial Dict object."""
    return cls(dict_obj,
               value_spec=value_spec,
               onchange_callback=onchange_callback,
               allow_partial=True,
               **kwargs)

  @classmethod
  def from_json(cls,
                json_value: typing.Any,
                *,
                allow_partial: bool = False,
                root_path: typing.Optional[object_utils.KeyPath] = None,
                **kwargs) -> 'Dict':
    """Class method that load an symbolic Dict from a JSON value.

    Args:
      json_value: Input JSON value, only JSON dict is acceptable.
      allow_partial: Whether to allow members of the dict to be partial.
      root_path: KeyPath of loaded object in its object tree.
      **kwargs: Allow passing through keyword arguments that are not applicable.

    Returns:
      A schemaless symbolic dict. For example::

        d = Dict.from_json({
          'a': {
            '_type': '__main__.Foo',
            'f1': 1,
            'f2': {
              'f21': True
            }
          }
        })

        assert d.value_spec is None
        # Okay:
        d.b = 1

        # a.f2 is bound by class Foo's field 'f2' definition (assume it defines
        # a schema for the Dict field).
        assert d.a.f2.value_spec is not None

        # Not okay:
        d.a.f2.abc = 1
    """
    return cls(json_value, allow_partial=allow_partial, root_path=root_path)

  def __init__(self,
               dict_obj: typing.Optional[typing.Dict[typing.Text,
                                                     typing.Any]] = None,
               value_spec: typing.Optional[schema_lib.Dict] = None,
               onchange_callback: typing.Optional[typing.Callable[
                   [typing.Dict[object_utils.KeyPath,
                                FieldUpdate]], None]] = None,
               **kwargs):
    """Constructor.

    Args:
      dict_obj: A dict as initial value for this Dict.
      value_spec: Value spec that applies to this Dict.
      onchange_callback: Callback when sub-tree has been modified.
      **kwargs: Key value pairs that will be inserted into the dict as initial
        value, which provides a syntax sugar for usage as below: d =
          pg.Dict(a=1, b=2)
    """
    allow_partial = kwargs.pop('allow_partial', False)
    accessor_writable = kwargs.pop('accessor_writable', True)
    sealed = kwargs.pop('sealed', False)
    root_path = kwargs.pop('root_path', None)

    # Skip schema check when dict_obj is validated against
    # schema externally. This flag is helpful to avoid duplicated schema
    # check in nested structures, which takes effect only when value_spec
    # is not None.
    pass_through = kwargs.pop('pass_through', False)

    # If True, the parent of dict items should be set to `self.sym_parent`,
    # This is useful when Dict is used as the field container of
    # pg.Object.
    self._set_raw_attr('_pass_through_parent',
                       kwargs.pop('pass_through_parent', False))

    if dict_obj is not None and not isinstance(dict_obj, dict):
      raise TypeError(
          f'Argument \'dict_obj\' must be dict type. '
          f'Encountered {type(dict_obj)}.')
    if value_spec and not isinstance(value_spec, schema_lib.Dict):
      raise TypeError(
          f'Argument \'value_spec\' must be a schema.Dict type. '
          f'Encountered {type(value_spec)}')

    # NOTE(daiyip): we call __init__ of superclasses explicitly instead of
    # calling super().__init__(...) since dict.__init__ does
    # not follow super(...).__init__ fashion, which will lead to
    # Symbolic.__init__ uncalled.
    Symbolic.__init__(
        self,
        allow_partial=allow_partial,
        accessor_writable=True,
        # We delay seal operation until members are filled.
        sealed=False,
        root_path=root_path)

    dict.__init__(self)
    self._value_spec = None
    self._onchange_callback = None

    # NOTE(daiyip): values in kwargs is prior to dict_obj.
    dict_obj = dict_obj or {}
    for k, v in kwargs.items():
      dict_obj[k] = v

    if value_spec:
      if pass_through:
        for k, v in dict_obj.items():
          super().__setitem__(k, self._relocate_if_symbolic(k, v))

        # NOTE(daiyip): when pass_through is on, we simply trust input
        # dict is validated and filled with values of their final form (
        # symbolic Dict/List vs. dict/list). This prevents members from
        # repeated validation and transformation.
        self._value_spec = value_spec
      else:
        for k, v in dict_obj.items():
          super().__setitem__(k, self._formalized_value(k, None, v))
        self.use_value_spec(value_spec, allow_partial)
    else:
      for k, v in dict_obj.items():
        self._set_item_without_permission_check(k, v)

    # NOTE(daiyip): We set onchange callback at the end of init to avoid
    # triggering during initialization.
    self._onchange_callback = onchange_callback
    self.set_accessor_writable(accessor_writable)
    self.seal(sealed)

  @property
  def value_spec(self) -> typing.Optional[schema_lib.Dict]:
    """Returns value spec of this dict.

    NOTE(daiyip): If this dict is schema-less, value_spec will be None.
    """
    return self._value_spec

  def use_value_spec(self,
                     value_spec: typing.Optional[schema_lib.Dict],
                     allow_partial: bool = False) -> 'Dict':
    """Applies a ``pg.typing.Dict`` as the value spec for current dict.

    Args:
      value_spec: A Dict ValueSpec to apply to this Dict.
        If current Dict is schema-less (whose immediate members are not
        validated against schema), and `value_spec` is not None, the value spec
        will be applied to the Dict.
        Or else if current Dict is already symbolic (whose immediate members
        are under the constraint of a Dict value spec), and `value_spec` is
        None, current Dict will become schema-less. However, the schema
        constraints for non-immediate members will remain.
      allow_partial: Whether allow partial dict based on the schema. This flag
        will override allow_partial flag in __init__ for spec-less Dict.

    Returns:
      Self.

    Raises:
      ValueError: validation failed due to value error.
      RuntimeError: Dict is already bound with another spec.
      TypeError: type errors during validation.
      KeyError: key errors during validation.
    """
    if value_spec is None:
      self._value_spec = None
      self._accessor_writable = True
      return self

    if self._value_spec and self._value_spec != value_spec:
      raise RuntimeError(
          self._error_message(
              f'Dict is already bound with a different value spec: '
              f'{self._value_spec}. New value spec: {value_spec}.'))

    self._allow_partial = allow_partial

    if _enabled_type_check():
      # NOTE(daiyip): self._value_spec will be set in Dict.custom_apply method
      # called by value_spec.apply, thus we don't need to set self._value_spec
      # explicitly.
      value_spec.apply(
          self,
          allow_partial=_allow_partial(self),
          child_transform=_symbolic_transform_fn(self._allow_partial),
          root_path=self.sym_path)
    else:
      self._value_spec = value_spec
    return self

  def _sym_missing(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns missing values.

    Returns:
      A dict of key to MISSING_VALUE.
    """
    missing = dict()
    if self._value_spec and self._value_spec.schema:
      matched_keys, _ = self._value_spec.schema.resolve(self.keys())
      for key_spec, keys in matched_keys.items():
        field = self._value_spec.schema[key_spec]
        if not keys:
          if isinstance(key_spec, schema_lib.ConstStrKey):
            missing[key_spec.text] = field.value.default
        else:
          for key in keys:
            v = self[key]
            if object_utils.MISSING_VALUE == v:
              missing[key] = field.value.default
            else:
              if isinstance(v, Symbolic):
                missing_child = v.sym_missing(flatten=False)
                if missing_child:
                  missing[key] = missing_child
    else:
      for k, v in self.items():
        if isinstance(v, Symbolic):
          missing_child = v.sym_missing(flatten=False)
          if missing_child:
            missing[k] = missing_child
    return missing

  def _sym_nondefault(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns non-default values as key/value pairs in a dict."""
    non_defaults = dict()
    if self._value_spec and self._value_spec.schema:
      dict_schema = self._value_spec.schema
      matched_keys, unmatched_keys = dict_schema.resolve(self.keys())
      assert not unmatched_keys
      for key_spec, keys in matched_keys.items():
        value_spec = dict_schema[key_spec].value
        for key in keys:
          v = self[key]
          child_has_non_defaults = False
          if isinstance(v, Symbolic):
            non_defaults_child = v.non_default_values(flatten=False)
            if non_defaults_child:
              non_defaults[key] = non_defaults_child
              child_has_non_defaults = True
          if not child_has_non_defaults and value_spec.default != v:
            non_defaults[key] = v
    else:
      for k, v in self.items():
        if isinstance(v, Symbolic):
          non_defaults_child = v.non_default_values(flatten=False)
          if non_defaults_child:
            non_defaults[k] = non_defaults_child
        else:
          non_defaults[k] = v
    return non_defaults

  def set_accessor_writable(self, writable: bool = True) -> 'Dict':
    """Sets accessor writable."""
    if self.accessor_writable == writable:
      return self
    for v in self.values():
      if isinstance(v, Symbolic):
        v.set_accessor_writable(writable)
    super().set_accessor_writable(writable)
    return self

  def seal(self, sealed: bool = True) -> 'Dict':
    """Seals or unseals current object from further modification."""
    if self.is_sealed == sealed:
      return self
    for v in self.values():
      if isinstance(v, Symbolic):
        v.seal(sealed)
    super().seal(sealed)
    return self

  def sym_attr_field(
      self, key: typing.Union[typing.Text, int]
      ) -> typing.Optional[schema_lib.Field]:
    """Returns the field definition for a symbolic attribute."""
    if self._value_spec is None or self._value_spec.schema is None:
      return None
    return self._value_spec.schema.get_field(key)

  def sym_hasattr(self, key: typing.Union[typing.Text, int]) -> bool:
    """Tests if a symbolic attribute exists."""
    return key in self

  def sym_keys(self) -> typing.Iterator[typing.Text]:
    """Iterates the keys of symbolic attributes."""
    if self._value_spec is None or self._value_spec.schema is None:
      for key in super().__iter__():
        yield key
    else:
      traversed = set()
      for key_spec in self._value_spec.schema.keys():
        if isinstance(key_spec, schema_lib.ConstStrKey) and key_spec in self:
          yield key_spec.text
          traversed.add(key_spec.text)

      if len(traversed) < len(self):
        for key in super().__iter__():
          if key not in traversed:
            yield key

  def sym_values(self) -> typing.Iterator[typing.Any]:
    """Iterates the values of symbolic attributes."""
    for k in self.sym_keys():
      yield self[k]

  def sym_items(self) -> typing.Iterator[
      typing.Tuple[typing.Text, typing.Any]]:
    """Iterates the (key, value) pairs of symbolic attributes."""
    for k in self.sym_keys():
      yield k, self[k]

  def sym_setparent(self, parent: 'Symbolic'):
    """Override set parent of Dict to handle the passing through scenario."""
    super().sym_setparent(parent)
    # NOTE(daiyip): when flag `pass_through_parent` is on, it sets the parent
    # of child symbolic values using its parent.
    if self._pass_through_parent:
      for v in self.values():
        if isinstance(v, Symbolic):
          v.sym_setparent(parent)

  def sym_eq(self, other: typing.Any) -> bool:
    """Tests symbolic equality."""
    return eq(self, other)

  def sym_hash(self) -> int:
    """Symbolic hashing."""
    return sym_hash(
        (self.__class__,
         tuple([sym_hash((k, v)) for k, v in self.sym_items()
                if v != schema_lib.MISSING_VALUE])))

  def _sym_getattr(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: typing.Text) -> typing.Any:
    """Gets symbolic attribute by key."""
    return self[key]

  def _sym_clone(self, deep: bool, memo=None) -> 'Dict':
    """Override Symbolic._sym_clone."""
    source = dict()
    for k, v in self.items():
      if deep or isinstance(v, Symbolic):
        v = clone(v, deep, memo)
      source[k] = v
    return Dict(
        source,
        value_spec=self._value_spec,
        allow_partial=self._allow_partial,
        accessor_writable=self._accessor_writable,
        sealed=self._sealed,
        # NOTE(daiyip): parent and root_path are reset to empty
        # for copy object.
        root_path=None,
        pass_through=True)

  def _update_children_paths(
      self,
      old_path: object_utils.KeyPath,
      new_path: object_utils.KeyPath) -> None:
    """Update children paths according to root_path of current node."""
    del old_path
    for k, v in self.items():
      if isinstance(v, Symbolic):
        v.sym_setpath(object_utils.KeyPath(k, new_path))

  def _set_item_without_permission_check(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: typing.Text, value: typing.Any) -> FieldUpdate:
    """Set item without permission check."""
    if not isinstance(key, str):
      raise KeyError(self._error_message(
          f'Key must be string type. Encountered {key!r}.'))

    field = None
    if self._value_spec and self._value_spec.schema:
      field = self._value_spec.schema.get_field(key)
      if not field:
        if (self.sym_parent is not None
            and self.sym_parent.sym_path == self.sym_path):
          container_cls = self.sym_parent.__class__
        else:
          container_cls = self.__class__
        raise KeyError(
            self._error_message(
                f'Key \'{key}\' is not allowed for {container_cls}.'))
    new_value = self._formalized_value(key, field, value)
    old_value = self.get(key, schema_lib.MISSING_VALUE)

    # Detach old value from object tree.
    if old_value is not new_value and isinstance(old_value, Symbolic):
      old_value.sym_setparent(None)
      old_value.sym_setpath(object_utils.KeyPath())

    if (schema_lib.MISSING_VALUE == new_value and
        (not field or isinstance(field.key, schema_lib.NonConstKey))):
      if key in self:
        super().__delitem__(key)
    else:
      super().__setitem__(key, new_value)

    # NOTE(daiyip): If current dict is the field dict of a symbolic object,
    # Use parent object as update target.
    target = self
    if (self.sym_parent
        and self.sym_parent.sym_path == self.sym_path):
      target = self.sym_parent
    return FieldUpdate(self.sym_path + key, target, field, old_value,
                       new_value)

  def _formalized_value(self, name: typing.Text,
                        field: typing.Optional[schema_lib.Field],
                        value: typing.Any) -> typing.Any:
    """Get transformed (formal) value from user input."""
    allow_partial = _allow_partial(self)
    if field and schema_lib.MISSING_VALUE == value:
      # NOTE(daiyip): default value is already in transformed form.
      value = field.default_value
    else:
      value = from_json(
          value,
          allow_partial=allow_partial,
          root_path=object_utils.KeyPath(name, self.sym_path))
    if field and _enabled_type_check():
      value = field.apply(
          value,
          allow_partial=allow_partial,
          transform_fn=_symbolic_transform_fn(self._allow_partial),
          root_path=object_utils.KeyPath(name, self.sym_path))
    return self._relocate_if_symbolic(name, value)

  @property
  def _subscribes_field_updates(self) -> bool:
    """Returns True if current dict subscribes field updates."""
    return self._onchange_callback is not None

  def _on_change(self, field_updates: typing.Dict[object_utils.KeyPath,
                                                  FieldUpdate]):
    """On change event of Dict."""
    if self._onchange_callback:
      self._onchange_callback(field_updates)

  def __setitem__(self, key: typing.Text, value: typing.Any) -> None:
    """Set item in this Dict.

    Args:
      key: String key. (Please be noted that key path is not supported.)
      value: Value to be inserted.

    Raises:
      WritePermissionError: when Dict cannot be modified by accessor or
        is sealed.
      KeyError: Key is not allowed according to the value spec.
      ValueError: Value is not acceptable according to the value spec.
    """
    if _is_sealed(self):
      raise WritePermissionError(
          self._error_message('Cannot modify field of a sealed Dict.'))

    if not _is_accessor_writable(self):
      raise WritePermissionError(
          self._error_message(
              'Cannot modify Dict field by attribute or key while '
              'accessor_writable is set to False. '
              'Use \'rebind\' method instead.'))

    update = self._set_item_without_permission_check(key, value)
    if _enabled_notification():
      self._notify_field_updates([update])

  def __setattr__(self, name: typing.Text, value: typing.Any) -> None:
    """Set attribute of this Dict.

    NOTE(daiyip): When setting attributes, public attributes (not started with
    '_') are set as dict fields, while private attributes (started with '_') are
    set on the object instance.

    Args:
      name: Name of attribute.
      value: Value of attribute.
    """
    if name.startswith('_'):
      super().__setattr__(name, value)
    else:
      self[name] = value

  def __delitem__(self, name: typing.Text) -> None:
    """Delete a key from the Dict.

    This is used to delete a key which resolves to a pg.typing.NonConstKey.

    Args:
      name: Key to delete.

    Raises:
      WritePermissionError: When Dict is sealed.
      KeyError: When key is not a NonConstKey.
    """
    if _is_sealed(self):
      raise WritePermissionError('Cannot del item from a sealed Dict.')

    if not _is_accessor_writable(self):
      raise WritePermissionError(
          self._error_message('Cannot del Dict field by attribute or key while '
                              'accessor_writable is set to False. '
                              'Use \'rebind\' method instead.'))

    if name not in self:
      raise KeyError(
          self._error_message(f'Key does not exist in Dict: {name!r}.'))

    update = self._set_item_without_permission_check(
        name, schema_lib.MISSING_VALUE)
    if _enabled_notification():
      self._notify_field_updates([update])

  def __delattr__(self, name: typing.Text) -> None:
    """Delete an attribute."""
    del self[name]

  def __getattr__(self, name: typing.Text) -> typing.Any:
    """Get attribute that is not defined as property."""
    if name in self:
      return self[name]
    raise AttributeError(
        f'Attribute \'{name}\' does not exist in {self.__class__!r}.')

  def __iter__(self):
    """Iterate keys in field declaration order."""
    return self.sym_keys()

  def keys(self) -> typing.Iterator[typing.Text]:
    """Returns an iterator of keys in current dict."""
    return self.sym_keys()

  def items(self) -> typing.Iterator[typing.Tuple[typing.Text, typing.Any]]:
    """Returns an iterator of (key, value) items in current dict."""
    return self.sym_items()

  def values(self) -> typing.Iterator[typing.Any]:
    """Returns an iterator of values in current dict.."""
    return self.sym_values()

  def copy(self) -> 'Dict':
    """Overriden copy using symbolic copy."""
    return self.sym_clone(deep=False)

  def pop(
      self, key: typing.Any, default: typing.Any = schema_lib.MISSING_VALUE
  ) -> typing.Any:
    """Pops a key from current dict."""
    if key in self:
      value = self[key]
      with allow_writable_accessors(True):
        del self[key]
      return value if value != schema_lib.MISSING_VALUE else default
    if default == schema_lib.MISSING_VALUE:
      raise KeyError(key)
    return default

  def clear(self) -> None:
    """Removes all the keys in current dict."""
    if _is_sealed(self):
      raise WritePermissionError('Cannot clear a sealed Dict.')
    value_spec = self._value_spec
    self._value_spec = None
    super().clear()

    if value_spec:
      self.use_value_spec(value_spec, self._allow_partial)

  def update(self, other: typing.Dict[typing.Text, typing.Any]) -> None:
    """Update Dict with the same semantic as update on standard dict."""
    self.rebind(
        other, raise_on_no_change=False, skip_notification=True)

  def sym_jsonify(
      self,
      hide_default_values: bool = False,
      **kwargs) -> object_utils.JSONValueType:
    """Converts current object to a dict with plain Python objects."""
    if self._value_spec and self._value_spec.schema:
      json_repr = dict()
      matched_keys, _ = self._value_spec.schema.resolve(self.keys())
      for key_spec, keys in matched_keys.items():
        # NOTE(daiyip): The key values of frozen field can safely be excluded
        # since they will be the same for a class.
        field = self._value_spec.schema[key_spec]
        if not field.frozen:
          for key in keys:
            value = self[key]
            if schema_lib.MISSING_VALUE == value:
              continue
            if hide_default_values and value == field.default_value:
              continue
            json_repr[key] = to_json(
                value, hide_default_values=hide_default_values, **kwargs)
      return json_repr
    else:
      return {k: to_json(v, **kwargs) for k, v in self.items()}

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: schema_lib.ValueSpec,
      allow_partial: bool,
      child_transform: typing.Optional[
          typing.Callable[[object_utils.KeyPath, schema_lib.Field, typing.Any],
                          typing.Any]] = None
  ) -> typing.Tuple[bool, 'Dict']:
    """Implement pg.typing.CustomTyping interface.

    Args:
      path: KeyPath of current object.
      value_spec: Origin value spec of the field.
      allow_partial: Whether allow partial object to be created.
      child_transform: Function to transform child node values in dict_obj into
        their final values. Transform function is called on leaf nodes first,
        then on their containers, recursively.

    Returns:
      A tuple (proceed_with_standard_apply, transformed value)
    """
    proceed_with_standard_apply = True
    if self._value_spec:
      if value_spec and not value_spec.is_compatible(self._value_spec):
        raise ValueError(
            object_utils.message_on_path(
                f'Dict cannot be applied to a different spec. '
                f'Encountered spec: {value_spec!r}.', path))
      if self._allow_partial == allow_partial:
        proceed_with_standard_apply = False
      else:
        self._allow_partial = allow_partial
    elif isinstance(value_spec, schema_lib.Dict):
      self._value_spec = value_spec
    return (proceed_with_standard_apply, self)

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      hide_default_values: bool = False,
      hide_missing_values: bool = False,
      exclude_keys: typing.Optional[typing.Set[typing.Text]] = None,
      cls_name: typing.Optional[typing.Text] = None,
      bracket_type: object_utils.BracketType = object_utils.BracketType.CURLY,
      **kwargs) -> typing.Text:
    """Formats this Dict."""
    cls_name = cls_name or ''
    exclude_keys = exclude_keys or set()
    def _indent(text, indent):
      return ' ' * 2 * indent + text

    field_list = []
    if self._value_spec and self._value_spec.schema:
      matched_keys, unmatched = self._value_spec.schema.resolve(self.keys())
      assert not unmatched
      for key_spec, keys in matched_keys.items():
        for key in keys:
          if key not in exclude_keys:
            field = self._value_spec.schema[key_spec]
            if schema_lib.MISSING_VALUE == self[key]:
              if hide_missing_values:
                continue
            elif hide_default_values and self[key] == field.default_value:
              continue
            field_list.append((field, key, self[key]))
    else:
      for k, v in self.items():
        if k not in exclude_keys:
          field_list.append((None, k, v))

    open_bracket, close_bracket = object_utils.bracket_chars(bracket_type)
    if not field_list:
      return f'{cls_name}{open_bracket}{close_bracket}'

    if compact:
      s = [f'{cls_name}{open_bracket}']
      kv_strs = []
      for f, k, v in field_list:
        v_str = object_utils.format(
            v,
            compact,
            verbose,
            root_indent + 1,
            hide_default_values=hide_default_values,
            hide_missing_values=hide_missing_values,
            **kwargs)
        kv_strs.append(f'{k}={v_str}')
      s.append(', '.join(kv_strs))
      s.append(close_bracket)
    else:
      s = [f'{cls_name}{open_bracket}\n']
      for i, (f, k, v) in enumerate(field_list):
        if i != 0:
          s.append(',\n')

        if verbose and f and typing.cast(schema_lib.Field, f).description:
          if i != 0:
            s.append('\n')
          s.append(_indent(
              f'# {typing.cast(schema_lib.Field, f).description}\n',
              root_indent + 1))
        v_str = object_utils.format(
            v,
            compact,
            verbose,
            root_indent + 1,
            hide_default_values=hide_default_values,
            hide_missing_values=hide_missing_values,
            **kwargs)
        s.append(_indent(f'{k} = {v_str}', root_indent + 1))
      s.append('\n')
      s.append(_indent(close_bracket, root_indent))
    return ''.join(s)

  def __repr__(self) -> typing.Text:
    """Operator repr()."""
    return self.format(compact=True)

  def __eq__(self, other: typing.Any) -> bool:
    """Operator ==."""
    if isinstance(other, dict):
      return dict.__eq__(self, other)
    return False

  def __ne__(self, other: typing.Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)

  def __hash__(self) -> int:
    """Overriden hashing function using symbolic hash."""
    return self.sym_hash()


class List(list, Symbolic, schema_lib.CustomTyping):
  """Symbolic list.

  ``pg.List`` implements a list type whose instances are symbolically
  programmable, which is a subclass of the built-in Python ``list``,
  and the subclass of ``pg.Symbolic``.

  ``pg.List`` can be used as a regular list::

    # Construct a symbolic list from an iterable object.
    l = pg.List(range(10))

  It also supports symbolic validation through the ``value_spec`` argument::

    l = pg.List([1, 2, 3], value_spec=pg.typing.List(
        pg.typing.Int(min_value=1),
        max_size=10
    ))

    # Raises: 0 is not in acceptable range.
    l.append(0)

  And can be symbolically manipulated::

    l = pg.List([{'foo': 1}])
    l.rebind({
      '[0].foo': 2
    })

    pg.query(l, where=lambda x: isinstance(x, int))

  The user call also subscribe changes to its sub-nodes::

    def on_change(updates):
      print(updates)

    l = pg.List([{'foo': 1}], onchange_callaback=on_change)

    # `on_change` will be triggered on item insertion.
    l.append({'bar': 2})

    # `on_change` will be triggered on item removal.
    l.pop(0)

    # `on_change` will also be triggered on subtree change.
    l.rebind({'[0].bar': 3})

  """

  @classmethod
  def partial(cls,
              items: typing.Optional[typing.List[typing.Any]] = None,
              *,
              value_spec: typing.Optional[schema_lib.List] = None,
              onchange_callback: typing.Optional[typing.Callable[
                  [typing.Dict[object_utils.KeyPath,
                               FieldUpdate]], None]] = None,
              **kwargs) -> 'List':
    """Class method that creates a partial List object."""
    return cls(items,
               value_spec=value_spec,
               onchange_callback=onchange_callback,
               allow_partial=True,
               **kwargs)

  @classmethod
  def from_json(cls,
                json_value: typing.Any,
                *,
                allow_partial: bool = False,
                root_path: typing.Optional[object_utils.KeyPath] = None,
                **kwargs) -> 'List':
    """Class method that load an symbolic List from a JSON value.

    Example::

        l = List.from_json([{
            '_type': '__main__.Foo',
            'f1': 1,
            'f2': {
              'f21': True
            }
          },
          1
        ])

        assert l.value_spec is None
        # Okay:
        l.append('abc')

        # [0].f2 is bound by class Foo's field 'f2' definition
        # (assuming it defines a schema for the Dict field).
        assert l[0].f2.value_spec is not None

        # Not okay:
        l[0].f2.abc = 1

    Args:
      json_value: Input JSON value, only JSON list is acceptable.
      allow_partial: Whether to allow elements of the list to be partial.
      root_path: KeyPath of loaded object in its object tree.
      **kwargs: Allow passing through keyword arguments that are not applicable.

    Returns:
      A schema-less symbolic list, but its items maybe symbolic.
    """
    return cls(json_value, allow_partial=allow_partial, root_path=root_path)

  def __init__(self,
               items: typing.Optional[typing.List[typing.Any]] = None,
               *,
               value_spec: typing.Optional[schema_lib.List] = None,
               onchange_callback: typing.Optional[typing.Callable[
                   [typing.Dict[object_utils.KeyPath,
                                FieldUpdate]], None]] = None,
               allow_partial: bool = False,
               accessor_writable: bool = True,
               sealed: bool = False,
               root_path: typing.Optional[object_utils.KeyPath] = None):
    """Constructor.

    Args:
      items: A list as initial value for this List.
      value_spec: Value spec that applies to this List.
      onchange_callback: Callback when sub-tree has been modified.
      allow_partial: Whether to allow unbound or partial fields. This takes
        effect only when value_spec is not None.
      accessor_writable: Whether to allow modification of this List using
        accessors (operator[]).
      sealed: Whether to seal this List after creation.
      root_path: KeyPath of this List in its object tree.
    """
    if items is not None and not isinstance(items, list):
      raise TypeError(f'Argument \'items\' must be list type. '
                      f'Encountered {type(items)}.')

    if value_spec and not isinstance(value_spec, schema_lib.List):
      raise TypeError(f'Argument \'value_spec\' must be a schema.List type.'
                      f'Encountered {type(value_spec)}.')

    # We delay seal operation until items are filled.
    Symbolic.__init__(self,
                      allow_partial=allow_partial,
                      accessor_writable=accessor_writable,
                      sealed=False,
                      root_path=root_path)

    self._value_spec = None
    self._onchange_callback = None

    list.__init__(self)
    if items:
      for item in items:
        self._set_item_without_permission_check(len(self), item)

    if value_spec:
      self.use_value_spec(value_spec, allow_partial)

    # NOTE(daiyip): We set onchange callback at the end of init to avoid
    # triggering during initialization.
    self._onchange_callback = onchange_callback
    self.seal(sealed)

  @property
  def max_size(self) -> typing.Optional[int]:
    """Returns max size of this list."""
    if self._value_spec:
      return typing.cast(schema_lib.ListKey,
                         self._value_spec.element.key).max_value
    return None

  def use_value_spec(self,
                     value_spec: typing.Optional[schema_lib.List],
                     allow_partial: bool = False) -> 'List':
    """Applies a ``pg.typing.List`` as the value spec for current list.

    Args:
      value_spec: A List ValueSpec to apply to this List.
        If current List is schema-less (whose immediate members are not
        validated against schema), and `value_spec` is not None, the value spec
        will be applied to the List.
        Or else if current List is already symbolic (whose immediate members
        are under the constraint of a List value spec), and `value_spec` is
        None, current List will become schema-less. However, the schema
        constraints for non-immediate members will remain.
      allow_partial: Whether allow partial dict based on the schema. This flag
        will override allow_partial flag in __init__ for spec-less List.

    Returns:
      Self.

    Raises:
      ValueError: schema validation failed due to value error.
      RuntimeError: List is already bound with another value_spec.
      TypeError: type errors during validation.
      KeyError: key errors during validation.
    """
    if value_spec is None:
      self._value_spec = None
      self._accessor_writable = True
      return self

    if self._value_spec and self._value_spec != value_spec:
      raise RuntimeError(
          self._error_message(
              f'List is already bound with a different value '
              f'spec: {self._value_spec}. New value spec: {value_spec}.'))
    self._allow_partial = allow_partial

    if _enabled_type_check():
      # NOTE(daiyip): self._value_spec will be set in Dict.custom_apply method
      # called by spec.apply, thus we don't need to set the _value_spec
      # explicitly.
      value_spec.apply(
          self,
          allow_partial=_allow_partial(self),
          child_transform=_symbolic_transform_fn(self._allow_partial),
          root_path=self.sym_path)
    else:
      self._value_spec = value_spec
    return self

  @property
  def value_spec(self) -> schema_lib.List:
    """Returns value spec of this List."""
    return self._value_spec

  def sym_attr_field(
      self, key: typing.Union[typing.Text, int]
      ) -> typing.Optional[schema_lib.Field]:
    """Returns the field definition for a symbolic attribute."""
    del key
    if self._value_spec is None:
      return None
    return self._value_spec.element

  def sym_hasattr(self, key: typing.Union[typing.Text, int]) -> bool:
    """Tests if a symbolic attribute exists."""
    return isinstance(key, int) and key >= -len(self) and key < len(self)

  def sym_keys(self) -> typing.Iterator[int]:
    """Symbolically iterates indices."""
    for i in range(len(self)):
      yield i

  def sym_values(self) -> typing.Iterator[typing.Any]:
    """Iterates the values of symbolic attributes."""
    return iter(self)

  def sym_items(self) -> typing.Iterator[typing.Tuple[int, typing.Any]]:
    """Iterates the (key, value) pairs of symbolic attributes."""
    return enumerate(self)

  def sym_eq(self, other: typing.Any) -> bool:
    """Tests symbolic equality."""
    return eq(self, other)

  def sym_hash(self) -> int:
    """Symbolically hashing."""
    return sym_hash((self.__class__, tuple([sym_hash(e) for e in self])))

  def _sym_getattr(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: int) -> typing.Any:
    """Gets symbolic attribute by index."""
    return self[key]

  def _sym_clone(self, deep: bool, memo=None) -> 'List':
    """Override Symbolic._clone."""
    source = []
    for v in self:
      if deep or isinstance(v, Symbolic):
        v = clone(v, deep, memo)
      source.append(v)
    return List(
        source,
        value_spec=self._value_spec,
        allow_partial=self._allow_partial,
        accessor_writable=self._accessor_writable,
        # NOTE(daiyip): parent and root_path are reset to empty
        # for copy object.
        root_path=None)

  def _sym_missing(self) -> typing.Dict[typing.Any, typing.Any]:
    """Returns missing fields."""
    missing = dict()
    for idx, elem in enumerate(self):
      if isinstance(elem, Symbolic):
        missing_child = elem.sym_missing(flatten=False)
        if missing_child:
          missing[idx] = missing_child
    return missing

  def _sym_nondefault(self) -> typing.Dict[int, typing.Any]:
    """Returns non-default values."""
    non_defaults = dict()
    for idx, elem in enumerate(self):
      if isinstance(elem, Symbolic):
        non_defaults_child = elem.non_default_values(flatten=False)
        if non_defaults_child:
          non_defaults[idx] = non_defaults_child
      else:
        non_defaults[idx] = elem
    return non_defaults

  def set_accessor_writable(self, writable: bool = True) -> 'List':
    """Sets accessor writable."""
    if self.accessor_writable == writable:
      return self
    for elem in self:
      if isinstance(elem, Symbolic):
        elem.set_accessor_writable(writable)
    super().set_accessor_writable(writable)
    return self

  def seal(self, sealed: bool = True) -> 'List':
    """Seal or unseal current object from further modification."""
    if self.is_sealed == sealed:
      return self
    for elem in self:
      if isinstance(elem, Symbolic):
        elem.seal(sealed)
    super().seal(sealed)
    return self

  def _update_children_paths(
      self,
      old_path: object_utils.KeyPath,
      new_path: object_utils.KeyPath) -> None:
    """Update children paths according to root_path of current node."""
    del old_path
    for idx, item in enumerate(self):
      if isinstance(item, Symbolic):
        item.sym_setpath(object_utils.KeyPath(idx, new_path))

  def _set_item_without_permission_check(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: int, value: typing.Any) -> FieldUpdate:
    """Set or add an item without permission check."""
    index = key
    if index >= len(self):
      index = len(self)
    should_insert = False
    if isinstance(value, Insertion):
      should_insert = True
      value = value.value
    new_value = self._formalized_value(index, value)
    old_value = schema_lib.MISSING_VALUE

    if index < len(self):
      if should_insert:
        old_value = schema_lib.MISSING_VALUE,
        list.insert(self, index, new_value)
      else:
        old_value = list.__getitem__(self, index)
        list.__setitem__(self, index, new_value)
        # Detach old value from object tree.
        if old_value is not new_value and isinstance(old_value, Symbolic):
          old_value.sym_setparent(None)
    else:
      super().append(new_value)
    return FieldUpdate(self.sym_path + index, self,
                       self._value_spec.element if self._value_spec else None,
                       old_value, new_value)

  def _formalized_value(self, idx: int, value: typing.Any):
    """Get transformed (formal) value from user input."""
    allow_partial = _allow_partial(self)
    value = from_json(
        value,
        allow_partial=allow_partial,
        root_path=object_utils.KeyPath(idx, self.sym_path))
    if self._value_spec and _enabled_type_check():
      value = self._value_spec.element.apply(
          value,
          allow_partial=allow_partial,
          transform_fn=_symbolic_transform_fn(self._allow_partial),
          root_path=object_utils.KeyPath(idx, self.sym_path))
    return self._relocate_if_symbolic(idx, value)

  @property
  def _subscribes_field_updates(self) -> bool:
    """Returns True if current list subscribes field updates."""
    return self._onchange_callback is not None

  def _on_change(self, field_updates: typing.Dict[object_utils.KeyPath,
                                                  FieldUpdate]):
    """On change event of List."""
    # Do nothing for now to handle changes of List.

    # NOTE(daiyip): Remove items that are MISSING_VALUES.
    keys_to_remove = []
    for i, item in enumerate(self):
      if schema_lib.MISSING_VALUE == item:
        keys_to_remove.append(i)
    if keys_to_remove:
      for i in reversed(keys_to_remove):
        list.__delitem__(self, i)

    # Update paths for children.
    for idx, item in enumerate(self):
      if isinstance(item, Symbolic) and item.sym_path.key != idx:
        item.sym_setpath(object_utils.KeyPath(idx, self.sym_path))

    if self._onchange_callback is not None:
      self._onchange_callback(field_updates)

  def __setitem__(self, index: int, value: typing.Any) -> None:
    """Set item in this List."""
    if _is_sealed(self):
      raise WritePermissionError(
          self._error_message('Cannot set item for a sealed List.'))

    if not _is_accessor_writable(self):
      raise WritePermissionError(
          self._error_message('Cannot modify List item by __setitem__ while '
                              'accessor_writable is set to False. '
                              'Use \'rebind\' method instead.'))

    update = self._set_item_without_permission_check(index, value)
    if _enabled_notification():
      self._notify_field_updates([update])

  def __delitem__(self, index: int) -> None:
    """Delete an item from the List."""
    if _is_sealed(self):
      raise WritePermissionError('Cannot del item from a sealed List.')

    if not _is_accessor_writable(self):
      raise WritePermissionError(
          self._error_message('Cannot delete List item while accessor_writable '
                              'is set to False. '
                              'Use \'rebind\' method instead.'))
    old_value = schema_lib.MISSING_VALUE
    if index < len(self):
      old_value = self[index]
    super().__delitem__(index)

    if _enabled_notification():
      self._notify_field_updates([
          FieldUpdate(self.sym_path + index, self,
                      self._value_spec.element if self._value_spec else None,
                      old_value, schema_lib.MISSING_VALUE)
      ])

  def append(self, value: typing.Any) -> None:
    """Appends an item."""
    if _is_sealed(self):
      raise WritePermissionError('Cannot append element on a sealed List.')
    if self.max_size is not None and len(self) >= self.max_size:
      raise ValueError(f'List reached its max size {self.max_size}.')

    update = self._set_item_without_permission_check(len(self), value)
    if _enabled_notification():
      self._notify_field_updates([update])

  def insert(self, index: int, value: typing.Any) -> None:
    """Inserts an item at a given position."""
    if _is_sealed(self):
      raise WritePermissionError('Cannot insert element on a sealed List.')
    if self.max_size is not None and len(self) >= self.max_size:
      raise ValueError(f'List reached its max size {self.max_size}.')

    update = self._set_item_without_permission_check(
        index, mark_as_insertion(value))
    if _enabled_notification():
      self._notify_field_updates([update])

  def pop(self, index: int) -> typing.Any:
    """Pop an item and return its value."""
    if index >= len(self):
      raise IndexError('pop index out of range')

    value = self[index]
    with allow_writable_accessors(True):
      del self[index]
    return value

  def remove(self, value: typing.Any) -> None:
    """Removes an item of given value."""
    for i, item in enumerate(self):
      if item == value:
        del self[i]

  def extend(self, other: typing.List[typing.Any]) -> None:
    if _is_sealed(self):
      raise WritePermissionError('Cannot extend a sealed List.')
    if self.max_size is not None and len(self) + len(other) > self.max_size:
      raise ValueError(
          f'Cannot extend List: the number of elements '
          f'({len(self) + len(other)}) exceeds max size ({self.max_size}).')
    updates = []
    for v in other:
      update = self._set_item_without_permission_check(len(self), v)
      updates.append(update)

    if _enabled_notification():
      self._notify_field_updates(updates)

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: schema_lib.ValueSpec,
      allow_partial: bool,
      child_transform: typing.Optional[
          typing.Callable[[object_utils.KeyPath, schema_lib.Field, typing.Any],
                          typing.Any]] = None
  ) -> typing.Tuple[bool, 'List']:
    """Implement pg.typing.CustomTyping interface.

    Args:
      path: KeyPath of current object.
      value_spec: Origin value spec of the field.
      allow_partial: Whether allow partial object to be created.
      child_transform: Function to transform child node values in dict_obj into
        their final values. Transform function is called on leaf nodes first,
        then on their containers, recursively.

    Returns:
      A tuple (proceed_with_standard_apply, transformed value)
    """
    proceed_with_standard_apply = True
    if self._value_spec:
      if value_spec and not value_spec.is_compatible(self._value_spec):
        raise ValueError(
            object_utils.message_on_path(
                f'List cannot be applied to an incompatible value spec. '
                f'Existing value spec: {self._value_spec!r}, '
                f'new value spec: {value_spec!r}.', path))
      if self._allow_partial == allow_partial:
        proceed_with_standard_apply = False
      else:
        self._allow_partial = allow_partial
    elif isinstance(value_spec, schema_lib.List):
      self._value_spec = value_spec
    return (proceed_with_standard_apply, self)

  def sym_jsonify(self, **kwargs) -> object_utils.JSONValueType:
    """Converts current list to a list of plain Python objects."""
    return [to_json(v, **kwargs) for v in self]

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      cls_name: typing.Optional[typing.Text] = None,
      bracket_type: object_utils.BracketType = object_utils.BracketType.SQUARE,
      **kwargs) -> typing.Text:
    """Formats this List."""

    def _indent(text, indent):
      return ' ' * 2 * indent + text

    cls_name = cls_name or ''
    open_bracket, close_bracket = object_utils.bracket_chars(bracket_type)
    s = [f'{cls_name}{open_bracket}']
    if compact:
      kv_strs = []
      for idx, elem in enumerate(self):
        v_str = object_utils.format(
            elem, compact, verbose, root_indent + 1, **kwargs)
        kv_strs.append(f'{idx}: {v_str}')
      s.append(', '.join(kv_strs))
      s.append(close_bracket)
    else:
      if self:
        for idx, elem in enumerate(self):
          if idx == 0:
            s.append('\n')
          else:
            s.append(',\n')
          v_str = object_utils.format(
              elem, compact, verbose, root_indent + 1, **kwargs)
          s.append(_indent(f'{idx} : {v_str}', root_indent + 1))
        s.append('\n')
        s.append(_indent(close_bracket, root_indent))
      else:
        s.append(close_bracket)
    return ''.join(s)

  def __copy__(self) -> 'List':
    """List.copy."""
    return self.sym_clone(deep=False)

  def __deepcopy__(self, memo) -> 'List':
    return self.sym_clone(deep=True, memo=memo)

  def __hash__(self) -> int:
    """Overriden hashing function."""
    return self.sym_hash()


class ObjectMeta(abc.ABCMeta):
  """Meta class for pg.Object."""

  @property
  def schema(cls) -> schema_lib.Schema:
    """Class level property for schema."""
    return getattr(cls, '__schema__', None)

  @property
  def sym_fields(cls) -> schema_lib.Dict:
    """Gets symbolic field."""
    return getattr(cls, '__sym_fields')

  @property
  def type_name(cls) -> typing.Text:
    """Class level property for type name.

    NOTE(daiyip): This is used for serialization/deserialization.

    Returns:
      String of <module>.<class> as identifier.
    """
    return f'{cls.__module__}.{cls.__name__}'

  @property
  def serialization_key(cls) -> typing.Text:
    """Gets serialization type key."""
    return getattr(cls, '__serialization_key__')

  @property
  def init_arg_list(cls) -> typing.List[typing.Text]:
    """Gets __init__ positional argument list."""
    return cls.schema.metadata['init_arg_list']


# Use ObjectMeta as meta class to inherit schema and type_name property.
class Object(Symbolic, metaclass=ObjectMeta):
  """Base class for symbolic user classes.

  PyGlove allow symbolic programming interfaces to be easily added to most
  Python classes in two ways:

  * Developing a dataclass-like symbolic class by subclassing ``pg.Object``.
  * Developing a class as usual and decorate it using :func:`pyglove.symbolize`.
    This also work with existing classes.

  By directly subclassing ``pg.Object``, programmers can create new symbolic
  classes with the least effort. For example::

    @pg.members([
        # Each tuple in the list defines a symbolic field for `__init__`.
        ('name', pg.typing.Str().noneable(), 'Name to greet'),
        ('time_of_day',
        pg.typing.Enum('morning', ['morning', 'afternnon', 'evening']),
        'Time of the day.')
    ])
    class Greeting(pg.Object):

      def __call__(self):
        # Values for symbolic fields can be accessed
        # as public data members of the symbolic object.
        print(f'Good {self.time_of_day}, {self.name}')

    # Create an object of Greeting and invoke it,
    # which shall print 'Good morning, Bob'.
    Greeting('Bob')()

  Symbolic fields can be inherited from the base symbolic class: the fields
  from the base class will be copied to the subclass in their declaration
  order, while the subclass can override the inherited fields with more
  restricted validation rules or different default values. For example::

    @pg.members([
        ('x', pg.typing.Int(max_value=10)),
        ('y', pg.typing.Float(min_value=0))
    ])
    class Foo(pg.Object)
      pass

    @pg.members([
        ('x', pg.typing.Int(min_value=1, default=1)),
        ('z', pg.typing.Str().noneable())
    ])
    class Bar(Foo)
      pass

    # Printing Bar's schema will show that there are 3 parameters defined:
    # x : pg.typing.Int(min_value=1, max_value=10, default=1))
    # y : pg.typing.Float(min_value=0)
    # z : pg.typing.Str().noneable()
    print(Bar.schema)
  """

  # Disable pytype attribute checking.
  _HAS_DYNAMIC_ATTRIBUTES = True

  # Class property that indicates whether to automatically register class
  # for deserialization.
  auto_register = True

  # Class property that indicates whether to allow attribute access on symbolic
  # members.
  allow_symbolic_attribute = True

  # Class property that indicates whether to allow to set or rebind symbolic
  # members by value assginment.
  allow_symbolic_assignment = False

  # Class property that indicates whether to allow use `sym_eq` for `__eq__`,
  # `sym_ne` for `__ne__`, and `sym_hash` for `__hash__`.
  allow_symbolic_comparison = True

  # Allow symbolic mutation using `rebind`.
  allow_symbolic_mutation = True

  @classmethod
  def __init_subclass__(cls):
    super().__init_subclass__()

    # Inherit schema from base classes that have schema
    # in the ordered of inheritance.
    # TODO(daiyip): size of base_schema_list can be reduced
    # by looking at their inheritance chains.
    base_schema_list = []
    for base in cls.__bases__:
      base_schema = getattr(base, 'schema', None)
      if isinstance(base_schema, schema_lib.Schema):
        base_schema_list.append(base_schema)

    cls_schema = _formalize_schema(
        schema_lib.create_schema(
            maybe_field_list=[],
            name=cls.type_name,
            base_schema_list=base_schema_list,
            allow_nonconst_keys=True,
            metadata={}))
    setattr(cls, '__schema__', cls_schema)
    setattr(cls, '__sym_fields', schema_lib.Dict(cls_schema))
    setattr(cls, '__serialization_key__', cls.type_name)
    cls_schema.metadata['init_arg_list'] = _auto_init_arg_list(cls)
    if cls.auto_register:
      _register_cls_for_deserialization(cls, cls.type_name)

    cls._update_init_signature_based_on_schema()
    cls._generate_sym_attributes_if_enabled()

  @classmethod
  def _update_init_signature_based_on_schema(cls):
    """Updates the signature of `__init__` if needed."""
    if (cls.__init__ is not Object.__init__
        and not hasattr(cls.__init__, '__sym_generated_init__')):
      # We only generate `__init__` from pg.Object subclass which does not
      # override the `__init__` method.
      # Functor and ClassWrapper override their `__init__` methods, therefore
      # they need to synchronize the __init__ signature by themselves.
      return
    signature = cls.schema.get_signature(
        cls.__module__, '__init__', f'{cls.__name__}.__init__')
    pseudo_init = signature.make_function(['pass'])

    # Create a new `__init__` that passes through all the arguments to
    # in `pg.Object.__init__`. This is needed for each class to use different
    # signature.
    @functools.wraps(pseudo_init)
    def _init(self, *args, **kwargs):
      # We pass through the arguments to `Object.__init__` instead of
      # `super()` since the parent class uses a generated `__init__` will
      # be delegated to `Object.__init__` eventually. Therefore, directly
      # calling `Object.__init__` is equivalent to calling `super().__init__`.
      Object.__init__(self, *args, **kwargs)
    setattr(_init, '__sym_generated_init__', True)
    setattr(cls, '__init__', _init)

  @classmethod
  def _generate_sym_attributes_if_enabled(cls):
    """Generates symbolic attributes based on schema if they are enabled."""
    def _create_sym_attribute(attr_name, field):
      return property(object_utils.make_function(
          attr_name,
          ['self'],
          [f'return self._sym_attributes[\'{attr_name}\']'],
          return_type=field.value.annotation))

    if cls.allow_symbolic_attribute:
      for key, field in cls.schema.fields.items():
        if isinstance(key, schema_lib.ConstStrKey):
          attr_name = str(key)
          if not hasattr(cls, attr_name):
            setattr(cls, attr_name, _create_sym_attribute(attr_name, field))

  @classmethod
  def partial(cls, *args, **kwargs) -> 'Object':
    """Class method that creates a partial object of current class."""
    return cls(*args, allow_partial=True, **kwargs)

  @classmethod
  def from_json(
      cls,
      json_value: typing.Any,
      *,
      allow_partial: bool = False,
      root_path: typing.Optional[object_utils.KeyPath] = None) -> 'Object':
    """Class method that load an symbolic Object from a JSON value.

    Example::

        @pg.members([
          ('f1', pg.typing.Int()),
          ('f2', pg.typing.Dict([
            ('f21', pg.typing.Bool())
          ]))
        ])
        class Foo(pg.Object):
          pass

        foo = Foo.from_json({
            'f1': 1,
            'f2': {
              'f21': True
            }
          })

        # or

        foo2 = symbolic.from_json({
            '_type': '__main__.Foo',
            'f1': 1,
            'f2': {
              'f21': True
            }
        })

        assert foo == foo2

    Args:
      json_value: Input JSON value, only JSON dict is acceptable.
      allow_partial: Whether to allow elements of the list to be partial.
      root_path: KeyPath of loaded object in its object tree.

    Returns:
      A symbolic Object instance.
    """
    return cls(allow_partial=allow_partial, root_path=root_path, **json_value)

  def __init__(
      self,
      *args,
      allow_partial: bool = False,
      sealed: typing.Optional[bool] = None,
      root_path: typing.Optional[object_utils.KeyPath] = None,
      explicit_init: bool = False,
      **kwargs):
    """Create an Object instance.

    Args:
      *args: positional arguments.
      allow_partial: If True, the object can be partial.
      sealed: If True, seal the object from future modification (unless under
        a `pg.seal(False)` context manager). If False, treat the object as
        unsealed. If None, it's determined by `cls.allow_symbolic_mutation`.
      root_path: The symbolic path for current object. By default it's None,
        which indicates that newly constructed object does not have a parent.
      explicit_init: Should set to `True` when `__init__` is called via
        `pg.Object.__init__` instead of `super().__init__`.
      **kwargs: key/value arguments that align with the schema. All required
        keys in the schema must be specified, and values should be acceptable
        according to their value spec.

    Raises:
      KeyError: When required key(s) are missing.
      ValueError: When value(s) are not acceptable by their value spec.
    """
    # Placeholder for Google-internal usage instrumentation.

    if sealed is None:
      sealed = not self.__class__.allow_symbolic_mutation

    if not isinstance(allow_partial, bool):
      raise TypeError(
          f'Expect bool type for argument \'allow_partial\' in '
          f'symbolic.Object.__init__ but encountered {allow_partial}.')

    # Create dummy `_sym_attributes` before enter super.__init__, which is
    # needed for '__getattr__' to function correctly. Calling to
    # `super.__init__` may slip into other base's `__init__` when
    # multi-inheritance is encountered.
    self._set_raw_attr('_sym_attributes', Dict())

    # We delay the seal attempt until members are all set.
    super().__init__(
        allow_partial=allow_partial,
        accessor_writable=self.__class__.allow_symbolic_assignment,
        sealed=sealed,
        root_path=root_path,
        init_super=not explicit_init)

    # Fill field_args and init_args from **kwargs.
    _, unmatched_keys = self.__class__.schema.resolve(list(kwargs.keys()))
    if unmatched_keys:
      arg_phrase = object_utils.auto_plural(len(unmatched_keys), 'argument')
      keys_str = object_utils.comma_delimited_str(unmatched_keys)
      raise TypeError(
          f'{self.__class__.__name__}.__init__() got unexpected '
          f'keyword {arg_phrase}: {keys_str}')

    field_args = {}
    # Fill field_args and init_args from *args.
    init_arg_names = self.__class__.init_arg_list
    if args:
      if not self.__class__.schema.fields:
        raise TypeError(f'{self.__class__.__name__}() takes no arguments.')
      elif init_arg_names and init_arg_names[-1].startswith('*'):
        vararg_name = init_arg_names[-1][1:]
        vararg_field = self.__class__.schema.get_field(vararg_name)
        assert vararg_field is not None

        num_named_args = len(init_arg_names) - 1
        field_args[vararg_name] = list(args[num_named_args:])
        args = args[:num_named_args]
      elif len(args) > len(init_arg_names):
        arg_phrase = object_utils.auto_plural(len(init_arg_names), 'argument')
        was_phrase = object_utils.auto_plural(len(args), 'was', 'were')
        raise TypeError(
            f'{self.__class__.__name__}.__init__() takes '
            f'{len(init_arg_names)} positional {arg_phrase} but {len(args)} '
            f'{was_phrase} given.')

      for i, arg_value in enumerate(args):
        arg_name = init_arg_names[i]
        field_args[arg_name] = arg_value

    for k, v in kwargs.items():
      if k in field_args:
        values_str = object_utils.comma_delimited_str([field_args[k], v])
        raise TypeError(
            f'{self.__class__.__name__}.__init__() got multiple values for '
            f'argument \'{k}\': {values_str}.')
      field_args[k] = v

    # Check missing arguments when partial binding is disallowed.
    if not _allow_partial(self):
      missing_args = []
      for field in self.__class__.schema.fields.values():
        if (not field.value.has_default
            and isinstance(field.key, schema_lib.ConstStrKey)
            and field.key not in field_args):
          missing_args.append(str(field.key))
      if missing_args:
        arg_phrase = object_utils.auto_plural(len(missing_args), 'argument')
        keys_str = object_utils.comma_delimited_str(missing_args)
        raise TypeError(
            f'{self.__class__.__name__}.__init__() missing {len(missing_args)} '
            f'required {arg_phrase}: {keys_str}.')

    self._set_raw_attr('_sym_attributes', Dict(
        field_args,
        value_spec=self.__class__.sym_fields,
        allow_partial=allow_partial,
        sealed=sealed,
        accessor_writable=self.__class__.allow_symbolic_assignment,
        root_path=root_path,
        pass_through_parent=True))
    self._sym_attributes.sym_setparent(self)
    self._on_init()
    self.seal(sealed)

  #
  # Events that subclasses can override.
  #

  def _on_init(self):
    """Event that is triggered at then end of __init__."""
    self._on_bound()

  def _on_bound(self) -> None:
    """Event that is triggered when any value in the subtree are set/updated.

    NOTE(daiyip): This is the best place to set derived members from members
    registered by the schema. It's called when any value in the sub-tree is
    modified, thus making sure derived members are up-to-date.

    When derived members are expensive to create/update, you can implement
    _init, _on_rebound, _on_subtree_rebound to update derived members only when
    they are impacted.

    _on_bound is not called on per-field basis, it's called at most once
    during a rebind call (though many fields may be updated)
    and during __init__.
    """

  def _on_change(self, field_updates: typing.Dict[object_utils.KeyPath,
                                                  FieldUpdate]):
    """Event that is triggered when field values in the subtree are updated.

    This event will be called
      * On per-field basis when object is modified via attribute.
      * In batch when multiple fields are modified via `rebind` method.

    When a field in an object tree is updated, all ancestors' `_on_change` event
    will be triggered in order, from the nearest one to furthest one.

    Args:
      field_updates: Updates made to the subtree. Key path is relative to
        current object.

    Returns:
      it will call `_on_bound` and return the return value of `_on_bound`.
    """
    del field_updates
    return self._on_bound()

  def _on_path_change(
      self, old_path: object_utils.KeyPath, new_path: object_utils.KeyPath):
    """Event that is triggered after the symbolic path changes."""
    del old_path, new_path

  def _on_parent_change(
      self,
      old_parent: typing.Optional[Symbolic],
      new_parent: typing.Optional[Symbolic]):
    """Event that is triggered after the symbolic parent changes."""
    del old_parent, new_parent

  @property
  def sym_init_args(self) -> Dict:
    """Returns symbolic attributes which are the arguments for `__init__`."""
    return self._sym_attributes

  def sym_hasattr(self, key: typing.Union[typing.Text, int]) -> bool:
    """Tests if a symbolic attribute exists."""
    if key == '_sym_attributes':
      raise ValueError(
          f'{self.__class__.__name__}.__init__ should call `super().__init__`.')
    return (isinstance(key, str)
            and not key.startswith('_') and key in self._sym_attributes)

  def sym_attr_field(
      self, key: typing.Union[typing.Text, int]
      ) -> typing.Optional[schema_lib.Field]:
    """Returns the field definition for a symbolic attribute."""
    return self._sym_attributes.sym_attr_field(key)

  def sym_keys(self) -> typing.Iterator[typing.Text]:
    """Iterates the keys of symbolic attributes."""
    return self._sym_attributes.sym_keys()

  def sym_values(self):
    """Iterates the values of symbolic attributes."""
    return self._sym_attributes.sym_values()

  def sym_items(self):
    """Iterates the (key, value) pairs of symbolic attributes."""
    return self._sym_attributes.sym_items()

  def sym_eq(self, other: typing.Any) -> bool:
    """Tests symbolic equality."""
    if (self is other
        or (isinstance(other, self.__class__)
            and eq(self._sym_attributes, other._sym_attributes))):  # pylint: disable=protected-access
      return True
    # Fall back to operator == when symbolic comparison is not the
    # default behavior.
    return not self.allow_symbolic_comparison and self == other

  def sym_hash(self) -> int:
    """Symbolically hashing."""
    return sym_hash((self.__class__, sym_hash(self._sym_attributes)))

  def sym_setparent(self, parent: 'Symbolic'):
    """Sets the parent of current node in the symbolic tree."""
    old_parent = self.sym_parent
    super().sym_setparent(parent)
    if old_parent is not parent:
      self._on_parent_change(old_parent, parent)

  def _sym_getattr(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: typing.Text) -> typing.Any:
    """Get symbolic field by key."""
    return self._sym_attributes[key]

  def _sym_rebind(
      self, path_value_pairs: typing.Dict[object_utils.KeyPath, typing.Any]
      ) -> typing.List[FieldUpdate]:
    """Rebind current object using object-form members."""
    return self._sym_attributes._sym_rebind(path_value_pairs)  # pylint: disable=protected-access

  def _sym_clone(self, deep: bool, memo=None) -> 'Object':
    """Copy flags."""
    kwargs = dict()
    for k, v in self._sym_attributes.items():
      if deep or isinstance(v, Symbolic):
        v = clone(v, deep, memo)
      kwargs[k] = v
    return self.__class__(allow_partial=self._allow_partial,
                          sealed=self._sealed,
                          **kwargs)  # pytype: disable=not-instantiable

  def _sym_missing(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns missing values."""
    return self._sym_attributes.sym_missing(flatten=False)

  def _sym_nondefault(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns non-default values."""
    return self._sym_attributes.sym_nondefault(flatten=False)

  def set_accessor_writable(self, writable: bool = True) -> 'Object':
    """Sets accessor writable."""
    self._sym_attributes.set_accessor_writable(writable)
    super().set_accessor_writable(writable)
    return self

  def seal(self, sealed: bool = True) -> 'Object':
    """Seal or unseal current object from further modification."""
    self._sym_attributes.seal(sealed)
    super().seal(sealed)
    return self

  def _update_children_paths(
      self,
      old_path: object_utils.KeyPath,
      new_path: object_utils.KeyPath) -> None:
    """Update children paths according to root_path of current node."""
    self._sym_attributes.sym_setpath(new_path)
    self._on_path_change(old_path, new_path)

  def _set_item_without_permission_check(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: typing.Text, value: typing.Any) -> FieldUpdate:
    """Set item without permission check."""
    return self._sym_attributes._set_item_without_permission_check(key, value)  # pylint: disable=protected-access

  @property
  def _subscribes_field_updates(self) -> bool:
    """Returns True if current object subscribes field updates.

    For pg.Object, this return True only when _on_change is overridden
    from subclass.
    """
    return self._on_change.__code__ is not Object._on_change.__code__  # pytype: disable=attribute-error

  def keys(self) -> typing.List[typing.Text]:
    """Returns all field names."""
    return list(self._sym_attributes.keys())

  def __iter__(self):
    """Iterates symbolic members in tuples (key, value)."""
    return self.sym_items()

  def __contains__(self, key: typing.Text) -> bool:
    """Returns True current object contains a symbolic attribute."""
    return self.sym_hasattr(key)

  def __setitem__(self, key: typing.Text, value: typing.Any) -> None:
    """Set field value by operator []."""
    self._sym_attributes[key] = value

  def __getitem__(self, key: typing.Text) -> typing.Any:
    """Get field value by operator []."""
    return self.sym_getattr(key)

  def __setattr__(self, name: typing.Text, value: typing.Any) -> None:
    """Set field value by attribute."""
    # NOTE(daiyip): two types of members are treated as regular members:
    # 1) All private members which prefixed with '_'.
    # 2) Public members that are not declared as symbolic members.
    if (not self.allow_symbolic_attribute
        or not self.__class__.schema.get_field(name)
        or name.startswith('_')):
      super().__setattr__(name, value)
    else:
      if _is_sealed(self):
        raise WritePermissionError(
            self._error_message(
                f'Cannot set attribute {name!r}: object is sealed.'))
      if not _is_accessor_writable(self):
        raise WritePermissionError(
            self._error_message(
                f'Cannot set attribute of <class {self.__class__.__name__}> '
                f'while `{self.__class__.__name__}.allow_symbolic_assignment` '
                f'is set to False.'))
      self._sym_attributes[name] = value

  def __getattribute__(self, name: typing.Text) -> typing.Any:
    """Override to accomondate symbolic attributes with variable keys."""
    try:
      return super().__getattribute__(name)
    except AttributeError as error:
      if not self.allow_symbolic_attribute or not self.sym_hasattr(name):
        raise error
      return self._sym_getattr(name)

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self.allow_symbolic_comparison:
      return self.sym_eq(other)
    # NOTE(daiyip): In Python2, not all classes (e.g. int) have `__eq__` method.
    # Therefore we always check the presence of `__eq__` and use it when
    # possible. Otherwise we downgrade the default behavior by returning
    # `NotImplemented`.
    if hasattr(super(), '__eq__'):
      return super().__eq__(other)

    # In Python, `__eq__` may returns NotImplemented to fallback to the
    # default equality check. Reference:
    # https://stackoverflow.com/questions/40780004/returning-notimplemented-from-eq
    return NotImplemented

  def __ne__(self, other: typing.Any) -> bool:
    """Operator!=."""
    r = self.__eq__(other)
    if r is NotImplemented:
      return r
    return not r

  def __hash__(self) -> int:
    """Hashing function."""
    if self.allow_symbolic_comparison:
      return self.sym_hash()
    return super().__hash__()

  def sym_jsonify(self, **kwargs) -> object_utils.JSONValueType:
    """Converts current object to a dict of plain Python objects."""
    return object_utils.merge([{
        _TYPE_NAME_KEY: self.__class__.serialization_key
    }, self._sym_attributes.to_json(**kwargs)])

  def format(self,
             compact: bool = False,
             verbose: bool = False,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Formats this object."""
    return self._sym_attributes.format(
        compact,
        verbose,
        root_indent,
        cls_name=self.__class__.__name__,
        bracket_type=object_utils.BracketType.ROUND,
        **kwargs)


class Functor(Object, object_utils.Functor):
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

  # `schema_lib.Schema` object for arguments.
  arg_schema = None

  def __init__(
      self,
      *args,
      root_path: typing.Optional[object_utils.KeyPath] = None,
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
    for i in range(len(args)):
      bound_kwargs[self.signature.args[i].name] = args[i]

    if varargs is not None:
      bound_kwargs[self.signature.varargs.name] = varargs

    for k, v in kwargs.items():
      if schema_lib.MISSING_VALUE != v:
        if k in bound_kwargs:
          raise TypeError(
              f'{self.signature.id}() got multiple values for keyword '
              f'argument {k!r}.')
        bound_kwargs[k] = v

    super().__init__(allow_partial=True,
                     root_path=root_path,
                     **bound_kwargs)

    self._override_args = override_args
    self._ignore_extra_args = ignore_extra_args
    self._bound_args = set(bound_kwargs.keys())

  def _sym_clone(self, deep: bool, memo: typing.Any) -> 'Functor':
    """Override to copy bound args."""
    other = super()._sym_clone(deep, memo)
    other._bound_args = set(self._bound_args)  # pylint: disable=protected-access
    return typing.cast(Functor, other)

  def _on_change(self, field_updates: typing.Dict[object_utils.KeyPath,
                                                  FieldUpdate]):
    """Custom handling field change to update bound args."""
    for relative_path, update in field_updates.items():
      assert relative_path
      if schema_lib.MISSING_VALUE == update.new_value:
        if len(relative_path) == 1:
          self._bound_args.discard(str(relative_path))
      else:
        self._bound_args.add(relative_path.keys[0])

  def __delattr__(self, name: typing.Text) -> None:
    """Discard a previously bound argument and reset to its default value."""
    del self._sym_attributes[name]
    self._bound_args.discard(name)

  def _sym_missing(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns missing values for Functor.

    Semantically unbound arguments are not missing, thus we only return partial
    bound arguments in `sym_missing`. As a result, a functor is partial only
    when any of its bound arguments is partial.

    Returns:
      A dict of missing key (or path) to missing value.
    """
    missing = dict()
    for k, v in self._sym_attributes.items():
      if schema_lib.MISSING_VALUE != v and isinstance(v, Symbolic):
        missing_child = v.sym_missing(flatten=False)
        if missing_child:
          missing[k] = missing_child
    return missing

  @property
  def unbound_args(self) -> typing.Set[typing.Text]:
    """Returns unbound argument names."""
    return set([name for name in self._sym_attributes.keys()
                if name not in self._bound_args])

  @property
  def bound_args(self) -> typing.Set[typing.Text]:
    """Returns bound argument names."""
    return self._bound_args

  @property
  def is_fully_bound(self) -> bool:
    """Returns if all arguments of functor is bound."""
    return not self.unbound_args

  @abc.abstractmethod
  def _call(self, *args, **kwargs) -> typing.Callable:  # pylint: disable=g-bare-generic
    """Actual function logic. Subclasses should implement this method."""

  def __call__(self, *args, **kwargs) -> typing.Any:
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
        k: v for k, v in self._sym_attributes.items() if k in self._bound_args
    }
    assert len(keyword_args) == len(self._bound_args)

    # Work out varargs when positional arguments are provided.
    varargs = None
    if self.signature.has_varargs:
      varargs = list(args[len(self.signature.args):])
      if _enabled_type_check():
        varargs = self.signature.varargs.value_spec.apply(
            varargs, root_path=self.sym_path + self.signature.varargs.name)
      args = args[:len(self.signature.args)]

    # Convert positional arguments to keyword arguments so we can map them back
    # later.
    for i in range(len(args)):
      arg_spec = self.signature.args[i]
      arg_name = arg_spec.name
      if arg_name in self._bound_args:
        if not override_args:
          raise TypeError(
              f'{self.signature.id}() got new value for argument {arg_name!r} '
              f'from position {i}, but \'override_args\' is set to False. '
              f'Old value: {keyword_args[arg_name]!r}, new value: {args[i]!r}.')
      arg_value = args[i]
      if _enabled_type_check():
        arg_value = arg_spec.value_spec.apply(
            arg_value, root_path=self.sym_path + arg_name)
      keyword_args[arg_name] = arg_value

    for arg_name, arg_value in kwargs.items():
      if arg_name in self._bound_args:
        if not override_args:
          raise TypeError(
              f'{self.signature.id}() got new value for argument {arg_name!r} '
              f'from keyword argument, while \'override_args\' is set to '
              f'False. Old value: {keyword_args[arg_name]!r}, '
              f'new value: {arg_value!r}.')
      arg_spec = self.signature.get_value_spec(arg_name)
      if arg_spec and _enabled_type_check():
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
      elif arg.value_spec.default != schema_lib.MISSING_VALUE:
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
    if self.signature.return_value and _enabled_type_check():
      return_value = self.signature.return_value.apply(
          return_value, root_path=self.sym_path + 'returns')
    if _is_tracking_origin() and isinstance(return_value, Symbolic):
      return_value.sym_setorigin(self, 'return')
    return return_value


_TUPLE_MARKER = '__tuple__'


@members([('value', schema_lib.Any(), 'Value to insert.')])
class Insertion(Object):
  """Class that marks a value to insert into a list.

  Example::

    l = pg.List([0, 1])
    l.rebind({
      0: pg.Insertion(2)
    })
    assert l == [2, 0, 1]
  """


def mark_as_insertion(value: typing.Any) -> Insertion:
  """Mark a value as an insertion to a List."""
  return Insertion(value=value)


class Diff(PureSymbolic, Object):
  """A value diff between two objects: a 'left' object and a 'right' object.

  If one of them is missing, it may be represented by pg.Diff.MISSING

  For example::

    >>> pg.Diff(3.14, 1.618)
    Diff(left=3.14, right=1.618)
    >>> pg.Diff('hello world', pg.Diff.MISSING)
    Diff(left='hello world', right=MISSING)
  """

  class _Missing:
    """Represents an absent party in a Diff."""

    def __repr__(self):
      return self.__str__()

    def __str__(self):
      return 'MISSING'

    def __eq__(self, other):
      return isinstance(other, Diff._Missing)

    def __ne__(self, other):
      return not self.__eq__(other)

  MISSING = _Missing()

  def _on_bound(self):
    super()._on_bound()
    if self.left == Diff.MISSING and self.right == Diff.MISSING:
      raise ValueError(
          'At least one of \'left\' and \'right\' should be specified.')
    if self.children:
      if not isinstance(self.left, type):
        raise ValueError(
            f'\'left\' must be a type when \'children\' is specified. '
            f'Encountered: {self.left!r}.')
      if not isinstance(self.right, type):
        raise ValueError(
            f'\'right\' must be a type when \'children\' is specified. '
            f'Encountered: {self.right!r}.')
    self._has_diff = None

  @property
  def is_leaf(self) -> bool:
    """Returns True if current Diff does not contain inner Diff object."""
    return not self.children

  def __bool__(self):
    """Returns True if there is a diff."""
    if self._has_diff is None:
      if ne(self.left, self.right):
        has_diff = True
      elif self.children:
        has_diff = any(bool(cd) for cd in self.children.values())
      else:
        has_diff = False
      self._has_diff = has_diff
    return self._has_diff

  def sym_eq(self, other: typing.Any):
    """Override symbolic equality."""
    if super().sym_eq(other):
      return True
    if not bool(self):
      return eq(self.left, other)

  @property
  def value(self):
    """Returns the value if left and right are the same."""
    if bool(self):
      raise ValueError(
          f'\'value\' cannot be accessed when \'left\' and \'right\' '
          f'are not the same. Left={self.left!r}, Right={self.right!r}.')
    return self.left

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs):
    """Override format to conditionally print the shared value or the diff."""
    if not bool(self):
      # When there is no diff, we simply return the value.
      return object_utils.format(
          self.value, compact, verbose, root_indent, **kwargs)
    if self.is_leaf:
      exclude_keys = kwargs.pop('exclude_keys', None)
      exclude_keys = exclude_keys or set()
      exclude_keys.add('children')
      return super().format(
          compact, verbose, root_indent, exclude_keys=exclude_keys, **kwargs)
    else:
      assert isinstance(self.left, type)
      assert isinstance(self.right, type)
      if self.left is self.right and issubclass(self.left, list):
        return self.children.format(
            compact=compact,
            verbose=verbose,
            root_indent=root_indent,
            cls_name='',
            bracket_type=object_utils.BracketType.SQUARE)
      if self.left is self.right:
        cls_name = self.left.__name__
      else:
        cls_name = f'{self.left.__name__}|{self.right.__name__}'
      return self.children.format(
          compact=compact,
          verbose=verbose,
          root_indent=root_indent,
          cls_name=cls_name,
          bracket_type=object_utils.BracketType.ROUND)


# NOTE(daiyip): we add the symbolic attribute to Diff after its declaration
# since we need to access Diff.MISSING as the default value for `left` and
# `right`.
members([
    ('left', schema_lib.Any(default=Diff.MISSING),
     'The left-hand object being compared.'),
    ('right', schema_lib.Any(default=Diff.MISSING),
     'The right-hand object being compared.'),
    ('children', schema_lib.Dict([
        (schema_lib.StrKey(), schema_lib.Object(Diff), 'Child node.')
    ]))
])(Diff)


#
# Function for rebinders.
#


def get_rebind_dict(
    rebinder: typing.Callable,  # pylint: disable=g-bare-generic
    target: Symbolic
) -> typing.Dict[typing.Text, typing.Any]:
  """Generate rebind dict using rebinder on target value.

  Args:
    rebinder: A callable object with signature:
      (key_path: object_utils.KeyPath, value: Any) -> Any or
      (key_path: object_utils.KeyPath, value: Any, parent: Any) -> Any.  If
        rebinder returns the same value from input, the value is considered
        unchanged. Otherwise it will be put into the returning rebind dict. See
        `Symbolic.rebind` for more details.
    target: Upon which value the rebind dict is computed.

  Returns:
    An ordered dict of key path string to updated value.
  """
  signature = schema_lib.get_signature(rebinder)
  if len(signature.args) == 2:
    select_fn = lambda k, v, p: rebinder(k, v)
  elif len(signature.args) == 3:
    select_fn = rebinder
  else:
    raise TypeError(
        f'Rebinder function \'{signature.id}\' should accept 2 or 3 arguments '
        f'(key_path, value, [parent]). Encountered: {signature.args}.')

  path_value_pairs = dict()

  def _fill_rebind_dict(path, value, parent):
    new_value = select_fn(path, value, parent)
    if new_value is not value:
      path_value_pairs[str(path)] = new_value
      return TraverseAction.CONTINUE
    return TraverseAction.ENTER

  traverse(target, _fill_rebind_dict)
  return path_value_pairs


#
#  Helper methods on operating symbolic.
#


class TraverseAction(enum.Enum):
  """Enum for the next action after a symbolic node is visited.

  See also: :func:`pyglove.traverse`.
  """

  # Traverse should immediately stop.
  STOP = 0

  # Traverse should enter sub-tree if sub-tree exists and traverse is in
  # pre-order. For post-order traverse, it has the same effect as CONTINUE.
  ENTER = 1

  # Traverse should continue to next node without entering the sub-tree.
  CONTINUE = 2


def traverse(x: typing.Any,
             preorder_visitor_fn: typing.Optional[
                 typing.Callable[[object_utils.KeyPath, typing.Any, typing.Any],
                                 typing.Optional[TraverseAction]]] = None,
             postorder_visitor_fn: typing.Optional[
                 typing.Callable[[object_utils.KeyPath, typing.Any, typing.Any],
                                 typing.Optional[TraverseAction]]] = None,
             root_path: typing.Optional[object_utils.KeyPath] = None,
             parent: typing.Optional[typing.Any] = None) -> bool:
  """Traverse a (maybe) symbolic value using visitor functions.

  Example::

    @pg.members([
      ('x', pg.typing.Int())
    ])
    class A(pg.Object):
      pass

    v = [{'a': A(1)}, A(2)]
    integers = []
    def track_integers(k, v, p):
      if isinstance(v, int):
        integers.append((k, v))
      return pg.TraverseAction.ENTER

    pg.traverse(v, track_integers)
    assert integers == [('[0].a.x', 1), ('[1].x', 2)]

  Args:
    x: Maybe symbolic value.
    preorder_visitor_fn: preorder visitor function. Function signature is
      `(path, value, parent) -> should_continue`.
    postorder_visitor_fn: postorder visitor function. Function signature is
      `(path, value, parent) -> should_continue`.
    root_path: KeyPath of root value.
    parent: Optional parent of the root node.

  Returns:
    True if both `preorder_visitor_fn` and `postorder_visitor_fn` return
      either `TraverseAction.ENTER` or `TraverseAction.CONTINUE` for all nodes.
      Otherwise False.
  """
  root_path = root_path or object_utils.KeyPath()

  def no_op_visitor(path, value, parent):
    del path, value, parent
    return TraverseAction.ENTER

  if preorder_visitor_fn is None:
    preorder_visitor_fn = no_op_visitor
  if postorder_visitor_fn is None:
    postorder_visitor_fn = no_op_visitor

  preorder_action = preorder_visitor_fn(root_path, x, parent)
  if preorder_action is None or preorder_action == TraverseAction.ENTER:
    if isinstance(x, dict):
      for k, v in x.items():
        if not traverse(v, preorder_visitor_fn, postorder_visitor_fn,
                        object_utils.KeyPath(k, root_path), x):
          preorder_action = TraverseAction.STOP
          break
    elif isinstance(x, list):
      for i, v in enumerate(x):
        if not traverse(v, preorder_visitor_fn, postorder_visitor_fn,
                        object_utils.KeyPath(i, root_path), x):
          preorder_action = TraverseAction.STOP
          break
    elif isinstance(x, Object):
      for k, v in x.sym_items():
        if not traverse(v, preorder_visitor_fn, postorder_visitor_fn,
                        object_utils.KeyPath(k, root_path), x):
          preorder_action = TraverseAction.STOP
          break
  postorder_action = postorder_visitor_fn(root_path, x, parent)
  if (preorder_action == TraverseAction.STOP or
      postorder_action == TraverseAction.STOP):
    return False
  return True


def diff(
    left: typing.Any,
    right: typing.Any,
    flatten: bool = False,
    collapse: typing.Union[
        bool,
        typing.Text,
        typing.Callable[[typing.Any, typing.Any], bool]] = 'same_type',
    mode: typing.Text = 'diff') -> object_utils.Nestable[Diff]:
  """Inspect the symbolic diff between two objects.

  For example::

    @pg.members([
      ('x', pg.typing.Any()),
      ('y', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    @pg.members([
      ('z', pg.typing.Any().noneable())
    ])
    class B(A):
      pass


    # Diff the same object.
    pg.diff(A(1, 2), A(1, 2))

    >> None

    # Diff the same object with mode 'same'.
    pg.diff(A(1, 2), A(1, 2), mode='same')

    >> A(
    >>   x = 1,
    >>   y = 2
    >> )

    # Diff different objects of the same type.
    pg.diff(A(1, 2), A(1, 3))

    >> A(
    >>   y = Diff(
    >>     left=2,
    >>     right=3
    >>   )
    >>  )

    # Diff objects of different type.
    pg.diff(A(1, 2), B(1, 3))

    >> Diff(
    >>    left = A(
    >>      x = 1,
    >>      y = 2
    >>    ),
    >>    right = B(
    >>      x = 1,
    >>      y = 3,
    >>      z = None
    >>    )

    # Diff objects of different type with collapse.
    pg.diff(A(1, 2), B(1, 3), collapse=True)

    >> A|B (
    >>   y = Diff(
    >>     left = 2,
    >>     right = 3,
    >>   ),
    >>   z = Diff(
    >>     left = MISSING,
    >>     right = None
    >>   )
    >> )

    # Diff objects of different type with collapse and flatten.
    # Object type is included in key '_type'.
    pg.diff(A(1, pg.Dict(a=1)), B(1, pg.Dict(a=2)), collapse=True, flatten=True)

    >> {
    >>    'y.a': Diff(1, 2),
    >>    'z', Diff(MISSING, None),
    >>    '_type': Diff(A, B)
    >> }

  Args:
    left: The left object to compare.
    right: The right object to compare.
    flatten: If True, returns a level-1 dict with diff keys flattened. Otherwise
      preserve the hierarchy of the diff result.
    collapse: One of a boolean value, string or a callable object that indicates
      whether to collapse two different values. The default value 'same_type'
      means only collapse when the two values are of the same type.
    mode: Diff mode, should be one of ['diff', 'same', 'both']. For 'diff' mode
      (the default), the return value contains only different values. For 'same'
      mode, the return value contains only same values. For 'both', the return
      value contains both different and same values.

  Returns:
    A `Diff` object when flatten is False. Otherwise a dict of string (key path)
    to `Diff`.
  """
  def _should_collapse(left, right):
    if isinstance(left, dict):
      if isinstance(right, dict):
        return True
    elif isinstance(left, list):
      return isinstance(right, list)

    if (isinstance(left, (dict, Symbolic))
        and isinstance(right, (dict, Symbolic))):
      if collapse == 'same_type':
        return type(left) is type(right)
      elif callable(collapse):
        return collapse(left, right)
      elif isinstance(collapse, bool):
        return collapse
      else:
        raise ValueError(f'Unsupported `collapse` value: {collapse!r}')
    else:
      return False

  def _add_child_diff(diff_container, key, value, child_has_diff):
    if ((mode != 'same' and child_has_diff)
        or (mode != 'diff' and not child_has_diff)):
      diff_container[key] = value

  def _get_container_ops(container):
    if isinstance(container, dict):
      return container.__contains__, container.__getitem__, container.items
    else:
      assert isinstance(container, Symbolic)
      return container.sym_hasattr, container.sym_getattr, container.sym_items

  def _diff(x, y) -> typing.Tuple[object_utils.Nestable[Diff], bool]:
    if x is y or x == y:
      return (Diff(x, y), False)
    if not _should_collapse(x, y):
      return (Diff(x, y), True)

    diff_value, has_diff = {}, False
    if isinstance(x, list):
      assert isinstance(y, list)
      def _child(l, index):
        return l[i] if index < len(l) else Diff.MISSING
      for i in range(max(len(x), len(y))):
        child_diff, child_has_diff = _diff(_child(x, i), _child(y, i))
        has_diff = has_diff or child_has_diff
        _add_child_diff(diff_value, str(i), child_diff, child_has_diff)
      diff_value = Diff(List, List, children=diff_value)
    else:
      assert isinstance(x, (dict, Symbolic))
      assert isinstance(y, (dict, Symbolic))

      x_haskey, _, x_items = _get_container_ops(x)
      y_haskey, y_getitem, y_items = _get_container_ops(y)

      for k, xv in x_items():
        yv = y_getitem(k) if y_haskey(k) else Diff.MISSING
        child_diff, child_has_diff = _diff(xv, yv)
        has_diff = has_diff or child_has_diff
        _add_child_diff(diff_value, k, child_diff, child_has_diff)

      for k, yv in y_items():
        if not x_haskey(k):
          child_diff, _ = _diff(Diff.MISSING, yv)
          has_diff = True
          _add_child_diff(diff_value, k, child_diff, True)

      xt, yt = type(x), type(y)
      same_type = xt is yt
      if not same_type:
        has_diff = True

      if flatten:
        # Put type difference with key '_type'. Since symbolic
        # fields will not start with underscore, so there should be
        # no clash.
        if not same_type or mode != 'diff':
          diff_value['_type'] = Diff(xt, yt)
      else:
        diff_value = Diff(xt, yt, children=diff_value)
    return diff_value, has_diff

  diff_value, _ = _diff(left, right)
  if flatten:
    diff_value = object_utils.flatten(diff_value)
  return diff_value


def query(
    x: typing.Any,
    path_regex: typing.Optional[typing.Text] = None,
    where: typing.Optional[typing.Union[typing.Callable[
        [typing.Any], bool], typing.Callable[[typing.Any, typing.Any],
                                             bool]]] = None,
    enter_selected: bool = False,
    custom_selector: typing.Optional[typing.Union[
        typing.Callable[[object_utils.KeyPath, typing.Any], bool],
        typing.Callable[[object_utils.KeyPath, typing.Any, typing.Any],
                        bool]]] = None
) -> typing.Dict[typing.Text, typing.Any]:
  """Queries a (maybe) symbolic value.

  Example::

      @pg.members([
          ('x', pg.typing.Int()),
          ('y', pg.typing.Int())
      ])
      class A(pg.Object):
        pass

      value = {
        'a1': A(x=0, y=1),
        'a2': [A(x=1, y=1), A(x=1, y=2)],
        'a3': {
          'p': A(x=2, y=1),
          'q': A(x=2, y=2)
        }
      }

      # Query by path regex.
      # Shall print:
      # {'a3.p': A(x=2, y=1)}
      print(pg.query(value, r'.*p'))

      # Query by value.
      # Shall print:
      # {
      #    'a2[1].y': 2,
      #    'a3.p.x': 2,
      #    'a3.q.x': 2,
      #    'a3.q.y': 2,
      # }
      print(pg.query(value, where=lambda v: v==2))

      # Query by path, value and parent.
      # Shall print:
      # {
      #    'a2[1].y': 2,
      # }
      print(pg.query(
          value, r'.*y',
          where=lambda v, p: v > 1 and isinstance(p, A) and p.x == 1))

  Args:
    x: A nested structure that may contains symbolic value.
    path_regex: Optional regex expression to constrain path.
    where: Optional callable to constrain value and parent when path matches
      with `path_regex` or `path_regex` is not provided. The signature is:

        `(value) -> should_select` or `(value, parent) -> should_select`

    enter_selected: If True, if a node is selected, enter the node and query
      its sub-nodes.
    custom_selector: Optional callable object as custom selector. When
      `custom_selector` is provided, `path_regex` and `where` must be None.
      The signature of `custom_selector` is:

        `(key_path, value) -> should_select`
        or `(key_path, value, parent) -> should_select`

  Returns:
    A dict of key path to value as results for selected values.
  """
  regex = re.compile(path_regex) if path_regex else None
  if custom_selector is not None:
    if path_regex is not None or where is not None:
      raise ValueError('\'path_regex\' and \'where\' must be None when '
                       '\'custom_selector\' is provided.')
    signature = schema_lib.get_signature(custom_selector)
    if len(signature.args) == 2:
      select_fn = lambda k, v, p: custom_selector(k, v)  # pytype: disable=wrong-arg-count
    elif len(signature.args) == 3:
      select_fn = custom_selector
    else:
      raise TypeError(
          f'Custom selector \'{signature.id}\' should accept 2 or 3 arguments. '
          f'(key_path, value, [parent]). Encountered: {signature.args}')
  else:
    if where is not None:
      signature = schema_lib.get_signature(where)
      if len(signature.args) == 1:
        where_fn = lambda v, p: where(v)  # pytype: disable=wrong-arg-count
      elif len(signature.args) == 2:
        where_fn = where
      else:
        raise TypeError(
            f'Where function \'{signature.id}\' should accept 1 or 2 '
            f'arguments: (value, [parent]). Encountered: {signature.args}.')
    else:
      where_fn = lambda v, p: True

    def select_fn(k, v, p):
      if regex is not None and not regex.match(str(k)):
        return False
      return where_fn(v, p)  # pytype: disable=wrong-arg-count

  results = {}

  def _preorder_visitor(path: object_utils.KeyPath, v: typing.Any,
                        parent: typing.Any) -> TraverseAction:
    if select_fn(path, v, parent):  # pytype: disable=wrong-arg-count
      results[str(path)] = v
      return TraverseAction.ENTER if enter_selected else TraverseAction.CONTINUE
    return TraverseAction.ENTER

  traverse(x, preorder_visitor_fn=_preorder_visitor)
  return results


def eq(left: typing.Any, right: typing.Any) -> bool:
  """Compares if two values are equal. Use symbolic equality if possible.

  Example::

    @pg.members([
      ('x', pg.typing.Any())
    ])
    class A(pg.Object):
      def sym_eq(self, right):
        if super().sym_eq(right):
          return True
        return pg.eq(self.x, right)

    class B:
      pass

    assert pg.eq(1, 1)
    assert pg.eq(A(1), A(1))
    # This is True since A has override `sym_eq`.
    assert pg.eq(A(1), 1)
    # Objects of B are compared by references.
    assert not pg.eq(A(B()), A(B()))

  Args:
    left: The left-hand value to compare.
    right: The right-hand value to compare.

  Returns:
    True if left and right is equal or symbolically equal. Otherwise False.
  """
  # NOTE(daiyip): the default behavior for dict/list/tuple comparison is that
  # it compares the elements using __eq__, __ne__. For symbolic comparison on
  # these container types, we need to change the behavior by using symbolic
  # comparison on their items.
  if left is right:
    return True
  if ((isinstance(left, list) and isinstance(right, list))
      or isinstance(left, tuple) and isinstance(right, tuple)):
    if len(left) != len(right):
      return False
    for x, y in zip(left, right):
      if ne(x, y):
        return False
    return True
  elif isinstance(left, dict):
    if (not isinstance(right, dict)
        or len(left) != len(right)
        or set(left.keys()) != set(right.keys())):
      return False
    for k, v in left.items():
      if ne(v, right[k]):
        return False
    return True
  elif isinstance(left, Object):
    return left.sym_eq(right)
  elif isinstance(right, Object):
    return right.sym_eq(left)
  return left == right


def ne(left: typing.Any, right: typing.Any) -> bool:
  """Compares if two values are not equal. Use symbolic equality if possible.

  Example::

    @pg.members([
      ('x', pg.typing.Any())
    ])
    class A(pg.Object):
      def sym_eq(self, right):
        if super().sym_eq(right):
          return True
        return pg.eq(self.x, right)

    class B:
      pass

    assert pg.ne(1, 2)
    assert pg.ne(A(1), A(2))
    # A has override `sym_eq`.
    assert not pg.ne(A(1), 1)
    # Objects of B are compared by references.
    assert pg.ne(A(B()), A(B()))

  Args:
    left: The left-hand value to compare.
    right: The right-hand value to compare.

  Returns:
    True if left and right is not equal or symbolically equal. Otherwise False.
  """
  return not eq(left, right)


def sym_hash(x: typing.Any) -> int:
  """Returns hash of value. Use symbolic hashing function if possible.

  Example::

    @pg.symbolize
    class A:
      def __init__(self, x):
        self.x = x

    assert hash(A(1)) != hash(A(1))
    assert pg.hash(A(1)) == pg.hash(A(1))
    assert pg.hash(pg.Dict(x=[A(1)])) == pg.hash(pg.Dict(x=[A(1)]))

  Args:
    x: Value for computing hash.

  Returns:
    The hash value for `x`.
  """
  if isinstance(x, Symbolic):
    return x.sym_hash()
  return hash(x)


def clone(
    x: typing.Any,
    deep: bool = False,
    memo: typing.Optional[typing.Any] = None,
    override: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None
) -> typing.Any:
  """Clones a value. Use symbolic clone if possible.

  Example::

    @pg.members([
      ('x', pg.typing.Int()),
      ('y', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    # B is not a symbolic object.
    class B:
      pass

    # Shallow copy on non-symbolic values (by reference).
    a = A(1, B())
    b = pg.clone(a)
    assert pg.eq(a, b)
    assert a.y is b.y

    # Deepcopy on non-symbolic values.
    c = pg.clone(a, deep=True)
    assert pg.ne(a, c)
    assert a.y is not c.y

    # Copy with override
    d = pg.clone(a, override={'x': 2})
    assert d.x == 2
    assert d.y is a.y

  Args:
    x: value to clone.
    deep: If True, use deep clone, otherwise use shallow clone.
    memo: Optional memo object for deep clone.
    override: Value to override if value is symbolic.

  Returns:
    Cloned instance.
  """
  if isinstance(x, Symbolic):
    return x.sym_clone(deep, memo, override)
  else:
    assert not override
    return copy.deepcopy(x, memo) if deep else copy.copy(x)


def is_deterministic(x: typing.Any) -> bool:
  """Returns if the input value is deterministic.

  Example::

    @pg.symbolize
    def foo(x, y):
      pass

    assert pg.is_deterministic(1)
    assert pg.is_deterministic(foo(1, 2))
    assert not pg.is_deterministic(pg.oneof([1, 2]))
    assert not pg.is_deterministic(foo(pg.oneof([1, 2]), 3))

  Args:
    x: Value to query against.

  Returns:
    True if value itself is not NonDeterministic and its child and nested
    child fields do not contain NonDeterministic values.
  """
  return not contains(x, type=NonDeterministic)


def is_pure_symbolic(x: typing.Any) -> bool:
  """Returns if the input value is pure symbolic.

  Example::

    class Bar(pg.PureSymbolic):
      pass

    @pg.symbolize
    def foo(x, y):
      pass

    assert not pg.is_pure_symbolic(1)
    assert not pg.is_pure_symbolic(foo(1, 2))
    assert pg.is_pure_symbolic(Bar())
    assert pg.is_pure_symbolic(foo(Bar(), 1))
    assert pg.is_pure_symbolic(foo(pg.oneof([1, 2]), 1))

  Args:
    x: Value to query against.

  Returns:
    True if value itself is PureSymbolic or its child and nested
    child fields contain PureSymbolic values.
  """
  def _check_pure_symbolic(k, v, p):
    del k, p
    if (isinstance(v, PureSymbolic)
        or (isinstance(v, Symbolic) and v.sym_puresymbolic)):
      return TraverseAction.STOP
    else:
      return TraverseAction.ENTER
  return not traverse(x, _check_pure_symbolic)


def is_abstract(x: typing.Any) -> bool:
  """Returns if the input value is abstract.

  Example::

    @pg.symbolize
    class Foo:
      def __init__(self, x):
        pass

    class Bar(pg.PureSymbolic):
      pass

    assert not pg.is_abstract(1)
    assert not pg.is_abstract(Foo(1))
    assert pg.is_abstract(Foo.partial())
    assert pg.is_abstract(Bar())
    assert pg.is_abstract(Foo(Bar()))
    assert pg.is_abstract(Foo(pg.oneof([1, 2])))

  Args:
    x: Value to query against.

  Returns:
    True if value itself is partial/PureSymbolic or its child and nested
    child fields contain partial/PureSymbolic values.
  """
  return object_utils.is_partial(x) or is_pure_symbolic(x)


def contains(
    x: typing.Any,
    value: typing.Any = None,
    type: typing.Optional[typing.Union[    # pylint: disable=redefined-builtin
        typing.Type[typing.Any],
        typing.Tuple[typing.Type[typing.Any]]]]=None
    ) -> bool:
  """Returns if a value contains values of specific type.

  Example::

    @pg.members([
        ('x', pg.typing.Any()),
        ('y', pg.typing.Any())
    ])
    class A(pg.Object):
      pass

    # Test if a symbolic tree contains a value.
    assert pg.contains(A('a', 'b'), 'a')
    assert not pg.contains(A('a', 'b'), A)

    # Test if a symbolic tree contains a type.
    assert pg.contains({'x': A(1, 2)}, type=A)
    assert pg.contains({'x': A(1, 2)}, type=int)
    assert pg.contains({'x': A(1, 2)}, type=(int, float))

  Args:
    x: The source value to query against.
    value: Value of sub-node to contain. Applicable when `type` is None.
    type: A type or a tuple of types for the sub-nodes. Applicable if
      not None.

  Returns:
    True if `x` itself or any of its sub-nodes equal to `value` or
    is an instance of `value_type`.
  """
  if type is not None:
    def _contains(k, v, p):
      del k, p
      if isinstance(v, type):
        return TraverseAction.STOP
      return TraverseAction.ENTER
  else:
    def _contains(k, v, p):
      del k, p
      if v == value:
        return TraverseAction.STOP
      return TraverseAction.ENTER
  return not traverse(x, _contains)


#
# Pluggable save/load handler.
#


def default_load_handler(path: typing.Text) -> typing.Any:
  """Default load handler from file."""
  with open(path, 'r') as f:
    content = f.read()
  return from_json_str(content, allow_partial=True)


def default_save_handler(
    value: typing.Any,
    path: typing.Text,
    indent: typing.Optional[int] = None) -> None:
  """Default save handler to file."""
  with open(path, 'w') as f:
    f.write(to_json_str(value, json_indent=indent))


_LOAD_HANDLER = default_load_handler
_SAVE_HANDLER = default_save_handler


def get_load_handler() -> typing.Optional[typing.Callable[..., typing.Any]]:
  """Returns global load handler."""
  return _LOAD_HANDLER


def get_save_handler() -> typing.Optional[typing.Callable[..., typing.Any]]:
  """Returns global save handler."""
  return _SAVE_HANDLER


def set_load_handler(
    load_handler: typing.Callable[..., typing.Any]
) -> typing.Optional[typing.Callable[..., typing.Any]]:
  """Sets global load handler.

  Args:
    load_handler: A callable object that takes arbitrary arguments and returns
      a value. `symbolic.load` method will pass through all arguments to this
      handler and return its return value.

  Returns:
    Previous global load handler.
  """
  if not callable(load_handler):
    raise ValueError('`load_handler` must be callable.')
  global _LOAD_HANDLER
  old_handler = _LOAD_HANDLER
  _LOAD_HANDLER = load_handler
  return old_handler


def set_save_handler(
    save_handler: typing.Callable[..., typing.Any]
) -> typing.Optional[typing.Callable[..., typing.Any]]:
  """Sets global save handler.

  Args:
    save_handler: A callable object that takes at least one argument as value to
      save. `symbolic.save` method will pass through all arguments to this
      handler and return its return value.

  Returns:
    Previous global save handler.
  """
  if not callable(save_handler):
    raise ValueError('`save_handler` must be callable.')
  global _SAVE_HANDLER
  old_handler = _SAVE_HANDLER
  _SAVE_HANDLER = save_handler
  return old_handler


def load(path: typing.Text, *args, **kwargs) -> typing.Any:
  """Load a symbolic value using the global load handler.

  Example::

      @pg.members([
        ('x', pg.typing.Any())
      ])
      class A(pg.Object):
        pass

      a1 = A(1)
      file = 'my_file.json'
      a1.save(file)
      a2 = pg.load(file)
      assert pg.eq(a1, a2)

  Args:
    path: A path string for loading an object.
    *args: Positional arguments that will be passed through to the global
      load handler.
    **kwargs: Keyword arguments that will be passed through to the global
      load handler.

  Returns:
    Return value from the global load handler.
  """
  value = _LOAD_HANDLER(path, *args, **kwargs)
  if _is_tracking_origin() and isinstance(value, Symbolic):
    value.sym_setorigin(path, 'load')
  return value


def save(value: typing.Any, path: typing.Text, *args, **kwargs) -> typing.Any:
  """Save a symbolic value using the global save handler.

  Example::

      @pg.members([
        ('x', pg.typing.Any())
      ])
      class A(pg.Object):
        pass

      a1 = A(1)
      file = 'my_file.json'
      a1.save(file)
      a2 = pg.load(file)
      assert pg.eq(a1, a2)

  Args:
    value: value to save.
    path: A path string for saving `value`.
    *args: Positional arguments that will be passed through to the global
      save handler.
    **kwargs: Keyword arguments that will be passed through to the global
      save handler.

  Returns:
    Return value from the global save handler.

  Raises:
    RuntimeError: if global save handler is not set.
  """
  return _SAVE_HANDLER(value, path, *args, **kwargs)
