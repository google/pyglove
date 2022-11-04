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
"""Utilities for handling schema for symbolic classes."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags


def update_schema(
    cls,
    fields: List[Union[
        pg_typing.Field,
        Tuple[Union[str, pg_typing.KeySpec], pg_typing.ValueSpec, str],
        Tuple[Union[str, pg_typing.KeySpec], pg_typing.ValueSpec, str, Any]]],
    metadata: Optional[Dict[str, Any]] = None,
    init_arg_list: Optional[Sequence[str]] = None,
    extend: bool = True,
    serialization_key: Optional[str] = None,
    additional_keys: Optional[List[str]] = None,
    add_to_registry: bool = True) -> None:
  """Updates the schema for a ``pg.Object`` subclass.

  This function allows the user to update the symbolic fields associated
  with a symbolic class. It was intended to support meta-programming
  scenarios in which symbolic fields are dynamically generated.

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
      a string. `value-spec` should be pg_typing.ValueSpec classes or
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
  cls_schema = formalize_schema(
      pg_typing.create_schema(
          maybe_field_list=fields,
          name=cls.type_name,
          base_schema_list=[cls.schema] if extend else [],
          allow_nonconst_keys=True,
          metadata=metadata))

  setattr(cls, '__schema__', cls_schema)
  setattr(cls, '__sym_fields', pg_typing.Dict(cls_schema))
  setattr(cls, '__serialization_key__', serialization_key or cls.type_name)

  if init_arg_list is None:
    init_arg_list = metadata.pop('init_arg_list', auto_init_arg_list(cls))
  validate_init_arg_list(init_arg_list, cls_schema)
  cls_schema.metadata['init_arg_list'] = init_arg_list

  if add_to_registry:
    register_cls_for_deserialization(cls, serialization_key, additional_keys)

  cls._update_init_signature_based_on_schema()  # pylint: disable=protected-access
  cls._generate_sym_attributes_if_enabled()  # pylint: disable=protected-access


def validate_init_arg_list(
    init_arg_list: List[str], cls_schema: pg_typing.Schema) -> None:
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
    if is_vararg and not isinstance(field.value, pg_typing.List):
      raise TypeError(
          f'Variable positional argument {arg!r} should be declared with '
          f'`pg.typing.List(...)`. Encountered {field.value!r}.')


def auto_init_arg_list(cls):
  """Generate the init_arg_list metadata from an pg.Object subclass."""
  # Inherit from the first non-empty base if they have the same signature.
  # This allows to bypass interface-only bases.
  init_arg_list = None
  for base_cls in cls.__bases__:
    schema = getattr(base_cls, 'schema', None)
    if isinstance(schema, pg_typing.Schema):
      if list(schema.keys()) == list(cls.schema.keys()):
        init_arg_list = base_cls.init_arg_list
      else:
        break
  if init_arg_list is None:
    # Automatically generate from the field definitions in their
    # declaration order from base classes to subclasses.
    init_arg_list = [str(key) for key in cls.schema.fields.keys()
                     if isinstance(key, pg_typing.ConstStrKey)]
  return init_arg_list


def register_cls_for_deserialization(
    cls,
    serialization_key: Optional[str] = None,
    additional_keys: Optional[List[str]] = None):
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
        key, cls, flags.is_repeated_class_registration_allowed())


def formalize_schema(schema: pg_typing.Schema) -> pg_typing.Schema:  # pylint: disable=redefined-outer-name
  """Formalize default values in the schema."""

  def _formalize_field(path: object_utils.KeyPath, node: Any) -> bool:
    """Formalize field."""
    if isinstance(node, pg_typing.Field):
      field = node
      if (not flags.is_empty_field_description_allowed()
          and not field.description):
        raise ValueError(
            f'Field description must not be empty (path={path}).')

      field.value.set_default(
          field.apply(
              field.default_value,
              allow_partial=True,
              transform_fn=base.symbolic_transform_fn(allow_partial=True)),
          use_default_apply=False)
      if isinstance(field.value, pg_typing.Dict):
        if field.value.schema is not None:
          field.value.schema.set_name(f'{schema.name}.{path.path}')
          object_utils.traverse(field.value.schema.fields, _formalize_field,
                                None, path)
      elif isinstance(field.value, pg_typing.List):
        _formalize_field(object_utils.KeyPath(0, path), field.value.element)
      elif isinstance(field.value, pg_typing.Tuple):
        for i, elem in enumerate(field.value.elements):
          _formalize_field(object_utils.KeyPath(i, path), elem)
      elif isinstance(field.value, pg_typing.Union):
        for i, c in enumerate(field.value.candidates):
          _formalize_field(
              object_utils.KeyPath(i, path),
              pg_typing.Field(field.key, c, 'Union sub-type.'))
    return True

  object_utils.traverse(schema.fields, _formalize_field)
  return schema
