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
"""Boilerplate class from symbolic object."""

import copy
import inspect

from typing import Any, List, Optional, Type

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import flags
from pyglove.core.symbolic import object as pg_object
from pyglove.core.symbolic import schema_utils


def boilerplate_class(
    cls_name: str,
    value: pg_object.Object,
    init_arg_list: Optional[List[str]] = None,
    **kwargs) -> Type[pg_object.Object]:
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
    **kwargs: Keyword arguments for infrequently used options. Acceptable
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
    A class which extends the input value's type, with its schema's default
      values set from the input value.

  Raises:
    TypeError: Keyword argumment provided is not supported.
  """
  if not isinstance(value, pg_object.Object):
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
  cls.__qualname__ = cls.__qualname__.replace(
      'boilerplate_class.<locals>._BoilerplateClass', cls_name)
  cls.__module__ = cls_module

  # Enable automatic registration for subclass.
  cls.auto_register = True

  allow_partial = value.allow_partial
  def _freeze_field(path: object_utils.KeyPath,
                    field: pg_typing.Field,
                    value: Any) -> Any:
    # We do not do validation since Object is already in valid form.
    del path
    if not isinstance(field.key, pg_typing.ListKey):
      # Recursively freeze dict field.
      if isinstance(field.value, pg_typing.Dict) and field.value.schema:
        field.value.schema.apply(
            value, allow_partial=allow_partial, child_transform=_freeze_field)
        field.value.set_default(value)
        if all(f.frozen for f in field.value.schema.values()):
          field.value.freeze()
      else:
        if value != pg_typing.MISSING_VALUE:
          field.value.freeze(copy.deepcopy(value), apply_before_use=False)
        else:
          field.value.set_default(
              pg_typing.MISSING_VALUE, use_default_apply=False)
    return value

  # NOTE(daiyip): we call `cls.__schema__.apply` to freeze fields that have
  # default values. But we no longer need to formalize `cls.__schema__`, since
  # it's copied from the boilerplate object's class which was already
  # formalized.
  with flags.allow_writable_accessors():
    cls.__schema__.apply(
        value._sym_attributes,  # pylint: disable=protected-access
        allow_partial=allow_partial,
        child_transform=_freeze_field,
    )

  if init_arg_list is not None:
    schema_utils.validate_init_arg_list(init_arg_list, cls.__schema__)
    cls.__schema__.metadata['init_arg_list'] = init_arg_list
  cls.register_for_deserialization(serialization_key, additional_keys)
  return cls
