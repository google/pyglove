# Copyright 2023 The PyGlove Authors
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
"""Compounding symbolic classes."""

import abc
import inspect
import types
from typing import Any, List, Optional, Tuple, Type, Union

from pyglove.core.symbolic import schema_utils
from pyglove.core.symbolic.object import Object
import pyglove.core.typing as pg_typing


class Compound(Object):
  """Base class for compound objects."""

  @property
  @abc.abstractmethod
  def decomposed(self) -> Any:
    """Returns the decomposed which is the created object by the factory."""

  def __init_subclass__(cls):
    # Bypass other classes' `__init_subclass__` method when `Compound`
    # is inherited as the first base class. This is to avoid side-effect
    # from the user class to compound with.
    Object.__init_subclass__(cls)

  def __init__(self, *args, **kwargs):
    # `explicit_init` allows the `__init__` of the other classes that sit after
    # `Compound` to be bypassed.
    Object.__init__(self, *args, explicit_init=True, **kwargs)


_COMPOUND_OWNED_ATTR_NAMES = frozenset(dir(Compound))


def compound_class(
    factory_fn: types.FunctionType,
    base_class: Optional[Type[Object]] = None,
    args: Optional[
        List[
            Union[
                Tuple[Tuple[str, pg_typing.KeySpec], pg_typing.ValueSpec, str],
                Tuple[
                    Tuple[str, pg_typing.KeySpec], pg_typing.ValueSpec, str, Any
                ],
            ]
        ]
    ] = None,  # pylint: disable=bad-continuation
    *,
    lazy_build: bool = True,
    auto_doc: bool = True,
    auto_typing: bool = True,
    serialization_key: Optional[str] = None,
    additional_keys: Optional[List[str]] = None,
    add_to_registry: bool = False
) -> Type[Compound]:
  """Creates a compound class from a factory function.

  Args:
    factory_fn: A function that produces a compound object.
    base_class: The base class of the compond class, which should be a
      ``pg.Object`` type. If None, it will be infererenced from the return
      annotation of `factory_fn`. If the annotation is not present or
      `auto_typing` is set to False, `base_class` must be present.
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
      `schema.Field` counterpart in `Functor.schema.fields` sorted by the
      declaration order of each argument in the function signature ( other than
      the order in `args`).  2) Default argument values are specified along with
      function definition as regular python functions, instead of being set at
      `schema.Field` level. But validation rules can be set using `args` and
      apply to argument values.
    lazy_build: If True, `factory_fn` will be called upon first use. Otherwise,
      it will be called at construction.
    auto_doc: If True, the descriptions of argument fields will be inherited
      from `factory_fn` docstr if they are not explicitly specified through
      ``args``.
    auto_typing: If True, the value spec for constraining each argument will be
      inferred from its annotation. Otherwise the value specs for all arguments
      will be ``pg.typing.Any()``.
    serialization_key: An optional string to be used as the serialization key
      for the class during `sym_jsonify`. If None, `cls.type_name` will be used.
      This is introduced for scenarios when we want to relocate a class, before
      the downstream can recognize the new location, we need the class to
      serialize it using previous key.
    additional_keys: An optional list of strings as additional keys to
      deserialize an object of the registered class. This can be useful when we
      need to relocate or rename the registered class while being able to load
      existing serialized JSON values.
    add_to_registry: If True, the newly created functor class will be added to
      the registry for deserialization.

  Returns:
    A callable that converts a factory function into a subclass of the base
      class.
  """

  if not inspect.isfunction(factory_fn):
    raise TypeError('Decorator `compound` is only applicable to functions.')

  schema = schema_utils.function_schema(
      factory_fn,
      args=args,
      returns=pg_typing.Object(base_class) if base_class else None,
      auto_doc=auto_doc,
      auto_typing=auto_typing,
  )

  # Inference the base_class from schema.
  return_spec = schema.metadata.get('returns', None)
  if isinstance(return_spec, pg_typing.Object):
    base_class = return_spec.cls
  else:
    raise ValueError(
        'Cannot inference the base class from return value annotation. '
        'Please either add an annotation for the return value or provide the '
        'value for the `base_class` argument.'
    )

  class _Compound(Compound, base_class):
    """The compound class bound to a factory function."""

    # Disable auto register so we can use function module and name
    # for registration later.
    auto_register = False

    # The compound class uses the function signature to decide its
    # schema, thus we do not infer its schema from the class annotations.
    auto_schema = False

    def _on_bound(self):
      super()._on_bound()
      self._sym_decomposed = None

      if not lazy_build:
        # Trigger build.
        _ = self.decomposed

    @property
    def decomposed(self):
      if self._sym_decomposed is None:
        # Build the compound object.
        self._sym_decomposed = factory_fn(**self.sym_init_args)
      return self._sym_decomposed

    def __getattribute__(self, name: str):
      if (
          name.startswith('_')
          or name in _COMPOUND_OWNED_ATTR_NAMES
          or name in self.sym_init_args
      ):
        return super().__getattribute__(name)
      # Redirect attribute to the compound object.
      return getattr(self.decomposed, name)

  cls = _Compound
  cls.__name__ = factory_fn.__name__
  cls.__qualname__ = factory_fn.__qualname__
  cls.__module__ = factory_fn.__module__
  cls.__doc__ = factory_fn.__doc__

  # Enable automatic registration of subclass.
  cls.auto_register = True
  cls.apply_schema(schema)

  if add_to_registry:
    cls.register_for_deserialization(serialization_key, additional_keys)
  return cls


def compound(
    base_class: Optional[Type[Object]] = None,
    args: Optional[
        List[
            Union[
                Tuple[Tuple[str, pg_typing.KeySpec], pg_typing.ValueSpec, str],
                Tuple[
                    Tuple[str, pg_typing.KeySpec], pg_typing.ValueSpec, str, Any
                ],
            ]
        ]
    ] = None,  # pylint: disable=bad-continuation
    **kwargs
):
  """Function decorator to create compound class.

  Example::

    @dataclasses.dataclass
    class Foo:
      x: int
      y: int

      def sum(self):
        return self.x + self.y

    @pg.compound
    def foo_with_equal_x_y(v: int) -> Foo:
      return Foo(v, v)

    f = foo_with_equal_x_y(1)

    # First of all, the objects of compound classes can be used as an in-place
    # replacement for the objects of regular classes, as they are subclasses of
    # the regular classes.
    assert issubclass(foo_with_equal_x_y, Foo)
    assert isinstance(f, Foo)

    # We can access symbolic attributes of the compound object.
    assert f.v == 1

    # We can also access the public APIs of the decomposed object.
    assert f.x == 1
    assert f.y == 1
    assert f.sum() == 2     

    # Or explicit access the decomposed object.
    assert f.decomposed == Foo(1, 1)

    # Moreover, symbolic power is fully unleashed to the compound class.
    f.rebind(v=2)
    assert f.x == 2
    assert f.y == 2
    assert f.sum() == 4

    # Err with runtime type check: 2.5 is not an integer.
    f.rebind(v=2.5)

  Args:
    base_class: The base class of the compond class, which should be a
      ``pg.Object`` type. If None, it will be infererenced from the return
      annotation of `factory_fn`. If the annotation is not present or
      `auto_typing` is set to False, `base_class` must be present.
    args: Symbolic args specification. See :class:`pg.compound_class` for
      details.
    **kwargs: Keyword arguments. See :class:`pg.compound_class` for details.

  Returns:

    A symbolic compound class that subclasses `base_class`.
  """
  if inspect.isfunction(base_class):
    assert args is None
    return compound_class(base_class, add_to_registry=True, **kwargs)
  return lambda fn: compound_class(  # pylint: disable=g-long-lambda  # pytype: disable=wrong-arg-types
      fn, base_class, args, add_to_registry=True, **kwargs
  )
