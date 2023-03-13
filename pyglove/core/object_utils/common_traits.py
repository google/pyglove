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
"""Common traits for Python objects.

This file defines interfaces for describing the common traits of a Python
object, for example, serialization (JSONConvertible), partiality (MaybePartial),
formatting (Formattable), functor (Functor).
"""

import abc
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

# Nestable[T] is a (maybe) nested structure of T, which could be T, a Dict
# a List or a Tuple of Nestable[T]. We use a Union to fool PyType checker to
# make Nestable[T] a valid type annotation without type check.
T = TypeVar('T')
Nestable = Union[Any, T]  # pytype: disable=not-supported-yet

# pylint: disable=invalid-name
JSONPrimitiveType = Union[int, float, bool, str]

# pytype doesn't support recursion. Use Any instead of 'JSONValueType'
# in List and Dict.
JSONListType = List[Any]
JSONDictType = Dict[str, Any]
JSONValueType = Union[JSONPrimitiveType, JSONListType, JSONDictType]

# pylint: enable=invalid-name


class _TypeRegistry:
  """A registry for mapping a string name to type definition.

  This class is used for looking up type definition by a string identifier for
  deserialization.
  """

  def __init__(self):
    """Constructor."""
    # NOTE(daiyip): the order of keys in the dict is preserved. As a result,
    # in `pg.wrapping.apply_wrappers`, the latest registered wrapper
    # class will always be picked up when there are multiple wrapper classes
    # registered for a user class.
    self._type_to_cls_map = dict()

  def register(
      self, type_name: str, cls: Type[Any], override_existing: bool = False
      ) -> None:
    """Register a ``symbolic.Object`` class with a type name.

    Args:
      type_name: String identifier for the class, which will be used as the
        value of `_type` property when deciding which class to construct object
        when converting a JSON value to object.
      cls: Class to register.
      override_existing: Whether allow to override existing value if type name
        is already registered.

    Raises:
      KeyError: If type_name is already registered and override_existing is set
        to False.
    """
    if type_name in self._type_to_cls_map and not override_existing:
      raise KeyError(
          f'Type {type_name!r} has already been registered with class '
          f'{self._type_to_cls_map[type_name].__name__}.')
    self._type_to_cls_map[type_name] = cls

  def is_registered(self, type_name: str) -> bool:
    """Returns whether a type name is registered."""
    return type_name in self._type_to_cls_map

  def class_from_typename(
      self, type_name: str) -> Optional[Type[Any]]:
    """Get class from type name."""
    return self._type_to_cls_map.get(type_name, None)

  def iteritems(self) -> Iterable[Tuple[str, Type[Any]]]:
    """Iterate type registry."""
    return self._type_to_cls_map.items()


class JSONConvertible(metaclass=abc.ABCMeta):
  """Interface for classes whose instances are convertible from/to JSON.

  A JSON convertible object is an object that can be converted into plain Python
  objects, hence can be serialized into or deserialized from JSON.

  Subclasses of ``JSONConvertible`` should implement:

    * ``to_json``: A method that returns a plain Python dict with a `_type`
      property whose value should identify the class.
    * ``from_json``: A class method that takes a plain Python dict and returns
      an instance of the class.

  Example::

    class MyObject(pg.JSONConvertible):

      def __init__(self, x: int):
        self.x = x

      def to_json(self, **kwargs):
        return {
          '_type': 'MyObject',
          'x': self.x
        }

      @classmethod
      def from_json(cls, json_value, **kwargs):
        return cls(json_value['x'])

  All symbolic types (see :class:`pyglove.Symbolic`) are JSON convertible.
  """

  # Registry for looking up the type definition for a string identifier during
  # deserialization. One key can be used for only one type, while the same type
  # can be registered with many different string identifiers, which can be
  # useful to allow backward compatibility of existing serialized strings.
  _TYPE_REGISTRY = _TypeRegistry()

  # Key in serialized dict that represents the class to restore.
  TYPE_NAME_KEY = '_type'

  # Marker (as the first element of a list) for serializing tuples.
  TUPLE_MARKER = '__tuple__'

  # Type converter that converts a complex type to basic JSON value type.
  # When this field is set by users, the converter will be invoked when a
  # complex value cannot be serialized by existing methods.
  TYPE_CONVERTER: Optional[
      Callable[[Type[Any]], Callable[[Any], JSONValueType]]] = None

  # Class property that indicates whether to automatically register class
  # for deserialization. Subclass can override.
  auto_register = True

  @classmethod
  def from_json(cls, json_value: JSONValueType, **kwargs) -> 'JSONConvertible':
    """Creates an instance of this class from a plain Python value.

    NOTE(daiyip): ``pg.Symbolic`` overrides ``from_json`` class method.

    Args:
      json_value: JSON value type.
      **kwargs: Keyword arguments as flags to control object creation.

    Returns:
      An instance of cls.
    """
    del kwargs
    assert isinstance(json_value, dict)
    init_args = {k: from_json(v) for k, v in json_value.items()
                 if k != JSONConvertible.TYPE_NAME_KEY}
    return cls(**init_args)

  @abc.abstractmethod
  def to_json(self, **kwargs) -> JSONValueType:
    """Returns a plain Python value as a representation for this object.

    A plain Python value are basic python types that can be serialized into
    JSON, e.g: ``bool``, ``int``, ``float``, ``str``, ``dict`` (with string
    keys), ``list``, ``tuple`` where the container types should have plain
    Python values as their values.

    Args:
      **kwargs: Keyword arguments as flags to control JSON conversion.

    Returns:
      A plain Python value.
    """

  @classmethod
  def register(
      cls,
      type_name: str,
      subclass: Type['JSONConvertible'],
      override_existing: bool = False
      ) -> None:
    """Registers a class with a type name.

    The type name will be used as the key for class lookup during
    deserialization. A class can be registered with multiple type names, but
    a type name should be uesd only for one class.

    Args:
      type_name: A global unique string identifier for subclass.
      subclass: A subclass of JSONConvertible.
      override_existing: If True, registering an type name is allowed.
        Otherwise an error will be raised.
    """
    cls._TYPE_REGISTRY.register(type_name, subclass, override_existing)

  @classmethod
  def is_registered(cls, type_name: str) -> bool:
    """Returns True if a type name is registered. Otherwise False."""
    return cls._TYPE_REGISTRY.is_registered(type_name)

  @classmethod
  def class_from_typename(
      cls, type_name: str) -> Optional[Type['JSONConvertible']]:
    """Gets the class for a registered type name.

    Args:
      type_name: A string as the global unique type identifier for requested
        class.

    Returns:
      A type object if registered, otherwise None.
    """
    return cls._TYPE_REGISTRY.class_from_typename(type_name)

  @classmethod
  def registered_types(cls) -> Iterable[Tuple[str, Type['JSONConvertible']]]:
    """Returns an iterator of registered (serialization key, class) tuples."""
    return cls._TYPE_REGISTRY.iteritems()

  @classmethod
  def to_json_dict(
      cls,
      fields: Dict[str, Any],
      **kwargs) -> Dict[str, JSONValueType]:
    """Helper method to create JSON dict from class and field."""
    type_name = f'{cls.__module__}.{cls.__name__}'
    json_dict = {JSONConvertible.TYPE_NAME_KEY: type_name}
    json_dict.update({k: to_json(v, **kwargs) for k, v in fields.items()})
    return json_dict

  @classmethod
  def __init_subclass__(cls):
    super().__init_subclass__()
    if not inspect.isabstract(cls) and cls.auto_register:
      type_name = f'{cls.__module__}.{cls.__name__}'
      JSONConvertible.register(type_name, cls, override_existing=True)


def registered_types() -> Iterable[Tuple[str, Type[JSONConvertible]]]:
  """Returns an iterator of registered (serialization key, class) tuples."""
  return JSONConvertible.registered_types()


def to_json(value: Any, **kwargs) -> Any:
  """Serializes a (maybe) JSONConvertible value into a plain Python object.

  Args:
    value: value to serialize. Applicable value types are:

      * Builtin python types: None, bool, int, float, string;
      * JSONConvertible types;
      * List types;
      * Tuple types;
      * Dict types.

    **kwargs: Keyword arguments to pass to value.to_json if value is
      JSONConvertible.

  Returns:
    JSON value.
  """
  if isinstance(value, (type(None), bool, int, float, str)):
    return value
  elif isinstance(value, JSONConvertible):
    return value.to_json(**kwargs)
  elif isinstance(value, tuple):
    return [JSONConvertible.TUPLE_MARKER] + to_json(list(value), **kwargs)
  elif isinstance(value, list):
    return [to_json(item, **kwargs) for item in value]
  elif isinstance(value, dict):
    return {k: to_json(v, **kwargs) for k, v in value.items()}
  else:
    if JSONConvertible.TYPE_CONVERTER is not None:
      converter = JSONConvertible.TYPE_CONVERTER(type(value))   # pylint: disable=not-callable
      if converter:
        return to_json(converter(value))
    raise ValueError(f'Cannot convert complex type {value} to JSON.')


def from_json(json_value: JSONValueType) -> Any:
  """Deserializes a (maybe) JSONConvertible value from JSON value.

  Args:
    json_value: Input JSON value.

  Returns:
    Deserialized value.
  """
  if isinstance(json_value, list):
    if json_value and json_value[0] == JSONConvertible.TUPLE_MARKER:
      if len(json_value) < 2:
        raise ValueError(
            f'Tuple should have at least one element '
            f'besides \'{JSONConvertible.TUPLE_MARKER}\'. '
            f'Encountered: {json_value}.')
      return tuple([from_json(v) for v in json_value[1:]])
    return [from_json(v) for v in json_value]
  elif isinstance(json_value, dict):
    if JSONConvertible.TYPE_NAME_KEY not in json_value:
      return {k: from_json(v) for k, v in json_value.items()}
    type_name = json_value[JSONConvertible.TYPE_NAME_KEY]
    cls = JSONConvertible.class_from_typename(type_name)
    if cls is None:
      raise TypeError(
          f'Type name \'{type_name}\' is not registered '
          f'with a `pg.JSONConvertible` subclass.')
    return cls.from_json(json_value)
  return json_value


class Formattable(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be pretty-formatted.

  This interface overrides the default ``__repr__`` and ``__str__`` method, thus
  all ``Formattable`` objects can be printed nicely.

  All symbolic types implement this interface.
  """

  @abc.abstractmethod
  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> str:
    """Formats this object into a string representation.

    Args:
      compact: If True, this object will be formatted into a single line.
      verbose: If True, this object will be formatted with verbosity.
        Subclasses should define `verbosity` on their own.
      root_indent: The start indent level for this object if the output is a
        multi-line string.
      **kwargs: Subclass specific keyword arguments.

    Returns:
      A string of formatted object.
    """

  def __str__(self) -> str:
    """Returns the full (maybe multi-line) representation of this object."""
    return self.format(compact=False, verbose=True)

  def __repr__(self) -> str:
    """Returns a single-line representation of this object."""
    return self.format(compact=True)


class MaybePartial(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be partially constructed.

  A ``MaybePartial`` object is an object whose ``__init__`` method can accept
  ``pg.MISSING_VALUE`` as its argument values. All symbolic types (see
  :class:`pyglove.Symbolic`) implements this interface, as their symbolic
  attributes can be partially filled.

  Example::

    d = pg.Dict(x=pg.MISSING_VALUE, y=1)
    assert d.is_partial
    assert 'x' in d.missing_values()
  """

  @property
  def is_partial(self) -> bool:
    """Returns True if this object is partial. Otherwise False.

    An object is considered partial when any of its required fields is missing,
    or at least one member is partial. The subclass can override this method
    to provide a more efficient solution.
    """
    return len(self.missing_values()) > 0  # pylint: disable=g-explicit-length-test

  @abc.abstractmethod
  def missing_values(self, flatten: bool = True) -> Dict[str, Any]:  # pylint: disable=redefined-outer-name
    """Returns missing values from this object.

    Args:
      flatten: If True, convert nested structures into a flattened dict using
        key path (delimited by '.' and '[]') as key.

    Returns:
      A dict of key to MISSING_VALUE.
    """


class Functor(metaclass=abc.ABCMeta):
  """Interface for functor."""

  @abc.abstractmethod
  def __call__(self, *args, **kwargs) -> Any:
    """Calls the functor.

    Args:
      *args: Any positional arguments.
      **kwargs: Any keyword arguments.

    Returns:
      Any value.
    """
