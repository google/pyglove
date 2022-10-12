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
"""Utility library for handling hierarchical Python objects.

Overview
--------

``pg.object_utils`` facilitates the handling of hierarchical
Python objects. It sits at the bottom of all PyGlove modules and empowers other
modules with the following features:

  +---------------------+----------------------------------------------+
  | Functionality       | API                                          |
  +=====================+==============================================+
  | Formatting          | :class:`pyglove.Formattable`,                |
  |                     |                                              |
  |                     | :func:`pyglove.format`,                      |
  |                     |                                              |
  |                     | :func:`pyglove.print`,                       |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.kvlist_str`,     |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.quote_if_str`,   |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.message_on_path` |
  +---------------------+----------------------------------------------+
  | Serialization       | :class:`pyglove.JSONConvertible`             |
  +---------------------+----------------------------------------------+
  | Partial construction| :class:`pyglove.MaybePartial`,               |
  |                     |                                              |
  |                     | :const:`pyglove.MISSING_VALUE`               |
  +---------------------+----------------------------------------------+
  | Hierarchical key    | :class:`pyglove.KeyPath`                     |
  | representation      |                                              |
  +---------------------+----------------------------------------------+
  | Hierarchical object | :func:`pyglove.object_utils.traverse`        |
  | traversal           |                                              |
  +---------------------+----------------------------------------------+
  | Hierarchical object | :func:`pyglove.object_utils.transform`,      |
  | transformation      |                                              |
  |                     | :func:`pyglove.object_utils.merge`,          |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.canonicalize`,   |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.flatten`         |
  +---------------------+----------------------------------------------+
"""

import abc
import copy
import enum
import operator
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Text, Tuple, Type, TypeVar, Union

# Nestable[T] is a (maybe) nested structure of T, which could be T, a Dict
# a List or a Tuple of Nestable[T]. We use a Union to fool PyType checker to
# make Nestable[T] a valid type annotation without type check.
T = TypeVar('T')
Nestable = Union[Any, T]  # pytype: disable=not-supported-yet

# pylint: disable=invalid-name
JSONPrimitiveType = Union[int, float, bool, Text]

# pytype doesn't support recursion. Use Any instead of 'JSONValueType'
# in List and Dict.
JSONListType = List[Any]
JSONDictType = Dict[Text, Any]
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
      self, type_name: Text, cls: Type[Any], override_existing: bool = False
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

  def is_registered(self, type_name: Text) -> bool:
    """Returns whether a type name is registered."""
    return type_name in self._type_to_cls_map

  def class_from_typename(
      self, type_name: Text) -> Optional[Type[Any]]:
    """Get class from type name."""
    return self._type_to_cls_map.get(type_name, None)

  def iteritems(self) -> Iterable[Tuple[Text, Type[Any]]]:
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

  @classmethod
  def from_json(cls, json_value: JSONValueType, **kwargs) -> 'JSONConvertible':
    """Creates an instance of this class from a plain Python value.

    Args:
      json_value: JSON value type.
      **kwargs: Keyword arguments as flags to control object creation.

    Returns:
      An instance of cls.
    """
    raise NotImplementedError('Subclass should override this method.')

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
      type_name: Text,
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
  def is_registered(cls, type_name: Text) -> bool:
    """Returns True if a type name is registered. Otherwise False."""
    return cls._TYPE_REGISTRY.is_registered(type_name)

  @classmethod
  def class_from_typename(
      cls, type_name: Text) -> Optional[Type['JSONConvertible']]:
    """Gets the class for a registered type name.

    Args:
      type_name: A string as the global unique type identifier for requested
        class.

    Returns:
      A type object if registered, otherwise None.
    """
    return cls._TYPE_REGISTRY.class_from_typename(type_name)

  @classmethod
  def registered_types(cls) -> Iterable[Tuple[Text, Type['JSONConvertible']]]:
    """Returns an iterator of registered (type name, class) tuples."""
    return cls._TYPE_REGISTRY.iteritems()


class MaybePartial(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be partially constructed.

  A ``MaybePartial`` object is an object whose ``__init__`` method can accept
  ``pg.MISSING_VALUE`` as its arguemnt values. All symbolic types (see
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
  def missing_values(self, flatten: bool = True) -> Dict[Text, Any]:  # pylint: disable=redefined-outer-name
    """Returns missing values from this object.

    Args:
      flatten: If True, convert nested structures into a flattened dict using
        key path (delimited by '.' and '[]') as key.

    Returns:
      A dict of key to MISSING_VALUE.
    """


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
             **kwargs) -> Text:
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

  def __str__(self) -> Text:
    """Returns the full (maybe multi-line) representation of this object."""
    return self.format(compact=False, verbose=True)

  def __repr__(self) -> Text:
    """Returns a single-line representation of this object."""
    return self.format(compact=True)


class Functor(metaclass=abc.ABCMeta):
  """Interface for functor."""

  # `schema.Signature` object for this functor class.
  signature = None

  @abc.abstractmethod
  def __call__(self, *args, **kwargs) -> Any:
    """Calls the functor.

    Args:
      *args: Any positional arguments.
      **kwargs: Any keyword arguments.

    Returns:
      Any value.
    """


class MissingValue(Formattable):
  """Value placeholder for an unassigned attribute."""

  def format(self, **kwargs):
    return 'MISSING_VALUE'

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, MissingValue)

  def __hash__(self) -> int:
    return hash(MissingValue.__module__ + MissingValue.__name__)


# A shortcut global object (constant) for referencing MissingValue.
MISSING_VALUE = MissingValue()


class StrKey(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be treated as str in ``KeyPath``.

  A :class:`pyglove.KeyPath` will format the path string using ``.`` (dot) as
  the delimiter for a key represented by this object. Otherwise ``[]`` (square
  brackets) will be used as the delimiters.

  Example::

    class MyKey(pg.object_utils.StrKey):

      def __init__(self, name):
        self.name = name

      def __str__(self):
        return f'__{self.name}__'

    path = pg.KeyPath(['a', MyKey('b')])
    print(str(path))   # Should print "a.__b__"

  """


class KeyPath(Formattable):
  """Represents a path of keys from the root to a node in a tree.

  ``KeyPath`` is an important concept in PyGlove, which is used for representing
  a symbolic object's location (see :meth:`pyglove.Symbolic.sym_path`) within
  its symbolic tree. For example::

    @pg.members([
        ('x', pg.typing.Int()),
        ('y', pg.typing.Str())
    ])
    class A(pg.Object):
      pass

    @pg.members([
        ('z', pg.typing.Object(A))
    ])
    class B(pg.Object):
      pass

    a = A(x=1, y='foo')
    b = B(z=a)
    assert a.sym_path == 'z' # The path to object `a` is 'z'.
    assert b.sym_path == ''  # The root object's KeyPath is empty.

  Since each node in a tree has a unique location, given the root we shall be
  able to use a ``KeyPath`` object to locate the node. With the example
  above, we can query the member ``x`` of object ``a`` via::

    pg.KeyPath.parse('z.x').query(b)  # Should return 1.

  Similarly, we can modify a symbolic object's sub-node based on a ``KeyPath``
  object. See :meth:`pyglove.Symbolic.rebind` for modifying sub-nodes in a
  symbolic tree.
  """

  def __init__(self,
               key_or_key_list: Optional[Union[Any, List[Any]]] = None,
               parent: Optional['KeyPath'] = None):
    """Constructor.

    Args:
      key_or_key_list: A single object as key, or a list/tuple of objects
        as keys in the path.
        When string types or StrKey objects are used as key, dot ('.') is used
        as the delimiter, otherwise square brackets ('[]') is used as the
        delimiter when formatting a KeyPath.
        For object type key, str(object) will be used to represent the key in
        string form.
      parent: Parent KeyPath.
    """
    if key_or_key_list is None:
      key_or_key_list = []
    elif not isinstance(key_or_key_list, (tuple, list)):
      key_or_key_list = [key_or_key_list]

    keys = []
    if parent:
      keys.extend(parent.keys)
    keys.extend(key_or_key_list)
    self._keys = keys
    # NOTE(daiyip): Lazy to build path string cache for fast access.
    self._path_str = None

  @classmethod
  def from_value(cls, value: Union['KeyPath', Text, int]) -> 'KeyPath':
    """Returns a KeyPath object from a KeyPath equivalence."""
    if isinstance(value, str):
      value = cls.parse(value)
    elif isinstance(value, int):
      value = cls(value)
    elif not isinstance(value, KeyPath):
      raise ValueError(f'{value!r} is not a valid KeyPath equivalence.')
    return value

  @classmethod
  def parse(cls,
            path_str: Text,
            parent: Optional['KeyPath'] = None) -> 'KeyPath':
    """Creates a ``KeyPath`` object from parsing a JSONPath-like string.

    The JSONPath (https://restfulapi.net/json-jsonpath/) like string is defined
    as following::

      <path>        := <empty> | {<dict-key>[.<dict-key>]*}
      <dict-key>     := <identifier>['[('<list-key>|<special-dict-key>)']']*
      <list-key>    := <number>
      <special-dict-key> := <string-with-delimiter-chars>
      <delimiter_chars> := '[' | ']' | '.'

    For example, following keys are valid path strings::

      ''               : An empty path representing the root of a path.
      'a'              : A path that contains a dict key 'a'.
      'a.b'            : A path that contains two dict keys 'a' and 'b'.
      'a[0]'           : A path that contains a dict key 'a' and a list key 0.
      'a.0.'           : A path that contains two dict keys 'a' and '0'.
      'a[0][1]'        : A path that contains a dict key 'a' and two list keys
                         0 and 1 for a multi-dimension list.
      'a[x.y].b'       : A path that contains three dict keys: 'a', 'x.y', 'b'.
                         Since 'x.y' has delimiter characters, it needs to be
                         enclosed in brackets.

    TODO(daiyip): Support paring ``KeyPath`` from keys of complex types.
    Now this method only supports parsing KeyPath of string and int keys.
    That being said, ``format``/``parse`` are not symmetric, while ``format``
    can convert a ``KeyPath`` that includes complex keys into a string,
    ``parse`` is not able to convert them back.

    Args:
      path_str: A JSON-path-like string.
      parent: Parent KeyPath object.

    Returns:
      A KeyPath object.

    Raises:
      ValueError: Path string is in bad format.
    """
    if not isinstance(path_str, str):
      raise ValueError(
          f'\'path_str\' must be a string type. Encountered: {path_str!r}')

    keys = []
    def _append_key(key, preserve_empty=False, maybe_numeric=False):
      """Helper method to append key."""
      if not (preserve_empty or key):
        return
      if maybe_numeric and key.lstrip('-').isdigit():
        key = int(key)
      keys.append(key)

    pos, key_start, unmatched_brackets = 0, 0, 0
    while pos != len(path_str):
      ch = path_str[pos]
      if ch == ']':
        unmatched_brackets -= 1
        if unmatched_brackets == 0:
          key = path_str[key_start:pos]
          _append_key(key, True, True)
          key_start = pos + 1
        elif unmatched_brackets < 0:
          raise ValueError(
              f'KeyPath parse failed: unmatched close bracket at position '
              f'{pos}:{path_str!r}')
      elif ch == '[':
        if unmatched_brackets == 0:
          key = path_str[key_start:pos]
          _append_key(key)
          key_start = pos + 1
        unmatched_brackets += 1
      elif ch == '.' and unmatched_brackets == 0:
        key = path_str[key_start:pos]
        _append_key(key)
        key_start = pos + 1
      pos += 1
    if key_start != len(path_str):
      _append_key(path_str[key_start:])
    if unmatched_brackets != 0:
      raise ValueError(
          f'KeyPath parse failed: unmatched open bracket at position '
          f'{key_start - 1}: {path_str!r}')
    return KeyPath(keys, parent)

  @property
  def keys(self) -> List[Any]:
    """A list of keys in this path."""
    return copy.copy(self._keys)

  @property
  def key(self) -> Any:
    """The rightmost key of this path."""
    if self.depth == 0:
      raise KeyError('Key of root KeyPath does not exist.')
    return self._keys[-1]

  @property
  def is_root(self) -> bool:
    """Returns True if this path is the root of a tree."""
    return not self._keys

  @property
  def depth(self) -> int:
    """The depth of this path."""
    return len(self._keys)

  @property
  def parent(self) -> 'KeyPath':
    """The ``KeyPath`` object for current node's parent.

    Example::

      path = pg.KeyPath.parse('a.b.c.')
      assert path.parent == 'a.b'

    Returns:
      A ``KeyPath`` object for the parent of current node.

    Raises:
      KeyError: If current path is the root.
    """
    if self.is_root:
      raise KeyError('Parent of a root KeyPath does not exist.')
    return KeyPath(self._keys[:-1])

  def __sub__(self, other: Union[None, int, Text, 'KeyPath']) -> 'KeyPath':
    """Finds the relative path of this path to the other.

    Example::

      path1 = pg.KeyPath.parse('a.b.c.d')
      path2 = pg.KeyPath.parse('a.b')
      assert path1 - path2 == 'c.d'

    Args:
      other: Object to subtract, which can be None, int (as a depth-1 KeyPath),
        string (parsed as a KeyPath) or a KeyPath object.

    Returns:
      Relative path of this path to the other.

    Raises:
      ValueError: This path is an ancestor node of the other path,
        or these two paths are in different branch.
    """
    if other is None:
      return self
    if isinstance(other, str):
      other = KeyPath.parse(other)
    elif isinstance(other, int):
      other = KeyPath(other)
    if not isinstance(other, KeyPath):
      raise TypeError(
          f'Cannot subtract KeyPath({self}) by {other!r}.')
    max_len = max(len(self), len(other))
    for pos in range(max_len):
      if pos >= len(self):
        raise ValueError(
            f'KeyPath subtraction failed: left path {self!r} '
            f'is an ancestor of right path {other!r}.')
      if pos >= len(other):
        return KeyPath(self.keys[pos:])

      if self.keys[pos] != other.keys[pos]:
        raise ValueError(
            f'KeyPath subtraction failed: left path {self!r} '
            f'and right path {other!r} are in different subtree.')
    return KeyPath()

  def __add__(self, other: Any) -> 'KeyPath':
    """Concatenates a KeyPath equivalent object.

    Args:
      other: Object to add, which can be None, int (as a 1-level KeyPath),
        string (parsed as a KeyPath), a KeyPath object, or any other object as
        a single key.

    Returns:
      Newly concatenated KeyPath.

    Raises:
      ValueError: If other is a string that cannot be parsed into a KeyPath.
    """
    if other is None:
      return self
    if isinstance(other, str):
      other = KeyPath.parse(other)
    elif not isinstance(other, KeyPath):
      other = KeyPath(other)
    assert isinstance(other, KeyPath)
    return KeyPath(other.keys, self)

  def query(self, src: Any) -> Any:
    """Query the value from the source object based on current path.

    Example::

      @pg.members([
          ('x', pg.typing.Int()),
          ('y', pg.typing.Str())
      ])
      class A(pg.Object):
        pass

      @pg.members([
          ('z', pg.typing.Object(A))
      ])
      class B(pg.Object):
        pass

      b = B(z=A(x=1, y='foo'))
      assert pg.KeyPath.parse('z.x').query(b) == 1

    Args:
      src: Source value to query.

    Returns:
      Value from src if path exists.

    Raises:
      KeyError: Path doesn't exist in src.
      RuntimeError: Called on a KeyPath that is considered as removed.
    """
    return self._query(0, src)

  def _query(self, key_pos: int, src: Any) -> Any:
    """Query the value of current path up to key_pos from an object.

    Args:
      key_pos: Start position in self._keys.
      src: Source value to query.

    Returns:
      Value from src if path exists.

    Raises:
      KeyError: Path doesn't exist in src.
    """
    if key_pos == len(self._keys):
      return src
    key = self.keys[key_pos]
    if hasattr(src, 'sym_getattr'):
      assert hasattr(src, 'sym_hasattr')
      if src.sym_hasattr(key):
        return self._query(key_pos + 1, src.sym_getattr(key))
    elif hasattr(src, '__getitem__'):
      if isinstance(key, int):
        if not hasattr(src, '__len__'):
          raise KeyError(
              f'Cannot query index ({key}) on object ({src!r}): '
              f'\'__len__\' does not exist.')
        if key < len(src):
          return self._query(key_pos + 1, src[key])
      else:
        if not hasattr(src, '__contains__'):
          raise KeyError(
              f'Cannot query key ({key!r}) on object ({src!r}): '
              f'\'__contains__\' does not exist.')
        if key in src:
          return self._query(key_pos + 1, src[key])
    else:
      raise KeyError(
          f'Cannot query sub-key {key!r} of object ({src!r}): '
          f'\'__getitem__\' does not exist. '
          f'(path={KeyPath(self.keys[:key_pos])})')
    raise KeyError(
        f'Path {KeyPath(self._keys[:key_pos + 1])!r} does not exist: '
        f'key {key!r} is absent from innermost value {src!r}.')

  def _has_special_chars(self, key):
    """Returns True if key has special characters."""
    return any([c in key for c in ['[', ']', '.']])

  def get(self, src: Any, default_value: Optional[Any] = None) -> Any:
    """Gets the value of current path from an object with a default value."""
    try:
      return self.query(src)
    except KeyError:
      return default_value

  def exists(self, src: Any) -> bool:
    """Returns whether current path exists in source object."""
    try:
      self.query(src)
      return True
    except KeyError:
      return False

  @property
  def path(self) -> Text:
    """JSONPath representation of current path."""
    if self._path_str is None:
      self._path_str = self.path_str()
    return self._path_str

  def path_str(self, preserve_complex_keys: bool = True) -> Text:
    """Returns JSONPath representation of current path.

    Args:
      preserve_complex_keys: if True, complex keys such as 'x.y' will be
      preserved by quoted in brackets.

      For example: KeyPath(['a', 'x.y', 'b']) will return 'a[x.y].b' when
      `preserve_complex_keys` is True, and `a.x.y.b` when
      `preserve_complex_keys` is False.

    Returns:
      Path string.
    """
    s = []
    for i, key in enumerate(self._keys):
      if ((isinstance(key, str)
           and not (preserve_complex_keys and self._has_special_chars(key)))
          or isinstance(key, StrKey)):
        if i != 0:
          s.append('.')
        s.append(str(key))
      else:
        s.append(f'[{key}]')
    return ''.join(s)

  def __len__(self) -> int:
    """Use depth as length of current path."""
    return self.depth

  def format(self, **kwargs):
    """Format current path."""
    return self.path

  def __hash__(self):
    """Hash function.

    Returns:
      return the hash value of its path.
    NOTE(daiyip): KeyPath shares the same hash of its JSONPath representation
    (relative form), thus we can lookup a dict with KeyPath key by string,
    and vice versa.
    """
    return hash(self.path)

  def __eq__(self, other: Any) -> bool:
    """Equality check.

    Args:
      other: A string or a KeyPath.

    Returns:
      Whether JSON-path representation (either absolute or relative form)
        of current path equals to other.
    """
    if isinstance(other, str):
      return self.path == other
    return isinstance(other, KeyPath) and self.keys == other.keys

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def __lt__(self, other: Any) -> bool:
    return self._compare(other, operator.lt)

  def __le__(self, other: Any) -> bool:
    return self._compare(other, operator.le)

  def __gt__(self, other: Any) -> bool:
    return self._compare(other, operator.gt)

  def __ge__(self, other: Any) -> bool:
    return self._compare(other, operator.ge)

  def _compare(
      self,
      other: Any,
      comparison: Callable[[Any, Any], bool]
      ) -> bool:
    """Compare to another KeyPath or a string.

    Args:
      other: A Keypath or a string.
      comparison: A comparison operator.

    Returns:
      Whether or not the comparison holds true.

    Raises:
      TypeError: The other object is neither a Keypath nor a string.
    """
    if isinstance(other, str):
      return comparison(self.path, other)
    if isinstance(other, KeyPath):
      return comparison(
          tuple(map(KeyPath._KeyComparisonWrapper, self.keys)),
          tuple(map(KeyPath._KeyComparisonWrapper, other.keys))
      )
    raise TypeError(
        f'Comparison is not supported between instances of \'KeyPath\' and '
        f'{type(other).__name__!r}.')

  class _KeyComparisonWrapper:
    """A wrapper around KeyPath keys enabling dynamic comparison."""

    def __init__(self, key: Any):
      self.key = key

    def __eq__(self, other: 'KeyPath._KeyComparisonWrapper') -> bool:
      return self._compare(other, operator.eq)

    def __ne__(self, other: 'KeyPath._KeyComparisonWrapper') -> bool:
      return self._compare(other, operator.ne)

    def __lt__(self, other: 'KeyPath._KeyComparisonWrapper') -> bool:
      return self._compare(other, operator.lt)

    def __le__(self, other: 'KeyPath._KeyComparisonWrapper') -> bool:
      return self._compare(other, operator.le)

    def __gt__(self, other: 'KeyPath._KeyComparisonWrapper') -> bool:
      return self._compare(other, operator.gt)

    def __ge__(self, other: 'KeyPath._KeyComparisonWrapper') -> bool:
      return self._compare(other, operator.ge)

    def _compare(
        self,
        other: 'KeyPath._KeyComparisonWrapper',
        comparison: Callable[[Any, Any], bool]
        ) -> bool:
      """Compare the key against another key from a different KeyPath."""
      is_int = lambda key: isinstance(key, int)
      is_str = lambda key: isinstance(key, str)
      is_int_or_str = lambda key: is_int(key) or is_str(key)
      if is_int(self.key) and is_int(other.key):
        # Both are ints. Compare numerically so that KeyPath(2) < KeyPath(10).
        return comparison(self.key, other.key)
      if is_int_or_str(self.key) and is_int_or_str(other.key):
        # One is a str; the other is an int or str. Compare lexicographically.
        return comparison(str(self.key), str(other.key))
      # One or both is a custom key. Delegate comparison to its magic methods.
      return comparison(self.key, other.key)


#
# Helper methods to traverse, transform and merge JSON values.
#


def traverse(value: Any,
             preorder_visitor_fn: Optional[Callable[[KeyPath, Any],
                                                    bool]] = None,
             postorder_visitor_fn: Optional[Callable[[KeyPath, Any],
                                                     bool]] = None,
             root_path: Optional[KeyPath] = None) -> bool:
  """Traverse a (maybe) hierarchical value.

  Example::

    def preorder_visit(path, value):
      print(path)

    tree = {'a': [{'c': [1, 2]}, {'d': {'g': (3, 4)}}], 'b': 'foo'}
    pg.object_utils.traverse(tree, preorder_visit)

    # Should print:
    # 'a'
    # 'a[0]'
    # 'a[0].c'
    # 'a[0].c[0]'
    # 'a[0].c[1]'
    # 'a[1]'
    # 'a[1].d'
    # 'a[1].d.g'
    # 'b'

  Args:
    value: A maybe hierarchical value to traverse.
    preorder_visitor_fn: Preorder visitor function.
      Function signature is (path, value) -> should_continue.
    postorder_visitor_fn: Postorder visitor function.
      Function signature is (path, value) -> should_continue.
    root_path: The key path of the root value.

  Returns:
    Whether visitor function returns True on all nodes.
  """
  root_path = root_path or KeyPath()
  def no_op_visitor(key, value):
    del key, value
    return True

  if preorder_visitor_fn is None:
    preorder_visitor_fn = no_op_visitor
  if postorder_visitor_fn is None:
    postorder_visitor_fn = no_op_visitor

  if not preorder_visitor_fn(root_path, value):
    return False
  if isinstance(value, dict):
    for k in value.keys():
      if not traverse(
          value[k], preorder_visitor_fn, postorder_visitor_fn,
          KeyPath(k, root_path)):
        return False
  elif isinstance(value, list):
    for i, v in enumerate(value):
      if not traverse(
          v, preorder_visitor_fn, postorder_visitor_fn, KeyPath(i, root_path)):
        return False
  if not postorder_visitor_fn(root_path, value):
    return False
  return True


def transform(value: Any,
              transform_fn: Callable[[KeyPath, Any], Any],
              root_path: Optional[KeyPath] = None,
              inplace: bool = True) -> Any:
  """Bottom-up (post-order) transform a (maybe) hierarchical value.

  Transform on value is in-place unless `transform_fn` returns a different
  instance.

  Example::

    def _remove_int(path, value):
      del path
      if isinstance(value, int):
        return pg.MISSING_VALUE
      return value

    inputs = {
        'a': {
            'b': 1,
            'c': [1, 'bar', 2, 3],
            'd': 'foo'
        },
        'e': 'bar',
        'f': 4
    }
    output = pg.object_utils.transform(inputs, _remove_int)
    assert output == {
        'a': {
            'c': ['bar'],
            'd': 'foo',
        },
        'e': 'bar'
    })

  Args:
    value: Any python value type. If value is a list of dict, transformation
      will occur recursively.
    transform_fn: Transform function in signature
      (path, value) -> new value
      If new value is MISSING_VALUE, key will be deleted.
    root_path: KeyPath of the root.
    inplace: If True, perform transformation in place.
  Returns:
    Transformed value.
  """
  def _transform(value: Any, current_path: KeyPath) -> Any:
    """Implementation of transform function."""
    new_value = value
    if isinstance(value, dict):
      if not inplace:
        new_value = value.__class__()
      deleted_keys = []
      for k, v in value.items():
        nv = _transform(v, KeyPath(k, current_path))
        if MISSING_VALUE != nv:
          if not inplace or value[k] is not nv:
            new_value[k] = nv
        elif inplace:
          deleted_keys.append(k)

      for k in deleted_keys:
        del value[k]
    elif isinstance(value, list):
      deleted_indices = []
      if not inplace:
        new_value = value.__class__()
      for i, v in enumerate(value):
        nv = _transform(v, KeyPath(i, current_path))
        if MISSING_VALUE != nv:
          if not inplace:
            new_value.append(nv)
          elif value[i] is not nv:
            value[i] = nv
        elif inplace:
          deleted_indices.append(i)
      for i in reversed(deleted_indices):
        del value[i]
    return transform_fn(current_path, new_value)
  return _transform(value, root_path or KeyPath())


def flatten(src: Any, flatten_complex_keys: bool = True) -> Any:
  """Flattens a (maybe) hierarchical value into a depth-1 dict.

  Example::

    inputs = {
        'a': {
            'e': 1,
            'f': [{
                'g': 2
            }, {
                'g[0]': 3
            }],
            'h': [],
            'i.j': {},
        },
        'b': 'hi',
        'c': None
    }
    output = pg.object_utils.flatten(inputs)
    assert output == {
        'a.e': 1,
        'a.f[0].g': 2,
        'a.f[1].g[0]': 3,
        'a.h': [],
        'a.i.j': {},
        'b': 'hi',
        'c': None
    }

  Args:
    src: source value to flatten.
    flatten_complex_keys: if True, complex keys such as 'x.y' will be flattened
    as 'x'.'y'. For example:
      {'a': {'b.c': 1}} will be flattened into {'a.b.c': 1} if this flag is on,
      otherwise it will be flattened as {'a[b.c]': 1}.

  Returns:
    For primitive value types, `src` itself will be returned.
    For list and dict types, an 1-depth dict will be returned.
    For tuple, a tuple of the same length, with each element flattened will be
    returned. The order of keys in nested ordered dict will be preserved,
    Keys of different depth are joined into a string using "." for dict
    properties and "[]" for list elements.
    For example, if src is::

      {
        "a": {
               "b": 4,
               "c": {"d": 10},
               "e": [1, 2]
               'f': [],
               'g.h': {},
             }
      }

    then the output dict is::

      {
        "a.b": 4,
        "a.c.d": 10,
        "a.e[0]": 1,
        "a.e[1]": 2,
        "a.f": [],
        "a.g.h": {},
      }

    when `flatten_complex_keys` is True, and::

      {
        "a.b": 4,
        "a.c.d": 10,
        "a.e[0]": 1,
        "a.e[1]": 2,
        "a.f": [],
        "a.[g.h]": {},
      }

    when `flatten_complex_keys` is False.

  Raises:
    ValueError: If any key from the nested dictionary contains ".".
  """
  # NOTE(daiyip): Comparing to list, tuple is treated as a single value,
  # whose index is not treated as key. That being said, there is no partial
  # update semantics on elements of tuple in `merge` method too.
  # Thus we simply flatten its elements and keep the tuple form.
  if isinstance(src, tuple):
    return tuple([flatten(elem) for elem in src])

  if not isinstance(src, (dict, list)) or not src:
    return src

  dest = dict()
  def _output_leaf(path: KeyPath, value: Any):
    if path and (not isinstance(value, (dict, list)) or not value):
      dest[path.path_str(not flatten_complex_keys)] = value
    return True
  traverse(src, postorder_visitor_fn=_output_leaf)
  return dest


def try_listify_dict_with_int_keys(
    src: Dict[Any, Any],
    convert_when_sparse: bool = False
    ) -> Tuple[Union[List[Any], Dict[Any, Any]], bool]:
  """Try to convert a dictionary with consequentive integer keys to a list.

  Args:
    src: A dict whose keys may be int type and their range form a perfect
      range(0, N) list unless convert_when_sparse is set to True.
    convert_when_sparse: When src is a int-key dict, force convert
      it to a list ordered by key, even it's sparse.

  Returns:
    converted list or src unchanged.
  """
  if not src:
    return (src, False)

  min_key = None
  max_key = None
  for key in src.keys():
    if not isinstance(key, int):
      return (src, False)
    if min_key is None or min_key > key:
      min_key = key
    if max_key is None or max_key < key:
      max_key = key
  if convert_when_sparse or (min_key == 0 and max_key == len(src) - 1):
    return ([src[key] for key in sorted(src.keys())], True)
  return (src, False)


def canonicalize(src: Any, sparse_list_as_dict: bool = True) -> Any:
  """Canonicalize (maybe) non-canonical hierarchical value.

    Non-canonical hierarchical values are dicts or nested structures of dicts
    that contain keys with '.' or '[<number>]'. Canonicalization is to unfold
    '.' and '[]' in their keys ('.' or '[]') into multi-level dicts.

    For example::

      [1, {
        "a.b[0]": {
          "e.f": 1
        }
        "a.b[0].c[x.y].d": 10
      }]

    will result in::

      [1, {
        "a": {
          "b": [{
            "c": {
              "x.y": {
                "d": 10
              }
            }
            "e": {
              "f": 1
            }
          }]
        }
      }]

   A sparse array indexer can be used in a non-canonical form. e.g::

     {
       'a[1]': 123,
       'a[5]': 234
     }

   This is to accommodate scenarios of list element update/append.
   When `sparse_list_as_dict` is set to true (by default), dict above will be
   converted to::

     {
       'a': [123, 234]
     }

   Otherwise sparse indexer will be kept so the container type will
   remain as a dict::

     {
       'a': {
         1: 123,
         5: 234
       }
     }

   (Please note that sparse indexer as key is not JSON serializable.)

   This is the reverse operation of method flatten.
   If input value is a simple type, the value itself will be returned.

  Args:
     src: A simple type or a nested structure of dict that may contains keys
       with JSON paths like 'a.b.c'
     sparse_list_as_dict: Whether convert sparse list to dict.
       When this is set to True, indices specified in the key path will be
       kept. Otherwise, a list will be returned with elements ordered by indices
       in the path.

  Returns:
     A nested structure of ordered dict that has only canonicalized keys or
     src itself if it's not a nested structure of dict. For dict of int keys
     whose values form a perfect range(0, N) will be returned as a list.

  Raises:
    KeyError: If key is empty or the same key yields conflicting values
      after resolving non-canonical paths. E.g: `{'': 1}` or
      `{'a.b': 1, 'a.b.c': True}`.
  """

  def _merge_fn(path, old_value, new_value):
    if old_value is not MISSING_VALUE and new_value is not MISSING_VALUE:
      raise KeyError(
          f'Path \'{path}\' is assigned with conflicting values. '
          f'Left value: {old_value}, Right value: {new_value}')
    # Always merge.
    return new_value if new_value is not MISSING_VALUE else old_value

  if isinstance(src, dict):
    # We keep order of keys.
    canonical_dict = dict()

    # Make deterministic traversal of dict.
    for key, value in src.items():
      if isinstance(key, str):
        path = KeyPath.parse(key)
      else:
        path = KeyPath(key)

      if len(path) == 1:
        # Key is already canonical.
        # NOTE(daiyip): pass through sparse_list_as_dict to canonicalize
        # value to keep consistency with the container.
        new_value = canonicalize(value, sparse_list_as_dict)
        if path.key not in canonical_dict:
          canonical_dict[path.key] = new_value
        else:
          old_value = canonical_dict[path.key]
          # merge dict is in-place.
          merge_tree(old_value, new_value, _merge_fn)
      else:
        # Key is a path.
        if path.is_root:
          raise KeyError(f'Key must not be empty. Encountered: {src}.')
        sub_root = dict()
        cur_dict = sub_root
        for token in path.keys[:-1]:
          cur_dict[token] = dict()
          cur_dict = cur_dict[token]
        cur_dict[path.key] = canonicalize(
            value, sparse_list_as_dict)
        # merge dict is in-place.
        merge_tree(canonical_dict, sub_root, _merge_fn)

    # NOTE(daiyip): We restore the list form of integer-keyed dict
    # if its keys form a perfect range(0, N), unless sparse_list_as_dict is set.
    def _listify_dict_equivalent(p, v):
      del p
      if isinstance(v, dict):
        v = try_listify_dict_with_int_keys(v, not sparse_list_as_dict)[0]
      return v

    return transform(canonical_dict, _listify_dict_equivalent)
  elif isinstance(src, list):
    return [canonicalize(item, sparse_list_as_dict) for item in src]
  else:
    return src


def merge(value_list: List[Any],
          merge_fn: Optional[Callable[[KeyPath, Any, Any], Any]] = None) -> Any:
  """Merge a list of hierarchical values.

  Example::

    original = {
        'a': 1,
        'b': 2,
        'c': {
            'd': 'foo',
            'e': 'bar'
        }
    }
    patch =  {
        'b': 3,
        'd': [1, 2, 3],
        'c': {
            'e': 'bar2',
            'f': 10
        }
    }
    output = pg.object_utils.merge([original, patch])
    assert output == {
        'a': 1,
        # b is updated.
        'b': 3,
        'c': {
            'd': 'foo',
            # e is updated.
            'e': 'bar2',
            # f is added.
            'f': 10
        },
        # d is inserted.
        'd': [1, 2, 3]
    })

  Args:
    value_list: A list of hierarchical values to merge. Later value will be
      treated as updates if it's a dict or otherwise a replacement of former
      value. The merge process will keep input values intact.
    merge_fn: A function to handle value merge that will be called for updated
      or added keys. If a branch is added/updated, the root of branch will be
      passed to merge_fn.
      the signature of function is:
      `(path, left_value, right_value) -> final_value`
      If a key is only present in src dict, old_value is MISSING_VALUE;
      If a key is only present in dest dict, new_value is MISSING_VALUE;
      otherwise both new_value and old_value are filled.
      If final_value is MISSING_VALUE for a path, it will be removed from its
      parent collection.

  Returns:
    A merged value.

  Raises:
    TypeError: If `value_list` is not a list.
    KeyError: If new key is found while not allowed.
  """
  if not isinstance(value_list, list):
    raise TypeError('value_list should be a list')

  if not value_list:
    return None

  new_value = canonicalize(value_list[0], sparse_list_as_dict=True)
  for value in value_list[1:]:
    if value is None:
      continue
    new_value = merge_tree(
        new_value,
        canonicalize(value, sparse_list_as_dict=True),
        merge_fn)

  def _listify_dict_equivalent(p, v):
    del p
    if isinstance(v, dict):
      v = try_listify_dict_with_int_keys(v, True)[0]
    return v
  return transform(new_value, _listify_dict_equivalent)


def _merge_dict_into_dict(
    dest: Dict[Any, Any],
    src: Dict[Any, Any],
    merge_fn: Callable[[KeyPath, Any, Any], Any],
    root_path: KeyPath) -> Dict[Any, Any]:
  """Merge a source dict into the destionation dict."""
  # NOTE(daiyip): When merge_fn is present, we iterate dest dict
  # to call merge_fn on keys that only appears in dest dict.
  keys_to_delete = []
  if merge_fn:
    for key in dest.keys():
      if key not in src:
        new_value = merge_tree(
            dest[key], MISSING_VALUE, merge_fn, KeyPath(key, root_path))
        if MISSING_VALUE != new_value:
          dest[key] = new_value
        else:
          keys_to_delete.append(key)

  # NOTE(daiyip): Merge keys from src dict to dest dict.
  for key, value in src.items():
    is_new = key not in dest
    if is_new or MISSING_VALUE == dest[key]:
      # Key exists in src but not dest (or dest[key] is MISSING_VALUE).
      new_value = value
      if merge_fn:
        new_value = merge_fn(KeyPath(key, root_path), MISSING_VALUE, value)
      if MISSING_VALUE != new_value:
        dest[key] = new_value
    else:
      # Key exists in both src and dest. Replacement scenario.
      old_value = dest[key]
      new_value = merge_tree(old_value, value,
                             merge_fn, KeyPath(key, root_path))
      if new_value is not MISSING_VALUE:
        if old_value is not new_value:
          dest[key] = new_value
      else:
        keys_to_delete.append(key)
  for key in keys_to_delete:
    del dest[key]
  return dest


def _merge_dict_into_list(
    dest: List[Any],
    src: Dict[int, Any],
    root_path: KeyPath) -> List[Any]:
  """Merge (possible) sparsed indexed list (in dict form) into a list."""
  for child_key in src.keys():
    if not isinstance(child_key, int):
      raise KeyError(
          f'Dict must use integers as keys when merging to a list. '
          f'Encountered: {src}, Path: {root_path!r}.')
  num_int_keys = len(src)
  if num_int_keys == len(src.keys()):
    old_size = len(dest)
    for int_key in sorted(src.keys()):
      if int_key < old_size:
        dest[int_key] = src[int_key]
      else:
        dest.append(src[int_key])
  return dest


def merge_tree(dest: Any,
               src: Any,
               merge_fn: Optional[Callable[[KeyPath, Any, Any], Any]] = None,
               root_path: Optional[KeyPath] = None) -> Any:
  """Deep merge two (maybe) hierarchical values.

  Args:
    dest: Destination value.
    src: Source value. When source value is a dict, it's considered as a
      patch (delta) to the destination when destination is a dict or list.
      For other source types, it's considered as a new value that will replace
      dest completely.
    merge_fn: A function to handle value merge that will be called for updated
      or added keys. If a branch is added/updated, the root of branch will be
      passed to merge_fn.
      the signature of function is: (path, left_value, right_value) ->
        final_value
        If a key is only present in src dict, old_value is MISSING_VALUE.
        If a key is only present in dest dict, new_value is MISSING_VALUE.
        Otherwise both new_value and old_value are filled.

        If final value is MISSING_VALUE, it will be removed from its parent
        collection.
   root_path: KeyPath of dest.

  Returns:
    Merged value.

  Raises:
    KeyError: Dict keys are not integers when merging into a list.
  """
  if not root_path:
    root_path = KeyPath()

  if isinstance(dest, dict) and isinstance(src, dict):
    # Merge dict into dict.
    return _merge_dict_into_dict(dest, src, merge_fn, root_path)

  if isinstance(dest, list) and isinstance(src, dict):
    # Merge (possible) sparse indexed list into a list.
    return _merge_dict_into_list(dest, src, root_path)

  # Merge at root level.
  if merge_fn:
    return merge_fn(root_path, dest, src)
  return src


def is_partial(value: Any) -> bool:
  """Returns True if a value is partially bound."""

  def _check_full_bound(path: KeyPath, value: Any) -> bool:
    del path
    if MISSING_VALUE == value:
      return False
    elif (isinstance(value, MaybePartial)
          and not isinstance(value, (dict, list))):
      return not value.is_partial
    return True
  return not traverse(value, _check_full_bound)


def kvlist_str(
    kvlist: List[Tuple[Text, Any, Any]],
    compact: bool = True,
    verbose: bool = False,
    root_indent: int = 0) -> Text:
  """Formats a list key/value pairs into a comma delimited string.

  Args:
    kvlist: List of tuples in format of
      (key, value, default_value or a tuple of default values)
    compact: If True, format value in kvlist in compact form.
    verbose: If True, format value in kvlist in verbose.
    root_indent: The indent should be applied for values in kvlist if they are
      multi-line.

  Returns:
    A formatted string from a list of key/value pairs delimited by comma.
  """
  s = []
  is_first = True
  for k, v, d in kvlist:
    if isinstance(d, tuple):
      include_pair = True
      for sd in d:
        if sd == v:
          include_pair = False
          break
    else:
      include_pair = v != d
    if include_pair:
      if not is_first:
        s.append(', ')
      if not isinstance(v, str):
        v = format(v, compact=compact, verbose=verbose, root_indent=root_indent)
      if k:
        s.append(f'{k}={v}')
      else:
        s.append(str(v))
      is_first = False
  return ''.join(s)


def quote_if_str(value: Any) -> Any:
  """Quotes the value if it is a str."""
  if isinstance(value, str):
    return f'\'{value}\''
  return value


def comma_delimited_str(value_list: Sequence[Any]) -> Text:
  """Gets comma delimited string."""
  return ', '.join(str(quote_if_str(v)) for v in value_list)


def auto_plural(
    number: int, singular: Text, plural: Optional[Text] = None) -> Text:
  """Use singular form if number is 1, otherwise use plural form."""
  if plural is None:
    plural = singular + 's'
  return singular if number == 1 else plural


def message_on_path(
    message: Text, path: KeyPath) -> Text:
  """Formats a message that is associated with a `KeyPath`."""
  if path is None:
    return message
  return f'{message} (path={path})'


class BracketType(enum.IntEnum):
  """Bracket types used for complex type formatting."""
  # Round bracket.
  ROUND = 0

  # Square bracket.
  SQUARE = 1

  # Curly bracket.
  CURLY = 2


_BRACKET_CHARS = [
    ('(', ')'),
    ('[', ']'),
    ('{', '}'),
]


def bracket_chars(bracket_type: BracketType) -> Tuple[Text, Text]:
  """Gets bracket character."""
  return _BRACKET_CHARS[int(bracket_type)]


def format(value: Any,              # pylint: disable=redefined-builtin
           compact: bool = False,
           verbose: bool = True,
           root_indent: int = 0,
           list_wrap_threshold: int = 80,
           strip_object_id: bool = False,
           exclude_keys: Optional[Set[Text]] = None,
           **kwargs) -> Text:
  """Formats a (maybe) hierarchical value with flags.

  Args:
    value: The value to format.
    compact: If True, this object will be formatted into a single line.
    verbose: If True, this object will be formatted with verbosity.
      Subclasses should define `verbosity` on their own.
    root_indent: The start indent level for this object if the output is a
      multi-line string.
    list_wrap_threshold: A threshold in number of characters for wrapping a
      list value in a single line.
    strip_object_id: If True, format object as '<class-name>(...)' other than
      'object at <address>'.
    exclude_keys: A set of keys to exclude from the top-level dict or object.
    **kwargs: Keyword arguments that will be passed through unto child
      ``Formattable`` objects.

  Returns:
    A string representation for `value`.
  """

  exclude_keys = exclude_keys or set()

  def _indent(text, indent: int) -> Text:
    return ' ' * 2 * indent + text

  def _format_child(v):
    return format(v, compact=compact, verbose=verbose,
                  root_indent=root_indent + 1,
                  list_wrap_threshold=list_wrap_threshold,
                  strip_object_id=strip_object_id,
                  **kwargs)

  if isinstance(value, Formattable):
    return value.format(compact=compact,
                        verbose=verbose,
                        root_indent=root_indent,
                        list_wrap_threshold=list_wrap_threshold,
                        strip_object_id=strip_object_id,
                        exclude_keys=exclude_keys,
                        **kwargs)
  elif isinstance(value, (list, tuple)):
    # Always try compact representation if length is not too long.
    open_bracket, close_bracket = bracket_chars(
        BracketType.SQUARE if isinstance(value, list) else BracketType.ROUND)
    s = [open_bracket]
    s.append(', '.join([_format_child(elem) for elem in value]))
    s.append(close_bracket)
    s = [''.join(s)]
    if not compact and len(s[-1]) > list_wrap_threshold:
      s = [f'{open_bracket}\n']
      s.append(',\n'.join([
          _indent(_format_child(elem), root_indent + 1)
          for elem in value
      ]))
      s.append('\n')
      s.append(_indent(close_bracket, root_indent))
  elif isinstance(value, dict):
    if compact or not value:
      s = ['{']
      s.append(', '.join([
          f'{k!r}: {_format_child(v)}'
          for k, v in value.items() if k not in exclude_keys
      ]))
      s.append('}')
    else:
      s = ['{\n']
      s.append(',\n'.join([
          _indent(f'{k!r}: {_format_child(v)}', root_indent + 1)
          for k, v in value.items() if k not in exclude_keys
      ]))
      s.append('\n')
      s.append(_indent('}', root_indent))
  else:
    if isinstance(value, str):
      # TODO(daiyip): since value can be unicode, we manually format the string
      # instead of `repr`, which adds 'u' prefix. We need to revisit unicode
      # handling as a horizontal topic for glove.
      s = [f'\'{value}\'']
    else:
      s = [repr(value) if compact else str(value)]
      if strip_object_id and 'object at 0x' in s[-1]:
        s = [f'{value.__class__.__name__}(...)']
  return ''.join(s)


def printv(v: Any, **kwargs):
  """Prints formatted value."""
  fs = kwargs.pop('file', sys.stdout)
  print(format(v, **kwargs), file=fs)


def make_function(
    name: Text,
    args: List[Text],
    body: List[Text],
    *,
    exec_globals: Optional[Dict[Text, Any]] = None,
    exec_locals: Optional[Dict[Text, Any]] = None,
    return_type: Any = MISSING_VALUE):
  """Creates a function dynamically from source."""
  if exec_locals is None:
    exec_locals = {}
  if return_type != MISSING_VALUE:
    exec_locals['_return_type'] = return_type
    return_annotation = '->_return_type'
  else:
    return_annotation = ''
  args = ', '.join(args)
  body = '\n'.join(f'  {line}' for line in body)
  fn_def = f' def {name}({args}){return_annotation}:\n{body}'
  local_vars = ', '.join(exec_locals.keys())
  fn_def = f'def _make_fn({local_vars}):\n{fn_def}\n return {name}'
  ns = {}
  exec(fn_def, exec_globals, ns)  # pylint: disable=exec-used
  return ns['_make_fn'](**exec_locals)
