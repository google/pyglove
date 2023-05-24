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
"""Handling locations in a hierarchical object."""

import abc
import copy
import operator
from typing import Any, Callable, List, Optional, Union
from pyglove.core.object_utils import common_traits


class KeyPath(common_traits.Formattable):
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
  def from_value(cls, value: Union['KeyPath', str, int]) -> 'KeyPath':
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
            path_str: str,
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

  def __sub__(self, other: Union[None, int, str, 'KeyPath']) -> 'KeyPath':
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
    # NOTE(daiyip): For contextual value (e.g. ``pg.ContextualValue``),
    # `query` returns its symbolic form instead of its evaluated value.
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
  def path(self) -> str:
    """JSONPath representation of current path."""
    if self._path_str is None:
      self._path_str = self.path_str()
    return self._path_str

  def path_str(self, preserve_complex_keys: bool = True) -> str:
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

  def format(self, *args, **kwargs):
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
