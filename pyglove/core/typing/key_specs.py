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
"""Concrete key specifications for field definition."""

import re
from typing import Any, Dict, Optional

from pyglove.core import object_utils
from pyglove.core.typing.class_schema import KeySpec


class KeySpecBase(KeySpec):
  """Base class for key specification subclasses."""

  def extend(self, base: KeySpec) -> KeySpec:
    """Extend current key spec based on a base spec."""
    if self != base:
      raise KeyError(f'{self} cannot extend {base} for keys are different.')
    return self

  def __repr__(self) -> str:
    """Operator repr."""
    return self.__str__()

  def __ne__(self, other: Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)


class ConstStrKey(KeySpecBase, object_utils.StrKey):
  """Class that represents a constant string key.

  Example::

      key = pg.typing.ConstStrKey('x')
      assert key == 'x'
      assert hash(key) == hash('x')
  """

  __serialization_key__ = 'pyglove.typing.ConstStrKey'

  @property
  def is_const(self) -> bool:
    return True

  def __init__(self, text: str):
    """Constructor.

    Args:
      text: string value of this key.

    Raises:
      KeyError: If key contains dots ('.'), which is not allowed.
    """
    if '.' in text:
      raise KeyError(f'\'.\' cannot be used in key. Encountered: {text!r}.')
    super().__init__()
    self._text = text

  @property
  def text(self) -> str:
    """Text of this const string key."""
    return self._text

  def match(self, key: Any) -> bool:
    """Whether can match against an input key."""
    return self._text == key

  def format(self, **kwargs) -> str:
    """Format this object."""
    return self._text

  def to_json(self, **kwargs: Any) -> Dict[str, Any]:
    return self.to_json_dict(
        fields=dict(text=self._text),
        **kwargs,
    )

  def __hash__(self) -> int:
    """Hash function.

    NOTE(daiyip): ConstStrKey shares the same hash with its text, which
    makes it easy to lookup a dict of string by an ConstStrKey object, and
    vice versa.

    Returns:
      Hash code.
    """
    return self._text.__hash__()

  def __eq__(self, other: Any) -> bool:
    """Operator ==."""
    if self is other:
      return True
    if isinstance(other, str):
      return self.text == other
    return isinstance(other, ConstStrKey) and other.text == self.text


class NonConstKey(KeySpecBase):
  """Base class of specification for non-const key.

  Subclasses: :class:`pyglove.typing.StrKey`.
  """

  @property
  def is_const(self) -> bool:
    return False


class StrKey(NonConstKey):
  """Class that represents a variable string key.

  Example::

      # Create a key spec that specifies all string keys started with 'foo'.
      key = pg.typing.StrKey('foo.*')

      assert key.match('foo')
      assert key.match('foo1')
      assert not key.match('bar')
  """

  __serialization_key__ = 'pyglove.typing.StrKey'

  def __init__(self, regex: Optional[str] = None):
    """Constructor.

    Args:
      regex: An optional regular expression. If set to None, any string value is
        acceptable.
    """
    super().__init__()
    self._regex = re.compile(regex) if regex else None

  def match(self, key: Any) -> bool:
    """Whether this key spec can match against input key."""
    if not isinstance(key, str):
      return False
    if self._regex:
      return self._regex.match(key) is not None
    return True

  @property
  def regex(self):
    """Returns regular expression of this key spec."""
    return self._regex

  def format(self, **kwargs):
    """Format this object."""
    regex_str = object_utils.kvlist_str([
        ('regex', object_utils.quote_if_str(
            self._regex.pattern if self._regex else None), None)
    ])
    return f'StrKey({regex_str})'

  def to_json(self, **kwargs: Any) -> Dict[str, Any]:
    regex = self._regex.pattern if self._regex is not None else None
    return self.to_json_dict(
        fields=dict(regex=(regex, None)),
        exclude_default=True,
        **kwargs,
    )

  def __hash__(self):
    """Hash function."""
    if self._regex:
      return self._regex.pattern.__hash__()
    else:
      return '.*'.__hash__()

  def __eq__(self, other):
    """Operator ==."""
    if self is other:
      return True
    return isinstance(other, StrKey) and self._regex == other.regex


class ListKey(NonConstKey):
  """Class that represents key specification for a list.

  Example::

      # Create a key spec that specifies list items from 1 to 5 (zero-based).
      key = pg.typing.ListKey(min_value=1, max_value=5)

      assert key.match(1)
      assert key.match(5)
      assert not key.match(0)
  """

  __serialization_key__ = 'pyglove.typing.ListKey'

  def __init__(
      self, min_value: int = 0, max_value: Optional[int] = None):
    """Constructor.

    Args:
      min_value: Min value that is acceptable for the list index.
      max_value: Max value that is acceptable for the list index. If None, there
        is no upper bound for list index.
    """
    super().__init__()
    self._min_value = min_value
    self._max_value = max_value

  def extend(self, base: KeySpec) -> 'ListKey':
    """Extend current key spec on top of base spec."""
    if not isinstance(base, ListKey):
      raise TypeError(f'{self} cannot extend {base}: incompatible type.')

    if self.min_value < base.min_value:
      raise TypeError(f'{self} cannot extend {base}: min_value is smaller.')
    if base.max_value is None:
      return self
    if self.max_value is None:
      self._max_value = base.max_value
    elif self.max_value > base.max_value:
      raise TypeError(f'{self} cannot extend {base}: max_value is greater.')
    return self

  @property
  def min_value(self) -> int:
    """Returns min value of acceptable list index value."""
    return self._min_value

  @property
  def max_value(self) -> Optional[int]:
    """Returns max value of acceptable list index value."""
    return self._max_value

  def match(self, key: Any) -> bool:
    """Returns whether this key spec can match against input key."""
    return (isinstance(key, int) and (self._min_value <= key) and
            (not self._max_value or self._max_value > key))

  def format(self, **kwargs):
    """Format this object."""
    return f'ListKey(min_value={self._min_value}, max_value={self._max_value})'

  def to_json(self, **kwargs: Any) -> Dict[str, Any]:
    return self.to_json_dict(
        fields=dict(
            min_value=(self._min_value, None),
            max_value=(self._max_value, None)),
        exclude_default=True,
        **kwargs,
    )

  def __eq__(self, other):
    """Operator==."""
    if self is other:
      return True
    return (isinstance(other, ListKey) and
            self._min_value == other.min_value and
            self._max_value == other.max_value)


class TupleKey(NonConstKey):
  """Class that represents a key specification for tuple.

  Example::

      # Create a key spec that specifies item 0 of a tuple.
      key = pg.typing.TupleKey(0)

      assert key.match(0)
      assert not key.match(1)
  """

  __serialization_key__ = 'pyglove.typing.TupleKey'

  def __init__(self, index: Optional[int] = None):
    """Constructor.

    Args:
      index: index of the tuple field that this key spec applies to.
        If None, this tuple value spec applies to all elements of a
        variable-length tuple.
    """
    super().__init__()
    self._index = index

  def extend(self, base: KeySpec) -> 'TupleKey':
    """Extends this key spec on top of a base spec."""
    if not isinstance(base, TupleKey):
      raise TypeError(f'{self} cannot extend {base}: incompatible type.')
    if self._index is None:
      self._index = base.index
    elif base.index is not None and base.index != self.index:
      raise KeyError(f'{self} cannot extend {base}: unmatched index.')
    return self

  @property
  def index(self) -> Optional[int]:
    """Returns the index of tuple field that the key applies to."""
    return self._index

  def match(self, key: Any) -> bool:
    """Returns whether this key spec can match against input key."""
    return isinstance(key, int) and (self._index is None or self._index == key)

  def format(self, **kwargs):
    """Format this object."""
    return 'TupleKey(index={self._index})'

  def to_json(self, **kwargs: Any) -> Dict[str, Any]:
    return self.to_json_dict(
        fields=dict(index=(self._index, None)),
        exclude_default=True,
        **kwargs,
    )

  def __eq__(self, other):
    """Operator==."""
    if self is other:
      return True
    return isinstance(other, TupleKey) and self._index == other.index


KeySpec.from_str = ConstStrKey
