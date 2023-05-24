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
"""Symbolic list."""

import dataclasses
import math
import typing
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union
from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags


# Special default value to detect missing keys in a List. Though its values is
# the same as `base._RAISE_IF_NOT_FOUND`, they are different instances so
# `pg.List` and `pg.Symbolic` could use their own instances to control error
# raising separately.
_RAISE_IF_NOT_FOUND = (pg_typing.MISSING_VALUE,)


class List(list, base.Symbolic, pg_typing.CustomTyping):
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
              items: Optional[Iterable[Any]] = None,
              *,
              value_spec: Optional[pg_typing.List] = None,
              onchange_callback: Optional[Callable[
                  [Dict[object_utils.KeyPath, base.FieldUpdate]], None]] = None,
              **kwargs) -> 'List':
    """Class method that creates a partial List object."""
    return cls(items,
               value_spec=value_spec,
               onchange_callback=onchange_callback,
               allow_partial=True,
               **kwargs)

  @classmethod
  def from_json(cls,
                json_value: Any,
                *,
                value_spec: Optional[pg_typing.List] = None,
                allow_partial: bool = False,
                root_path: Optional[object_utils.KeyPath] = None,
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
      value_spec: An optional `pg.typing.List` object as the schema for the
        list.
      allow_partial: Whether to allow elements of the list to be partial.
      root_path: KeyPath of loaded object in its object tree.
      **kwargs: Allow passing through keyword arguments that are not applicable.

    Returns:
      A schema-less symbolic list, but its items maybe symbolic.
    """
    return cls(json_value,
               value_spec=value_spec,
               allow_partial=allow_partial,
               root_path=root_path)

  def __init__(
      self,
      items: Optional[Iterable[Any]] = None,
      *,
      value_spec: Optional[pg_typing.List] = None,
      onchange_callback: Optional[Callable[
          [Dict[object_utils.KeyPath, base.FieldUpdate]], None]] = None,
      allow_partial: bool = False,
      accessor_writable: bool = True,
      sealed: bool = False,
      root_path: Optional[object_utils.KeyPath] = None):
    """Constructor.

    Args:
      items: A optional iterable object as initial value for this list.
      value_spec: Value spec that applies to this List.
      onchange_callback: Callback when sub-tree has been modified.
      allow_partial: Whether to allow unbound or partial fields. This takes
        effect only when value_spec is not None.
      accessor_writable: Whether to allow modification of this List using
        accessors (operator[]).
      sealed: Whether to seal this List after creation.
      root_path: KeyPath of this List in its object tree.
    """
    if value_spec and not isinstance(value_spec, pg_typing.List):
      raise TypeError(
          f'Argument \'value_spec\' must be a `pg.typing.List` object. '
          f'Encountered {value_spec}.')

    # We delay seal operation until items are filled.
    base.Symbolic.__init__(
        self,
        allow_partial=allow_partial,
        accessor_writable=accessor_writable,
        sealed=False,
        root_path=root_path)

    self._value_spec = None
    self._onchange_callback = None

    list.__init__(self)
    if items:
      # Copy the symbolic form instead of evaluated form.
      if isinstance(items, List):
        items = items.sym_values()

      for item in items:
        self._set_item_without_permission_check(len(self), item)

    if value_spec:
      self.use_value_spec(value_spec, allow_partial)

    # NOTE(daiyip): We set onchange callback at the end of init to avoid
    # triggering during initialization.
    self._onchange_callback = onchange_callback
    self.seal(sealed)

  @property
  def max_size(self) -> Optional[int]:
    """Returns max size of this list."""
    if self._value_spec:
      return typing.cast(pg_typing.ListKey,
                         self._value_spec.element.key).max_value
    return None

  def use_value_spec(self,
                     value_spec: Optional[pg_typing.List],
                     allow_partial: bool = False) -> 'List':
    """Applies a ``pg.List`` as the value spec for current list.

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

    if not isinstance(value_spec, pg_typing.List):
      raise ValueError(
          self._error_message(
              f'Value spec for list must be a `pg.typing.List` object. '
              f'Encountered: {value_spec!r}'))

    if self._value_spec and self._value_spec != value_spec:
      raise RuntimeError(
          self._error_message(
              f'List is already bound with a different value '
              f'spec: {self._value_spec}. New value spec: {value_spec}.'))
    self._allow_partial = allow_partial

    if flags.is_type_check_enabled():
      # NOTE(daiyip): self._value_spec will be set in List.custom_apply method
      # called by spec.apply, thus we don't need to set the _value_spec
      # explicitly.
      value_spec.apply(
          self,
          allow_partial=base.accepts_partial(self),
          child_transform=base.symbolic_transform_fn(self._allow_partial),
          root_path=self.sym_path)
    else:
      self._value_spec = value_spec
    return self

  @property
  def value_spec(self) -> Optional[pg_typing.List]:
    """Returns value spec of this List."""
    return self._value_spec

  def sym_attr_field(self, key: Union[str, int]) -> Optional[pg_typing.Field]:
    """Returns the field definition for a symbolic attribute."""
    del key
    if self._value_spec is None:
      return None
    return self._value_spec.element

  def sym_hasattr(self, key: Union[str, int]) -> bool:
    """Tests if a symbolic attribute exists."""
    return isinstance(key, int) and key >= -len(self) and key < len(self)

  def sym_keys(self) -> Iterator[int]:
    """Symbolically iterates indices."""
    for i in range(len(self)):
      yield i

  def sym_values(self) -> Iterator[Any]:
    """Iterates the values of symbolic attributes."""
    for i in range(len(self)):
      yield super().__getitem__(i)

  def sym_items(self) -> Iterator[Tuple[int, Any]]:
    """Iterates the (key, value) pairs of symbolic attributes."""
    for i in range(len(self)):
      yield (i, super().__getitem__(i))

  def sym_hash(self) -> int:
    """Symbolically hashing."""
    return base.sym_hash(
        (self.__class__, tuple([base.sym_hash(e) for e in self.sym_values()]))
    )

  def _sym_getattr(self, key: int) -> Any:   # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
    """Gets symbolic attribute by index."""
    return super().__getitem__(key)

  def _sym_clone(self, deep: bool, memo=None) -> 'List':
    """Override Symbolic._clone."""
    source = []
    for v in self.sym_values():
      if deep or isinstance(v, base.Symbolic):
        v = base.clone(v, deep, memo)
      source.append(v)
    return List(
        source,
        value_spec=self._value_spec,
        allow_partial=self._allow_partial,
        accessor_writable=self._accessor_writable,
        # NOTE(daiyip): parent and root_path are reset to empty
        # for copy object.
        root_path=None)

  def _sym_missing(self) -> Dict[Any, Any]:
    """Returns missing fields."""
    missing = dict()
    for idx, elem in self.sym_items():
      if isinstance(elem, base.Symbolic):
        missing_child = elem.sym_missing(flatten=False)
        if missing_child:
          missing[idx] = missing_child
    return missing

  def _sym_rebind(
      self, path_value_pairs: typing.Dict[object_utils.KeyPath, Any]
      ) -> typing.List[base.FieldUpdate]:
    """Subclass specific rebind implementation."""
    updates = []

    # Apply the updates in reverse order, so the operated path will not alter
    # from insertions and deletions.
    path_value_pairs = sorted(
        path_value_pairs.items(), key=lambda x: x[0], reverse=True)
    for k, v in path_value_pairs:
      update = self._set_item_of_current_tree(k, v)
      if update is not None:
        updates.append(update)
    # Reverse the updates so the update is from the smallest number to
    # the largest.
    updates.reverse()
    return updates

  def _sym_nondefault(self) -> Dict[int, Any]:
    """Returns non-default values."""
    non_defaults = dict()
    for idx, elem in self.sym_items():
      if isinstance(elem, base.Symbolic):
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
    for elem in self.sym_values():
      if isinstance(elem, base.Symbolic):
        elem.set_accessor_writable(writable)
    super().set_accessor_writable(writable)
    return self

  def seal(self, sealed: bool = True) -> 'List':
    """Seal or unseal current object from further modification."""
    if self.is_sealed == sealed:
      return self
    for elem in self.sym_values():
      if isinstance(elem, base.Symbolic):
        elem.seal(sealed)
    super().seal(sealed)
    return self

  def _update_children_paths(
      self,
      old_path: object_utils.KeyPath,
      new_path: object_utils.KeyPath) -> None:
    """Update children paths according to root_path of current node."""
    del old_path
    for idx, item in self.sym_items():
      if isinstance(item, base.Symbolic):
        item.sym_setpath(object_utils.KeyPath(idx, new_path))

  def _set_item_without_permission_check(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: int, value: Any) -> Optional[base.FieldUpdate]:
    """Set or add an item without permission check."""
    assert isinstance(key, int), key
    index = key
    if index >= len(self):
      # Appending MISSING_VALUE is considered no-op.
      if value == pg_typing.MISSING_VALUE:
        return None
      index = len(self)
    should_insert = False
    if isinstance(value, Insertion):
      should_insert = True
      value = value.value

    old_value = pg_typing.MISSING_VALUE
    # Replace an existing value.
    if index < len(self) and not should_insert:
      old_value = list.__getitem__(self, index)
      # Generates no update as old value is the same as the new value.
      if old_value is value:
        return None

    new_value = self._formalized_value(index, value)
    if index < len(self):
      if should_insert:
        list.insert(self, index, new_value)
      else:
        list.__setitem__(self, index, new_value)
        # Detach old value from object tree.
        if isinstance(old_value, base.Symbolic):
          old_value.sym_setparent(None)
    else:
      super().append(new_value)
    return base.FieldUpdate(
        self.sym_path + index, self,
        self._value_spec.element if self._value_spec else None,
        old_value, new_value)

  def _formalized_value(self, idx: int, value: Any):
    """Get transformed (formal) value from user input."""
    allow_partial = base.accepts_partial(self)
    value = base.from_json(
        value,
        allow_partial=allow_partial,
        root_path=object_utils.KeyPath(idx, self.sym_path))
    if self._value_spec and flags.is_type_check_enabled():
      value = self._value_spec.element.apply(
          value,
          allow_partial=allow_partial,
          transform_fn=base.symbolic_transform_fn(self._allow_partial),
          root_path=object_utils.KeyPath(idx, self.sym_path))
    return self._relocate_if_symbolic(idx, value)

  @property
  def _subscribes_field_updates(self) -> bool:
    """Returns True if current list subscribes field updates."""
    return self._onchange_callback is not None

  def _on_change(self,
                 field_updates: Dict[object_utils.KeyPath, base.FieldUpdate]):
    """On change event of List."""
    # Do nothing for now to handle changes of List.

    # NOTE(daiyip): Remove items that are MISSING_VALUES.
    keys_to_remove = []
    for i, item in self.sym_items():
      if pg_typing.MISSING_VALUE == item:
        keys_to_remove.append(i)
    if keys_to_remove:
      for i in reversed(keys_to_remove):
        list.__delitem__(self, i)

    # Update paths for children.
    for idx, item in self.sym_items():
      if isinstance(item, base.Symbolic) and item.sym_path.key != idx:
        item.sym_setpath(object_utils.KeyPath(idx, self.sym_path))

    if self._onchange_callback is not None:
      self._onchange_callback(field_updates)

  def _parse_slice(self, index: slice) -> Tuple[int, int, int]:
    start = index.start if index.start is not None else 0
    start = max(-len(self), start)
    start = min(len(self), start)
    if start < 0:
      start += len(self)

    stop = index.stop if index.stop is not None else len(self)
    stop = max(-len(self), stop)
    stop = min(len(self), stop)
    if stop < 0:
      stop += len(self)

    step = index.step if index.step is not None else 1
    return start, stop, step

  def _sym_value(self, key: int, default: Any) -> Any:  # pytype: disable=signature-mismatch
    try:
      v = super().__getitem__(key)
    except IndexError:
      return default

    def _eval(i, v):
      if isinstance(v, base.ContextualValue):
        return self.sym_contextual_getattr(
            i, default=default, getter=v, start=self.sym_parent
        )
      return v

    if isinstance(key, slice):
      return [
          _eval(k, v[i]) for i, k in enumerate(range(*self._parse_slice(key)))
      ]
    return _eval(key, v)

  def __getitem__(self, index) -> Any:
    """Gets the item at a given position."""
    v = self.sym_value(index, _RAISE_IF_NOT_FOUND)
    if v is _RAISE_IF_NOT_FOUND:
      raise IndexError('list index out of range')
    return v

  def __setitem__(self, index, value: Any) -> None:
    """Set item in this List."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError(
          self._error_message('Cannot set item for a sealed List.'))

    if not base.writtable_via_accessors(self):
      raise base.WritePermissionError(
          self._error_message('Cannot modify List item by __setitem__ while '
                              'accessor_writable is set to False. '
                              'Use \'rebind\' method instead.'))
    if isinstance(index, slice):
      start, stop, step = self._parse_slice(index)
      replacements = [self._formalized_value(i, v) for i, v in enumerate(value)]
      if step < 0:
        replacements.reverse()
        step = -step
      slice_size = math.ceil((stop - start) * 1.0 / step)
      if step == 1:
        if slice_size < len(replacements):
          for i in range(slice_size, len(replacements)):
            replacements[i] = Insertion(replacements[i])
        else:
          replacements.extend(
              [pg_typing.MISSING_VALUE
               for _ in range(slice_size - len(replacements))])
      elif slice_size != len(replacements):
        raise ValueError(
            f'attempt to assign sequence of size {len(replacements)} to '
            f'extended slice of size {slice_size}')
      updates = []
      for i, r in enumerate(replacements):
        update = self._set_item_without_permission_check(start + i * step, r)
        if update is not None:
          updates.append(update)
      if flags.is_change_notification_enabled() and updates:
        self._notify_field_updates(updates)
    elif isinstance(index, int):
      if index < -len(self) or index >= len(self):
        raise IndexError(
            f'list assignment index out of range. '
            f'Length={len(self)}, index={index}')
      update = self._set_item_without_permission_check(index, value)
      if flags.is_change_notification_enabled() and update:
        self._notify_field_updates([update])
    else:
      raise TypeError(
          f'list assignment index must be an integer. Encountered {index!r}.')

  def __delitem__(self, index: int) -> None:
    """Delete an item from the List."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot delete item from a sealed List.')

    if not base.writtable_via_accessors(self):
      raise base.WritePermissionError(
          self._error_message('Cannot delete List item while accessor_writable '
                              'is set to False. '
                              'Use \'rebind\' method instead.'))
    if not isinstance(index, int):
      raise TypeError(
          f'list index must be an integer. Encountered {index!r}.')

    if index < -len(self) or index >= len(self):
      raise IndexError(
          f'list index out of range. '
          f'Length={len(self)}, index={index}')

    old_value = self.sym_getattr(index)
    super().__delitem__(index)

    if flags.is_change_notification_enabled():
      self._notify_field_updates([
          base.FieldUpdate(
              self.sym_path + index, self,
              self._value_spec.element if self._value_spec else None,
              old_value, pg_typing.MISSING_VALUE)
      ])

  def __add__(self, other: Iterable[Any]) -> 'List':
    """Returns a concatenated List of self and other."""
    concatenated = self.copy()
    concatenated.extend(other)
    return concatenated

  def __mul__(self, n: int) -> 'List':
    """Returns a repeated Lit of self."""
    result = List()
    for _ in range(n):
      result.extend(self)
    if self._value_spec is not None:
      result.use_value_spec(self._value_spec)
    return result

  def __rmul__(self, n: int) -> 'List':
    """Returns a repeated Lit of self."""
    return self.__mul__(n)

  def copy(self) -> 'List':
    """Shallow current list."""
    return List(super().copy(), value_spec=self._value_spec)

  def append(self, value: Any) -> None:
    """Appends an item."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot append element on a sealed List.')
    if self.max_size is not None and len(self) >= self.max_size:
      raise ValueError(f'List reached its max size {self.max_size}.')

    update = self._set_item_without_permission_check(len(self), value)
    if flags.is_change_notification_enabled() and update:
      self._notify_field_updates([update])

  def insert(self, index: int, value: Any) -> None:
    """Inserts an item at a given position."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError(
          'Cannot insert element into a sealed List.')
    if self.max_size is not None and len(self) >= self.max_size:
      raise ValueError(f'List reached its max size {self.max_size}.')

    update = self._set_item_without_permission_check(
        index, mark_as_insertion(value))
    if flags.is_change_notification_enabled() and update:
      self._notify_field_updates([update])

  def pop(self, index: int = -1) -> Any:
    """Pop an item and return its value."""
    if index < -len(self) or index >= len(self):
      raise IndexError('pop index out of range')
    index = (index + len(self)) % len(self)
    value = self[index]
    with flags.allow_writable_accessors(True):
      del self[index]
    return value

  def remove(self, value: Any) -> None:
    """Removes the first occurrence of the value."""
    for i, item in self.sym_items():
      if item == value:
        if (self._value_spec and self._value_spec.min_size == len(self)):
          raise ValueError(
              f'Cannot remove item: min size ({self._value_spec.min_size}) '
              f'is reached.')
        del self[i]
        return
    raise ValueError(f'{value!r} not in list.')

  def extend(self, other: Iterable[Any]) -> None:
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot extend a sealed List.')
    other = list(other)
    if self.max_size is not None and len(self) + len(other) > self.max_size:
      raise ValueError(
          f'Cannot extend List: the number of elements '
          f'({len(self) + len(other)}) exceeds max size ({self.max_size}).')
    updates = []
    # Extend on the symbolic form instead of the evaluated form.
    iter_other = other.sym_values() if isinstance(other, List) else other
    for v in iter_other:
      update = self._set_item_without_permission_check(len(self), v)
      if update is not None:
        updates.append(update)

    if flags.is_change_notification_enabled() and updates:
      self._notify_field_updates(updates)

  def clear(self) -> None:
    """Clears the list."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot clear a sealed List.')
    if self._value_spec and self._value_spec.min_size > 0:
      raise ValueError(
          f'List cannot be cleared: min size is {self._value_spec.min_size}.')
    super().clear()

  def sort(self, *, key=None, reverse=False) -> None:
    """Sorts the items of the list in place.."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot sort a sealed List.')
    super().sort(key=key, reverse=reverse)

  def reverse(self) -> None:
    """Reverse the elements of the list in place."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot reverse a sealed List.')
    super().reverse()

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[
          Callable[[object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
  ) -> Tuple[bool, 'List']:
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
                f'List (spec={self._value_spec!r}) cannot be assigned to an '
                f'incompatible field (spec={value_spec!r}).', path))
      if self._allow_partial == allow_partial:
        proceed_with_standard_apply = False
      else:
        self._allow_partial = allow_partial
    elif isinstance(value_spec, pg_typing.List):
      self._value_spec = value_spec
    return (proceed_with_standard_apply, self)

  def sym_jsonify(self, **kwargs) -> object_utils.JSONValueType:
    """Converts current list to a list of plain Python objects."""
    return [base.to_json(v, **kwargs) for v in self.sym_values()]

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      python_format: bool = False,
      cls_name: Optional[str] = None,
      bracket_type: object_utils.BracketType = object_utils.BracketType.SQUARE,
      **kwargs) -> str:
    """Formats this List."""

    def _indent(text, indent):
      return ' ' * 2 * indent + text

    cls_name = cls_name or ''
    open_bracket, close_bracket = object_utils.bracket_chars(bracket_type)
    s = [f'{cls_name}{open_bracket}']
    if compact:
      kv_strs = []
      for idx, elem in self.sym_items():
        v_str = object_utils.format(
            elem, compact, verbose, root_indent + 1,
            python_format=python_format, **kwargs)
        if python_format:
          kv_strs.append(v_str)
        else:
          kv_strs.append(f'{idx}: {v_str}')
      s.append(', '.join(kv_strs))
      s.append(close_bracket)
    else:
      if self:
        for idx, elem in self.sym_items():
          if idx == 0:
            s.append('\n')
          else:
            s.append(',\n')
          v_str = object_utils.format(
              elem, compact, verbose, root_indent + 1,
              python_format=python_format, **kwargs)
          if python_format:
            s.append(_indent(v_str, root_indent + 1))
          else:
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


base.Symbolic.ListType = List


@dataclasses.dataclass
class Insertion:
  """Class that marks a value to insert into a list.

  Example::

    l = pg.List([0, 1])
    l.rebind({
      0: pg.Insertion(2)
    })
    assert l == [2, 0, 1]
  """

  value: Any


def mark_as_insertion(value: Any) -> Insertion:
  """Mark a value as an insertion to a List."""
  return Insertion(value=value)
