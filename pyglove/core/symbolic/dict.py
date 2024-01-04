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
"""Symbolic dict."""

import typing
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags


class Dict(dict, base.Symbolic, pg_typing.CustomTyping):
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
              dict_obj: Optional[typing.Dict[str, Any]] = None,
              value_spec: Optional[pg_typing.Dict] = None,
              *,
              onchange_callback: Optional[Callable[
                  [typing.Dict[object_utils.KeyPath, base.FieldUpdate]], None]
              ] = None,  # pylint: disable=bad-continuation
              **kwargs) -> 'Dict':
    """Class method that creates a partial Dict object."""
    return cls(dict_obj,
               value_spec=value_spec,
               onchange_callback=onchange_callback,
               allow_partial=True,
               **kwargs)

  @classmethod
  def from_json(cls,
                json_value: Any,
                *,
                value_spec: Optional[pg_typing.Dict] = None,
                allow_partial: bool = False,
                root_path: Optional[object_utils.KeyPath] = None,
                **kwargs) -> 'Dict':
    """Class method that load an symbolic Dict from a JSON value.

    Args:
      json_value: Input JSON value, only JSON dict is acceptable.
      value_spec: An optional value spec to apply.
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
    return cls(json_value,
               value_spec=value_spec,
               allow_partial=allow_partial,
               root_path=root_path)

  def __init__(self,
               dict_obj: Union[
                   None,
                   Iterable[Tuple[str, Any]],
                   typing.Dict[str, Any]] = None,
               *,
               value_spec: Optional[pg_typing.Dict] = None,
               onchange_callback: Optional[Callable[
                   [typing.Dict[object_utils.KeyPath, base.FieldUpdate]], None]
               ] = None,  # pylint: disable=bad-continuation
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
    if value_spec and not isinstance(value_spec, pg_typing.Dict):
      raise TypeError(
          f'Argument \'value_spec\' must be a `pg.typing.Dict` object. '
          f'Encountered {value_spec}')

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
    self._set_raw_attr(
        '_as_object_attributes_container',
        kwargs.pop('as_object_attributes_container', False),
    )

    # We copy the symbolic form of dict values instead of their evaluated
    # values.
    if isinstance(dict_obj, Dict):
      dict_obj = {k: v for k, v in dict_obj.sym_items()}
    elif dict_obj is not None:
      dict_obj = dict(dict_obj)

    # NOTE(daiyip): we call __init__ of superclasses explicitly instead of
    # calling super().__init__(...) since dict.__init__ does
    # not follow super(...).__init__ fashion, which will lead to
    # Symbolic.__init__ uncalled.
    base.Symbolic.__init__(
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
  def value_spec(self) -> Optional[pg_typing.Dict]:
    """Returns value spec of this dict.

    NOTE(daiyip): If this dict is schema-less, value_spec will be None.
    """
    return self._value_spec

  def use_value_spec(self,
                     value_spec: Optional[pg_typing.Dict],
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

    if not isinstance(value_spec, pg_typing.Dict):
      raise ValueError(
          self._error_message(
              f'Value spec for list must be a `pg.typing.Dict` object. '
              f'Encountered: {value_spec!r}'))

    if self._value_spec and self._value_spec != value_spec:
      raise RuntimeError(
          self._error_message(
              f'Dict is already bound with a different value spec: '
              f'{self._value_spec}. New value spec: {value_spec}.'))

    self._allow_partial = allow_partial

    if flags.is_type_check_enabled():
      # NOTE(daiyip): self._value_spec will be set in Dict.custom_apply method
      # called by value_spec.apply, thus we don't need to set self._value_spec
      # explicitly.
      value_spec.apply(
          self,
          allow_partial=base.accepts_partial(self),
          child_transform=base.symbolic_transform_fn(self._allow_partial),
          root_path=self.sym_path)
    else:
      self._value_spec = value_spec
    return self

  def _sym_parent_for_children(self) -> Optional[base.Symbolic]:
    if self._as_object_attributes_container:
      return self.sym_parent
    return self

  def _sym_rebind(
      self, path_value_pairs: typing.Dict[object_utils.KeyPath, Any]
      ) -> List[base.FieldUpdate]:
    """Subclass specific rebind implementation."""
    updates = []
    for k, v in path_value_pairs.items():
      update = self._set_item_of_current_tree(k, v)
      if update is not None:
        updates.append(update)
    return updates

  def _sym_missing(self) -> typing.Dict[str, Any]:
    """Returns missing values.

    Returns:
      A dict of key to MISSING_VALUE.
    """
    missing = dict()
    if self._value_spec and self._value_spec.schema:
      matched_keys, _ = self._value_spec.schema.resolve(self.keys())
      for key_spec, keys in matched_keys.items():
        field = self._value_spec.schema[key_spec]
        assert keys or isinstance(key_spec, pg_typing.NonConstKey), key_spec
        if keys:
          for key in keys:
            v = self.sym_getattr(key)
            if object_utils.MISSING_VALUE == v:
              missing[key] = field.value.default
            else:
              if isinstance(v, base.Symbolic):
                missing_child = v.sym_missing(flatten=False)
                if missing_child:
                  missing[key] = missing_child
    else:
      for k, v in self.sym_items():
        if isinstance(v, base.Symbolic):
          missing_child = v.sym_missing(flatten=False)
          if missing_child:
            missing[k] = missing_child
    return missing

  def _sym_nondefault(self) -> typing.Dict[str, Any]:
    """Returns non-default values as key/value pairs in a dict."""
    non_defaults = dict()
    if self._value_spec is not None and self._value_spec.schema:
      dict_schema = self._value_spec.schema
      matched_keys, _ = dict_schema.resolve(self.keys())
      for key_spec, keys in matched_keys.items():
        value_spec = dict_schema[key_spec].value
        for key in keys:
          diff = self._diff_base(self.sym_getattr(key), value_spec.default)
          if pg_typing.MISSING_VALUE != diff:
            non_defaults[key] = diff
    else:
      for k, v in self.sym_items():
        if isinstance(v, base.Symbolic):
          non_defaults_child = v.sym_nondefault(flatten=False)
          if non_defaults_child:
            non_defaults[k] = non_defaults_child
        else:
          non_defaults[k] = v
    return non_defaults

  def _diff_base(self, value: Any, base_value: Any) -> Any:
    """Computes the diff between a value and a base value."""
    if base.eq(value, base_value):
      return pg_typing.MISSING_VALUE

    if (isinstance(value, list)
        or not isinstance(value, base.Symbolic)
        or pg_typing.MISSING_VALUE == base_value):
      return value

    if value.__class__ is base_value.__class__:
      getter = lambda x, k: x.sym_getattr(k)
    elif isinstance(value, dict) and isinstance(base_value, dict):
      getter = lambda x, k: x[k]
    else:
      return value

    diff = {}
    for k, v in value.sym_items():
      base_v = getter(base_value, k)
      child_diff = self._diff_base(v, base_v)
      if pg_typing.MISSING_VALUE != child_diff:
        diff[k] = child_diff
    return diff

  def seal(self, sealed: bool = True) -> 'Dict':
    """Seals or unseals current object from further modification."""
    if self.is_sealed == sealed:
      return self
    for v in self.sym_values():
      if isinstance(v, base.Symbolic):
        v.seal(sealed)
    super().seal(sealed)
    return self

  def sym_attr_field(
      self, key: Union[str, int]
      ) -> Optional[pg_typing.Field]:
    """Returns the field definition for a symbolic attribute."""
    if self._value_spec is None or self._value_spec.schema is None:
      return None
    return self._value_spec.schema.get_field(key)  # pytype: disable=attribute-error

  def sym_hasattr(self, key: Union[str, int]) -> bool:
    """Tests if a symbolic attribute exists."""
    return key in self

  def sym_keys(self) -> Iterator[str]:
    """Iterates the keys of symbolic attributes."""
    if self._value_spec is None or self._value_spec.schema is None:
      for key in super().__iter__():
        yield key
    else:
      traversed = set()
      for key_spec in self._value_spec.schema.keys():  # pytype: disable=attribute-error
        if isinstance(key_spec, pg_typing.ConstStrKey) and key_spec in self:
          yield key_spec.text
          traversed.add(key_spec.text)

      if len(traversed) < len(self):
        for key in super().__iter__():
          if key not in traversed:
            yield key

  def sym_values(self) -> Iterator[Any]:
    """Iterates the values of symbolic attributes."""
    for k in self.sym_keys():
      yield self._sym_getattr(k)

  def sym_items(self) -> Iterator[
      Tuple[str, Any]]:
    """Iterates the (key, value) pairs of symbolic attributes."""
    for k in self.sym_keys():
      yield k, self._sym_getattr(k)

  def sym_setparent(self, parent: base.Symbolic):
    """Override set parent of Dict to handle the passing through scenario."""
    super().sym_setparent(parent)
    # NOTE(daiyip): when flag `as_object_attributes_container` is on, it sets
    # the parent of child symbolic values using its parent.
    if self._as_object_attributes_container:
      for v in self.sym_values():
        if isinstance(v, base.TopologyAware):
          v.sym_setparent(parent)

  def sym_hash(self) -> int:
    """Symbolic hashing."""
    return base.sym_hash(
        (self.__class__,
         tuple([(k, base.sym_hash(v)) for k, v in self.sym_items()
                if v != pg_typing.MISSING_VALUE])))

  def _sym_getattr(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: str) -> Any:
    """Gets symbolic attribute by key."""
    return super().__getitem__(key)

  def _sym_clone(self, deep: bool, memo=None) -> 'Dict':
    """Override Symbolic._sym_clone."""
    source = dict()
    for k, v in self.sym_items():
      if deep or isinstance(v, base.Symbolic):
        v = base.clone(v, deep, memo)
      source[k] = v
    return Dict(
        source,
        value_spec=self._value_spec,
        allow_partial=self._allow_partial,
        accessor_writable=self._accessor_writable,
        sealed=self._sealed,
        onchange_callback=self._onchange_callback,
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
    for k, v in self.sym_items():
      if isinstance(v, base.TopologyAware):
        v.sym_setpath(object_utils.KeyPath(k, new_path))

  def _set_item_without_permission_check(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, key: str, value: Any) -> Optional[base.FieldUpdate]:
    """Set item without permission check."""
    if not isinstance(key, str):
      raise KeyError(self._error_message(
          f'Key must be string type. Encountered {key!r}.'))

    old_value = self.get(key, pg_typing.MISSING_VALUE)
    if old_value is value:
      return None

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

    # Detach old value from object tree.
    if isinstance(old_value, base.TopologyAware):
      old_value.sym_setparent(None)
      old_value.sym_setpath(object_utils.KeyPath())

    if (pg_typing.MISSING_VALUE == value and
        (not field or isinstance(field.key, pg_typing.NonConstKey))):
      if key in self:
        # Using pg.MISSING_VALUE for deleting keys.
        super().__delitem__(key)
        new_value = pg_typing.MISSING_VALUE
      else:
        # This condition could trigger when copying a partial Dict to a Dict
        # without schema.
        return None
    else:
      new_value = self._formalized_value(key, field, value)
      super().__setitem__(key, new_value)

    # NOTE(daiyip): If current dict is the field dict of a symbolic object,
    # Use parent object as update target.
    target = self
    if (self.sym_parent is not None
        and self.sym_parent.sym_path == self.sym_path):
      target = self.sym_parent
    return base.FieldUpdate(
        self.sym_path + key, target, field, old_value, new_value)

  def _formalized_value(self, name: str,
                        field: Optional[pg_typing.Field],
                        value: Any) -> Any:
    """Get transformed (formal) value from user input."""
    allow_partial = base.accepts_partial(self)
    if field and pg_typing.MISSING_VALUE == value:
      # NOTE(daiyip): default value is already in transformed form.
      value = field.default_value
    else:
      value = base.from_json(
          value,
          allow_partial=allow_partial,
          root_path=object_utils.KeyPath(name, self.sym_path))
    if field and flags.is_type_check_enabled():
      value = field.apply(
          value,
          allow_partial=allow_partial,
          transform_fn=base.symbolic_transform_fn(self._allow_partial),
          root_path=object_utils.KeyPath(name, self.sym_path))
    return self._relocate_if_symbolic(name, value)

  @property
  def _subscribes_field_updates(self) -> bool:
    """Returns True if current dict subscribes field updates."""
    return self._onchange_callback is not None

  def _on_change(self, field_updates: typing.Dict[object_utils.KeyPath,
                                                  base.FieldUpdate]):
    """On change event of Dict."""
    if self._onchange_callback:
      self._onchange_callback(field_updates)

  def _init_kwargs(self) -> typing.Dict[str, Any]:
    kwargs = super()._init_kwargs()
    if not self._accessor_writable:
      kwargs['accessor_writable'] = False
    if self._onchange_callback is not None:
      kwargs['onchange_callback'] = self._onchange_callback
    # NOTE(daiyip): We do not serialize ValueSpec for now as in most use
    # cases they come from the subclasses of `pg.Object`.
    return kwargs

  def __getstate__(self) -> Any:
    """Customizes pickle.dump."""
    return dict(value=dict(self), kwargs=self._init_kwargs())

  def __setstate__(self, state) -> None:
    """Customizes pickle.load."""
    self.__init__(state['value'], **state['kwargs'])

  def __getitem__(self, key: str) -> Any:
    """Get item in this Dict."""
    try:
      return self.sym_inferred(key)
    except AttributeError as e:
      raise KeyError(key) from e

  def __setitem__(self, key: str, value: Any) -> None:
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
    # NOTE(daiyip): THIS IS A WORKAROUND FOR WORKING WITH PICKLE.
    # `pg.Dict` is a subclass of `dict`, therefore, when pickle loads a dict,
    # it will try to set its items directly by calling `__setitem__` without
    # calling `pg.Dict.__init__` at the first place. As a result, an error will
    # raise, which complains about that an attribute set up during `__init__` is
    # not available. A mitigation to this issue is to detect such calls in
    # `__setitem__` as the follows, and simply do nothing, which will give a
    # chance to `pg.Dict.__getstate__` to deal with the restoration logic as
    # an object (instead of a dict).
    if not hasattr(self, '_sym_parent'):
      return

    if base.treats_as_sealed(self):
      raise base.WritePermissionError(
          self._error_message('Cannot modify field of a sealed Dict.'))

    if not base.writtable_via_accessors(self):
      raise base.WritePermissionError(
          self._error_message(
              'Cannot modify Dict field by attribute or key while '
              'accessor_writable is set to False. '
              'Use \'rebind\' method instead.'))

    update = self._set_item_without_permission_check(key, value)
    if flags.is_change_notification_enabled() and update:
      self._notify_field_updates([update])

  def __setattr__(self, name: str, value: Any) -> None:
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

  def __delitem__(self, name: str) -> None:
    """Delete a key from the Dict.

    This is used to delete a key which resolves to a pg.typing.NonConstKey.

    Args:
      name: Key to delete.

    Raises:
      WritePermissionError: When Dict is sealed.
      KeyError: When key is not a NonConstKey.
    """
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot del item from a sealed Dict.')

    if not base.writtable_via_accessors(self):
      raise base.WritePermissionError(
          self._error_message('Cannot del Dict field by attribute or key while '
                              'accessor_writable is set to False. '
                              'Use \'rebind\' method instead.'))

    if name not in self:
      raise KeyError(
          self._error_message(f'Key does not exist in Dict: {name!r}.'))

    update = self._set_item_without_permission_check(
        name, pg_typing.MISSING_VALUE)
    if flags.is_change_notification_enabled() and update:
      self._notify_field_updates([update])

  def __delattr__(self, name: str) -> None:
    """Delete an attribute."""
    del self[name]

  def __getattr__(self, name: str) -> Any:
    """Get attribute that is not defined as property."""
    if name in self:
      return self.sym_inferred(name)
    raise AttributeError(
        f'Attribute \'{name}\' does not exist in {self.__class__!r}.')

  def __iter__(self):
    """Iterate keys in field declaration order."""
    return self.sym_keys()

  def keys(self) -> Iterator[str]:  # pytype: disable=signature-mismatch
    """Returns an iterator of keys in current dict."""
    return self.sym_keys()

  def items(self) -> Iterator[Tuple[str, Any]]:  # pytype: disable=signature-mismatch
    """Returns an iterator of (key, value) items in current dict."""
    return self.sym_items()

  def values(self) -> Iterator[Any]:  # pytype: disable=signature-mismatch
    """Returns an iterator of values in current dict.."""
    return self.sym_values()

  def copy(self) -> 'Dict':
    """Overridden copy using symbolic copy."""
    return self.sym_clone(deep=False)

  def pop(
      self, key: Any, default: Any = base.RAISE_IF_NOT_FOUND  # pylint: disable=protected-access
  ) -> Any:
    """Pops a key from current dict."""
    if key in self:
      value = self[key]
      with flags.allow_writable_accessors(True):
        del self[key]
      return value if value != pg_typing.MISSING_VALUE else default
    if default is base.RAISE_IF_NOT_FOUND:
      raise KeyError(key)
    return default

  def popitem(self) -> Tuple[str, Any]:
    if self._value_spec is not None:
      raise ValueError(
          '\'popitem\' cannot be performed on a Dict with value spec.')
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot pop item from a sealed Dict.')
    return super().popitem()

  def clear(self) -> None:
    """Removes all the keys in current dict."""
    if base.treats_as_sealed(self):
      raise base.WritePermissionError('Cannot clear a sealed Dict.')
    value_spec = self._value_spec
    self._value_spec = None
    super().clear()

    if value_spec:
      self.use_value_spec(value_spec, self._allow_partial)

  def setdefault(self, key: str, default: Any = None) -> Any:
    """Sets default as the value to key if not present."""
    value = pg_typing.MISSING_VALUE
    if key in self:
      value = self.sym_getattr(key)
    if value == pg_typing.MISSING_VALUE:
      self[key] = default
      value = default
    return value

  def update(self,
             other: Union[
                 None,
                 typing.Dict[str, Any],
                 Iterable[Tuple[str, Any]]] = None,
             **kwargs) -> None:  # pytype: disable=signature-mismatch
    """Update Dict with the same semantic as update on standard dict."""
    updates = dict(other) if other else {}
    updates.update(kwargs)
    self.rebind(
        updates, raise_on_no_change=False, skip_notification=True)

  def sym_jsonify(
      self,
      hide_default_values: bool = False,
      exclude_keys: Optional[Sequence[str]] = None,
      use_inferred: bool = False,
      **kwargs) -> object_utils.JSONValueType:
    """Converts current object to a dict with plain Python objects."""
    exclude_keys = set(exclude_keys or [])
    if self._value_spec and self._value_spec.schema:
      json_repr = dict()
      matched_keys, _ = self._value_spec.schema.resolve(self.keys())  # pytype: disable=attribute-error
      for key_spec, keys in matched_keys.items():
        # NOTE(daiyip): The key values of frozen field can safely be excluded
        # since they will be the same for a class.
        field = self._value_spec.schema[key_spec]
        if not field.frozen:
          for key in keys:
            if key not in exclude_keys:
              value = self.sym_getattr(key)
              if use_inferred and isinstance(value, base.Inferential):
                value = self.sym_inferred(key, default=value)
              if pg_typing.MISSING_VALUE == value:
                continue
              if hide_default_values and base.eq(value, field.default_value):
                continue
              json_repr[key] = base.to_json(
                  value, hide_default_values=hide_default_values,
                  use_inferred=use_inferred,
                  **kwargs)
      return json_repr
    else:
      return {
          k: base.to_json(
              self.sym_inferred(k, default=v) if (
                  use_inferred and isinstance(v, base.Inferential)) else v,
              hide_default_values=hide_default_values,
              use_inferred=use_inferred,
              **kwargs)
          for k, v in self.sym_items()
          if k not in exclude_keys
      }

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[
          Callable[[object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
  ) -> Tuple[bool, 'Dict']:
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
                f'Dict (spec={self._value_spec!r}) cannot be assigned to an '
                f'incompatible field (spec={value_spec!r}).', path))
      if self._allow_partial == allow_partial:
        proceed_with_standard_apply = False
      else:
        self._allow_partial = allow_partial
    elif isinstance(value_spec, pg_typing.Dict):
      self._value_spec = value_spec
    return (proceed_with_standard_apply, self)

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      python_format: bool = False,
      markdown: bool = False,
      hide_default_values: bool = False,
      hide_missing_values: bool = False,
      include_keys: Optional[Set[str]] = None,
      exclude_keys: Optional[Set[str]] = None,
      use_inferred: bool = False,
      cls_name: Optional[str] = None,
      bracket_type: object_utils.BracketType = object_utils.BracketType.CURLY,
      key_as_attribute: bool = False,
      extra_blankline_for_field_docstr: bool = False,
      **kwargs,
  ) -> str:
    """Formats this Dict."""
    cls_name = cls_name or ''
    exclude_keys = exclude_keys or set()
    def _indent(text, indent):
      return ' ' * 2 * indent + text

    def _should_include_key(key):
      if include_keys:
        return key in include_keys
      return key not in exclude_keys

    field_list = []
    if self._value_spec and self._value_spec.schema:
      matched_keys, unmatched = self._value_spec.schema.resolve(self.keys())  # pytype: disable=attribute-error
      assert not unmatched
      for key_spec, keys in matched_keys.items():
        for key in keys:
          if _should_include_key(key):
            field = self._value_spec.schema[key_spec]
            v = self.sym_getattr(key)
            if use_inferred and isinstance(v, base.Inferential):
              v = self.sym_inferred(key, default=v)
            if pg_typing.MISSING_VALUE == v:
              if hide_missing_values:
                continue
            elif hide_default_values and base.eq(v, field.default_value):
              continue
            field_list.append((field, key, v))
    else:
      for k, v in self.sym_items():
        if _should_include_key(k):
          if use_inferred and isinstance(v, base.Inferential):
            v = self.sym_inferred(k, default=v)
          field_list.append((None, k, v))

    open_bracket, close_bracket = object_utils.bracket_chars(bracket_type)
    if not field_list:
      return object_utils.maybe_markdown_quote(
          f'{cls_name}{open_bracket}{close_bracket}', markdown
      )

    if compact:
      s = [f'{cls_name}{open_bracket}']
      kv_strs = []
      for _, k, v in field_list:
        v_str = object_utils.format(
            v,
            compact,
            verbose,
            root_indent + 1,
            hide_default_values=hide_default_values,
            hide_missing_values=hide_missing_values,
            python_format=python_format,
            use_inferred=use_inferred,
            extra_blankline_for_field_docstr=extra_blankline_for_field_docstr,
            **kwargs)
        if not python_format or key_as_attribute:
          kv_strs.append(f'{k}={v_str}')
        else:
          kv_strs.append(f'\'{k}\': {v_str}')

      s.append(', '.join(kv_strs))
      s.append(close_bracket)
    else:
      s = [f'{cls_name}{open_bracket}\n']
      for i, (f, k, v) in enumerate(field_list):
        if i != 0:
          s.append(',\n')

        if verbose and f and typing.cast(pg_typing.Field, f).description:
          if i != 0 and extra_blankline_for_field_docstr:
            s.append('\n')
          description = typing.cast(pg_typing.Field, f).description
          for line in description.split('\n'):
            s.append(_indent(f'# {line}\n', root_indent + 1))
        v_str = object_utils.format(
            v,
            compact,
            verbose,
            root_indent + 1,
            hide_default_values=hide_default_values,
            hide_missing_values=hide_missing_values,
            python_format=python_format,
            use_inferred=use_inferred,
            extra_blankline_for_field_docstr=extra_blankline_for_field_docstr,
            **kwargs)
        if not python_format:
          # Format in PyGlove's format (default).
          s.append(_indent(f'{k} = {v_str}', root_indent + 1))
        elif key_as_attribute:
          # Format `pg.Objects` under Python format.
          s.append(_indent(f'{k}={v_str}', root_indent + 1))
        else:
          # Format regular `pg.Dict` under Python format.
          s.append(_indent(f'\'{k}\': {v_str}', root_indent + 1))
      s.append('\n')
      s.append(_indent(close_bracket, root_indent))
    return object_utils.maybe_markdown_quote(''.join(s), markdown)

  def __repr__(self) -> str:
    """Operator repr()."""
    return base.Symbolic.__repr__(self)

  def __eq__(self, other: Any) -> bool:
    """Operator ==."""
    if isinstance(other, dict):
      return dict.__eq__(self, other)
    return False

  def __ne__(self, other: Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)

  def __hash__(self) -> int:
    """Overridden hashing function using symbolic hash."""
    return self.sym_hash()


base.Symbolic.DictType = Dict
