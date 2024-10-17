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
"""Symbolic type base."""

import abc
import copy
import enum
import inspect
import json
import os
import re
import sys
import typing
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union

from pyglove.core import io as pg_io
from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import flags
from pyglove.core.symbolic.origin import Origin
from pyglove.core.symbolic.pure_symbolic import NonDeterministic
from pyglove.core.symbolic.pure_symbolic import PureSymbolic
from pyglove.core.views import html


class WritePermissionError(Exception):
  """Exception raisen when write access to object fields is not allowed."""


class FieldUpdate(object_utils.Formattable):
  """Class that describes an update to a field in an object tree."""

  def __init__(self,
               path: object_utils.KeyPath,
               target: 'Symbolic',
               field: Optional[pg_typing.Field],
               old_value: Any,
               new_value: Any):
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

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs,
  ) -> str:
    """Formats this object."""
    return object_utils.kvlist_str(
        [
            ('parent_path', self.target.sym_path, None),
            ('path', self.path, None),
            ('old_value', self.old_value, object_utils.MISSING_VALUE),
            ('new_value', self.new_value, object_utils.MISSING_VALUE),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        **kwargs
    )

  def __eq__(self, other: Any) -> bool:
    """Operator ==."""
    if not isinstance(other, self.__class__):
      return False
    return (self.path == other.path and self.target is other.target and
            self.field is other.field and self.old_value == other.old_value and
            self.new_value == other.new_value)

  def __ne__(self, other: Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)


class DescendantQueryOption(enum.Enum):
  """Options for querying descendants through `sym_descendant`."""

  # Returning all matched descendants.
  ALL = 0

  # Returning only the immediate matched descendants.
  IMMEDIATE = 1

  # Returning only the leaf matched descendants.
  LEAF = 2


class TopologyAware(metaclass=abc.ABCMeta):
  """Interface for objects that are aware of the topology it is part of."""

  @property
  @abc.abstractmethod
  def sym_parent(self) -> Optional['TopologyAware']:
    """Returns the parent of this object."""

  @abc.abstractmethod
  def sym_setparent(self, parent: Optional['TopologyAware']) -> None:
    """Sets the parent of this object."""

  @property
  @abc.abstractmethod
  def sym_path(self) -> object_utils.KeyPath:
    """Returns the path of this object under its topology."""

  @abc.abstractmethod
  def sym_setpath(self, path: object_utils.KeyPath) -> None:
    """Sets the path of this object under its topology."""


class Inferential(TopologyAware, pg_typing.CustomTyping):
  """Interface for values that could be dynamically inferred upon read.

  Inferential values are objects whose values are not determined directly but
  are instead derived from other sources, such as references (:class:`pg.Ref`)
  to other objects or computed based on their context
  (:class:`pg.symbolic.ValueFromParentChain`) such as the symbolic tree they
  reside in.

  When inferential values are utilized as symbolic attributes, we can obtain
  their original definition by invoking :meth:`pg.Symbolic.sym_getattr`, and
  their inferred values can be retrieved by calling
  :meth:`pg.Symbolic.sym_inferred`. The values retrieved from :class:`pg.Dict`,
  :class:`pg.List` and :class:`pg.Object` through `__getitem__` or
  `__getattribute__` are all inferred values.
  """

  @abc.abstractmethod
  def infer(self, **kwargs) -> Any:
    """Returns the inferred value.

    Args:
      **kwargs: Optional keyword arguments for inference, which are usually
        inferential subclass specific.

    Returns:
      Inferred value.

    Raises:
      AttributeError: If the value cannot be inferred.
    """


# Value marker for raising errors if a attribute does not exist or cannot
# be computed.
RAISE_IF_NOT_FOUND = (pg_typing.MISSING_VALUE,)


class Symbolic(
    TopologyAware,
    object_utils.Formattable,
    object_utils.JSONConvertible,
    object_utils.MaybePartial,
    html.HtmlTreeView.Extension
):
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

  # Do not include comments in str output.
  __str_format_kwargs__ = dict(compact=False, verbose=False)

  # Symbolic sub-types that will be set when they are defined.
  # pylint: disable=invalid-name

  DictType = None
  ListType = None
  ObjectType = None

  # pylint: enable=invalid-name

  def __init__(self,
               *,
               allow_partial: bool,
               accessor_writable: bool,
               sealed: bool,
               root_path: Optional[object_utils.KeyPath],
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

    origin = Origin(None, '__init__') if flags.is_tracking_origin() else None
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

  def _init_kwargs(self) -> Dict[str, Any]:
    kwargs = {}
    def add_if_nondefault(key, attrname, default):
      v = getattr(self, attrname)
      if v != default:
        kwargs[key] = v

    add_if_nondefault('allow_partial', '_allow_partial', False)
    add_if_nondefault('sealed', '_sealed', False)
    return kwargs

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
    return bool(self.sym_missing(flatten=False))

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

  def sym_missing(self, flatten: bool = True) -> Dict[str, Any]:
    """Returns missing values."""
    missing = getattr(self, '_sym_missing_values')
    if missing is None:
      missing = self._sym_missing()
      self._set_raw_attr('_sym_missing_values', missing)
    if flatten:
      missing = object_utils.flatten(missing)
    return missing

  def sym_nondefault(self, flatten: bool = True) -> Dict[Union[int, str], Any]:
    """Returns missing values."""
    nondefault = getattr(self, '_sym_nondefault_values')
    if nondefault is None:
      nondefault = self._sym_nondefault()
      self._set_raw_attr('_sym_nondefault_values', nondefault)
    if flatten:
      nondefault = object_utils.flatten(nondefault)
    return nondefault

  @property
  def sym_field(self) -> Optional[pg_typing.Field]:
    """Returns the symbolic field for current object."""
    if self.sym_parent is None:
      return None
    return self.sym_parent.sym_attr_field(self.sym_path.key)

  @property
  def sym_root(self) -> 'Symbolic':
    """Returns the root of the symbolic tree."""
    root = self
    while root.sym_parent is not None:
      root = root.sym_parent
    return root

  def sym_ancestor(
      self,
      where: Optional[Callable[[Any], bool]] = None,
      ) -> Optional['Symbolic']:
    """Returns the nearest ancestor of specific classes."""
    ancestor = self.sym_parent
    where = where or (lambda x: True)
    while ancestor is not None and not where(ancestor):
      ancestor = ancestor.sym_parent
    return ancestor

  def sym_descendants(
      self,
      where: Optional[Callable[[Any], bool]] = None,
      option: DescendantQueryOption = DescendantQueryOption.ALL,
      include_self: bool = False) -> List[Any]:
    """Returns all descendants of specific classes.

    Args:
      where: Optional callable object as the filter of descendants to return.
      option: Descendant query options, indicating whether all matched,
        immediate matched or only the matched leaf nodes will be returned.
      include_self: If True, `self` will be included in the query, otherwise
        only strict descendants are included.

    Returns:
      A list of objects that match the descendant_cls.
    """
    descendants = []
    where = where or (lambda x: True)

    def visit(k, v, p):
      del k, p
      if not where(v):
        return TraverseAction.ENTER

      if not include_self and self is v:
        return TraverseAction.ENTER

      if option == DescendantQueryOption.IMMEDIATE:
        descendants.append(v)
        return TraverseAction.CONTINUE

      # Dealing with option = ALL or LEAF.
      leaf_descendants = []
      if isinstance(v, Symbolic):
        leaf_descendants = v.sym_descendants(where, option)

      if option is DescendantQueryOption.ALL or not leaf_descendants:
        descendants.append(v)
      descendants.extend(leaf_descendants)
      return TraverseAction.CONTINUE

    traverse(self, visit)
    return descendants

  @abc.abstractmethod
  def sym_attr_field(self, key: Union[str, int]) -> Optional[pg_typing.Field]:
    """Returns the field definition for a symbolic attribute."""

  def sym_has(self, path: Union[object_utils.KeyPath, str, int]) -> bool:
    """Returns True if a path exists in the sub-tree.

    Args:
      path: A KeyPath object or equivalence.

    Returns:
      True if the path exists in current sub-tree, otherwise False.
    """
    return object_utils.KeyPath.from_value(path).exists(self)

  def sym_get(
      self,
      path: Union[object_utils.KeyPath, str, int],
      default: Any = RAISE_IF_NOT_FOUND,
      use_inferred: bool = False) -> Any:
    """Returns a sub-node by path.

    NOTE: there is no `sym_set`, use `sym_rebind`.

    Args:
      path: A KeyPath object or equivalence.
      default: Default value if path does not exists. If absent, `KeyError` will
        be thrown.
      use_inferred: If True, return inferred value instead of the symbolic form
        of `pg.Inferential` objects.

    Returns:
      Value of symbolic attribute specified by path if found, otherwise the
      default value if it's specified.

    Raises:
      KeyError if `path` does not exist and `default` is not specified.
    """
    path = object_utils.KeyPath.from_value(path)
    if default is RAISE_IF_NOT_FOUND:
      return path.query(self, use_inferred=use_inferred)
    else:
      return path.get(self, default, use_inferred=use_inferred)

  @abc.abstractmethod
  def sym_hasattr(self, key: Union[str, int]) -> bool:
    """Returns if a symbolic attribute exists."""

  def sym_getattr(
      self, key: Union[str, int], default: Any = RAISE_IF_NOT_FOUND
  ) -> Any:
    """Gets a symbolic attribute.

    Args:
      key: Key of symbolic attribute.
      default: Default value if attribute does not exist. If absent,

    Returns:
      Value of symbolic attribute if found, otherwise the default value
      if it's specified.

    Raises:
      AttributeError if `key` does not exist and `default` is not provided.
    """
    if not self.sym_hasattr(key):
      if default is RAISE_IF_NOT_FOUND:
        raise AttributeError(
            self._error_message(
                f'{self.__class__!r} object has no symbolic attribute {key!r}.'
            )
        )
      return default
    return self._sym_getattr(key)

  def sym_inferrable(self, key: Union[str, int], **kwargs) -> bool:
    """Returns True if the attribute under key can be inferred."""
    return (
        self.sym_inferred(key, pg_typing.MISSING_VALUE, **kwargs)
        != pg_typing.MISSING_VALUE
    )

  def sym_inferred(
      self,
      key: Union[str, int],
      default: Any = RAISE_IF_NOT_FOUND,
      **kwargs,
  ) -> Any:
    """Returns the inferred value of the attribute under key."""
    if default is RAISE_IF_NOT_FOUND:
      return self._sym_inferred(key, **kwargs)
    else:
      try:
        return self._sym_inferred(key, **kwargs)
      except Exception:  # pylint: disable=broad-exception-caught
        return default

  def _sym_inferred(self, key: Union[str, int], **kwargs) -> Any:
    v = self.sym_getattr(key)
    if isinstance(v, Inferential):
      v = v.infer(**kwargs)
    return v

  @abc.abstractmethod
  def sym_keys(self) -> Iterator[Union[str, int]]:
    """Iterates the keys of symbolic attributes."""

  @abc.abstractmethod
  def sym_values(self) -> Iterator[Any]:
    """Iterates the values of symbolic attributes."""

  @abc.abstractmethod
  def sym_items(self) -> Iterator[Tuple[Union[str, int], Any]]:
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
      value: Any = None,
      type: Union[None, Type[Any], Tuple[Type[Any], ...]] = None   # pylint: disable=redefined-builtin
      ) -> bool:
    """Returns True if the object contains sub-nodes of given value or type."""
    return contains(self, value, type)

  @property
  def sym_path(self) -> object_utils.KeyPath:
    """Returns the path of current object from the root of its symbolic tree."""
    return getattr(self, '_sym_path')

  def sym_setpath(
      self, path: Optional[Union[str, object_utils.KeyPath]]) -> None:
    """Sets the path of current node in its symbolic tree."""
    if self.sym_path != path:
      old_path = self.sym_path
      self._set_raw_attr('_sym_path', path)
      self._update_children_paths(old_path, path)

  def sym_rebind(
      self,
      path_value_pairs: Optional[Union[
          Dict[
              Union[object_utils.KeyPath, str, int],
              Any],
          Callable]] = None,  # pylint: disable=g-bare-generic
      *,
      raise_on_no_change: bool = True,
      notify_parents: bool = True,
      skip_notification: Optional[bool] = None,
      **kwargs,
  ) -> 'Symbolic':
    """Mutates the sub-nodes of current object. Please see `rebind`."""
    assert Symbolic.DictType is not None
    if callable(path_value_pairs):
      path_value_pairs = get_rebind_dict(path_value_pairs, self)
    elif path_value_pairs is None:
      path_value_pairs = {}
    elif isinstance(path_value_pairs, Symbolic.DictType):
      # Rebind work on symbolic form, thus we get their symbol instead of
      # their evaluated value when building the rebind dict.
      sd = typing.cast(Symbolic.DictType, path_value_pairs)
      path_value_pairs = {k: v for k, v in sd.sym_items()}
    if not isinstance(path_value_pairs, dict):
      raise ValueError(
          self._error_message(
              f'Argument \'path_value_pairs\' should be a dict. '
              f'Encountered {path_value_pairs}'))
    path_value_pairs.update(kwargs)
    path_value_pairs = {object_utils.KeyPath.from_value(k): v
                        for k, v in path_value_pairs.items()}

    if not path_value_pairs and raise_on_no_change:
      raise ValueError(self._error_message('There are no values to rebind.'))
    updates = self._sym_rebind(path_value_pairs)
    if skip_notification is None:
      skip_notification = not flags.is_change_notification_enabled()
    if not skip_notification:
      self._notify_field_updates(updates, notify_parents=notify_parents)
    return self

  def sym_clone(self,
                deep: bool = False,
                memo: Optional[Any] = None,
                override: Optional[Dict[str, Any]] = None):
    """Clones current object symbolically."""
    assert deep or not memo
    new_value = self._sym_clone(deep, memo)
    if override:
      new_value.sym_rebind(override, raise_on_no_change=False)
    if flags.is_tracking_origin():
      new_value.sym_setorigin(self, 'deepclone' if deep else 'clone')
    return new_value

  @abc.abstractmethod
  def sym_jsonify(self,
                  *,
                  hide_default_values: bool = False,
                  **kwargs) -> object_utils.JSONValueType:
    """Converts representation of current object to a plain Python object."""

  def sym_ne(self, other: Any) -> bool:
    """Returns if this object does not equal to another object symbolically."""
    return ne(self, other)

  def sym_eq(self, other: Any) -> bool:
    """Returns if this object equals to another object symbolically."""
    return eq(self, other)

  def sym_gt(self, other: Any) -> bool:
    """Returns if this object is symbolically greater than another object."""
    return gt(self, other)

  def sym_lt(self, other: Any) -> bool:
    """Returns True if this object is symbolically less than other object."""
    return lt(self, other)

  @abc.abstractmethod
  def sym_hash(self) -> int:
    """Computes the symbolic hash of current object."""

  @property
  def sym_origin(self) -> Optional[Origin]:
    """Returns the symbolic origin of current object."""
    return getattr(self, '_sym_origin')

  def sym_setorigin(
      self,
      source: Any,
      tag: str,
      stacktrace: Optional[bool] = None,
      stacklimit: Optional[int] = None,
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
        determined by the value passed to `pg.set_origin_stacktrace_limit`,
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
      current_source = self.sym_origin.source  # pytype: disable=attribute-error  # always-use-property-annotation
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

  def missing_values(self, flatten: bool = True) -> Dict[str, Any]:
    """Alias for `sym_missing`."""
    return self.sym_missing(flatten)

  def non_default_values(
      self, flatten: bool = True) -> Dict[Union[int, str], Any]:
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
      path_value_pairs: Optional[Union[
          Dict[
              Union[object_utils.KeyPath, str, int],
              Any],
          Callable]] = None,  # pylint: disable=g-bare-generic
      *,
      raise_on_no_change: bool = True,
      notify_parents: bool = True,
      skip_notification: Optional[bool] = None,
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
      d = pg.Dict(value_spec=pg.typing.Dict([
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
        value_spec = pg.typing.List(pg.typing.Dict([
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
      notify_parents: If True (default), parents will be notified upon change.
        Otherwisee only the current object and the impacted children will
        be notified. A most common use case for setting this flag to False
        is when users want to rebind a child within the parent `_on_bound`
        method.
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
        path_value_pairs,
        raise_on_no_change=raise_on_no_change,
        notify_parents=notify_parents,
        skip_notification=skip_notification,
        **kwargs)

  def clone(
      self,
      deep: bool = False,
      memo: Optional[Any] = None,
      override: Optional[Dict[str, Any]] = None
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
    return to_json(self, **kwargs)

  def to_json_str(self, json_indent: Optional[int] = None, **kwargs) -> str:
    """Serializes current object into a JSON string."""
    return to_json_str(self, json_indent=json_indent, **kwargs)

  def _html_tree_view_content(
      self,
      *,
      view: html.HtmlTreeView,
      name: Optional[str] = None,
      parent: Any = None,
      root_path: Optional[object_utils.KeyPath] = None,
      extra_flags: Optional[Dict[str, Any]],
      **kwargs,
  ) -> html.Html:
    """Returns the content HTML for a symbolic object.."""
    extra_flags = extra_flags or {}
    hide_frozen = extra_flags.get('hide_frozen', True)
    hide_default_values = extra_flags.get('hide_default_values', False)
    use_inferred = extra_flags.get('use_inferred', False)

    kv = {}
    for k, v in self.sym_items():
      # Apply frozen filter.
      field = self.sym_attr_field(k)
      if hide_frozen and field and field.frozen:
        continue

      # Apply inferred value.
      if use_inferred and isinstance(v, Inferential):
        v = self.sym_inferred(k, default=v)

      # Apply default value filter.
      if field and hide_default_values and eq(v, field.default_value):
        continue
      kv[k] = v
    return view.complex_value(
        kv, name=name, parent=self, root_path=root_path,
        extra_flags=extra_flags, **kwargs
    )

  @classmethod
  def load(cls, *args, **kwargs) -> Any:
    """Loads an instance of this type using the global load handler."""
    value = load(*args, **kwargs)
    if not isinstance(value, cls):
      raise TypeError(f'Value is not of type {cls!r}: {value!r}.')
    return value

  def save(self, *args, **kwargs) -> Any:
    """Saves current object using the global save handler."""
    return save(self, *args, **kwargs)

  def inspect(
      self,
      path_regex: Optional[str] = None,
      where: Optional[Union[Callable[[Any], bool],
                            Callable[[Any, Any], bool]]] = None,
      custom_selector: Optional[Union[
          Callable[[object_utils.KeyPath, Any], bool],
          Callable[[object_utils.KeyPath, Any, Any], bool]]] = None,
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
    object_utils.print(v, file=file, **kwargs)

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
  def _sym_rebind(
      self, path_value_pairs: Dict[object_utils.KeyPath, Any]
      ) -> List[FieldUpdate]:
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

  @abc.abstractmethod
  def _sym_missing(self) -> Dict[str, Any]:
    """Returns missing values."""

  @abc.abstractmethod
  def _sym_nondefault(self) -> Dict[Union[int, str], Any]:
    """Returns non-default values."""

  @abc.abstractmethod
  def _sym_getattr(self, key: Union[str, int]) -> Any:
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
  def _set_item_without_permission_check(
      self, key: Union[str, int], value: Any) -> Optional[FieldUpdate]:
    """Child should implement: set an item without permission check."""

  @abc.abstractmethod
  def _on_change(self, field_updates: Dict[object_utils.KeyPath, FieldUpdate]):
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

  def _set_raw_attr(self, name: str, value: Any) -> 'Symbolic':
    """Set raw property without trigger __setattr__."""
    # `object.__setattr__` adds a property to the instance without side effects.
    object.__setattr__(self, name, value)
    return self

  def _relocate_if_symbolic(self, key: Union[str, int], value: Any) -> Any:
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

    if isinstance(value, TopologyAware):
      value.sym_setpath(object_utils.KeyPath(key, self.sym_path))
      value.sym_setparent(self._sym_parent_for_children())
    return value

  def _sym_parent_for_children(self) -> Optional['Symbolic']:
    """Returns the symbolic parent for children."""
    return self

  def _set_item_of_current_tree(
      self, path: object_utils.KeyPath, value: Any) -> Optional[FieldUpdate]:
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

    if treats_as_sealed(parent_node):
      raise WritePermissionError(
          f'Cannot rebind key {path.key!r} of '
          f'sealed {parent_node.__class__.__name__}: {parent_node!r}. '
          f'(path=\'{path.parent}\')')
    return parent_node._set_item_without_permission_check(path.key, value)  # pylint: disable=protected-access

  def _notify_field_updates(
      self,
      field_updates: List[FieldUpdate],
      notify_parents: bool = True) -> None:
    """Notify field updates."""
    per_target_updates = dict()

    def _get_target_updates(
        target: 'Symbolic'
    ) -> Dict[object_utils.KeyPath, FieldUpdate]:
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

      # If `notify_parents` is set to False, stop notifications once `self`
      # is processed.
      if target is self and not notify_parents:
        break

  def _error_message(self, message: str) -> str:
    """Create error message to include path information."""
    return object_utils.message_on_path(message, self.sym_path)


#
# Function for rebinders.
#


def get_rebind_dict(
    rebinder: Callable,  # pylint: disable=g-bare-generic
    target: Symbolic
) -> Dict[str, Any]:
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
  signature = pg_typing.signature(
      rebinder, auto_typing=False, auto_doc=False
  )
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


def traverse(x: Any,
             preorder_visitor_fn: Optional[
                 Callable[[object_utils.KeyPath, Any, Any],
                          Optional[TraverseAction]]] = None,
             postorder_visitor_fn: Optional[
                 Callable[[object_utils.KeyPath, Any, Any],
                          Optional[TraverseAction]]] = None,
             root_path: Optional[object_utils.KeyPath] = None,
             parent: Optional[Any] = None) -> bool:
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
    elif isinstance(x, Symbolic.ObjectType):  # pytype: disable=wrong-arg-types
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


def query(
    x: Any,
    path_regex: Optional[str] = None,
    where: Optional[Union[Callable[[Any], bool],
                          Callable[[Any, Any], bool]]] = None,
    enter_selected: bool = False,
    custom_selector: Optional[Union[
        Callable[[object_utils.KeyPath, Any], bool],
        Callable[[object_utils.KeyPath, Any, Any], bool]]] = None
) -> Dict[str, Any]:
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
    signature = pg_typing.signature(
        custom_selector, auto_typing=False, auto_doc=False
    )
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
      signature = pg_typing.signature(where)
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

  def _preorder_visitor(path: object_utils.KeyPath, v: Any,
                        parent: Any) -> TraverseAction:
    if select_fn(path, v, parent):  # pytype: disable=wrong-arg-count
      results[str(path)] = v
      return TraverseAction.ENTER if enter_selected else TraverseAction.CONTINUE
    return TraverseAction.ENTER

  traverse(x, preorder_visitor_fn=_preorder_visitor)
  return results


def eq(left: Any, right: Any) -> bool:
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
      or (isinstance(left, tuple) and isinstance(right, tuple))):
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
    # NOTE(daiyip): pg.Dict.__getitem__ will trigger inferred value
    # evaluation, therefore we always get its symbolic form during traversal.
    left_items = left.sym_items if isinstance(left, Symbolic) else left.items
    right_item = (
        right.sym_getattr if isinstance(right, Symbolic) else right.__getitem__)
    for k, v in left_items():
      if ne(v, right_item(k)):
        return False
    return True
  # We compare sym_eq with Symbolic.sym_eq to avoid endless recursion.
  elif (hasattr(left, 'sym_eq')
        and not inspect.isclass(left)
        and left.sym_eq.__code__ is not Symbolic.sym_eq.__code__):
    return left.sym_eq(right)
  elif (hasattr(right, 'sym_eq')
        and not inspect.isclass(right)
        and right.sym_eq.__code__ is not Symbolic.sym_eq.__code__):
    return right.sym_eq(left)
  # Compare two maybe callable objects.
  return pg_typing.callable_eq(left, right)


def ne(left: Any, right: Any) -> bool:
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


def lt(left: Any, right: Any) -> bool:
  """Returns True if a value is symbolically less than the other value.

  Symbolic values are comparable by their symbolic representations. For common
  types such as numbers and string, symbolic comparison returns the same value
  as value comparisons. For example::

    assert pg.lt(False, True) == Flase < True
    assert pg.lt(0.1, 1) == 0.1 < 1
    assert pg.lt('a', 'ab') == 'a' < 'ab'

  However, symbolic comparison can be applied on hierarchical values, for
  example::

    assert pg.lt(['a'], ['a', 'b'])
    assert pg.lt(['a', 'b', 'c'], ['b'])
    assert pg.lt({'x': 1}, {'x': 2})
    assert pg.lt({'x': 1}, {'y': 1})
    assert pg.lt(A(x=1), A(x=2))

  Also, symbolic values of different types can be compared, for example::

    assert pg.lt(pg.MISSING_VALUE, None)
    assert pg.lt(None, 1)
    assert pg.lt(1, 'abc')
    assert pg.lt('abc', [])
    assert pg.lt([], {})
    assert pg.lt([], A(x=1))

  The high-level idea is that a value with lower information entropy is less
  than a value with higher information entropy. As a result, we know that
  `pg.MISSING_VALUE` is the smallest among all values.

  The order of symbolic representation are defined by the following rules:

  1) If x and y are comparable by their values, they will be compared using
     operator <. (e.g. bool, int, float, str)
  2) If x and y are not directly comparable and are different in their types,
     they will be compared based on their types. The order of different types
     are: pg.MISSING_VALUE, NoneType, bool, int, float, str, list, tuple, set,
     dict, functions/classes. When different functions/classes compare, their
     order is determined by their qualified name.
  3) If x and y are of the same type, which are symbolic containers (e.g. list,
     dict, pg.Symbolic objects), their order will be determined by the order of
     their first sub-nodes which are different. Therefore ['b'] is greater than
     ['a', 'b'], though the later have 2 elements.
  4) Non-symbolic classes can define method `sym_lt` to enable symbolic
     comparison.

  Args:
    left: The left-hand value to compare.
    right: The right-hand value to compare.

  Returns:
    True if the left value is symbolically less than the right value.
  """
  # A fast type check can eliminate most
  if type(left) is not type(right):
    tol = _type_order(left)
    tor = _type_order(right)
    # When tol == tor, this means different types are treated as same symbols.
    # E.g. list and pg.List.
    if tol != tor:
      return tol < tor

  # Most symbolic nodes are leaf, which are primitive types, therefore
  # we detect such types to make `lt` to run faster.
  if isinstance(left, (int, float, bool, str)):
    return left < right
  elif isinstance(left, list):
    min_len = min(len(left), len(right))
    for i in range(min_len):
      l, r = left[i], right[i]
      if not eq(l, r):
        return lt(l, r)
    # `left` and `right` are equal so far, so `left` is less than `right`
    # only when left has a smaller length.
    return len(left) < len(right)
  elif isinstance(left, dict):
    lkeys = list(left.keys())
    rkeys = list(right.keys())
    min_len = min(len(lkeys), len(rkeys))
    for i in range(min_len):
      kl, kr = lkeys[i], rkeys[i]
      if kl == kr:
        if not eq(left[kl], right[kr]):
          return lt(left[kl], right[kr])
      else:
        return kl < kr
    # `left` and `right` are equal so far, so `left is less than `right`
    # only when left has fewer keys.
    return len(lkeys) < len(rkeys)
  elif hasattr(left, 'sym_lt'):
    return left.sym_lt(right)
  return left < right


def gt(left: Any, right: Any) -> bool:
  """Returns True if a value is symbolically greater than the other value.

  Refer to :func:`pyglove.lt` for the definition of symbolic comparison.

  Args:
    left: The left-hand value to compare.
    right: The right-hand value to compare.

  Returns:
    True if the left value is symbolically greater than the right value.
  """
  return lt(right, left)   # pylint: disable=arguments-out-of-order


def _type_order(value: Any) -> str:
  """Returns the ordering string of value's type."""
  if isinstance(value, object_utils.MissingValue):
    type_order = 0
  elif value is None:
    type_order = 1
  elif isinstance(value, (bool, int, float)):
    type_order = 2
  elif isinstance(value, str):
    type_order = 3
  elif isinstance(value, list):
    type_order = 4
  elif isinstance(value, tuple):
    type_order = 5
  elif isinstance(value, set):
    type_order = 6
  elif isinstance(value, dict):
    type_order = 7
  else:
    type_order = type(value).__qualname__
  return str(type_order)


def sym_hash(x: Any) -> int:
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
  if inspect.isfunction(x):
    return hash(x.__code__.co_code)
  if inspect.ismethod(x):
    return hash((sym_hash(x.__self__), x.__code__.co_code))  # pytype: disable=attribute-error
  return hash(x)


def clone(
    x: Any,
    deep: bool = False,
    memo: Optional[Any] = None,
    override: Optional[Dict[str, Any]] = None
) -> Any:
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
  elif isinstance(x, list):
    assert not override, override
    return [clone(v, deep, memo) for v in x]
  elif isinstance(x, tuple):
    assert not override, override
    return tuple([clone(v, deep, memo) for v in x])
  elif isinstance(x, dict):
    assert not override, override
    return {k: clone(v, deep, memo) for k, v in x.items()}
  else:
    assert not override, override
    return copy.deepcopy(x, memo) if deep else copy.copy(x)


def is_deterministic(x: Any) -> bool:
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


def is_pure_symbolic(x: Any) -> bool:
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


def is_abstract(x: Any) -> bool:
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
    x: Any,
    value: Any = None,
    type: Optional[Union[    # pylint: disable=redefined-builtin
        Type[Any],
        Tuple[Type[Any]]]]=None
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


def from_json(json_value: Any,
              *,
              allow_partial: bool = False,
              root_path: Optional[object_utils.KeyPath] = None,
              auto_import: bool = True,
              auto_dict: bool = False,
              **kwargs) -> Any:
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
    auto_import: If True, when a '_type' is not registered, PyGlove will
      identify its parent module and automatically import it. For example,
      if the type is 'foo.bar.A', PyGlove will try to import 'foo.bar' and
      find the class 'A' within the imported module.
    auto_dict: If True, dict with '_type' that cannot be loaded will remain
      as dict, with '_type' renamed to 'type_name'.
    **kwargs: Allow passing through keyword arguments to from_json of specific
      types.

  Returns:
    Deserialized value, which is
    * pg.Dict for dict.
    * pg.List for list.
    * symbolic.Object for dict with '_type' property.
    * value itself.
  """
  assert Symbolic.DictType is not None
  if isinstance(json_value, Symbolic):
    return json_value

  typename_resolved = kwargs.pop('_typename_resolved', False)
  if not typename_resolved:
    json_value = object_utils.json_conversion.resolve_typenames(
        json_value, auto_import=auto_import, auto_dict=auto_dict
    )

  kwargs.update({
      'allow_partial': allow_partial,
      'root_path': root_path,
  })
  if isinstance(json_value, list):
    if (json_value
        and json_value[0] == object_utils.JSONConvertible.TUPLE_MARKER):
      if len(json_value) < 2:
        raise ValueError(
            object_utils.message_on_path(
                f'Tuple should have at least one element '
                f'besides \'{object_utils.JSONConvertible.TUPLE_MARKER}\'. '
                f'Encountered: {json_value}', root_path))
      kwargs.pop('root_path')
      return tuple([
          from_json(
              v,
              root_path=object_utils.KeyPath(i, root_path),
              _typename_resolved=True,
              **kwargs
          )
          for i, v in enumerate(json_value[1:])
      ])
    return Symbolic.ListType(json_value, **kwargs)  # pytype: disable=not-callable   # pylint: disable=not-callable
  elif isinstance(json_value, dict):
    if object_utils.JSONConvertible.TYPE_NAME_KEY not in json_value:
      return Symbolic.DictType.from_json(json_value, **kwargs)
    return object_utils.from_json(json_value, _typename_resolved=True, **kwargs)
  return json_value


def from_json_str(json_str: str,
                  *,
                  allow_partial: bool = False,
                  root_path: Optional[object_utils.KeyPath] = None,
                  auto_import: bool = True,
                  auto_dict: bool = False,
                  **kwargs) -> Any:
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
    auto_import: If True, when a '_type' is not registered, PyGlove will
      identify its parent module and automatically import it. For example,
      if the type is 'foo.bar.A', PyGlove will try to import 'foo.bar' and
      find the class 'A' within the imported module.
    auto_dict: If True, dict with '_type' that cannot be loaded will remain
      as dict, with '_type' renamed to 'type_name'.
    **kwargs: Additional keyword arguments that will be passed to
      ``pg.from_json``.

  Returns:
    A deserialized value.
  """
  return from_json(
      json.loads(json_str),
      allow_partial=allow_partial,
      root_path=root_path,
      auto_import=auto_import,
      auto_dict=auto_dict,
      **kwargs
  )


def to_json(value: Any, **kwargs) -> Any:
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
      * Tuple types;
      * Dict types.

    **kwargs: Keyword arguments to pass to value.to_json if value is
      JSONConvertible.

  Returns:
    JSON value.
  """
  # NOTE(daiyip): special handling `sym_jsonify` since symbolized
  # classes may have conflicting `to_json` method in their existing classes.
  if isinstance(value, Symbolic):
    return value.sym_jsonify(**kwargs)
  return object_utils.to_json(value, **kwargs)


def to_json_str(value: Any,
                *,
                json_indent=None,
                **kwargs) -> str:
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


def load(path: str, *args, **kwargs) -> Any:
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
  load_handler = flags.get_load_handler() or default_load_handler
  value = load_handler(path, *args, **kwargs)
  if flags.is_tracking_origin() and isinstance(value, Symbolic):
    value.sym_setorigin(path, 'load')
  return value


def save(value: Any, path: str, *args, **kwargs) -> Any:
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
  save_handler = flags.get_save_handler() or default_save_handler
  return save_handler(value, path, *args, **kwargs)


def default_load_handler(
    path: str,
    file_format: Literal['json', 'txt'] = 'json',
    **kwargs) -> Any:
  """Default load handler from file."""
  content = pg_io.readfile(path)
  if file_format == 'json':
    return from_json_str(content, allow_partial=True, **kwargs)
  elif file_format == 'txt':
    return content
  else:
    raise ValueError(f'Unsupported `file_format`: {file_format!r}.')


def default_save_handler(
    value: Any,
    path: str,
    *,
    indent: Optional[int] = None,
    file_format: Literal['json', 'txt'] = 'json',
    **kwargs) -> None:
  """Default save handler to file."""
  if file_format == 'json':
    content = to_json_str(value, json_indent=indent, **kwargs)
  elif file_format == 'txt':
    content = value if isinstance(value, str) else object_utils.format(
        value, compact=False, verbose=True)
  else:
    raise ValueError(f'Unsupported `file_format`: {file_format!r}.')

  pg_io.mkdirs(os.path.dirname(path), exist_ok=True)
  pg_io.writefile(path, content)


#
# Internal helper methods.
#


def accepts_partial(value: Symbolic) -> bool:
  """Returns True if partial values is allowed in scope."""
  allow_in_scope = flags.is_under_partial_scope()
  return value.allow_partial if allow_in_scope is None else allow_in_scope


def writtable_via_accessors(value: Symbolic) -> bool:
  """Returns True if partial values is allowed in scope."""
  writable_in_scope = flags.is_under_accessor_writable_scope()
  if writable_in_scope is None:
    return value.accessor_writable
  return writable_in_scope


def treats_as_sealed(value: Symbolic) -> bool:
  """Returns True if current object is treated as sealed in scope."""
  sealed_in_scope = flags.is_under_sealed_scope()
  return value.sym_sealed if sealed_in_scope is None else sealed_in_scope


def symbolic_transform_fn(allow_partial: bool):
  """Symbolic object transform function builder."""

  def _fn(
      path: object_utils.KeyPath, field: pg_typing.Field, value: Any) -> Any:
    """Transform schema-less List and Dict to symbolic."""
    if isinstance(value, Symbolic):
      return value
    if isinstance(value, dict):
      value_spec = pg_typing.ensure_value_spec(
          field.value, pg_typing.Dict(), path)
      value = Symbolic.DictType(   # pytype: disable=not-callable  # pylint: disable=not-callable
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
      value_spec = pg_typing.ensure_value_spec(
          field.value, pg_typing.List(pg_typing.Any()), path)
      value = Symbolic.ListType(   # pytype: disable=not-callable  # pylint: disable=not-callable
          value,
          value_spec=value_spec,
          allow_partial=allow_partial,
          root_path=path)
    return value

  return _fn
