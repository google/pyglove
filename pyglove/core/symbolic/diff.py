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
"""Symbolic differences."""

from typing import Any, Callable, Tuple, Union

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import list as pg_list
from pyglove.core.symbolic import object as pg_object
from pyglove.core.symbolic.pure_symbolic import PureSymbolic


class Diff(PureSymbolic, pg_object.Object):
  """A value diff between two objects: a 'left' object and a 'right' object.

  If one of them is missing, it may be represented by pg.Diff.MISSING

  For example::

    >>> pg.Diff(3.14, 1.618)
    Diff(left=3.14, right=1.618)
    >>> pg.Diff('hello world', pg.Diff.MISSING)
    Diff(left='hello world', right=MISSING)
  """

  class _Missing:
    """Represents an absent party in a Diff."""

    def __repr__(self):
      return self.__str__()

    def __str__(self):
      return 'MISSING'

    def __eq__(self, other):
      return isinstance(other, Diff._Missing)

    def __ne__(self, other):
      return not self.__eq__(other)

  MISSING = _Missing()

  def _on_bound(self):
    super()._on_bound()
    if self.children:
      if not isinstance(self.left, type):
        raise ValueError(
            f'\'left\' must be a type when \'children\' is specified. '
            f'Encountered: {self.left!r}.')
      if not isinstance(self.right, type):
        raise ValueError(
            f'\'right\' must be a type when \'children\' is specified. '
            f'Encountered: {self.right!r}.')
    self._has_diff = None

  @property
  def is_leaf(self) -> bool:
    """Returns True if current Diff does not contain inner Diff object."""
    return not self.children

  def __bool__(self):
    """Returns True if there is a diff."""
    if self._has_diff is None:
      if base.ne(self.left, self.right):
        has_diff = True
      elif self.children:
        has_diff = any(bool(cd) for cd in self.children.values())
      else:
        has_diff = False
      self._has_diff = has_diff
    return self._has_diff

  def sym_eq(self, other: Any):
    """Override symbolic equality."""
    if super().sym_eq(other):
      return True
    if not bool(self):
      return base.eq(self.left, other)

  @property
  def value(self):
    """Returns the value if left and right are the same."""
    if bool(self):
      raise ValueError(
          f'\'value\' cannot be accessed when \'left\' and \'right\' '
          f'are not the same. Left={self.left!r}, Right={self.right!r}.')
    return self.left

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs):
    """Override format to conditionally print the shared value or the diff."""
    if not bool(self):
      if self.value == Diff.MISSING:
        return 'No diff'
      # When there is no diff, but the same value needs to be displayed
      # we simply return the value.
      return object_utils.format(
          self.value, compact, verbose, root_indent, **kwargs)
    if self.is_leaf:
      exclude_keys = kwargs.pop('exclude_keys', None)
      exclude_keys = exclude_keys or set()
      exclude_keys.add('children')
      return super().format(
          compact, verbose, root_indent, exclude_keys=exclude_keys, **kwargs)
    else:
      assert isinstance(self.left, type)
      assert isinstance(self.right, type)
      if self.left is self.right and issubclass(self.left, list):
        return self.children.format(
            compact=compact,
            verbose=verbose,
            root_indent=root_indent,
            cls_name='',
            bracket_type=object_utils.BracketType.SQUARE)
      if self.left is self.right:
        cls_name = self.left.__name__
      else:
        cls_name = f'{self.left.__name__}|{self.right.__name__}'
      return self.children.format(
          compact=compact,
          verbose=verbose,
          root_indent=root_indent,
          cls_name=cls_name,
          bracket_type=object_utils.BracketType.ROUND)


# NOTE(daiyip): we add the symbolic attribute to Diff after its declaration
# since we need to access Diff.MISSING as the default value for `left` and
# `right`.
pg_object.members([
    ('left', pg_typing.Any(default=Diff.MISSING),
     'The left-hand object being compared.'),
    ('right', pg_typing.Any(default=Diff.MISSING),
     'The right-hand object being compared.'),
    ('children', pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Object(Diff), 'Child node.')
    ]))
])(Diff)


def diff(
    left: Any,
    right: Any,
    flatten: bool = False,
    collapse: Union[bool, str, Callable[[Any, Any], bool]] = 'same_type',
    mode: str = 'diff') -> object_utils.Nestable[Diff]:
  """Inspect the symbolic diff between two objects.

  For example::

    @pg.members([
      ('x', pg.Any()),
      ('y', pg.Any())
    ])
    class A(pg.Object):
      pass

    @pg.members([
      ('z', pg.Any().noneable())
    ])
    class B(A):
      pass


    # Diff the same object.
    pg.diff(A(1, 2), A(1, 2))

    >> No diff

    # Diff the same object with mode 'same'.
    pg.diff(A(1, 2), A(1, 2), mode='same')

    >> A(1, 2)

    # Diff different objects of the same type.
    pg.diff(A(1, 2), A(1, 3))

    >> A(
    >>   y = Diff(
    >>     left=2,
    >>     right=3
    >>   )
    >>  )

    # Diff objects of different type.
    pg.diff(A(1, 2), B(1, 3))

    >> Diff(
    >>    left = A(
    >>      x = 1,
    >>      y = 2
    >>    ),
    >>    right = B(
    >>      x = 1,
    >>      y = 3,
    >>      z = None
    >>    )

    # Diff objects of different type with collapse.
    pg.diff(A(1, 2), B(1, 3), collapse=True)

    >> A|B (
    >>   y = Diff(
    >>     left = 2,
    >>     right = 3,
    >>   ),
    >>   z = Diff(
    >>     left = MISSING,
    >>     right = None
    >>   )
    >> )

    # Diff objects of different type with collapse and flatten.
    # Object type is included in key '_type'.
    pg.diff(A(1, pg.Dict(a=1)), B(1, pg.Dict(a=2)), collapse=True, flatten=True)

    >> {
    >>    'y.a': Diff(1, 2),
    >>    'z', Diff(MISSING, None),
    >>    '_type': Diff(A, B)
    >> }

  Args:
    left: The left object to compare.
    right: The right object to compare.
    flatten: If True, returns a level-1 dict with diff keys flattened. Otherwise
      preserve the hierarchy of the diff result.
    collapse: One of a boolean value, string or a callable object that indicates
      whether to collapse two different values. The default value 'same_type'
      means only collapse when the two values are of the same type.
    mode: Diff mode, should be one of ['diff', 'same', 'both']. For 'diff' mode
      (the default), the return value contains only different values. For 'same'
      mode, the return value contains only same values. For 'both', the return
      value contains both different and same values.

  Returns:
    A `Diff` object when flatten is False. Otherwise a dict of string (key path)
    to `Diff`.
  """
  def _should_collapse(left, right):
    if isinstance(left, dict):
      if isinstance(right, dict):
        return True
    elif isinstance(left, list):
      return isinstance(right, list)

    if (isinstance(left, (dict, base.Symbolic))
        and isinstance(right, (dict, base.Symbolic))):
      if collapse == 'same_type':
        return type(left) is type(right)
      elif callable(collapse):
        return collapse(left, right)
      elif isinstance(collapse, bool):
        return collapse
      else:
        raise ValueError(f'Unsupported `collapse` value: {collapse!r}')
    else:
      return False

  def _add_child_diff(diff_container, key, value, child_has_diff):
    if ((mode != 'same' and child_has_diff)
        or (mode != 'diff' and not child_has_diff)):
      diff_container[key] = value

  def _get_container_ops(container):
    if isinstance(container, dict):
      return container.__contains__, container.__getitem__, container.items
    else:
      assert isinstance(container, base.Symbolic)
      return container.sym_hasattr, container.sym_getattr, container.sym_items

  def _diff(x, y) -> Tuple[object_utils.Nestable[Diff], bool]:
    if x is y or x == y:
      return (Diff(x, y), False)
    if not _should_collapse(x, y):
      return (Diff(x, y), True)

    diff_value, has_diff = {}, False
    if isinstance(x, list):
      assert isinstance(y, list)
      def _child(l, index):
        return l[i] if index < len(l) else Diff.MISSING
      for i in range(max(len(x), len(y))):
        child_diff, child_has_diff = _diff(_child(x, i), _child(y, i))
        has_diff = has_diff or child_has_diff
        _add_child_diff(diff_value, str(i), child_diff, child_has_diff)
      diff_value = Diff(pg_list.List, pg_list.List, children=diff_value)
    else:
      assert isinstance(x, (dict, base.Symbolic))
      assert isinstance(y, (dict, base.Symbolic))

      x_haskey, _, x_items = _get_container_ops(x)
      y_haskey, y_getitem, y_items = _get_container_ops(y)

      for k, xv in x_items():
        yv = y_getitem(k) if y_haskey(k) else Diff.MISSING
        child_diff, child_has_diff = _diff(xv, yv)
        has_diff = has_diff or child_has_diff
        _add_child_diff(diff_value, k, child_diff, child_has_diff)

      for k, yv in y_items():
        if not x_haskey(k):
          child_diff, _ = _diff(Diff.MISSING, yv)
          has_diff = True
          _add_child_diff(diff_value, k, child_diff, True)

      xt, yt = type(x), type(y)
      same_type = xt is yt
      if not same_type:
        has_diff = True

      if flatten:
        # Put type difference with key '_type'. Since symbolic
        # fields will not start with underscore, so there should be
        # no clash.
        if not same_type or mode != 'diff':
          diff_value['_type'] = Diff(xt, yt)
      else:
        diff_value = Diff(xt, yt, children=diff_value)
    return diff_value, has_diff

  diff_value, has_diff = _diff(left, right)
  if not has_diff and mode == 'diff':
    diff_value = Diff()
  if flatten:
    diff_value = object_utils.flatten(diff_value)
  return diff_value
