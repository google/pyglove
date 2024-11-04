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
"""Symbolic reference."""

import functools
import numbers
from typing import Any, Callable, List, Optional, Tuple
from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic.object import Object
from pyglove.core.views.html import tree_view


class Ref(Object, base.Inferential, tree_view.HtmlTreeView.Extension):
  """Symbolic reference.

  When adding a symbolic node to a symbolic tree, it undergoes a copy operation
  if it already has a parent, ensuring that all symbolic objects have a single
  parent. Additionally, list and dict objects are automatically converted to
  ``pg.List`` and ``pg.Dict``, respectively, to enable symbolic operability.

  However, these two conventions come with certain costs. The act of making
  copies incurs a runtime cost, and it also introduces challenges in sharing
  states across different symbolic objects. To address this issue, symbolic
  reference is introduced. This feature allows a symbolic node to refer to
  value objects without the need for transformation or copying, even when the
  symbolic node itself is copied. For example::

    class A(pg.Object):
    x: int

    a = pg.Ref(A(1))
    b = pg.Dict(x=a)
    c = pg.Dict(y=a)

    assert b.x is a
    assert c.y is a
    assert b.clone().x is a
    assert c.clone(deep=True).y is a

  In this example, ``pg.Ref`` is used to create a symbolic reference to the
  object ``A(1)``, and the ``pg.Dict`` objects `b` and `c` can then reference
  `a` without creating additional copies. This mechanism not only mitigates
  the runtime cost but also facilitates seamless sharing of states among various
  symbolic objects.

  Another useful scenario arises when we wish to utilize regular Python list
  and dict objects. In this case, ``pg.Ref`` enables us to access the list/dict
  object as fields in the symbolic tree without requiring them to be transformed
  into ``pg.List`` and ``pg.Dict``. This allows for seamless integration of
  standard Python containers within the symbolic structure::

    d = pg.Dict(x=pg.Ref({1: 2}))
    assert isinstance(d.x, dict)
    assert not isinstance(d.x, pg.Dict)

    e = pg.Dict(x=pg.Ref([0, 1, 2]]))
    assert isinstance(e.x, list)
    assert not isinstance(e.x, pg.List)

  Please be aware that ``pg.Ref`` objects are treated as leaf nodes in the
  symbolic tree, even when they reference other symbolic objects. As a result,
  the ``rebind()`` method cannot modify the value they are pointing to.

  For primitive types, ``pg.Ref()`` returns their values directly without
  creating a reference. For example, ``pg.Ref(1)`` and ``pg.Ref('abc')`` will
  simply return the values 1 and 'abc', respectively, without any additional
  referencing.
  """

  def __new__(cls, value: Any, **kwargs):
    del kwargs
    if isinstance(value, (bool, numbers.Number, str)):
      return value
    return object.__new__(cls)

  @object_utils.explicit_method_override
  def __init__(self, value: Any, **kwargs) -> None:
    super().__init__(**kwargs)
    if isinstance(value, Ref):
      value = value.value
    self._value = value

  def _on_parent_change(
      self,
      old_parent: Optional[base.Symbolic],
      new_parent: Optional[base.Symbolic]) -> None:
    if (new_parent is not None
        and isinstance(self._value, base.Symbolic)
        and self._value.sym_root is new_parent.sym_root):
      raise NotImplementedError('Self-referential object is not supported.')

  @property
  def value(self) -> Any:
    """Returns the referenced value."""
    return self._value

  def infer(self, **kwargs) -> Any:
    """Returns the referenced value."""
    return self._value

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool = False,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
      ) -> Tuple[bool, Any]:
    """Validate candidates during value_spec binding time."""
    del child_transform
    # Check if the field being assigned could accept the referenced value.
    # We do not do any transformation, thus not passing the child transform.
    value_spec.apply(
        self._value,
        allow_partial=allow_partial)
    return (False, self)

  def _sym_clone(self, deep: bool, memo: Any = None) -> 'Ref':
    # Always create a new object.
    # TODO(daiyip): support deep clone with the update of reference when
    # the original value is updated.
    return Ref(self._value, allow_partial=self.allow_partial)

  def sym_eq(self, other: Any) -> bool:
    return isinstance(other, Ref) and self.value is other.value

  def sym_jsonify(self, **kwargs: Any) -> Any:
    raise TypeError(f'{self!r} cannot be serialized at the moment.')

  def __getstate__(self):
    raise TypeError(f'{self!r} cannot be pickled at the moment.')

  def format(
      self,
      compact: bool = False,
      verbose: bool = False,
      root_indent: int = 0,
      **kwargs: Any,
  ) -> str:
    value_str = object_utils.format(
        self._value,
        compact=compact, verbose=verbose, root_indent=root_indent + 1)
    if compact:
      return f'{self.__class__.__name__}({value_str})'
    else:
      return (
          f'{self.__class__.__name__}(\n'
          + '  ' * (root_indent + 1)
          + f'value = {value_str}\n'
          + '  ' * root_indent
          + ')'
      )

  def _html_tree_view_content(
      self,
      *,
      view: tree_view.HtmlTreeView,
      **kwargs: Any) -> tree_view.Html:
    """Overrides `_html_content` to render the referenced value."""
    return view.content(self._value, **kwargs)

  def _html_tree_view_summary(
      self,
      *,
      view: tree_view.HtmlTreeView,
      title: Optional[str] = None,
      **kwargs: Any) -> Optional[tree_view.Html]:
    """Overrides `_html_content` to render the referenced value."""
    return view.summary(
        self,
        title=title or f'{type(self._value).__name__}(...)',
        **kwargs
    )

  @classmethod
  @functools.cache
  def _html_tree_view_config(cls) -> dict[str, Any]:
    return tree_view.HtmlTreeView.get_kwargs(
        super()._html_tree_view_config(),
        dict(
            css_classes=['ref'],
        )
    )

  @classmethod
  @functools.cache
  def _html_tree_view_css_styles(cls) -> List[str]:
    return super()._html_tree_view_css_styles() + [
        """
        /* Ref styles. */
        .ref.summary-title::before {
          content: 'ref: ';
          color: #aaa;
        }
        """
    ]


def maybe_ref(value: Any) -> Optional[Ref]:
  """Returns a reference if a value is not symbolic or already has a parent."""
  if isinstance(value, base.Symbolic):
    if value.sym_parent is None:
      return value
  return Ref(value)


def deref(value: base.Symbolic, recursive: bool = False) -> Any:
  """Dereferences a symbolic value that may contain pg.Ref.

  Args:
    value: The input symbolic value.
    recursive: If True, dereference `pg.Ref` in the entire tree. Otherwise
      Only dereference the root node.

  Returns:
    The dereferenced root, or dereferenced tree if recursive is True.
  """
  if isinstance(value, Ref):
    value = value.value

  if recursive:
    def _deref(k, v, p):
      del k, p
      if isinstance(v, Ref):
        return deref(v.value, recursive=True)
      return v
    return value.rebind(_deref, raise_on_no_change=False)
  return value

