# Copyright 2025 The PyGlove Authors
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
"""Contextual objects.

This module provides support for defining and working with `ContextualObject`s,
whose attributes can dynamically adapt based on context managers and parent
objects.

# Overview:

Contextual objects are specialized objects whose attribute values can be:
1. Dynamically overridden using the `pg.contextual_override` context manager.
2. Accessed via parent objects using the `pg.contextual_attribute` placeholder.

This flexibility is particularly useful in scenarios where attributes need
to respond to dynamic runtime conditions or inherit behavior from hierarchical
structures.

Example:

```python
class A(pg.ContextualObject):
  x: int
  y: Any = pg.contextual_attribute()

a = A(1)
print(a.x) # 1
with pg.contextual_override(x=2):
  print(a.x) # 2
print(a.x) #1

pg.Dict(y=3, a=a)
print(a.y) # 3. (Accessing parent's y)
```
"""

import threading
from typing import Annotated, Any, ContextManager, Dict, Optional, Type
from pyglove.core import utils as pg_utils
from pyglove.core.symbolic import base
from pyglove.core.symbolic import inferred as pg_inferred
from pyglove.core.symbolic import object as pg_object
from pyglove.core.views.html import tree_view


class ContextualObject(pg_object.Object):
  """Base class for contextual objects.

  Contextual objects are objects whose attributes can be dynamically overridden
  using `pg.contextual_override` or resolved through
  `pg.contextual_attribute`, allowing them to inherit values from their
  containing objects.

  Usages:

  ```
  # Define a contextual object.
  class A(pg.ContextualObject):
    x: int
    y: Any = pg.contextual_attribute()

  # Create an instance of A
  a = A(1)
  print(a.x)  # Outputs: 1
  print(a.y)  # Raises an error, as `a` has no containing object.

  # Define another contextual object containing an instance of A
  class B(pg.ContextualObject):
    y: int
    a: A

  # Create an instance of B, containing "a"
  b = B(y=2, a=a)
  print(a.y)  # Outputs: 2, as "y" is resolved from the containing object (B).

  # Contextual overrides are thread-specific
  with pg.contextual_override(x=2):
    print(a.x)  # Outputs: 2

  # Thread-specific behavior of `pg.contextual_override`
  def foo(a):
    print(a.x)

  with pg.contextual_override(x=3):
    t = threading.Thread(target=foo, args=(a,))
    t.start()
    t.join()
    # Outputs: 1, because `pg.contextual_override` is limited to the current
    # thread to avoid clashes in multi-threaded environments.

  # To propagate the override to a new thread, use `pg.with_contextual_override`
  with pg.contextual_override(x=3):
    t = threading.Thread(target=pg.with_contextual_override(foo), args=(a,))
    t.start()
    t.join()
    # Outputs: 3, as the override is explicitly propagated.
  ```
  """

  # Override __repr__ format to use inferred values when available.
  __repr_format_kwargs__ = dict(
      compact=True,
      use_inferred=True,
  )

  # Override __str__ format to use inferred values when available.
  __str_format_kwargs__ = dict(
      compact=False,
      verbose=False,
      use_inferred=True,
  )

  def _on_bound(self):
    super()._on_bound()
    self._contextual_overrides = threading.local()

  def _sym_inferred(self, key: str, **kwargs):
    """Override to allow attribute to access scoped value.

    Args:
      key: attribute name.
      **kwargs: Optional keyword arguments for value inference.

    Returns:
      The value of the symbolic attribute. If not available, returns the
        default value.

    Raises:
      AttributeError: If the attribute does not exist or contextual attribute
        is not ready.
    """
    if key not in self._sym_attributes:
      raise AttributeError(key)

    # Step 1: Try use value from `self.override`.
    # The reason is that `self.override` is short-lived and explicitly specified
    # by the user in scenarios like `LangFunc.render`, which should not be
    # affected by `pg.contextual_override`.
    v = pg_utils.contextual.get_scoped_value(self._contextual_overrides, key)
    if v is not None:
      return v.value

    # Step 2: Try use value from `pg.contextual_override` with `override_attrs`.
    # This gives users a chance to override the bound attributes of components
    # from the top, allowing change of bindings without modifying the code
    # that produces the components.
    override = pg_utils.contextual.get_contextual_override(key)
    if override and override.override_attrs:
      return override.value

    # Step 3: Try use value from the symbolic tree, starting from self to
    # the root of the tree.
    # Step 4: If the value is not present, use the value from `context()` (
    # override_attrs=False).
    # Step 5: Otherwise use the default value from `ContextualAttribute`.
    return super()._sym_inferred(key, context_override=override, **kwargs)

  def override(
      self,
      **kwargs
  ) -> ContextManager[Dict[str, pg_utils.contextual.ContextualOverride]]:
    """Context manager to override the attributes of this component."""
    vs = {
        k: pg_utils.contextual.ContextualOverride(v)
        for k, v in kwargs.items()
    }
    return pg_utils.contextual.contextual_scope(
        self._contextual_overrides, **vs
    )

  def __getattribute__(self, name: str) -> Any:
    """Override __getattribute__ to deal with class attribute override."""
    if not name.startswith('_') and hasattr(self.__class__, name):
      tls = self.__dict__.get('_contextual_overrides', None)
      if tls is not None:
        v = pg_utils.contextual.get_scoped_value(tls, name)
        if v is not None:
          return v.value
    return super().__getattribute__(name)


class ContextualAttribute(
    pg_inferred.ValueFromParentChain, tree_view.HtmlTreeView.Extension
):
  """Attributes whose values are inferred from the containing objects."""

  NO_DEFAULT = (pg_utils.MISSING_VALUE,)

  type: Annotated[Optional[Type[Any]], 'An optional type constraint.'] = None

  default: Any = NO_DEFAULT

  def value_from(
      self,
      parent,
      *,
      context_override: Optional[pg_utils.contextual.ContextualOverride] = None,
      **kwargs,
  ):
    if (parent not in (None, self.sym_parent)
        and isinstance(parent, ContextualObject)):
      # Apply original search logic along the contextual object containing
      # chain.
      return super().value_from(parent, **kwargs)
    elif parent is None:
      # When there is no value inferred from the symbolic tree.
      # Search context override, and then attribute-level default.
      if context_override:
        return context_override.value
      if self.default == ContextualAttribute.NO_DEFAULT:
        return pg_utils.MISSING_VALUE
      return self.default
    else:
      return pg_utils.MISSING_VALUE

  def _html_tree_view_content(
      self,
      *,
      view: tree_view.HtmlTreeView,
      parent: Any = None,
      root_path: Optional[pg_utils.KeyPath] = None,
      **kwargs,
  ) -> tree_view.Html:
    inferred_value = pg_utils.MISSING_VALUE
    if isinstance(parent, base.Symbolic) and root_path:
      inferred_value = parent.sym_inferred(
          root_path.key, pg_utils.MISSING_VALUE
      )

    if inferred_value is not pg_utils.MISSING_VALUE:
      kwargs.pop('name', None)
      return view.render(
          inferred_value, parent=self,
          root_path=pg_utils.KeyPath('<inferred>', root_path),
          **view.get_passthrough_kwargs(**kwargs)
      )
    return tree_view.Html.element(
        'div',
        [
            '(not available)',
        ],
        css_classes=['unavailable-contextual'],
    )

  def _html_tree_view_config(self) -> Dict[str, Any]:
    return tree_view.HtmlTreeView.get_kwargs(
        super()._html_tree_view_config(),
        dict(
            collapse_level=1,
        )
    )

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .contextual-attribute {
          color: purple;
        }
        .unavailable-contextual {
          color: gray;
          font-style: italic;
        }
        """
    ]


# NOTE(daiyip): Returning Any instead of `pg.ContextualAttribute` to
# avoid pytype check error as `contextual_attribute()` can be assigned to any
# type.
def contextual_attribute(
    type: Optional[Type[Any]] = None,  # pylint: disable=redefined-builtin
    default: Any = ContextualAttribute.NO_DEFAULT,
) -> Any:
  """Value marker for a contextual attribute."""
  return ContextualAttribute(type=type, default=default, allow_partial=True)

