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
"""Derived value from other hyper primitives."""

import abc
import copy
from typing import Any, Callable, List, Optional, Tuple, Union

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core import utils


@symbolic.members([(
    'reference_paths',
    pg_typing.List(pg_typing.Object(utils.KeyPath)),
    (
        'Paths of referenced values, which are relative paths searched from '
        'current node to root.'
    ),
)])
class DerivedValue(symbolic.Object, pg_typing.CustomTyping):
  """Base class of value that references to other values in object tree."""

  @abc.abstractmethod
  def derive(self, *args: Any) -> Any:
    """Derive the value from referenced values."""

  def resolve(
      self, reference_path_or_paths: Optional[Union[str, List[str]]] = None
  ) -> Union[
      Tuple[symbolic.Symbolic, utils.KeyPath],
      List[Tuple[symbolic.Symbolic, utils.KeyPath]],
  ]:
    """Resolve reference paths based on the location of this node.

    Args:
      reference_path_or_paths: (Optional) a string or KeyPath as a reference
        path or a list of strings or KeyPath objects as a list of
        reference paths.
        If this argument is not provided, prebound reference paths of this
        object will be used.

    Returns:
      A tuple (or list of tuple) of (resolved parent, resolved full path)
    """
    single_input = False
    if reference_path_or_paths is None:
      reference_paths = self.reference_paths
    elif isinstance(reference_path_or_paths, str):
      reference_paths = [utils.KeyPath.parse(reference_path_or_paths)]
      single_input = True
    elif isinstance(reference_path_or_paths, utils.KeyPath):
      reference_paths = [reference_path_or_paths]
      single_input = True
    elif isinstance(reference_path_or_paths, list):
      paths = []
      for path in reference_path_or_paths:
        if isinstance(path, str):
          path = utils.KeyPath.parse(path)
        elif not isinstance(path, utils.KeyPath):
          raise ValueError('Argument \'reference_path_or_paths\' must be None, '
                           'a string, KeyPath object, a list of strings, or a '
                           'list of KeyPath objects.')
        paths.append(path)
      reference_paths = paths
    else:
      raise ValueError('Argument \'reference_path_or_paths\' must be None, '
                       'a string, KeyPath object, a list of strings, or a '
                       'list of KeyPath objects.')

    resolved_paths = []
    for reference_path in reference_paths:
      parent = self.sym_parent
      while parent is not None and not reference_path.exists(parent):
        parent = getattr(parent, 'sym_parent', None)
      if parent is None:
        raise ValueError(
            f'Cannot resolve \'{reference_path}\': parent not found.')
      resolved_paths.append((parent, parent.sym_path + reference_path))
    return resolved_paths if not single_input else resolved_paths[0]

  def __call__(self):
    """Generate value by deriving values from reference paths."""
    referenced_values = []
    for reference_path, (parent, _) in zip(
        self.reference_paths, self.resolve()):
      referenced_value = reference_path.query(parent)

      # Make sure referenced value does not have referenced value.
      # NOTE(daiyip): We can support dependencies between derived values
      # in future if needed.
      if not utils.traverse(referenced_value, self._contains_not_derived_value):
        raise ValueError(
            f'Derived value (path={referenced_value.sym_path}) should not '
            f'reference derived values. '
            f'Encountered: {referenced_value}, '
            f'Referenced at path {self.sym_path}.')
      referenced_values.append(referenced_value)
    return self.derive(*referenced_values)

  def _contains_not_derived_value(
      self, path: utils.KeyPath, value: Any
  ) -> bool:
    """Returns whether a value contains derived value."""
    if isinstance(value, DerivedValue):
      return False
    elif isinstance(value, symbolic.Object):
      for k, v in value.sym_items():
        if not utils.traverse(
            v,
            self._contains_not_derived_value,
            root_path=utils.KeyPath(k, path),
        ):
          return False
    return True


class ValueReference(DerivedValue):
  """Class that represents a value referencing another value."""

  def _on_bound(self):
    """Custom init."""
    super()._on_bound()
    if len(self.reference_paths) != 1:
      raise ValueError(
          f'Argument \'reference_paths\' should have exact 1 '
          f'item. Encountered: {self.reference_paths}')

  def derive(self, referenced_value: Any) -> Any:
    """Derive value by return a copy of the referenced value."""
    return copy.copy(referenced_value)

  def custom_apply(
      self,
      path: utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[
          Callable[[utils.KeyPath, pg_typing.Field, Any], Any]
      ] = None,
  ) -> Tuple[bool, 'DerivedValue']:
    """Implement pg_typing.CustomTyping interface."""
    # TODO(daiyip): perform possible static analysis on referenced paths.
    del path, value_spec, allow_partial, child_transform
    return (False, self)


def reference(reference_path: str) -> ValueReference:
  """Create a referenced value from a referenced path."""
  return ValueReference(reference_paths=[reference_path])
