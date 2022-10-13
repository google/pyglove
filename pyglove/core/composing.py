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
"""Harnessing symbolic compositions.

This sub-package provides utilities for handling manual or automatic symbolic
compositions.
"""

import dataclasses
import enum
import random
import types
from typing import Any, Callable, List, Optional, Tuple, Union

from pyglove.core import object_utils
from pyglove.core import symbolic


class MutationType(str, enum.Enum):
  """Mutation type."""

  # Replace current node.
  REPLACE = 0

  # Insert an element before current node (in a list).
  INSERT = 1

  # Delete current node (in a list.)
  DELETE = 2


def mutate(value: symbolic.Symbolic,
           replacer: Callable[
               [
                   object_utils.KeyPath,    # Location.
                   Any,                     # Old value.
                                            # pg.MISSING_VALUE for insertion.
                   symbolic.Symbolic,       # Parent node.
               ],
               Any                          # Replacement.
           ],
           weights: Optional[Callable[
               [
                   MutationType,            # Mutation type.
                   object_utils.KeyPath,    # Location.
                   Any,                     # Value.
                   symbolic.Symbolic,       # Parent.
               ],
               float                        # Mutation weight.
           ]] = None,   # pylint:disable=bad-whitespace
           random_generator: Union[
               None, types.ModuleType, random.Random] = None,
           ) -> symbolic.Symbolic:  # pylint:disable=bad-whitespace
  """Mutates a symbolic value.

  `pg.mutate` provides an interface for programs to mutate an arbitrary
  symbolic composition. It allows users to provide a weighting function to
  determine the probabilities of all points in the symbolic tree to mutate on.
  Then it randomly choose a point based on the probabilities, and invoke the
  user provided `replacer` function to generate a replaced value. `pg.mutate`
  automatically handle node insertion and deletion in lists. The `replacer`
  function will receive `pg.MISSING_VALUE` to signal an insertion, with the
  location pointing to the place where it shall be inserted. Users can use
  `parent` and key to inspect its surrounding elements when needed.

  Example::

    replacer = pg.replacers.FieldLevelSeenObjects()
    for _ in range(10):
      value = pg.mutate(value, replacer)

  Args:
    value: A symbolic value to mutate.
    replacer: A callable object that takes information of the value to operate
      (e.g. location, old value, parent node) and returns a new value as a
      replacement for the node. Such information allows users to not only access
      the mutation node, but the entire symbolic tree if needed, allowing
      complex mutation rules to be written with ease - for example - check
      adjacent nodes while modifying a list element. This function is designed
      to take care of both node replacements and node insertions. When insertion
      happens, the old value for the location will be `pg.MISSING_VALUE`. See
      `pg.composing.SeenObjectReplacer` as an example.
    weights: An optional callable object that returns the unnormalized (e.g.
      the sum of all probabilities don't have to sum to 1.0) mutation
      probabilities for all the nodes in the symbolic tree, based on (mutation
      type, location, old value, parent node), If None, all the locations and
      mutation types will be sampled uniformly.
    random_generator: An optional random generator that chooses the mutation
      point based on computed probabilities. If None, the process-level random
      generator will be used.

  Returns:
    The mutated the input.
  """
  # Find mutation point.
  weights = weights or (lambda mt, k, v, p: 1.0)
  random_generator = random_generator or random
  points, weights = _mutation_points_and_weights(value, weights)
  [point] = random_generator.choices(points, weights, k=1)

  # Mutating value.
  if point.mutation_type == MutationType.REPLACE:
    assert point.location, point
    value.rebind({
        str(point.location): replacer(
            point.location, point.old_value, point.parent)})
  elif point.mutation_type == MutationType.INSERT:
    assert isinstance(point.parent, symbolic.List), point
    assert point.old_value == object_utils.MISSING_VALUE, point
    assert isinstance(point.location.key, int), point
    with symbolic.allow_writable_accessors():
      point.parent.insert(
          point.location.key,
          replacer(point.location, point.old_value, point.parent))
  else:
    assert point.mutation_type == MutationType.DELETE, point
    assert isinstance(point.parent, symbolic.List), point
    assert isinstance(point.location.key, int), point
    with symbolic.allow_writable_accessors():
      del point.parent[point.location.key]
  return value


@dataclasses.dataclass
class _MutationPoint:
  """Internal class that encapsulates the information for a mutation point.

  Attributes:
    mutation_type: The type of the mutation.
    location: The location where the mutation will take place.
    old_value: The value of the mutation point before mutation.
    parent: The parent node of the mutation point.
  """

  mutation_type: MutationType
  location: object_utils.KeyPath
  old_value: Any
  parent: symbolic.Symbolic


def _mutation_points_and_weights(
    value: symbolic.Symbolic,
    weights: Callable[
        [
            MutationType,           # Mutation type.
            object_utils.KeyPath,   # Location.
            Any,                    # Value.
            symbolic.Symbolic,      # Parent.
        ],
        float                       # Mutation weight.
    ]) -> Tuple[List[_MutationPoint], List[float]]:
  """Returns mutation points with weights for a symbolic tree."""
  mutation_points: List[_MutationPoint] = []
  mutation_weights: List[float] = []

  def _choose_mutation_point(k: object_utils.KeyPath,
                             v: Any,
                             p: Optional[symbolic.Symbolic]):
    """Visiting function for a symbolic node."""
    def _add_point(mt: MutationType, k=k, v=v, p=p):
      assert p is not None
      mutation_points.append(_MutationPoint(mt, k, v, p))
      mutation_weights.append(weights(mt, k, v, p))

    if p is not None:
      # Stopping mutating current branch if metadata said so.
      f = p.sym_attr_field(k.key)
      if f and f.metadata and 'no_mutation' in f.metadata:
        return symbolic.TraverseAction.CONTINUE
      _add_point(MutationType.REPLACE)

    # Special handle list traversal to add insertion and deletion.
    if isinstance(v, symbolic.List):
      if v.value_spec:
        spec = v.value_spec
        reached_max_size = spec.max_size and len(v) == spec.max_size
        reached_min_size = spec.min_size and len(v) == spec.min_size
      else:
        reached_max_size = False
        reached_min_size = False

      for i, cv in enumerate(v):
        ck = object_utils.KeyPath(i, parent=k)
        if not reached_max_size:
          _add_point(
              MutationType.INSERT, k=ck, v=object_utils.MISSING_VALUE, p=v)

        if not reached_min_size:
          _add_point(MutationType.DELETE, k=ck, v=cv, p=v)

        # Replace type and value will be added in traverse.
        symbolic.traverse(cv, _choose_mutation_point, root_path=ck, parent=v)
        if not reached_max_size and i == len(v) - 1:
          _add_point(MutationType.INSERT,
                     k=object_utils.KeyPath(i + 1, parent=k),
                     v=object_utils.MISSING_VALUE,
                     p=v)
      return symbolic.TraverseAction.CONTINUE
    return symbolic.TraverseAction.ENTER

  # First-order traverse the symbolic tree to compute
  # the mutation points and weights.
  symbolic.traverse(value, _choose_mutation_point)
  return mutation_points, mutation_weights
