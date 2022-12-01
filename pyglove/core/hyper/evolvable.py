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
"""Evolvable symbolic values."""

import dataclasses
import enum
import random
import types
from typing import Any, Callable, List, Optional, Tuple, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper import custom


class MutationType(str, enum.Enum):
  """Mutation type."""
  REPLACE = 0
  INSERT = 1
  DELETE = 2


@dataclasses.dataclass
class MutationPoint:
  """Internal class that encapsulates the information for a mutation point.

  Attributes:
    mutation_type: The type of the mutation.
    location: The location where the mutation will take place.
    old_value: The value of the mutation point before mutation.
    parent: The parent node of the mutation point.
  """
  mutation_type: 'MutationType'
  location: object_utils.KeyPath
  old_value: Any
  parent: symbolic.Symbolic


class Evolvable(custom.CustomHyper):
  """Hyper primitive for evolving an arbitrary symbolic object."""

  def _on_bound(self):
    super()._on_bound()
    self._weights = self.weights or (lambda mt, k, v, p: 1.0)

  def custom_decode(self, dna: geno.DNA) -> Any:
    assert isinstance(dna.value, str)
    # TODO(daiyip): consider compression.
    return symbolic.from_json_str(dna.value)

  def custom_encode(self, value: Any) -> geno.DNA:
    return geno.DNA(symbolic.to_json_str(value))

  def mutation_points_and_weights(
      self,
      value: symbolic.Symbolic) -> Tuple[List[MutationPoint], List[float]]:
    """Returns mutation points with weights for a symbolic tree."""
    mutation_points: List[MutationPoint] = []
    mutation_weights: List[float] = []

    def _choose_mutation_point(k: object_utils.KeyPath,
                               v: Any,
                               p: Optional[symbolic.Symbolic]):
      """Visiting function for a symbolic node."""
      def _add_point(mt: MutationType, k=k, v=v, p=p):
        assert p is not None
        mutation_points.append(MutationPoint(mt, k, v, p))
        mutation_weights.append(self._weights(mt, k, v, p))

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
            _add_point(MutationType.INSERT,
                       k=ck, v=object_utils.MISSING_VALUE, p=v)

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

  def first_dna(self) -> geno.DNA:
    """Returns the first DNA of current sub-space."""
    return self.custom_encode(self.initial_value)

  def random_dna(
      self,
      random_generator: Union[types.ModuleType, random.Random, None] = None,
      previous_dna: Optional[geno.DNA] = None) -> geno.DNA:
    """Generates a random DNA."""
    random_generator = random_generator or random
    if previous_dna is None:
      return self.first_dna()
    return self.custom_encode(
        self.mutate(self.custom_decode(previous_dna), random_generator))

  def mutate(
      self,
      value: symbolic.Symbolic,
      random_generator: Union[types.ModuleType, random.Random, None] = None
      ) -> symbolic.Symbolic:
    """Returns the next value for a symbolic value."""
    r = random_generator or random
    points, weights = self.mutation_points_and_weights(value)
    [point] = r.choices(points, weights, k=1)

    # Mutating value.
    if point.mutation_type == MutationType.REPLACE:
      assert point.location, point
      value.rebind({
          str(point.location): self.node_transform(
              point.location, point.old_value, point.parent)})
    elif point.mutation_type == MutationType.INSERT:
      assert isinstance(point.parent, symbolic.List), point
      assert point.old_value == object_utils.MISSING_VALUE, point
      assert isinstance(point.location.key, int), point
      with symbolic.allow_writable_accessors():
        point.parent.insert(
            point.location.key,
            self.node_transform(point.location, point.old_value, point.parent))
    else:
      assert point.mutation_type == MutationType.DELETE, point
      assert isinstance(point.parent, symbolic.List), point
      assert isinstance(point.location.key, int), point
      with symbolic.allow_writable_accessors():
        del point.parent[point.location.key]
    return value


# We defer members declaration for Evolvable since the weights will reference
# the definition of MutationType.
symbolic.members([
    ('initial_value', pg_typing.Object(symbolic.Symbolic),
     'Symbolic value to involve.'),
    ('node_transform', pg_typing.Callable(
        [],
        returns=pg_typing.Any()),
     ''),
    ('weights', pg_typing.Callable(
        [
            pg_typing.Object(MutationType),
            pg_typing.Object(object_utils.KeyPath),
            pg_typing.Any().noneable(),
            pg_typing.Object(symbolic.Symbolic)
        ], returns=pg_typing.Float(min_value=0.0)).noneable(),
     ('An optional callable object that returns the unnormalized (e.g. '
      'the sum of all probabilities do not have to sum to 1.0) mutation '
      'probabilities for all the nodes in the symbolic tree, based on '
      '(mutation type, location, old value, parent node). If None, all the '
      'locations and mutation types will be sampled uniformly.')),
])(Evolvable)


def evolve(
    initial_value: symbolic.Symbolic,
    node_transform: Callable[
        [
            object_utils.KeyPath,    # Location.
            Any,                     # Old value.
                                     # pg.MISSING_VALUE for insertion.
            symbolic.Symbolic,       # Parent node.
        ],
        Any                          # Replacement.
    ],
    *,
    weights: Optional[Callable[
        [
            MutationType,  # Mutation type.
            object_utils.KeyPath,    # Location.
            Any,                     # Value.
            symbolic.Symbolic,       # Parent.
        ],
        float                        # Mutation weight.
    ]] = None,  # pylint: disable=bad-whitespace
    name: Optional[str] = None,
    hints: Optional[Any] = None) -> Evolvable:
  """An evolvable symbolic value.

  Example::

    @pg.symbolize
    @dataclasses.dataclass
    class Foo:
      x: int
      y: int

    @pg.symbolize
    @dataclasses.dataclass
    class Bar:
      a: int
      b: int

    # Defines possible transitions.
    def node_transform(location, value, parent):
      if isinstance(value, Foo)
        return Bar(value.x, value.y)
      if location.key == 'x':
        return random.choice([1, 2, 3])
      if location.key == 'y':
        return random.choice([3, 4, 5])

    v = pg.evolve(Foo(1, 3), node_transform)

  See also:

    * :class:`pyglove.hyper.Evolvable`
    * :func:`pyglove.oneof`
    * :func:`pyglove.manyof`
    * :func:`pyglove.permutate`
    * :func:`pyglove.floatv`

  Args:
    initial_value: The initial value to evolve.
    node_transform: A callable object that takes information of the value to
      operate (e.g. location, old value, parent node) and returns a new value as
      a replacement for the node. Such information allows users to not only
      access the mutation node, but the entire symbolic tree if needed, allowing
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
    name: An optional name of the decision point.
    hints: An optional hints for the decision point.

  Returns:
    A `pg.hyper.Evolvable` object.
  """
  return Evolvable(
      initial_value=initial_value, node_transform=node_transform,
      weights=weights, name=name, hints=hints)
