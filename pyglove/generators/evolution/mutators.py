# Copyright 2019 The PyGlove Authors
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
"""Mutators for evolutionary algorithms."""

import random
from typing import List, Optional, Tuple

import pyglove.core as pg
from pyglove.generators.evolution import base


# We disable implicit str concat as it is commonly used class schema docstr.
# pylint: disable=implicit-str-concat


def where_fn_spec():
  """Returns ValueSpec for 'where' function."""
  return pg.typing.Callable([pg.typing.Object(pg.DNA)],
                            returns=pg.typing.Bool()).noneable()


@pg.members([
    ('where', where_fn_spec(),
     'A callable to determine which nodes of the DNA are mutable. By default, '
     'all nodes are mutable. When writing a custom `where`, it can be '
     'assumed that the DNA arg has a `spec` field with its DNASpec.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed for mutation.')
], init_arg_list=['where', 'seed'])
class Uniform(base.Mutator):
  """Mutates a DNA by randomizing a branch of the DNA.

  This is a minimal mutator. It acts as follows. PyGlove represents a DNA as a
  tree, with information at each node, where child nodes are conditional on the
  value of parent nodes. This mutator will pick a node uniformly at random and
  mutate the subtree rooted at that node (inclusive), respecting dependencies
  specified in the DNASpec.

  However, in general, we recommended that you write your own Mutator subclass
  so you can tailor it to your search space. This would allow you, for example:
  i) to modify a value drawing from a custom distribution: e.g. a
  gaussian-distributed additive change may be more appropriate in many cases.
  ii) to choose a node in the tree with a non-uniform distribution. E.g. you
  may want to modify some nodes more frequently if they encode areas of the
  space that should be explored more thoroughly.
  iii) perform mutations that implement a different type of locality than that
  represented by the tree structure. E.g. if two nodes at the same level need
  to be modified in a coordinated way.
  """

  def _on_bound(self):
    super()._on_bound()
    if self.seed is None:
      self._random = random
    else:
      self._random = random.Random(self.seed)

  def mutate(self, dna: pg.DNA, step: int = 0) -> pg.DNA:
    """Mutates the DNA at a given step."""
    del step
    dna = dna.clone(deep=True)  # Prevent overwriting argument.
    child_nodes, parent_nodes, child_indexes = self._get_relationships(dna)
    if not child_nodes:
      raise RuntimeError(f'Immutable DNA: {dna!r}')
    child_node, parent_node, child_index = self._random.choice(list(zip(
        child_nodes, parent_nodes, child_indexes)))
    if parent_node is None:
      # The node mutated ("child") is the root of the DNA tree.
      return pg.random_dna(child_node.spec, self._random)
    else:
      # The node mutated is not the root of the DNA tree.
      if _node_needs_distinct(child_node.spec):
        # The approach taken here is inefficient in the special case when there
        # are many choices. If a random choice is likely to succeed, that
        # scenario can be sped up by redrawing random choices until success.
        # Consider adding a branch to handle that case, depending on need.

        # Compute mutated node value, enforcing distinct constraint.
        distinct_candidates = (set(range(len(child_node.spec.candidates)))
                               - set([c.value for c in parent_node.children]))
        if distinct_candidates:
          new_child_value = self._random.choice(list(distinct_candidates))
          # Create a new sub-tree.
          new_child_node = pg.DNA(
              new_child_value,
              children=[pg.random_dna(
                  child_node.spec.candidates[new_child_value], self._random)])
          new_child_node.use_spec(child_node.spec)
        else:
          new_child_node = None
      else:
        new_child_node = pg.random_dna(
            child_node.spec, self._random)
      if new_child_node is not None:
        # NOTE(daiyip): we update the children without invalidating the internal
        # states of the DNA for better performance.
        parent_node.children.rebind(
            {child_index: new_child_node}, skip_notification=True)
        if _node_needs_sorting(child_node.spec):
          parent_node.rebind(
              children=sorted(parent_node.children, key=lambda c: c.value),
              skip_notification=True)
      return dna

  def _get_relationships(self, dna: pg.DNA) -> Tuple[
      List[pg.DNA], List[Optional[pg.DNA]], List[Optional[int]]]:
    """Extracts the parent-child node relationships in a DNA.

    Note that PyGlove represents the nodes in a DNA instance as DNA instances
    themselves.

    Args:
      dna: the DNA that will be mutated.

    Returns:
      A tuple of 3 lists of the same length with corresponding elements:
      -child_nodes: a list of every node in the DNA.
      -parent_nodes: a list of the parent node of the corresponding node in
        `child_nodes`.
      -child_indexes: a list of indexes. For all j, child_nodes[j] is the i-th
        child of parent_nodes[j], where i = child_indexes[j].
      Note that the root is included as a "child" with a `None` parent.
    """
    # This method uses the word "child" and "parent" to refer to the node
    # relationships in the tree structure of a DNA. This should not be confused
    # with the standard use of "child" and "parent" as the genealogic
    # relationship of DNAs generated by an evolutionary algorithm.

    def is_mutable_node(obj):
      return self._is_mutable_node(obj)

    results = pg.query(dna, where=is_mutable_node, enter_selected=True)
    child_nodes = list(results.values())
    parent_nodes = [n.parent_dna for n in child_nodes]
    child_indexes = [
        n.sym_path.key if n.parent_dna else None for n in child_nodes]

    return child_nodes, parent_nodes, child_indexes

  def _is_mutable_node(self, obj: pg.Object) -> bool:
    """Returns whether the branch contains mutateble values."""
    if not isinstance(obj, pg.DNA):
      return False
    if (obj.sym_parent is None and
        # `_immutable_root` is only set by unit tests.
        getattr(self, '_immutable_root', None)):
      return False
    if self.where and not self.where(obj):
      return False
    return (isinstance(obj.spec, pg.geno.Choices) or
            isinstance(obj.spec, pg.geno.Float))


@pg.members([
    ('where', where_fn_spec(),
     'A callable to determine which nodes of the DNA are mutable. By default, '
     'all nodes are mutable. When writing a custom `where`, it can be '
     'assumed that the DNA arg has a `spec` field with its DNASpec.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed for mutation.')
], init_arg_list=['where', 'seed'])
class Swap(base.Mutator):
  """Specialized mutator that swaps DNA branches rooted at sibling nodes."""

  def _on_bound(self):
    super()._on_bound()
    if self.seed is None:
      self._random = random
    else:
      self._random = random.Random(self.seed)

  def mutate(self, dna: pg.DNA, step: int = 0) -> pg.DNA:
    """Mutates the DNA. If impossible, returns a clone."""
    dna = dna.clone(deep=True)  # Prevent overwriting argument.
    parent_node_candidates = self._get_candidate_nodes(dna)
    self._random.shuffle(parent_node_candidates)
    parent_node = None
    child_indexes = []
    for parent_node in parent_node_candidates:
      if not parent_node.spec.sorted:
        # If no sorting is required, any swap is valid.
        child_indexes = self._random.sample(range(len(parent_node.children)), 2)
        break  # Found a pair to swap.

    if child_indexes:
      # Swap the two indexes.
      assert len(child_indexes) == 2
      child0 = parent_node.children[child_indexes[0]]
      child1 = parent_node.children[child_indexes[1]]
      parent_node.children.rebind({child_indexes[0]: child1})
      parent_node.children.rebind({child_indexes[1]: child0})
    return dna

  def _get_candidate_nodes(self, dna: pg.DNA) -> List[pg.DNA]:
    """Returns a list of nodes with potentially swappable children."""
    def is_candidate_node(obj):
      if not isinstance(obj, pg.DNA):
        return False
      if self.where and not self.where(obj):
        return False
      return (isinstance(obj.spec, pg.geno.Choices) and
              obj.spec.num_choices > 1)
    selected_nodes = pg.query(
        dna, where=is_candidate_node, enter_selected=True)
    return list(selected_nodes.values())


def _node_needs_distinct(dna_spec: pg.DNASpec) -> bool:
  """Returns whether this node requires distinct children."""
  return (isinstance(dna_spec, pg.geno.Choices)
          and dna_spec.is_subchoice and dna_spec.distinct)


def _node_needs_sorting(dna_spec: pg.DNASpec) -> bool:
  """Returns whether this node requires distinct children."""
  return (isinstance(dna_spec, pg.geno.Choices)
          and dna_spec.is_subchoice and dna_spec.sorted)
