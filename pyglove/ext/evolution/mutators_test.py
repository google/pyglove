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
"""Tests for mutators."""

import random
import re
import time
import unittest

import pyglove.core as pg


from pyglove.ext.evolution import mutators


class CoverageTestCase(unittest.TestCase):
  """Class to test coverage of methods with random outputs.

  Usage:

  class MyCoverageTestCase(CoverageTestCase):
    ...
    def testMyFunc(self):
      self.assertEventually(func=MyFunc, allowed=[0, 1, 2], required=[1, 2])
  """

  def assertEventually(  # pylint: disable=g-unreachable-test-method
      self, func, required, allowed, timeout_secs=300.0):
    """Tests that calls to the given function meet required and allowed values.

    Args:
      func: function to test.
      required: iterable of required values. Must be hashable and non-empty.
      allowed: iterable of allowed values. Must be hashable and non-empty.
      timeout_secs: fails if more than this time is required.
    """
    required = set(required)
    assert required
    seen = set()
    start_time = time.time()
    while timeout_secs is None or time.time() - start_time < timeout_secs:
      if seen == required:
        return  # Success.
      value = func()
      if value not in allowed:
        self.fail(msg=(f'Disallowed value: {value}.'))
      if value in required:
        seen.add(value)
    missing = [v for v in required if v not in seen]
    self.fail(
        msg=f'Timed out. Missing values: {str([str(v) for v in missing])}.')


class UniformTest(CoverageTestCase):
  """Tests the Uniform.

  In this test, we generally have specialized tests for the root node because
  the implementation uses a special case to handle it.
  """

  def testOriginalDNARemainsUnchanged(self):
    random.seed()
    dna_spec = pg.geno.manyof(10, [
        pg.geno.constant(), pg.geno.constant(), pg.geno.constant()
    ], distinct=False, sorted=False)
    mutator = mutators.Uniform()
    dna = pg.DNA([0, 2, 1, 1, 2, 0, 0, 1, 2, 1])
    dna.use_spec(dna_spec)
    expected = pg.DNA([0, 2, 1, 1, 2, 0, 0, 1, 2, 1])
    _ = mutator.mutate(dna)
    self.assertEqual(dna, expected)

  def testRootNodeValueMutability(self):
    """Tests that the value of a root node can be mutated."""
    random.seed()
    dna_spec = pg.geno.oneof([pg.geno.constant(), pg.geno.constant()])
    mutator = mutators.Uniform()
    dna = pg.DNA(1)
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return mutated_dna.value != dna.value
    self.assertEventually(func=mutate, required=[True], allowed=[True, False])

  def testRootNodeBranchMutability(self):
    """Tests that a mutation at the root affects the children."""
    random.seed()
    dna_spec = pg.geno.oneof([
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()]),
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()])
    ])
    mutator = mutators.Uniform()
    dna = pg.DNA([(0, [1])])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return (
          # This mutation happened at the root.
          mutated_dna.value != dna.value and
          # This mutation also modified the branch.
          mutated_dna.children[0].value != dna.children[0].value)
    self.assertEventually(func=mutate, required=[True], allowed=[True, False])

  def testRootNodeWithoutValueBranchMutability(self):
    """Tests that a value-less root node is mutable."""
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.constant(), pg.geno.constant()
    ], distinct=False, sorted=False)
    mutator = mutators.Uniform()
    dna = pg.DNA([1, 0])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      # If both children changed, the mutation happened at the root.
      return (mutated_dna.children[0].value != dna.children[0].value and
              mutated_dna.children[1].value != dna.children[1].value)
    self.assertEventually(func=mutate, required=[True], allowed=[True, False])

  def testNonRootNodeValueMutability(self):
    """Tests that the value of a non-root node can be mutated."""
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.constant(), pg.geno.constant()
    ], distinct=False, sorted=False)
    mutator = mutators.Uniform()
    mutator._immutable_root = True  # Prevent root from being mutated.
    dna = pg.DNA([1, 0])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return mutated_dna != dna
    self.assertEventually(func=mutate, required=[True], allowed=[True, False])

  def testNonRootNodeBranchMutability(self):
    """Tests that a mutation at a non-root node affects its children."""
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()]),
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()])
    ], distinct=False, sorted=False,)
    mutator = mutators.Uniform()
    mutator._immutable_root = True  # Prevent root from being mutated.
    dna = pg.DNA([(0, [0]), (0, [0])])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      mutated_nonroot_index = None
      num_different_nonroot_nodes = 0
      for nonroot_index in range(2):
        if (mutated_dna.children[nonroot_index].value !=
            dna.children[nonroot_index].value):
          num_different_nonroot_nodes += 1
          mutated_nonroot_index = nonroot_index
      assert num_different_nonroot_nodes <= 1  # Because root cannot be mutated.
      if num_different_nonroot_nodes == 0:
        # Not the case we are interested in: no first-level nodes were modified.
        return False
      return (
          # Check whether the nonroot node's child has changed too.
          mutated_dna.children[mutated_nonroot_index].children[0].value !=
          dna.children[mutated_nonroot_index].children[0].value)
    self.assertEventually(func=mutate, required=[True], allowed=[True, False])

  def testRootNodeWithNontrivialConditionValueCoverage(self):
    random.seed()
    dna_spec = pg.geno.oneof([
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()]),
        pg.geno.constant()
    ])
    mutator = mutators.Uniform()
    dna = pg.DNA([(0, 0)])
    dna.use_spec(dna_spec)
    expected_mutated_dnas = [
        pg.DNA([(0, 0)]),
        pg.DNA([(0, 1)]),
        pg.DNA([1]),
    ]
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return expected_mutated_dnas.index(mutated_dna)
    expected_indexes = range(len(expected_mutated_dnas))
    self.assertEventually(
        func=mutate, required=expected_indexes, allowed=expected_indexes)

  def testNonRootNodeWithNontrivialConditionValueCoverage(self):
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()]),
        pg.geno.constant()
    ], distinct=False)
    mutator = mutators.Uniform()
    dna = pg.DNA([(0, 0), (0, 0)])
    dna.use_spec(dna_spec)
    expected_mutated_first_child = [
        pg.DNA([(0, 0)]),
        pg.DNA([(0, 1)]),
        pg.DNA([1]),
    ]
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return expected_mutated_first_child.index(mutated_dna.children[0])
    expected_indexes = range(len(expected_mutated_first_child))
    self.assertEventually(
        func=mutate, required=expected_indexes, allowed=expected_indexes)

  def testFloatValueCoverage(self):
    """Tests that all float values can be reached."""
    random.seed()
    dna_spec = pg.geno.floatv(0.0, 1.0)
    mutator = mutators.Uniform()
    dna = pg.DNA(0.2)
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return round(mutated_dna.value * 10.0)
    expected = range(11)
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testChoicesValueCoverage(self):
    """Tests that all choices can be reached."""
    random.seed()
    dna_spec = pg.geno.space([
        pg.geno.manyof(2, [
            pg.geno.constant(), pg.geno.constant(), pg.geno.constant()
        ], distinct=False, sorted=False)
    ])
    mutator = mutators.Uniform()
    dna = pg.DNA([1, 0])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return mutated_dna.children[1].value
    expected = [0, 1, 2]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testSortedChoicesValueCoverage(self):
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.constant(), pg.geno.constant(), pg.geno.constant()
    ], distinct=False, sorted=True)
    mutator = mutators.Uniform()
    dna = pg.DNA([0, 1])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return (mutated_dna.children[0].value, mutated_dna.children[1].value)
    expected = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testNestedSortedChoicesValueCoverage(self):
    random.seed()
    dna_spec = pg.dna_spec(
        pg.oneof([
            pg.manyof(
                2, [1, 2, 3], choices_sorted=True, choices_distinct=False),
            1
        ]))

    mutator = mutators.Uniform()
    dna = pg.DNA((0, [0, 1]))
    dna.use_spec(dna_spec)
    def mutate():
      return mutator.mutate(dna)
    expected = [
        pg.DNA((0, [0, 0])),
        pg.DNA((0, [0, 1])),
        pg.DNA((0, [0, 2])),
        pg.DNA((0, [1, 1])),
        pg.DNA((0, [1, 2])),
        pg.DNA((0, [2, 2])),
        pg.DNA(1)
    ]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testDistinctChoicesValueCoverage(self):
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.constant(), pg.geno.constant(), pg.geno.constant()
    ], distinct=True, sorted=False)
    mutator = mutators.Uniform()
    dna = pg.DNA([0, 2])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return (mutated_dna.children[0].value, mutated_dna.children[1].value)
    expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testNestedDistinctChoicesValueCoverage(self):
    random.seed()
    dna_spec = pg.dna_spec(
        pg.oneof([
            pg.manyof(
                2, [1, 2, 3], choices_distinct=True),
            1
        ]))

    mutator = mutators.Uniform()
    dna = pg.DNA((0, [0, 1]))
    dna.use_spec(dna_spec)
    def mutate():
      return mutator.mutate(dna)
    expected = [
        pg.DNA((0, [0, 1])),
        pg.DNA((0, [1, 0])),
        pg.DNA((0, [0, 2])),
        pg.DNA((0, [2, 0])),
        pg.DNA((0, [1, 2])),
        pg.DNA((0, [2, 1])),
        pg.DNA(1)
    ]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testSortedDistinctChoicesValueCoverage(self):
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.constant(), pg.geno.constant(), pg.geno.constant()
    ], distinct=True, sorted=True)
    mutator = mutators.Uniform()
    dna = pg.DNA([0, 2])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return (mutated_dna.children[0].value, mutated_dna.children[1].value)
    expected = [(0, 1), (0, 2), (1, 2)]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testNestedSortedDistinctChoicesValueCoverage(self):
    random.seed()
    dna_spec = pg.dna_spec(
        pg.oneof([
            pg.manyof(
                2, [1, 2, 3], choices_sorted=True, choices_distinct=True),
            1
        ]))

    mutator = mutators.Uniform()
    dna = pg.DNA((0, [0, 1]))
    dna.use_spec(dna_spec)
    def mutate():
      return mutator.mutate(dna)
    expected = [
        pg.DNA((0, [0, 1])),
        pg.DNA((0, [0, 2])),
        pg.DNA((0, [1, 2])),
        pg.DNA(1)
    ]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testDistinctChoicesSubtree(self):
    """Tests conditional subtrees of a node with distinct constraints."""
    random.seed()
    dna_spec = pg.geno.manyof(2, [
        pg.geno.constant(),
        pg.geno.oneof([pg.geno.constant(), pg.geno.constant()])
    ], distinct=True, sorted=False)
    mutator = mutators.Uniform()
    dna = pg.DNA([0, (1, 0)])
    dna.use_spec(dna_spec)
    expected_mutated_dnas = [
        pg.DNA([0, (1, 0)]),
        pg.DNA([0, (1, 1)]),
        pg.DNA([(1, 0), 0]),
        pg.DNA([(1, 1), 0]),
    ]
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return expected_mutated_dnas.index(mutated_dna)
    expected_indexes = range(len(expected_mutated_dnas))
    self.assertEventually(
        func=mutate, required=expected_indexes, allowed=expected_indexes)

  def testTargeting(self):
    random.seed()
    dna_spec = pg.geno.space([
        pg.geno.oneof([   # No hints, so not targeted.
            pg.geno.constant(), pg.geno.constant()
        ]),
        pg.geno.oneof([   # Targeted by regex.
            pg.geno.constant(), pg.geno.constant()
        ], hints='bcde'),
        pg.geno.oneof([   # Not targeted by regex.
            pg.geno.constant(), pg.geno.constant()
        ], hints='abcd')
    ])

    def where_fn(dna):
      """Targets nodes according to DNASpec `hints` tag."""
      if dna.spec.hints is None:
        return False  # No hints to target.
      return re.match('^bc.*', dna.spec.hints)
    mutator = mutators.Uniform(where=where_fn)

    dna = pg.DNA([0, 0, 1])
    dna.use_spec(dna_spec)
    expected_mutated_dnas = [
        pg.DNA([0, 0, 1]),
        pg.DNA([0, 1, 1]),
    ]
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return expected_mutated_dnas.index(mutated_dna)
    expected_indexes = range(len(expected_mutated_dnas))
    self.assertEventually(
        func=mutate, required=expected_indexes, allowed=expected_indexes)

  def testCustomDecisionPoint(self):
    def random_dna(random_generator, previous_dna):
      del previous_dna
      return pg.DNA(value=str(random_generator.randint(0, 3)))

    dna_spec = pg.geno.oneof([
        pg.geno.constant(),
        pg.geno.custom(random_dna_fn=random_dna),
        pg.geno.constant(),
    ])
    mutator = mutators.Uniform()
    dna = pg.DNA(0, spec=dna_spec)
    expected_mutated_dnas = [
        pg.DNA(0),
        pg.DNA((1, '0')),
        pg.DNA((1, '1')),
        pg.DNA((1, '2')),
        pg.DNA((1, '3')),
        pg.DNA(2),
    ]
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return expected_mutated_dnas.index(mutated_dna)
    expected_indexes = range(len(expected_mutated_dnas))
    self.assertEventually(
        func=mutate, required=expected_indexes, allowed=expected_indexes)


class SwapTest(CoverageTestCase):
  """Tests the Swap."""

  def testOriginalDNARemainsUnchanged(self):
    """Tests that a mutable root node is mutated."""
    dna_spec = pg.geno.manyof(
        3, [pg.geno.constant() for _ in range(10)],
        distinct=False, sorted=False)
    mutator = mutators.Swap(seed=1)
    dna = pg.DNA([1, 8, 9])
    dna.use_spec(dna_spec)
    expected = pg.DNA([1, 8, 9])
    _ = mutator.mutate(dna)
    self.assertEqual(dna, expected)

  def testRandomSeed(self):
    dna_spec = pg.geno.manyof(
        3, [pg.geno.constant() for _ in range(10)],
        distinct=False, sorted=False)
    mutator = mutators.Swap(seed=1)
    self.assertEqual(
        mutator.mutate(pg.DNA([1, 8, 9], spec=dna_spec)),
        pg.DNA([9, 8, 1]))
    self.assertEqual(
        mutator.mutate(pg.DNA([1, 8, 9], spec=dna_spec)),
        pg.DNA([8, 1, 9]))

  def testCoverage(self):
    random.seed()
    dna_spec = pg.geno.manyof(
        3, [pg.geno.constant() for _ in range(10)],
        distinct=False, sorted=False)
    mutator = mutators.Swap()
    dna = pg.DNA([1, 8, 9])
    dna.use_spec(dna_spec)
    def mutate():
      mutated_dna = mutator.mutate(dna)
      return (mutated_dna.children[0].value, mutated_dna.children[1].value,
              mutated_dna.children[2].value)
    expected = [(1, 9, 8), (9, 8, 1), (8, 1, 9)]
    self.assertEventually(func=mutate, required=expected, allowed=expected)

  def testHandlesLackOfCandidates(self):
    dna_spec = pg.geno.oneof([pg.geno.constant() for _ in range(10)])
    mutator = mutators.Swap()
    dna = pg.DNA([5])
    dna.use_spec(dna_spec)
    mutated_dna = mutator.mutate(dna)
    self.assertEqual(mutated_dna, dna)

  def testTargeting(self):
    random.seed()
    dna_spec = pg.geno.space([
        pg.geno.manyof(2, [  # No hints, so not targeted.
            pg.geno.constant(), pg.geno.constant()
        ]),
        pg.geno.manyof(2, [  # Targeted by regex.
            pg.geno.constant(), pg.geno.constant()
        ], hints='bcde'),
        pg.geno.manyof(2, [  # Not targeted by regex.
            pg.geno.constant(), pg.geno.constant()
        ], hints='abcd')
    ])

    def where_fn(dna):
      """Targets nodes according to DNASpec `hints` tag."""
      if dna.spec.hints is None:
        return False  # No hints to target.
      return re.match('^bc.*', dna.spec.hints)
    mutator = mutators.Swap(where=where_fn)

    dna = pg.DNA([[0, 1], [0, 1], [0, 1]])
    dna.use_spec(dna_spec)
    mutated_dna = mutator.mutate(dna)
    self.assertEqual(mutated_dna, pg.DNA([[0, 1], [1, 0], [0, 1]]))


if __name__ == '__main__':
  unittest.main()
