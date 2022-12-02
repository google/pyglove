# Copyright 2021 The PyGlove Authors
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
"""Test for evolution recombinators."""

import unittest
import pyglove.core as pg
from pyglove.ext.evolution import recombinators


class RecombinationTest(unittest.TestCase):
  """Base class for recombination test."""

  def assert_eventual(
      self, recombinator, inputs, expected_outputs, max_attempts=200):
    """Asserts that the output should eventually cover all expected outputs."""
    expected_outputs = {tuple(sorted(o)): 0 for o in expected_outputs}
    matched = 0
    for _ in range(max_attempts):
      outputs = tuple(sorted(recombinator(inputs)))
      assert outputs in expected_outputs, outputs
      prev = expected_outputs[outputs]
      if prev == 0:
        matched += 1
      expected_outputs[outputs] = prev + 1
      if matched == len(expected_outputs):
        break
    remaining = [k for k, v in expected_outputs.items() if v == 0]
    assert not remaining, remaining


class UniformTest(RecombinationTest):
  """Tests for uniform recombinator."""

  def test_flat_search_space(self):
    dna_spec = pg.dna_spec(pg.List(
        [pg.oneof(range(10))]
        + [pg.floatv(0.1, 1.)]
        + [pg.manyof(2, range(10), False, False)]))
    r = recombinators.Uniform(seed=10)
    x = pg.DNA([1, 0.2, [1, 2]], spec=dna_spec)
    y = pg.DNA([3, 0.5, [4, 5]], spec=dna_spec)
    z = pg.DNA([2, 0.6, [3, 3]], spec=dna_spec)
    def _make_output(a, b, c, d):
      return (pg.DNA([a, b, [c, d]]),)
    self.assert_eventual(
        r, [x, y, z],
        [_make_output(*v) for v in pg.iter(pg.List([
            pg.oneof([1, 3, 2]), pg.oneof([0.2, 0.5, 0.6]),
            pg.oneof([1, 4, 3]), pg.oneof([2, 5, 3])]))],
        max_attempts=500)

  def test_hierarchical_search_space(self):
    dna_spec = pg.dna_spec(pg.manyof(3, [
        pg.oneof(['foo', pg.oneof([1, 2])]),
        pg.oneof(range(10)),
        'bar'
    ], False, False))
    r = recombinators.Uniform(seed=1)
    x = pg.DNA([(0, 0), 2, (1, 4)], spec=dna_spec)
    y = pg.DNA([2, (0, 1, 1), (1, 5)], spec=dna_spec)
    z = pg.DNA([(1, 3), 2, (0, 1, 0)], spec=dna_spec)
    def _make_output(a, b, c):
      return (pg.DNA([a, b, c]),)
    self.assert_eventual(
        r, [x, y, z],
        [_make_output(*v) for v in pg.iter(pg.List([
            pg.oneof([(0, 0), 2, (1, 3)]), pg.oneof([2, (0, 1, 1)]),
            pg.oneof([(1, 4), (1, 5), (0, 1, 0)])]))],
        max_attempts=500)

  def test_distinct_multi_choices(self):
    dna_spec = pg.dna_spec(pg.manyof(3, range(10), distinct=True))
    r = recombinators.Uniform(seed=1)
    x = pg.DNA([2, 1, 5], spec=dna_spec)
    y = pg.DNA([1, 0, 7], spec=dna_spec)
    z = pg.DNA([0, 1, 2], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([0, 1, 2]),),
            (pg.DNA([0, 1, 5]),),
            (pg.DNA([0, 1, 7]),),
            (pg.DNA([1, 0, 2]),),
            (pg.DNA([1, 0, 5]),),
            (pg.DNA([1, 0, 7]),),
            (pg.DNA([2, 0, 5]),),
            (pg.DNA([2, 0, 7]),),
            (pg.DNA([2, 1, 5]),),
            (pg.DNA([2, 1, 7]),),
        ])

  def test_sorted_multi_choices(self):
    dna_spec = pg.dna_spec(pg.manyof(3, range(10), distinct=False, sorted=True))
    r = recombinators.Uniform(seed=1)
    x = pg.DNA([1, 1, 5], spec=dna_spec)
    y = pg.DNA([0, 1, 2], spec=dna_spec)
    z = pg.DNA([2, 2, 2], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([0, 1, 2]),),
            (pg.DNA([0, 1, 5]),),
            (pg.DNA([0, 2, 2]),),
            (pg.DNA([0, 2, 5]),),
            (pg.DNA([1, 1, 2]),),
            (pg.DNA([1, 1, 5]),),
            (pg.DNA([1, 2, 2]),),
            (pg.DNA([1, 2, 5]),),
            (pg.DNA([2, 2, 2]),),
            (pg.DNA([2, 2, 5]),),
        ])

  def test_distinct_sorted_multi_choices(self):
    dna_spec = pg.dna_spec(pg.manyof(3, range(10), distinct=True, sorted=True))
    r = recombinators.Uniform(seed=1)
    x = pg.DNA([0, 1, 2], spec=dna_spec)
    y = pg.DNA([0, 2, 5], spec=dna_spec)
    z = pg.DNA([1, 5, 7], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([0, 1, 2]),),
            (pg.DNA([0, 1, 5]),),
            (pg.DNA([0, 1, 7]),),
            (pg.DNA([0, 2, 5]),),
            (pg.DNA([0, 2, 7]),),
            (pg.DNA([0, 5, 7]),),
            (pg.DNA([1, 2, 5]),),
            (pg.DNA([1, 2, 7]),),
            (pg.DNA([1, 5, 7]),),
        ])

  def test_distinct_multi_choices_without_rearrangement(self):
    dna_spec = pg.dna_spec(pg.permutate(range(4)))
    r = recombinators.Uniform(seed=1)
    x = pg.DNA([3, 2, 0, 1], spec=dna_spec)
    y = pg.DNA([0, 3, 1, 2], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([3, 2, 0, 1]),),
            (pg.DNA([0, 3, 1, 2]),),
        ])

  def test_carry_over_unrecombined_parts(self):
    dna_spec = pg.dna_spec(
        pg.Dict(x=pg.floatv(0.1, 1.0), y=pg.oneof(range(10))))
    r = recombinators.Uniform(
        where=lambda xs: [x for x in xs if isinstance(x, pg.geno.Float)],
        seed=1)
    x = pg.DNA([0.2, 1], spec=dna_spec)
    y = pg.DNA([0.5, 2], spec=dna_spec)
    z = pg.DNA([0.3, 7], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([0.2, 1]), pg.DNA([0.2, 2]), pg.DNA([0.2, 7])),
            (pg.DNA([0.5, 1]), pg.DNA([0.5, 2]), pg.DNA([0.5, 7])),
            (pg.DNA([0.3, 1]), pg.DNA([0.3, 2]), pg.DNA([0.3, 7])),
        ])

  def test_complex_case(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=pg.oneof(range(10), name='excluded'),
        y=pg.oneof([
            pg.manyof(2, [
                pg.oneof(range(10)),
                'foo',
                'bar'
            ]),
            1,
            2
        ]),
        z='constant'))

    r = recombinators.Uniform(
        where=lambda xs: [x for x in xs if x.name != 'excluded'], seed=1)
    x = pg.DNA([7, 1], spec=dna_spec)
    y = pg.DNA([5, (0, [(0, 4), 2])], spec=dna_spec)
    z = pg.DNA([3, (0, [1, (0, 7)])], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([7, 1]), pg.DNA([5, 1]), pg.DNA([3, 1])),
            (pg.DNA([7, (0, [(0, 4), 2])]),
             pg.DNA([5, (0, [(0, 4), 2])]),
             pg.DNA([3, (0, [(0, 4), 2])])),
            (pg.DNA([7, (0, [1, (0, 7)])]),
             pg.DNA([5, (0, [1, (0, 7)])]),
             pg.DNA([3, (0, [1, (0, 7)])])),
            (pg.DNA([7, (0, [1, 2])]),
             pg.DNA([5, (0, [1, 2])]),
             pg.DNA([3, (0, [1, 2])])),
        ], max_attempts=500)


class SampleTest(RecombinationTest):
  """Test sample recombinator."""

  def test_simple_case(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=pg.oneof(range(10)),
        y=pg.oneof(range(10))))

    r = recombinators.Sample(lambda parents: [0.0, 0.5, 0.5], seed=1)
    a = pg.DNA([1, 9], spec=dna_spec)
    b = pg.DNA([2, 3], spec=dna_spec)
    c = pg.DNA([5, 4], spec=dna_spec)
    self.assert_eventual(
        r, [a, b, c],
        [
            (pg.DNA([2, 3]),),
            (pg.DNA([2, 4]),),
            (pg.DNA([5, 3]),),
            (pg.DNA([5, 4]),),
        ])

  def test_complex_case(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=pg.oneof(range(10), name='excluded'),
        y=pg.oneof([
            pg.manyof(2, [
                pg.oneof(range(10)),
                'foo',
                'bar'
            ]),
            1,
            2
        ]),
        z='constant'))

    r = recombinators.Sample(
        weights=lambda parents: [0.0, 0.5, 0.5],
        where=lambda xs: [x for x in xs if x.name != 'excluded'], seed=1)
    a = pg.DNA([7, 1], spec=dna_spec)
    b = pg.DNA([5, (0, [(0, 4), 2])], spec=dna_spec)
    c = pg.DNA([3, (0, [1, (0, 7)])], spec=dna_spec)
    self.assert_eventual(
        r, [a, b, c],
        [
            # NOTE(daiyip): dict item y=2 shall never be sampled since
            # a's weight is 0.0.
            (pg.DNA([7, (0, [(0, 4), 2])]),
             pg.DNA([5, (0, [(0, 4), 2])]),
             pg.DNA([3, (0, [(0, 4), 2])])),
            (pg.DNA([7, (0, [1, (0, 7)])]),
             pg.DNA([5, (0, [1, (0, 7)])]),
             pg.DNA([3, (0, [1, (0, 7)])])),
            (pg.DNA([7, (0, [1, 2])]),
             pg.DNA([5, (0, [1, 2])]),
             pg.DNA([3, (0, [1, 2])])),
        ], max_attempts=500)


class AverageTest(RecombinationTest):
  """Test average recombinator."""

  def test_floats_only(self):
    dna_spec = pg.dna_spec(
        pg.Dict(x=pg.floatv(0.1, 0.5), y=pg.floatv(0.0, 1.0)))

    r = recombinators.Average()
    x = pg.DNA([0.1, 0.5], spec=dna_spec)
    y = pg.DNA([0.5, 0.2], spec=dna_spec)
    z = pg.DNA([0.3, 0.8], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([0.3, 0.5]),)
        ],)

  def test_carry_over_unrecombined_parts(self):
    dna_spec = pg.dna_spec(
        pg.Dict(x=pg.floatv(0.1, 0.5, name='excluded'), y=pg.floatv(0.0, 1.0)))

    r = recombinators.Average(
        where=lambda xs: [x for x in xs if x.name != 'excluded'])
    x = pg.DNA([0.1, 0.5], spec=dna_spec)
    y = pg.DNA([0.5, 0.2], spec=dna_spec)
    z = pg.DNA([0.3, 0.8], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [(pg.DNA([0.1, 0.5]), pg.DNA([0.5, 0.5]), pg.DNA([0.3, 0.5]))])


class WeightedAverageTest(RecombinationTest):
  """Test weighted average recombinator."""

  def test_floats_only(self):
    dna_spec = pg.dna_spec(
        pg.Dict(x=pg.floatv(0.0, 1.0), y=pg.floatv(0.0, 1.0)))

    r = recombinators.WeightedAverage(lambda x: [0.2, 0.2, 0.6])
    x = pg.DNA([0.1, 0.5], spec=dna_spec)
    y = pg.DNA([0.5, 0.2], spec=dna_spec)
    z = pg.DNA([0.3, 0.8], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [
            (pg.DNA([0.3, 0.62]),)
        ],)

  def test_carry_over_unrecombined_parts(self):
    dna_spec = pg.dna_spec(
        pg.Dict(x=pg.floatv(0.1, 0.5, name='excluded'), y=pg.floatv(0.0, 1.0)))

    r = recombinators.WeightedAverage(
        lambda x: [0.2, 0.2, 0.6],
        where=lambda xs: [x for x in xs if x.name != 'excluded'])
    x = pg.DNA([0.1, 0.5], spec=dna_spec)
    y = pg.DNA([0.5, 0.2], spec=dna_spec)
    z = pg.DNA([0.3, 0.8], spec=dna_spec)
    self.assert_eventual(
        r, [x, y, z],
        [(pg.DNA([0.1, 0.62]), pg.DNA([0.5, 0.62]), pg.DNA([0.3, 0.62]))])


class KPointTest(RecombinationTest):
  """Test KPoint recombinator."""

  def test_single_point(self):
    dna_spec = pg.dna_spec(
        pg.List([pg.oneof(range(100))] * 4
                + [pg.floatv(0., 1.)]))

    r = recombinators.KPoint(1, seed=2)
    x = pg.DNA([0, 1, 3, 1, 0.5], spec=dna_spec)
    y = pg.DNA([1, 2, 5, 4, 0.3], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, 5, 4, 0.3]), pg.DNA([1, 1, 3, 1, 0.5])),
            (pg.DNA([0, 1, 5, 4, 0.3]), pg.DNA([1, 2, 3, 1, 0.5])),
            (pg.DNA([0, 1, 3, 4, 0.3]), pg.DNA([1, 2, 5, 1, 0.5])),
            (pg.DNA([0, 1, 3, 1, 0.3]), pg.DNA([1, 2, 5, 4, 0.5])),
        ])

  def test_two_point(self):
    dna_spec = pg.dna_spec(pg.List([pg.oneof(range(100))] * 4))

    r = recombinators.KPoint(2, seed=2)
    x = pg.DNA([0, 1, 3, 1], spec=dna_spec)
    y = pg.DNA([1, 2, 3, 4], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, 3, 1]), pg.DNA([1, 1, 3, 4])),
            (pg.DNA([0, 1, 3, 1]), pg.DNA([1, 2, 3, 4])),
        ])

  def test_k_points(self):
    dna_spec = pg.dna_spec(pg.List([pg.oneof(range(100))] * 4))

    # Downgrade into alternating position.
    r = recombinators.KPoint(5, seed=2)
    x = pg.DNA([0, 1, 3, 1], spec=dna_spec)
    y = pg.DNA([1, 2, 5, 4], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, 3, 4]), pg.DNA([1, 1, 5, 1])),
        ])

  def test_independent_multi_choice(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=[pg.oneof(range(10))] * 2,
        y=pg.manyof(2, range(10), distinct=False)))

    r = recombinators.KPoint(2, seed=2)
    x = pg.DNA([0, 1, [3, 1]], spec=dna_spec)
    y = pg.DNA([1, 2, [5, 4]], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, [3, 1]]), pg.DNA([1, 1, [5, 4]])),
            (pg.DNA([0, 2, [5, 1]]), pg.DNA([1, 1, [3, 4]])),
            (pg.DNA([0, 1, [5, 1]]), pg.DNA([1, 2, [3, 4]])),
        ])

  def test_distinct_multi_choice(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=[pg.oneof(range(10))] * 2,
        y=pg.manyof(3, range(10), distinct=True)))

    r = recombinators.KPoint(2, seed=2)
    x = pg.DNA([0, 1, [3, 1, 2]], spec=dna_spec)
    y = pg.DNA([1, 2, [1, 3, 4]], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, [3, 1, 2]]), pg.DNA([1, 1, [1, 3, 4]])),
        ])

  def test_sorted_multi_choice(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=[pg.oneof(range(10))] * 2,
        y=pg.manyof(3, range(10), sorted=True)))

    r = recombinators.KPoint(2, seed=2)
    x = pg.DNA([0, 1, [1, 2, 3]], spec=dna_spec)
    y = pg.DNA([1, 2, [1, 3, 4]], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, [1, 2, 3]]), pg.DNA([1, 1, [1, 3, 4]])),
        ])


class SegmentedTest(RecombinationTest):
  """Test for alternating position recombinator."""

  def test_basics(self):
    dna_spec = pg.dna_spec(pg.List([pg.oneof(range(100))] * 4))
    r = recombinators.Segmented(lambda xs: [1, 3])
    x = pg.DNA([0, 1, 3, 1], spec=dna_spec)
    y = pg.DNA([1, 2, 5, 4], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, 5, 1]), pg.DNA([1, 1, 3, 4]))
        ])


class PartiallyMappedTest(RecombinationTest):
  """Test for partially mapped crossover (PMX)."""

  def test_simple_case(self):
    dna_spec = pg.dna_spec(pg.permutate(range(5)))
    r = recombinators.PartiallyMapped(seed=3)
    x = pg.DNA([0, 1, 2, 4, 3], spec=dna_spec)
    y = pg.DNA([1, 2, 3, 4, 0], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 2, 3, 4, 1]), pg.DNA([3, 1, 2, 4, 0]))
        ])

  def test_non_matching_multi_choices(self):
    dna_spec = pg.dna_spec(
        pg.oneof([
            pg.permutate(range(5)),
            pg.permutate(range(5))
        ]))

    r = recombinators.PartiallyMapped(seed=3)
    x = pg.DNA((0, [0, 1, 2, 4, 3]), spec=dna_spec)
    y = pg.DNA((1, [1, 2, 3, 4, 0]), spec=dna_spec)

    # Output the parents since two multi-choices are active at different branch.
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA((0, [0, 1, 2, 4, 3])), pg.DNA((1, [1, 2, 3, 4, 0]))),
        ])

  def test_multi_choices_with_different_decisions(self):
    dna_spec = pg.dna_spec(pg.Dict(
        x=pg.oneof([1, 2, 3]),
        y=pg.permutate(range(3)),
        z=pg.floatv(0., 1.)))

    r = recombinators.PartiallyMapped(seed=3)
    x = pg.DNA([0, [0, 1, 2], 0.2], spec=dna_spec)
    y = pg.DNA([1, [1, 2, 0], 0.5], spec=dna_spec)

    # The output is a product of non-permutated parts of the parents and the
    # permutated parts.
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, [0, 1, 2], 0.2]),
             pg.DNA([1, [0, 1, 2], 0.5]),
             pg.DNA([1, [1, 2, 0], 0.5]),
             pg.DNA([0, [1, 2, 0], 0.2]))
        ])

  def test_nested_matching_multi_choices(self):
    dna_spec = pg.dna_spec(pg.oneof([
        1,
        pg.permutate(range(5))
    ]))

    r = recombinators.PartiallyMapped(seed=3)
    x = pg.DNA((1, [0, 1, 2, 4, 3]), spec=dna_spec)
    y = pg.DNA((1, [1, 2, 3, 4, 0]), spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA((1, [1, 2, 3, 4, 0])), pg.DNA((1, [0, 1, 2, 4, 3])))
        ])

  def test_where_clause(self):
    dna_spec = pg.dna_spec(pg.List([
        pg.permutate(range(5), name='a'),
        pg.permutate(range(5), name='b'),
        pg.permutate(range(5), name='c')
    ]))

    r = recombinators.PartiallyMapped(
        where=lambda xs: [x for x in xs if x.name != 'b'], seed=5)
    x = pg.DNA([[0, 1, 2, 3, 4],
                [1, 2, 3, 4, 0],
                [2, 3, 4, 0, 1]], spec=dna_spec)
    y = pg.DNA([[3, 4, 0, 1, 2],
                [4, 0, 1, 2, 3],
                [0, 1, 2, 3, 4]], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            # Crossover happens on the 2rd column. Without the `where`
            # statement it will happen on the 1st and 3rd column.
            (pg.DNA([[3, 4, 0, 1, 2], [4, 0, 1, 2, 3], [2, 3, 0, 1, 4]]),
             pg.DNA([[2, 3, 0, 1, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]]),
             pg.DNA([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 0, 1, 4]]),
             pg.DNA([[3, 4, 0, 1, 2], [4, 0, 1, 2, 3], [0, 1, 4, 2, 3]]),
             pg.DNA([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [0, 1, 4, 2, 3]]),
             pg.DNA([[1, 4, 2, 3, 0], [4, 0, 1, 2, 3], [0, 1, 2, 3, 4]]),
             pg.DNA([[2, 3, 0, 1, 4], [4, 0, 1, 2, 3], [0, 1, 2, 3, 4]]),
             pg.DNA([[1, 4, 2, 3, 0], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]]))
        ])


class OrderCrossoverTest(RecombinationTest):
  """Test for order crossover (OX)."""

  def test_correctness(self):
    """Refer to https://www.hindawi.com/journals/cin/2017/7430125."""
    r = recombinators.Order()
    self.assertEqual(
        r.order_crossover([[3, 4, 8, 2, 7, 1, 6, 5],
                           [4, 2, 5, 1, 6, 8, 3, 7]], 3, 6),
        [[4, 2, 7, 1, 6, 8, 5, 3], [5, 6, 8, 2, 7, 1, 3, 4]])

  def test_simple_case(self):
    dna_spec = pg.dna_spec(pg.permutate(range(5)))
    r = recombinators.Order(seed=4)
    x = pg.DNA([0, 1, 2, 4, 3], spec=dna_spec)
    y = pg.DNA([1, 2, 3, 4, 0], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([1, 2, 4, 3, 0]), pg.DNA([2, 1, 3, 4, 0]))
        ])


class CycleCrossoverTest(RecombinationTest):
  """Test for cycle crossover (CX)."""

  def test_correctness(self):
    """Refer to https://www.hindawi.com/journals/cin/2017/7430125."""
    r = recombinators.Cycle(seed=4)
    self.assertEqual(
        r.permutate(None, [[1, 2, 3, 4, 5, 6, 7, 8],
                           [8, 5, 2, 1, 3, 6, 4, 7]]),
        [[1, 5, 2, 4, 3, 6, 7, 8], [8, 2, 3, 1, 5, 6, 4, 7]])

  def test_simple_case(self):
    dna_spec = pg.dna_spec(pg.permutate(range(5)))
    r = recombinators.Cycle(seed=4)
    x = pg.DNA([0, 1, 2, 3, 4], spec=dna_spec)
    y = pg.DNA([1, 0, 3, 4, 2], spec=dna_spec)
    self.assert_eventual(
        r, [x, y],
        [
            (pg.DNA([0, 1, 3, 4, 2]), pg.DNA([1, 0, 2, 3, 4]))
        ])


if __name__ == '__main__':
  unittest.main()
