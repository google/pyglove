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
"""Common recombinators for evolutionary algorithms.

Types of recombinators
**********************

This file implements 3 types of recombinators: point-wise, segment-wise and
permutation.

Point-wise recombinators
========================
A point-wise recombinator works for an arbitrary number of parents, it
crossovers the parents' DNA into offspring's DNA point-by-point. For example,
`Uniform` takes a random parent's decisions per decision point, `Sample` samples
one of the parents' decision with probabilities computed from a weighting
function provided by the user, and `Average` works for float decision points
by averaging parents' values into the child's. When dealing with categorical
decision points (e.g. `pg.oneof`, `pg.manyof`),  `Sample` can be very sample
efficient when the decision points are orthgonal to each other.

Segment-wise recombinators
==========================
A segment-wise recombinator works on 2 parents, it cuts the parents' DNA into
multiple segments, and chooses alternating semgents from the parents. For
example, if the i'th segment of one parent is taken as the i'th segment of a
child, the (i + 1)'th segment of the child will be taken from another parent.
Each recombination produces 2 children which start with a segment from different
parents. For example::

    parent 1:   1  2  | 3  4  5  6  | 7  8  9
    parent 2:   10 20 | 30 40 50 60 | 70 80 90

    child 1:    1  2  | 30 40 50 60 | 7  8  9
    child 2:    10 20 | 3  4  5  6  | 70 80 90

Well-known segment-wise recombinators are single-point crossover (SPX),
two-point crossover (TPX), K-point crossover and more generally, the segmented
crossover. While each of the former three chooses a fixed number of cutting
points randomly, the segmented crossover allows the user to specify a function
to produce the cutting points, which can be done dynamically based on the global
state and step. Also the customized cutting point can be effective when the user
knows how the search space should be partitioned based on the application.

Permutations
============

A permutation recombinator works on 2 parents by permutating the order of the
subchoices of multi-choice decision points, which can be useful in applications
in which the order of choices matters (e.g. the traveling salesman problem).
Well-know permutation crossovers are partially mapped crossover (PMX), order
crossover (OX) and cycle crossover (CX).

Which recombinators to use?
***************************

Semantics on hyper primitves
============================

+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
|             |           |    #    |        |       |         manyof        |
|             |           | parents | floatv | oneof |  sorted Y |  sorted N |
|             |           |         |        |       |  distinct |  distinct |
|             |           |         |        |       |  Y  |  N  |  Y  |  N  |
+=============+===========+=========+========+=======+=====+=====+=====+=====+
|Point-wise   | Uniform   |   > 0   |   X    |   X   |  X  |  X  |  X  |  X  |
+             +-----------+---------+--------+-------+-----+-----+-----+-----+
|             | Sample    |   > 0   |   X    |   X   |  X  |  X  |  X  |  X  |
+             +-----------+---------+--------+-------+-----+-----+-----+-----+
|             | Average   |   > 0   |   X    |       |     |     |     |     |
+             +-----------+---------+--------+-------+-----+-----+-----+-----+
|             | W-Average |   > 0   |   X    |       |     |     |     |     |
+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
|Segment-wise | KPoint    |    2    |   X    |   X   | X (treated as a |  X  |
|             |           |         |        |       | single decision |     |
+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
|             | Segmented |    2    |   X    |   X   | point)          |  X  |
+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
|Permutation  | PMX       |    2    |        |       |     |     |  X  |     |
+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
|             | Order     |    2    |        |       |     |     |  X  |     |
+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
|             | Cycle     |    2    |        |       |     |     |  X  |     |
+-------------+-----------+---------+--------+-------+-----+-----+-----+-----+
* A blank cell means the recombinator works as a no-op on the hyper primitive.

Disruptiveness of recombinations
================================

A recombinator is more disruptive if it produces children that has more
differences from the parents. In that regard, `Uniform` is more disruptive than
`Sample` when a fitness-based weighting function i used. `KPoint` with a larger
K is more disruptive than with a smaller K.

The more disruptive an operation is, the more diversity the operation produces.
Historically studies shows that with small populations, more disruptive
recombination such as `Uniform` or `K-Point` (k >> 2) may yield better results
because they help overcome the limited information capacity of smaller
populations and the tendency for more homogencity. With larger populations, less
disruptive recombinations like 2-point are more likely to work better.
(Reference: Holland, John H. (1975). Adaptation in Natural and Artificial
Systems, The University of Michigan Press.)

When we need faster convergence, `Sample` with a fitness-based weighting
function will be very effective, which can be used to implement a point-wise
greedy strategy.

On recombination for float values
=================================

Adding an `Average` or `WeightedAverage` when you have float decision points in
the search space (created by `pg.floatv`). It is a no-op when there is no float
decision points in the space.
"""

import abc
import random
import typing
from typing import List, Optional, Union

import pyglove.core as pg

from pyglove.ext import scalars
from pyglove.ext.evolution import base
from pyglove.ext.evolution import where


# We disable implicit str concat as it is commonly used class schema docstr.
# pylint: disable=implicit-str-concat


#
# Point-wise recombinators.
#


@pg.members([
    ('where', where.where_spec(),
     'A callable object that takes a list of decision points as input, and '
     'returns a list of decision points as applicable points for '
     'recombination. If not specified, `pg.evolution.where.ALL` will be used.')
])
class PointWise(base.Recombinator):
  """Base class for point-wise recombinators.

  A point-wise recombinator operates on decision points (abbr. points) from the
  search space one by one. It crossovers the values from the parents into the
  offspring's decision on each target point. For those points that are not
  targeted, the decisions from the parents will be copied to the offsprings.
  Therefore, N parents can result in N children when at least 1 decision points
  are not applicable for crossover, otherwise 1 child will be produced.

  Example::

    parents = [pg.DNA([0, 1, 0.1, 0.2]), pg.DNA([2, 3, 0.5, 0.4])]
    # `children` will be [pg.DNA(0, 1, 0.3, 0.3), pg.geno(2, 3, 0.3, 0.3)].
    children = pg.evolution.recombinators.Average()(parents)

    parents = [pg.DNA([0.1, 0.2]), pg.DNA([0.5, 0.4])]
    # `children` will be [pg.DNA(0.3, 0.3)].
    children = pg.evolution.recombinators.Average()(parents)

  Targeted decision points are points that are applicable for current
  recombinator according to its semantics, as well as passing the `where`
  statement if it's specified. The `PointWise` base class depends on the
  `applicable_decision_points` method to select all applicable points from the
  search space. If not overridden by subclasses, the method returns all decision
  points in the search space, with multi-choice folded into a single decision
  point. When the `where` argument is provided by the user, it further filters
  out unwanted decision points, which is useful when we want to limit the range
  of points for crossover.

  Example::

    search_space = pg.Dict(
        x=pg.oneof(range(5)),
        y=pg.oneof(range(5), hints='excluded'))
    parents = [pg.DNA([0, 1]), pg.DNA([2, 3])]

    # The children will be one of the following:
    #   [DNA([0, 1]), DNA([2, 1])] and [DNA([0, 3]), DNA([2, 3])].
    children = pg.evolution.recombinators.Uniform(
        where=lambda xs: [x for in xs if x.hints != 'excluded'])(parents)
  """

  def _on_bound(self):
    super()._on_bound()
    self._applicable_decision_points = pg.typing.CallableWithOptionalKeywordArgs(
        self.applicable_decision_points, ['global_state', 'step'])
    self._merge = pg.typing.CallableWithOptionalKeywordArgs(
        self.merge, ['global_state', 'step'])

  def recombine(
      self,
      parents: List[pg.DNA],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[pg.DNA]:
    if not parents:
      return []
    dna_spec = parents[0].spec
    target_decision_points = self.where([
        x for x in self._applicable_decision_points(
            dna_spec, global_state=global_state, step=step)
    ], global_state=global_state, step=step)
    parent_dicts = [
        p.to_dict('dna_spec',
                  multi_choice_key='parent',
                  include_inactive_decisions=True) for p in parents]

    for dp in target_decision_points:
      parent_decisions = [p[dp] for p in parent_dicts]
      if all(d is None for d in parent_decisions):
        decision = None
      else:
        decision = self._merge(
            dp, parent_decisions, global_state=global_state, step=step)

      # Use the recombined decision to update the decision points on all parent
      # dicts, each dict will be used for creating a child later.
      if decision is not None:
        # NOTE(daiyip): for conditional search spaces, if a parent's decision
        # for the outer space has changed, it should not contribute to the
        # decisions within the subspace. This is done by setting the decision
        # of inner points to None for that parent. When the inner points are
        # merged later, only the parents who has originally selected the
        # subspace will have not-None values on that child decision point.
        for old_decision, parent_dict in zip(parent_decisions, parent_dicts):
          if isinstance(dp, pg.geno.Choices) and dp.num_choices > 1:
            for i in range(dp.num_choices):
              if old_decision is None or old_decision[i] != decision[i]:
                for child_dp in self._applicable_decision_points(
                    dp.subchoice(i), global_state=global_state, step=step):
                  parent_dict[child_dp] = None
          else:
            if old_decision != decision:
              for child_dp in self._applicable_decision_points(
                  dp, global_state=global_state, step=step):
                parent_dict[child_dp] = None
          # Update each parent dict for creating a child later.
          parent_dict[dp] = decision
    return list(set(pg.DNA.from_dict(pd, dna_spec) for pd in parent_dicts))

  def applicable_decision_points(
      self,
      dna_spec: pg.geno.DNASpec,
      global_state: pg.geno.AttributeDict,
      step: int) -> List[pg.geno.DecisionPoint]:
    """Returns applicable decision points for this recombinator.

    The default behavior is to return all decision points in the search space,
    with multi-choice subchoices folded into a single decision point. Subclasses
    can override this method to select applicable points according to their
    semantics.

    Args:
      dna_spec: The root DNASpec.
      global_state: An optional keyword argument as the global state. Subclass
        can omit.
      step: An optional keyword argument as current step. Subclass can omit.

    Returns:
      A list of targeted decision points for point-wise recombination, which
        will be further filtered by the `where` statement later.
    """
    applicable_points = []
    for dp in dna_spec.decision_points:
      # Fold multi-choice subchoices into a single decision point.
      if isinstance(dp, pg.geno.Choices) and dp.is_subchoice:
        if dp.subchoice_index == 0:
          applicable_points.append(dp.parent_spec)
      else:
        applicable_points.append(dp)
    return applicable_points

  @abc.abstractmethod
  def merge(
      self,
      decision_point: pg.geno.DecisionPoint,
      parent_decisions: List[Union[int, List[int], float, None]],
      global_state: pg.geno.AttributeDict,
      step: int) -> Union[int, List[int], float]:
    """Implementation of point-wise decision making.

    Args:
      decision_point: Decision point for recombination.
      parent_decisions: A list of parent's decisions. Each item should be an
        int as an active single-choice decision, a list of int as active multi-
        choice decisions, a float as an active float decision, or None for
        inactive decision point (whose parent space is not chosen).
      global_state: An optional keyword argument as the global state. Subclass
        can omit.
      step: An optional keyword argument as the current step. Subclass can omit.

    Returns:
      An int, list of int or float as the decision made for the decision point.
    """


@pg.members([
    ('seed', pg.typing.Int().noneable(), 'Random seed.'),
], init_arg_list=['where', 'seed'])
class Uniform(PointWise):
  """Uniform crossover (UX) with equal probability.

  The uniform recombinator makes decision at each position on whether the two
  parents' DNA should be swapped or not. It produces 2 children.

  Reference:

  G. Syswerda. 1989. Uniform crossover in genetic algorithms.
  In Proceedings of the 3rd International Conference on Genetic Algorithms.
  Morgan Kaufman, 2–9.

  https://ci.nii.ac.jp/naid/10000012509/
  """

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)

  def merge(  # pytype: disable=signature-mismatch
      self,
      decision_point: pg.geno.DecisionPoint,
      parent_decisions: List[Union[int, List[int], float, None]]
      ) -> Union[int, List[int], float]:
    if (isinstance(decision_point, pg.geno.Choices)
        and decision_point.num_choices > 1):
      return _merge_multi_choice(
          decision_point,
          parent_decisions,
          [1.] * len(parent_decisions),
          self._random)
    else:
      return self._random.choice([v for v in parent_decisions if v is not None])


def _merge_multi_choice(
    decision_point: pg.geno.Choices,
    parent_decisions: List[Union[None, List[int]]],
    weights: List[float],
    rand,
    max_rearrange_attempts: int = 8) -> List[int]:
  """Merge multi choice decisions with possible constraints."""
  results = []
  adjusted_weights = [
      0 if p is None else w for p, w in zip(parent_decisions, weights)]

  def _merge_next(index, attempts):
    if index == decision_point.num_choices:
      return True
    candidates = [None if p is None else p[index] for p in parent_decisions]
    while attempts < max_rearrange_attempts:
      decision = rand.choices(candidates, weights=adjusted_weights, k=1)[0]
      if ((not decision_point.distinct or decision not in results)
          and (not decision_point.sorted
               or not results or decision >= results[-1])):
        results.append(decision)
        break
      attempts += 1
    if attempts == max_rearrange_attempts:
      return False
    return _merge_next(index + 1, attempts)
  if _merge_next(0, 0):
    return results
  return rand.choices(parent_decisions, weights=adjusted_weights, k=1)[0]


@pg.members([
    ('weights', base.operation_spec(
        pg.typing.Object(pg.DNA), pg.typing.Float(min_value=0.0)),
     'A callable object that takes the parents DNA as input, and output the '
     'a list of float as their weights.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed.')
], init_arg_list=['weights', 'where', 'seed'])
class Sample(PointWise):
  """Point-wise crossover that sample values from parents by weights.

  The `Sample` recombinator works similarly as the `Uniform` recombinator,
  except that it takes a user function to compute the weights, based on which
  each parent's decision will be sampled. `Uniform` can be represented as
  `Sample(lambda xs: [1] * len(xs))`, whose `weights` function generates the
  sampling weights in uniform distribution.
  """

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)
    self._weights = base.make_operation_compatible(self.weights)
    self._parent_weights = []

  def _on_input(self, inputs: List[pg.DNA]) -> None:
    super()._on_input(inputs)
    self._parent_weights = self._weights(inputs)

  def merge(    # pytype: disable=signature-mismatch
      self,
      decision_point: pg.geno.DecisionPoint,
      parent_decisions: List[Union[int, List[int], float, None]]
      ) -> Union[int, List[int], float]:
    assert len(parent_decisions) == len(self._parent_weights)
    adjusted_weights = [0 if parent_decisions[i] is None else w
                        for i, w in enumerate(self._parent_weights)]
    if (isinstance(decision_point, pg.geno.Choices)
        and decision_point.num_choices > 1):
      return _merge_multi_choice(
          decision_point,
          parent_decisions,
          adjusted_weights,
          self._random)
    else:
      return self._random.choices(
          parent_decisions, weights=adjusted_weights, k=1)[0]


class Numeric(PointWise):
  """Base class for numerical recombinators.

  A numeric recombinator operates on `pg.geno.Float` decision points, by
  recombining parents' values into child values.

  Decisions of `pg.geno.Choices` will be copied over from N parents to N
  children on an one-to-one basis. If there is no choice decision points
  in the search space and  `where` clause does not filer any float points out,
  there will be a single child produced.

  Example::

    parents = [pg.DNA([0, 1, 0.1, 0.2]), pg.DNA([2, 3, 0.5, 0.4])]
    # `children` will be [pg.DNA(0, 1, 0.3, 0.3), pg.geno(2, 3, 0.3, 0.3)].
    children = pg.evolution.recombinators.Average()(parents)

    parents = [pg.DNA([0.1, 0.2]), pg.DNA([0.5, 0.4])]
    # `children` will be [pg.DNA(0.3, 0.3)].
    children = pg.evolution.recombinators.Average()(parents)
  """

  def applicable_decision_points(    # pytype: disable=signature-mismatch
      self, dna_spec: pg.geno.DNASpec) -> List[pg.geno.DecisionPoint]:
    return [dp for dp in dna_spec.decision_points
            if isinstance(dp, pg.geno.Float)]


@pg.members([], init_arg_list=['where'])
class Average(Numeric):
  """Average crossover.

  Average crossover performs point-wise average of the parents' decisions on
  float decision points. We include this recombinator to help float decisions
  converge.

  References:
  https://link.springer.com/content/pdf/10.1007/s00500-006-0049-7.pdf
  """

  def merge(   # pytype: disable=signature-mismatch
      self,
      decision_point: pg.geno.DecisionPoint,
      parent_decisions: List[Optional[float]]) -> float:
    del decision_point
    parent_decisions = [d for d in parent_decisions if d is not None]
    return sum(parent_decisions) / len(parent_decisions)


@pg.members([
    ('weights', base.operation_spec(
        pg.typing.Object(pg.DNA), pg.typing.Float(min_value=0.0)),
     'A callable object that takes the parents DNA as input, and output the '
     'a list of float as their weights.')
], init_arg_list=['weights', 'where'])
class WeightedAverage(Numeric):
  """Weighted-average crossover.

  Similar as the `Average` crossover, a weighted-average crossover operates on
  all float decision points and carries over other parts of the chromosome from
  each parent to a child. Thus it produces the same number of children as the
  parents.

  It uses formula `cv[i] = sum(pw[j] * pv[j][i]) / sum(pw[j])` to compute the
  values for all or selected float points. `cv[i]` is the value of the i-th
  float point of the child DNA. `pw[j]` is the weight computed from the j-th
  parent, and `pv[j][i]` is the j-th parent's value on the i-th float point.

  References:
  https://link.springer.com/content/pdf/10.1007/s00500-006-0049-7.pdf
  """

  def _on_bound(self):
    super()._on_bound()
    self._weights = base.make_operation_compatible(self.weights)
    self._parent_weights = []

  def _on_input(self, inputs: List[pg.DNA]) -> None:
    super()._on_input(inputs)
    self._parent_weights = self._weights(inputs)

  def merge(   # pytype: disable=signature-mismatch
      self,
      decision_point: pg.geno.Float,
      parent_decisions: List[Optional[float]]) -> float:
    del decision_point
    decision = 0.0
    denominator = 0.0
    for d, w in zip(parent_decisions, self._parent_weights):
      if d is not None:
        decision += w * d
        denominator += w
    return decision / denominator


#
# Segment-wise recombinators.
#


class SegmentWise(base.Recombinator):
  """Base for recombinators that interleavingly take segments from parents' DNA.

  A segment-wise recombinator usually involve 2 parents. It cuts both parents'
  DNA at K positions, which forms K + 1 segments. Then it takes the K + 1
  segments from both parents in an interleaving manner. For example, a two-point
  crossover that takes places at cutting points 2 and 5 will result in
  2 children as follows::

       Parent 1:   [1,  2,  | 3,  4,  5,  | 6,  7]
       Parent 2:   [10, 20, | 30, 40, 50, | 60, 70]
                            |             |
       Child 1:    [1,  2,  | 30, 40, 50, | 6,  7]
       Child 2:    [10, 20, | 3,  4,  5,  | 60, 70]

  One important aspect of recombination is that the children produced from
  recombination should be valid. Decisions can be moved around when and only
  when they are independent from other decisions. In PyGlove, there are multiple
  interdependent DNA sequences: 1) conditional space, whose DNA is represented
  by tuples (e.g. (1, 2) means inner-choice 2 is made under outer-choice 1).
  2) subchoices of a multi-choice which has sorted and/or distinct constraint.
  Each interdependent DNA group, like a sub-tree for conditional space, or a
  list of subchoices for constrained multi-choice will be treated as a single
  position when the DNA is being cut into segments. For example,
  DNA([0, 1, [1, 2, 0, 3]]) has length 3 if the multi-choice is sorted or
  distinct, otherwise its length is 6.

  The varitions of segment-wise recombination differ from each other in how they
  choose the cutting strategies. `KPoint` randomly choose K cutting points in
  the DNA sequence, `AlternatingPosition` cuts the DNA at every position thus
  forming a list of length-1 segments. `Segment` lets the user to customize
  cutting strategies based on a list of applicable decision points.
  """

  NUM_PARENTS = 2

  def _on_bound(self):
    super()._on_bound()
    self._cutting_indices_fn = pg.typing.CallableWithOptionalKeywordArgs(
        self.cutting_indices, ['global_state', 'step'])

  def recombine(
      self,
      parents: List[pg.DNA],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[pg.DNA]:
    x, y = parents
    dna_spec = x.spec

    # Find top-level decision points that are independent from each other.
    independent_decision_points = []
    for dp in dna_spec.decision_points:
      if dp.parent_choice is None:
        # For multi-choices which have distinct or sorted constraint, treat
        # the multi-choice as an indepedent point, otherwise treat each
        # subchoices as an independent point.
        if (isinstance(dp, pg.geno.Choices)
            and dp.is_subchoice and (dp.distinct or dp.sorted)):
          if dp.subchoice_index == 0:
            independent_decision_points.append(dp.parent_spec)
        else:
          independent_decision_points.append(dp)

    # Compute segment ending indices.
    segment_ends = self._cutting_indices_fn(
        independent_decision_points, global_state=global_state, step=step)
    segment_ends.append(len(independent_decision_points))

    child1, child2 = dict(), dict()
    start = 0
    for i, cp in enumerate(segment_ends):
      segments = independent_decision_points[start:cp]
      for dp in segments:
        child1[dp] = x[dp] if i % 2 == 0 else y[dp]
        child2[dp] = y[dp] if i % 2 == 0 else x[dp]
      start = cp
    return [pg.DNA.from_dict(child1, dna_spec),
            pg.DNA.from_dict(child2, dna_spec)]

  @abc.abstractmethod
  def cutting_indices(
      self,
      independent_decision_points: List[pg.geno.DecisionPoint],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[int]:
    """Implementation of getting the indices of the cutting points.

    Args:
      independent_decision_points: A list of independent decision points.
      global_state: An optional keyword argument as the global state. Subclass
        can omit.
      step: An optional keyword argument as the curent step. Subclass can omit.

    Returns:
      A list of integers as the cutting points.
    """


@pg.members([
    ('k', scalars.scalar_spec(pg.typing.Int(min_value=1)),
     'Number of point for crossover.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed.')
], init_arg_list=['k', 'seed'])
class KPoint(SegmentWise):
  """K-point crossover.

  K-point crossover is one of the basic crossovers in evolutionary algorithms.
  It cuts both parents' DNA at K positions, which forms K + 1 segments. Then it
  takes the K + 1 segments from both parents in an interleaving manner.
  For example, a two-point crossover that takes places at cutting points 2 and 5
  will result in 2 children as follows::

       Parent 1:   [1,  2,  | 3,  4,  5,  | 6,  7]
       Parent 2:   [10, 20, | 30, 40, 50, | 60, 70]
                            |             |
       Child 1:    [1,  2,  | 30, 40, 50, | 6,  7]
       Child 2:    [10, 20, | 3,  4,  5,  | 60, 70]

  2 children as follows:
  When K=1, we get a single-point crossover. Similarily, when K=2, we get a
  two-point crossover.

  When K equals or is greater than the length of DNA sequence, we get an
  alternating-position (APX) crossover.

  Reference:
  https://mitpress.mit.edu/books/introduction-genetic-algorithms
  https://dl.acm.org/doi/abs/10.5555/93126.93134
  https://www.intechopen.com/chapters/335
  """

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)

  def cutting_indices(
      self,
      independent_decision_points: List[pg.geno.DecisionPoint],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[int]:
    """Returns the indices of cutting points for a list decision points."""
    del global_state
    k = scalars.scalar_value(self.k, step)
    if len(independent_decision_points) > k + 1:
      indices = sorted(self._random.sample(
          list(range(1, len(independent_decision_points))), k=k))
    else:
      indices = list(range(1, len(independent_decision_points)))
    return indices


@pg.members([
    ('cutting_points', pg.typing.Callable(
        [pg.typing.List(pg.typing.Object(pg.geno.DecisionPoint))],
        returns=pg.typing.List(pg.typing.Int(min_value=0))),
     'A callable that returns cutting points for a list of decision points.')
], init_arg_list=['cutting_points'])
class Segmented(SegmentWise):
  """Segmented crossover.

  Instead of using a predefined cutting strategy, the segmented recombinator
  allows the user to customize how the cutting points should be chosen, which
  can be used to implement fixed cutting point strategies (e.g nodes + edges)
  as well as decision points' based cutting point strategies.

  Example::

    # A fixed single-point crossover at the mid of the chrosome:
    pg.evolution.Segmented(lambda xs: [len(xs) / 2])

    # A dynamic multi-point crossover based on hints.
    pg.evolution.Segmented(lambda xs: [x.hints == 'block_end' for x in xs])

  """

  def cutting_indices(
      self,
      independent_decision_points: List[pg.geno.DecisionPoint],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[int]:
    """Returns the indices of cutting points for a list decision points."""
    return self.cutting_points(independent_decision_points)


#
# Permutation recombinators.
#


@pg.members([
    ('where', where.where_spec(where.ANY),
     'A callable object as the decision point filter. By default, '
     '`pg.evolution.where.ANY` will be used.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed.')
])
class Permutation(base.Recombinator):
  """Base for recombinators that permutate the multi-choice subchoices.

  A permutation recombinator operates on target permutation decision points.
  A permutation point is a multi-choice decision point that has distinct but
  unsorted subchoices, also with a `num_choices` equals to the number of its
  candidates. A permutation decision point can be created via `pg.permutate`
  or ``pg.manyof(k=len(candidates), candidates)``.

  Example::

    pg.Dict(x=pg.manyof(2, range(5)), y=pg.permutate(range(3)))

  contains 1 permutation decision point, as the first ``pg.manyof`` has only
  2 subchoices while the number of candidates is 5.

  A permutation point is targeted if it's included in the return value of the
  `where` function when it's specified. By default, the `where` function is set
  to `where.ANY`, which returns 1 random point among all the permutation points
  in the search space. When users specify the `where` argument to select more
  than 1 permutation points, the original DNA from each parent will be merged
  with each permutation proposal to generate a child. There will be a multiply
  effect between the number of parents and the number of proposals for each
  permutation, but the proposals generated from different permutation points
  will not be multiplied. That being said, if there are N parents, N propoals
  per crossover and K crossovers (K=1 by default), the max number of children
  will be `N * M * K`. Most permutation recombinators (PMX, OX, CX, etc.)
  operates on 2 parents and produces 2 children in a single crossover,
  resulting `N * M * K = 2 * 2 = 4` children.

  For example, if DNA([0, 1, [1, 2, 3, 0]]) and DNA([2, 3, [0, 1, 2, 3]]) are
  recombinated on [1, 2, 3, 0] and [0, 1, 2, 3], which outputs [0, 2, 3, 1] and
  [3, 1, 2, 0] as recombined results. Then there will be 4 DNA in the output::

    DNA([0, 1, [0, 2, 3, 1]])
    DNA([0, 1, [3, 1, 2, 0]])
    DNA([2, 3, [0, 2, 3, 1]])
    DNA([2, 3, [3, 1, 2, 0]])

  It's worthy noting that though common permutation operations takes 2 parents,
  the `Permutation` base class is designed to support arbitrary number of
  parents, use the `NUM_PARENTS` property to specify intended parent number if
  subclass needs a fixed parent number.
  """

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)
    if self.where.sym_hasattr('seed'):
      self.where.rebind(seed=self.seed, skip_notification=True)

  def recombine(
      self,
      parents: List[pg.DNA],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[pg.DNA]:
    if not parents:
      return []

    def possible_permutation_points(
        dna_spec: pg.geno.DNASpec, dna_list: List[pg.DNA]):
      """Find possible permutation point under DNASpec."""
      if (any(dna is None or dna_spec is not dna.spec
              for dna in dna_list)):
        return []

      # All DNA should have the same value at current position to get into.
      if len(set(dna.value for dna in dna_list if dna)) != 1:
        return []

      permutation_candidates = []
      multi_choice_spec = dna_list[0].multi_choice_spec
      if (multi_choice_spec is not None
          and len(multi_choice_spec.candidates) == multi_choice_spec.num_choices
          and multi_choice_spec.distinct and not multi_choice_spec.sorted):
        permutation_candidates.append(multi_choice_spec)

      # Traverse children.
      for i in range(len(dna_list[0].children)):
        sub_spec = dna_list[0].children[i].spec
        sub_dna_list = [dna.children[i] for dna in dna_list]
        permutation_candidates.extend(
            possible_permutation_points(sub_spec, sub_dna_list))
      return permutation_candidates

    dna_spec = parents[0].spec

    # We perform one read-only pass to extract all possible permutation points
    # from matched parent DNA trees and use the `where` clause to select
    # permutation points.
    permutation_points = self.where(
        possible_permutation_points(dna_spec, parents),
        global_state=global_state, step=step)

    # If there exists any permutation candidate, we random choose one as the
    # permutation target, and do a write-only pass
    if permutation_points:
      outputs = []

      # Get parents' decisions at the permutation point.
      for permutation_point in permutation_points:
        parent_dicts = [p.to_dict('dna_spec', 'dna', 'parent') for p in parents]

        # Permutate original parents' decisions and get child decisions.
        # Each child is a list of integer, representing the DNA value for
        # subchoices of a multi-choice.
        permutation_proposals: List[List[int]] = self.permutate(
            permutation_point,
            [[dna.value for dna in p[permutation_point]] for p in parent_dicts])

        # For each child proposal, merge it back with each parent's DNA to
        # produce an output. Therefore if there are N parents and M permuation
        # proposals, the maximum number of output is N * M. Duplicates will be
        # removed so it may lead to a smaller number than N * M.
        for parent_dict in parent_dicts:
          subdna_map = {d.value: d for d in parent_dict[permutation_point]}
          for proposal in permutation_proposals:
            parent_dict[permutation_point] = [subdna_map[v] for v in proposal]
            outputs.append(pg.DNA.from_dict(parent_dict, dna_spec))
      outputs = list(set(outputs))
      return outputs
    return parents

  @abc.abstractmethod
  def permutate(
      self,
      multi_choice_spec: pg.geno.Choices,
      parents: List[List[int]]) -> List[List[int]]:
    """"Permutate decisions for a multi_choice_spec."""


@pg.members([], init_arg_list=['where', 'seed'])
class PartiallyMapped(Permutation):
  """Partially mapped crossover (PMX).

  The partially mapped crossover (PMX) was proposed by D. Goldberg and
  R. Lingle, “Alleles, Loci and the Traveling Salesman Problem,” in
  Proceedings of the 1st International Conference on Genetic Algorithms and
  Their Applications, vol. 1985, pp. 154–159, Los Angeles, USA.

  Reference: https://dl.acm.org/doi/10.5555/645511.657095
  """

  # PMX supports two parents only.
  NUM_PARENTS = 2

  def permutate(
      self,
      multi_choice_spec: pg.geno.Choices,
      parents: List[List[int]]) -> List[List[int]]:
    del multi_choice_spec
    size = len(parents[0])
    start, end = 0, size
    while end - start == size:
      start, end = sorted(self._random.sample(list(range(0, size)), k=2))
    return self.partially_mapped_crossover(parents, start, end)

  def partially_mapped_crossover(
      self, parents: List[List[int]], start: int, end: int) -> List[List[int]]:
    """Cross over and remap the rest elements at given cutting points."""
    assert len(parents) == 2
    size = len(parents[0])

    children, assigned, indices = [], [], []
    for i in range(2):
      child = [None] * size
      child[start:end] = parents[(i + 1) % 2][start:end]
      children.append(child)

      assigned.append(set(c for c in children[i] if c is not None))
      indices.append({v: j for j, v in enumerate(parents[i])})

    positions = list(range(start)) + list(range(end, size))
    for i in range(2):
      for j in positions:
        v = parents[i][j]
        k = (i + 1) % 2
        while v in assigned[i]:
          v = parents[(k + 1) % 2][indices[k][v]]
        children[i][j] = v
        assigned[i].add(v)
    return children


@pg.members([], init_arg_list=['where', 'seed'])
class Order(Permutation):
  """Order crossover (OX).

  The order crossover (OX) was proposed by L. Davis, “Applying adaptive
  algorithms to epistatic domains,” IJCAI, vol. 85, pp. 162–164, 1985.

  It builds offspring by choosing a subtour of a parent and preserving the
  relative order of bits of the other parent.

  Reference: https://dl.acm.org/doi/10.5555/1625135.1625164
  """

  # PMX supports two parents only.
  NUM_PARENTS = 2

  def permutate(
      self,
      multi_choice_spec: pg.geno.Choices,
      parents: List[List[int]]) -> List[List[int]]:
    del multi_choice_spec
    size = len(parents[0])
    start, end = 0, size
    while end - start == size:
      start, end = sorted(self._random.sample(list(range(0, size)), k=2))
    return self.order_crossover(parents, start, end)

  def order_crossover(
      self, parents: List[List[int]], start: int, end: int) -> List[List[int]]:
    """Cross over and remap the rest elements at given cutting points."""
    assert len(parents) == 2
    size = len(parents[0])

    children, crossovered, indices = [], [], []
    for i in range(2):
      child = [None] * size
      child[start:end] = parents[(i + 1) % 2][start:end]
      children.append(child)

      crossovered.append(set(c for c in children[i] if c is not None))
      indices.append({v: j for j, v in enumerate(parents[i])})

    positions = list(range(end, size)) + list(range(start))
    for i in range(2):
      parent_pos = end % size
      for j in positions:
        v = parents[i][parent_pos]
        while v in crossovered[i]:
          parent_pos = (parent_pos + 1) % size
          v = parents[i][parent_pos]
        children[i][j] = v
        parent_pos = (parent_pos + 1) % size
    return children


@pg.members([], init_arg_list=['where', 'seed'])
class Cycle(Permutation):
  """Cycle crossover (CX).

  The cycle crossover (CX) operator was first proposed by I. M. Oliver,
  D. J. d. Smith, and R. C. J. Holland, “Study of permutation crossover
  operators on the traveling salesman problem,” in Genetic algorithms and
  their applications: proceedings of the second International Conference
  on Genetic Algorithms: July 28-31, 1987 at the Massachusetts Institute of
  Technology, Cambridge, MA, USA, 1987.

  Reference: https://dl.acm.org/doi/10.5555/42512.42542.
  """

  # PMX supports two parents only.
  NUM_PARENTS = 2

  def permutate(
      self,
      multi_choice_spec: pg.geno.Choices,
      parents: List[List[int]]) -> List[List[int]]:
    del multi_choice_spec
    return self.cycle_crossover(parents)

  def cycle_crossover(
      self, parents: List[List[int]]) -> List[List[int]]:
    """Cycle crossover."""
    size = len(parents[0])
    children, indices, selected = [], [], []
    for i in range(2):
      children.append([None] * size)
      indices.append({v: i for i, v in enumerate(parents[i])})
      selected.append(set())

    def pick(child_id, parent_id, index):
      if children[child_id][index] is None:
        x = parents[parent_id][index]
        y = parents[(parent_id + 1) % 2][index]

        children[child_id][index] = x
        selected[child_id].add(x)

        pick(child_id, parent_id, indices[parent_id][y])
        pick((child_id + 1) % 2, (parent_id + 1) % 2, index)

    for i in range(size):
      if children[0][i] is None:
        child_id = self._random.choice([0, 1])
        pick(child_id, 0, i)
    return typing.cast(List[List[int]], children)

