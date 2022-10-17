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
"""Interfaces and utilities for decision point selection."""

import abc
import random
import types
from typing import List, Optional

import pyglove.core as pg
from pyglove.ext import scalars


class DecisionPointFilter(pg.Object):
  """Base class for decision point filter.

  A decision pointer filter is used for deciding which decision points should
  be included during recombination and mutation. It takes a list of decision
  points as the candidates for filtering, and returns a list of decision points
  as the target that evolutionary operations should work on.
  """

  def _on_bound(self):
    super()._on_bound()
    self._call = pg.typing.CallableWithOptionalKeywordArgs(
        self.call, ['global_state', 'step'])

  def __call__(
      self,
      decision_points: List[pg.geno.DecisionPoint],
      global_state: Optional[pg.geno.AttributeDict] = None,
      step: int = 0) -> List[pg.geno.DecisionPoint]:
    """Filtering decision points based on global state and current step.

    Args:
      decision_points: A list of decision points as candidates for filtering.
      global_state: An optional keyword argument as the global state.
      step: An optional keyword argument as current step of evolution.

    Returns:
      A list of decision points that should be kept.
    """
    return self._call(decision_points, global_state=global_state, step=step)

  @abc.abstractmethod
  def call(
      self,
      decision_points: List[pg.geno.DecisionPoint],
      global_state: Optional[pg.geno.AttributeDict] = None,
      step: int = 0) -> List[pg.geno.DecisionPoint]:
    """Implementation of filtering logic. Subclass to override.

    Args:
      decision_points: A list of decision points as candidates for filtering.
      global_state: An optional keyword argument as the global state.
      step: An optional keyword argument as current step of evolution.

    Returns:
      A list of decision points that should be kept.
    """


@pg.members([
    ('fn', pg.typing.Callable(
        [pg.typing.List(pg.typing.Object(pg.geno.DecisionPoint))],
        returns=pg.typing.List(pg.typing.Object(pg.geno.DecisionPoint))))
])
class Lambda(DecisionPointFilter):
  """Decision point filter that is converted from a callable object."""

  def _on_bound(self):
    super()._on_bound()
    self._fn = pg.typing.CallableWithOptionalKeywordArgs(
        self.fn, ['global_state', 'step'])

  def call(
      self,
      decision_points: List[pg.geno.DecisionPoint],
      global_state: Optional[pg.geno.AttributeDict] = None,
      step: int = 0) -> List[pg.geno.DecisionPoint]:
    return self._fn(decision_points, global_state=global_state, step=step)


# Register typing conversion from function to DecisionPointFilter
# so users can use lambda functions for where statement.
pg.typing.register_converter(types.FunctionType, DecisionPointFilter, Lambda)


class All(DecisionPointFilter):
  """Include all decision points."""

  def call(  # pytype: disable=signature-mismatch
      self,
      decision_points: List[pg.geno.DecisionPoint]
      ) -> List[pg.geno.DecisionPoint]:
    return decision_points


ALL = All()


@pg.members([
    ('k', scalars.scalar_spec(pg.typing.Int(min_value=0)).set_default(1),
     'Number of decision points to return.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed.')
])
class Any(DecisionPointFilter):
  """Include any K (1 by default) decision point(s)."""

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)

  def call(  # pytype: disable=signature-mismatch
      self,
      decision_points: List[pg.geno.DecisionPoint],
      step: int
      ) -> List[pg.geno.DecisionPoint]:
    k = scalars.scalar_value(self.k, step)
    if k >= len(decision_points):
      return decision_points
    else:
      indices = sorted(
          self._random.sample(list(range(len(decision_points))), k=k))
      return [decision_points[i] for i in indices]


ANY = Any()


def where_spec(default=ALL) -> pg.typing.ValueSpec:
  """Returns the value spec for a decision point filter."""
  return pg.typing.Object(DecisionPointFilter, default=default)
