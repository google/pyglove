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

import random
import re
import unittest

from pyglove.core import composing
from pyglove.core import symbolic
from pyglove.core import typing


class Layer(symbolic.Object):
  pass


@symbolic.members([
    ('layers', typing.List(typing.Object(Layer))),
])
class Sequential(Layer):
  pass


class Activation(Layer):
  pass


class ReLU(Activation):
  pass


class Swish(Activation):
  pass


@symbolic.members([
    ('filters', typing.Int(min_value=1)),
    # `kernel_size` is marked as no_mutation, which should not appear as a
    # mutation candidate.
    ('kernel_size', typing.Int(min_value=1), '', {'no_mutation': True}),
    ('activation', typing.Object(Activation).noneable())
])
class Conv(Layer):
  pass


def mutate_at_location(mutation_type: composing.MutationType, location: str):
  def _weights(mt, k, v, p):
    del v, p
    if mt == mutation_type and re.match(location, str(k)):
      return 1.0
    return 0.0
  return _weights


class MutateTest(unittest.TestCase):
  """Tests for `pg.mutate`."""

  def setUp(self):
    super().setUp()
    self._seed_program = Sequential([
        Conv(16, 3, ReLU()),
        Conv(32, 5, Swish()),
        Sequential([
            Conv(64, 7)
        ])
    ])

  def testReplace(self):
    self.assertEqual(
        composing.mutate(
            self._seed_program.clone(deep=True),
            weights=mutate_at_location(
                composing.MutationType.REPLACE, r'^layers\[1\]$'),
            replacer=lambda k, v, p: ReLU()),
        Sequential([
            Conv(16, 3, ReLU()),
            ReLU(),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def testInsertion(self):
    self.assertEqual(
        composing.mutate(
            self._seed_program.clone(deep=True),
            weights=mutate_at_location(
                composing.MutationType.INSERT, r'^layers\[1\]$'),
            replacer=lambda k, v, p: ReLU()),
        Sequential([
            Conv(16, 3, ReLU()),
            ReLU(),
            Conv(32, 5, Swish()),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def testDelete(self):
    self.assertEqual(
        composing.mutate(
            self._seed_program.clone(deep=True),
            weights=mutate_at_location(
                composing.MutationType.DELETE, r'^layers\[1\]$'),
            replacer=lambda k, v, p: ReLU()),
        Sequential([
            Conv(16, 3, ReLU()),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def testRandomGenerator(self):
    self.assertEqual(
        composing.mutate(
            self._seed_program.clone(deep=True),
            weights=mutate_at_location(
                composing.MutationType.REPLACE, r'^layers\[.*\]$'),
            replacer=lambda k, v, p: ReLU(),
            random_generator=random.Random(1)),
        Sequential([
            ReLU(),
            Conv(32, 5, Swish()),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def testMutationPointsAndWeights(self):
    points, weights = composing._mutation_points_and_weights(
        self._seed_program, lambda *x: 1.0)
    # NOTE(daiyip): Conv.kernel_size is marked with 'no_mutation', thus
    # it should not show here.
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (composing.MutationType.REPLACE, 'layers'),
        (composing.MutationType.INSERT, 'layers[0]'),
        (composing.MutationType.DELETE, 'layers[0]'),
        (composing.MutationType.REPLACE, 'layers[0]'),
        (composing.MutationType.REPLACE, 'layers[0].filters'),
        (composing.MutationType.REPLACE, 'layers[0].activation'),
        (composing.MutationType.INSERT, 'layers[1]'),
        (composing.MutationType.DELETE, 'layers[1]'),
        (composing.MutationType.REPLACE, 'layers[1]'),
        (composing.MutationType.REPLACE, 'layers[1].filters'),
        (composing.MutationType.REPLACE, 'layers[1].activation'),
        (composing.MutationType.INSERT, 'layers[2]'),
        (composing.MutationType.DELETE, 'layers[2]'),
        (composing.MutationType.REPLACE, 'layers[2]'),
        (composing.MutationType.REPLACE, 'layers[2].layers'),
        (composing.MutationType.INSERT, 'layers[2].layers[0]'),
        (composing.MutationType.DELETE, 'layers[2].layers[0]'),
        (composing.MutationType.REPLACE, 'layers[2].layers[0]'),
        (composing.MutationType.REPLACE, 'layers[2].layers[0].filters'),
        (composing.MutationType.REPLACE, 'layers[2].layers[0].activation'),
        (composing.MutationType.INSERT, 'layers[2].layers[1]'),
        (composing.MutationType.INSERT, 'layers[3]'),
    ])
    self.assertEqual(weights, [1.0] * len(points))

  def testMutationPointsAndWeightsWithHonoringListSize(self):
    # Non-typed list. There is no size limit.
    l = symbolic.List([1])
    points, _ = composing._mutation_points_and_weights(l, lambda *x: 1.0)
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (composing.MutationType.INSERT, '[0]'),
        (composing.MutationType.DELETE, '[0]'),
        (composing.MutationType.REPLACE, '[0]'),
        (composing.MutationType.INSERT, '[1]'),
    ])

    # Typed list with size limit.
    value_spec = typing.List(typing.Int(), min_size=1, max_size=3)
    l = symbolic.List([1, 2], value_spec=value_spec)
    points, _ = composing._mutation_points_and_weights(l, lambda *x: 1.0)
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (composing.MutationType.INSERT, '[0]'),
        (composing.MutationType.DELETE, '[0]'),
        (composing.MutationType.REPLACE, '[0]'),
        (composing.MutationType.INSERT, '[1]'),
        (composing.MutationType.DELETE, '[1]'),
        (composing.MutationType.REPLACE, '[1]'),
        (composing.MutationType.INSERT, '[2]'),
    ])
    l = symbolic.List([1], value_spec=value_spec)
    points, _ = composing._mutation_points_and_weights(l, lambda *x: 1.0)
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (composing.MutationType.INSERT, '[0]'),
        (composing.MutationType.REPLACE, '[0]'),
        (composing.MutationType.INSERT, '[1]'),
    ])
    l = symbolic.List([1, 2, 3], value_spec=value_spec)
    points, _ = composing._mutation_points_and_weights(l, lambda *x: 1.0)
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (composing.MutationType.DELETE, '[0]'),
        (composing.MutationType.REPLACE, '[0]'),
        (composing.MutationType.DELETE, '[1]'),
        (composing.MutationType.REPLACE, '[1]'),
        (composing.MutationType.DELETE, '[2]'),
        (composing.MutationType.REPLACE, '[2]'),
    ])


if __name__ == '__main__':
  unittest.main()
