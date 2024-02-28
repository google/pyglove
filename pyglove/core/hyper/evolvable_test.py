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
"""Tests for pyglove.hyper.Evolvable."""

import random
import re
import unittest

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper.evolvable import evolve
from pyglove.core.hyper.evolvable import MutationType


class Layer(symbolic.Object):
  pass


@symbolic.members([
    ('layers', pg_typing.List(pg_typing.Object(Layer))),
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
    ('filters', pg_typing.Int(min_value=1)),
    # `kernel_size` is marked as no_mutation, which should not appear as a
    # mutation candidate.
    ('kernel_size', pg_typing.Int(min_value=1), '', {'no_mutation': True}),
    ('activation', pg_typing.Object(Activation).noneable())
])
class Conv(Layer):
  pass


def seed_program():
  return Sequential([
      Conv(16, 3, ReLU()),
      Conv(32, 5, Swish()),
      Sequential([
          Conv(64, 7)
      ])
  ])


def mutate_at_location(mutation_type: MutationType, location: str):
  def _weights(mt, k, v, p):
    del v, p
    if mt == mutation_type and re.match(location, str(k)):
      return 1.0
    return 0.0
  return _weights


class EvolvableTest(unittest.TestCase):
  """Tests for pg.hyper.Evolvable."""

  def test_basics(self):
    v = evolve(
        seed_program(), lambda k, v, p: ReLU(),
        weights=mutate_at_location(MutationType.REPLACE, r'^layers\[.*\]$'))
    self.assertEqual(
        seed_program(), v.custom_decode(v.custom_encode(seed_program())))
    self.assertEqual(v.first_dna(), v.custom_encode(seed_program()))
    self.assertEqual(v.random_dna(), v.custom_encode(seed_program()))
    self.assertEqual(
        v.random_dna(random.Random(1), v.first_dna()),
        v.custom_encode(
            Sequential([
                ReLU(),
                Conv(32, 5, Swish()),
                Sequential([
                    Conv(64, 7)
                ])
            ])))

  def test_replace(self):
    v = evolve(
        seed_program(), lambda k, v, p: ReLU(),
        weights=mutate_at_location(MutationType.REPLACE, r'^layers\[1\]$'))
    self.assertEqual(
        v.mutate(seed_program()),
        Sequential([
            Conv(16, 3, ReLU()),
            ReLU(),
            Sequential([
                Conv(64, 7)
            ])
        ]))

    # Mutating at root.
    v = evolve(
        seed_program(), lambda k, v, p: ReLU(),
        weights=lambda mt, k, v, p: 1.0 if p is None else 0.0)
    self.assertEqual(
        v.mutate(seed_program()),
        ReLU()
    )

  def test_insertion(self):
    v = evolve(
        seed_program(), lambda k, v, p: ReLU(),
        weights=mutate_at_location(MutationType.INSERT, r'^layers\[1\]$'))
    self.assertEqual(
        v.mutate(seed_program()),
        Sequential([
            Conv(16, 3, ReLU()),
            ReLU(),
            Conv(32, 5, Swish()),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def test_delete(self):
    v = evolve(
        seed_program(), lambda k, v, p: ReLU(),
        weights=mutate_at_location(MutationType.DELETE, r'^layers\[1\]$'))
    self.assertEqual(
        v.mutate(seed_program(), random.Random(1)),
        Sequential([
            Conv(16, 3, ReLU()),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def test_random_generator(self):
    v = evolve(
        seed_program(), lambda k, v, p: ReLU(),
        weights=mutate_at_location(MutationType.REPLACE, r'^layers\[.*\]$'))
    self.assertEqual(
        v.mutate(seed_program(), random_generator=random.Random(1)),
        Sequential([
            ReLU(),
            Conv(32, 5, Swish()),
            Sequential([
                Conv(64, 7)
            ])
        ]))

  def test_mutation_points_and_weights(self):
    v = evolve(seed_program(), lambda k, v, p: v, weights=lambda *x: 1.0)
    points, weights = v.mutation_points_and_weights(seed_program())

    # NOTE(daiyip): Conv.kernel_size is marked with 'no_mutation', thus
    # it should not show here.
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (MutationType.REPLACE, ''),
        (MutationType.REPLACE, 'layers'),
        (MutationType.INSERT, 'layers[0]'),
        (MutationType.DELETE, 'layers[0]'),
        (MutationType.REPLACE, 'layers[0]'),
        (MutationType.REPLACE, 'layers[0].filters'),
        (MutationType.REPLACE, 'layers[0].activation'),
        (MutationType.INSERT, 'layers[1]'),
        (MutationType.DELETE, 'layers[1]'),
        (MutationType.REPLACE, 'layers[1]'),
        (MutationType.REPLACE, 'layers[1].filters'),
        (MutationType.REPLACE, 'layers[1].activation'),
        (MutationType.INSERT, 'layers[2]'),
        (MutationType.DELETE, 'layers[2]'),
        (MutationType.REPLACE, 'layers[2]'),
        (MutationType.REPLACE, 'layers[2].layers'),
        (MutationType.INSERT, 'layers[2].layers[0]'),
        (MutationType.DELETE, 'layers[2].layers[0]'),
        (MutationType.REPLACE, 'layers[2].layers[0]'),
        (MutationType.REPLACE, 'layers[2].layers[0].filters'),
        (MutationType.REPLACE, 'layers[2].layers[0].activation'),
        (MutationType.INSERT, 'layers[2].layers[1]'),
        (MutationType.INSERT, 'layers[3]'),
    ])
    self.assertEqual(weights, [1.0] * len(points))

  def test_mutation_points_and_weights_with_honoring_list_size(self):
    # Non-typed list. There is no size limit.
    v = evolve(
        symbolic.List([]), lambda k, v, p: v,
        weights=lambda *x: 1.0)
    points, _ = v.mutation_points_and_weights(symbolic.List([1]))
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (MutationType.REPLACE, ''),
        (MutationType.INSERT, '[0]'),
        (MutationType.DELETE, '[0]'),
        (MutationType.REPLACE, '[0]'),
        (MutationType.INSERT, '[1]'),
    ])

    # Typed list with size limit.
    value_spec = pg_typing.List(pg_typing.Int(), min_size=1, max_size=3)
    points, _ = v.mutation_points_and_weights(
        symbolic.List([1, 2], value_spec=value_spec))
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (MutationType.REPLACE, ''),
        (MutationType.INSERT, '[0]'),
        (MutationType.DELETE, '[0]'),
        (MutationType.REPLACE, '[0]'),
        (MutationType.INSERT, '[1]'),
        (MutationType.DELETE, '[1]'),
        (MutationType.REPLACE, '[1]'),
        (MutationType.INSERT, '[2]'),
    ])
    points, _ = v.mutation_points_and_weights(
        symbolic.List([1], value_spec=value_spec))
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (MutationType.REPLACE, ''),
        (MutationType.INSERT, '[0]'),
        (MutationType.REPLACE, '[0]'),
        (MutationType.INSERT, '[1]'),
    ])
    points, _ = v.mutation_points_and_weights(
        symbolic.List([1, 2, 3], value_spec=value_spec))
    self.assertEqual([(p.mutation_type, p.location) for p in points], [
        (MutationType.REPLACE, ''),
        (MutationType.DELETE, '[0]'),
        (MutationType.REPLACE, '[0]'),
        (MutationType.DELETE, '[1]'),
        (MutationType.REPLACE, '[1]'),
        (MutationType.DELETE, '[2]'),
        (MutationType.REPLACE, '[2]'),
    ])


if __name__ == '__main__':
  unittest.main()
