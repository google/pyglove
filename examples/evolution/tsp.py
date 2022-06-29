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
"""Solving the travelling salesman problem (TSP) with evolution.

Reference: https://en.wikipedia.org/wiki/Travelling_salesman_problem

A more detailed tutorial explaning how to use PyGlove to solve TSP can be
found here:
https://colab.research.google.com/github/google/pyglove/blob/main/docs/notebooks/evolution/tsp.ipynb
"""

import math
from typing import List
import pyglove as pg


@pg.symbolize
class City:
  """Represents a city with location (x, y) on the map."""

  def __init__(self, x: int, y: int):
    self.x = x
    self.y = y

  def distance(self, other: 'City') -> float:
    return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@pg.symbolize
class Route:
  """Represents a route that traverse the cities in their appearing order."""

  def __init__(self, cities: List[City]):
    self.cities = cities

  def length(self) -> float:
    l = 0
    for i in range(0, len(self.cities)):
      l += self.cities[i].distance(self.cities[(i + 1) % len(self.cities)])
    return l


def tsp(cities: List[City], num_trials: int = 500) -> Route:
  """Returns the best route found."""
  best_route, min_len = None, None

  # The route space is a Route object
  # with all possible permutations generated
  # from given cities.
  route_space = Route(pg.permutate(cities))

  def evolution(op, population_size=50, tournament_size=20, seed=None):
    return pg.evolution.Evolution(
        (pg.evolution.selectors.Random(tournament_size, seed=seed)
         >> pg.evolution.selectors.Top(2)
         >> op),
        population_init=(pg.geno.Random(seed=seed), population_size),
        population_update=pg.evolution.selectors.Last(population_size))

  search_algorithm = evolution(
      pg.evolution.recombinators.PartiallyMapped()
      >> pg.evolution.mutators.Swap())

  # `pg.sample` is the way to sample an example
  # route from the route space. Each iteration
  # will emit a `feedback` object, which can be
  # used to pass the reward to the controller.
  for route, feedback in pg.sample(
      route_space,
      search_algorithm,
      num_examples=num_trials):
    l = route.length()
    if min_len is None or min_len > l:
      best_route, min_len = route, l
    # We negate the route length as the reward since
    # the algorithm is to maximize the reward value.
    feedback(-l)

  print(f'Best route length: {min_len}.')
  print(best_route)
  return best_route


def main():
  # Generating 25 cities.
  cities = list(pg.random_sample(
      City(x=pg.oneof(range(100)), y=pg.oneof(range(100))), 25, seed=1))
  tsp(cities)


if __name__ == '__main__':
  main()
