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
"""Tests for hill climbing algorithm."""

import math
import threading
import time
import unittest
import pyglove.core as pg
from pyglove.ext.evolution import base


@pg.members([
    ('n', pg.typing.Int()),
])
class LastN(base.Selector):
  """A last N selector."""

  def select(self, population, step):
    del step
    if len(population) > self.n:
      return population[len(population) - self.n:]
    return population


class FirstAndLast(base.Selector):
  """A selector that selects the first and the last individual in population."""

  def select(self, population, step):
    del step
    return [population[0], population[-1]]


class Average(base.Recombinator):
  """A recombinator to produce child values by averaging parents values."""

  def recombine(self, parents, step):
    dna_data = []
    for v in zip(*[p.to_numbers() for p in parents]):
      nv = int(sum(v) / len(v))
      dna_data.append(nv)
    return [pg.DNA(dna_data).use_spec(parents[0].spec)]


class NextValue(base.Mutator):
  """A mutator to increase current DNA by 1."""

  def mutate(self, dna, step):
    return dna.next_dna()


def search_space():
  """A simple search space used in the tests."""
  return pg.dna_spec(pg.Dict(
      x=pg.oneof(range(10)),
      y=pg.oneof(range(10))))


class EvolutionTest(unittest.TestCase):
  """Tests for the `Evolution` class."""

  def test_basics(self):
    init_algo = pg.geno.Random(seed=0)
    algo = base.Evolution(
        FirstAndLast() >> Average() >> NextValue(),
        population_init=(init_algo, 3),
        population_update=LastN(5))

    algo.setup(search_space())
    self.assertEqual(len(algo.population), 0)
    self.assertEqual(algo.num_proposals, 0)
    self.assertEqual(algo.num_feedbacks, 0)

    dna_list = []
    for _ in range(10):
      dna = algo.propose()
      algo.feedback(dna, 0.)
      dna_list.append(dna)

    self.assertEqual(len(algo.population), 5)
    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(algo.num_feedbacks, 10)
    self.assertEqual(init_algo.num_proposals, 3)
    self.assertEqual(init_algo.num_feedbacks, 3)
    self.assertEqual(dna_list, [
        # Initial population.
        pg.DNA([6, 6]),
        pg.DNA([0, 4]),
        pg.DNA([8, 7]),
        # New population.
        pg.DNA([7, 7]),
        pg.DNA([6, 7]),
        pg.DNA([6, 7]),
        pg.DNA([3, 6]),
        pg.DNA([5, 7]),
        pg.DNA([6, 8]),
        pg.DNA([6, 8]),
    ])
    self.assertEqual(
        [base.get_feedback_sequence_number(d) for d in algo.population],
        [6, 7, 8, 9, 10])
    self.assertEqual(
        [base.is_initial_population(d) for d in dna_list],
        [True] * 3 + [False] * 7)

  def test_population_init_without_population_size(self):
    @pg.geno.dna_generator
    def init_population(dna_spec):
      for i in range(10):
        yield pg.DNA([i, i]).use_spec(dna_spec)

    init_algo = init_population()  # pylint: disable=no-value-for-parameter
    algo = base.Evolution(
        FirstAndLast() >> Average() >> NextValue(),
        population_init=init_algo)

    algo.setup(search_space())
    self.assertEqual(len(algo.population), 0)
    self.assertEqual(algo.num_proposals, 0)
    self.assertEqual(algo.num_feedbacks, 0)

    dna_list = []
    for _ in range(12):
      dna = algo.propose()
      algo.feedback(dna, 0.)
      dna_list.append(dna)

    self.assertEqual(len(algo.population), 12)
    self.assertEqual(algo.num_proposals, 12)
    self.assertEqual(algo.num_feedbacks, 12)
    self.assertEqual(init_algo.num_proposals, 10)
    self.assertEqual(init_algo.num_feedbacks, 10)
    self.assertEqual(dna_list, [
        # Initial population.
        pg.DNA([0, 0]),
        pg.DNA([1, 1]),
        pg.DNA([2, 2]),
        pg.DNA([3, 3]),
        pg.DNA([4, 4]),
        pg.DNA([5, 5]),
        pg.DNA([6, 6]),
        pg.DNA([7, 7]),
        pg.DNA([8, 8]),
        pg.DNA([9, 9]),
        # New population.
        pg.DNA([4, 5]),
        pg.DNA([2, 3]),
    ])
    self.assertEqual(
        [base.is_initial_population(d) for d in dna_list],
        [True] * 10 + [False] * 2)

  def test_reproduction_with_multiple_dnas(self):
    # We produce multiple children by not having a recombinator, so the
    # parents will be used for mutation.
    algo = base.Evolution(
        FirstAndLast() >> NextValue(),
        population_init=(pg.geno.Sweeping(), 3),
        population_update=LastN(5))

    ssd = search_space()
    algo.setup(ssd)
    dna_list = []
    for _ in range(10):
      dna = algo.propose()
      algo.feedback(dna, 0.)
      dna_list.append(dna)

    self.assertEqual(algo.num_generations, math.ceil(1 + (10 - 3) / 2))
    # Make sure trial ID is incremented correctly.
    self.assertEqual(
        [base.get_proposal_id(dna) for dna in dna_list],
        list(range(1, 11)))
    self.assertEqual(dna_list, [
        # Initial population.
        pg.DNA([0, 0]),
        pg.DNA([0, 1]),
        pg.DNA([0, 2]),
        # Evolved population.
        pg.DNA([0, 1]),
        pg.DNA([0, 3]),
        pg.DNA([0, 1]),
        pg.DNA([0, 4]),
        pg.DNA([0, 3]),
        pg.DNA([0, 5]),
        pg.DNA([0, 4]),
    ])
    for i, dna in enumerate(dna_list):
      self.assertEqual(dna.metadata, {
          'proposal_id': i + 1,
          'generation_id': max(0, int(1 + (i - 3) / 2)) + 1,
          'feedback_sequence_number': i + 1,
          'initial_population': i < 3,
          'reward': 0.
      })
    self.assertEqual(algo.global_state, {
        'num_generations': 5
    })

  def test_reproduction_with_the_same_dna(self):
    init_algo = pg.geno.Random(seed=0)
    algo = base.Evolution(
        LastN(1),
        population_init=(init_algo, 1))

    algo.setup(search_space())
    first_dna = algo.propose()
    algo.feedback(first_dna, 0.)
    dna_id_set = set([id(first_dna)])

    for _ in range(5):
      dna = algo.propose()
      self.assertEqual(dna, first_dna)
      self.assertNotIn(id(dna), dna_id_set)
      dna_id_set.add(id(dna))
      algo.feedback(dna, 0.)

  def test_batched_proposal(self):
    init_algo = pg.geno.Random(seed=0)
    algo = base.Evolution(
        (FirstAndLast() >> Average() >> NextValue()) * 4,
        population_init=(init_algo, 3),
        population_update=LastN(5))

    algo.setup(search_space())
    self.assertEqual(len(algo.population), 0)
    self.assertEqual(algo.num_proposals, 0)
    self.assertEqual(algo.num_feedbacks, 0)

    dna_list = []
    for _ in range(10):
      dna = algo.propose()
      algo.feedback(dna, 0.)
      dna_list.append(dna)

    self.assertEqual(len(algo.population), 5)
    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(algo.num_feedbacks, 10)
    # The initial proposal is batched before any feedback.
    self.assertEqual(init_algo.num_proposals, 3)
    self.assertEqual(init_algo.num_feedbacks, 3)
    self.assertEqual(dna_list, [
        # Initial population.
        pg.DNA([6, 6]),
        pg.DNA([0, 4]),
        pg.DNA([8, 7]),
        # New population.
        pg.DNA([7, 7]),
        pg.DNA([7, 7]),
        pg.DNA([7, 7]),
        pg.DNA([7, 7]),
        pg.DNA([7, 8]),
        pg.DNA([7, 8]),
        pg.DNA([7, 8]),
    ])

  def test_recovery(self):
    algo1 = base.Evolution(
        FirstAndLast() >> Average() >> NextValue(),
        population_init=(pg.geno.Sweeping(), 3),
        population_update=LastN(5))

    ssd = search_space()
    algo1.setup(ssd)

    history = []
    for _ in range(10):
      dna = algo1.propose()
      reward = 0.
      algo1.feedback(dna, 0.)
      history.append((dna, reward))

    history.extend([(algo1.propose(), None) for _ in range(3)])
    algo2 = algo1.clone(deep=True)
    algo2.setup(ssd)
    self.assertEqual(algo2.num_proposals, 0)
    algo2.recover(history)
    self.assertEqual(algo1.population, algo2.population)
    self.assertEqual(algo1.num_proposals, algo2.num_proposals)
    self.assertEqual(algo1.num_feedbacks, algo2.num_feedbacks)
    self.assertEqual(algo1.num_generations, algo2.num_generations)
    self.assertTrue(algo2._population_initialized)
    self.assertEqual(
        [base.get_proposal_id(d) for d in algo1.population],
        [base.get_proposal_id(d) for d in algo2.population])
    self.assertEqual(
        [base.get_feedback_sequence_number(d) for d in algo1.population],
        [base.get_feedback_sequence_number(d) for d in algo2.population])

    # Test cover with a random initializer.
    algo1 = base.Evolution(
        NextValue(),
        population_init=(pg.geno.Random(seed=1), 50))
    ssd = search_space()
    algo1.setup(ssd)

    history = []
    for _ in range(10):
      dna = algo1.propose()
      reward = 0.
      algo1.feedback(dna, reward)
      history.append((dna, reward))
    algo2 = algo1.clone(deep=True)
    algo2.setup(ssd)
    algo2.recover(history)
    self.assertEqual(algo1.propose(), algo2.propose())

  def test_bad_population_initializer(self):
    @pg.geno.dna_generator
    def bad_init(unused_spec):
      if True:  # pylint: disable=using-constant-test
        raise ValueError('bad initializer')
      yield pg.DNA(0)

    algo = base.Evolution(
        LastN(1),
        population_init=bad_init.partial())

    algo.setup(search_space())

    with self.assertRaisesRegex(ValueError, 'bad initializer'):
      algo.propose()

    with self.assertRaisesRegex(
        ValueError, 'Error happened earlier: bad initializer'):
      algo.propose()

  def test_thread_safety(self):
    def run_in_parallel(num_workers, evaluation_time):
      algo = base.Evolution(
          FirstAndLast() >> Average() >> NextValue(),
          population_init=(pg.geno.Random(), 1),
          population_update=LastN(5))

      ssd = search_space()
      algo.setup(ssd)
      lock = threading.Lock()
      errors = []

      def thread_fun():
        for _ in range(20):
          try:
            dna = algo.propose()
            time.sleep(evaluation_time)
            algo.feedback(dna, 1)
          except Exception as e:  # pylint: disable=broad-except
            with lock:
              errors.append(e)
            break

      threads = [
          threading.Thread(target=thread_fun)
          for _ in range(num_workers)
      ]
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      return errors

    # Should run fine.
    self.assertEqual(run_in_parallel(100, 0.01), [])

  def test_empty_children(self):
    @pg.geno.dna_generator
    def small_population(dna_spec):
      yield dna_spec.first_dna()

    algo = base.Evolution(
        LastN(1) >> NextValue(),
        population_init=small_population.partial(),
        population_update=LastN(2))

    algo.setup(search_space())

    self.assertIsNotNone(algo.propose())
    with self.assertRaisesRegex(
        ValueError, 'There is no child reproduced'):
      _ = algo.propose()


class OperationInterfaceTest(unittest.TestCase):
  """Tests for Operation interface."""

  def test_make_operation_comptabile(self):
    self.assertIsNone(base.make_operation_compatible(None))

    op = LastN(1)
    self.assertIs(base.make_operation_compatible(op), op)

    op = base.make_operation_compatible(lambda x: x[1:])
    self.assertEqual(op([0, 1]), [1])
    self.assertEqual(
        op([0, 1], global_state=pg.geno.AttributeDict()), [1])
    self.assertEqual(op([0, 1], step=1), [1])
    self.assertEqual(
        op([0, 1], global_state=pg.geno.AttributeDict(), step=1), [1])

    op = base.make_operation_compatible(lambda x, step: x[step:])
    self.assertEqual(op([0, 1], step=0), [0, 1])
    self.assertEqual(op([0, 1], step=1), [1])
    self.assertEqual(
        op([0, 1], global_state=pg.geno.AttributeDict(), step=1), [1])

    op = base.make_operation_compatible(
        lambda x, global_state: [x[0] + global_state.get('k', 0)])
    self.assertEqual(
        op([0, 1], global_state=pg.geno.AttributeDict()), [0])
    global_state = pg.geno.AttributeDict(k=2)
    self.assertEqual(op([0, 1], global_state=global_state), [2])
    self.assertEqual(op([0, 1], global_state=global_state, step=1), [2])

  def test_input_output_type_check(self):

    class Op1(base.Operation):

      @property
      def input_element_type(self):
        return int

      @property
      def output_element_type(self):
        return float

      def call(self, inputs):
        return inputs

    op = Op1()
    with self.assertRaisesRegex(
        TypeError, 'The input is expected to be a list of .*'):
      op(['a', 'b', 'c'])

    op = Op1()
    with self.assertRaisesRegex(
        TypeError, 'The output is expected to be a list of .*'):
      op([1, 2, 3])

  def test_call_with_different_signature(self):

    class Op1(base.Operation):

      def call(self, inputs):
        return inputs

    self.assertEqual(Op1()([0, 1]), [0, 1])
    self.assertEqual(Op1()([0, 1], global_state=None), [0, 1])
    self.assertEqual(Op1()([0, 1], step=1), [0, 1])
    self.assertEqual(Op1()([0, 1], global_state=None, step=1), [0, 1])

    class Op2(base.Operation):

      def call(self, inputs, global_state):
        return [i + global_state.get('a', 0) for i in inputs]

    self.assertEqual(Op2()([0, 1]), [0, 1])
    self.assertEqual(
        Op2()([0, 1], global_state=pg.geno.AttributeDict(a=1)), [1, 2])
    self.assertEqual(Op2()([0, 1], step=2), [0, 1])
    self.assertEqual(
        Op2()([0, 1], global_state=pg.geno.AttributeDict(a=1), step=2),
        [1, 2])

    class Op3(base.Operation):

      def call(self, inputs, step):
        return [i + step for i in inputs]

    self.assertEqual(Op3()([0, 1]), [0, 1])
    self.assertEqual(
        Op3()([0, 1], global_state=pg.geno.AttributeDict(a=1)), [0, 1])
    self.assertEqual(Op3()([0, 1], step=2), [2, 3])
    self.assertEqual(
        Op3()([0, 1], global_state=pg.geno.AttributeDict(a=1), step=2), [2, 3])

    class Op4(base.Operation):

      def call(self, inputs, global_state, step):
        return [i + global_state.get('a', 0) + step for i in inputs]

    self.assertEqual(Op4()([0, 1]), [0, 1])
    self.assertEqual(
        Op4()([0, 1], global_state=pg.geno.AttributeDict(a=1)), [1, 2])
    self.assertEqual(Op4()([0, 1], step=2), [2, 3])
    self.assertEqual(
        Op4()([0, 1], global_state=pg.geno.AttributeDict(a=1), step=2),
        [3, 4])

  def test_not_implemented_call_method(self):
    with self.assertRaises(NotImplementedError):
      base.Operation()([])

  def test_selector_interface(self):

    class Selector1(base.Selector):

      def select(self, inputs):
        return inputs

    self.assertEqual(Selector1()([1, 2, 3]), [1, 2, 3])

    class Selector2(base.Selector):

      def select(self, inputs, global_state):
        return inputs[:global_state.end_pos]

    self.assertEqual(
        Selector2()(
            [1, 2, 3], global_state=pg.geno.AttributeDict(end_pos=2)),
        [1, 2])

    class Selector3(base.Selector):

      def select(self, inputs, step):
        return inputs[step:]

    self.assertEqual(Selector3()([1, 2, 3], step=1), [2, 3])

    class Selector4(base.Selector):

      def select(self, inputs, global_state, step):
        return inputs[step:global_state.end_pos]

    self.assertEqual(
        Selector4()(
            [1, 2, 3],
            global_state=pg.geno.AttributeDict(end_pos=2),
            step=1),
        [2])

  def test_recombinator_interface(self):

    class Recombinator1(base.Recombinator):

      def recombine(self, inputs):
        return [pg.DNA(sum([d.value for d in inputs]) / len(inputs))]

    self.assertEqual(
        Recombinator1()([pg.DNA(1), pg.DNA(2), pg.DNA(3)]),
        [pg.DNA(2.)])

    class Recombinator2(base.Recombinator):

      def recombine(self, inputs, global_state):
        inputs = inputs[:global_state.end_pos]
        return [pg.DNA(sum([d.value for d in inputs]) / len(inputs))]

    self.assertEqual(
        Recombinator2()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)],
            global_state=pg.geno.AttributeDict(end_pos=2)),
        [pg.DNA(1.5)])

    class Recombinator3(base.Recombinator):

      def recombine(self, inputs, step):
        inputs = inputs[step:]
        return [pg.DNA(sum([d.value for d in inputs]) / len(inputs))]

    self.assertEqual(
        Recombinator3()([pg.DNA(1), pg.DNA(2), pg.DNA(3)], step=1),
        [pg.DNA(2.5)])

    class Recombinator4(base.Recombinator):

      def recombine(self, inputs, global_state, step):
        inputs = inputs[step:global_state.end_pos]
        return [pg.DNA(sum([d.value for d in inputs]) / len(inputs))]

    self.assertEqual(
        Recombinator4()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)],
            global_state=pg.geno.AttributeDict(end_pos=2),
            step=1),
        [pg.DNA(2.)])

    class Recombinator5(base.Recombinator):

      NUM_PARENTS = 2

      def recombine(self, inputs, global_state, step):
        return inputs

    self.assertEqual(
        Recombinator5()([pg.DNA(1), pg.DNA(2)]),
        [pg.DNA(1), pg.DNA(2)])

    with self.assertRaisesRegex(
        ValueError, '.* supports recombination on exact 2 parents.'):
      Recombinator5()([pg.DNA(1)])

  def test_mutator_interface(self):
    # Test element-wise mutations.

    class Mutator1(base.Mutator):

      def mutate(self, dna):
        return pg.DNA(dna.value + 1)

    self.assertEqual(
        Mutator1()([pg.DNA(1), pg.DNA(2), pg.DNA(3)]),
        [pg.DNA(2), pg.DNA(3), pg.DNA(4)])

    class Mutator2(base.Mutator):

      def mutate(self, dna, global_state):
        return pg.DNA(dna.value + global_state.bias)

    self.assertEqual(
        Mutator2()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)],
            global_state=pg.geno.AttributeDict(bias=-2)),
        [pg.DNA(-1), pg.DNA(0), pg.DNA(1)])

    class Mutator3(base.Mutator):

      def mutate(self, dna, step):
        return pg.DNA(dna.value + step)

    self.assertEqual(
        Mutator3()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)], step=2),
        [pg.DNA(3), pg.DNA(4), pg.DNA(5)])

    class Mutator4(base.Mutator):

      def mutate(self, dna, global_state, step):
        return pg.DNA(dna.value + step + global_state.bias)

    self.assertEqual(
        Mutator4()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)],
            global_state=pg.geno.AttributeDict(bias=2), step=1),
        [pg.DNA(4), pg.DNA(5), pg.DNA(6)])

    # Mutator that returns a list of DNA for each mutation.
    class Mutator5(base.Mutator):

      def mutate(self, dna):
        return [pg.DNA(dna.value), pg.DNA(dna.value * 2)]

    self.assertEqual(
        Mutator5()([pg.DNA(1), pg.DNA(2)]),
        [pg.DNA(1), pg.DNA(2), pg.DNA(2), pg.DNA(4)])

    # Test for mutate_list interface.
    class Mutator6(base.Mutator):

      def mutate_list(self, dna_list):
        return [pg.DNA(dna.value + 1) for dna in dna_list]

    self.assertEqual(
        Mutator6()([pg.DNA(1), pg.DNA(2), pg.DNA(3)]),
        [pg.DNA(2), pg.DNA(3), pg.DNA(4)])

    class Mutator7(base.Mutator):

      def mutate_list(self, dna_list, global_state):
        return [pg.DNA(dna.value + global_state.bias) for dna in dna_list]

    self.assertEqual(
        Mutator7()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)],
            global_state=pg.geno.AttributeDict(bias=-2)),
        [pg.DNA(-1), pg.DNA(0), pg.DNA(1)])

    class Mutator8(base.Mutator):

      def mutate_list(self, dna_list, step):
        return [pg.DNA(dna.value + step) for dna in dna_list]

    self.assertEqual(
        Mutator8()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)], step=2),
        [pg.DNA(3), pg.DNA(4), pg.DNA(5)])

    class Mutator9(base.Mutator):

      def mutate_list(self, dna_list, global_state, step):
        return [pg.DNA(d.value + step + global_state.bias) for d in dna_list]

    self.assertEqual(
        Mutator9()(
            [pg.DNA(1), pg.DNA(2), pg.DNA(3)],
            global_state=pg.geno.AttributeDict(bias=2), step=1),
        [pg.DNA(4), pg.DNA(5), pg.DNA(6)])

    # Test instantiate the base class with unimplemented `mutate` method.
    with self.assertRaises(NotImplementedError):
      base.Mutator()([pg.DNA(1)])


class CompositionalOperationsTest(unittest.TestCase):
  """Tests for compositional operations."""

  def test_identity(self):
    inputs = [pg.DNA(i) for i in range(5)]
    op = base.Identity()
    self.assertEqual(op(inputs), inputs)

  def test_lambda(self):
    inputs = [pg.DNA(i) for i in range(5)]
    op = base.Lambda(lambda x: x[::2])
    self.assertEqual(op(inputs),
                     [pg.DNA(0), pg.DNA(2), pg.DNA(4)])

  def test_choice(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = base.Choice([
        (FirstAndLast(), 0.0),
        (LastN(4), 1.0),
        (FirstAndLast(), 1.0),
    ])
    self.assertEqual(
        op(inputs),
        [pg.DNA(6), pg.DNA(9)])

    # Case when probability is a function of step.
    op = base.Choice([
        (FirstAndLast(), lambda t: 0.0 if t == 0 else 1.0)
    ])
    self.assertEqual(op(inputs, step=0), inputs)
    self.assertEqual(op(inputs, step=1), [pg.DNA(0), pg.DNA(9)])

    # Test `Operation.with_prob`.
    op = FirstAndLast().with_prob(0.9, seed=1)
    self.assertEqual(
        op(inputs), [pg.DNA(0), pg.DNA(9)])

    op = FirstAndLast().with_prob(0.01, seed=1)
    self.assertIs(op(inputs), inputs)

  def test_conditional(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = base.Conditional(
        lambda x, step: len(x) > step,
        FirstAndLast(),
        LastN(2))
    self.assertEqual(op(inputs, step=5), [pg.DNA(0), pg.DNA(9)])
    self.assertEqual(op(inputs, step=10), [pg.DNA(8), pg.DNA(9)])

    # Test `if_true`.
    op = FirstAndLast().if_true(lambda x, step: len(x) > step)
    self.assertEqual(op(inputs, step=5), [pg.DNA(0), pg.DNA(9)])
    self.assertIs(op(inputs, step=10), inputs)

    # Test `if_false`.
    op = FirstAndLast().if_false(lambda x, step: len(x) > step)
    self.assertIs(op(inputs, step=5), inputs)
    self.assertEqual(op(inputs, step=10), [pg.DNA(0), pg.DNA(9)])

  def test_element_wise(self):
    inputs = [[1, 2], [[1], [2]], [[1, 2], [3, 4]]]
    op = base.Identity().for_each(LastN(1))
    self.assertEqual(op(inputs), [[2], [[2]], [[3, 4]]])

  def test_flatten(self):
    inputs = [[1], 2, [[3], [4]]]

    op = base.Flatten()
    self.assertEqual(op(inputs), [1, 2, 3, 4])

    op = base.Flatten(max_level=1)
    self.assertEqual(op(inputs), [1, 2, [3], [4]])

    op = base.Identity().flatten()
    self.assertEqual(op(inputs), [1, 2, 3, 4])

  def test_global_state_getter(self):
    op = base.GlobalStateGetter('foo')
    self.assertEqual(
        op([], global_state=pg.geno.AttributeDict(foo=[1, 2, 3])),
        [1, 2, 3])

    # Test `GlobalStateGetter` with default value.
    op = base.GlobalStateGetter('bar', default=[1])
    self.assertEqual(
        op([], global_state=pg.geno.AttributeDict()), [1])

    # Test `GlobalStateGetter` withut default value.
    op = base.GlobalStateGetter('bar')
    with self.assertRaises(KeyError):
      op([], global_state=pg.geno.AttributeDict())

    # Test `global_state` operator.
    op = base.Identity().global_state('foo')
    self.assertEqual(
        op([], global_state=pg.geno.AttributeDict(
            foo=[1, 2, 3])), [1, 2, 3])

  def test_global_state_setter(self):
    op = base.GlobalStateSetter('foo')
    global_state = pg.geno.AttributeDict()
    self.assertEqual(op([1, 2, 3], global_state=global_state), [])
    self.assertEqual(global_state, {
        'foo': [1, 2, 3]
    })

    # Constant value.
    op = base.GlobalStateSetter('foo', 1)
    global_state = pg.geno.AttributeDict()
    self.assertEqual(op([1, 2, 3], global_state=global_state), [])
    self.assertEqual(global_state, {
        'foo': 1
    })

    # Test `as_global_state` operator.
    op = base.ElementWise(lambda x: x + 1).as_global_state('foo')
    self.assertEqual(op([1, 2, 3], global_state=global_state), [])
    self.assertEqual(global_state, {
        'foo': [2, 3, 4]
    })

    # Test `set_global_state` operator
    op = base.Identity().set_global_state('foo', 1)
    self.assertEqual(op([1, 2, 3], global_state=global_state), [1, 2, 3])
    self.assertEqual(global_state, {
        'foo': 1
    })

  def test_until_change(self):
    inputs = [pg.DNA(0, spec=pg.dna_spec(pg.oneof(range(10))))]
    op = base.UntilChange(base.Choice([(NextValue(), 0.2)]))
    self.assertEqual(op(inputs), [pg.DNA(1)])

    # Test 'Operation.until_change'.
    op = base.Choice([(NextValue(), 0.01)], seed=1).until_change()
    self.assertEqual(op(inputs), [pg.DNA(1)])

    op = base.Choice([(NextValue(), 0.01)], seed=1).until_change(max_attempts=1)
    self.assertEqual(op(inputs), [pg.DNA(0)])

  def test_pipeline(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = LastN(5) >> FirstAndLast()
    self.assertEqual(op(inputs), [pg.DNA(5), pg.DNA(9)])
    self.assertIs(op >> None, op)
    self.assertIs(None >> op, op)

    # Test __rshift__.
    op = LastN(5) >> (lambda x: x[:2])
    self.assertEqual(op(inputs), [pg.DNA(5), pg.DNA(6)])

    # Test __rrshift__.
    op = (lambda x: x[:2]) >> LastN(1)
    self.assertEqual(op(inputs), [pg.DNA(1)])

  def test_power(self):
    inputs = [pg.DNA(0, spec=pg.dna_spec(pg.oneof(range(10))))]
    op = NextValue() ** 3
    self.assertEqual(op(inputs), [pg.DNA(3)])

    # Test when `k` is a step-controlled value.
    op = NextValue() ** (lambda t: 1 if t == 0 else 2)
    self.assertEqual(op(inputs, step=0), [pg.DNA(1)])
    self.assertEqual(op(inputs, step=1), [pg.DNA(2)])

  def test_concatenation(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = FirstAndLast() + FirstAndLast()
    self.assertEqual(
        op(inputs), [pg.DNA(0), pg.DNA(9), pg.DNA(0), pg.DNA(9)])
    self.assertIs(op + None, op)
    self.assertIs(None + op, op)

    # Test __radd__.
    op = (lambda x: x[:2]) + FirstAndLast()
    self.assertEqual(
        op(inputs), [pg.DNA(0), pg.DNA(1), pg.DNA(0), pg.DNA(9)])

  def test_slice(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = base.Identity()[-2:]
    self.assertEqual(op(inputs), [pg.DNA(8), pg.DNA(9)])
    op = base.Identity()[-1]
    self.assertEqual(op(inputs), [pg.DNA(9)])

  def test_repeat(self):
    inputs = [pg.DNA(0, spec=pg.dna_spec(pg.oneof(range(10))))]
    op = NextValue() * 3
    self.assertEqual(op(inputs), [pg.DNA(1), pg.DNA(1), pg.DNA(1)])

    # Test when `k` is a step-controlled value.
    op = NextValue() * (lambda t: 1 if t == 0 else 2)
    self.assertEqual(op(inputs, step=0), [pg.DNA(1)])
    self.assertEqual(op(inputs, step=1), [pg.DNA(1), pg.DNA(1)])

  def test_union(self):
    inputs = [pg.DNA(i) for i in range(10)]
    # Test union with a single child.
    op = base.Union([lambda x: [x[0]] * 3])
    self.assertEqual(op(inputs), [pg.DNA(0)])
    self.assertIs(op | None, op)
    self.assertIs(None | op, op)

    # Test union with overlaps.
    op = FirstAndLast() | LastN(2)  # pylint: disable=unsupported-binary-operation
    self.assertEqual(op(inputs), [pg.DNA(0), pg.DNA(9), pg.DNA(8)])

    # Test __ror__.
    op = (lambda x: x[:1]) | LastN(2)  # pylint: disable=unsupported-binary-operation
    self.assertEqual(op(inputs), [pg.DNA(0), pg.DNA(8), pg.DNA(9)])

  def test_intersection(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = LastN(3) & FirstAndLast()
    self.assertEqual(op(inputs), [pg.DNA(9)])
    self.assertIs(op & None, op)
    self.assertIs(None & op, op)

    op = LastN(3) & (lambda x: x[:3])
    self.assertEqual(op(inputs), [])

    op = (lambda x: x[:-2]) & LastN(3)
    self.assertEqual(op(inputs), [pg.DNA(7)])

  def test_difference(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = FirstAndLast() - LastN(3)
    self.assertEqual(op(inputs), [pg.DNA(0)])
    self.assertIs(op - None, op)
    self.assertEqual(None - op, base.Inversion(op))

    op = (lambda x: x[3:-4]) - LastN(3)
    self.assertEqual(op(inputs), [pg.DNA(3), pg.DNA(4), pg.DNA(5)])

  def test_symmetric_difference(self):
    inputs = [pg.DNA(1), pg.DNA(2), pg.DNA(2), pg.DNA(0)]
    op = FirstAndLast() ^ LastN(3)
    self.assertEqual(op(inputs), [pg.DNA(1), pg.DNA(2), pg.DNA(2)])
    self.assertIs(op ^ None, op)
    self.assertIs(None ^ op, op)

    op = (lambda x: x[:3]) ^ LastN(3)
    self.assertEqual(op(inputs), [pg.DNA(1), pg.DNA(0)])

  def test_inversion(self):
    inputs = [pg.DNA(i) for i in range(5)]
    op = ~FirstAndLast()  # pylint: disable=invalid-unary-operand-type
    self.assertEqual(
        op(inputs), [pg.DNA(1), pg.DNA(2), pg.DNA(3)])
    # double negative is positive.
    self.assertEqual((-op)(inputs), FirstAndLast()(inputs))
    self.assertEqual(-op, base.Inversion(op))
    self.assertEqual(~op, base.Inversion(op))

  def test_composition(self):
    inputs = [pg.DNA(i) for i in range(10)]
    op = ~LastN(7) >> base.Choice([  # pylint: disable=invalid-unary-operand-type
        (lambda x: [pg.DNA(d.value + 1) for d in x], 1.0),
        (lambda x: [pg.DNA(d.value + 2) for d in x], 1.0),
        (lambda x: [pg.DNA(d.value + 3) for d in x], 1.0),
    ], limit=2) ** 3 | FirstAndLast()
    self.assertEqual(
        op(inputs),
        [
            pg.DNA(9),  # This is the inputs[0] + (1 + 2) * 3
            pg.DNA(10),
            pg.DNA(11),
            pg.DNA(0),
            pg.DNA(9)   # This is the inputs[9]
        ])

if __name__ == '__main__':
  unittest.main()
