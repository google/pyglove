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
"""Tests for pyglove.hyper.DynamicEvaluationContext."""

import threading
import unittest

from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core.hyper.categorical import manyof
from pyglove.core.hyper.categorical import oneof
from pyglove.core.hyper.categorical import OneOf
from pyglove.core.hyper.categorical import permutate
from pyglove.core.hyper.custom import CustomHyper
from pyglove.core.hyper.dynamic_evaluation import dynamic_evaluate as pg_dynamic_evaluate
from pyglove.core.hyper.dynamic_evaluation import DynamicEvaluationContext
from pyglove.core.hyper.dynamic_evaluation import trace as pg_trace
from pyglove.core.hyper.numerical import floatv
from pyglove.core.hyper.object_template import template as pg_template


class DynamicEvaluationTest(unittest.TestCase):
  """Dynamic evaluation test."""

  def test_dynamic_evaluate(self):
    with self.assertRaisesRegex(
        ValueError, '\'evaluate_fn\' must be either None or a callable object'):
      with pg_dynamic_evaluate(1):
        pass

    with self.assertRaisesRegex(
        ValueError, '\'exit_fn\' must be a callable object'):
      with pg_dynamic_evaluate(None, exit_fn=1):
        pass

  def test_evaluated_values_during_collect(self):
    with DynamicEvaluationContext().collect():
      self.assertEqual(oneof([0, 1]), 0)
      self.assertEqual(oneof([{'x': oneof(['a', 'b'])}, 1]),
                       {'x': 'a'})
      self.assertEqual(manyof(2, [0, 1, 3]), [0, 1])
      self.assertEqual(manyof(4, [0, 1, 3], distinct=False),
                       [0, 0, 0, 0])
      self.assertEqual(permutate([0, 1, 2]), [0, 1, 2])
      self.assertEqual(floatv(0.0, 1.0), 0.0)

  def test_per_thread_collect_and_apply(self):
    def thread_fun():
      context = DynamicEvaluationContext()
      with context.collect():
        oneof(range(10))

      with context.apply([3]):
        self.assertEqual(oneof(range(10)), 3)

    threads = []
    for _ in range(10):
      thread = threading.Thread(target=thread_fun)
      threads.append(thread)
      thread.start()
    for t in threads:
      t.join()

  def test_process_wise_collect_and_apply(self):
    def thread_fun():
      _ = oneof(range(10))

    context = DynamicEvaluationContext(per_thread=False)
    with context.collect() as hyper_dict:
      threads = []
      for _ in range(10):
        thread = threading.Thread(target=thread_fun)
        threads.append(thread)
        thread.start()
      for t in threads:
        t.join()
    self.assertEqual(len(hyper_dict), 10)

  def test_search_space_defined_without_hyper_name(self):
    def fun():
      x = oneof([1, 2, 3]) + 1
      y = sum(manyof(2, [2, 4, 6, 8], name='y'))
      z = floatv(min_value=1.0, max_value=2.0)
      return x + y + z

    # Test dynamic evaluation by allowing reentry (all hyper primitives will
    # be registered twice).
    context = DynamicEvaluationContext()
    with context.collect() as hyper_dict:
      result = fun()
      result = fun()

    # 1 + 1 + 2 + 4 + 1.0
    self.assertEqual(result, 9.0)
    self.assertEqual(hyper_dict, {
        'decision_0': oneof([1, 2, 3]),
        'y': manyof(2, [2, 4, 6, 8], name='y'),
        'decision_1': floatv(min_value=1.0, max_value=2.0),
        'decision_2': oneof([1, 2, 3]),
        'decision_3': floatv(min_value=1.0, max_value=2.0),
    })

    with context.apply(geno.DNA.parse(
        [1, [0, 2], 1.5, 0, 1.8])):
      # 2 + 1 + 2 + 6 + 1.5
      self.assertEqual(fun(), 12.5)
      # 1 + 1 + 2 + 6 + 1.8
      self.assertEqual(fun(), 11.8)

  def test_search_space_defined_with_hyper_name(self):
    def fun():
      x = oneof([1, 2, 3], name='a') + 1
      y = sum(manyof(2, [2, 4, 6, 8], name='b'))
      z = floatv(min_value=1.0, max_value=2.0, name='c')
      return x + y + z

    # Test dynamic evaluation by disallowing reentry (all hyper primitives will
    # be registered only once).
    context = DynamicEvaluationContext(require_hyper_name=True)
    with context.collect() as hyper_dict:
      with self.assertRaisesRegex(
          ValueError, '\'name\' must be specified for hyper primitive'):
        oneof([1, 2, 3])
      result = fun()
      result = fun()

    # 1 + 1 + 2 + 4 + 1.0
    self.assertEqual(result, 9.0)
    self.assertEqual(hyper_dict, symbolic.Dict(
        a=oneof([1, 2, 3], name='a'),
        b=manyof(2, [2, 4, 6, 8], name='b'),
        c=floatv(min_value=1.0, max_value=2.0, name='c')))
    with context.apply(geno.DNA.parse([1, [0, 2], 1.5])):
      # We can call fun multiple times since decision will be bound to each
      # name just once.
      # 2 + 1 + 2 + 6 + 1.5
      self.assertEqual(fun(), 12.5)
      self.assertEqual(fun(), 12.5)
      self.assertEqual(fun(), 12.5)

  def test_hierarchical_search_space(self):
    def fun():
      return oneof([
          lambda: sum(manyof(2, [2, 4, 6, 8])),
          lambda: oneof([3, 7]),
          lambda: floatv(min_value=1.0, max_value=2.0),
          10]) + oneof([11, 22])

    context = DynamicEvaluationContext()
    with context.collect() as hyper_dict:
      result = fun()
    # 2 + 4 + 11
    self.assertEqual(result, 17)
    self.assertEqual(hyper_dict, {
        'decision_0': oneof([
            # NOTE(daiyip): child decisions within candidates are always in
            # form of list.
            {
                'decision_1': manyof(2, [2, 4, 6, 8]),
            },
            {
                'decision_2': oneof([3, 7])
            },
            {
                'decision_3': floatv(min_value=1.0, max_value=2.0)
            },
            10,
        ]),
        'decision_4': oneof([11, 22])
    })

    with context.apply(geno.DNA.parse([(0, [1, 3]), 0])):
      # 4 + 8 + 11
      self.assertEqual(fun(), 23)

    # Use list-form decisions.
    with context.apply([0, 1, 3, 0]):
      # 4 + 8 + 11
      self.assertEqual(fun(), 23)

    with context.apply(geno.DNA.parse([(1, 1), 1])):
      # 7 + 22
      self.assertEqual(fun(), 29)

    with context.apply(geno.DNA.parse([(2, 1.5), 0])):
      # 1.5 + 11
      self.assertEqual(fun(), 12.5)

    with context.apply(geno.DNA.parse([3, 1])):
      # 10 + 22
      self.assertEqual(fun(), 32)

    with self.assertRaisesRegex(
        ValueError, '`decisions` should be a DNA or a list of numbers.'):
      with context.apply(3):
        fun()

    with self.assertRaisesRegex(
        ValueError, 'No decision is provided for .*'):
      with context.apply(geno.DNA.parse(3)):
        fun()

    with self.assertRaisesRegex(
        ValueError, 'Expect float-type decision for .*'):
      with context.apply([2, 0, 1]):
        fun()

    with self.assertRaisesRegex(
        ValueError, 'Expect int-type decision in range .*'):
      with context.apply([5, 0.5, 0]):
        fun()

    with self.assertRaisesRegex(
        ValueError, 'Found extra decision values that are not used.*'):
      with context.apply(geno.DNA.parse([(1, 1), 1, 1])):
        fun()

  def test_hierarchical_search_space_with_hyper_name(self):
    def fun():
      return oneof([
          lambda: sum(manyof(2, [2, 4, 6, 8], name='a1')),
          lambda: oneof([3, 7], name='a2'),
          lambda: floatv(min_value=1.0, max_value=2.0, name='a3.xx'),
          10], name='a') + oneof([11, 22], name='b')

    context = DynamicEvaluationContext(require_hyper_name=True)
    with context.collect() as hyper_dict:
      result = fun()
      result = fun()

    # 2 + 4 + 11
    self.assertEqual(result, 17)
    self.assertEqual(hyper_dict, {
        'a': oneof([
            # NOTE(daiyip): child decisions within candidates are always in
            # form of list.
            {'a1': manyof(2, [2, 4, 6, 8], name='a1')},
            {'a2': oneof([3, 7], name='a2')},
            {'a3.xx': floatv(min_value=1.0, max_value=2.0, name='a3.xx')},
            10,
        ], name='a'),
        'b': oneof([11, 22], name='b')
    })

    with context.apply(geno.DNA.parse([(0, [1, 3]), 0])):
      # 4 + 8 + 11
      self.assertEqual(fun(), 23)
      self.assertEqual(fun(), 23)
      self.assertEqual(fun(), 23)

    # Use list form.
    with context.apply([0, 1, 3, 0]):
      # 4 + 8 + 11
      self.assertEqual(fun(), 23)
      self.assertEqual(fun(), 23)
      self.assertEqual(fun(), 23)

    with context.apply(geno.DNA.parse([(1, 1), 1])):
      # 7 + 22
      self.assertEqual(fun(), 29)
      self.assertEqual(fun(), 29)

    with context.apply(geno.DNA.parse([(2, 1.5), 0])):
      # 1.5 + 11
      self.assertEqual(fun(), 12.5)
      self.assertEqual(fun(), 12.5)

    with context.apply(geno.DNA.parse([3, 1])):
      # 10 + 22
      self.assertEqual(fun(), 32)
      self.assertEqual(fun(), 32)

    with self.assertRaisesRegex(
        ValueError, '`decisions` should be a DNA or a list of numbers.'):
      with context.apply(3):
        fun()

    with self.assertRaisesRegex(
        ValueError, 'DNA value type mismatch'):
      with context.apply(geno.DNA.parse(3)):
        fun()

    with self.assertRaisesRegex(
        ValueError, 'Found extra decision values that are not used'):
      with context.apply(context.dna_spec.first_dna()):
        # Do not consume any decision points from the search space.
        _ = 1

    with self.assertRaisesRegex(
        ValueError,
        'Hyper primitive .* is not defined during search space inspection'):
      with context.apply(context.dna_spec.first_dna()):
        # Do not consume any decision points from the search space.
        _ = oneof(range(5), name='uknown')

  def test_where_statement(self):
    context = DynamicEvaluationContext(
        where=lambda x: getattr(x, 'name') != 'x')
    with context.collect():
      self.assertEqual(oneof(range(10)), 0)
      self.assertIsInstance(oneof(range(5), name='x'), OneOf)

    with context.apply([1]):
      self.assertEqual(oneof(range(10)), 1)
      self.assertIsInstance(oneof(range(5), name='x'), OneOf)

  def test_trace(self):
    def fun():
      return oneof([-1, 0, 1]) * oneof([-1, 0, 3]) + 1

    self.assertEqual(
        pg_trace(fun).hyper_dict,
        {
            'decision_0': oneof([-1, 0, 1]),
            'decision_1': oneof([-1, 0, 3])
        })

  def test_dynamic_evaluation_with_custom_hyper(self):

    class IntList(CustomHyper):

      def custom_decode(self, dna):
        return [int(x) for x in dna.value.split(':')]

      def first_dna(self):
        return geno.DNA('0:1:2:3')

    def fun():
      return sum(IntList()) + oneof([0, 1]) + floatv(-1., 1.)

    context = DynamicEvaluationContext()
    with context.collect():
      fun()

    self.assertEqual(
        context.hyper_dict,
        {
            'decision_0': IntList(),
            'decision_1': oneof([0, 1]),
            'decision_2': floatv(-1., 1.)
        })
    with context.apply(geno.DNA(['1:2:3:4', 1, 0.5])):
      self.assertEqual(fun(), 1 + 2 + 3 + 4 + 1 + 0.5)

    with self.assertRaisesRegex(
        ValueError, 'Expect string-type decision for .*'):
      with context.apply(geno.DNA([0, 1, 0.5])):
        fun()

    class IntListWithoutFirstDNA(CustomHyper):

      def custom_decode(self, dna):
        return [int(x) for x in dna.value.split(':')]

    context = DynamicEvaluationContext()
    with self.assertRaisesRegex(
        NotImplementedError,
        '.* must implement method `next_dna` to be used in '
        'dynamic evaluation mode'):
      with context.collect():
        IntListWithoutFirstDNA()

  def test_dynamic_evaluation_with_external_dna_spec(self):
    def fun():
      return oneof(range(5), name='x') + oneof(range(3), name='y')

    context = pg_trace(fun, require_hyper_name=True, per_thread=True)
    self.assertFalse(context.is_external)
    self.assertIsNotNone(context.hyper_dict)

    search_space_str = symbolic.to_json_str(context.dna_spec)

    context2 = DynamicEvaluationContext(
        require_hyper_name=True, per_thread=True,
        dna_spec=symbolic.from_json_str(search_space_str))
    self.assertTrue(context2.is_external)
    self.assertIsNone(context2.hyper_dict)

    with self.assertRaisesRegex(
        ValueError,
        '`collect` cannot be called .* is using an external DNASpec'):
      with context2.collect():
        fun()

    with context2.apply(geno.DNA([1, 2])):
      self.assertEqual(fun(), 3)

  def test_search_space_partitioning_without_hyper_name(self):
    def fun():
      return sum([
          oneof([1, 2, 3], hints='ssd1'),
          oneof([4, 5], hints='ssd2'),
      ])

    context1 = DynamicEvaluationContext(where=lambda x: x.hints == 'ssd1')
    context2 = DynamicEvaluationContext(where=lambda x: x.hints == 'ssd2')
    with context1.collect():
      with context2.collect():
        self.assertEqual(fun(), 1 + 4)

    self.assertEqual(
        context1.hyper_dict, {
            'decision_0': oneof([1, 2, 3], hints='ssd1')
        })
    self.assertEqual(
        context2.hyper_dict, {
            'decision_0': oneof([4, 5], hints='ssd2')
        })
    with context1.apply(geno.DNA(2)):
      with context2.apply(geno.DNA(1)):
        self.assertEqual(fun(), 3 + 5)

  def test_search_space_partitioning_with_hyper_name(self):
    def fun():
      return sum([
          oneof([1, 2, 3], name='x', hints='ssd1'),
          oneof([4, 5], name='y', hints='ssd2'),
      ])

    context1 = DynamicEvaluationContext(where=lambda x: x.hints == 'ssd1')
    context2 = DynamicEvaluationContext(where=lambda x: x.hints == 'ssd2')
    with context1.collect():
      with context2.collect():
        self.assertEqual(fun(), 1 + 4)

    self.assertEqual(
        context1.hyper_dict, {
            'x': oneof([1, 2, 3], name='x', hints='ssd1')
        })
    self.assertEqual(
        context2.hyper_dict, {
            'y': oneof([4, 5], name='y', hints='ssd2')
        })
    with context1.apply(geno.DNA(2)):
      with context2.apply(geno.DNA(1)):
        self.assertEqual(fun(), 3 + 5)

  def test_hierarchial_search_space_with_partitioning(self):
    def fun():
      return sum([
          oneof([
              lambda: oneof([1, 2, 3], name='y', hints='ssd1'),
              lambda: oneof([4, 5, 6], name='z', hints='ssd1'),
          ], name='x', hints='ssd1'),
          oneof([7, 8], name='p', hints='ssd2'),
          oneof([9, 10], name='q', hints='ssd2'),
      ])
    context1 = DynamicEvaluationContext(where=lambda x: x.hints == 'ssd1')
    context2 = DynamicEvaluationContext(where=lambda x: x.hints == 'ssd2')
    with context1.collect():
      with context2.collect():
        self.assertEqual(fun(), 1 + 7 + 9)

    self.assertEqual(
        context1.hyper_dict, {
            'x': oneof([
                {'y': oneof([1, 2, 3], name='y', hints='ssd1')},
                {'z': oneof([4, 5, 6], name='z', hints='ssd1')},
            ], name='x', hints='ssd1')
        })
    self.assertEqual(
        context2.hyper_dict, {
            'p': oneof([7, 8], name='p', hints='ssd2'),
            'q': oneof([9, 10], name='q', hints='ssd2')
        })
    with context1.apply(geno.DNA((1, 1))):
      with context2.apply(geno.DNA([0, 1])):
        self.assertEqual(fun(), 5 + 7 + 10)

  def test_search_space_partitioning_with_different_per_thread_settings(self):
    context1 = DynamicEvaluationContext(per_thread=True)
    context2 = DynamicEvaluationContext(per_thread=False)

    def fun():
      return oneof([1, 2, 3])

    with self.assertRaisesRegex(
        ValueError,
        'Nested dynamic evaluation contexts must be either .*'):
      with context1.collect():
        with context2.collect():
          fun()

  def test_manual_decision_point_registration(self):
    context = DynamicEvaluationContext()
    self.assertEqual(
        context.add_decision_point(oneof([1, 2, 3])), 1)
    self.assertEqual(
        context.add_decision_point(oneof(['a', 'b'], name='x')), 'a')
    self.assertEqual(
        context.add_decision_point(pg_template(1)), 1)

    with self.assertRaisesRegex(
        ValueError, 'Found different hyper primitives under the same name'):
      context.add_decision_point(oneof(['foo', 'bar'], name='x'))

    self.assertEqual(context.hyper_dict, {
        'decision_0': oneof([1, 2, 3]),
        'x': oneof(['a', 'b'], name='x'),
    })

    with self.assertRaisesRegex(
        ValueError, '`evaluate` needs to be called under the `apply` context'):
      context.evaluate(oneof([1, 2, 3]))

    with context.apply([1, 1]):
      self.assertEqual(context.evaluate(context.hyper_dict['decision_0']), 2)
      self.assertEqual(context.evaluate(context.hyper_dict['x']), 'b')


if __name__ == '__main__':
  unittest.main()
