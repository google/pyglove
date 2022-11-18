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
"""Tests for pyglove.geno.Choices."""

import inspect
import random
import unittest

from pyglove.core import symbolic
from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import Choices
from pyglove.core.geno.categorical import manyof
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.custom import custom
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.space import constant


class ChoicesTest(unittest.TestCase):
  """Tests for `pg.geno.Choices`."""

  def test_init(self):
    _ = oneof([constant(), constant()])
    _ = oneof([
        constant(),
        oneof([constant(), constant()]),
        manyof(2, [constant(), constant()]),
        floatv(-1.0, 1.0),
        custom()
    ])

    with self.assertRaisesRegex(
        ValueError,
        'There are not enough candidates .* to make .* distinct choices.'):
      _ = manyof(2, [constant()])

    with self.assertRaisesRegex(
        ValueError,
        'The length of \'candidates\' .* should be equal to the length of '
        '\'literal_values\''):
      _ = oneof([constant(), constant()], literal_values=[1])

    with self.assertRaisesRegex(
        ValueError, 'Multi-choice spec cannot be a subchoice.'):
      _ = Choices(2, candidates=[constant(), constant()], subchoice_index=0)

  def test_basics(self):
    choice = oneof([constant(), constant()],
                   location='a.b', literal_values=[1, 'foo'])
    self.assertEqual(choice.location.keys, ['a', 'b'])
    self.assertEqual(choice.literal_values, [1, 'foo'])
    self.assertEqual(choice.num_choices, 1)
    self.assertIsNone(choice.hints)
    self.assertIsNone(choice.name)
    self.assertIs(choice.subchoice(0), choice)

    self.assertFalse(choice.is_space)
    self.assertTrue(choice.is_categorical)
    self.assertFalse(choice.is_subchoice)
    self.assertFalse(choice.is_numerical)
    self.assertFalse(choice.is_custom_decision_point)

    choice = manyof(2, [constant(), constant()], name='x', hints='abc')

    self.assertIsNone(choice.literal_values)
    self.assertEqual(choice.num_choices, 2)
    self.assertEqual(choice.hints, 'abc')

    self.assertFalse(choice.is_space)
    self.assertTrue(choice.is_categorical)
    self.assertFalse(choice.is_subchoice)
    self.assertFalse(choice.is_numerical)
    self.assertFalse(choice.is_custom_decision_point)

  def test_subchoice(self):
    choice = oneof([constant(), constant()])
    self.assertIs(choice.subchoice(0), choice)
    self.assertFalse(choice.is_subchoice)

    choice = manyof(2, [constant(), constant()])
    self.assertFalse(choice.is_subchoice)

    subchoice = choice.subchoice(0)
    self.assertEqual(subchoice.num_choices, 1)
    self.assertTrue(symbolic.eq(subchoice.candidates, choice.candidates))
    self.assertTrue(
        symbolic.eq(subchoice.literal_values, choice.literal_values))
    self.assertTrue(subchoice.is_subchoice)
    self.assertIs(subchoice.parent_spec, choice)

    with self.assertRaisesRegex(
        ValueError, '\'subchoice\' should not be called on a subchoice'):
      subchoice.subchoice(0)

  def test_len(self):
    self.assertEqual(len(oneof([constant(), constant()])), 1)
    self.assertEqual(len(manyof(2, [constant(), constant()])), 2)
    self.assertEqual(
        len(manyof(2, [oneof([constant(), constant()]),
                       oneof([constant(), constant()])])),
        6)  # r0, r1, r0.c0, r0.c1, r1.c0, r1.c1.

  def test_id(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], location='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')

    self.assertEqual(a.id, 'a')
    self.assertEqual(x.id, 'a[=0/3]')
    self.assertEqual(y.id, 'a[=1/3].y')
    self.assertEqual(z.id, 'a[=2/3]')

  def test_get(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], location='y', name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')

    # Get by decision index.
    self.assertIs(x.get(0), x)
    self.assertIs(y.get(0), y.subchoice(0))
    self.assertIs(y.get(1), y.subchoice(1))
    self.assertIs(a.get(0), a)

    # Get by ID.
    self.assertIs(a.get('a'), a)
    self.assertIs(a.get('a[=0/3]'), x)
    self.assertEqual(a.get('a[=1/3].y'), [y.subchoice(0), y.subchoice(1)])
    self.assertIs(a.get('a[=2/3]'), z)

    # Get by name.
    self.assertIsNone(a.get('x'))  # `x` does not specify a name.
    self.assertEqual(a.get('y'), [y.subchoice(0), y.subchoice(1)])

  def test_getitem(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], location='y', name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')

    # Get by index and slice.
    self.assertIs(a[0], a)
    self.assertEqual(y[:], [y.subchoice(0), y.subchoice(1)])

    # Get by ID.
    self.assertIs(a['a'], a)
    self.assertIs(a['a[=0/3]'], x)
    self.assertEqual(a['a[=1/3].y'], [y.subchoice(0), y.subchoice(1)])
    self.assertIs(a['a[=2/3]'], z)

    # Get by name.
    self.assertEqual(a['y'], [y.subchoice(0), y.subchoice(1)])
    with self.assertRaises(KeyError):
      _ = a['x']    # `x` does not specify a name.

  def test_decision_ids(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], location='y', name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    self.assertEqual(
        a.decision_ids,
        [
            'a',
            'a[=0/3]',
            'a[=1/3].y',
            'a[=2/3]',
        ])

  def test_decision_points(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    self.assertEqual(
        a.decision_points,
        [
            a, x, y.subchoice(0), y.subchoice(1), z,
        ])

  def test_named_decision_points(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    self.assertEqual(
        a.named_decision_points,
        {'y': [y.subchoice(0), y.subchoice(1)]})

  def test_set_get_userdata(self):
    s = oneof([constant(), constant()])
    s.set_userdata('mydata', 1)
    self.assertEqual(s.userdata.mydata, 1)
    self.assertEqual(s.userdata['mydata'], 1)
    with self.assertRaises(KeyError):
      _ = s.userdata.nonexist_data
    with self.assertRaises(KeyError):
      _ = s.userdata['nonexist_data']

  def test_parent_spec(self):
    a = oneof([
        oneof([              # x
            constant(),
            constant()
        ]),
        manyof(2, [          # y
            constant(),
            constant(),
            constant()
        ]),
        custom(),            # z
    ])
    x = a.candidates[0].elements[0]
    y = a.candidates[1].elements[0]
    z = a.candidates[2].elements[0]

    self.assertIsNone(a.parent_spec)
    self.assertIs(a.candidates[0].parent_spec, a)
    self.assertIs(x.parent_spec, a.candidates[0])
    self.assertIs(y.parent_spec, a.candidates[1])
    self.assertIs(z.parent_spec, a.candidates[2])

  def test_parent_choice(self):
    a = oneof([
        oneof([              # x
            constant(),
            constant()
        ]),
        manyof(2, [          # y
            constant(),
            constant(),
            constant()
        ]),
        custom(),            # z
    ])
    x = a.candidates[0].elements[0]
    y = a.candidates[1].elements[0]
    z = a.candidates[2].elements[0]

    self.assertIsNone(a.parent_choice)
    self.assertIsNone(a.subchoice(0).parent_choice)
    self.assertIs(x.parent_choice, a)
    self.assertIs(x.subchoice(0).parent_choice, a)
    self.assertIs(y.parent_choice, a)
    self.assertIs(y.subchoice(0).parent_choice, a)
    self.assertIs(y.subchoice(1).parent_choice, a)
    self.assertIs(z.parent_choice, a)
    self.assertIs(z.parent_choice, a)

  def test_validate(self):
    c = oneof([
        oneof([
            constant(),
            constant()
        ]),
        manyof(2, [
            constant(),
            constant(),
            constant()
        ]),
        constant(),
        custom(),
        floatv(min_value=-1.0, max_value=1.0)
    ])
    c.validate(DNA((0, 1)))
    c.validate(DNA((1, [0, 1])))
    c.validate(DNA((3, 'abc')))
    c.validate(DNA((4, 0.0)))

    with self.assertRaisesRegex(
        ValueError, 'Expect an integer for a single choice'):
      c.validate(DNA(None))

    with self.assertRaisesRegex(
        ValueError,
        'No child DNA provided for child space'):
      c.validate(DNA(0))

    with self.assertRaisesRegex(
        ValueError, 'Choice out of range'):
      c.validate(DNA((5, 10)))

    with self.assertRaisesRegex(
        ValueError,
        'Expect an integer for a single choice, but encountered a list'):
      c.validate(DNA([2, 0]))

    with self.assertRaisesRegex(
        ValueError,
        'Child DNA .* provided while there is no child decision point'):
      c.validate(DNA((2, 0)))

    with self.assertRaisesRegex(
        ValueError, 'CustomDecisionPoint expects string type DNA'):
      c.validate(DNA((3, 0.2)))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than '):
      c.validate(DNA((4, 1.5)))

  def test_first_dna(self):
    self.assertEqual(oneof([constant(), constant()]).first_dna(), DNA(0))

    c = manyof(3, [
        oneof([
            constant(),
            constant()
        ]),
        manyof(2, [
            constant(),
            constant(),
            constant()
        ]),
        floatv(min_value=0.0, max_value=1.0),
    ])
    self.assertEqual(c.first_dna(), DNA([(0, 0), (1, [0, 1]), (2, 0.0)]))

  def test_next_dna(self):
    c = oneof([
        oneof([
            constant(),
            constant()
        ]),
        manyof(2, [
            constant(),
            constant(),
            constant()
        ]),
    ])
    self.assertEqual(c.next_dna(), DNA((0, 0)))
    self.assertEqual(c.next_dna(DNA((0, 1))), DNA((1, [0, 1])))
    self.assertEqual(c.next_dna(DNA((1, [0, 2]))), DNA((1, [1, 0])))
    self.assertIsNone(c.next_dna(DNA((1, [2, 1]))))

  def test_iter_dna(self):
    c = oneof([
        oneof([
            constant(),
            constant()
        ]),
        manyof(2, [
            constant(),
            constant(),
            constant()
        ]),
    ])
    self.assertEqual(list(c.iter_dna()), [
        DNA((0, 0)),
        DNA((0, 1)),
        DNA((1, [0, 1])),
        DNA((1, [0, 2])),
        DNA((1, [1, 0])),
        DNA((1, [1, 2])),
        DNA((1, [2, 0])),
        DNA((1, [2, 1])),
    ])
    self.assertEqual(list(c.iter_dna(DNA((1, [1, 2])))), [
        DNA((1, [2, 0])),
        DNA((1, [2, 1])),
    ])
    with self.assertRaisesRegex(ValueError, 'Choice value .* is out of range'):
      next(c.iter_dna(DNA(None)))

    with self.assertRaisesRegex(ValueError, 'Choice value .* is out of range'):
      next(c.iter_dna(DNA((2, 1))))

    with self.assertRaisesRegex(ValueError, 'Choice value .* is out of range'):
      next(c.iter_dna(DNA((-1, 1))))

  def test_random_dna(self):
    c = oneof([
        oneof([
            constant(),
            constant()
        ]),
        manyof(2, [
            constant(),
            constant(),
            constant()
        ]),
    ])
    self.assertEqual(c.random_dna(random.Random(1)), DNA((0, 0)))

  def test_inspection(self):
    self.assertEqual(
        str(oneof([constant(), constant()], location='a.b')),
        inspect.cleandoc("""
            Choices(num_choices=1, [
              (0): Space()
              (1): Space()
            ], id='a.b')"""))

  def test_format_literal(self):
    c = oneof([constant(), constant()],
              literal_values=['0/2 (\'foo\nbar\')', '1/2 (1)'])
    self.assertEqual(c.literal_values, ['\'foo\nbar\'', '1'])
    self.assertEqual(c.format_candidate(0), '0/2 (\'foo\nbar\')')
    self.assertEqual(c.format_candidate(0, 'choice'), '0/2')
    self.assertEqual(c.format_candidate(0, 'literal'), '\'foo\nbar\'')

  def test_candidate_index(self):
    c = oneof([constant(), constant()],
              literal_values=['0/2 (\'foo\nbar\')', 2])
    self.assertEqual(c.candidate_index(2), 1)
    self.assertEqual(c.candidate_index('0/2'), 0)
    self.assertEqual(c.candidate_index('0/2 (\'foo\nbar\')'), 0)
    self.assertEqual(c.candidate_index('\'foo\nbar\''), 0)
    with self.assertRaisesRegex(ValueError, 'There is no candidate.*'):
      _ = c.candidate_index(0)
    with self.assertRaisesRegex(
        ValueError, 'Candidate index out of range at choice'):
      _ = c.candidate_index('2/2')
    with self.assertRaisesRegex(
        ValueError, 'Number of candidates .* does not match with DNASpec'):
      _ = c.candidate_index('1/3')
    with self.assertRaisesRegex(
        ValueError,
        'The literal value from the input .* does not match with .*'):
      _ = c.candidate_index('0/2 (No match)')


def assert_produces_dnas(self, dna_spec, expected_dna_list):
  expected_dna_list = [DNA(e) for e in expected_dna_list]
  produced_dna_list = list(dna_spec.iter_dna())
  self.assertEqual(produced_dna_list, expected_dna_list)
  self.assertEqual(dna_spec.space_size, len(expected_dna_list))


class SingleChoiceTest(unittest.TestCase):
  """Tests for single choices."""

  def test_flat_categorical(self):
    c = oneof([constant(), constant()])
    self.assertEqual(len(c), 1)
    assert_produces_dnas(self, c, [0, 1])

  def test_nested_categorical(self):
    c = oneof([
        oneof([constant(), constant()]),
        manyof(2, [constant(), constant()]),
    ])
    self.assertEqual(len(c), 1 + 1 + 2)
    assert_produces_dnas(self, c, [
        (0, 0),
        (0, 1),
        (1, [0, 1]),
        (1, [1, 0]),
    ])

  def test_nested_numerical(self):
    c = oneof([
        floatv(-1.0, 1.0),
        constant(),
    ])
    self.assertEqual(len(c), 1 + 1)
    self.assertEqual(c.space_size, -1)
    self.assertEqual(c.first_dna(), DNA((0, -1.0)))


class MultiChoicesTest(unittest.TestCase):
  """Tests for multiple choices."""

  def test_nondistinct_unsorted(self):
    c = manyof(2, [
        constant(),
        oneof([
            constant(),
            constant(),
        ]),
        oneof([
            constant(),
            constant(),
            constant(),
        ])
    ], distinct=False, sorted=False)
    self.assertEqual(len(c), 2 * (1 + 2))
    assert_produces_dnas(self, c, [
        [0, 0],
        [0, (1, 0)],
        [0, (1, 1)],
        [0, (2, 0)],
        [0, (2, 1)],
        [0, (2, 2)],
        [(1, 0), 0],
        [(1, 0), (1, 0)],
        [(1, 0), (1, 1)],
        [(1, 0), (2, 0)],
        [(1, 0), (2, 1)],
        [(1, 0), (2, 2)],
        [(1, 1), 0],
        [(1, 1), (1, 0)],
        [(1, 1), (1, 1)],
        [(1, 1), (2, 0)],
        [(1, 1), (2, 1)],
        [(1, 1), (2, 2)],
        [(2, 0), 0],
        [(2, 0), (1, 0)],
        [(2, 0), (1, 1)],
        [(2, 0), (2, 0)],
        [(2, 0), (2, 1)],
        [(2, 0), (2, 2)],
        [(2, 1), 0],
        [(2, 1), (1, 0)],
        [(2, 1), (1, 1)],
        [(2, 1), (2, 0)],
        [(2, 1), (2, 1)],
        [(2, 1), (2, 2)],
        [(2, 2), 0],
        [(2, 2), (1, 0)],
        [(2, 2), (1, 1)],
        [(2, 2), (2, 0)],
        [(2, 2), (2, 1)],
        [(2, 2), (2, 2)]
    ])

  def test_distinct_unsorted(self):
    c = manyof(2, [
        constant(),
        oneof([
            constant(),
            constant(),
        ]),
        oneof([
            constant(),
            constant(),
            constant(),
        ])
    ], distinct=True, sorted=False)
    self.assertEqual(len(c), 6)
    assert_produces_dnas(self, c, [
        [0, (1, 0)],
        [0, (1, 1)],
        [0, (2, 0)],
        [0, (2, 1)],
        [0, (2, 2)],
        [(1, 0), 0],
        [(1, 0), (2, 0)],
        [(1, 0), (2, 1)],
        [(1, 0), (2, 2)],
        [(1, 1), 0],
        [(1, 1), (2, 0)],
        [(1, 1), (2, 1)],
        [(1, 1), (2, 2)],
        [(2, 0), 0],
        [(2, 0), (1, 0)],
        [(2, 0), (1, 1)],
        [(2, 1), 0],
        [(2, 1), (1, 0)],
        [(2, 1), (1, 1)],
        [(2, 2), 0],
        [(2, 2), (1, 0)],
        [(2, 2), (1, 1)]
    ])

  def test_nondistinct_sorted(self):
    c = manyof(2, [
        constant(),
        oneof([
            constant(),
            constant(),
        ]),
        oneof([
            constant(),
            constant(),
            constant(),
        ])
    ], distinct=False, sorted=True)
    self.assertEqual(len(c), 6)
    assert_produces_dnas(self, c, [
        [0, 0],
        [0, (1, 0)],
        [0, (1, 1)],
        [0, (2, 0)],
        [0, (2, 1)],
        [0, (2, 2)],
        [(1, 0), (1, 0)],
        [(1, 0), (1, 1)],
        [(1, 0), (2, 0)],
        [(1, 0), (2, 1)],
        [(1, 0), (2, 2)],
        [(1, 1), (1, 0)],
        [(1, 1), (1, 1)],
        [(1, 1), (2, 0)],
        [(1, 1), (2, 1)],
        [(1, 1), (2, 2)],
        [(2, 0), (2, 0)],
        [(2, 0), (2, 1)],
        [(2, 0), (2, 2)],
        [(2, 1), (2, 0)],
        [(2, 1), (2, 1)],
        [(2, 1), (2, 2)],
        [(2, 2), (2, 0)],
        [(2, 2), (2, 1)],
        [(2, 2), (2, 2)]
    ])

  def test_distinct_sorted(self):
    c = manyof(2, [
        constant(),
        oneof([constant(), constant()]),
        oneof([constant(), constant(), constant()])
    ], distinct=True, sorted=True)
    self.assertEqual(len(c), 6)
    assert_produces_dnas(self, c, [
        [0, (1, 0)],
        [0, (1, 1)],
        [0, (2, 0)],
        [0, (2, 1)],
        [0, (2, 2)],
        [(1, 0), (2, 0)],
        [(1, 0), (2, 1)],
        [(1, 0), (2, 2)],
        [(1, 1), (2, 0)],
        [(1, 1), (2, 1)],
        [(1, 1), (2, 2)]
    ])


if __name__ == '__main__':
  unittest.main()
