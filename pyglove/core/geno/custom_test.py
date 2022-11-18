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
"""Tests for pyglove.geno.CustomDecisionPoint."""

import random
import unittest

from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.custom import custom
from pyglove.core.geno.space import constant


class CustomDecisionPointTest(unittest.TestCase):
  """Tests for `pg.geno.CustomDecisionPoint`."""

  def test_basics(self):
    x = custom(hyper_type='CustomA', hints=1, location='a.b')
    self.assertEqual(x.location.keys, ['a', 'b'])
    self.assertEqual(x.hyper_type, 'CustomA')
    self.assertEqual(x.hints, 1)
    self.assertIsNone(x.name)

    self.assertFalse(x.is_space)
    self.assertFalse(x.is_categorical)
    self.assertFalse(x.is_subchoice)
    self.assertFalse(x.is_numerical)
    self.assertTrue(x.is_custom_decision_point)

  def test_len(self):
    self.assertEqual(len(custom()), 1)

  def test_space_size(self):
    self.assertEqual(custom().space_size, -1)

  def test_id(self):
    x = custom(location='x')
    a = oneof([constant(), x], location='a')
    self.assertEqual(a.id, 'a')
    self.assertEqual(x.id, 'a[=1/2].x')

  def test_get(self):
    x = custom(location='x', name='foo')
    a = oneof([x, constant()], location='a')

    # Get by ID.
    self.assertIs(a.get('a'), a)
    self.assertIs(a.get('a[=0/2].x'), x)
    self.assertIs(x.get('a[=0/2].x'), x)

    # Get by name.
    self.assertIs(a.get('foo'), x)

  def test_getitem(self):
    x = custom(location='x', name='foo')
    _ = oneof([x, constant()], location='a')

    # Get by ID.
    self.assertIs(x['a[=0/2].x'], x)

    # Get by name.
    self.assertIs(x['foo'], x)

    with self.assertRaises(KeyError):
      _ = x['y']    # There is no decision point named `y` under x.

  def test_decision_ids(self):
    x = custom(location='x', name='foo')
    _ = oneof([x, constant()], location='a')
    self.assertEqual(
        x.decision_ids,
        [
            'a[=0/2].x',
        ])

  def test_decision_points(self):
    x = custom(location='x')
    _ = oneof([x, constant()], location='a')
    self.assertEqual(x.decision_points, [x])

  def test_named_decision_points(self):
    x = custom(location='x', name='foo')
    _ = oneof([x, constant()], location='a')
    self.assertEqual(x.named_decision_points, {'foo': x})

  def test_set_get_userdata(self):
    x = custom(location='x', name='foo')
    x.set_userdata('mydata', 1)
    self.assertEqual(x.userdata.mydata, 1)
    self.assertEqual(x.userdata['mydata'], 1)
    with self.assertRaises(KeyError):
      _ = x.userdata.nonexist_data
    with self.assertRaises(KeyError):
      _ = x.userdata['nonexist_data']

  def test_parent_spec(self):
    x = custom(location='x')
    self.assertIsNone(x.parent_spec)
    a = oneof([x, constant()], location='a')
    self.assertIs(x.parent_spec, a.candidates[0])

  def test_parent_choice(self):
    x = custom(location='x')
    self.assertIsNone(x.parent_choice)
    a = oneof([x, constant()], location='a')
    self.assertIs(x.parent_choice, a)

  def test_validate(self):
    x = custom(location='x')
    x.validate(DNA('abc'))

    with self.assertRaisesRegex(
        ValueError, 'CustomDecisionPoint expects string type DNA'):
      x.validate(DNA(None))

    with self.assertRaisesRegex(
        ValueError, 'CustomDecisionPoint expects string type DNA'):
      x.validate(DNA(1))

  def test_first_dna(self):
    x = custom('MyCustomDecisionPoint')
    with self.assertRaisesRegex(
        NotImplementedError,
        '`next_dna` is not supported on \'MyCustomDecisionPoint\'.'):
      x.first_dna()

    def next_dna(dna):
      del dna
      return DNA('0')

    x = custom(next_dna_fn=next_dna)
    self.assertEqual(x.first_dna(), DNA('0'))

  def test_next_dna(self):
    x = custom()
    with self.assertRaisesRegex(
        NotImplementedError,
        '`next_dna` is not supported on \'CustomDecisionPoint\'.'):
      x.next_dna()

    def next_dna(dna):
      if dna is None:
        return DNA('0')
      value = int(dna.value)
      if value == 10:
        return None
      return DNA(str(value + 1))

    x = custom(next_dna_fn=next_dna)
    self.assertEqual(x.next_dna(), DNA('0'))
    self.assertEqual(x.next_dna(DNA('8')), DNA('9'))
    self.assertIsNone(x.next_dna(DNA('10')))

  def test_iter_dna(self):
    x = custom()
    with self.assertRaisesRegex(
        NotImplementedError,
        '`next_dna` is not supported on \'CustomDecisionPoint\'.'):
      list(x.iter_dna())

    def next_dna(dna):
      if dna is None:
        return DNA('0')
      value = int(dna.value)
      if value == 10:
        return None
      return DNA(str(value + 1))

    x = custom(next_dna_fn=next_dna)
    self.assertEqual(list(x.iter_dna()), [DNA(str(i)) for i in range(11)])

  def test_random_dna(self):
    x = custom()
    with self.assertRaisesRegex(
        NotImplementedError,
        '`random_dna` is not supported on \'CustomDecisionPoint\'.'):
      x.random_dna()

    def random_dna(r, old_dna):
      del old_dna
      return DNA(str(r.randint(0, 10)))

    x = custom(random_dna_fn=random_dna)
    self.assertEqual(x.random_dna(random.Random(1)), DNA('2'))

  def test_inspection(self):
    self.assertEqual(
        str(custom('CustomA', location='a.b')),
        'CustomDecisionPoint(id=\'a.b\', hyper_type=\'CustomA\')')


if __name__ == '__main__':
  unittest.main()
