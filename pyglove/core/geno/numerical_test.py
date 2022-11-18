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
"""Tests for pyglove.geno.Float."""

import random
import unittest

from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.space import constant


class FloatTest(unittest.TestCase):
  """Tests for `pg.geno.Float`."""

  def test_init(self):
    floatv(-1.0, 1.0)
    floatv(0.0, 1.0, scale='linear')
    floatv(1e-8, 1.0, scale='log')
    floatv(1e-8, 1.0, scale='rlog')

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'min_value\' .* should be no greater than \'max_value\''):
      floatv(1.0, -1.0)

    with self.assertRaisesRegex(ValueError, '\'min_value\' must be positive'):
      floatv(min_value=0.0, max_value=1.0, scale='log')

    with self.assertRaisesRegex(
        ValueError, 'Value .* is not in candidate list'):
      floatv(min_value=0.0, max_value=1.0, scale='unsupported_scale')

  def test_basics(self):
    x = floatv(0.1, 1.0, location='a.b')
    self.assertEqual(x.location.keys, ['a', 'b'])
    self.assertIsNone(x.scale)
    self.assertIsNone(x.name)

    self.assertFalse(x.is_space)
    self.assertFalse(x.is_categorical)
    self.assertFalse(x.is_subchoice)
    self.assertTrue(x.is_numerical)
    self.assertFalse(x.is_custom_decision_point)

  def test_len(self):
    self.assertEqual(len(floatv(0.1, 1.0)), 1)

  def test_space_size(self):
    self.assertEqual(floatv(0.1, 1.0).space_size, -1)

  def test_id(self):
    x = floatv(-1.0, 1.0, location='x')
    a = oneof([constant(), x], location='a')
    self.assertEqual(a.id, 'a')
    self.assertEqual(x.id, 'a[=1/2].x')

  def test_get(self):
    x = floatv(-1.0, 1.0, location='x', name='foo')
    a = oneof([x, constant()], location='a')

    # Get by ID.
    self.assertIs(a.get('a'), a)
    self.assertIs(a.get('a[=0/2].x'), x)
    self.assertIs(x.get('a[=0/2].x'), x)

    # Get by name.
    self.assertIs(a.get('foo'), x)

  def test_getitem(self):
    x = floatv(-1.0, 1.0, location='x', name='foo')
    _ = oneof([x, constant()], location='a')

    # Get by ID.
    self.assertIs(x['a[=0/2].x'], x)

    # Get by name.
    self.assertIs(x['foo'], x)

    with self.assertRaises(KeyError):
      _ = x['y']    # There is no decision point named `y` under x.

  def test_decision_ids(self):
    x = floatv(-1.0, 1.0, location='x')
    _ = oneof([x, constant()], location='a')
    self.assertEqual(
        x.decision_ids,
        [
            'a[=0/2].x',
        ])

  def test_decision_points(self):
    x = floatv(-1.0, 1.0, location='x')
    _ = oneof([x, constant()], location='a')
    self.assertEqual(x.decision_points, [x])

  def test_named_decision_points(self):
    x = floatv(-1.0, 1.0, location='x', name='foo')
    _ = oneof([x, constant()], location='a')
    self.assertEqual(x.named_decision_points, {'foo': x})

  def test_set_get_userdata(self):
    x = floatv(-1.0, 1.0)
    x.set_userdata('mydata', 1)
    self.assertEqual(x.userdata.mydata, 1)
    self.assertEqual(x.userdata['mydata'], 1)
    with self.assertRaises(KeyError):
      _ = x.userdata.nonexist_data
    with self.assertRaises(KeyError):
      _ = x.userdata['nonexist_data']

  def test_parent_spec(self):
    x = floatv(-1.0, 1.0, location='x')
    self.assertIsNone(x.parent_spec)
    a = oneof([x, constant()], location='a')
    self.assertIs(x.parent_spec, a.candidates[0])

  def test_parent_choice(self):
    x = floatv(-1.0, 1.0, location='x')
    self.assertIsNone(x.parent_choice)
    a = oneof([x, constant()], location='a')
    self.assertIs(x.parent_choice, a)

  def test_validate(self):
    x = floatv(min_value=0.0, max_value=1.0)
    x.validate(DNA(0.5))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      x.validate(DNA(None))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      x.validate(DNA(1))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no less than 0.*'):
      x.validate(DNA(-1.0))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than 1.*'):
      x.validate(DNA(1.5))

    with self.assertRaisesRegex(
        ValueError, 'Float DNA should have no children'):
      x.validate(DNA((1.0, [1, 2])))

  def test_first_dna(self):
    x = floatv(min_value=0.0, max_value=1.0)
    self.assertEqual(x.first_dna(), DNA(0.0))

  def test_next_dna(self):
    x = floatv(min_value=0.0, max_value=1.0)
    self.assertEqual(x.next_dna(), DNA(0.0))
    with self.assertRaisesRegex(
        NotImplementedError, '`next_dna` is not supported on `Float` yet'):
      x.next_dna(DNA(0.0))

  def test_iter_dna(self):
    x = floatv(min_value=0.0, max_value=1.0)
    with self.assertRaisesRegex(
        NotImplementedError, '`next_dna` is not supported on `Float` yet'):
      list(x.iter_dna())

  def test_random_dna(self):
    x = floatv(min_value=0.0, max_value=1.0)
    self.assertEqual(
        x.random_dna(random.Random(1)), DNA(0.13436424411240122))

  def test_inspection(self):
    self.assertEqual(
        str(floatv(min_value=0.0, max_value=1.0, location='a.b')),
        'Float(id=\'a.b\', min_value=0.0, max_value=1.0)')


if __name__ == '__main__':
  unittest.main()
