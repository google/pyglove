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
import unittest

from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core import utils
from pyglove.core.hyper.numerical import Float
from pyglove.core.hyper.numerical import floatv


class FloatTest(unittest.TestCase):
  """Test for hyper.Float."""

  def test_basics(self):
    v = floatv(0.0, 1.0)
    self.assertEqual(v.min_value, 0.0)
    self.assertEqual(v.max_value, 1.0)
    self.assertIsNone(v.scale)
    self.assertTrue(v.is_leaf)

    with self.assertRaisesRegex(
        ValueError, '\'min_value\' .* is greater than \'max_value\' .*'):
      floatv(min_value=1.0, max_value=0.0)

  def test_scale(self):
    self.assertEqual(floatv(-1.0, 1.0, 'linear').scale, 'linear')
    with self.assertRaisesRegex(
        ValueError, '\'min_value\' must be positive'):
      floatv(-1.0, 1.0, 'log')

  def test_dna_spec(self):
    self.assertTrue(
        symbolic.eq(
            floatv(0.0, 1.0).dna_spec('a'),
            geno.Float(
                location=utils.KeyPath('a'), min_value=0.0, max_value=1.0
            ),
        )
    )

  def test_decode(self):
    v = floatv(0.0, 1.0)
    self.assertEqual(v.decode(geno.DNA(0.0)), 0.0)
    self.assertEqual(v.decode(geno.DNA(1.0)), 1.0)

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      v.decode(geno.DNA(1))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no less than'):
      v.decode(geno.DNA(-1.0))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than'):
      v.decode(geno.DNA(2.0))

  def test_encode(self):
    v = floatv(0.0, 1.0)
    self.assertEqual(v.encode(0.0), geno.DNA(0.0))
    self.assertEqual(v.encode(1.0), geno.DNA(1.0))

    with self.assertRaisesRegex(
        ValueError, 'Value should be float to be encoded'):
      v.encode('abc')

    with self.assertRaisesRegex(
        ValueError, 'Value should be no less than'):
      v.encode(-1.0)

    with self.assertRaisesRegex(
        ValueError, 'Value should be no greater than'):
      v.encode(2.0)

  def test_assignment_compatibility(self):
    sd = symbolic.Dict.partial(
        value_spec=pg_typing.Dict([
            ('a', pg_typing.Int()),
            ('b', pg_typing.Float()),
            ('c', pg_typing.Union([pg_typing.Str(), pg_typing.Float()])),
            ('d', pg_typing.Any()),
            ('e', pg_typing.Float(max_value=0.0)),
            ('f', pg_typing.Float(min_value=1.0))
        ]))
    v = floatv(min_value=0.0, max_value=1.0)
    sd.b = v
    sd.c = v
    sd.d = v

    self.assertEqual(sd.b.sym_path, 'b')
    self.assertEqual(sd.c.sym_path, 'c')
    self.assertEqual(sd.d.sym_path, 'd')
    with self.assertRaisesRegex(
        TypeError, 'Source spec Float\\(\\) is not compatible with '
        'destination spec Int\\(\\)'):
      sd.a = v

    with self.assertRaisesRegex(
        ValueError,
        'Float.max_value .* should be no greater than the max value'):
      sd.e = v

    with self.assertRaisesRegex(
        ValueError,
        'Float.min_value .* should be no less than the min value'):
      sd.f = v

  def test_custom_apply(self):
    v = floatv(min_value=0.0, max_value=1.0)
    self.assertIs(pg_typing.Object(Float).apply(v), v)
    self.assertIs(pg_typing.Float().apply(v), v)
    with self.assertRaisesRegex(
        TypeError, r'Source spec Float\(\) is not compatible'):
      pg_typing.Int().apply(v)

    with self.assertRaisesRegex(
        ValueError, r'.* should be no less than the min value'):
      pg_typing.Float(min_value=2.0).apply(v)


if __name__ == '__main__':
  unittest.main()
