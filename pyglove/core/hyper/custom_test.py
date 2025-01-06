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
import unittest

from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core import utils
from pyglove.core.hyper.categorical import oneof
from pyglove.core.hyper.custom import CustomHyper
from pyglove.core.hyper.iter import iterate
from pyglove.core.hyper.object_template import materialize


class IntSequence(CustomHyper):

  def custom_decode(self, dna):
    return [int(v) for v in dna.value.split(',')]


class IntSequenceWithEncode(IntSequence):

  def custom_encode(self, value):
    return geno.DNA(','.join([str(v) for v in value]))

  def next_dna(self, dna):
    if dna is None:
      return geno.DNA(','.join([str(i) for i in range(5)]))
    v = self.custom_decode(dna)
    v.append(len(v))
    return self._create_dna(v)

  def random_dna(self, random_generator, previous_dna):
    del previous_dna
    k = random_generator.randint(0, 10)
    v = random_generator.choices(list(range(10)), k=k)
    return self._create_dna(v)

  def _create_dna(self, numbers):
    return geno.DNA(','.join([str(n) for n in numbers]))


class CustomHyperTest(unittest.TestCase):
  """Test for CustomHyper."""

  def test_dna_spec(self):
    self.assertTrue(
        symbolic.eq(
            IntSequence(hints='x').dna_spec('a'),
            geno.CustomDecisionPoint(
                hyper_type='IntSequence', location=utils.KeyPath('a'), hints='x'
            ),
        )
    )

  def test_decode(self):
    self.assertEqual(IntSequence().decode(geno.DNA('0,1,2')), [0, 1, 2])
    self.assertEqual(IntSequence().decode(geno.DNA('0')), [0])
    with self.assertRaisesRegex(ValueError, '.* expects string type DNA'):
      IntSequence().decode(geno.DNA(1))

  def test_encode(self):
    self.assertEqual(
        IntSequenceWithEncode().encode([0, 1, 2]), geno.DNA('0,1,2'))

    with self.assertRaisesRegex(
        NotImplementedError, '\'custom_encode\' is not supported by'):
      _ = IntSequence().encode([0, 1, 2])

  def test_random_dna(self):
    self.assertEqual(
        geno.random_dna(
            IntSequenceWithEncode().dna_spec('a'), random.Random(1)),
        geno.DNA('5,8'))

    with self.assertRaisesRegex(
        NotImplementedError, '`random_dna` is not implemented in .*'):
      geno.random_dna(IntSequence().dna_spec('a'))

  def test_iter(self):
    self.assertEqual(IntSequenceWithEncode().first_dna(), geno.DNA('0,1,2,3,4'))
    self.assertEqual(
        list(iterate(IntSequenceWithEncode(), 3)),
        [[0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4, 5],
         [0, 1, 2, 3, 4, 5, 6]])

    with self.assertRaisesRegex(
        NotImplementedError, '`next_dna` is not implemented in .*'):
      next(iterate(IntSequence()))

  def test_interop_with_other_primitives(self):
    v = oneof([IntSequence(), 1, 2])
    self.assertEqual(materialize(v, geno.DNA(1)), 1)
    self.assertEqual(materialize(v, geno.DNA((0, '3,4'))), [3, 4])


if __name__ == '__main__':
  unittest.main()
