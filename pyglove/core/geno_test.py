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
"""Tests for pyglove.geno."""

import inspect
import random
import unittest
from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing


class DNATest(unittest.TestCase):
  """Tests for geno.DNA."""

  def testBasics(self):
    """Test basic interfaces of DNA."""
    dna = geno.DNA(1, [geno.DNA(2), geno.DNA(3.0), geno.DNA('abc')])
    self.assertEqual(dna.value, 1)
    self.assertIsNotNone(dna.children)
    self.assertEqual(len(dna.children), 3)
    self.assertIsNone(dna.spec)
    self.assertIsNone(dna.parent_dna)
    self.assertIs(dna.children[0].parent_dna, dna)
    self.assertIs(dna.root, dna)
    self.assertIs(dna.children[0].root, dna)

    # Test __contains__.
    self.assertIn(2, dna)
    self.assertIn(3.0, dna)
    self.assertIn('abc', dna)
    self.assertIn(geno.DNA(2), dna)
    self.assertNotIn(0.1, dna)
    self.assertNotIn('foo', dna)
    with self.assertRaisesRegex(
        ValueError, 'DNA.__contains__ does not accept .*'):
      _ = (1, 1) in dna

    # Test __iter__.
    values = []
    for child in dna:
      values.append(child.value)
    self.assertEqual(values, [2, 3.0, 'abc'])

    # Test __getitem__.
    self.assertEqual(dna[0], geno.DNA(2))
    with self.assertRaisesRegex(IndexError, 'list index out of range'):
      _ = dna[3]

    # Test is_leaf.
    self.assertFalse(dna.is_leaf)
    self.assertTrue(dna.children[0].is_leaf)
    self.assertTrue(dna.children[1].is_leaf)
    self.assertTrue(dna.children[2].is_leaf)

    # Test methods that need spec to be invovled.
    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      dna.get('a')

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      _ = dna.literal_value

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      _ = dna.decision_ids

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      _ = dna.to_dict()

  def testInspection(self):
    """Test inspection on DNA."""
    self.assertEqual(geno.DNA(0).to_json(type_info=False), 0)
    self.assertEqual(geno.DNA([0, 1]).to_json(type_info=False), [0, 1])
    self.assertEqual(
        geno.DNA([0, (0, 1)]).to_json(type_info=False), [0, (0, 1)])
    self.assertEqual(
        geno.DNA((0, 0.5)).to_json(type_info=False), (0, 0.5))

    self.assertEqual(
        str(geno.DNA((0, [1, 0.5, 'abc']))), 'DNA(0, [1, 0.5, \'abc\'])')
    self.assertEqual(
        geno.DNA(0.0).use_spec(
            geno.space([
                geno.floatv(0.0, 1.0, location='a')
            ])).format(as_dict=True), 'DNA({\n'
        '  \'a\': 0.0\n'
        '})')

  def testInit(self):
    """Test DNA.__init__."""
    dna_spec = geno.floatv(min_value=0.0, max_value=1.0)
    dna = geno.DNA(0.5, spec=dna_spec)
    self.assertEqual(dna.spec, dna_spec)

    # Reduant None-value ancestors. (single node)
    # should be reduced to the DNA value itself.
    self.assertEqual(
        geno.DNA(None, [geno.DNA(None, [geno.DNA(None, [
            geno.DNA(1),
        ])])]), geno.DNA(1))

    # Reduant None-value ancestors. (multiple nodes)
    # should be reduced to 2-level DNA: a None parent and children with values.
    self.assertEqual(
        geno.DNA(
            None,
            [geno.DNA(None,
                      [geno.DNA(None, [geno.DNA(1), geno.DNA('abc')])])]),
        geno.DNA(None, [geno.DNA(1), geno.DNA('abc')]))

    # No redudant node in the DNA tree, remains the same.
    self.assertEqual(
        geno.DNA(None, [
            geno.DNA(1),
            geno.DNA(None,
                     [geno.DNA(2), geno.DNA(None, [geno.DNA(3)])])
        ]),
        geno.DNA(None, [
            geno.DNA(1),
            geno.DNA(None,
                     [geno.DNA(2), geno.DNA(None, [geno.DNA(3)])])
        ]))

    # Tests for compositional values.
    self.assertEqual(
        geno.DNA([1, 2]), geno.DNA(None, [geno.DNA(1), geno.DNA(2)]))

    self.assertEqual(geno.DNA((1, 1)), geno.DNA(1, [geno.DNA(1)]))

    self.assertEqual(
        geno.DNA((1, 2, [3, 4])),
        geno.DNA(1, [geno.DNA(2, [geno.DNA(3), geno.DNA(4)])]))

    self.assertEqual(
        geno.DNA([(1, [2, (3, 4)]), 5, (6, [7, 8])]),
        geno.DNA(None, [
            geno.DNA(1, [geno.DNA(2), geno.DNA(3, [geno.DNA(4)])]),
            geno.DNA(5),
            geno.DNA(6, [geno.DNA(7), geno.DNA(8)])
        ]))

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):
      geno.DNA(ValueError())

    with self.assertRaisesRegex(
        ValueError,
        '\'children\' .* must be None when \'value\' .* is compositional.'):
      geno.DNA([1, 2], [geno.DNA(1)])

    with self.assertRaisesRegex(
        ValueError,
        'Tuple as conditional choices must have at least 2 items'):
      geno.DNA((1,))

    with self.assertRaisesRegex(
        ValueError, 'Tuple as conditional choices only allow multiple '
        'choices to be used at the last position'):
      geno.DNA([(1, [1, 2], 3)])

  def testFromFn(self):
    """Test DNA.from_fn."""
    dna_spec = geno.space([
        geno.oneof([
            geno.manyof(2, [geno.constant(), geno.constant()]),
            geno.constant()]),
        geno.floatv(0.0, 1.0)
    ])

    stats = dict(invocation=0)
    def dna_emitter(spec):
      stats['invocation'] += 1
      return spec.first_dna()

    # `dna_emitter` should be called only twice, once on
    # the top-most Choices and once on the Float. Since both
    # of them returns a DNA, we use its child DNA directly
    # instead of stepping into the child DNASpec.
    dna = geno.DNA.from_fn(dna_spec, dna_emitter)
    self.assertEqual(stats['invocation'], 2)
    self.assertEqual(dna, geno.DNA([(0, [0, 1]), 0.0]))

    def zero_emitter(spec):
      if isinstance(spec, geno.Float):
        return spec.min_value
      return range(spec.num_choices)

    dna = geno.DNA.from_fn(dna_spec, zero_emitter)
    self.assertEqual(dna, geno.DNA([(0, [0, 1]), 0.0]))

    def choice_out_of_range(spec):
      if isinstance(spec, geno.Float):
        return spec.min_value
      return [len(spec.candidates)] * spec.num_choices

    with self.assertRaisesRegex(ValueError, 'Choice out of range.'):
      geno.DNA.from_fn(dna_spec, choice_out_of_range)

    def bad_choice_value(spec):
      del spec
      return [0.1]

    with self.assertRaisesRegex(ValueError, 'Choice value should be int'):
      geno.DNA.from_fn(dna_spec, bad_choice_value)

    def unmatched_num_choices(spec):
      del spec
      return [0]

    with self.assertRaisesRegex(
        ValueError,
        'Number of DNA child values does not match the number of choices'):
      geno.DNA.from_fn(dna_spec, unmatched_num_choices)

    with self.assertRaisesRegex(
        TypeError, 'Argument \'dna_spec\' should be DNASpec type.'):
      geno.DNA.from_fn(1, unmatched_num_choices)

  def testSerialization(self):
    """Test serialization/deserialization."""
    dna = geno.DNA([0, (1, 2), [3, (4, 5, 'abc')]])
    dna.set_metadata('a', 1, cloneable=True)
    dna_str = symbolic.to_json_str(dna)
    dna2 = symbolic.from_json_str(dna_str)
    self.assertEqual(dna, dna2)
    self.assertEqual(dna.metadata, dna2.metadata)

    # Serialization using non-compact form.
    dna_str_verbose = symbolic.to_json_str(dna, compact=False)
    self.assertGreater(len(dna_str_verbose), len(dna_str))
    dna3 = symbolic.from_json_str(dna_str_verbose)
    self.assertEqual(dna, dna3)
    self.assertEqual(dna.metadata, dna3.metadata)

  def testUseSpec(self):
    """Test DNA with spec."""
    # Test single choice.
    spec = geno.oneof([
        geno.constant(),
        geno.space([
            geno.floatv(0.0, 1.0, hints=0),
            geno.floatv(0.0, 1.0),
            geno.custom()
        ]),
        geno.space([geno.floatv(0.0, 1.0, hints=2)]),
    ], hints=1)
    self.assertEqual(geno.DNA(0).use_spec(spec).spec, spec)

    dna = geno.DNA((1, [0.0, 1.0, 'abc'])).use_spec(spec)
    self.assertIs(dna.spec, spec)
    self.assertTrue(symbolic.eq(
        dna.children[0].spec,
        geno.floatv(min_value=0.0, max_value=1.0, hints=0)))
    self.assertTrue(symbolic.eq(
        dna.children[1].spec,
        geno.floatv(min_value=0.0, max_value=1.0)))
    self.assertTrue(symbolic.eq(
        dna.children[2].spec,
        geno.custom()))

    dna = geno.DNA((2, 0.0)).use_spec(spec)
    self.assertIs(dna.spec, spec)
    self.assertTrue(symbolic.eq(
        dna.children[0].spec,
        geno.floatv(min_value=0.0, max_value=1.0, hints=2)))

    with self.assertRaisesRegex(ValueError, 'Unsupported spec'):
      geno.DNA(None).use_spec(1)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      geno.DNA(None).use_spec(spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      geno.DNA(1.5).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError,
        'Value of DNA is out of range according to the DNA spec'):
      geno.DNA(3).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'There is no DNA spec for child DNA values'):
      geno.DNA((0, 0)).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError,
        'Number of elements in child templates .* does not match '
        'with the length of children .* from DNA'):
      geno.DNA((1, 0)).use_spec(spec)

    # Test multiple choices.
    spec = geno.manyof(2, [
        geno.constant(),
        geno.constant(),
        geno.constant(),
    ], sorted=True, distinct=True, hints=0)
    dna = geno.DNA([0, 1]).use_spec(spec)
    self.assertIs(dna.spec, spec)
    child_spec = spec.clone(
        override=dict(num_choices=1, location='[0]', subchoice_index=0))
    self.assertTrue(symbolic.eq(dna.children[0].spec, child_spec))
    with self.assertRaisesRegex(
        ValueError, 'Cannot apply multi-choice DNA spec on value'):
      geno.DNA(1).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError,
        'Number of choices .* does not match with the number '
        'of child values'):
      geno.DNA([0, 0, 1]).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not sorted'):
      geno.DNA([1, 0]).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not distinct'):
      geno.DNA([0, 0]).use_spec(spec)

    # Test float.
    spec = geno.floatv(min_value=0.0, max_value=1.0, hints=1)
    self.assertIs(geno.DNA(0.5).use_spec(spec).spec, spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      geno.DNA(None).use_spec(spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      geno.DNA(0).use_spec(spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      geno.DNA([0, 1]).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no less than .*'):
      geno.DNA(-0.1).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than .*'):
      geno.DNA(2.).use_spec(spec)

    # Test custom decision point.
    spec = geno.custom()
    self.assertIs(geno.DNA('abc').use_spec(spec).spec, spec)

    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      geno.DNA(1).use_spec(geno.space([geno.custom()]))

    # Test complex spec.
    spec = geno.space([
        geno.oneof([
            geno.manyof(2, [
                geno.constant(),
                geno.floatv(0.0, 1.0),
                geno.constant(),
                geno.custom()
            ], sorted=True, distinct=True, location='b'),
            geno.floatv(0.0, 1.0, location='c'),
            geno.constant(),
        ], location='a'),
        geno.floatv(-1.0, 0.0, location='d')
    ])
    dna = geno.DNA([(0, [0, (1, 0.5, 'abc')]), -0.5]).use_spec(spec)
    self.assertIsNotNone(dna.spec)

    with self.assertRaisesRegex(
        ValueError, 'Argument \'spec\' must not be None.'):
      geno.DNA([2, -0.5]).use_spec(None)

    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch.'):
      geno.DNA(1).use_spec(spec)

    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch.'):
      geno.DNA([0.5, -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError,
        'Number of choices .* does not match with the number of '
        'child values'):
      geno.DNA([(0, 0), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError,
        'Number of choices .* does not match with the number of '
        'child values'):
      geno.DNA([(0, 2, 1), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError,
        'Value of DNA is out of range according to the DNA spec.'):
      geno.DNA([4, -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Encountered more than 1 value.'):
      geno.DNA([1, [0.0, 1.0]]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not sorted'):
      geno.DNA([(0, [(1, 0.5), 0]), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not distinct'):
      geno.DNA([(0, [(1, 0.2), (1, 0.5)]), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Length of DNA child values .* is different from '
        'the number of elements.'):
      geno.DNA([0, 1]).use_spec(geno.space([]))

  def testDictConversion(self):
    """Test geno.DNA.from_dict and geno.DNA.to_dict."""
    spec = geno.space([
        geno.oneof([
            # NOTE(daiyip): multiple choices under conditions.
            geno.manyof(3, [
                geno.constant(),
                geno.floatv(0.0, 1.0, name='c'),
                geno.constant(),
                geno.custom(name='f')
            ], literal_values=[
                123, 'Float(...)', 'xyz', 'Custom(...)'
            ], location='b', name='b'),
            geno.constant(),
        ], literal_values=[
            'Something complex',
            '\'foo\''
        ], location='a', name='a'),
        geno.floatv(-1.0, 0.0, location='d', name='d'),
        # NOTE(daiyip): multiple choices NOT under conditions.
        geno.manyof(2, [
            geno.constant(), geno.constant(), geno.constant()
        ], literal_values=[
            '\'a\'',
            0.333,
            '\'c\'',
        ], location='e', name='e')
    ])
    dna = geno.DNA(
        [(0, [0, (1, 0.5), (3, 'abc')]), -0.5, [0, 1]]).use_spec(spec)

    # Test cases for to_dict using 'id' as key type.
    # Test case 1: key_type='id', value_type='value'
    self.assertEqual(
        dna.to_dict(), {
            'a': 0,
            'a[=0/2].b[0]': 0,
            'a[=0/2].b[1]': 1,
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[2]': 3,
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e[0]': 0,
            'e[1]': 1,
        })

    # Test case 2: key_type='id', value_type='dna'
    self.assertEqual(
        dna.to_dict(value_type='dna'), {
            'a': geno.DNA((0, [0, (1, 0.5), (3, 'abc')])),
            'a[=0/2].b[0]': geno.DNA(0),
            'a[=0/2].b[1]': geno.DNA((1, 0.5)),
            'a[=0/2].b[1][=1/4]': geno.DNA(0.5),
            'a[=0/2].b[2]': geno.DNA((3, 'abc')),
            'a[=0/2].b[2][=3/4]': geno.DNA('abc'),
            'd': geno.DNA(-0.5),
            'e[0]': geno.DNA(0),
            'e[1]': geno.DNA(1),
        })

    # Test case 3: key_type='id', value_type='choice'
    self.assertEqual(
        dna.to_dict(value_type='choice'), {
            'a': '0/2',
            'a[=0/2].b[0]': '0/4',
            'a[=0/2].b[1]': '1/4',
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[2]': '3/4',
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e[0]': '0/3',
            'e[1]': '1/3',
        })

    # Test case 4: key_type='id', value_type='literal'
    self.assertEqual(
        dna.to_dict(value_type='literal'), {
            'a': 'Something complex',
            'a[=0/2].b[0]': 123,
            'a[=0/2].b[1]': 'Float(...)',
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[2]': 'Custom(...)',
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e[0]': '\'a\'',
            'e[1]': 0.333,
        })

    # Test case 5: key_type='id', value_type='choice_literal'
    self.assertEqual(
        dna.to_dict(value_type='choice_and_literal'), {
            'a': '0/2 (Something complex)',
            'a[=0/2].b[0]': '0/4 (123)',
            'a[=0/2].b[1]': '1/4 (Float(...))',
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[2]': '3/4 (Custom(...))',
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e[0]': '0/3 (\'a\')',
            'e[1]': '1/3 (0.333)',
        })

    # Test cases for to_dict using 'name_or_id' as key type.
    # Test case 1: key_type='name_or_id', value_type='value'
    self.assertEqual(
        dna.to_dict(key_type='name_or_id'), {
            'a': 0,
            'b': [0, 1, 3],
            'c': 0.5,
            'd': -0.5,
            'e': [0, 1],
            'f': 'abc'
        })

    # Test case 2: key_type='name_or_id', value_type='dna'
    self.assertEqual(
        dna.to_dict(key_type='name_or_id', value_type='dna'), {
            'a': geno.DNA((0, [0, (1, 0.5), (3, 'abc')])),
            'b': [geno.DNA(0), geno.DNA((1, 0.5)), geno.DNA((3, 'abc'))],
            'c': geno.DNA(0.5),
            'd': geno.DNA(-0.5),
            'e': [geno.DNA(0), geno.DNA(1)],
            'f': geno.DNA('abc'),
        })

    # Test case 3: key_type='name_or_id', value_type='choice'
    self.assertEqual(
        dna.to_dict(key_type='name_or_id', value_type='choice'), {
            'a': '0/2',
            'b': ['0/4', '1/4', '3/4'],
            'c': 0.5,
            'd': -0.5,
            'e': ['0/3', '1/3'],
            'f': 'abc'
        })

    # Test case 4: key_type='name_or_id', value_type='literal'
    self.assertEqual(
        dna.to_dict(key_type='name_or_id', value_type='literal'), {
            'a': 'Something complex',
            'b': [123, 'Float(...)', 'Custom(...)'],
            'c': 0.5,
            'd': -0.5,
            'e': ['\'a\'', 0.333],
            'f': 'abc'
        })

    # Test case 5: key_type='name_or_id', value_type='choice_literal'
    self.assertEqual(
        dna.to_dict(key_type='name_or_id', value_type='choice_and_literal'), {
            'a': '0/2 (Something complex)',
            'b': ['0/4 (123)', '1/4 (Float(...))', '3/4 (Custom(...))'],
            'c': 0.5,
            'd': -0.5,
            'e': ['0/3 (\'a\')', '1/3 (0.333)'],
            'f': 'abc'
        })

    # Test cases for using 'dna_spec' as key types.
    # Test case 1: key_type='dna_spec', value_type='value'
    named = spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec'), {
            named['a']: 0,
            named['b'][0]: 0,
            named['b'][1]: 1,
            named['b'][2]: 3,
            named['c'][1]: 0.5,
            named['d']: -0.5,
            named['e'][0]: 0,
            named['e'][1]: 1,
            named['f'][2]: 'abc',
        })

    # Test case 2: key_type='dna_spec', value_type='dna'
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='dna'), {
            named['a']: geno.DNA((0, [0, (1, 0.5), (3, 'abc')])),
            named['b'][0]: geno.DNA(0),
            named['b'][1]: geno.DNA((1, 0.5)),
            named['b'][2]: geno.DNA((3, 'abc')),
            named['c'][1]: geno.DNA(0.5),
            named['d']: geno.DNA(-0.5),
            named['e'][0]: geno.DNA(0),
            named['e'][1]: geno.DNA(1),
            named['f'][2]: geno.DNA('abc')
        })

    # Test case 3: key_type='dna_spec', value_type='choice'
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='choice'), {
            named['a']: '0/2',
            named['b'][0]: '0/4',
            named['b'][1]: '1/4',
            named['b'][2]: '3/4',
            named['c'][1]: 0.5,
            named['d']: -0.5,
            named['e'][0]: '0/3',
            named['e'][1]: '1/3',
            named['f'][2]: 'abc'
        })

    # Test case 4: key_type='dna_spec', value_type='literal'
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='literal'), {
            named['a']: 'Something complex',
            named['b'][0]: 123,
            named['b'][1]: 'Float(...)',
            named['b'][2]: 'Custom(...)',
            named['c'][1]: 0.5,
            named['d']: -0.5,
            named['e'][0]: '\'a\'',
            named['e'][1]: 0.333,
            named['f'][2]: 'abc'
        })

    # Test case 5: key_type='dna_spec', value_type='choice_and_literal'
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='choice_and_literal'), {
            named['a']: '0/2 (Something complex)',
            named['b'][0]: '0/4 (123)',
            named['b'][1]: '1/4 (Float(...))',
            named['b'][2]: '3/4 (Custom(...))',
            named['c'][1]: 0.5,
            named['d']: -0.5,
            named['e'][0]: '0/3 (\'a\')',
            named['e'][1]: '1/3 (0.333)',
            named['f'][2]: 'abc'
        })

    # Test cases for collapsing choice decisions:
    # Test case 1: key_type='id', value_type='value',
    self.assertEqual(
        dna.to_dict(multi_choice_key='parent'), {
            'a': 0,
            'a[=0/2].b': [0, 1, 3],
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e': [0, 1]
        })

    # Test case 2: key_type='dna_spec', value_type='dna',
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='dna',
                    multi_choice_key='both'),
        {
            named['a']: geno.DNA((0, [0, (1, 0.5), (3, 'abc')])),
            named['b'][0].parent_spec: [geno.DNA(0),
                                        geno.DNA((1, 0.5)),
                                        geno.DNA((3, 'abc'))],
            named['b'][0]: geno.DNA(0),
            named['b'][1]: geno.DNA((1, 0.5)),
            named['b'][2]: geno.DNA((3, 'abc')),
            named['c'][1]: geno.DNA(0.5),
            named['d']: geno.DNA(-0.5),
            named['e'][0].parent_spec: [geno.DNA(0), geno.DNA(1)],
            named['e'][0]: geno.DNA(0),
            named['e'][1]: geno.DNA(1),
            named['f'][2]: geno.DNA('abc'),
        })

    # Test cases for including inactive decisions:
    # Test case 1: key_type='id', value_type='value',
    self.assertEqual(
        dna.to_dict(include_inactive_decisions=True), {
            'a': 0,
            'a[=0/2].b[0]': 0,
            'a[=0/2].b[0][=1/4]': None,
            'a[=0/2].b[0][=3/4]': None,
            'a[=0/2].b[1]': 1,
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[1][=3/4]': None,
            'a[=0/2].b[2]': 3,
            'a[=0/2].b[2][=1/4]': None,
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e[0]': 0,
            'e[1]': 1
        })

    # Test case 2: key_type='id', value_type='dna',
    # include_inactive_decisions=True
    # multi_choice_key='parent'
    self.assertEqual(
        dna.to_dict(value_type='dna',
                    multi_choice_key='parent',
                    include_inactive_decisions=True),
        {
            'a': geno.DNA((0, [0, (1, 0.5), (3, 'abc')])),
            'a[=0/2].b': [geno.DNA(0),
                          geno.DNA((1, 0.5)),
                          geno.DNA((3, 'abc'))],
            'a[=0/2].b[0][=1/4]': None,
            'a[=0/2].b[0][=3/4]': None,
            'a[=0/2].b[1][=1/4]': geno.DNA(0.5),
            'a[=0/2].b[1][=3/4]': None,
            'a[=0/2].b[2][=1/4]': None,
            'a[=0/2].b[2][=3/4]': geno.DNA('abc'),
            'd': geno.DNA(-0.5),
            'e': [geno.DNA(0), geno.DNA(1)],
        })

    # Test bad inputs for `to_dict`.
    with self.assertRaisesRegex(
        ValueError, '\'key_type\' must be either .*'):
      dna.to_dict(key_type='foo')

    with self.assertRaisesRegex(
        ValueError, '\'value_type\' must be either .*'):
      dna.to_dict(value_type='foo')

    with self.assertRaisesRegex(
        ValueError, '\'multi_choice_key\' must be either .*'):
      dna.to_dict(multi_choice_key='foo')

    # Test DNA.from_dict()
    for kt in ['id', 'name_or_id', 'dna_spec']:
      for vt in ['value', 'dna', 'choice', 'literal', 'choice_and_literal']:
        for mk in ['subchoice', 'parent', 'both']:
          for inactive in [True, False]:
            self.assertEqual(
                geno.DNA.from_dict(
                    dna.to_dict(key_type=kt, value_type=vt,
                                multi_choice_key=mk,
                                include_inactive_decisions=inactive),
                    spec, use_ints_as_literals=(vt == 'literal')),
                dna)

    # Test DNA.from_dict with filter_fn.
    self.assertEqual(
        dna.to_dict(
            value_type='dna',
            multi_choice_key='parent',
            include_inactive_decisions=True,
            filter_fn=lambda x: isinstance(x, geno.CustomDecisionPoint)),
        {
            'a[=0/2].b[0][=3/4]': None,
            'a[=0/2].b[1][=3/4]': None,
            'a[=0/2].b[2][=3/4]': geno.DNA('abc'),
        })

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      geno.DNA(0).to_dict()

    # Test DNA.from_parameters()
    dna2 = geno.DNA.from_parameters(dna.parameters(), spec)
    self.assertEqual(dna, dna2)

    dna3 = geno.DNA.from_parameters(
        dna.parameters(use_literal_values=True), spec, use_literal_values=True)
    self.assertEqual(dna, dna3)
    with self.assertRaisesRegex(
        ValueError, 'Value for .* is not found in the dictionary .*'):
      geno.DNA.from_parameters({'x': 1}, spec)
    with self.assertRaisesRegex(
        ValueError, 'There is no candidate in .*'):
      geno.DNA.from_parameters({'a': 'foo'}, spec)
    with self.assertRaisesRegex(
        ValueError,
        'Number of candidates .* for Choice .* does not match with DNASpec'):
      geno.DNA.from_parameters({'a': '0/3'}, spec)
    float_spec = geno.space([
        geno.floatv(location='a', min_value=0.0, max_value=1.0)])
    with self.assertRaisesRegex(
        ValueError, 'Value for .* is not found in the dictionary .*'):
      geno.DNA.from_parameters({'b': '0/3'}, float_spec)
    with self.assertRaisesRegex(
        ValueError, 'The decision for .* should be no less than .*'):
      geno.DNA.from_parameters({'a': -1.0}, float_spec)
    with self.assertRaisesRegex(
        ValueError, 'The decision for .* should be no greater than .*'):
      geno.DNA.from_parameters({'a': 2.0}, float_spec)

    custom_spec = geno.space([geno.custom(location='a')])
    with self.assertRaisesRegex(
        ValueError, 'The decision for .* should be a string'):
      geno.DNA.from_parameters({'a': 1}, custom_spec)

  def testSubNodesAccess(self):
    """Test sub nodes access."""
    spec = geno.space([
        geno.oneof([
            # NOTE(daiyip): multiple choices under conditions.
            geno.manyof(2, [
                geno.custom(),
                geno.floatv(0.0, 1.0, name='z'),
                geno.constant(),
            ], location='b', name='y'),
            geno.constant()
        ], literal_values=[
            '0/2 (Something complex)', '1/2 (\'foo\')'
        ], location='a', name='x'),
        geno.floatv(-1.0, 0.0, location='d', name='p'),
        # NOTE(daiyip): multiple choices NOT under conditions.
        geno.manyof(2, [
            geno.constant(), geno.constant(), geno.constant()
        ], literal_values=[
            '0/3 (\'a\')',
            '1/3 (\'b\')',
            '2/3 (\'c\')',
        ], location='e', name='q')
    ])
    dna = geno.DNA([(0, [(0, 'abc'), (1, 0.5)]), -0.5, [0, 1]]).use_spec(spec)

    # Test decision ids.
    self.assertEqual(dna.decision_ids, [
        'a',
        'a[=0/2].b',
        'a[=0/2].b[0][=0/3]',
        'a[=0/2].b[0][=1/3]',
        'a[=0/2].b[1][=0/3]',
        'a[=0/2].b[1][=1/3]',
        'd',
        'e'])

    # Test __getitem__ by name.
    self.assertEqual(dna['x'], geno.DNA((0, [(0, 'abc'), (1, 0.5)])))
    self.assertEqual(dna['y'], [geno.DNA((0, 'abc')), geno.DNA((1, 0.5))])
    self.assertEqual(dna['z'], geno.DNA(0.5))
    self.assertEqual(dna['p'], geno.DNA(-0.5))
    self.assertEqual(dna['q'], [geno.DNA(0), geno.DNA(1)])

    # Test __getitem__ by id (Text).
    self.assertEqual(dna['a'], geno.DNA((0, [(0, 'abc'), (1, 0.5)])))
    self.assertEqual(dna['a[=0/2].b'], [geno.DNA((0, 'abc')),
                                        geno.DNA((1, 0.5))])
    self.assertEqual(dna['a[=0/2].b[0]'], geno.DNA((0, 'abc')))
    self.assertEqual(dna['a[=0/2].b[0][=0/3]'], geno.DNA('abc'))
    self.assertIsNone(dna['a[=0/2].b[0][=1/3]'])
    self.assertIsNone(dna['a[=0/2].b[1][=0/3]'])
    self.assertEqual(dna['a[=0/2].b[1]'], geno.DNA((1, 0.5)))
    self.assertEqual(dna['a[=0/2].b[1][=1/3]'], geno.DNA(0.5))
    self.assertEqual(dna['d'], geno.DNA(-0.5))
    self.assertEqual(dna['e'], [geno.DNA(0), geno.DNA(1)])
    self.assertEqual(dna['e[0]'], geno.DNA(0))

    # Test __getitem__ by id (KeyPath).
    self.assertEqual(dna[dna.spec['a[=0/2].b[1][=1/3]'].id], geno.DNA(0.5))

    # Test __getitem__ by DNASpec.
    self.assertEqual(dna[dna.spec['a[=0/2].b[1][=1/3]']], geno.DNA(0.5))

    # Test DNA.get.
    self.assertEqual(dna.get('a[=0/2].b'), [geno.DNA((0, 'abc')),
                                            geno.DNA((1, 0.5))])
    self.assertEqual(dna.get('a[=0/2].b[1]'), geno.DNA((1, 0.5)))
    self.assertIsNone(dna.get('a[=0/2].b[0][=1/3]'))
    self.assertIsNone(dna.get('xyz'))

    # Test literal value.
    self.assertEqual(
        dna.literal_value,
        [
            ['0/2 -> 0/3 -> abc', '0/2 -> 1/3 -> 0.5'],
            '-0.5',
            ["'a'", "'b'"]
        ])
    self.assertEqual(dna['y'][0].literal_value, '0/3 -> abc')
    self.assertEqual(dna['z'].literal_value, '0.5')
    self.assertEqual(dna['q'][0].literal_value, '\'a\'')

    # Test is_subchoice.
    self.assertFalse(dna.is_subchoice)
    self.assertTrue(dna['y'][0].is_subchoice)
    self.assertFalse(dna['z'].is_subchoice)
    self.assertTrue(dna['q'][1].is_subchoice)

    # Test root.
    self.assertIs(dna['y'][0].root, dna)
    self.assertIs(dna['z'].root, dna)
    self.assertIs(dna['q'][0].root, dna)

  def testHash(self):
    """Test DNA.__hash__."""
    self.assertEqual(hash(geno.DNA(None)), hash(geno.DNA(None)))
    self.assertEqual(hash(geno.DNA(1)), hash(geno.DNA(1)))
    self.assertEqual(
        hash(
            geno.DNA(1, [geno.DNA(2, [geno.DNA(3), geno.DNA(4)]),
                         geno.DNA(5)])),
        hash(
            geno.DNA(1, [geno.DNA(2, [geno.DNA(3), geno.DNA(4)]),
                         geno.DNA(5)])))

  def testCmp(self):
    """Test DNA.__cmp__."""
    self.assertLess(geno.DNA(None), geno.DNA(0))
    self.assertLess(geno.DNA(0), geno.DNA(1))
    self.assertLess(geno.DNA([0, 1, 2]), geno.DNA([0, 2, 1]))
    self.assertLess(geno.DNA((0, 'abc')), geno.DNA((0, 'abd')))
    self.assertLess(geno.DNA([0, 1]), geno.DNA(0))
    self.assertLess(geno.DNA((0, 0, 0)), geno.DNA((0, 0, 1, 0)))
    with self.assertRaisesRegex(
        ValueError,
        'The two input DNA have different number of children'):
      _ = geno.DNA((0, 0, 0)) < geno.DNA((0, 0, 0, 0))

  def testToNumbers(self):
    """Test to_numbers."""
    self.assertEqual(geno.DNA(None).to_numbers(), [])
    self.assertEqual(geno.DNA(None).to_numbers(flatten=False), [])
    self.assertEqual(geno.DNA(1).to_numbers(), [1])
    self.assertEqual(geno.DNA(1).to_numbers(flatten=False), 1)
    self.assertEqual(geno.DNA(0.5).to_numbers(), [0.5])
    self.assertEqual(geno.DNA(0.5).to_numbers(flatten=False), 0.5)
    self.assertEqual(geno.DNA([0, 1, 2]).to_numbers(), [0, 1, 2])
    self.assertEqual(
        geno.DNA((0, [1, 2])).to_numbers(flatten=False),
        (0, [1, 2]))
    self.assertEqual(
        geno.DNA([0, 1, 2]).to_numbers(flatten=False), [0, 1, 2])
    self.assertEqual(
        geno.DNA([(0, 1), 2, (3, [4, 0.5])]).to_numbers(),
        [0, 1, 2, 3, 4, 0.5])
    self.assertEqual(
        geno.DNA([(0, 1), 'abc', (3, [4, 0.5])]).to_numbers(flatten=False),
        [(0, 1), 'abc', (3, [4, 0.5])])

  def testFromNumbers(self):
    """Test from_numbers."""
    spec = geno.space([
        geno.manyof(2, [
            geno.oneof([
                geno.floatv(0.0, 1.0),
                geno.constant()
            ]),
            geno.constant(),
        ]),
        geno.floatv(-1., 1.)
    ])
    self.assertEqual(
        geno.DNA.from_numbers([0, 0, 0.1, 1, 0.2], spec),
        geno.DNA([[(0, 0, 0.1), 1], 0.2]))

    with self.assertRaisesRegex(
        ValueError, 'Candidate index out of range at choice .*'):
      geno.DNA.from_numbers([0, 4, 0.1, 1, 0.2], spec)

    with self.assertRaisesRegex(
        ValueError, 'The input .* is too short'):
      geno.DNA.from_numbers([0, 0], spec)

    with self.assertRaisesRegex(
        ValueError, 'The input .* is too long'):
      geno.DNA.from_numbers([0, 0, 0.1, 1, 0.2, 0], spec)

  def testMetadata(self):
    """Test DNA metadata."""
    dna = geno.DNA(None)
    self.assertEqual(len(dna.metadata), 0)
    self.assertEqual(dna.set_metadata('a', 1).metadata.a, 1)

    dna = geno.DNA(None, metadata=dict(a=1))
    self.assertEqual(dna.metadata, dict(a=1))
    self.assertEqual(dna.set_metadata('a', 2).set_metadata('b', 3).metadata,
                     dict(a=2, b=3))
    # Neither metadata keys are cloneable.
    self.assertEqual(dna.clone(deep=True).metadata, {})

    dna_str = symbolic.to_json_str(dna)
    dna2 = symbolic.from_json_str(dna_str)
    self.assertEqual(dna2.metadata, dict(a=2, b=3))

    dna = geno.DNA(None, metadata=dict(a=1))
    dna.set_metadata('b', 2, cloneable=True)
    dna.set_metadata('c', 3, cloneable=False)
    self.assertEqual(dna._cloneable_metadata_keys, set('b'))    # pylint: disable=protected-access
    # Only metadata 'b' is cloneable.
    self.assertEqual(dna.clone(deep=True).metadata, dict(b=2))

    json_value = symbolic.to_json(dna)
    self.assertEqual(json_value['_cloneable_metadata_keys'], ['b'])
    dna2 = symbolic.from_json(json_value)
    # All metadata should be present during serialization but only cloneable
    # keys should be present after cloning.
    self.assertEqual(dna2.metadata, dict(a=1, b=2, c=3))
    self.assertEqual(dna2._cloneable_metadata_keys, set('b'))    # pylint: disable=protected-access
    self.assertEqual(dna2.clone(deep=True).metadata, dict(b=2))

  def testUserData(self):
    """Test DNA userdata."""
    dna = geno.DNA(None)
    self.assertEqual(len(dna.userdata), 0)
    self.assertEqual(dna.set_userdata('a', 1).userdata.a, 1)
    self.assertEqual(dna.set_userdata('b', 'foo', True).userdata.b, 'foo')

    # Test clone carries over the userdata 'b'
    dna2 = dna.clone()
    self.assertNotIn('a', dna2.userdata)
    self.assertEqual(dna2.userdata.b, 'foo')

  def testIterDNA(self):
    """Test iter_dna from current DNA."""
    dna_spec = geno.oneof([
        geno.constant(), geno.constant(), geno.constant(),
        geno.constant(), geno.constant(), geno.constant()])
    dna = geno.DNA(3, spec=dna_spec)
    self.assertEqual(dna.next_dna(), geno.DNA(4))
    self.assertEqual(list(dna.iter_dna()), [geno.DNA(4), geno.DNA(5)])

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      geno.DNA(3).next_dna()

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      next(geno.DNA(3).iter_dna())

  def testNamedDecisions(self):
    """Test named decisions in DNA."""
    t = geno.space([
        geno.manyof(3, [
            geno.space([
                geno.manyof(2, [
                    geno.floatv(0.1, 0.9, name='c', location='z'),
                    geno.oneof([
                        geno.constant(),
                        geno.constant(),
                        geno.constant(),
                    ], name='d'),
                    geno.constant(),
                    geno.constant(),
                ], location='y', name='b'),
                geno.oneof([
                    geno.constant(),
                    geno.constant()
                ], location='q')
            ]),
            geno.constant(),
            geno.constant()
        ], distinct=False, location='x', name='a'),
        geno.floatv(location='m', name='e', min_value=0.0, max_value=1.0)
    ])
    dna = geno.DNA([
        [(0, [[(0, 0.1), (1, 0)], 0]),
         (0, [[(0, 0.2), (1, 1)], 0]),
         (0, [[(0, 0.3), (1, 2)], 0])],
        0.0
    ], spec=t)
    self.assertEqual(
        list(dna.named_decisions.keys()), ['a', 'b', 'c', 'd', 'e'])
    self.assertEqual(dna.named_decisions['a'], [
        geno.DNA((0, [[(0, 0.1), (1, 0)], 0])),
        geno.DNA((0, [[(0, 0.2), (1, 1)], 0])),
        geno.DNA((0, [[(0, 0.3), (1, 2)], 0])),
    ])
    self.assertEqual(dna.named_decisions['b'], [
        geno.DNA((0, 0.1)),
        geno.DNA((1, 0)),
        geno.DNA((0, 0.2)),
        geno.DNA((1, 1)),
        geno.DNA((0, 0.3)),
        geno.DNA((1, 2))
    ])
    self.assertEqual(dna.named_decisions['c'], [
        geno.DNA(0.1), None, geno.DNA(0.2), None, geno.DNA(0.3), None
    ])
    self.assertEqual(dna.named_decisions['d'], [
        geno.DNA(0), None, geno.DNA(1), None, geno.DNA(2)
    ])
    self.assertEqual([d.parent_dna for d in dna.named_decisions['d'] if d], [
        geno.DNA((1, 0)),
        geno.DNA((1, 1)),
        geno.DNA((1, 2))
    ])
    self.assertEqual(dna.named_decisions['e'], geno.DNA(0.0))

  def testMultiChoiceSpec(self):
    """Test `DNA.multi_choice_spec` property."""
    dna_spec = geno.manyof(2, [
        geno.space([
            geno.oneof([geno.constant(), geno.constant()]),
            geno.oneof([geno.constant(), geno.constant()])
        ]),
        geno.manyof(2, [
            geno.constant(),
            geno.constant()
        ])
    ])
    dna = geno.DNA([(0, [1, 1]), (1, [0, 1])], spec=dna_spec)
    self.assertIs(dna.multi_choice_spec, dna_spec)
    self.assertTrue(dna.is_multi_choice_container)

    self.assertIsNone(dna.children[0].multi_choice_spec)
    self.assertFalse(dna.children[0].is_multi_choice_container)

    self.assertIsNone(dna.children[0].children[0].multi_choice_spec)
    self.assertFalse(dna.children[0].children[0].is_multi_choice_container)

    self.assertIsNotNone(dna.children[1].multi_choice_spec)
    self.assertTrue(dna.children[1].is_multi_choice_container)

    self.assertIsNone(dna.children[1].children[0].multi_choice_spec)
    self.assertFalse(dna.children[1].children[0].is_multi_choice_container)


class DNASpecTest(unittest.TestCase):
  """Tests for geno.DNASpec classes."""

  def testSpace(self):
    """Test geno.Space."""
    template = geno.constant()
    self.assertTrue(template.is_constant)
    template.validate(geno.DNA(None))
    self.assertEqual(template.first_dna(), geno.DNA(None))
    self.assertIsNone(template.next_dna(template.first_dna()))

    with self.assertRaisesRegex(ValueError, 'Unexpected float value'):
      template.validate(geno.DNA(1.0))

    with self.assertRaisesRegex(
        ValueError, 'Extra DNA values encountered'):
      template.validate(geno.DNA(1))

    template = geno.space([
        geno.oneof([
            geno.oneof([
                geno.constant(),
                geno.constant()
            ]),
            geno.constant()
        ]),
        geno.floatv(min_value=0.0, max_value=1.0),
    ])
    self.assertFalse(template.is_constant)
    self.assertIsNone(template.parent_spec)
    self.assertIs(template.elements[0].parent_spec, template)
    self.assertIs(template.elements[0].candidates[0].parent_spec,
                  template.elements[0])
    template.validate(geno.DNA([(0, 1), 0.5]))
    template.validate(geno.DNA([1, 0.5]))
    self.assertTrue(symbolic.eq(
        template[1], geno.floatv(min_value=0.0, max_value=1.0)))
    self.assertTrue(symbolic.eq(
        template.first_dna(), geno.DNA([(0, 0), 0.0])))

    with self.assertRaisesRegex(
        ValueError,
        'Number of child values in DNA \\(.*\\) does not match '
        'the number of elements \\(.*\\)'):
      template.validate(geno.DNA(None))

    with self.assertRaisesRegex(
        ValueError,
        'Number of child values in DNA \\(.*\\) does not match '
        'the number of elements \\(.*\\)'):
      template.validate(geno.DNA(1))

    with self.assertRaisesRegex(ValueError, 'Expect integer for Choices'):
      template.validate(geno.DNA([0, 0.5]))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      template.validate(geno.DNA([1, 0]))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than '):
      template.validate(geno.DNA([1, 1.5]))

    with self.assertRaisesRegex(
        NotImplementedError, '`next_dna` is not supported on `Float` yet'):
      template.next_dna(template.first_dna())

    # Test space_size, length and iter_dna.
    t = geno.constant()
    self.assertEqual(t.space_size, 1)
    self.assertEqual(len(t), 0)

    t = geno.space([
        geno.manyof(2, [
            geno.constant(),
            geno.constant(),
            geno.constant(),
        ], distinct=False, sorted=False),
        geno.oneof([
            geno.constant(),
            geno.constant()
        ])
    ])
    self.assertEqual(len(t), 2 + 1)
    self._assert_all_dnas(t, [
        [[0, 0], 0],
        [[0, 0], 1],
        [[0, 1], 0],
        [[0, 1], 1],
        [[0, 2], 0],
        [[0, 2], 1],
        [[1, 0], 0],
        [[1, 0], 1],
        [[1, 1], 0],
        [[1, 1], 1],
        [[1, 2], 0],
        [[1, 2], 1],
        [[2, 0], 0],
        [[2, 0], 1],
        [[2, 1], 0],
        [[2, 1], 1],
        [[2, 2], 0],
        [[2, 2], 1],
    ])
    t = geno.space([
        geno.floatv(min_value=0., max_value=1.),
        geno.oneof([geno.constant()])
    ])
    self.assertEqual(t.space_size, -1)
    self.assertEqual(len(t), 2)

    # Test space with custom decision points.
    t = geno.space([
        geno.oneof([
            geno.constant(),
            geno.custom()
        ]),
        geno.custom()
    ])
    self.assertEqual(t.space_size, -1)
    self.assertEqual(len(t), 3)

    dna = geno.DNA([(1, 'abc'), 'def'])
    t.validate(dna)

  def testChoices(self):
    """Test geno.Choices."""

    # Single choice without literal values.
    choices = geno.oneof([geno.constant(), geno.constant()], location='a.b')

    self.assertIs(choices.choice_spec(0), choices)
    self.assertEqual(choices[0], choices)
    self.assertEqual(choices[:], [choices])

    self.assertEqual(
        str(choices),
        inspect.cleandoc("""
            Choices(num_choices=1, [
              (0): Space()
              (1): Space()
            ], id='a.b')"""))

    with self.assertRaisesRegex(
        ValueError, 'Multi-choice spec cannot be a subchoice.'):
      choices = geno.Choices(
          2, [geno.constant(), geno.constant()], subchoice_index=0)

    with self.assertRaisesRegex(
        ValueError, 'The length of \'candidates\' .* should be equal to '
        'the length of \'literal_values\' .*.'):
      geno.oneof([geno.constant()], literal_values=['0/2 (0)', '1/2 (1)'])

    self.assertEqual(choices.location.keys, ['a', 'b'])
    self.assertTrue(symbolic.eq(choices.candidates[0], geno.Space(index=0)))
    choices.validate(geno.DNA(0))
    choices.validate(geno.DNA(1))
    with self.assertRaisesRegex(ValueError, 'Expect integer for Choices'):
      choices.validate(geno.DNA(1.0))

    with self.assertRaisesRegex(ValueError, 'Expect integer for Choices'):
      choices.validate(geno.DNA(None))

    with self.assertRaisesRegex(ValueError, 'Choice out of range.'):
      choices.validate(geno.DNA(2))

    with self.assertRaisesRegex(ValueError, 'Extra DNA values encountered'):
      choices.validate(geno.DNA((0, 1)))

    with self.assertRaisesRegex(ValueError, 'Expect integer for Choices'):
      choices.validate(geno.DNA([0, 1]))

    # Multiple choices with literal values.
    choices = geno.manyof(2, [
        geno.constant(), geno.constant(), geno.constant(),
    ], literal_values=[1, 2, 3], distinct=True, sorted=True)

    self.assertIs(choices[1], choices.choice_spec(1))
    self.assertEqual(choices[1:], [choices.choice_spec(1)])
    self.assertTrue(symbolic.eq(
        choices.choice_spec(0),
        choices.clone(deep=True, override={'location': object_utils.KeyPath(0),
                                           'num_choices': 1,
                                           'subchoice_index': 0})))

    self.assertEqual(choices.format_candidate(0), '0/3 (1)')
    self.assertEqual(choices.format_candidate(0, 'choice'), '0/3')
    self.assertEqual(choices.format_candidate(0, 'literal'), 1)
    with self.assertRaisesRegex(
        ValueError,
        '`display_format` must be either \'choice\', \'literal\', or '
        '\'choice_and_literal\''):
      choices.format_candidate(0, 'unsupported_format')

    self.assertEqual(choices.candidate_index(1), 0)
    self.assertEqual(choices.candidate_index('1/3 (2)'), 1)
    self.assertEqual(choices.candidate_index('2/3'), 2)

    with self.assertRaisesRegex(
        ValueError,
        'There is no candidate in .*'):
      choices.candidate_index('foo')
    with self.assertRaisesRegex(
        ValueError,
        'Candidate index out of range'):
      choices.candidate_index('3/3')
    with self.assertRaisesRegex(
        ValueError,
        'The literal value from the input .* does not match with the '
        'literal value .* from the chosen candidate.'):
      choices.candidate_index('1/3 (1)')
    with self.assertRaisesRegex(
        ValueError,
        'The value for Choice .* should be either an integer, '
        'a float or a string'):
      choices.candidate_index(ValueError())

    choices.validate(geno.DNA([0, 1]))
    self.assertEqual(
        str(choices),
        inspect.cleandoc("""
            Choices(num_choices=2, [
              (0): 1
              (1): 2
              (2): 3
            ], sorted=True)"""))

    with self.assertRaisesRegex(
        ValueError,
        'Number of DNA child values does not match the number of choices'):
      choices.validate(geno.DNA(None))

    with self.assertRaisesRegex(
        ValueError,
        'Number of DNA child values does not match the number of choices.'):
      choices.validate(geno.DNA(0))

    with self.assertRaisesRegex(ValueError, 'Choice out of range.'):
      choices.validate(geno.DNA([2, 3]))

    with self.assertRaisesRegex(
        ValueError, 'DNA child values should be distinct.'):
      choices.validate(geno.DNA([0, 0]))

    with self.assertRaisesRegex(
        ValueError, 'DNA child values should be sorted.'):
      choices.validate(geno.DNA([1, 0]))

    with self.assertRaisesRegex(ValueError, 'Choice value should be int'):
      choices.validate(geno.DNA([0.1, 1]))

    with self.assertRaisesRegex(
        ValueError, 'Expect .* choices but encountered .* sub-DNA'):
      choices.next_dna(geno.DNA([0, 1, 2]))

    with self.assertRaisesRegex(
        ValueError, 'Choice value .* is out of range'):
      choices.next_dna(geno.DNA([0, 4]))

    # Test __len__, space_size and iter_dna
    t = geno.oneof([geno.constant(), geno.constant()])
    self.assertEqual(len(t), 1)
    self._assert_all_dnas(t, [0, 1])

    t = geno.oneof([
        geno.constant(),
        geno.oneof([
            geno.constant(),
            geno.constant(),
        ])
    ])
    self.assertEqual(len(t), 2)
    self._assert_all_dnas(t, [0, (1, 0), (1, 1)])

    t = geno.oneof([
        geno.constant(),
        geno.floatv(min_value=0.0, max_value=1.0),
    ])

    self.assertEqual(len(t), 2)
    self.assertEqual(t.space_size, -1)
    self.assertEqual(t.next_dna(t.first_dna()), geno.DNA((1, 0.0)))

    t = geno.manyof(2, [
        geno.constant(),
        geno.oneof([
            geno.constant(),
            geno.constant(),
        ]),
        geno.oneof([
            geno.constant(),
            geno.constant(),
            geno.constant(),
        ])
    ], distinct=False, sorted=False)
    self.assertEqual(len(t), 2 * (1 + 2))
    self._assert_all_dnas(t, [
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

    t = geno.manyof(2, [
        geno.constant(),
        geno.oneof([
            geno.constant(),
            geno.constant(),
        ]),
        geno.oneof([
            geno.constant(),
            geno.constant(),
            geno.constant(),
        ])
    ], distinct=True, sorted=False)
    self.assertEqual(len(t), 6)
    self._assert_all_dnas(t, [
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

    t = geno.manyof(2, [
        geno.constant(),
        geno.oneof([
            geno.constant(),
            geno.constant(),
        ]),
        geno.oneof([
            geno.constant(),
            geno.constant(),
            geno.constant(),
        ])
    ], distinct=False, sorted=True)
    self.assertEqual(len(t), 6)
    self._assert_all_dnas(t, [
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

    t = geno.manyof(2, [
        geno.constant(),
        geno.oneof([geno.constant(), geno.constant()]),
        geno.oneof([geno.constant(), geno.constant(), geno.constant()])
    ], distinct=True, sorted=True)
    self.assertEqual(len(t), 6)
    self._assert_all_dnas(t, [
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

  def testFloat(self):
    """Test geno.Float."""
    float_value = geno.floatv(min_value=0.0, max_value=1.0, location='a.b')
    self.assertEqual(float_value.location.keys, ['a', 'b'])
    self.assertTrue(float_value.is_leaf)
    self.assertIsNone(float_value.scale)
    self.assertEqual(float_value.space_size, -1)

    self.assertEqual(float_value.first_dna(), geno.DNA(0.0))
    with self.assertRaisesRegex(
        NotImplementedError, '`next_dna` is not supported on `Float` yet'):
      float_value.next_dna(float_value.first_dna())

    float_value.validate(geno.DNA(0.5))
    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      float_value.validate(geno.DNA(None))
    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      float_value.validate(geno.DNA(1))
    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no less than 0.*'):
      float_value.validate(geno.DNA(-1.0))
    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than 1.*'):
      float_value.validate(geno.DNA(1.5))
    with self.assertRaisesRegex(
        ValueError, 'Float DNA should have no children'):
      float_value.validate(geno.DNA((1.0, [1, 2])))

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'min_value\' \\(.*\\) should be no greater than '
        '\'max_value\''):
      _ = geno.floatv(min_value=100.0, max_value=0.0)

    with self.assertRaisesRegex(
        ValueError, '\'min_value\' must be positive'):
      _ = geno.floatv(min_value=0.0, max_value=1.0, scale='log')

  def testCustomDecisionPoint(self):
    """Test geno.CustomDecisionPoint."""
    custom_dp = geno.CustomDecisionPoint(
        hyper_type='CustomA', hints=1, location='a.b')
    self.assertEqual(custom_dp.location.keys, ['a', 'b'])
    self.assertEqual(custom_dp.hyper_type, 'CustomA')
    self.assertEqual(custom_dp.hints, 1)
    self.assertTrue(custom_dp.is_leaf)
    self.assertEqual(len(custom_dp), 1)
    self.assertEqual(custom_dp.space_size, -1)
    self.assertIs(geno.DNA('abc').use_spec(custom_dp).spec, custom_dp)

    with self.assertRaisesRegex(
        NotImplementedError,
        '`next_dna` is not supported on `CustomDecisionPoint`'):
      custom_dp.next_dna()

    custom_dp.validate(geno.DNA('abc'))
    with self.assertRaisesRegex(
        ValueError,
        'CustomDecisionPoint expects string type DNA'):
      custom_dp.validate(geno.DNA(1))

  def testSerialization(self):
    """Test serialization of DNASpec."""
    spec = geno.space([
        geno.oneof([geno.constant(), geno.constant()], hints=1),
        geno.manyof(2, [geno.constant(), geno.constant()]),
        geno.floatv(0., 1.)
    ])
    json_dict = spec.to_json()
    json_dict['userdata'] = None
    self.assertTrue(symbolic.eq(
        symbolic.from_json(json_dict), spec))

  def testInspection(self):
    """Test inspection of DNASpecs."""
    template = geno.space([
        geno.oneof([
            geno.oneof([geno.constant(), geno.constant()]),
            geno.constant()
        ]),
        geno.floatv(min_value=0.0, max_value=1.0)
    ], location='a')

    # Test compact version, which overrides symbolic.Object.format.
    self.assertEqual(
        str(template),
        inspect.cleandoc("""Space({
                 0 = \'\': Choices(num_choices=1, [
                   (0): Space({
                     0 = \'\': Choices(num_choices=1, [
                       (0): Space()
                       (1): Space()
                     ])
                   })
                   (1): Space()
                 ])
                 1 = \'\': Float(min_value=0.0, max_value=1.0)
               })"""))

    # Test non-compact version.
    self.assertEqual(
        template.format(compact=False, verbose=True),
        symbolic.Object.format(template, compact=False, verbose=True))

  def testUserData(self):
    """Test geno.DNASpec.user_data interface."""
    float_value = geno.floatv(min_value=0.0, max_value=1.0)
    self.assertTrue(float_value.location.is_root)
    float_value.set_userdata('a', 1)
    float_value.set_userdata('b', 'foo')
    self.assertEqual(float_value.userdata['a'], 1)

    # Test clone that does not carry the userdata.
    float_value_2 = float_value.clone()
    self.assertNotIn('a', float_value_2.userdata)

  def _assert_all_dnas(self, dna_spec, expected_dna_list):
    expected_dna_list = [geno.DNA(e) for e in expected_dna_list]
    produced_dna_list = list(dna_spec.iter_dna())
    self.assertEqual(produced_dna_list, expected_dna_list)
    self.assertEqual(dna_spec.space_size, len(expected_dna_list))

  def testSubDNASpecAccess(self):
    """Test accessing child decision points."""
    t = geno.space([
        geno.oneof([
            geno.constant(), geno.constant()
        ], location='x', name='a'),
        geno.oneof([
            geno.manyof(2, [
                geno.space([
                    geno.floatv(min_value=-1.0, max_value=1.0)
                ], location='y'),
                geno.constant()
            ], location='z', name='b')
        ], location='p'),
        geno.floatv(min_value=0.0, max_value=1.0, location='q')
    ])

    # Test DNASpec.decision_ids.
    self.assertEqual(t.decision_ids, [
        'x',
        'p',
        'p[=0/1].z',
        'p[=0/1].z[0][=0/2].y',
        'p[=0/1].z[1][=0/2].y',
        'q'
    ])

    # Test DNASpec.__getitem__.
    self.assertTrue(symbolic.eq(
        t['x'],
        geno.oneof([
            geno.constant(), geno.constant()
        ], location='x', name='a')))

    self.assertTrue(symbolic.eq(
        t['p'],
        geno.oneof([
            geno.space([
                geno.manyof(2, [
                    geno.space([
                        geno.floatv(min_value=-1.0, max_value=1.0)
                    ], location='y'),
                    geno.constant()
                ], location='z', name='b')
            ])
        ], location='p')))

    self.assertTrue(symbolic.eq(
        t['p[=0/1].z'], [
            geno.Choices(num_choices=1, subchoice_index=0, candidates=[
                geno.space([
                    geno.floatv(min_value=-1.0, max_value=1.0)
                ], location='y'),
                geno.constant()
            ], location=object_utils.KeyPath(0), name='b'),
            geno.Choices(num_choices=1, subchoice_index=1, candidates=[
                geno.space([
                    geno.floatv(min_value=-1.0, max_value=1.0)
                ], location='y'),
                geno.constant()
            ], location=object_utils.KeyPath(1), name='b')
        ]))
    self.assertTrue(symbolic.eq(
        t['p[=0/1].z[0][=0/2].y'],
        geno.floatv(min_value=-1.0, max_value=1.0)))

    self.assertIsNot(t['p[=0/1].z[0][=0/2].y'], t['p[=0/1].z[1][=0/2].y'])
    self.assertTrue(symbolic.eq(
        t['p[=0/1].z[0][=0/2].y'],
        t['p[=0/1].z[1][=0/2].y']))

    # Test DNASpec.get.
    self.assertTrue(symbolic.eq(
        t.get('p[=0/1].z[0][=0/2].y'),
        geno.floatv(min_value=-1.0, max_value=1.0)))
    self.assertIsNone(t.get('xyz'))

    # Test DNASpec.decision_points
    self.assertIs(t.decision_points[0], t.elements[0])
    self.assertIs(t.decision_points[1], t.elements[1])
    self.assertIs(t.decision_points[2],
                  t.decision_points[1].candidates[0].elements[0].choice_spec(0))
    self.assertIs(t.decision_points[3],
                  t.decision_points[2].candidates[0].elements[0])
    self.assertIs(t.decision_points[4],
                  t.decision_points[1].candidates[0].elements[0].choice_spec(1))
    self.assertIs(t.decision_points[5],
                  t.decision_points[4].candidates[0].elements[0])
    self.assertIs(t.decision_points[6], t.elements[2])

    # Test DNASpec.parent_spec and DNASpec.parent_choice.
    spec = t.decision_points
    self.assertIs(spec[0].parent_spec, t)
    self.assertIsNone(spec[0].parent_choice)

    self.assertIs(spec[1].parent_spec, t)
    self.assertIsNone(spec[1].parent_choice)

    self.assertIs(spec[2].parent_spec, spec[1].candidates[0].elements[0])
    self.assertIs(spec[2].parent_choice, spec[1])

    self.assertIs(spec[3].parent_spec, spec[2].candidates[0])
    self.assertIs(spec[3].parent_choice, spec[2])

    self.assertIs(spec[4].parent_spec, spec[1].candidates[0].elements[0])
    self.assertIs(spec[4].parent_choice, spec[1])

    self.assertIs(spec[5].parent_spec, spec[4].candidates[0])
    self.assertIs(spec[5].parent_choice, spec[4])

    self.assertIs(spec[6].parent_spec, t)
    self.assertIsNone(spec[6].parent_choice)

  def testNamedDecisionPoints(self):
    """Test geno.Space.named_decision_points."""
    t = geno.space([
        geno.manyof(3, [
            geno.space([
                geno.manyof(2, [
                    geno.floatv(name='c', min_value=0.1, max_value=0.9),
                    geno.oneof([
                        geno.constant(),
                        geno.constant(),
                        geno.constant(),
                    ], name='d'),
                    geno.constant(),
                    geno.constant(),
                ], name='b'),
                geno.oneof([
                    geno.constant(),
                    geno.constant()
                ])
            ]),
            geno.constant(),
            geno.constant()
        ], distinct=False, name='a'),
        geno.floatv(name='e', min_value=0.0, max_value=1.0)
    ])
    self.assertEqual(
        list(t.named_decision_points.keys()), ['a', 'b', 'c', 'd', 'e'])

    self.assertEqual(
        t.named_decision_points['a'], t.elements[0].choice_specs)

    def _flatten(list_of_list):
      r = []
      for l in list_of_list:
        r.extend(l)
      return r

    self.assertEqual(
        t.named_decision_points['b'],
        _flatten([s.candidates[0].elements[0].choice_specs
                  for s in t.named_decision_points['a']]))

    self.assertEqual(
        t.named_decision_points['c'],
        [s.candidates[0].elements[0] for s in t.named_decision_points['b']])

    self.assertEqual(
        t.named_decision_points['d'],
        [s.candidates[1].elements[0] for s in t.named_decision_points['b']])

    self.assertIs(t.named_decision_points['e'], t.elements[1])

  def testDecisionIdUpdateUponSearchSpaceChange(self):
    """Test decision IDs."""
    t = geno.space([
        geno.oneof([
            geno.constant(),
            geno.constant()
        ], location='x')
    ])
    self.assertEqual(t.decision_ids, ['x'])
    _ = geno.oneof([t], location='y')
    self.assertEqual(t.decision_ids, ['y[=0/1].x'])

  def testLiteralValueBackwardCompatibility(self):
    """Test backward compatibility on literal value."""
    t = geno.oneof([
        geno.constant(),
        geno.constant(),
    ], literal_values=['0/2 (\'foo\nbar\')', '1/2 (1)'])
    self.assertEqual(t.literal_values, ['\'foo\nbar\'', '1'])
    self.assertEqual(t.format_candidate(0), '0/2 (\'foo\nbar\')')

  def testSpaceValidation(self):
    """Test validation on name collisions in a space."""
    # Test decision point clash on the same name.
    with self.assertRaisesRegex(
        ValueError, 'Found 2 decision point definitions clash on name'):
      geno.space([
          geno.oneof([geno.constant()], name='x'),
          geno.oneof([
              geno.space([
                  geno.floatv(0., 1., name='x')
              ])])
      ])

    # Test decision points clash between name and id.
    with self.assertRaisesRegex(
        ValueError,
        'Found 2 decision point definitions clash between name .* and id .* '):
      geno.space([
          geno.oneof([geno.constant()], name='x'),
          geno.oneof([geno.constant()], location='x')
      ])

    # Should be okay: location and name are the same for a decision point.
    geno.space([
        geno.oneof([geno.constant()], location='x', name='x'),
        geno.manyof(2, [geno.constant(), geno.constant()], location='y')
    ])


class ConditionalKeyTest(unittest.TestCase):
  """Tests for conditional key."""

  def testConditionalKey(self):
    key = geno.ConditionalKey(1, 5)
    self.assertEqual(key.index, 1)
    self.assertEqual(key.num_choices, 5)
    self.assertEqual(str(key), '=1/5')


class DNAGeneraterTest(unittest.TestCase):
  """Tests for DNAGenerator."""

  def testGeneratorBase(self):
    """Tests DNAGenerator the base class."""
    dna_spec = geno.oneof([geno.constant(), geno.constant()])

    # Test for DNAGenerator that does not need feedbacks.
    class DummyGenerator(geno.DNAGenerator):

      def _propose(self):
        return geno.DNA(0)

    a = DummyGenerator()
    a.setup(dna_spec)
    self.assertFalse(a.needs_feedback)
    self.assertFalse(a.multi_objective)
    self.assertEqual(a.num_proposals, 0)
    dna = a.propose()
    self.assertEqual(dna, geno.DNA(0))
    self.assertEqual(a.num_proposals, 1)
    self.assertEqual(a.num_feedbacks, 0)
    a.feedback(dna, 0)
    self.assertEqual(a.num_feedbacks, 1)

    # Test for DNAGenerator that optimizes a single objective.
    class DummySingleObjectiveOptimizer(geno.DNAGenerator):

      def _setup(self):
        self.max_reward = None

      def _propose(self):
        return geno.DNA(1)

      def _feedback(self, dna, reward):
        if self.max_reward is None or reward > self.max_reward:
          self.max_reward = reward

    b = DummySingleObjectiveOptimizer()
    b.setup(dna_spec)
    self.assertTrue(b.needs_feedback)
    self.assertFalse(b.multi_objective)
    self.assertEqual(b.num_proposals, 0)
    dna = b.propose()
    self.assertEqual(dna, geno.DNA(1))
    self.assertEqual(b.num_proposals, 1)
    self.assertEqual(b.num_feedbacks, 0)
    b.feedback(dna, 1.2)
    self.assertEqual(b.num_feedbacks, 1)
    self.assertEqual(b.max_reward, 1.2)

    with self.assertRaisesRegex(
        ValueError,
        '.* is single objective, but the reward .*'
        'contains multiple objectives'):
      b.feedback(dna, (0, 1))

    # Test for DNAGenerator that optimizes multiple objectives.
    class DummyMultiObjectiveOptimizer(geno.DNAGenerator):

      @property
      def multi_objective(self):
        return True

      def _setup(self):
        self.rewards = []

      def _propose(self):
        return geno.DNA(1)

      def _feedback(self, dna, reward):
        self.rewards.append(reward)

    c = DummyMultiObjectiveOptimizer()
    c.setup(dna_spec)
    self.assertTrue(c.needs_feedback)
    self.assertTrue(c.multi_objective)
    self.assertEqual(c.num_proposals, 0)
    dna = c.propose()
    self.assertEqual(dna, geno.DNA(1))
    self.assertEqual(c.num_proposals, 1)
    self.assertEqual(c.num_feedbacks, 0)
    c.feedback(dna, (0.9, 1.2))
    self.assertEqual(c.num_feedbacks, 1)
    self.assertEqual(c.rewards, [(0.9, 1.2)])
    c.feedback(dna, 1.)
    self.assertEqual(c.num_feedbacks, 2)
    self.assertEqual(c.rewards, [(0.9, 1.2), (1.,)])

  def testDNAGeneratorDecorator(self):
    """Test dna_generator decorator."""
    @geno.dna_generator
    def first_ten(dna_spec):
      dna = None
      for _ in range(10):
        dna = dna_spec.next_dna(dna)
        if dna is None:
          break
        yield dna

    algo = first_ten()    # pylint: disable=no-value-for-parameter
    dna_spec = geno.oneof([geno.constant()] * 8)
    algo.setup(dna_spec)

    dna_list = []
    with self.assertRaises(StopIteration):
      while True:
        dna_list.append(algo.propose())
    self.assertEqual(
        dna_list,
        [geno.DNA(i) for i in range(8)])

    # Test error preservation from early proposal.
    @geno.dna_generator
    def bad_generator(unused_spec):
      if True:  # pylint: disable=using-constant-test
        raise ValueError('bad initializer')
      yield geno.DNA(0)

    algo = bad_generator.partial()
    algo.setup(None)

    with self.assertRaisesRegex(
        ValueError, 'bad initializer'):
      algo.propose()

    with self.assertRaisesRegex(
        ValueError, 'Error happened earlier: bad initializer'):
      algo.propose()


class RandomTest(unittest.TestCase):
  """Test the random algorithms."""

  def testRandomGenerator(self):
    """Test the random algorithm."""
    algo = geno.Random(seed=123)
    dna_spec = geno.space([
        # Single choice.
        geno.oneof([
            geno.constant(),
            geno.constant(),
            geno.constant()
        ]),
        # Multi-choice.
        geno.manyof(3, [
            geno.manyof(4, [
                geno.constant(),
                geno.constant(),
                geno.constant()
            ], distinct=False, sorted=True),
            geno.manyof(3, [
                geno.constant(),
                geno.constant(),
                geno.constant()
            ], distinct=False, sorted=False),
            geno.manyof(3, [
                geno.constant(),
                geno.constant(),
                geno.constant()
            ], distinct=True, sorted=True)
        ], distinct=True, sorted=False),
        geno.floatv(0.0, 1.0)
    ])
    algo.setup(dna_spec)
    result = algo.propose()
    expected = geno.DNA([
        0,                       # Single choice.
        [                        # Distinct, unsorted multi-choices.
            (1, [1, 0, 0]),      # Non-distinct, unsorted multi-choices.
            (0, [1, 1, 2, 2]),   # Non-distinct, sorted multi-choices.
            (2, [0, 1, 2])       # Distinct, sorted multi-choices.
        ],
        0.1350574593038607])
    self.assertEqual(result, expected)

    # Test recover scenario when random seed is provided.
    # The recovered algorithm should produce the same sequence of random
    # examples after recovery.
    algo1 = geno.Random(seed=1)
    algo1.setup(dna_spec)
    dna_list = [algo1.propose() for _ in range(10)]
    algo2 = algo1.clone(deep=True)
    algo2.setup(dna_spec)
    algo2.recover([(dna, 0.) for dna in dna_list])
    self.assertEqual(algo1._random.getstate(), algo2._random.getstate())
    self.assertEqual(algo1.propose(), algo2.propose())

  def testRandomDNA(self):
    """Test geno.random_dna."""
    # NOTE(daiyip): since different semantics of DNASpec is already tested
    # in RandomTest, we only test the correctness of the `random_dna` workflow.
    spec = geno.oneof([geno.constant(), geno.constant(), geno.constant()])
    self.assertEqual(geno.random_dna(spec, random.Random(123)),
                     geno.DNA(0))

    with self.assertRaisesRegex(
        NotImplementedError, '`random_dna` is not supported'):
      _ = geno.random_dna(geno.custom())

  def testRandomDNAWithPreviousDNA(self):
    """Test geno.random_dna with previous DNA."""
    def custom_random_dna_fn(random_generator, previous_dna):
      del random_generator
      if previous_dna is None:
        return geno.DNA('abc')
      return geno.DNA(previous_dna.value + 'x')

    spec = geno.space([
        geno.oneof([
            geno.oneof([
                geno.custom(random_dna_fn=custom_random_dna_fn),
                geno.constant()
            ]),
            geno.floatv(0.1, 1.0),
            geno.constant()
        ]),
        geno.manyof(2, [
            geno.constant(),
            geno.constant(),
            geno.constant()
        ])
    ])
    dna = geno.random_dna(
        spec, random.Random(1),
        previous_dna=geno.DNA([(0, 0, 'xyz'), [0, 1]]))
    self.assertEqual(dna, geno.DNA([(0, 0, 'xyzx'), [1, 0]]))


@geno.dna_generator
def dummy_generator(unused_dna_spec):
  yield geno.DNA([0, 0])
  yield geno.DNA([0, 0])
  yield geno.DNA([1, 1])
  yield geno.DNA([2, 2])
  yield geno.DNA([1, 1])
  yield geno.DNA([3, 0])


@symbolic.members([
    ('num_dups', typing.Int(min_value=1))
])
class DuplicatesGenerator(geno.DNAGenerator):

  def _propose(self):
    return geno.DNA(self.num_feedbacks // self.num_dups)

  def _feedback(self, dna, reward):
    pass


class DedupingTest(unittest.TestCase):
  """Tests for Deduping generator."""

  def testDefaultHashFn(self):
    dedup = geno.Deduping(dummy_generator.partial())
    dedup.setup(None)
    self.assertEqual(
        list(iter(dedup)),
        [geno.DNA([0, 0]), geno.DNA([1, 1]), geno.DNA([2, 2]),
         geno.DNA([3, 0])])

  def testCustomHashFn(self):
    dedup = geno.Deduping(
        dummy_generator.partial(),
        hash_fn=lambda x: x.children[0].value - x.children[1].value)
    dedup.setup(None)
    self.assertEqual(
        list(iter(dedup)),
        [geno.DNA([0, 0]), geno.DNA([3, 0])])

  def testCustomNumDuplicates(self):
    dedup = geno.Deduping(
        dummy_generator.partial(),
        hash_fn=lambda x: x.children[0].value - x.children[1].value,
        max_duplicates=2)
    dedup.setup(None)
    self.assertEqual(
        list(iter(dedup)),
        [geno.DNA([0, 0]), geno.DNA([0, 0]), geno.DNA([3, 0])])

  def testGeneratorWithFeedback(self):
    dedup = geno.Deduping(DuplicatesGenerator(2))
    dedup.setup(None)
    it = iter(dedup)
    x1, f1 = next(it)
    self.assertEqual(x1, geno.DNA(0))

    # f1 is not yet called, so the hash is not in cache yet, the generator
    # will return the same `geno.DNA(0)`.
    x2, f2 = next(it)
    self.assertEqual(x2, geno.DNA(0))
    f1(0)
    f2(0)

    # Both f1, f2 are called, so the next proposal will be
    # DNA(num_feedbacks / 2), which is DNA(1)
    x3, f3 = next(it)
    self.assertEqual(x3, geno.DNA(1))

    self.assertEqual(dedup.generator.num_proposals, 3)
    self.assertEqual(dedup.generator.num_feedbacks, 2)

    # Once f3 is called, the next proposal will still be DNA(3 // 2) == DNA(1)
    # Since all subsequent call will return the same DNA, which is already
    # duplicated with x3, so StopIteration will be raised.
    f3(0)
    with self.assertRaises(StopIteration):
      _ = next(it)

    # The inner generator of dedup should have made another proposals for
    # `max_proposal_attempts` times.
    self.assertEqual(dedup.generator.num_proposals,
                     3 + dedup.max_proposal_attempts)
    self.assertEqual(dedup.generator.num_feedbacks, 3)

  def testAutoReward(self):
    dedup = geno.Deduping(DuplicatesGenerator(4),
                          hash_fn=lambda x: x.value,
                          auto_reward_fn=sum)
    dedup.setup(None)

    for i, (x, f) in enumerate(dedup):
      if i == 13:
        break
      # NOTE(daiyi): This logic will be taken care of by `pg.sample`.
      reward = x.value if 'reward' not in x.metadata else x.metadata['reward']
      f(reward)
    self.assertEqual(
        dedup._cache,
        {
            0: [0, 0, 0, 0],
            1: [1, 1, 2, 4],
            2: [2, 2, 4, 8],
            3: [3]
        })

  def testRecover(self):
    dedup = geno.Deduping(geno.Random(seed=1))
    dna_spec = geno.floatv(0.1, 0.5)
    dedup.setup(dna_spec)

    history = []
    for i, x in enumerate(dedup):
      history.append((x, None))
      if i == 10:
        break

    dedup2 = dedup.clone(deep=True)
    dedup2.setup(dna_spec)
    dedup2.recover(history)
    self.assertEqual(
        dedup2._cache, dedup._cache)
    self.assertEqual(
        dedup2.generator._random.getstate(),
        dedup.generator._random.getstate())


if __name__ == '__main__':
  unittest.main()
