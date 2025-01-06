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

from pyglove.core import symbolic
from pyglove.core import utils
from pyglove.core.geno.base import ConditionalKey
from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import manyof
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.custom import custom
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.space import constant
from pyglove.core.geno.space import Space


class DNATest(unittest.TestCase):
  """Tests for `pg.DNA`."""

  def _dna(self):
    spec = Space([
        oneof([
            manyof(2, [
                custom(),
                floatv(0.0, 1.0, name='z'),
                constant(),
            ], location='b', name='y'),
            constant()
        ], literal_values=[
            '0/2 (Something complex)', '1/2 (\'foo\')'
        ], location='a', name='x'),
        floatv(-1.0, 0.0, location='d', name='p'),
        # NOTE(daiyip): multiple choices NOT under conditions.
        manyof(2, [
            constant(), constant(), constant()
        ], literal_values=[
            '0/3 (\'a\')',
            '1/3 (\'b\')',
            '2/3 (\'c\')',
        ], location='e', name='q')
    ])
    return DNA([(0, [(0, 'abc'), (1, 0.5)]), -0.5, [0, 1]]).use_spec(spec)

  def test_init(self):
    dna_spec = floatv(min_value=0.0, max_value=1.0)
    dna = DNA(0.5, spec=dna_spec)
    self.assertEqual(dna.spec, dna_spec)

    # Reduant None-value ancestors. (single node)
    # should be reduced to the DNA value itself.
    self.assertEqual(
        DNA(None, [DNA(None, [DNA(None, [DNA(1)])])]), DNA(1))

    # Reduant None-value ancestors. (multiple nodes)
    # should be reduced to 2-level DNA: a None parent and children with values.
    self.assertEqual(
        DNA(None, [DNA(None, [DNA(None, [DNA(1), DNA('abc')])])]),
        DNA(None, [DNA(1), DNA('abc')]))

    # No redundant node in the DNA tree, remains the same.
    self.assertEqual(
        DNA(None, [DNA(1), DNA(None, [DNA(2), DNA(None, [DNA(3)])])]),
        DNA(None, [DNA(1), DNA(None, [DNA(2), DNA(None, [DNA(3)])])]))

    # Tests for compositional values.
    self.assertEqual(DNA([1, 2]), DNA(None, [DNA(1), DNA(2)]))
    self.assertEqual(DNA((1, 1)), DNA(1, [DNA(1)]))
    self.assertEqual(DNA((1, 2, [3, 4])), DNA(1, [DNA(2, [DNA(3), DNA(4)])]))

    self.assertEqual(
        DNA([(1, [2, (3, 4)]), 5, (6, [7, 8])]),
        DNA(None, [
            DNA(1, [DNA(2), DNA(3, [DNA(4)])]),
            DNA(5),
            DNA(6, [DNA(7), DNA(8)])
        ]))

  def test_bad_init(self):
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      DNA(ValueError())

    with self.assertRaisesRegex(
        ValueError,
        '\'children\' .* must be None when \'value\' .* is compositional.'):
      DNA([1, 2], [DNA(1)])

    with self.assertRaisesRegex(
        ValueError,
        'Tuple as conditional choices must have at least 2 items'):
      DNA((1,))

    with self.assertRaisesRegex(
        ValueError, 'Tuple as conditional choices only allow multiple '
        'choices to be used at the last position'):
      DNA([(1, [1, 2], 3)])

  def test_basics(self):
    dna = DNA(1, [DNA(2), DNA(3.0), DNA('abc')])
    self.assertEqual(dna.value, 1)
    self.assertIsNotNone(dna.children)
    self.assertEqual(len(dna.children), 3)
    self.assertIsNone(dna.spec)
    self.assertIsNone(dna.parent_dna)
    self.assertIs(dna.children[0].parent_dna, dna)
    self.assertIs(dna.root, dna)
    self.assertIs(dna.children[0].root, dna)

  def test_contains(self):
    dna = DNA(1, [DNA(2), DNA(3.0), DNA('abc')])
    self.assertIn(2, dna)
    self.assertIn(3.0, dna)
    self.assertIn('abc', dna)
    self.assertIn(DNA(2), dna)
    self.assertNotIn(0.1, dna)
    self.assertNotIn('foo', dna)
    with self.assertRaisesRegex(
        ValueError, 'DNA.__contains__ does not accept .*'):
      _ = (1, 1) in dna

  def test_iter(self):
    dna = DNA(1, [DNA(2), DNA(3.0), DNA('abc')])
    values = []
    for child in dna:
      values.append(child.value)
    self.assertEqual(values, [2, 3.0, 'abc'])

  def test_is_leaf(self):
    dna = DNA(1, [DNA(2), DNA(3.0), DNA('abc')])
    self.assertFalse(dna.is_leaf)
    self.assertTrue(dna.children[0].is_leaf)
    self.assertTrue(dna.children[1].is_leaf)
    self.assertTrue(dna.children[2].is_leaf)

  def test_methods_that_require_spec(self):
    dna = DNA(1, [DNA(2), DNA(3.0), DNA('abc')])
    with self.assertRaisesRegex(ValueError, '.* is not bound with a DNASpec'):
      dna.get('a')

    with self.assertRaisesRegex(ValueError, '.* is not bound with a DNASpec'):
      _ = dna.literal_value

    with self.assertRaisesRegex(ValueError, '.* is not bound with a DNASpec'):
      _ = dna.decision_ids

    with self.assertRaisesRegex(ValueError, '.* is not bound with a DNASpec'):
      _ = dna.to_dict()

  def test_inspection(self):
    self.assertEqual(DNA(0).to_json(type_info=False), 0)
    self.assertEqual(DNA([0, 1]).to_json(type_info=False), [0, 1])
    self.assertEqual(
        DNA([0, (0, 1)]).to_json(type_info=False), [0, (0, 1)])
    self.assertEqual(
        DNA((0, 0.5)).to_json(type_info=False), (0, 0.5))

    self.assertEqual(
        str(DNA((0, [1, 0.5, 'abc']))), 'DNA(0, [1, 0.5, \'abc\'])')
    self.assertEqual(
        DNA(0.0).use_spec(
            Space([
                floatv(0.0, 1.0, location='a')
            ])).format(as_dict=True), 'DNA({\n'
        '  \'a\': 0.0\n'
        '})')

  def test_from_fn(self):
    dna_spec = Space([
        oneof([
            manyof(2, [constant(), constant()]),
            constant()]),
        floatv(0.0, 1.0)
    ])

    stats = dict(invocation=0)
    def dna_emitter(spec):
      stats['invocation'] += 1
      return spec.first_dna()

    # `dna_emitter` should be called only twice, once on
    # the top-most Choices and once on the Float. Since both
    # of them returns a DNA, we use its child DNA directly
    # instead of stepping into the child DNASpec.
    dna = DNA.from_fn(dna_spec, dna_emitter)
    self.assertEqual(stats['invocation'], 2)
    self.assertEqual(dna, DNA([(0, [0, 1]), 0.0]))

    def zero_emitter(spec):
      if spec.is_numerical:
        return spec.min_value
      return range(spec.num_choices)

    dna = DNA.from_fn(dna_spec, zero_emitter)
    self.assertEqual(dna, DNA([(0, [0, 1]), 0.0]))

    def choice_out_of_range(spec):
      if spec.is_numerical:
        return spec.min_value
      return [len(spec.candidates)] * spec.num_choices

    with self.assertRaisesRegex(ValueError, 'Choice out of range.'):
      DNA.from_fn(dna_spec, choice_out_of_range)

    def bad_choice_value(spec):
      del spec
      return [0.1]

    with self.assertRaisesRegex(ValueError, 'Choice value should be int'):
      DNA.from_fn(dna_spec, bad_choice_value)

    def unmatched_num_choices(spec):
      del spec
      return [0]

    with self.assertRaisesRegex(
        ValueError,
        'Number of DNA child values does not match the number of choices'):
      DNA.from_fn(dna_spec, unmatched_num_choices)

    with self.assertRaisesRegex(
        TypeError, 'Argument \'dna_spec\' should be DNASpec type.'):
      DNA.from_fn(1, unmatched_num_choices)

  def test_serialization(self):
    dna = DNA([0, (1, 2), [3, (4, 5, 'abc')]])
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

  def test_decision_ids(self):
    self.assertEqual(
        self._dna().decision_ids,
        [
            'a',
            'a[=0/2].b',
            'a[=0/2].b[0][=0/3]',
            'a[=0/2].b[0][=1/3]',
            'a[=0/2].b[1][=0/3]',
            'a[=0/2].b[1][=1/3]',
            'd',
            'e'
        ])

  def test_getitem(self):
    # Get by index.
    dna = DNA(1, [DNA(2), DNA(3.0), DNA('abc')])
    self.assertEqual(dna[0], DNA(2))
    with self.assertRaisesRegex(IndexError, 'list index out of range'):
      _ = dna[3]

    # Get by name.
    dna = self._dna()
    self.assertEqual(dna['x'], DNA((0, [(0, 'abc'), (1, 0.5)])))
    self.assertEqual(dna['y'], [DNA((0, 'abc')), DNA((1, 0.5))])
    self.assertEqual(dna['z'], DNA(0.5))
    self.assertEqual(dna['p'], DNA(-0.5))
    self.assertEqual(dna['q'], [DNA(0), DNA(1)])

    # Get by id (Text).
    self.assertEqual(dna['a'], DNA((0, [(0, 'abc'), (1, 0.5)])))
    self.assertEqual(dna['a[=0/2].b'], [DNA((0, 'abc')),
                                        DNA((1, 0.5))])
    self.assertEqual(dna['a[=0/2].b[0]'], DNA((0, 'abc')))
    self.assertEqual(dna['a[=0/2].b[0][=0/3]'], DNA('abc'))
    self.assertIsNone(dna['a[=0/2].b[0][=1/3]'])
    self.assertIsNone(dna['a[=0/2].b[1][=0/3]'])
    self.assertEqual(dna['a[=0/2].b[1]'], DNA((1, 0.5)))
    self.assertEqual(dna['a[=0/2].b[1][=1/3]'], DNA(0.5))
    self.assertEqual(dna['d'], DNA(-0.5))
    self.assertEqual(dna['e'], [DNA(0), DNA(1)])
    self.assertEqual(dna['e[0]'], DNA(0))

    # Get by id (KeyPath).
    self.assertEqual(dna[dna.spec['a[=0/2].b[1][=1/3]'].id], DNA(0.5))

    # Get by DNASpec.
    self.assertEqual(dna[dna.spec['a[=0/2].b[1][=1/3]']], DNA(0.5))

  def test_get(self):
    dna = self._dna()
    self.assertEqual(dna.get('a[=0/2].b'), [DNA((0, 'abc')),
                                            DNA((1, 0.5))])
    self.assertEqual(dna.get('a[=0/2].b[1]'), DNA((1, 0.5)))
    self.assertIsNone(dna.get('a[=0/2].b[0][=1/3]'))
    self.assertIsNone(dna.get('xyz'))

  def test_literal_value(self):
    dna = self._dna()
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

  def test_is_subchoice(self):
    dna = self._dna()
    self.assertFalse(dna.is_subchoice)
    self.assertTrue(dna['y'][0].is_subchoice)
    self.assertFalse(dna['z'].is_subchoice)
    self.assertTrue(dna['q'][1].is_subchoice)

  def test_root(self):
    dna = self._dna()
    self.assertIs(dna['y'][0].root, dna)
    self.assertIs(dna['z'].root, dna)
    self.assertIs(dna['q'][0].root, dna)

  def test_hash(self):
    self.assertEqual(hash(DNA(None)), hash(DNA(None)))
    self.assertEqual(hash(DNA(1)), hash(DNA(1)))
    self.assertEqual(hash(DNA(1, [DNA(2, [DNA(3), DNA(4)]), DNA(5)])),
                     hash(DNA(1, [DNA(2, [DNA(3), DNA(4)]), DNA(5)])))

  def test_cmp(self):
    self.assertLess(DNA(None), DNA(0))
    self.assertLess(DNA(0), DNA(1))
    self.assertLess(DNA([0, 1, 2]), DNA([0, 2, 1]))
    self.assertLess(DNA((0, 'abc')), DNA((0, 'abd')))
    self.assertLess(DNA((0, 1)), DNA((0, 'abd')))
    self.assertLess(DNA([0, 1]), DNA(0))
    self.assertLess(DNA((0, 0, 0)), DNA((0, 0, 1, 0)))

    self.assertGreater(DNA(0), DNA(None))
    self.assertGreater(DNA(1), DNA(0))
    self.assertGreater(DNA([0, 2, 1]), DNA([0, 1, 2]))
    self.assertGreater(DNA((0, 'abd')), DNA((0, 'abc')))
    self.assertGreater(DNA((0, 'abd')), DNA((0, 1)))
    self.assertGreater(DNA(0), DNA([0, 1]))
    self.assertGreater(DNA((0, 0, 1, 0)), DNA((0, 0, 0)))

    with self.assertRaisesRegex(
        ValueError,
        'The two input DNA have different number of children'):
      _ = DNA((0, 0, 0)) < DNA((0, 0, 0, 0))

  def test_to_numbers(self):
    self.assertEqual(DNA(None).to_numbers(), [])
    self.assertEqual(DNA(None).to_numbers(flatten=False), [])
    self.assertEqual(DNA(1).to_numbers(), [1])
    self.assertEqual(DNA(1).to_numbers(flatten=False), 1)
    self.assertEqual(DNA(0.5).to_numbers(), [0.5])
    self.assertEqual(DNA(0.5).to_numbers(flatten=False), 0.5)
    self.assertEqual(DNA([0, 1, 2]).to_numbers(), [0, 1, 2])
    self.assertEqual(
        DNA((0, [1, 2])).to_numbers(flatten=False),
        (0, [1, 2]))
    self.assertEqual(
        DNA([0, 1, 2]).to_numbers(flatten=False),
        [0, 1, 2])
    self.assertEqual(
        DNA([(0, 1), 2, (3, [4, 0.5])]).to_numbers(),
        [0, 1, 2, 3, 4, 0.5])
    self.assertEqual(
        DNA([(0, 1), 'abc', (3, [4, 0.5])]).to_numbers(flatten=False),
        [(0, 1), 'abc', (3, [4, 0.5])])

  def test_from_numbers(self):
    spec = Space([
        manyof(2, [
            oneof([
                floatv(0.0, 1.0),
                constant()
            ]),
            constant(),
        ]),
        floatv(-1., 1.)
    ])
    self.assertEqual(
        DNA.from_numbers([0, 0, 0.1, 1, 0.2], spec),
        DNA([[(0, 0, 0.1), 1], 0.2]))

    with self.assertRaisesRegex(
        ValueError, 'Candidate index out of range at choice .*'):
      DNA.from_numbers([0, 4, 0.1, 1, 0.2], spec)

    with self.assertRaisesRegex(
        ValueError, 'The input .* is too short'):
      DNA.from_numbers([0, 0], spec)

    with self.assertRaisesRegex(
        ValueError, 'The input .* is too long'):
      DNA.from_numbers([0, 0, 0.1, 1, 0.2, 0], spec)

  def test_metadata(self):
    dna = DNA(None)
    self.assertEqual(len(dna.metadata), 0)
    self.assertEqual(dna.set_metadata('a', 1).metadata.a, 1)

    dna = DNA(None, metadata=dict(a=1))
    self.assertEqual(dna.metadata, dict(a=1))
    self.assertEqual(dna.set_metadata('a', 2).set_metadata('b', 3).metadata,
                     dict(a=2, b=3))
    # Neither metadata keys are cloneable.
    self.assertEqual(dna.clone(deep=True).metadata, {})

    dna_str = symbolic.to_json_str(dna)
    dna2 = symbolic.from_json_str(dna_str)
    self.assertEqual(dna2.metadata, dict(a=2, b=3))

    dna = DNA(None, metadata=dict(a=1))
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

  def test_userdata(self):
    dna = DNA(None)
    self.assertEqual(len(dna.userdata), 0)
    self.assertEqual(dna.set_userdata('a', 1).userdata.a, 1)
    self.assertEqual(dna.set_userdata('b', 'foo', True).userdata.b, 'foo')

    # Test clone carries over the userdata 'b'
    dna2 = dna.clone()
    self.assertNotIn('a', dna2.userdata)
    self.assertEqual(dna2.userdata.b, 'foo')

  def test_iter_dna(self):
    dna_spec = oneof([
        constant(), constant(), constant(),
        constant(), constant(), constant()])
    dna = DNA(3, spec=dna_spec)
    self.assertEqual(dna.next_dna(), DNA(4))
    self.assertEqual(list(dna.iter_dna()), [DNA(4), DNA(5)])

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      DNA(3).next_dna()

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      next(DNA(3).iter_dna())

  def test_named_decisions(self):
    t = Space([
        manyof(3, [
            Space([
                manyof(2, [
                    floatv(0.1, 0.9, name='c', location='z'),
                    oneof([
                        constant(),
                        constant(),
                        constant(),
                    ], name='d'),
                    constant(),
                    constant(),
                ], location='y', name='b'),
                oneof([
                    constant(),
                    constant()
                ], location='q')
            ]),
            constant(),
            constant()
        ], distinct=False, location='x', name='a'),
        floatv(location='m', name='e', min_value=0.0, max_value=1.0)
    ])
    dna = DNA([
        [(0, [[(0, 0.1), (1, 0)], 0]),
         (0, [[(0, 0.2), (1, 1)], 0]),
         (0, [[(0, 0.3), (1, 2)], 0])],
        0.0
    ], spec=t)
    self.assertEqual(
        list(dna.named_decisions.keys()), ['a', 'b', 'c', 'd', 'e'])
    self.assertEqual(dna.named_decisions['a'], [
        DNA((0, [[(0, 0.1), (1, 0)], 0])),
        DNA((0, [[(0, 0.2), (1, 1)], 0])),
        DNA((0, [[(0, 0.3), (1, 2)], 0])),
    ])
    self.assertEqual(dna.named_decisions['b'], [
        DNA((0, 0.1)),
        DNA((1, 0)),
        DNA((0, 0.2)),
        DNA((1, 1)),
        DNA((0, 0.3)),
        DNA((1, 2))
    ])
    self.assertEqual(dna.named_decisions['c'], [
        DNA(0.1), None, DNA(0.2), None, DNA(0.3), None
    ])
    self.assertEqual(dna.named_decisions['d'], [
        DNA(0), None, DNA(1), None, DNA(2)
    ])
    self.assertEqual([d.parent_dna for d in dna.named_decisions['d'] if d], [
        DNA((1, 0)),
        DNA((1, 1)),
        DNA((1, 2))
    ])
    self.assertEqual(dna.named_decisions['e'], DNA(0.0))

  def test_multi_choice_spec(self):
    dna_spec = manyof(2, [
        Space([
            oneof([constant(), constant()]),
            oneof([constant(), constant()])
        ]),
        manyof(2, [
            constant(),
            constant()
        ])
    ])
    dna = DNA([(0, [1, 1]), (1, [0, 1])], spec=dna_spec)
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


class DNAUseSpecTest(unittest.TestCase):
  """Tests for `pg.DNA.use_spec`."""

  def test_use_spec_on_oneof(self):
    spec = oneof([
        constant(),
        Space([
            floatv(0.0, 1.0, hints=0),
            floatv(0.0, 1.0),
            custom()
        ]),
        Space([floatv(0.0, 1.0, hints=2)]),
    ], hints=1)
    self.assertEqual(DNA(0).use_spec(spec).spec, spec)

    dna = DNA((1, [0.0, 1.0, 'abc'])).use_spec(spec)
    self.assertIs(dna.spec, spec)
    self.assertTrue(symbolic.eq(
        dna.children[0].spec,
        floatv(min_value=0.0, max_value=1.0, hints=0)))
    self.assertTrue(symbolic.eq(
        dna.children[1].spec,
        floatv(min_value=0.0, max_value=1.0)))
    self.assertTrue(symbolic.eq(
        dna.children[2].spec,
        custom()))

    dna = DNA((2, 0.0)).use_spec(spec)
    self.assertIs(dna.spec, spec)
    self.assertTrue(symbolic.eq(
        dna.children[0].spec,
        floatv(min_value=0.0, max_value=1.0, hints=2)))

    with self.assertRaisesRegex(
        ValueError, 'Argument \'spec\' must be a `pg.DNASpec` object.'):
      DNA(None).use_spec(1)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      DNA(None).use_spec(spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      DNA(1.5).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError,
        'Value of DNA is out of range according to the DNA spec'):
      DNA(3).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'There is no DNA spec for child DNA values'):
      DNA((0, 0)).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError,
        'Number of elements in child templates .* does not match '
        'with the length of children .* from DNA'):
      DNA((1, 0)).use_spec(spec)

  def test_use_spec_on_manyof(self):
    spec = manyof(2, [
        constant(),
        constant(),
        constant(),
    ], sorted=True, distinct=True, hints=0)
    dna = DNA([0, 1]).use_spec(spec)
    self.assertIs(dna.spec, spec)
    child_spec = spec.clone(
        override=dict(num_choices=1, location='[0]', subchoice_index=0))
    self.assertTrue(symbolic.eq(dna.children[0].spec, child_spec))
    with self.assertRaisesRegex(
        ValueError, 'Cannot apply multi-choice DNA spec on value'):
      DNA(1).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError,
        'Number of choices .* does not match with the number '
        'of child values'):
      DNA([0, 0, 1]).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not sorted'):
      DNA([1, 0]).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not distinct'):
      DNA([0, 0]).use_spec(spec)

  def test_use_spec_on_float(self):
    spec = floatv(min_value=0.0, max_value=1.0, hints=1)
    self.assertIs(DNA(0.5).use_spec(spec).spec, spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      DNA(None).use_spec(spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      DNA(0).use_spec(spec)
    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      DNA([0, 1]).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no less than .*'):
      DNA(-0.1).use_spec(spec)
    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than .*'):
      DNA(2.).use_spec(spec)

  def test_use_spec_on_custom_decision_point(self):
    spec = custom()
    self.assertIs(DNA('abc').use_spec(spec).spec, spec)

    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch'):
      DNA(1).use_spec(Space([custom()]))

  def test_use_spec_on_complex(self):
    spec = Space([
        oneof([
            manyof(2, [
                constant(),
                floatv(0.0, 1.0),
                constant(),
                custom()
            ], sorted=True, distinct=True, location='b'),
            floatv(0.0, 1.0, location='c'),
            constant(),
        ], location='a'),
        floatv(-1.0, 0.0, location='d')
    ])
    dna = DNA([(0, [0, (1, 0.5, 'abc')]), -0.5]).use_spec(spec)
    self.assertIsNotNone(dna.spec)

    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch.'):
      DNA(1).use_spec(spec)

    with self.assertRaisesRegex(ValueError, 'DNA value type mismatch.'):
      DNA([0.5, -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError,
        'Number of choices .* does not match with the number of '
        'child values'):
      DNA([(0, 0), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError,
        'Number of choices .* does not match with the number of '
        'child values'):
      DNA([(0, 2, 1), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError,
        'Value of DNA is out of range according to the DNA spec.'):
      DNA([4, -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Encountered more than 1 value.'):
      DNA([1, [0.0, 1.0]]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not sorted'):
      DNA([(0, [(1, 0.5), 0]), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Child values .* are not distinct'):
      DNA([(0, [(1, 0.2), (1, 0.5)]), -0.5]).use_spec(spec)

    with self.assertRaisesRegex(
        ValueError, 'Length of DNA child values .* is different from '
        'the number of elements.'):
      DNA([0, 1]).use_spec(Space([]))


class DNADictConversionTest(unittest.TestCase):
  """Tests for `pg.DNA.to_dict` and `pg.DNA.from_dict`."""

  def _dna(self):
    spec = Space([
        oneof([
            # NOTE(daiyip): multiple choices under conditions.
            manyof(3, [
                constant(),
                floatv(0.0, 1.0, name='c'),
                constant(),
                custom(name='f')
            ], literal_values=[
                123, 'Float(...)', 'xyz', 'Custom(...)'
            ], location='b', name='b'),
            constant(),
        ], literal_values=[
            'Something complex',
            '\'foo\''
        ], location='a', name='a'),
        floatv(-1.0, 0.0, location='d', name='d'),
        # NOTE(daiyip): multiple choices NOT under conditions.
        manyof(2, [
            constant(), constant(), constant()
        ], literal_values=[
            '\'a\'',
            0.333,
            '\'c\'',
        ], location='e', name='e')
    ])
    return DNA([(0, [0, (1, 0.5), (3, 'abc')]), -0.5, [0, 1]]).use_spec(spec)

  def test_id_and_value(self):
    self.assertEqual(
        self._dna().to_dict(),
        {
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

  def test_id_and_dna(self):
    self.assertEqual(
        self._dna().to_dict(value_type='dna'),
        {
            'a': DNA((0, [0, (1, 0.5), (3, 'abc')])),
            'a[=0/2].b[0]': DNA(0),
            'a[=0/2].b[1]': DNA((1, 0.5)),
            'a[=0/2].b[1][=1/4]': DNA(0.5),
            'a[=0/2].b[2]': DNA((3, 'abc')),
            'a[=0/2].b[2][=3/4]': DNA('abc'),
            'd': DNA(-0.5),
            'e[0]': DNA(0),
            'e[1]': DNA(1),
        })

  def test_id_and_choice(self):
    self.assertEqual(
        self._dna().to_dict(value_type='choice'),
        {
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

  def test_id_and_literal(self):
    self.assertEqual(
        self._dna().to_dict(value_type='literal'),
        {
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

  def test_id_and_choice_literal(self):
    self.assertEqual(
        self._dna().to_dict(value_type='choice_and_literal'),
        {
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

  def test_name_or_id_and_value(self):
    self.assertEqual(
        self._dna().to_dict(key_type='name_or_id'),
        {
            'a': 0,
            'b': [0, 1, 3],
            'c': 0.5,
            'd': -0.5,
            'e': [0, 1],
            'f': 'abc'
        })

  def test_name_or_id_and_dna(self):
    self.assertEqual(
        self._dna().to_dict(key_type='name_or_id', value_type='dna'),
        {
            'a': DNA((0, [0, (1, 0.5), (3, 'abc')])),
            'b': [DNA(0), DNA((1, 0.5)), DNA((3, 'abc'))],
            'c': DNA(0.5),
            'd': DNA(-0.5),
            'e': [DNA(0), DNA(1)],
            'f': DNA('abc'),
        })

  def test_name_or_id_and_choice(self):
    self.assertEqual(
        self._dna().to_dict(key_type='name_or_id', value_type='choice'),
        {
            'a': '0/2',
            'b': ['0/4', '1/4', '3/4'],
            'c': 0.5,
            'd': -0.5,
            'e': ['0/3', '1/3'],
            'f': 'abc'
        })

  def test_name_or_id_and_literal(self):
    self.assertEqual(
        self._dna().to_dict(key_type='name_or_id', value_type='literal'),
        {
            'a': 'Something complex',
            'b': [123, 'Float(...)', 'Custom(...)'],
            'c': 0.5,
            'd': -0.5,
            'e': ['\'a\'', 0.333],
            'f': 'abc'
        })

  def test_name_or_id_and_choice_literal(self):
    self.assertEqual(
        self._dna().to_dict(
            key_type='name_or_id',
            value_type='choice_and_literal'),
        {
            'a': '0/2 (Something complex)',
            'b': ['0/4 (123)', '1/4 (Float(...))', '3/4 (Custom(...))'],
            'c': 0.5,
            'd': -0.5,
            'e': ['0/3 (\'a\')', '1/3 (0.333)'],
            'f': 'abc'
        })

  def test_dna_spec_and_value(self):
    dna = self._dna()
    named = dna.spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec'),
        {
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

  def test_dna_spec_and_dna(self):
    dna = self._dna()
    named = dna.spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='dna'),
        {
            named['a']: DNA((0, [0, (1, 0.5), (3, 'abc')])),
            named['b'][0]: DNA(0),
            named['b'][1]: DNA((1, 0.5)),
            named['b'][2]: DNA((3, 'abc')),
            named['c'][1]: DNA(0.5),
            named['d']: DNA(-0.5),
            named['e'][0]: DNA(0),
            named['e'][1]: DNA(1),
            named['f'][2]: DNA('abc')
        })

  def test_dna_spec_and_choice(self):
    dna = self._dna()
    named = dna.spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='choice'),
        {
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

  def test_dna_spec_and_literal(self):
    dna = self._dna()
    named = dna.spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='literal'),
        {
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

  def test_dna_spec_and_choice_literal(self):
    dna = self._dna()
    named = dna.spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='choice_and_literal'),
        {
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

  def test_collapse_multi_choices(self):
    self.assertEqual(
        self._dna().to_dict(multi_choice_key='parent'),
        {
            'a': 0,
            'a[=0/2].b': [0, 1, 3],
            'a[=0/2].b[1][=1/4]': 0.5,
            'a[=0/2].b[2][=3/4]': 'abc',
            'd': -0.5,
            'e': [0, 1]
        })

  def test_keep_both_parent_and_subchoices(self):
    dna = self._dna()
    named = dna.spec.named_decision_points
    self.assertEqual(
        dna.to_dict(key_type='dna_spec', value_type='dna',
                    multi_choice_key='both'),
        {
            named['a']: DNA((0, [0, (1, 0.5), (3, 'abc')])),
            named['b'][0].parent_spec: [DNA(0),
                                        DNA((1, 0.5)),
                                        DNA((3, 'abc'))],
            named['b'][0]: DNA(0),
            named['b'][1]: DNA((1, 0.5)),
            named['b'][2]: DNA((3, 'abc')),
            named['c'][1]: DNA(0.5),
            named['d']: DNA(-0.5),
            named['e'][0].parent_spec: [DNA(0), DNA(1)],
            named['e'][0]: DNA(0),
            named['e'][1]: DNA(1),
            named['f'][2]: DNA('abc'),
        })

  def test_include_inactive_decisions(self):
    self.assertEqual(
        self._dna().to_dict(include_inactive_decisions=True),
        {
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

  def test_include_inactive_decisions_and_collapse_multi_choices(self):
    self.assertEqual(
        self._dna().to_dict(
            value_type='dna',
            multi_choice_key='parent',
            include_inactive_decisions=True),
        {
            'a': DNA((0, [0, (1, 0.5), (3, 'abc')])),
            'a[=0/2].b': [DNA(0),
                          DNA((1, 0.5)),
                          DNA((3, 'abc'))],
            'a[=0/2].b[0][=1/4]': None,
            'a[=0/2].b[0][=3/4]': None,
            'a[=0/2].b[1][=1/4]': DNA(0.5),
            'a[=0/2].b[1][=3/4]': None,
            'a[=0/2].b[2][=1/4]': None,
            'a[=0/2].b[2][=3/4]': DNA('abc'),
            'd': DNA(-0.5),
            'e': [DNA(0), DNA(1)],
        })

  def test_from_dict(self):
    dna = self._dna()
    spec = dna.spec
    for kt in ['id', 'name_or_id', 'dna_spec']:
      for vt in ['value', 'dna', 'choice', 'literal', 'choice_and_literal']:
        for mk in ['subchoice', 'parent', 'both']:
          for inactive in [True, False]:
            self.assertEqual(
                DNA.from_dict(
                    dna.to_dict(key_type=kt, value_type=vt,
                                multi_choice_key=mk,
                                include_inactive_decisions=inactive),
                    spec, use_ints_as_literals=(vt == 'literal')),
                dna)

  def test_from_dict_with_filter_fn(self):
    self.assertEqual(
        self._dna().to_dict(
            value_type='dna',
            multi_choice_key='parent',
            include_inactive_decisions=True,
            filter_fn=lambda x: x.is_custom_decision_point),
        {
            'a[=0/2].b[0][=3/4]': None,
            'a[=0/2].b[1][=3/4]': None,
            'a[=0/2].b[2][=3/4]': DNA('abc'),
        })

  def test_bad_to_dict(self):
    dna = self._dna()
    with self.assertRaisesRegex(
        ValueError, '\'key_type\' must be either .*'):
      dna.to_dict(key_type='foo')

    with self.assertRaisesRegex(
        ValueError, '\'value_type\' must be either .*'):
      dna.to_dict(value_type='foo')

    with self.assertRaisesRegex(
        ValueError, '\'multi_choice_key\' must be either .*'):
      dna.to_dict(multi_choice_key='foo')

    with self.assertRaisesRegex(
        ValueError, '.* is not bound with a DNASpec'):
      DNA(0).to_dict()

  def test_bad_from_dict(self):
    # Categorical.
    spec = Space([oneof([constant(), constant()], location='a')])
    with self.assertRaisesRegex(ValueError, 'Candidate index out of range'):
      DNA.from_dict({'a': '2/2'}, spec)

    # Numerical.
    spec = Space([floatv(location='a', min_value=0.0, max_value=1.0)])
    with self.assertRaisesRegex(
        ValueError, 'Value for .* is not found in the dictionary .*'):
      DNA.from_dict({'b': '0/3'}, spec)
    with self.assertRaisesRegex(
        ValueError, 'The decision for .* should be no less than .*'):
      DNA.from_dict({'a': -1.0}, spec)
    with self.assertRaisesRegex(
        ValueError, 'The decision for .* should be no greater than .*'):
      DNA.from_dict({'a': 2.0}, spec)

    # Custom decision point.
    custom_spec = Space([custom(location='a')])
    with self.assertRaisesRegex(
        ValueError, 'The decision for .* should be a string'):
      DNA.from_dict({'a': 1}, custom_spec)

    # Complex case:
    spec = self._dna().spec
    with self.assertRaisesRegex(
        ValueError, 'Value for .* is not found in the dictionary .*'):
      DNA.from_parameters({'x': 1}, spec)
    with self.assertRaisesRegex(
        ValueError, 'There is no candidate in .*'):
      DNA.from_parameters({'a': 'foo'}, spec)
    with self.assertRaisesRegex(
        ValueError,
        'Number of candidates .* for Choice .* does not match with DNASpec'):
      DNA.from_parameters({'a': '0/3'}, spec)

  def test_from_parameters(self):
    dna = self._dna()
    spec = dna.spec

    dna2 = DNA.from_parameters(dna.parameters(), spec)
    self.assertEqual(dna, dna2)

    dna3 = DNA.from_parameters(
        dna.parameters(use_literal_values=True), spec, use_literal_values=True)
    self.assertEqual(dna, dna3)


class ConditionalKeyTest(unittest.TestCase):
  """Tests for ConditionalKey."""

  def test_basics(self):
    key = ConditionalKey(1, 5)
    self.assertEqual(key.index, 1)
    self.assertEqual(key.num_choices, 5)

  def test_to_str(self):
    key = utils.KeyPath(['a', ConditionalKey(1, 5), 'b'])
    self.assertEqual(str(key), 'a[=1/5].b')


if __name__ == '__main__':
  unittest.main()
