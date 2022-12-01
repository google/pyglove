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
"""Tests for pyglove.hyper.Choices."""

import unittest
from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper.categorical import manyof
from pyglove.core.hyper.categorical import ManyOf
from pyglove.core.hyper.categorical import oneof
from pyglove.core.hyper.categorical import OneOf
from pyglove.core.hyper.numerical import floatv


class OneOfTest(unittest.TestCase):
  """Tests for pg.oneof."""

  def test_dna_spec(self):

    class C:
      pass

    self.assertTrue(symbolic.eq(
        oneof(candidates=[
            'foo',
            {
                'a': floatv(min_value=0.0, max_value=1.0),
                'b': oneof(candidates=[1, 2, 3]),
                'c': C()
            },
            [floatv(min_value=1.0, max_value=2.0), 1.0],
        ]).dna_spec('a.b'),
        geno.Choices(
            num_choices=1,
            candidates=[
                geno.constant(),
                geno.Space(elements=[
                    geno.Float(min_value=0.0, max_value=1.0, location='a'),
                    geno.Choices(
                        num_choices=1,
                        candidates=[
                            geno.constant(),
                            geno.constant(),
                            geno.constant()
                        ],
                        literal_values=[1, 2, 3],
                        location='b'),
                ]),
                geno.Space(elements=[
                    geno.Float(min_value=1.0, max_value=2.0, location='[0]')
                ])
            ],
            literal_values=[
                '\'foo\'',
                ('{a=Float(min_value=0.0, max_value=1.0), '
                 'b=OneOf(candidates=[0: 1, 1: 2, 2: 3]), '
                 'c=C(...)}'),
                '[0: Float(min_value=1.0, max_value=2.0), 1: 1.0]',
            ],
            location='a.b')))

  def test_decode(self):
    choice_value = oneof(candidates=[
        'foo',
        {
            'a': floatv(min_value=0.0, max_value=1.0),
            'b': oneof(candidates=[1, 2, 3]),
        },
        [floatv(min_value=1.0, max_value=2.0), 1.0],
    ])

    self.assertEqual(choice_value.decode(geno.DNA.parse(0)), 'foo')

    self.assertEqual(
        choice_value.decode(geno.DNA.parse((1, [0.5, 0]))), {
            'a': 0.5,
            'b': 1
        })

    self.assertEqual(choice_value.decode(geno.DNA.parse((2, 1.5))), [1.5, 1.0])

    with self.assertRaisesRegex(ValueError, 'Choice out of range'):
      choice_value.decode(geno.DNA.parse(5))

    with self.assertRaisesRegex(
        ValueError, 'Encountered extra DNA value to decode'):
      choice_value.decode(geno.DNA.parse((0, 1)))

    with self.assertRaisesRegex(
        ValueError,
        'The length of child values .* is different from the number '
        'of hyper primitives'):
      choice_value.decode(geno.DNA.parse((1, 0)))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      choice_value.decode(geno.DNA.parse((1, [1, 0])))

    with self.assertRaisesRegex(
        ValueError,
        'The length of child values .* is different from the number '
        'of hyper primitives'):
      choice_value.decode(geno.DNA.parse((1, [0.5, 1, 2])))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      choice_value.decode(geno.DNA.parse(2))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than'):
      choice_value.decode(geno.DNA.parse((2, 5.0)))

  def test_encode(self):
    choice_value = oneof(candidates=[
        'foo',
        {
            'a': floatv(min_value=0.0, max_value=1.0),
            'b': oneof(candidates=[1, 2, 3]),
        },
        [floatv(min_value=1.0, max_value=2.0), 1.0],
    ])
    self.assertEqual(choice_value.encode('foo'), geno.DNA(0))
    self.assertEqual(
        choice_value.encode({
            'a': 0.5,
            'b': 1
        }), geno.DNA.parse((1, [0.5, 0])))
    self.assertEqual(choice_value.encode([1.5, 1.0]), geno.DNA.parse((2, 1.5)))

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      choice_value.encode(['bar'])

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      print(choice_value.encode({'a': 0.5}))

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      choice_value.encode({'a': 1.8, 'b': 1})

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      choice_value.encode([1.0])

  def test_assignment_compatibility(self):
    sd = symbolic.Dict.partial(
        value_spec=pg_typing.Dict([
            ('a', pg_typing.Str()),
            ('b', pg_typing.Int()),
            ('c', pg_typing.Union([pg_typing.Str(), pg_typing.Int()])),
            ('d', pg_typing.Any())
        ]))
    choice_value = oneof(candidates=[1, 'foo'])
    sd.c = choice_value
    sd.d = choice_value

    with self.assertRaisesRegex(
        TypeError, 'Cannot bind an incompatible value spec'):
      sd.a = choice_value

    with self.assertRaisesRegex(
        TypeError, 'Cannot bind an incompatible value spec'):
      sd.b = choice_value

  def test_custom_apply(self):
    o = oneof([1, 2])
    self.assertIs(pg_typing.Object(OneOf).apply(o), o)
    self.assertIs(pg_typing.Int().apply(o), o)
    with self.assertRaisesRegex(
        TypeError, r'Cannot bind an incompatible value spec Float\(\)'):
      pg_typing.Float().apply(o)


class ManyOfTest(unittest.TestCase):
  """Test for pg.manyof."""

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError, '.* candidates cannot produce .* distinct choices'):
      manyof(3, [1, 2], distinct=True)

  def test_dna_spec(self):
    # Test simple choice list without nested encoders.
    self.assertTrue(symbolic.eq(
        manyof(
            2, ['foo', 1, 2, 'bar'], sorted=True, distinct=True).dna_spec(),
        geno.manyof(2, [
            geno.constant(),
            geno.constant(),
            geno.constant(),
            geno.constant()
        ], literal_values=[
            '\'foo\'', 1, 2, '\'bar\''
        ], sorted=True, distinct=True)))

    # Test complex choice list with nested encoders.
    self.assertTrue(symbolic.eq(
        oneof([
            'foo',
            {
                'a': floatv(min_value=0.0, max_value=1.0),
                'b': oneof(candidates=[1, 2, 3]),
            },
            [floatv(min_value=1.0, max_value=2.0, scale='linear'), 1.0],
        ]).dna_spec('a.b'),
        geno.oneof([
            geno.constant(),
            geno.space([
                geno.floatv(min_value=0.0, max_value=1.0, location='a'),
                geno.oneof([
                    geno.constant(),
                    geno.constant(),
                    geno.constant()
                ], literal_values=[1, 2, 3], location='b')
            ]),
            geno.floatv(1.0, 2.0, scale='linear', location='[0]')
        ], literal_values=[
            '\'foo\'',
            ('{a=Float(min_value=0.0, max_value=1.0), '
             'b=OneOf(candidates=[0: 1, 1: 2, 2: 3])}'),
            '[0: Float(min_value=1.0, max_value=2.0, scale=\'linear\'), 1: 1.0]',
        ], location='a.b')))

  def test_decode(self):
    choice_list = manyof(2, [
        'foo', 1, 2, 'bar'
    ], choices_sorted=True, choices_distinct=True)
    self.assertTrue(choice_list.is_leaf)
    self.assertEqual(choice_list.decode(geno.DNA.parse([0, 1])), ['foo', 1])

    with self.assertRaisesRegex(
        ValueError,
        'Number of DNA child values does not match the number of choices'):
      choice_list.decode(geno.DNA.parse([1, 0, 0]))

    with self.assertRaisesRegex(ValueError, 'Choice value should be int'):
      choice_list.decode(geno.DNA.parse([0, 0.1]))

    with self.assertRaisesRegex(ValueError, 'Choice out of range'):
      choice_list.decode(geno.DNA.parse([0, 5]))

    with self.assertRaisesRegex(
        ValueError, 'DNA child values should be sorted'):
      choice_list.decode(geno.DNA.parse([1, 0]))

    with self.assertRaisesRegex(
        ValueError, 'DNA child values should be distinct'):
      choice_list.decode(geno.DNA.parse([0, 0]))

    choice_list = manyof(1, [
        'foo',
        {
            'a': floatv(min_value=0.0, max_value=1.0),
            'b': oneof(candidates=[1, 2, 3]),
        },
        [floatv(min_value=1.0, max_value=2.0), 1.0],
    ])
    self.assertFalse(choice_list.is_leaf)
    self.assertEqual(choice_list.decode(geno.DNA.parse(0)), ['foo'])

    self.assertEqual(
        choice_list.decode(geno.DNA.parse((1, [0.5, 0]))), [{
            'a': 0.5,
            'b': 1
        }])

    self.assertEqual(choice_list.decode(geno.DNA.parse((2, 1.5))), [[1.5, 1.0]])

    with self.assertRaisesRegex(ValueError, 'Choice out of range'):
      choice_list.decode(geno.DNA.parse(5))

    with self.assertRaisesRegex(
        ValueError, 'Encountered extra DNA value to decode'):
      choice_list.decode(geno.DNA.parse((0, 1)))

    with self.assertRaisesRegex(
        ValueError,
        'The length of child values .* is different from the number '
        'of hyper primitives'):
      choice_list.decode(geno.DNA.parse((1, 0)))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      choice_list.decode(geno.DNA.parse((1, [1, 0])))

    with self.assertRaisesRegex(
        ValueError,
        'The length of child values .* is different from the number '
        'of hyper primitives'):
      choice_list.decode(geno.DNA.parse((1, [0.5, 1, 2])))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      choice_list.decode(geno.DNA.parse(2))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than'):
      choice_list.decode(geno.DNA.parse((2, 5.0)))

  def test_encode(self):
    choice_list = manyof(1, [
        'foo',
        {
            'a': floatv(min_value=0.0, max_value=1.0),
            'b': oneof(candidates=[1, 2, 3]),
        },
        [floatv(min_value=1.0, max_value=2.0), 1.0],
    ])
    self.assertEqual(choice_list.encode(['foo']), geno.DNA(0))
    self.assertEqual(
        choice_list.encode([{
            'a': 0.5,
            'b': 1
        }]), geno.DNA.parse((1, [0.5, 0])))
    self.assertEqual(choice_list.encode([[1.5, 1.0]]), geno.DNA.parse((2, 1.5)))

    with self.assertRaisesRegex(
        ValueError, 'Cannot encode value: value should be a list type'):
      choice_list.encode('bar')

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      choice_list.encode(['bar'])

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      print(choice_list.encode([{'a': 0.5}]))

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      choice_list.encode([{'a': 1.8, 'b': 1}])

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      choice_list.encode([[1.0]])

    choice_list = manyof(2, ['a', 'b', 'c'])
    self.assertEqual(choice_list.encode(['a', 'c']), geno.DNA.parse([0, 2]))
    with self.assertRaisesRegex(
        ValueError,
        'Length of input list is different from the number of choices'):
      choice_list.encode(['a'])

  def test_assignment_compatibility(self):
    """Test drop-in type compatibility."""
    sd = symbolic.Dict.partial(
        value_spec=pg_typing.Dict([
            ('a', pg_typing.Int()),
            ('b', pg_typing.List(pg_typing.Int())),
            ('c', pg_typing.List(pg_typing.Union(
                [pg_typing.Str(), pg_typing.Int()]))),
            ('d', pg_typing.Any())
        ]))
    choice_list = manyof(2, [1, 'foo'])
    sd.c = choice_list
    sd.d = choice_list

    with self.assertRaisesRegex(
        TypeError, 'Cannot bind an incompatible value spec Int\\(\\)'):
      sd.a = choice_list

    with self.assertRaisesRegex(
        TypeError,
        'Cannot bind an incompatible value spec List\\(Int\\(\\)\\)'):
      sd.b = choice_list

  def test_custom_apply(self):
    l = manyof(2, [1, 2, 3])
    self.assertIs(pg_typing.Object(ManyOf).apply(l), l)
    self.assertIs(pg_typing.List(pg_typing.Int()).apply(l), l)
    with self.assertRaisesRegex(
        TypeError, r'Cannot bind an incompatible value spec List\(Float\(\)\)'):
      pg_typing.List(pg_typing.Float()).apply(l)

    class A:
      pass

    class B:
      pass

    t = oneof([B()])
    self.assertEqual(
        pg_typing.Union([pg_typing.Object(A), pg_typing.Object(B)]).apply(t), t)


if __name__ == '__main__':
  unittest.main()
