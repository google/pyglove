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
"""Tests for pyglove.hyper."""

import threading
import unittest

from pyglove.core import geno
from pyglove.core import hyper
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as schema


symbolic.allow_empty_field_description()
symbolic.allow_repeated_class_registration()


class ObjectTemplateTest(unittest.TestCase):
  """Test for hyper.ObjectTemplate."""

  def testConstantTemplate(self):
    """Test basics."""

    @symbolic.members([('x', schema.Int())])
    class A(symbolic.Object):
      pass

    t = hyper.ObjectTemplate({'a': A(x=1)})
    self.assertEqual(t.value, {'a': A(x=1)})
    self.assertEqual(len(t.hyper_primitives), 0)
    self.assertTrue(t.is_constant)
    self.assertTrue(symbolic.eq(t.dna_spec(), geno.Space(elements=[])))
    self.assertEqual(t.root_path, '')
    self.assertEqual(t.decode(geno.DNA(None)), {'a': A(x=1)})
    with self.assertRaisesRegex(
        ValueError, 'Encountered extra DNA value to decode'):
      t.decode(geno.DNA(0))
    self.assertEqual(t.encode({'a': A(x=1)}), geno.DNA(None))
    with self.assertRaisesRegex(
        ValueError, 'Unmatched Object type between template and input'):
      t.encode({'a': 1})

  def testSimpleTemplate(self):
    """Test simple template."""
    v = symbolic.Dict({
        'a': hyper.oneof(candidates=[0, 2.5]),
        'b': hyper.floatv(min_value=0.0, max_value=1.0)
    })
    t = hyper.ObjectTemplate(v)
    self.assertEqual(t.value, v)
    self.assertFalse(t.is_constant)
    self.assertEqual(len(t.hyper_primitives), 2)
    self.assertTrue(symbolic.eq(
        t.dna_spec(),
        geno.Space(elements=[
            geno.Choices(
                location='a',
                num_choices=1,
                candidates=[geno.constant(), geno.constant()],
                literal_values=[0, 2.5]),
            geno.Float(location='b', min_value=0.0, max_value=1.0)
        ])))

    # Test decode.
    self.assertEqual(t.decode(geno.DNA.parse([0, 0.5])), {'a': 0, 'b': 0.5})
    self.assertEqual(t.decode(geno.DNA.parse([1, 0.3])), {'a': 2.5, 'b': 0.3})

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      t.decode(geno.DNA.parse([0, 0]))

    with self.assertRaisesRegex(ValueError, 'Expect integer for OneOf'):
      t.decode(geno.DNA.parse([0.5, 0.0]))

    with self.assertRaisesRegex(
        ValueError,
        'The length of child values .* is different from the number '
        'of hyper primitives'):
      t.decode(geno.DNA.parse([0]))

    # Test encode.
    self.assertEqual(t.encode({'a': 0, 'b': 0.5}), geno.DNA.parse([0, 0.5]))

    with self.assertRaisesRegex(
        ValueError,
        'Cannot encode value: no candidates matches with the value'):
      t.encode({'a': 5, 'b': 0.5})

    # Test set_dna.
    dna = geno.DNA.parse([0, 0.5])
    t.set_dna(dna)

    # Test __call__
    self.assertEqual(t(), {'a': 0, 'b': 0.5})

    # Check after call, child DNA are properly set.
    self.assertEqual(t.dna, dna)
    self.assertEqual(t.hyper_primitives[0][1].dna, dna.children[0])
    self.assertEqual(t.hyper_primitives[1][1].dna, dna.children[1])

    t.set_dna(None)
    with self.assertRaisesRegex(
        ValueError, '\'set_dna\' should be called to set a DNA'):
      t()

  def testWhere(self):
    """Test template with where clause."""
    @symbolic.functor()
    def foo(a, b):
      return a + b

    ssd = foo(
        a=hyper.oneof([
            hyper.oneof([0, 1]),
            2
        ]),
        b=hyper.oneof([3, 4]))

    # Test template that operates on all.
    t = hyper.template(ssd)
    self.assertEqual(t.decode(geno.DNA.parse([(0, 1), 0])), foo(a=1, b=3))
    self.assertEqual(t.encode(foo(a=0, b=4)), geno.DNA.parse([(0, 0), 1]))

    # Test template that operates on `foo.a`.
    t = hyper.template(ssd, lambda v: v.sym_path != 'b')
    self.assertEqual(t.decode(geno.DNA(1)), foo(a=2, b=hyper.oneof([3, 4])))
    self.assertEqual(t.decode(geno.DNA.parse((0, 0))),
                     foo(a=0, b=hyper.oneof([3, 4])))
    self.assertEqual(t.encode(foo(a=1, b=hyper.oneof([3, 4]))),
                     geno.DNA.parse((0, 1)))

    # Test template that operates on `foo.a.candidates[0]` (the nested oneof).
    t = hyper.template(ssd, lambda v: len(v.sym_path) == 3)
    self.assertEqual(t.decode(geno.DNA(1)),
                     foo(a=hyper.oneof([1, 2]), b=hyper.oneof([3, 4])))
    self.assertEqual(t.encode(foo(a=hyper.oneof([0, 2]),
                                  b=hyper.oneof([3, 4]))),
                     geno.DNA(0))

    # Test template that operates on `foo.b`.
    t = hyper.template(ssd, lambda v: v.sym_path == 'b')
    self.assertEqual(t.decode(geno.DNA(0)),
                     foo(a=hyper.oneof([hyper.oneof([0, 1]), 2]), b=3))

    self.assertEqual(t.encode(foo(a=hyper.oneof([hyper.oneof([0, 1]), 2]),
                                  b=4)),
                     geno.DNA(1))

  def testDerived(self):
    """Test template with derived value."""

    @symbolic.members([(schema.StrKey(), schema.Int())])
    class A(symbolic.Object):
      pass

    v = symbolic.Dict({
        'a': hyper.oneof(candidates=[0, 1]),
        'b': hyper.floatv(min_value=0.0, max_value=1.0),
        'c': hyper.ValueReference(['a']),
        'd': A(x=1)
    })
    t = hyper.ObjectTemplate(v, compute_derived=True)
    self.assertEqual(t.value, v)
    self.assertFalse(t.is_constant)
    self.assertEqual(len(t.hyper_primitives), 2)
    self.assertTrue(symbolic.eq(
        t.dna_spec(),
        geno.Space(elements=[
            geno.Choices(
                location='a',
                num_choices=1,
                candidates=[geno.constant(), geno.constant()],
                literal_values=[0, 1]),
            geno.Float(location='b', min_value=0.0, max_value=1.0)
        ])))

    # Test decode.
    self.assertEqual(
        t.decode(geno.DNA.parse([0, 0.5])), {
            'a': 0,
            'b': 0.5,
            'c': 0,
            'd': A(x=1)
        })

    # Test encode.
    self.assertEqual(
        t.encode({
            'a': 0,
            'b': 0.5,
            'c': 0,
            'd': A(x=1)
        }), geno.DNA.parse([0, 0.5]))

    with self.assertRaisesRegex(
        ValueError,
        'Unmatched derived value between template and input.'):
      t.encode({'a': 0, 'b': 0.5, 'c': 1, 'd': A(x=1)})

    with self.assertRaisesRegex(
        ValueError,
        'Unmatched Object keys between template value and input value'):
      t.encode({'a': 0, 'b': 0.5, 'c': 0, 'd': A(y=1)})

  def testDropInCompatibility(self):
    """Test drop in compatibility."""
    sd = symbolic.Dict.partial(
        value_spec=schema.Dict([('a', schema.Dict([(
            'x', schema.Int())])), ('b', schema.Int())]))
    sd.a = hyper.ObjectTemplate({'x': hyper.oneof(candidates=[1, 2, 3, 4])})
    sd.a = hyper.ObjectTemplate({'x': 1})
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sd.a = hyper.ObjectTemplate({'x': 'foo'})

  def testCustomApply(self):
    """Test custom_apply to ValueSpec."""
    t = hyper.template(symbolic.Dict())
    self.assertIs(schema.Object(hyper.Template).apply(t), t)
    self.assertIs(schema.Dict().apply(t), t)
    with self.assertRaisesRegex(
        ValueError, 'Dict cannot be applied to a different spec'):
      schema.Int().apply(t)


class ManyOfTest(unittest.TestCase):
  """Test for hyper.ManyOf."""

  def testBasics(self):
    """Test basics of ManyOf."""
    with self.assertRaisesRegex(
        ValueError, '.* candidates cannot produce .* distinct choices'):
      hyper.manyof(3, [1, 2], distinct=True)

  def testDNASpec(self):
    """Test ManyOf.dna_spec()."""

    # Test simple choice list without nested encoders.
    self.assertTrue(symbolic.eq(
        hyper.manyof(
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
        hyper.oneof([
            'foo',
            {
                'a': hyper.floatv(min_value=0.0, max_value=1.0),
                'b': hyper.oneof(candidates=[1, 2, 3]),
            },
            [hyper.floatv(min_value=1.0, max_value=2.0, scale='linear'), 1.0],
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

  def testDecode(self):
    """Test ManyOf.decode()."""
    choice_list = hyper.manyof(2, [
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

    choice_list = hyper.manyof(1, [
        'foo',
        {
            'a': hyper.floatv(min_value=0.0, max_value=1.0),
            'b': hyper.oneof(candidates=[1, 2, 3]),
        },
        [hyper.floatv(min_value=1.0, max_value=2.0), 1.0],
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

  def testEncode(self):
    """Test ManyOf.encode()."""
    choice_list = hyper.manyof(1, [
        'foo',
        {
            'a': hyper.floatv(min_value=0.0, max_value=1.0),
            'b': hyper.oneof(candidates=[1, 2, 3]),
        },
        [hyper.floatv(min_value=1.0, max_value=2.0), 1.0],
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

    choice_list = hyper.manyof(2, ['a', 'b', 'c'])
    self.assertEqual(choice_list.encode(['a', 'c']), geno.DNA.parse([0, 2]))
    with self.assertRaisesRegex(
        ValueError,
        'Length of input list is different from the number of choices'):
      choice_list.encode(['a'])

  def testDropInCompatibility(self):
    """Test drop-in type compatibility."""
    sd = symbolic.Dict.partial(
        value_spec=schema.Dict([(
            'a', schema.Int()), ('b', schema.List(schema.Int(
            ))), ('c', schema.List(schema.Union(
                [schema.Str(), schema.Int()]))), ('d', schema.Any())]))
    choice_list = hyper.manyof(2, [1, 'foo'])
    sd.c = choice_list
    sd.d = choice_list

    with self.assertRaisesRegex(
        TypeError, 'Cannot bind an incompatible value spec Int\\(\\)'):
      sd.a = choice_list

    with self.assertRaisesRegex(
        TypeError,
        'Cannot bind an incompatible value spec List\\(Int\\(\\)\\)'):
      sd.b = choice_list

  def testCustomApply(self):
    """test custom_apply on value specs."""
    l = hyper.manyof(2, [1, 2, 3])
    self.assertIs(schema.Object(hyper.ManyOf).apply(l), l)
    self.assertIs(schema.List(schema.Int()).apply(l), l)
    with self.assertRaisesRegex(
        TypeError, r'Cannot bind an incompatible value spec List\(Float\(\)\)'):
      schema.List(schema.Float()).apply(l)

    class A:
      pass

    class B:
      pass

    t = hyper.oneof([B()])
    self.assertEqual(
        schema.Union([schema.Object(A), schema.Object(B)]).apply(t), t)


class OneOfTest(unittest.TestCase):
  """Tests for hyper.OneOf."""

  def testDNASpec(self):
    """Test OneOf.dna_spec()."""

    class C:
      pass

    self.assertTrue(symbolic.eq(
        hyper.oneof(candidates=[
            'foo',
            {
                'a': hyper.floatv(min_value=0.0, max_value=1.0),
                'b': hyper.oneof(candidates=[1, 2, 3]),
                'c': C()
            },
            [hyper.floatv(min_value=1.0, max_value=2.0), 1.0],
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

  def testDecode(self):
    """Test OneOf.decode()."""
    choice_value = hyper.oneof(candidates=[
        'foo',
        {
            'a': hyper.floatv(min_value=0.0, max_value=1.0),
            'b': hyper.oneof(candidates=[1, 2, 3]),
        },
        [hyper.floatv(min_value=1.0, max_value=2.0), 1.0],
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

  def testEncode(self):
    """Test OneOf.encode()."""
    choice_value = hyper.oneof(candidates=[
        'foo',
        {
            'a': hyper.floatv(min_value=0.0, max_value=1.0),
            'b': hyper.oneof(candidates=[1, 2, 3]),
        },
        [hyper.floatv(min_value=1.0, max_value=2.0), 1.0],
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

  def testDropInCompatibility(self):
    """Test drop-in type compatibility."""
    sd = symbolic.Dict.partial(
        value_spec=schema.Dict([('a', schema.Str()), (
            'b', schema.Int()), (
                'c',
                schema.Union([schema.Str(), schema.Int()])), ('d',
                                                              schema.Any())]))
    choice_value = hyper.oneof(candidates=[1, 'foo'])
    sd.c = choice_value
    sd.d = choice_value

    with self.assertRaisesRegex(
        TypeError, 'Cannot bind an incompatible value spec'):
      sd.a = choice_value

    with self.assertRaisesRegex(
        TypeError, 'Cannot bind an incompatible value spec'):
      sd.b = choice_value

  def testCustomApply(self):
    """test custom_apply on value specs."""
    o = hyper.oneof([1, 2])
    self.assertIs(schema.Object(hyper.OneOf).apply(o), o)
    self.assertIs(schema.Int().apply(o), o)
    with self.assertRaisesRegex(
        TypeError, r'Cannot bind an incompatible value spec Float\(\)'):
      schema.Float().apply(o)


class FloatTest(unittest.TestCase):
  """Test for hyper.Float."""

  def setUp(self):
    """Setup test."""
    super().setUp()
    self._float = hyper.floatv(min_value=0.0, max_value=1.0)

  def testBasics(self):
    """Test Float basics."""
    self.assertEqual(self._float.min_value, 0.0)
    self.assertEqual(self._float.max_value, 1.0)
    self.assertIsNone(self._float.scale)
    self.assertTrue(self._float.is_leaf)

    with self.assertRaisesRegex(
        ValueError, '\'min_value\' .* is greater than \'max_value\' .*'):
      hyper.floatv(min_value=1.0, max_value=0.0)

  def testScale(self):
    self.assertEqual(hyper.floatv(-1.0, 1.0, 'linear').scale, 'linear')
    with self.assertRaisesRegex(
        ValueError, '\'min_value\' must be positive'):
      hyper.floatv(-1.0, 1.0, 'log')

  def testDNASpec(self):
    """Test Float.dna_spec()."""
    self.assertTrue(symbolic.eq(
        self._float.dna_spec('a'),
        geno.Float(
            location=object_utils.KeyPath('a'),
            min_value=self._float.min_value,
            max_value=self._float.max_value)))

  def testDecode(self):
    """Test Float.decode()."""
    self.assertEqual(self._float.decode(geno.DNA(0.0)), 0.0)
    self.assertEqual(self._float.decode(geno.DNA(1.0)), 1.0)

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      self._float.decode(geno.DNA(1))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no less than'):
      self._float.decode(geno.DNA(-1.0))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than'):
      self._float.decode(geno.DNA(2.0))

  def testEncode(self):
    """Test Float.encode()."""
    self.assertEqual(self._float.encode(0.0), geno.DNA(0.0))
    self.assertEqual(self._float.encode(1.0), geno.DNA(1.0))

    with self.assertRaisesRegex(
        ValueError, 'Value should be float to be encoded'):
      self._float.encode('abc')

    with self.assertRaisesRegex(
        ValueError, 'Value should be no less than'):
      self._float.encode(-1.0)

    with self.assertRaisesRegex(
        ValueError, 'Value should be no greater than'):
      self._float.encode(2.0)

  def testDropInCompatibility(self):
    """Test drop-in type compatibility."""
    sd = symbolic.Dict.partial(
        value_spec=schema.Dict([('a', schema.Int()), ('b', schema.Float(
        )), ('c',
             schema.Union([schema.Str(), schema.Float()])), (
                 'd', schema.Any()), ('e', schema.Float(
                     max_value=0.0)), ('f', schema.Float(min_value=1.0))]))
    float_value = hyper.floatv(min_value=0.0, max_value=1.0)
    sd.b = float_value
    sd.c = float_value
    sd.d = float_value

    self.assertEqual(sd.b.sym_path, 'b')
    self.assertEqual(sd.c.sym_path, 'c')
    self.assertEqual(sd.d.sym_path, 'd')
    with self.assertRaisesRegex(
        TypeError, 'Source spec Float\\(\\) is not compatible with '
        'destination spec Int\\(\\)'):
      sd.a = float_value

    with self.assertRaisesRegex(
        ValueError,
        'Float.max_value .* should be no greater than the max value'):
      sd.e = float_value

    with self.assertRaisesRegex(
        ValueError,
        'Float.min_value .* should be no less than the min value'):
      sd.f = float_value

  def testCustomApply(self):
    """test custom_apply on value specs."""
    f = hyper.float_value(min_value=0.0, max_value=1.0)
    self.assertIs(schema.Object(hyper.Float).apply(f), f)
    self.assertIs(schema.Float().apply(f), f)
    with self.assertRaisesRegex(
        TypeError, r'Source spec Float\(\) is not compatible'):
      schema.Int().apply(f)

    with self.assertRaisesRegex(
        ValueError, r'.* should be no less than the min value'):
      schema.Float(min_value=2.0).apply(f)


class CustomHyperTest(unittest.TestCase):
  """Test for hyper.CustomHyper."""

  def setUp(self):
    """Setup test."""
    super().setUp()

    class IntSequence(hyper.CustomHyper):

      def custom_decode(self, dna):
        return [int(v) for v in dna.value.split(',')]

    class IntSequenceWithEncode(IntSequence):

      def custom_encode(self, value):
        return geno.DNA(','.join([str(v) for v in value]))

    self._int_sequence = IntSequence(hints='1,2,-3,4,5,-2,7')
    self._int_sequence_with_encode = IntSequenceWithEncode(
        hints='1,2,-3,4,5,-2,7')

  def testDNASpec(self):
    """Test CustomHyper.dna_spec()."""
    self.assertTrue(symbolic.eq(
        self._int_sequence.dna_spec('a'),
        geno.CustomDecisionPoint(
            location=object_utils.KeyPath('a'),
            hints='1,2,-3,4,5,-2,7')))

  def testDecode(self):
    """Test CustomHyper.decode()."""
    self.assertEqual(
        self._int_sequence.decode(geno.DNA('0,1,2')), [0, 1, 2])
    self.assertEqual(
        self._int_sequence.decode(geno.DNA('0')), [0])
    with self.assertRaisesRegex(
        ValueError, '.* expects string type DNA'):
      self._int_sequence.decode(geno.DNA(1))

  def testEncode(self):
    """Test CustomHyper.encode()."""
    self.assertEqual(
        self._int_sequence_with_encode.encode([0, 1, 2]),
        geno.DNA('0,1,2'))

    with self.assertRaisesRegex(
        NotImplementedError, '\'custom_encode\' is not supported by'):
      _ = self._int_sequence.encode([0, 1, 2])

  def testCooperation(self):
    """Test cooperation with pg.oneof."""
    hv = hyper.oneof([
        self._int_sequence,
        1,
        2
    ])
    self.assertEqual(hyper.materialize(hv, geno.DNA(1)), 1)
    self.assertEqual(hyper.materialize(hv, geno.DNA((0, '3,4'))), [3, 4])


class TunableValueHelpersTests(unittest.TestCase):
  """Tests for helper methods on tunable values."""

  def testDNASpec(self):
    """Test hyper.dna_spec."""
    v = symbolic.Dict(a=hyper.oneof([0, 1]))
    self.assertTrue(symbolic.eq(
        hyper.dna_spec(v),
        geno.Space(elements=[
            geno.Choices(location='a', num_choices=1, candidates=[
                geno.constant(),
                geno.constant()
            ], literal_values=[0, 1])
        ])))

  def testMaterialize(self):
    """Test hyper.materialize."""
    v = symbolic.Dict(a=hyper.oneof([1, 3]))
    # Materialize using DNA.
    self.assertEqual(
        hyper.materialize(v, geno.DNA.parse([0])),
        {'a': 1})
    # Materialize using parameter dict with use_literal_values set to False.
    self.assertEqual(
        hyper.materialize(v, {'a': '1/2'}, use_literal_values=False),
        {'a': 3})
    # Materialize using parameter dict with use_literal_values set to True.
    self.assertEqual(
        hyper.materialize(v, {'a': '1/2 (3)'}, use_literal_values=True),
        {'a': 3})

    # Bad parameters.
    with self.assertRaisesRegex(
        TypeError,
        '\'parameters\' must be a DNA or a dict of string to DNA values. '):
      hyper.materialize(v, 1)

  def testIterate(self):
    """Test hyper.iterate."""
    # Test iterate with default algorithm (Sweeping)
    v = hyper.oneof(range(100))
    examples = list(hyper.iterate(v))
    self.assertEqual(examples, list(range(100)))

    examples = list(hyper.iterate(v, 10))
    self.assertEqual(examples, list(range(10)))

    class ConstantAlgorithm(geno.DNAGenerator):
      """An algorithm that always emit a constant DNA."""

      def _on_bound(self):
        self._rewards = []

      def _propose(self):
        if len(self._rewards) == 100:
          raise StopIteration()
        return geno.DNA(0)

      def _feedback(self, dna, reward):
        self._rewards.append(reward)

      @property
      def rewards(self):
        return self._rewards

    # Test iterate with a custom algorithm.
    v = hyper.oneof([1, 3])
    algo = ConstantAlgorithm()
    examples = []
    for i, (x, feedback) in enumerate(hyper.iterate(v, 5, algo)):
      examples.append(x)
      feedback(float(i))
      self.assertEqual(feedback.dna, geno.DNA(0))
    self.assertEqual(len(examples), 5)
    self.assertEqual(examples, [1] * 5)
    self.assertEqual(algo.rewards, [float(i) for i in range(5)])

    for x, feedback in hyper.iterate(v, algorithm=algo):
      examples.append(x)
      feedback(0.)
    self.assertEqual(len(examples), 100)

    # Test iterate with dynamic evaluation.
    def foo():
      return hyper.oneof([1, 3])
    examples = []
    for x in hyper.iterate(hyper.trace(foo)):
      with x():
        examples.append(foo())
    self.assertEqual(examples, [1, 3])

    with self.assertRaisesRegex(
        ValueError, '\'hyper_value\' is a constant value'):
      next(hyper.iterate('foo', algo))

    # Test iterate on DNAGenerator that generate a no-op feedback.
    class ConstantAlgorithm2(geno.DNAGenerator):
      """An algorithm that always emit a constant DNA."""

      def propose(self):
        return geno.DNA(0)

    algo = ConstantAlgorithm2()
    examples = []
    for x, feedback in hyper.iterate(
        v, 10, algorithm=algo, force_feedback=True):
      examples.append(x)
      # No op.
      feedback(0.)
    self.assertEqual(len(examples), 10)

    # Test iterate with continuation.
    class ConstantAlgorithm3(geno.DNAGenerator):
      """An algorithm that always emit a constant DNA."""

      def setup(self, dna_spec):
        super().setup(dna_spec)
        self.num_trials = 0

      def propose(self):
        self.num_trials += 1
        return geno.DNA(0)

    algo = ConstantAlgorithm3()
    for unused_x in hyper.iterate(v, 10, algo):
      pass
    for unused_x in hyper.iterate(v, 10, algo):
      pass
    self.assertEqual(algo.num_trials, 20)
    with self.assertRaisesRegex(
        ValueError, '.* has been set up with a different DNASpec'):
      next(hyper.iterate(hyper.oneof([2, 3]), 10, algo))

  def testRandomSample(self):
    """Test hyper.random_sample."""
    self.assertEqual(
        list(hyper.random_sample(hyper.one_of([0, 1]), 3, seed=123)),
        [0, 1, 0])


class DynamicEvaluationTest(unittest.TestCase):
  """Dynamic evaluation test."""

  def testDynamicEvaluate(self):
    """Test dynamic_evaluate."""
    with self.assertRaisesRegex(
        ValueError,
        '\'evaluate_fn\' must be either None or a callable object'):
      with hyper.dynamic_evaluate(1):
        pass

    with self.assertRaisesRegex(
        ValueError,
        '\'exit_fn\' must be a callable object'):
      with hyper.dynamic_evaluate(None, exit_fn=1):
        pass

  def testDynamicEvaluatedValues(self):
    """Test dynamically evaluated values."""
    with hyper.DynamicEvaluationContext().collect():
      self.assertEqual(hyper.oneof([0, 1]), 0)
      self.assertEqual(hyper.oneof([{'x': hyper.oneof(['a', 'b'])}, 1]),
                       {'x': 'a'})
      self.assertEqual(hyper.manyof(2, [0, 1, 3]), [0, 1])
      self.assertEqual(hyper.manyof(4, [0, 1, 3], distinct=False),
                       [0, 0, 0, 0])
      self.assertEqual(hyper.permutate([0, 1, 2]), [0, 1, 2])
      self.assertEqual(hyper.floatv(0.0, 1.0), 0.0)

  def testDefineByRunPerThread(self):
    """Test DynamicEvaluationContext per-thread."""
    def thread_fun():
      context = hyper.DynamicEvaluationContext()
      with context.collect():
        hyper.oneof(range(10))

      with context.apply([3]):
        self.assertEqual(hyper.oneof(range(10)), 3)

    threads = []
    for _ in range(10):
      thread = threading.Thread(target=thread_fun)
      threads.append(thread)
      thread.start()
    for t in threads:
      t.join()

  def testDefineByRunPerProcess(self):
    """Test DynamicEvaluationContext per-process."""
    def thread_fun():
      _ = hyper.oneof(range(10))

    context = hyper.DynamicEvaluationContext(per_thread=False)
    with context.collect() as hyper_dict:
      threads = []
      for _ in range(10):
        thread = threading.Thread(target=thread_fun)
        threads.append(thread)
        thread.start()
      for t in threads:
        t.join()

    self.assertEqual(len(hyper_dict), 10)

  def testIndependentDecisions(self):
    """Test the search space of independent decisions."""
    def fun():
      x = hyper.oneof([1, 2, 3]) + 1
      y = sum(hyper.manyof(2, [2, 4, 6, 8], name='y'))
      z = hyper.floatv(min_value=1.0, max_value=2.0)
      return x + y + z

    # Test dynamic evaluation by allowing reentry (all hyper primitives will
    # be registered twice).
    context = hyper.DynamicEvaluationContext()
    with context.collect() as hyper_dict:
      result = fun()
      result = fun()

    # 1 + 1 + 2 + 4 + 1.0
    self.assertEqual(result, 9.0)
    self.assertEqual(hyper_dict, {
        'decision_0': hyper.oneof([1, 2, 3]),
        'y': hyper.manyof(2, [2, 4, 6, 8], name='y'),
        'decision_1': hyper.floatv(min_value=1.0, max_value=2.0),
        'decision_2': hyper.oneof([1, 2, 3]),
        'decision_3': hyper.floatv(min_value=1.0, max_value=2.0),
    })

    with context.apply(geno.DNA.parse(
        [1, [0, 2], 1.5, 0, 1.8])):
      # 2 + 1 + 2 + 6 + 1.5
      self.assertEqual(fun(), 12.5)
      # 1 + 1 + 2 + 6 + 1.8
      self.assertEqual(fun(), 11.8)

  def testIndependentDecisionsWithRequiringHyperName(self):
    """Test independent decisions with requiring hyper primitive name."""
    def fun():
      x = hyper.oneof([1, 2, 3], name='a') + 1
      y = sum(hyper.manyof(2, [2, 4, 6, 8], name='b'))
      z = hyper.floatv(min_value=1.0, max_value=2.0, name='c')
      return x + y + z

    # Test dynamic evaluation by disallowing reentry (all hyper primitives will
    # be registered only once).
    context = hyper.DynamicEvaluationContext(require_hyper_name=True)
    with context.collect() as hyper_dict:
      with self.assertRaisesRegex(
          ValueError, '\'name\' must be specified for hyper primitive'):
        hyper.oneof([1, 2, 3])
      result = fun()
      result = fun()

    # 1 + 1 + 2 + 4 + 1.0
    self.assertEqual(result, 9.0)
    self.assertEqual(hyper_dict, symbolic.Dict(
        a=hyper.oneof([1, 2, 3], name='a'),
        b=hyper.manyof(2, [2, 4, 6, 8], name='b'),
        c=hyper.floatv(min_value=1.0, max_value=2.0, name='c')))
    with context.apply(geno.DNA.parse([1, [0, 2], 1.5])):
      # We can call fun multiple times since decision will be bound to each
      # name just once.
      # 2 + 1 + 2 + 6 + 1.5
      self.assertEqual(fun(), 12.5)
      self.assertEqual(fun(), 12.5)
      self.assertEqual(fun(), 12.5)

  def testHierarchicalDecisions(self):
    """Test hierarchical search space."""
    def fun():
      return hyper.oneof([
          lambda: sum(hyper.manyof(2, [2, 4, 6, 8])),
          lambda: hyper.oneof([3, 7]),
          lambda: hyper.floatv(min_value=1.0, max_value=2.0),
          10]) + hyper.oneof([11, 22])

    context = hyper.DynamicEvaluationContext()
    with context.collect() as hyper_dict:
      result = fun()
    # 2 + 4 + 11
    self.assertEqual(result, 17)
    self.assertEqual(hyper_dict, {
        'decision_0': hyper.oneof([
            # NOTE(daiyip): child decisions within candidates are always in
            # form of list.
            {
                'decision_1': hyper.manyof(2, [2, 4, 6, 8]),
            },
            {
                'decision_2': hyper.oneof([3, 7])
            },
            {
                'decision_3': hyper.floatv(min_value=1.0, max_value=2.0)
            },
            10,
        ]),
        'decision_4': hyper.oneof([11, 22])
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

  def testHierarchicalDecisionsWithRequiringHyperName(self):
    """Test hierarchical search space."""
    def fun():
      return hyper.oneof([
          lambda: sum(hyper.manyof(2, [2, 4, 6, 8], name='a1')),
          lambda: hyper.oneof([3, 7], name='a2'),
          lambda: hyper.floatv(min_value=1.0, max_value=2.0, name='a3.xx'),
          10], name='a') + hyper.oneof([11, 22], name='b')

    context = hyper.DynamicEvaluationContext(require_hyper_name=True)
    with context.collect() as hyper_dict:
      result = fun()
      result = fun()

    # 2 + 4 + 11
    self.assertEqual(result, 17)
    self.assertEqual(hyper_dict, {
        'a': hyper.oneof([
            # NOTE(daiyip): child decisions within candidates are always in
            # form of list.
            {'a1': hyper.manyof(2, [2, 4, 6, 8], name='a1')},
            {'a2': hyper.oneof([3, 7], name='a2')},
            {'a3.xx': hyper.floatv(min_value=1.0, max_value=2.0, name='a3.xx')},
            10,
        ], name='a'),
        'b': hyper.oneof([11, 22], name='b')
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
        _ = hyper.oneof(range(5), name='uknown')

  def testWhereStatement(self):
    """Test `where`."""
    context = hyper.DynamicEvaluationContext(
        where=lambda x: getattr(x, 'name') != 'x')
    with context.collect():
      self.assertEqual(hyper.oneof(range(10)), 0)
      self.assertIsInstance(hyper.oneof(range(5), name='x'), hyper.OneOf)

    with context.apply([1]):
      self.assertEqual(hyper.oneof(range(10)), 1)
      self.assertIsInstance(hyper.oneof(range(5), name='x'), hyper.OneOf)

  def testTrace(self):
    """Test `trace`."""
    def fun():
      return hyper.oneof([-1, 0, 1]) * hyper.oneof([-1, 0, 3]) + 1

    self.assertEqual(
        hyper.trace(fun).hyper_dict,
        {
            'decision_0': hyper.oneof([-1, 0, 1]),
            'decision_1': hyper.oneof([-1, 0, 3])
        })

  def testCustomHyper(self):
    """Test dynamic evaluation with custom hyper."""

    class IntList(hyper.CustomHyper):

      def custom_decode(self, dna):
        return [int(x) for x in dna.value.split(':')]

      def first_dna(self):
        return geno.DNA('0:1:2:3')

    def fun():
      return sum(IntList()) + hyper.oneof([0, 1]) + hyper.floatv(-1., 1.)

    context = hyper.DynamicEvaluationContext()
    with context.collect():
      fun()

    self.assertEqual(
        context.hyper_dict,
        {
            'decision_0': IntList(),
            'decision_1': hyper.oneof([0, 1]),
            'decision_2': hyper.floatv(-1., 1.)
        })
    with context.apply(geno.DNA(['1:2:3:4', 1, 0.5])):
      self.assertEqual(fun(), 1 + 2 + 3 + 4 + 1 + 0.5)

    with self.assertRaisesRegex(
        ValueError, 'Expect string-type decision for .*'):
      with context.apply(geno.DNA([0, 1, 0.5])):
        fun()

    class IntListWithoutFirstDNA(hyper.CustomHyper):

      def custom_decode(self, dna):
        return [int(x) for x in dna.value.split(':')]

    context = hyper.DynamicEvaluationContext()
    with self.assertRaisesRegex(
        NotImplementedError,
        '.* must implement method `first_dna` to be used in '
        'dynamic evaluation mode'):
      with context.collect():
        IntListWithoutFirstDNA()

  def testExternalDNASpec(self):
    """Test dynamic evalaution with external DNASpec."""

    def fun():
      return hyper.oneof(range(5), name='x') + hyper.oneof(range(3), name='y')

    context = hyper.trace(fun, require_hyper_name=True, per_thread=True)
    self.assertFalse(context.is_external)
    self.assertIsNotNone(context.hyper_dict)

    search_space_str = symbolic.to_json_str(context.dna_spec)

    context2 = hyper.DynamicEvaluationContext(
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

  def testNestedDynamicEvaluationSimple(self):
    """Test nested dynamic evaluation context."""
    def fun():
      return sum([
          hyper.oneof([1, 2, 3], hints='ssd1'),
          hyper.oneof([4, 5], hints='ssd2'),
      ])

    context1 = hyper.DynamicEvaluationContext(
        where=lambda x: x.hints == 'ssd1')
    context2 = hyper.DynamicEvaluationContext(
        where=lambda x: x.hints == 'ssd2')
    with context1.collect():
      with context2.collect():
        self.assertEqual(fun(), 1 + 4)

    self.assertEqual(
        context1.hyper_dict, {
            'decision_0': hyper.oneof([1, 2, 3], hints='ssd1')
        })
    self.assertEqual(
        context2.hyper_dict, {
            'decision_0': hyper.oneof([4, 5], hints='ssd2')
        })
    with context1.apply(geno.DNA(2)):
      with context2.apply(geno.DNA(1)):
        self.assertEqual(fun(), 3 + 5)

  def testNestedDynamicEvaluationWithRequiredHyperName(self):
    """Test nested dynamic evaluation context with required hyper name."""
    def fun():
      return sum([
          hyper.oneof([1, 2, 3], name='x', hints='ssd1'),
          hyper.oneof([4, 5], name='y', hints='ssd2'),
      ])

    context1 = hyper.DynamicEvaluationContext(
        where=lambda x: x.hints == 'ssd1')
    context2 = hyper.DynamicEvaluationContext(
        where=lambda x: x.hints == 'ssd2')
    with context1.collect():
      with context2.collect():
        self.assertEqual(fun(), 1 + 4)

    self.assertEqual(
        context1.hyper_dict, {
            'x': hyper.oneof([1, 2, 3], name='x', hints='ssd1')
        })
    self.assertEqual(
        context2.hyper_dict, {
            'y': hyper.oneof([4, 5], name='y', hints='ssd2')
        })
    with context1.apply(geno.DNA(2)):
      with context2.apply(geno.DNA(1)):
        self.assertEqual(fun(), 3 + 5)

  def testNestedSearchSpaceInNestedDynamicEvaluationContext(self):
    """Test nested search space in nested dynamic evaluation context."""
    def fun():
      return sum([
          hyper.oneof([
              lambda: hyper.oneof([1, 2, 3], name='y', hints='ssd1'),
              lambda: hyper.oneof([4, 5, 6], name='z', hints='ssd1'),
          ], name='x', hints='ssd1'),
          hyper.oneof([7, 8], name='p', hints='ssd2'),
          hyper.oneof([9, 10], name='q', hints='ssd2'),
      ])
    context1 = hyper.DynamicEvaluationContext(
        where=lambda x: x.hints == 'ssd1')
    context2 = hyper.DynamicEvaluationContext(
        where=lambda x: x.hints == 'ssd2')
    with context1.collect():
      with context2.collect():
        self.assertEqual(fun(), 1 + 7 + 9)

    self.assertEqual(
        context1.hyper_dict, {
            'x': hyper.oneof([
                {'y': hyper.oneof([1, 2, 3], name='y', hints='ssd1')},
                {'z': hyper.oneof([4, 5, 6], name='z', hints='ssd1')},
            ], name='x', hints='ssd1')
        })
    self.assertEqual(
        context2.hyper_dict, {
            'p': hyper.oneof([7, 8], name='p', hints='ssd2'),
            'q': hyper.oneof([9, 10], name='q', hints='ssd2')
        })
    with context1.apply(geno.DNA((1, 1))):
      with context2.apply(geno.DNA([0, 1])):
        self.assertEqual(fun(), 5 + 7 + 10)

  def testNestedDynamicEvaluationWithDifferentPerThreadSetting(self):
    """Test nested dynamic evaluation context with different per-thread."""
    context1 = hyper.DynamicEvaluationContext(per_thread=True)
    context2 = hyper.DynamicEvaluationContext(per_thread=False)

    def fun():
      return hyper.oneof([1, 2, 3])

    with self.assertRaisesRegex(
        ValueError,
        'Nested dynamic evaluation contexts must be either .*'):
      with context1.collect():
        with context2.collect():
          fun()

  def testDynamicEvaluationWithManualRegistry(self):
    """Test dynamic evaluation context with manual registration."""
    context = hyper.DynamicEvaluationContext()
    self.assertEqual(
        context.add_decision_point(hyper.oneof([1, 2, 3])), 1)
    self.assertEqual(
        context.add_decision_point(hyper.oneof(['a', 'b'], name='x')), 'a')
    self.assertEqual(
        context.add_decision_point(hyper.template(1)), 1)

    with self.assertRaisesRegex(
        ValueError, 'Found different hyper primitives under the same name'):
      context.add_decision_point(hyper.oneof(['foo', 'bar'], name='x'))

    self.assertEqual(context.hyper_dict, {
        'decision_0': hyper.oneof([1, 2, 3]),
        'x': hyper.oneof(['a', 'b'], name='x'),
    })

    with self.assertRaisesRegex(
        ValueError, '`evaluate` needs to be called under the `apply` context'):
      context.evaluate(hyper.oneof([1, 2, 3]))

    with context.apply([1, 1]):
      self.assertEqual(context.evaluate(context.hyper_dict['decision_0']), 2)
      self.assertEqual(context.evaluate(context.hyper_dict['x']), 'b')


class ValueReferenceTest(unittest.TestCase):
  """Tests for hyper.ValueReference classes."""

  def testResolve(self):
    """Test ValueReference.resolve."""
    sd = symbolic.Dict({'c': [
        {
            'x': [{
                'z': 0
            }],
        },
        {
            'x': [{
                'z': 1
            }]
        },
    ]})
    sd.a = hyper.ValueReference(reference_paths=['c[0].x[0].z'])
    self.assertEqual(sd.a.resolve(), [(sd, 'c[0].x[0].z')])

    # References refer to the same relative path under different parent.
    ref = hyper.ValueReference(reference_paths=['x[0].z'])
    sd.c[0].y = ref
    sd.c[1].y = ref
    self.assertEqual(sd.c[0].y.resolve(), [(sd.c[0], 'c[0].x[0].z')])
    self.assertEqual(sd.c[1].y.resolve(), [(sd.c[1], 'c[1].x[0].z')])
    # Resolve references from this point.
    self.assertEqual(sd.c[0].y.resolve(object_utils.KeyPath(0)), (sd.c, 'c[0]'))
    self.assertEqual(sd.c[0].y.resolve('[0]'), (sd.c, 'c[0]'))
    self.assertEqual(sd.c[0].y.resolve(['[0]', '[1]']), [(sd.c, 'c[0]'),
                                                         (sd.c, 'c[1]')])

    # Bad inputs.
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'reference_path_or_paths\' must be None, a string, KeyPath '
        'object, a list of strings, or a list of KeyPath objects.'):
      sd.c[0].y.resolve([1])

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'reference_path_or_paths\' must be None, a string, KeyPath '
        'object, a list of strings, or a list of KeyPath objects.'):
      sd.c[0].y.resolve(1)

    with self.assertRaisesRegex(
        ValueError, 'Cannot resolve .*: parent not found.'):
      hyper.ValueReference(reference_paths=['x[0].z']).resolve()

  def testCall(self):
    """Test ValueReference.__call__."""

    @symbolic.members([('a', schema.Int(), 'Field a.')])
    class A(symbolic.Object):
      pass

    sd = symbolic.Dict({'c': [
        {
            'x': [{
                'z': 0
            }],
        },
        {
            'x': [{
                'z': A(a=1)
            }]
        },
    ]})
    sd.a = hyper.ValueReference(reference_paths=['c[0].x[0].z'])
    self.assertEqual(sd.a(), 0)

    # References refer to the same relative path under different parent.
    ref = hyper.ValueReference(reference_paths=['x[0]'])
    sd.c[0].y = ref
    sd.c[1].y = ref
    self.assertEqual(sd.c[0].y(), {'z': 0})
    self.assertEqual(sd.c[1].y(), {'z': A(a=1)})

    # References to another reference is not supported.
    sd.c[1].z = hyper.ValueReference(reference_paths=['y'])
    with self.assertRaisesRegex(
        ValueError,
        'Derived value .* should not reference derived values'):
      sd.c[1].z()

    sd.c[1].z = hyper.ValueReference(reference_paths=['c'])
    with self.assertRaisesRegex(
        ValueError,
        'Derived value .* should not reference derived values'):
      sd.c[1].z()

  def testSchemaCheck(self):
    """Test for schema checking on derived value."""
    sd = symbolic.Dict.partial(
        x=0,
        value_spec=schema.Dict([('x', schema.Int()), ('y', schema.Int()),
                                ('z', schema.Str())]))

    sd.y = hyper.ValueReference(['x'])
    # TODO(daiyip): Enable this test once static analysis is done
    # on derived values.
    # with self.assertRaisesRegexp(
    #     TypeError, ''):
    #   sd.z = hyper.ValueReference(['x'])

  def testBadInit(self):
    """Test bad __init__."""
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'reference_paths\' should have exact 1 item'):
      hyper.ValueReference([])

if __name__ == '__main__':
  unittest.main()
