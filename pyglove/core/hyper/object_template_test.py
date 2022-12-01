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
"""Tests for pyglove.hyper.ObjectTemplate."""

import unittest

from pyglove.core import geno
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper.categorical import oneof
from pyglove.core.hyper.derived import ValueReference
from pyglove.core.hyper.numerical import floatv
from pyglove.core.hyper.object_template import dna_spec
from pyglove.core.hyper.object_template import materialize
from pyglove.core.hyper.object_template import ObjectTemplate
from pyglove.core.hyper.object_template import template


class ObjectTemplateTest(unittest.TestCase):
  """Tests for pg.hyper.ObjectTemplate."""

  def test_constant_template(self):

    @symbolic.members([('x', pg_typing.Int())])
    class A(symbolic.Object):
      pass

    t = ObjectTemplate({'a': A(x=1)})
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

  def test_simple_template(self):
    v = symbolic.Dict({
        'a': oneof(candidates=[0, 2.5]),
        'b': floatv(min_value=0.0, max_value=1.0)
    })
    t = ObjectTemplate(v)
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

  def test_template_with_where_clause(self):
    @symbolic.functor()
    def foo(a, b):
      return a + b

    ssd = foo(
        a=oneof([
            oneof([0, 1]),
            2
        ]),
        b=oneof([3, 4]))

    # Test template that operates on all.
    t = template(ssd)
    self.assertEqual(t.decode(geno.DNA.parse([(0, 1), 0])), foo(a=1, b=3))
    self.assertEqual(t.encode(foo(a=0, b=4)), geno.DNA.parse([(0, 0), 1]))

    # Test template that operates on `foo.a`.
    t = template(ssd, lambda v: v.sym_path != 'b')
    self.assertEqual(t.decode(geno.DNA(1)), foo(a=2, b=oneof([3, 4])))
    self.assertEqual(t.decode(geno.DNA.parse((0, 0))),
                     foo(a=0, b=oneof([3, 4])))
    self.assertEqual(t.encode(foo(a=1, b=oneof([3, 4]))),
                     geno.DNA.parse((0, 1)))

    # Test template that operates on `foo.a.candidates[0]` (the nested oneof).
    t = template(ssd, lambda v: len(v.sym_path) == 3)
    self.assertEqual(t.decode(geno.DNA(1)),
                     foo(a=oneof([1, 2]), b=oneof([3, 4])))
    self.assertEqual(t.encode(foo(a=oneof([0, 2]),
                                  b=oneof([3, 4]))),
                     geno.DNA(0))

    # Test template that operates on `foo.b`.
    t = template(ssd, lambda v: v.sym_path == 'b')
    self.assertEqual(t.decode(geno.DNA(0)),
                     foo(a=oneof([oneof([0, 1]), 2]), b=3))

    self.assertEqual(t.encode(foo(a=oneof([oneof([0, 1]), 2]),
                                  b=4)),
                     geno.DNA(1))

  def test_template_with_derived_value(self):
    @symbolic.members([(pg_typing.StrKey(), pg_typing.Int())])
    class A(symbolic.Object):
      pass

    v = symbolic.Dict({
        'a': oneof(candidates=[0, 1]),
        'b': floatv(min_value=0.0, max_value=1.0),
        'c': ValueReference(['a']),
        'd': A(x=1)
    })
    t = ObjectTemplate(v, compute_derived=True)
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

  def test_assignment_compatibility(self):
    sd = symbolic.Dict.partial(
        value_spec=pg_typing.Dict([
            ('a', pg_typing.Dict([
                ('x', pg_typing.Int())
            ])),
            ('b', pg_typing.Int())
        ]))
    sd.a = ObjectTemplate({'x': oneof(candidates=[1, 2, 3, 4])})
    sd.a = ObjectTemplate({'x': 1})
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sd.a = ObjectTemplate({'x': 'foo'})

  def test_custom_apply(self):
    t = template(symbolic.Dict())
    self.assertIs(pg_typing.Object(ObjectTemplate).apply(t), t)
    self.assertIs(pg_typing.Dict().apply(t), t)
    with self.assertRaisesRegex(
        ValueError, 'Dict .* cannot be assigned to an incompatible field'):
      pg_typing.Int().apply(t)


class ObjectTemplateHelperTests(unittest.TestCase):
  """Tests for object template related helpers."""

  def test_dna_spec(self):
    self.assertTrue(symbolic.eq(
        dna_spec(symbolic.Dict(a=oneof([0, 1]))),
        geno.Space(elements=[
            geno.Choices(location='a', num_choices=1, candidates=[
                geno.constant(),
                geno.constant()
            ], literal_values=[0, 1])
        ])))

  def test_materialize(self):
    v = symbolic.Dict(a=oneof([1, 3]))
    # Materialize using DNA.
    self.assertEqual(
        materialize(v, geno.DNA.parse([0])),
        {'a': 1})
    # Materialize using parameter dict with use_literal_values set to False.
    self.assertEqual(
        materialize(v, {'a': '1/2'}, use_literal_values=False),
        {'a': 3})
    # Materialize using parameter dict with use_literal_values set to True.
    self.assertEqual(
        materialize(v, {'a': '1/2 (3)'}, use_literal_values=True),
        {'a': 3})

    # Bad parameters.
    with self.assertRaisesRegex(
        TypeError,
        '\'parameters\' must be a DNA or a dict of string to DNA values. '):
      materialize(v, 1)


if __name__ == '__main__':
  unittest.main()
