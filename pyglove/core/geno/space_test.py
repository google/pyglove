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
"""Tests for pyglove.geno.Space."""

import inspect
import random
import unittest

from pyglove.core import symbolic
from pyglove.core.geno.base import DNA
from pyglove.core.geno.categorical import manyof
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.custom import custom
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.space import constant
from pyglove.core.geno.space import Space


class SpaceTest(unittest.TestCase):
  """Tests for `pg.geno.Space`."""

  def test_init(self):
    _ = Space([
        oneof([constant(), constant()]),
        manyof(2, [constant(), constant()]),
        floatv(0.1, 1.0),
        custom()
    ])

    # Okay: location and name are the same for a decision point.
    _ = Space([
        oneof([constant()], location='x', name='x'),
        manyof(2, [constant(), constant()], location='y')
    ])

    # Test decision point clash on the same name.
    with self.assertRaisesRegex(
        ValueError, 'Found 2 decision point definitions clash on name'):
      _ = Space([
          oneof([constant()], name='x'),
          oneof([
              Space([
                  floatv(0., 1., name='x')
              ])])
      ])

    # Test decision points clash between name and id.
    with self.assertRaisesRegex(
        ValueError,
        'Found 2 decision point definitions clash between name .* and id .* '):
      _ = Space([
          oneof([constant()], name='x'),
          oneof([constant()], location='x')
      ])

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      _ = Space([constant()])

  def test_basics(self):
    space = constant()
    self.assertTrue(space.is_space)
    self.assertFalse(space.is_categorical)
    self.assertFalse(space.is_subchoice)
    self.assertFalse(space.is_numerical)
    self.assertFalse(space.is_custom_decision_point)

  def test_is_constant(self):
    self.assertTrue(constant().is_constant)
    self.assertFalse(Space([oneof([constant(), constant()])]).is_constant)
    self.assertFalse(Space([floatv(0.1, 0.2)]).is_constant)
    self.assertFalse(Space([custom()]).is_constant)

  def test_space_size(self):
    self.assertEqual(constant().space_size, 1)
    a = oneof([constant(), constant()])
    b = manyof(2, [constant(), constant(), constant()])
    c = floatv(-1.0, 1.0)
    d = custom()
    self.assertEqual(Space([a]).space_size, a.space_size)
    self.assertEqual(Space([b]).space_size, b.space_size)
    self.assertEqual(Space([c]).space_size, c.space_size)
    self.assertEqual(Space([d]).space_size, d.space_size)
    self.assertEqual(Space([a, b]).space_size, a.space_size * b.space_size)
    self.assertEqual(Space([a, b, c]).space_size, -1)
    self.assertEqual(Space([a, b, d]).space_size, -1)

  def test_len(self):
    self.assertEqual(len(constant()), 0)
    self.assertEqual(len(Space([
        oneof([
            oneof([constant(), constant()]),
            floatv(-1.0, 1.0)
        ]),
        manyof(2, [constant(), constant()]),
        floatv(-1.0, 1.0)
    ])), 6)

  def test_id(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()])
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    b = manyof(2, [constant(), constant()], location='b')
    c = floatv(-1.0, 1.0, location='c')
    d = custom(location='d')
    space = Space([a, b, c, d], location='root')

    self.assertEqual(space.id, 'root')
    self.assertEqual(a.id, 'root.a')
    self.assertEqual(x.id, 'root.a[=0/3]')
    self.assertEqual(y.id, 'root.a[=1/3]')
    self.assertEqual(z.id, 'root.a[=2/3]')
    self.assertEqual(b.id, 'root.b')
    self.assertEqual(c.id, 'root.c')
    self.assertEqual(d.id, 'root.d')

  def test_get(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    b = manyof(2, [constant(), constant()], location='b')
    c = floatv(-1.0, 1.0, location='c')
    d = custom(location='d', name='d')
    space = Space([a, b, c, d], location='root')

    # Get by ID.
    self.assertIs(space.get('root.a'), a)
    self.assertIs(space.get('root.a[=0/3]'), x)
    self.assertEqual(
        space.get('root.a[=1/3]'), [y.subchoice(0), y.subchoice(1)])
    self.assertIs(space.get('root.a[=2/3]'), z)
    self.assertEqual(space.get('root.b'), [b.subchoice(0), b.subchoice(1)])
    self.assertIs(space.get('root.c'), c)
    self.assertIs(space.get('root.d'), d)

    # Get by name.
    self.assertIsNone(space.get('a'))  # `a` does not specify a name.
    self.assertEqual(space.get('y'), [y.subchoice(0), y.subchoice(1)])
    self.assertIs(space.get('d'), d)

  def test_getitem(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    b = manyof(2, [constant(), constant()], location='b')
    c = floatv(-1.0, 1.0, location='c')
    d = custom(location='d', name='d')
    space = Space([a, b, c, d], location='root')

    # Get by index and slice.
    self.assertIs(space[0], a)
    self.assertEqual(space[1:], [b, c, d])

    # Get by ID.
    self.assertIs(space['root.c'], c)

    # Get by name.
    self.assertEqual(space['y'], [y.subchoice(0), y.subchoice(1)])
    with self.assertRaises(KeyError):
      _ = space['a']    # `a` does not specify a name.

  def test_decision_ids(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    b = manyof(2, [constant(), constant()], location='b')
    c = floatv(-1.0, 1.0, location='c')
    d = custom(location='d', name='d')
    space = Space([a, b, c, d], location='root')

    self.assertEqual(
        space.decision_ids,
        [
            'root.a',
            'root.a[=0/3]',
            'root.a[=1/3]',
            'root.a[=2/3]',
            'root.b',
            'root.c',
            'root.d',
        ])

  def test_decision_points(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    b = manyof(2, [constant(), constant()], location='b')
    c = floatv(-1.0, 1.0, location='c')
    d = custom(location='d', name='d')
    space = Space([a, b, c, d], location='root')

    self.assertEqual(
        space.decision_points,
        [
            a, x, y.subchoice(0), y.subchoice(1), z,
            b.subchoice(0), b.subchoice(1), c, d,
        ])

  def test_named_decision_points(self):
    x = oneof([constant(), constant()])
    y = manyof(2, [constant(), constant()], name='y')
    z = floatv(-1.0, 1.0)
    a = oneof([x, y, z], location='a')
    b = manyof(2, [constant(), constant()], location='b')
    c = floatv(-1.0, 1.0, location='c')
    d = custom(location='d', name='d')
    space = Space([a, b, c, d], location='root')
    self.assertEqual(
        space.named_decision_points,
        {'d': d, 'y': [y.subchoice(0), y.subchoice(1)]})

  def test_set_get_userdata(self):
    s = constant()
    s.set_userdata('mydata', 1)
    self.assertEqual(s.userdata.mydata, 1)
    self.assertEqual(s.userdata['mydata'], 1)
    with self.assertRaises(KeyError):
      _ = s.userdata.nonexist_data
    with self.assertRaises(KeyError):
      _ = s.userdata['nonexist_data']

  def test_parent_spec(self):
    a = Space([                                   # a
        oneof([                                   # b
            oneof([                               # x
                constant(),
                constant()
            ]),
            manyof(2, [                           # y
                constant(),
                constant(),
                constant()
            ]),
            custom(),                             # z
        ]),
        floatv(min_value=0.0, max_value=1.0),     # c
        custom(),                                 # d
    ])
    b = a.elements[0]
    c = a.elements[1]
    d = a.elements[2]
    x = b.candidates[0].elements[0]
    y = b.candidates[1].elements[0]
    z = b.candidates[2].elements[0]

    self.assertIsNone(a.parent_spec)
    self.assertIs(b.parent_spec, a)
    self.assertIs(b.candidates[0].parent_spec, b)
    self.assertIs(x.parent_spec, b.candidates[0])
    self.assertIs(y.parent_spec, b.candidates[1])
    self.assertIs(z.parent_spec, b.candidates[2])
    self.assertIs(c.parent_spec, a)
    self.assertIs(d.parent_spec, a)

  def test_parent_choice(self):
    a = Space([                                # a
        oneof([                                # b
            oneof([                            # x
                constant(),
                constant()
            ]),
            manyof(2, [                        # y
                constant(),
                constant(),
                constant()
            ]),
            custom(),                          # z
        ]),
        floatv(min_value=0.0, max_value=1.0),  # c
        custom(),                              # d
    ])
    b = a.elements[0]
    c = a.elements[1]
    d = a.elements[2]
    x = b.candidates[0].elements[0]
    y = b.candidates[1].elements[0]
    z = b.candidates[2].elements[0]

    self.assertIsNone(a.parent_choice)
    self.assertIsNone(b.parent_choice)
    self.assertIsNone(b.subchoice(0).parent_choice)
    self.assertIs(x.parent_choice, b)
    self.assertIs(x.subchoice(0).parent_choice, b)
    self.assertIs(y.parent_choice, b)
    self.assertIs(y.subchoice(0).parent_choice, b)
    self.assertIs(y.subchoice(1).parent_choice, b)
    self.assertIs(z.parent_choice, b)
    self.assertIs(z.parent_choice, b)
    self.assertIsNone(c.parent_choice)
    self.assertIsNone(d.parent_choice)

  def test_validate(self):
    space = constant()
    space.validate(DNA(None))

    with self.assertRaisesRegex(ValueError, 'Extra DNA values encountered'):
      space.validate(DNA(1))

    space = Space([
        oneof([
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
        ]),
        floatv(min_value=0.0, max_value=1.0),
    ])
    space.validate(DNA([(0, 1), 0.5]))
    space.validate(DNA([(1, [0, 1]), 0.5]))
    space.validate(DNA([(3, 'abc'), 0.1]))
    space.validate(DNA([(4, 0.0), 0.1]))

    with self.assertRaisesRegex(
        ValueError,
        'Number of child values in DNA \\(.*\\) does not match '
        'the number of elements \\(.*\\)'):
      space.validate(DNA(None))

    with self.assertRaisesRegex(
        ValueError,
        'Number of child values in DNA \\(.*\\) does not match '
        'the number of elements \\(.*\\)'):
      space.validate(DNA(1))

    with self.assertRaisesRegex(
        ValueError, 'No child DNA provided for child space'):
      space.validate(DNA([0, 0.5]))

    with self.assertRaisesRegex(ValueError, 'Expect float value'):
      space.validate(DNA([2, 0]))

    with self.assertRaisesRegex(
        ValueError, 'DNA value should be no greater than '):
      space.validate(DNA([2, 1.5]))

    with self.assertRaisesRegex(
        ValueError, 'CustomDecisionPoint expects string type DNA'):
      space.validate(DNA([(3, 0.2), 0.1]))

  def test_first_dna(self):
    self.assertEqual(constant().first_dna(), DNA(None))

    space = Space([
        oneof([
            oneof([
                constant(),
                constant()
            ]),
            manyof(2, [
                constant(),
                constant(),
                constant()
            ])
        ]),
        floatv(min_value=0.0, max_value=1.0),
    ])
    self.assertEqual(space.first_dna(), DNA([(0, 0), 0.0]))

  def test_next_dna(self):
    space = constant()
    self.assertIsNone(space.next_dna(space.first_dna()))

    space = Space([floatv(-1.0, 1.0)])
    self.assertEqual(space.next_dna(), DNA(-1.0))
    with self.assertRaisesRegex(
        NotImplementedError, '`next_dna` is not supported on `Float` yet.'):
      space.next_dna(DNA(-1.0))

    space = Space([custom()])
    with self.assertRaisesRegex(
        NotImplementedError,
        '`next_dna` is not supported on \'CustomDecisionPoint\'.'):
      space.next_dna()

    space = Space([
        oneof([
            oneof([
                constant(),
                constant()
            ]),
            manyof(2, [
                constant(),
                constant(),
                constant()
            ]),
        ]),
        oneof([constant(), constant()])
    ])
    self.assertEqual(space.next_dna(), DNA([(0, 0), 0]))
    self.assertEqual(
        space.next_dna(DNA([(0, 1), 0])), DNA([(0, 1), 1]))
    self.assertEqual(
        space.next_dna(DNA([(0, 1), 1])), DNA([(1, [0, 1]), 0]))
    self.assertEqual(
        space.next_dna(DNA([(1, [0, 2]), 1])), DNA([(1, [1, 0]), 0]))

  def test_iter_dna(self):
    self.assertEqual(list(constant().iter_dna()), [DNA(None)])
    space = Space([
        oneof([
            oneof([
                constant(),
                constant()
            ]),
            manyof(2, [
                constant(),
                constant(),
                constant()
            ]),
        ]),
        oneof([constant(), constant()])
    ])
    self.assertEqual(list(space.iter_dna()), [
        DNA([(0, 0), 0]),
        DNA([(0, 0), 1]),
        DNA([(0, 1), 0]),
        DNA([(0, 1), 1]),
        DNA([(1, [0, 1]), 0]),
        DNA([(1, [0, 1]), 1]),
        DNA([(1, [0, 2]), 0]),
        DNA([(1, [0, 2]), 1]),
        DNA([(1, [1, 0]), 0]),
        DNA([(1, [1, 0]), 1]),
        DNA([(1, [1, 2]), 0]),
        DNA([(1, [1, 2]), 1]),
        DNA([(1, [2, 0]), 0]),
        DNA([(1, [2, 0]), 1]),
        DNA([(1, [2, 1]), 0]),
        DNA([(1, [2, 1]), 1])
    ])
    self.assertEqual(list(space.iter_dna(DNA([(1, [1, 2]), 1]))), [
        DNA([(1, [2, 0]), 0]),
        DNA([(1, [2, 0]), 1]),
        DNA([(1, [2, 1]), 0]),
        DNA([(1, [2, 1]), 1])
    ])

  def test_random_dna(self):
    r = random.Random(1)
    self.assertEqual(constant().random_dna(r), DNA(None))
    space = Space([
        oneof([
            oneof([
                constant(),
                constant()
            ]),
            manyof(2, [
                constant(),
                constant(),
                constant()
            ]),
        ]),
        oneof([constant(), constant()])
    ])
    self.assertEqual(space.random_dna(r), DNA([(0, 0), 1]))

  def test_inspection(self):
    space = Space([
        oneof([
            oneof([constant(), constant()]),
            constant()
        ]),
        floatv(min_value=0.0, max_value=1.0)
    ], location='a')

    # Test compact version, which overrides symbolic.Object.format.
    self.assertEqual(
        str(space),
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
        space.format(compact=False, verbose=True),
        symbolic.Object.format(space, compact=False, verbose=True))

  def test_serialization(self):
    space = Space([
        oneof([constant(), constant()], hints=1),
        manyof(2, [constant(), constant()]),
        floatv(0., 1.)
    ])
    json_dict = space.to_json()
    json_dict['userdata'] = None
    self.assertTrue(symbolic.eq(symbolic.from_json(json_dict), space))


if __name__ == '__main__':
  unittest.main()
