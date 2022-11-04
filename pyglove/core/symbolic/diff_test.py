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
"""Tests for pyglove.diff."""

import unittest

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.diff import Diff
from pyglove.core.symbolic.diff import diff as pg_diff
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.object import members as pg_members
from pyglove.core.symbolic.object import Object


class DiffTest(unittest.TestCase):
  """Tests for `pg.diff`."""

  def test_diff_class(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    self.assertTrue(bool(Diff(1, 2)))
    self.assertTrue(bool(Diff(A, B)))
    self.assertTrue(bool(Diff(A(1), A(2))))
    self.assertTrue(bool(Diff(A, A, children={'x': Diff(1, 2)})))
    self.assertFalse(bool(Diff(1, 1)))
    self.assertFalse(bool(Diff(A, A)))
    self.assertFalse(bool(Diff(A(1), A(1))))

    self.assertTrue(Diff(1, 2).is_leaf)
    self.assertFalse(
        Diff(int, int, children={'x': Diff(1, 2)}).is_leaf)

    self.assertEqual(Diff(1, 1), 1)
    self.assertEqual(Diff(1, 1), Diff(1, 1))
    self.assertNotEqual(Diff(1, 1), 2)
    self.assertNotEqual(Diff(1, 1), Diff(1, 2))

    self.assertEqual(
        repr(Diff(A(1), A(1))), 'A(x=1)')
    self.assertEqual(
        repr(Diff(A(1), A(2))), 'Diff(left=A(x=1), right=A(x=2))')
    self.assertEqual(
        repr(Diff(A, A, children={'x': Diff(1, 2)})),
        'A(x=Diff(left=1, right=2))')
    self.assertEqual(
        repr(Diff(A, B, children={
            'x': Diff(1, 2),
            'y': Diff(Diff.MISSING, 3)
        })),
        'A|B(x=Diff(left=1, right=2), y=Diff(left=MISSING, right=3))')
    self.assertEqual(
        repr(Diff(List, List, children={
            '0': Diff(1, 2),
            '1': Diff(Diff.MISSING, 3)
        })),
        '[0=Diff(left=1, right=2), 1=Diff(left=MISSING, right=3)]')

    with self.assertRaisesRegex(
        ValueError,
        'At least one of \'left\' and \'right\' should be specified.'):
      Diff()

    with self.assertRaisesRegex(
        ValueError,
        '\'left\' must be a type when \'children\' is specified.'):
      Diff(1, int, children={'x': Diff(3, 4)})

    with self.assertRaisesRegex(
        ValueError,
        '\'right\' must be a type when \'children\' is specified.'):
      Diff(int, 2, children={'x': Diff(3, 4)})

    with self.assertRaisesRegex(
        ValueError,
        '\'value\' cannot be accessed when \'left\' and \'right\' '
        'are not the same.'):
      _ = Diff(1, 2).value

  def test_diff_on_simple_types(self):
    self.assertEqual(pg_diff(1, 1), Diff(1, 1))
    self.assertEqual(pg_diff(1, 1, mode='same'), Diff(1, 1))
    self.assertEqual(pg_diff(1, 1, flatten=True, mode='same'), Diff(1, 1))
    self.assertEqual(pg_diff(1, 2), Diff(1, 2))
    self.assertEqual(pg_diff(1, 2, mode='same'), Diff(1, 2))
    self.assertEqual(pg_diff(1, 2, flatten=True, mode='same'), Diff(1, 2))

  def test_diff_on_list(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    # List vs. list.
    self.assertEqual(pg_diff([A(1)], [A(1)]), Diff([A(1)], [A(1)]))
    self.assertEqual(
        pg_diff([A(1)], [A(0)]),
        Diff(left=List, right=List, children={
            '0': Diff(A, A, children={
                'x': Diff(1, 0)
            })}))

    # List vs. other types.
    self.assertEqual(pg_diff([A(1)], 1), Diff([A(1)], 1))

  def test_diff_on_dict(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    # Dict vs. dict.
    self.assertEqual(
        pg_diff({'a': A(1)}, {'a': A(1)}), Diff({'a': A(1)}, {'a': A(1)}))
    self.assertEqual(
        pg_diff({'a': A(1), 'b': A(2), 'c': A(3)},
                {'a': A(1), 'b': A(3), 'd': A(4)}),
        Diff(dict, dict, children={
            'b': Diff(A, A, children={
                'x': Diff(2, 3)
            }),
            'c': Diff(A(3)),
            'd': Diff(right=A(4)),
        }))

    # Dict vs. symbolic object.
    self.assertEqual(
        pg_diff(A(1), {'x': 1}),
        Diff(A(1), {'x': 1}))

    self.assertEqual(
        pg_diff(A(1), {'x': 1}, collapse=True),
        Diff(A, dict))

    self.assertEqual(
        pg_diff(A(1), {'x': 2}, collapse=True),
        Diff(A, dict, children={
            'x': Diff(1, 2)
        }))

    # Dict vs. other types.
    self.assertEqual(pg_diff({'x': 1}, 1), Diff({'x': 1}, 1))

  def test_diff_on_symbolic_object(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    # Same types.
    self.assertEqual(pg_diff(A(1), A(1)), Diff(A(1), A(1)))
    self.assertEqual(
        pg_diff(B(1, 2), B(1, 3)),
        Diff(B, B, children={'y': Diff(2, 3)}))

    # Different types without collapse.
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2)),
        Diff(B(1, 2), C(1, 2)))

    # Different types with always collapse.
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True),
        Diff(B, C))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3)),
        pg_diff(B(1, 2), C(1, 3)))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True),
        Diff(B, C, children={
            'y': Diff(2, 3)
        }))

  def test_diff_between_symbolic_and_dict(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    a = A(A(1))
    d = Dict(x=Dict(x=1))
    self.assertEqual(pg_diff(a, d), Diff(left=a, right=d))
    self.assertEqual(
        pg_diff(a, d, collapse=True),
        Diff(A, Dict, children=dict(x=Diff(A, Dict))))

  def test_collapse(self):

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class D(Object):
      pass

    def collapse_subclass(x, y):
      return issubclass(type(x), type(y)) or issubclass(type(y), type(x))

    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=collapse_subclass),
        Diff(B, C))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=collapse_subclass),
        Diff(B, C, children={
            'y': Diff(2, 3)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), D(1, 3), collapse=collapse_subclass),
        Diff(B(1, 2), D(1, 3)))

    # Test bad collapse option.
    with self.assertRaisesRegex(
        ValueError, 'Unsupported `collapse` value'):
      pg_diff(B(1, 2), C(1, 2), collapse='unsupported_option')

  def test_mode(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    self.assertEqual(pg_diff(A(1), A(1), mode='diff'), Diff(A(1), A(1)))
    self.assertEqual(pg_diff(A(1), A(1), mode='same'), Diff(A(1), A(1)))
    self.assertEqual(pg_diff(A(1), A(1), mode='both'), Diff(A(1), A(1)))

    self.assertEqual(
        pg_diff(B(1, 2), B(1, 3), mode='diff'),
        Diff(B, B, children={'y': Diff(2, 3)}))
    self.assertEqual(
        pg_diff(B(1, 2), B(1, 3), mode='same'),
        Diff(B, B, children={'x': Diff(1, 1)}))
    self.assertEqual(
        pg_diff(B(1, 2), B(2, 2), mode='both'),
        Diff(B, B, children={
            'x': Diff(1, 2),
            'y': Diff(2, 2)
        }))

    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True, mode='diff'), Diff(B, C))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True, mode='same'),
        Diff(B, C, children={
            'x': Diff(1, 1),
            'y': Diff(2, 2)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True, mode='both'),
        Diff(B, C, children={
            'x': Diff(1, 1),
            'y': Diff(2, 2)
        }))

    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, mode='diff'),
        Diff(B, C, children={
            'y': Diff(2, 3)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, mode='same'),
        Diff(B, C, children={
            'x': Diff(1, 1)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, mode='both'),
        Diff(B, C, children={
            'x': Diff(1, 1),
            'y': Diff(2, 3)
        }))

  def test_flatten(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    self.assertEqual(
        pg_diff(A(Dict(a=1, b=2, c=3)), A(Dict(a=1, b=3, d=4)), flatten=True),
        {
            'x.b': Diff(2, 3),
            'x.c': Diff(3, Diff.MISSING),
            'x.d': Diff(Diff.MISSING, 4)
        })
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, flatten=True),
        {
            'y': Diff(2, 3),
            '_type': Diff(B, C),
        })


if __name__ == '__main__':
  unittest.main()
