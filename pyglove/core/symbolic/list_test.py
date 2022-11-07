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
"""Tests for pyglove.List."""

import copy
import inspect
import io
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags
from pyglove.core.symbolic import object as pg_object
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.list import Insertion
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.pure_symbolic import NonDeterministic
from pyglove.core.symbolic.pure_symbolic import PureSymbolic


MISSING_VALUE = object_utils.MISSING_VALUE


class ListTest(unittest.TestCase):
  """Tests for `pg.Dict`."""

  def test_init(self):
    # Schemaless list.
    sl = List()
    self.assertIsNone(sl.value_spec)
    self.assertEqual(len(sl), 0)

    # Schemaless list created from a regular list.
    sl = List([1, 2, 3])
    self.assertIsNone(sl.value_spec)
    self.assertEqual(sl, [1, 2, 3])

    # Schemaless list created from iterable values.
    sl = List(range(3))
    self.assertIsNone(sl.value_spec)
    self.assertEqual(sl, [0, 1, 2])

    # Schematized list.
    vs = pg_typing.List(pg_typing.Int())
    sl = List([1], value_spec=vs)
    self.assertIs(sl.value_spec, vs)
    self.assertEqual(sl, [1])

    with self.assertRaisesRegex(
        TypeError, '.* must be a `pg.typing.List` object.'):
      List(value_spec=pg_typing.Int())

  def test_partial(self):
    spec = pg_typing.List(pg_typing.Dict([
        ('a', pg_typing.Int()),
        ('b', pg_typing.Int())
    ]))
    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified.'):
      _ = List([{'a': 1}], value_spec=spec)
    self.assertTrue(List.partial([{'a': 1}], value_spec=spec).is_partial)

    with flags.allow_partial(True):
      self.assertTrue(List.partial([{'a': 1}], value_spec=spec).is_partial)

    sl = List.partial([{'a': 1}], value_spec=spec)
    self.assertTrue(sl.is_partial)
    sl.rebind({'[0].b': 2})
    self.assertFalse(sl.is_partial)

  def test_missing_value_as_values(self):
    sl = List([1, MISSING_VALUE, 2, MISSING_VALUE])
    self.assertEqual(sl, List([1, 2]))
    sl.rebind({2: MISSING_VALUE}, raise_on_no_change=False)
    self.assertEqual(sl, List([1, 2]))

    sl = List(value_spec=pg_typing.List(pg_typing.Int(default=0)))
    sl.append(MISSING_VALUE)
    self.assertEqual(sl, [])

  def test_runtime_type_check(self):
    spec = pg_typing.List(pg_typing.Dict([
        ('a', pg_typing.Int(min_value=0)),
        ('b', pg_typing.List(pg_typing.Int(), min_size=2)),
        ('c', pg_typing.Dict([
            ('x', pg_typing.Int())]))
    ]))
    with self.assertRaisesRegex(ValueError, 'Required value is not specified'):
      List([dict(b=[0, 1], c=dict(x=1))], value_spec=spec)

    with self.assertRaisesRegex(ValueError, '.* is out of range'):
      List([dict(a=-1, b=[0, 1], c=dict(x=1))], value_spec=spec)

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is less than min size'):
      List([dict(a=1, b=[0], c=dict(x=1))], value_spec=spec)

    with flags.enable_type_check(False):
      self.assertEqual(List([0], value_spec=spec), [0])

  def test_symbolization_on_nested_containers(self):
    sl = List([{'c': True, 'd': []}])
    self.assertIsInstance(sl[0], Dict)
    self.assertIsInstance(sl[0].d, List)

  def test_implicit_copy_during_assignment(self):

    class A:
      pass

    # There is no impliit copy when assigning a root symbolic object to
    # another tree.
    sl = List([dict(x=dict(), y=list(), z=A())])
    sl2 = List([sl])

    self.assertIs(sl, sl2[0])

    # There is an implicit copy when assigning a symbolic object with
    # a parent to another tree.
    sl3 = List([sl])
    self.assertIsNot(sl, sl3[0])
    self.assertIsNot(sl[0], sl3[0][0])
    self.assertIsNot(sl[0].x, sl3[0][0].x)
    self.assertIsNot(sl[0].y, sl3[0][0].y)
    # Non-symbolic member is copy by reference.
    self.assertIs(sl[0].z, sl3[0][0].z)

    self.assertEqual(sl[0], sl3[0][0])
    self.assertEqual(sl[0].x, sl3[0][0].x)
    self.assertEqual(sl[0].y, sl3[0][0].y)

  def test_spec_compatibility_during_assignment(self):
    sd = Dict(x=[], value_spec=pg_typing.Dict([
        ('x', pg_typing.List(pg_typing.Int(min_value=0)))]))
    sl = List([0], value_spec=pg_typing.List(pg_typing.Int()))
    with self.assertRaisesRegex(
        ValueError, 'List .* cannot be assigned to an incompatible field'):
      # A field of non-negative list cannot accept a list of possible
      # negative integers.
      sd.x = sl

  def test_inspect(self):
    s = io.StringIO()
    sl = List([dict(x=1, y=dict(a='foo'))])
    sl.inspect(file=s, compact=True)
    self.assertEqual(s.getvalue(), '[0: {x=1, y={a=\'foo\'}}]\n')

    s = io.StringIO()
    sl.inspect(where=lambda v: v == 1, file=s)
    self.assertEqual(s.getvalue(), '{\n  \'[0].x\': 1\n}\n')

  def test_setitem(self):
    # Set item in a schemaless dict.
    sl = List([0])
    sl[0] = 1
    self.assertEqual(sl, [1])
    with self.assertRaisesRegex(
        TypeError, 'list assignment index must be an integer'):
      sl['abc'] = 1

    with self.assertRaisesRegex(
        IndexError, 'list assignment index out of range'):
      sl[1] = 2
    self.assertEqual(sl, [1])

    # Set item in a schematized list.
    sl = List([0], value_spec=pg_typing.List(pg_typing.Int(min_value=0)))
    self.assertEqual(sl, [0])
    sl[0] = 1
    self.assertEqual(sl, [1])
    with self.assertRaisesRegex(ValueError, '.* is out of range'):
      sl[0] = -1

    # Slicing.
    context = Dict(updates=0)
    def on_list_change(field_updates):
      context.updates = len(field_updates)

    sl = List([0, 1, 2], value_spec=pg_typing.List(pg_typing.Int()))
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sl[2:] = [1.0, 2.0]

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[2:] = [4, 5, 6]
    self.assertEqual(sl, [0, 1, 4, 5, 6])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[:1] = [4, 5, 6]
    self.assertEqual(sl, [4, 5, 6, 1, 2])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[5:] = [4, 5, 6]
    self.assertEqual(sl, [0, 1, 2, 4, 5, 6])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[-1:] = [4, 5, 6]
    self.assertEqual(sl, [0, 1, 4, 5, 6])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[-5:] = [4, 5, 6]
    self.assertEqual(sl, [4, 5, 6])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[:] = [4, 5, 6]
    self.assertEqual(sl, [4, 5, 6])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[:-2] = [4, 5, 6]
    self.assertEqual(sl, [4, 5, 6, 1, 2])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[:-5] = [4, 5, 6]
    self.assertEqual(sl, [4, 5, 6, 0, 1, 2])
    self.assertEqual(context.updates, 3)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[::2] = [4, 6]
    self.assertEqual(sl, [4, 1, 6])
    self.assertEqual(context.updates, 2)

    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl[::-2] = [4, 6]
    self.assertEqual(sl, [6, 1, 4])
    self.assertEqual(context.updates, 2)

    sl = List([0, 1, 2])
    with self.assertRaisesRegex(
        ValueError,
        'attempt to assign sequence of size .* to extended slice of size .*'):
      sl[::3] = [4, 6]
    self.assertEqual(sl, [0, 1, 2])

  def test_getitem(self):
    sl = List([0])
    self.assertEqual(sl[0], 0)
    self.assertEqual(sl[-1], 0)
    with self.assertRaisesRegex(IndexError, 'list index out of range'):
      _ = sl[1]

    # Slicing.
    sl = List([0, 1, 2])
    self.assertEqual(sl[:-1], [0, 1])
    self.assertEqual(sl[1:], [1, 2])
    self.assertEqual(sl[:], [0, 1, 2])

  def test_delitem(self):
    # Delete an item from a schemaless dict.
    sl = List([0])
    del sl[0]
    self.assertEqual(len(sl), 0)

    sl = List([0])
    del sl[-1]
    with self.assertRaisesRegex(TypeError, 'list index must be an integer'):
      del sl['x']

    with self.assertRaisesRegex(IndexError, 'list index out of range'):
      del sl[0]

    sl.append(1)
    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot delete item from a sealed List.'):
      del sl[0]

  def test_insert(self):
    sl = List([0, 1, 2])
    sl.insert(1, 3)
    self.assertEqual(sl, [0, 3, 1, 2])

    sl = List([0, 1, 2], value_spec=pg_typing.List(pg_typing.Int(), max_size=4))
    sl.insert(1, 3)
    self.assertEqual(sl, [0, 3, 1, 2])
    with self.assertRaisesRegex(ValueError, 'List reached its max size 4'):
      sl.insert(0, 5)
    self.assertEqual(sl, [0, 3, 1, 2])

    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot insert element into a sealed List.'):
      sl.insert(0, 6)

  def test_append(self):
    sl = List([0, 1, 2])
    sl.append(3)
    self.assertEqual(sl, [0, 1, 2, 3])

    sl = List([0, 1, 2], value_spec=pg_typing.List(pg_typing.Int(), max_size=4))
    sl.append(3)
    self.assertEqual(sl, [0, 1, 2, 3])
    sl.pop()
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sl.append('foo')
    sl.append(3)
    with self.assertRaisesRegex(ValueError, 'List reached its max size '):
      sl.append('foo')
    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot append element on a sealed List.'):
      sl.append(4)

  def test_extend(self):
    sl = List([0, 1])
    sl.extend([2, 3])
    self.assertEqual(sl, [0, 1, 2, 3])

    sl = List([0, 1], value_spec=pg_typing.List(pg_typing.Int(), max_size=4))
    with self.assertRaisesRegex(
        ValueError,
        'Cannot extend List: the number of elements .* exceeds max size'):
      sl.extend([2, 3, 4])
    self.assertEqual(sl, [0, 1])
    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot extend a sealed List.'):
      sl.extend([0, 1])

  def test_concatenate(self):
    sl = List([0]) + List([1])
    self.assertIsInstance(sl, List)
    self.assertEqual(sl, [0, 1])

    spec = pg_typing.List(pg_typing.Int(min_value=0))
    sl = List([0], value_spec=spec) + [1, 2]
    self.assertIsInstance(sl, List)
    self.assertIs(sl.value_spec, spec)
    self.assertEqual(sl, [0, 1, 2])

    self.assertEqual(
        List([0], value_spec=spec) + range(2), List([0, 0, 1], value_spec=spec))
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered'):
      _ = List([0], value_spec=spec) + [1.0]

  def test_multiply(self):
    class A:
      pass

    a = A()
    sl = List([a]) * 0
    self.assertIsInstance(sl, List)
    self.assertEqual(len(sl), 0)

    sl = List([a]) * 3
    self.assertIsInstance(sl, List)
    # This is true only when copy is done by reference.
    self.assertEqual(sl, [a, a, a])

    spec = pg_typing.List(pg_typing.Int(min_value=0), min_size=1, max_size=10)
    sl = List([0], value_spec=spec) * 2
    self.assertIsInstance(sl, List)
    self.assertIs(sl.value_spec, spec)
    self.assertEqual(sl, [0, 0])

    # Right hand multiplication.
    sl = 2 * List([0], value_spec=spec)
    self.assertIsInstance(sl, List)
    self.assertIs(sl.value_spec, spec)
    self.assertEqual(sl, [0, 0])

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is less than min size'):
      _ = List([0], value_spec=spec) * 0

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is greater than max size'):
      _ = List([0], value_spec=spec) * 11

  def test_remove(self):
    sl = List([0, 1, 2])
    sl.remove(1)
    self.assertEqual(sl, [0, 2])
    with self.assertRaisesRegex(ValueError, '.* not in list'):
      sl.remove(3)

    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot delete item from a sealed List.'):
      sl.remove(2)

    sl = List([0, 1], value_spec=pg_typing.List(pg_typing.Int(), min_size=2))
    with self.assertRaisesRegex(
        ValueError, 'Cannot remove item: min size .* is reached'):
      sl.remove(1)

  def test_pop(self):
    sl = List([0, 1, 2])
    self.assertEqual(sl.pop(), 2)
    self.assertEqual(sl, [0, 1])

    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot delete item from a sealed List.'):
      sl.pop(-1)

    with self.assertRaisesRegex(IndexError, 'pop index out of range'):
      _ = sl.pop(5)

    with self.assertRaisesRegex(IndexError, 'pop index out of range'):
      _ = sl.pop(-3)

  def test_clear(self):
    sl = List([0, 1])
    sl.clear()
    self.assertEqual(len(sl), 0)

    sl = List([0, 1])
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot clear a sealed List.'):
        sl.clear()

    sl = List([0, 1], value_spec=pg_typing.List(pg_typing.Int(), size=2))
    with self.assertRaisesRegex(
        ValueError, 'List cannot be cleared: min size is 2'):
      sl.clear()
    self.assertEqual(len(sl), 2)

  def test_use_value_spec(self):
    spec = pg_typing.List(pg_typing.Int(min_value=0))

    sl = List([0])
    with self.assertRaisesRegex(
        ValueError, 'Value spec for list must be a `pg.typing.List` object'):
      sl.use_value_spec(pg_typing.Int())
    sl.use_value_spec(spec)

    # Apply the same schema twice to verify its eligibility.
    sl.use_value_spec(spec)

    with self.assertRaisesRegex(
        RuntimeError, 'List is already bound with a different value spec: .*'):
      sl.use_value_spec(pg_typing.List(pg_typing.Float()))

    # Remove schema constraint and insert new keys.
    sl.use_value_spec(None)
    sl[0] = -1
    self.assertEqual(sl[0], -1)

    with flags.enable_type_check(False):
      # Shall not trigger error, since type/value check is not enabled.
      sl.use_value_spec(spec)

  def test_index(self):
    sl = List([0, 1, 2, 1])
    self.assertEqual(sl.index(1), 1)
    with self.assertRaisesRegex(
        ValueError, '3 is not in list'):
      _ = sl.index(3)

  def test_count(self):
    sl = List([0, 1, 2, 1])
    self.assertEqual(sl.count(1), 2)
    self.assertEqual(sl.count(3), 0)

  def test_copy(self):

    class A:
      pass

    sl = List([0, A(), dict(x=A()), list([A()])])
    sl2 = sl.copy()
    self.assertIsInstance(sl2, List)
    self.assertEqual(sl, sl2)
    self.assertIsNot(sl, sl2)
    self.assertIs(sl[1], sl2[1])
    self.assertIsNot(sl[2], sl2[2])
    self.assertIsNot(sl[3], sl2[3])
    self.assertIs(sl[2].x, sl2[2].x)
    self.assertIs(sl[3][0], sl2[3][0])

    sl = List([0], value_spec=pg_typing.List(pg_typing.Int()))
    sl2 = sl.copy()
    self.assertIs(sl.value_spec, sl2.value_spec)

    # Test copy.copy.
    sl = List([0, A(), dict(x=A()), list([A()])])
    sl3 = copy.copy(sl)
    self.assertIsInstance(sl3, List)
    self.assertEqual(sl, sl3)
    self.assertIsNot(sl, sl3)
    self.assertIs(sl[1], sl3[1])
    self.assertIsNot(sl[2], sl3[2])
    self.assertIsNot(sl[3], sl3[3])
    self.assertIs(sl[2].x, sl3[2].x)
    self.assertIs(sl[3][0], sl3[3][0])

    # Test copy.deepcopy.

    class B:

      def __init__(self, v):
        self.v = v

      def __eq__(self, other):
        return isinstance(other, B) and self.v == other.v

    sl = List([0, B(1), dict(x=B(2)), list([B(3)])])
    sl4 = copy.deepcopy(sl)
    self.assertIsInstance(sl4, List)
    self.assertEqual(sl, sl4)
    self.assertIsNot(sl, sl4)
    self.assertIsNot(sl[1], sl4[1])
    self.assertIsNot(sl[2], sl4[2])
    self.assertIsNot(sl[3], sl4[3])
    self.assertIsNot(sl[2].x, sl4[2].x)
    self.assertIsNot(sl[3][0], sl4[3][0])

  def test_sort(self):
    sl = List([0, 2, 1, 3])
    sl.sort()
    self.assertEqual(sl, [0, 1, 2, 3])

    sl.append(-1)
    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot sort a sealed List'):
      sl.sort()

  def test_reverse(self):
    sl = List([0, 2, 1, 3])
    sl.reverse()
    self.assertEqual(sl, [3, 1, 2, 0])

    sl.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot reverse a sealed List'):
      sl.reverse()

  def test_in(self):
    sl = List([0, 1, 2])
    self.assertIn(1, sl)
    self.assertNotIn(3, sl)

  def test_iter(self):
    sl = List([0, 1, 2, 3])
    self.assertEqual(list(sl), [0, 1, 2, 3])

  def test_non_default(self):
    sl = List([0])
    self.assertEqual(sl.non_default_values(), {'[0]': 0})

    sl = List([dict(a=0)], value_spec=pg_typing.List(pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Dict([
            ('c', pg_typing.Int(default=1)),
            ('d', pg_typing.Str(default='foo'))
        ]))
    ])))
    self.assertEqual(sl.non_default_values(), {})
    sl[0].b = dict(d='bar')
    self.assertEqual(sl.non_default_values(), {'[0].b.d': 'bar'})
    self.assertEqual(
        sl.non_default_values(flatten=False), {0: {'b': {'d': 'bar'}}})

    # After rebind, the non_default_values are updated.
    sl.rebind({'[0].b.d': 'foo', '[0].a': 1})
    self.assertEqual(sl.non_default_values(flatten=False), {0: {'a': 1}})

  def test_missing_values(self):
    sl = List([0])
    self.assertEqual(len(sl.missing_values()), 0)

    sl = List.partial([dict(a=1)], value_spec=pg_typing.List(pg_typing.Dict([
        ('a', pg_typing.Int()),
        ('b', pg_typing.Int()),
        ('c', pg_typing.Dict([
            ('d', pg_typing.Int(default=0)),
            ('e', pg_typing.Str())
        ]))
    ])))
    self.assertEqual(
        sl.missing_values(), {'[0].b': MISSING_VALUE, '[0].c.e': MISSING_VALUE})

    self.assertEqual(
        sl.missing_values(flatten=False),
        {0: {'b': MISSING_VALUE, 'c': {'e': MISSING_VALUE}}})

    # After rebind, `missing_values` is updated.
    sl.rebind({'[0].c.e': 'foo'})
    self.assertEqual(
        sl.missing_values(flatten=False), {0: {'b': MISSING_VALUE}})

  def test_sym_has(self):
    sl = List([dict(x=[dict(y=1)])])
    self.assertTrue(sl.sym_has(0))
    self.assertTrue(sl.sym_has('[0].x'))
    self.assertTrue(sl.sym_has('[0].x[0]'))
    self.assertTrue(sl.sym_has('[0].x[0].y'))
    self.assertTrue(sl.sym_has(object_utils.KeyPath.parse('[0].x[0].y')))

  def test_sym_get(self):
    sl = List([dict(x=[dict(y=1)])])
    self.assertEqual(sl.sym_get(0), dict(x=[dict(y=1)]))
    self.assertEqual(sl.sym_get('[0].x'), [dict(y=1)])
    self.assertEqual(sl.sym_get('[0].x[0]'), dict(y=1))
    self.assertEqual(sl.sym_get('[0].x[0].y'), 1)
    with self.assertRaisesRegex(KeyError, 'Path .* does not exist'):
      sl.sym_get('[0].y')

  def test_sym_hasattr(self):
    sl = List([dict(x=0)])
    self.assertTrue(sl.sym_hasattr(0))
    self.assertFalse(sl.sym_hasattr(1))
    self.assertFalse(sl.sym_hasattr('[0].x'))

  def test_sym_getattr(self):
    sl = List([dict(x=0)])
    self.assertEqual(sl.sym_getattr(0), dict(x=0))
    self.assertIsNone(sl.sym_getattr(1, None))
    self.assertIsNone(sl.sym_getattr('x', None))
    with self.assertRaisesRegex(
        AttributeError,
        '.* object has no symbolic attribute 1.'):
      sl.sym_getattr(1)

  def test_sym_field(self):
    sl = List([dict(x=[], y={})])
    self.assertIsNone(sl.sym_field)

    spec = pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.List(pg_typing.Int())),
        ('y', pg_typing.Dict())
    ]))
    sl.use_value_spec(spec)
    self.assertIsNone(sl.sym_field)
    self.assertIs(sl[0].sym_field, spec.element)
    self.assertIs(sl[0].x.sym_field, spec.element.value.schema.get_field('x'))

  def test_sym_attr_field(self):
    sl = List([dict(x=1, y={})])
    self.assertIsNone(sl.sym_attr_field(0))

    spec = pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict())
    ]))
    sl.use_value_spec(spec)
    self.assertIs(sl.sym_attr_field(0), spec.element)
    self.assertIs(
        sl[0].sym_attr_field('x'), spec.element.value.schema.get_field('x'))

  def test_sym_keys(self):
    sl = List(['a', 'b'])
    self.assertEqual(next(sl.sym_keys()), 0)
    self.assertEqual(list(sl.sym_keys()), [0, 1])

  def test_sym_values(self):
    sl = List(['a', 'b'])
    self.assertEqual(next(sl.sym_values()), 'a')
    self.assertEqual(list(sl.sym_values()), ['a', 'b'])

    sl = List([dict(x=1), dict(x=2)], value_spec=pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int(default=0))
    ])))
    self.assertEqual(next(sl.sym_values()), dict(x=1, y=0))
    self.assertEqual(list(sl.sym_values()), [dict(x=1, y=0), dict(x=2, y=0)])

  def test_sym_items(self):
    sl = List(['a', 'b'])
    self.assertEqual(next(sl.sym_items()), (0, 'a'))
    self.assertEqual(list(sl.sym_items()), [(0, 'a'), (1, 'b')])

    sl = List([dict(x=1), dict(x=2)], value_spec=pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int(default=0))
    ])))
    self.assertEqual(next(sl.sym_items()), (0, dict(x=1, y=0)))
    self.assertEqual(
        list(sl.sym_items()), [(0, dict(x=1, y=0)), (1, dict(x=2, y=0))])

  def test_sym_jsonify(self):
    # Refer to SerializationTest for more detailed tests.
    sl = List([0])
    self.assertEqual(sl.sym_jsonify(), [0])

  def test_sym_rebind(self):
    # Refer to RebindTest for more detailed tests.
    sl = List([0, 1, 2])
    sl.sym_rebind({
        0: MISSING_VALUE,
        1: 3,
        2: Insertion(4),
        4: 5
    })
    self.assertEqual(sl, [3, 4, 2, 5])

  def test_sym_clone(self):
    class A():
      pass

    sl = List([[], dict(), A()])
    sl2 = sl.clone()
    self.assertEqual(sl, sl2)
    self.assertIsNot(sl, sl2)
    # Symbolic members are always copied by value.
    self.assertIsNot(sl[0], sl2[0])
    self.assertIsNot(sl[1], sl2[1])

    # Non-symbolic members are copied by reference.
    self.assertIs(sl[2], sl2[2])

    spec = pg_typing.List(pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Any())
    ]))
    sl = List([Dict(x=list(), z=dict(), y=A())], value_spec=spec)
    sl2 = sl.sym_clone(deep=True)

    # Instances of `A` are compared by reference.
    # During deep clone A() is copied which results in a different instance.
    self.assertNotEqual(sl, sl2)
    self.assertIs(sl.value_spec, sl2.value_spec)
    self.assertIsNot(sl, sl2)
    self.assertIsNot(sl[0].x, sl2[0].x)
    self.assertIsNot(sl[0].y, sl2[0].y)
    self.assertIsNot(sl[0].z, sl2[0].z)

  def test_sym_origin(self):
    # Refer `object_test.test_sym_origin` for more details.
    sl = List([0])
    sl.sym_setorigin(List.__init__, 'constructor')
    self.assertEqual(sl.sym_origin.source, List.__init__)
    self.assertEqual(sl.sym_origin.tag, 'constructor')

  def test_sym_partial(self):
    # Refer to `test_partial` for more details.
    sl = List.partial([dict(x=1)], value_spec=pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict([
            ('z', pg_typing.Int())
        ])),
    ])))
    self.assertTrue(sl.sym_partial)
    sl.rebind({'[0].y.z': 2})
    self.assertFalse(sl.sym_partial)

  def test_sym_missing(self):
    # Refer to `test_missing_values` for more details.
    sl = List.partial([dict(x=1)], value_spec=pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int()),
    ])))
    self.assertEqual(sl.sym_missing(), {'[0].y': MISSING_VALUE})

  def test_sym_nondefault(self):
    # Refer to `test_non_default_values` for more details.
    sl = List([dict(x=1)], value_spec=pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int(default=0)),
        ('y', pg_typing.Dict([
            ('z', pg_typing.Int(default=1))
        ])),
    ])))
    self.assertEqual(sl.sym_nondefault(), {'[0].x': 1})
    sl.rebind({'[0].y.z': 2, '[0].x': 0})
    self.assertEqual(sl.sym_nondefault(), {'[0].y.z': 2})

  def test_sym_puresymbolic(self):
    self.assertFalse(List().sym_puresymbolic)

    class A(PureSymbolic):
      pass

    self.assertTrue(List([A()]).sym_puresymbolic)

  def test_sym_abstract(self):
    self.assertFalse(List().sym_abstract)
    self.assertTrue(List.partial([{}], value_spec=pg_typing.List(
        pg_typing.Dict([('x', pg_typing.Int())]))).sym_abstract)

    class A(PureSymbolic):
      pass

    self.assertTrue(List([A()]).sym_puresymbolic)

  def test_is_deterministic(self):

    class X(NonDeterministic):
      pass

    self.assertTrue(List().is_deterministic)
    self.assertFalse(List([X()]).is_deterministic)
    self.assertFalse(List([dict(x=X())]).is_deterministic)

  def test_sym_contains(self):
    sl = List([dict(x=dict(y=[dict(z=1)]))])
    self.assertTrue(sl.sym_contains(value=1))
    self.assertFalse(sl.sym_contains(value=2))
    self.assertTrue(sl.sym_contains(type=int))
    self.assertFalse(sl.sym_contains(type=str))

  def test_sym_eq(self):
    # Use cases that `__eq__` and `sym_eq` have the same results.
    self.assertEqual(List(), List())
    self.assertTrue(List().sym_eq(List()))
    self.assertTrue(base.eq(List(), List()))

    self.assertEqual(List([0]), List([0]))
    self.assertTrue(List([0]).sym_eq(List([0])))
    self.assertTrue(base.eq(List[0], List[0]))

    self.assertEqual(
        List([0]),
        List([0], value_spec=pg_typing.List(pg_typing.Int())))
    self.assertTrue(base.eq(
        List([0]),
        List([0], value_spec=pg_typing.List(pg_typing.Int()))))
    self.assertEqual(List([[]]), List([List()]))
    self.assertTrue(base.eq(List([[]]), List([List()])))

    # Use case that `__eq__` rules both Python equality and `pg.eq`.
    class A:

      def __init__(self, value):
        self.value = value

      def __eq__(self, other):
        return ((isinstance(other, A) and self.value == other.value)
                or self.value == other)

    self.assertEqual(List([A(1)]), List([1]))
    self.assertTrue(base.eq(List([A(1)]), List([1])))

    # Use case that `sym_eq` only rule `pg.eq` but not Python equality.
    class B:

      def __init__(self, value):
        self.value = value

      def sym_eq(self, other):
        return ((isinstance(other, A) and self.value == other.value)
                or self.value == other)

    self.assertNotEqual(List([B(1)]), List([1]))
    self.assertTrue(base.eq(List([B(1)]), List([1])))

  def test_sym_ne(self):
    # Refer test_sym_eq for more details.
    self.assertNotEqual(List(), 1)
    self.assertTrue(List().sym_ne(1))
    self.assertTrue(base.ne(List(), 1))

    self.assertNotEqual(List(), List([0]))
    self.assertTrue(List().sym_ne(List([0])))
    self.assertTrue(base.ne(List(), List([0])))

    self.assertNotEqual(List([0]), List([1]))
    self.assertTrue(List([0]).sym_ne(List([1])))
    self.assertTrue(base.ne(List([0]), List([1])))

  def test_sym_lt(self):
    self.assertFalse(List().sym_lt(MISSING_VALUE))
    self.assertFalse(List().sym_lt(None))
    self.assertFalse(List().sym_lt(True))
    self.assertFalse(List().sym_lt(1))
    self.assertFalse(List().sym_lt(2.0))
    self.assertFalse(List().sym_lt('abc'))
    self.assertTrue(List().sym_lt(tuple()))
    self.assertTrue(List().sym_lt(set()))
    self.assertTrue(List().sym_lt(Dict()))

    self.assertTrue(List().sym_lt(List([0])))
    self.assertTrue(List([0]).sym_lt(List([1])))
    self.assertTrue(List([0]).sym_lt(List([0, 1])))
    self.assertTrue(List([0, 1]).sym_lt(List([1])))
    self.assertFalse(List([0]).sym_lt(List([0])))
    self.assertFalse(List([1]).sym_lt(List([0])))
    self.assertFalse(List([0, 1]).sym_lt(List([0])))
    self.assertFalse(List([1]).sym_lt(List([0, 1])))

    class A:
      pass

    self.assertTrue(List().sym_lt(A()))

  def test_sym_gt(self):
    self.assertTrue(List().sym_gt(MISSING_VALUE))
    self.assertTrue(List().sym_gt(None))
    self.assertTrue(List().sym_gt(True))
    self.assertTrue(List().sym_gt(1))
    self.assertTrue(List().sym_gt(2.0))
    self.assertTrue(List().sym_gt('abc'))
    self.assertFalse(List().sym_gt((1,)))
    self.assertFalse(List().sym_gt(set()))
    self.assertFalse(List().sym_gt({}))

    self.assertTrue(List([0]).sym_gt(List()))
    self.assertTrue(List([1]).sym_gt(List([0])))
    self.assertTrue(List([1]).sym_gt(List([0, 1])))
    self.assertTrue(List([0, 1]).sym_gt(List([0])))
    self.assertFalse(List().sym_gt(List([0])))
    self.assertFalse(List([0]).sym_gt(List([0])))
    self.assertFalse(List([0]).sym_gt(List([0, 1])))
    self.assertFalse(List([0, 1]).sym_gt(List([1])))
    self.assertFalse(List([0]).sym_gt(List([0])))
    self.assertFalse(List([0]).sym_gt(List([1])))
    self.assertFalse(List([0]).sym_gt(List([0, 1])))

    class A:
      pass

    self.assertFalse(List().sym_gt(A()))

  def test_sym_hash(self):
    self.assertEqual(hash(List()), hash(List()))
    self.assertEqual(hash(List([0])), hash(List([0])))
    self.assertEqual(hash(List([dict(x=1)])), hash(List([dict(x=1)])))
    self.assertNotEqual(hash(List()), hash(List([0])))
    self.assertNotEqual(hash(List([0])), hash(List([1])))

    class A:
      pass

    a = A()
    b = A()
    self.assertNotEqual(hash(a), hash(b))
    self.assertNotEqual(hash(List([a])), hash(List([b])))

    class B:

      def __init__(self, value):
        self.value = value

      def __hash__(self):
        return hash((B, self.value))

    a = B(1)
    b = B(1)
    self.assertEqual(hash(a), hash(b))
    self.assertEqual(hash(List([B(1)])), hash(List([B(1)])))
    self.assertNotEqual(hash(List([B(1)])), hash(List([B(2)])))

  def test_sym_parent(self):
    sl = List([[0], dict(x=1)])
    self.assertIsNone(sl.sym_parent)

    self.assertIs(sl[0].sym_parent, sl)
    self.assertIs(sl[1].sym_parent, sl)

    pl = List([sl])
    self.assertIs(sl.sym_parent, pl)

  def test_sym_path(self):
    sl = List([dict(a=dict()), [dict(b=[0])]])
    self.assertEqual(sl.sym_path, '')
    self.assertEqual(sl[0].sym_path, '[0]')
    self.assertEqual(sl[0].a.sym_path, '[0].a')
    self.assertEqual(sl[1].sym_path, '[1]')
    self.assertEqual(sl[1][0].b.sym_path, '[1][0].b')

    sl.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(sl.sym_path, 'a')
    self.assertEqual(sl[0].sym_path, 'a[0]')
    self.assertEqual(sl[0].a.sym_path, 'a[0].a')
    self.assertEqual(sl[1].sym_path, 'a[1]')
    self.assertEqual(sl[1][0].b.sym_path, 'a[1][0].b')

  def test_accessor_writable(self):
    sl = List([0], accessor_writable=False)
    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot modify List item .* while accessor_writable is set to False.'):
      sl[0] = 2

    with flags.allow_writable_accessors(True):
      sl[0] = 2
      self.assertEqual(sl[0], 2)

    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot modify List item .* while accessor_writable is set to False.'):
      sl[0] = 1

    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot delete List item while accessor_writable is set to False.'):
      del sl[0]

    with flags.allow_writable_accessors(True):
      del sl[0]
      self.assertEqual(len(sl), 0)

    # Append/pop/extend is not impacted by accessor_writable flag.
    sl.append(0)
    sl.pop()
    sl.extend([1])
    self.assertEqual(sl, [1])

    sl.rebind({0: 2})
    self.assertEqual(sl, [2])

    # Delete key with rebind.
    sl.rebind({0: MISSING_VALUE})
    self.assertEqual(0, len(sl))

    sl.append(0)
    sl.set_accessor_writable(True)
    sl[0] = 1
    self.assertEqual(sl[0], 1)
    with flags.allow_writable_accessors(False):
      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot modify List .* while accessor_writable is set to False.'):
        sl[0] = 2

      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot delete List item while accessor_writable is set to False.'):
        del sl[0]

  def test_mark_missing_values(self):
    # For schemaless Dict.
    sl = List([0])
    self.assertEqual(len(sl), 1)

    # Set field to MISSING_VALUE will delete field.
    sl[0] = MISSING_VALUE
    self.assertEqual(len(sl), 0)

    # Clear will empty the list.
    sl.clear()
    self.assertEqual(len(sl), 0)

    # Mark missing values in symbolic list.
    spec = pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict([
            ('z', pg_typing.Bool(True)),
            ('p', pg_typing.Str())]))
    ]))
    sl = List.partial([{}], value_spec=spec)
    self.assertEqual(sl, [{
        'x': MISSING_VALUE,
        'y': {
            'z': True,
            'p': MISSING_VALUE,
        }
    }])
    sl[0].y.z = False

    # Assign MISSING_VALUE to a field with default value
    # will reset field to default value
    sl[0].y.z = MISSING_VALUE
    self.assertEqual(sl[0].y.z, True)

  def test_seal(self):
    sl = List([0], sealed=True)
    self.assertTrue(sl.is_sealed)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot append element on a sealed List'):
      sl.append(1)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot insert element into a sealed List'):
      sl.insert(0, 1)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot extend a sealed List'):
      sl.extend([0, 1])

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot delete item from a sealed List'):
      sl.pop()

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot set item for a sealed List'):
      sl[0] = 1

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot set item for a sealed List'):
      sl[:] = [1]

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot delete item from a sealed List'):
      del sl[0]

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot rebind key .* of sealed List'):
      sl.rebind({0: 3})

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot clear a sealed List.'):
      sl.clear()

    # Unseal.
    sl.seal(False)
    self.assertFalse(sl.is_sealed)

    # Test repeated seal has no side effect.
    sl.seal(False)
    self.assertFalse(sl.is_sealed)
    sl[0] = 2
    sl.append(3)
    sl.extend([4, 5, 6])
    sl.pop()
    del sl[-1]
    sl.rebind({-1: 7})
    self.assertEqual(sl, [2, 3, 7])

    sl = List([0], sealed=True)
    with flags.as_sealed(False):
      sl[0] = 2
      sl.append(3)
      sl.extend([4, 5, 6])
      sl.pop()
      del sl[-1]
      sl.rebind({-1: 7})
      self.assertEqual(sl, [2, 3, 7])

      sl.clear()
      self.assertEqual(len(sl), 0)

      # Object-level is_sealed flag is not modified.
      self.assertTrue(sl.is_sealed)

    # Test nested sealed List.
    sl = List([[]])
    self.assertFalse(sl.is_sealed)
    self.assertFalse(sl[0].is_sealed)
    sl.seal()
    self.assertTrue(sl.is_sealed)
    self.assertTrue(sl[0].is_sealed)


class RebindTest(unittest.TestCase):
  """Dedicated tests for `pg.List.rebind`."""

  def test_rebind_on_schemaless_lists(self):
    sl = List(['foo', 2, {}, []])
    sl.rebind({
        # Update sl[0].
        0: 'bar',
        # Delete sl[1] ('foo')
        '[1]': MISSING_VALUE,
        # Insert before sl[2] ({})
        '[2]': Insertion(100),
        # Insert before sl[3] ([])
        '[3]': Insertion(200),
        # Insert sl[2].x,
        '[2].x': True,
        # Insert sl[3][0].
        '[3][0]': 300,
    })
    self.assertEqual(sl, ['bar', 100, {'x': True}, 200, [300]])

  def test_rebind_with_typing(self):
    spec = pg_typing.List(pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Str(regex='foo.*')),
        ('c', pg_typing.Dict([
            ('x', pg_typing.Int(min_value=1, default=1)),
            ('y', pg_typing.Bool()),
        ]))
    ]))
    sl = List([dict(b='foo', c=dict(x=1, y=True))], value_spec=spec)
    sl.rebind({
        '[0].b': 'foo1',
        '[0].c.x': 2
    })
    self.assertEqual(sl, [{
        'a': 0,
        'b': 'foo1',
        'c': {'x': 2, 'y': True}
    }])
    with self.assertRaisesRegex(
        ValueError, '.* does not match regular expression'):
      sl.rebind({'[0].b': 'bar'})

    with self.assertRaisesRegex(ValueError, '.* is out of range'):
      sl.rebind({'[0].c.x': 0})

  def test_rebind_with_no_updates(self):
    def on_list_change(field_updates):
      del field_updates
      assert False
    sl = List([0, 1], onchange_callback=on_list_change)
    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind'):
      sl.rebind()
    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind'):
      sl.rebind(lambda k, v, p: v)
    sl.rebind(raise_on_no_change=False)

  def test_rebind_with_skipping_notification(self):
    def on_list_change(field_updates):
      del field_updates
      assert False
    sl = List([0, 1, 2], onchange_callback=on_list_change)
    sl.rebind({0: 100}, skip_notification=True)
    self.assertEqual(sl, [100, 1, 2])

  def test_rebind_with_field_updates_notification(self):
    updates = []
    def on_dict_change(field_updates):
      updates.append(field_updates)

    sl = List([
        1,
        Dict(x=1,
             y=Dict(onchange_callback=on_dict_change),
             onchange_callback=on_dict_change),
        List([Dict(p=1, onchange_callback=on_dict_change)],
             onchange_callback=on_dict_change),
        'foo',
    ], onchange_callback=on_dict_change)
    sl.rebind({
        '[0]': 2,
        '[1].x': 2,
        '[1].y.z': 1,
        '[2][0].p': MISSING_VALUE,
        '[2][0].q': 2,
        '[3]': 'foo',  # Unchanged.
        '[4]': Insertion('bar')
    })
    self.assertEqual(updates, [
        {  # Notification to `sl[2][0]`.
            'p': base.FieldUpdate(
                object_utils.KeyPath.parse('[2][0].p'),
                target=sl[2][0],
                field=None,
                old_value=1,
                new_value=MISSING_VALUE),
            'q': base.FieldUpdate(
                object_utils.KeyPath.parse('[2][0].q'),
                target=sl[2][0],
                field=None,
                old_value=MISSING_VALUE,
                new_value=2),
        },
        {  # Notification to `sl.c`.
            '[0].p': base.FieldUpdate(
                object_utils.KeyPath.parse('[2][0].p'),
                target=sl[2][0],
                field=None,
                old_value=1,
                new_value=MISSING_VALUE),
            '[0].q': base.FieldUpdate(
                object_utils.KeyPath.parse('[2][0].q'),
                target=sl[2][0],
                field=None,
                old_value=MISSING_VALUE,
                new_value=2),
        },
        {  # Notification to `sl[1].y`.
            'z': base.FieldUpdate(
                object_utils.KeyPath.parse('[1].y.z'),
                target=sl[1].y,
                field=None,
                old_value=MISSING_VALUE,
                new_value=1),
        },
        {  # Notification to `sl.b`.
            'x': base.FieldUpdate(
                object_utils.KeyPath.parse('[1].x'),
                target=sl[1],
                field=None,
                old_value=1,
                new_value=2),
            'y.z': base.FieldUpdate(
                object_utils.KeyPath.parse('[1].y.z'),
                target=sl[1].y,
                field=None,
                old_value=MISSING_VALUE,
                new_value=1),
        },
        {  # Notification to `sl`.
            '[0]': base.FieldUpdate(
                object_utils.KeyPath.parse('[0]'),
                target=sl,
                field=None,
                old_value=1,
                new_value=2),
            '[1].x': base.FieldUpdate(
                object_utils.KeyPath.parse('[1].x'),
                target=sl[1],
                field=None,
                old_value=1,
                new_value=2),
            '[1].y.z': base.FieldUpdate(
                object_utils.KeyPath.parse('[1].y.z'),
                target=sl[1].y,
                field=None,
                old_value=MISSING_VALUE,
                new_value=1),
            '[2][0].p': base.FieldUpdate(
                object_utils.KeyPath.parse('[2][0].p'),
                target=sl[2][0],
                field=None,
                old_value=1,
                new_value=MISSING_VALUE),
            '[2][0].q': base.FieldUpdate(
                object_utils.KeyPath.parse('[2][0].q'),
                target=sl[2][0],
                field=None,
                old_value=MISSING_VALUE,
                new_value=2),
            '[4]': base.FieldUpdate(
                object_utils.KeyPath.parse('[4]'),
                target=sl,
                field=None,
                old_value=MISSING_VALUE,
                new_value='bar')
        }
    ])

  def test_rebind_with_fn(self):
    sl = List([0, dict(x=1, y='foo', z=[2, 3, 4])])
    def increment(k, v, p):
      del k, p
      if isinstance(v, int):
        return v + 1
      return v
    sl.rebind(increment)
    self.assertEqual(sl, List([1, Dict(x=2, y='foo', z=[3, 4, 5])]))

  def test_notify_on_change(self):
    context = Dict(num_changes=0)
    def increment_change(unused_updates):
      context.num_changes += 1

    sl = List([0], onchange_callback=increment_change)
    sl.append(1)
    sl[1] = 3
    sl.insert(0, 2)
    del sl[0]
    sl.remove(3)
    self.assertEqual(context.num_changes, 5)

    context.num_changes = 0
    sl = List([0], onchange_callback=increment_change)
    with flags.notify_on_change(False):
      sl.append(1)
      sl[1] = 3
      sl.insert(0, 2)
      del sl[0]
      sl.remove(3)
    self.assertEqual(context.num_changes, 0)

  def test_bad_rebind(self):
    # Rebind is invalid on root object.
    with self.assertRaisesRegex(
        KeyError, 'Root key .* cannot be used in .*rebind.'):
      List().rebind({'': 1})

    # Rebind is invalid on non-symbolic object.
    with self.assertRaisesRegex(
        KeyError, 'Cannot rebind key .* is not a symbolic type.'):
      List([0]).rebind({'[0].x': 1})

    with self.assertRaisesRegex(
        ValueError, 'Argument \'path_value_pairs\' should be a dict.'):
      List().rebind(1)

    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind.'):
      List().rebind({})

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):
      List([0], value_spec=pg_typing.List(pg_typing.Int())).rebind({0: 1.0})


class SerializationTest(unittest.TestCase):
  """Dedicated tests for `pg.List` serialization."""

  def test_schemaless(self):
    sl = List([0, 'foo', None, dict(x=1)])

    # Key order is preserved.
    self.assertEqual(sl.to_json_str(), '[0, "foo", null, {"x": 1}]')

  def test_schematized(self):
    sl = List.partial(
        [dict(x=1)],
        value_spec=pg_typing.List(pg_typing.Dict([
            ('w', pg_typing.Str()),
            ('x', pg_typing.Int()),
            ('y', pg_typing.Str().noneable()),
            # Frozen field shall not be written.
            ('z', pg_typing.Bool(True).freeze()),
        ])))
    self.assertEqual(sl.to_json_str(), '[{"x": 1, "y": null}]')

  def test_serialization_with_converter(self):

    class A:

      def __init__(self, value: float):
        self.value = value

      def __eq__(self, other):
        return isinstance(other, A) and other.value == self.value

    spec = pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Object(A))
    ]))
    sl = List([dict(x=1, y=A(2.0))], value_spec=spec)
    with self.assertRaisesRegex(
        ValueError, 'Cannot convert complex type .* to JSON.'):
      sl.to_json_str()

    pg_typing.register_converter(A, float, convert_fn=lambda x: x.value)
    pg_typing.register_converter(float, A, convert_fn=A)

    self.assertEqual(sl.to_json(), [{'x': 1, 'y': 2.0}])
    self.assertEqual(List.from_json(sl.to_json(), value_spec=spec), sl)

    self.assertEqual(sl.to_json_str(), '[{"x": 1, "y": 2.0}]')
    self.assertEqual(base.from_json_str(sl.to_json_str(), value_spec=spec), sl)

  def test_serialization_with_tuple(self):
    self.assertEqual(base.to_json_str((1, 2)), '["__tuple__", 1, 2]')
    self.assertEqual(base.from_json_str('["__tuple__", 1]'), (1,))
    with self.assertRaisesRegex(
        ValueError,
        'Tuple should have at least one element besides \'__tuple__\'.'):
      base.from_json_str('["__tuple__"]')

  def test_hide_default_values(self):
    sl = List.partial(
        [dict(x=1)],
        value_spec=pg_typing.List(pg_typing.Dict([
            ('w', pg_typing.Str()),
            ('x', pg_typing.Int()),
            ('y', pg_typing.Str().noneable()),
            # Frozen field shall not be written.
            ('z', pg_typing.Bool(True).freeze()),
        ])))
    self.assertEqual(sl.to_json_str(hide_default_values=True), '[{"x": 1}]')

  def test_from_json(self):
    spec = pg_typing.List(pg_typing.Dict([
        ('w', pg_typing.Str()),
        ('x', pg_typing.Int()),
        ('y', pg_typing.Str().noneable()),
        # Frozen field shall not be written.
        ('z', pg_typing.Bool(True).freeze()),
    ]))
    self.assertEqual(
        base.from_json_str('[{"w": "foo", "x": 1}]').use_value_spec(spec),
        List.partial([dict(w='foo', x=1)], value_spec=spec))

  def test_to_json_on_regular_list(self):
    self.assertEqual(base.to_json_str([0, None, True]), '[0, null, true]')

  def test_unsupported_types(self):

    class A:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert complex type .* to JSON.'):
      base.to_json(List([A()]))


class FormatTest(unittest.TestCase):
  """Dedicated tests for `pg.Dict.format`."""

  def setUp(self):
    super().setUp()

    @pg_object.members([
        ('x', pg_typing.Int(1)),
        ('y', pg_typing.Str()),
        ('z', pg_typing.Dict([
            ('p', pg_typing.List(pg_typing.Bool().noneable())),
            ('q', pg_typing.Enum('foo', ['foo', 'bar'])),
            ('t', pg_typing.Str())
        ]))
    ])
    class A(pg_object.Object):
      pass

    spec = pg_typing.List(pg_typing.Dict([
        ('a1', pg_typing.Int(1), 'Field a1.'),
        ('a2', pg_typing.Dict([
            ('b1', pg_typing.Dict([
                ('c1', pg_typing.List(pg_typing.Dict([
                    ('d1', pg_typing.Str(), 'Field d1.'),
                    ('d2', pg_typing.Bool(True)),
                    ('d3', pg_typing.Object(A)),
                ])), 'Field c1.')]), 'Field b1.')]),
         'Field a2.')
    ]))
    self._list = List.partial([{
        'a1': 1,
        'a2': {
            'b1': {
                'c1': [{
                    'd3': A.partial(x=2, z={'p': [None, True], 't': 'foo'})
                }]
            }
        }
    }], value_spec=spec)

  def test_compact(self):
    self.assertEqual(
        self._list.format(compact=True),
        '[0: {a1=1, a2={b1={c1=[0: {d1=MISSING_VALUE, d2=True, d3='
        'A(x=2, y=MISSING_VALUE, z={p=[0: None, 1: True], '
        'q=\'foo\', t=\'foo\'})}]}}}]')

  def test_noncompact_nonverbose(self):
    self.assertEqual(
        self._list.format(compact=False, verbose=False),
        inspect.cleandoc("""[
          0 : {
            a1 = 1,
            a2 = {
              b1 = {
                c1 = [
                  0 : {
                    d1 = MISSING_VALUE(Str()),
                    d2 = True,
                    d3 = A(
                      x = 2,
                      y = MISSING_VALUE(Str()),
                      z = {
                        p = [
                          0 : None,
                          1 : True
                        ],
                        q = 'foo',
                        t = 'foo'
                      }
                    )
                  }
                ]
              }
            }
          }
        ]"""))

  def test_noncompact_verbose(self):
    self.assertEqual(
        self._list.format(compact=False, verbose=True),
        inspect.cleandoc("""[
          0 : {
            # Field a1.
            a1 = 1,

            # Field a2.
            a2 = {
              # Field b1.
              b1 = {
                # Field c1.
                c1 = [
                  0 : {
                    # Field d1.
                    d1 = MISSING_VALUE(Str()),
                    d2 = True,
                    d3 = A(
                      x = 2,
                      y = MISSING_VALUE(Str()),
                      z = {
                        p = [
                          0 : None,
                          1 : True
                        ],
                        q = 'foo',
                        t = 'foo'
                      }
                    )
                  }
                ]
              }
            }
          }
        ]"""))

  def test_noncompact_verbose_hide_default_and_missing_values(self):
    self.assertEqual(
        self._list.format(
            compact=False,
            verbose=True,
            hide_default_values=True,
            hide_missing_values=True),
        inspect.cleandoc("""[
          0 : {
            # Field a2.
            a2 = {
              # Field b1.
              b1 = {
                # Field c1.
                c1 = [
                  0 : {
                    d3 = A(
                      x = 2,
                      z = {
                        p = [
                          0 : None,
                          1 : True
                        ],
                        t = 'foo'
                      }
                    )
                  }
                ]
              }
            }
          }
        ]"""))


if __name__ == '__main__':
  unittest.main()
