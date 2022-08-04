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
"""Tests for pyglove.symbolic."""

import abc
import inspect
import os
import tempfile
import typing
import unittest

from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as schema


symbolic.allow_empty_field_description()
symbolic.allow_repeated_class_registration()


class DictTest(unittest.TestCase):
  """Tests for symbolic.Dict basics."""

  def testSchemaless(self):
    """Tests for schema-less Dict."""
    sd = symbolic.Dict()
    self.assertTrue(sd.accessor_writable)
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd, {})
    self.assertFalse(sd.is_sealed)

    # Test basic dict interfaces.
    # Test __setitem__.
    sd['b'] = {'c': True, 'd': []}
    with self.assertRaisesRegex(KeyError, 'Key must be string type'):
      sd[0] = 1

    # Test __setattr__.
    sd.a = 0

    # Test __len__.
    self.assertEqual(len(sd), 2)
    # Test __getitem__.
    self.assertEqual(sd['a'], 0)
    # Test __getattr__.
    self.assertEqual(sd.b.d, [])

    # Test get.
    self.assertEqual(sd.get('a', None), 0)
    self.assertIsNone(sd.get('x', None))

    # Tet __iter__.
    # Insertion order is preserved.
    self.assertEqual([k for k in sd], ['b', 'a'])

    # Test __contains__.
    self.assertIn('a', sd)
    self.assertNotIn('x', sd)

    # Test keys and values (insertion order are preserved.)
    self.assertEqual(list(sd.keys()), ['b', 'a'])
    self.assertEqual(list(sd.values()), [{'c': True, 'd': []}, 0])
    self.assertEqual(list(sd.items()), [('b', {'c': True, 'd': []}), ('a', 0)])

    # Test nested dict/list is converted into symbolic Dict/List.
    self.assertIsInstance(sd.b, symbolic.Dict)
    self.assertIsInstance(sd.b.d, symbolic.List)

    # Test __delitem__
    del sd['a']
    self.assertNotIn('a', sd)

    # __delitem__ on non-existing key.
    with self.assertRaisesRegex(KeyError, 'Key does not exist in Dict'):
      del sd['xyz']

    # __delattr__ on non-existing key.
    with self.assertRaisesRegex(KeyError, 'Key does not exist in Dict'):
      del sd.xyz

    # Test __delattr__
    del sd.b.c
    self.assertNotIn('c', sd.b)

    # Test pop.
    self.assertEqual(sd.pop('a', 'foo'), 'foo')
    with self.assertRaisesRegex(KeyError, 'a'):
      sd.pop('a')
    sd.a = 1
    self.assertEqual(sd.pop('a'), 1)

    # Test rebind
    sd.rebind({
        'x': 'bar',
        # Delete b.d.
        'b.d': schema.MISSING_VALUE
    })
    self.assertEqual(sd, {'x': 'bar', 'b': {}})

    # Test update
    sd.update({'x': schema.MISSING_VALUE, 'b.c': False})
    self.assertEqual(sd, {'b': {'c': False,}})
    sd.update({})
    self.assertEqual(sd, {'b': {'c': False,}})

    # Test clear.
    sd.clear()

    # Test empty.
    self.assertEqual(0, len(sd))

    # Test invalid args in constructor.
    with self.assertRaisesRegex(
        TypeError, 'Argument \'dict_obj\' must be dict type.'):
      symbolic.Dict(1)

  def testsymbolic(self):
    """Tests for Dict with value_spec."""
    value_spec = schema.Dict([
        ('a2', schema.Int()),
        ('a1',
         schema.Dict([('b1',
                       schema.Dict([('c1',
                                     schema.List(
                                         schema.Dict([
                                             ('d1', schema.Str()),
                                             ('d2', schema.Bool(True)),
                                         ])))]))])),
        (schema.StrKey(regex='x.*'), schema.Str())
    ])

    sd = symbolic.Dict.partial(value_spec=value_spec)
    self.assertTrue(sd.accessor_writable)
    self.assertIsNotNone(sd.value_spec)
    self.assertEqual(sd, {
        'a2': schema.MISSING_VALUE,
        'a1': {
            'b1': {
                'c1': schema.MISSING_VALUE
            }
        }
    })
    self.assertFalse(sd.is_sealed)

    # Test content-based __hash__.
    self.assertEqual(hash(sd), hash(symbolic.Dict(a1={'b1': {}})))

    # Test basic dict interfaces.
    # Test __setitem__.
    sd['x2'] = 'foo'
    sd['x1'] = 'bar'
    sd['a2'] = 1

    # Test __setattr__.
    sd.a1.b1.c1 = [{'d1': 'foo'}]

    self.assertEqual(
        sd, {
            'a2': 1,
            'a1': {
                'b1': {
                    'c1': [{
                        'd1': 'foo',
                        'd2': True
                    }]
                }
            },
            'x2': 'foo',
            'x1': 'bar'
        })

    # Test __len__.
    self.assertEqual(len(sd), 4)

    # Test __getitem__.
    self.assertEqual(sd['a2'], 1)

    # Test __getattr__.
    self.assertEqual(sd.a1.b1.c1[0].d1, 'foo')

    # Test get.
    self.assertEqual(sd.get('a2', None), 1)
    self.assertIsNone(sd.get('x', None))

    # Test __iter__.
    self.assertEqual([k for k in sd], ['a2', 'a1', 'x2', 'x1'])

    # Test __contains__.
    self.assertIn('a1', sd)
    self.assertNotIn('x', sd)

    # Test keys and values
    # The order are preserved to follow:
    # 1. const-keys in declaration order.
    # 2. non-const-keys in insertion order.
    self.assertEqual(list(sd.keys()), ['a2', 'a1', 'x2', 'x1'])
    self.assertEqual(
        list(sd.values()),
        [1, {
            'b1': {
                'c1': [{
                    'd1': 'foo',
                    'd2': True
                }]
            }
        }, 'foo', 'bar'])
    self.assertEqual(
        list(sd.items()), [('a2', 1),
                           ('a1', {
                               'b1': {
                                   'c1': [{
                                       'd1': 'foo',
                                       'd2': True
                                   }]
                               }
                           }), ('x2', 'foo'), ('x1', 'bar')])

    # Test nested dict/list is converted into symbolic Dict/List.
    self.assertIsInstance(sd.a1, symbolic.Dict)
    self.assertIsInstance(sd.a1.b1, symbolic.Dict)
    self.assertIsInstance(sd.a1.b1.c1, symbolic.List)
    self.assertIsInstance(sd.a1.b1.c1[0], symbolic.Dict)

    # Test __delitem__
    # Reset to default value.
    del sd['a2']
    self.assertEqual(sd['a2'], schema.MISSING_VALUE)

    # __delitem__ on non-existing key.
    with self.assertRaisesRegex(KeyError, 'Key does not exist in Dict'):
      del sd['xyz']

    # __delattr__ on non-existing key.
    with self.assertRaisesRegex(KeyError, 'Key does not exist in Dict'):
      del sd.xyz

    # Test __delattr__
    # Reset to default value.
    del sd.a1.b1
    self.assertEqual(sd.a1.b1, {'c1': schema.MISSING_VALUE})

    # Test pop.
    self.assertEqual(sd.pop('a2', 123), 123)
    # a2 is restored to MISSING_VALUE.
    self.assertEqual(sd.pop('a2'), schema.MISSING_VALUE)
    sd.a2 = 1
    self.assertEqual(sd.pop('a2'), 1)
    sd.rebind({'a1.b1.c1': [{'d2': False}]})
    self.assertFalse(sd.a1.b1.c1[0].pop('d2'))
    # d2 is restored to the default value True.
    self.assertTrue(sd.a1.b1.c1[0].pop('d2'))

    # Test simple rebind
    # More tests on `rebind` are in RebindTest.
    sd.rebind({'a2': 2, 'a1.b1.c1': [{'d2': False}]})
    self.assertEqual(
        sd, {
            'a2': 2,
            'a1': {
                'b1': {
                    'c1': [{
                        'd1': schema.MISSING_VALUE,
                        'd2': False
                    }]
                }
            },
            'x2': 'foo',
            'x1': 'bar',
        })

    # Test update
    sd.update({
        'a2': schema.MISSING_VALUE,
        # Reset a2.b1.c1[0].d2 to default value (True).
        'a1.b1.c1[0].d2': schema.MISSING_VALUE,
        'x2': 'foo2',
        # Even marking_missing_values is set to True,
        # 'x1' as a non-const key will be removed.
        'x1': schema.MISSING_VALUE
    })
    self.assertEqual(
        sd, {
            'a2': schema.MISSING_VALUE,
            'a1': {
                'b1': {
                    'c1': [{
                        'd1': schema.MISSING_VALUE,
                        'd2': True,
                    }]
                }
            },
            'x2': 'foo2'
        })

    # Test clear.
    # This set all members to their default values.
    sd.clear()
    self.assertEqual(sd, {
        'a2': schema.MISSING_VALUE,
        'a1': {
            'b1': {
                'c1': schema.MISSING_VALUE
            }
        }
    })

    # Test invalid args in constructor.
    with self.assertRaisesRegex(
        TypeError, 'Argument \'value_spec\' must be a schema.Dict type.'):
      symbolic.Dict({'a': 1}, 1)

  def testDictWithSpecialCharsAsKeys(self):
    """Test dict with special characters as keys."""
    # Test schema-free dict with special characters.
    sd = symbolic.Dict()
    sd['x.y'] = symbolic.Dict(z=symbolic.List())
    self.assertIn('x.y', sd)
    self.assertEqual(sd['x.y'].z.sym_path, '[x.y].z')
    self.assertEqual(sd.rebind({'[x.y].z': 2}), {
        'x.y': {
            'z': 2
        }
    })

    # Test dict special characters against a Dict schema.
    sd.clone().use_value_spec(schema.Dict())
    sd.clone().use_value_spec(schema.Dict([
        (schema.StrKey(), schema.Dict())
    ]))

  def testMixTypeCompatibility(self):
    """Test compatibility with Any type and Union type."""
    # Test compatibility with Any type.
    spec = schema.Dict([('x', schema.Any())])
    sd = symbolic.Dict(x={'a': 1, 'b': 2}, value_spec=spec)
    self.assertIsNone(sd.x.value_spec)

    # Test compatibility with Union type.
    spec = schema.Dict([('x',
                         schema.Union([
                             schema.Int(),
                             schema.Bool(),
                             schema.Dict([('a', schema.Int()),
                                          ('b', schema.Str().noneable())])
                         ]))])
    sd = symbolic.Dict(x={'a': 1}, value_spec=spec)
    self.assertEqual(sd.x, {'a': 1, 'b': None})
    self.assertEqual(
        sd.x.value_spec,
        schema.Dict([('a', schema.Int()), ('b', schema.Str().noneable())]))

  def testUseValueSpec(self):
    """Tests for Dict.use_value_spec."""
    spec = schema.Dict([
        ('a', schema.Int(min_value=0)),
        ('b', schema.Bool().noneable()),
    ])

    # Cannot apply schema multiple times.
    sd = symbolic.Dict(a=1, b=True)
    sd.use_value_spec(spec)
    # Apply the same schema twice to verify its eligibility.
    sd.use_value_spec(spec)

    with self.assertRaisesRegex(
        RuntimeError,
        'Dict is already bound with a different value spec: .*'):
      sd.use_value_spec(
          schema.Dict([
              ('a', schema.Int(min_value=2)),
              ('b', schema.Bool().noneable()),
          ]))

    # Remove schema constraint and insert new keys.
    sd.use_value_spec(None)
    sd['c'] = 1
    self.assertIn('c', sd)

  def testAccessorWritable(self):
    """Tests for Dict with accessor writable set to False."""
    sd = symbolic.Dict(a=0, accessor_writable=False)
    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot modify Dict field by attribute or key while accessor_writable '
        'is set to False.'):
      sd.a = 2

    with symbolic.allow_writable_accessors(True):
      sd.a = 2
      self.assertEqual(sd.a, 2)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot modify Dict field by attribute or key while accessor_writable '
        'is set to False.'):
      sd['a'] = 1

    with symbolic.allow_writable_accessors(True):
      sd['a'] = 1
      self.assertEqual(sd.a, 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot del Dict field by attribute or key while accessor_writable is '
        'set to False.'):
      del sd.a

    with symbolic.allow_writable_accessors(True):
      del sd.a
      self.assertNotIn('a', sd)
      sd.a = 1

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot del Dict field by attribute or key while accessor_writable is '
        'set to False.'):
      del sd['a']

    with symbolic.allow_writable_accessors(True):
      del sd['a']
      self.assertNotIn('a', sd)

    sd.rebind(a=2)
    self.assertEqual(sd.a, 2)

    # Delete key with rebind.
    sd.rebind(a=schema.MISSING_VALUE)
    self.assertEqual(0, len(sd))

    sd.set_accessor_writable(True)
    sd.a = 1
    self.assertEqual(sd.a, 1)
    with symbolic.allow_writable_accessors(False):
      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify Dict field by attribute or key while '
          'accessor_writable is set to False.'):
        sd.a = 2

      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify Dict field by attribute or key while '
          'accessor_writable is set to False.'):
        sd['a'] = 2

      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot del Dict field by attribute or key while accessor_writable is '
          'set to False.'):
        del sd.a

      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot del Dict field by attribute or key while accessor_writable is '
          'set to False.'):
        del sd['a']

  def testMarkMissingValues(self):
    # Test missing values are marked.

    # For schema-less Dict.
    sd = symbolic.Dict(x=1, y=2)
    self.assertIn('x', sd)
    self.assertIn('y', sd)

    # Set field to MISSING_VALUE will delete field.
    sd.x = schema.MISSING_VALUE
    self.assertNotIn('x', sd)

    # Clear will empty the dict.
    sd.clear()
    self.assertEqual(0, len(sd))

    # Mark missing values in symbolic Dict. (DEFAULT)
    value_spec = schema.Dict([('x', schema.Int()),
                              ('y',
                               schema.Dict([('z', schema.Bool(True)),
                                            ('p', schema.Str())]))])
    sd = symbolic.Dict.partial(value_spec=value_spec)
    self.assertEqual(sd, {
        'x': schema.MISSING_VALUE,
        'y': {
            'z': True,
            'p': schema.MISSING_VALUE,
        }
    })
    sd.y.z = False

    # Assign MISSING_VALUE to a field with default value
    # will reset field to default value
    sd.y.z = schema.MISSING_VALUE
    self.assertEqual(sd.y.z, True)

    # Do not mark missing values in schema-less Dict.
    sd = symbolic.Dict.partial(value_spec=value_spec)
    sd.y = schema.MISSING_VALUE
    sd.y.z = False

    # Assign MISSING_VALUE to a field with default value
    # will reset field to default value.
    sd.y.z = schema.MISSING_VALUE
    self.assertEqual(sd.y.z, True)

    # Clear will reset default values.
    sd.clear()
    self.assertEqual(sd, {
        'x': schema.MISSING_VALUE,
        'y': {
            'z': True,
            'p': schema.MISSING_VALUE,
        }
    })

  def testSealed(self):
    """Test seal/unseal a Dict."""
    sd = symbolic.Dict({'a': 0}, sealed=True)
    self.assertTrue(sd.is_sealed)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
      sd.a = 1

    with symbolic.as_sealed(False):
      sd.a = 2
      self.assertEqual(sd.a, 2)
      # Object-level is_sealed flag is not modified.
      self.assertTrue(sd.is_sealed)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
      sd['a'] = 1

    with symbolic.as_sealed(False):
      sd['a'] = 1
      self.assertEqual(sd['a'], 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      sd.rebind(a=1)

    with symbolic.as_sealed(False):
      sd.rebind(a=2)
      self.assertEqual(sd.a, 2)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      sd.update({'a': 1})

    with symbolic.as_sealed(False):
      sd.update({'a': 1})
      self.assertEqual(sd.a, 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot del item from a sealed Dict.'):
      del sd['a']

    with symbolic.as_sealed(False):
      del sd['a']
      self.assertNotIn('a', sd)
      sd.a = 1

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot del item from a sealed Dict.'):
      del sd.a

    with symbolic.as_sealed(False):
      del sd.a
      self.assertNotIn('a', sd)
      sd.a = 1

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot clear a sealed Dict.'):
      sd.clear()

    with symbolic.as_sealed(False):
      sd.clear()
      self.assertEqual(sd, {})

    # Unseal.
    sd.seal(False)
    self.assertFalse(sd.is_sealed)

    # Test repeated seal has no side effect.
    sd.seal(False)
    self.assertFalse(sd.is_sealed)

    sd.a = 2
    sd['b'] = 1
    self.assertEqual(sd.a, 2)
    self.assertEqual(sd.b, 1)
    sd.rebind(b=2)

    with symbolic.as_sealed(True):
      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify field of a sealed Dict.'):
        sd.a = 1

      # Object-level sealed state is not changed,
      self.assertFalse(sd.is_sealed)

    # Seal again.
    self.assertEqual(sd.b, 2)
    sd.seal()
    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      sd.rebind(a=0)

    # Test nested sealed Dict.
    sd = symbolic.Dict(a=symbolic.Dict())
    self.assertFalse(sd.is_sealed)
    self.assertFalse(sd.a.is_sealed)
    sd.seal()
    self.assertTrue(sd.is_sealed)
    self.assertTrue(sd.a.is_sealed)

  def testCustomApply(self):
    """Test Dict.custom_apply."""
    sd = symbolic.Dict.partial(
        value_spec=schema.Dict([('a', schema.Int()), (
            'b', schema.Dict()), ('c', schema.Dict([('x', schema.Int())]))]))
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sd.a = symbolic.Dict()

    sd.b = symbolic.Dict()
    sd.c = symbolic.Dict(x=1)

    # Target field cannot accept Dict of different schema.
    with self.assertRaisesRegex(
        KeyError, 'Keys .* are not allowed in Schema'):
      sd.c = symbolic.Dict(y=1)

    # Target field can accept Dict with the same schema.
    sd.c = symbolic.Dict(x=1, value_spec=schema.Dict([('x', schema.Int())]))

    # Target field cannot accept Dict with incompatible schema.
    with self.assertRaisesRegex(
        ValueError, 'Dict cannot be applied to a different spec.'):
      sd.c = symbolic.Dict(x=1, value_spec=schema.Dict())

    # Target field can accept any Dict.
    sd.b = symbolic.Dict(x=1, value_spec=schema.Dict([('x', schema.Int())]))

  def testSymbolicOperations(self):
    """Tests for symbolic operations."""
    a = symbolic.Dict(x=1, y=symbolic.Dict(z=2))
    self.assertTrue(a.sym_hasattr('x'))
    self.assertFalse(a.sym_hasattr('a'))
    self.assertEqual(a.sym_getattr('x'), 1)
    with self.assertRaisesRegex(
        AttributeError,
        '.* object has no symbolic attribute \'a\'.'):
      a.sym_getattr('a')

    self.assertIs(a.y.sym_parent, a)
    self.assertEqual(a.y.sym_path, 'y')
    a.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(a.sym_path, 'a')
    self.assertEqual(a.y.sym_path, 'a.y')

    a.sym_rebind(x=2)
    self.assertEqual(a.x, 2)
    self.assertEqual(next(a.sym_keys()), 'x')
    self.assertEqual(list(a.sym_keys()), ['x', 'y'])
    self.assertEqual(next(a.sym_values()), 2)
    self.assertEqual(list(a.sym_values()), [2, symbolic.Dict(z=2)])
    self.assertEqual(list(a.sym_items()),
                     [('x', 2), ('y', symbolic.Dict(z=2))])

    self.assertEqual(a.sym_clone(), symbolic.Dict(x=2, y={'z': 2}))
    self.assertTrue(a.sym_eq(a.clone()))
    self.assertTrue(a.sym_ne(symbolic.Dict))
    self.assertEqual(a.sym_hash(),
                     a.sym_clone(deep=True).sym_hash())

    a.sym_setorigin(symbolic.Dict.__init__, 'constructor')
    self.assertEqual(a.sym_origin.source, symbolic.Dict.__init__)
    self.assertEqual(a.sym_origin.tag, 'constructor')

    # Test symbolic and non-symbolic eq/ne.
    @symbolic.members([
        ('x', schema.Int())
    ])
    class A(symbolic.Object):

      def __eq__(self, other):
        if isinstance(other, int):
          return self.x == other
        return isinstance(other, A) and self.x == other.x

      def __ne__(self, other):
        return not self.__eq__(other)

    self.assertEqual(symbolic.Dict(a=A(x=1)), {'a': 1})
    self.assertFalse(symbolic.eq(symbolic.Dict(a=A(x=1)), {'a': 1}))
    self.assertTrue(symbolic.ne(symbolic.Dict(a=A(x=1)), {'a': 1}))
    self.assertNotEqual(symbolic.Dict(a=A(x=1)), {'a': 1.0})


class ListTest(unittest.TestCase):
  """Tests for symbolic.List basics."""

  def testSchemaless(self):
    """Tests for schema-less List."""
    sl = symbolic.List([0, 'foo'])
    self.assertTrue(sl.accessor_writable)
    self.assertIsNone(sl.value_spec)
    self.assertEqual(sl, [0, 'foo'])
    self.assertFalse(sl.is_sealed)

    # Test basic list interfaces.
    # Test __setitem__.
    sl[0] = {'c': True, 'd': []}

    # Old value is detached from object tree after reassignment.
    d = sl[0].d
    self.assertIsNotNone(d.sym_parent)
    sl[0].d = symbolic.List()
    self.assertIsNone(d.sym_parent)

    x = sl[0]
    self.assertIsNotNone(x.sym_parent)
    sl[0] = {'c': True, 'd': [1]}
    self.assertIsNone(x.sym_parent)

    # Test __getitem__.
    self.assertEqual(sl[0], {'c': True, 'd': [1]})
    self.assertIsInstance(sl[0], symbolic.Dict)
    self.assertIsInstance(sl[0].d, symbolic.List)

    # Test __len__.
    self.assertEqual(len(sl), 2)

    # Test __contains__.
    self.assertIn({'c': True, 'd': [1]}, sl)
    self.assertIn('foo', sl)
    self.assertNotIn(0, sl)
    self.assertNotIn(True, sl)

    # Test __eq__.
    self.assertEqual(sl, [{'c': True, 'd': [1]}, 'foo'])

    # Test __hash__.
    self.assertEqual(hash(sl), hash(sl))

    # Test __delitem__
    del sl[0]
    self.assertEqual(sl, ['foo'])

    # __delitem__ on non-existing key.
    with self.assertRaisesRegex(
        IndexError, 'list assignment index out of range'):
      del sl[2]

    # Test append.
    sl.append(1)
    self.assertEqual(sl, ['foo', 1])

    # Test pop.
    self.assertEqual(sl.pop(1), 1)
    self.assertEqual(sl, ['foo'])
    with self.assertRaisesRegex(IndexError, 'pop index out of range'):
      sl.pop(10)

    # Test remove.
    sl.insert(0, 'bar')
    self.assertEqual(sl, ['bar', 'foo'])

    # Test remove.
    sl.remove('foo')
    self.assertEqual(sl, ['bar'])

    # Test rebind
    sl.rebind({0: 1, 4: 2, 8: 3})
    self.assertEqual(sl, [1, 2, 3])

    # Test __init__.
    with self.assertRaisesRegex(
        TypeError, 'Argument \'items\' must be list type'):
      symbolic.List(1)

  def testsymbolic(self):
    """Tests for symbolic List."""
    value_spec = schema.List(
        schema.List(
            schema.Dict([
                ('a1', schema.List(schema.Int())),
                ('a2', schema.Bool(True)),
            ])),
        max_size=2)

    sl = symbolic.List.partial([[{}]], value_spec=value_spec)
    self.assertTrue(sl.accessor_writable)
    self.assertIsNotNone(sl.value_spec)
    self.assertFalse(sl.is_sealed)
    self.assertEqual(sl, [[{'a1': schema.MISSING_VALUE, 'a2': True}]])
    self.assertEqual(hash(sl), hash(symbolic.List(
        [[{'a1': schema.MISSING_VALUE, 'a2': True}]])))
    self.assertIsInstance(sl[0], symbolic.List)
    self.assertIsInstance(sl[0][0], symbolic.Dict)

    # Test basic list interfaces.
    # Test __setitem__.
    sl[0][0].a1 = [1, 2, 3]

    # Test __getitem__.
    self.assertIsInstance(sl[0][0].a1, symbolic.List)

    # Test __len__.
    self.assertEqual(len(sl), 1)
    self.assertEqual(len(sl[0][0].a1), 3)

    # Test __contains__.
    self.assertIn({'a1': [1, 2, 3], 'a2': True}, sl[0])
    self.assertNotIn(True, sl)

    # Test __delitem__
    # Reset to default value.
    del sl[0][0].a1[1]
    self.assertEqual(sl[0][0].a1, [1, 3])

    # __delitem__ on non-existing key.
    with self.assertRaisesRegex(
        IndexError, 'list assignment index out of range'):
      del sl[10]

    # Test append.
    sl.append([])
    self.assertEqual(len(sl), 2)
    self.assertIsInstance(sl[1], symbolic.List)

    with self.assertRaisesRegex(ValueError, 'List reached its max size'):
      sl.append([])

    # Test extend.
    with self.assertRaisesRegex(
        ValueError, 'Cannot extend List: the number of elements .* exceeds'):
      sl.extend([[]])

    # Test pop.
    sl.pop(1)
    self.assertEqual(len(sl), 1)

    # Test insert.
    sl.insert(0, [])
    self.assertEqual(sl, [[], [{'a1': [1, 3], 'a2': True}]])

    # Test remove.
    sl.remove([])
    self.assertEqual(len(sl), 1)

    # Test simple rebind
    # More tests on `rebind` are in RebindTest.
    sl.rebind({'[0][0].a2': False, '[2]': []})
    self.assertEqual(sl, [[{'a1': [1, 3], 'a2': False}], []])

    # Test __init__.
    with self.assertRaisesRegex(
        TypeError, 'Argument \'value_spec\' must be a schema.List type'):
      symbolic.List([0], value_spec=1)

    with self.assertRaisesRegex(
        ValueError, 'Length of list .* is greater than max size'):
      symbolic.List([0, 1], value_spec=schema.List(schema.Int(), max_size=1))

  def testMixTypeCompatibility(self):
    """Test compatibility with Any type and Union type."""
    # Test compatibility with Any type.
    spec = schema.Dict([('x', schema.Any())])
    sd = symbolic.Dict(x=[1, 2], value_spec=spec)
    self.assertIsNone(sd.x.value_spec)

    # Test compatibility with Union type.
    spec = schema.Dict([('x',
                         schema.Union([
                             schema.Int(),
                             schema.Bool(),
                             schema.List(
                                 schema.Dict([('a', schema.Int()),
                                              ('b', schema.Str().noneable())]))
                         ]))])
    sd = symbolic.Dict(x=[{'a': 1}], value_spec=spec)
    self.assertEqual(sd.x, [{'a': 1, 'b': None}])
    self.assertEqual(
        sd.x.value_spec,
        schema.List(
            schema.Dict([('a', schema.Int()), ('b', schema.Str().noneable())])))

  def testUseValueSpec(self):
    """Tests for List.use_value_spec."""
    spec = schema.List(
        schema.Dict([
            ('a', schema.Int(min_value=0)),
            ('b', schema.Bool().noneable()),
        ]))

    # Cannot apply schema multiple times.
    sl = symbolic.List([{'a': 1}])
    sl.use_value_spec(spec)
    # Apply the same schema twice to verify its eligibility.
    sl.use_value_spec(spec)
    with self.assertRaisesRegex(
        RuntimeError,
        'List is already bound with a different value spec: .*'):
      sl.use_value_spec(schema.List(schema.Int()))

    # Remove schema constraint from sl
    sl.use_value_spec(None)
    sl.append(1)
    self.assertEqual(sl[1], 1)

    # Fail to use_value_spec if schema does not match.
    sl2 = symbolic.List([1])
    with self.assertRaisesRegex(
        TypeError, 'Expect <(type|class) \'dict\'> but encountered '
        '<(type|class) \'int\'>: .*'):
      sl2.use_value_spec(spec)

  def testAccessorWritable(self):
    """Tests for List with accessor writable set to False."""
    sl = symbolic.List([1, 2, 3], accessor_writable=False)
    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot modify List item by __setitem__ while accessor_writable '
        'is set to False.'):
      sl[0] = 0

    with symbolic.allow_writable_accessors(True):
      sl[0] = 0
      self.assertEqual(sl[0], 0)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot delete List item while accessor_writable '
        'is set to False.'):
      del sl[0]

    with symbolic.allow_writable_accessors(True):
      del sl[0]
      self.assertEqual(sl, [2, 3])
      sl.insert(0, 1)

    sl.rebind({0: 0})
    self.assertEqual(sl[0], 0)

    # Delete key with rebind.
    sl.rebind({
        0: schema.MISSING_VALUE,
        2: schema.MISSING_VALUE,
    })
    self.assertEqual(sl, [2])

  def testMarkMissingValues(self):
    # Test mark_missing_values flags.

    # For schema-less List.
    sl = symbolic.List([1, 2, 3, 4, 5])

    # Set element to MISSING_VALUE will delete element.
    sl[0] = schema.MISSING_VALUE

    self.assertEqual(sl, [2, 3, 4, 5])
    sl.rebind({
        0: schema.MISSING_VALUE,  # value 2
        1: schema.MISSING_VALUE,  # value 3
        3: schema.MISSING_VALUE,  # value 5
    })
    self.assertEqual(sl, [4])

    # Mark missing values in symbolic List. (DEFAULT)
    value_spec = schema.List(
        schema.Dict([('x', schema.Int()),
                     ('y',
                      schema.Dict([('z', schema.Bool(True)),
                                   ('p', schema.Str())]))]))
    sl = symbolic.List.partial([{}], value_spec=value_spec)
    self.assertEqual(sl, [{
        'x': schema.MISSING_VALUE,
        'y': {
            'z': True,
            'p': schema.MISSING_VALUE,
        }
    }])
    sl.rebind({1: {'x': 1, 'y': {'z': False}}})
    self.assertEqual(sl, [{
        'x': schema.MISSING_VALUE,
        'y': {
            'z': True,
            'p': schema.MISSING_VALUE,
        }
    }, {
        'x': 1,
        'y': {
            'z': False,
            'p': schema.MISSING_VALUE,
        }
    }])

  def testSealed(self):
    """Test seal/unseal a List."""
    sl = symbolic.List([0, 99], sealed=True)
    self.assertTrue(sl.is_sealed)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot set item for a sealed List.'):
      sl[0] = 1

    with symbolic.as_sealed(False):
      sl[0] = 1
      self.assertEqual(sl[0], 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key 0 of sealed List.'):
      sl.rebind({0: 1})

    with symbolic.as_sealed(False):
      sl.rebind({0: 0})
      self.assertEqual(sl[0], 0)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot del item from a sealed List.'):
      del sl[0]

    with symbolic.as_sealed(False):
      del sl[0]
      self.assertEqual(len(sl), 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot insert element on a sealed List.'):
      sl.insert(0, 1)

    with symbolic.as_sealed(False):
      sl.insert(0, 1)
      self.assertEqual(len(sl), 2)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot extend a sealed List.'):
      sl.extend([1])

    with symbolic.as_sealed(False):
      sl.extend([1])
      self.assertEqual(len(sl), 3)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot del item from a sealed List.'):
      sl.pop(0)

    with symbolic.as_sealed(False):
      sl.pop(0)
      self.assertEqual(len(sl), 2)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot del item from a sealed List.'):
      sl.remove(99)

    with symbolic.as_sealed(False):
      sl.remove(99)
      self.assertEqual(len(sl), 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot append element on a sealed List.'):
      sl.append(11)

    with symbolic.as_sealed(False):
      sl.append(11)
      self.assertEqual(len(sl), 2)

    # Unseal.
    sl.seal(False)
    self.assertFalse(sl.is_sealed)
    # Repeated seal should have no side effect.
    sl.seal(False)
    self.assertFalse(sl.is_sealed)
    sl[0] = 1
    self.assertEqual(sl[0], 1)
    sl.rebind({1: 88, 2: 77})
    self.assertEqual(sl, [1, 88, 77])

    # Seal again.
    sl.seal()
    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key 0 of sealed List.'):
      sl.rebind({0: 1})

    # Seal nested structures.
    sl = symbolic.List([{'x': 1}])
    self.assertFalse(sl.is_sealed)
    self.assertFalse(sl[0].is_sealed)
    sl.seal()
    self.assertTrue(sl.is_sealed)
    self.assertTrue(sl[0].is_sealed)

  def testCustomApply(self):
    """Test List.custom_apply."""
    sd = symbolic.Dict.partial(
        value_spec=schema.Dict([(
            'a', schema.Int()), ('b', schema.List(schema.Any(
            ))), ('c', schema.List(schema.Dict([('x', schema.Int())])))]))
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sd.a = symbolic.List()

    sd.b = symbolic.List([1])
    sd.b = symbolic.List(['str'])
    sd.c = symbolic.List([{'x': 1}])

    # Target field cannot accept List of different schema.
    with self.assertRaisesRegex(
        KeyError, 'Keys .* are not allowed in Schema'):
      sd.c = symbolic.List([{'y': 1}])

    # Target field can accept List with the same schema.
    sd.c = symbolic.List(
        [symbolic.Dict(x=1, value_spec=schema.Dict([('x', schema.Int())]))])

    # Target field cannot accept List with incompatible schema.
    with self.assertRaisesRegex(
        ValueError,
        'List cannot be applied to an incompatible value spec.'):
      sd.c = symbolic.List([1], value_spec=schema.List(schema.Int()))

  def testSymbolicOperations(self):
    """Tests for symbolic operations."""
    a = symbolic.List([symbolic.Dict(), 1])
    self.assertTrue(a.sym_hasattr(0))
    self.assertFalse(a.sym_hasattr(2))
    self.assertFalse(a.sym_hasattr('abc'))
    self.assertEqual(a.sym_getattr(1), 1)
    with self.assertRaisesRegex(
        AttributeError, '.* object has no symbolic attribute 2.'):
      a.sym_getattr(2)

    self.assertIs(a[0].sym_parent, a)
    self.assertEqual(a[0].sym_path, '[0]')
    a.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(a.sym_path, 'a')
    self.assertEqual(a[0].sym_path, 'a[0]')

    a.sym_rebind({'[1]': 2})
    self.assertEqual(a[1], 2)
    self.assertEqual(next(a.sym_keys()), 0)
    self.assertEqual(list(a.sym_keys()), [0, 1])
    self.assertEqual(next(a.sym_values()), symbolic.Dict())
    self.assertEqual(list(a.sym_values()), [symbolic.Dict(), 2])
    self.assertEqual(list(a.sym_items()), [(0, symbolic.Dict()), (1, 2)])

    self.assertEqual(a.sym_clone(),
                     symbolic.List([symbolic.Dict(), 2]))
    self.assertTrue(a.sym_eq(a.clone()))
    self.assertTrue(a.sym_ne(symbolic.List))
    self.assertEqual(a.sym_hash(),
                     a.sym_clone(deep=True).sym_hash())

    a.sym_setorigin(symbolic.List.__init__, 'constructor')
    self.assertEqual(a.sym_origin.source, symbolic.List.__init__)
    self.assertEqual(a.sym_origin.tag, 'constructor')

    # Test symbolic and non-symbolic eq/ne.
    @symbolic.members([
        ('x', schema.Int())
    ])
    class A(symbolic.Object):

      def __eq__(self, other):
        if isinstance(other, int):
          return self.x == other
        return isinstance(other, A) and self.x == other.x

      def __ne__(self, other):
        return not self.__eq__(other)

    self.assertEqual(symbolic.List([A(x=1)]), [1])
    self.assertFalse(symbolic.eq(symbolic.List([A(x=1)]), [1]))
    self.assertTrue(symbolic.ne(symbolic.List([A(x=1)]), [1]))
    self.assertNotEqual(symbolic.List([A(x=1)]), [1.0])

    # Test symbolic eq/ne on symbolic objects that does not use
    # symbolic comparison as the default eq/ne behavior.

    @symbolic.members([
        ('x', schema.Int())
    ])
    class B(symbolic.Object):
      allow_symbolic_comparison = False

      def __eq__(self, other):
        return isinstance(other, int) and self.x == other

      def __ne__(self, other):
        return not self.__eq__(other)

    self.assertEqual(B(1), 1)
    self.assertNotEqual(B(1), B(1))
    self.assertNotEqual([B(1)], [B(1)])
    self.assertNotEqual((B(1),), (B(1),))
    self.assertNotEqual(dict(x=B(1)), dict(x=B(1)))

    self.assertTrue(symbolic.eq(B(1), B(1)))
    self.assertTrue(symbolic.eq(B(1), 1))
    self.assertTrue(symbolic.eq(1, B(1)))
    self.assertTrue(symbolic.eq([B(1)], [B(1)]))
    self.assertTrue(symbolic.eq([B(1)], [1]))
    self.assertTrue(symbolic.eq([1], [B(1)]))
    self.assertTrue(symbolic.eq((B(1),), (B(1),)))
    self.assertTrue(symbolic.eq((B(1),), (1,)))
    self.assertTrue(symbolic.eq((1,), (B(1),)))
    self.assertTrue(symbolic.eq(dict(x=B(1)), dict(x=B(1))))
    self.assertTrue(symbolic.eq(dict(x=B(1)), dict(x=1)))
    self.assertTrue(symbolic.eq(dict(x=1), dict(x=B(1))))

    self.assertTrue(symbolic.ne([B(1)], (B(1),)))
    self.assertTrue(symbolic.ne([B(1)], [B(1), B(1)]))
    self.assertTrue(symbolic.ne([B(1)], [2]))
    self.assertTrue(symbolic.ne((B(1),), [B(1)]))
    self.assertTrue(symbolic.ne((B(1),), (B(1), B(1))))
    self.assertTrue(symbolic.ne((B(1),), 1))
    self.assertTrue(symbolic.ne(dict(x=B(1)), [B(1)]))
    self.assertTrue(symbolic.ne(dict(x=B(1)), dict(x=B(1), y=B(1))))
    self.assertTrue(symbolic.ne(dict(x=B(1)), dict(y=B(1))))
    self.assertTrue(symbolic.ne(dict(x=B(1)), dict(x=2)))


class ObjectTest(unittest.TestCase):
  """Basic tests for symbolic.Object."""

  def testBasics(self):
    """Test basic interfaces of symbolic.Object."""

    @symbolic.members([
        ('c', schema.Dict([('d', schema.Enum('foo', ['foo', 'bar']))])),
        ('a', schema.Int()),
        ('b', schema.Str().noneable())
    ])
    class A(symbolic.Object):
      pass

    # Verify A's schema.
    self.assertEqual(
        A.schema,
        schema.create_schema([
            ('c', schema.Dict([('d', schema.Enum('foo', ['foo', 'bar']))])),
            ('a', schema.Int()), ('b', schema.Str().noneable())
        ]))
    self.assertEqual(A.schema.metadata['init_arg_list'], ['c', 'a', 'b'])

    # Test __init__.
    self.assertEqual(
        A(dict(d='bar'), 1),
        A(c=dict(d='bar'), a=1, b=None))

    # Test bad __init__.
    with self.assertRaisesRegex(
        TypeError, '.* missing 1 required argument: \'a\''):
      A()

    with self.assertRaisesRegex(
        TypeError, 'Expect bool type for argument \'allow_partial\''):
      A(a=1, allow_partial='no')

    with self.assertRaisesRegex(
        TypeError, '.* got unexpected keyword arguments'):
      A(x=1, y=2)

    with self.assertRaisesRegex(
        TypeError, '.* takes 3 positional arguments but 4 were given'):
      A(1, 2, 3, 4)

    with self.assertRaisesRegex(
        TypeError, '.* got multiple values for argument \'c\''):
      A(1, c=2)

    class B(symbolic.Object):
      pass

    with self.assertRaisesRegex(
        TypeError, '.* takes no arguments.'):
      B(1)

    # Construct A by putting argument 'a' before 'c'
    # and verify key order aligns with field order.
    a = A(a=1, c={'d': 'foo'})
    self.assertEqual(a.keys(), ['c', 'a', 'b'])

    # Test __getattr__.
    self.assertEqual(a.a, 1)
    self.assertIsNone(a.b)
    with self.assertRaisesRegex(
        AttributeError, '\'A\' object has no attribute \'d\''):
      _ = a.d

    # Test __getitem__.
    self.assertEqual(a['c'], {'d': 'foo'})

    # Test __setattr__.
    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot set attribute of .* while .*allow_symbolic_assignment.* '
        'is set to False'):
      a.b = 'hi'

    # Test detached object has no parent.
    c = a.c
    a.rebind(c=symbolic.Dict(d='bar'))
    self.assertIsNone(c.sym_parent)

    a = A(a=0)
    a.set_accessor_writable(True)
    a.b = 'hi'
    self.assertEqual(a.b, 'hi')

    # Test partial.
    a = A.partial()
    self.assertEqual(a.a, schema.MISSING_VALUE)

    # Test keys()
    self.assertEqual(a.keys(), ['c', 'a', 'b'])

    # Test __contains__.
    self.assertIn('b', a)
    self.assertNotIn('d', a)

    # Test __iter__.
    self.assertEqual([(k, v) for k, v in a], [('c', {
        'd': 'foo'
    }), ('a', schema.MISSING_VALUE), ('b', None)])

    # Test __eq__.
    self.assertEqual(A(a=1), A(a=1))
    self.assertNotEqual(A.partial(b='foo'), A.partial())

    # Test __hash__.
    self.assertEqual(hash(A(a=1)), hash(A(a=1)))
    self.assertNotEqual(hash(A.partial(b='foo')), hash(A.partial()))

    # Test bad arguments for `pg.members`.
    with self.assertRaisesRegex(
        TypeError, 'Unsupported keyword arguments'):

      @symbolic.members([
          ('x', schema.Int())
      ], unsupported_keyword=1)
      class C(symbolic.Object):  # pylint: disable=unused-variable
        pass

    # Test missing description for pg.members.
    symbolic.allow_empty_field_description(False)

    class D(symbolic.Object):
      pass

    with self.assertRaisesRegex(
        ValueError, 'Field description must not be empty'):
      symbolic.members([('a', schema.Int())])(D)
    symbolic.allow_empty_field_description(True)

  def testInitArgList(self):
    """Test init_arg_list."""

    class A(symbolic.Object):
      pass

    self.assertEqual(A.init_arg_list, [])
    self.assertEqual(A(), A())

    @symbolic.members([
        ('x', schema.Any()),
        ('z', schema.Any()),
    ])
    class B(A):
      pass

    self.assertEqual(B.init_arg_list, ['x', 'z'])
    self.assertEqual(B(1, 2), B(x=1, z=2))
    self.assertEqual(B(1, 2).sym_init_args, symbolic.Dict(x=1, z=2))

    @symbolic.members([
        ('x', schema.Int()),
        ('y', schema.Any()),
    ])
    class C(B):
      pass

    self.assertEqual(C.init_arg_list, ['x', 'z', 'y'])
    self.assertEqual(C(1, 2, 3), C(x=1, z=2, y=3))
    self.assertEqual(C(1, 2, 3).sym_init_args, symbolic.Dict(x=1, z=2, y=3))

    class D(C):
      pass

    self.assertEqual(D.init_arg_list, ['x', 'z', 'y'])
    self.assertEqual(D(1, 2, 3), D(x=1, z=2, y=3))

    @symbolic.members([
        ('p', schema.Any()),
        ('y', schema.Int()),
    ])
    class E(D):
      pass

    self.assertEqual(E.init_arg_list, ['x', 'z', 'y', 'p'])
    self.assertEqual(E(1, 2, 3, 4), E(x=1, z=2, y=3, p=4))

    @symbolic.members([
    ], init_arg_list=['x', 'y', 'z'])
    class F(E):
      pass

    self.assertEqual(F.init_arg_list, ['x', 'y', 'z'])
    self.assertEqual(F(1, 2, 3, p=4), F(x=1, y=2, z=3, p=4))

    with self.assertRaisesRegex(
        TypeError, '.* takes 3 positional arguments but 4 were given'):
      _ = F(1, 2, 3, 4)

    # Inherit init_arg_list from the parent class if there is no
    # extra field declared.
    @symbolic.members([])
    class G(F):
      pass

    self.assertEqual(G.init_arg_list, ['x', 'y', 'z'])

    # Inherit init_arg_list from the parent class if new fields are only
    # overrides of existing fields.
    @symbolic.members([
        ('z', schema.Int(default=1))
    ])
    class H(G):
      pass

    self.assertEqual(H.init_arg_list, ['x', 'y', 'z'])

    # Invalidate init_arg_list as new field is added.
    @symbolic.members([
        ('q', schema.Int())
    ])
    class I(H):
      pass

    self.assertEqual(I.init_arg_list, ['x', 'z', 'y', 'p', 'q'])

    # Use metadata to rearrange init_arg_list.
    # This is for legacy use cases.
    @symbolic.members([
        ('q', schema.Int())
    ], metadata={
        'init_arg_list': ['z', 'x', 'y', 'q', 'p']
    })
    class J(I):
      pass

    self.assertEqual(J.init_arg_list, ['z', 'x', 'y', 'q', 'p'])

    # Bad argument name in init_arg_list.
    with self.assertRaisesRegex(
        TypeError, 'Argument \'z\' from `init_arg_list` is not defined.'):

      @symbolic.members([
          ('x', schema.Int()),
          ('y', schema.Int())
      ], init_arg_list=['x', 'z'])
      class K(symbolic.Object):  # pylint: disable=unused-variable
        pass

    # Bad variable positional argument.
    with self.assertRaisesRegex(
        TypeError,
        'Variable positional argument \'y\' should be declared '
        'with `pg.typing.List'):

      @symbolic.members([
          ('x', schema.Int()),
          ('y', schema.Int())
      ], init_arg_list=['x', '*y'])
      class L(symbolic.Object):  # pylint: disable=unused-variable
        pass

  def testParentOfMembers(self):
    """Test parent property of Object members."""

    class A(symbolic.Object):
      pass

    @symbolic.members([
        ('x', schema.Any().noneable()),
        ('y', schema.Any().noneable())
    ])
    class B(symbolic.Object):
      allow_symbolic_assignment = True

    # Test parent of members set at __init__ time.
    b = B(x=A())
    self.assertIs(b.x.sym_parent, b)

    # Test parent of members set at __setitem__ time.
    b.x = A()
    self.assertIs(b.x.sym_parent, b)

    # Test parent of members set at rebind time.
    b.rebind(y=A())
    self.assertIs(b.y.sym_parent, b)

  def testWildcardKey(self):
    """Test wildcard key as object members."""

    @symbolic.members([(schema.StrKey(regex='.*foo'), schema.Int())])
    class A(symbolic.Object):
      pass

    a = A(foo=1)
    self.assertEqual(a.foo, 1)
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      A(foo='abc')

    a = A()
    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot set attribute of .* while .*allow_symbolic_assignment.* '
        'is set to False'):
      a.foo = 'abc'

    a = A()
    a.set_accessor_writable(True)
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      a.foo = 'abc'

    a = A()
    # Okay: 'bar' is treated as non symbolic members.
    a.bar = 'abc'
    # Okay: private member are not treated as symbolic member though
    # it matches the key regex.
    a._foo = 'abc'

  def testAbsentDecorator(self):
    """Test symbolic.Object with absent @schema decorator."""

    class A(symbolic.Object):
      pass

    # Class without members.
    self.assertEqual(A.schema, schema.Schema([]))

    @symbolic.members([('x', schema.Int())])
    class B(symbolic.Object):
      pass

    self.assertEqual(B.schema, schema.create_schema([('x', schema.Int())]))

    # Class with inherited members
    class C(B):
      pass

    self.assertEqual(C.schema, B.schema)

    # Class with further inheritance.
    @symbolic.members([('y', schema.Str())])
    class D(C):
      pass

    self.assertEqual(
        D.schema,
        schema.create_schema([('x', schema.Int()), ('y', schema.Str())]))

  def testMixTypeCompatibility(self):
    """Test compatibility with Any type and Union type."""

    class A(symbolic.Object):
      pass

    spec = schema.Dict([('x', schema.Any())])
    sd = symbolic.Dict(x=A(), value_spec=spec)
    self.assertIsInstance(sd.x, A)

    # Test compatibility with Union type.
    spec = schema.Dict([
        ('x', schema.Union([schema.Int(),
                            schema.Bool(),
                            schema.Object(A)]))
    ])
    sd = symbolic.Dict(x=A(), value_spec=spec)
    self.assertEqual(sd.x, A())

  def testAllowSymbolicAssignment(self):
    """Tests for Object.allow_symbolic_assignment."""

    @symbolic.members([('x', schema.Int(default=1))])
    class A(symbolic.Object):
      pass

    a = A(x=1)
    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot set attribute of <class A> while '
        '`A.allow_symbolic_assignment` is set to False'):
      a.x = 2

    with symbolic.allow_writable_accessors(True):
      a.x = 2
      self.assertEqual(a.x, 2)

    with self.assertRaisesRegex(AttributeError, 'can\'t delete attribute'):
      del a.x

    a.rebind({'x': 2})
    self.assertEqual(a.x, 2)

    # We still can access non-symbolic members.
    a.y = 1
    with symbolic.allow_writable_accessors(False):
      a.z = 2
    self.assertEqual(a.y, 1)
    self.assertEqual(a.z, 2)

  def testMarkMissingValues(self):
    # Test missing values are marked.

    # Mark missing values in symbolic Dict. (DEFAULT)
    @symbolic.members([
        ('x', schema.List(schema.Dict([('q', schema.Int())]))),
        ('y', schema.Dict([('z', schema.Bool(True)), ('p', schema.Str())]))
    ])
    class A(symbolic.Object):
      pass

    a = A.partial(x=[{}])
    self.assertEqual(a.x, [{'q': schema.MISSING_VALUE}])
    self.assertEqual(a.y, {
        'z': True,
        'p': schema.MISSING_VALUE,
    })

    a.rebind({'y.z': False})
    self.assertFalse(a.y.z)

    # Assign MISSING_VALUE to a field with default value
    # will reset field to default value
    a.rebind({'y.z': schema.MISSING_VALUE})
    self.assertEqual(a.y.z, True)

  def testSealed(self):
    """Test seal/unseal a Object."""

    @symbolic.members([('a', schema.Int()), ('b', schema.Dict())])
    class A(symbolic.Object):
      allow_symbolic_assignment = True
      allow_symbolic_mutation = False

    a = A(a=0, b={})

    self.assertTrue(a.is_sealed)
    self.assertTrue(a.b.is_sealed)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError,
        'Cannot set attribute .*: object is sealed.'):
      a.a = 1

    with symbolic.as_sealed(False):
      a.a = 2
      self.assertEqual(a.a, 2)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
      a['a'] = 1

    with symbolic.as_sealed(False):
      a['a'] = 1
      self.assertEqual(a.a, 1)

    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      a.rebind(a=1)

    with symbolic.as_sealed(False):
      a.rebind(a=2)
      self.assertEqual(a.a, 2)

    # Unseal.
    a.seal(False)
    self.assertFalse(a.is_sealed)
    self.assertFalse(a.b.is_sealed)
    a.a = 2
    self.assertEqual(a.a, 2)
    a.rebind(a=3)
    self.assertEqual(a.a, 3)

    # Seal again.
    a.seal()
    with self.assertRaisesRegex(
        symbolic.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      a.rebind(a=0)

  def testInheritance(self):
    """Test schema inheritance."""

    class A(symbolic.Object):
      pass

    self.assertEqual(A.schema, schema.create_schema([]))

    @symbolic.members([
        ('a', schema.Int(min_value=0)),
        ('b', schema.Bool().noneable()),
        ('c', schema.Str()),
        ('d', schema.Object(A)),
        ('e', schema.Union([
            schema.Int(),
            schema.Tuple([schema.Int(), schema.Int()])])),
        ('f', schema.Enum('a', ['a', 'b', 'c'])),
        ('g', schema.Any())
    ])
    class B(A):
      pass

    class C(B):
      pass

    self.assertEqual(C.schema, B.schema)

    @symbolic.members([
        ('a', schema.Int(max_value=10, default=5)),
        ('b', schema.Bool()),
        ('c', schema.Str(regex='foo.*')),
        ('d', schema.Object(B)),
        ('e', schema.Tuple([schema.Int(), schema.Int()])),
        ('f', schema.Enum('a', ['a', 'b'])),
        ('g', schema.Int())
    ])
    class D(C):
      pass

    self.assertEqual(
        D.schema,
        schema.create_schema([
            ('a', schema.Int(min_value=0, max_value=10, default=5)),
            ('b', schema.Bool()),
            ('c', schema.Str(regex='foo.*')),
            ('d', schema.Object(B)),
            ('e', schema.Tuple([schema.Int(),
                                schema.Int()])),
            ('f', schema.Enum('a', ['a', 'b'])),
            ('g', schema.Int())]))

    # pylint: disable=unused-variable
    with self.assertRaisesRegex(
        TypeError,
        '.* cannot extend .*: min_value .* is greater than max_value .* '
        'after extension.'):

      @symbolic.members([
          ('a', schema.Int(max_value=-1)),
      ])
      class E1(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible type'):

      @symbolic.members([('b', schema.Int())])
      class E2(C):
        pass

    with self.assertRaisesRegex(
        TypeError,
        '.* cannot extend .*: None is not allowed in base spec'):

      @symbolic.members([('c', schema.Str().noneable())])
      class E3(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: incompatible class.'):

      @symbolic.members([
          ('d', schema.Object(symbolic.Object)),
      ])
      class E4(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec.'):

      @symbolic.members([
          ('e', schema.Int().noneable()),
      ])
      class E51(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: no compatible type found in Union'):

      @symbolic.members([
          ('e', schema.Str()),
      ])
      class E52(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: None is not allowed in base spec'):

      @symbolic.members([
          ('e', schema.Tuple([schema.Int().noneable(), schema.Int()]))
      ])
      class E53(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: no compatible type found in Union.'):

      @symbolic.members([('e', schema.Tuple([schema.Int(), schema.Str()]))])
      class E54(C):
        pass

    with self.assertRaisesRegex(
        TypeError, '.* cannot extend .*: values in base should be super set.'):

      @symbolic.members([('f', schema.Enum('a', ['a', 'b', 'c', 'd']))])
      class E6(C):
        pass

    @symbolic.members([('x', schema.Int())])
    class E7(symbolic.Object):

      def __init__(self, x):
        self.x = x

    with self.assertRaisesRegex(
        ValueError, r'.* should call `super.*.__init__`.'):
      E7(x=1)

  def testMultiInheritance(self):
    """Test multi-inheritance."""

    @symbolic.members([('a', schema.Int())])
    class A(symbolic.Object):

      def _on_bound(self):
        super()._on_bound()
        self.x = self.a

    @symbolic.members([('b', schema.Float())])
    class B(symbolic.Object):

      def _on_bound(self):
        super()._on_bound()
        self.y = self.b

    # Multi-inheritance among two symbolic bases.
    class C(A, B):
      pass

    # Test schema without `members` decorator.
    self.assertEqual(
        C.schema,
        schema.create_schema([('a', schema.Int()), ('b', schema.Float())]))

    c = C(a=1, b=2.0)
    self.assertEqual(c.a, 1)
    self.assertEqual(c.b, 2.0)
    # Test both `_on_bound` have been triggered.
    self.assertEqual(c.a, c.x)
    self.assertEqual(c.b, c.y)

    class A2:
      """Non-symbolic base."""

      def __init__(self):
        super().__init__()
        self.x = getattr(self, 'x', 1)

    # Test multi-inheritance among non-symbolic/symbolic classes.
    class C2(B, A2):
      pass

    c2 = C2(b=2.0)  # pylint: disable=unexpected-keyword-arg
    # Test B.__init__ is triggered.
    self.assertEqual(c2.b, 2.0)
    # Test A2.__init__ is triggered.
    self.assertEqual(c2.x, 1)

    class C3(A2, B):
      pass

    # A2.__init__ rejects argument 'b'.
    with self.assertRaisesRegex(
        TypeError, r'__init__\(\) got an unexpected keyword argument'):
      C3(b=2.0)  # pylint: disable=unexpected-keyword-arg

    # B.__init__ expects argument 'b'.
    with self.assertRaisesRegex(
        TypeError, r'__init__\(\) missing 1 required argument: \'b\''):
      C3()

    # Test multi-inheritance among bases with different arguments.
    class A3:
      """Non-symbolic base with argument."""

      def __init__(self, a):
        super().__init__()
        self.a = a

    # Since symbolic object has flag `explicit_init` which can decide
    # whether to relay __init__ call to the next base class or not, it's
    # suggested to put symbolic base at the front.
    class C4(B, A3):
      """Mixed subclass with explicit __init__."""

      def __init__(self, a, b):
        A3.__init__(self, a)
        B.__init__(self, b=b, explicit_init=True)  # pylint: disable=non-parent-init-called

    c4 = C4(1, 2.0)
    self.assertEqual(c4.a, 1)
    self.assertEqual(c4.b, 2.0)

    # Bad __init__ without explicit_init=True.
    class C5(B, A3):
      """Mixed subclass with explicit __init__."""

      def __init__(self, a, b):
        A3.__init__(self, a)
        # BE CAREFUL: explicit_init is not called, thus B.__init__
        # will eventually invoke `super().__init__`
        # which will implicit call into A3.__init__ again with no argument.
        B.__init__(self, b=b)  # pylint: disable=non-parent-init-called

    # A3.__init__ (2nd call) expects argument 'a'.
    with self.assertRaisesRegex(
        TypeError, r'__init__\(\) '
        r'(takes exactly 2 arguments|missing 1 required positional argument)'):
      C5(1, 2.0)

    # Bad order.
    # Reverse base class order which creates issue
    # since A3.__init__ relay call to B.__init__ (implicitly by super.__init__)
    class C6(A3, B):
      """Mixed subclass: symbolic base is put after non-symbolic base."""

      def __init__(self, a, b):
        A3.__init__(self, a)
        B.__init__(self, b=b, explicit_init=True)  # pylint: disable=non-parent-init-called

    # B.__init__ (1st call) expects 'b', (`A3.__init__` calls
    # `super.__init__`, which implicitly call `B.__init__` with no argument.)
    with self.assertRaisesRegex(
        TypeError, r'C6.__init__\(\) missing 1 required argument: \'b\''):
      C6(1, 2.0)

  def testDerivedFields(self):
    """Test auto update of derived fields."""

    @symbolic.members([('a', schema.Int()), ('b', schema.Int())])
    class A(symbolic.Object):

      def _on_bound(self):
        self._sum = self.a + self.b

      @property
      def sum(self):
        return self._sum

    a = A(a=0, b=1)
    self.assertEqual(a.sum, 1)
    self.assertEqual(a.rebind(a=1).sum, 2)

  def testInnerAttributeError(self):
    """Tests the error message when attribute error happen inside a property."""

    class A(symbolic.Object):

      @property
      def foo(self):
        # `_bar` does not exist in A.
        return self._bar

    with self.assertRaisesRegex(
        AttributeError, '\'A\' object has no attribute \'_bar\''):
      _ = A().foo

  def testAutomaticRegistration(self):
    """Test automatic registration."""

    class A(symbolic.Object):
      pass

    self.assertIs(
        object_utils.JSONConvertible.class_from_typename(A.type_name), A)

  def testSymbolicOperations(self):
    """Tests for symbolic operations."""

    @symbolic.members([
        ('x', schema.Any(default=1)),
        ('y', schema.List(schema.Str()))
    ])
    class A(symbolic.Object):
      pass

    self.assertTrue(A.partial().sym_partial)
    self.assertTrue(A.partial().sym_abstract)
    self.assertEqual(A.partial(x=1).sym_missing(), {
        'y': schema.MISSING_VALUE,
    })
    self.assertEqual(A(x=1, y=['a']).sym_nondefault(), {
        'y[0]': 'a'
    })

    a = A(x=1, y=['foo'])
    self.assertFalse(a.sym_partial)
    self.assertFalse(a.sym_abstract)
    self.assertFalse(a.sym_sealed)
    self.assertTrue(a.sym_hasattr('x'))
    self.assertFalse(a.sym_hasattr('z'))
    self.assertFalse(a.sym_hasattr(0))
    self.assertEqual(a.sym_getattr('x'), 1)
    with self.assertRaisesRegex(
        AttributeError, '.* object has no symbolic attribute \'z\'.'):
      a.sym_getattr('z')

    self.assertIs(a.y.sym_parent, a)
    self.assertEqual(a.y.sym_path, 'y')
    a.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(a.sym_path, 'a')
    self.assertEqual(a.y.sym_path, 'a.y')

    a.sym_rebind({'y[1]': 'bar'})
    self.assertEqual(a.y[1], 'bar')
    self.assertEqual(next(a.sym_keys()), 'x')
    self.assertEqual(list(a.sym_keys()), ['x', 'y'])
    self.assertEqual(next(a.sym_values()), 1)
    self.assertEqual(list(a.sym_values()), [1, symbolic.List(['foo', 'bar'])])
    self.assertEqual(list(a.sym_items()), [
        ('x', 1), ('y', symbolic.List(['foo', 'bar']))])
    self.assertTrue(a.sym_contains('bar'))
    self.assertTrue(a.sym_contains(type=int))

    self.assertEqual(a.sym_clone(), A(x=1, y=['foo', 'bar']))
    self.assertTrue(a.sym_eq(a.clone()))
    self.assertTrue(a.sym_ne(A(x=2, y=['bar'])))
    self.assertEqual(a.sym_hash(),
                     a.sym_clone(deep=True).sym_hash())

    a.sym_setorigin(A.__init__, 'constructor')
    self.assertEqual(a.sym_origin.source, A.__init__)
    self.assertEqual(a.sym_origin.tag, 'constructor')

    @symbolic.members([
        ('z', schema.Int())
    ])
    class B(symbolic.Object):

      def sym_eq(self, other):
        if isinstance(other, int):
          return self.z == other
        return isinstance(other, A) and self.z == other.z

    self.assertEqual(A(x=B(z=1), y=['foo']), A(x=1, y=['foo']))
    self.assertNotEqual(A(x=B(z=1), y=['foo']), A(x=1.0, y=['foo']))
    self.assertTrue(symbolic.eq(
        A(x=B(z=1), y=['foo']),
        A(x=1, y=['foo'])))
    self.assertFalse(symbolic.ne(
        A(x=B(z=1), y=['foo']),
        A(x=1, y=['foo'])))


# NOTE(daiyip): when a function is converted into a functor, additional call is
# required to invoke the function. Lint doesn't handle decorated function like
# this well, thus we disable not-callable and no-value-for-parameter throughout
# the FunctorTest.

# pylint: disable=not-callable
# pylint: disable=no-value-for-parameter


class FunctorTest(unittest.TestCase):
  """Tests for symbolic.Functor."""

  def testBasics(self):
    """Test functor basics."""

    @symbolic.functor
    def _f0(a, b=symbolic.Dict(c=1)):
      return a + b

    self.assertEqual(_f0().missing_values(), {})
    self.assertFalse(_f0().is_partial)
    self.assertEqual(0, len(_f0().bound_args))
    self.assertEqual(_f0().unbound_args, set(['a', 'b']))
    self.assertFalse(_f0().is_fully_bound)

    with self.assertRaisesRegex(
        TypeError, 'takes 2 positional arguments but 3 were given'):
      _f0()(1, 2, 3)

    @symbolic.members([
        ('x', schema.Int()),
        ('y', schema.Int()),
    ])
    class _A(symbolic.Object):
      pass

    f0 = _f0(_A.partial(x=1))
    self.assertEqual(f0.missing_values(), {'a.y': schema.MISSING_VALUE})
    self.assertTrue(f0.is_partial)
    self.assertEqual(f0.bound_args, set(['a']))
    self.assertEqual(f0.unbound_args, set(['b']))
    self.assertFalse(f0.is_fully_bound)

    # Make sure `bound_args` is correctly set while `clone` constructs
    # a new object with non-args arguments.
    # (e.g. sealed, mark_missing_values, etc.)
    self.assertEqual(f0.clone(deep=True).bound_args, set(['a']))
    self.assertEqual(f0.clone(deep=False).bound_args, set(['a']))

    # Test bound args is updated on assignment and reset.
    f0.b = 2
    self.assertEqual(f0.bound_args, set(['a', 'b']))
    self.assertEqual(f0.clone(deep=True).bound_args, set(['a', 'b']))
    self.assertEqual(f0.clone(deep=False).bound_args, set(['a', 'b']))
    del f0.b  # Reset argument `b` to default value.
    self.assertEqual(f0.b, symbolic.Dict(c=1))
    self.assertEqual(f0.bound_args, set(['a']))
    self.assertEqual(f0.clone(deep=True).bound_args, set(['a']))
    self.assertEqual(f0.clone(deep=False).bound_args, set(['a']))
    f0.a = schema.MISSING_VALUE
    self.assertEqual(0, len(f0.bound_args))

    # Test bound args is updated on `rebind`.
    f0.rebind(a=2)
    self.assertEqual(f0.bound_args, set(['a']))
    f0.rebind(a=schema.MISSING_VALUE)
    self.assertEqual(0, len(f0.bound_args))
    f0.rebind(b=1)
    self.assertEqual(f0(1), 2)

    # Test bound args is updated when `rebind` on argument members.
    f0 = _f0()
    self.assertEqual(f0.b, symbolic.Dict(c=1))
    self.assertEqual(0, len(f0.bound_args))
    f0.rebind({'b.c': 2})
    self.assertEqual(f0.b, symbolic.Dict(c=2))
    self.assertEqual(f0.bound_args, set(['b']))

    with self.assertRaisesRegex(
        TypeError, '.* got an unexpected keyword argument'):
      @symbolic.functor(unsupported_keyword=1)
      def foo(x):   # pylint: disable=unused-variable
        return x

  def testAutoSchematization(self):
    """Test functor with no arguments."""

    @symbolic.functor()
    def _f1(a, b, *args, c=0, **kwargs):
      return a + b + c + sum(args) + sum(kwargs.values())

    self.assertEqual(
        list(_f1.schema.values()), [
            schema.Field('a', schema.Any(), 'Argument \'a\'.'),
            schema.Field('b', schema.Any(), 'Argument \'b\'.'),
            schema.Field('args', schema.List(schema.Any(), default=[]),
                         'Wildcard positional arguments.'),
            schema.Field('c', schema.Any(default=0), 'Argument \'c\'.'),
            schema.Field(schema.StrKey(), schema.Any(),
                         'Wildcard keyword arguments.'),
        ])
    self.assertEqual(_f1.signature.args, [
        schema.Argument('a', schema.Any()),
        schema.Argument('b', schema.Any())
    ])
    self.assertEqual(
        _f1.signature.varargs,
        schema.Argument('args', schema.List(schema.Any(), default=[])))
    self.assertEqual(
        _f1.signature.varkw,
        schema.Argument('kwargs', schema.Any()))
    self.assertEqual(
        _f1.signature.kwonlyargs,
        [schema.Argument('c', schema.Any(default=0))])
    self.assertIsNone(_f1.signature.return_value, None)
    self.assertTrue(_f1.signature.has_varargs)
    self.assertIsInstance(_f1(), symbolic.Functor)
    self.assertEqual(_f1()(1, 2), 3)
    self.assertEqual(_f1(b=1)(1), 2)
    self.assertEqual(_f1(b=1)(a=1), 2)

    self.assertEqual(_f1(1, 2, 3, 4)(), 10)
    self.assertEqual(_f1(1, 2, 3, 4, c=5)(), 15)
    self.assertEqual(_f1(1, 2, 3, 4, c=5, x=5)(), 20)
    self.assertEqual(_f1(1, 2, 3, 4)(2, 3, 4, override_args=True), 9)
    self.assertEqual(_f1(1, 2, 3, 4)(c=1), 11)
    self.assertEqual(_f1(1, 2, 3, 4)(x=2), 12)
    self.assertEqual(_f1(b=1)(2, c=4), 7)

  def testFullSchematization(self):
    """Test functor with all args symbolic."""

    @symbolic.functor([
        ('a', schema.Int()),
        ('b', schema.Int()),
    ], returns=schema.Int())
    def _f2(a=1, b=2):
      return a + b

    self.assertEqual(_f2.signature.args, [
        schema.Argument('a', schema.Int(default=1)),
        schema.Argument('b', schema.Int(default=2)),
    ])
    self.assertEqual(
        list(_f2.schema.values()), [
            schema.Field('a', schema.Int(default=1)),
            schema.Field('b', schema.Int(default=2)),
        ])
    self.assertEqual(_f2.signature.return_value, schema.Int())
    self.assertFalse(_f2.signature.has_varargs)
    self.assertFalse(_f2.signature.has_varkw)
    self.assertEqual(_f2()(), 3)
    self.assertEqual(_f2(a=2)(b=2), 4)
    self.assertEqual(_f2(a=3, b=2)(), 5)
    self.assertEqual(_f2(1, 2)(), 3)

    # Override default value.
    self.assertEqual(_f2(a=2)(), 4)
    self.assertEqual(_f2(a=2, override_args=True)(a=3), 5)  # pylint: disable=unexpected-keyword-arg
    self.assertEqual(_f2(a=1, b=1, override_args=True)(a=2, b=2), 4)  # pylint: disable=unexpected-keyword-arg
    self.assertEqual(_f2(2, 4, override_args=True)(1), 5)  # pylint: disable=unexpected-keyword-arg

  def testPartialSchematization(self):
    """Test functor with some args symbolic."""

    @symbolic.functor([
        ('c', schema.Int()),
        ('a', schema.Int()),
    ])
    def _f3(a, b=1, c=1):
      return a + b + c

    self.assertEqual(_f3.signature.args, [
        schema.Argument('a', schema.Int()),
        schema.Argument('b', schema.Any(default=1)),
        schema.Argument('c', schema.Int(default=1)),
    ])
    self.assertFalse(_f3.signature.has_varargs)
    self.assertFalse(_f3.signature.has_varkw)
    self.assertEqual(
        list(_f3.schema.values()), [
            schema.Field('a', schema.Int()),
            schema.Field('b', schema.Any(default=1), 'Argument \'b\'.'),
            schema.Field('c', schema.Int(default=1)),
        ])
    self.assertEqual(_f3()(a=2), 4)
    # Override 'a' with 2, provide 'b' with 2, use 'c' from default value 1.
    self.assertEqual(_f3()(2, 2), 5)

    # 'a' is not provided.
    with self.assertRaisesRegex(
        TypeError, 'missing 1 required positional argument'):
      _f3()()

  def testArgOverride(self):
    """Test functor with/without `override_args` flag."""

    @symbolic.functor()
    def _f7(x, *args, a=1, **kwargs):
      return x + sum(args) + a + sum(kwargs.values())

    self.assertEqual(_f7()(1), 2)
    # Override default value is not treated as override.
    self.assertEqual(_f7()(1, a=2), 3)
    self.assertEqual(_f7(1, override_args=True)(2), 3)  # pylint: disable=unexpected-keyword-arg
    self.assertEqual(_f7(1, override_args=True)(a=2), 3)  # pylint: disable=unexpected-keyword-arg
    with self.assertRaisesRegex(
        TypeError,
        '.* got new value for argument \'x\' from position 0'):
      _f7(0)(1)

    with self.assertRaisesRegex(
        TypeError,
        '.* got new value for argument \'a\' from keyword argument'):
      _f7(0, a=1)(a=2)

  def testIgnoreExtraArgs(self):
    """Test functor with/without `ignore_extra_args` flag."""
    # Test ignore extra args.
    @symbolic.functor()
    def _f8(a=1):
      return a

    self.assertEqual(_f8(ignore_extra_args=True)(1, c=0), 1)  # pylint: disable=unexpected-keyword-arg
    self.assertEqual(_f8()(1, c=0, ignore_extra_args=True), 1)
    self.assertEqual(_f8()(1, 2, ignore_extra_args=True), 1)
    with self.assertRaisesRegex(
        TypeError, '.* got an unexpected keyword argument \'c\''):
      _f8()(1, c=0)

  def testValidateArgs(self):
    """Test schema validation on function arguments."""

    @symbolic.functor([
        ('a', schema.Int(min_value=0)),
        ('b', schema.Int(max_value=10)),
        ('args', schema.List(schema.Int())),
        (schema.StrKey(), schema.Int(max_value=5))
    ])
    def _f10(a, *args, b, **kwargs):
      return a + b + sum(args) + sum(kwargs.values())

    self.assertEqual(_f10(1, 2, b=3)(c=4), 10)
    # Validate during pre-binding.
    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range .*min=0'):
      _f10(-1, b=1)

    with self.assertRaisesRegex(
        TypeError, 'Expect .*int.* but encountered .*float'):
      _f10(1, 0.1, b=1)

    with self.assertRaisesRegex(
        ValueError, 'Value 11 is out of range .*max=10'):
      _f10(1, b=11)

    with self.assertRaisesRegex(
        ValueError, 'Value 10 is out of range .*max=5'):
      _f10(1, b=1, c=10)

    # Validate during late binding.
    with self.assertRaisesRegex(
        ValueError, 'Value -1 is out of range .*min=0'):
      _f10(b=2)(-1)

    with self.assertRaisesRegex(
        TypeError, 'Expect .*int.* but encountered .*float'):
      _f10(1, b=1)(1, 0.1, override_args=True)

    with self.assertRaisesRegex(
        ValueError, 'Value 11 is out of range .*max=10'):
      _f10(1)(b=11)   # pylint: disable=missing-kwoa

    with self.assertRaisesRegex(
        ValueError, 'Value 10 is out of range .*max=5'):
      _f10(1, b=1)(c=10)

    with self.assertRaisesRegex(
        ValueError, 'Value 6 is out of range .*max=5'):
      _f10(1, b=1)(c=6)

  def testAsFunctor(self):
    """Test `symbolic.as_functor`."""
    f11 = symbolic.as_functor(lambda x: x)
    self.assertIsInstance(f11, symbolic.Functor)
    self.assertEqual(f11.signature.args, [schema.Argument('x', schema.Any())])
    self.assertIsNone(f11.signature.return_value)
    self.assertEqual(f11(1), 1)

  def testCustomBaseClass(self):
    """Test `symbolic.functor` with custom `base_class`."""

    class MyFunctor(symbolic.Functor):
      pass

    @symbolic.functor(base_class=MyFunctor)
    def my_fun():
      return 0

    self.assertIsInstance(my_fun(), MyFunctor)

  def testAutomaticRegistration(self):
    """Test automatic registration."""

    @symbolic.functor()
    def foo():
      return 0

    self.assertIs(
        object_utils.JSONConvertible.class_from_typename(foo.type_name), foo)

  def testInvalidUseCases(self):
    """Test invalid use cases."""
    # @functor decorator is not applicable to class.
    with self.assertRaisesRegex(TypeError, '.* is not a method.'):

      @symbolic.functor([('a', schema.Int())])
      class _A:
        pass

      del _A

    # @functor decorator found extra symbolic argument.
    with self.assertRaisesRegex(
        KeyError, '.* found extra symbolic argument \'a\'.'):

      @symbolic.functor([('a', schema.Int())])
      def bar1():  # pylint: disable=unused-variable
        pass

    # @functor decorator has multiple StrKey.
    with self.assertRaisesRegex(
        KeyError,
        '.* multiple StrKey found in symbolic arguments declaration.'):

      @symbolic.functor([
          ('a', schema.Int()),
          (schema.StrKey(), schema.Any()),
          (schema.StrKey(), schema.Any()),
      ])  # pylint: disable=unused-variable
      def bar2(a, **kwargs):
        del a, kwargs

    with self.assertRaisesRegex(
        KeyError, '.* multiple symbolic fields found for argument.'):

      @symbolic.functor([
          ('a', schema.Int()),
          ('a', schema.Str()),
      ])  # pylint: disable=unused-variable
      def bar3(a):
        del a

    with self.assertRaisesRegex(
        ValueError,
        '.* the default value .* of symbolic argument .* does not equal '
        'to the default value .* specified at function signature'):

      @symbolic.functor([
          ('a', schema.Int(default=2)),
      ])  # pylint: disable=unused-variable
      def bar4(a=1):
        del a

    with self.assertRaisesRegex(
        ValueError, 'return value spec should not have default value'):

      @symbolic.functor([
          ('a', schema.Int()),
      ], returns=schema.Any(default=None))
      def bar5(a=1):  # pylint: disable=unused-variable
        del a

    @symbolic.functor([
        ('a', schema.Int()),
    ])
    def bar6(a=1):  # pylint: disable=unused-variable
      del a

    with self.assertRaisesRegex(
        TypeError, '.* got multiple values for keyword argument \'a\''):
      bar6(1, a=1)  # pylint: disable=redundant-keyword-arg

    with self.assertRaisesRegex(
        TypeError, '.* takes 1 positional argument but 2 were given'):
      bar6(1, 2)  # pylint: disable=too-many-function-args

    with self.assertRaisesRegex(
        TypeError, '.* takes 1 positional argument but 2 were given'):
      bar6()(1, 2)  # pylint: disable=too-many-function-args


# pylint: enable=not-callable
# pylint: enable=no-value-for-parameter

#
# Tests for mixture of symbolic types on special topics.
#


class RebindTest(unittest.TestCase):
  """Tests for symbolic.Symbolic.rebind."""

  def setUp(self):
    """Test setup."""
    super().setUp()

    @symbolic.members([('a', schema.Int()), ('b', schema.Bool(True)),
                       ('c', schema.Str().noneable())])
    class A(symbolic.Object):
      pass

    @symbolic.members([('d',
                        schema.List(
                            schema.Dict([
                                ('e', schema.Enum('foo', ['foo', 'bar'])),
                                ('f', schema.Float(1.0)),
                            ]))),
                       ('g',
                        schema.Dict([
                            ('h',
                             schema.Union([
                                 schema.Int(),
                                 schema.Tuple([schema.Int(),
                                               schema.Int()])
                             ]).noneable()),
                            ('i', schema.Object(object_utils.KeyPath))
                        ])), ('j', schema.Any())])
    class B(A):
      pass

    # pylint: disable=invalid-name
    self._A = A
    self._B = B

  def testSchemaless(self):
    """Test rebind on a mixture of schemaless List and symbolic object."""
    sl = symbolic.List([{
        'x': self._A(a=0),
        'y': 1,
        'z': [1, True, False, 0]
    }])
    self.assertEqual(sl, [{
        'x': self._A(a=0, b=True, c=None),
        'y': 1,
        'z': [1, True, False, 0]
    }])
    sl.rebind({
        '[0].x.c': 'foo',
        '[0].y': 'bar',
        # Delete element 0 (1) in z.
        '[0].z[0]': schema.MISSING_VALUE,
        # Replace element 1 (True) into 1.0
        '[0].z[1]': 1.0,
        # Insert 'hello' at position 2 (between True and False).
        '[0].z[2]': symbolic.mark_as_insertion('hello'),
        # Append 5 at the end of z.
        '[0].z[10]': 5
    })
    self.assertEqual(sl, [{
        'x': self._A(a=0, b=True, c='foo'),
        'y': 'bar',
        'z': [1.0, 'hello', False, 0, 5]
    }])
    sl.rebind({
        # Reset x to a different type.
        '[0].x': 0,
        # Remove field y.
        '[0].y': schema.MISSING_VALUE,
        # Reset z to a different type.
        '[0].z': (1, 2)
    })
    self.assertEqual(sl, [{'x': 0, 'z': (1, 2)}])

    sl.rebind({'[0]': schema.MISSING_VALUE})
    self.assertEqual(0, len(sl))

  def testsymbolic(self):
    """Test rebind on a symbolic object."""
    b = self._B.partial(d=[{}])
    self.assertEqual(
        b,
        self._B.partial(
            a=schema.MISSING_VALUE,
            b=True,
            c=None,
            d=[{
                'e': 'foo',
                'f': 1.0
            }],
            g={
                'h': schema.MISSING_VALUE,
                'i': schema.MISSING_VALUE,
            },
            j=schema.MISSING_VALUE))
    b.rebind({
        # Choose tuple for union.
        'g.h': (1, 1),
        # Implicit string => KeyPath conversion.
        'g.i': 'a.b.c',
        # Any type.
        'j': 123
    })
    self.assertEqual(b.a, schema.MISSING_VALUE)
    self.assertEqual(b.g.i.keys, ['a', 'b', 'c'])
    self.assertEqual(b.g.h, (1, 1))
    self.assertEqual(b.j, 123)

    b.rebind({
        # Rebind from tuple to int.
        'g.h': 1
    })
    self.assertEqual(b.g.h, 1)

  def testRebindUsingRebinder(self):
    """Test rebind using path_value_paris function."""
    b = self._B(a=1, d=[{}], g={'i': 'x.y'}, j={'h': 'foo', 'm': 1.0})

    # Change value based on field path.
    # `g.i` will be impacted.
    b.rebind(lambda k, v: 'y.z' if k == 'g.i' else v)

    # Change value based on field name.
    # `g.h` and `j.h` will be impacted.
    b.rebind(lambda k, v: (1, 1) if k and k.key == 'h' else v)
    self.assertEqual(
        b,
        self._B(
            a=1, d=[{}], g={
                'h': (1, 1),
                'i': 'y.z'
            }, j={
                'h': (1, 1),
                'm': 1.0
            }))

    # Change all float values to 0.0 recursively.
    # `d.f` and 'j.m' will be impacted.
    b.rebind(lambda k, v: 0.0 if isinstance(v, float) else v)
    self.assertEqual(
        b,
        self._B(
            a=1,
            d=[{
                'f': 0.0
            }],
            g={
                'h': (1, 1),
                'i': 'y.z'
            },
            j={
                'h': (1, 1),
                'm': 0.0
            }))

    # Change value based on parent path.
    def modify_float_under_j(k, v, p):
      del k
      if isinstance(v, float) and p.sym_path.key == 'j':
        return 2.0
      return v

    b.rebind(modify_float_under_j)
    self.assertEqual(
        b,
        self._B(
            a=1,
            d=[{
                'f': 0.0
            }],
            g={
                'h': (1, 1),
                'i': 'y.z'
            },
            j={
                'h': (1, 1),
                'm': 2.0
            }))

    with self.assertRaisesRegex(
        TypeError, 'Rebinder function .* should accept 2 or 3 arguments'):
      b.rebind(lambda v: v)

  def testRebindNothing(self):
    """Test rebind nothing."""
    b = self._B.partial()
    self.assertEqual(b.rebind(raise_on_no_change=False), self._B.partial())
    with self.assertRaisesRegex(ValueError, 'There are no values to rebind'):
      symbolic.Dict().rebind(lambda k, v: v)
    with self.assertRaisesRegex(ValueError, 'There are no values to rebind'):
      symbolic.List().rebind(lambda k, v: v)

  def testRebindWithSkippingNotification(self):
    """Test rebind with skip_notification flag."""

    @symbolic.members([
        ('x', schema.Int())
    ])
    class A(symbolic.Object):

      def _on_init(self):
        super()._on_init()
        self._num_changes = 0

      def _on_change(self, updates):
        super()._on_change(updates)
        self._num_changes += 1

      @property
      def num_changes(self):
        return self._num_changes

    a = A(x=1)
    self.assertEqual(a.num_changes, 0)
    a.rebind(x=2)
    self.assertEqual(a.num_changes, 1)
    a.rebind(x=3, skip_notification=True)
    self.assertEqual(a.num_changes, 1)
    a.rebind(x=4)
    self.assertEqual(a.num_changes, 2)

  def testInvalidRebind(self):
    """Test invalid rebind use cases."""

    class A:

      def __init__(self):
        self.x = 0

    @symbolic.members([('a', schema.Object(A))])
    class B(symbolic.Object):
      pass

    b = B(a=A())
    # Rebind is invalid on root object.
    with self.assertRaisesRegex(
        KeyError, 'Root key .* cannot be used in .*rebind.'):
      b.rebind({'': 1})

    # Rebind is invalid on non-symbolic object.
    with self.assertRaisesRegex(
        KeyError, 'Cannot rebind key .* is not a symbolic type.'):
      b.rebind({'a.x': 1})

    # Dict-specific invalid rebind.
    with self.assertRaisesRegex(
        ValueError,
        'Either argument \'path_value_pairs\' or \'\\*\\*kwargs\' '
        'shall be specified'):
      symbolic.Dict().rebind({'a': 1}, a=1)

    with self.assertRaisesRegex(
        TypeError, 'Argument \'path_value_pairs\' should be a dict.'):
      symbolic.Dict().rebind(1)

    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind.'):
      symbolic.Dict().rebind({})

    with self.assertRaisesRegex(
        KeyError,
        'Keys in argument \'path_value_pairs\' of Dict.rebind must be string'):
      symbolic.Dict().rebind({1: 1})

    d = symbolic.Dict(a=1, value_spec=schema.Dict([('a', schema.Int())]))
    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified.'):
      d.rebind({'a': schema.MISSING_VALUE})

    # List-specific invalid rebind.
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'path_value_pairs\' must be a non-empty dict'):
      symbolic.List().rebind(1)

    with self.assertRaisesRegex(
        KeyError,
        'Keys in argument \'path_value_paris\' of List.rebind must be either '
        'int or string type'):
      symbolic.List().rebind({int: 1})

  def testResetSemantics(self):
    """Test rebind with reset to default values."""
    a = self._A(a=1, b=False, c='bar')
    a.rebind({
        # Reset b to default value (True).
        'b': schema.MISSING_VALUE,
        # Reset c to default value (None)
        'c': schema.MISSING_VALUE
    })
    self.assertTrue(a.b)
    self.assertIsNone(a.c)


class EventTest(unittest.TestCase):
  """Tests for symbolic.Symbolic._on_change event."""

  def testFieldUpdate(self):
    """Test symbolic.FieldUpdate."""
    d = symbolic.Dict({'a': 1})
    update = symbolic.FieldUpdate(object_utils.KeyPath('a'), d, None, 1, 2)
    self.assertEqual(update, update)
    self.assertEqual(
        update,
        symbolic.FieldUpdate(object_utils.KeyPath('a'), d, None, 1, 2))
    self.assertNotEqual(update, None)
    self.assertEqual(
        str(update),
        'FieldUpdate(parent_path=, path=a, old_value=1, new_value=2)')

  def testSubscribesFieldUpdates(self):
    """Test symbolic._subscribes_field_updates property."""
    self.assertFalse(symbolic.Dict()._subscribes_field_updates)  # pylint: disable=protected-access
    self.assertTrue(
        symbolic.Dict(
            onchange_callback=lambda updates: None)._subscribes_field_updates)  # pylint: disable=protected-access

    self.assertFalse(symbolic.List()._subscribes_field_updates)  # pylint: disable=protected-access
    self.assertTrue(
        symbolic.List(
            onchange_callback=lambda updates: None)._subscribes_field_updates)  # pylint: disable=protected-access

    @symbolic.members([('x', schema.Int())])
    class A(symbolic.Object):
      pass

    @symbolic.members([('x', schema.Int())])
    class B(symbolic.Object):

      def _on_change(self, field_updates):
        pass

    self.assertFalse(A(x=1)._subscribes_field_updates)  # pylint: disable=protected-access
    self.assertTrue(B(x=1)._subscribes_field_updates)  # pylint: disable=protected-access

  def testOnChange(self):
    """Test _on_change event."""
    object_updates = []

    @symbolic.members([('x', schema.Int(1)),
                       ('y', schema.Bool().noneable()),
                       ('z', schema.Str())])
    class A(symbolic.Object):

      def _on_change(self, field_updates):
        object_updates.append(field_updates)

    value_spec = schema.Dict([
        ('a1', schema.Int()),
        ('a2',
         schema.Dict([('b1',
                       schema.Dict([('c1',
                                     schema.List(
                                         schema.Dict([('d1', schema.Str('foo')),
                                                      ('d2', schema.Bool(True)),
                                                      ('d3', schema.Object(A))
                                                     ])))]))]))
    ])

    root_updates = []

    def _onchange(field_updates):
      root_updates.append(field_updates)

    list_updates = []

    def _onchange_list(field_updates):
      list_updates.append(field_updates)

    child_dict_updates = []

    def _onchange_child(field_updates):
      child_dict_updates.append(field_updates)

    sd = symbolic.Dict.partial(
        {
            'a2': {
                'b1': {
                    'c1':
                        symbolic.List([
                            symbolic.Dict(
                                d3=A.partial(),
                                allow_partial=True,
                                onchange_callback=_onchange_child)
                        ], allow_partial=True, onchange_callback=_onchange_list)
                }
            }
        },
        value_spec=value_spec,
        onchange_callback=_onchange)

    # There are no updates in object A.
    self.assertEqual(object_updates, [])

    # innermost Dict get updated after bind with List.
    self.assertEqual(
        child_dict_updates,
        [
            # Set default value from outer space (parent List) for field d1.
            {
                'd1':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d1',
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                        old_value=schema.MISSING_VALUE,
                        new_value='foo')
            },
            # Set default value from outer space (parent List) for field d2.
            {
                'd2':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d2',
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                        old_value=schema.MISSING_VALUE,
                        new_value=True)
            }
        ])

    # list get updated after bind with parent structures.
    self.assertEqual(list_updates, [{
        '[0].d1':
            symbolic.FieldUpdate(
                path='a2.b1.c1[0].d1',
                target=sd.a2.b1.c1[0],
                field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                old_value=schema.MISSING_VALUE,
                new_value='foo')
    }, {
        '[0].d2':
            symbolic.FieldUpdate(
                path='a2.b1.c1[0].d2',
                target=sd.a2.b1.c1[0],
                field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                old_value=schema.MISSING_VALUE,
                new_value=True)
    }])

    # There are no updates in root.
    self.assertEqual(root_updates, [])

    child_dict_updates = []
    list_updates = []

    sd.rebind({
        'a1': 1,
        'a2.b1.c1[0].d1': 'bar',
        'a2.b1.c1[0].d2': False,
        'a2.b1.c1[0].d3.z': 'foo',
    })

    # Inspect root object changes.
    self.assertEqual(len(root_updates), 1)
    self.assertEqual(len(root_updates[0]), 4)

    self.assertEqual(
        root_updates[0], {
            'a1':
                symbolic.FieldUpdate(
                    path='a1',
                    target=sd,
                    field=sd.value_spec.schema['a1'],
                    old_value=schema.MISSING_VALUE,
                    new_value=1),
            'a2.b1.c1[0].d1':
                symbolic.FieldUpdate(
                    path='a2.b1.c1[0].d1',
                    target=sd.a2.b1.c1[0],
                    field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                    old_value='foo',
                    new_value='bar'),
            'a2.b1.c1[0].d2':
                symbolic.FieldUpdate(
                    path='a2.b1.c1[0].d2',
                    target=sd.a2.b1.c1[0],
                    field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                    old_value=True,
                    new_value=False),
            'a2.b1.c1[0].d3.z':
                symbolic.FieldUpdate(
                    path='a2.b1.c1[0].d3.z',
                    target=sd.a2.b1.c1[0].d3,
                    field=sd.a2.b1.c1[0].d3.__class__.schema['z'],
                    old_value=schema.MISSING_VALUE,
                    new_value='foo')
        })

    # Inspect list node changes.
    self.assertEqual(
        list_updates,
        [
            # Root object rebind.
            {
                '[0].d1':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d1',
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                        old_value='foo',
                        new_value='bar'),
                '[0].d2':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d2',
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                        old_value=True,
                        new_value=False),
                '[0].d3.z':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d3.z',
                        target=sd.a2.b1.c1[0].d3,
                        field=sd.a2.b1.c1[0].d3.__class__.schema['z'],
                        old_value=schema.MISSING_VALUE,
                        new_value='foo')
            }
        ])

    # Inspect leaf node changes.
    self.assertEqual(
        child_dict_updates,
        [
            # Root object rebind.
            {
                'd1':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d1',
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                        old_value='foo',
                        new_value='bar'),
                'd2':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d2',
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                        old_value=True,
                        new_value=False),
                'd3.z':
                    symbolic.FieldUpdate(
                        path='a2.b1.c1[0].d3.z',
                        target=sd.a2.b1.c1[0].d3,
                        field=sd.a2.b1.c1[0].d3.__class__.schema['z'],
                        old_value=schema.MISSING_VALUE,
                        new_value='foo')
            }
        ])

  def testOnChangeEventOrder(self):
    """Test the trigger order of `_on_change` event."""

    change_order = []

    @symbolic.members([
        (schema.StrKey(), schema.Any())
    ])
    class Node(symbolic.Object):

      def _on_change(self, field_updates):
        change_order.append(self.sym_path)

    node = Node(
        b=Node(z=1, x=1),
        a=Node(y=1, x=Node(p=1), z=1),
        c=Node(z=Node(q=1), y=1),
        d=Node())
    node.rebind({
        'c.y': 2,
        'a.z': 2,
        'b.z': 2,
        'a.x.p': 2,
        'c.z.q': 2
    })
    self.assertEqual(change_order, [
        'c.z', 'c', 'b', 'a.x', 'a', ''
    ])

  def testOnParentChange(self):
    """Test on_parent_change event."""

    class A(symbolic.Object):

      def _on_parent_change(self, old_parent, new_parent):
        self.old_parent = old_parent
        self.new_parent = new_parent

    x = A()
    y = symbolic.Dict(x=x)
    self.assertIsNone(x.old_parent)
    self.assertIs(x.new_parent, y)
    self.assertEqual(x.sym_path, 'x')

    y.x = A()
    self.assertIs(x.old_parent, y)
    self.assertIsNone(x.new_parent)
    self.assertEqual(x.sym_path, object_utils.KeyPath())

  def testOnPathChange(self):
    """Test on_path_change event."""

    class A(symbolic.Object):

      def _on_path_change(self, old_path, new_path):
        self.old_path = old_path
        self.new_path = new_path

    x = A()
    x.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(x.old_path, object_utils.KeyPath())
    self.assertEqual(x.new_path, 'a')

    y = symbolic.Dict(x=x)
    self.assertEqual(x.old_path, 'a')
    self.assertEqual(x.new_path, 'x')

    _ = symbolic.Dict(y=y)
    self.assertEqual(x.old_path, 'x')
    self.assertEqual(x.new_path, 'y.x')


class ContextManagerTest(unittest.TestCase):
  """Tests for symbolic context managers."""

  def testNotifyOnChange(self):
    """Test `notify_on_change` method."""
    context = symbolic.Dict(num_changes=0)
    def increment_change(unused_updates):
      context.num_changes += 1

    d = symbolic.Dict(a=1, onchange_callback=increment_change)

    # Change notification triggers for Dict.
    d.a = 2
    d.b = 4
    del d['a']
    d.a = 4
    self.assertEqual(context.num_changes, 4)

    # Change notification does not trigger.
    context.num_changes = 0
    with symbolic.notify_on_change(False):
      d.a = 2
      d.b = 4
      del d['a']
      d.a = 4
    self.assertEqual(context.num_changes, 0)

    # Change notification triggers for List.
    context.num_changes = 0
    l = symbolic.List([0], onchange_callback=increment_change)
    l.append(1)
    l[1] = 3
    l.insert(0, 2)
    del l[0]
    l.remove(3)
    self.assertEqual(context.num_changes, 5)

    # Change notification for Object.
    @symbolic.members([
        ('x', schema.Int())
    ])
    class A(symbolic.Object):

      def _on_change(self, unused_updates):
        context.num_changes += 1

    context.num_changes = 0
    a = A(x=1)
    with symbolic.notify_on_change(False):
      a.rebind(x=5)
      with symbolic.notify_on_change(True):
        a.rebind(x=6)
      a.rebind(x=7)
    self.assertEqual(context.num_changes, 1)

  def testEnableTypeCheck(self):
    """Test `enable_type_check` context manager."""
    # List.
    with symbolic.enable_type_check(False):
      l = symbolic.List([1.0], value_spec=schema.List(schema.Int()))
      l.append(2.0)
      # Test nesting.
      with symbolic.enable_type_check(True):
        with self.assertRaisesRegex(
            TypeError, 'Expect .* but encountered .*'):
          l.append(3.0)
      l.append(4.0)

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):
      l.append(5.0)

    # Dict.
    with symbolic.enable_type_check(False):
      d = symbolic.Dict(x=1.0, value_spec=schema.Dict([
          ('x', schema.Int()),
          ('y', schema.Str()),
      ]))
      with self.assertRaisesRegex(
          KeyError, 'Key \'z\' is not allowed for .*Dict'):
        d.z = True

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):
      d.y = 0.1

    # Object.
    @symbolic.members([
        ('x', schema.Int())
    ])
    class Foo(symbolic.Object):
      pass

    with symbolic.enable_type_check(False):
      f = Foo(x='foo')
      f.rebind(x=1.0)
      with self.assertRaisesRegex(
          KeyError, 'Key .* is not allowed for .*Foo'):
        f.rebind(y=None)

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):
      f.rebind(x='bar')

  def testAsSealed(self):
    """Test `as_sealed` method."""
    sd = symbolic.Dict()
    sd2 = symbolic.Dict().seal()
    with symbolic.as_sealed(True):
      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify field of a sealed Dict.'):
        sd.a = 1
      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify field of a sealed Dict.'):
        sd2.a = 1
      with symbolic.as_sealed(False):
        sd.a = 1
        sd2.a = 1
        with symbolic.as_sealed(None):
          sd.a = 1
          with self.assertRaisesRegex(
              symbolic.WritePermissionError,
              'Cannot modify field of a sealed Dict.'):
            sd2.a = 1

  def testAllowPartialValues(self):
    """Test `allow_partial_values` method."""

    @symbolic.members([
        ('x', schema.Int()),
        ('y', schema.Int())
    ])
    class A(symbolic.Object):
      pass

    with symbolic.allow_partial_values(True):
      a = A(x=1)
      with symbolic.allow_partial_values(False):
        with self.assertRaisesRegex(
            ValueError, 'Required value is not specified'):
          a.rebind(x=schema.MISSING_VALUE)
      a.rebind(x=schema.MISSING_VALUE)
      self.assertEqual(a.x, schema.MISSING_VALUE)
      self.assertEqual(a.y, schema.MISSING_VALUE)

  def testAllowWritableAccessors(self):
    """Test `allow_writable_accessors` method."""
    sd = symbolic.Dict()
    sd2 = symbolic.Dict(accessor_writable=False)
    with symbolic.allow_writable_accessors(False):
      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify Dict field by attribute or key while '
          'accessor_writable is set to False.'):
        sd.a = 2
      with self.assertRaisesRegex(
          symbolic.WritePermissionError,
          'Cannot modify Dict field by attribute or key while '
          'accessor_writable is set to False.'):
        sd2.a = 2
      with symbolic.allow_writable_accessors(True):
        sd.a = 2
        sd2.a = 2
        with symbolic.allow_writable_accessors(None):
          # Honor object-level 'accessor_writable' flags.
          sd.a = 1
          with self.assertRaisesRegex(
              symbolic.WritePermissionError,
              'Cannot modify Dict field by attribute or key while '
              'accessor_writable is set to False.'):
            sd2.a = 1


class OriginTest(unittest.TestCase):
  """Tests related to origin."""

  def testOrigin(self):
    """Test Origin class."""
    a = symbolic.Dict(a=1)

    # Test basics.
    o = symbolic.Origin(a, '__init__')
    self.assertIs(o.source, a)
    self.assertEqual(o.tag, '__init__')
    self.assertIsNone(o.stack)
    self.assertIsNone(o.stacktrace)

    o = symbolic.Origin(a, '__init__', stacktrace=True, stacklimit=3)
    self.assertIs(o.source, a)
    self.assertEqual(o.tag, '__init__')
    self.assertEqual(len(o.stack), 3)
    self.assertIsNotNone(o.stacktrace)

    symbolic.set_stacktrace_limit(2)
    o = symbolic.Origin(a, '__init__', stacktrace=True)
    self.assertEqual(len(o.stack), 2)

    # Test eq/ne:
    self.assertEqual(
        symbolic.Origin(None, '__init__'),
        symbolic.Origin(None, '__init__'))

    self.assertEqual(
        symbolic.Origin(a, 'builder'),
        symbolic.Origin(a, 'builder'))

    self.assertNotEqual(
        symbolic.Origin(a, 'builder'),
        symbolic.Origin(a, 'return'))

    self.assertNotEqual(
        symbolic.Origin(a, 'builder'),
        symbolic.Origin(symbolic.Dict(a=1), 'builder'))

    # Test format:
    o = symbolic.Origin(None, '__init__')
    self.assertEqual(o.format(), 'Origin(tag=\'__init__\')')

    o = symbolic.Origin('/path/to/file', 'load')
    self.assertEqual(
        o.format(),
        'Origin(tag=\'load\', source=\'/path/to/file\')')

    o = symbolic.Origin(a, 'builder')
    self.assertEqual(
        o.format(compact=True),
        'Origin(tag=\'builder\', source={a=1} at 0x%x)' % id(a))

  def testTrackOrigin(self):
    """Test origin tracking."""
    @symbolic.functor
    def f1():
      return f2()

    @symbolic.functor()
    def f2():
      return symbolic.Dict(a=1)

    with symbolic.track_origin():
      p = f1()
      x = p()
      y = x()
      z = y.clone()
      q = z.clone(deep=True)

    self.assertIs(q.sym_origin.source, z)
    self.assertEqual(q.sym_origin.tag, 'deepclone')
    self.assertIsNotNone(q.sym_origin.stacktrace)

    # Test chain.
    self.assertEqual(q.sym_origin.chain(), [
        symbolic.Origin(z, tag='deepclone'),
        symbolic.Origin(y, tag='clone'),
        symbolic.Origin(x, tag='return'),
        symbolic.Origin(p, tag='return'),
        symbolic.Origin(None, tag='__init__')
    ])

    self.assertEqual(q.sym_origin.chain('return'), [
        symbolic.Origin(x, tag='return'),
        symbolic.Origin(p, tag='return')
    ])

  def testDoesNotTrackOrigin(self):
    """Test chain when origin tracking is off."""
    @symbolic.functor
    def f1():
      return f2()

    @symbolic.functor()
    def f2():
      return symbolic.Dict(a=1)

    with symbolic.track_origin(True):
      with symbolic.track_origin(False):
        p = f1()
        q = p()

    self.assertIsNone(q.sym_origin)

    # Set origin by user.
    q.sym_setorigin(p, 'builder')
    self.assertIs(q.sym_origin.source, p)
    self.assertEqual(q.sym_origin.tag, 'builder')
    self.assertIsNone(q.sym_origin.stack)
    self.assertIsNone(q.sym_origin.stacktrace)

    # Set origin with a different description.
    q.sym_setorigin(p, 'builder2', stacktrace=True)
    self.assertIs(q.sym_origin.source, p)
    self.assertEqual(q.sym_origin.tag, 'builder2')
    self.assertIsNotNone(q.sym_origin.stack)
    self.assertIsNotNone(q.sym_origin.stacktrace)

    # Set origin with a different source.
    with self.assertRaisesRegex(
        ValueError, 'Cannot set the origin with a different source value'):
      q.sym_setorigin(f1(), 'builder3')


class SerializationTest(unittest.TestCase):

  def testDict(self):
    """Test serialization for symbolic.Dict."""

    sd = symbolic.Dict()
    sd.b = 0
    sd.c = None
    sd.a = 'foo'

    # Test key order is preserved.
    self.assertEqual(
        symbolic.to_json_str(sd), '{"b": 0, "c": null, "a": "foo"}')

    # Test symbolic dict.
    sd = symbolic.Dict.partial(
        x=1,
        value_spec=schema.Dict([
            ('w', schema.Str()),
            ('x', schema.Int()),
            ('y', schema.Str().noneable()),
            ('z', schema.Bool(True).freeze()),
        ]))

    self.assertEqual(sd, {
        'w': schema.MISSING_VALUE,
        'x': 1,
        'y': None,
        'z': True
    })

    self.assertEqual(
        symbolic.to_json_str(sd), '{"x": 1, "y": null}')

    # Test hide default values.
    self.assertEqual(
        symbolic.to_json_str(sd, hide_default_values=True), '{"x": 1}')

    # Test Dict.from_json.
    self.assertEqual(
        symbolic.from_json_str('{"x": 1}').use_value_spec(
            sd.value_spec, allow_partial=True),
        symbolic.Dict(sd, value_spec=sd.value_spec, allow_partial=True))

    # Test to_json_str on a regular dict.
    self.assertEqual(
        symbolic.to_json_str({
            'x': 1,
            'y': None,
            'z': True
        }), '{"x": 1, "y": null, "z": true}')

    # Test to_json_str on for convert
    class A:
      pass
    with self.assertRaisesRegex(
        ValueError, 'Cannot convert complex type .* to JSON.'):
      symbolic.to_json_str({'x': 1, 'y': A()})
    schema.register_converter(A, float, convert_fn=lambda x: 2)
    self.assertEqual(
        symbolic.to_json_str({'x': 1, 'y': A()}),
        '{"x": 1, "y": 2}')
    self.assertEqual(
        symbolic.to_json_str({'x': 1, 'y': {'y1': 3, 'y2': A()}}),
        '{"x": 1, "y": {"y1": 3, "y2": 2}}')

  def testList(self):
    """Test serialization for List."""
    sl = symbolic.List()
    sl.append(1)
    sl.extend([2, schema.MISSING_VALUE, 3])

    self.assertEqual(symbolic.to_json_str(sl), '[1, 2, 3]')

    self.assertEqual(
        symbolic.List.from_json([1, 2, 3]), symbolic.List([1, 2, 3]))

    self.assertEqual(
        symbolic.from_json_str('[1, 2, 3]'), symbolic.List([1, 2, 3]))

  def testObject(self):
    """Test serialization for Object."""

    @symbolic.members([
        ('a', schema.Int()),
        ('b', schema.Str('foo')),
        ('c', schema.Bool().noneable()),
    ])
    class A(symbolic.Object):
      pass

    self.assertEqual(
        symbolic.to_json_str(A(a=1)),
        '{"_type": "pyglove.core.symbolic_test.A", "a": 1, "b": "foo", "c": null}')

    self.assertEqual(
        symbolic.from_json_str(
            '{"_type": "pyglove.core.symbolic_test.A", "a": 1, "b": "foo", "c": null}'), A(a=1))

    with self.assertRaisesRegex(
        TypeError,
        'Type name \'.*\' is not registered with a symbolic.Object'):
      symbolic.from_json_str('{"_type": "pyglove.core.symbolic_test.NotExisted", "a": 1}')

  def testTuple(self):
    """Test serialization for tuple."""
    self.assertEqual(symbolic.to_json_str((1, 2)), '["__tuple__", 1, 2]')

    self.assertEqual(symbolic.from_json_str('["__tuple__", 1]'), (1,))

    with self.assertRaisesRegex(
        ValueError,
        'Tuple should have at least one element besides \'__tuple__\'.'):
      symbolic.from_json_str('["__tuple__"]')

  def testMixture(self):
    """Test mixture of types."""

    @symbolic.members([('x', schema.Int()),
                       ('y', schema.Object(object_utils.KeyPath))])
    class A(symbolic.Object):
      pass

    @symbolic.members([('a', schema.Object(A))])
    class B(symbolic.Object):
      pass

    sd = symbolic.Dict({
        'f1': [A(x=1, y='x.y')],
        'f2': (1, A(x=0, y='c.d')),
        'f3': B(a=A(x=0, y='a.b'))
    })
    json_str = sd.to_json_str()
    sd2 = symbolic.from_json_str(json_str)
    self.assertEqual(sd, sd2)

  def testUnsupportedTypes(self):
    """Test unsupported types."""

    class A:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert complex type .* to JSON.'):
      symbolic.to_json(A())

  def testCustomSerializationKeys(self):
    """Test custom serialization key and additional keys."""
    # Test custom serialization key for symbolic class.

    @symbolic.members([
        ('x', schema.Int())
    ], serialization_key='main.Foo', additional_keys=['main.Bar'])
    class A(symbolic.Object):
      pass

    a = A(x=1)
    v = a.to_json()
    self.assertEqual(v['_type'], 'main.Foo')
    self.assertEqual(symbolic.from_json(v), a)
    v['_type'] = A.type_name
    self.assertEqual(symbolic.from_json(v), a)
    v['_type'] = 'main.Bar'
    self.assertEqual(symbolic.from_json(v), a)

    # Test custom serialization key for functor.
    @symbolic.functor(serialization_key='a.foo', additional_keys=['a.bar'])
    def foo(b):
      return b

    b = foo(1)
    v = b.to_json()
    self.assertEqual(v['_type'], 'a.foo')
    self.assertEqual(symbolic.from_json(v), b)
    v['_type'] = foo.type_name
    self.assertEqual(symbolic.from_json(v), b)
    v['_type'] = 'a.bar'
    self.assertEqual(symbolic.from_json(v), b)


class CopyTest(unittest.TestCase):
  """Tests for copy semantics on symbolic.Symbolic."""

  def setUp(self):
    super().setUp()

    class A:
      __metaclass__ = abc.ABCMeta

    @symbolic.members([
        ('x', schema.Int(1)), ('y', schema.Str()),
        ('z',
         schema.Dict([('p', schema.List(schema.Bool().noneable())),
                      ('q', schema.Enum('foo', ['foo', 'bar']))]))
    ])
    class B(symbolic.Object, A):
      pass

    class C(A):

      def __init__(self, x):
        self.x = x

      def __eq__(self, other):
        return isinstance(other, self.__class__) and other.x == self.x

      def __ne__(self, other):
        return not self.__eq__(other)

    value_spec = schema.Dict([
        ('a1', schema.Int()),
        ('a2',
         schema.Dict([('b1',
                       schema.Dict([('c1',
                                     schema.List(
                                         schema.Dict([('d1', schema.Str()),
                                                      ('d2', schema.Bool(True)),
                                                      ('d3', schema.Object(A)),
                                                      ('d4', schema.Object(A))
                                                     ])))]))]))
    ])
    # pylint: disable=invalid-name
    self._A = A
    self._B = B
    self._C = C
    self._value_spec = value_spec

  def testImplicitCopyDuringAssignment(self):
    """Test implicit copy during assignment."""

    # Test implicit copy for Dict.

    # There is no impliit copy when assigning a root symbolic object to
    # another tree.
    sd = symbolic.Dict({'a': 1})
    sd2 = symbolic.Dict({'b': sd})
    self.assertEqual(id(sd), id(sd2.b))

    # There is an implicit copy when assigning a symbolic object with
    # a parent to another tree.
    sd = symbolic.Dict(
        {'a': {
            'b': [self._B(y='foo', z={'p': []}),
                  self._C(1)]
        }})
    sd2 = symbolic.Dict({'c': sd.a})
    self.assertEqual(sd.a, sd2.c)
    self.assertNotEqual(id(sd.a), id(sd2.c))
    self.assertNotEqual(id(sd.a.b), id(sd2.c.b))
    self.assertNotEqual(id(sd.a.b[0]), id(sd2.c.b[0]))
    # Non-symbolic member is copy by reference.
    self.assertEqual(id(sd.a.b[1]), id(sd2.c.b[1]))

    # Test implicit copy for List with Object.
    @symbolic.members([('x', schema.Int())])
    class A(symbolic.Object):
      pass

    # There is no impliit copy when assigning a root symbolic object to
    # another tree.
    a = A(x=1)
    sl = symbolic.List([a])
    self.assertEqual(id(a), id(sl[0]))

    # There is an implicit copy when assigning a symbolic object with
    # a parent to another tree.
    sd = symbolic.Dict({'a': sl[0]})
    self.assertEqual(sl[0], sd.a)
    self.assertNotEqual(id(sl[0]), id(sd.a))

  def testCopy(self):
    """Test Dict.__copy__."""
    # For schema-less Dict.
    sd = symbolic.Dict.partial({
        'a1': 1,
        'a2': {
            'b1': {
                'c1': [{
                    'd1': 'bar',
                    'd3': self._B.partial(x=2, z={
                        'p': [None],
                        'q': 'bar'
                    }),
                    'd4': self._C(1)
                }]
            }
        }
    })
    sd2 = sd.copy()

    self.assertEqual(sd, sd2)
    self.assertNotEqual(id(sd), id(sd2))
    # For symbolic members, shallow copy will (shallow) copy the member.
    self.assertNotEqual(id(sd.a2.b1.c1[0].d3), id(sd2.a2.b1.c1[0].d3))

    # For non-symbolic members, shallow copy simple uses the same objects.
    self.assertEqual(id(sd.a2.b1.c1[0].d4), id(sd2.a2.b1.c1[0].d4))
    self.assertEqual(sd.a2.b1.c1[0].d3.sym_path, 'a2.b1.c1[0].d3')
    self.assertIsNone(sd2.value_spec)

    # Check root_path and parent is correctly set.
    c1_copy = sd.a2.b1.c1.clone()
    self.assertEqual(0, len(c1_copy.sym_path))
    self.assertIsNone(c1_copy.sym_parent)
    self.assertEqual(c1_copy[0].d3.sym_path, '[0].d3')

    # For symbolic Dict.
    sd = symbolic.Dict.partial(
        {
            'a1': 2,
            'a2': {
                'b1': {
                    'c1': [{
                        'd1':
                            'foo',
                        'd3':
                            self._B.partial(
                                x=4, z={
                                    'p': [None, True],
                                    'q': 'bar'
                                }),
                        'd4':
                            self._C(1)
                    }]
                }
            }
        },
        value_spec=self._value_spec)

    # Shallow copy using copy module.
    sd2 = sd.clone()
    self.assertEqual(sd, sd2)
    self.assertNotEqual(id(sd), id(sd2))

    # For symbolic members, shallow copy will (shallow) copy the member.
    self.assertNotEqual(id(sd.a2), id(sd2.a2))
    self.assertNotEqual(id(sd.a2.b1.c1[0].d3), id(sd2.a2.b1.c1[0].d3))

    # For non-symbolic members, shallow copy simple uses the same objects.
    self.assertEqual(id(sd.a2.b1.c1[0].d4), id(sd2.a2.b1.c1[0].d4))

    # Use the same spec instance.
    self.assertEqual(id(sd.value_spec), id(sd2.value_spec))

    # Shallow copy with override.
    sd3 = sd.clone(override={
        'a1': 3,
        'a2.b1.c1[0].d1': 'bar'
    })
    self.assertEqual(sd3.a1, 3)
    self.assertEqual(sd3.a2.b1.c1[0].d1, 'bar')

  def testDeepCopy(self):
    """Test Dict.__deepcopy__."""
    # For schema-less Dict.
    sd = symbolic.Dict.partial({
        'a1': 1,
        'a2': {
            'b1': {
                'c1': [{
                    'd1': 'bar',
                    'd3': self._B.partial(x=2, z={
                        'p': [None],
                        'q': 'bar'
                    }),
                    'd4': self._C(1)
                }]
            }
        }
    })
    sd2 = sd.clone(deep=True)

    self.assertEqual(sd, sd2)
    self.assertNotEqual(id(sd), id(sd2))
    self.assertIsNone(sd2.value_spec)

    # New instances are created for both symbolic and non-symbolic
    # members, recursively.
    self.assertNotEqual(id(sd.a2), id(sd2.a2))
    self.assertNotEqual(id(sd.a2.b1.c1[0].d3), id(sd2.a2.b1.c1[0].d3))
    self.assertNotEqual(id(sd.a2.b1.c1[0].d4), id(sd2.a2.b1.c1[0].d4))

    # For symbolic Dict.
    sd = symbolic.Dict.partial(
        {
            'a1': 2,
            'a2': {
                'b1': {
                    'c1': [{
                        'd1':
                            'foo',
                        'd3':
                            self._B.partial(
                                x=4, z={
                                    'p': [None, True],
                                    'q': 'bar'
                                }),
                        'd4':
                            self._C(1)
                    }]
                }
            }
        },
        value_spec=self._value_spec)

    sd2 = sd.clone(deep=True)
    self.assertEqual(sd, sd2)
    self.assertNotEqual(id(sd), id(sd2))
    self.assertNotEqual(id(sd.a2.b1.c1[0].d3), id(sd2.a2.b1.c1[0].d3))
    self.assertNotEqual(id(sd.a2.b1.c1[0].d4), id(sd2.a2.b1.c1[0].d4))
    self.assertEqual(id(sd.value_spec), id(sd2.value_spec))

    # Deep copy with override.
    sd3 = sd.clone(override={
        'a1': 3,
        'a2.b1.c1[0].d1': 'bar'
    })
    self.assertEqual(sd3.a1, 3)
    self.assertEqual(sd3.a2.b1.c1[0].d1, 'bar')


class InspectionTest(unittest.TestCase):
  """Tests for inspection methods."""

  def testMissingValues(self):
    """Tests Dict.missing_values."""

    @symbolic.members([
        ('x', schema.Int(1)), ('y', schema.Str()),
        ('z',
         schema.Dict([('p', schema.List(schema.Bool().noneable())),
                      ('q', schema.Enum('foo', ['foo', 'bar']))]))
    ])
    class A(symbolic.Object):
      pass

    value_spec = schema.Dict([
        ('a1', schema.Int()),
        ('a2', schema.Dict([
            ('b1', schema.Dict([
                ('c1', schema.List(schema.Dict([
                    ('d1', schema.Str()),
                    ('d2', schema.Bool(True)),
                    ('d3', schema.Object(A))
                ])))
            ]))
        ]))
    ])
    sd = symbolic.Dict.partial(value_spec=value_spec)
    self.assertEqual(
        sd.missing_values(flatten=True), {
            'a1': schema.MISSING_VALUE,
            'a2.b1.c1': schema.MISSING_VALUE,
        })
    self.assertEqual(
        sd.missing_values(flatten=False), {
            'a1': schema.MISSING_VALUE,
            'a2': {
                'b1': {
                    'c1': schema.MISSING_VALUE,
                }
            }
        })
    missing_a2 = sd.a2.missing_values(flatten=False)
    sd.rebind({
        'a1': 1
    })
    self.assertIs(sd.a2.missing_values(flatten=False), missing_a2)
    self.assertEqual(
        sd.missing_values(flatten=True), {
            'a2.b1.c1': schema.MISSING_VALUE,
        })

    sd.a2.b1.c1 = [{'d3': A.partial(x=2)}]
    self.assertEqual(
        sd.a2.b1.c1[0].missing_values(flatten=True), {
            'd1': schema.MISSING_VALUE,
            'd3.y': schema.MISSING_VALUE,
            'd3.z.p': schema.MISSING_VALUE
        })
    self.assertEqual(
        sd.a2.b1.c1[0].missing_values(flatten=False), {
            'd1': schema.MISSING_VALUE,
            'd3': {
                'y': schema.MISSING_VALUE,
                'z': {
                    'p': schema.MISSING_VALUE
                }
            }
        })

    # Inspect schema-less dict while its member is symbolic.
    sd = symbolic.Dict(
        a=symbolic.Dict.partial(
            value_spec=schema.Dict([('x', schema.Int())])))
    self.assertEqual(
        sd.missing_values(flatten=True), {
            'a.x': schema.MISSING_VALUE,
        })

    # Inspect schema-less list when its member is symbolic.
    sl = symbolic.List([
        symbolic.Dict.partial(value_spec=schema.Dict([('x', schema.Int())]))
    ])
    self.assertEqual(
        sl.missing_values(flatten=True), {'[0].x': schema.MISSING_VALUE})

  def testNonDefaultValues(self):
    """Test Dict.non_default_values."""

    @symbolic.members([
        ('x', schema.Int(1)), ('y', schema.Str()),
        ('z', schema.Dict([
            ('p', schema.List(schema.Bool().noneable())),
            ('q', schema.Enum('foo', ['foo', 'bar']))
        ]))
    ])
    class A(symbolic.Object):
      pass

    value_spec = schema.Dict([
        ('a1', schema.Int(0)),
        ('a2', schema.Dict([
            ('b1', schema.Dict([
                ('c1', schema.List(schema.Dict([
                    ('d1', schema.Str('foo')),
                    ('d2', schema.Bool(True)),
                    ('d3', schema.Object(A))
                ])))
            ]))
        ]))
    ])
    sd = symbolic.Dict.partial(
        {
            'a1': 1,
            'a2': {
                'b1': {
                    'c1': [{
                        'd1': 'bar',
                        'd3': A.partial(x=2, z={
                            'p': [None],
                            'q': 'bar'
                        })
                    }]
                }
            }
        },
        value_spec=value_spec)

    self.assertEqual(
        sd.non_default_values(flatten=True), {
            'a1': 1,
            'a2.b1.c1[0].d1': 'bar',
            'a2.b1.c1[0].d3.x': 2,
            'a2.b1.c1[0].d3.z.p[0]': None,
            'a2.b1.c1[0].d3.z.q': 'bar',
        })

    self.assertEqual(
        sd.non_default_values(flatten=False), {
            'a1': 1,
            'a2': {
                'b1': {
                    'c1': {
                        0: {
                            'd1': 'bar',
                            'd3': {
                                'x': 2,
                                'z': {
                                    'p': {
                                        0: None
                                    },
                                    'q': 'bar'
                                }
                            }
                        }
                    }
                }
            }
        })

    sd.rebind({
        'a1': 0,
        'a2.b1.c1[0].d1': 'bar2',
    })
    self.assertEqual(
        sd.non_default_values(flatten=True), {
            'a2.b1.c1[0].d1': 'bar2',
            'a2.b1.c1[0].d3.x': 2,
            'a2.b1.c1[0].d3.z.p[0]': None,
            'a2.b1.c1[0].d3.z.q': 'bar',
        })

    # Inspect schema-less dict while its member is symbolic.
    sd = symbolic.Dict(
        a=symbolic.Dict.partial(
            x=0,
            y=2,
            value_spec=schema.Dict([('x', schema.Int(default=1)),
                                    ('y', schema.Int(default=2))])),
        b=1)
    self.assertEqual(sd.non_default_values(flatten=True), {'a.x': 0, 'b': 1})

    # Inspect schema-less list while its member is symbolic.
    sl = symbolic.List([
        symbolic.Dict.partial(
            x=0,
            y=2,
            value_spec=schema.Dict([('x', schema.Int(default=1)),
                                    ('y', schema.Int(default=2))])), 1
    ])
    self.assertEqual(
        sl.non_default_values(flatten=True), {
            '[0].x': 0,
            '[1]': 1
        })

  def testContains(self):
    """Test contains."""

    @symbolic.members([
        ('x', schema.Any()),
        ('y', schema.Any())
    ])
    class A(symbolic.Object):
      pass

    self.assertTrue(symbolic.contains(A('a', 'b'), 'a'))
    self.assertTrue(symbolic.contains(A('a', 'b'), type=A))
    self.assertTrue(symbolic.contains(A([{'a': 1}], 'a'), 1))
    self.assertTrue(symbolic.contains(A(1, 2), type=int))
    self.assertTrue(symbolic.contains(A([{'a': 1}], 'a'), type=int))
    self.assertTrue(symbolic.contains(A('a', 'b'), type=(int, str)))
    self.assertFalse(symbolic.contains(A('a', 'b'), type=int))

  def testAbstractCheck(self):
    """Test pure symbolic check."""

    class X(symbolic.Object, symbolic.NonDeterministic):
      pass

    @symbolic.members([
        ('x', schema.Object(X))
    ])
    class Y(symbolic.Object, symbolic.NonDeterministic):
      pass

    class Z(symbolic.Object, symbolic.PureSymbolic):
      pass

    @symbolic.members([
        ('b', schema.Any())
    ])
    class A(symbolic.Object):
      pass

    @symbolic.functor()
    def foo(x, y):
      del x, y

    l = A(b={'c': [Y(x=X())], 'd': X(), 'e': Z()})

    # Test Symbolic.is_abstract.
    self.assertFalse(A(b=1).is_abstract)
    # Partial bound functor is not abstract.
    self.assertFalse(foo(1).is_abstract)  # pylint: disable=no-value-for-parameter
    # Functor with partial argument is abstract.
    self.assertTrue(foo(A.partial(), 2).is_abstract)
    self.assertTrue(A.partial().is_abstract)
    self.assertTrue(X().is_abstract)
    self.assertTrue(Z().is_abstract)
    self.assertTrue(l.is_abstract)
    l.rebind({
        'b.c[0]': 1,
        'b.d': 2
    })
    self.assertTrue(l.is_abstract)
    l.rebind({
        'b.e': 3
    })
    self.assertFalse(l.is_abstract)
    l.rebind({
        'b.e': Z()
    })
    self.assertTrue(l.is_abstract)
    l.rebind({
        'b.e': X()
    })

    # Test symbolic.is_abstract
    self.assertFalse(symbolic.is_abstract('abc'))
    self.assertFalse(symbolic.is_abstract(A(b=1)))
    self.assertTrue(symbolic.is_abstract(A.partial()))
    self.assertTrue(symbolic.is_abstract(l))
    self.assertTrue(symbolic.is_abstract(X()))
    self.assertTrue(symbolic.is_abstract(Z()))

    # Test symbolic.is_pure_symbolic
    self.assertFalse(symbolic.is_pure_symbolic('abc'))
    self.assertFalse(symbolic.is_pure_symbolic(A(b=1)))
    self.assertFalse(symbolic.is_pure_symbolic(A.partial()))
    self.assertTrue(symbolic.is_pure_symbolic(l))
    self.assertTrue(symbolic.is_pure_symbolic(X()))
    self.assertTrue(symbolic.is_pure_symbolic(Z()))

    # Test Symbolic.is_pure_symbolic
    self.assertFalse(A(b=1).is_pure_symbolic)
    self.assertFalse(A.partial().is_pure_symbolic)
    self.assertTrue(l.is_pure_symbolic)
    self.assertTrue(X().is_pure_symbolic)
    self.assertTrue(Z().is_pure_symbolic)

    # Test symbolic.is_deterministic.
    self.assertTrue(symbolic.is_deterministic('abc'))
    self.assertTrue(symbolic.is_deterministic(A(b=1)))
    self.assertFalse(symbolic.is_deterministic(l))
    self.assertFalse(symbolic.is_deterministic(X()))
    self.assertTrue(symbolic.is_deterministic(Z()))

    # Test Symbolic.is_deterministic.
    self.assertTrue(A(b=1).is_deterministic)
    self.assertFalse(l.is_deterministic)
    self.assertFalse(X().is_deterministic)

  def testEqual(self):
    """Test equal."""

    @symbolic.members([
        ('x', schema.Int(1)), ('y', schema.Str()),
        ('z',
         schema.Dict([('p', schema.List(schema.Bool().noneable())),
                      ('q', schema.Enum('foo', ['foo', 'bar']))]))
    ])
    class A(symbolic.Object):
      pass

    value_spec = schema.Dict([
        ('a1', schema.Int()),
        ('a2',
         schema.Dict([('b1',
                       schema.Dict([('c1',
                                     schema.List(
                                         schema.Dict([('d1', schema.Str()),
                                                      ('d2', schema.Bool(True)),
                                                      ('d3', schema.Object(A))
                                                     ])))]))]))
    ])

    # Comparing differences across locations.
    def _factory_fn1(x, p0):
      return symbolic.Dict.partial(
          {'a2': {
              'b1': {
                  'c1': [{
                      'd3': A.partial(x=x, z={'p': [p0]})
                  }]
              }
          }})

    self.assertEqual(_factory_fn1(0, True), _factory_fn1(0, True))
    self.assertNotEqual(_factory_fn1(1, True), _factory_fn1(0, True))
    self.assertNotEqual(_factory_fn1(0, False), _factory_fn1(0, True))

    # Dict with value_spec have default values.
    self.assertNotEqual(
        symbolic.Dict.partial(value_spec=value_spec),
        symbolic.Dict.partial())

    # Tests for `symbolic.eq` and `symbolic.ne`
    class B(symbolic.Object):

      def __eq__(self, other):
        """Overriding __eq__ should not impact `symbolic.eq`."""
        return self is other

      def __ne__(self, other):
        return not self.__eq__(other)

    b = B()
    self.assertEqual(b, b)
    self.assertNotEqual(b, B())
    self.assertTrue(symbolic.eq(B(), B()))
    self.assertFalse(symbolic.eq(1, B()))
    self.assertFalse(symbolic.eq(B(), 1))
    self.assertTrue(symbolic.ne(B(), 1))

  def testFormat(self):
    """Test format."""

    @symbolic.members([
        ('x', schema.Int(1)), ('y', schema.Str()),
        ('z',
         schema.Dict([('p', schema.List(schema.Bool().noneable())),
                      ('q', schema.Enum('foo', ['foo', 'bar'])),
                      ('t', schema.Str())]))
    ])
    class A(symbolic.Object):
      pass

    value_spec = schema.Dict([
        ('a1', schema.Int(1), 'Field a1.'),
        ('a2',
         schema.Dict([('b1',
                       schema.Dict([('c1',
                                     schema.List(
                                         schema.Dict([
                                             ('d1', schema.Str(), 'Field d1.'),
                                             ('d2', schema.Bool(True)),
                                             ('d3', schema.Object(A))
                                         ])), 'Field c1.')]), 'Field b1.')]),
         'Field a2.')
    ])

    d = symbolic.Dict.partial(
        {
            'a1': 1,
            'a2': {
                'b1': {
                    'c1': [{
                        'd3': A.partial(x=2, z={
                            'p': [None, True],
                            't': 'foo'
                        })
                    }]
                }
            }
        },
        value_spec=value_spec)

    # Format compact.
    self.assertEqual(
        d.format(compact=True),
        '{a1=1, a2={b1={c1=[0: {d1=MISSING_VALUE, d2=True, d3='
        'A(x=2, y=MISSING_VALUE, z={p=[0: None, 1: True], '
        'q=\'foo\', t=\'foo\'})}]}}}')

    # Format non-compact, non-verbose.
    self.assertEqual(
        d.format(compact=False, verbose=False),
        inspect.cleandoc("""{
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
        }"""))

    # Format non-compact, verbose.
    self.assertEqual(
        d.format(compact=False, verbose=True),
        inspect.cleandoc("""{
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
        }"""))

    # Format non-compact, verbose, hide default values and hide missing values.
    self.assertEqual(
        d.format(
            compact=False,
            verbose=True,
            hide_default_values=True,
            hide_missing_values=True),
        inspect.cleandoc("""{
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
        }"""))

  def testTraverse(self):
    """Test traverse."""

    @symbolic.members([('x', schema.Any())])
    class A(symbolic.Object):
      pass

    def visit_all(visited_keys):
      visited_keys[:] = []

      def _fn(k, v, p):
        del v, p
        visited_keys.append(str(k))
        return symbolic.TraverseAction.ENTER

      return _fn

    def visit_all_implicit(visited_keys):
      visited_keys[:] = []

      def _fn(k, v, p):
        del v, p
        visited_keys.append(str(k))

      return _fn

    def stop_after(path, visited_keys):
      visited_keys[:] = []

      def _fn(k, v, p):
        del v, p
        visited_keys.append(str(k))
        if k == path:
          return symbolic.TraverseAction.STOP
        return symbolic.TraverseAction.ENTER

      return _fn

    def enter_if_shallower_than(depth, visited_keys):
      visited_keys[:] = []

      def _fn(k, v, p):
        del v, p
        visited_keys.append(str(k))
        if len(k) < depth:
          return symbolic.TraverseAction.ENTER
        return symbolic.TraverseAction.CONTINUE

      return _fn

    preorder_paths = []
    postorder_paths = []
    v = [A(x={'y': A(x=0), 'z': 'foo'}), 1, 'bar']

    ret = symbolic.traverse(
        v,
        visit_all(preorder_paths),
        visit_all(postorder_paths)
    )
    self.assertTrue(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[0].x.y',
        '[0].x.y.x',
        '[0].x.z',
        '[1]',
        '[2]',
    ])
    self.assertEqual(postorder_paths, [
        '[0].x.y.x',
        '[0].x.y',
        '[0].x.z',
        '[0].x',
        '[0]',
        '[1]',
        '[2]',
        '',
    ])

    preorder_paths = []
    postorder_paths = []
    ret = symbolic.traverse(
        v,
        visit_all_implicit(preorder_paths),
        visit_all_implicit(postorder_paths)
    )
    self.assertTrue(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[0].x.y',
        '[0].x.y.x',
        '[0].x.z',
        '[1]',
        '[2]',
    ])
    self.assertEqual(postorder_paths, [
        '[0].x.y.x',
        '[0].x.y',
        '[0].x.z',
        '[0].x',
        '[0]',
        '[1]',
        '[2]',
        '',
    ])

    ret = symbolic.traverse(v, stop_after('[0].x.y', preorder_paths),
                            visit_all(postorder_paths))
    self.assertFalse(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[0].x.y',
    ])
    self.assertEqual(postorder_paths, ['[0].x.y', '[0].x', '[0]', ''])

    ret = symbolic.traverse(v, enter_if_shallower_than(2, preorder_paths),
                            visit_all(postorder_paths))
    self.assertTrue(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[1]',
        '[2]',
    ])
    self.assertEqual(postorder_paths, [
        '[0].x',
        '[0]',
        '[1]',
        '[2]',
        '',
    ])

  def testDiff(self):
    """Test diff."""

    @symbolic.members([('x', schema.Any())])
    class A(symbolic.Object):
      pass

    @symbolic.members([('x', schema.Any()), ('y', schema.Any())])
    class B(symbolic.Object):
      pass

    class C(B):
      pass

    @symbolic.members([('x', schema.Any()), ('y', schema.Any())])
    class D(symbolic.Object):
      pass

    # Test Diff.
    self.assertTrue(bool(symbolic.Diff(1, 2)))
    self.assertTrue(bool(symbolic.Diff(A, B)))
    self.assertTrue(bool(symbolic.Diff(A(1), A(2))))
    self.assertTrue(
        bool(symbolic.Diff(A, A, children={'x': symbolic.Diff(1, 2)})))
    self.assertFalse(bool(symbolic.Diff(1, 1)))
    self.assertFalse(bool(symbolic.Diff(A, A)))
    self.assertFalse(bool(symbolic.Diff(A(1), A(1))))

    self.assertTrue(symbolic.Diff(1, 2).is_leaf)
    self.assertFalse(
        symbolic.Diff(int, int, children={'x': symbolic.Diff(1, 2)}).is_leaf)

    self.assertEqual(symbolic.Diff(1, 1), 1)
    self.assertEqual(symbolic.Diff(1, 1), symbolic.Diff(1, 1))
    self.assertNotEqual(symbolic.Diff(1, 1), 2)
    self.assertNotEqual(symbolic.Diff(1, 1), symbolic.Diff(1, 2))

    self.assertEqual(
        repr(symbolic.Diff(A(1), A(1))), 'A(x=1)')
    self.assertEqual(
        repr(symbolic.Diff(A(1), A(2))), 'Diff(left=A(x=1), right=A(x=2))')
    self.assertEqual(
        repr(symbolic.Diff(A, A, children={'x': symbolic.Diff(1, 2)})),
        'A(x=Diff(left=1, right=2))')
    self.assertEqual(
        repr(symbolic.Diff(A, B, children={
            'x': symbolic.Diff(1, 2),
            'y': symbolic.Diff(symbolic.Diff.MISSING, 3)
        })),
        'A|B(x=Diff(left=1, right=2), y=Diff(left=MISSING, right=3))')
    self.assertEqual(
        repr(symbolic.Diff(symbolic.List, symbolic.List, children={
            '0': symbolic.Diff(1, 2),
            '1': symbolic.Diff(symbolic.Diff.MISSING, 3)
        })),
        '[0=Diff(left=1, right=2), 1=Diff(left=MISSING, right=3)]')

    with self.assertRaisesRegex(
        ValueError,
        'At least one of \'left\' and \'right\' should be specified.'):
      symbolic.Diff()

    with self.assertRaisesRegex(
        ValueError,
        '\'left\' must be a type when \'children\' is specified.'):
      symbolic.Diff(1, int, children={'x': symbolic.Diff(3, 4)})

    with self.assertRaisesRegex(
        ValueError,
        '\'right\' must be a type when \'children\' is specified.'):
      symbolic.Diff(int, 2, children={'x': symbolic.Diff(3, 4)})

    with self.assertRaisesRegex(
        ValueError,
        '\'value\' cannot be accessed when \'left\' and \'right\' '
        'are not the same.'):
      _ = symbolic.Diff(1, 2).value

    # Simple type
    self.assertEqual(symbolic.diff(1, 1), symbolic.Diff(1, 1))
    self.assertEqual(
        symbolic.diff(1, 1, mode='same'), symbolic.Diff(1, 1))
    self.assertEqual(
        symbolic.diff(1, 1, flatten=True, mode='same'), symbolic.Diff(1, 1))
    self.assertEqual(symbolic.diff(1, 2), symbolic.Diff(1, 2))
    self.assertEqual(symbolic.diff(1, 2, mode='same'), symbolic.Diff(1, 2))
    self.assertEqual(
        symbolic.diff(1, 2, flatten=True, mode='same'), symbolic.Diff(1, 2))

    # List types.
    # List vs. list.
    self.assertEqual(
        symbolic.diff([A(1)], [A(1)]),
        symbolic.Diff([A(1)], [A(1)]))
    self.assertEqual(
        symbolic.diff([A(1)], [A(0)]),
        symbolic.Diff(left=symbolic.List, right=symbolic.List, children={
            '0': symbolic.Diff(A, A, children={
                'x': symbolic.Diff(1, 0)
            })}))
    # List vs. other types.
    self.assertEqual(
        symbolic.diff([A(1)], 1),
        symbolic.Diff([A(1)], 1))

    # Dict types.
    # Dict vs. dict.
    self.assertEqual(
        symbolic.diff({'a': A(1)}, {'a': A(1)}),
        symbolic.Diff({'a': A(1)}, {'a': A(1)}))
    self.assertEqual(
        symbolic.diff({'a': A(1), 'b': A(2), 'c': A(3)},
                      {'a': A(1), 'b': A(3), 'd': A(4)}),
        symbolic.Diff(dict, dict, children={
            'b': symbolic.Diff(A, A, children={
                'x': symbolic.Diff(2, 3)
            }),
            'c': symbolic.Diff(A(3)),
            'd': symbolic.Diff(right=A(4)),
        }))

    # Dict vs. symbolic object.
    self.assertEqual(
        symbolic.diff(A(1), {'x': 1}),
        symbolic.Diff(A(1), {'x': 1}))

    self.assertEqual(
        symbolic.diff(A(1), {'x': 1}, collapse=True),
        symbolic.Diff(A, dict))

    self.assertEqual(
        symbolic.diff(A(1), {'x': 2}, collapse=True),
        symbolic.Diff(A, dict, children={
            'x': symbolic.Diff(1, 2)
        }))

    # Dict vs. other types.
    self.assertEqual(
        symbolic.diff({'x': 1}, 1),
        symbolic.Diff({'x': 1}, 1))

    # Symbolic types.
    # Same types.
    self.assertEqual(symbolic.diff(A(1), A(1)), symbolic.Diff(A(1), A(1)))
    self.assertEqual(
        symbolic.diff(B(1, 2), B(1, 3)),
        symbolic.Diff(B, B, children={'y': symbolic.Diff(2, 3)}))

    # Different types without collapse.
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 2)),
        symbolic.Diff(B(1, 2), C(1, 2)))

    # Different types with always collapse.
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 2), collapse=True),
        symbolic.Diff(B, C))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3)),
        symbolic.diff(B(1, 2), C(1, 3)))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3), collapse=True),
        symbolic.Diff(B, C, children={
            'y': symbolic.Diff(2, 3)
        }))

    # Different types with custom collapse function.
    def collapse_subclass(x, y):
      return issubclass(type(x), type(y)) or issubclass(type(y), type(x))

    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 2), collapse=collapse_subclass),
        symbolic.Diff(B, C))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3), collapse=collapse_subclass),
        symbolic.Diff(B, C, children={
            'y': symbolic.Diff(2, 3)
        }))
    self.assertEqual(
        symbolic.diff(B(1, 2), D(1, 3), collapse=collapse_subclass),
        symbolic.Diff(B(1, 2), D(1, 3)))

    # Test bad collapse option.
    with self.assertRaisesRegex(
        ValueError, 'Unsupported `collapse` value'):
      symbolic.diff(B(1, 2), C(1, 2), collapse='unsupported_option')

    # Test diff mode.
    self.assertEqual(
        symbolic.diff(A(1), A(1), mode='diff'),
        symbolic.Diff(A(1), A(1)))
    self.assertEqual(
        symbolic.diff(A(1), A(1), mode='same'),
        symbolic.Diff(A(1), A(1)))
    self.assertEqual(
        symbolic.diff(A(1), A(1), mode='both'),
        symbolic.Diff(A(1), A(1)))
    self.assertEqual(
        symbolic.diff(B(1, 2), B(1, 3), mode='diff'),
        symbolic.Diff(B, B, children={'y': symbolic.Diff(2, 3)}))
    self.assertEqual(
        symbolic.diff(B(1, 2), B(1, 3), mode='same'),
        symbolic.Diff(B, B, children={'x': symbolic.Diff(1, 1)}))
    self.assertEqual(
        symbolic.diff(B(1, 2), B(2, 2), mode='both'),
        symbolic.Diff(B, B, children={
            'x': symbolic.Diff(1, 2),
            'y': symbolic.Diff(2, 2)
        }))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 2), collapse=True, mode='diff'),
        symbolic.Diff(B, C))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 2), collapse=True, mode='same'),
        symbolic.Diff(B, C, children={
            'x': symbolic.Diff(1, 1),
            'y': symbolic.Diff(2, 2)
        }))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 2), collapse=True, mode='both'),
        symbolic.Diff(B, C, children={
            'x': symbolic.Diff(1, 1),
            'y': symbolic.Diff(2, 2)
        }))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3), collapse=True, mode='diff'),
        symbolic.Diff(B, C, children={
            'y': symbolic.Diff(2, 3)
        }))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3), collapse=True, mode='same'),
        symbolic.Diff(B, C, children={
            'x': symbolic.Diff(1, 1)
        }))
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3), collapse=True, mode='both'),
        symbolic.Diff(B, C, children={
            'x': symbolic.Diff(1, 1),
            'y': symbolic.Diff(2, 3)
        }))

    # Test flatten.
    self.assertEqual(
        symbolic.diff(A(symbolic.Dict(a=1, b=2, c=3)),
                      A(symbolic.Dict(a=1, b=3, d=4)), flatten=True),
        {
            'x.b': symbolic.Diff(2, 3),
            'x.c': symbolic.Diff(3, symbolic.Diff.MISSING),
            'x.d': symbolic.Diff(symbolic.Diff.MISSING, 4)
        })
    self.assertEqual(
        symbolic.diff(B(1, 2), C(1, 3), collapse=True, flatten=True),
        {
            'y': symbolic.Diff(2, 3),
            '_type': symbolic.Diff(B, C),
        })

  def testQuery(self):
    """Test query."""

    @symbolic.members([('x', schema.Int())])
    class A(symbolic.Object):
      pass

    @symbolic.members([
        ('a', schema.Object(A)),
        ('y', schema.Str()),
        ('z', schema.Int())
    ])
    class B(symbolic.Object):
      pass

    d = symbolic.List(
        [symbolic.Dict(a=A(x=0), b=B(a=A(x=1), y='foo', z=2))])

    # Query without path and value constraints.
    self.assertEqual(symbolic.query(d), {'': d})

    # Query with path regex.
    self.assertEqual(symbolic.query(d, r'.*y'), {'[0].b.y': 'foo'})

    # Query with value constraint.
    self.assertEqual(
        symbolic.query(d, where=lambda v: isinstance(v, int)), {
            '[0].a.x': 0,
            '[0].b.a.x': 1,
            '[0].b.z': 2
        })

    # Query with both path and value constraints.
    self.assertEqual(
        symbolic.query(d, r'.*a', where=lambda v: isinstance(v, int)), {
            '[0].a.x': 0,
            '[0].b.a.x': 1,
        })

    # Query with value and parent constraints.
    self.assertEqual(
        symbolic.query(
            d, where=lambda v, p: isinstance(v, int) and not isinstance(p, A)),
        {
            '[0].b.z': 2,
        })

    # Query with `enter_selected` flag.
    self.assertEqual(
        symbolic.query(
            d, where=lambda v: isinstance(v, symbolic.Object),
            enter_selected=True),
        {
            '[0].a': A(x=0),
            '[0].b': B(a=A(x=1), y='foo', z=2),
            '[0].b.a': A(x=1),
        })

    # Query with custom selector using (key_path, value).
    self.assertEqual(
        symbolic.query(
            d, custom_selector=lambda k, v: len(k) == 2 and isinstance(v, A)),
        {'[0].a': A(x=0)})

    # Query with custom selector using (key_path, value, parent).
    self.assertEqual(
        symbolic.query(
            d,
            custom_selector=(
                lambda k, v, p: len(k) > 2 and isinstance(p, A) and v > 0)),
        {'[0].b.a.x': 1})

    # Query with no match.
    self.assertEqual(0, len(symbolic.query(d, r'xx')))

    with self.assertRaisesRegex(
        TypeError, 'Where function .* should accept 1 or 2 arguments'):
      symbolic.query(d, where=lambda: True)

    with self.assertRaisesRegex(
        TypeError, 'Custom selector .* should accept 2 or 3 arguments'):
      symbolic.query(d, custom_selector=lambda: True)

    with self.assertRaisesRegex(
        ValueError, '\'path_regex\' and \'where\' must be None when '
        '\'custom_selector\' is provided'):
      symbolic.query(d, path_regex=r'x', custom_selector=lambda: True)

  def testInspect(self):
    """Test `symbolic.inspect`."""

    class StringStream:
      """String stream for testing purpose."""

      def __init__(self):
        self.str = ''

      def write(self, content):
        self.str += content

      def reset(self):
        self.str = ''

    string_stream = StringStream()
    d = symbolic.Dict(x=1, y={'a': 'foo'})
    d.inspect(file=string_stream, compact=True)
    self.assertEqual(string_stream.str, '{x=1, y={a=\'foo\'}}\n')
    string_stream.reset()
    d.inspect(where=lambda v: v == 1, file=string_stream)
    self.assertEqual(string_stream.str, '{\n  \'x\': 1\n}\n')

  def testSignature(self):
    """Test inspecting `pg.Object.__init__`'s signature."""

    @symbolic.members([
        ('x', schema.Int(default=1)),
        ('y', schema.Any()),
        ('z', schema.List(schema.Int())),
        (schema.StrKey(), schema.Str())
    ])
    class A(symbolic.Object):
      pass

    signature = inspect.signature(A.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'x', 'y', 'z', 'kwargs'])

    self.assertEqual(signature.parameters['self'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['self'].annotation,
                     inspect.Signature.empty)
    self.assertEqual(signature.parameters['self'].default,
                     inspect.Signature.empty)

    self.assertEqual(signature.parameters['x'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['x'].annotation, int)
    self.assertEqual(signature.parameters['x'].default, 1)

    self.assertEqual(signature.parameters['y'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].default, schema.MISSING_VALUE)

    self.assertEqual(signature.parameters['z'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['z'].annotation, typing.List[int])
    self.assertEqual(signature.parameters['z'].default, schema.MISSING_VALUE)

    self.assertEqual(signature.parameters['kwargs'].kind,
                     inspect.Parameter.VAR_KEYWORD)
    self.assertEqual(signature.parameters['kwargs'].annotation, str)

    @symbolic.members([
        ('x', schema.Int(default=1)),
        ('y', schema.Any()),
        ('z', schema.List(schema.Int())),
    ], init_arg_list=['y', '*z'])
    class B(symbolic.Object):
      pass

    signature = inspect.signature(B.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'y', 'z', 'x'])

    self.assertEqual(signature.parameters['self'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['self'].annotation,
                     inspect.Signature.empty)
    self.assertEqual(signature.parameters['self'].default,
                     inspect.Signature.empty)

    self.assertEqual(signature.parameters['y'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].default, inspect.Signature.empty)

    self.assertEqual(signature.parameters['z'].kind,
                     inspect.Parameter.VAR_POSITIONAL)
    self.assertEqual(signature.parameters['z'].annotation, int)
    self.assertEqual(signature.parameters['z'].default, inspect.Signature.empty)

    self.assertEqual(signature.parameters['x'].kind,
                     inspect.Parameter.KEYWORD_ONLY)
    self.assertEqual(signature.parameters['x'].annotation, int)
    self.assertEqual(signature.parameters['x'].default, 1)

    class C(B):
      """Custom __init__."""

      def __init__(self, a, b):
        super().__init__(b, x=a)

    signature = inspect.signature(C.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'a', 'b'])

    @symbolic.functor([
        ('a', schema.Int())
    ])
    def foo(a=1, *b, c, **d):  # pylint: disable=keyword-arg-before-vararg
      del a, b, c, d

    signature = inspect.signature(foo.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'a', 'b', 'c', 'd'])
    self.assertEqual(signature.parameters['self'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['self'].annotation,
                     inspect.Signature.empty)
    self.assertEqual(signature.parameters['self'].default,
                     inspect.Signature.empty)

    self.assertEqual(signature.parameters['a'].kind,
                     inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['a'].annotation, int)
    self.assertEqual(signature.parameters['a'].default, 1)

    self.assertEqual(signature.parameters['b'].kind,
                     inspect.Parameter.VAR_POSITIONAL)
    self.assertEqual(signature.parameters['b'].annotation,
                     inspect.Signature.empty)

    self.assertEqual(signature.parameters['c'].kind,
                     inspect.Parameter.KEYWORD_ONLY)
    self.assertEqual(signature.parameters['c'].annotation,
                     inspect.Signature.empty)
    self.assertEqual(signature.parameters['c'].default,
                     inspect.Signature.empty)

    self.assertEqual(signature.parameters['d'].kind,
                     inspect.Parameter.VAR_KEYWORD)
    self.assertEqual(signature.parameters['d'].annotation,
                     inspect.Signature.empty)


class BoilerplateClassTest(unittest.TestCase):
  """Tests for symbolic.boilerplate_class."""

  def setUp(self):
    """Setup test."""
    super().setUp()

    @symbolic.members([
        ('a', schema.Int()),
        ('b', schema.Union([schema.Int(), schema.Str()])),
        ('c', schema.Dict([
            ('d', schema.List(schema.Dict([
                ('e', schema.Float()),
                ('f', schema.Bool())
            ])))
        ]))
    ])
    class A(symbolic.Object):
      pass

    self._a = A.partial(b='foo', c={'d': [{'e': 1.0, 'f': True}]})

    # pylint: disable=invalid-name
    B = symbolic.boilerplate_class('B', self._a)
    self._A = A
    self._B = B

    C = symbolic.boilerplate_class('C', self._a, init_arg_list=['a', 'c', 'b'])
    self._C = C

  def testAutomaticRegistration(self):
    """Test automatic registration."""
    self.assertIs(
        object_utils.JSONConvertible.class_from_typename(self._B.type_name),
        self._B)

  def testBasics(self):
    """Test if B is created as expected."""
    self.assertTrue(issubclass(self._B, self._A))
    self.assertEqual(self._B.type_name, 'pyglove.core.symbolic_test.B')

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'value\' must be an instance of symbolic.Object'):
      symbolic.boilerplate_class('A', 1)

    with self.assertRaisesRegex(
        TypeError, 'Unsupported keyword arguments'):
      symbolic.boilerplate_class('A', self._a, unsupported_keyword=1)

  def testSchema(self):
    """Test schema is correctly set with default values."""
    # Boilerplate class' schema should carry the default value and be frozen.
    self.assertEqual(
        self._B.schema,
        schema.create_schema([
            ('a', schema.Int()),
            ('b', schema.Union(
                [schema.Int(), schema.Str()], default='foo').freeze()),
            ('c', schema.Dict([
                ('d', schema.List(schema.Dict([
                    ('e', schema.Float()),
                    ('f', schema.Bool())
                ]), default=symbolic.List(
                    [symbolic.Dict(e=1.0, f=True)])).freeze())
            ]).freeze())
        ]))

    # Original class' schema should remain unchanged.
    self.assertEqual(
        self._A.schema,
        schema.create_schema([
            ('a', schema.Int()),
            ('b', schema.Union([schema.Int(), schema.Str()])),
            ('c', schema.Dict([
                ('d', schema.List(schema.Dict([
                    ('e', schema.Float()),
                    ('f', schema.Bool())
                ])))
            ]))
        ]))

  def testInit(self):
    """Test created objet should have default values bound."""
    b = self._B(0)
    self.assertEqual(
        b,
        self._B.partial(
            a=0, b='foo', c={'d': [{
                'e': 1.0,
                'f': True
            }]}))
    self.assertEqual(self._C.init_arg_list, ['a', 'c', 'b'])

  def testDoNotModifyOriginalObject(self):
    """Test rebind of produced object does not change original object."""
    b = self._B(a=1)
    with self.assertRaisesRegex(
        ValueError, 'Frozen field is not assignable.'):
      b.rebind(b=1)

    b.rebind({'c.d[0].f': False})
    self.assertFalse(b.c.d[0].f)

    # Default value of the boilerplate class remain unchanged.
    self.assertEqual(
        self._B.schema['c'].default_value,
        symbolic.Dict.partial({'d': [{
            'e': 1.0,
            'f': True,
        }]}, value_spec=self._B.schema['c'].value))

    # Original object remain unchanged.
    self.assertTrue(self._a.c.d[0].f)

  def testSerialization(self):
    """Serialized boilerplate class objects should not carry frozen fields."""
    b = self._B(a=1)
    self.assertEqual(b.to_json(), {
        '_type': 'pyglove.core.symbolic_test.B',
        'a': 1
    })


class LoadSaveHanlderTest(unittest.TestCase):
  """Tests for global load and save handler."""

  def testDefaultLoadSaveHandler(self):
    """Tests for default load/save handler."""

    @symbolic.members([
        ('a', schema.Int()),
        ('b', schema.List(schema.Int()))
    ])
    class A(symbolic.Object):
      pass

    tmp_dir = tempfile.gettempdir()

    # Test save/load in JSON.
    path = os.path.join(tmp_dir, 'a.json')
    symbolic.save(A(a=1, b=[0, 1]), path)
    with open(path) as f:
      content = f.read()
    self.assertEqual(content, '{"_type": "pyglove.core.symbolic_test.A", "a": 1, "b": [0, 1]}')

    # Test tracking origin.
    with symbolic.track_origin():
      a = symbolic.load(path)
    self.assertEqual(a, A(a=1, b=[0, 1]))
    self.assertEqual(a.sym_origin.source, path)
    self.assertEqual(a.sym_origin.tag, 'load')

  def testCustomLoadSave(self):
    """Test custom load/save handler."""
    repo = {}
    def _load(name):
      return repo[name]
    old_loader = symbolic.set_load_handler(_load)
    self.assertIs(old_loader, symbolic.default_load_handler)
    self.assertIs(symbolic.get_load_handler(), _load)

    def _save(value, name):
      repo[name] = value
    old_saver = symbolic.set_save_handler(_save)
    self.assertIs(old_saver, symbolic.default_save_handler)
    self.assertIs(symbolic.get_save_handler(), _save)

    # Test 'symbolic.save/load'.
    symbolic.save([1, 2, 3], 'foo')
    self.assertEqual(symbolic.load('foo'), [1, 2, 3])

    # Test 'symbolic.save/load'.
    @symbolic.members([
        ('x', schema.Int(1)),
        ('y', schema.Str())
    ])
    class A(symbolic.Object):
      pass

    A(y='abc').save('bar')
    self.assertEqual(A.load('bar'), A(y='abc'))
    with self.assertRaisesRegex(
        TypeError, 'Value is not of type .*'):
      symbolic.Dict.load('bar')

    # Test 'save/load' with empty save/load handler.
    with self.assertRaisesRegex(
        ValueError, '`load_handler` must be callable.'):
      symbolic.set_load_handler(None)

    with self.assertRaisesRegex(
        ValueError, '`save_handler` must be callable.'):
      symbolic.set_save_handler(None)

    symbolic.set_load_handler(symbolic.default_load_handler)
    symbolic.set_save_handler(symbolic.default_save_handler)


if __name__ == '__main__':
  unittest.main()
