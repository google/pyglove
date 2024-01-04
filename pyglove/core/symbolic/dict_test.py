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
"""Tests for pyglove.Dict."""

import copy
import inspect
import io
import pickle
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags
from pyglove.core.symbolic import inferred
from pyglove.core.symbolic import object as pg_object
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.pure_symbolic import NonDeterministic
from pyglove.core.symbolic.pure_symbolic import PureSymbolic


MISSING_VALUE = object_utils.MISSING_VALUE


class DictTest(unittest.TestCase):
  """Tests for `pg.Dict`."""

  def test_init(self):
    # Schemaless dict.
    sd = Dict()
    self.assertIsNone(sd.value_spec)
    self.assertEqual(len(sd), 0)

    # Schemaless dict created from a regular dict.
    sd = Dict({'a': 1})
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd, dict(a=1))

    # Schemaless dict created from key value pairs.
    sd = Dict((('a', 1),))
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd, dict(a=1))

    # Schemaless dict created from keyword args.
    sd = Dict(a=1)
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd, dict(a=1))

    # Schemaless dict created from both a regular dict and keyword args.
    sd = Dict({'a': 1}, a=2)
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd, dict(a=2))

    # Schematized dict.
    vs = pg_typing.Dict([('a', pg_typing.Int())])
    sd = Dict(a=1, value_spec=vs)
    self.assertIs(sd.value_spec, vs)
    self.assertEqual(sd, dict(a=1))

    # Dict with inferred value
    sd = Dict(Dict(x=inferred.ValueFromParentChain()))
    self.assertEqual(sd, dict(x=inferred.ValueFromParentChain()))

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified.'):
      Dict(value_spec=vs)

    # Schematized dict with default value.
    vs = pg_typing.Dict([('a', pg_typing.Int(default=0))])
    sd = Dict(value_spec=vs)
    self.assertIs(sd.value_spec, vs)
    self.assertEqual(sd, dict(a=0))

    # Bad init.
    with self.assertRaisesRegex(TypeError, '.* is not iterable'):
      Dict(1)

    with self.assertRaisesRegex(
        TypeError, '.* must be a `pg.typing.Dict` object'):
      Dict(value_spec=pg_typing.Int())

  def test_partial(self):
    spec = pg_typing.Dict([
        ('a', pg_typing.Int()),
        ('b', pg_typing.Int())
    ])
    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified.'):
      _ = Dict(a=1, value_spec=spec)
    self.assertTrue(Dict.partial(value_spec=spec).is_partial)

    with flags.allow_partial(True):
      self.assertTrue(Dict(value_spec=spec).is_partial)

    sd = Dict.partial(a=1, value_spec=spec)
    self.assertTrue(sd.is_partial)
    sd.rebind(b=2)
    self.assertFalse(sd.is_partial)

  def test_missing_value_as_values(self):
    sd = Dict(x=1, y=MISSING_VALUE)
    self.assertEqual(sd, Dict(x=1))
    sd.rebind(z=MISSING_VALUE, raise_on_no_change=False)
    self.assertEqual(sd, Dict(x=1))

    sd = Dict(x=1, value_spec=pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Int())
    ]))
    sd['y'] = MISSING_VALUE
    sd['x'] = MISSING_VALUE
    self.assertNotIn('y', sd)
    self.assertNotIn('x', sd)

  def test_runtime_type_check(self):
    sd = Dict.partial(
        value_spec=pg_typing.Dict([
            ('a', pg_typing.Int()),
            ('b', pg_typing.Dict()),
            ('c', pg_typing.Dict([
                ('x', pg_typing.Int())]))
        ]))
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      sd.a = Dict()

    sd.b = Dict()
    sd.c = Dict(x=1)

    # Target field cannot accept Dict of different schema.
    with self.assertRaisesRegex(
        KeyError, 'Keys .* are not allowed in Schema'):
      sd.c = Dict(y=1)

    # Target field can accept Dict with the same pg_typing.
    sd.c = Dict(x=1, value_spec=pg_typing.Dict([('x', pg_typing.Int())]))

    # Target field cannot accept Dict with incompatible pg_typing.
    with self.assertRaisesRegex(
        ValueError, 'Dict .* cannot be assigned to an incompatible field'):
      sd.c = Dict(x=1, value_spec=pg_typing.Dict())

    # Target field can accept any Dict.
    sd.b = Dict(x=1, value_spec=pg_typing.Dict([('x', pg_typing.Int())]))

    # Test compatibility with Any type.
    spec = pg_typing.Dict([('x', pg_typing.Any())])
    sd = Dict(x={'a': 1, 'b': 2}, value_spec=spec)
    self.assertIsNone(sd.x.value_spec)

    # Test compatibility with Union type.
    spec = pg_typing.Dict([
        ('x', pg_typing.Union([
            pg_typing.Int(),
            pg_typing.Bool(),
            pg_typing.Dict([
                ('a', pg_typing.Int()),
                ('b', pg_typing.Str().noneable())
            ])
        ]))])
    sd = Dict(x={'a': 1}, value_spec=spec)
    self.assertEqual(sd.x, {'a': 1, 'b': None})
    self.assertEqual(
        sd.x.value_spec,
        pg_typing.Dict([
            ('a', pg_typing.Int()),
            ('b', pg_typing.Str().noneable())
        ]))

  def test_symbolization_on_nested_containers(self):
    sd = Dict(b={'c': True, 'd': []}, a=0)
    self.assertIsInstance(sd.b, Dict)
    self.assertIsInstance(sd.b.d, List)

  def test_implicit_copy_during_assignment(self):

    class A:
      pass

    # There is no impliit copy when assigning a root symbolic object to
    # another tree.
    sd = Dict(x=dict(), y=list(), z=A())
    sd2 = Dict(b=sd)

    self.assertIs(sd, sd2.b)

    # There is an implicit copy when assigning a symbolic object with
    # a parent to another tree.
    sd3 = Dict(b=sd)
    self.assertIsNot(sd, sd3.b)
    self.assertIsNot(sd.x, sd3.b.x)
    self.assertIsNot(sd.y, sd3.b.y)
    # Non-symbolic member is copied by reference.
    self.assertIs(sd.z, sd3.b.z)

    self.assertEqual(sd, sd3.b)
    self.assertEqual(sd.x, sd3.b.x)
    self.assertEqual(sd.y, sd3.b.y)

  def test_dict_with_special_chars_in_keys(self):
    # Test schema-free dict with special characters.
    sd = Dict()
    sd['x.y'] = Dict(z=List())
    self.assertIn('x.y', sd)
    self.assertEqual(sd['x.y'].z.sym_path, '[x.y].z')
    self.assertEqual(sd.rebind({'[x.y].z': 2}), {
        'x.y': {
            'z': 2
        }
    })

    # Test dict special characters against a Dict pg_typing.
    sd.clone().use_value_spec(pg_typing.Dict())
    sd.clone().use_value_spec(pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Dict())
    ]))

  def test_spec_compatibility_during_assignment(self):
    sl = List([], value_spec=pg_typing.List(pg_typing.Dict([
        ('x', pg_typing.Int(min_value=0))])))
    sd = Dict(x=0, value_spec=pg_typing.Dict([('x', pg_typing.Int())]))
    with self.assertRaisesRegex(
        ValueError, 'Dict .* cannot be assigned to an incompatible field'):
      # A list of dicts with non-negative values cannot accept a dict that
      # possibly contains negative integers.
      sl.append(sd)

  def test_inspect(self):
    s = io.StringIO()
    sd = Dict(x=1, y={'a': 'foo'})
    sd.inspect(file=s, compact=True)
    self.assertEqual(s.getvalue(), '{x=1, y={a=\'foo\'}}\n')

    s = io.StringIO()
    sd.inspect(where=lambda v: v == 1, file=s)
    self.assertEqual(s.getvalue(), '{\n  \'x\': 1\n}\n')

  def test_list(self):
    sd = Dict(a=1, b=2)
    self.assertEqual(list(sd), ['a', 'b'])

    spec = pg_typing.Dict([
        ('b', pg_typing.Bool(default=True)),
        ('a', pg_typing.Int()),
    ])
    sd = Dict(a=1, value_spec=spec)
    self.assertEqual(list(sd), ['b', 'a'])

    sd = Dict.partial(value_spec=spec)
    self.assertEqual(list(sd), ['b', 'a'])
    self.assertEqual(sd.a, MISSING_VALUE)

  def test_len(self):
    sd = Dict(a=1, b=2)
    self.assertEqual(len(sd), 2)

    spec = pg_typing.Dict([
        ('b', pg_typing.Bool(default=True)),
        ('a', pg_typing.Int()),
    ])
    sd = Dict(a=1, value_spec=spec)
    self.assertEqual(len(sd), 2)

    sd = Dict.partial(value_spec=spec)
    self.assertEqual(len(sd), 2)
    self.assertEqual(sd.a, MISSING_VALUE)

  def test_in(self):
    sd = Dict(a=1, b=2)
    self.assertIn('a', sd)
    self.assertNotIn('x', sd)

    spec = pg_typing.Dict([
        ('b', pg_typing.Bool(default=True)),
        ('a', pg_typing.Int()),
    ])
    sd = Dict(a=1, value_spec=spec)
    self.assertIn('a', sd)
    self.assertIn('b', sd)
    self.assertNotIn('x', sd)

    sd = Dict.partial(value_spec=spec)
    self.assertIn('a', sd)
    self.assertIn('b', sd)
    self.assertNotIn('x', sd)
    self.assertEqual(sd.a, MISSING_VALUE)

  def test_setitem(self):
    # Set item in a schemaless dict.
    sd = Dict()
    sd['a'] = 0
    sd['a'] = 0
    self.assertEqual(sd, dict(a=0))
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
        sd['b'] = 1
    with self.assertRaisesRegex(KeyError, 'Key must be string type'):
      sd[0] = 1

    # Set item in a schematized dict.
    sd = Dict(value_spec=pg_typing.Dict([('a', pg_typing.Int(default=0))]))
    self.assertEqual(sd, dict(a=0))
    sd['a'] = 1
    self.assertEqual(sd, dict(a=1))
    with self.assertRaisesRegex(KeyError, 'Key \'b\' is not allowed'):
      sd['b'] = 2

    # Set item in a schematized dict with StrKey.
    sd = Dict(value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        (pg_typing.StrKey('x.*'), pg_typing.Int())
    ]))
    self.assertEqual(sd, dict(a=0))
    sd['a'] = 1
    self.assertEqual(sd, dict(a=1))
    sd['x1'] = 2
    sd['x2'] = 3
    self.assertEqual(sd, dict(a=1, x1=2, x2=3))
    with self.assertRaisesRegex(KeyError, 'Key \'y1\' is not allowed'):
      sd['y1'] = 4

    # Set item with an inferred value.
    sd = Dict(x=1)
    sd.x = inferred.ValueFromParentChain()
    self.assertEqual(sd.sym_getattr('x'), inferred.ValueFromParentChain())

  def test_getitem(self):
    sd = Dict(a=1)
    self.assertEqual(sd['a'], 1)
    with self.assertRaisesRegex(KeyError, 'x'):
      _ = sd['x']

    sd = Dict(x=inferred.ValueFromParentChain())
    with self.assertRaisesRegex(KeyError, 'x'):
      _ = sd['x']

    sdd = Dict(x=Dict(foo=0), y=sd)
    self.assertIs(sd['x'], sdd['x'])

  def test_delitem(self):
    # Delete an item from a schemaless dict.
    sd = Dict(a=1, b=0)
    del sd['a']
    self.assertEqual(len(sd), 1)

    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot del item from a sealed Dict.'):
        del sd['b']
    del sd['b']
    self.assertEqual(len(sd), 0)

    with self.assertRaisesRegex(KeyError, 'Key does not exist in Dict'):
      del sd['x']

    # Delete a required field from a schematized dict.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int())
    ]))
    with self.assertRaisesRegex(ValueError, 'Required value is not specified.'):
      del sd['a']

    # Deleting a key with default value restores its default value.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0))
    ]))
    del sd['a']
    self.assertEqual(sd, dict(a=0))

    # Deleting a dynamic key.
    sd = Dict(x1=1, x2=2, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        (pg_typing.StrKey('x.*'), pg_typing.Int())
    ]))
    del sd['x1']
    self.assertEqual(sd, dict(a=0, x2=2))

  def test_setattr(self):
    sd = Dict()
    sd.a = 0
    self.assertEqual(sd, dict(a=0))
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
        sd.b = 1

    # Set item in a schematized dict.
    sd = Dict(value_spec=pg_typing.Dict([('a', pg_typing.Int(default=0))]))
    self.assertEqual(sd, dict(a=0))
    sd.a = 1
    self.assertEqual(sd, dict(a=1))
    with self.assertRaisesRegex(KeyError, 'Key \'b\' is not allowed'):
      sd.b = 2

    # Set item in a schematized dict with StrKey.
    sd = Dict(value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        (pg_typing.StrKey('x.*'), pg_typing.Int())
    ]))
    self.assertEqual(sd, dict(a=0))
    sd.a = 1
    self.assertEqual(sd, dict(a=1))
    sd.x1 = 2
    sd.x2 = 3
    self.assertEqual(sd, dict(a=1, x1=2, x2=3))
    with self.assertRaisesRegex(KeyError, 'Key \'y1\' is not allowed'):
      sd.y1 = 4

  def test_getattr(self):
    sd = Dict(a=1)
    self.assertEqual(sd.a, 1)
    with self.assertRaisesRegex(AttributeError, 'Attribute .* does not exist'):
      _ = sd.x

    # Test inferred value.
    sd = Dict(x=inferred.ValueFromParentChain())
    with self.assertRaisesRegex(
        AttributeError, '`x` is not found under its context'
    ):
      _ = sd.x

    p = Dict(x=Dict(p='foo'), y=Dict(z=sd))
    self.assertEqual(sd.x, Dict(p='foo'))
    self.assertIs(p.x, sd.x)

  def test_delattr(self):
    # Delete an item from a schemaless dict.
    sd = Dict(a=1, b=0)
    del sd.a
    self.assertEqual(len(sd), 1)
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot del item from a sealed Dict.'):
        del sd.b
    del sd.b
    self.assertEqual(len(sd), 0)
    with self.assertRaisesRegex(KeyError, 'Key does not exist in Dict'):
      del sd.x

    # Delete a required field from a schematized dict.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int())
    ]))
    with self.assertRaisesRegex(ValueError, 'Required value is not specified.'):
      del sd.a

    # Deleting a key with default value restores its default value.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0))
    ]))
    del sd.a
    self.assertEqual(sd, dict(a=0))

    # Deleting a dynamic key.
    sd = Dict(x1=1, x2=2, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        (pg_typing.StrKey('x.*'), pg_typing.Int())
    ]))
    del sd.x1
    self.assertEqual(sd, dict(a=0, x2=2))

  def test_pop(self):
    sd = Dict(b=0, a=1, c=2)
    self.assertEqual(sd.pop('a'), 1)
    self.assertEqual(list(sd.keys()), ['b', 'c'])

    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot del item from a sealed Dict.'):
        sd.pop('b')

    self.assertIsNone(sd.pop('a', None))
    with self.assertRaisesRegex(KeyError, 'a'):
      sd.pop('a')

    # Poping a required key triggers value error.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int())
    ]))
    with self.assertRaisesRegex(ValueError, 'Required value is not specified.'):
      sd.pop('a')

    # Poping a key with default value restores its default value.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0))
    ]))
    self.assertEqual(sd.pop('a'), 1)
    self.assertEqual(sd, dict(a=0))

  def test_popitem(self):
    sd = Dict(b=0, a=1)
    self.assertEqual(sd.popitem(), ('a', 1))
    self.assertEqual(list(sd.keys()), ['b'])

    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot pop item from a sealed Dict.'):
        sd.popitem()

    self.assertEqual(sd.popitem(), ('b', 0))
    self.assertEqual(len(sd), 0)

    with self.assertRaisesRegex(
        KeyError, 'dictionary is empty'):
      sd.popitem()

    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int())
    ]))
    with self.assertRaisesRegex(
        ValueError, 'cannot be performed on a Dict with value spec'):
      sd.popitem()

  def test_clear(self):
    sd = Dict(b=0, a=1, c=2)
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot clear a sealed Dict.'):
        sd.clear()
    sd.clear()
    self.assertEqual(len(sd), 0)

    # Clearing a dict with required keys triggers value error.
    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int())
    ]))
    with self.assertRaisesRegex(ValueError, 'Required value is not specified.'):
      sd.clear()

    # Clearing a dict with all keys with default values restore their default
    # values.
    sd = Dict(a=1, b=2, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Int(default=1))
    ]))
    sd.clear()
    self.assertEqual(sd, dict(a=0, b=1))

  def test_copy(self):

    class A:
      pass

    sd = Dict(a=0, b=A(), c=dict(d=A()))
    sd2 = sd.copy()
    self.assertIsInstance(sd2, Dict)
    self.assertEqual(sd, sd2)
    self.assertIsNot(sd, sd2)
    # Shallow copy of regular objet.
    self.assertIs(sd.b, sd2.b)
    self.assertIs(sd.c.d, sd2.c.d)

    # Deep copy of symbolic object.
    self.assertIsNot(sd.c, sd2.c)

    sd = Dict(a=0, value_spec=pg_typing.Dict([('a', pg_typing.Int())]))
    sd2 = sd.copy()
    self.assertIs(sd.value_spec, sd2.value_spec)

    # Test copy.copy
    sd = Dict(a=0, b=A(), c=dict(d=A()))
    sd3 = copy.copy(sd)
    self.assertIsInstance(sd3, Dict)
    self.assertEqual(sd, sd3)
    self.assertIsNot(sd, sd3)
    # Shallow copy of regular objet.
    self.assertIs(sd.b, sd3.b)
    self.assertIs(sd.c.d, sd3.c.d)
    # Deep copy of symbolic object.
    self.assertIsNot(sd.c, sd3.c)

    # Test copy.deepcopy.

    class B:

      def __init__(self, v):
        self.v = v

      def __eq__(self, other):
        return isinstance(other, B) and self.v == other.v

    sd = Dict(a=0, b=B(1), c=dict(d=B(2)))
    sd4 = copy.deepcopy(sd)
    self.assertIsInstance(sd4, Dict)
    self.assertEqual(sd, sd4)
    self.assertIsNot(sd, sd4)
    # Deep copy of regular objet.
    self.assertIsNot(sd.b, sd4.b)
    self.assertIsNot(sd.c.d, sd4.c.d)
    # Deep copy of symbolic object.
    self.assertIsNot(sd.c, sd4.c)

  def test_setdefault(self):
    sd = Dict(a=0, b=1)
    self.assertEqual(sd.setdefault('a'), 0)
    self.assertIsNone(sd.setdefault('x'))
    self.assertIsNone(sd.x)

    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
        sd.setdefault('y')

    sd = Dict(a=0, value_spec=pg_typing.Dict([('a', pg_typing.Int())]))
    self.assertEqual(sd.setdefault('a'), 0)
    with self.assertRaisesRegex(KeyError, 'Key .* is not allowed'):
      sd.setdefault('y', 1)

    sd = Dict.partial(value_spec=pg_typing.Dict([('a', pg_typing.Int())]))
    self.assertEqual(sd.setdefault('a', 1), 1)

    sd = Dict()
    sd.setdefault('x', inferred.ValueFromParentChain())
    self.assertEqual(sd.sym_getattr('x'), inferred.ValueFromParentChain())
    sd.setdefault('x', 1)
    self.assertEqual(sd.sym_getattr('x'), inferred.ValueFromParentChain())

  def test_update(self):
    sd = Dict(b=0, a=1, c=2)
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot rebind key .* of sealed Dict'):
        sd.update(a=1)
    sd.update({'a': MISSING_VALUE, 'b': 1, 'd': 3})
    self.assertEqual(sd, {'b': 1, 'c': 2, 'd': 3})

    sd.update((('c', 3),))
    self.assertEqual(sd, {'b': 1, 'c': 3, 'd': 3})

    sd.update(b=2, d=MISSING_VALUE, e=4)
    self.assertEqual(sd, {'b': 2, 'c': 3, 'e': 4})

    sd.update({'b': 3}, b=4)
    self.assertEqual(sd, {'b': 4, 'c': 3, 'e': 4})

    sd = Dict(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0))
    ]))
    sd.update(a=2)
    self.assertEqual(sd, dict(a=2))
    with self.assertRaisesRegex(KeyError, 'Key \'b\' is not allowed.'):
      sd.update(b=2)

  def test_use_value_spec(self):
    spec = pg_typing.Dict([
        ('a', pg_typing.Int(min_value=0)),
        ('b', pg_typing.Bool().noneable()),
    ])
    sd = Dict(a=1, b=True)
    with self.assertRaisesRegex(
        ValueError, 'Value spec for list must be a `pg.typing.Dict` object'):
      sd.use_value_spec(pg_typing.Int())
    sd.use_value_spec(spec)

    # Apply the same schema twice to verify its eligibility.
    sd.use_value_spec(spec)

    with self.assertRaisesRegex(
        RuntimeError, 'Dict is already bound with a different value spec: .*'):
      sd.use_value_spec(
          pg_typing.Dict([
              ('a', pg_typing.Int(min_value=2)),
              ('b', pg_typing.Bool().noneable()),
          ]))

    # Remove schema constraint and insert new keys.
    sd.use_value_spec(None)
    sd['c'] = 1
    self.assertIn('c', sd)

    with flags.enable_type_check(False):
      # Shall not trigger error, since type/value check is not enabled.
      sd.use_value_spec(spec)

  def test_iter(self):
    sd = Dict(b={'c': True, 'd': []}, a=0)
    self.assertEqual(list(sd), ['b', 'a'])

  def test_reversed(self):
    sd = Dict(b={'c': True, 'd': []}, a=0)
    self.assertEqual(list(reversed(sd)), ['a', 'b'])

  def test_keys(self):
    sd = Dict(b={'c': True, 'd': []}, a=0)
    self.assertEqual(list(sd.keys()), ['b', 'a'])

    # For schematized dict, key order is determined by schema.
    sd = Dict(a=1, b=2, c=3, value_spec=pg_typing.Dict([
        ('b', pg_typing.Int()),
        ('a', pg_typing.Int()),
        ('c', pg_typing.Int()),
    ]))
    self.assertEqual(list(sd.keys()), ['b', 'a', 'c'])

  def test_values(self):
    sd = Dict(b={'c': True, 'd': []}, a=0)
    self.assertEqual(list(sd.values()), [{'c': True, 'd': []}, 0])

    # For schematized dict, value order is determined by schema.
    sd = Dict(a=1, b=2, c=3, value_spec=pg_typing.Dict([
        ('b', pg_typing.Int()),
        ('a', pg_typing.Int()),
        ('c', pg_typing.Int()),
    ]))
    self.assertEqual(list(sd.values()), [2, 1, 3])

  def test_items(self):
    sd = Dict(b={'c': True, 'd': []}, a=0)
    self.assertEqual(list(sd.items()), [('b', {'c': True, 'd': []}), ('a', 0)])

    # For schematized dict, item order is determined by schema.
    sd = Dict(a=1, b=2, c=3, value_spec=pg_typing.Dict([
        ('b', pg_typing.Int()),
        ('a', pg_typing.Int()),
        ('c', pg_typing.Int()),
    ]))
    self.assertEqual(list(sd.items()), [('b', 2), ('a', 1), ('c', 3)])

  def test_non_default(self):
    sd = Dict(a=1)
    self.assertEqual(len(sd.non_default_values()), 1)

    sd = Dict(a=1, b=0, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Int(default=0))
    ]))
    self.assertEqual(sd.non_default_values(), dict(a=1))

    sd = Dict(a=1, b=dict(c=2), value_spec=pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Dict([
            ('c', pg_typing.Int(default=1))
        ]))
    ]))
    self.assertEqual(sd.non_default_values(), {'a': 1, 'b.c': 2})
    self.assertEqual(
        sd.non_default_values(flatten=False), {'a': 1, 'b': {'c': 2}})

    # After rebind, the non_default_values are updated.
    sd.rebind({'b.c': 1})
    self.assertEqual(sd.non_default_values(flatten=False), {'a': 1})

    # A non-schematized dict has a schematized child.
    sd = Dict(x=sd)
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd.non_default_values(), {'x.a': 1})

  def test_missing_values(self):
    sd = Dict(a=1)
    self.assertEqual(len(sd.missing_values()), 0)

    sd = Dict.partial(a=1, value_spec=pg_typing.Dict([
        ('a', pg_typing.Int()),
        ('b', pg_typing.Int())
    ]))
    self.assertEqual(sd.missing_values(), dict(b=MISSING_VALUE))

    sd = Dict.partial(b=dict(d=1), value_spec=pg_typing.Dict([
        ('a', pg_typing.Int()),
        ('b', pg_typing.Dict([
            ('c', pg_typing.Int()),
            ('d', pg_typing.Int())
        ]))
    ]))
    self.assertEqual(
        sd.missing_values(),
        {'a': MISSING_VALUE, 'b.c': MISSING_VALUE})
    self.assertEqual(
        sd.missing_values(flatten=False),
        {'a': MISSING_VALUE, 'b': {'c': MISSING_VALUE}})

    # After rebind, `missing_values` is updated.
    sd.rebind({'b.c': 1})
    self.assertEqual(
        sd.missing_values(flatten=False), {'a': MISSING_VALUE})

    # A non-schematized dict has a schematized child.
    sd = Dict(x=sd)
    self.assertIsNone(sd.value_spec)
    self.assertEqual(sd.missing_values(), {'x.a': MISSING_VALUE})

  def test_sym_has(self):
    sd = Dict(x=1, y=Dict(z=2))
    self.assertTrue(sd.sym_has('x'))
    self.assertTrue(sd.sym_has('y.z'))
    self.assertTrue(sd.sym_has(object_utils.KeyPath.parse('y.z')))
    self.assertFalse(sd.sym_has('x.z'))

  def test_sym_get(self):
    sd = Dict(x=1, y=Dict(z=2))
    self.assertEqual(sd.sym_get('x'), 1)
    self.assertEqual(sd.sym_get('y.z'), 2)
    self.assertIsNone(sd.sym_get('x.z', None))
    with self.assertRaisesRegex(
        KeyError, 'Cannot query sub-key \'z\' of object.'):
      sd.sym_get('x.z')

  def test_sym_hasattr(self):
    sd = Dict(x=1, y=Dict(z=2))
    self.assertTrue(sd.sym_hasattr('x'))
    self.assertFalse(sd.sym_hasattr('y.z'))
    self.assertFalse(sd.sym_hasattr('a'))

  def test_sym_getattr(self):
    sd = Dict(x=1, y=Dict(z=2))
    self.assertEqual(sd.sym_getattr('x'), 1)
    self.assertIsNone(sd.sym_getattr('a', None))
    with self.assertRaisesRegex(
        AttributeError,
        '.* object has no symbolic attribute \'a\'.'):
      sd.sym_getattr('a')

    sd = Dict(x=inferred.ValueFromParentChain())
    self.assertEqual(sd.sym_getattr('x'), inferred.ValueFromParentChain())

  def test_sym_inferred(self):
    sd = Dict(x=1, y=inferred.ValueFromParentChain())  # pylint: disable=no-value-for-parameter
    self.assertEqual(sd.sym_inferred('x'), 1)
    with self.assertRaisesRegex(AttributeError, 'y'):
      _ = sd.sym_inferred('y')
    with self.assertRaisesRegex(AttributeError, 'z'):
      _ = sd.sym_inferred('z')

    sd = Dict(y=1, x=Dict(x=Dict(y=inferred.ValueFromParentChain())))
    self.assertEqual(sd.x.x.y, 1)

  def test_sym_field(self):
    sd = Dict(x=1, y=Dict(z=2))
    self.assertIsNone(sd.sym_field)

    spec = pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict())
    ])
    sd.use_value_spec(spec)
    self.assertIsNone(sd.sym_field)
    self.assertIs(sd.y.sym_field, spec.schema.get_field('y'))

  def test_sym_attr_field(self):
    sd = Dict(x=1, y=Dict(z=2))
    self.assertIsNone(sd.sym_attr_field('x'))

    spec = pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict())
    ])
    sd.use_value_spec(spec)
    self.assertIs(sd.sym_attr_field('x'), spec.schema.get_field('x'))

  def test_sym_keys(self):
    sd = Dict(x=1, y=2)
    self.assertEqual(next(sd.sym_keys()), 'x')
    self.assertEqual(list(sd.sym_keys()), ['x', 'y'])

    sd = Dict(x=1, z=3, y=2, value_spec=pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Int())
    ]))
    self.assertEqual(next(sd.sym_keys()), 'x')
    self.assertEqual(list(sd.sym_keys()), ['x', 'z', 'y'])

    sd = Dict(x=1, y=inferred.ValueFromParentChain())
    self.assertEqual(next(sd.sym_keys()), 'x')
    self.assertEqual(list(sd.sym_keys()), ['x', 'y'])

  def test_sym_values(self):
    sd = Dict(x=1, y=2)
    self.assertEqual(next(sd.sym_values()), 1)
    self.assertEqual(list(sd.sym_values()), [1, 2])

    sd = Dict(x=1, z=3, y=2, value_spec=pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Int())
    ]))
    self.assertEqual(next(sd.sym_values()), 1)
    self.assertEqual(list(sd.sym_values()), [1, 3, 2])

    sd = Dict(x=1, y=inferred.ValueFromParentChain())
    self.assertEqual(next(sd.sym_values()), 1)
    self.assertEqual(
        list(sd.sym_values()), [1, inferred.ValueFromParentChain()]
    )

  def test_sym_items(self):
    sd = Dict(x=1, y=2)
    self.assertEqual(next(sd.sym_items()), ('x', 1))
    self.assertEqual(list(sd.sym_items()), [('x', 1), ('y', 2)])

    sd = Dict(x=1, z=3, y=2, value_spec=pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Int())
    ]))
    self.assertEqual(next(sd.sym_items()), ('x', 1))
    self.assertEqual(list(sd.sym_items()), [('x', 1), ('z', 3), ('y', 2)])

    sd = Dict(x=1, y=inferred.ValueFromParentChain())
    self.assertEqual(next(sd.sym_items()), ('x', 1))
    self.assertEqual(
        list(sd.sym_items()), [('x', 1), ('y', inferred.ValueFromParentChain())]
    )

  def test_sym_jsonify(self):
    # Refer to SerializationTest for more detailed tests.
    sd = Dict(x=1, y=inferred.ValueFromParentChain())
    self.assertEqual(
        sd.sym_jsonify(),
        {'x': 1, 'y': inferred.ValueFromParentChain().to_json()},
    )

  def test_sym_rebind(self):
    # Refer to RebindTest for more detailed tests.
    sd = Dict(x=1, y=2)
    sd.sym_rebind(x=2)
    self.assertEqual(sd, dict(x=2, y=2))

  def test_sym_clone(self):
    class A:
      pass

    sd = Dict(x=[], y=dict(), z=A())
    sd2 = sd.clone()
    self.assertEqual(sd, sd2)
    self.assertIsNot(sd, sd2)
    # Symbolic members are always copied by value.
    self.assertIsNot(sd.x, sd2.x)
    self.assertIsNot(sd.y, sd2.y)

    # Non-symbolic members are copied by reference.
    self.assertIs(sd.z, sd2.z)

    value_spec = pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Any())
    ])
    sd = Dict(x=list(), z=dict(), y=A(), value_spec=value_spec)
    sd2 = sd.sym_clone(deep=True)

    # Instances of `A` are compared by reference.
    # During deep clone A() is copied which results in a different instance.
    self.assertNotEqual(sd, sd2)
    self.assertIs(sd.value_spec, sd2.value_spec)
    self.assertIsNot(sd, sd2)
    self.assertIsNot(sd.x, sd2.x)
    self.assertIsNot(sd.y, sd2.y)
    self.assertIsNot(sd.z, sd2.z)

  def test_sym_origin(self):
    # Refer `object_test.test_sym_origin` for more details.
    sd = Dict(x=1)
    sd.sym_setorigin(Dict.__init__, 'constructor')
    self.assertEqual(sd.sym_origin.source, Dict.__init__)
    self.assertEqual(sd.sym_origin.tag, 'constructor')

  def test_sym_partial(self):
    # Refer to `test_partial` for more details.
    sd = Dict.partial(x=1, value_spec=pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict([
            ('z', pg_typing.Int())
        ])),
    ]))
    self.assertTrue(sd.sym_partial)
    sd.rebind({'y.z': 2})
    self.assertFalse(sd.sym_partial)

  def test_sym_missing(self):
    # Refer to `test_missing_values` for more details.
    sd = Dict.partial(x=1, value_spec=pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int()),
    ]))
    self.assertEqual(sd.sym_missing(), {'y': MISSING_VALUE})

    # Test inferred value as the default value.
    sd = Dict(
        x=inferred.ValueFromParentChain(),
        value_spec=pg_typing.Dict([
            ('x', pg_typing.Int()),
        ]),
    )
    self.assertEqual(sd.sym_missing(), {})

  def test_sym_nondefault(self):

    class A(pg_object.Object):
      x: int
      use_symbolic_comparison = False

    class B(pg_object.Object):
      y: int = 1
      use_symbolic_comparison = True

    sd = Dict(x=1, y=dict(a1=A(1)), value_spec=pg_typing.Dict([
        ('x', pg_typing.Int(default=0)),
        ('y', pg_typing.Dict([
            ('z', pg_typing.Int(default=1)),
            ('a1', pg_typing.Object(A)),
            ('a2', pg_typing.Object(A, default=A(1))),
            ('b', pg_typing.Object(B, default=B(2))),
        ])),
    ]))
    self.assertTrue(base.eq(sd.sym_nondefault(), {'x': 1, 'y.a1': A(1)}))
    sd.rebind({'y.z': 2, 'y.a2': A(2), 'y.b': B(1)}, x=0)
    self.assertTrue(
        base.eq(
            sd.sym_nondefault(),
            {'y.z': 2, 'y.a1': A(1), 'y.a2.x': 2, 'y.b.y': 1}
        )
    )

    # Test inferred value as the default value.
    sd = Dict(
        x=1,
        value_spec=pg_typing.Dict([
            ('x', pg_typing.Int(default=inferred.ValueFromParentChain())),
        ]),
    )
    self.assertEqual(sd.sym_nondefault(), {'x': 1})
    sd.rebind(x=inferred.ValueFromParentChain())
    self.assertEqual(sd.sym_nondefault(), {})

    # Test inferred value as the specified value.
    sd = Dict(
        x=inferred.ValueFromParentChain(),
        value_spec=pg_typing.Dict([
            ('x', pg_typing.Int(default=1)),
        ]),
    )
    self.assertEqual(
        sd.sym_nondefault(), {'x': inferred.ValueFromParentChain()}
    )
    sd.rebind(x=1)
    self.assertEqual(sd.sym_nondefault(), {})

  def test_sym_puresymbolic(self):
    self.assertFalse(Dict(x=1).sym_puresymbolic)

    class A(PureSymbolic):
      pass

    self.assertTrue(Dict(x=A()).sym_puresymbolic)

  def test_sym_abstract(self):
    self.assertFalse(Dict(x=1).sym_abstract)
    self.assertTrue(Dict.partial(value_spec=pg_typing.Dict([
        ('x', pg_typing.Int())
    ])).sym_abstract)

    class A(PureSymbolic):
      pass

    self.assertTrue(Dict(x=A()).sym_abstract)

  def test_is_deterministic(self):

    class X(NonDeterministic):
      pass

    self.assertTrue(Dict().is_deterministic)
    self.assertFalse(Dict(x=X()).is_deterministic)
    self.assertFalse(Dict(x=[dict(y=X())]).is_deterministic)

  def test_sym_contains(self):
    sd = Dict(x=dict(y=[dict(z=1)]))
    self.assertTrue(sd.sym_contains(value=1))
    self.assertFalse(sd.sym_contains(value=2))
    self.assertTrue(sd.sym_contains(type=int))
    self.assertFalse(sd.sym_contains(type=str))

  def test_sym_eq(self):
    # Use cases that `__eq__` and `sym_eq` have the same results.
    self.assertEqual(Dict(), Dict())
    self.assertTrue(Dict().sym_eq(Dict()))
    self.assertTrue(base.eq(Dict(), Dict()))

    self.assertEqual(Dict(a=1), Dict(a=1))
    self.assertTrue(Dict(a=1).sym_eq(Dict(a=1)))
    self.assertTrue(base.eq(Dict(a=1), Dict(a=1)))
    self.assertTrue(
        base.eq(
            Dict(x=inferred.ValueFromParentChain()),
            Dict(x=inferred.ValueFromParentChain()),
        )
    )
    self.assertEqual(
        Dict(a=1),
        Dict(a=1, value_spec=pg_typing.Dict([('a', pg_typing.Int())])))
    self.assertTrue(base.eq(
        Dict(a=1),
        Dict(a=1, value_spec=pg_typing.Dict([('a', pg_typing.Int())]))))
    self.assertEqual(Dict(a=Dict()), Dict(a=dict()))
    self.assertTrue(base.eq(Dict(a=Dict()), Dict(a=dict())))

    # Use case that `__eq__` rules both Python equality and `pg.eq`.
    class A:

      def __init__(self, value):
        self.value = value

      def __eq__(self, other):
        return ((isinstance(other, A) and self.value == other.value)
                or self.value == other)

    self.assertEqual(Dict(a=A(1)), Dict(a=1))
    self.assertTrue(base.eq(Dict(a=A(1)), Dict(a=1)))

    # Use case that `sym_eq` only rule `pg.eq` but not Python equality.
    class B:

      def __init__(self, value):
        self.value = value

      def sym_eq(self, other):
        return ((isinstance(other, A) and self.value == other.value)
                or self.value == other)

    self.assertNotEqual(Dict(a=B(1)), Dict(a=1))
    self.assertTrue(base.eq(Dict(a=B(1)), Dict(a=1)))

  def test_sym_ne(self):
    # Refer test_sym_eq for more details.
    self.assertNotEqual(Dict(), 1)
    self.assertTrue(base.ne(Dict(), 1))
    self.assertNotEqual(Dict(), Dict(a=1))
    self.assertTrue(base.ne(Dict(), Dict(a=1)))
    self.assertNotEqual(Dict(a=0), Dict(a=1))
    self.assertTrue(base.ne(Dict(a=0), Dict(a=1)))

  def test_sym_lt(self):
    self.assertFalse(Dict().sym_lt(MISSING_VALUE))
    self.assertFalse(Dict().sym_lt(None))
    self.assertFalse(Dict().sym_lt(True))
    self.assertFalse(Dict().sym_lt(1))
    self.assertFalse(Dict().sym_lt(2.0))
    self.assertFalse(Dict().sym_lt('abc'))
    self.assertFalse(Dict().sym_lt([]))
    self.assertFalse(Dict().sym_lt(tuple()))
    self.assertFalse(Dict().sym_lt(set()))
    self.assertFalse(Dict().sym_lt(Dict()))

    self.assertTrue(Dict().sym_lt(Dict(a=0)))
    self.assertTrue(Dict(a=0).sym_lt(Dict(a=1)))
    self.assertTrue(Dict(a=0).sym_lt(Dict(a=0, b=1)))
    self.assertTrue(Dict(a=0, b=1).sym_lt(Dict(a=1)))
    self.assertFalse(Dict(a=0).sym_lt(Dict(a=0)))
    self.assertFalse(Dict(a=1).sym_lt(Dict(a=0)))
    self.assertFalse(Dict(a=0, b=1).sym_lt(Dict(a=0)))
    self.assertFalse(Dict(a=1).sym_lt(Dict(a=0, b=1)))

    class A:
      pass

    self.assertTrue(Dict().sym_lt(A()))

  def test_sym_gt(self):
    self.assertTrue(Dict().sym_gt(MISSING_VALUE))
    self.assertTrue(Dict().sym_gt(None))
    self.assertTrue(Dict().sym_gt(True))
    self.assertTrue(Dict().sym_gt(1))
    self.assertTrue(Dict().sym_gt(2.0))
    self.assertTrue(Dict().sym_gt('abc'))
    self.assertTrue(Dict().sym_gt([]))
    self.assertTrue(Dict().sym_gt((1,)))
    self.assertTrue(Dict().sym_gt(set()))

    self.assertTrue(Dict(a=0).sym_gt(Dict()))
    self.assertTrue(Dict(a=1).sym_gt(Dict(a=0)))
    self.assertTrue(Dict(a=0, b=1).sym_gt(Dict(a=0)))
    self.assertTrue(Dict(b=0).sym_gt(Dict(a=0)))
    self.assertFalse(Dict().sym_gt(Dict(a=0)))
    self.assertFalse(Dict(a=0).sym_gt(Dict(a=1)))
    self.assertFalse(Dict(a=0).sym_gt(Dict(a=0, b=1)))
    self.assertFalse(Dict(a=0, b=1).sym_gt(Dict(a=1)))
    self.assertFalse(Dict(a=0).sym_gt(Dict(a=0)))
    self.assertFalse(Dict(a=0).sym_gt(Dict(a=1)))
    self.assertFalse(Dict(a=0).sym_gt(Dict(a=0, b=1)))

    class A:
      pass

    self.assertFalse(Dict().sym_gt(A()))

  def test_sym_hash(self):
    self.assertEqual(hash(Dict()), hash(Dict()))
    self.assertEqual(hash(Dict(a=1)), hash(Dict(a=1)))
    self.assertEqual(hash(Dict(a=dict(x=1))), hash(Dict(a=dict(x=1))))
    self.assertNotEqual(hash(Dict()), hash(Dict(a=1)))
    self.assertNotEqual(hash(Dict(a=1)), hash(Dict(a=2)))

    class A:
      pass

    a = A()
    b = A()
    self.assertNotEqual(hash(Dict(x=a)), hash(Dict(x=b)))

    class B:

      def __init__(self, value):
        self.value = value

      def __hash__(self):
        return hash((B, self.value))

    a = B(1)
    b = B(1)
    self.assertEqual(hash(a), hash(b))
    self.assertEqual(hash(Dict(x=a)), hash(Dict(x=b)))
    self.assertNotEqual(hash(Dict(x=B(1))), hash(Dict(x=B(2))))

    class C(pg_object.Object):
      x: str
      use_symbolic_comparison = False

    self.assertEqual(Dict(x=C('abc')).sym_hash(), Dict(x=C('abc')).sym_hash())

  def test_sym_parent(self):
    sd = Dict(x=dict(a=1), y=[])
    self.assertIsNone(sd.sym_parent)

    self.assertIs(sd.x.sym_parent, sd)
    self.assertIs(sd.y.sym_parent, sd)

    pd = Dict(a=sd)
    self.assertIs(sd.sym_parent, pd)

  def test_sym_root(self):
    sd = Dict(x=dict(a=1), y=[])
    self.assertIs(sd.sym_root, sd)
    self.assertIs(sd.x.sym_parent, sd)
    self.assertIs(sd.y.sym_parent, sd)

    pd = Dict(a=sd)
    self.assertIs(sd.sym_root, pd)

  def test_sym_path(self):
    sd = Dict(x=dict(a=dict()), y=[dict(b=dict())])
    self.assertEqual(sd.sym_path, '')
    self.assertEqual(sd.x.sym_path, 'x')
    self.assertEqual(sd.x.a.sym_path, 'x.a')
    self.assertEqual(sd.y[0].b.sym_path, 'y[0].b')

    sd.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(sd.sym_path, 'a')
    self.assertEqual(sd.x.sym_path, 'a.x')
    self.assertEqual(sd.x.a.sym_path, 'a.x.a')
    self.assertEqual(sd.y[0].b.sym_path, 'a.y[0].b')

  def test_accessor_writable(self):
    sd = Dict(a=0, accessor_writable=False)
    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot modify Dict field by attribute or key while accessor_writable '
        'is set to False.'):
      sd.a = 2

    with flags.allow_writable_accessors(True):
      sd.a = 2
      self.assertEqual(sd.a, 2)

    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot modify Dict field by attribute or key while accessor_writable '
        'is set to False.'):
      sd['a'] = 1

    with flags.allow_writable_accessors(True):
      sd['a'] = 1
      self.assertEqual(sd.a, 1)

    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot del Dict field by attribute or key while accessor_writable is '
        'set to False.'):
      del sd.a

    with flags.allow_writable_accessors(True):
      del sd.a
      self.assertNotIn('a', sd)
      sd.a = 1

    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot del Dict field by attribute or key while accessor_writable is '
        'set to False.'):
      del sd['a']

    with flags.allow_writable_accessors(True):
      del sd['a']
      self.assertNotIn('a', sd)

    sd.rebind(a=2)
    self.assertEqual(sd.a, 2)

    # Delete key with rebind.
    sd.rebind(a=MISSING_VALUE)
    self.assertEqual(0, len(sd))

    sd.set_accessor_writable(True)
    sd.a = 1
    self.assertEqual(sd.a, 1)
    with flags.allow_writable_accessors(False):
      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot modify Dict field by attribute or key while '
          'accessor_writable is set to False.'):
        sd.a = 2

      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot modify Dict field by attribute or key while '
          'accessor_writable is set to False.'):
        sd['a'] = 2

      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot del Dict field by attribute or key while accessor_writable '
          'is set to False.'):
        del sd.a

      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot del Dict field by attribute or key while accessor_writable '
          'is set to False.'):
        del sd['a']

    # Test with inferred value.
    class Unresolvable(inferred.InferredValue):

      def infer(self):
        raise ValueError()

    # Test accessor writable within a sub-tree.
    x = Unresolvable()
    self.assertFalse(x.accessor_writable)

    sd = Dict(x=x)  # pylint: disable=no-value-for-parameter

    self.assertTrue(sd.accessor_writable)
    self.assertFalse(x.accessor_writable)

    sd.set_accessor_writable(False)
    self.assertFalse(sd.accessor_writable)
    self.assertFalse(sd.accessor_writable)

    sd.set_accessor_writable(True)
    self.assertTrue(sd.accessor_writable)
    self.assertFalse(x.accessor_writable)

    x.set_accessor_writable(True)
    self.assertTrue(sd.accessor_writable)
    self.assertTrue(x.accessor_writable)

    sd.set_accessor_writable(False)
    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot modify Dict field by attribute or key while '
        'accessor_writable is set to False.',
    ):
      sd.x = 1

  def test_mark_missing_values(self):
    # For schemaless Dict.
    sd = Dict(x=1, y=2)
    self.assertIn('x', sd)
    self.assertIn('y', sd)

    # Set field to MISSING_VALUE will delete field.
    sd.x = MISSING_VALUE
    self.assertNotIn('x', sd)

    # Clear will empty the dict.
    sd.clear()
    self.assertEqual(0, len(sd))

    # Mark missing values in symbolic Dict.
    value_spec = pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Dict([
            ('z', pg_typing.Bool(True)),
            ('p', pg_typing.Str())]))
    ])
    sd = Dict.partial(value_spec=value_spec)
    self.assertEqual(sd, {
        'x': MISSING_VALUE,
        'y': {
            'z': True,
            'p': MISSING_VALUE,
        }
    })
    sd.y.z = False

    # Assign MISSING_VALUE to a field with default value
    # will reset field to default value
    sd.y.z = MISSING_VALUE
    self.assertEqual(sd.y.z, True)

    # Clear will reset default values.
    sd.clear()
    self.assertEqual(sd, {
        'x': MISSING_VALUE,
        'y': {
            'z': True,
            'p': MISSING_VALUE,
        }
    })

  def test_seal(self):
    sd = Dict({'a': 0}, sealed=True)
    self.assertTrue(sd.is_sealed)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
      sd.a = 1

    with flags.as_sealed(False):
      sd.a = 2
      self.assertEqual(sd.a, 2)
      # Object-level is_sealed flag is not modified.
      self.assertTrue(sd.is_sealed)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
      sd['a'] = 1

    with flags.as_sealed(False):
      sd['a'] = 1
      self.assertEqual(sd['a'], 1)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      sd.rebind(a=1)

    with flags.as_sealed(False):
      sd.rebind(a=2)
      self.assertEqual(sd.a, 2)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      sd.update({'a': 1})

    with flags.as_sealed(False):
      sd.update({'a': 1})
      self.assertEqual(sd.a, 1)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot del item from a sealed Dict.'):
      del sd['a']

    with flags.as_sealed(False):
      del sd['a']
      self.assertNotIn('a', sd)
      sd.a = 1

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot del item from a sealed Dict.'):
      del sd.a

    with flags.as_sealed(False):
      del sd.a
      self.assertNotIn('a', sd)
      sd.a = 1

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot clear a sealed Dict.'):
      sd.clear()

    with flags.as_sealed(False):
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

    with flags.as_sealed(True):
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot modify field of a sealed Dict.'):
        sd.a = 1

      # Object-level sealed state is not changed,
      self.assertFalse(sd.is_sealed)

    # Seal again.
    self.assertEqual(sd.b, 2)
    sd.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot rebind key .* of sealed Dict.'):
      sd.rebind(a=0)

    # Test nested sealed Dict.
    sd = Dict(a=Dict())
    self.assertFalse(sd.is_sealed)
    self.assertFalse(sd.a.is_sealed)
    sd.seal()
    self.assertTrue(sd.is_sealed)
    self.assertTrue(sd.a.is_sealed)

    # Test with inferred value.
    class Unresolvable(inferred.InferredValue):

      def infer(self):
        raise ValueError()

    sd = Dict(x=Unresolvable())
    sd.seal()
    self.assertTrue(sd.sym_getattr('x').is_sealed)
    sd.seal(False)
    self.assertFalse(sd.sym_getattr('x').is_sealed)


class RebindTest(unittest.TestCase):
  """Dedicated tests for `pg.Dict.rebind`."""

  def test_rebind_on_schemaless_dicts(self):
    sd = Dict(a=Dict(b=1, c=2), d=3)
    sd.rebind({
        # Update a.b.
        'a.b': 'bar',
        # Delete a.c.
        'a.c': MISSING_VALUE,
        # Insert a.x.
        'a.x': 1,
        # Insert y.
        'y': 2,
        # No-op.
        'z': MISSING_VALUE
    })
    self.assertEqual(sd, {
        'a': {
            'b': 'bar',
            'x': 1
        },
        'd': 3,
        'y': 2
    })

  def test_rebind_with_kwargs(self):
    # Rebind using only kwargs.
    sd = Dict(a=1, b=2, c=3)
    sd.rebind(a=2, c=4)
    self.assertEqual(sd, dict(a=2, b=2, c=4))

    # Rebind using both update dict and kwargs.
    sd = Dict(a=1, b=2, c=3)
    sd.rebind({'a': 2, 'b': 3}, a=3, c=3)
    self.assertEqual(sd, dict(a=3, b=3, c=3))

  def test_rebind_with_context_value(self):
    sd = Dict(a=1, b=2)
    sd.rebind(a=inferred.ValueFromParentChain())
    self.assertEqual(sd, dict(a=inferred.ValueFromParentChain(), b=2))

    sd = Dict(
        a=1,
        b=2,
        value_spec=pg_typing.Dict(
            [('a', pg_typing.Int()), ('b', pg_typing.Int())]
        ),
    )
    sd.rebind(a=inferred.ValueFromParentChain())
    self.assertEqual(sd, dict(a=inferred.ValueFromParentChain(), b=2))

  def test_rebind_with_typing(self):
    spec = pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Str(regex='foo.*')),
        ('c', pg_typing.Dict([
            ('x', pg_typing.Int(min_value=1, default=1)),
            ('y', pg_typing.Bool()),
        ]))
    ])
    sd = Dict(b='foo', c=dict(x=1, y=True), value_spec=spec)
    sd.rebind({
        'b': 'foo1',
        'c.x': 2
    })
    self.assertEqual(sd, {
        'a': 0,
        'b': 'foo1',
        'c': {'x': 2, 'y': True}
    })
    with self.assertRaisesRegex(
        ValueError, '.* does not match regular expression'):
      sd.rebind({'b': 'bar'})

    with self.assertRaisesRegex(ValueError, '.* is out of range'):
      sd.rebind({'c.x': 0})

  def test_rebind_with_reset_default(self):
    spec = pg_typing.Dict([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Str()),
        ('c', pg_typing.Dict([
            ('x', pg_typing.Int(default=1)),
            ('y', pg_typing.Bool(default=False)),
        ]))
    ])
    sd = Dict(a=1, b='foo', c=dict(x=0, y=True), value_spec=spec)

    # Reset the default value of `a` and `c`, and update `b`.
    sd.rebind({
        'a': MISSING_VALUE,
        'b': 'bar',
        'c': MISSING_VALUE,
    })
    self.assertEqual(sd, {
        'a': 0,
        'b': 'bar',
        'c': {'x': 1, 'y': False}
    })

  def test_rebind_on_sealed_dict(self):
    sd = Dict(a=1, b=2)
    with flags.as_sealed():
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot rebind key .* of sealed Dict'):
        sd.rebind(a=2)
    sd.rebind(a=2)
    self.assertEqual(sd.a, 2)

  def test_rebind_with_no_updates(self):
    def on_dict_change(field_updates):
      del field_updates
      assert False
    sd = Dict(a=1, b=2, c=3, onchange_callback=on_dict_change)
    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind'):
      sd.rebind()
    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind'):
      sd.rebind(lambda k, v, p: v)
    sd.rebind(a=1, b=2, raise_on_no_change=False)

  def test_rebind_with_skipping_notification(self):
    def on_dict_change(field_updates):
      del field_updates
      assert False
    sd = Dict(a=1, b=2, c=3, onchange_callback=on_dict_change)
    sd.rebind(a=2, skip_notification=True)
    self.assertEqual(sd, dict(a=2, b=2, c=3))

  def test_rebind_without_notifying_parents(self):
    updates = []
    def on_dict_change(x):
      def _on_change(field_updates):
        del field_updates
        updates.append(x)
      return _on_change

    c = Dict(x=1, onchange_callback=on_dict_change('c'))
    b = Dict(c=c, onchange_callback=on_dict_change('b'))
    a = Dict(b=b, onchange_callback=on_dict_change('a'))
    _ = Dict(a=a, onchange_callback=on_dict_change('y'))

    a.rebind({'b.c.x': 2}, notify_parents=False)
    self.assertEqual(updates, ['c', 'b', 'a'])

    updates[:] = []
    a.rebind({'b.c.x': 3}, notify_parents=True)
    self.assertEqual(updates, ['c', 'b', 'a', 'y'])

    updates[:] = []
    a.rebind(b=1, notify_parents=False)
    self.assertEqual(updates, ['a'])

  def test_rebind_with_field_updates_notification(self):
    updates = []
    def on_dict_change(field_updates):
      updates.append(field_updates)

    sd = Dict(
        a=1,
        b=Dict(x=1,
               y=Dict(onchange_callback=on_dict_change),
               onchange_callback=on_dict_change),
        c=List([Dict(p=1, onchange_callback=on_dict_change)],
               onchange_callback=on_dict_change),
        d='foo',
        onchange_callback=on_dict_change)
    sd.rebind({
        'a': 2,
        'b.x': 2,
        'b.y.z': 1,
        'c[0].p': MISSING_VALUE,
        'c[0].q': 2,
        'd': 'foo',  # Unchanged.
        'e': 'bar'
    })
    self.assertEqual(updates, [
        {  # Notification to `sd.c[0]`.
            'p': base.FieldUpdate(
                object_utils.KeyPath.parse('c[0].p'),
                target=sd.c[0],
                field=None,
                old_value=1,
                new_value=MISSING_VALUE),
            'q': base.FieldUpdate(
                object_utils.KeyPath.parse('c[0].q'),
                target=sd.c[0],
                field=None,
                old_value=MISSING_VALUE,
                new_value=2),
        },
        {  # Notification to `sd.c`.
            '[0].p': base.FieldUpdate(
                object_utils.KeyPath.parse('c[0].p'),
                target=sd.c[0],
                field=None,
                old_value=1,
                new_value=MISSING_VALUE),
            '[0].q': base.FieldUpdate(
                object_utils.KeyPath.parse('c[0].q'),
                target=sd.c[0],
                field=None,
                old_value=MISSING_VALUE,
                new_value=2),
        },
        {  # Notification to `sd.b.y`.
            'z': base.FieldUpdate(
                object_utils.KeyPath.parse('b.y.z'),
                target=sd.b.y,
                field=None,
                old_value=MISSING_VALUE,
                new_value=1),
        },
        {  # Notification to `sd.b`.
            'x': base.FieldUpdate(
                object_utils.KeyPath.parse('b.x'),
                target=sd.b,
                field=None,
                old_value=1,
                new_value=2),
            'y.z': base.FieldUpdate(
                object_utils.KeyPath.parse('b.y.z'),
                target=sd.b.y,
                field=None,
                old_value=MISSING_VALUE,
                new_value=1),
        },
        {  # Notification to `sd`.
            'a': base.FieldUpdate(
                object_utils.KeyPath.parse('a'),
                target=sd,
                field=None,
                old_value=1,
                new_value=2),
            'b.x': base.FieldUpdate(
                object_utils.KeyPath.parse('b.x'),
                target=sd.b,
                field=None,
                old_value=1,
                new_value=2),
            'b.y.z': base.FieldUpdate(
                object_utils.KeyPath.parse('b.y.z'),
                target=sd.b.y,
                field=None,
                old_value=MISSING_VALUE,
                new_value=1),
            'c[0].p': base.FieldUpdate(
                object_utils.KeyPath.parse('c[0].p'),
                target=sd.c[0],
                field=None,
                old_value=1,
                new_value=MISSING_VALUE),
            'c[0].q': base.FieldUpdate(
                object_utils.KeyPath.parse('c[0].q'),
                target=sd.c[0],
                field=None,
                old_value=MISSING_VALUE,
                new_value=2),
            'e': base.FieldUpdate(
                object_utils.KeyPath.parse('e'),
                target=sd,
                field=None,
                old_value=MISSING_VALUE,
                new_value='bar')
        }
    ])

  def test_rebind_with_fn(self):
    sd = Dict(a=1, b=dict(x=2, y='foo', z=[0, 1, 2]))
    def increment(k, v, p):
      del k, p
      if isinstance(v, int):
        return v + 1
      return v
    sd.rebind(increment)
    self.assertEqual(sd, Dict(a=2, b=Dict(x=3, y='foo', z=[1, 2, 3])))

  def test_notify_on_change(self):
    context = Dict(num_changes=0)
    def increment_change(unused_updates):
      context.num_changes += 1

    sd = Dict(a=1, onchange_callback=increment_change)
    sd.a = 2
    sd.b = 4
    del sd['a']
    sd.a = 4
    self.assertEqual(context.num_changes, 4)

    context.num_changes = 0
    sd = Dict(a=1, onchange_callback=increment_change)
    with flags.notify_on_change(False):
      sd.a = 2
      sd.b = 4
      del sd['a']
      sd.a = 4
    self.assertEqual(context.num_changes, 0)

  def test_bad_rebind(self):
    # Rebind is invalid on root object.
    with self.assertRaisesRegex(
        KeyError, 'Root key .* cannot be used in .*rebind.'):
      Dict().rebind({'': 1})

    # Rebind is invalid on non-symbolic object.
    with self.assertRaisesRegex(
        KeyError, 'Cannot rebind key .* is not a symbolic type.'):
      Dict(a=1).rebind({'a.x': 1})

    with self.assertRaisesRegex(
        ValueError, 'Argument \'path_value_pairs\' should be a dict.'):
      Dict().rebind(1)

    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind.'):
      Dict().rebind({})

    with self.assertRaisesRegex(
        KeyError, 'Key must be string type. Encountered 1'):
      Dict().rebind({1: 1})

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified.'):
      Dict(a=1, value_spec=pg_typing.Dict([('a', pg_typing.Int())])).rebind({
          'a': MISSING_VALUE})


class SerializationTest(unittest.TestCase):
  """Dedicated tests for `pg.Dict` serialization."""

  def test_schemaless(self):
    sd = Dict()
    sd.b = 0
    sd.c = None
    sd.a = 'foo'

    # Key order is preserved.
    self.assertEqual(sd.to_json_str(), '{"b": 0, "c": null, "a": "foo"}')

  def test_schematized(self):
    sd = Dict.partial(
        x=1,
        value_spec=pg_typing.Dict([
            ('w', pg_typing.Str()),
            ('x', pg_typing.Int()),
            ('y', pg_typing.Str().noneable()),
            # Frozen field shall not be written.
            ('z', pg_typing.Bool(True).freeze()),
        ]))
    self.assertEqual(sd.to_json_str(), '{"x": 1, "y": null}')

  def test_serialization_with_converter(self):

    class A:

      def __init__(self, value: float):
        self.value = value

      def __eq__(self, other):
        return isinstance(other, A) and other.value == self.value

    spec = pg_typing.Dict([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Object(A))
    ])
    sd = Dict(x=1, y=A(2.0), value_spec=spec)
    with self.assertRaisesRegex(
        ValueError, 'Cannot encode opaque object .* with pickle'):
      sd.to_json_str()

    pg_typing.register_converter(A, float, convert_fn=lambda x: x.value)
    pg_typing.register_converter(float, A, convert_fn=A)

    self.assertEqual(sd.to_json(), {'x': 1, 'y': 2.0})
    self.assertEqual(Dict.from_json(sd.to_json(), value_spec=spec), sd)

    self.assertEqual(sd.to_json_str(), '{"x": 1, "y": 2.0}')
    self.assertEqual(base.from_json_str(sd.to_json_str(), value_spec=spec), sd)

  def test_hide_default_values(self):

    class A(pg_object.Object):
      x: int = 1
      use_symbolic_comparison = False

    sd = Dict.partial(
        x=1,
        value_spec=pg_typing.Dict([
            ('v', pg_typing.Object(A, default=A(1))),
            ('w', pg_typing.Str()),
            ('x', pg_typing.Int()),
            ('y', pg_typing.Str().noneable()),
            # Frozen field shall not be written.
            ('z', pg_typing.Bool(True).freeze()),
        ]))
    self.assertEqual(sd.to_json_str(hide_default_values=True), '{"x": 1}')

  def test_use_inferred(self):
    # Schematized dict.
    sd = Dict(
        x=1,
        y=Dict(
            x=inferred.ValueFromParentChain(),
            value_spec=pg_typing.Dict([('x', pg_typing.Int())])
        )
    )
    self.assertEqual(
        sd.to_json_str(),
        ('{"x": 1, "y": {"x": {"_type": "'
         + inferred.ValueFromParentChain.__type_name__
         + '"}}}')
    )
    self.assertEqual(
        sd.to_json_str(use_inferred=True),
        '{"x": 1, "y": {"x": 1}}'
    )

    # Non-schematized dict.
    sd = Dict(x=1, y=Dict(x=inferred.ValueFromParentChain()))
    self.assertEqual(
        sd.to_json_str(),
        ('{"x": 1, "y": {"x": {"_type": "'
         + inferred.ValueFromParentChain.__type_name__
         + '"}}}')
    )
    self.assertEqual(
        sd.to_json_str(use_inferred=True),
        '{"x": 1, "y": {"x": 1}}'
    )

  def test_from_json(self):
    spec = pg_typing.Dict([
        ('w', pg_typing.Str()),
        ('x', pg_typing.Int()),
        ('y', pg_typing.Str().noneable()),
        # Frozen field shall not be written.
        ('z', pg_typing.Bool(True).freeze()),
        ('p', pg_typing.Int(default=inferred.ValueFromParentChain())),
    ])
    self.assertEqual(
        base.from_json_str('{"x": 1}').use_value_spec(spec, allow_partial=True),
        Dict.partial(x=1, value_spec=spec))

  def test_to_json_on_regular_dict(self):
    self.assertEqual(
        base.to_json_str({
            'x': 1,
            'y': None,
            'z': True
        }), '{"x": 1, "y": null, "z": true}')

  def test_unsupported_types(self):

    class A:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot encode opaque object .* with pickle'):
      base.to_json(Dict(x=A()))


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

    value_spec = pg_typing.Dict([
        ('a1', pg_typing.Int(1), 'Field a1.'),
        ('a2', pg_typing.Dict([
            ('b1', pg_typing.Dict([
                ('c1', pg_typing.List(pg_typing.Dict([
                    ('d1', pg_typing.Str(), 'Field d1.'),
                    ('d2', pg_typing.Bool(True), 'Field d2.\nA bool value.'),
                    ('d3', pg_typing.Object(A)),
                ])), 'Field c1.')]), 'Field b1.')]),
         'Field a2.')
    ])
    self._dict = Dict.partial({
        'a1': 1,
        'a2': {
            'b1': {
                'c1': [{
                    'd3': A.partial(x=2, z={'p': [None, True], 't': 'foo'})
                }]
            }
        }
    }, value_spec=value_spec)

  def test_compact(self):
    self.assertEqual(
        self._dict.format(compact=True),
        '{a1=1, a2={b1={c1=[0: {d1=MISSING_VALUE, d2=True, d3='
        'A(x=2, y=MISSING_VALUE, z={p=[0: None, 1: True], '
        'q=\'foo\', t=\'foo\'})}]}}}')

  def test_compact_include_keys(self):
    self.assertEqual(
        self._dict.format(compact=True, include_keys=set(['a1'])), '{a1=1}')

  def test_compact_exclude_keys(self):
    self.assertEqual(
        self._dict.format(compact=True, exclude_keys=set(['a2'])), '{a1=1}')

  def test_compact_python_format(self):
    self.assertEqual(
        self._dict.format(compact=True, python_format=True, markdown=True),
        "`{'a1': 1, 'a2': {'b1': {'c1': [{'d1': MISSING_VALUE, "
        "'d2': True, 'd3': A(x=2, y=MISSING_VALUE, z={'p': [None, True], "
        "'q': 'foo', 't': 'foo'})}]}}}`",
    )

  def test_noncompact_python_format(self):
    self.assertEqual(
        self._dict.format(
            compact=False, verbose=False, python_format=True, markdown=True
        ),
        inspect.cleandoc("""
            ```
            {
              'a1': 1,
              'a2': {
                'b1': {
                  'c1': [
                    {
                      'd1': MISSING_VALUE(Str()),
                      'd2': True,
                      'd3': A(
                        x=2,
                        y=MISSING_VALUE(Str()),
                        z={
                          'p': [
                            None,
                            True
                          ],
                          'q': 'foo',
                          't': 'foo'
                        }
                      )
                    }
                  ]
                }
              }
            }
            ```
            """),
    )

  def test_noncompact_nonverbose(self):
    self.assertEqual(
        self._dict.format(compact=False, verbose=False),
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

  def test_noncompact_verbose(self):
    self.assertEqual(
        self._dict.format(
            compact=False, verbose=True),
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
                  # Field d2.
                  # A bool value.
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

  def test_noncompact_verbose_with_extra_blankline_for_field_docstr(self):
    self.assertEqual(
        self._dict.format(
            compact=False, verbose=True, extra_blankline_for_field_docstr=True),
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

                  # Field d2.
                  # A bool value.
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

  def test_noncompact_verbose_hide_default_and_missing_values(self):
    self.assertEqual(
        self._dict.format(
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

  def test_noncompact_with_inferred_value(self):
    self.assertEqual(
        Dict(x=1, y=inferred.ValueFromParentChain()).format(compact=False),
        inspect.cleandoc("""{
            x = 1,
            y = ValueFromParentChain()
          }
        """),
    )
    self.assertEqual(
        Dict(x=1, y=inferred.ValueFromParentChain()).format(
            compact=False, use_inferred=True),
        inspect.cleandoc("""{
            x = 1,
            y = ValueFromParentChain()
          }
        """),
    )
    self.assertEqual(
        Dict(y=2, p=Dict(x=1, y=inferred.ValueFromParentChain())).format(
            compact=False, use_inferred=True),
        inspect.cleandoc("""{
            y = 2,
            p = {
              x = 1,
              y = 2
            }
          }
        """),
    )


def _on_change_callback(updates):
  del updates


class PickleTest(unittest.TestCase):

  def assert_pickle_correctness(self, d: Dict) -> Dict:
    payload = pickle.dumps(d)
    d2 = pickle.loads(payload)
    self.assertEqual(d, d2)
    self.assertEqual(d.sym_sealed, d2.sym_sealed)
    self.assertEqual(d.allow_partial, d2.allow_partial)
    # For now, deserialized `pg.Dict` does not carry value spec, which requires
    # the user to call `use_spec` on it.
    self.assertIsNone(d2.value_spec)
    self.assertEqual(d.accessor_writable, d2.accessor_writable)
    self.assertIs(d._onchange_callback, d2._onchange_callback)
    return d2

  def test_basic(self):
    self.assert_pickle_correctness(Dict(x=1, y=2, z=Dict(p='str')))

  def test_sealed(self):
    self.assert_pickle_correctness(Dict(x=1).seal())

  def test_partial(self):
    self.assert_pickle_correctness(
        Dict.partial(
            x=1,
            value_spec=pg_typing.Dict([('x', int)])))

  def test_accessor_writable(self):
    d = self.assert_pickle_correctness(Dict(x=1).set_accessor_writable(False))
    with self.assertRaises(base.WritePermissionError):
      d.y = 1

  def test_with_value_spec(self):
    self.assert_pickle_correctness(
        Dict(
            x=1, y=2, z=Dict(p='str'),
            value_spec=pg_typing.Dict([
                ('x', int),
                ('y', int),
                ('z', pg_typing.Dict([('p', str)])),
            ])))

  def test_with_onchange_callback(self):
    self.assert_pickle_correctness(
        Dict(x=1, y=2, onchange_callback=_on_change_callback))


if __name__ == '__main__':
  unittest.main()
