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
"""Tests for pyglove.object_utils."""

import inspect
import typing
import unittest
from pyglove.core import object_utils


class JSONConvertibleTest(unittest.TestCase):
  """Tests for JSONConvertible type registry."""

  def testRegistry(self):
    """Test bad registration cases."""

    class A(object_utils.JSONConvertible):

      def __init__(self, x):
        self.x = x

      @classmethod
      def from_json(cls, json_dict):
        return A(x=json_dict.pop('x'))

      def to_json(self):
        return {
            '_type': 'A',
            'x': self.x
        }

    object_utils.JSONConvertible.register('A', A)
    self.assertTrue(object_utils.JSONConvertible.is_registered('A'))
    self.assertIs(object_utils.JSONConvertible.class_from_typename('A'), A)
    self.assertIn(
        ('A', A),
        list(object_utils.JSONConvertible.registered_types()))

    class B(A):
      pass

    with self.assertRaisesRegex(
        NotImplementedError, 'Subclass should override this method'):
      _ = object_utils.JSONConvertible.from_json(1)

    with self.assertRaisesRegex(
        KeyError, 'Type .* has already been registered with class .*'):
      object_utils.JSONConvertible.register('A', B)

    object_utils.JSONConvertible.register('A', B, override_existing=True)
    self.assertIn(
        ('A', B),
        list(object_utils.JSONConvertible.registered_types()))


class MissingValueTest(unittest.TestCase):
  """Tests for class MissingValue."""

  def testBasics(self):
    """Test basic functionalities."""
    self.assertEqual(object_utils.MissingValue(), object_utils.MissingValue())
    self.assertNotEqual(object_utils.MissingValue(), 1)
    self.assertNotEqual(object_utils.MissingValue(), {})

    self.assertEqual(str(object_utils.MissingValue()), 'MISSING_VALUE')
    self.assertEqual(repr(object_utils.MissingValue()), 'MISSING_VALUE')


class KeyPathTest(unittest.TestCase):
  """Tests for class KeyPath."""

  def testBasics(self):
    """Test basic functionalities of KeyPath."""

    # Root element.
    r = object_utils.KeyPath()
    self.assertTrue(r.is_root)
    self.assertFalse(r)
    self.assertEqual(r, '')
    self.assertEqual(r.depth, 0)
    self.assertEqual(len(r), 0)
    self.assertEqual(r.path, '')
    with self.assertRaisesRegex(
        KeyError, 'Parent of a root KeyPath does not exist.'):
      _ = r.parent

    with self.assertRaisesRegex(
        KeyError, 'Key of root KeyPath does not exist.'):
      _ = r.key

    # 1-level deep.
    a = object_utils.KeyPath('a')
    self.assertFalse(a.is_root)
    self.assertEqual(a.key, 'a')
    self.assertEqual(a.path, 'a')
    self.assertEqual(a, 'a')  # Relative path compare.
    self.assertNotEqual(a, '')
    self.assertEqual(a.depth, 1)
    self.assertEqual(len(a), 1)
    self.assertEqual(a.parent, r)

    a2 = object_utils.KeyPath(0)
    self.assertFalse(a2.is_root)
    self.assertEqual(a2.key, 0)
    self.assertEqual(str(a2), '[0]')
    self.assertEqual(a2.path, '[0]')
    self.assertEqual(a2, '[0]')
    self.assertNotEqual(a2, '')
    self.assertEqual(a2.depth, 1)
    self.assertEqual(len(a2), 1)
    self.assertEqual(a2.parent, r)

    a3 = object_utils.KeyPath('x.y')
    self.assertFalse(a3.is_root)
    self.assertEqual(a3.key, 'x.y')
    self.assertEqual(a3.path, '[x.y]')
    self.assertEqual(a3.path_str(False), 'x.y')
    self.assertEqual(a3, '[x.y]')  # Relative path compare.
    self.assertNotEqual(a3, 'x.y')
    self.assertEqual(a3.depth, 1)
    self.assertEqual(len(a3), 1)
    self.assertEqual(a3.parent, r)

    # Multiple levels.
    b = object_utils.KeyPath([1, 'b'])
    self.assertEqual(b, '[1].b')
    self.assertEqual(b.path, '[1].b')
    self.assertNotEqual(a, b)
    self.assertEqual(len(b), 2)
    self.assertEqual(b.parent, '[1]')

    c = object_utils.KeyPath('c', b)
    self.assertEqual(c.key, 'c')
    self.assertEqual(c, '[1].b.c')
    self.assertEqual(c.keys, [1, 'b', 'c'])
    self.assertEqual(len(c), 3)
    self.assertEqual(c.parent, b)

    d = object_utils.KeyPath(['d', 0], c)
    self.assertEqual(d.key, 0)
    self.assertEqual(d, '[1].b.c.d[0]')
    self.assertEqual(d.keys, [1, 'b', 'c', 'd', 0])
    self.assertEqual(d.parent, '[1].b.c.d')
    self.assertEqual(len(d), 5)

    d2 = object_utils.KeyPath(('d', 0), c)
    self.assertEqual(d, d2)

  def testComplexKeyType(self):

    class A:

      def __init__(self, text):
        self._text = text

      def __str__(self):
        return f'A({self._text})'

    p = object_utils.KeyPath([A('a'), A('b'), 'c'])
    self.assertEqual(p.path, '[A(a)][A(b)].c')

    # Key may have '.' in their string form.
    p = object_utils.KeyPath([A('a.*'), A('$b')])
    self.assertEqual(p.path, '[A(a.*)][A($b)]')

    # NOTE: We cannot really parse KeyPath with complex types.

    class B(object_utils.StrKey):
      """Class that implements StrKey will be treated as string key."""

      def __init__(self, text):
        self._text = text

      def __str__(self):
        return f'B({self._text})'

    p = object_utils.KeyPath([B('a'), B('b'), 'c'])
    self.assertEqual(p.path, 'B(a).B(b).c')

  def testParse(self):
    """Test KeyPath.parse method."""
    self.assertEqual(object_utils.KeyPath.parse('a').keys, ['a'])
    self.assertEqual(len(object_utils.KeyPath.parse('')), 0)

    self.assertEqual(object_utils.KeyPath.parse('a').keys, ['a'])
    self.assertEqual(object_utils.KeyPath.parse('[a ]').keys, ['a '])
    self.assertEqual(object_utils.KeyPath.parse('[0].a').keys, [0, 'a'])
    self.assertEqual(object_utils.KeyPath.parse('[0][1].a').keys, [0, 1, 'a'])
    self.assertEqual(
        object_utils.KeyPath.parse('a.b[1].c').keys, ['a', 'b', 1, 'c'])
    self.assertEqual(
        object_utils.KeyPath.parse('a[x[0]].b[y.z].c').keys,
        ['a', 'x[0]', 'b', 'y.z', 'c'])

    with self.assertRaisesRegex(
        ValueError, '\'path_str\' must be a string type.'):
      object_utils.KeyPath.parse(0)

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched open bracket at position 0'):
      object_utils.KeyPath.parse('[0')

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched open bracket at position 0'):
      object_utils.KeyPath.parse('[[0]')

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched close bracket at position 3'):
      object_utils.KeyPath.parse('[0]]')

  def testFromValue(self):
    """Test KeyPath.from_value."""
    self.assertEqual(
        object_utils.KeyPath.from_value('x.y'),
        object_utils.KeyPath(['x', 'y']))

    self.assertEqual(
        object_utils.KeyPath.from_value(1),
        object_utils.KeyPath([1]))

    path = object_utils.KeyPath(['x'])
    self.assertIs(
        object_utils.KeyPath.from_value(path),
        path)

    with self.assertRaisesRegex(
        ValueError, '.* is not a valid KeyPath equivalence'):
      object_utils.KeyPath.from_value(0.1)

  def testArithmetics(self):
    """Test KeyPath arithmetics."""

    # Test operator +.
    self.assertEqual(object_utils.KeyPath('a') + 'b.c', 'a.b.c')
    self.assertEqual(object_utils.KeyPath('a') + '[0].b', 'a[0].b')
    self.assertEqual(object_utils.KeyPath('a') + None, 'a')
    self.assertEqual(object_utils.KeyPath('a') + 1, 'a[1]')
    self.assertEqual(
        object_utils.KeyPath('a') + object_utils.KeyPath('b'), 'a.b')
    self.assertEqual(object_utils.KeyPath.parse('a.b') + 1.0, 'a.b[1.0]')

    # Test operator -.
    self.assertEqual(object_utils.KeyPath('a') - object_utils.KeyPath('a'), '')
    self.assertEqual(object_utils.KeyPath('a') - 'a', '')
    self.assertEqual(object_utils.KeyPath('a') - '', 'a')
    self.assertEqual(object_utils.KeyPath('a') - None, 'a')
    self.assertEqual(object_utils.KeyPath('a') - object_utils.KeyPath(), 'a')
    self.assertEqual(object_utils.KeyPath.parse('a.b.c.d') - 'a.b', 'c.d')
    self.assertEqual(object_utils.KeyPath.parse('[0].a') - 0, 'a')

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = object_utils.KeyPath('a') - 'b'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = object_utils.KeyPath.parse('a.b') - 'a.c'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = object_utils.KeyPath.parse('a[0]') - 'a[1]'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* is an ancestor'):
      _ = object_utils.KeyPath.parse('a.b') - 'a.b.c'

    with self.assertRaisesRegex(TypeError, 'Cannot subtract KeyPath'):
      _ = object_utils.KeyPath.parse('a.b') - 1.0

  def testHash(self):
    """Test KeyPath.__hash__ method."""
    self.assertIn(object_utils.KeyPath.parse('a.b.c'), {'a.b.c': 1})
    self.assertNotIn(object_utils.KeyPath.parse('a.b.c'), {'a.b': 1})

  def testCompare(self):
    """Test KeyPath.__eq__, __ne__, __lt__, __le__, __gt__, and __ge__."""
    keypath = object_utils.KeyPath.parse
    # Equality should only hold true for KeyPaths that are identical.
    self.assertEqual(object_utils.KeyPath(), object_utils.KeyPath.parse(''))
    self.assertEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].b[3][4]'))
    self.assertNotEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].a[3][4]'))
    self.assertNotEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].b[4][4]'))
    # Earlier keys in the path should be prioritized over later ones.
    self.assertLess(object_utils.KeyPath(), keypath('a'))
    self.assertLess(keypath('a'), keypath('a.a'))
    self.assertLess(keypath('a.a'), keypath('a.b'))
    self.assertGreater(keypath('a'), object_utils.KeyPath())
    self.assertGreater(keypath('a[1].b'), keypath('a[1].a'))
    self.assertGreater(keypath('a[1].a'), keypath('a[1]'))
    # Numbers should be compared numerically - not lexicographically.
    self.assertLessEqual(keypath('a[1]'), keypath('a[2]'))
    self.assertLessEqual(keypath('a[2]'), keypath('a[10]'))
    self.assertGreaterEqual(keypath('a[10]'), keypath('a[2]'))
    self.assertGreaterEqual(keypath('a[2]'), keypath('a[1]'))
    # It should be possible to compare numeric keys with string keys.
    self.assertLess(keypath('a[1]'), keypath('a.b'))
    self.assertGreater(keypath('a.b'), keypath('a[1]'))
    self.assertLessEqual(keypath('a[1]'), keypath('a.b'))
    self.assertGreaterEqual(keypath('a.b'), keypath('a[1]'))

  def testQuery(self):
    """Test KeyPath.query method family."""

    def query_shall_succeed(path_str, obj, expected_value):
      self.assertEqual(
          object_utils.KeyPath.parse(path_str).query(obj), expected_value)

    def query_shall_fail(path_str,
                         obj,
                         error='Cannot query sub-key .* of object .*'):
      with self.assertRaisesRegex(KeyError, error):
        object_utils.KeyPath.parse(path_str).query(obj)

    def get_shall_succeed(path_str, obj, default, expected_value):
      self.assertEqual(
          object_utils.KeyPath.parse(path_str).get(obj, default),
          expected_value)

    def assert_exists(path_str, obj, should_exists):
      self.assertEqual(
          object_utils.KeyPath.parse(path_str).exists(obj), should_exists)

    # Query at root level.
    query_shall_succeed('', 1, 1)
    query_shall_succeed('', None, None)
    query_shall_succeed('', [1, 2], [1, 2])
    query_shall_succeed('', {'a': 'foo'}, {'a': 'foo'})

    # Query simple types with more than 1 depth.
    query_shall_fail('a', 1)
    query_shall_fail('[0]', None)

    # Query complex types.
    class Foo:
      """Custom object."""

      def __init__(self, values):
        self._values = values

      def __getitem__(self, key):
        return self._values[key]

      # NOTE(daiyip): __len__ and __contains__ is intentional omitted.

    class Bar:
      """Custom object with sym_getattr."""

      def __init__(self, **kwargs):
        self._map = kwargs

      def sym_hasattr(self, name):
        return name in self._map

      def sym_getattr(self, name):
        return self._map[name]

    src = {'a': [{'c': 'foo'},
                 {'d': [1, 2]}],
           'b': True,
           'e': Foo([1, 2, 3]),
           'f': Bar(x=0, y=1)}

    query_shall_succeed('', src, src)
    query_shall_succeed('a', src, src['a'])
    query_shall_succeed('a[0]', src, src['a'][0])
    query_shall_succeed('a[0].c', src, src['a'][0]['c'])
    query_shall_succeed('a[1].d[1]', src, src['a'][1]['d'][1])
    query_shall_succeed('b', src, src['b'])
    query_shall_succeed('f.x', src, 0)

    query_shall_fail('c', src, 'Path .* does not exist: key .* is absent')
    query_shall_fail('a.c', src, 'Path .* does not exist: key .* is absent')
    query_shall_fail('a[2]', src, 'Path .* does not exist: key .* is absent')
    query_shall_fail('a[1].e', src, 'Path .* does not exist: key .* is absent')
    query_shall_fail(
        'e[0]', src, 'Cannot query index .* on object .*: '
        '\'__len__\' does not exist')
    query_shall_fail(
        'e.f', src, 'Cannot query key .* on object .*: '
        '\'__contains__\' does not exist')
    query_shall_fail(
        'f.z', src, 'Path .* does not exist: key .* is absent')
    # Test get method.
    get_shall_succeed('', src, None, src)
    get_shall_succeed('a[1].d[1]', src, None, src['a'][1]['d'][1])
    get_shall_succeed('c', src, None, None)
    get_shall_succeed('b.c', src, 1, 1)

    # Test exists method.
    assert_exists('', src, True)
    assert_exists('a[1].d[1]', src, True)
    assert_exists('c', src, False)
    assert_exists('b.c', src, False)


class TraverseTest(unittest.TestCase):
  """Tests for traverse method."""

  def testSimpleTypes(self):
    """Test traverse on simple types."""
    visited = []

    def visit_fn(p, v):
      visited[:] = [p, v]
      return True

    self.assertTrue(object_utils.traverse(None, visit_fn))
    self.assertEqual(visited, [object_utils.KeyPath(), None])

    self.assertTrue(object_utils.traverse(1, visit_fn))
    self.assertEqual(visited, [object_utils.KeyPath(), 1])

    self.assertTrue(object_utils.traverse('abc', visit_fn))
    self.assertEqual(visited, [object_utils.KeyPath(), 'abc'])

    self.assertTrue(object_utils.traverse(True, visit_fn))
    self.assertEqual(visited, [object_utils.KeyPath(), True])

    self.assertTrue(object_utils.traverse((1, 2), visit_fn))
    self.assertEqual(visited, [object_utils.KeyPath(), (1, 2)])

  def testComplexTypes(self):
    """Test traverse complex types."""
    previsited = []
    sequence_id = [0]

    def previsit_fn(p, v):
      previsited.append((sequence_id[0], p, v))
      sequence_id[0] += 1
      return True

    postvisited = []

    def postvisit_fn(p, v):
      postvisited.append((sequence_id[0], p, v))
      sequence_id[0] += 1
      return True

    # Pre-order
    tree = {'a': [{'c': [1, 2]}, {'d': {'g': (3, 4)}}], 'b': 'foo'}
    self.assertTrue(object_utils.traverse(tree, previsit_fn, postvisit_fn))
    self.assertEqual(previsited, [
        (0, '', tree),
        (1, 'a', tree['a']),
        (2, 'a[0]', tree['a'][0]),
        (3, 'a[0].c', tree['a'][0]['c']),
        (4, 'a[0].c[0]', tree['a'][0]['c'][0]),
        (6, 'a[0].c[1]', tree['a'][0]['c'][1]),
        (10, 'a[1]', tree['a'][1]),
        (11, 'a[1].d', tree['a'][1]['d']),
        (12, 'a[1].d.g', tree['a'][1]['d']['g']),
        (17, 'b', tree['b']),
    ])

    self.assertEqual(postvisited, [
        (5, 'a[0].c[0]', tree['a'][0]['c'][0]),
        (7, 'a[0].c[1]', tree['a'][0]['c'][1]),
        (8, 'a[0].c', tree['a'][0]['c']),
        (9, 'a[0]', tree['a'][0]),
        (13, 'a[1].d.g', tree['a'][1]['d']['g']),
        (14, 'a[1].d', tree['a'][1]['d']),
        (15, 'a[1]', tree['a'][1]),
        (16, 'a', tree['a']),
        (18, 'b', tree['b']),
        (19, '', tree),
    ])

  def testShortcircuit(self):
    """Test shorcircut at a given node."""
    previsited = []
    sequence_id = [0]

    def previsit_fn(p, v):
      previsited.append((sequence_id[0], p, v))
      sequence_id[0] += 1
      return True

    postvisited = []

    def postvisit_fn(p, v):
      postvisited.append((sequence_id[0], p, v))
      sequence_id[0] += 1
      return p != 'a[0]'

    # Pre-order
    tree = {'a': [{'c': [1, 2]}, {'d': {'g': (3, 4)}}], 'b': 'foo'}
    self.assertFalse(object_utils.traverse(tree, previsit_fn, postvisit_fn))
    self.assertEqual(previsited, [
        (0, '', tree),
        (1, 'a', tree['a']),
        (2, 'a[0]', tree['a'][0]),
        (3, 'a[0].c', tree['a'][0]['c']),
        (4, 'a[0].c[0]', tree['a'][0]['c'][0]),
        (6, 'a[0].c[1]', tree['a'][0]['c'][1]),
    ])

    self.assertEqual(postvisited, [
        (5, 'a[0].c[0]', tree['a'][0]['c'][0]),
        (7, 'a[0].c[1]', tree['a'][0]['c'][1]),
        (8, 'a[0].c', tree['a'][0]['c']),
        (9, 'a[0]', tree['a'][0]),
    ])


class ListifyDictWithIntKeysTest(unittest.TestCase):
  """Tests for try_listify_dict_with_int_keys."""

  def testEmptyDict(self):
    """Test listify empty dict."""
    self.assertEqual(
        object_utils.try_listify_dict_with_int_keys({}), ({}, False))

  def testNonIntKeys(self):
    """Test listify non-int keys dict."""
    # Str type key.
    self.assertEqual(
        object_utils.try_listify_dict_with_int_keys({'a': 1}), ({
            'a': 1
        }, False))

    # Not all keys are int type.
    self.assertEqual(
        object_utils.try_listify_dict_with_int_keys({
            0: 2,
            'a': 1
        }), ({
            0: 2,
            'a': 1
        }, False))

  def testSparseIndices(self):
    """Test listify int key dict with sparse indices."""
    self.assertEqual(
        object_utils.try_listify_dict_with_int_keys({
            0: 1,
            2: 2
        }), ({
            0: 1,
            2: 2
        }, False))

    self.assertEqual(
        object_utils.try_listify_dict_with_int_keys({
            0: 1,
            2: 2
        },
                                                    convert_when_sparse=True),
        ([1, 2], True))

  def testDenseIndices(self):
    """Test listify int key dict with dense indices."""
    self.assertEqual(
        object_utils.try_listify_dict_with_int_keys({
            0: 1,
            1: 2,
            2: 3
        }), ([1, 2, 3], True))


class TransformTest(unittest.TestCase):
  """Tests for transform method."""

  def testSimpleTypes(self):
    """Test transform on simple types."""
    self.assertEqual(object_utils.transform(1, lambda k, v: v + 1), 2)
    self.assertIsNone(object_utils.transform(None, lambda k, v: None))
    self.assertEqual(object_utils.transform(True, lambda k, v: not v), False)
    self.assertEqual(object_utils.transform('foo', lambda k, v: len(v)), 3)

  def testComplexTypes(self):
    """Test transform on simple types."""

    def _remove_int(path, value):
      del path
      if isinstance(value, int):
        return object_utils.MISSING_VALUE
      return value

    self.assertEqual(
        object_utils.transform(
            {
                'a': {
                    'b': 1,
                    'c': [1, 'bar', 2, 3],
                    'd': 'foo'
                },
                'e': 'bar',
                'f': 4
            }, _remove_int), {
                'a': {
                    'c': ['bar'],
                    'd': 'foo',
                },
                'e': 'bar'
            })

    def _listify_dict_equivalent(path, value):
      del path
      if isinstance(value, dict):
        value, _ = object_utils.try_listify_dict_with_int_keys(value, True)
      return value

    self.assertEqual(
        object_utils.transform({
            0: {
                'b': {
                    0: 1,
                    3: 2
                },
                'd': 'foo'
            },
            1: {}
        }, _listify_dict_equivalent), [{
            'b': [1, 2],
            'd': 'foo'
        }, {}])

  def testMaintainKeyOrder(self):
    """Test transform keep key order for dict."""
    self.assertEqual(
        list(
            object_utils.flatten(
                object_utils.transform(
                    {
                        'b': 1,
                        'a': 2,
                        'c': {
                            'f': 'foo',
                            'e': 'bar'
                        }
                    }, lambda p, v: v)).keys()),
        ['b', 'a', 'c.f', 'c.e'])

  def testInPlaceFlag(self):
    """Test inplace flag."""
    d1 = {
        'a': 1,
        'b': {
            'c': {}
        }
    }
    d2 = object_utils.transform(d1, lambda p, v: v, inplace=True)
    self.assertIs(d1, d2)
    self.assertIs(d1['b'], d2['b'])
    self.assertIs(d1['b']['c'], d2['b']['c'])

    d3 = object_utils.transform(d1, lambda p, v: v, inplace=False)
    self.assertEqual(d1, d3)
    self.assertIsNot(d1, d3)
    self.assertIsNot(d1['b']['c'], d3['b']['c'])


class FlattenTest(unittest.TestCase):
  """Tests for class flatten method."""

  def testSimpleTypes(self):
    """Test flattening simple types."""
    self.assertIsNone(object_utils.flatten(None))
    self.assertEqual(object_utils.flatten(1), 1)
    self.assertEqual(object_utils.flatten(1.0), 1.0)
    self.assertEqual(object_utils.flatten(True), True)
    self.assertEqual(object_utils.flatten('abc'), 'abc')

  def testComplexTypes(self):
    """Test flattening complex types."""
    # Test lists.
    self.assertEqual(object_utils.flatten([]), [])
    self.assertEqual(object_utils.flatten([1, 2]), {
        '[0]': 1,
        '[1]': 2,
    })
    self.assertEqual(object_utils.flatten([[1], 2]), {
        '[0][0]': 1,
        '[1]': 2,
    })

    # Test dicts.
    self.assertEqual(object_utils.flatten({}), {})
    self.assertEqual(
        object_utils.flatten({
            'a': 1,
            'b': None,
            'c': {
                'd': True
            }
        }), {
            'a': 1,
            'b': None,
            'c.d': True
        })

    # Test list/dict hybrid.
    self.assertEqual(
        object_utils.flatten({
            'a': {
                'e': 1,
                'f': [{
                    'g': 2
                }, {
                    'g[0]': 3
                }],
                'h': [],
                'i.j': {},
            },
            'b': 'hi',
            'c': None
        }), {
            'a.e': 1,
            'a.f[0].g': 2,
            'a.f[1].g[0]': 3,
            'a.h': [],
            'a.i.j': {},
            'b': 'hi',
            'c': None
        })

    self.assertEqual(
        object_utils.flatten({
            'a': {
                'e': 1,
                'f': [{
                    'g': 2
                }, {
                    'g[0]': 3
                }],
                'h': [],
                'i.j': {},
            },
            'b': 'hi',
            'c': None
        }, False), {
            'a.e': 1,
            'a.f[0].g': 2,
            'a.f[1][g[0]]': 3,
            'a.h': [],
            'a[i.j]': {},
            'b': 'hi',
            'c': None
        })

    # Test tuples:
    self.assertEqual(object_utils.flatten((1, True)), (1, True))
    self.assertEqual(
        object_utils.flatten(([1, 2, 3], 'foo')), ({
            '[0]': 1,
            '[1]': 2,
            '[2]': 3,
        }, 'foo'))


class CanonicalizeTest(unittest.TestCase):
  """Test canonicalize method."""

  def testSimpleTypes(self):
    """Test canonicalize on simple types."""
    self.assertIsNone(object_utils.canonicalize(None))
    self.assertEqual(object_utils.canonicalize('foo'), 'foo')
    self.assertEqual(object_utils.canonicalize([1, 2, [4, 5]]), [1, 2, [4, 5]])

  def testSparseList(self):
    """Test canonicalize sparse list."""

    # List is root element.
    self.assertEqual(
        object_utils.canonicalize({
            '[0]': 'a',
            '[1]': 'b',
            '[3]': 'c',
        }, sparse_list_as_dict=True),
        {
            0: 'a',
            1: 'b',
            3: 'c',
        })

    self.assertEqual(
        object_utils.canonicalize({
            '[0]': 'a',
            '[1]': 'b',
            '[3]': 'c',
        }, sparse_list_as_dict=False),
        ['a', 'b', 'c'])

    # List is intermediate element.
    self.assertEqual(
        object_utils.canonicalize({
            'a[0]': 'a',
            'a[1]': 'b',
            'a[3]': 'c',
        },
                                  sparse_list_as_dict=True),
        {'a': {
            0: 'a',
            1: 'b',
            3: 'c'
        }})

    self.assertEqual(
        object_utils.canonicalize({
            'a[0]': 'a',
            'a[1]': 'b',
            'a[3]': 'c',
        },
                                  sparse_list_as_dict=False),
        {'a': ['a', 'b', 'c']})

  def testComplexTypes(self):
    """Test canonicalize on complex types."""
    # Nested structures of dict.
    self.assertEqual(
        object_utils.canonicalize({
            'a': {
                'b': [1, {
                    'c.d': True
                }],
                'e.g': 1,
                '[e.h].i': 2
            },
            'a.e.f': 'hi',
            'a[e.h].j': 3
        }), {'a': {
            'b': [1, {
                'c': {
                    'd': True
                },
            }],
            'e': {
                'f': 'hi',
                'g': 1
            },
            'e.h': {
                'i': 2,
                'j': 3
            }
        }})

    # Canonicalize a canonical dict should return identity.
    self.assertEqual(
        object_utils.canonicalize(
            {'a': {
                'b': [1, {
                    'c': {
                        'd': True
                    }
                }],
                'e': {
                    'f': 'hi',
                    'g': 1
                }
            }}),
        {'a': {
            'b': [1, {
                'c': {
                    'd': True
                }
            }],
            'e': {
                'f': 'hi',
                'g': 1
            }
        }})

    # Test that overlaped keys are merged correctly
    self.assertDictEqual(
        object_utils.canonicalize({
            'a.c': False,
            'a': {'b': 1},
            'a.d': False,
        }), {
            'a': {
                'b': 1,
                'c': False,
                'd': False
            }
        })

  def testIncompatibleValues(self):
    """Test canonicalize with incompatible keys."""
    # Should raise error if a key yield incompatible values.
    with self.assertRaisesRegex(
        KeyError, 'Path \'.*\' is assigned with conflicting values.'):
      object_utils.canonicalize({'a.b': 1, 'a.b.c': True})

    with self.assertRaisesRegex(KeyError, 'Key must not be empty.'):
      object_utils.canonicalize({'': 1})

  def testDictKeyOrder(self):
    """Test dict key order is prereserved."""
    self.assertEqual(
        list(object_utils.canonicalize({
            'b': 0,
            'a1': 1,
            'a': 2,
        }).keys()), ['b', 'a1', 'a'])

    self.assertEqual(
        list(
            object_utils.canonicalize({
                'b': 0,
                'a1': 1,
                'a': 2
            }).keys()), ['b', 'a1', 'a'])


class MergeTest(unittest.TestCase):
  """Test merge_dict method."""

  def testInvalidInput(self):
    """Test invalid input."""
    with self.assertRaisesRegex(TypeError, 'value_list should be a list'):
      object_utils.merge('abc')

  def testEmptyDicts(self):
    """Test merge of empty dicts."""
    # Merge variations of empty dicts.
    self.assertIsNone(object_utils.merge([]))
    self.assertEqual(object_utils.merge([None, {}]), {})

    # Test merge non-empty with empties.
    a = {'a': {'b': 1}}
    self.assertEqual(object_utils.merge([None, a]), a)
    self.assertEqual(object_utils.merge([a, None]), a)

  def testSparseListUpdate(self):
    """Test sparse list update."""

  def testSideEffectFree(self):
    """Test merge keep input dicts in-tact."""
    a = {'a': 1, 'b': 2}
    b = {'c': 3}
    a2 = dict(a)
    b2 = dict(b)
    self.assertEqual(object_utils.merge([a, b]), {'a': 1, 'b': 2, 'c': 3})
    self.assertEqual(a, a2)
    self.assertEqual(b, b2)

  def testCanonicalDicts(self):
    """Test merge_dict on already canonical dicts."""
    # Test merge with standard (canonical) dict merges:
    self.assertEqual(
        object_utils.merge([
            # original.
            {
                'a': 1,
                'b': 2,
                'c': {
                    'd': 'foo',
                    'e': 'bar'
                }
            },
            # patch.
            {
                'b': 3,
                'd': [1, 2, 3],
                'c': {
                    'e': 'bar2',
                    'f': 10
                }
            }
        ]),
        {
            'a': 1,
            # b is updated.
            'b': 3,
            'c': {
                'd': 'foo',
                # e is updated.
                'e': 'bar2',
                # f is added.
                'f': 10
            },
            # d is inserted.
            'd': [1, 2, 3]
        })

  def testNoncanonicalDicts(self):
    """Test merge_dict on non-canonical dicts."""
    # Test merge with noncanonical dict merges.
    self.assertEqual(
        object_utils.merge([
            # original.
            {
                'a': 1,
                'b': 2,
                'c': {
                    'd': 'foo',
                },
                'c.e': 'bar'
            },
            # patch.
            {
                'b': 3,
                'd': [1, 2, 3],
                'c': {
                    'e': 'bar2',
                },
                'c.f': 10
            }
        ]),
        {
            'a': 1,
            # b is updated.
            'b': 3,
            'c': {
                'd': 'foo',
                # e is updated.
                'e': 'bar2',
                # f is added.
                'f': 10
            },
            # d is inserted.
            'd': [1, 2, 3]
        })

  def testMergeList(self):
    """Test merge semantics with list."""

    # Merge list at root level.
    self.assertEqual(
        object_utils.merge([[1, 2, 3], {
            '[0]': -1,
            '[3]': 4
        }]), [-1, 2, 3, 4])

    # Replace entire list.
    self.assertEqual(
        object_utils.merge([{
            'a': [0, 1, 2],
        }, {
            'a': [3]
        }]), {'a': [3]})

    # Update single element and append.
    self.assertEqual(
        object_utils.merge([{
            'a': [0, 1, 2],
        }, {
            'a[0]': -1,
            'a[10]': 3,
        }]), {'a': [-1, 1, 2, 3]})

    with self.assertRaisesRegex(
        KeyError, 'Dict must use integers as keys when merging to a list.'):
      object_utils.merge([[0, 1, 2], {'a': 1}])

  def testDeleteSemantics(self):
    """Test delete semantics."""

    def _remove_fixed_key(path, old_value, new_value):
      if path.key == 'c' or path.key == 'b':
        return object_utils.MISSING_VALUE
      return new_value if object_utils.MISSING_VALUE != new_value else old_value

    self.assertEqual(
        object_utils.merge([{
            'a': {
                'c': True,
                'd': False,
            },
            'b': 2,
            'c': 3,
            'd': 4
        }, {
            'a': {
                'c': 1,
                'e': 'foo'
            },
            'b': 1,
        }],
                           merge_fn=_remove_fixed_key),
        {
            'a': {
                # NOTE(daiyip): 'a.c' is removed by _remove_fixed_keys.
                'd': False,
                'e': 'foo'
            },
            # 'b' and 'c' are removed by _remove_fixed_keys.
            'd': 4
        })

  def testCustomizedMergeFn(self):
    """Test merge with custom merge_fn."""
    added_keys = []
    updated_keys = []

    def _merge_fn(path, old_value, new_value):
      if new_value is object_utils.MISSING_VALUE:
        return old_value

      if old_value is object_utils.MISSING_VALUE:
        added_keys.append(path.path)
      else:
        updated_keys.append(path.path)
        if isinstance(new_value, int) and isinstance(old_value, int):
          return new_value + old_value
      return new_value

    self.assertEqual(
        object_utils.merge([{
            'a': 1,
            'b': {
                'e': 2,
                'f': 3,
            }
        }, {
            'b.f': 4,
            'b.g': 5,
            'c': 0
        }],
                           merge_fn=_merge_fn),
        {
            # a is untouched.
            'a': 1,
            'b': {
                # e is untoched.
                'e': 2,
                # f is assigned with new value + old_value
                'f': 7,
                # g is added.
                'g': 5
            },
            # c is added.
            'c': 0
        })

    self.assertEqual(added_keys, ['b.g', 'c'])
    self.assertEqual(updated_keys, ['b.f'])

  def testMergeKeyOrder(self):
    """Test key order is expected after merge."""

    # For ordered dict, order of keys are preserved.
    # Newly added keys are appended.
    self.assertEqual(
        list(
            object_utils.merge([
                {'b': 1, 'a': 2, 'c': 1},
                {'a': 'foo', 'd': 'bar'}
            ]).keys()),
        ['b', 'a', 'c', 'd'])

    # For dict, the order of keys is preserved.
    # Newly added keys are appended.
    self.assertEqual(
        list(
            object_utils.merge([
                {'b': 1, 'a': 2, 'c': 1},
                {'a': 'foo', 'f': 'something', 'd': 'bar'}
            ]).keys()),
        ['b', 'a', 'c', 'f', 'd'])


class PartialTest(unittest.TestCase):
  """Test object_utils.is_partial."""

  def testSimpleTypes(self):
    """Test simple types."""
    self.assertFalse(object_utils.is_partial(1))
    self.assertFalse(object_utils.is_partial(True))
    self.assertFalse(object_utils.is_partial(1.0))
    self.assertFalse(object_utils.is_partial('abc'))
    self.assertFalse(object_utils.is_partial(None))
    self.assertTrue(object_utils.is_partial(object_utils.MISSING_VALUE))

  def testComplexTypes(self):
    """Test complex types."""

    class A:
      pass

    class B(object_utils.MaybePartial):

      def missing_values(self):
        return {'SOME_KEY': 1}

    self.assertFalse(object_utils.is_partial([1, 2]))
    self.assertFalse(object_utils.is_partial({'foo': 'bar'}))
    self.assertFalse(object_utils.is_partial([1, {'foo': A()}]))
    self.assertTrue(
        object_utils.is_partial({'foo': object_utils.MISSING_VALUE}))
    self.assertTrue(object_utils.is_partial([object_utils.MISSING_VALUE]))
    self.assertTrue(object_utils.is_partial([{'a': 1, 'b': [B()]}]))


class StringHelperTest(unittest.TestCase):
  """Tests for string helper methods in object_utils."""

  def testKVListStr(self):
    """Test object_utils.kvlist_str."""
    self.assertEqual(
        object_utils.kvlist_str([
            ('', 'foo', None),
            ('a', 1, None),
            ('b', 'str', (None, 'str')),
            ('c', True, False),
        ]), 'foo, a=1, c=True')

  def testQuoteIfStr(self):
    """Test object_utils.quote_if_str."""
    self.assertEqual(object_utils.quote_if_str(1), 1)
    self.assertEqual(object_utils.quote_if_str('foo'), '\'foo\'')

  def testMessageOnPath(self):
    """Test object_utils.message_on_path."""
    self.assertEqual(object_utils.message_on_path('hi.', None), 'hi.')
    self.assertEqual(
        object_utils.message_on_path('hi.', object_utils.KeyPath()),
        'hi. (path=)')

  def testCommaDelimitedStr(self):
    """Test object_utils.comma_delimited_str."""
    self.assertEqual(
        object_utils.comma_delimited_str([1, 2, 'abc']), '1, 2, \'abc\'')

  def testAutoPlural(self):
    """Test object_utils.auto_plural."""
    self.assertEqual(object_utils.auto_plural(2, 'number'), 'numbers')
    self.assertEqual(object_utils.auto_plural(2, 'was', 'were'), 'were')


class FormatTest(unittest.TestCase):
  """Tests for object_utils.format."""

  def testFormattable(self):
    """Test Formattable interface."""

    class A(object_utils.Formattable):

      def format(self, compact=True, **kwargs):
        if compact:
          return 'A()'
        else:
          return 'A(...)'

    self.assertEqual(str(A()), 'A(...)')
    self.assertEqual(repr(A()), 'A()')

  def testSimpleTypes(self):
    """Test format on simple types."""
    self.assertEqual(object_utils.format(True, compact=True), 'True')
    self.assertEqual(object_utils.format(1, compact=True), '1')
    self.assertEqual(object_utils.format(1.0, compact=True), '1.0')
    self.assertEqual(object_utils.format('foo', compact=True), '\'foo\'')

    # Compact=False has no impact on simple types.
    self.assertEqual(object_utils.format(True, compact=False), 'True')
    self.assertEqual(object_utils.format(1, compact=False), '1')
    self.assertEqual(object_utils.format(1.0, compact=False), '1.0')
    self.assertEqual(object_utils.format('foo', compact=False), '\'foo\'')

    # Verbose has no impact on simple types.
    self.assertEqual(object_utils.format(True, verbose=True), 'True')
    self.assertEqual(object_utils.format(1, verbose=True), '1')
    self.assertEqual(object_utils.format(1.0, verbose=True), '1.0')
    self.assertEqual(object_utils.format('foo', verbose=True), '\'foo\'')

    # Root indent has no impact on simple types.
    self.assertEqual(object_utils.format(True, root_indent=4), 'True')
    self.assertEqual(object_utils.format(1, root_indent=4), '1')
    self.assertEqual(object_utils.format(1.0, root_indent=4), '1.0')
    self.assertEqual(object_utils.format('foo', root_indent=4), '\'foo\'')

  def testComplexTypes(self):
    """Test format on complex types."""

    class CustomFormattable(object_utils.Formattable):
      """Custom formattable."""

      def format(self, custom_param=None, **kwargs):
        return f'CustomFormattable({custom_param})'

    class A:
      pass

    self.assertEqual(
        object_utils.format(
            {
                'a': CustomFormattable(),
                'b': {
                    'c': [1, 2, 3],
                    'd': ['foo', 'bar', 3, 4, 5]
                }
            },
            compact=True,
            custom_param='foo'),
        "{'a': CustomFormattable(foo), 'b': {'c': [1, 2, 3], "
        "'d': ['foo', 'bar', 3, 4, 5]}}")

    self.assertEqual(
        object_utils.format(
            {
                'a': A(),
                'b': {
                    'c': [1, 2, 3],
                    'd': ['foo', 'bar', 3, 4, 5]
                }
            },
            compact=False,
            list_wrap_threshold=15,
            strip_object_id=True),
        inspect.cleandoc("""{
          'a': A(...),
          'b': {
            'c': [1, 2, 3],
            'd': [
              'foo',
              'bar',
              3,
              4,
              5
            ]
          }
        }"""))

  def testExcludeKeys(self):
    """Test format with excluded keys."""

    class A:
      pass

    class B(object_utils.Formattable):
      """Custom formattable."""

      def format(self, custom_param=None, exclude_keys=None, **kwargs):
        exclude_keys = exclude_keys or set()
        kv = dict(a=1, b=2, c=3)
        kv_pairs = [(k, v, None) for k, v in kv.items()
                    if k not in exclude_keys]
        return f'B({object_utils.kvlist_str(kv_pairs, compact=True)})'

    self.assertEqual(
        object_utils.format(B(), compact=False, exclude_keys=set(['a', 'c'])),
        'B(b=2)')
    self.assertEqual(
        object_utils.format(
            {
                'a': A(),
                'b': B(),
                'c': {
                    'd': [1, 2, 3],
                }
            },
            compact=False,
            list_wrap_threshold=15,
            strip_object_id=True,
            # 'a' should be removed, but 'b.a', 'c.d' should be kept as they are
            # not at the top level.
            exclude_keys=set(['a', 'd'])),
        inspect.cleandoc("""{
          'b': B(a=1, b=2, c=3),
          'c': {
            'd': [1, 2, 3]
          }
        }"""))


class MakeFunctionTest(unittest.TestCase):
  """Tests for object_utils.make_function."""

  def testMakeFunction(self):
    func1 = object_utils.make_function(
        'foo',
        ['x: typing.Optional[int]', 'y: int = 0'],
        ['return x + y'],
        exec_globals=None,
        exec_locals={'typing': typing},
        return_type=int)

    signature = inspect.signature(func1)
    self.assertEqual(list(signature.parameters.keys()), ['x', 'y'])
    self.assertEqual(signature.parameters['x'].annotation, typing.Optional[int])
    self.assertEqual(signature.parameters['y'].annotation, int)
    self.assertEqual(signature.parameters['y'].default, 0)
    self.assertIs(signature.return_annotation, int)
    self.assertEqual(func1(1, 2), 3)

    func2 = object_utils.make_function(
        'foo',
        ['x', 'y'],
        ['return x + y'])
    signature = inspect.signature(func2)
    self.assertEqual(list(signature.parameters.keys()), ['x', 'y'])
    self.assertEqual(
        signature.parameters['x'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(signature.parameters['y'].default, inspect.Signature.empty)
    self.assertIs(signature.return_annotation, inspect.Signature.empty)
    self.assertEqual(func2(1, 2), 3)


if __name__ == '__main__':
  unittest.main()
