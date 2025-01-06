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
from pyglove.core.utils import formatting
from pyglove.core.utils import value_location


KeyPath = value_location.KeyPath
KeyPathSet = value_location.KeyPathSet


class KeyPathTest(unittest.TestCase):
  """Tests for class KeyPath."""

  def test_basics(self):
    # Root element.
    r = KeyPath()
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
    a = KeyPath('a')
    self.assertFalse(a.is_root)
    self.assertEqual(a.key, 'a')
    self.assertEqual(a.path, 'a')
    self.assertEqual(a, 'a')  # Relative path compare.
    self.assertNotEqual(a, '')
    self.assertEqual(a.depth, 1)
    self.assertEqual(len(a), 1)
    self.assertEqual(a.parent, r)

    a2 = KeyPath(0)
    self.assertFalse(a2.is_root)
    self.assertEqual(a2.key, 0)
    with formatting.str_format(markdown=True):
      self.assertEqual(str(a2), '[0]')
    with formatting.repr_format(markdown=True):
      self.assertEqual(repr(a2), '[0]')
    self.assertEqual(a2.path, '[0]')
    self.assertEqual(a2, '[0]')
    self.assertNotEqual(a2, '')
    self.assertEqual(a2.depth, 1)
    self.assertEqual(len(a2), 1)
    self.assertEqual(a2.parent, r)

    a3 = KeyPath('x.y')
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
    b = KeyPath([1, 'b'])
    self.assertEqual(b, '[1].b')
    self.assertEqual(b.path, '[1].b')
    self.assertNotEqual(a, b)
    self.assertEqual(len(b), 2)
    self.assertEqual(b.parent, '[1]')

    c = KeyPath('c', b)
    self.assertEqual(c.key, 'c')
    self.assertEqual(c, '[1].b.c')
    self.assertEqual(c.keys, [1, 'b', 'c'])
    self.assertEqual(len(c), 3)
    self.assertEqual(c.parent, b)

    d = KeyPath(['d', 0], c)
    self.assertEqual(d.key, 0)
    self.assertEqual(d, '[1].b.c.d[0]')
    self.assertEqual(d.keys, [1, 'b', 'c', 'd', 0])
    self.assertEqual(d.parent, '[1].b.c.d')
    self.assertEqual(len(d), 5)

    d2 = KeyPath(('d', 0), c)
    self.assertEqual(d, d2)

  def test_complex_key_type(self):

    class A:

      def __init__(self, text):
        self._text = text

      def __str__(self):
        return f'A({self._text})'

    p = KeyPath([A('a'), A('b'), 'c'])
    self.assertEqual(p.path, '[A(a)][A(b)].c')

    # Key may have '.' in their string form.
    p = KeyPath([A('a.*'), A('$b')])
    self.assertEqual(p.path, '[A(a.*)][A($b)]')

    # NOTE: We cannot really parse KeyPath with complex types.

    class B(value_location.StrKey):
      """Class that implements StrKey will be treated as string key."""

      def __init__(self, text):
        self._text = text

      def __str__(self):
        return f'B({self._text})'

    p = KeyPath([B('a'), B('b'), 'c'])
    self.assertEqual(p.path, 'B(a).B(b).c')

  def test_parse(self):
    """Test KeyPath.parse method."""
    self.assertEqual(KeyPath.parse('a').keys, ['a'])
    self.assertEqual(len(KeyPath.parse('')), 0)

    self.assertEqual(KeyPath.parse('a').keys, ['a'])
    self.assertEqual(KeyPath.parse('[a ]').keys, ['a '])
    self.assertEqual(KeyPath.parse('[0].a').keys, [0, 'a'])
    self.assertEqual(
        KeyPath.parse('[0][1].a').keys, [0, 1, 'a'])
    self.assertEqual(
        KeyPath.parse('a.b[1].c').keys, ['a', 'b', 1, 'c'])
    self.assertEqual(
        KeyPath.parse('a[x[0]].b[y.z].c').keys,
        ['a', 'x[0]', 'b', 'y.z', 'c'])

    with self.assertRaisesRegex(
        ValueError, '\'path_str\' must be a string type.'):
      KeyPath.parse(0)

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched open bracket at position 0'):
      KeyPath.parse('[0')

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched open bracket at position 0'):
      KeyPath.parse('[[0]')

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched close bracket at position 3'):
      KeyPath.parse('[0]]')

  def test_from_value(self):
    """Test KeyPath.from_value."""
    self.assertEqual(
        KeyPath.from_value('x.y'),
        KeyPath(['x', 'y']))

    self.assertEqual(
        KeyPath.from_value(1),
        KeyPath([1]))

    path = KeyPath(['x'])
    self.assertIs(
        KeyPath.from_value(path),
        path)

    with self.assertRaisesRegex(
        ValueError, '.* is not a valid KeyPath equivalence'):
      KeyPath.from_value(0.1)

  def test_arithmetics(self):
    """Test KeyPath arithmetics."""

    # Test operator +.
    self.assertEqual(KeyPath('a') + 'b.c', 'a.b.c')
    self.assertEqual(KeyPath('a') + '[0].b', 'a[0].b')
    self.assertEqual(KeyPath('a') + None, 'a')
    self.assertEqual(KeyPath('a') + 1, 'a[1]')
    self.assertEqual(
        KeyPath('a') + KeyPath('b'), 'a.b')
    self.assertEqual(KeyPath.parse('a.b') + 1.0, 'a.b[1.0]')

    # Test operator -.
    self.assertEqual(
        KeyPath('a') - KeyPath('a'), '')
    self.assertEqual(KeyPath('a') - 'a', '')
    self.assertEqual(KeyPath('a') - '', 'a')
    self.assertEqual(KeyPath('a') - None, 'a')
    self.assertEqual(
        KeyPath('a') - KeyPath(), 'a')
    self.assertEqual(KeyPath.parse('a.b.c.d') - 'a.b', 'c.d')
    self.assertEqual(KeyPath.parse('[0].a') - 0, 'a')

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = KeyPath('a') - 'b'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = KeyPath.parse('a.b') - 'a.c'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = KeyPath.parse('a[0]') - 'a[1]'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* is an ancestor'):
      _ = KeyPath.parse('a.b') - 'a.b.c'

    with self.assertRaisesRegex(TypeError, 'Cannot subtract KeyPath'):
      _ = KeyPath.parse('a.b') - 1.0

  def test_is_relative_to(self):
    self.assertTrue(
        KeyPath.parse('a.b.c').is_relative_to(
            KeyPath())
    )
    self.assertTrue(
        KeyPath.parse('a.b.c').is_relative_to(
            KeyPath.parse('a.b'))
    )
    self.assertTrue(
        KeyPath.parse('a.b.c').is_relative_to(
            KeyPath.parse('a.b.c'))
    )
    self.assertFalse(
        KeyPath.parse('a.b').is_relative_to(
            KeyPath.parse('a.b.c'))
    )
    self.assertFalse(
        KeyPath.parse('a.b.d').is_relative_to(
            KeyPath.parse('a.b.c'))
    )

  def test_hash(self):
    self.assertIn(KeyPath.parse('a.b.c'), {'a.b.c': 1})
    self.assertNotIn(KeyPath.parse('a.b.c'), {'a.b': 1})

  def test_comparison(self):
    keypath = KeyPath.parse
    # Equality should only hold true for KeyPaths that are identical.
    self.assertEqual(
        KeyPath(), KeyPath.parse(''))
    self.assertEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].b[3][4]'))
    self.assertNotEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].a[3][4]'))
    self.assertNotEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].b[4][4]'))
    # Earlier keys in the path should be prioritized over later ones.
    self.assertLess(KeyPath(), 'a')
    self.assertLess(KeyPath(), keypath('a'))
    self.assertLess(keypath('a'), keypath('a.a'))
    self.assertLess(keypath('a.a'), keypath('a.b'))
    self.assertGreater(keypath('a'), KeyPath())
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

    class CustomKey(value_location.StrKey):

      def __init__(self, text):
        self.text = text

      def __lt__(self, other):
        if isinstance(other, CustomKey):
          return self.text < other.text
        return False

    self.assertLess(
        KeyPath([CustomKey('a'), 'b']),
        KeyPath([CustomKey('b'), 'b']))

    with self.assertRaisesRegex(
        TypeError, 'Comparison is not supported between instances'):
      _ = KeyPath() < 1

  def test_query(self):

    def query_shall_succeed(path_str, obj, expected_value, use_inferred=False):
      self.assertEqual(
          KeyPath.parse(path_str).query(obj, use_inferred),
          expected_value)

    def query_shall_fail(path_str,
                         obj,
                         error='Cannot query sub-key .* of object .*'):
      with self.assertRaisesRegex(KeyError, error):
        KeyPath.parse(path_str).query(obj)

    def get_shall_succeed(path_str, obj, default, expected_value):
      self.assertEqual(
          KeyPath.parse(path_str).get(obj, default),
          expected_value)

    def assert_exists(path_str, obj, should_exists):
      self.assertEqual(
          KeyPath.parse(path_str).exists(obj), should_exists)

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

      def sym_inferred(self, key):
        return Bar(z=self._map[key])

      def __contains__(self, key):
        return key in self._map

      def __eq__(self, other):
        return self._map == other._map

      def __ne__(self, other):
        return not self.__eq__(other)

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
    query_shall_succeed('f.x.z', src, Bar(z=0), use_inferred=True)

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
    query_shall_fail('f.x.z', src, 'Cannot query sub-key .* does not exist')

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

  def test_message_on_path(self):
    self.assertEqual(value_location.message_on_path('hi.', None), 'hi.')
    self.assertEqual(
        value_location.message_on_path('hi.', KeyPath()),
        'hi. (path=)')
    self.assertEqual(
        value_location.message_on_path('hi.', KeyPath(['a'])),
        'hi. (path=a)')


class KeyPathSetTest(unittest.TestCase):
  """Tests for class KeyPathSet."""

  def test_empty_set(self):
    s1 = KeyPathSet()
    self.assertFalse(s1)
    self.assertNotIn('', s1)
    self.assertNotIn(KeyPath(), s1)
    self.assertNotIn('abc', s1)
    self.assertNotIn(1, s1)
    self.assertEqual(list(s1), [])
    self.assertIs(s1.subtree(KeyPath()), s1)
    self.assertFalse(s1.subtree('a.b.c'))
    self.assertEqual(s1, KeyPathSet())
    self.assertNotEqual(s1, 1)
    self.assertNotEqual(s1, KeyPathSet([1]))

  def test_add(self):
    s1 = KeyPathSet(
        ['a.b.c', 1, KeyPath([1, 'x']), 'a.b']
    )
    self.assertEqual(
        s1._trie,
        {
            1: {
                'x': {
                    '$': True,
                },
                '$': True
            },
            'a': {
                'b': {
                    'c': {
                        '$': True
                    },
                    '$': True,
                },
            }
        }
    )
    self.assertNotIn('', s1)
    self.assertNotIn('a', s1)
    self.assertIn(KeyPath(['a', 'b']), s1)
    self.assertIn('a.b.c', s1)
    self.assertIn(1, s1)
    self.assertIn('[1]', s1)
    self.assertIn('[1].x', s1)
    self.assertIn(KeyPath([1, 'x']), s1)

    self.assertTrue(s1.add(''))
    self.assertIn('', s1)
    self.assertFalse(s1.add('a.b.c'))

    # Test include_intermediate.
    s1 = KeyPathSet()
    self.assertTrue(s1.add('a.b.c', include_intermediate=True))
    self.assertIn('a', s1)
    self.assertIn('a.b', s1)
    self.assertIn('a.b.c', s1)

  def test_remove(self):
    s1 = KeyPathSet(
        ['a.b.c', 1, KeyPath([1, 'x']), 'a.b', 'c.d']
    )
    self.assertFalse(s1.remove('b'))
    self.assertFalse(s1.remove('c'))
    self.assertTrue(s1.remove('a.b.c'))
    self.assertTrue(s1.remove('a.b'))
    self.assertTrue(s1.remove(1))
    self.assertEqual(
        s1._trie,
        {
            1: {
                'x': {
                    '$': True,
                },
            },
            'c': {
                'd': {
                    '$': True,
                },
            },
        }
    )
    self.assertNotIn(1, s1)
    self.assertTrue(s1.has_prefix(1))

  def test_iter(self):
    self.assertEqual(
        list(KeyPathSet(['', 'a.b.c', 1, KeyPath([1, 'x']), 'a.b'])),
        [
            KeyPath(), KeyPath.parse('a.b.c'), KeyPath.parse('a.b'),
            KeyPath([1]), KeyPath([1, 'x'])
        ]
    )

  def test_has_prefix(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b'])
    self.assertTrue(s1.has_prefix('a'))
    self.assertTrue(s1.has_prefix('a.b'))
    self.assertTrue(s1.has_prefix('a.b.c'))
    self.assertTrue(s1.has_prefix(KeyPath(['a'])))
    self.assertTrue(s1.has_prefix(1))
    self.assertFalse(s1.has_prefix(2))
    self.assertFalse(s1.has_prefix('a.b.c.d'))

  def test_subpaths(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b'])
    self.assertIs(s1.subtree(''), s1)
    self.assertEqual(
        s1.subtree('a'), KeyPathSet(['b.c', 'b'])
    )
    self.assertEqual(s1.subtree(1), KeyPathSet(['', 'x']))
    self.assertEqual(
        s1.subtree(1), KeyPathSet(['', 'x'])
    )

  def test_clear(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b'])
    s1.clear()
    self.assertEqual(s1, KeyPathSet())

  def test_copy(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b'])
    s2 = s1.copy()
    self.assertIsNot(s1, s2)
    self.assertIsNot(s1._trie, s2._trie)
    self.assertIsNot(s1._trie['a'], s2._trie['a'])
    self.assertIsNot(s1._trie['a']['b'], s2._trie['a']['b'])

  def test_update(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b'])
    s1.update(KeyPathSet(['a.b.d', 'a.c', '']))
    self.assertEqual(
        s1._trie,
        {
            1: {
                'x': {
                    '$': True,
                },
                '$': True,
            },
            'a': {
                'b': {
                    'c': {
                        '$': True
                    },
                    'd': {
                        '$': True
                    },
                    '$': True,
                },
                'c': {
                    '$': True
                },
            },
            '$': True,
        }
    )

  def test_union(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b'])
    s2 = s1.union(KeyPathSet(['a.b.d', 'a.c', '']))
    self.assertEqual(
        list(s2),
        [
            KeyPath.parse('a.b.c'),
            KeyPath.parse('a.b'),
            KeyPath.parse('a.b.d'),
            KeyPath.parse('a.c'),
            KeyPath([1]),
            KeyPath([1, 'x']),
            KeyPath(),
        ]
    )
    self.assertIsNot(s2._trie['a'], s1._trie['a'])
    self.assertIsNot(s2._trie['a']['b'], s1._trie['a']['b'])
    self.assertIsNot(
        s2._trie['a']['b']['c'], s1._trie['a']['b']['c']
    )

  def test_difference(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b', ''])
    s2 = s1.difference(
        KeyPathSet(['a.b', 'a.c.b', '[1].x', ''])
    )
    self.assertEqual(
        s2._trie,
        {
            1: {
                '$': True
            },
            'a': {
                'b': {
                    'c': {
                        '$': True
                    }
                }
            }
        }
    )
    self.assertIsNot(s2._trie['a'], s1._trie['a'])
    self.assertIsNot(s2._trie['a']['b'], s1._trie['a']['b'])
    self.assertIsNot(s2._trie[1], s1._trie[1])

    s1.difference_update(KeyPathSet(['a.b', 'a.c.b', '[1].x', '']))
    self.assertEqual(list(s1), ['a.b.c', '[1]'])

  def test_intersection(self):
    s1 = KeyPathSet(['a.b.c', 1, KeyPath([1, 'x']), 'a.b', 'a.c.d', ''])
    s2 = s1.intersection(
        KeyPathSet(['a.b', 'a.b.d', 'a.c.b', '[1].x', ''])
    )
    self.assertEqual(
        s2._trie,
        {
            1: {
                'x': {
                    '$': True,
                },
            },
            'a': {
                'b': {
                    '$': True,
                },
            },
            '$': True,
        }
    )
    self.assertIsNot(s2._trie['a'], s1._trie['a'])
    self.assertIsNot(s2._trie['a']['b'], s1._trie['a']['b'])
    self.assertIsNot(s2._trie[1], s1._trie[1])

    s1.intersection_update(
        KeyPathSet(['a.b', 'a.c.b', '[1].x', ''])
    )
    self.assertEqual(list(s1), ['a.b', '[1].x', ''])

  def test_rebase(self):
    s1 = KeyPathSet(['x.y', 'y', 'y.z.w'])
    s1.rebase('a.b.')
    self.assertEqual(
        list(s1),
        [
            KeyPath.parse('a.b.x.y'),
            KeyPath.parse('a.b.y'),
            KeyPath.parse('a.b.y.z.w'),
        ]
    )

  def test_operator_add(self):
    self.assertEqual(
        KeyPathSet(['a.b.c', 'a.b']) + KeyPathSet(['a.b', '[1].a', '']),
        KeyPathSet(['a.b.c', 'a.b', '[1].a', ''])
    )
    self.assertEqual(
        KeyPath.parse('x[0]') + KeyPathSet(['a.b.c', 'a.b']),
        KeyPathSet(['x[0].a.b.c', 'x[0].a.b'])
    )

  def test_format(self):
    self.assertEqual(
        KeyPathSet(['a.b.c', 'a.b']).format(),
        'KeyPathSet([a.b.c, a.b])'
    )

  def test_from_value(self):
    """Test KeyPathSet.from_value."""
    s = KeyPathSet(['a.b.c'])
    self.assertIs(
        KeyPathSet.from_value(s), s
    )
    self.assertEqual(
        KeyPathSet.from_value(['a.b']),
        KeyPathSet(['a.b'])
    )
    self.assertEqual(
        KeyPathSet.from_value(['a.b'], include_intermediate=True),
        KeyPathSet(['a', 'a.b', ''])
    )
    with self.assertRaisesRegex(
        ValueError, 'Cannot convert .* to KeyPathSet'
    ):
      KeyPathSet.from_value(1)

if __name__ == '__main__':
  unittest.main()
