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
"""Tests for pyglove.object_utils.value_location."""

import unittest
from pyglove.core.object_utils import value_location


class KeyPathTest(unittest.TestCase):
  """Tests for class KeyPath."""

  def test_basics(self):
    # Root element.
    r = value_location.KeyPath()
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
    a = value_location.KeyPath('a')
    self.assertFalse(a.is_root)
    self.assertEqual(a.key, 'a')
    self.assertEqual(a.path, 'a')
    self.assertEqual(a, 'a')  # Relative path compare.
    self.assertNotEqual(a, '')
    self.assertEqual(a.depth, 1)
    self.assertEqual(len(a), 1)
    self.assertEqual(a.parent, r)

    a2 = value_location.KeyPath(0)
    self.assertFalse(a2.is_root)
    self.assertEqual(a2.key, 0)
    self.assertEqual(str(a2), '[0]')
    self.assertEqual(a2.path, '[0]')
    self.assertEqual(a2, '[0]')
    self.assertNotEqual(a2, '')
    self.assertEqual(a2.depth, 1)
    self.assertEqual(len(a2), 1)
    self.assertEqual(a2.parent, r)

    a3 = value_location.KeyPath('x.y')
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
    b = value_location.KeyPath([1, 'b'])
    self.assertEqual(b, '[1].b')
    self.assertEqual(b.path, '[1].b')
    self.assertNotEqual(a, b)
    self.assertEqual(len(b), 2)
    self.assertEqual(b.parent, '[1]')

    c = value_location.KeyPath('c', b)
    self.assertEqual(c.key, 'c')
    self.assertEqual(c, '[1].b.c')
    self.assertEqual(c.keys, [1, 'b', 'c'])
    self.assertEqual(len(c), 3)
    self.assertEqual(c.parent, b)

    d = value_location.KeyPath(['d', 0], c)
    self.assertEqual(d.key, 0)
    self.assertEqual(d, '[1].b.c.d[0]')
    self.assertEqual(d.keys, [1, 'b', 'c', 'd', 0])
    self.assertEqual(d.parent, '[1].b.c.d')
    self.assertEqual(len(d), 5)

    d2 = value_location.KeyPath(('d', 0), c)
    self.assertEqual(d, d2)

  def test_complex_key_type(self):

    class A:

      def __init__(self, text):
        self._text = text

      def __str__(self):
        return f'A({self._text})'

    p = value_location.KeyPath([A('a'), A('b'), 'c'])
    self.assertEqual(p.path, '[A(a)][A(b)].c')

    # Key may have '.' in their string form.
    p = value_location.KeyPath([A('a.*'), A('$b')])
    self.assertEqual(p.path, '[A(a.*)][A($b)]')

    # NOTE: We cannot really parse KeyPath with complex types.

    class B(value_location.StrKey):
      """Class that implements StrKey will be treated as string key."""

      def __init__(self, text):
        self._text = text

      def __str__(self):
        return f'B({self._text})'

    p = value_location.KeyPath([B('a'), B('b'), 'c'])
    self.assertEqual(p.path, 'B(a).B(b).c')

  def test_parse(self):
    """Test KeyPath.parse method."""
    self.assertEqual(value_location.KeyPath.parse('a').keys, ['a'])
    self.assertEqual(len(value_location.KeyPath.parse('')), 0)

    self.assertEqual(value_location.KeyPath.parse('a').keys, ['a'])
    self.assertEqual(value_location.KeyPath.parse('[a ]').keys, ['a '])
    self.assertEqual(value_location.KeyPath.parse('[0].a').keys, [0, 'a'])
    self.assertEqual(
        value_location.KeyPath.parse('[0][1].a').keys, [0, 1, 'a'])
    self.assertEqual(
        value_location.KeyPath.parse('a.b[1].c').keys, ['a', 'b', 1, 'c'])
    self.assertEqual(
        value_location.KeyPath.parse('a[x[0]].b[y.z].c').keys,
        ['a', 'x[0]', 'b', 'y.z', 'c'])

    with self.assertRaisesRegex(
        ValueError, '\'path_str\' must be a string type.'):
      value_location.KeyPath.parse(0)

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched open bracket at position 0'):
      value_location.KeyPath.parse('[0')

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched open bracket at position 0'):
      value_location.KeyPath.parse('[[0]')

    with self.assertRaisesRegex(
        ValueError,
        'KeyPath parse failed: unmatched close bracket at position 3'):
      value_location.KeyPath.parse('[0]]')

  def test_from_value(self):
    """Test KeyPath.from_value."""
    self.assertEqual(
        value_location.KeyPath.from_value('x.y'),
        value_location.KeyPath(['x', 'y']))

    self.assertEqual(
        value_location.KeyPath.from_value(1),
        value_location.KeyPath([1]))

    path = value_location.KeyPath(['x'])
    self.assertIs(
        value_location.KeyPath.from_value(path),
        path)

    with self.assertRaisesRegex(
        ValueError, '.* is not a valid KeyPath equivalence'):
      value_location.KeyPath.from_value(0.1)

  def test_arithmetics(self):
    """Test KeyPath arithmetics."""

    # Test operator +.
    self.assertEqual(value_location.KeyPath('a') + 'b.c', 'a.b.c')
    self.assertEqual(value_location.KeyPath('a') + '[0].b', 'a[0].b')
    self.assertEqual(value_location.KeyPath('a') + None, 'a')
    self.assertEqual(value_location.KeyPath('a') + 1, 'a[1]')
    self.assertEqual(
        value_location.KeyPath('a') + value_location.KeyPath('b'), 'a.b')
    self.assertEqual(value_location.KeyPath.parse('a.b') + 1.0, 'a.b[1.0]')

    # Test operator -.
    self.assertEqual(
        value_location.KeyPath('a') - value_location.KeyPath('a'), '')
    self.assertEqual(value_location.KeyPath('a') - 'a', '')
    self.assertEqual(value_location.KeyPath('a') - '', 'a')
    self.assertEqual(value_location.KeyPath('a') - None, 'a')
    self.assertEqual(
        value_location.KeyPath('a') - value_location.KeyPath(), 'a')
    self.assertEqual(value_location.KeyPath.parse('a.b.c.d') - 'a.b', 'c.d')
    self.assertEqual(value_location.KeyPath.parse('[0].a') - 0, 'a')

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = value_location.KeyPath('a') - 'b'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = value_location.KeyPath.parse('a.b') - 'a.c'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* are in different subtree.'):
      _ = value_location.KeyPath.parse('a[0]') - 'a[1]'

    with self.assertRaisesRegex(
        ValueError, 'KeyPath subtraction failed: .* is an ancestor'):
      _ = value_location.KeyPath.parse('a.b') - 'a.b.c'

    with self.assertRaisesRegex(TypeError, 'Cannot subtract KeyPath'):
      _ = value_location.KeyPath.parse('a.b') - 1.0

  def test_hash(self):
    self.assertIn(value_location.KeyPath.parse('a.b.c'), {'a.b.c': 1})
    self.assertNotIn(value_location.KeyPath.parse('a.b.c'), {'a.b': 1})

  def test_comparison(self):
    keypath = value_location.KeyPath.parse
    # Equality should only hold true for KeyPaths that are identical.
    self.assertEqual(
        value_location.KeyPath(), value_location.KeyPath.parse(''))
    self.assertEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].b[3][4]'))
    self.assertNotEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].a[3][4]'))
    self.assertNotEqual(keypath('a[1][2].b[3][4]'), keypath('a[1][2].b[4][4]'))
    # Earlier keys in the path should be prioritized over later ones.
    self.assertLess(value_location.KeyPath(), 'a')
    self.assertLess(value_location.KeyPath(), keypath('a'))
    self.assertLess(keypath('a'), keypath('a.a'))
    self.assertLess(keypath('a.a'), keypath('a.b'))
    self.assertGreater(keypath('a'), value_location.KeyPath())
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
        value_location.KeyPath([CustomKey('a'), 'b']),
        value_location.KeyPath([CustomKey('b'), 'b']))

    with self.assertRaisesRegex(
        TypeError, 'Comparison is not supported between instances'):
      _ = value_location.KeyPath() < 1

  def test_query(self):

    def query_shall_succeed(path_str, obj, expected_value, use_inferred=False):
      self.assertEqual(
          value_location.KeyPath.parse(path_str).query(obj, use_inferred),
          expected_value)

    def query_shall_fail(path_str,
                         obj,
                         error='Cannot query sub-key .* of object .*'):
      with self.assertRaisesRegex(KeyError, error):
        value_location.KeyPath.parse(path_str).query(obj)

    def get_shall_succeed(path_str, obj, default, expected_value):
      self.assertEqual(
          value_location.KeyPath.parse(path_str).get(obj, default),
          expected_value)

    def assert_exists(path_str, obj, should_exists):
      self.assertEqual(
          value_location.KeyPath.parse(path_str).exists(obj), should_exists)

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


if __name__ == '__main__':
  unittest.main()
