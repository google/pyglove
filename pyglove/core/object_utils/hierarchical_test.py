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
"""Tests for pyglove.object_utils.hierarchical."""

import unittest
from pyglove.core.object_utils import common_traits
from pyglove.core.object_utils import hierarchical
from pyglove.core.object_utils import value_location


class TraverseTest(unittest.TestCase):
  """Tests for traverse method."""

  def test_simple_types(self):
    visited = []

    def visit_fn(p, v):
      visited[:] = [p, v]
      return True

    self.assertTrue(hierarchical.traverse(None, visit_fn))
    self.assertEqual(visited, [value_location.KeyPath(), None])

    self.assertTrue(hierarchical.traverse(1, visit_fn))
    self.assertEqual(visited, [value_location.KeyPath(), 1])

    self.assertTrue(hierarchical.traverse('abc', visit_fn))
    self.assertEqual(visited, [value_location.KeyPath(), 'abc'])

    self.assertTrue(hierarchical.traverse(True, visit_fn))
    self.assertEqual(visited, [value_location.KeyPath(), True])

    self.assertTrue(hierarchical.traverse((1, 2), visit_fn))
    self.assertEqual(visited, [value_location.KeyPath(), (1, 2)])

  def test_complex_types(self):
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
    self.assertTrue(hierarchical.traverse(tree, previsit_fn, postvisit_fn))
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

  def test_shortcircuit(self):
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
    self.assertFalse(hierarchical.traverse(tree, previsit_fn, postvisit_fn))
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

  def test_empty_dict(self):
    self.assertEqual(
        hierarchical.try_listify_dict_with_int_keys({}), ({}, False))

  def test_non_int_keys(self):
    # Str type key.
    self.assertEqual(
        hierarchical.try_listify_dict_with_int_keys({'a': 1}), ({
            'a': 1
        }, False))

    # Not all keys are int type.
    self.assertEqual(
        hierarchical.try_listify_dict_with_int_keys({
            0: 2,
            'a': 1
        }), ({
            0: 2,
            'a': 1
        }, False))

  def test_sparse_indices(self):
    self.assertEqual(
        hierarchical.try_listify_dict_with_int_keys({
            0: 1,
            2: 2
        }), ({
            0: 1,
            2: 2
        }, False))

    self.assertEqual(
        hierarchical.try_listify_dict_with_int_keys({
            0: 1,
            2: 2
        }, convert_when_sparse=True),
        ([1, 2], True))

  def test_dense_indices(self):
    self.assertEqual(
        hierarchical.try_listify_dict_with_int_keys({
            0: 1,
            1: 2,
            2: 3
        }), ([1, 2, 3], True))


class TransformTest(unittest.TestCase):
  """Tests for transform method."""

  def test_simple_types(self):
    self.assertEqual(hierarchical.transform(1, lambda k, v: v + 1), 2)
    self.assertIsNone(hierarchical.transform(None, lambda k, v: None))
    self.assertEqual(hierarchical.transform(True, lambda k, v: not v), False)
    self.assertEqual(hierarchical.transform('foo', lambda k, v: len(v)), 3)

  def test_complex_types(self):
    def _remove_int(path, value):
      del path
      if isinstance(value, int):
        return hierarchical.MISSING_VALUE
      return value

    self.assertEqual(
        hierarchical.transform(
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
        value, _ = hierarchical.try_listify_dict_with_int_keys(value, True)
      return value

    self.assertEqual(
        hierarchical.transform({
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

  def test_maintain_key_order(self):
    self.assertEqual(
        list(
            hierarchical.flatten(
                hierarchical.transform(
                    {
                        'b': 1,
                        'a': 2,
                        'c': {
                            'f': 'foo',
                            'e': 'bar'
                        }
                    }, lambda p, v: v)).keys()),
        ['b', 'a', 'c.f', 'c.e'])

  def test_inplace_flag(self):
    d1 = {
        'a': 1,
        'b': [
            {'c': [1]}
        ]
    }
    d2 = hierarchical.transform(d1, lambda p, v: v, inplace=True)
    self.assertIs(d1, d2)
    self.assertIs(d1['b'], d2['b'])
    self.assertIs(d1['b'][0], d2['b'][0])
    self.assertIs(d1['b'][0]['c'], d2['b'][0]['c'])

    d3 = hierarchical.transform(
        d1, lambda p, v: v + 1 if isinstance(v, int) else v, inplace=True)
    self.assertEqual(d3, {
        'a': 2,
        'b': [
            {'c': [2]}
        ]
    })
    self.assertIs(d1, d3)
    self.assertIs(d1['b'], d3['b'])
    self.assertIs(d1['b'][0], d3['b'][0])
    self.assertIs(d1['b'][0]['c'], d3['b'][0]['c'])

    d4 = hierarchical.transform(d1, lambda p, v: v, inplace=False)
    self.assertEqual(d1, d4)
    self.assertIsNot(d1, d4)
    self.assertIsNot(d1['b'], d4['b'])
    self.assertIsNot(d1['b'][0], d4['b'][0])
    self.assertIsNot(d1['b'][0]['c'], d4['b'][0]['c'])


class FlattenTest(unittest.TestCase):
  """Tests for class flatten method."""

  def test_simple_types(self):
    self.assertIsNone(hierarchical.flatten(None))
    self.assertEqual(hierarchical.flatten(1), 1)
    self.assertEqual(hierarchical.flatten(1.0), 1.0)
    self.assertEqual(hierarchical.flatten(True), True)
    self.assertEqual(hierarchical.flatten('abc'), 'abc')

  def test_complex_types(self):
    # Test lists.
    self.assertEqual(hierarchical.flatten([]), [])
    self.assertEqual(hierarchical.flatten([1, 2]), {
        '[0]': 1,
        '[1]': 2,
    })
    self.assertEqual(hierarchical.flatten([[1], 2]), {
        '[0][0]': 1,
        '[1]': 2,
    })

    # Test dicts.
    self.assertEqual(hierarchical.flatten({}), {})
    self.assertEqual(
        hierarchical.flatten({
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
        hierarchical.flatten({
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
        hierarchical.flatten({
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
    self.assertEqual(hierarchical.flatten((1, True)), (1, True))
    self.assertEqual(
        hierarchical.flatten(([1, 2, 3], 'foo')), ({
            '[0]': 1,
            '[1]': 2,
            '[2]': 3,
        }, 'foo'))


class CanonicalizeTest(unittest.TestCase):
  """Test canonicalize method."""

  def test_simple_types(self):
    self.assertIsNone(hierarchical.canonicalize(None))
    self.assertEqual(hierarchical.canonicalize('foo'), 'foo')
    self.assertEqual(hierarchical.canonicalize([1, 2, [4, 5]]), [1, 2, [4, 5]])

  def test_sparse_list(self):
    # List is root element.
    self.assertEqual(
        hierarchical.canonicalize({
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
        hierarchical.canonicalize({
            '[0]': 'a',
            '[1]': 'b',
            '[3]': 'c',
        }, sparse_list_as_dict=False),
        ['a', 'b', 'c'])

    # List is intermediate element.
    self.assertEqual(
        hierarchical.canonicalize({
            'a[0]': 'a',
            'a[1]': 'b',
            'a[3]': 'c',
        }, sparse_list_as_dict=True),
        {'a': {
            0: 'a',
            1: 'b',
            3: 'c'
        }})

    self.assertEqual(
        hierarchical.canonicalize({
            'a[0]': 'a',
            'a[1]': 'b',
            'a[3]': 'c',
        }, sparse_list_as_dict=False),
        {'a': ['a', 'b', 'c']})

  def test_complex_types(self):
    # Nested structures of dict.
    self.assertEqual(
        hierarchical.canonicalize({
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
        hierarchical.canonicalize(
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
        hierarchical.canonicalize({
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

  def test_incompatible_values(self):
    with self.assertRaisesRegex(
        KeyError, 'Path \'.*\' is assigned with conflicting values.'):
      hierarchical.canonicalize({'a.b': 1, 'a.b.c': True})

    with self.assertRaisesRegex(KeyError, 'Key must not be empty.'):
      hierarchical.canonicalize({'': 1})

  def test_dict_key_order(self):
    self.assertEqual(
        list(hierarchical.canonicalize({
            'b': 0,
            'a1': 1,
            'a': 2,
        }).keys()), ['b', 'a1', 'a'])

    self.assertEqual(
        list(
            hierarchical.canonicalize({
                'b': 0,
                'a1': 1,
                'a': 2
            }).keys()), ['b', 'a1', 'a'])


class MergeTest(unittest.TestCase):
  """Test merge_dict method."""

  def test_invalid_input(self):
    with self.assertRaisesRegex(TypeError, 'value_list should be a list'):
      hierarchical.merge('abc')

  def test_empty_dicts(self):
    # Merge variations of empty dicts.
    self.assertIsNone(hierarchical.merge([]))
    self.assertEqual(hierarchical.merge([None, {}]), {})

    # Test merge non-empty with empties.
    a = {'a': {'b': 1}}
    self.assertEqual(hierarchical.merge([None, a]), a)
    self.assertEqual(hierarchical.merge([a, None]), a)

  def test_side_effect_free(self):
    a = {'a': 1, 'b': 2}
    b = {'c': 3}
    a2 = dict(a)
    b2 = dict(b)
    self.assertEqual(hierarchical.merge([a, b]), {'a': 1, 'b': 2, 'c': 3})
    self.assertEqual(a, a2)
    self.assertEqual(b, b2)

  def test_canonical_dicts(self):
    # Test merge with standard (canonical) dict merges:
    self.assertEqual(
        hierarchical.merge([
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

  def test_noncanonical_dicts(self):
    # Test merge with noncanonical dict merges.
    self.assertEqual(
        hierarchical.merge([
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

  def test_merge_list(self):
    # Merge list at root level.
    self.assertEqual(
        hierarchical.merge([[1, 2, 3], {
            '[0]': -1,
            '[3]': 4
        }]), [-1, 2, 3, 4])

    # Replace entire list.
    self.assertEqual(
        hierarchical.merge([{
            'a': [0, 1, 2],
        }, {
            'a': [3]
        }]), {'a': [3]})

    # Update single element and append.
    self.assertEqual(
        hierarchical.merge([{
            'a': [0, 1, 2],
        }, {
            'a[0]': -1,
            'a[10]': 3,
        }]), {'a': [-1, 1, 2, 3]})

    with self.assertRaisesRegex(
        KeyError, 'Dict must use integers as keys when merging to a list.'):
      hierarchical.merge([[0, 1, 2], {'a': 1}])

  def test_delete_semantics(self):
    def _remove_fixed_key(path, old_value, new_value):
      if path.key == 'c' or path.key == 'b':
        return hierarchical.MISSING_VALUE
      return new_value if hierarchical.MISSING_VALUE != new_value else old_value

    self.assertEqual(
        hierarchical.merge([{
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
        }], merge_fn=_remove_fixed_key),
        {
            'a': {
                # NOTE(daiyip): 'a.c' is removed by _remove_fixed_keys.
                'd': False,
                'e': 'foo'
            },
            # 'b' and 'c' are removed by _remove_fixed_keys.
            'd': 4
        })

  def test_customized_merge_fn(self):
    added_keys = []
    updated_keys = []

    def _merge_fn(path, old_value, new_value):
      if new_value is hierarchical.MISSING_VALUE:
        return old_value

      if old_value is hierarchical.MISSING_VALUE:
        added_keys.append(path.path)
      else:
        updated_keys.append(path.path)
        if isinstance(new_value, int) and isinstance(old_value, int):
          return new_value + old_value
      return new_value

    self.assertEqual(
        hierarchical.merge([{
            'a': 1,
            'b': {
                'e': 2,
                'f': 3,
            }
        }, {
            'b.f': 4,
            'b.g': 5,
            'c': 0
        }], merge_fn=_merge_fn),
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

  def test_merge_key_order(self):
    # For ordered dict, order of keys are preserved.
    # Newly added keys are appended.
    self.assertEqual(
        list(
            hierarchical.merge([
                {'b': 1, 'a': 2, 'c': 1},
                {'a': 'foo', 'd': 'bar'}
            ]).keys()),
        ['b', 'a', 'c', 'd'])

    # For dict, the order of keys is preserved.
    # Newly added keys are appended.
    self.assertEqual(
        list(
            hierarchical.merge([
                {'b': 1, 'a': 2, 'c': 1},
                {'a': 'foo', 'f': 'something', 'd': 'bar'}
            ]).keys()),
        ['b', 'a', 'c', 'f', 'd'])


class PartialTest(unittest.TestCase):
  """Test hierarchical.is_partial."""

  def test_simple_types(self):
    self.assertFalse(hierarchical.is_partial(1))
    self.assertFalse(hierarchical.is_partial(True))
    self.assertFalse(hierarchical.is_partial(1.0))
    self.assertFalse(hierarchical.is_partial('abc'))
    self.assertFalse(hierarchical.is_partial(None))
    self.assertTrue(hierarchical.is_partial(hierarchical.MISSING_VALUE))

  def test_complex_types(self):

    class A:
      pass

    class B(common_traits.MaybePartial):

      def missing_values(self):
        return {'SOME_KEY': 1}

    self.assertFalse(hierarchical.is_partial([1, 2]))
    self.assertFalse(hierarchical.is_partial({'foo': 'bar'}))
    self.assertFalse(hierarchical.is_partial([1, {'foo': A()}]))
    self.assertTrue(
        hierarchical.is_partial({'foo': hierarchical.MISSING_VALUE}))
    self.assertTrue(hierarchical.is_partial([hierarchical.MISSING_VALUE]))
    self.assertTrue(hierarchical.is_partial([{'a': 1, 'b': [B()]}]))


if __name__ == '__main__':
  unittest.main()
