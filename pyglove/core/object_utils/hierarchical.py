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
"""Operating hierarchical object."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pyglove.core.object_utils import common_traits
from pyglove.core.object_utils.missing import MISSING_VALUE
from pyglove.core.object_utils.value_location import KeyPath


def traverse(value: Any,
             preorder_visitor_fn: Optional[Callable[[KeyPath, Any],
                                                    bool]] = None,
             postorder_visitor_fn: Optional[Callable[[KeyPath, Any],
                                                     bool]] = None,
             root_path: Optional[KeyPath] = None) -> bool:
  """Traverse a (maybe) hierarchical value.

  Example::

    def preorder_visit(path, value):
      print(path)

    tree = {'a': [{'c': [1, 2]}, {'d': {'g': (3, 4)}}], 'b': 'foo'}
    pg.object_utils.traverse(tree, preorder_visit)

    # Should print:
    # 'a'
    # 'a[0]'
    # 'a[0].c'
    # 'a[0].c[0]'
    # 'a[0].c[1]'
    # 'a[1]'
    # 'a[1].d'
    # 'a[1].d.g'
    # 'b'

  Args:
    value: A maybe hierarchical value to traverse.
    preorder_visitor_fn: Preorder visitor function.
      Function signature is (path, value) -> should_continue.
    postorder_visitor_fn: Postorder visitor function.
      Function signature is (path, value) -> should_continue.
    root_path: The key path of the root value.

  Returns:
    Whether visitor function returns True on all nodes.
  """
  root_path = root_path or KeyPath()
  def no_op_visitor(key, value):
    del key, value
    return True

  if preorder_visitor_fn is None:
    preorder_visitor_fn = no_op_visitor
  if postorder_visitor_fn is None:
    postorder_visitor_fn = no_op_visitor

  if not preorder_visitor_fn(root_path, value):
    return False
  if isinstance(value, dict):
    for k in value.keys():
      if not traverse(
          value[k], preorder_visitor_fn, postorder_visitor_fn,
          KeyPath(k, root_path)):
        return False
  elif isinstance(value, list):
    for i, v in enumerate(value):
      if not traverse(
          v, preorder_visitor_fn, postorder_visitor_fn, KeyPath(i, root_path)):
        return False
  if not postorder_visitor_fn(root_path, value):
    return False
  return True


def transform(value: Any,
              transform_fn: Callable[[KeyPath, Any], Any],
              root_path: Optional[KeyPath] = None,
              inplace: bool = True) -> Any:
  """Bottom-up (post-order) transform a (maybe) hierarchical value.

  Transform on value is in-place unless `transform_fn` returns a different
  instance.

  Example::

    def _remove_int(path, value):
      del path
      if isinstance(value, int):
        return pg.MISSING_VALUE
      return value

    inputs = {
        'a': {
            'b': 1,
            'c': [1, 'bar', 2, 3],
            'd': 'foo'
        },
        'e': 'bar',
        'f': 4
    }
    output = pg.object_utils.transform(inputs, _remove_int)
    assert output == {
        'a': {
            'c': ['bar'],
            'd': 'foo',
        },
        'e': 'bar'
    })

  Args:
    value: Any python value type. If value is a list of dict, transformation
      will occur recursively.
    transform_fn: Transform function in signature
      (path, value) -> new value
      If new value is MISSING_VALUE, key will be deleted.
    root_path: KeyPath of the root.
    inplace: If True, perform transformation in place.
  Returns:
    Transformed value.
  """
  def _transform(value: Any, current_path: KeyPath) -> Any:
    """Implementation of transform function."""
    new_value = value
    if isinstance(value, dict):
      if not inplace:
        new_value = value.__class__()
      deleted_keys = []
      for k, v in value.items():
        nv = _transform(v, KeyPath(k, current_path))
        if MISSING_VALUE != nv:
          if not inplace or value[k] is not nv:
            new_value[k] = nv
        elif inplace:
          deleted_keys.append(k)

      for k in deleted_keys:
        del value[k]
    elif isinstance(value, list):
      deleted_indices = []
      if not inplace:
        new_value = value.__class__()
      for i, v in enumerate(value):
        nv = _transform(v, KeyPath(i, current_path))
        if MISSING_VALUE != nv:
          if not inplace:
            new_value.append(nv)
          elif value[i] is not nv:
            value[i] = nv
        elif inplace:
          deleted_indices.append(i)
      for i in reversed(deleted_indices):
        del value[i]
    return transform_fn(current_path, new_value)
  return _transform(value, root_path or KeyPath())


def flatten(src: Any, flatten_complex_keys: bool = True) -> Any:
  """Flattens a (maybe) hierarchical value into a depth-1 dict.

  Example::

    inputs = {
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
    }
    output = pg.object_utils.flatten(inputs)
    assert output == {
        'a.e': 1,
        'a.f[0].g': 2,
        'a.f[1].g[0]': 3,
        'a.h': [],
        'a.i.j': {},
        'b': 'hi',
        'c': None
    }

  Args:
    src: source value to flatten.
    flatten_complex_keys: if True, complex keys such as 'x.y' will be flattened
    as 'x'.'y'. For example:
      {'a': {'b.c': 1}} will be flattened into {'a.b.c': 1} if this flag is on,
      otherwise it will be flattened as {'a[b.c]': 1}.

  Returns:
    For primitive value types, `src` itself will be returned.
    For list and dict types, an 1-depth dict will be returned.
    For tuple, a tuple of the same length, with each element flattened will be
    returned. The order of keys in nested ordered dict will be preserved,
    Keys of different depth are joined into a string using "." for dict
    properties and "[]" for list elements.
    For example, if src is::

      {
        "a": {
               "b": 4,
               "c": {"d": 10},
               "e": [1, 2]
               'f': [],
               'g.h': {},
             }
      }

    then the output dict is::

      {
        "a.b": 4,
        "a.c.d": 10,
        "a.e[0]": 1,
        "a.e[1]": 2,
        "a.f": [],
        "a.g.h": {},
      }

    when `flatten_complex_keys` is True, and::

      {
        "a.b": 4,
        "a.c.d": 10,
        "a.e[0]": 1,
        "a.e[1]": 2,
        "a.f": [],
        "a.[g.h]": {},
      }

    when `flatten_complex_keys` is False.

  Raises:
    ValueError: If any key from the nested dictionary contains ".".
  """
  # NOTE(daiyip): Comparing to list, tuple is treated as a single value,
  # whose index is not treated as key. That being said, there is no partial
  # update semantics on elements of tuple in `merge` method too.
  # Thus we simply flatten its elements and keep the tuple form.
  if isinstance(src, tuple):
    return tuple([flatten(elem) for elem in src])

  if not isinstance(src, (dict, list)) or not src:
    return src

  dest = dict()
  def _output_leaf(path: KeyPath, value: Any):
    if path and (not isinstance(value, (dict, list)) or not value):
      dest[path.path_str(not flatten_complex_keys)] = value
    return True
  traverse(src, postorder_visitor_fn=_output_leaf)
  return dest


def try_listify_dict_with_int_keys(
    src: Dict[Any, Any],
    convert_when_sparse: bool = False
    ) -> Tuple[Union[List[Any], Dict[Any, Any]], bool]:
  """Try to convert a dictionary with consequentive integer keys to a list.

  Args:
    src: A dict whose keys may be int type and their range form a perfect
      range(0, N) list unless convert_when_sparse is set to True.
    convert_when_sparse: When src is a int-key dict, force convert
      it to a list ordered by key, even it's sparse.

  Returns:
    converted list or src unchanged.
  """
  if not src:
    return (src, False)

  min_key = None
  max_key = None
  for key in src.keys():
    if not isinstance(key, int):
      return (src, False)
    if min_key is None or min_key > key:
      min_key = key
    if max_key is None or max_key < key:
      max_key = key
  if convert_when_sparse or (min_key == 0 and max_key == len(src) - 1):
    return ([src[key] for key in sorted(src.keys())], True)
  return (src, False)


def canonicalize(src: Any, sparse_list_as_dict: bool = True) -> Any:
  """Canonicalize (maybe) non-canonical hierarchical value.

    Non-canonical hierarchical values are dicts or nested structures of dicts
    that contain keys with '.' or '[<number>]'. Canonicalization is to unfold
    '.' and '[]' in their keys ('.' or '[]') into multi-level dicts.

    For example::

      [1, {
        "a.b[0]": {
          "e.f": 1
        }
        "a.b[0].c[x.y].d": 10
      }]

    will result in::

      [1, {
        "a": {
          "b": [{
            "c": {
              "x.y": {
                "d": 10
              }
            }
            "e": {
              "f": 1
            }
          }]
        }
      }]

   A sparse array indexer can be used in a non-canonical form. e.g::

     {
       'a[1]': 123,
       'a[5]': 234
     }

   This is to accommodate scenarios of list element update/append.
   When `sparse_list_as_dict` is set to true (by default), dict above will be
   converted to::

     {
       'a': [123, 234]
     }

   Otherwise sparse indexer will be kept so the container type will
   remain as a dict::

     {
       'a': {
         1: 123,
         5: 234
       }
     }

   (Please note that sparse indexer as key is not JSON serializable.)

   This is the reverse operation of method flatten.
   If input value is a simple type, the value itself will be returned.

  Args:
     src: A simple type or a nested structure of dict that may contains keys
       with JSON paths like 'a.b.c'
     sparse_list_as_dict: Whether convert sparse list to dict.
       When this is set to True, indices specified in the key path will be
       kept. Otherwise, a list will be returned with elements ordered by indices
       in the path.

  Returns:
     A nested structure of ordered dict that has only canonicalized keys or
     src itself if it's not a nested structure of dict. For dict of int keys
     whose values form a perfect range(0, N) will be returned as a list.

  Raises:
    KeyError: If key is empty or the same key yields conflicting values
      after resolving non-canonical paths. E.g: `{'': 1}` or
      `{'a.b': 1, 'a.b.c': True}`.
  """

  def _merge_fn(path, old_value, new_value):
    if old_value is not MISSING_VALUE and new_value is not MISSING_VALUE:
      raise KeyError(
          f'Path \'{path}\' is assigned with conflicting values. '
          f'Left value: {old_value}, Right value: {new_value}')
    # Always merge.
    return new_value if new_value is not MISSING_VALUE else old_value

  if isinstance(src, dict):
    # We keep order of keys.
    canonical_dict = dict()

    # Make deterministic traversal of dict.
    for key, value in src.items():
      if isinstance(key, str):
        path = KeyPath.parse(key)
      else:
        path = KeyPath(key)

      if len(path) == 1:
        # Key is already canonical.
        # NOTE(daiyip): pass through sparse_list_as_dict to canonicalize
        # value to keep consistency with the container.
        new_value = canonicalize(value, sparse_list_as_dict)
        if path.key not in canonical_dict:
          canonical_dict[path.key] = new_value
        else:
          old_value = canonical_dict[path.key]
          # merge dict is in-place.
          merge_tree(old_value, new_value, _merge_fn)
      else:
        # Key is a path.
        if path.is_root:
          raise KeyError(f'Key must not be empty. Encountered: {src}.')
        sub_root = dict()
        cur_dict = sub_root
        for token in path.keys[:-1]:
          cur_dict[token] = dict()
          cur_dict = cur_dict[token]
        cur_dict[path.key] = canonicalize(
            value, sparse_list_as_dict)
        # merge dict is in-place.
        merge_tree(canonical_dict, sub_root, _merge_fn)

    # NOTE(daiyip): We restore the list form of integer-keyed dict
    # if its keys form a perfect range(0, N), unless sparse_list_as_dict is set.
    def _listify_dict_equivalent(p, v):
      del p
      if isinstance(v, dict):
        v = try_listify_dict_with_int_keys(v, not sparse_list_as_dict)[0]
      return v

    return transform(canonical_dict, _listify_dict_equivalent)
  elif isinstance(src, list):
    return [canonicalize(item, sparse_list_as_dict) for item in src]
  else:
    return src


def merge(value_list: List[Any],
          merge_fn: Optional[Callable[[KeyPath, Any, Any], Any]] = None) -> Any:
  """Merge a list of hierarchical values.

  Example::

    original = {
        'a': 1,
        'b': 2,
        'c': {
            'd': 'foo',
            'e': 'bar'
        }
    }
    patch =  {
        'b': 3,
        'd': [1, 2, 3],
        'c': {
            'e': 'bar2',
            'f': 10
        }
    }
    output = pg.object_utils.merge([original, patch])
    assert output == {
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

  Args:
    value_list: A list of hierarchical values to merge. Later value will be
      treated as updates if it's a dict or otherwise a replacement of former
      value. The merge process will keep input values intact.
    merge_fn: A function to handle value merge that will be called for updated
      or added keys. If a branch is added/updated, the root of branch will be
      passed to merge_fn.
      the signature of function is:
      `(path, left_value, right_value) -> final_value`
      If a key is only present in src dict, old_value is MISSING_VALUE;
      If a key is only present in dest dict, new_value is MISSING_VALUE;
      otherwise both new_value and old_value are filled.
      If final_value is MISSING_VALUE for a path, it will be removed from its
      parent collection.

  Returns:
    A merged value.

  Raises:
    TypeError: If `value_list` is not a list.
    KeyError: If new key is found while not allowed.
  """
  if not isinstance(value_list, list):
    raise TypeError('value_list should be a list')

  if not value_list:
    return None

  new_value = canonicalize(value_list[0], sparse_list_as_dict=True)
  for value in value_list[1:]:
    if value is None:
      continue
    new_value = merge_tree(
        new_value,
        canonicalize(value, sparse_list_as_dict=True),
        merge_fn)

  def _listify_dict_equivalent(p, v):
    del p
    if isinstance(v, dict):
      v = try_listify_dict_with_int_keys(v, True)[0]
    return v
  return transform(new_value, _listify_dict_equivalent)


def _merge_dict_into_dict(
    dest: Dict[Any, Any],
    src: Dict[Any, Any],
    merge_fn: Callable[[KeyPath, Any, Any], Any],
    root_path: KeyPath) -> Dict[Any, Any]:
  """Merge a source dict into the destionation dict."""
  # NOTE(daiyip): When merge_fn is present, we iterate dest dict
  # to call merge_fn on keys that only appears in dest dict.
  keys_to_delete = []
  if merge_fn:
    for key in dest.keys():
      if key not in src:
        new_value = merge_tree(
            dest[key], MISSING_VALUE, merge_fn, KeyPath(key, root_path))
        if MISSING_VALUE != new_value:
          dest[key] = new_value
        else:
          keys_to_delete.append(key)

  # NOTE(daiyip): Merge keys from src dict to dest dict.
  for key, value in src.items():
    is_new = key not in dest
    if is_new or MISSING_VALUE == dest[key]:
      # Key exists in src but not dest (or dest[key] is MISSING_VALUE).
      new_value = value
      if merge_fn:
        new_value = merge_fn(KeyPath(key, root_path), MISSING_VALUE, value)
      if MISSING_VALUE != new_value:
        dest[key] = new_value
    else:
      # Key exists in both src and dest. Replacement scenario.
      old_value = dest[key]
      new_value = merge_tree(old_value, value,
                             merge_fn, KeyPath(key, root_path))
      if new_value is not MISSING_VALUE:
        if old_value is not new_value:
          dest[key] = new_value
      else:
        keys_to_delete.append(key)
  for key in keys_to_delete:
    del dest[key]
  return dest


def _merge_dict_into_list(
    dest: List[Any],
    src: Dict[int, Any],
    root_path: KeyPath) -> List[Any]:
  """Merge (possible) sparsed indexed list (in dict form) into a list."""
  for child_key in src.keys():
    if not isinstance(child_key, int):
      raise KeyError(
          f'Dict must use integers as keys when merging to a list. '
          f'Encountered: {src}, Path: {root_path!r}.')
  num_int_keys = len(src)
  if num_int_keys == len(src.keys()):
    old_size = len(dest)
    for int_key in sorted(src.keys()):
      if int_key < old_size:
        dest[int_key] = src[int_key]
      else:
        dest.append(src[int_key])
  return dest


def merge_tree(dest: Any,
               src: Any,
               merge_fn: Optional[Callable[[KeyPath, Any, Any], Any]] = None,
               root_path: Optional[KeyPath] = None) -> Any:
  """Deep merge two (maybe) hierarchical values.

  Args:
    dest: Destination value.
    src: Source value. When source value is a dict, it's considered as a
      patch (delta) to the destination when destination is a dict or list.
      For other source types, it's considered as a new value that will replace
      dest completely.
    merge_fn: A function to handle value merge that will be called for updated
      or added keys. If a branch is added/updated, the root of branch will be
      passed to merge_fn.
      the signature of function is: (path, left_value, right_value) ->
        final_value
        If a key is only present in src dict, old_value is MISSING_VALUE.
        If a key is only present in dest dict, new_value is MISSING_VALUE.
        Otherwise both new_value and old_value are filled.

        If final value is MISSING_VALUE, it will be removed from its parent
        collection.
   root_path: KeyPath of dest.

  Returns:
    Merged value.

  Raises:
    KeyError: Dict keys are not integers when merging into a list.
  """
  if not root_path:
    root_path = KeyPath()

  if isinstance(dest, dict) and isinstance(src, dict):
    # Merge dict into dict.
    return _merge_dict_into_dict(dest, src, merge_fn, root_path)

  if isinstance(dest, list) and isinstance(src, dict):
    # Merge (possible) sparse indexed list into a list.
    return _merge_dict_into_list(dest, src, root_path)

  # Merge at root level.
  if merge_fn:
    return merge_fn(root_path, dest, src)
  return src


def is_partial(value: Any) -> bool:
  """Returns True if a value is partially bound."""

  def _check_full_bound(path: KeyPath, value: Any) -> bool:
    del path
    if MISSING_VALUE == value:
      return False
    elif (isinstance(value, common_traits.MaybePartial)
          and not isinstance(value, (dict, list))):
      return not value.is_partial
    return True
  return not traverse(value, _check_full_bound)
