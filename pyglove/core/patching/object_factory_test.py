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
"""Tests for pyglove.object_factory."""

import unittest
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.patching.object_factory import ObjectFactory
from pyglove.core.patching.rule_based import patcher as pg_patcher


@pg_patcher([
    ('value', pg_typing.Int())
])
def update_a(unused_src, value):
  return {'a': value}


class ObjectFactoryTest(unittest.TestCase):
  """ObjectFactory test."""

  def test_factory_with_base_value_in_object_form(self):
    v = ObjectFactory(symbolic.Dict, symbolic.Dict(a=1))()
    self.assertEqual(v, {'a': 1})

  def test_factory_with_base_value_in_callable_form(self):
    v = ObjectFactory(symbolic.Dict, lambda: symbolic.Dict(a=1))()
    self.assertEqual(v, {'a': 1})

  def test_factory_with_base_value_in_file_form(self):
    file_db = {}
    def save_handler(v, filepath):
      file_db[filepath] = v

    def load_handler(filepath):
      return file_db[filepath]

    old_save_handler = symbolic.set_save_handler(save_handler)
    old_load_handler = symbolic.set_load_handler(load_handler)

    filepath = 'myfile.json'
    symbolic.Dict(a=1).save(filepath)
    v = ObjectFactory(symbolic.Dict, filepath)()
    self.assertEqual(v, symbolic.Dict(a=1))

    symbolic.set_save_handler(old_save_handler)
    symbolic.set_load_handler(old_load_handler)

  def test_factory_with_patchers(self):
    v = ObjectFactory(
        symbolic.Dict, symbolic.Dict(a=1), ['update_a?value=2'])()
    self.assertEqual(v, {'a': 2})

  def test_factory_with_hierarchical_params_override(self):
    # Using dict for `params_override`.
    v = ObjectFactory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override={
            'a': {
                'x': 2
            }
        })()
    self.assertEqual(v, {'a': {'x': 2, 'y': 2}})

  def test_factory_with_flattened_params_override(self):
    v = ObjectFactory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override={
            'a.x': 2
        })()
    self.assertEqual(v, {'a': {'x': 2, 'y': 2}})

  def test_factory_with_serialized_hierarchical_params_override(self):
    v = ObjectFactory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override='{"a": {"x": 2}}')()
    self.assertEqual(v, {'a': {'x': 2, 'y': 2}})

  def test_factory_with_serialized_flattened_params_override(self):
    v = ObjectFactory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override='{"a.x": 2}')()
    self.assertEqual(v, {'a': {'x': 2, 'y': 2}})

    with self.assertRaisesRegex(
        TypeError,
        'Loaded value .* is not an instance of .*'):
      ObjectFactory(
          symbolic.Dict, symbolic.Dict(a=1),
          params_override='1')()

  def test_factory_with_patchers_and_params_override(self):
    v = ObjectFactory(
        symbolic.Dict, symbolic.Dict(a=1, b=2),
        ['update_a?value=2'],
        params_override={'b': 3, 'c': 0})()
    self.assertEqual(v, {
        'a': 2,
        'b': 3,
        'c': 0
    })

  def test_factory_with_mismatch_object_type(self):
    with self.assertRaisesRegex(
        TypeError,
        '.* is neither an instance of .*, nor a factory or a path '
        'of JSON file that produces an instance of .*'):
      ObjectFactory(symbolic.Dict, symbolic.List())()


if __name__ == '__main__':
  unittest.main()
