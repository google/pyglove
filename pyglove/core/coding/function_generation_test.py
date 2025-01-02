# Copyright 2024 The PyGlove Authors
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
import inspect
import typing
import unittest

from pyglove.core.coding import function_generation


class MakeFunctionTest(unittest.TestCase):
  """Tests for function_generation.make_function."""

  def test_make_function_with_type_annotations(self):
    func = function_generation.make_function(
        'foo',
        ['x: typing.Optional[int]', 'y: int = 0'],
        ['return x + y'],
        exec_globals=None,
        exec_locals={'typing': typing},
        return_type=int)

    signature = inspect.signature(func)
    self.assertEqual(list(signature.parameters.keys()), ['x', 'y'])
    self.assertEqual(signature.parameters['x'].annotation, typing.Optional[int])
    self.assertEqual(signature.parameters['y'].annotation, int)
    self.assertEqual(signature.parameters['y'].default, 0)
    self.assertIs(signature.return_annotation, int)
    self.assertEqual(func(1, 2), 3)

  def test_make_function_without_type_annotations(self):
    func = function_generation.make_function(
        'foo',
        ['x', 'y'],
        ['return x + y'])
    signature = inspect.signature(func)
    self.assertEqual(list(signature.parameters.keys()), ['x', 'y'])
    self.assertEqual(
        signature.parameters['x'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(signature.parameters['y'].default, inspect.Signature.empty)
    self.assertIs(signature.return_annotation, inspect.Signature.empty)
    self.assertEqual(func(1, 2), 3)


if __name__ == '__main__':
  unittest.main()
