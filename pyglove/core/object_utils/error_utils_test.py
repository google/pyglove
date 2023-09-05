# Copyright 2023 The PyGlove Authors
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
"""Tests for pyglove.object_utils.error_utils."""

import unittest
from pyglove.core.object_utils import error_utils


class CatchErrorsTest(unittest.TestCase):

  def assert_caught_error(self, func, errors_to_catch):
    with error_utils.catch_errors(errors_to_catch) as context:
      func()
    self.assertIsNotNone(context.error)

  def assert_propagate_error(self, func, errors_to_catch):

    with self.assertRaises(Exception):
      with error_utils.catch_errors(errors_to_catch):
        func()

  def test_catch_errors(self):
    def foo():
      raise ValueError('this is an error')

    self.assert_caught_error(foo, ValueError)
    self.assert_caught_error(foo, (ValueError,))
    self.assert_caught_error(foo, (KeyError, ValueError))
    self.assert_caught_error(foo, (ValueError, '.* an error'))
    self.assert_caught_error(foo, (KeyError, (ValueError, '.* an error'),))

    self.assert_propagate_error(foo, KeyError)
    self.assert_propagate_error(foo, (ValueError, 'something else'))

  def test_catch_errors_with_error_handler(self):
    errors = []
    def handle_error(error):
      errors.append(error)

    def foo():
      raise ValueError()

    with error_utils.catch_errors(ValueError, handle_error) as context:
      foo()
    self.assertEqual(errors, [context.error])

  def test_catch_errors_bad_inputs(self):
    with self.assertRaisesRegex(
        TypeError, 'Each error specification should be either .*'):
      with error_utils.catch_errors([(ValueError, 'abc', 'abc')]):
        pass

    with self.assertRaisesRegex(
        TypeError, 'Each error specification should be either .*'):
      with error_utils.catch_errors([(ValueError, 1)]):
        pass

    with self.assertRaisesRegex(
        TypeError, 'Exception contains non-except types'):
      with error_utils.catch_errors([str, ValueError]):
        pass


if __name__ == '__main__':
  unittest.main()
