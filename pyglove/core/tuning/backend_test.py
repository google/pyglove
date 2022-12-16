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
"""Tests for pyglove.core.tuning.backend."""

import unittest
from pyglove.core.tuning import backend
from pyglove.core.tuning import local_backend   # pylint: disable=unused-import


class BackendTest(unittest.TestCase):
  """Tests for pluggable backend."""

  def test_pluggable_backend(self):
    self.assertEqual(backend.available_backends(), ['in-memory'])

    @backend.add_backend('test')
    class TestBackend(backend.Backend):  # pylint: disable=unused-variable
      """A fake backend factory for testing."""

      def __init__(self, **kwargs):
        pass

      @classmethod
      def poll_result(cls, name):
        return None

      def next(self):
        return None

    self.assertEqual(backend.available_backends(), ['in-memory', 'test'])
    self.assertEqual(backend.default_backend(), 'in-memory')
    backend.set_default_backend('test')
    self.assertEqual(backend.default_backend(), 'test')

    with self.assertRaisesRegex(
        ValueError, 'Backend .* does not exist'):
      backend.set_default_backend('non-exist-backend')

    with self.assertRaisesRegex(
        TypeError, '.* is not a `pg.tuning.Backend` subclass'):

      @backend.add_backend('bad')
      class BadBackend:  # pylint: disable=unused-variable
        pass
    backend.set_default_backend('in-memory')
    self.assertEqual(backend.default_backend(), 'in-memory')


if __name__ == '__main__':
  unittest.main()
