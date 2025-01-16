# Copyright 2025 The PyGlove Authors
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
import concurrent.futures
import unittest
from pyglove.core.utils import contextual


class ContextualTest(unittest.TestCase):

  def test_contextual_override(self):
    with contextual.contextual_override(x=3, y=3, z=3) as parent_override:
      self.assertEqual(
          parent_override,
          dict(
              x=contextual.ContextualOverride(
                  3, cascade=False, override_attrs=False
              ),
              y=contextual.ContextualOverride(
                  3, cascade=False, override_attrs=False
              ),
              z=contextual.ContextualOverride(
                  3, cascade=False, override_attrs=False
              ),
          ),
      )
      self.assertEqual(
          contextual.get_contextual_override('y'),
          contextual.ContextualOverride(3, cascade=False, override_attrs=False),
      )
      self.assertEqual(contextual.contextual_value('x'), 3)
      self.assertIsNone(contextual.contextual_value('f', None))
      with self.assertRaisesRegex(KeyError, '.* does not exist'):
        contextual.contextual_value('f')

      self.assertEqual(contextual.all_contextual_values(), dict(x=3, y=3, z=3))

      # Test nested contextual override with override_attrs=True (default).
      with contextual.contextual_override(
          y=4, z=4, override_attrs=True) as nested_override:
        self.assertEqual(
            nested_override,
            dict(
                x=contextual.ContextualOverride(
                    3, cascade=False, override_attrs=False
                ),
                y=contextual.ContextualOverride(
                    4, cascade=False, override_attrs=True
                ),
                z=contextual.ContextualOverride(
                    4, cascade=False, override_attrs=True
                ),
            ),
        )

    # Test nested contextual override with cascade=True.
    with contextual.contextual_override(x=3, y=3, z=3, cascade=True):
      with contextual.contextual_override(y=4, z=4, cascade=True):
        self.assertEqual(contextual.contextual_value('x'), 3)
        self.assertEqual(contextual.contextual_value('y'), 3)
        self.assertEqual(contextual.contextual_value('z'), 3)

  def test_with_contextual_override(self):
    def func(i):
      del i
      return contextual.contextual_value('x')

    pool = concurrent.futures.ThreadPoolExecutor()
    with contextual.contextual_override(x=3):
      self.assertEqual(contextual.with_contextual_override(func)(0), 3)
      self.assertEqual(
          list(pool.map(contextual.with_contextual_override(func), range(1))),
          [3]
      )


if __name__ == '__main__':
  unittest.main()
