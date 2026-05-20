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
import sys
import unittest
import pyglove.core as pg
from pyglove.dev import reloader


class ReloaderTest(unittest.TestCase):

  def test_module_dependencies(self):
    dependencies = reloader.module_dependencies(pg)
    self.assertEqual(
        dependencies,
        [
            sys.modules['pyglove.core.symbolic'],
            sys.modules['pyglove.core.typing'],
            sys.modules['pyglove.core.geno'],
            sys.modules['pyglove.core.hyper'],
            sys.modules['pyglove.core.tuning'],
            sys.modules['pyglove.core.detouring'],
            sys.modules['pyglove.core.patching'],
            sys.modules['pyglove.core.utils'],
            sys.modules['pyglove.core.views'],
            sys.modules['pyglove.core.views.html.controls'],
            sys.modules['pyglove.core.io'],
            sys.modules['pyglove.core.coding'],
            sys.modules['pyglove.core.logging'],
            sys.modules['pyglove.core.monitoring'],
        ],
    )

  def test_module_dependencies_transitive(self):
    dependencies = reloader.module_dependencies(
        pg.symbolic,
        transitive=True,
        filter=lambda m: m.__name__.startswith('pyglove.core.symbolic'))

    def index(module_name):
      return dependencies.index(
          sys.modules['pyglove.core.symbolic.' + module_name])

    self.assertLess(index('base'), index('list'))
    self.assertLess(index('origin'), index('base'))
    self.assertLess(index('pure_symbolic'), index('base'))
    self.assertLess(index('object'), index('class_wrapper'))

  def test_module_dependencies_transitive_multiple(self):
    dependencies = reloader.module_dependencies(
        (pg.symbolic, pg.symbolic.origin),
        transitive=True,
        filter=lambda m: m.__name__.startswith('pyglove.core.symbolic'))

    def index(module_name):
      return dependencies.index(
          sys.modules['pyglove.core.symbolic.' + module_name])

    self.assertLess(index('base'), index('list'))
    self.assertLess(index('origin'), index('base'))
    self.assertLess(index('pure_symbolic'), index('base'))
    self.assertLess(index('object'), index('class_wrapper'))

  def test_reload(self):
    _ = reloader.reload(pg)


if __name__ == '__main__':
  unittest.main()
