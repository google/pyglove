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
"""Tests for symbolic_program/base.py."""

import inspect
import unittest
from pyglove.ext.mutfun import base


class AssignTest(unittest.TestCase):
  """Tests for Assign."""

  def test_python_repr(self):
    x = base.Assign('x', 1)
    self.assertEqual(base.python_repr(x), 'x = 1')

    y = base.Assign('y', x)
    self.assertEqual(base.python_repr(y), 'y = x = 1')

  def test_verbose_format(self):
    x = base.Assign('x', 1)
    self.assertEqual(
        x.format(verbose=True),
        'Assign(name=\'x\', value=1)')

  def test_compile(self):
    x = base.Assign('x', 1)
    x.compile()
    y = base.Assign('y', base.Var('x'))
    with self.assertRaisesRegex(
        ValueError, 'Undefined variable \'x\''):
      y.compile()
    y.compile(set(['x']))

  def test_seen_vars(self):
    y = base.Assign('y', 1)
    self.assertEqual(y.seen_vars, set())

  def test_evaluate(self):
    x = base.Assign('x', 1)
    variables = dict()
    self.assertEqual(x.evaluate(variables), 1)
    self.assertEqual(variables, dict(x=1))


class VarTest(unittest.TestCase):
  """Tests for Var."""

  def test_python_repr(self):
    x = base.Var('x')
    self.assertEqual(str(x), 'x')

  def test_compile(self):
    x = base.Var('x')
    with self.assertRaisesRegex(
        ValueError, 'Undefined variable \'x\''):
      x.compile()
    x.compile(set(['x']))

  def test_seen_vars(self):
    x = base.Var('x')
    self.assertEqual(x.seen_vars, set())

  def test_evaluate(self):
    x = base.Var('x')
    with self.assertRaisesRegex(KeyError, 'x'):
      x.evaluate(dict())
    self.assertEqual(x.evaluate(dict(x=1)), 1)


class FunctionTest(unittest.TestCase):
  """Tests for Function."""

  def setUp(self):
    super().setUp()
    self._f = base.Function('f', [
        base.Assign('y', base.Var('x')),
        base.Var('y')
    ], args=['x'])

  def test_python_repr(self):
    self.assertEqual(
        base.python_repr(self._f),
        inspect.cleandoc("""
        def f(x):
          y = x
          return y
        """))

  def test_compile(self):
    self._f.compile()
    with self.assertRaisesRegex(
        ValueError, 'Undefined variable \'x\''):
      base.Function('g', [base.Var('x')]).compile()

  def test_seen_vars(self):
    self.assertEqual(self._f.instructions[0].seen_vars, set(['x']))
    self.assertEqual(self._f.instructions[1].seen_vars, set(['x', 'y']))

  def test_evaluate(self):
    self.assertEqual(self._f.evaluate(dict(x=1)), 1)


if __name__ == '__main__':
  unittest.main()
