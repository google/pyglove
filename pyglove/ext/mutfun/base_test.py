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
from typing import Any, Dict
import unittest

import pyglove.core as pg
from pyglove.ext.mutfun import base


@pg.members([
    ('x', pg.typing.Any()),
    ('y', pg.typing.Any())
])
class Xor(base.Instruction):

  def python_repr(self, block_indent: int = 0) -> str:
    return base.indent(
        base.python_repr(self.x) + ' ^ ' + base.python_repr(self.y),
        block_indent)

  def evaluate(self, variables: Dict[str, Any]) -> Any:
    return base.evaluate(self.x, variables) ^ base.evaluate(self.y, variables)


class CodeTest(unittest.TestCase):
  """Tests for Code."""

  def setUp(self):
    super().setUp()
    self._f = base.Function('f', [
        base.Assign('y', base.Var('x')),                                # 0
        base.Assign('z', base.Var('x')),                                # 1
        base.Function('g', [                                            # 2
            base.Assign('b', base.Var('a')),                            # 2.0
            Xor(base.Var('b'), 2)                                       # 2.1
        ], args=['a']),
        base.Assign('p', Xor(base.Var('y'), 1)),                        # 3
        base.Assign('a', Xor(base.Var('x'), base.Var('y'))),            # 4
        base.Assign('b', Xor(base.Var('x'), base.Var('z'))),            # 5
        base.Assign('c', Xor(base.Var('a'), base.Var('b'))),            # 6
        base.Assign('c', Xor(base.Var('c'), 1)),                        # 7
        base.FunctionCall('g', [Xor(base.Var('x'), base.Var('c'))]),    # 8
        base.Assign('p', 1),                                            # 9
        base.FunctionCall('g', [base.Var('p')]),                        # 10
    ], args=['x'])

  def test_parent_func(self):
    f = self._f
    self.assertIs(f.body[0].parent_func(), f)
    self.assertIs(f.body[0].value.parent_func(), f)
    self.assertIs(f.body[2].parent_func(), f)
    self.assertIs(f.body[2].body[0].parent_func(),
                  f.body[2])
    self.assertIs(f.body[3].value.x.parent_func(), f)
    self.assertIs(f.body[8].args[0].x.parent_func(), f)

  def test_parent_code(self):
    f = self._f
    self.assertIs(f.body[0].value.parent_code(), f.body[0])
    self.assertIs(f.body[2].parent_code(), f)
    self.assertIs(f.body[2].body[0].parent_code(),
                  f.body[2])
    self.assertIs(f.body[3].value.x.parent_code(),
                  f.body[3].value)
    self.assertIs(f.body[3].value.parent_code(),
                  f.body[3])
    self.assertIs(f.body[8].args[0].parent_code(),
                  f.body[8])
    self.assertIs(f.body[8].args[0].x.parent_code(),
                  f.body[8].args[0])

  def test_line(self):
    def assert_line(x, y):
      self.assertIs(x.line(), y)

    f = self._f
    assert_line(f.body[0], f.body[0])
    assert_line(f.body[0].value, f.body[0])
    assert_line(f.body[8].args[0].x, f.body[8])

    # Nested function.
    assert_line(f.body[2].body[0],
                f.body[2].body[0])
    assert_line(f.body[2].body[1].x,
                f.body[2].body[1])

  def test_line_number(self):
    f = self._f
    # Test top-level instructions per line.
    for i, ins in enumerate(f.body):
      self.assertEqual(ins.line_number(), i)

    # Test top-level instructions within the nested function.
    for i, ins in enumerate(f.body[2].body):
      self.assertEqual(ins.line_number(), i)

    # Test cherry-picked inner instructions.
    self.assertEqual(f.body[6].value.x.line_number(), 6)
    self.assertEqual(f.body[8].args[0].x.line_number(), 8)

  def test_preceding_lines(self):
    def assert_preceding_lines(x, y):
      self.assertEqual(list(x.preceding_lines()), y)

    f = self._f
    # Test top-level instructions per line.
    for i in range(len(f.body)):
      assert_preceding_lines(f.body[i], f.body[:i])

    # Test top-level instructions within the nested function.
    for i in range(len(f.body[2].body)):
      assert_preceding_lines(f.body[2].body[i],
                             f.body[2].body[:i])

    # Test cherry-picked inner instructions.
    assert_preceding_lines(f.body[1].value, f.body[:1])
    assert_preceding_lines(f.body[8].args[0].x, f.body[:8])

  def test_preceding_lines_reversed(self):
    def assert_preceding_lines_reversed(x, y):
      self.assertEqual(list(x.preceding_lines_reversed()), list(reversed(y)))

    f = self._f
    # Test top-level instructions per line.
    for i in range(len(f.body)):
      assert_preceding_lines_reversed(f.body[i], f.body[:i])

    # Test top-level instructions within the nested function.
    for i in range(len(f.body[2].body)):
      assert_preceding_lines_reversed(f.body[2].body[i], f.body[2].body[:i])

    # Test cherry-picked inner instructions.
    assert_preceding_lines_reversed(f.body[1].value, f.body[:1])
    assert_preceding_lines_reversed(f.body[8].args[0].x, f.body[:8])

  def test_succeeding_lines(self):
    def assert_succeeding_lines(x, y):
      self.assertEqual(list(x.succeeding_lines()), y)

    f = self._f
    # Test top-level instructions per line.
    for i in range(len(f.body)):
      assert_succeeding_lines(f.body[i], f.body[i + 1:])

    # Test top-level instructions within the nested function.
    for i in range(len(f.body[2].body)):
      assert_succeeding_lines(f.body[2].body[i],
                              f.body[2].body[i + 1:])

    # Test cherry-picked inner instructions.
    assert_succeeding_lines(f.body[1].value, f.body[2:])
    assert_succeeding_lines(f.body[8].args[0].x, f.body[9:])

  def test_input_vars(self):
    def assert_input_vars(x, y):
      self.assertEqual(x.input_vars(), set(y))

    f = self._f
    assert_input_vars(f, [])
    assert_input_vars(f.body[0], ['x'])
    assert_input_vars(f.body[1], ['x'])
    assert_input_vars(f.body[2], [])
    assert_input_vars(f.body[2].body[0], ['a'])
    assert_input_vars(f.body[2].body[1], ['b'])
    assert_input_vars(f.body[3], ['y'])
    assert_input_vars(f.body[4], ['x', 'y'])
    assert_input_vars(f.body[5], ['x', 'z'])
    assert_input_vars(f.body[6], ['a', 'b'])
    assert_input_vars(f.body[7], ['c'])
    assert_input_vars(f.body[8], ['g', 'x', 'c'])

    # Test a bad case.
    g = base.Function('g', [
        base.Assign('y', base.Var('x'))
    ])
    with self.assertRaisesRegex(ValueError, 'Undefined variable .*'):
      g.body[0].input_defs()

  def test_input_vars_transitive(self):
    def assert_input_vars(x, y):
      self.assertEqual(x.input_vars(transitive=True), set(y))

    f = self._f
    assert_input_vars(f, [])
    assert_input_vars(f.body[0], ['x'])
    assert_input_vars(f.body[1], ['x'])
    assert_input_vars(f.body[2], [])
    assert_input_vars(f.body[2].body[0], ['a'])
    assert_input_vars(f.body[2].body[1], ['a', 'b'])
    assert_input_vars(f.body[3], ['x', 'y'])
    assert_input_vars(f.body[4], ['x', 'y'])
    assert_input_vars(f.body[5], ['x', 'z'])
    assert_input_vars(f.body[6], ['a', 'b', 'x', 'y', 'z'])
    assert_input_vars(f.body[7], ['a', 'b', 'x', 'y', 'z', 'c'])
    assert_input_vars(f.body[8], ['g', 'x', 'y', 'z', 'a', 'b', 'c'])

  def test_output_vars(self):
    def assert_output_vars(x, y):
      self.assertEqual(x.output_vars(), set(y))

    f = self._f
    assert_output_vars(f, ['f'])
    assert_output_vars(f.body[0], ['y'])
    assert_output_vars(f.body[1], ['z'])
    assert_output_vars(f.body[2], ['g'])
    assert_output_vars(f.body[2].body[0], ['b'])
    assert_output_vars(f.body[2].body[1], [])
    assert_output_vars(f.body[3], ['p'])
    assert_output_vars(f.body[4], ['a'])
    assert_output_vars(f.body[5], ['b'])
    assert_output_vars(f.body[6], ['c'])
    assert_output_vars(f.body[7], ['c'])
    assert_output_vars(f.body[8], [])

  def test_output_vars_transitive(self):
    def assert_output_vars(x, y):
      self.assertEqual(x.output_vars(transitive=True), set(y))

    f = self._f
    assert_output_vars(f, ['f'])
    assert_output_vars(f.body[0], ['y', 'p', 'a', 'c'])
    assert_output_vars(f.body[1], ['z', 'b', 'c'])
    assert_output_vars(f.body[2], ['g'])
    assert_output_vars(f.body[2].body[0], ['b'])
    assert_output_vars(f.body[2].body[1], [])
    assert_output_vars(f.body[3], ['p'])
    assert_output_vars(f.body[4], ['a', 'c'])
    assert_output_vars(f.body[5], ['b', 'c'])
    assert_output_vars(f.body[6], ['c'])
    assert_output_vars(f.body[7], ['c'])
    assert_output_vars(f.body[8], [])

  def test_seen_vars(self):
    def assert_seen_vars(x, y):
      self.assertEqual(x.seen_vars(), set(y))

    f = self._f
    assert_seen_vars(f, [])
    assert_seen_vars(f.body[0], ['f', 'x'])
    assert_seen_vars(f.body[1], ['f', 'x', 'y'])
    assert_seen_vars(f.body[2], ['f', 'x', 'y', 'z'])
    assert_seen_vars(f.body[2].body[0], ['g', 'a'])
    assert_seen_vars(f.body[2].body[1], ['g', 'a', 'b'])
    assert_seen_vars(f.body[3], ['f', 'x', 'y', 'z', 'g'])
    assert_seen_vars(f.body[4], ['f', 'x', 'y', 'z', 'g', 'p'])
    assert_seen_vars(f.body[5], ['f', 'x', 'y', 'z', 'g', 'p', 'a'])
    assert_seen_vars(f.body[6],
                     ['f', 'x', 'y', 'z', 'g', 'p', 'a', 'b'])
    assert_seen_vars(f.body[7],
                     ['f', 'x', 'y', 'z', 'g', 'p', 'a', 'b', 'c'])
    assert_seen_vars(f.body[8],
                     ['f', 'x', 'y', 'z', 'g', 'p', 'a', 'b', 'c'])

  def test_input_defs(self):
    def assert_input_defs(x, y):
      self.assertEqual(x.input_defs(), y)

    f = self._f
    assert_input_defs(f.body[0], [])
    assert_input_defs(f.body[1], [])
    assert_input_defs(f.body[2], [])
    assert_input_defs(f.body[3], [f.body[0]])
    assert_input_defs(f.body[4], [f.body[0]])
    assert_input_defs(f.body[5], [f.body[1]])
    assert_input_defs(f.body[6], [
        f.body[0],
        f.body[1],
        f.body[4],
        f.body[5]
        ])
    assert_input_defs(f.body[7], [
        f.body[0],
        f.body[1],
        f.body[4],
        f.body[5],
        f.body[6]
        ])
    assert_input_defs(f.body[8], [
        f.body[0],
        f.body[1],
        f.body[2],
        f.body[4],
        f.body[5],
        f.body[6],
        f.body[7]
        ])
    assert_input_defs(f.body[9], [])
    assert_input_defs(f.body[10], [
        f.body[2],
        f.body[9]
        ])

  def test_input_defs_non_transitive(self):
    def assert_input_defs(x, y):
      self.assertEqual(x.input_defs(transitive=False), y)

    f = self._f
    assert_input_defs(f.body[0], [])
    assert_input_defs(f.body[1], [])
    assert_input_defs(f.body[2], [])
    assert_input_defs(f.body[3], [f.body[0]])
    assert_input_defs(f.body[4], [f.body[0]])
    assert_input_defs(f.body[5], [f.body[1]])
    assert_input_defs(f.body[6], [f.body[4], f.body[5]])
    assert_input_defs(f.body[7], [f.body[6]])
    assert_input_defs(f.body[8], [f.body[2], f.body[7]])
    assert_input_defs(f.body[9], [])
    assert_input_defs(f.body[10], [f.body[2], f.body[9]])

  def test_output_refs(self):
    def assert_output_refs(x, y):
      self.assertEqual(x.output_refs(), y)

    f = self._f
    assert_output_refs(f.body[0], [
        f.body[3].value.x,
        f.body[4].value.y,
        f.body[6].value.x,
        f.body[7].value.x,
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[1], [
        f.body[5].value.y,
        f.body[6].value.y,
        f.body[7].value.x,
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[2], [
        f.body[8],
        f.body[10]
    ])
    assert_output_refs(f.body[3], [])
    assert_output_refs(f.body[4], [
        f.body[6].value.x,
        f.body[7].value.x,
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[5], [
        f.body[6].value.y,
        f.body[7].value.x,
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[6], [
        f.body[7].value.x,
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[7], [
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[8], [])
    assert_output_refs(f.body[9], [
        f.body[10].args[0],
    ])
    assert_output_refs(f.body[10], [])

  def test_output_refs_non_transitive(self):
    def assert_output_refs(x, y):
      self.assertEqual(x.output_refs(transitive=False), y)

    f = self._f
    assert_output_refs(f.body[0], [
        f.body[3].value.x,
        f.body[4].value.y,
    ])
    assert_output_refs(f.body[1], [
        f.body[5].value.y,
    ])
    assert_output_refs(f.body[2], [
        f.body[8],
        f.body[10]
    ])
    assert_output_refs(f.body[3], [])
    assert_output_refs(f.body[4], [
        f.body[6].value.x,
    ])
    assert_output_refs(f.body[5], [
        f.body[6].value.y,
    ])
    assert_output_refs(f.body[6], [
        f.body[7].value.x,
    ])
    assert_output_refs(f.body[7], [
        f.body[8].args[0].y,
    ])
    assert_output_refs(f.body[8], [])
    assert_output_refs(f.body[9], [
        f.body[10].args[0],
    ])
    assert_output_refs(f.body[10], [])

  def test_verbose_format(self):
    x = base.Assign('x', 1)
    self.assertEqual(
        x.format(verbose=True),
        'Assign(name=\'x\', value=1)')

  def test_compile(self):
    # Compile the whole function.
    self._f.compile()

    # Compile the last statement.
    self._f.body[-1].compile()

    # Test bad cases.
    g = base.Function('g', [
        base.Assign('y', base.Var('x'))
    ])
    with self.assertRaisesRegex(ValueError, 'Undefined variables .* found'):
      g.body[0].compile()

  def test_select_types(self):
    self.assertEqual(
        list(base.Instruction.select_types(lambda cls: cls is Xor)),
        [Xor])
    self.assertEqual(
        list(
            base.Instruction.select_types(lambda cls: cls is base.Instruction)
        ),
        [],
    )


class AssignTest(unittest.TestCase):
  """Tests for Assign."""

  def test_python_repr(self):
    x = base.Assign('x', 1)
    self.assertEqual(base.python_repr(x), 'x = 1')

    y = base.Assign('y', x)
    self.assertEqual(base.python_repr(y), 'y = x = 1')

  def test_evaluate(self):
    x = base.Assign('x', 1)
    variables = dict()
    self.assertEqual(x.evaluate(variables), 1)
    self.assertEqual(variables, dict(x=1))


class IfTest(unittest.TestCase):
  """Tests for If."""

  def test_evaluate(self):
    one = base.Assign('x', 1)
    two = base.Assign('y', 2)
    i = base.If(predicate=2, true_branch=[one], false_branch=[two])
    variables = dict(x=2, y=3)
    self.assertIsNone(i.evaluate(variables))
    self.assertEqual(variables, dict(x=1, y=3))

    variables = dict(x=2, y=3)
    i = base.If(predicate=False, true_branch=[one])
    self.assertIsNone(i.evaluate(variables))
    self.assertEqual(variables, dict(x=2, y=3))

    variables = dict(x=2, y=3)
    i = base.If(predicate=False, true_branch=[one], false_branch=[two])
    self.assertIsNone(i.evaluate(variables))
    self.assertEqual(variables, dict(x=2, y=2))

    variables = dict()
    i = base.If(
        predicate=one,
        true_branch=[
            base.Assign('y', Xor(base.Var('x'), 1)),
            base.Assign('z', base.Var('y')),
        ],
    )
    self.assertIsNone(i.evaluate(variables))
    self.assertEqual(variables, dict(x=1, y=0, z=0))

  def test_python_repr(self):
    g = Xor(base.Var('b'), 2)
    one = base.Assign('x', 1)
    two = base.Assign('x', 2)
    i = base.If(predicate=g, true_branch=[one], false_branch=[two])
    self.assertEqual(
        i.python_repr(0),
        inspect.cleandoc("""
            if b ^ 2:
              x = 1
            else:
              x = 2
            """),
    )
    nested = base.If(predicate=True, true_branch=[one])
    self.assertEqual(
        nested.python_repr(0),
        inspect.cleandoc("""
            if True:
              x = 1
            """),
    )
    composite_if = base.If(
        predicate=g, true_branch=[nested], false_branch=[two]
    )
    self.assertEqual(
        composite_if.python_repr(0),
        inspect.cleandoc("""
            if b ^ 2:
              if True:
                x = 1
            else:
              x = 2
            """),
    )


class WhileTest(unittest.TestCase):
  """Tests for While loop."""

  def setUp(self):
    super().setUp()
    one = base.Assign('x', 1)
    xor = base.Assign('y', Xor(base.Var('y'), base.Var('x')))
    self._while_true = base.While(predicate=True, body=[one, xor])
    self._while_false = base.While(predicate=False, body=[one])
    self._while_true_to_false = base.While(predicate=xor, body=[one])

  def test_python_repr(self):
    self.assertEqual(
        self._while_true.python_repr(0),
        inspect.cleandoc("""
          while True:
            x = 1
            y = y ^ x
          """),
    )
    self.assertEqual(
        self._while_false.python_repr(0),
        inspect.cleandoc("""
          while False:
            x = 1
          """),
    )
    self.assertEqual(
        self._while_true_to_false.python_repr(0),
        inspect.cleandoc("""
          while y = y ^ x:
            x = 1
          """),
    )

  def test_evaluate(self):
    variables = dict(x=0)
    self.assertIsNone(self._while_false.evaluate(variables))
    self.assertEqual(variables, dict(x=0))
    variables = dict(x=0, y=1)
    self.assertIsNone(self._while_true_to_false.evaluate(variables))
    self.assertEqual(variables, dict(x=1, y=0))


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

  def test_evaluate(self):
    variables = dict(x=1)
    self.assertIs(self._f.evaluate(variables), self._f)
    self.assertEqual(variables, dict(x=1, f=self._f))

  def test_call(self):
    self.assertEqual(self._f(1), 1)
    with self.assertRaisesRegex(
        ValueError, 'Expected .* arguments .* but received .*'):
      self._f(1, 2)

  def test_compile(self):
    self._f.compile()
    with self.assertRaisesRegex(
        ValueError, 'Undefined variables .* found at .*'):
      base.Function('g', [base.Var('x')]).compile()


class InstructionTest(unittest.TestCase):
  """Tests for Instruction."""

  def test_parent_instruction(self):
    f = base.Function('f', [
        base.Assign('y', Xor(base.Var('x'), 1)),
        base.Assign('z', Xor(base.Var('x'), Xor(base.Var('y'), 1))),
        Xor(base.Var('y'), base.Var('z')),
    ], args=['x'])
    self.assertIsNone(f.body[0].value.parent_instruction())
    self.assertIs(f.body[0].value.x.parent_instruction(), f.body[0].value)
    self.assertIs(f.body[1].value.y.parent_instruction(), f.body[1].value)
    self.assertIs(f.body[1].value.y.x.parent_instruction(), f.body[1].value.y)
    self.assertIsNone(f.body[2].parent_instruction())
    self.assertIs(f.body[2].x.parent_instruction(), f.body[2])
    self.assertIs(f.body[2].y.parent_instruction(), f.body[2])


class VarTest(unittest.TestCase):
  """Tests for Var."""

  def test_python_repr(self):
    x = base.Var('x')
    self.assertEqual(str(x), 'x')

  def test_evaluate(self):
    x = base.Var('x')
    with self.assertRaisesRegex(KeyError, 'x'):
      x.evaluate(dict())
    self.assertEqual(x.evaluate(dict(x=1)), 1)


class FunctionCallTest(unittest.TestCase):
  """Tests for FunctionCall."""

  def test_python_repr(self):
    f = base.FunctionCall('f', [1, 2, 3])
    self.assertEqual(base.python_repr(f), 'f(1, 2, 3)')

  def test_evaluate(self):
    main = base.Function('main', [
        base.Function('f', [base.Var('x')], args=['x']),
        base.FunctionCall('f', [1]),
    ], args=['x'])
    self.assertEqual(main(1), 1)

    with self.assertRaisesRegex(
        ValueError, 'Arguments mismatch for function \'f\'.'):
      _ = base.FunctionCall('f', [1, 2]).evaluate(dict(f=main.body[0]))

    with self.assertRaisesRegex(
        ValueError, '.* is not a Function object'):
      _ = base.FunctionCall('f', [1]).evaluate(dict(f=1))

    with self.assertRaisesRegex(
        ValueError, 'Undefined function \'h\'.'):
      _ = base.FunctionCall('h', [1]).evaluate(dict())

  def test_prune(self):
    f = base.Function('f', [
        base.Assign('y', 1),
        base.Assign('y', base.Var('x')),
        base.Assign('z', Xor(base.Var('y'), 1)),
        base.Assign('z', 2),
        Xor(base.Var('y'), base.Var('z'))
    ], args=['x'])
    f.prune()
    self.assertTrue(
        pg.eq(f, base.Function('f', [
            base.Assign('y', base.Var('x')),
            base.Assign('z', 2),
            Xor(base.Var('y'), base.Var('z'))
        ], args=['x'])))


if __name__ == '__main__':
  unittest.main()
