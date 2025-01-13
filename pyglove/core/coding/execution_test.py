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
import dataclasses
import inspect
import time
import unittest
from pyglove.core.coding import errors
from pyglove.core.coding import execution
from pyglove.core.coding import permissions


class EvaluateTest(unittest.TestCase):

  def test_with_context(self):
    with execution.context(x=1, y=0):
      with execution.context(x=2, z=2):
        self.assertEqual(
            execution.evaluate(
                inspect.cleandoc(
                    """
                    p = x + y + z
                    """
                ),
                # Override value from the context.
                global_vars=dict(z=3),
                outputs_intermediate=True,
            ),
            dict(p=2 + 0 + 3, __result__=2 + 0 + 3, __stdout__=''),
        )

  def test_basics(self):
    self.assertEqual(
        execution.evaluate(
            inspect.cleandoc(
                """
                x = 1
                y = x + 1
                print(y)
                z = x + y
                """
            ),
            outputs_intermediate=True,
        ),
        dict(x=1, y=2, z=3, __result__=3, __stdout__='2\n'),
    )
    self.assertEqual(
        execution.evaluate(
            inspect.cleandoc(
                """
                x = 1
                y = x + 1
                print(y)
                z = x + y
                """
            ),
        ),
        3,
    )
    with self.assertRaisesRegex(errors.CodeError, 'ValueError'):
      execution.evaluate(
          inspect.cleandoc(
              """
              def foo():
                raise ValueError("intentional error")
              foo()
              """
          ),
          permission=permissions.CodePermission.ALL
      )

  def test_class_def(self):
    ret = execution.evaluate(
        inspect.cleandoc(
            """
            @dataclasses.dataclass
            class A:
              x: int
              y: int
              def __call__(self):
                return self.x + self.y
            """
        ),
        permission=permissions.CodePermission.ALL,
        global_vars=dict(dataclasses=dataclasses),
        outputs_intermediate=True,
    )
    self.assertEqual(list(ret.keys()), ['A', '__result__', '__stdout__'])
    self.assertTrue(inspect.isclass(ret['A']))
    self.assertIs(ret['__result__'], ret['A'])
    self.assertEqual(ret['__stdout__'], '')

  def test_function_def(self):
    ret = execution.evaluate(
        inspect.cleandoc(
            """
            def foo(x, y):
              return x + y

            def bar(z):
              return z + foo(z, z)
            """
        ),
        permission=permissions.CodePermission.ALL,
        outputs_intermediate=True,
    )
    self.assertEqual(
        list(ret.keys()), ['foo', 'bar', '__result__', '__stdout__']
    )
    self.assertTrue(inspect.isfunction(ret['foo']))
    self.assertTrue(inspect.isfunction(ret['bar']))
    self.assertIs(ret['__result__'], ret['bar'])

  def test_function_def_and_call(self):
    code = inspect.cleandoc(
        """
        def foo(x, y):
          return x + y

        def bar(z):
          print(f'z is {z}')
          return z + foo(z, z)

        bar(1)
        """
    )
    ret = execution.evaluate(
        code,
        permission=permissions.CodePermission.ALL,
        outputs_intermediate=True,
    )
    self.assertEqual(
        list(ret.keys()), ['foo', 'bar', '__result__', '__stdout__']
    )
    self.assertEqual(ret['__result__'], 3)
    ret = execution.evaluate(
        code,
        permission=permissions.CodePermission.ALL,
        returns_stdout=True,
    )
    self.assertEqual(ret, 'z is 1\n')

  def test_complex(self):
    ret = execution.evaluate(
        inspect.cleandoc(
            """
            @dataclasses.dataclass
            class A:
              x: int
              y: int
              def __call__(self, z):
                return self.x + self.y + z

            def foo(x, y):
              return x + y
            k = A(1, 2)
            k(foo(3, 4))
            """
        ),
        permission=permissions.CodePermission.ALL,
        global_vars=dict(dataclasses=dataclasses),
        outputs_intermediate=True,
    )
    self.assertEqual(
        list(ret.keys()), ['A', 'foo', 'k', '__result__', '__stdout__']
    )
    self.assertTrue(inspect.isclass(ret['A']))
    self.assertTrue(inspect.isfunction(ret['foo']))
    self.assertIsInstance(ret['k'], ret['A'])
    self.assertEqual(ret['__result__'], 10)

  def test_run_with_error(self):
    with self.assertRaisesRegex(
        errors.CodeError, 'NameError: name .* is not defined'
    ):
      execution.evaluate(
          inspect.cleandoc(
              """
              x = 1
              y = x + z
              """
          ),
          permission=permissions.CodePermission.ALL,
      )
    with self.assertRaisesRegex(errors.CodeError, 'ValueError'):
      execution.evaluate(
          'raise ValueError()', permission=permissions.CodePermission.ALL
      )


@dataclasses.dataclass
class Foo:
  x: int
  y: int


class SandboxCallTest(unittest.TestCase):

  def test_basics(self):
    def f(x, y):
      return x + y
    self.assertEqual(execution.sandbox_call(f, 1, y=2), 3)

  def test_complex_type(self):
    def f(x, y):
      return Foo(x, y)

    self.assertEqual(execution.sandbox_call(f, 1, 2), Foo(1, 2))

  def test_timeout(self):
    def f(x):
      time.sleep(x)

    self.assertIsNone(execution.sandbox_call(f, 0, timeout=1))
    with self.assertRaises(TimeoutError):
      execution.sandbox_call(f, 2, timeout=1)

  def test_raise(self):
    def f(x):
      if x == 0:
        raise ValueError()

    self.assertIsNone(execution.sandbox_call(f, 1))
    with self.assertRaises(ValueError):
      execution.sandbox_call(f, 0)


class CallTest(unittest.TestCase):

  def test_call_without_sandboxing(self):
    def foo(x, y):
      return x + y

    self.assertEqual(
        execution.maybe_sandbox_call(foo, 1, y=2, sandbox=False),
        3
    )

  def test_call_with_sandboxing(self):
    def foo(x, y):
      return x + y

    self.assertEqual(
        execution.maybe_sandbox_call(foo, 1, y=2, sandbox=True),
        3
    )

    def make_cls():
      @dataclasses.dataclass
      class A:
        x: str
      return A

    with self.assertRaises(errors.SerializationError):
      execution.maybe_sandbox_call(make_cls, sandbox=True)

  def test_call_with_automatic_sandboxing(self):
    def foo(x, y):
      return x + y

    self.assertEqual(
        execution.maybe_sandbox_call(foo, 1, y=2),
        3
    )

    def make_cls():
      @dataclasses.dataclass
      class A:
        x: str
      return A

    self.assertTrue(inspect.isclass(execution.maybe_sandbox_call(make_cls)))


class RunTest(unittest.TestCase):

  def test_run_without_sandboxing(self):
    self.assertEqual(
        execution.run(
            'x + y',
            global_vars=dict(x=1, y=2),
            sandbox=False,
        ),
        3,
    )

  def test_run_with_sandboxing(self):
    self.assertEqual(
        execution.run(
            'x + y',
            global_vars=dict(x=1, y=2),
            sandbox=True,
        ),
        3,
    )

  def test_run_with_automatic_sandboxing(self):
    self.assertEqual(
        execution.run(
            'x + y',
            global_vars=dict(x=1, y=2),
        ),
        3,
    )

    r = execution.run(
        inspect.cleandoc("""
            def foo(x, y):
              return x + y

            @dataclasses.dataclass
            class A:
              x: str
            """),
        global_vars=dict(dataclasses=dataclasses),
        outputs_intermediate=True,
    )
    self.assertTrue(inspect.isfunction(r['foo']))
    self.assertTrue(inspect.isclass(r['A']))


if __name__ == '__main__':
  unittest.main()
