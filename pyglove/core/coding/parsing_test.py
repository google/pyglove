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
import inspect
import unittest
from pyglove.core.coding import errors
from pyglove.core.coding import parsing
from pyglove.core.coding import permissions


class ParsePythonCodeTest(unittest.TestCase):

  def test_no_permission_check(self):
    ast = parsing.parse(inspect.cleandoc("""
        if x > 0:
          print(x)
        """))
    self.assertIsNotNone(ast)

  def assert_allowed(self, code: str, permission: permissions.CodePermission):
    ast = parsing.parse(inspect.cleandoc(code), permission)
    self.assertIsNotNone(ast)

  def assert_not_allowed(
      self, code: str, permission: permissions.CodePermission
  ):
    with self.assertRaisesRegex(errors.CodeError, '.* is not allowed'):
      parsing.parse(inspect.cleandoc(code), permission)

  def test_parse_with_allowed_code(self):
    self.assert_allowed(
        """
        x = 1
        """,
        permissions.CodePermission.ASSIGN,
    )
    self.assert_allowed(
        """
        if x > 0:
          pass
        """,
        permissions.CodePermission.CONDITION,
    )
    self.assert_allowed(
        """
        for i in [1, 2, 3]:
          pass
        """,
        permissions.CodePermission.LOOP,
    )
    self.assert_allowed(
        """
        foo(x=1, y=bar(2))
        """,
        permissions.CodePermission.CALL,
    )
    self.assert_allowed(
        """
        assert x > 1
        """,
        permissions.CodePermission.EXCEPTION,
    )
    self.assert_allowed(
        """
        class A:
          pass
        """,
        permissions.CodePermission.CLASS_DEFINITION,
    )
    self.assert_allowed(
        """
        def foo(x, y):
          return x + y
        """,
        permissions.CodePermission.FUNCTION_DEFINITION,
    )
    self.assert_allowed(
        """
        import re
        """,
        permissions.CodePermission.IMPORT,
    )

  def test_parse_with_not_allowed_code(self):
    self.assert_not_allowed(
        """
        x = 1
        """,
        permissions.CodePermission.CONDITION,
    )
    self.assert_not_allowed(
        """
        if x > 0:
          pass
        """,
        permissions.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        for i in range(5):
          pass
        """,
        permissions.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        assert x > 1
        """,
        permissions.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        class A:
          pass
        """,
        permissions.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        def foo(x, y):
          return x + y
        """,
        permissions.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        import re
        """,
        permissions.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        range(5)
        """,
        permissions.CodePermission.ASSIGN,
    )


if __name__ == '__main__':
  unittest.main()
