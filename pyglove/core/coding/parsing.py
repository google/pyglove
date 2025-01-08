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
"""Python code parser with permission control."""

import ast
import sys
from typing import Optional

from pyglove.core.coding import errors
from pyglove.core.coding import permissions


class _CodeValidator(ast.NodeVisitor):
  """Python AST node visitor for ensuring code are permitted."""

  def __init__(
      self,
      code: str,
      permission: permissions.CodePermission
  ):
    super().__init__()
    self.code = code
    self.permission = permission

  def verify(
      self,
      node,
      flag: permissions.CodePermission,
      node_type,
      error_message: str,
  ) -> None:
    """Verifies if the node type is permitted based on flag."""
    if isinstance(node_type, (tuple, list)):
      node_type = tuple(t for t in node_type if t is not None)
    if isinstance(node, node_type) and not (self.permission & flag):
      if sys.version_info >= (3, 10):
        error = SyntaxError(
            error_message,
            (
                '<generated-code>',
                node.lineno,
                node.col_offset,
                self._code_line(node.lineno),
                node.end_lineno,
                node.end_col_offset,
            )
        )
      else:
        error = SyntaxError(
            error_message,
            (
                '<generated-code>',
                node.lineno,
                node.col_offset,
                self._code_line(node.lineno),
            )
        )
        setattr(error, 'end_lineno', node.lineno)
        setattr(error, 'end_col_offset', node.col_offset)
      raise error

  def _code_line(self, lineno):
    return self.code.split('\n')[lineno - 1]

  def generic_visit(self, node):
    self.verify(
        node,
        permissions.CodePermission.ASSIGN,
        (ast.Assign),
        'Assignment is not allowed.',
    )

    self.verify(
        node,
        permissions.CodePermission.CONDITION,
        # Match is not supported until Python 3.10.
        (ast.If, getattr(ast, 'Match', None)),
        'Condition is not allowed.',
    )

    self.verify(
        node,
        permissions.CodePermission.LOOP,
        (ast.For, ast.While, ast.AsyncFor, ast.AsyncWith),
        'Loop is not allowed.',
    )

    self.verify(
        node,
        permissions.CodePermission.EXCEPTION,
        (ast.Try, ast.Raise, ast.Assert),
        'Exception is not allowed.',
    )

    self.verify(
        node,
        permissions.CodePermission.CALL,
        ast.Call,
        'Call is not allowed.',
    )
    self.verify(
        node,
        permissions.CodePermission.CLASS_DEFINITION,
        ast.ClassDef,
        'Class definition is not allowed.',
    )

    self.verify(
        node,
        permissions.CodePermission.FUNCTION_DEFINITION,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Lambda,
            ast.Return,
            ast.Yield,
            ast.YieldFrom,
        ),
        'Function definition is not allowed.',
    )

    self.verify(
        node,
        permissions.CodePermission.IMPORT,
        (ast.Import, ast.ImportFrom),
        '`import` is not allowed.',
    )

    super().generic_visit(node)


def parse(
    code: str,
    permission: Optional[permissions.CodePermission] = None,
) -> ast.AST:
  try:
    parsed_code = ast.parse(code, mode='exec')
    if permission is not None:
      _CodeValidator(code, permission).visit(parsed_code)
  except SyntaxError as e:
    raise errors.CodeError(code, e) from e
  return parsed_code
