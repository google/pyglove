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
"""Base for symbolic instructions."""

import abc
from typing import Any, Dict, Optional, Set
import pyglove.core as pg


class Instruction(pg.Object):
  """Base class for all instructions."""

  @abc.abstractmethod
  def evaluate(self, variables: Dict[str, Any]) -> Any:
    """Evaluates current instruction with variable dictionary."""

  @abc.abstractmethod
  def python_repr(self, block_indent: int = 0) -> str:
    """Returns a Python code representation of current instruction."""

  def format(self,
             compact: bool = True,
             verbose: bool = False,
             root_indent: int = 0,
             **kwargs) -> str:
    """Overrides pg.Symbolic.format to support Python program representation."""
    if verbose:
      return super().format(compact, verbose, root_indent=root_indent, **kwargs)
    return self.python_repr(root_indent)

  @property
  def seen_vars(self) -> Set[str]:
    """Returns seen variables prior to this instruction."""
    # Find parent function first.
    parent = self.sym_parent
    while parent is not None and not isinstance(parent, Function):
      parent = parent.sym_parent

    if parent is not None:
      context = pg.Dict(stop_traversal=False, seen_vars=set(parent.args))
      def list_variable(k, v, p):
        del k, p
        if v is self:
          context.stop_traversal = True
          return pg.TraverseAction.STOP
        if not context.stop_traversal:
          if isinstance(v, Assign):
            context.seen_vars.add(v.name)
        return pg.TraverseAction.ENTER
      pg.traverse(parent, postorder_visitor_fn=list_variable)
      return context.seen_vars
    return set()

  def compile(self, seen_vars: Optional[Set[str]] = None):
    """Compiles current instruction with defined variable names."""
    seen_vars: Set[str] = seen_vars or set()
    def check_variable(k: pg.KeyPath, v: Any, p: pg.Symbolic):
      del k
      if isinstance(v, Assign):
        seen_vars.add(v.name)
      elif isinstance(v, Var):
        if v.name not in seen_vars:
          raise ValueError(
              f'Undefined variable \'{v.name}\' used in {p}.')
    pg.traverse(self, check_variable)


#
# Helper methods.
#


def evaluate(value: Any, variables: Dict[str, Any]) -> Any:
  """Evaluates an instruction or a normal value."""
  if isinstance(value, Instruction):
    return value.evaluate(variables)
  return value


def python_repr(value: Any, block_indent: int = 0) -> str:
  """Returns Python code representation of a value."""
  return indent(str(value), block_indent)


def indent(text: str, block_indent: int) -> str:
  """Indents a text."""
  return ' ' * 2 * block_indent + text


#
# Common instructions.
#


@pg.members([
    ('name', pg.typing.Str()),
    ('instructions', pg.typing.List(pg.typing.Object(Instruction))),
    ('args', pg.typing.List(pg.typing.Str(), default=[])),
])
class Function(Instruction):
  """A function that contains a list of instructions."""

  def evaluate(self, variables: Dict[str, Any]) -> Any:
    value = None
    for instruction in self.instructions:
      value = instruction.evaluate(variables)
    return value

  def python_repr(self, block_indent: int = 0) -> str:
    args = ', '.join(self.args)
    r = indent(f'def {self.name}({args}):\n', block_indent)
    for i, instruction in enumerate(self.instructions):
      if i == len(self.instructions) - 1:
        r += indent('return ', block_indent + 1)
        r += instruction.python_repr(0)
      else:
        r += indent(instruction.python_repr(0) + '\n', block_indent + 1)
    return r

  def compile(self, seen_vars: Optional[Set[str]] = None):
    return super().compile(set(self.args).union(seen_vars or set()))


@pg.members([
    ('name', pg.typing.Str()),
    ('value', pg.typing.Any())
])
class Assign(Instruction):
  """Assignment instruction."""

  def evaluate(self, variables: Dict[str, Any]) -> Any:
    value = evaluate(self.value, variables)
    variables[self.name] = value
    return value

  def python_repr(self, block_indent: int = 0) -> str:
    return indent(f'{self.name} = {self.value}', block_indent)


@pg.members([
    ('name', pg.typing.Str())
])
class Var(Instruction):
  """Reference to a variable by name."""

  def evaluate(self, variables: Dict[str, Any]) -> Any:
    return variables[self.name]

  def python_repr(self, block_indent: int = 0) -> str:
    return self.name
