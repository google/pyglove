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
"""Base types for representing mutable functions."""

import abc
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type
import pyglove.core as pg


class Code(pg.Object):
  """Interface for code entity."""

  # This allows Code objects to be stored in dict/set by their ID.
  # We can always use pg.eq/ne for symbolic comparison.
  use_symbolic_comparison = False

  @abc.abstractmethod
  def evaluate(self, context: Dict[str, Any]) -> Any:
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

  def parent_code(self) -> Optional['Code']:
    """Returns the parent code entity of current code entity."""
    parent = self.sym_parent
    while parent is not None and not isinstance(parent, Code):
      parent = parent.sym_parent
    return parent

  def parent_func(self) -> Optional['Function']:
    """Returns the parent function of current code entity."""
    parent = self.sym_parent
    while parent is not None and not isinstance(parent, Function):
      parent = parent.sym_parent
    return parent

  def line(self) -> 'Code':
    """Returns the top-level code entity of current line."""
    parent = self.sym_parent
    line = self
    while parent is not None and not isinstance(parent, Function):
      if isinstance(parent, Code):
        line = parent
      parent = parent.sym_parent
    return line

  def line_number(self) -> int:
    """Returns the 0-based line number of current line within its function."""
    index = self.line().sym_path.key
    assert isinstance(index, int), index
    return index

  def preceding_lines(self) -> Iterable['Code']:
    """Iterates the preceding lines (first first)."""
    parent_func = self.parent_func()
    if parent_func is not None:
      for i in range(self.line_number()):
        yield parent_func.body[i]

  def preceding_lines_reversed(self) -> Iterable['Code']:
    """Iterates the preceding lines in reversed order (recent first)."""
    parent_func = self.parent_func()
    if parent_func is not None:
      for i in reversed(range(self.line_number())):
        yield parent_func.body[i]

  def succeeding_lines(self) -> Iterable['Code']:
    """Iterates the top-level instructions of succeeding lines."""
    parent_func = self.parent_func()
    if parent_func is not None:
      for i in range(self.line_number() + 1, len(parent_func.body)):
        yield parent_func.body[i]

  def input_vars(self, transitive: bool = False) -> Set[str]:
    """Returns the input context from this code entity.

    Args:
      transitive: If True, transitive input context will be included.

    Returns:
      A set of context.
    """
    input_vars = set()
    def list_var_refs(k, v, p):
      del k, p
      # NOTE(daiyip): we do not step into function definitions as for now
      # closure is not supported.
      if isinstance(v, Function):
        return pg.TraverseAction.CONTINUE
      if isinstance(v, SymbolReference):
        input_vars.add(v.name)
      return pg.TraverseAction.ENTER
    pg.traverse(self, list_var_refs)

    if transitive:
      parent_func = self.parent_func()
      if parent_func is not None:
        unresolved_vars = input_vars.copy()
        for i in reversed(range(self.line_number())):
          line = parent_func.body[i]
          line_output_vars = line.output_vars()
          if line_output_vars & unresolved_vars:
            line_input_vars = line.input_vars()
            input_vars.update(line_input_vars)
            unresolved_vars -= line_output_vars
            unresolved_vars.update(line_input_vars)
        assert unresolved_vars.issubset(set(parent_func.args)), unresolved_vars
    return input_vars

  def output_vars(self, transitive: bool = False) -> Set[str]:
    """Returns the output context from this instruction.

    Args:
      transitive: If True, transitive output context will be included.

    Returns:
      A set of output variable names.
    """
    output_vars = set()
    def list_var_defs(k, v, p):
      del k, p
      if isinstance(v, SymbolDefinition):
        output_vars.add(v.name)
      # NOTE(daiyip): we do not step into function definitions as for now
      # closure is not supported.
      if isinstance(v, Function):
        return pg.TraverseAction.CONTINUE
      return pg.TraverseAction.ENTER
    pg.traverse(self.line(), list_var_defs)

    if transitive:
      parent_func = self.parent_func()
      if parent_func is not None:
        for i in (range(self.line_number(), len(parent_func.body))):
          line = parent_func.body[i]
          line_input_vars = line.input_vars()
          if output_vars & line_input_vars:
            output_vars.update(line.output_vars())
    return output_vars

  def seen_vars(self) -> Set[str]:
    """Returns seen context prior to this instruction."""
    seen = set()
    parent_func = self.parent_func()
    if parent_func is not None:
      seen.add(parent_func.name)
      seen.update(parent_func.args)
      for line in self.preceding_lines():
        seen.update(line.output_vars())
    return seen

  def input_defs(
      self, transitive: bool = True) -> List['SymbolDefinition']:
    """Returns the symbol definitions for the inputs of this code entity.

    Args:
      transitive: If True, transitive inputs will be included.
        Otherwise, only the direct dependencies will be included.

    Returns:
      A list of `SymbolDefinition` in their declaration order that produce
      the inputs required for current code entity.
    """
    parent_func = self.parent_func()
    var_producers: Dict[str, Set[SymbolDefinition]] = {
        arg: set() for arg in parent_func.args}
    var_producers[parent_func.name] = set()

    def analyze_var_producers(k: pg.KeyPath, v: Any, p: pg.Symbolic):
      del k, p
      if v is self:
        return pg.TraverseAction.STOP
      if isinstance(v, SymbolDefinition):
        var_entry = set([v])
        if transitive:
          for var_name in v.input_vars():
            var_entry.update(var_producers[var_name])
        var_producers[v.name] = var_entry
      # NOTE(daiyip): do not enter child function definitions.
      if v is not parent_func and isinstance(v, Function):
        return pg.TraverseAction.CONTINUE
      return pg.TraverseAction.ENTER

    pg.traverse(parent_func, analyze_var_producers)

    dependencies: Set[SymbolDefinition] = set()
    for var_name in self.input_vars():
      if var_name not in var_producers:
        raise ValueError(
            f'Undefined variable {repr(var_name)} found in function '
            f'\'{parent_func.name}\' line#{self.line_number()}')
      dependencies.update(var_producers[var_name])
    return sorted(dependencies, key=lambda x: x.line_number())

  def output_refs(
      self, transitive: bool = True) -> List['SymbolReference']:
    """Returns the references to the symbols that this code outputs.

    Args:
      transitive: If True, transitive symbol references will be included.
        Otherwise, only the direct dependencies will be included.

    Returns:
      A list of ``Var` or ``FunctionCall`` in their definition order that
      consume the outputs of current instruction. Users can use
      :meth:`parent_instruction` or :meth:`line` to get their context.
    """
    parent_func = self.parent_func()
    references: List[SymbolReference] = []

    if parent_func is not None:
      output_vars = self.output_vars()
      def find_references(code: Code):
        refs = []
        def identify_reference(k, v, p):
          del k, p
          if isinstance(v, SymbolReference):
            if v.name in output_vars:
              refs.append(v)
        pg.traverse(code, identify_reference)
        return refs

      for line in self.succeeding_lines():
        ins_refs = find_references(line)
        references.extend(ins_refs)
        # Deal with reassignment.
        # For example:
        # ```
        # LN#1: x = 1
        # LN#2: x = 2
        # ```
        # After LN#2, the output vars from LN#1 should be cleared.
        new_assigned = line.output_vars()
        if ins_refs and transitive:
          output_vars.update(new_assigned)
        else:
          output_vars -= new_assigned
    return references

  def compile(self) -> None:
    """Compiles current instruction."""
    if not self.seen_vars().issuperset(self.input_vars()):
      raise ValueError(
          f'Undefined variables {self.input_vars() - self.seen_vars()} '
          f'found at \'{self.sym_path}\'.')


#
# Symbol definitions.
#


@pg.members([
    ('name', pg.typing.Str(), 'Name of the symbol to define.')
])
class SymbolDefinition(Code):
  """Base class for symbol definition."""


def instruction_operrand():
  """Returns value spec for instruction operrand."""
  return pg.typing.Any()


@pg.members([
    ('value', instruction_operrand()),
])
class Assign(SymbolDefinition):
  """Assignment instruction."""

  def evaluate(self, context: Dict[str, Any]) -> Any:
    value = evaluate(self.value, context)
    context[self.name] = value
    return value

  def python_repr(self, block_indent: int = 0) -> str:
    return indent(f'{self.name} = {self.value}', block_indent)


@pg.members([
    ('body', pg.typing.List(pg.typing.Object(Code))),
    ('args', pg.typing.List(pg.typing.Str(), default=[])),
])
class Function(SymbolDefinition):
  """A function that contains a list of instructions."""

  def evaluate(self, context: Dict[str, Any]) -> Any:
    context[self.name] = self
    return self

  def __call__(self,
               *args,
               context: Optional[Dict[str, Any]] = None) -> Any:
    """Invokes current function."""
    if len(args) != len(self.args):
      raise ValueError(
          f'Expected {len(self.args)} arguments ({repr(self.args)}) '
          f'but received {len(args)} ({repr(args)}).')

    local_context = {k: v for k, v in zip(self.args, args)}
    if context:
      local_context.update(context)
    value = None
    for instruction in self.body:
      value = instruction.evaluate(local_context)
    return value

  def python_repr(self, block_indent: int = 0) -> str:
    args = ', '.join(self.args)
    r = indent(f'def {self.name}({args}):\n', block_indent)
    for i, instruction in enumerate(self.body):
      if i == len(self.body) - 1:
        r += indent('return ', block_indent + 1)
        r += instruction.python_repr(0)
      else:
        r += indent(instruction.python_repr(0) + '\n', block_indent + 1)
    return r

  def compile(self):
    """Compiles current function."""
    seen_vars = set(self.args)
    for line in self.body:
      if not seen_vars.issuperset(line.input_vars()):
        diff = line.input_vars() - seen_vars
        raise ValueError(
            f'Undefined variables {diff} found at \'{line.sym_path}\'.')
      seen_vars |= line.output_vars()

  def prune(self):
    """Prune useless instructions."""
    if self.body:
      effective_lines = set(
          x.line_number() for x in self.body[-1].input_defs())
      ineffective_lines = set(
          range(len(self.body) - 1)) - effective_lines
      for i in sorted(ineffective_lines, reverse=True):
        with pg.allow_writable_accessors(True):
          del self.body[i]


#
# Instruction
#


class Instruction(Code):
  """Base class for all instructions."""

  def parent_instruction(self) -> Optional['Instruction']:
    """Returns the parent instruction of current instruction."""
    parent = self.sym_parent
    while parent is not None and not isinstance(parent, Instruction):
      parent = parent.sym_parent
    return parent

  @classmethod
  def select_types(
      cls,
      where: Callable[[Type['Instruction']], bool] = lambda x: True
      ) -> Iterable[Type['Instruction']]:
    """Selects all instruction types that match the condition."""
    for _, t in pg.registered_types():
      if issubclass(t, Instruction) and t is not Instruction and where(t):
        yield t


#
# Control flow
#


class ControlFlow(Instruction):
  """Base class for control flows."""


@pg.members([
    ('predicate', instruction_operrand()),
    ('true_branch', pg.typing.List(pg.typing.Object(Code))),
    ('false_branch', pg.typing.List(pg.typing.Object(Code)).noneable()),
])
class If(ControlFlow):
  """If statement."""

  def evaluate(self, context: Dict[str, Any]):
    cond = evaluate(self.predicate, context)
    if cond:
      for s in self.true_branch:
        evaluate(s, context)
    elif self.false_branch is not None:
      for i in self.false_branch:
        evaluate(i, context)
    return None

  def python_repr(self, block_indent: int = 0) -> str:
    expr = str(self.predicate)
    r = indent(f'if {expr}:\n', block_indent)
    for i, instruction in enumerate(self.true_branch):
      r += instruction.python_repr(block_indent + 1) + (
          '' if i == len(self.true_branch) - 1 else '\n'
      )
    if self.false_branch:
      r += indent('\nelse:\n', block_indent)
      for j, instruction in enumerate(self.false_branch):
        r += instruction.python_repr(block_indent + 1) + (
            '' if j == len(self.false_branch) - 1 else '\n'
        )
    return r


@pg.members([
    ('predicate', instruction_operrand()),
    ('body', pg.typing.List(pg.typing.Object(Code))),
])
class While(ControlFlow):
  """While loop control flow."""

  def evaluate(self, context: Dict[str, Any]):
    while evaluate(self.predicate, context):
      for s in self.body:
        evaluate(s, context)
    return None

  def python_repr(self, block_indent: int = 0) -> str:
    expr = str(self.predicate)
    r = indent(f'while {expr}:\n', block_indent)
    for i, instruction in enumerate(self.body):
      r += instruction.python_repr(block_indent + 1) + (
          '' if i == len(self.body) - 1 else '\n'
      )
    return r


#
# Symbol reference.
#


@pg.members([
    ('name', pg.typing.Str(), 'Name of the symbol to use.')
])
class SymbolReference(Instruction):
  """Base class for symbol references."""


class Var(SymbolReference):
  """Reference to a variable by name."""

  def evaluate(self, context: Dict[str, Any]) -> Any:
    return context[self.name]

  def python_repr(self, block_indent: int = 0) -> str:
    return self.name


@pg.members([
    ('args', pg.typing.List(instruction_operrand(), default=[]),
     'Argument values for calling the function'),
])
class FunctionCall(SymbolReference):
  """Function call."""

  def python_repr(self, block_indent: int = 0) -> str:
    args = ', '.join([str(x) for x in self.args])
    return indent(f'{self.name}({args})', block_indent)

  def evaluate(self, context: Dict[str, Any]) -> Any:
    if self.name not in context:
      raise ValueError(f'Undefined function \'{self.name}\'.')
    func = context[self.name]
    if not isinstance(func, Function):
      raise ValueError(f'\'{self.name}\' is not a Function object.')

    if len(func.args) != len(self.args):
      raise ValueError(
          f'Arguments mismatch for function {repr(self.name)}. '
          f'Expected: {repr(func.args)}, Actual: {repr(self.args)}.')
    args = [evaluate(arg, context) for arg in self.args]
    return func(args, context=context)


#
# Helper methods.
#


def evaluate(value: Any, context: Dict[str, Any]) -> Any:
  """Evaluates an instruction or a normal value."""
  if isinstance(value, Code):
    return value.evaluate(context)
  return value


def python_repr(value: Any, block_indent: int = 0) -> str:
  """Returns Python code representation of a value."""
  return indent(str(value), block_indent)


def indent(text: str, block_indent: int) -> str:
  """Indents a text."""
  return ' ' * 2 * block_indent + text
