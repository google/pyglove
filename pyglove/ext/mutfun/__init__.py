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
"""General mutable functions for evolution and symbolic regression.

While users can use hyper primitives such as ``pg.oneof`` to fine-tune specific
parts of pre-existing programs, there are other use cases where it's necessary
to create a program from scratch or transform an existing program into something
drastically different. For example, one may need to find an activation formula
for neural networks or symbolically regress a function with only a few
observations. To address these needs, pg.mutfun has been introduced.

``pg.mutfun`` provides functions and instructions represented by symbolic
objects that allow for maximum flexibility in manipulating a program.
This includes but is not limited to inserting new lines, deleting existing ones,
replacing operations, and creating new functions. With ``pg.mutfun``, users have
access to APIs that make these tasks more manageable. They can easily identify
all downstream instructions that depend on the current instruction, or all
upstream instructions that the current instruction depends on, or finding out
all defined variables up to current instruction.

A mutfun program is a :class:`Function <pyglove.mutfun.Function>` object
illustrated as below::

  f = pg.mutfun.Function('f',
      [
          pg.mutfun.Assign('y', pg.mutfun.Var('x') + 1)
          pg.mutfun.Assign('z', pg.mutfun.Var('x') ** 2)
          pg.mutfun.Var('y') * pg.mutfun.Var('z')
      ], args=['x'])

  assert f(2) == (2 + 1) * 2 ** 2
  print(f)

  >> def f(x):
  >>   y = x + 1
  >>   z = x ** 2
  >>   return y + z

:doc:`../../../notebooks/evolution/function_regression` provides an example for
evolving and doing symbolic regression on mutable functions.

Class hierarchy:

.. graphviz::
   :align: center

    digraph codetypes {
      node [shape="box"];
      edge [arrowtail="empty" arrowhead="none" dir="back" style="dashed"];

      code [label="Code" href="code.html"]
      symbol_def [label="SymbolDefinition" href="symbol_definition.html"];
      assign [label="Assign" href="assign.html"];
      function [label="Function" href="function.html"];
      instruction [label="Instruction" href="instruction.html"];
      symbol_ref [label="SymbolReference" href="symbol_reference.html"];
      var [label="Var" href="var.html"];
      function_call [label="FunctionCall" href="function_call.html"]
      operator [label="Operator", href="operator.html"]
      unary_operator [label="UnaryOperator", href="unary_operator.html"]
      binary_operator [label="BinaryOperator", href="binary_operator.html"]
      user_defined [label="<User-defined instructions>"]

      code -> symbol_def;
      code -> instruction;
      symbol_def -> assign;
      symbol_def -> function;
      instruction -> symbol_ref;
      instruction -> operator;
      instruction -> user_defined;
      symbol_ref -> var;
      symbol_ref -> function_call;
      operator -> unary_operator;
      operator -> binary_operator;
      function -> code [arrowtail="diamond" style="none" label="body"];
    }
"""

# pylint: disable=g-bad-import-order

# Base and common instructions.
from pyglove.ext.mutfun.base import Code

from pyglove.ext.mutfun.base import SymbolDefinition
from pyglove.ext.mutfun.base import Function
from pyglove.ext.mutfun.base import Assign

from pyglove.ext.mutfun.base import Instruction
from pyglove.ext.mutfun.base import SymbolReference
from pyglove.ext.mutfun.base import Var
from pyglove.ext.mutfun.base import FunctionCall

from pyglove.ext.mutfun.base import evaluate
from pyglove.ext.mutfun.base import python_repr

# Basic operators.
from pyglove.ext.mutfun.basic_ops import Operator
from pyglove.ext.mutfun.basic_ops import UnaryOperator
from pyglove.ext.mutfun.basic_ops import Negate
from pyglove.ext.mutfun.basic_ops import BinaryOperator
from pyglove.ext.mutfun.basic_ops import Add
from pyglove.ext.mutfun.basic_ops import Substract
from pyglove.ext.mutfun.basic_ops import Multiply
from pyglove.ext.mutfun.basic_ops import Divide
from pyglove.ext.mutfun.basic_ops import FloorDivide
from pyglove.ext.mutfun.basic_ops import Mod
from pyglove.ext.mutfun.basic_ops import Power
from pyglove.ext.mutfun.basic_ops import Equals
from pyglove.ext.mutfun.basic_ops import NotEquals
from pyglove.ext.mutfun.basic_ops import GreaterThan
from pyglove.ext.mutfun.basic_ops import LessThan


# pylint: enable=g-bad-import-order
