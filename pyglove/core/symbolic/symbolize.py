# Copyright 2021 The PyGlove Authors
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
"""Symbolizing existing Python types."""

import inspect
from pyglove.core.symbolic import base
from pyglove.core.symbolic import class_wrapper
from pyglove.core.symbolic import dict as pg_dict
from pyglove.core.symbolic import list as pg_list
from pyglove.core.symbolic.functor import functor_class


# NOTE(daiyip): we removed the pytype annotation for the return value here,
# since there were often false-positive pytype errors. Also it takes a long time
# for checking type errors.
def symbolize(*args, **kwargs):
  """Make a symbolic class/function out of a regular Python class/function.

  ``pg.symbolize`` is introduced for the purpose of making existing
  classes/functions symbolically programmable. For use cases that build
  symbolic classes from scratch (native PyGlove classes), extending `pg.Object`
  with `@pg.members` that declares the symbolic properties is the recommended
  way, which automatically generates the ``__init__`` method and allow
  symbolic attributes to be accessed via `self.<member>`.

  ``pg.symbolize`` can be invoked as a class/function decorator, or as a
  function. When it is used as a decorator, the decorated class or function
  will be converted to a symbolic type (via :func:`pyglove.wrap` and
  :func:`pyglove.functor_class`). This is preferred when user can modify the
  files of existing classes/functions. For example::

    @pg.symbolize
    def foo(a, b):
      return a + b

    f = foo(1, 2)
    f.rebind(a=2)
    f()           # Returns 4

    @pg.symbolize([
      # (Optional) add symbolic constraint for __init__ argument 'a'.
      ('a', pg.typing.Int(min_value=0), 'Description for `a`.')
    ])
    class Foo:
      def __init__(self, a, b):
        self._a = a
        self._b = b

      def result(self):
        return self._a + self._b

    f = Foo(1, 2)
    f.rebind(a=2, b=3)
    f.result()   # Returns 5

  When it used as a function, the input class or function will not be modified.
  Instead, a new symbolic type will be created and returned. This is helpful
  when users want to create new symbolic types from existing classes/functions
  without modifying their original source code. For example::

    def foo(a, b):
      return a + b

    # Create a new symbolic type with constraint on 'a'.
    symbolic_foo = pg.symbolize(foo, [
        ('a', pg.typing.Int(min_value=0))
    ], returns=pg.typing.Int())
    foo(1, 2)    # Returns 3 (foo is kept intact).

    f = symbolic_foo(1, 2)
    f.rebind(a=2)
    f()          # Returns 4.

    class Foo:
      def __init__(self, a, b):
        self._a = a
        self._b = b

      def result(self):
        return self._a + self._b

    SymbolicFoo = pg.symbolize(Foo)
    f = SymbolicFoo(2, 2)
    f.rebind(a=3)
    f.result()   # Returns 5.

  Args:
    *args:  The positional arguments for `symbolize` are:

      * `class_or_fn`: applicable when `symbolize` is called in function mode.
      * `constraints`: an optional list of tuples that allows users to specify
        the constraints for arguments from the `__init__` method (for class)
        or the arguments from the function signature (for function).
        Each tuple should be in format:

           `(<arg_name>, <value_spec>, [description], [arg_metadata])`

        Where `arg_name` is an argument name that is acceptable to the
        `__init__` method of the class, or the function signature;
        'value_spec' is a `pg.ValueSpec` object that validates the value of
        the argument.
        `description` and `arg_metadata` are optional, for documentation and
        meta-programming purposes.
    **kwargs: Keyword arguments will be passsed through to :func:`pyglove.wrap`
      (for symbolizing classes) and :func:`pyglove.functor_class` (for
      symbolizing functions).

  Returns:
    A Symbolic subclass for the decorated/input type.

  Raises:
    TypeError: input type cannot be symbolized, or it's not a type.
  """
  cls_or_fn = None
  if args:
    if inspect.isclass(args[0]) or inspect.isfunction(args[0]):
      cls_or_fn = args[0]
      if cls_or_fn is dict or cls_or_fn is list:
        if len(args) != 1 or kwargs:
          raise ValueError(
              f'Constraints are not supported in symbolic {cls_or_fn!r}. '
              f'Encountered: constraints={args[1]!r}.')
        return pg_dict.Dict if cls_or_fn is dict else pg_list.List
      args = args[1:]
      if len(args) > 1:
        raise ValueError(
            f'Only `constraint` is supported as positional arguments. '
            f'Encountered {args!r}.')
    elif not isinstance(args[0], list):
      raise TypeError(f'{args[0]!r} cannot be symbolized.')

  def _symbolize(cls_or_fn):
    if inspect.isclass(cls_or_fn):
      if (issubclass(cls_or_fn, base.Symbolic)
          and not issubclass(cls_or_fn, class_wrapper.ClassWrapper)):
        raise ValueError(
            f'Cannot symbolize {cls_or_fn!r}: {cls_or_fn.__name__} is already '
            f'a dataclass-like symbolic class derived from `pg.Object`. '
            f'Consider to use `pg.members` to add new symbolic attributes.')
      return class_wrapper.wrap(cls_or_fn, *args, **kwargs)
    assert inspect.isfunction(cls_or_fn), (
        f'Unexpected: {cls_or_fn!r} should be a class or function.')
    return functor_class(
        cls_or_fn, add_to_registry=True, *args, **kwargs)

  if cls_or_fn is not None:
    # When `cls_or_fn` is provided, `symbolize` is called under function mode
    # such as `SymbolicFoo = pg.symbolize(Foo)` or being used as a decorator
    # with no arguments, e.g:
    # ```
    #   @symbolize
    #   class Foo:
    #     pass
    # ```
    # In both case, we return the symbolic type of `cls_or_fn`.
    return _symbolize(cls_or_fn)
  else:
    # Otherwise a factory method is returned to create the symbolic type from
    # a late-bound `cls_or_fn` input, which is the case when `symbolize` is used
    # as a decorator with provided arguments.
    return _symbolize
