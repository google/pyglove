# Copyright 2022 The PyGlove Authors
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
"""Pytype support."""

import typing

if typing.TYPE_CHECKING:

  _GenericCallable = typing.TypeVar('_GenericCallable')

  class Decorator(object):
    """A type annotation for decorators that do not change signatures.

    This is a stand-in for using `Callable[[T], T]` to represent a decorator.

    Given a decorator function, which takes in a callable and returns a callable
    with the same signature, apply this class as a decorator to that function.
    This can also be used for decorator factories.

    Examples:

    Plain decorator (decorator matches Callable[[T], T]):

    >>> @pg.typing.Decorator
    ... def my_decorator(func):
    ...   def wrapper(...):
    ...     ...
    ...   return wrapper

    Decorator factory (factory matches Callable[..., Callable[[T], T]]):

    >>> def my_decorator_factory(foo: int):
    ...
    ...   @py.typing.Decorator
    ...   def my_decorator(func):
    ...     ...
    ...   return my_decorator

    This class only exists at build time, for typechecking. At runtime, the
    'Decorator' member of this module is a simple identity function.
    """

    def __init__(
        self,
        decorator: typing.Callable[[_GenericCallable], _GenericCallable]):  # pylint: disable=unused-argument
      ...  # pylint: disable=pointless-statement

    def __call__(self, func: _GenericCallable) -> _GenericCallable:
      ...  # pytype: disable=bad-return-type  # pylint: disable=pointless-statement

else:
  Decorator = lambda d: d
