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
"""Customizable contextual value markers."""
from typing import Any
from pyglove.core.symbolic import base
from pyglove.core.symbolic import functor


class ContextualGetter(functor.Functor, base.ContextualValue):
  """Base for functor-based contextual getter."""

  def value_from(self, context: base.GetAttributeContext) -> Any:
    return self.__call__(context)


def contextual_getter(args=None, returns=None, **kwargs):
  """Decorator that makes ContextualGetter class from function.

  Examples::

    @pg.contextual_getter
    def static_value(self, name, context, value):
      return value

    class A(pg.Object):
      x: pg.ContextualValue = static_value(value=1)

  Args:
    args: A list of tuples that defines the schema for function arguments.
      Please see `functor_class` for detailed explanation of `args`. If None, it
      will be inferenced from the function argument annotations.
    returns: Optional value spec for return value. If None, it will be inferred
      from the function return value annotation.
    **kwargs: Additional keyword argments for controlling the behavior of
      functor creation. Please refer to :func:`pg.symbolic.functor_class` for
      more details.

  Returns:
    A function that converts a regular function into a ``pg.ContextualGetter``
      subclass.
  """
  return functor.functor(args, returns, base_class=ContextualGetter, **kwargs)
