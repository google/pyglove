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
"""Callable extensions."""

from typing import Any, Callable, List

from pyglove.core.typing import callable_signature


class CallableWithOptionalKeywordArgs:
  """Helper class for invoking callable objects with optional keyword args.

  Examples::

    f = pg.typing.CallableWithOptionalKeywordArgs(lambda x: x ** 2, 'y')
    # Returns 4. Keyword 'y' is ignored.
    f(2, y=3)
  """

  def __init__(self,
               func: Callable[..., Any],
               optional_keywords: List[str]):
    sig = callable_signature.signature(
        func, auto_typing=False, auto_doc=False
    )

    # Check for variable keyword arguments.
    if sig.has_varkw:
      absent_keywords = []
    else:
      all_keywords = set(sig.arg_names)
      absent_keywords = [k for k in optional_keywords if k not in all_keywords]
    self._absent_keywords = absent_keywords
    self._func = func

  def __call__(self, *args, **kwargs):
    """Delegate the call to function."""
    for k in self._absent_keywords:
      kwargs.pop(k, None)
    return self._func(*args, **kwargs)
