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
"""Utilities for working with errors."""

import contextlib
import dataclasses
import inspect
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


@dataclasses.dataclass()
class CatchErrorsContext:
  """Context for pg.catch_errors."""
  error: Optional[Exception] = None


@contextlib.contextmanager
def catch_errors(
    errors: Union[
        Union[Type[Exception], Tuple[Exception, str]],
        Sequence[Union[Type[Exception], Tuple[Exception, str]]],
    ],
    error_handler: Optional[Callable[[Exception], None]] = None
):
  """Context manager for catching user-specified exceptions.

  Examples::

    with pg.object_utils.catch_errors(
        [
            RuntimeErrror,
            (ValueError, 'Input is wrong.')
        ],
    ) as error_context:
      do_something()

    if error_context.error:
      # Error branch.
      handle_error(error_context.error)

  Args:
    errors: A sequence of exception types or tuples of exception type and error
      messages (described in regular expression) as the desired exception types
      to catch. If an error is raised within the scope which does not match with
      the specification, it will be propagated to the outer scope.
    error_handler: An optional callable object to handle the error on failure.
      It's usually provided if the user want to create a context manager based
      on `pg.catch_errors` with specific error handling logics.

  Yields:
    A CatchErrorsContext object.
  """
  if not isinstance(errors, (tuple, list)):
    errors = [errors]
  elif (
      isinstance(errors, tuple)
      and len(errors) == 2
      and isinstance(errors[1], str)  # pytype: disable=not-indexable
  ):
    errors = [errors]

  error_mapping: Dict[Type[Exception], List[str]] = {}
  for error_type in errors:
    regex = None
    if isinstance(error_type, tuple):
      if len(error_type) != 2 or not isinstance(error_type[1], str):  # pytype: disable=not-indexable
        raise TypeError(
            'Each error specification should be either an Exception type or '
            'a tuple of Exception type and error message (regular expression) '
            f'to match. Encountered: {error_type!r}.'
        )
      error_type, regex = error_type
    if not (inspect.isclass(error_type) and issubclass(error_type, Exception)):
      raise TypeError(f'Exception contains non-except types: {error_type!r}.')
    if error_type not in error_mapping:
      error_mapping[error_type] = []
    if regex is not None:
      error_mapping[error_type].append(regex)

  context = CatchErrorsContext()
  try:
    yield context
  except tuple(error_mapping.keys()) as e:
    error_message = str(e)
    found_match = False
    for error_type, error_regexes in error_mapping.items():
      if isinstance(e, error_type):
        if not error_regexes:
          found_match = True
        else:
          for regex in error_regexes:
            if re.match(regex, error_message):
              found_match = True
              break
    if found_match:
      context.error = e
      if error_handler is not None:
        error_handler(e)
    else:
      raise e
