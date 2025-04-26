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
"""Logging for PyGlove.

This module allows PyGlove to use external created logger for logging PyGlove
events without introducing library dependencies in PyGlove.
"""

import inspect
import logging
from typing import Any, Callable, List, Union


_DEFAULT_LOGGER = logging.getLogger()


def register_frame_to_skip(
    method: Union[Callable[..., Any], List[Callable[..., Any]]]
) -> bool:
  """Skips the source of the given method when logging.

  Args:
    method: The method to skip. Can be a single method or a list of methods.

  Returns:
    True if the method is registered to skip.

  Raises:
    TypeError: The source file of the method cannot be inspected.
  """
  register_fn = getattr(
      _DEFAULT_LOGGER.__class__, 'register_frame_to_skip', None
  )
  if register_fn is None:
    return False
  methods = [method] if not isinstance(method, list) else method
  for m in methods:
    register_fn(inspect.getsourcefile(m), m.__name__)
  return True


def set_logger(logger: logging.Logger) -> None:
  """Sets current logger."""
  global _DEFAULT_LOGGER
  _DEFAULT_LOGGER = logger

  # Skip logging frames in pyglove.logging.
  register_frame_to_skip([debug, info, warning, error, critical])


def get_logger() -> logging.Logger:
  """Gets the current logger."""
  return _DEFAULT_LOGGER


def debug(msg: str, *args, **kwargs) -> None:
  """Logs debug message.

  Args:
    msg: Message with possible format string.
    *args: Values for variables in the format string.
    **kwargs: Keyword arguments for the logger.
  """
  _DEFAULT_LOGGER.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
  """Logs info message.

  Args:
    msg: Message with possible format string.
    *args: Values for variables in the format string.
    **kwargs: Keyword arguments for the logger.
  """
  _DEFAULT_LOGGER.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
  """Logs warning message.

  Args:
    msg: Message with possible format string.
    *args: Values for variables in the format string.
    **kwargs: Keyword arguments for the logger.
  """
  _DEFAULT_LOGGER.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
  """Logs error message.

  Args:
    msg: Message with possible format string.
    *args: Values for variables in the format string.
    **kwargs: Keyword arguments for the logger.
  """
  _DEFAULT_LOGGER.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
  """Logs critical message.

  Args:
    msg: Message with possible format string.
    *args: Values for variables in the format string.
    **kwargs: Keyword arguments for the logger.
  """
  _DEFAULT_LOGGER.critical(msg, *args, **kwargs)
