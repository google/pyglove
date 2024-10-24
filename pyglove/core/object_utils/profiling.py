# Copyright 2024 The PyGlove Authors
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
"""Utilities for profiling."""

import dataclasses
import time
from typing import Dict, List, Optional

from pyglove.core.object_utils import thread_local


class TimeIt:
  """Context manager for timing the execution of a code block."""

  @dataclasses.dataclass(frozen=True)
  class Status:
    """Status of a single `pg.timeit`."""
    name: str
    elapse: float
    has_ended: bool
    error: Optional[Exception]

    @property
    def has_started(self) -> bool:
      """Returns whether the context has started."""
      return self.elapse > 0

    @property
    def has_error(self) -> bool:
      """Returns whether the context has error."""
      return self.error is not None

  @dataclasses.dataclass
  class StatusSummary:
    """Aggregated summary for repeated calls for `pg.timeit`."""

    @dataclasses.dataclass
    class Entry:
      """Aggregated status from the `pg.timeit` calls of the same name."""

      num_started: int = 0
      num_ended: int = 0
      num_failed: int = 0
      avg_duration: float = 0.0

      def update(self, status: 'TimeIt.Status'):
        self.avg_duration = (
            (self.avg_duration * self.num_started + status.elapse)
            / (self.num_started + 1)
        )
        self.num_started += 1
        if status.has_ended:
          self.num_ended += 1
        if status.has_error:
          self.num_failed += 1

    breakdown: dict[str, 'TimeIt.StatusSummary.Entry'] = (
        dataclasses.field(default_factory=dict)
    )

    def aggregate(self, timeit_obj: 'TimeIt'):
      for k, v in timeit_obj.status().items():
        if k not in self.breakdown:
          self.breakdown[k] = TimeIt.StatusSummary.Entry()
        self.breakdown[k].update(v)

  def __init__(self, name: str):
    self._name = name
    self._start_time = None
    self._end_time = None
    self._child_contexts = {}
    self._error = None
    self._parent = None

  @property
  def name(self) -> str:
    """Returns the name of the context."""
    return self._name

  @property
  def children(self) -> List['TimeIt']:
    """Returns child contexts."""
    return list(self._child_contexts.values())

  def add(self, context: 'TimeIt'):
    """Adds a child context."""
    if context.name in self._child_contexts:
      raise ValueError(f'`timeit` with name {context.name!r} already exists.')
    self._child_contexts[context.name] = context

  def start(self):
    """Starts timing."""
    self._start_time = time.time()

  def end(self, error: Optional[BaseException] = None) -> bool:
    """Ends timing."""
    if not self.has_ended:
      self._end_time = time.time()
      self._error = error
      return True
    return False

  @property
  def has_started(self) -> bool:
    """Returns whether the context has started."""
    return self._start_time is not None

  @property
  def has_ended(self) -> bool:
    """Returns whether the context has ended."""
    return self._end_time is not None

  @property
  def start_time(self) -> Optional[float]:
    """Returns start time."""
    return self._start_time

  @property
  def end_time(self) -> Optional[float]:
    """Returns end time."""
    return self._end_time

  @property
  def error(self) -> Optional[BaseException]:
    """Returns error."""
    return self._error

  @property
  def has_error(self) -> bool:
    """Returns whether the context has error."""
    return self._error is not None

  @property
  def elapse(self) -> float:
    """Returns the elapse since start until end."""
    if self._start_time is None:
      return 0
    if self._end_time is None:
      return time.time() - self._start_time
    return self._end_time - self._start_time  # pytype: disable=unsupported-operands

  def status(self) -> Dict[str, Status]:
    """Gets the status of all `timeit` under this context."""
    result = {
        self.name: TimeIt.Status(
            name=self.name, elapse=self.elapse,
            has_ended=self.has_ended, error=self._error,
        )
    }
    for child in self._child_contexts.values():
      child_result = child.status()
      for k, v in child_result.items():
        result[f'{self.name}.{k}'] = v
    return result

  def __enter__(self):
    parent = thread_local.thread_local_get('__timing_context__', None)
    if parent is not None:
      parent.add(self)
      self._parent = parent
    thread_local.thread_local_set('__timing_context__', self)
    self.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, traceback
    self.end(exc_value)
    if self._parent is None:
      thread_local.thread_local_del('__timing_context__')
    else:
      thread_local.thread_local_set('__timing_context__', self._parent)


def timeit(name: str) -> TimeIt:
  """Context manager to time a block of code."""
  return TimeIt(name)
