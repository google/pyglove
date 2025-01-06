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
"""Tracking the origin of a symbolic object."""

import traceback
from typing import Any, Callable, List, Optional

from pyglove.core import utils
from pyglove.core.symbolic import flags


class Origin(utils.Formattable):
  """Class that represents the origin of a symbolic value.

  Origin is used for debugging the creation chain of a symbolic value, as
  well as keeping track of the factory or builder in creational design patterns.
  An `Origin` object records the source value, a string tag, and optional
  stack information on where a symbolic value is created.

  Built-in tags are '__init__', 'clone', 'deepclone' and 'return'.
  Users can pass custom tags to the `sym_setorigin` method of a symbolic value
  for tracking its source in their own scenarios.

  When origin tracking is enabled by calling `pg.track_origin(True)`, the
  `sym_setorigin` method of symbolic values will be automatically called during
  object creation, cloning or being returned from a functor. The stack
  information can be obtained by `origin.stack` or `origin.stacktrace`.
  """

  def __init__(self,
               source: Any,
               tag: str,
               stacktrace: Optional[bool] = None,
               stacklimit: Optional[int] = None,
               stacktop: int = -1):
    """Constructor.

    Args:
      source: Source value for the origin.
      tag: A descriptive tag of the origin. Built-in tags are:
        '__init__', 'clone', 'deepclone', 'return'. Users can manually
        call `sym_setorigin` with custom tag value.
      stacktrace: If True, enable stack trace for the origin. If None, enable
        stack trace if `pg.tracek_origin()` is called. Otherwise stack trace is
        disabled.
      stacklimit: An optional integer to limit the stack depth. If None, it's
        determined by the value passed to `pg.set_origin_stacktrace_limit`,
        which is 10 by default.
      stacktop: A negative integer to indicate the stack top among the stack
        frames that we want to present to user, by default it's 2-level up from
        the stack within current `sym_setorigin` call.
    """
    if not isinstance(tag, str):
      raise ValueError(f'`tag` must be a string. Encountered: {tag!r}.')

    self._source = source
    self._tag = tag
    self._stack = None
    self._stacktrace = None

    if stacktrace is None:
      stacktrace = flags.is_tracking_origin()

    if stacklimit is None:
      stacklimit = flags.get_origin_stacktrace_limit()

    if stacktrace:
      self._stack = traceback.extract_stack(limit=stacklimit - stacktop)
      if stacktop < 0:
        self._stack = self._stack[:stacktop]

  @property
  def source(self) -> Any:
    """Returns the source object."""
    return self._source

  @property
  def root(self) -> 'Origin':
    """Returns the root source of the origin."""
    current = self
    while True:
      parent = getattr(current.source, 'sym_origin', None)
      if parent is None:
        break
      current = parent
    return current

  def history(
      self,
      condition: Optional[Callable[['Origin'], bool]] = None) -> List['Origin']:
    """Returns a history of origins with an optional filter.
    
    Args:
      condition: An optional callable object with signature
        (origin) -> should_list. If None, all origins will be listed.

    Returns:
      A list of filtered origin from the earliest (root) to the most recent.
    """
    condition = condition or (lambda o: True)
    current = self
    history = []
    while current is not None:
      if condition(current):
        history.append(current)
      current = getattr(current.source, 'sym_origin', None)
    history.reverse()
    return history

  @property
  def tag(self) -> str:
    """Returns tag."""
    return self._tag

  @property
  def stack(self) -> Optional[List[traceback.FrameSummary]]:
    """Returns the frame summary of original stack."""
    return self._stack

  @property
  def stacktrace(self) -> Optional[str]:
    """Returns stack trace string."""
    if self._stack is None:
      return None
    if self._stacktrace is None:
      self._stacktrace = ''.join(traceback.format_list(self._stack))
    return self._stacktrace

  def chain(self, tag: Optional[str] = None) -> List['Origin']:
    """Get the origin list from the neareast to the farthest filtered by tag."""
    origins = []
    o = self
    while o is not None:
      if tag is None or tag == o.tag:
        origins.append(o)
      o = getattr(o.source, 'sym_origin', None)
    return origins

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs,
  ) -> str:
    """Formats this object."""
    if isinstance(self._source, (str, type(None))):
      source_str = self._source
    else:
      source_info = utils.format(
          self._source, compact, verbose, root_indent + 1, **kwargs
      )
      source_str = utils.RawText(f'{source_info} at 0x{id(self._source):8x}')

    return utils.kvlist_str(
        [
            ('tag', self._tag, None),
            ('source', source_str, None),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
    )

  def __eq__(self, other: Any) -> bool:
    """Operator ==."""
    if not isinstance(other, self.__class__):
      return False
    return self._source is other.source and self._tag == other.tag

  def __ne__(self, other: Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)
