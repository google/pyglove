# Copyright 2025 The PyGlove Authors
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
"""Utility library for text coloring."""

import re
from typing import List, Optional

try:
  import termcolor      # pylint: disable=g-import-not-at-top
except ImportError:
  termcolor = None


# Regular expression for ANSI color characters.
_ANSI_COLOR_REGEX = re.compile(r'\x1b\[[0-9;]*m')


def decolor(text: str) -> str:
  """De-colors a string that may contains ANSI color characters."""
  return re.sub(_ANSI_COLOR_REGEX, '', text)


def colored(
    text: str,
    color: Optional[str] = None,
    background: Optional[str] = None,
    styles: Optional[List[str]] = None,
) -> str:
  """Returns the colored text with ANSI color characters.

  Args:
    text: A string that may or may not already has ANSI color characters.
    color: A string for text colors. Applicable values are:
      'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    background: A string for background colors. Applicable values are:
      'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    styles: A list of strings for applying styles on the text.
      Applicable values are:
      'bold', 'dark', 'underline', 'blink', 'reverse', 'concealed'.

  Returns:
    A string with ANSI color characters embracing the entire text.
  """
  if not termcolor:
    return text
  return termcolor.colored(
      text,
      color=color,
      on_color=('on_' + background) if background else None,
      attrs=styles
  )


def colored_block(
    text: str,
    block_start: str,
    block_end: str,
    color: Optional[str] = None,
    background: Optional[str] = None,
    styles: Optional[List[str]] = None,
) -> str:
  """Apply colors to text blocks.

  Args:
    text: A string that may or may not already has ANSI color characters.
    block_start: A string that signals the start of a block. E.g. '{{'
    block_end: A string that signals the end of a block. E.g. '}}'.
    color: A string for text colors. Applicable values are:
      'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    background: A string for background colors. Applicable values are:
      'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    styles: A list of strings for applying styles on the text.
      Applicable values are:
      'bold', 'dark', 'underline', 'blink', 'reverse', 'concealed'.

  Returns:
    A string with ANSI color characters embracing the matched text blocks.
  """
  if not color and not background and not styles:
    return text

  s = []
  start_index = 0
  end_index = 0
  previous_color = None

  def write_nonblock_text(text: str, previous_color: Optional[str]):
    if previous_color:
      s.append(previous_color)
    s.append(text)

  while start_index < len(text):
    start_index = text.find(block_start, end_index)
    if start_index == -1:
      write_nonblock_text(text[end_index:], previous_color)
      break

    # Deal with text since last block.
    since_last_block = text[end_index:start_index]
    write_nonblock_text(since_last_block, previous_color)
    colors = re.findall(_ANSI_COLOR_REGEX, since_last_block)
    if colors:
      previous_color = colors[-1]

    # Match block.
    end_index = text.find(block_end, start_index + len(block_start))
    if end_index == -1:
      write_nonblock_text(text[start_index:], previous_color)
      break
    end_index += len(block_end)

    # Write block text.
    block = text[start_index:end_index]
    block = colored(
        block, color=color, background=background, styles=styles)
    s.append(block)
  return ''.join(s)
