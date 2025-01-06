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
"""Utilities for formatting objects."""

import abc
import enum
import io
import sys
from typing import Any, Callable, ContextManager, Dict, List, Optional, Sequence, Set, Tuple
from pyglove.core.utils import thread_local


_TLS_STR_FORMAT_KWARGS = '_str_format_kwargs'
_TLS_REPR_FORMAT_KWARGS = '_repr_format_kwargs'


CustomFormatFn = Callable[
    [
        Any,    # Value to format.
        int,    # Root indent.
    ],
    # Returns a string or None. If None, the default `pg.format` will be used.
    Optional[str]
]


def str_format(**kwargs) -> ContextManager[Dict[str, Any]]:
  """Context manager for setting the default format kwargs for __str__."""
  return thread_local.thread_local_arg_scope(_TLS_STR_FORMAT_KWARGS, **kwargs)


def repr_format(**kwargs) -> ContextManager[Dict[str, Any]]:
  """Context manager for setting the default format kwargs for __repr__."""
  return thread_local.thread_local_arg_scope(_TLS_REPR_FORMAT_KWARGS, **kwargs)


class Formattable(metaclass=abc.ABCMeta):
  """Interface for classes whose instances can be pretty-formatted.

  This interface overrides the default ``__repr__`` and ``__str__`` method, thus
  all ``Formattable`` objects can be printed nicely.

  All symbolic types implement this interface.
  """

  # Additional format keyword arguments for `__str__`.
  __str_format_kwargs__ = dict(compact=False, verbose=True)

  # Additional format keyword arguments for `__repr__`.
  __repr_format_kwargs__ = dict(compact=True)

  @abc.abstractmethod
  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> str:
    """Formats this object into a string representation.

    Args:
      compact: If True, this object will be formatted into a single line.
      verbose: If True, this object will be formatted with verbosity.
        Subclasses should define `verbosity` on their own.
      root_indent: The start indent level for this object if the output is a
        multi-line string.
      **kwargs: Subclass specific keyword arguments.

    Returns:
      A string of formatted object.
    """

  def __str__(self) -> str:
    """Returns the full (maybe multi-line) representation of this object."""
    # NOTE(daiyip): we delegate the formatting logic to `pg.format` instead of
    # `Formattable.format` as `pg.format` could add common functionalities such
    # as `markdown` support.
    return format(self, **self.__str_kwargs__())

  def __str_kwargs__(self) -> Dict[str, Any]:
    """Returns the default format kwargs for __str__."""
    kwargs = dict(self.__str_format_kwargs__)
    kwargs.update(thread_local.thread_local_kwargs(_TLS_STR_FORMAT_KWARGS))
    return kwargs

  def __repr__(self) -> str:
    """Returns a single-line representation of this object."""
    # NOTE(daiyip): we delegate the formatting logic to `pg.format` instead of
    # `Formattable.format` as `pg.format` could add common functionalities such
    # as `markdown` support.
    return format(self, **self.__repr_kwargs__())

  def __repr_kwargs__(self) -> Dict[str, Any]:
    """Returns the default format kwargs for __repr__."""
    kwargs = dict(self.__repr_format_kwargs__)
    kwargs.update(thread_local.thread_local_kwargs(_TLS_REPR_FORMAT_KWARGS))
    return kwargs


class RawText(Formattable):
  """Raw text."""

  def __init__(self, text: str):
    self.text = text

  def format(self, *args, **kwargs):
    del args, kwargs
    return self.text

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, RawText):
      return self.text == other.text
    elif isinstance(other, str):
      return self.text == other
    return False

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)


class BracketType(enum.IntEnum):
  """Bracket types used for complex type formatting."""
  # Round bracket.
  ROUND = 0

  # Square bracket.
  SQUARE = 1

  # Curly bracket.
  CURLY = 2


_BRACKET_CHARS = [
    ('(', ')'),
    ('[', ']'),
    ('{', '}'),
]


def bracket_chars(bracket_type: BracketType) -> Tuple[str, str]:
  """Gets bracket character."""
  return _BRACKET_CHARS[int(bracket_type)]


def kvlist_str(
    kvlist: List[Tuple[str, Any, Any]],
    compact: bool = True,
    verbose: bool = False,
    root_indent: int = 0,
    *,
    label: Optional[str] = None,
    bracket_type: BracketType = BracketType.ROUND,
    custom_format: Optional[CustomFormatFn] = None,
    **kwargs,
) -> str:
  """Formats a list key/value pairs into a comma delimited string.

  Args:
    kvlist: List of tuples in format of
      (key, value, default_value or a tuple of default values)
    compact: If True, format value in kvlist in compact form.
    verbose: If True, format value in kvlist in verbose.
    root_indent: The indent should be applied for values in kvlist if they are
      multi-line.
    label: (Optional) If not None, add label to brace all kv pairs.
    bracket_type: Bracket type used for embracing the kv pairs. Applicable only
      when `name` is not None.
    custom_format: An optional custom format function, which will be applied to
      each value (and child values) in kvlist. If the function returns None, it
      will fall back to the default `pg.format`.
    **kwargs: Keyword arguments that will be passed through unto child
      ``Formattable`` objects.
  Returns:
    A formatted string from a list of key/value pairs delimited by comma.
  """
  s = io.StringIO()
  is_first = True
  bracket_start, bracket_end = bracket_chars(bracket_type)

  child_indent = (root_indent + 1) if label else root_indent
  body = io.StringIO()

  for k, v, d in kvlist:
    if isinstance(d, tuple):
      include_pair = True
      for sd in d:
        if sd == v:
          include_pair = False
          break
    else:
      include_pair = v != d
    if include_pair:
      if not is_first:
        body.write(',')
        body.write(' ' if compact else '\n')
      v = format(
          v, compact=compact,
          verbose=verbose,
          root_indent=child_indent,
          custom_format=custom_format,
          **kwargs
      )
      if not compact:
        body.write(_indent('', child_indent))
      if k:
        body.write(f'{k}={str_ext(v, custom_format, child_indent)}')
      else:
        body.write(str_ext(v, custom_format, child_indent))
      is_first = False

  if label and not is_first and not compact:
    body.write('\n')

  body = body.getvalue()
  if label is None:
    return body
  else:
    s.write(label)
    s.write(bracket_start)
    if body:
      if not compact:
        s.write('\n')
      s.write(body)
      if not compact:
        s.write(_indent('', root_indent))
    s.write(bracket_end)
    return s.getvalue()


def quote_if_str(value: Any) -> Any:
  """Quotes the value if it is a str."""
  if isinstance(value, str):
    return repr(value)
  return value


def comma_delimited_str(value_list: Sequence[Any]) -> str:
  """Gets comma delimited string."""
  return ', '.join(str(quote_if_str(v)) for v in value_list)


def auto_plural(
    number: int, singular: str, plural: Optional[str] = None) -> str:
  """Use singular form if number is 1, otherwise use plural form."""
  if plural is None:
    plural = singular + 's'
  return singular if number == 1 else plural


def format(   # pylint: disable=redefined-builtin
    value: Any,
    compact: bool = False,
    verbose: bool = True,
    root_indent: int = 0,
    list_wrap_threshold: int = 80,
    strip_object_id: bool = False,
    include_keys: Optional[Set[str]] = None,
    exclude_keys: Optional[Set[str]] = None,
    markdown: bool = False,
    max_str_len: Optional[int] = None,
    max_bytes_len: Optional[int] = None,
    *,
    custom_format: Optional[CustomFormatFn] = None,
    **kwargs,
) -> str:
  """Formats a (maybe) hierarchical value with flags.

  Args:
    value: The value to format.
    compact: If True, this object will be formatted into a single line.
    verbose: If True, this object will be formatted with verbosity. Subclasses
      should define `verbosity` on their own.
    root_indent: The start indent level for this object if the output is a
      multi-line string.
    list_wrap_threshold: A threshold in number of characters for wrapping a list
      value in a single line.
    strip_object_id: If True, format object as '<class-name>(...)' other than
      'object at <address>'.
    include_keys: A set of keys to include from the top-level dict or object.
    exclude_keys: A set of keys to exclude from the top-level dict or object.
      Applicable only when `include_keys` is set to None.
    markdown: If True, use markdown notion to quote the formatted object.
    max_str_len: The max length of the string to be formatted. If the string is
      longer than this length, it will be truncated.
    max_bytes_len: The max length of the bytes to be formatted. If the bytes is
      longer than this length, it will be truncated.
    custom_format: An optional custom format function, which will be applied to
      each value (and child values) in kvlist. If the function returns None, it
      will fall back to the default `pg.format`.
    **kwargs: Keyword arguments that will be passed through unto child
      ``Formattable`` objects.

  Returns:
    A string representation for `value`.
  """
  # We allow custom_format to intercept the entire value if it's present.
  if custom_format is not None:
    result = custom_format(value, root_indent)
    if result is not None:
      return maybe_markdown_quote(result, markdown)

  exclude_keys = exclude_keys or set()

  def _should_include_key(key: str) -> bool:
    if include_keys:
      return key in include_keys
    return key not in exclude_keys

  def _format_child(v):
    return format(
        v,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent + 1,
        list_wrap_threshold=list_wrap_threshold,
        strip_object_id=strip_object_id,
        max_str_len=max_str_len,
        max_bytes_len=max_bytes_len,
        custom_format=custom_format,
        **kwargs
    )

  # `markdown` is only applied at the outter most level,
  # so we disable markdown format in the sub-tree for `str` and `repr`.
  with str_format(markdown=False), repr_format(markdown=False):
    if isinstance(value, Formattable):
      s = value.format(
          compact=compact,
          verbose=verbose,
          root_indent=root_indent,
          list_wrap_threshold=list_wrap_threshold,
          strip_object_id=strip_object_id,
          include_keys=include_keys,
          exclude_keys=exclude_keys,
          max_str_len=max_str_len,
          max_bytes_len=max_bytes_len,
          custom_format=custom_format,
          **kwargs
      )
    elif isinstance(value, (list, tuple)):
      # Always try compact representation if length is not too long.
      open_bracket, close_bracket = bracket_chars(
          BracketType.SQUARE if isinstance(value, list) else BracketType.ROUND)
      s = [open_bracket]
      s.append(', '.join([_format_child(elem) for elem in value]))
      s.append(close_bracket)
      s = [''.join(s)]
      if not compact and len(s[-1]) > list_wrap_threshold:
        s = [f'{open_bracket}\n']
        s.append(',\n'.join([
            _indent(_format_child(elem), root_indent + 1)
            for elem in value
        ]))
        s.append('\n')
        s.append(_indent(close_bracket, root_indent))
    elif isinstance(value, dict):
      if compact or not value:
        s = ['{']
        s.append(', '.join([
            f'{k!r}: {_format_child(v)}'
            for k, v in value.items() if _should_include_key(k)
        ]))
        s.append('}')
      else:
        s = ['{\n']
        s.append(',\n'.join([
            _indent(f'{k!r}: {_format_child(v)}', root_indent + 1)
            for k, v in value.items() if _should_include_key(k)
        ]))
        s.append('\n')
        s.append(_indent('}', root_indent))
    else:
      if isinstance(value, str):
        if max_str_len is not None and len(value) > max_str_len:
          value = value[:max_str_len] + '...'
        s = [repr_ext(value, custom_format, root_indent)]
      elif isinstance(value, bytes):
        if max_bytes_len is not None and len(value) > max_bytes_len:
          value = value[:max_bytes_len] + b'...'
        s = [repr_ext(value, custom_format, root_indent)]
      else:
        s = [repr_ext(value, custom_format, root_indent)
             if compact else str_ext(value, custom_format, root_indent)]
        if strip_object_id and 'object at 0x' in s[-1]:
          s = [f'{value.__class__.__name__}(...)']
  return maybe_markdown_quote(''.join(s), markdown)


def _maybe_custom_format(
    v: Any,
    default_fn: Callable[[Any], str],
    custom_format: Optional[CustomFormatFn],
    root_indent: int,
) -> str:
  if custom_format is None:
    return default_fn(v)
  x = custom_format(v, root_indent)
  if x is None:
    return default_fn(v)
  return x


def str_ext(
    v: Any,
    custom_format: Optional[CustomFormatFn] = None,
    root_indent: int = 0,
) -> str:
  """"str operator with special format support."""
  return _maybe_custom_format(v, str, custom_format, root_indent)


def repr_ext(
    v: Any,
    custom_format: Optional[CustomFormatFn] = None,
    root_indent: int = 0,
) -> str:
  """repr operator with special format support."""
  return _maybe_custom_format(v, repr, custom_format, root_indent)


def maybe_markdown_quote(s: str, markdown: bool = True) -> str:
  """Maybe quote the formatted string with markdown."""
  if not markdown:
    return s
  if '\n' not in s:
    return f'`{s}`'
  else:
    return f'```\n{s}\n```'


def camel_to_snake(text: str, separator: str = '_') -> str:
  """Returns the snake case version of a camel case string."""
  chunks = []
  chunk_start = 0
  last_upper = 0
  length = len(text)
  for i, c in enumerate(text):
    if c.isupper():
      if last_upper < i - 1 or (i < length - 1 and text[i + 1].islower()):
        chunks.append(text[chunk_start:i])
        chunk_start = i
      last_upper = i
  chunks.append(text[chunk_start:])
  return (separator.join(c for c in chunks if c)).lower()


def printv(v: Any, **kwargs):
  """Prints formatted value."""
  fs = kwargs.pop('file', sys.stdout)
  print(format(v, **kwargs), file=fs)


def _indent(text: str, indent: int) -> str:
  return ' ' * 2 * indent + text
