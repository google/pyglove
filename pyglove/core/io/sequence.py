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
"""Pluggable record IO."""

import abc
import collections
import os
import re
from typing import Any, Callable, Iterator, Optional, Union

from pyglove.core.io import file_system


class Sequence(metaclass=abc.ABCMeta):
  """Interface for a sequence of records."""

  def __init__(
      self,
      perms: Optional[int] = None,
      serializer: Optional[Callable[[Any], Union[bytes, str]]] = None,
      deserializer: Optional[Callable[[Union[bytes, str]], Any]] = None
  ):
    self._perms = perms
    self._serializer = serializer
    self._deserializer = deserializer

  def add(self, record: Any) -> None:
    """Adds a record to the reader."""
    if self._serializer:
      record = self._serializer(record)
    if not isinstance(record, (str, bytes)):
      raise ValueError(
          f'Cannot write record with type {type(record)}. '
          'Did you forget to pass a serializer?'
      )
    self._add(record)

  @abc.abstractmethod
  def _add(self, record: Union[bytes, str]) -> None:
    """Adds a raw record to the reader."""

  @abc.abstractmethod
  def __len__(self):
    """Gets the number of records in the reader."""

  def __iter__(self) -> Iterator[Any]:
    """Iterates over the records in the reader."""
    for record in self._iter():
      if self._deserializer:
        yield self._deserializer(record)
      else:
        yield record

  @abc.abstractmethod
  def _iter(self) -> Iterator[Union[bytes, str]]:
    """Iterates over the raw records in the reader."""

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the reader."""

  @abc.abstractmethod
  def flush(self) -> None:
    """Flushes the read records to the storage."""

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback
    self.close()


class SequenceIO(metaclass=abc.ABCMeta):
  """Interface for a record IO system."""

  @abc.abstractmethod
  def open(
      self,
      path: Union[str, os.PathLike[str]],
      mode: str,
      *,
      perms: Optional[int],
      serializer: Optional[Callable[[Any], Union[bytes, str]]],
      deserializer: Optional[Callable[[Union[bytes, str]], Any]],
      **kwargs
  ) -> Sequence:
    """Opens a sequence for reading or writing."""


_SHARDED_FILE_EXT = re.compile(r'(.*)-(\d+)-of-(\d+)$')


class _SequenceIORegistry(object):
  """Registry for record IO systems."""

  def __init__(self):
    self._registry = {}

  def add(self, extension: str, sequence_io: SequenceIO) -> None:
    """Adds a record IO system with a prefix."""
    self._registry[extension] = sequence_io

  def get(self, path: Union[str, os.PathLike[str]]) -> SequenceIO:
    """Gets the record IO system for a path."""
    path = file_system.resolve_path(path)
    parts = path.split('.')
    if parts:
      extension = parts[-1].lower()
      if '@' in extension:
        extension = extension.split('@')[0]
      match = _SHARDED_FILE_EXT.match(extension)
      if match:
        extension = match.group(1)
      if extension in self._registry:
        return self._registry[extension]
    return LineSequenceIO()


_registry = _SequenceIORegistry()


def add_sequence_io(extension: str, sequence_io: SequenceIO) -> None:
  """Adds a record IO system with a prefix."""
  _registry.add(extension, sequence_io)


def open_sequence(
    path: Union[str, os.PathLike[str]],
    mode: str = 'r',
    *,
    perms: Optional[int] = 0o664,  # Default to world-readable.
    serializer: Optional[
        Callable[[Any], Union[bytes, str]]
    ] = None,
    deserializer: Optional[
        Callable[[Union[bytes, str]], Any]
    ] = None,
    make_dirs_if_not_exist: bool = True,
) -> Sequence:
  """Open sequence for reading or writing.

  Args:
    path: The path to the sequence.
    mode: The mode of the sequence.
    perms: (Optional) The permissions of the sequence.
    serializer: (Optional) A serializer function for converting a structured
      object to a string or bytes.
    deserializer: (Optional) A deserializer function for converting a string or
      bytes to a structured object.
    make_dirs_if_not_exist: (Optional) Whether to create the directories
      if they do not exist. Applicable when opening in write or append mode.

  Returns:
    A sequence for reading or writing.
  """
  if 'w' in mode or 'a' in mode:
    parent_dir = os.path.dirname(path)
    if make_dirs_if_not_exist:
      file_system.mkdirs(parent_dir, exist_ok=True)
  return _registry.get(path).open(
      path, mode, perms=perms, serializer=serializer, deserializer=deserializer
  )


class MemorySequence(Sequence):
  """An in-memory sequence."""

  def __init__(
      self,
      path: str,
      mode: str,
      records: list[Union[str, bytes]],
      *,
      perms: Optional[int],
      serializer: Optional[Callable[[Any], Union[bytes, str]]],
      deserializer: Optional[Callable[[Union[bytes, str]], Any]]
  ):
    super().__init__(perms, serializer, deserializer)
    self._path = path
    self._mode = mode
    self._records = records
    self._closed = False

  def _add(self, record: Union[str, bytes]) -> None:
    if 'w' not in self._mode and 'a' not in self._mode:
      raise ValueError(
          f'Cannot write record {record!r} to memory sequence {self._path!r} '
          f'with mode {self._mode!r}.'
      )
    if self._closed:
      raise ValueError(
          f'Cannot write record {record!r} to a closed writer for '
          f'{self._path!r}.'
      )
    self._records.append(record)

  def __len__(self):
    return len(self._records)

  def _iter(self):
    if 'r' not in self._mode:
      raise ValueError(
          f'Cannot read memory sequence {self._path!r} with '
          f'mode {self._mode!r}.'
      )
    if self._closed:
      raise ValueError(
          f'Cannot iterate over a closed sequence reader {self._path!r}.'
      )
    return iter(self._records)

  def flush(self):
    pass

  def close(self) -> None:
    self._closed = True


class MemorySequenceIO(SequenceIO):
  """Memory-based record IO."""

  def __init__(self):
    super().__init__()
    self._root = collections.defaultdict(list)

  def open(
      self,
      path: Union[str, os.PathLike[str]],
      mode: str,
      *,
      perms: Optional[int],
      serializer: Optional[Callable[[Any], Union[bytes, str]]],
      deserializer: Optional[Callable[[Union[bytes, str]], Any]],
      **kwargs
  ) -> Sequence:
    """Opens a reader for a sequence."""
    del kwargs
    if 'w' in mode:
      self._root[path] = []
    return MemorySequence(
        path, mode, self._root[path],
        perms=perms, serializer=serializer, deserializer=deserializer
    )


add_sequence_io('mem', MemorySequenceIO())


class LineSequence(Sequence):
  """A new-line broken sequence."""

  def __init__(
      self,
      path: str,
      mode: str,
      perms: Optional[int],
      serializer: Optional[Callable[[Any], Union[bytes, str]]],
      deserializer: Optional[Callable[[Union[bytes, str]], Any]],
  ) -> None:
    super().__init__(perms, serializer, deserializer)
    self._path = path
    self._mode = mode
    self._file = file_system.open(path, mode)

  def __len__(self):
    raise NotImplementedError(
        '__len__ is not supported for LineSequence. '
        'Use `len(list(iter(sequence)))` instead.'
    )

  def _iter(self):
    while True:
      line = self._file.readline()
      if not line:
        break
      yield line.rstrip('\n')

  def _add(self, record: Union[str, bytes]) -> None:
    self._file.write(record.rstrip('\n'))
    self._file.write('\n')

  def flush(self) -> None:
    self._file.flush()

  def close(self) -> None:
    self._file.close()
    if ('w' in self._mode or 'a' in self._mode) and self._perms is not None:
      file_system.chmod(self._path, self._perms)


class LineSequenceIO(SequenceIO):
  """Line-based record IO."""

  def open(
      self,
      path: Union[str, os.PathLike[str]],
      mode: str,
      *,
      perms: Optional[int],
      serializer: Optional[Callable[[Any], Union[bytes, str]]],
      deserializer: Optional[Callable[[Union[bytes, str]], Any]],
      **kwargs
  ) -> Sequence:
    """Opens a reader for a sequence."""
    del kwargs
    return LineSequence(path, mode, perms, serializer, deserializer)

