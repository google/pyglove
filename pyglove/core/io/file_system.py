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
"""Pluggable file system."""

import abc
import io
import os
from typing import Any, Literal, Optional, Union


class File(metaclass=abc.ABCMeta):
  """Interface for a file."""

  @abc.abstractmethod
  def read(self, size: Optional[int] = None) -> Union[str, bytes]:
    """Reads the content of the file."""

  @abc.abstractmethod
  def readline(self) -> Union[str, bytes]:
    """Reads the next line."""

  @abc.abstractmethod
  def write(self, content: Union[str, bytes]) -> None:
    """Writes the content of the file."""

  @abc.abstractmethod
  def seek(self, offset: int, whence: Literal[0, 1, 2] = 0) -> int:
    """Changes the current position of the file.

    Args:
      offset: Offset from the position to a reference point.
      whence: The reference point, with 0 meaning the beginning of the file,
        1 meaning the current position, or 2 meaning the end of the file.

    Returns:
      The position from the beginning of the file.
    """

  @abc.abstractmethod
  def tell(self) -> int:
    """Returns the current position of the file."""

  @abc.abstractmethod
  def flush(self) -> None:
    """Flushes the written content to the storage."""

  @abc.abstractmethod
  def close(self) -> None:
    """Closes the file."""

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback
    self.close()


class FileSystem(metaclass=abc.ABCMeta):
  """Interface for a file system."""

  @abc.abstractmethod
  def open(
      self, path: Union[str, os.PathLike[str]], mode: str = 'r', **kwargs
  ) -> File:
    """Opens a file with a path."""

  @abc.abstractmethod
  def chmod(self, path: Union[str, os.PathLike[str]], mode: int) -> None:
    """Changes the permission of a file."""

  @abc.abstractmethod
  def exists(self, path: Union[str, os.PathLike[str]]) -> bool:
    """Returns True if a path exists."""

  @abc.abstractmethod
  def listdir(self, path: Union[str, os.PathLike[str]]) -> list[str]:
    """Lists all files or sub-directories."""

  @abc.abstractmethod
  def isdir(self, path: Union[str, os.PathLike[str]]) -> bool:
    """Returns True if a path is a directory."""

  @abc.abstractmethod
  def mkdir(
      self, path: Union[str, os.PathLike[str]], mode: int = 0o777
  ) -> None:
    """Makes a directory based on a path."""

  @abc.abstractmethod
  def mkdirs(
      self,
      path: Union[str, os.PathLike[str]],
      mode: int = 0o777,
      exist_ok: bool = True,
  ) -> None:
    """Makes a directory chain based on a path."""

  @abc.abstractmethod
  def rm(self, path: Union[str, os.PathLike[str]]) -> None:
    """Removes a file based on a path."""

  @abc.abstractmethod
  def rmdir(self, path: Union[str, os.PathLike[str]]) -> bool:
    """Removes a directory based on a path."""

  @abc.abstractmethod
  def rmdirs(self, path: Union[str, os.PathLike[str]]) -> None:
    """Removes a directory chain based on a path."""


def resolve_path(path: Union[str, os.PathLike[str]]) -> str:
  if isinstance(path, str):
    return path
  elif hasattr(path, '__fspath__'):
    return path.__fspath__()
  else:
    raise ValueError(f'Unsupported path: {path!r}.')


#
# The standard file system.
#


class StdFile(File):
  """The standard file."""

  def __init__(self, file_object) -> None:
    super().__init__()
    self._file_object = file_object

  def read(self, size: Optional[int] = None) -> Union[str, bytes]:
    return self._file_object.read(size)

  def readline(self) -> Union[str, bytes]:
    return self._file_object.readline()

  def write(self, content: Union[str, bytes]) -> None:
    self._file_object.write(content)

  def seek(self, offset: int, whence: Literal[0, 1, 2] = 0) -> int:
    return self._file_object.seek(offset, whence)

  def tell(self) -> int:
    return self._file_object.tell()

  def flush(self) -> None:
    self._file_object.flush()

  def close(self) -> None:
    self._file_object.close()


class StdFileSystem(FileSystem):
  """The standard file system."""

  def open(
      self, path: Union[str, os.PathLike[str]], mode: str = 'r', **kwargs
  ) -> File:
    return StdFile(io.open(path, mode, **kwargs))

  def chmod(self, path: Union[str, os.PathLike[str]], mode: int) -> None:
    os.chmod(path, mode)

  def exists(self, path: Union[str, os.PathLike[str]]) -> bool:
    return os.path.exists(path)

  def listdir(self, path: Union[str, os.PathLike[str]]) -> list[str]:
    return os.listdir(path)

  def isdir(self, path: Union[str, os.PathLike[str]]) -> bool:
    return os.path.isdir(path)

  def mkdir(
      self, path: Union[str, os.PathLike[str]], mode: int = 0o777
  ) -> None:
    os.mkdir(path, mode)

  def mkdirs(
      self,
      path: Union[str, os.PathLike[str]],
      mode: int = 0o777,
      exist_ok: bool = True,
  ) -> None:
    os.makedirs(path, mode, exist_ok)

  def rm(self, path: Union[str, os.PathLike[str]]) -> None:
    os.remove(path)

  def rmdir(self, path: Union[str, os.PathLike[str]]) -> None:  # pytype: disable=signature-mismatch
    os.rmdir(path)

  def rmdirs(self, path: Union[str, os.PathLike[str]]) -> None:
    os.removedirs(path)


#
# Memory file system.
#


class MemoryFile(File):
  """Memory file."""

  def __init__(self, buffer: io.IOBase):
    super().__init__()
    self._buffer = buffer
    self._pos = 0

  def read(self, size: Optional[int] = None) -> Union[str, bytes]:
    return self._buffer.read(size)

  def readline(self) -> Union[str, bytes]:
    return self._buffer.readline()

  def write(self, content: Union[str, bytes]) -> None:
    self._buffer.write(content)

  def seek(self, offset: int, whence: Literal[0, 1, 2] = 0) -> int:
    return self._buffer.seek(offset, whence)

  def tell(self) -> int:
    return self._buffer.tell()

  def flush(self) -> None:
    pass

  def close(self) -> None:
    self.seek(0)


class MemoryFileSystem(FileSystem):
  """The in-memory file system."""

  def __init__(self, prefix: str = '/mem/'):
    super().__init__()
    self._root = {}
    self._prefix = prefix

  def _internal_path(self, path: Union[str, os.PathLike[str]]) -> str:
    return '/' + resolve_path(path).lstrip(self._prefix)

  def _locate(self, path: Union[str, os.PathLike[str]]) -> Any:
    current = self._root
    for x in self._internal_path(path).split('/'):
      if not x:
        continue
      if x not in current:
        return None
      current = current[x]
    return current

  def open(
      self, path: Union[str, os.PathLike[str]], mode: str = 'r', **kwargs
  ) -> File:
    file = self._locate(path)
    if isinstance(file, dict):
      raise IsADirectoryError(path)
    if 'w' in mode and file is None:
      parent_dir, name = self._parent_and_name(path)
      if isinstance(parent_dir, dict):
        buffer = io.BytesIO() if 'b' in mode else io.StringIO()
        file = MemoryFile(buffer)
        parent_dir[name] = file

    if file is None:
      raise FileNotFoundError(path)
    return file

  def chmod(self, path: Union[str, os.PathLike[str]], mode: int) -> None:
    # No-op.
    del path, mode

  def exists(self, path: Union[str, os.PathLike[str]]) -> bool:
    return self._locate(path) is not None

  def listdir(self, path: Union[str, os.PathLike[str]]) -> list[str]:
    d = self._locate(path)
    if not isinstance(d, dict):
      raise FileNotFoundError(path)
    return list(d.keys())

  def isdir(self, path: Union[str, os.PathLike[str]]) -> bool:
    return isinstance(self._locate(path), dict)

  def _parent_and_name(
      self, path: Union[str, os.PathLike[str]]
  ) -> tuple[dict[str, Any], str]:
    path = resolve_path(path)
    rpos = path.rfind('/')
    assert rpos >= 0, path
    name = path[rpos + 1:]
    parent_dir = self._locate(path[:rpos])
    if parent_dir is None:
      raise FileNotFoundError(path)
    return parent_dir, name

  def mkdir(
      self, path: Union[str, os.PathLike[str]], mode: int = 0o777
  ) -> None:
    del mode
    parent_dir, name = self._parent_and_name(path)
    if name in parent_dir:
      raise FileExistsError(path)
    parent_dir[name] = {}

  def mkdirs(
      self,
      path: Union[str, os.PathLike[str]],
      mode: int = 0o777,
      exist_ok: bool = True,
  ) -> None:
    del mode
    current = self._root
    dirs = self._internal_path(path).split('/')
    for i, x in enumerate(dirs):
      if not x:
        continue
      entry = current.get(x)
      if entry is None:
        entry = {}
        current[x] = entry
      elif isinstance(entry, dict) and i == len(dirs) - 1 and not exist_ok:
        raise FileExistsError(path)
      elif not isinstance(entry, dict):
        raise NotADirectoryError(path)
      current = entry

  def rm(self, path: Union[str, os.PathLike[str]]) -> None:
    parent_dir, name = self._parent_and_name(path)
    entry = parent_dir.get(name)
    if entry is None:
      raise FileNotFoundError(path)
    elif isinstance(entry, dict):
      raise IsADirectoryError(path)
    del parent_dir[name]

  def rmdir(self, path: Union[str, os.PathLike[str]]) -> None:  # pytype: disable=signature-mismatch
    parent_dir, name = self._parent_and_name(path)
    entry = parent_dir.get(name)
    if entry is None:
      raise FileNotFoundError(path)
    elif not isinstance(entry, dict):
      raise NotADirectoryError(path)
    elif entry:
      raise OSError(f'Directory not empty: {path!r}')
    del parent_dir[name]

  def rmdirs(self, path: Union[str, os.PathLike[str]]) -> None:
    def _rmdir(dir_dict, subpath: str) -> bool:
      if not subpath:
        if dir_dict:
          raise OSError(f'Directory not empty: {path!r}')
        return True
      subpath = subpath.lstrip('/')
      pos = subpath.find('/')
      if pos < 0:
        pos = len(subpath)
      name = subpath[:pos]
      subpath = subpath[pos + 1:] if pos < len(subpath) else ''
      if name not in dir_dict:
        raise FileNotFoundError(path)
      elif not isinstance(dir_dict[name], dict):
        raise NotADirectoryError(path)
      if _rmdir(dir_dict[name], subpath):
        del dir_dict[name]
      return not dir_dict
    _rmdir(self._root, self._internal_path(path))


class _FileSystemRegistry:
  """File system registry."""

  def __init__(self):
    self._filesystems = []

  def add(self, prefix, fs: FileSystem) -> None:
    """Gets the file system applicable for path."""
    self._filesystems.append((prefix, fs))
    self._filesystems.sort(key=lambda x: x[0], reverse=True)

  def get(self, path: Union[str, os.PathLike[str]]) -> FileSystem:
    """Gets the file system for a path."""
    path = resolve_path(path)
    for prefix, fs in self._filesystems:
      if path.startswith(prefix):
        return fs
    return StdFileSystem()


_fs = _FileSystemRegistry()


def add_file_system(prefix: str, fs: FileSystem) -> None:
  """Adds a file system with a prefix."""
  _fs.add(prefix, fs)


# Register a memory file system.
add_file_system('/mem/', MemoryFileSystem('/mem/'))


#
# APIs for file IO.
#


def open(path: Union[str, os.PathLike[str]], mode: str = 'r', **kwargs) -> File:  # pylint:disable=redefined-builtin
  """Opens a file with a path."""
  return _fs.get(path).open(path, mode, **kwargs)


def chmod(path: Union[str, os.PathLike[str]], mode: int) -> None:
  """Changes the permission of a file."""
  _fs.get(path).chmod(path, mode)


def readfile(
    path: Union[str, os.PathLike[str]],
    mode: str = 'r',
    nonexist_ok: bool = False,
    **kwargs,
) -> Union[bytes, str, None]:
  """Reads content from a file."""
  try:
    with _fs.get(path).open(path, mode=mode, **kwargs) as f:
      return f.read()
  except FileNotFoundError as e:
    if nonexist_ok:
      return None
    raise e


def writefile(
    path: Union[str, os.PathLike[str]],
    content: Union[str, bytes],
    *,
    mode: str = 'w',
    perms: Optional[int] = 0o664,   # Default to world-readable.
    **kwargs,
) -> None:
  """Writes content to a file."""
  with _fs.get(path).open(path, mode=mode, **kwargs) as f:
    f.write(content)
  if perms is not None:
    chmod(path, perms)


def rm(path: Union[str, os.PathLike[str]]) -> None:
  """Removes a file."""
  _fs.get(path).rm(path)


def path_exists(path: Union[str, os.PathLike[str]]) -> bool:
  """Returns True if path exists."""
  return _fs.get(path).exists(path)


def listdir(
    path: Union[str, os.PathLike[str]], fullpath: bool = False
) -> list[str]:  # pylint: disable=redefined-builtin
  """Lists all files or sub-directories under a dir."""
  entries = _fs.get(path).listdir(path)
  if fullpath:
    return [os.path.join(path, entry) for entry in entries]
  return entries


def isdir(path: Union[str, os.PathLike[str]]) -> bool:
  """Returns True if path is a directory."""
  return _fs.get(path).isdir(path)


def mkdir(path: Union[str, os.PathLike[str]], mode: int = 0o777) -> None:
  """Makes a directory."""
  _fs.get(path).mkdir(path, mode=mode)


def mkdirs(
    path: Union[str, os.PathLike[str]],
    mode: int = 0o777,
    exist_ok: bool = True,
) -> None:
  """Makes a directory chain."""
  _fs.get(path).mkdirs(path, mode=mode, exist_ok=exist_ok)


def rmdir(path: Union[str, os.PathLike[str]]) -> bool:
  """Removes a directory."""
  return _fs.get(path).rmdir(path)


def rmdirs(path: Union[str, os.PathLike[str]]) -> bool:
  """Removes a directory chain until a parent directory is not empty."""
  return _fs.get(path).rmdirs(path)  # pytype: disable=bad-return-type
