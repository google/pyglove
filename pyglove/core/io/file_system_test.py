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

import os
import pathlib
import tempfile
import unittest
from pyglove.core.io import file_system


class StdFileSystemTest(unittest.TestCase):

  def test_file(self):
    tmp_dir = tempfile.mkdtemp()
    fs = file_system.StdFileSystem()

    file1 = os.path.join(tmp_dir, 'file1')
    with self.assertRaises(FileNotFoundError):
      fs.open(file1)

    with fs.open(file1, 'w') as f:
      f.write('hello\npyglove')

    with fs.open(file1, 'r') as f:
      self.assertEqual(f.readline(), 'hello\n')
      self.assertEqual(f.tell(), 6)
      self.assertEqual(f.seek(8, 0), 8)
      self.assertEqual(f.read(), 'glove')

    self.assertTrue(fs.exists(file1))
    self.assertFalse(fs.exists('/not-exist-dir'))
    fs.rm(file1)
    self.assertFalse(fs.exists(file1))

  def test_file_system(self):
    tmp_dir = tempfile.mkdtemp()
    fs = file_system.StdFileSystem()

    # Create a directory.
    dir_a = os.path.join(tmp_dir, 'a')
    fs.mkdir(dir_a)
    self.assertTrue(fs.exists(dir_a))
    self.assertTrue(fs.isdir(dir_a))
    self.assertEqual(len(fs.listdir(dir_a)), 0)

    # Create a file under the directory.
    file1 = os.path.join(dir_a, 'file1')
    self.assertFalse(fs.exists(file1))
    with fs.open(file1, 'w') as f:
      f.write('hello')
    self.assertTrue(fs.exists(file1))
    self.assertFalse(fs.isdir(file1))

    fs.mkdirs(os.path.join(dir_a, 'b/c/d'))
    self.assertEqual(sorted(fs.listdir(dir_a)), ['b', 'file1'])   # pylint: disable=g-generic-assert

    # Test rm.
    with self.assertRaises(FileNotFoundError):
      fs.rm(os.path.join(dir_a, 'file2'))
    with self.assertRaises((IsADirectoryError, PermissionError)):
      fs.rm(os.path.join(dir_a, 'b'))

    # Test rmdir.
    with self.assertRaises(FileNotFoundError):
      fs.rmdir(os.path.join(dir_a, 'c'))
    with self.assertRaises(FileNotFoundError):
      fs.rmdir(os.path.join(dir_a, 'file2'))
    fs.rmdir(os.path.join(dir_a, 'b/c/d'))

    # Test rmdirs.
    fs.rmdirs(os.path.join(dir_a, 'b/c'))
    self.assertEqual(sorted(fs.listdir(dir_a)), ['file1'])   # pylint: disable=g-generic-assert


class MemoryFileSystemTest(unittest.TestCase):

  def test_file(self):
    fs = file_system.MemoryFileSystem()

    file1 = os.path.join('/mem', 'file1')
    with self.assertRaises(FileNotFoundError):
      fs.open(file1)

    with self.assertRaisesRegex(ValueError, 'Unsupported path'):
      fs.open(1)

    with fs.open(file1, 'w') as f:
      f.write('hello\npyglove')

    with fs.open(file1, 'r') as f:
      self.assertEqual(f.readline(), 'hello\n')
      self.assertEqual(f.tell(), 6)
      self.assertEqual(f.seek(8, 0), 8)
      self.assertEqual(f.read(), 'glove')

    self.assertTrue(fs.exists(file1))
    self.assertFalse(fs.exists('/not-exist-dir'))
    fs.rm(file1)
    self.assertFalse(fs.exists(file1))

  def test_file_system(self):
    fs = file_system.MemoryFileSystem()

    # Create a directory.
    dir_a = os.path.join('/mem', 'a')
    fs.mkdir(dir_a)
    self.assertTrue(fs.exists(dir_a))
    self.assertTrue(fs.isdir(dir_a))
    self.assertEqual(len(fs.listdir(dir_a)), 0)

    with self.assertRaises(FileExistsError):
      fs.mkdir(dir_a)

    with self.assertRaises(FileNotFoundError):
      fs.mkdir('/mem/b/c')

    with self.assertRaises(IsADirectoryError):
      fs.open(dir_a)

    # Create a file under the directory.
    file1 = os.path.join(dir_a, 'file1')
    self.assertFalse(fs.exists(file1))
    with fs.open(file1, 'w') as f:
      f.write('hello')
      f.flush()
    self.assertTrue(fs.exists(file1))
    self.assertFalse(fs.isdir(file1))

    # Make dirs.
    fs.mkdirs(os.path.join(dir_a, 'b/c/d'))

    fs.mkdirs(os.path.join(dir_a, 'b/c/d'))
    with self.assertRaises(FileExistsError):
      fs.mkdirs(os.path.join(dir_a, 'b/c/d'), exist_ok=False)
    with self.assertRaises(NotADirectoryError):
      fs.mkdirs(os.path.join(file1, 'e'))

    with self.assertRaises(FileNotFoundError):
      fs.listdir('/mem/not-exist')
    self.assertEqual(sorted(fs.listdir(dir_a)), ['b', 'file1'])   # pylint: disable=g-generic-assert

    # Test rm.
    with self.assertRaises(FileNotFoundError):
      fs.rm(os.path.join(dir_a, 'file2'))
    with self.assertRaises(IsADirectoryError):
      fs.rm(os.path.join(dir_a, 'b'))

    # Test rmdir.
    with self.assertRaisesRegex(OSError, 'Directory not empty'):
      fs.rmdir(dir_a)
    with self.assertRaises(FileNotFoundError):
      fs.rmdir(os.path.join(dir_a, 'c'))
    with self.assertRaises(FileNotFoundError):
      fs.rmdir(os.path.join(dir_a, 'file2'))
    with self.assertRaises(NotADirectoryError):
      fs.rmdir(os.path.join(dir_a, 'file1'))
    fs.rmdir(os.path.join(dir_a, 'b/c/d'))

    # Test rmdirs.
    with self.assertRaisesRegex(OSError, 'Directory not empty'):
      fs.rmdirs(dir_a)

    with self.assertRaises(NotADirectoryError):
      fs.rmdirs(os.path.join(file1, 'a'))

    with self.assertRaises(FileNotFoundError):
      fs.rmdirs('a/b/d')

    fs.rmdirs(os.path.join(dir_a, 'b/c'))
    self.assertEqual(fs.listdir(dir_a), ['file1'])   # pylint: disable=g-generic-assert


class FileIoApiTest(unittest.TestCase):

  def test_standard_filesystem(self):
    file1 = os.path.join(tempfile.mkdtemp(), 'file1')
    with self.assertRaises(FileNotFoundError):
      file_system.readfile(file1)
    self.assertIsNone(file_system.readfile(file1, nonexist_ok=True))
    with file_system.open(file1, 'w') as f:
      self.assertIsInstance(f, file_system.StdFile)
      f.write('foo')
    self.assertTrue(file_system.path_exists(file1))
    self.assertEqual(file_system.readfile(file1), 'foo')
    file_system.writefile(file1, 'bar')
    self.assertEqual(file_system.readfile(file1), 'bar')
    self.assertFalse(file_system.isdir(file1))
    file_system.rm(file1)
    self.assertFalse(file_system.path_exists(file1))

    dir1 = os.path.join(tempfile.mkdtemp(), 'dir1')
    file_system.mkdir(dir1)
    self.assertEqual(file_system.listdir(dir1), [])
    file_system.mkdirs(os.path.join(dir1, 'a/b/c'))
    file2 = os.path.join(dir1, 'file2')
    file_system.writefile(file2, 'baz')
    self.assertEqual(sorted(file_system.listdir(dir1)), ['a', 'file2'])  # pylint: disable=g-generic-assert
    self.assertEqual(    # pylint: disable=g-generic-assert
        sorted(file_system.listdir(dir1, fullpath=True)),
        [os.path.join(dir1, 'a'), os.path.join(dir1, 'file2')]
    )
    file_system.rmdir(os.path.join(dir1, 'a/b/c'))
    file_system.rmdirs(os.path.join(dir1, 'a/b'))
    self.assertTrue(file_system.path_exists(dir1))
    self.assertEqual(sorted(file_system.listdir(dir1)), ['file2'])  # pylint: disable=g-generic-assert
    file_system.rm(file2)
    self.assertFalse(file_system.path_exists(file2))

  def test_memory_filesystem(self):
    file1 = pathlib.Path('/mem/file1')
    with self.assertRaises(FileNotFoundError):
      file_system.readfile(file1)
    self.assertIsNone(file_system.readfile(file1, nonexist_ok=True))
    with file_system.open(file1, 'w') as f:
      self.assertIsInstance(f, file_system.MemoryFile)
      f.write('foo')
      f.flush()
    self.assertTrue(file_system.path_exists(file1))
    self.assertEqual(file_system.readfile(file1), 'foo')
    file_system.writefile(file1, 'bar')
    self.assertEqual(file_system.readfile(file1), 'bar')
    self.assertFalse(file_system.isdir(file1))
    file_system.rm(file1)
    self.assertFalse(file_system.path_exists(file1))

    dir1 = os.path.join('/mem/dir1')
    file_system.mkdir(dir1)
    self.assertEqual(file_system.listdir(dir1), [])
    file_system.mkdirs(os.path.join(dir1, 'a/b/c'))
    file2 = os.path.join(dir1, 'file2')
    file_system.writefile(file2, 'baz')
    self.assertEqual(sorted(file_system.listdir(dir1)), ['a', 'file2'])  # pylint: disable=g-generic-assert
    file_system.rmdir(os.path.join(dir1, 'a/b/c'))
    file_system.rmdirs(os.path.join(dir1, 'a/b'))
    self.assertTrue(file_system.path_exists(dir1))
    self.assertEqual(sorted(file_system.listdir(dir1)), ['file2'])  # pylint: disable=g-generic-assert
    file_system.rm(file2)
    self.assertFalse(file_system.path_exists(file2))


if __name__ == '__main__':
  unittest.main()
