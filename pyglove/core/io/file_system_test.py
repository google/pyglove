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

  def test_rename(self):
    tmp_dir = tempfile.mkdtemp()
    fs = file_system.StdFileSystem()

    _ = fs.mkdirs(os.path.join(tmp_dir, 'a/b'))
    file_foo = os.path.join(tmp_dir, 'a/foo.txt')
    file_bar = os.path.join(tmp_dir, 'a/bar.txt')

    with fs.open(file_foo, 'w') as f:
      f.write('foo')
    with fs.open(file_bar, 'w') as f:
      f.write('bar')

    # Rename file to a new name.
    file_foo_new = os.path.join(tmp_dir, 'a/foo-new.txt')
    fs.rename(file_foo, file_foo_new)
    self.assertFalse(fs.exists(file_foo))
    self.assertTrue(fs.exists(file_foo_new))

    # Rename file to an existing file name.
    fs.rename(file_foo_new, file_bar)
    self.assertFalse(fs.exists(file_foo_new))
    with fs.open(file_bar, 'r') as f:
      self.assertEqual(f.read(), 'foo')

    # Rename directory to a new name.
    dir_b = os.path.join(tmp_dir, 'a/b')
    dir_c = os.path.join(tmp_dir, 'a/c')
    fs.rename(dir_b, dir_c)
    self.assertFalse(fs.exists(dir_b))
    self.assertTrue(fs.exists(dir_c))
    self.assertTrue(fs.isdir(dir_c))

    # Rename directory to an existing empty directory.
    dir_d = os.path.join(tmp_dir, 'a/d')
    fs.mkdirs(dir_d)
    fs.rename(dir_c, dir_d)
    self.assertFalse(fs.exists(dir_c))
    self.assertTrue(fs.exists(dir_d))

    # Rename directory to a non-empty directory.
    dir_x = os.path.join(tmp_dir, 'x')
    dir_a = os.path.join(tmp_dir, 'a')
    fs.mkdirs(os.path.join(dir_x, 'y'))
    with self.assertRaises(OSError):
      fs.rename(dir_a, dir_x)
    self.assertTrue(fs.exists(dir_a))
    self.assertTrue(fs.exists(os.path.join(dir_x, 'y')))

    # Errors
    dir_u = os.path.join(tmp_dir, 'u')
    dir_u_v = os.path.join(dir_u, 'v')
    file_u_a = os.path.join(dir_u, 'a.txt')
    fs.mkdirs(dir_u_v)
    with fs.open(file_u_a, 'w') as f:
      f.write('a')

    with self.assertRaises((OSError, NotADirectoryError)):
      fs.rename(dir_u, file_u_a)

    with self.assertRaises(IsADirectoryError):
      fs.rename(file_u_a, dir_u_v)

    with self.assertRaises(FileNotFoundError):
      fs.rename(
          os.path.join(tmp_dir, 'non-existent'),
          os.path.join(tmp_dir, 'y')
      )


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

  def test_glob(self):
    fs = file_system.MemoryFileSystem()
    fs.mkdirs('/mem/a/b/c')
    with fs.open('/mem/a/foo.txt', 'w') as f:
      f.write('foo')
    with fs.open('/mem/a/bar.json', 'w') as f:
      f.write('bar')
    with fs.open('/mem/a/b/baz.txt', 'w') as f:
      f.write('baz')

    self.assertEqual(
        sorted(fs.glob('/mem/a/*')),
        ['/mem/a/b', '/mem/a/b/baz.txt', '/mem/a/b/c',
         '/mem/a/bar.json', '/mem/a/foo.txt'])
    self.assertEqual(
        sorted(fs.glob('/mem/a/*.txt')),
        ['/mem/a/b/baz.txt', '/mem/a/foo.txt'])
    self.assertEqual(
        sorted(fs.glob('/mem/a/b/*')),
        ['/mem/a/b/baz.txt', '/mem/a/b/c'])
    self.assertEqual(fs.glob('/mem/a/b/*.txt'), ['/mem/a/b/baz.txt'])
    self.assertEqual(fs.glob('/mem/a/b/c/*'), [])
    self.assertEqual(fs.glob('/mem/a/???.txt'), ['/mem/a/foo.txt'])
    self.assertEqual(fs.glob('/mem/a/bar.*'), ['/mem/a/bar.json'])
    self.assertEqual(
        sorted(fs.glob('/mem/a/*.*')),
        ['/mem/a/b/baz.txt', '/mem/a/bar.json', '/mem/a/foo.txt'])

  def test_rename(self):
    fs = file_system.MemoryFileSystem()
    fs.mkdirs('/mem/a/b')
    with fs.open('/mem/a/foo.txt', 'w') as f:
      f.write('foo')
    with fs.open('/mem/a/bar.txt', 'w') as f:
      f.write('bar')

    # Rename file to a new name.
    fs.rename('/mem/a/foo.txt', '/mem/a/foo-new.txt')
    self.assertFalse(fs.exists('/mem/a/foo.txt'))
    self.assertTrue(fs.exists('/mem/a/foo-new.txt'))

    # Rename file to an existing file name.
    fs.rename('/mem/a/foo-new.txt', '/mem/a/bar.txt')
    self.assertFalse(fs.exists('/mem/a/foo-new.txt'))
    with fs.open('/mem/a/bar.txt', 'r') as f:
      self.assertEqual(f.read(), 'foo')

    # Rename directory to a new name.
    fs.rename('/mem/a/b', '/mem/a/c')
    self.assertFalse(fs.exists('/mem/a/b'))
    self.assertTrue(fs.exists('/mem/a/c'))
    self.assertTrue(fs.isdir('/mem/a/c'))

    # Rename directory to an existing empty directory.
    fs.mkdirs('/mem/a/d')
    fs.rename('/mem/a/c', '/mem/a/d')
    self.assertFalse(fs.exists('/mem/a/c'))
    self.assertTrue(fs.exists('/mem/a/d'))

    # Rename directory to a non-empty directory.
    fs.mkdirs('/mem/x/y')
    with self.assertRaisesRegex(OSError, "Directory not empty: '/mem/x'"):
      fs.rename('/mem/a', '/mem/x')
    self.assertTrue(fs.exists('/mem/a'))
    self.assertTrue(fs.exists('/mem/x/y'))

    # Errors
    fs.mkdirs('/mem/u/v')
    with fs.open('/mem/u/a.txt', 'w') as f:
      f.write('a')

    with self.assertRaisesRegex(
        OSError, "Cannot move directory '/mem/u' to a subdirectory of itself"):
      fs.rename('/mem/u', '/mem/u/v/w')

    with self.assertRaisesRegex(
        NotADirectoryError,
        "Cannot rename directory '/mem/u' to non-directory '/mem/u/a.txt'"):
      fs.rename('/mem/u', '/mem/u/a.txt')

    with self.assertRaisesRegex(
        IsADirectoryError,
        "Cannot rename non-directory '/mem/u/a.txt' to directory '/mem/u/v'"):
      fs.rename('/mem/u/a.txt', '/mem/u/v')

    with self.assertRaises(FileNotFoundError):
      fs.rename('/mem/non-existent', '/mem/y')


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

    # Test glob with standard file system.
    glob_dir = os.path.join(tempfile.mkdtemp(), 'glob')
    file_system.mkdirs(os.path.join(glob_dir, 'a/b'))
    file_system.writefile(os.path.join(glob_dir, 'a/foo.txt'), 'foo')
    self.assertEqual(
        sorted(file_system.glob(os.path.join(glob_dir, 'a/*'))),
        [os.path.join(glob_dir, 'a/b'), os.path.join(glob_dir, 'a/foo.txt')])

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

    # Test glob with memory file system.
    file_system.mkdirs('/mem/g/a/b')
    file_system.writefile('/mem/g/a/foo.txt', 'foo')
    file_system.rename('/mem/g/a/foo.txt', '/mem/g/a/foo2.txt')
    file_system.writefile('/mem/g/a/b/bar.txt', 'bar')
    self.assertEqual(
        sorted(file_system.glob('/mem/g/a/*')),
        ['/mem/g/a/b', '/mem/g/a/b/bar.txt', '/mem/g/a/foo2.txt'])


if __name__ == '__main__':
  unittest.main()
