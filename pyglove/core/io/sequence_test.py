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

import os
import tempfile
import unittest
from pyglove.core.io import sequence as sequence_io
# We need to import this module to register the default parser/serializer.
import pyglove.core.symbolic as pg_symbolic


class LineSequenceIOTest(unittest.TestCase):

  def test_read_write(self):
    tmp_dir = tempfile.gettempdir()
    file1 = os.path.join(tmp_dir, 'abc', 'file1')
    with pg_symbolic.open_jsonl(file1, 'w') as f:
      self.assertIsInstance(f, sequence_io.LineSequence)
      f.add(1)
      f.add(' foo')
      f.add(' bar ')
      f.flush()
      self.assertTrue(os.path.exists(file1))

      f.add('baz\n')
      f.add(dict(x=1))

    self.assertTrue(os.path.exists(file1))
    with pg_symbolic.open_jsonl(file1, 'r') as f:
      self.assertIsInstance(f, sequence_io.LineSequence)
      self.assertEqual(list(iter(f)), [1, ' foo', ' bar ', 'baz\n', dict(x=1)])

    with pg_symbolic.open_jsonl(file1, 'a') as f:
      f.add('qux')

    with pg_symbolic.open_jsonl(file1, 'r') as f:
      with self.assertRaisesRegex(
          NotImplementedError, '__len__ is not supported'
      ):
        _ = len(f)
      self.assertEqual(
          list(iter(f)), [1, ' foo', ' bar ', 'baz\n', dict(x=1), 'qux']
      )

  def test_read_write_with_raw_texts(self):
    tmp_dir = tempfile.gettempdir()
    file2 = os.path.join(tmp_dir, 'file2')
    with sequence_io.open_sequence(file2, 'w') as f:
      self.assertIsInstance(f, sequence_io.LineSequence)
      with self.assertRaisesRegex(
          ValueError, 'Cannot write record with type'
      ):
        f.add(1)
      f.add('foo\nbar\n')

    with sequence_io.open_sequence(file2, 'r') as f:
      with self.assertRaisesRegex(
          ValueError, 'not writable'
      ):
        f.add('baz')
      self.assertEqual(list(iter(f)), ['foo', 'bar'])


class MemorySequenceIOTest(unittest.TestCase):

  def test_read_write(self):
    with sequence_io.open_sequence('/file1.mem@123', 'w') as f:
      self.assertIsInstance(f, sequence_io.MemorySequence)
      f.add(' foo')
      f.add(' bar ')
      f.flush()
      f.add('baz')
      with self.assertRaisesRegex(
          ValueError, 'Cannot write record with type'
      ):
        f.add(1)
      with self.assertRaisesRegex(
          ValueError, 'Cannot read memory sequence'
      ):
        next(iter(f))

    with self.assertRaisesRegex(
        ValueError, 'Cannot write record .* to a closed writer'
    ):
      f.add('qux')

    with sequence_io.open_sequence('/file1.mem@123', 'a') as f:
      self.assertIsInstance(f, sequence_io.MemorySequence)
      f.add('qux')

    with sequence_io.open_sequence('/file1.mem@123') as f:
      self.assertIsInstance(f, sequence_io.MemorySequence)
      self.assertEqual(len(f), 4)
      self.assertEqual(list(f), [' foo', ' bar ', 'baz', 'qux'])
      with self.assertRaisesRegex(
          ValueError, 'Cannot write record .* to memory sequence'
      ):
        f.add('abc')

    with self.assertRaisesRegex(
        ValueError, 'Cannot iterate over a closed sequence reader'
    ):
      next(iter(f))

    with sequence_io.open_sequence('/file1.mem@123', 'w') as f:
      f.add('abc')

    with sequence_io.open_sequence('/file1.mem@123', 'r') as f:
      self.assertEqual(list(iter(f)), ['abc'])

  def test_sharded_file_name(self):
    with sequence_io.open_sequence('/file1.mem-00000-of-00001', 'w') as f:
      self.assertIsInstance(f, sequence_io.MemorySequence)


if __name__ == '__main__':
  unittest.main()
