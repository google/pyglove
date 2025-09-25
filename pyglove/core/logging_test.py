# Copyright 2019 The PyGlove Authors
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
import io
import logging
import unittest
from pyglove.core import logging as pg_logging


class LoggingTest(unittest.TestCase):
  """Tests for pg.logging."""

  def testLogging(self):
    self.assertIs(pg_logging.get_logger(), logging.getLogger())
    string_io = io.StringIO()
    logger = pg_logging.use_stream(string_io, name='logger1', fmt='')
    self.assertIs(pg_logging.get_logger(), logger)

    pg_logging.debug('x=%s', 1)
    pg_logging.info('y=%s', 2)
    pg_logging.warning('z=%s', 3)
    pg_logging.error('p=%s', 4)
    pg_logging.critical('q=%s', 5)

    self.assertEqual(string_io.getvalue(), '\n'.join([
        'y=2',
        'z=3',
        'p=4',
        'q=5',
    ]) + '\n')

    string_io = io.StringIO()
    with pg_logging.redirect_stream(
        string_io, level=logging.DEBUG, name='logger2', fmt=''
    ):
      pg_logging.debug('x=%s', 6)
      self.assertEqual(string_io.getvalue(), '\n'.join([
          'x=6',
      ]) + '\n')

    pg_logging.use_stdout()
    pg_logging.info('y=%s', 7)

if __name__ == '__main__':
  unittest.main()
