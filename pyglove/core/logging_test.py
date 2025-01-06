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
    string_io = io.StringIO()
    logger = logging.getLogger('logger1')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(stream=string_io)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    self.assertIs(pg_logging.get_logger(), logging.getLogger())
    pg_logging.set_logger(logger)
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
    logger = logging.getLogger('logger2')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stream=string_io)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    pg_logging.set_logger(logger)

    pg_logging.debug('x=%s', 6)
    self.assertEqual(string_io.getvalue(), '\n'.join([
        'x=6',
    ]) + '\n')


if __name__ == '__main__':
  unittest.main()
