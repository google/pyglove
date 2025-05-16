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

import inspect
import unittest
from pyglove.core.symbolic import base
from pyglove.core.symbolic import error_info as error_info_lib  # pylint: disable=g-bad-import-order


class ErrorInfoTest(unittest.TestCase):
  """Tests for ErrorInfo."""

  def test_from_exception(self):

    def foo():
      return 1 / 0

    def bar():
      try:
        foo()
      except ZeroDivisionError as e:
        raise ValueError('Bad call to `foo`') from e

    error_info = None
    try:
      bar()
    except ValueError as e:
      error_info = error_info_lib.ErrorInfo.from_exception(e)
    self.assertIsNotNone(error_info)
    self.assertEqual(error_info.tag, 'ValueError.ZeroDivisionError')
    self.assertEqual(error_info.description, 'Bad call to `foo`')
    self.assertIn('Traceback (most recent call last)', error_info.stacktrace)

  def test_to_json(self):
    error_info = error_info_lib.ErrorInfo(
        tag='ValueError.ZeroDivisionError',
        description='Bad call to `foo`',
        stacktrace='Traceback (most recent call last)',
    )
    json_dict = error_info.to_json()
    error_info2 = base.from_json(json_dict)
    self.assertIsNot(error_info2, error_info)
    self.assertEqual(error_info2, error_info)
    json_dict['_type'] = 'pyglove.core.utils.error_utils.ErrorInfo'
    error_info2 = base.from_json(json_dict)
    self.assertEqual(error_info2, error_info)

  def test_format(self):
    error_info = error_info_lib.ErrorInfo(
        tag='ValueError.ZeroDivisionError',
        description='Bad call to `foo`',
        stacktrace='Traceback (most recent call last)',
    )
    self.assertEqual(
        error_info.format(compact=False),
        inspect.cleandoc(
            """
            ErrorInfo(
              tag = 'ValueError.ZeroDivisionError',
              description = 'Bad call to `foo`',
              stacktrace = 'Traceback (most recent call last)'
            )
            """
        )
    )

  def test_to_html(self):
    error_info = error_info_lib.ErrorInfo(
        tag='ValueError.ZeroDivisionError',
        description='Bad call to `foo`',
        stacktrace='Traceback (most recent call last)',
    )
    html = error_info.to_html()
    self.assertIn('ErrorInfo', html.content)
    self.assertIn('ValueError.ZeroDivisionError', html.content)
    self.assertIn('Bad call to `foo`', html.content)
    self.assertIn('Traceback (most recent call last)', html.content)


if __name__ == '__main__':
  unittest.main()
