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
"""Tests for pyglove.core.typing.annotated."""

import typing
import unittest

from pyglove.core.typing import annotated
from pyglove.core.typing import annotation_conversion  # pylint: disable=unused-import
from pyglove.core.typing import value_specs as vs


class AnnotatedTest(unittest.TestCase):

  def test_subscription(self):
    # Field type only.
    x = annotated.Annotated[int]
    self.assertEqual(x.value_spec, vs.Int())
    self.assertIsNone(x.docstring)
    self.assertEqual(x.metadata, {})

    # Field type and docstring
    x = annotated.Annotated[int, 'hello']
    self.assertEqual(x.value_spec, vs.Int())
    self.assertEqual(x.docstring, 'hello')
    self.assertEqual(x.metadata, {})

    # Field type, docstring and metadata
    x = annotated.Annotated[int, 'hello', dict(foo=1)]
    self.assertEqual(x.value_spec, vs.Int())
    self.assertEqual(x.docstring, 'hello')
    self.assertEqual(x.metadata, dict(foo=1))

  def test_bad_subscription(self):
    with self.assertRaisesRegex(
        TypeError, '`pg.typing.Annotated` accepts 1 to 3 type arguments'):
      _ = annotated.Annotated[int, 'hello', dict(foo=1), 1]

    with self.assertRaisesRegex(
        TypeError, 'Cannot convert 1'):
      _ = annotated.Annotated[1]

    with self.assertRaisesRegex(
        TypeError, 'The second type argument .* must be a str'):
      _ = annotated.Annotated[int, 1]

    with self.assertRaisesRegex(
        TypeError, 'The third type argument .* must be a dict with str keys'):
      _ = annotated.Annotated[int, 'foo', [1, 2]]

  def test_type_checking(self):
    typing.TYPE_CHECKING = True

    self.assertIs(annotated.Annotated[str], str)
    self.assertIs(annotated.Annotated[str, 'hello'], str)
    self.assertIs(
        annotated.Annotated[vs.Str().noneable(), 'hello'], typing.Optional[str])

    typing.TYPE_CHECKING = False


if __name__ == '__main__':
  unittest.main()
