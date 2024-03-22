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
"""Tests for pyglove.symbolize."""

import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import list as pg_list  # pylint: disable=unused-import
from pyglove.core.symbolic import schema_utils


class CallableSchemaTest(unittest.TestCase):
  """Tests for `callable_schema`."""

  def test_function_schema(self):
    def foo(x: int, *args, y: str, **kwargs) -> float:
      """A function.

      Args:
        x: Input 1.
        *args: Variable positional args.
        y: Input 2.
        **kwargs: Variable keyword args.

      Returns:
        The result.
      """
      del x, y, args, kwargs

    schema = schema_utils.callable_schema(foo, auto_typing=True, auto_doc=True)
    self.assertEqual(schema.name, f'{foo.__module__}.{foo.__qualname__}')
    self.assertEqual(schema.description, 'A function.')
    self.assertEqual(
        list(schema.fields.values()),
        [
            pg_typing.Field('x', pg_typing.Int(), description='Input 1.'),
            pg_typing.Field(
                'args',
                pg_typing.List(pg_typing.Any(), default=[]),
                description='Variable positional args.',
            ),
            pg_typing.Field('y', pg_typing.Str(), description='Input 2.'),
            pg_typing.Field(
                pg_typing.StrKey(),
                pg_typing.Any(),
                description='Variable keyword args.',
            ),
        ],
    )

  def test_class_init_schema(self):
    class A:

      def __init__(self, x: int, *args, y: str, **kwargs) -> float:
        """Constructor.

        Args:
          x: Input 1.
          *args: Variable positional args.
          y: Input 2.
          **kwargs: Variable keyword args.

        Returns:
          The result.
        """
        del x, y, args, kwargs

    schema = schema_utils.callable_schema(
        A.__init__, auto_typing=True, auto_doc=True, remove_self=True
    )
    self.assertEqual(schema.name, f'{A.__module__}.{A.__init__.__qualname__}')
    self.assertEqual(schema.description, 'Constructor.')
    self.assertEqual(
        list(schema.fields.values()),
        [
            pg_typing.Field('x', pg_typing.Int(), description='Input 1.'),
            pg_typing.Field(
                'args',
                pg_typing.List(pg_typing.Any(), default=[]),
                description='Variable positional args.',
            ),
            pg_typing.Field('y', pg_typing.Str(), description='Input 2.'),
            pg_typing.Field(
                pg_typing.StrKey(),
                pg_typing.Any(),
                description='Variable keyword args.',
            ),
        ],
    )


class SchemaDescriptionFromDocStrTest(unittest.TestCase):
  """Tests for `schema_description_from_docstr`."""

  def test_none_doc_str(self):
    self.assertIsNone(schema_utils.schema_description_from_docstr(None))

  def test_short_description_only(self):
    docstr = object_utils.DocStr.parse(
        """This is a function.""")
    self.assertEqual(
        schema_utils.schema_description_from_docstr(docstr),
        'This is a function.')
    self.assertEqual(
        schema_utils.schema_description_from_docstr(
            docstr, include_long_description=True),
        'This is a function.')

  def test_long_description_only(self):
    docstr = object_utils.DocStr.parse(
        """This is a function.
        
        This is the longer explanation of the function.
        
        """)
    self.assertEqual(
        schema_utils.schema_description_from_docstr(docstr),
        'This is a function.')
    self.assertEqual(
        schema_utils.schema_description_from_docstr(
            docstr, include_long_description=True),
        ('This is a function.\n\n'
         'This is the longer explanation of the function.'))


if __name__ == '__main__':
  unittest.main()
