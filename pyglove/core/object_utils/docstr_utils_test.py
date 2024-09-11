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
"""Tests for pyglove.object_utils.docstr_utils."""

import inspect
import unittest
from pyglove.core.object_utils import docstr_utils


class DocStrTest(unittest.TestCase):
  """Tests for DocStr."""

  def test_parse(self):
    doc = """Test doc string."""
    docstr = docstr_utils.DocStr.parse(doc)
    # Returns the first style that fit.
    self.assertEqual(docstr.style, docstr_utils.DocStrStyle.REST)

    docstr = docstr_utils.DocStr.parse(doc, docstr_utils.DocStrStyle.GOOGLE)
    self.assertEqual(docstr.style, docstr_utils.DocStrStyle.GOOGLE)

  def test_docstr(self):
    def my_sum(x, y, *args, **kwargs):
      """Returns the sum of two integers.

      This function will return the sum of two integers.

      Examples:

        ```
          ret = sum(1, 2)
          print(ret)
        ```

      Args:
        x: An integer.
        y: Another integer.
        *args: Variable positional args.
        **kwargs: Variable keyword args.

      Returns:
        The sum of both.

      Raises:
        ValueError: when either `x` and `y` is not an integer.
      """
      del args, kwargs
      return x + y

    docstr = docstr_utils.docstr(my_sum)
    self.assertEqual(
        docstr.description,
        inspect.cleandoc('''
        Returns the sum of two integers.

        This function will return the sum of two integers.
        ''')
    )
    self.assertEqual(docstr, docstr_utils.DocStr(
        style=docstr_utils.DocStrStyle.GOOGLE,
        short_description='Returns the sum of two integers.',
        long_description='This function will return the sum of two integers.',
        examples=[
            docstr_utils.DocStrExample(
                description='```\n  ret = sum(1, 2)\n  print(ret)\n```')
        ],
        args={
            'x': docstr_utils.DocStrArgument(
                name='x',
                description='An integer.',
            ),
            'y': docstr_utils.DocStrArgument(
                name='y',
                description='Another integer.',
            ),
            '*args': docstr_utils.DocStrArgument(
                name='*args',
                description='Variable positional args.',
            ),
            '**kwargs': docstr_utils.DocStrArgument(
                name='**kwargs',
                description='Variable keyword args.',
            )
        },
        returns=docstr_utils.DocStrReturns(
            description='The sum of both.',
        ),
        raises=[
            docstr_utils.DocStrRaises(
                type_name='ValueError',
                description='when either `x` and `y` is not an integer.',
            )
        ]
    ))
    sig = inspect.signature(my_sum)
    self.assertIsNotNone(docstr.parameter(sig.parameters['x']))
    self.assertIsNotNone(docstr.parameter(sig.parameters['y']))
    self.assertIsNotNone(docstr.parameter(sig.parameters['args']))
    self.assertIsNotNone(docstr.parameter(sig.parameters['kwargs']))

    class Foo:
      pass

    self.assertIsNone(docstr_utils.docstr(Foo))
    self.assertIsNone(docstr_utils.docstr(None))

    class Bar:
      """bar."""

    self.assertEqual(docstr_utils.docstr(Bar).description, 'bar.')

    # pylint: disable=g-classes-have-attributes
    # pylint: disable=g-short-docstring-punctuation
    # pylint: disable=g-space-before-docstring-summary
    # pylint: disable=g-no-space-after-docstring-summary
    class Baz:
      """
      Args:
        x: int
      """
    # pylint: enable=g-space-before-docstring-summary
    # pylint: enable=g-short-docstring-punctuation
    # pylint: enable=g-classes-have-attributes
    self.assertIsNone(docstr_utils.docstr(Baz).description)


if __name__ == '__main__':
  unittest.main()
