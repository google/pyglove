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

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.base import eq as pg_eq
from pyglove.core.symbolic.base import from_json as pg_from_json
from pyglove.core.symbolic.class_wrapper import ClassWrapper
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.functor import Functor
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.object import Object
from pyglove.core.symbolic.symbolize import symbolize as pg_symbolize


class SymbolizeRegularTypesTest(unittest.TestCase):
  """Tests for `pg.symbolize`."""

  def test_symboliz_dict(self):
    self.assertIs(pg_symbolize(dict), Dict)
    with self.assertRaisesRegex(
        ValueError, 'Constraints are not supported in symbolic'):
      pg_symbolize(dict, pg_typing.Dict())

  def test_symbolize_list(self):
    self.assertIs(pg_symbolize(list), List)
    with self.assertRaisesRegex(
        ValueError, 'Constraints are not supported in symbolic'):
      pg_symbolize(list, pg_typing.List(pg_typing.Any()))


class SymbolizeFunctionsTest(unittest.TestCase):
  """Tests for symbolizing functions."""

  def test_symbolize_a_function_by_function_call(self):
    def f(x, y):
      del x, y
    self.assertTrue(issubclass(pg_symbolize(f), Functor))

  def test_symbolize_a_function_by_decorator_without_call(self):
    @pg_symbolize
    def f(x, y):
      del x, y
    self.assertTrue(issubclass(f, Functor))

  def test_symbolize_a_function_by_decorator_without_args(self):
    @pg_symbolize()
    def f(x, y):
      del x, y
    self.assertTrue(issubclass(f, Functor))

  def test_symbolize_a_function_by_decorator_with_typing(self):
    @pg_symbolize([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Str())
    ], returns=pg_typing.Int())
    def f(x, y):
      del x, y
    self.assertTrue(issubclass(f, Functor))
    self.assertEqual(f.signature.args, [
        (pg_typing.Argument('x', pg_typing.Int())),
        (pg_typing.Argument('y', pg_typing.Str()))
    ])
    self.assertEqual(f.signature.return_value, pg_typing.Int())

  def test_symbolize_with_serialization_key(self):
    @pg_symbolize(serialization_key='BAR', additional_keys=['RRR'])
    def bar(a, b):
      del a, b

    b = bar(3, 4)
    v = b.to_json()
    self.assertEqual(v['_type'], 'BAR')

    # Deserialize with type name: OK
    v['_type'] = bar.type_name
    self.assertTrue(pg_eq(pg_from_json(v), b))

    # Deserialize with additional key: OK
    v['_type'] = 'RRR'
    self.assertTrue(pg_eq(pg_from_json(v), b))


class SymbolizeClassesTest(unittest.TestCase):
  """Tests for symbolizing classes.

  For more detailed cases, see 'functor_test.py'.
  """

  def test_symbolize_a_class_by_function_call(self):

    class A:
      def __init__(self, x, y):
        pass

    A1 = pg_symbolize(A)  # pylint: disable=invalid-name
    self.assertIsInstance(A1(1, 2), A)
    self.assertIsInstance(A1(1, 2), ClassWrapper)
    self.assertEqual(list(A1.schema.fields.keys()), ['x', 'y'])
    self.assertEqual(
        [f.value for f in A1.schema.fields.values()],
        [pg_typing.Any(), pg_typing.Any()])

  def test_symbolize_a_class_by_decorator_without_call(self):

    @pg_symbolize
    class A:
      def __init__(self, x, y):
        pass

    self.assertIsInstance(A(1, 2), ClassWrapper)
    self.assertEqual(list(A.schema.fields.keys()), ['x', 'y'])
    self.assertEqual(
        [f.value for f in A.schema.fields.values()],
        [pg_typing.Any(), pg_typing.Any()])

  def test_symbolize_a_class_by_decorator_without_args(self):

    @pg_symbolize()
    class A:
      def __init__(self, x, y):
        pass

    self.assertIsInstance(A(1, 2), ClassWrapper)
    self.assertEqual(list(A.schema.fields.keys()), ['x', 'y'])
    self.assertEqual(
        [f.value for f in A.schema.fields.values()],
        [pg_typing.Any(), pg_typing.Any()])

  def test_symbolize_a_class_by_decorator_with_typing(self):

    @pg_symbolize([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Str()),
    ])
    class A:
      def __init__(self, x, y):
        pass

    self.assertEqual(list(A.schema.fields.keys()), ['x', 'y'])
    self.assertEqual(
        [f.value for f in A.schema.fields.values()],
        [pg_typing.Int(), pg_typing.Str()])

  def test_symbolize_pg_object_subclass(self):

    class A(Object):
      pass

    with self.assertRaisesRegex(
        ValueError,
        'Cannot symbolize .* is already a dataclass-like symbolic class'):
      pg_symbolize(A)

  def test_symbolize_an_existing_symbolized_class(self):

    @pg_symbolize
    class A:
      pass

    self.assertIs(pg_symbolize(A), A)

  def test_symbolize_with_serialization_key(self):

    @pg_symbolize(serialization_key='BB', additional_keys=['BBB'])
    class B:

      def __init__(self, x):
        self.x = x

    b = B(1)
    v = b.to_json()
    self.assertEqual(v['_type'], 'BB')

    # Deserilaize with serialization key: OK
    self.assertTrue(pg_eq(pg_from_json(v), b))

    # Deserialize with type name: OK
    v['_type'] = B.type_name
    self.assertTrue(pg_eq(pg_from_json(v), b))

    # Deserialize with additional key: OK
    v['_type'] = 'BBB'
    self.assertTrue(pg_eq(pg_from_json(v), b))

  def test_bad_symbolize(self):

    class A:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Only `constraint` is supported as positional arguments'):
      pg_symbolize(A, 1, 2)

    with self.assertRaisesRegex(TypeError, '.* cannot be symbolized'):
      pg_symbolize(1)


if __name__ == '__main__':
  unittest.main()
