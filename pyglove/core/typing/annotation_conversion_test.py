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
"""Tests for pyglove.core.typing.annotation_conversion."""

import inspect
import typing
import unittest

from pyglove.core.typing import annotation_conversion
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs
from pyglove.core.typing.class_schema import Field
from pyglove.core.typing.class_schema import ValueSpec


class FieldFromAnnotationTest(unittest.TestCase):
  """Tests for Field.fromAnnotation."""

  def test_from_annotated(self):
    self.assertEqual(
        Field.from_annotation(
            'x', typing.Annotated[str, 'A str'], auto_typing=True),
        Field('x', vs.Str(), 'A str'))

    self.assertEqual(
        Field.from_annotation(
            'x', typing.Annotated[str, 'A str', dict(x=1)], auto_typing=True),
        Field('x', vs.Str(), 'A str', dict(x=1)))

  def test_from_regular_annotation(self):
    self.assertEqual(
        Field.from_annotation('x', str, auto_typing=True),
        Field('x', vs.Str()))

    self.assertEqual(
        Field.from_annotation('x', str, 'A str', auto_typing=True),
        Field('x', vs.Str(), 'A str'))

    self.assertEqual(
        Field.from_annotation('x', str, 'A str', dict(x=1), auto_typing=True),
        Field('x', vs.Str(), 'A str', dict(x=1)))


class ValueSpecFromAnnotationTest(unittest.TestCase):
  """Tests for ValueSpec.fromAnnotation."""

  def assert_from_annotation_with_default_value(self, annotation, expected):
    self.assertEqual(
        ValueSpec.from_annotation(annotation, accept_value_as_annotation=True),
        expected)

  def test_from_values(self):
    self.assert_from_annotation_with_default_value(1, vs.Int(1))
    self.assert_from_annotation_with_default_value(1.0, vs.Float(1.0))
    self.assert_from_annotation_with_default_value(False, vs.Bool(False))
    self.assert_from_annotation_with_default_value('abc', vs.Str('abc'))
    self.assert_from_annotation_with_default_value(
        [], vs.List(vs.Any(), default=[]))
    self.assert_from_annotation_with_default_value(
        [1, 2], vs.List(vs.Int(), default=[1, 2]))
    self.assert_from_annotation_with_default_value(
        {}, vs.Dict().set_default({}))
    self.assert_from_annotation_with_default_value(
        (1, 1.0, 'b'),
        vs.Tuple([vs.Int(), vs.Float(), vs.Str()]).set_default((1, 1.0, 'b')))

    class A:
      pass

    a = A()
    self.assert_from_annotation_with_default_value(
        a,
        vs.Object(A, default=a))

    f = lambda: 0
    self.assert_from_annotation_with_default_value(
        f,
        vs.Callable(default=f))

    with self.assertRaisesRegex(TypeError, 'Cannot convert .*'):
      _ = ValueSpec.from_annotation(1, auto_typing=True)

  def test_no_annotation(self):
    self.assertEqual(
        ValueSpec.from_annotation(inspect.Parameter.empty, False), vs.Any())
    self.assertEqual(
        ValueSpec.from_annotation(inspect.Parameter.empty, True), vs.Any())

  def test_none(self):
    self.assertEqual(
        ValueSpec.from_annotation(None, False), vs.Any().freeze(None))
    self.assertEqual(
        ValueSpec.from_annotation(None, True), vs.Any().freeze(None))
    self.assertEqual(
        ValueSpec.from_annotation(
            None, accept_value_as_annotation=True), vs.Any().noneable())

  def test_any(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Any, False),
        vs.Any(annotation=typing.Any))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Any, True),
        vs.Any(annotation=typing.Any))

  def test_bool(self):
    self.assertEqual(ValueSpec.from_annotation(bool, True), vs.Bool())
    self.assertEqual(
        ValueSpec.from_annotation(bool, False), vs.Any(annotation=bool))
    self.assertEqual(
        ValueSpec.from_annotation(bool, False, True), vs.Any(annotation=bool))

  def test_int(self):
    self.assertEqual(ValueSpec.from_annotation(int, True), vs.Int())
    self.assertEqual(ValueSpec.from_annotation(int, True, True), vs.Int())
    self.assertEqual(
        ValueSpec.from_annotation(int, False), vs.Any(annotation=int))
    self.assertEqual(
        ValueSpec.from_annotation(int, False, True), vs.Any(annotation=int))

  def test_float(self):
    self.assertEqual(ValueSpec.from_annotation(float, True), vs.Float())
    self.assertEqual(ValueSpec.from_annotation(float, True, False), vs.Float())
    self.assertEqual(
        ValueSpec.from_annotation(float, False), vs.Any(annotation=float))
    self.assertEqual(
        ValueSpec.from_annotation(float, False, True), vs.Any(annotation=float))

  def test_str(self):
    self.assertEqual(ValueSpec.from_annotation(str, True), vs.Str())
    self.assertEqual(ValueSpec.from_annotation(str, True, False), vs.Str())
    self.assertEqual(
        ValueSpec.from_annotation(str, False), vs.Any(annotation=str))
    self.assertEqual(
        ValueSpec.from_annotation(str, False, True), vs.Any(annotation=str))

    self.assertEqual(
        ValueSpec.from_annotation('A', False, False), vs.Any(annotation='A'))
    self.assertEqual(
        ValueSpec.from_annotation('A', False, True), vs.Str('A'))
    self.assertEqual(
        ValueSpec.from_annotation('A', True), vs.Object('A'))
    self.assertEqual(
        ValueSpec.from_annotation('A', True, True), vs.Str('A'))

  def test_list(self):
    self.assertEqual(ValueSpec.from_annotation(list, True), vs.List(vs.Any()))
    self.assertEqual(
        ValueSpec.from_annotation(typing.List, True), vs.List(vs.Any()))
    self.assertEqual(
        ValueSpec.from_annotation(list[int], True), vs.List(vs.Int()))

  def test_tuple(self):
    self.assertEqual(ValueSpec.from_annotation(tuple, True), vs.Tuple(vs.Any()))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Tuple, True), vs.Tuple(vs.Any()))
    self.assertEqual(
        ValueSpec.from_annotation(tuple[int], True), vs.Tuple([vs.Int()]))
    self.assertEqual(
        ValueSpec.from_annotation(tuple[int, ...], True), vs.Tuple(vs.Int()))
    with self.assertRaisesRegex(
        TypeError, 'Tuple with ellipsis should have exact 2 type arguments'):
      ValueSpec.from_annotation(tuple[...], True)

  def test_sequence(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Sequence[int], True),
        vs.Union([vs.List(vs.Int()), vs.Tuple(vs.Int())]))

  def test_dict(self):
    self.assertEqual(ValueSpec.from_annotation(dict, True), vs.Dict())
    self.assertEqual(ValueSpec.from_annotation(typing.Dict, True), vs.Dict())
    self.assertEqual(
        ValueSpec.from_annotation(typing.Dict[str, int], True),
        vs.Dict([(ks.StrKey(), vs.Int())]))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Mapping[str, int], True),
        vs.Dict([(ks.StrKey(), vs.Int())]))
    with self.assertRaisesRegex(
        TypeError, 'Dict type field with non-string key is not supported'):
      ValueSpec.from_annotation(dict[int, int], True)

  def test_callable(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Callable, True), vs.Callable())
    self.assertEqual(
        ValueSpec.from_annotation(typing.Callable[..., typing.Any], True),
        vs.Callable(returns=vs.Any(annotation=typing.Any)))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Callable[int, int], True),
        vs.Callable(args=[vs.Int()], returns=vs.Int()))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Callable[(int, int), int], True),
        vs.Callable(args=[vs.Int(), vs.Int()], returns=vs.Int()))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Callable[[int, int], int], True),
        vs.Callable(args=[vs.Int(), vs.Int()], returns=vs.Int()))

  def test_class(self):
    class Foo:
      pass

    self.assertEqual(
        ValueSpec.from_annotation(Foo, True), vs.Object(Foo))
    self.assertEqual(
        ValueSpec.from_annotation('Foo', True), vs.Object('Foo'))
    self.assertEqual(
        ValueSpec.from_annotation(Foo, False), vs.Any(annotation=Foo))

  def test_generic_class(self):
    X = typing.TypeVar('X')
    Y = typing.TypeVar('Y')

    class Foo(typing.Generic[X, Y]):
      pass

    self.assertEqual(
        ValueSpec.from_annotation(Foo[int, str], True), vs.Object(Foo[int, str])
    )

  def test_type(self):
    class Foo:
      pass

    self.assertEqual(
        ValueSpec.from_annotation(typing.Type[Foo], True), vs.Type(Foo)
    )
    self.assertEqual(ValueSpec.from_annotation(type[Foo], True), vs.Type(Foo))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Type, True), vs.Type(typing.Any)
    )
    self.assertEqual(ValueSpec.from_annotation(type, True), vs.Type(typing.Any))

  def test_optional(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Optional[int], True),
        vs.Int().noneable())
    if annotation_conversion._UnionType:
      self.assertEqual(
          ValueSpec.from_annotation(int | None, True),
          vs.Int().noneable())

  def test_union(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Union[int, str], True),
        vs.Union([vs.Int(), vs.Str()]))
    self.assertEqual(
        ValueSpec.from_annotation(typing.Union[int, str, None], True),
        vs.Union([vs.Int(), vs.Str()]).noneable())
    if annotation_conversion._UnionType:
      self.assertEqual(
          ValueSpec.from_annotation(int | str, True),
          vs.Union([vs.Int(), vs.Str()]))


if __name__ == '__main__':
  unittest.main()
