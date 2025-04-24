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
import sys
import typing
import unittest

from pyglove.core import coding
from pyglove.core.typing import annotated
from pyglove.core.typing import annotation_conversion
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs
from pyglove.core.typing.class_schema import Field
from pyglove.core.typing.class_schema import ForwardRef
from pyglove.core.typing.class_schema import ValueSpec


class Foo:
  class Bar:
    pass


_MODULE = sys.modules[__name__]


class AnnotationFromStrTest(unittest.TestCase):
  """Tests for annotation_from_str."""

  def test_basic_types(self):
    self.assertIsNone(annotation_conversion.annotation_from_str('None'))
    self.assertEqual(annotation_conversion.annotation_from_str('str'), str)
    self.assertEqual(annotation_conversion.annotation_from_str('int'), int)
    self.assertEqual(annotation_conversion.annotation_from_str('float'), float)
    self.assertEqual(annotation_conversion.annotation_from_str('bool'), bool)
    self.assertEqual(annotation_conversion.annotation_from_str('list'), list)
    self.assertEqual(
        annotation_conversion.annotation_from_str('list[int]'), list[int]
    )
    self.assertEqual(annotation_conversion.annotation_from_str('tuple'), tuple)
    self.assertEqual(
        annotation_conversion.annotation_from_str('tuple[int]'), tuple[int]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str('tuple[int, ...]'),
        tuple[int, ...]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str('tuple[int, str]'),
        tuple[int, str]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str('list[Foo]', _MODULE),
        list[Foo]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str('list[Foo.Bar]', _MODULE),
        list[Foo.Bar]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str('list[Foo.Baz]', _MODULE),
        list[typing.ForwardRef('Foo.Baz', False, _MODULE)]
    )

  def test_generic_types(self):
    self.assertEqual(
        annotation_conversion.annotation_from_str('typing.List[str]', _MODULE),
        typing.List[str]
    )

  def test_union(self):
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.Union[str, typing.Union[int, float]]', _MODULE),
        typing.Union[str, int, float]
    )
    if sys.version_info >= (3, 10):
      self.assertEqual(
          annotation_conversion.annotation_from_str(
              'str | int | float', _MODULE),
          typing.Union[str, int, float]
      )

  def test_literal(self):
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.Literal[1, True, "a", \'"b"\', "\\"c\\"", "\\\\"]',
            _MODULE
        ),
        typing.Literal[1, True, 'a', '"b"', '"c"', '\\']
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.Literal[(1, 1), f"A {[1]}"]', _MODULE),
        typing.Literal[(1, 1), 'A [1]']
    )
    with self.assertRaisesRegex(SyntaxError, 'Expected "\\["'):
      annotation_conversion.annotation_from_str('typing.Literal', _MODULE)

    with self.assertRaisesRegex(SyntaxError, 'Unexpected end of annotation'):
      annotation_conversion.annotation_from_str('typing.Literal[1', _MODULE)

    with self.assertRaisesRegex(
        coding.CodeError, 'Function definition is not allowed'
    ):
      annotation_conversion.annotation_from_str(
          'typing.Literal[lambda x: x]', _MODULE
      )

  def test_callable(self):
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.Callable[int, int]', _MODULE),
        typing.Callable[[int], int]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.Callable[[int], int]', _MODULE),
        typing.Callable[[int], int]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.Callable[..., None]', _MODULE),
        typing.Callable[..., None]
    )

  def test_forward_ref(self):
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'AAA', _MODULE),
        typing.ForwardRef(
            'AAA', False, _MODULE
        )
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.List[AAA]', _MODULE),
        typing.List[
            typing.ForwardRef(
                'AAA', False, _MODULE
            )
        ]
    )

  def test_reloading(self):
    setattr(_MODULE, '__reloading__', True)
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.List[Foo]', _MODULE),
        typing.List[
            typing.ForwardRef(
                'Foo', False, _MODULE
            )
        ]
    )
    self.assertEqual(
        annotation_conversion.annotation_from_str(
            'typing.List[Foo.Bar]', _MODULE),
        typing.List[
            typing.ForwardRef(
                'Foo.Bar', False, _MODULE
            )
        ]
    )
    delattr(_MODULE, '__reloading__')

  def test_bad_annotation(self):
    with self.assertRaisesRegex(SyntaxError, 'Expected type identifier'):
      annotation_conversion.annotation_from_str('typing.List[]')

    with self.assertRaisesRegex(SyntaxError, 'Expected "]"'):
      annotation_conversion.annotation_from_str('typing.List[int')

    with self.assertRaisesRegex(SyntaxError, 'Unexpected end of annotation'):
      annotation_conversion.annotation_from_str('typing.List[int]1', _MODULE)

    with self.assertRaisesRegex(SyntaxError, 'Expected "]"'):
      annotation_conversion.annotation_from_str('typing.Callable[[x')

    with self.assertRaisesRegex(TypeError, '.* does not exist'):
      annotation_conversion.annotation_from_str('typing.Foo', _MODULE)


class FieldFromAnnotationTest(unittest.TestCase):
  """Tests for Field.fromAnnotation."""

  def test_from_pg_annotated(self):
    self.assertEqual(
        Field.from_annotation(
            'x', annotated.Annotated[str, 'A str'], auto_typing=True),
        Field('x', vs.Str(), 'A str'))
    self.assertEqual(
        Field.from_annotation(
            'x',
            annotated.Annotated[vs.Str().noneable(), 'A str'],
            auto_typing=True),
        Field('x', vs.Str().noneable(), 'A str'))
    self.assertEqual(
        Field.from_annotation(
            'x',
            annotated.Annotated[vs.Str().noneable(), 'A str', dict(foo=1)],
            auto_typing=True),
        Field('x', vs.Str().noneable(), 'A str', dict(foo=1)))

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

  def test_with_parent_module(self):
    self.assertEqual(
        Field.from_annotation(
            'x', 'ValueSpecBase', auto_typing=True, parent_module=vs
        ),
        Field('x', vs.Object(vs.ValueSpecBase))
    )


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
        ValueSpec.from_annotation(None, False), vs.Object(type(None)))
    self.assertEqual(
        ValueSpec.from_annotation('None', True), vs.Object(type(None)))
    self.assertEqual(
        ValueSpec.from_annotation(None, True), vs.Object(type(None)))
    self.assertEqual(
        ValueSpec.from_annotation(None, accept_value_as_annotation=True),
        vs.Any().noneable()
    )

  def test_any(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Any, False),
        vs.Any(annotation=typing.Any))
    self.assertEqual(
        ValueSpec.from_annotation('typing.Any', True, parent_module=_MODULE),
        vs.Any(annotation=typing.Any)
    )
    self.assertEqual(
        ValueSpec.from_annotation(typing.Any, True),
        vs.Any(annotation=typing.Any))
    self.assertEqual(
        ValueSpec.from_annotation(vs.Any, True),
        vs.Any(annotation=vs.Any))

  def test_bool(self):
    self.assertEqual(ValueSpec.from_annotation(bool, True), vs.Bool())
    self.assertEqual(ValueSpec.from_annotation('bool', True), vs.Bool())
    self.assertEqual(
        ValueSpec.from_annotation(bool, False), vs.Any(annotation=bool))
    self.assertEqual(
        ValueSpec.from_annotation(bool, False, True), vs.Any(annotation=bool))

  def test_int(self):
    self.assertEqual(ValueSpec.from_annotation(int, True), vs.Int())
    self.assertEqual(ValueSpec.from_annotation('int', True), vs.Int())
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
        ValueSpec.from_annotation('A', False, False),
        vs.Any(annotation='A')
    )
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

  def test_enum(self):
    self.assertEqual(
        ValueSpec.from_annotation(typing.Literal[None, 1, 'foo'], True),
        vs.Enum[None, 1, 'foo'])

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
    self.assertEqual(
        ValueSpec.from_annotation(Foo, True), vs.Object(Foo))
    self.assertEqual(
        ValueSpec.from_annotation('Foo', True), vs.Object('Foo'))
    self.assertEqual(
        ValueSpec.from_annotation(Foo, False), vs.Any(annotation=Foo))
    self.assertEqual(
        ValueSpec.from_annotation(
            typing.ForwardRef('Foo'), True), vs.Object(Foo))
    self.assertEqual(
        ValueSpec.from_annotation(
            ForwardRef(sys.modules[__name__], 'Foo'),
            True
        ), vs.Object(Foo))

  def test_generic_class(self):
    X = typing.TypeVar('X')
    Y = typing.TypeVar('Y')

    class Bar(typing.Generic[X, Y]):
      pass

    self.assertEqual(
        ValueSpec.from_annotation(Bar[int, str], True), vs.Object(Bar[int, str])
    )

  def test_type(self):
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

  def test_final(self):
    self.assertEqual(
        ValueSpec.from_annotation(
            typing.Final[int], True
        ).set_default(1),
        vs.Int().freeze(1)
    )


if __name__ == '__main__':
  unittest.main()
