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

from __future__ import annotations
import sys
import typing
from typing import List, Literal, Union
import unittest

from pyglove.core import symbolic as pg
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


class AnnotationFutureConversionTest(unittest.TestCase):

  # Class with forward declaration must not be defined in functions.
  class A(pg.Object):
    a: typing.Optional[AnnotationFutureConversionTest.A]
    b: List[AnnotationFutureConversionTest.A]

  def assert_value_spec(self, cls, field_name, expected_value_spec):
    self.assertEqual(cls.__schema__[field_name].value, expected_value_spec)

  def test_basics(self):

    class Foo(pg.Object):
      a: int
      b: float
      c: bool
      d: str
      e: typing.Any
      f: typing.Dict[str, typing.Any]
      g: typing.List[str]
      h: typing.Tuple[int, int]
      i: typing.Callable[[int, int], None]

    self.assert_value_spec(Foo, 'a', vs.Int())
    self.assert_value_spec(Foo, 'b', vs.Float())
    self.assert_value_spec(Foo, 'c', vs.Bool())
    self.assert_value_spec(Foo, 'd', vs.Str())
    self.assert_value_spec(Foo, 'e', vs.Any(annotation=typing.Any))
    self.assert_value_spec(
        Foo, 'f', vs.Dict([(ks.StrKey(), vs.Any(annotation=typing.Any))])
    )
    self.assert_value_spec(Foo, 'g', vs.List(vs.Str()))
    self.assert_value_spec(Foo, 'h', vs.Tuple([vs.Int(), vs.Int()]))
    self.assert_value_spec(
        Foo, 'i',
        vs.Callable([vs.Int(), vs.Int()], returns=vs.Object(type(None)))
    )

  def test_list(self):
    if sys.version_info >= (3, 10):

      class Bar(pg.Object):
        x: list[int | None]

      self.assert_value_spec(Bar, 'x', vs.List(vs.Int().noneable()))

  def test_var_length_tuple(self):

    class Foo(pg.Object):
      x: typing.Tuple[int, ...]

    self.assert_value_spec(Foo, 'x', vs.Tuple(vs.Int()))

    if sys.version_info >= (3, 10):

      class Bar(pg.Object):
        x: tuple[int, ...]

      self.assert_value_spec(Bar, 'x', vs.Tuple(vs.Int()))

  def test_optional(self):

    class Foo(pg.Object):
      x: typing.Optional[int]

    self.assert_value_spec(Foo, 'x', vs.Int().noneable())

    if sys.version_info >= (3, 10):
      class Bar(pg.Object):
        x: int | None

      self.assert_value_spec(Bar, 'x', vs.Int().noneable())

  def test_union(self):

    class Foo(pg.Object):
      x: Union[int, typing.Union[str, bool], None]

    self.assert_value_spec(
        Foo, 'x', vs.Union([vs.Int(), vs.Str(), vs.Bool()]).noneable()
    )

    if sys.version_info >= (3, 10):

      class Bar(pg.Object):
        x: int | str | bool

      self.assert_value_spec(
          Bar, 'x', vs.Union([vs.Int(), vs.Str(), vs.Bool()])
      )

  def test_literal(self):

    class Foo(pg.Object):
      x: Literal[1, True, 'abc']

    self.assert_value_spec(
        Foo, 'x', vs.Enum(vs.MISSING_VALUE, [1, True, 'abc'])
    )

  def test_self_referencial(self):
    self.assert_value_spec(
        self.A, 'a', vs.Object(self.A).noneable()
    )
    self.assert_value_spec(
        self.A, 'b', vs.List(vs.Object(self.A))
    )

if __name__ == '__main__':
  unittest.main()
