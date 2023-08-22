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
"""PyGlove runtime typing example."""

import pyglove as pg


class Foo(pg.Object):
  """A symbolic class."""

  x: pg.typing.List[int] | None
  y: pg.typing.Dict[str, pg.typing.Any]
  z: pg.typing.Annotated[
      pg.typing.Int(min_value=0),   # Field value spec.
      'Field z',                    # Field docstring.
      dict(p=1)                     # Meta-data
  ]
  p: pg.typing.Enum[['foo', 'bar', 'baz']] = 'baz'
  q: pg.typing.Dict[{
      'a': int | None,
      'b': pg.typing.Int[0, 100],
      'c': pg.typing.Tuple[int, ...]
  }] = dict(a=1, b=10, c=(1, 2))


# `pg.typing` could also be used for static type analysis.
def add(
    x: pg.typing.Int[0, None],
    y: pg.typing.Float[None, 1.0]) -> pg.typing.Any:
  return x + y


def main() -> None:
  foo = Foo([0, 1], dict(x=1), 1)
  print(foo)

  try:
    Foo(None, dict(y=1), -1)
  except ValueError as e:
    print('Expected error', e)

  # There is no runtime check for regular Python function even type annotation
  # is given.
  print(add(-1, 2.0))

  # But we can create a symbolic function which does runtime value check.
  prime_add = pg.symbolize(add)
  try:
    prime_add(-1, 2.0)()
  except ValueError as e:
    print('Expected error', e)


if __name__ == '__main__':
  main()

