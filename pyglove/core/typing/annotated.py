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
"""pg.typing.Annotated: A drop-in replacement of typing.Annotated."""

import typing
from typing import Any, Dict, Optional, Tuple

from pyglove.core.typing import class_schema
from pyglove.core.typing import value_specs as vs


class Annotated(vs.Generic):
  """The PyGlove enhanced `typing.Annotated` for defining a class field.

  Example::

    class A(pg.Object):
      x = pg.typing.Annotated[
          int,                                # Field type or value spec.
          'Docstring for field `x`.',         # Field docstring.
          dict(foo=1, bar=2)                  # Field metadata
      ]
    ]
  """

  def __init__(
      self,
      t: vs.ValueSpecOrAnnotation,
      docstring: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None
      ):
    super().__init__()
    self._value_spec = class_schema.ValueSpec.from_annotation(
        t, auto_typing=True)
    self._docstring = docstring
    self._metadata = metadata or {}

  @property
  def value_spec(self) -> class_schema.ValueSpec:
    """Returns the value spec for the field."""
    return self._value_spec

  @property
  def docstring(self) -> Optional[str]:
    """Returns the docstring for the field."""
    return self._docstring

  @property
  def metadata(self) -> Dict[str, Any]:
    """Returns the metadata for the field."""
    return self._metadata

  @classmethod
  def with_type_args(cls, type_args: Tuple[Any, ...]) -> Any:
    if len(type_args) == 1:
      t = type_args[0]
      docstring, metadata = None, None
    elif len(type_args) == 2:
      t, docstring = type_args
      metadata = None
    elif len(type_args) == 3:
      t, docstring, metadata = type_args
    else:
      raise TypeError(
          '`pg.typing.Annotated` accepts 1 to 3 type arguments ',
          '(<field type>, [field docstring], [field metadata]). '
          f'Encountered: {type_args!r}')
    if docstring is not None and not isinstance(docstring, str):
      raise TypeError(
          'The second type argument (`docstring`) must be a str. '
          f'Encountered: {docstring!r}')
    if metadata is not None and not isinstance(metadata, dict):
      raise TypeError(
          'The third type argument (`metadata`) must be a dict with str keys. '
          f'Encountered: {metadata!r}')

    t = class_schema.ValueSpec.from_annotation(t, auto_typing=True)
    annotated = Annotated(t, docstring=docstring, metadata=metadata)

    # This makes `pg.typing.Annotated` compatible with third-party type checking
    # solutions.
    if typing.TYPE_CHECKING:
      return annotated.value_spec.annotation
    return annotated
