# Copyright 2021 The PyGlove Authors
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
"""Symbolic types for reprenting unknown types and objects."""

from typing import Annotated, Any, ClassVar, Literal
from pyglove.core import typing as pg_typing
from pyglove.core import utils
from pyglove.core.symbolic import list as pg_list  # pylint: disable=unused-import
from pyglove.core.symbolic import object as pg_object


class UnknownSymbol(pg_object.Object, pg_typing.CustomTyping):
  """Interface for symbolic representation of unknown symbols."""
  auto_register = False

  def custom_apply(self, *args, **kwargs) -> tuple[bool, Any]:
    """Bypass PyGlove type check."""
    return (False, self)


class UnknownType(UnknownSymbol):
  """Symbolic object for representing unknown types."""

  auto_register = True
  __serialization_key__ = 'unknown_type'

  # TODO(daiyip): Revisit the design on how `pg.typing.Object()` handles
  # UnknownType. This hacky solution should be removed in the future.
  __no_type_check__ = True

  name: str
  args: list[Any] = []

  def sym_jsonify(self, **kwargs) -> utils.JSONValueType:
    json_dict = {'_type': 'type', 'name': self.name}
    if self.args:
      json_dict['args'] = utils.to_json(self.args, **kwargs)
    return json_dict

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs
  ) -> str:
    s = f'<unknown-type {self.name}>'
    if self.args:
      s += f'[{", ".join(repr(x) for x in self.args)}]'
    return s

  def __call__(self, **kwargs):
    return UnknownTypedObject(
        type_name=self.name, **kwargs
    )


class UnknownCallable(UnknownSymbol):
  """Symbolic object for representing unknown callables."""

  auto_register = False
  name: str
  CALLABLE_TYPE: ClassVar[Literal['function', 'method']]

  def sym_jsonify(self, **kwargs) -> utils.JSONValueType:
    return {'_type': self.CALLABLE_TYPE, 'name': self.name}

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs
  ) -> str:
    return f'<unknown-{self.CALLABLE_TYPE} {self.name}>'


class UnknownFunction(UnknownCallable):
  """Symbolic objject for representing unknown functions."""

  auto_register = True
  __serialization_key__ = 'unknown_function'
  CALLABLE_TYPE = 'function'


class UnknownMethod(UnknownCallable):
  """Symbolic object for representing unknown methods."""

  auto_register = True
  __serialization_key__ = 'unknown_method'
  CALLABLE_TYPE = 'method'


class UnknownTypedObject(UnknownSymbol):
  """Symbolic object for representing objects of unknown-type."""

  auto_register = True
  __serialization_key__ = 'unknown_object'

  type_name: str
  __kwargs__: Annotated[
      Any,
      (
          'Fields of the original object will be kept as symbolic attributes '
          'of this object so they can be accessed through `__getattr__`.'
      )
  ]

  def sym_jsonify(self, **kwargs) -> utils.JSONValueType:
    """Converts current object to a dict of plain Python objects."""
    json_dict = self._sym_attributes.to_json(
        exclude_keys=set(['type_name']), **kwargs
    )
    assert isinstance(json_dict, dict)
    json_dict[utils.JSONConvertible.TYPE_NAME_KEY] = self.type_name
    return json_dict

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs
  ) -> str:
    exclude_keys = kwargs.pop('exclude_keys', set())
    exclude_keys.add('type_name')
    kwargs['exclude_keys'] = exclude_keys
    return self._sym_attributes.format(
        compact,
        verbose,
        root_indent,
        cls_name=f'<unknown-type {self.type_name}>',
        key_as_attribute=True,
        bracket_type=utils.BracketType.ROUND,
        **kwargs,
    )
