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
"""Utilities for working with docstrs."""

import dataclasses
import enum
from typing import Any, Dict, List, Optional

import docstring_parser


class DocStrStyle(enum.Enum):
  """Docstring style."""
  REST = 1
  GOOGLE = 2
  NUMPYDOC = 3
  EPYDOC = 4


@dataclasses.dataclass
class DocStrEntry:
  """An entry in a docstring."""
  description: str


@dataclasses.dataclass
class DocStrArgument(DocStrEntry):
  """An entry in the "Args" section of a docstring."""
  name: str
  type_name: Optional[str] = None
  default: Optional[str] = None
  is_optional: Optional[bool] = None


@dataclasses.dataclass
class DocStrReturns(DocStrEntry):
  """An entry in the "Returns"/"Yields" section of a docstring."""
  name: Optional[str] = None
  type_name: Optional[str] = None
  is_yield: bool = False


@dataclasses.dataclass
class DocStrRaises(DocStrEntry):
  """An entry in the "Raises" section of a docstring."""
  type_name: Optional[str] = None


@dataclasses.dataclass
class DocStrExample(DocStrEntry):
  """An entry in the "Examples" section of a docstring."""


@dataclasses.dataclass
class DocStr:
  """Docstring."""
  style: DocStrStyle
  short_description: Optional[str]
  long_description: Optional[str]
  examples: List[DocStrExample]
  args: Dict[str, DocStrArgument]
  returns: Optional[DocStrReturns]
  raises: List[DocStrRaises]
  blank_after_short_description: bool = True

  @classmethod
  def parse(cls, text: str, style: Optional[DocStrStyle] = None) -> 'DocStr':
    """Parses a docstring."""
    result = docstring_parser.parse(text, _to_parser_style(style))
    return cls(
        style=_from_parser_style(result.style),
        short_description=result.short_description,
        long_description=result.long_description,
        examples=[
            DocStrExample(description=e.description)
            for e in result.examples
        ],
        args={  # pylint: disable=g-complex-comprehension
            p.arg_name: DocStrArgument(
                name=p.arg_name, description=p.description,
                type_name=p.type_name, default=p.default,
                is_optional=p.is_optional)
            for p in result.params
        },
        returns=DocStrReturns(  # pylint: disable=g-long-ternary
            name=result.returns.return_name,
            description=result.returns.description,
            is_yield=result.returns.is_generator) if result.returns else None,
        raises=[
            DocStrRaises(type_name=r.type_name, description=r.description)
            for r in result.raises
        ],
        blank_after_short_description=result.blank_after_short_description)


def docstr(symbol: Any) -> Optional[DocStr]:
  """Gets structure docstring of a Python symbol."""
  docstr_text = getattr(symbol, '__doc__', None)
  return DocStr.parse(docstr_text) if docstr_text else None


_PARSER_STYLE_MAPPING = [
    (DocStrStyle.REST, docstring_parser.DocstringStyle.REST),
    (DocStrStyle.GOOGLE, docstring_parser.DocstringStyle.GOOGLE),
    (DocStrStyle.NUMPYDOC, docstring_parser.DocstringStyle.NUMPYDOC),
    (DocStrStyle.EPYDOC, docstring_parser.DocstringStyle.EPYDOC),
]


def _to_parser_style(
    style: Optional[DocStrStyle]) -> docstring_parser.DocstringStyle:
  """Returns parser style from DocStrStyle."""
  if style is None:
    return docstring_parser.DocstringStyle.AUTO
  for s, ps in _PARSER_STYLE_MAPPING:
    if style == s:
      return ps
  raise ValueError(f'Unsupported style {style}.')


def _from_parser_style(
    style: docstring_parser.DocstringStyle) -> DocStrStyle:
  """Returns DocStrStyle from parser style."""
  for s, ps in _PARSER_STYLE_MAPPING:
    if style == ps:
      return s
  raise ValueError(f'Unsupported parser style {style}.')

