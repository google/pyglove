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
"""Representing missing value for a field."""

from typing import Any, Dict
from pyglove.core.utils import formatting
from pyglove.core.utils import json_conversion


class MissingValue(formatting.Formattable, json_conversion.JSONConvertible):
  """Value placeholder for an unassigned attribute."""

  def format(self, *args, **kwargs):  # pytype: disable=signature-mismatch
    return 'MISSING_VALUE'

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, MissingValue)

  def __hash__(self) -> int:
    return hash(MissingValue.__module__ + MissingValue.__name__)

  def to_json(self, **kwargs) -> Dict[str, Any]:
    return self.to_json_dict(fields=dict(), **kwargs)


# A shortcut global object (constant) for referencing MissingValue.
MISSING_VALUE = MissingValue()
