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
"""Object factory based on patchers."""

from typing import Any, Callable, Dict, Optional, Type, Union
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core.patching import rule_based


def from_maybe_serialized(
    source: Union[Any, str],
    value_type: Optional[Type[Any]] = None) -> Any:
  """Load value from maybe serialized form (e.g. JSON file or JSON string).

  Args:
    source: Source of value. It can be value (non-string type) itself, or a
      filepath, or a JSON string from where the value will be loaded.
    value_type: An optional type to constrain the value.

  Returns:
    Value from source.
  """
  if isinstance(source, str):
    if source.endswith('.json'):
      value = symbolic.load(source)
    else:
      value = symbolic.from_json_str(source)
  else:
    value = source
  if value_type is not None and not isinstance(value, value_type):
    raise TypeError(
        f'Loaded value {value!r} is not an instance of {value_type!r}.')
  return value


@symbolic.functor()
def ObjectFactory(    # pylint: disable=invalid-name
    value_type: Type[symbolic.Symbolic],
    base_value: Union[symbolic.Symbolic,
                      Callable[[], symbolic.Symbolic],
                      str],
    patches: Optional[rule_based.PatchType] = None,
    params_override: Optional[Union[Dict[str, Any], str]] = None) -> Any:
  """A factory to create symbolic object from a base value and patches.

  Args:
    value_type: Type of return value.
    base_value: An instance of `value_type`,
      or a callable object that produces an instance of `value_type`,
      or a string as the path to the serialized value.
    patches: Optional patching rules. See :func:`patch` for details.
    params_override: A rebind dict (or a JSON string as serialized rebind dict)
      as an additional patch to the value,

  Returns:
    Value after applying `patchers` and `params_override` based on `base_value`.
  """
  # Step 1: Load base value.
  if not isinstance(base_value, value_type) and callable(base_value):
    value = base_value()
  elif isinstance(base_value, str):
    value = symbolic.load(base_value)
  else:
    value = base_value

  if not isinstance(value, value_type):
    raise TypeError(
        f'{base_value!r} is neither an instance of {value_type!r}, '
        f'nor a factory or a path of JSON file that produces an '
        f'instance of {value_type!r}.')

  # Step 2: Patch with patchers if available.
  if patches is not None:
    value = rule_based.patch(value, patches)

  # Step 3: Patch with additional parameter override dict if available.
  if params_override:
    value = value.rebind(
        object_utils.flatten(from_maybe_serialized(params_override, dict)),
        raise_on_no_change=False)
  return value

