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
"""Utilities for code generation."""

from typing import Any, Dict, List, Optional
from pyglove.core.object_utils.missing import MISSING_VALUE


def make_function(
    name: str,
    args: List[str],
    body: List[str],
    *,
    exec_globals: Optional[Dict[str, Any]] = None,
    exec_locals: Optional[Dict[str, Any]] = None,
    return_type: Any = MISSING_VALUE):
  """Creates a function dynamically from source."""
  if exec_locals is None:
    exec_locals = {}
  if return_type != MISSING_VALUE:
    exec_locals['_return_type'] = return_type
    return_annotation = '->_return_type'
  else:
    return_annotation = ''
  args = ', '.join(args)
  body = '\n'.join(f'  {line}' for line in body)
  fn_def = f' def {name}({args}){return_annotation}:\n{body}'
  local_vars = ', '.join(exec_locals.keys())
  fn_def = f'def _make_fn({local_vars}):\n{fn_def}\n return {name}'
  ns = {}
  exec(fn_def, exec_globals, ns)  # pylint: disable=exec-used
  return ns['_make_fn'](**exec_locals)
