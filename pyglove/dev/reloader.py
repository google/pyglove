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
"""Utilities for reloading modules."""

import contextlib
import getpass
import importlib
import inspect
import re
import sys
import time
import types
from typing import Callable, List, Optional, Sequence, Union


def reload(
    module: Union[
        types.ModuleType,                         # Module
        str,                                      # Module name.
        Sequence[Union[types.ModuleType, str]],   # List of module/module names.
        None
    ] = None,   # pylint: disable=bad-whitespace
    *,
    workspace: Optional[str] = None,
    user: Optional[str] = None,
    cl: Optional[int] = None,
    reset_flags: bool = True,
    reload_pattern: str = 'pyglove.*',
    behavior: Optional[str] = 'preferred',
    verbose: bool = False,
    ) -> Union[types.ModuleType, List[types.ModuleType]]:
  """Reloads a module with refreshing its sub-modules based on filter.

  Args:
    module: The root module(s) to reload. If None, module `pyglove` will be
      reloaded.
    workspace: Cider-V workspace to sync code from. If None, use a specific
      CL when `cl` is specified, or sync code from HEAD.
    user: The user LDAP. If None, the current user will be used.
    cl: A Change Number to sync code from. If None, refer to `workspace`.
    reset_flags: If True, removes all the flags in the module that is being
      reloaded. This is to avoid flags being defined twice when reloading.
    reload_pattern: An optional regular expression to whitelist the dependent
      module names that need to be reloaded. If None, it will reload all the
      dependent modules of `module`.
    behavior: The adhoc_import behavior string. Among 'preferred' or None (
      'fallback').
    verbose: If True, print the reloaded sub-modules.

  Returns:
    The reloaded module(s).
  """
  reload_multiple = isinstance(module, (list, tuple))

  if module is None:
    module = sys.modules['pyglove']

  modules = list(module) if isinstance(module, (list, tuple)) else [module]

  regex = re.compile(reload_pattern)
  filter_fn = lambda m: regex.match(m.__name__)

  import_lib = adhoc_import_lib()

  def _reload(m: types.ModuleType):
    try:
      setattr(m, '__reloading__', True)
      if import_lib is None:
        return importlib.reload(m)
      else:
        return import_lib.Reload(m, reset_flags=reset_flags)
    finally:
      delattr(m, '__reloading__')

  start_time = time.time()
  with adhoc_import(workspace, user, cl=cl, behavior=behavior):
    # Step 1: Load module from names.
    for i, m in enumerate(modules):
      if isinstance(m, str):
        if verbose:
          print(f'Loading [{m}]...')
        modules[i] = importlib.import_module(m)

    # Step 2: Compute and reload dependencies.
    for m in module_dependencies(modules, transitive=True, filter=filter_fn):  # pyrefly: ignore[bad-argument-type]
      if verbose:
        print(f'Reloading [{m.__name__}]...')
      _ = _reload(m)

    # Reload the root modules.
    reloaded_modules = []
    for m in modules:
      if verbose:
        print(f'Reloading [{m.__name__}]...')
      reloaded_modules.append(_reload(m))  # pyrefly: ignore[bad-argument-type]

  elapse = time.time() - start_time
  print(f'Sync completed in {elapse:.2f} seconds.')
  return reloaded_modules if reload_multiple else reloaded_modules[0]


_BUILTIN_MODULE_NAMES = frozenset(sys.builtin_module_names)


def module_dependencies(
    module: Union[types.ModuleType, Sequence[types.ModuleType]],
    transitive: bool = False,
    filter: Optional[Callable[[types.ModuleType], bool]] = None  # pylint: disable=redefined-builtin
    ) -> List[types.ModuleType]:
  """Returns a list of module dependencies for a given module."""
  if transitive and not filter:
    raise ValueError(
        '`filter` must be provided when `transitive` is set to True.')

  filter = filter or (lambda m: True)

  dependencies = []
  seen = set()
  max_depth = None if transitive else 1

  def _visit(m: types.ModuleType, depth: int) -> None:
    if max_depth is not None and depth >= max_depth:
      return

    if not hasattr(m, '__file__'):
      return

    try:
      lines = inspect.getsource(m).split('\n')
    except OSError:
      return

    for line in lines:
      symbols = _imported_symbols(line)

      for symbol in symbols:
        dependency = _dependent_module(symbol)
        if not dependency or not filter(dependency):
          continue

        if dependency not in seen:
          seen.add(dependency)
          _visit(dependency, depth + 1)
          dependencies.append(dependency)

  if not isinstance(module, (list, tuple)):
    module = [module]  # pyrefly: ignore[bad-assignment]

  for m in module:
    _visit(m, 0)
  return dependencies


_IMPORT_REGEX = re.compile('^import (.*)')
_FROM_IMPORT_REGEX = re.compile('^from (.*) import (.*)')


def _imported_symbols(import_statement: str) -> List[str]:
  """Gets the fully qualified names of the imported symbols."""
  m = _FROM_IMPORT_REGEX.match(import_statement)
  if m:
    parent_module = m.group(1).strip()
    symbol_names = (
        m.group(2).split(' as ')[0]   # Remove 'as' sub-statements.
        .split('#')[0]                # Remove comments.
        .split(','))
    return [
        f'{parent_module}.{symbol_name.strip()}'
        for symbol_name in symbol_names
    ]

  m = _IMPORT_REGEX.match(import_statement)
  if m:
    symbol_name = (
        m.group(1).split(' as ')[0]   # Remove 'as' sub-statements.
        .split('#')[0]                # Remove comments.
        .split(','))
    return [n.strip() for n in symbol_name]
  return []


def _dependent_module(symbol_name: str):
  """Gets the immediate module for a fully qualified symbol name."""
  if symbol_name.startswith(('_', '.')):
    return None

  module = sys.modules.get(symbol_name)
  if module is None:
    module_name = symbol_name[:symbol_name.rindex('.')]
    if module_name.endswith('_pb2'):
      return None
    module = sys.modules.get(module_name)
  if (module is not None
      and (module.__name__ in _BUILTIN_MODULE_NAMES
           or not hasattr(module, '__file__')
           or module.__name__.endswith('_pb2'))):
    return None
  return module


def adhoc_import(
    workspace: Optional[str],
    user: Optional[str] = None,
    cl: Optional[int] = None,
    behavior: Optional[str] = 'preferred'):
  """Returns a context manager for importing libraries."""
  import_lib = adhoc_import_lib()
  if import_lib is None:
    return contextlib.nullcontext()
  # Placeholder for Google-internal adhoc import logic.


def adhoc_import_lib():
  try:
    _ = get_ipython()  # pytype: disable=name-error
    # pytype: disable=import-error
    from colabtools import adhoc_import as import_lib   # pylint: disable=g-import-not-at-top
    # pytype: enable=import-error
    return import_lib
  except NameError:
    return None
