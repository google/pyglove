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
"""Sphinx source files (.rst) generation script for API references."""

import abc
import dataclasses
import importlib
import inspect
import os
import re
import sys
import typing
from typing import Dict, Iterator, List, Optional, Set

from absl import app
from absl import flags
import jinja2


flags.DEFINE_string(
    'import_root',
    '.',
    'Root directory for importing PyGlove.')

flags.DEFINE_string(
    'template_root',
    'docs/api/_templates',
    'Root directory for doc templates.')

flags.DEFINE_string(
    'output_root',
    'docs/api',
    'Root directory for generated files.')

flags.DEFINE_bool(
    'overwrite',
    False,
    'Force overwrite existing .rst files in the output directory.')


FLAGS = flags.FLAGS


MODULES_TO_DOCUMENT = frozenset({
    # Core.
    'detouring',
    'geno',
    'hyper',
    'io',
    'patching',
    'symbolic',
    'views',
    'tuning',
    'typing',
    'utils',
    # Ext.
    'early_stopping',
    'evolution',
    'evolution.selectors',
    'evolution.mutators',
    'evolution.recombinators',
    'mutfun',
    'scalars',
    # generators.
    'generators',
})


def to_snake_case(maybe_camle_case_str):
  """Converts a maybe camle-case string to snake case."""
  s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', maybe_camle_case_str)
  s = re.sub('__([A-Z])', r'_\1', s)
  s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s)
  return s.lower()


class Templates:
  """Api document templates."""

  def __init__(self, template_root_dir: str):
    self._template_root = template_root_dir
    self._default_template_root = os.path.join(self._template_root, '_default')
    self._template_cache = {}

  def get_template(self, api: 'Api') -> jinja2.Template:
    """Returns the template for an API entry."""
    relative_path = api.doc_handle + '.rst'
    template_path = os.path.join(self._template_root, relative_path)
    if os.path.exists(template_path):
      return self._load_template(template_path)
    return self._load_default_template(api.default_template_filename)

  def _load_template(self, template_filepath):
    """Loads a template by file."""
    if template_filepath not in self._template_cache:
      with open(template_filepath, 'r') as f:
        t = jinja2.Template(f.read())
      self._template_cache[template_filepath] = t
    return self._template_cache[template_filepath]

  def _load_default_template(self, filename: str) -> jinja2.Template:
    """Returns default template for an API type."""
    return self._load_template(
        os.path.join(self._default_template_root, filename))


@dataclasses.dataclass
class Api(metaclass=abc.ABCMeta):
  """Base class for an API entry."""
  name: str
  qualname: Optional[str]

  def __post_init__(self):
    self._access_paths: Set[str] = set()
    self._source_category = None

  def add_access_path(self, access_path):
    self._access_paths.add(access_path)

  @property
  def access_paths(self) -> List[str]:
    """Returns all access paths. E.g. ['pg.Dict', 'pg.symbolic.Dict']."""
    def order(path):
      names = path.split('.')
      return (len(names), names[-1] != self.name)
    return sorted(self._access_paths, key=order)

  @property
  def preferred_path(self) -> str:
    """Returns preferred path."""
    return self.access_paths[0]

  @property
  @abc.abstractmethod
  def canonical_path(self) -> str:
    """Returns the canonical path. E.g 'pg.symbolic.Dict' for Dict."""

  @property
  @abc.abstractmethod
  def doc_handle(self) -> str:
    """Returns a doc handle to the api (e.g. 'core/symbolic/dict')."""

  def relative_handle(self, base_handle: str) -> str:
    """Returns the relative handle to a base handle."""
    return os.path.relpath(self.doc_handle, base_handle)

  @property
  def doc_dir(self) -> str:
    dir_segments = self.canonical_path.split('.')[1:]
    if not dir_segments:
      return ''
    if isinstance(self, Leaf):
      dir_segments.pop(-1)
    return os.path.join(self.source_category, *dir_segments)

  @property
  @abc.abstractmethod
  def default_template_filename(self) -> str:
    """Default template file name (under `_templates/_default`)."""

  def set_source_category(self, source_category) -> None:
    self._source_category = source_category

  @property
  @abc.abstractmethod
  def template_variable(self) -> str:
    """Variable name from the template file for referencing this entry."""

  @property
  def source_category(self) -> Optional[str]:
    """Gets source category."""
    if self._source_category is None:
      assert self.qualname is not None, self
      names = self.qualname.split('.')
      if len(names) >= 2:
        assert names[0] in ['pyglove', 'pg']
        self._source_category = names[1]
    return self._source_category

  #
  # Methods/Properties that can be called in .rst files.
  #
  @property
  def rst_label(self) -> str:
    """Api label in .rst files."""
    return self.canonical_path

  @property
  def rst_import_name(self) -> str:
    """Api import name in .rst files. E.g. 'pyglove.symbolic.Dict'."""
    paths = self.canonical_path.split('.')
    assert paths[0] == 'pg'
    paths[0] = 'pyglove'
    return '.'.join(paths)

  @property
  def rst_access_paths(self) -> str:
    """Returns a ReStructureText markdown for access_paths."""
    return ', '.join([f'``{name}``' for name in self.access_paths])


@dataclasses.dataclass
class Leaf(Api):
  """Base class for a non-module API."""

  def __post_init__(self):
    super().__post_init__()
    self._doc_handlename = None

  @property
  def doc_handlename(self) -> Optional[str]:
    return self._doc_handlename

  def set_doc_handlename(self, doc_handlename: str) -> None:
    self._doc_handlename = doc_handlename

  @property
  def canonical_path(self) -> str:
    """Returns the canonical path. E.g 'pg.symbolic.Dict' for Dict."""
    for path in self.access_paths:
      if len(path.split('.')) <= 2:
        continue
      return path
    assert False, self

  @property
  def doc_handle(self) -> str:
    """Returns doc handle (e.g. core/typing/dict) relative to base_handle."""
    assert self.doc_handlename is not None, self
    return os.path.join(self.doc_dir, self.doc_handlename)

  @abc.abstractmethod
  def doc_handlename_suffix_candidates(self) -> Iterator[str]:
    """Returns the suffix candidates for filename if conflict is found."""


@dataclasses.dataclass
class Class(Leaf):
  """API entry for class."""

  def doc_handlename_suffix_candidates(self) -> Iterator[str]:
    yield ''
    yield '_class'
    yield '_the_class'

  @property
  def default_template_filename(self) -> str:
    return 'class.rst'

  @property
  def template_variable(self) -> str:
    return 'cls'


@dataclasses.dataclass
class Function(Leaf):
  """API entry for a function."""

  def doc_handlename_suffix_candidates(self) -> Iterator[str]:
    yield ''
    yield '_function'
    yield '_the_function'

  @property
  def default_template_filename(self) -> str:
    return 'function.rst'

  @property
  def template_variable(self) -> str:
    return 'func'


@dataclasses.dataclass
class Object(Leaf):
  """API entry for a constant."""

  def doc_handlename_suffix_candidates(self) -> Iterator[str]:
    yield ''
    yield '_object'
    yield '_the_object'

  @property
  def default_template_filename(self) -> str:
    return 'object.rst'

  @property
  def template_variable(self) -> str:
    return 'obj'


@dataclasses.dataclass
class NamedEntry:
  name: str
  api: Api


@dataclasses.dataclass
class Module(Api):
  """API entry for a module."""

  def __post_init__(self):
    super().__post_init__()
    self._children: List[NamedEntry] = []
    self._name_to_api: Dict[str, Api] = {}

  def __getitem__(self, key):
    """Returns child API entry."""
    return self._name_to_api[key]   # pytype: disable=attribute-error

  @property
  def template_variable(self) -> str:
    return 'module'

  @property
  def canonical_path(self) -> str:
    if self.access_paths:
      return self.access_paths[0]
    return ''

  @property
  def default_template_filename(self) -> str:
    return 'module.rst'

  @property
  def doc_handle(self) -> str:
    return os.path.join(self.doc_dir, 'index')

  def add_child(self, entry: NamedEntry):
    """Adds a child API entry."""
    self._children.append(entry)
    self._name_to_api[entry.name] = entry.api

    # Object's source category cannot be inferred from qualname, since
    # it does not exist. Therefore, we carry the source category from the
    # parent module.
    if self.source_category is not None and isinstance(entry.api, Object):
      entry.api.set_source_category(self.source_category)

  @property
  def classes(self) -> List[NamedEntry]:
    """Returns all class APIs."""
    return [c for c in self._children if isinstance(c.api, Class)]

  @property
  def functions(self) -> List[NamedEntry]:
    """Returns all class APIs."""
    return [c for c in self._children if isinstance(c.api, Function)]

  @property
  def objects(self) -> List[NamedEntry]:
    """Returns all object APIs."""
    return [c for c in self._children if isinstance(c.api, Object)]

  @property
  def modules(self) -> List[NamedEntry]:
    """Returns all child module APIs."""
    return [c for c in self._children if isinstance(c.api, Module)]

  @property
  def children(self) -> List[NamedEntry]:
    return self._children

  def all_apis(self, memo: Optional[Set[int]] = None) -> List[Leaf]:
    """Returns all leaf APIs."""
    if memo is None:
      memo = set()
    apis = []
    for c in self.children:
      if isinstance(c.api, Leaf) and id(c.api) not in memo:
        apis.append(c.api)
        memo.add(id(c))
    for m in self.modules:
      module_api = m.api
      assert isinstance(module_api, Module)
      apis.extend(module_api.all_apis(memo))
    return apis


def get_api(pg) -> Module:
  """Get API entries from PyGlove module."""
  symbol_to_api = {}

  def get_api_from_module(module, path):
    module_api = Module(path.key, module.__name__)
    module_api.add_access_path(str(path))
    for name in dir(module):
      if name.startswith('_'):
        continue
      child = getattr(module, name)
      child_path = path + name

      child_api = symbol_to_api.get(id(child), None)
      if child_api is None:
        if (inspect.ismodule(child)
            and '.'.join(child_path.keys[1:]) in MODULES_TO_DOCUMENT):
          child_api = get_api_from_module(child, child_path)
        if inspect.isclass(child):
          child_api = Class(
              child.__name__, f'{child.__module__}.{child.__name__}')
        elif inspect.isfunction(child):
          child_api = Function(
              child.__name__, f'{child.__module__}.{child.__name__}')
        elif (not inspect.ismodule(child)
              and not isinstance(child, typing._Final)):  # pytype: disable=module-attr  # pylint: disable=protected-access
          child_api = Object(name, None)
        if child_api:
          symbol_to_api[id(child)] = child_api
      if child_api:
        module_api.add_child(NamedEntry(name, child_api))
        child_api.add_access_path(str(child_path))
    return module_api

  root = get_api_from_module(pg, pg.KeyPath(['pg']))

  # Deciding the filenames for each API.
  filename_cache = {}
  for api in root.all_apis():
    doc_dir = api.doc_dir
    for suffix in api.doc_handlename_suffix_candidates():
      snake_case_name = to_snake_case(api.name)
      fullpath = os.path.join(doc_dir, snake_case_name + suffix)
      if fullpath not in filename_cache or filename_cache[fullpath] is api:
        filename_cache[fullpath] = api
        api.set_doc_handlename(snake_case_name + suffix)
        break
  return root


def generate_doc(
    templates: Templates,
    api: Api,
    output_root: str,
    overwrite: bool = False) -> bool:
  """Generates document (.rst) file for API."""
  api_doc_file = os.path.join(output_root, api.doc_handle + '.rst')
  if not os.path.exists(api_doc_file) or overwrite:
    print(f'Generating {api_doc_file!r} ...')
    t = templates.get_template(api)
    parent_dir = os.path.dirname(api_doc_file)
    if not os.path.exists(parent_dir):
      os.makedirs(parent_dir)
    with open(api_doc_file, 'w') as f:
      kwargs = {api.template_variable: api}
      f.write(t.render(**kwargs))
    return True
  else:
    print(f'Skipping {api_doc_file!r} as it already exists.')
    return False


def generate_api_docs(
    import_root: Optional[str] = None,
    template_root: Optional[str] = None,
    output_root: Optional[str] = None,
    overwrite: bool = True) -> Module:
  """Generate API docs."""

  if import_root is not None:
    # Set parent directory to path in order to import pyglove.
    sys.path.insert(0, import_root)

  if template_root is None:
    template_root = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '_templates')

  if output_root is None:
    output_root = os.path.abspath(os.path.dirname(__file__))

  pg = importlib.import_module('pyglove')
  templates = Templates(template_root)
  stats = pg.Dict(num_files=0, num_skipped=0)
  def docgen(api):
    if not generate_doc(templates, api, output_root, overwrite):
      stats.num_skipped += 1
    stats.num_files += 1

  def visit(module: Module):
    docgen(module)
    for c in module.children:
      if isinstance(c.api, Module):
        visit(c.api)
      else:
        docgen(c.api)

  # api = get_api(pg)
  # e = api['evolution']
  # print(e, e.source_category, e.access_paths, e.canonical_path)
  pyglove_api = get_api(pg)
  visit(pyglove_api)
  print(f'API doc generation completed '
        f'(generated={stats.num_files - stats.num_skipped}, '
        f'skipped={stats.num_skipped})')
  return pyglove_api


def main(*unused_args, **unused_kwargs):
  """Generates PyGlove API docs."""
  import_root = os.path.abspath(FLAGS.import_root)
  template_root = os.path.abspath(FLAGS.template_root)
  output_root = os.path.abspath(FLAGS.output_root)

  print(f'IMPORT_ROOT: {import_root}')
  print(f'TEMPLATE_ROOT: {template_root}')
  print(f'OUTPUT_ROOT: {output_root}')
  generate_api_docs(import_root, template_root, output_root, FLAGS.overwrite)


if __name__ == '__main__':
  app.run(main)
