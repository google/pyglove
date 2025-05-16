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
#
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
"""Sphinx configuration."""

import importlib
import inspect
import os
import re
import sys

from sphinx.domains import python as sphinx_python

PyXRefRole = sphinx_python.PyXRefRole


# Set parent directory to path in order to import pyglove.
sys.path.insert(0, os.path.abspath('..'))
pyglove_module = importlib.import_module('pyglove')
access_path_to_api = {}


def generate_api_docs(_):
  docgen = importlib.import_module('docs.api.docgen')
  print('Generating API docs from templates...')
  pyglove_api = docgen.generate_api_docs()
  for api in pyglove_api.all_apis():
    if api.qualname:
      access_path_to_api[api.qualname] = api
    for path in api.access_paths:
      access_path_to_api[path] = api


def setup(app):
  app.connect('builder-inited', generate_api_docs)
  app.add_role_to_domain('py', 'class', PgXRefRole())
  app.add_role_to_domain('py', 'const', PgXRefRole())
  app.add_role_to_domain('py', 'obj', PgXRefRole())
  app.add_role_to_domain('py', 'data', PgXRefRole())
  app.add_role_to_domain('py', 'func', PgXRefRole())
  app.add_role_to_domain('py', 'meth', PgXRefRole())
  app.add_role_to_domain('py', 'attr', PgXRefRole())
  app.add_role_to_domain('py', 'mod', PgXRefRole())

# Consider to include versioning.
branch = os.getenv('BRANCH') or 'main'
_GIT_ROOT = f'https://github.com/google/pyglove/blob/{branch}'


def get_git_location(module_name, entity_name):
  """Get Github location for a symbol."""
  segs = module_name.split('.')
  if not segs or segs[0] not in ['pyglove', 'pg']:
    return None
  module = pyglove_module
  for seg in segs[1:]:
    module = getattr(module, seg, None)
    if module is None:
      return None
  if '.' in entity_name:
    cls_name, attrname = entity_name.split('.')
    cls = getattr(module, cls_name, None)
    target = getattr(cls, attrname, None)
    if hasattr(target, 'fget'):
      target = target.fget
  else:
    target = getattr(module, entity_name, None)

  print(f'Resolving {module_name}.{entity_name}...')
  if target is None:
    return None
  try:
    file = inspect.getsourcefile(target)
    lines = inspect.getsourcelines(target)
  except TypeError:
    # e.g. target is constant number, etc.
    return None
  file = os.path.relpath(file, os.path.abspath('..'))
  if not file.startswith('pyglove'):
    return None
  start, end = lines[1], lines[1] + len(lines[0]) - 1
  return f'{_GIT_ROOT}/{file}#L{start}-L{end}'


def linkcode_resolve(domain, info):
  """Link source code to github source files."""
  assert domain == 'py', 'expected only Python objects'
  module_name, entity_name = info['module'], info['fullname']
  try:
    return get_git_location(module_name, entity_name)
  except Exception as e:  # pylint: disable=broad-except
    print(f'Found error when resolving {module_name}.{entity_name}: {e}')
    return None


class PgXRefRole(PyXRefRole):
  """Custom XRefRole for PyGlove code.

  This role is introduced to consistently use the preferred name for the same
  symbol, though they could be referenced with different paths. For example,
  both :class:`pg.DNA` and :class:`pyglove.geno.DNA` will refer to class
  ``pg.DNA``, while "pg.DNA" will be used as the the title for both references.
  """

  def process_link(self, env, refnode, has_explicit_title: bool,
                   title: str, target: str) -> tuple[str, str]:
    """Processes link."""
    title, target = super().process_link(
        env, refnode, has_explicit_title, title, target)

    def noramlized_title_and_target(api, attr_name=None):
      new_title = title
      if not has_explicit_title:
        new_title = api.preferred_path
      new_target = api.canonical_path
      if attr_name:
        new_target = '.'.join([new_target, attr_name])
      new_target = re.sub(r'^pg\.', 'pyglove.', new_target)
      return (new_title, new_target)

    # Try with target directly.
    if target in access_path_to_api:
      return noramlized_title_and_target(access_path_to_api[target])

    # Replace target prefix and try again.
    target = re.sub(r'^pyglove\.(?:core|ext)?\.?', 'pg.', target)
    if target in access_path_to_api:
      return noramlized_title_and_target(access_path_to_api[target])
    else:
      # new_target may be class members.
      name_items = target.split('.')
      class_name, attr_name = '.'.join(name_items[:-1]), name_items[-1]
      if class_name in access_path_to_api:
        return noramlized_title_and_target(
            access_path_to_api[class_name], attr_name)
    return (title, target)


# -- Project information -----------------------------------------------------

project = 'PyGlove'
copyright = 'Copyright 2023, The PyGlove Authors'    # pylint: disable=redefined-builtin
author = 'The PyGlove Authors'

# The short X.Y version
version = ''

# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # pip install autodocsumm
    'autodocsumm',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'myst_nb'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}


suppress_warnings = [
    # 'ref.citation',  # Many duplicated citations in numpy/scipy docstrings.
    # 'ref.footnote',  # Many unreferenced footnotes in numpy/scipy docstrings
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = ['.rst', '.ipynb', '.md']

# The main toctree document.
main_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # Temporarily disable sources for faster sphinx-build
    # 'api/core/symbolic',
    # 'api/core/utils',
    # 'api/core/typing',
    # 'api/core/detouring',
    # 'api/core/wrapping',
    # 'api/core/patching',
    # 'api/core/geno',
    # 'api/core/hyper',
    # 'api/core/tuning',
    # 'api/ext/evolution',
    # 'api/ext/scalars',
    # 'api/ext/early_stopping',
    # Manual sections.
    # 'overview',
    # 'symbolic_oop',
    # 'smart_programs',
    # 'symbolic_computing',
    # 'topics',
    # Sometimes sphinx reads its own outputs as inputs!
    'build/html',
    'build/jupyter_execute',
    # Ignore markdown source for notebooks; myst-nb builds from the ipynb
    # These are kept in sync using the jupytext pre-commit hook.
    'notebooks/*.md',
    # Ignore API templates.
    'api/_templates',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

html_show_sourcelink = True

autosummary_generate = True

napolean_use_rtype = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'furo'
html_title = 'PyGlove'

html_static_path = ['_static']
html_theme_options = {
    # furo
    'light_logo': 'logo_light.svg',
    'dark_logo': 'logo_dark.svg',
    'sidebar_hide_name': True,
    'navigation_with_keys': True,

    'light_css_variables': {
        'sidebar-item-font-size': '81.25%',
        'code-font-size': '75%'
    },
}
html_favicon = '_static/favicon.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
# always_document_param_types = True

add_module_names = False

jupyter_execute_notebooks = 'off'
