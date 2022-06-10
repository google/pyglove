# Copyright 2019 The PyGlove Authors
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
"""Setup for pip package."""

from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('pyglove/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g['__version__']
    raise ValueError('`__version__` not defined in `pyglove/__init__.py`')


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


_VERSION = _get_version()

setup(
    name='pyglove',
    version=_VERSION,
    url='https://github.com/google/pyglove',
    license='Apache License 2.0',
    author='PyGlove Authors',
    description='PyGlove: A library for manipulating Python objects.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='pyglove-authors@google.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(include=['pyglove*']),
    install_requires=[],
    extras_require={},
    requires_python='>=3.7',
    include_package_data=True,
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='ai machine learning automl mutable symbolic framework meta-programming',
)
