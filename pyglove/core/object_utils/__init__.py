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
"""Utility library for handling hierarchical Python objects.

Overview
--------

``pg.object_utils`` facilitates the handling of hierarchical
Python objects. It sits at the bottom of all PyGlove modules and empowers other
modules with the following features:

  +---------------------+----------------------------------------------+
  | Functionality       | API                                          |
  +=====================+==============================================+
  | Formatting          | :class:`pyglove.Formattable`,                |
  |                     |                                              |
  |                     | :func:`pyglove.format`,                      |
  |                     |                                              |
  |                     | :func:`pyglove.print`,                       |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.kvlist_str`,     |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.quote_if_str`,   |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.message_on_path` |
  +---------------------+----------------------------------------------+
  | Serialization       | :class:`pyglove.JSONConvertible`             |
  +---------------------+----------------------------------------------+
  | Partial construction| :class:`pyglove.MaybePartial`,               |
  |                     |                                              |
  |                     | :const:`pyglove.MISSING_VALUE`               |
  +---------------------+----------------------------------------------+
  | Hierarchical key    | :class:`pyglove.KeyPath`                     |
  | representation      |                                              |
  +---------------------+----------------------------------------------+
  | Hierarchical object | :func:`pyglove.object_utils.traverse`        |
  | traversal           |                                              |
  +---------------------+----------------------------------------------+
  | Hierarchical object | :func:`pyglove.object_utils.transform`,      |
  | transformation      |                                              |
  |                     | :func:`pyglove.object_utils.merge`,          |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.canonicalize`,   |
  |                     |                                              |
  |                     | :func:`pyglove.object_utils.flatten`         |
  +---------------------+----------------------------------------------+
"""

# pylint: disable=g-bad-import-order

# Common traits.
from pyglove.core.object_utils.common_traits import Nestable
from pyglove.core.object_utils.common_traits import JSONValueType

from pyglove.core.object_utils.common_traits import JSONConvertible
from pyglove.core.object_utils.common_traits import Formattable
from pyglove.core.object_utils.common_traits import MaybePartial
from pyglove.core.object_utils.common_traits import Functor

# Value location.
from pyglove.core.object_utils.value_location import KeyPath
from pyglove.core.object_utils.value_location import StrKey

# Value markers.
from pyglove.core.object_utils.missing import MissingValue
from pyglove.core.object_utils.missing import MISSING_VALUE

# Handling hierarchical.
from pyglove.core.object_utils.hierarchical import traverse
from pyglove.core.object_utils.hierarchical import transform
from pyglove.core.object_utils.hierarchical import flatten
from pyglove.core.object_utils.hierarchical import canonicalize
from pyglove.core.object_utils.hierarchical import merge
from pyglove.core.object_utils.hierarchical import merge_tree
from pyglove.core.object_utils.hierarchical import is_partial
from pyglove.core.object_utils.hierarchical import try_listify_dict_with_int_keys

# Handling formatting.
from pyglove.core.object_utils.formatting import format            # pylint: disable=redefined-builtin
from pyglove.core.object_utils.formatting import printv as print   # pylint: disable=redefined-builtin
from pyglove.core.object_utils.formatting import kvlist_str
from pyglove.core.object_utils.formatting import quote_if_str
from pyglove.core.object_utils.formatting import comma_delimited_str
from pyglove.core.object_utils.formatting import auto_plural
from pyglove.core.object_utils.formatting import message_on_path
from pyglove.core.object_utils.formatting import BracketType
from pyglove.core.object_utils.formatting import bracket_chars

# Handling code generation.
from pyglove.core.object_utils.codegen import make_function


# pylint: enable=g-bad-import-order
