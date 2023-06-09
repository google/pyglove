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
# pylint: disable=line-too-long
"""Utility library that provides common traits for objects in Python.

Overview
--------

``pg.object_utils`` sits at the bottom of all PyGlove modules and empowers other
modules with the following features:

  +---------------------+--------------------------------------------+
  | Functionality       | API                                        |
  +=====================+============================================+
  | Formatting          | :class:`pg.Formattable`,                   |
  |                     |                                            |
  |                     | :func:`pg.format`,                         |
  |                     |                                            |
  |                     | :func:`pg.print`,                          |
  |                     |                                            |
  |                     | :func:`pg.object_utils.kvlist_str`,        |
  |                     |                                            |
  |                     | :func:`pg.object_utils.quote_if_str`,      |
  |                     |                                            |
  |                     | :func:`pg.object_utils.message_on_path`    |
  +---------------------+--------------------------------------------+
  | Serialization       | :class:`pg.JSONConvertible`,               |
  |                     |                                            |
  |                     | :func:`pg.registered_types`,               |
  |                     |                                            |
  |                     | :func:`pg.object_utils.to_json`,           |
  |                     |                                            |
  |                     | :func:`pg.object_utils.from_json`,         |
  +---------------------+--------------------------------------------+
  | Partial construction| :class:`pg.MaybePartial`,                  |
  |                     |                                            |
  |                     | :const:`pg.MISSING_VALUE`.                 |
  +---------------------+--------------------------------------------+
  | Hierarchical key    | :class:`pg.KeyPath`                        |
  | representation      |                                            |
  +---------------------+--------------------------------------------+
  | Hierarchical object | :func:`pg.object_utils.traverse`           |
  | traversal           |                                            |
  +---------------------+--------------------------------------------+
  | Hierarchical object | :func:`pg.object_utils.transform`,         |
  | transformation      |                                            |
  |                     | :func:`pg.object_utils.merge`,             |
  |                     |                                            |
  |                     | :func:`pg.object_utils.canonicalize`,      |
  |                     |                                            |
  |                     | :func:`pg.object_utils.flatten`            |
  +---------------------+--------------------------------------------+
  | Code generation     | :class:`pg.object_utils.make_function`     |
  +---------------------+--------------------------------------------+
  | Docstr handling     | :class:`pg.docstr`,                        |
  +---------------------+--------------------------------------------+
"""
# pylint: enable=line-too-long
# pylint: disable=g-bad-import-order

# Common traits.
from pyglove.core.object_utils.common_traits import Nestable
from pyglove.core.object_utils.common_traits import JSONValueType

from pyglove.core.object_utils.common_traits import JSONConvertible
from pyglove.core.object_utils.common_traits import from_json
from pyglove.core.object_utils.common_traits import to_json

from pyglove.core.object_utils.common_traits import Formattable
from pyglove.core.object_utils.common_traits import MaybePartial
from pyglove.core.object_utils.common_traits import Functor

from pyglove.core.object_utils.common_traits import registered_types

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

# Handling thread local values.
from pyglove.core.object_utils.thread_local import thread_local_value_scope
from pyglove.core.object_utils.thread_local import thread_local_has
from pyglove.core.object_utils.thread_local import thread_local_set
from pyglove.core.object_utils.thread_local import thread_local_get
from pyglove.core.object_utils.thread_local import thread_local_del
from pyglove.core.object_utils.thread_local import thread_local_increment
from pyglove.core.object_utils.thread_local import thread_local_decrement
from pyglove.core.object_utils.thread_local import thread_local_push
from pyglove.core.object_utils.thread_local import thread_local_pop

# Handling docstrings.
from pyglove.core.object_utils.docstr_utils import DocStr
from pyglove.core.object_utils.docstr_utils import DocStrStyle
from pyglove.core.object_utils.docstr_utils import DocStrEntry
from pyglove.core.object_utils.docstr_utils import DocStrExample
from pyglove.core.object_utils.docstr_utils import DocStrArgument
from pyglove.core.object_utils.docstr_utils import DocStrReturns
from pyglove.core.object_utils.docstr_utils import DocStrRaises
from pyglove.core.object_utils.docstr_utils import docstr

# pylint: enable=g-bad-import-order
