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

``pg.utils`` sits at the bottom of all PyGlove modules and empowers other
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
  |                     | :func:`pg.utils.kvlist_str`,        |
  |                     |                                            |
  |                     | :func:`pg.utils.quote_if_str`,      |
  |                     |                                            |
  |                     | :func:`pg.utils.message_on_path`    |
  +---------------------+--------------------------------------------+
  | Serialization       | :class:`pg.JSONConvertible`,               |
  |                     |                                            |
  |                     | :func:`pg.registered_types`,               |
  |                     |                                            |
  |                     | :func:`pg.utils.to_json`,           |
  |                     |                                            |
  |                     | :func:`pg.utils.from_json`,         |
  +---------------------+--------------------------------------------+
  | Partial construction| :class:`pg.MaybePartial`,                  |
  |                     |                                            |
  |                     | :const:`pg.MISSING_VALUE`.                 |
  +---------------------+--------------------------------------------+
  | Hierarchical key    | :class:`pg.KeyPath`                        |
  | representation      |                                            |
  +---------------------+--------------------------------------------+
  | Hierarchical object | :func:`pg.utils.traverse`           |
  | traversal           |                                            |
  +---------------------+--------------------------------------------+
  | Hierarchical object | :func:`pg.utils.transform`,         |
  | transformation      |                                            |
  |                     | :func:`pg.utils.merge`,             |
  |                     |                                            |
  |                     | :func:`pg.utils.canonicalize`,      |
  |                     |                                            |
  |                     | :func:`pg.utils.flatten`            |
  +---------------------+--------------------------------------------+
  | Docstr handling     | :class:`pg.docstr`,                        |
  +---------------------+--------------------------------------------+
  | Error handling     | :class:`pg.catch_errors`,                   |
  +---------------------+--------------------------------------------+
"""
# pylint: enable=line-too-long
# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member

# Handling JSON conversion.
from pyglove.core.utils.json_conversion import Nestable
from pyglove.core.utils.json_conversion import JSONValueType

from pyglove.core.utils.json_conversion import JSONConvertible
from pyglove.core.utils.json_conversion import from_json
from pyglove.core.utils.json_conversion import to_json
from pyglove.core.utils.json_conversion import registered_types

# Handling formatting.
from pyglove.core.utils.formatting import Formattable
from pyglove.core.utils.formatting import format  # pylint: disable=redefined-builtin
from pyglove.core.utils.formatting import printv as print  # pylint: disable=redefined-builtin
from pyglove.core.utils.formatting import kvlist_str
from pyglove.core.utils.formatting import quote_if_str
from pyglove.core.utils.formatting import maybe_markdown_quote
from pyglove.core.utils.formatting import comma_delimited_str
from pyglove.core.utils.formatting import camel_to_snake
from pyglove.core.utils.formatting import auto_plural
from pyglove.core.utils.formatting import BracketType
from pyglove.core.utils.formatting import bracket_chars
from pyglove.core.utils.formatting import RawText

# Context managers for defining the default format for __str__ and __repr__.
from pyglove.core.utils.formatting import str_format
from pyglove.core.utils.formatting import repr_format

# Value location.
from pyglove.core.utils.value_location import KeyPath
from pyglove.core.utils.value_location import KeyPathSet
from pyglove.core.utils.value_location import StrKey
from pyglove.core.utils.value_location import message_on_path

# Value markers.
from pyglove.core.utils.missing import MissingValue
from pyglove.core.utils.missing import MISSING_VALUE

# Handling hierarchical.
from pyglove.core.utils.hierarchical import traverse
from pyglove.core.utils.hierarchical import transform
from pyglove.core.utils.hierarchical import flatten
from pyglove.core.utils.hierarchical import canonicalize
from pyglove.core.utils.hierarchical import merge
from pyglove.core.utils.hierarchical import merge_tree
from pyglove.core.utils.hierarchical import is_partial
from pyglove.core.utils.hierarchical import try_listify_dict_with_int_keys

# Common traits.
from pyglove.core.utils.common_traits import MaybePartial
from pyglove.core.utils.common_traits import Functor

from pyglove.core.utils.common_traits import explicit_method_override
from pyglove.core.utils.common_traits import ensure_explicit_method_override

# Handling thread local values.
from pyglove.core.utils.thread_local import thread_local_value_scope
from pyglove.core.utils.thread_local import thread_local_has
from pyglove.core.utils.thread_local import thread_local_set
from pyglove.core.utils.thread_local import thread_local_get
from pyglove.core.utils.thread_local import thread_local_del
from pyglove.core.utils.thread_local import thread_local_increment
from pyglove.core.utils.thread_local import thread_local_decrement
from pyglove.core.utils.thread_local import thread_local_push
from pyglove.core.utils.thread_local import thread_local_pop
from pyglove.core.utils.thread_local import thread_local_peek

# Handling docstrings.
from pyglove.core.utils.docstr_utils import DocStr
from pyglove.core.utils.docstr_utils import DocStrStyle
from pyglove.core.utils.docstr_utils import DocStrEntry
from pyglove.core.utils.docstr_utils import DocStrExample
from pyglove.core.utils.docstr_utils import DocStrArgument
from pyglove.core.utils.docstr_utils import DocStrReturns
from pyglove.core.utils.docstr_utils import DocStrRaises
from pyglove.core.utils.docstr_utils import docstr

# Handling exceptions.
from pyglove.core.utils.error_utils import catch_errors
from pyglove.core.utils.error_utils import CatchErrorsContext
from pyglove.core.utils.error_utils import ErrorInfo

# Timing.
from pyglove.core.utils.timing import timeit
from pyglove.core.utils.timing import TimeIt

# Value override from context manager.
from pyglove.core.utils.contextual import ContextualOverride
from pyglove.core.utils.contextual import contextual_override
from pyglove.core.utils.contextual import with_contextual_override
from pyglove.core.utils.contextual import get_contextual_override
from pyglove.core.utils.contextual import contextual_value
from pyglove.core.utils.contextual import all_contextual_values

# Text color.
from pyglove.core.utils.text_color import colored
from pyglove.core.utils.text_color import colored_block
from pyglove.core.utils.text_color import decolor

# pylint: enable=g-importing-member
# pylint: enable=g-bad-import-order
