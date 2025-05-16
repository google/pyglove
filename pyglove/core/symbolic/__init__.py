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
"""Symbolic Object Model.

This module enables Symbolic Object Model, which defines and implements the
symbolic interfaces for common Python types (e.g. symbolic class, symbolic
function and symbolic container types). Based on symbolic types, symbolic
objects can be created, which can be then inspected, manipulated symbolically.
"""

# pylint: disable=g-bad-import-order

# Global flags.
from pyglove.core.symbolic.flags import allow_empty_field_description
from pyglove.core.symbolic.flags import is_empty_field_description_allowed

from pyglove.core.symbolic.flags import allow_repeated_class_registration
from pyglove.core.symbolic.flags import is_repeated_class_registration_allowed

from pyglove.core.symbolic.flags import set_origin_stacktrace_limit
from pyglove.core.symbolic.flags import get_origin_stacktrace_limit

from pyglove.core.symbolic.flags import set_load_handler
from pyglove.core.symbolic.flags import get_load_handler

from pyglove.core.symbolic.flags import set_save_handler
from pyglove.core.symbolic.flags import get_save_handler

# Context managers.

from pyglove.core.symbolic.flags import track_origin
from pyglove.core.symbolic.flags import is_tracking_origin

from pyglove.core.symbolic.flags import enable_type_check
from pyglove.core.symbolic.flags import is_type_check_enabled

from pyglove.core.symbolic.flags import allow_writable_accessors
from pyglove.core.symbolic.flags import is_under_accessor_writable_scope

from pyglove.core.symbolic.flags import as_sealed
from pyglove.core.symbolic.flags import is_under_sealed_scope

from pyglove.core.symbolic.flags import allow_partial
from pyglove.core.symbolic.flags import is_under_partial_scope

from pyglove.core.symbolic.flags import notify_on_change
from pyglove.core.symbolic.flags import is_change_notification_enabled

from pyglove.core.symbolic.flags import auto_call_functors
from pyglove.core.symbolic.flags import should_call_functors_during_init

# Symbolic types and their definition helpers.
from pyglove.core.symbolic.base import Symbolic
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.dict import Dict

from pyglove.core.symbolic.object import ObjectMeta
from pyglove.core.symbolic.object import Object
from pyglove.core.symbolic.object import members
from pyglove.core.symbolic.object import use_init_args

from pyglove.core.symbolic.functor import Functor
from pyglove.core.symbolic.functor import functor
from pyglove.core.symbolic.functor import functor_class
from pyglove.core.symbolic.functor import as_functor

from pyglove.core.symbolic.class_wrapper import ClassWrapper
from pyglove.core.symbolic.class_wrapper import wrap
from pyglove.core.symbolic.class_wrapper import wrap_module
from pyglove.core.symbolic.class_wrapper import apply_wrappers

from pyglove.core.symbolic.symbolize import symbolize

from pyglove.core.symbolic.compounding import compound
from pyglove.core.symbolic.compounding import compound_class
from pyglove.core.symbolic.boilerplate import boilerplate_class

from pyglove.core.symbolic.contextual_object import ContextualObject
from pyglove.core.symbolic.contextual_object import ContextualAttribute
from pyglove.core.symbolic.contextual_object import contextual_attribute

# Inferential types.
from pyglove.core.symbolic.base import Inferential

from pyglove.core.symbolic.inferred import InferredValue
from pyglove.core.symbolic.inferred import ValueFromParentChain

# Reference type.
from pyglove.core.symbolic.ref import Ref
from pyglove.core.symbolic.ref import maybe_ref
from pyglove.core.symbolic.ref import deref

# Symbolic operations.
from pyglove.core.symbolic.base import traverse
from pyglove.core.symbolic.base import query

from pyglove.core.symbolic.base import eq
from pyglove.core.symbolic.base import ne
from pyglove.core.symbolic.base import lt
from pyglove.core.symbolic.base import gt
from pyglove.core.symbolic.base import sym_hash as hash  # pylint: disable=redefined-builtin
from pyglove.core.symbolic.base import contains
from pyglove.core.symbolic.diff import diff

from pyglove.core.symbolic.base import is_deterministic
from pyglove.core.symbolic.base import is_pure_symbolic
from pyglove.core.symbolic.base import is_abstract

from pyglove.core.symbolic.base import clone
from pyglove.core.symbolic.base import from_json
from pyglove.core.symbolic.base import from_json_str
from pyglove.core.symbolic.base import to_json
from pyglove.core.symbolic.base import to_json_str
from pyglove.core.symbolic.base import load
from pyglove.core.symbolic.base import save
from pyglove.core.symbolic.base import open_jsonl

# Interfaces for pure symbolic objects.
from pyglove.core.symbolic.pure_symbolic import PureSymbolic
from pyglove.core.symbolic.pure_symbolic import NonDeterministic

# Symbolic helper classes.
from pyglove.core.symbolic.base import FieldUpdate
from pyglove.core.symbolic.base import DescendantQueryOption
from pyglove.core.symbolic.base import TraverseAction
from pyglove.core.symbolic.list import Insertion
from pyglove.core.symbolic.diff import Diff
from pyglove.core.symbolic.origin import Origin

# Symbolic helper methods.
from pyglove.core.symbolic.base import default_load_handler
from pyglove.core.symbolic.base import default_save_handler
from pyglove.core.symbolic.list import mark_as_insertion

# Error types.
from pyglove.core.symbolic.base import WritePermissionError
from pyglove.core.symbolic.error_info import ErrorInfo

# pylint: enable=g-bad-import-order
