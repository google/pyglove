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

"""Core PyGlove.

As the prefix implies, this package is the minimal PyGlove. The purpose of
having this layer is to allow users to use PyGlove's core functionalities
without extra dependencies beyond the Python interpreter. Therefore, it can be
a lightweight Swiss-Army knife for general Python programming.

Though introduced for AutoML, PyGlove is general designed and implemented for
enabling program search on an arbitrary Python program. Its method was first
published at NeurIPS 2020 (https://arxiv.org/abs/2101.08809). The core PyGlove
builds its capabilities around Symbolic Programming, including but not limited
to symbolic program declaration, program manipulation, program generation and
runtime type checking. It also introduced features in advanced function binding
and class detouring, which may be applicable to general Python coding scenarios.

Here lists the sub-modules included in the core PyGlove library:

  pyglove/core
   |__ symbolic        :  Symbolic program constructs and operations.
   |__ typing          :  Runtime type check and value validation.
   |__ geno            :  Genome types for program generation/manipulation.
   |__ hyper           :  Encoding/Decoding between a program and a genome.
   |__ tuning          :  Interface for program tuning with a local backend.
   |__ detouring       :  Detouring classes creation without symbolic types.
   |__ patching        :  Patching a program with URL-like strings.
   |__ object_utils    :  Utility libary on operating with Python objects.

"""

# NOTE(daiyip): We disable bad-import-order to preserve the relation of
# imported symbols
# pylint: disable=g-bad-import-order
# pylint: disable=unused-import
# pylint: disable=reimported
# pylint: disable=g-import-not-at-top


#
# Symbols from 'symbolic' sub-module.
#


from pyglove.core import symbolic

# Global flags.
allow_empty_field_description = symbolic.allow_empty_field_description
allow_repeated_class_registration = symbolic.allow_repeated_class_registration
set_origin_stacktrace_limit = symbolic.set_origin_stacktrace_limit

# Context manager for scoped flags.
allow_partial = symbolic.allow_partial
allow_writable_accessors = symbolic.allow_writable_accessors
auto_call_functors = symbolic.auto_call_functors
notify_on_change = symbolic.notify_on_change
enable_type_check = symbolic.enable_type_check
track_origin = symbolic.track_origin
as_sealed = symbolic.as_sealed

# Symbolic types.
Symbolic = symbolic.Symbolic
PureSymbolic = symbolic.PureSymbolic

Dict = symbolic.Dict
dict = Dict   # pylint: disable=redefined-builtin

List = symbolic.List
list = List   # pylint: disable=redefined-builtin

Object = symbolic.Object
ClassWrapper = symbolic.ClassWrapper
Functor = symbolic.Functor

# Inferentiable base classes.
Inferentiable = symbolic.Inferential
InferredValue = symbolic.InferredValue

# Reference types.
Ref = symbolic.Ref
maybe_ref = symbolic.maybe_ref

# Decorator for declaring symbolic. members for `pg.Object`.
members = symbolic.members

# Decorator for updating the __init__ signature of `pg.Object`.
use_init_args = symbolic.use_init_args

#
# Methods for making symbolic types.
#

# Function/Decorator for symbolizing existing types.
symbolize = symbolic.symbolize

# Decorator for converting a function into `pg.Functor`.
functor = symbolic.functor

# Method for making a functor class out from a function.
functor_class = symbolic.functor_class

# Function for wrapping a single class.
wrap = symbolic.wrap

# Function for wrapping multiple classes under a module.
wrap_module = symbolic.wrap_module

# Decorator for creating compound class.
compound = symbolic.compound

# Function for making compound class.
compound_class = symbolic.compound_class

# Method for declaring a boilerplated class from a symbolic instance.
boilerplate_class = symbolic.boilerplate_class


#
# Context manager for swapping wrapped class with their wrappers.
#

apply_wrappers = symbolic.apply_wrappers


# Methods for symbolic operations.
eq = symbolic.eq
ne = symbolic.ne
lt = symbolic.lt
gt = symbolic.gt
hash = symbolic.hash  # pylint: disable=redefined-builtin
clone = symbolic.clone

# Methods for querying symbolic types.
TraverseAction = symbolic.TraverseAction
traverse = symbolic.traverse
query = symbolic.query
contains = symbolic.contains

is_abstract = symbolic.is_abstract
is_pure_symbolic = symbolic.is_pure_symbolic
is_deterministic = symbolic.is_deterministic

# Method for differentiating symbolic types.
Diff = symbolic.Diff
diff = symbolic.diff

# Methods for serializing/deserializing symbolic types.
from_json = symbolic.from_json
from_json_str = symbolic.from_json_str
to_json = symbolic.to_json
to_json_str = symbolic.to_json_str
save = symbolic.save
load = symbolic.load
get_load_handler = symbolic.get_load_handler
set_load_handler = symbolic.set_load_handler
get_save_handler = symbolic.get_save_handler
set_save_handler = symbolic.set_save_handler

# Auxiliary classes:
Origin = symbolic.Origin
FieldUpdate = symbolic.FieldUpdate
Insertion = symbolic.Insertion
WritePermissionError = symbolic.WritePermissionError

#
# Symbols from 'typing' sub-module.
#

# NOTE(daiyip): we introduce 'typing' as an alias for 'schema' sub-module, since
# it may be easier to comprehend when users use pytype.
from pyglove.core import typing

# Promote the following concepts to top level as they may be used in pytype
# annotation frequently.
KeySpec = typing.KeySpec
ValueSpec = typing.ValueSpec
Field = typing.Field
Schema = typing.Schema
CustomTyping = typing.CustomTyping

get_converter = typing.get_converter
register_converter = typing.register_converter
get_signature = typing.get_signature


#
# Symbols from 'geno' sub-module.
#

from pyglove.core import geno

DNA = geno.DNA
DNASpec = geno.DNASpec
DNAGenerator = geno.DNAGenerator

random_dna = geno.random_dna


#
# Symbols from 'hyper' sub-module.
#

from pyglove.core import hyper

# Methods for creating hyper values.
template = hyper.template

oneof = hyper.oneof
manyof = hyper.manyof
permutate = hyper.permutate
floatv = hyper.floatv
evolve = hyper.evolve

# Aliases for backward compatibility.
one_of = hyper.one_of
sublist_of = hyper.sublist_of
float_value = hyper.float_value
search_space = hyper.search_space

# Helper methods for operating with hyper values.
dna_spec = hyper.search_space
materialize = hyper.materialize

iter = hyper.iterate  # pylint: disable=redefined-builtin
random_sample = hyper.random_sample


#
# Symbols from 'tuning' sub-module.
#

from pyglove.core import tuning
sample = tuning.sample
poll_result = tuning.poll_result


#
# Symbols from 'detouring' sub-module.
#

from pyglove.core import detouring
detour = detouring.detour


#
# Symbols from 'patching' sub-module.
#

from pyglove.core import patching

patch = patching.patch
patcher = patching.patcher
patch_on_key = patching.patch_on_key
patch_on_path = patching.patch_on_path
patch_on_value = patching.patch_on_value
patch_on_type = patching.patch_on_type
patch_on_member = patching.patch_on_member
ObjectFactory = patching.ObjectFactory


#
# Symbols from 'object_utils' sub-module.
#

from pyglove.core import object_utils
KeyPath = object_utils.KeyPath
MISSING_VALUE = object_utils.MISSING_VALUE

Formattable = object_utils.Formattable
MaybePartial = object_utils.MaybePartial
JSONConvertible = object_utils.JSONConvertible
DocStr = object_utils.DocStr

registered_types = object_utils.registered_types
explicit_method_override = object_utils.explicit_method_override

is_partial = object_utils.is_partial
format = object_utils.format   # pylint: disable=redefined-builtin
print = object_utils.print    # pylint: disable=redefined-builtin
docstr = object_utils.docstr
catch_errors = object_utils.catch_errors

#
# Symbols from `io` sub-module.
#

from pyglove.core import io


#
# Symbols from `logging.py`.
#

from pyglove.core import logging

# pylint: enable=g-import-not-at-top
# pylint: enable=reimported
# pylint: enable=unused-import
# pylint: enable=g-bad-import-order
