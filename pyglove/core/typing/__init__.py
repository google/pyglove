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
"""Symbolic typing.

Overview
--------

To enable symbolic programmability on classes and functions, PyGlove intercepts
the assignment operation on the attributes of symbolic objects, to achieve two
goals:

  * Enable automatic type checking, value validation, conversion and
    transformation on attribute values. See `Runtime typing`_ for more details.

  * Allow symbolic attributes to be placeheld by special symbols, in order to
    represent abstract concepts such as a space of objects. E.g. hyper
    primitives like :func:`pyglove.oneof` placeholds a symbolic attribute to
    create a space of values for that attribute. See `Symbolic placeholding`_
    for more details.


Runtime typing
**************

Symbolic objects are intended to be manipulated after creation. Without a
runtime  typing system, things can go wrong easily. For instance, an `int`
attribute which was mistakenly modified at early program stages can be very
difficut to debug at later stages. PyGlove introduces a runtime type system
that automatically validates symbolic objects upon creation and modification,
minimizing boilerplated code for input validation, so the developer can focus on
the main business logic.

Understanding Schema
^^^^^^^^^^^^^^^^^^^^

PyGlove's runtime type system is based on the concept of ``Schema`` (
class :class:`pyglove.Schema`), which defines what symbolic attributes are held
by a symbolic type (e.g. a symbolic dict, a symbolic list or a symbolic class)
and what values each attribute accepts. A ``Schema`` object consists of a list
of ``Field`` (class :class:`pyglove.Field`), which define the acceptable
keys (class :class:`pyglove.KeySpec`) and their values (class
:class:`pyglove.ValueSpec`) for these types. A ``Schema`` object is usually
created automatically and associated with a symbolic type upon its declaration,
through decorators such as :func:`pyglove.members`, :func:`pyglove.symbolize` or
:func:`pyglove.functor`. For example::

  @pg.members([
      ('x', pg.typing.Int(default=1)),
      ('y', pg.typing.Float().noneable())
  ])
  class A(pg.Object):
    pass

  print(A.__schema__)

  @pg.symbolize([
      ('a', pg.typing.Int()),
      ('b', pg.typing.Float())
  ])
  def foo(a, b):
    return a + b

  print(foo.__schema__)


The first argument of all the decorators takes a list of field definitions,
with each described by a tuple of 4 items::

    (key specification, value specification, doc string, field metadata)

The **key specification** and **value specification** are required, while the
doc string and the field metadata are optional. The key specification defines
acceptable identifiers for this field, and the value specification defines the
attribute's value type, its default value, validation rules. The doc string will
serve as the description for the field, and the field metadata can be used for
field-based code generation.

The following code snippet illustrates all supported ``KeySpec`` and
``ValueSpec`` subclasses and their usage with a manually created schema::

    schema = pg.typing.create_schema([
        # Primitive types.
        ('a', pg.typing.Bool(default=True).noneable()),
        ('b', True),       # Equivalent to ('b', pg.typing.Bool(default=True)).
        ('c', pg.typing.Int()),
        ('d', 0),          # Equivalent to ('d', pg.typing.Int(default=0)).
        ('e', pg.typing.Int(
            min_value=0,
            max_value=10).noneable()),
        ('f', pg.typing.Float()),
        ('g', 1.0),        # Equivalent to ('g', pg.typing.Float(default=0.0)).
        ('h', pg.typing.Str()),
        ('i', 'foo'),      # Equivalent to ('i', pg.typing.Str(default='foo').
        ('j', pg.typing.Str(regex='foo.*')),

        # Enum type.
        ('l', pg.typing.Enum('foo', ['foo', 'bar', 0, 1]))

        # List type.
        ('m', pg.typing.List(pg.typing.Int(), size=2, default=[])),
        ('n', pg.typing.List(pg.typing.Dict([
            ('n1', pg.typing.List(pg.typing.Int())),
            ('n2', pg.typing.Str().noneable())
        ]), min_size=1, max_size=10, default=[])),

        # Dict type.
        ('o', pg.typing.Dict([
            ('o1', pg.typing.Int()),
            ('o2', pg.typing.List(pg.typing.Dict([
                ('o21', 1),
                ('o22', 1.0),
            ]))),
            ('o3', pg.typing.Dict([
                # Use of regex key,
                (pg.typing.StrKey('n3.*'), pg.typing.Int())
            ]))
        ]))

        # Tuple type.
        ('p', pg.typing.Tuple([
            ('p1', pg.typing.Int()),
            ('p2', pg.typing.Str())
        ]))

        # Object type.
        ('q', pg.typing.Object(A, default=A()))

        # Type type.
        ('r', pg.typing.Type(int))

        # Callable type.
        ('s', pg.typing.Callable([pg.typing.Int(), pg.typing.Int()],
                                  kw=[('a', pg.typing.Str())])),

        # Functor type (same as Callable, but only for symbolic.Functor).
        ('t', pg.typing.Functor([pg.typing.Str()],
                                 kwargs=[('a', pg.typing.Str())]))

        # Union type.
        ('u', pg.typing.Union([
            pg.typing.Int(),
            pg.typing.Str()
        ], default=1),

        # Any type.
        ('v', pg.typing.Any(default=1))
    ])


Schema inheritance
''''''''''''''''''

Symbolic attributes can be inherited during subclassing. Accordingly, the
schema that defines a symbolic class' attributes can be inherited too by its
subclasses. The fields from the bases' schema will be carried over into the
subclasses' schema, while the subclass can override, by redefining that field
with the same key. The subclass cannot override its base classes' field with
arbitrary value specs, it must be overriding non-frozen fields with more
restrictive validation rules of the same type, or change their default values.
See :meth:`pyglove.ValueSpec.extend` for more details.

The code snippet below illustrates schema inheritance during subclassing::

  @pg.members([
      ('x', pg.typing.Int(min_value=1)),
      ('y', pg.typing.Float()),
  ])
  class A(pg.Object):
    pass

  @pg.members([
      # Further restrict inherited 'x' by specifying the max value, as well
      # as providing a default value.
      ('x', pg.typing.Int(max_value=5, default=2)),
      ('z', pg.typing.Str('foo').freeze())
  ])
  class B(A):
    pass

  assert B.__schema__.fields.keys() == ['x', 'y', 'z']

  @pg.members([
      # Raises: 'z' is frozen in class B and cannot be extended further.
      ('z', pg.typing.Str())
  ])
  class C(B):
    pass


Automatic type conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Type conversion is another mechanism to extend the typing system, which allows
the user to register converters between types. So when a value is assigned to a
attribute whose value specification does not match with the input value type, a
conversion will take place automatically if there is a converter from the input
value type to the type required by the value specification. For example::

  class A:
    def __init__(self, str):
      self._str = str

    def __str__(self):
      return self._str

    def __eq__(self, other):
      return isinstance(other, self.__class__) and self._str == other._str

  pg.typing.register_converter(A, str, str)
  pg.typing.register_converter(str, A, A)

  assert pg.typing.Str().accept(A('abc')) == 'abc'
  assert pg.typing.Object(A).accept('abc') == A('abc')


See :func:`pyglove.typing.register_converter` for more details.


Symbolic placeholding
*********************

Symbolic placeholding enables scenarios that requires an abstract (non-material)
representation of objects - for example - a search space or a partially bound
object. Such objects with placeheld attributes can only be used symbolically,
which means they can be symbolically queried or manipulated, but not yet
materialized to execute the business logic. For example::

  @pg.functor()
  def foo(x, y):
    return x + y

  f = f(x=pg.oneof(range(5)), y=pg.floatv(0.0, 2.0))

  # Error: `x` and `y` are not materialized, thus the functor cannot be
  # executed.
  f()

  # But we can symbolically manipulated it:
  f.rebind(x=1, y=2)

  # Returns 3
  f()


Symbolic placeholding is achieved by the :class:`pyglove.CustomTyping`
interface, which was intended as a mechanism to extend the typing system
horizontally without modifying the :class:`pyglove.ValueSpec` of existing
classes' attributes.

This is done by allowing the `CustomTyping` subclasses to take over value
validation and transformation from an attribute's original ``ValueSpec``
via :meth:`pyglove.CustomTyping.custom_apply` method.

Following is an example of using `CustomTyping` to extend the schema system::

  class FloatTensor(pg.typing.CustomTyping):

    def __init__(self, tensor):
      self._tensor = tensor

    def custom_apply(
        self, path, value_spec, allow_partial, child_transform):
      if isinstane(value_spec, pg.typing.Float):
        # Validate initial tensor, we can also add an TF operator to guard
        # tensor value to go beyond value_spec.min_value and max_value.
        value_spec.apply(self._tensor.numpy())
        # Shortcircuit .apply and returns object itself as final value.
        return (False, self)
      else:
        raise ValueError('FloatTensor can only be applied to float type.')

  @pg.members([
      ('x', pg.typing.Float())
  ])
  class Foo(pg.Object):
    pass

  # FloatTensor can be accepted for all symbolic attributes
  # with Float value spec.
  f = Foo(x=FloatTensor(tf.constant(1.0)))
"""

# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top

# Missing value.
from pyglove.core.typing.typed_missing import MISSING_VALUE  # Non-typed.
from pyglove.core.typing.typed_missing import MissingValue   # Typed.

# Class schema.
from pyglove.core.typing.class_schema import KeySpec
from pyglove.core.typing.class_schema import ValueSpec
from pyglove.core.typing.class_schema import Field
from pyglove.core.typing.class_schema import FieldKeyDef
from pyglove.core.typing.class_schema import FieldValueDef
from pyglove.core.typing.class_schema import FieldDef
from pyglove.core.typing.class_schema import Schema
from pyglove.core.typing.class_schema import create_field
from pyglove.core.typing.class_schema import create_schema
from pyglove.core.typing.class_schema import ForwardRef

# Concrete key specifications.
from pyglove.core.typing.key_specs import ConstStrKey
from pyglove.core.typing.key_specs import NonConstKey
from pyglove.core.typing.key_specs import StrKey
from pyglove.core.typing.key_specs import ListKey
from pyglove.core.typing.key_specs import TupleKey

# Concrete value specifications.
from pyglove.core.typing.value_specs import PrimitiveType
from pyglove.core.typing.value_specs import Bool
from pyglove.core.typing.value_specs import Str
from pyglove.core.typing.value_specs import Number
from pyglove.core.typing.value_specs import Int
from pyglove.core.typing.value_specs import Float
from pyglove.core.typing.value_specs import Enum
from pyglove.core.typing.value_specs import List
from pyglove.core.typing.value_specs import Tuple
from pyglove.core.typing.value_specs import Dict
from pyglove.core.typing.value_specs import Object
from pyglove.core.typing.value_specs import Callable
from pyglove.core.typing.value_specs import Functor
from pyglove.core.typing.value_specs import Type
from pyglove.core.typing.value_specs import Union
from pyglove.core.typing.value_specs import Any

from pyglove.core.typing.value_specs import ensure_value_spec

# Generic type aliases.
from pyglove.core.typing.value_specs import GenericMeta
from pyglove.core.typing.value_specs import Generic
from pyglove.core.typing.value_specs import GenericTypeAlias

from pyglove.core.typing.value_specs import Sequence
from pyglove.core.typing.value_specs import Optional

# `pg.typing.Any` is evaluated to `typing.Any` during type checking.
import typing as _typing
if _typing.TYPE_CHECKING:
  Any = _typing.Any

# Annotated.
from pyglove.core.typing.annotated import Annotated

# Type conversion.
from pyglove.core.typing.type_conversion import register_converter
from pyglove.core.typing.type_conversion import get_converter
from pyglove.core.typing.type_conversion import get_json_value_converter

# Inspect helpers.
from pyglove.core.typing.inspect import is_subclass
from pyglove.core.typing.inspect import is_instance
from pyglove.core.typing.inspect import get_outer_class
from pyglove.core.typing.inspect import get_type
from pyglove.core.typing.inspect import get_type_args
from pyglove.core.typing.inspect import is_generic
from pyglove.core.typing.inspect import has_generic_bases
from pyglove.core.typing.inspect import callable_eq

# Annotation conversion.
import pyglove.core.typing.annotation_conversion  # pylint: disable=unused-import

# Interface for custom typing.
from pyglove.core.typing.custom_typing import CustomTyping

# Annotation conversion
from pyglove.core.typing.annotation_conversion import annotation_from_str

# Callable signature.
from pyglove.core.typing.callable_signature import Argument
from pyglove.core.typing.callable_signature import CallableType
from pyglove.core.typing.callable_signature import Signature
from pyglove.core.typing.callable_signature import signature
from pyglove.core.typing.callable_signature import schema

# For backward compatibility.
get_signature = signature

# Callable extensions.
from pyglove.core.typing.callable_ext import PresetArgValue
from pyglove.core.typing.callable_ext import enable_preset_args
from pyglove.core.typing.callable_ext import preset_args
from pyglove.core.typing.callable_ext import CallableWithOptionalKeywordArgs

# JSON schema conversion.
from pyglove.core.typing.json_schema import to_json_schema

# PyType support.
from pyglove.core.typing.pytype_support import *

# pylint: enable=g-import-not-at-top
# pylint: enable=g-importing-member
# pylint: enable=g-bad-import-order
