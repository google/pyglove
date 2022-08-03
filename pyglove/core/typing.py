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
"""Formal validation and symbolic placeholding.

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

  print(A.schema)

  @pg.symbolize([
      ('a', pg.typing.Int()),
      ('b', pg.typing.Float())
  ])
  def foo(a, b):
    return a + b

  print(foo.schema)


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

  assert B.schema.fields.keys() == ['x', 'y', 'z']

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
import abc
import calendar
import collections
import copy
import datetime
import enum
import inspect
import numbers
import re
import sys
import typing

from pyglove.core import object_utils


class KeySpec(object_utils.Formattable):
  """Interface for key specifications.

  A key specification determines what keys are acceptable for a symbolic
  field (see :class:`pyglove.Field`). Usually, symbolic attributes have an 1:1
  relationship with symbolic fields. But in some cases (e.g. a dict with dynamic
  keys), a field can be used to describe a group of symbolic attributes::

      # A dictionary that accepts key 'x' with float value
      # or keys started with 'foo' with int values.
      d = pg.Dict(value_spec=pg.typing.Dict([
         ('x', pg.typing.Float(min_value=0.0)),
         (pg.typing.StrKey('foo.*'), pg.typing.Int())
      ]))

  You may noticed that the code above pass a string 'x' for the key spec for a
  field definition. The string is automatically converted to
  :class:`pyglove.typing.ConstStrKey`.

  PyGlove's Builtin key specifications are:

  +---------------------------+----------------------------------------------+
  | ``KeySpec`` type          | Class                                        |
  +===========================+==============================================+
  | Fixed string identifier   | :class:`pyglove.typing.ConstStrKey`          |
  +---------------------------+----------------------------------------------+
  | Dynamic string identifier | :class:`pyglove.typing.StrKey`               |
  +---------------------------+----------------------------------------------+
  | Key of a list             | :class:`pyglove.typing.ListKey`              |
  +---------------------------+----------------------------------------------+
  | Key of a tuple            | :class:`pyglove.typing.TupleKey`             |
  +---------------------------+----------------------------------------------+

  In most scenarios, the user either use a string or a ``StrKey`` as the key
  spec, while other ``KeySpec`` subclasses (e.g. ``ListKey`` and ``TupleKey``)
  are used internally to constrain list size and tuple items.
  """

  @abc.abstractmethod
  def match(self, key: typing.Any) -> bool:
    """Returns whether current key specification can match a key."""

  @abc.abstractmethod
  def extend(self, base: 'KeySpec') -> 'KeySpec':
    """Extend base key specification and returns self.

    NOTE(daiyip): When a ``Field`` extends a base Field (from a base schema),
    it calls ``extend`` on both its ``KeySpec`` and ``ValueSpec``.
    ``KeySpec.extend`` is to determine whether the ``Field`` key is allowed to
    be extended, and ``ValueSpec.extend`` is to determine the final
    ``ValueSpec`` after extension.

    Args:
      base: A base ``KeySpec`` object.

    Returns:
      An ``KeySpec`` object derived from this key spec by extending the base.
    """


class ValueSpec(object_utils.Formattable):
  """Interface for value specifications.

  A value specification defines what values are acceptable for a symbolic
  field (see :class:`pyglove.Field`). When assignments take place on the
  attributes for the field, the associated ValueSpec object will kick in to
  intercept the process and take care of the following aspects:

    * Type check
    * Noneable check
    * Value validation and transformation
    * Default value assignment

  See :meth:`.apply` for more details.

  Different aspects of assignment interception are handled by the following
  methods:

  +-----------------------+-------------------------------------------------+
  | Aspect name           | Property/Method                                 |
  +=======================+=================================================+
  | Type check            | :attr:`.value_type`                             |
  +-----------------------+-------------------------------------------------+
  | Noneable check        | :attr:`.is_noneable`                            |
  +-----------------------+-------------------------------------------------+
  | Type-specific value   | :meth:`.apply`                                  |
  | validation and        |                                                 |
  | transformation        |                                                 |
  +-----------------------+-------------------------------------------------+
  | User customized value | :attr:`.user_validator`                         |
  | validation            |                                                 |
  +-----------------------+-------------------------------------------------+
  | Default value lookup  | :attr:`.default`                                |
  +-----------------------+-------------------------------------------------+

  There are many ``ValueSpec`` subclasses, each correspond to a commonly used
  Python type, e.g. `Bool`, `Int`, `Float` and etc. PyGlove's builtin value
  specifications are:

    +---------------------------+----------------------------------------------+
    | ``ValueSpec`` type        | Class                                        |
    +===========================+==============================================+
    | bool                      | :class:`pyglove.typing.Bool`                 |
    +---------------------------+----------------------------------------------+
    | int                       | :class:`pyglove.typing.Int`                  |
    +---------------------------+----------------------------------------------+
    | float                     | :class:`pyglove.typing.Float`                |
    +---------------------------+----------------------------------------------+
    | str                       | :class:`pyglove.typing.Str`                  |
    +---------------------------+----------------------------------------------+
    | enum                      | :class:`pyglove.typing.Enum`                 |
    +---------------------------+----------------------------------------------+
    | list                      | :class:`pyglove.typing.List`                 |
    +---------------------------+----------------------------------------------+
    | tuple                     | :class:`pyglove.typing.Tuple`                |
    +---------------------------+----------------------------------------------+
    | dict                      | :class:`pyglove.typing.Dict`                 |
    +---------------------------+----------------------------------------------+
    | instance of a class       | :class:`pyglove.typing.Object`               |
    +---------------------------+----------------------------------------------+
    | callable                  | :class:`pyglove.typing.Callable`             |
    +---------------------------+----------------------------------------------+
    | functor                   | :class:`pyglove.typing.Functor`              |
    +---------------------------+----------------------------------------------+
    | type                      | :class:`pyglove.typing.Type`                 |
    +---------------------------+----------------------------------------------+
    | union                     | :class:`pyglove.typing.Union`                |
    +---------------------------+----------------------------------------------+
    | any                       | :class:`pyglove.typing.Any`                  |
    +---------------------------+----------------------------------------------+

  **Construction**

  A value specification is an instance of a ``ValueSpec`` subclass. All
  :class:`pyglove.ValueSpec` subclasses follow a common pattern to construct::

      pg.typing.<ValueSpecClass>(
          [validation-rules],
          [default=<default>],
          [user_validator=<user_validator>])

  After creation, a ``ValueSpec`` object can be modified with chaining.
  The code below creates an int specification with default value 1 and can
  accept None::

      pg.typing.Int().noneable().set_default(1)


  **Usage**

  To apply a value specification on an user input to get the accepted value,
  :meth:`pyglove.ValueSpec.apply` shall be used::

      value == pg.typing.Int(min_value=1).apply(4)
      assert value == 4

  **Extension**

  Besides, a ``ValueSpec`` object can extend another ``ValueSpec`` object to
  obtain a more restricted ``ValueSpec`` object. For example::

      pg.typing.Int(min_value=1).extend(pg.typing.Int(max_value=5))

  will end up with::

      pg.typing.Int(min_value=1, max_value=5)

  which will be useful when subclass adds additional restrictions to an
  inherited symbolic attribute from its base class. For some use cases, a value
  spec can be frozen to avoid subclass extensions::

      pg.typing.Int().freeze(1)


  ``ValueSpec`` objects can be created and modified with chaining. For example::

      pg.typing.Int().noneable().set_default(1)

  The code above creates an int specification with default value 1 and can
  accept None.
  """

  @property
  @abc.abstractmethod
  def value_type(self) -> typing.Union[
      typing.Type[typing.Any],
      typing.Tuple[typing.Type[typing.Any], ...]]:  # pyformat: disable
    """Returns acceptable value type(s)."""

  @abc.abstractmethod
  def noneable(self) -> 'ValueSpec':
    """Marks none-able and returns `self`."""

  @property
  @abc.abstractmethod
  def is_noneable(self) -> bool:
    """Returns True if current value spec accepts None."""

  @abc.abstractmethod
  def set_default(self,
                  default: typing.Any,
                  use_default_apply: bool = True) -> 'ValueSpec':
    """Sets the default value and returns `self`.

    Args:
      default: Default value.
      use_default_apply: If True, invoke `apply` to the value, otherwise use
        default value as is.

    Returns:
      ValueSpec itself.

    Raises:
      ValueError: If default value cannot be applied when use_default_apply
        is set to True.
    """

  @property
  @abc.abstractmethod
  def default(self) -> typing.Any:
    """Returns the default value.

    If no default is provided, MISSING_VALUE will be returned for non-dict
    types. For Dict type, a dict that may contains nested MISSING_VALUE
    will be returned.
    """

  @property
  def has_default(self) -> bool:
    """Returns True if the default value is provided."""
    return self.default != MISSING_VALUE

  @abc.abstractmethod
  def freeze(
      self,
      permanent_value: typing.Any = object_utils.MISSING_VALUE,
      apply_before_use: bool = True) -> 'ValueSpec':
    """Sets the default value using a permanent value and freezes current spec.

    A frozen value spec will not accept any value that is not the default
    value. A frozen value spec is useful when a subclass fixes the value of a
    symoblic attribute and want to prevent it from being modified.

    Args:
      permanent_value: A permanent value used for current spec.
        If MISSING_VALUE, freeze the value spec with current default value.
      apply_before_use: If True, invoke `apply` on permanent value
        when permanent_value is provided, otherwise use it as is.

    Returns:
      ValueSpec itself.

    Raises:
      ValueError if current default value is MISSING_VALUE and the permanent
        value is not specified.
    """

  @property
  @abc.abstractmethod
  def frozen(self) -> bool:
    """Returns True if current value spec is frozen."""

  @property
  @abc.abstractmethod
  def annotation(self) -> typing.Any:
    """Returns PyType annotation. MISSING_VALUE if annotation is absent."""

  @property
  @abc.abstractmethod
  def user_validator(
      self) -> typing.Optional[typing.Callable[[typing.Any], None]]:
    """Returns a user validator which is used for custom validation logic."""

  @abc.abstractmethod
  def is_compatible(self, other: 'ValueSpec') -> bool:
    """Returns True if values acceptable to `other` is acceptable to this spec.

    Args:
      other: Other value spec.

    Returns:
      True if values that is applicable to the other value spec can be applied
        to current spec. Otherwise False.
    """

  @abc.abstractmethod
  def extend(self, base: 'ValueSpec') -> 'ValueSpec':
    """Extends a base spec with current spec's rules.

    Args:
      base: Base ValueSpec to extend.

    Returns:
      ValueSpec itself.

    Raises:
      TypeError: When this value spec cannot extend from base.
    """

  @abc.abstractmethod
  def apply(
      self,
      value: typing.Any,
      allow_partial: bool = False,
      child_transform: typing.Optional[typing.Callable[
          [object_utils.KeyPath, 'Field', typing.Any], typing.Any]] = None,
      root_path: typing.Optional[object_utils.KeyPath] = None) -> typing.Any:
    """Validates, completes and transforms the input value.

    Here is the procedure of ``apply``::

       (1). Choose the default value if the input value is ``MISSING_VALUE``
       (2). Check whether the input value is None.
         (2.a) Input value is None and ``value_spec.is_noneable()`` is False,
               raises Error.
         (2.b) Input value is not None or ``value_spec.is_noneable()`` is True,
               goto step (3).
       (3). Run ``value_spec.custom_apply`` if the input value is a
            ``CustomTyping`` instance.
         (3.a). If ``value_spec.custom_apply`` returns a value that indicates to
                proceed with standard apply, goto step (4).
         (3.b). Else goto step (6)
       (4). Check the input value type against the ``value_spec.value_type``.
         (4.a). If their value type matches, go to step (5)
         (4.b). Else if there is a converter registered between input value type
                and the value spec's value type, perform the conversion, and go
                to step (5). (see pg.typing.register_converter)
         (4.c)  Otherwise raises type mismatch.
       (5). Perform type-specific and user validation and transformation.
            For complex types such as Dict, List, Tuple, call `child_spec.apply`
            recursively on the child fields.
       (6). Perform user transform and returns final value
            (invoked at Field.apply.)

    Args:
      value: Input value to apply.
      allow_partial: If True, partial value is allowed. This is useful for
        container types (dict, list, tuple).
      child_transform: Function to transform child node values into final
        values.
        (NOTE: This transform will not be performed on current value. Instead
        transform on current value is done by Field.apply, which has adequate
        information to call transform with both KeySpec and ValueSpec).
      root_path: Key path of current node.

    Returns:
      Final value:

        * When allow_partial is set to False (default), only input value that
          has no missing values can be applied.
        * When allow_partial is set to True, missing fields will be placeheld
          using MISSING_VALUE.

    Raises:
      KeyError: If additional key is found in value, or required key is missing
        and allow_partial is set to False.
      TypeError: If type of value is not the same as spec required.
      ValueError: If value is not acceptable, or value is MISSING_VALUE while
        allow_partial is set to False.
    """

  def __ne__(self, other: typing.Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)

  def __repr__(self) -> typing.Text:
    """Operator repr."""
    return self.format(compact=True)

  def __str__(self) -> typing.Text:
    """Operator str."""
    return self.format(compact=False, verbose=True)


class CustomTyping(metaclass=abc.ABCMeta):
  """Interface of custom value type.

  Instances of subclasses of CustomTyping can be assigned to fields of
  any ValueSpec, and take over `apply` via `custom_apply` method.

  As a result, CustomTyping makes the schema system extensible without modifying
  existing value specs. For example, value generators can extend CustomTyping
  and be assignable to any fields.
  """

  @abc.abstractmethod
  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: ValueSpec,
      allow_partial: bool,
      child_transform: typing.Optional[typing.Callable[
          [object_utils.KeyPath, 'Field', typing.Any], typing.Any]] = None
  ) -> typing.Tuple[bool, typing.Any]:
    """Custom apply on a value based on its original value spec.

    Args:
      path: KeyPath of current object under its object tree.
      value_spec: Original value spec for this field.
      allow_partial: Whether allow partial object to be created.
      child_transform: Function to transform child node values into their final
        values. Transform function is called on leaf nodes first, then on their
        parents, recursively.

    Returns:
      A tuple (proceed_with_standard_apply, value_to_proceed).
        If proceed_with_standard_apply is set to False, value_to_proceed
        will be used as final value.

    Raises:
      Error when the value is not compatible with the value spec.
    """


# Shortcut variable as type-agnostic missing value.
MISSING_VALUE = object_utils.MISSING_VALUE


class MissingValue(object_utils.MissingValue, object_utils.Formattable):
  """Class represents missing value **for a specific value spec**."""

  def __init__(self, value_spec: ValueSpec):
    """Constructor."""
    self._value_spec = value_spec

  @property
  def value_spec(self) -> ValueSpec:
    """Returns value spec of current missing value."""
    return self._value_spec

  def __eq__(self, other: typing.Any) -> bool:
    """Operator ==.

    NOTE: `MissingValue(value_spec) and `object_utils.MissingValue` are
    considered equal, but `MissingValue(value_spec1)` and
    `MissingValue(value_spec2)` are considered different. That being said,
    the 'eq' operation is not transitive.

    However in practice this is not a problem, since user always compare
    against `schema.MISSING_VALUE` which is `object_utils.MissingValue`.
    Therefore the `__hash__` function returns the same value with
    `object_utils.MissingValue`.

    Args:
      other: the value to compare against.

    Returns:
      True if the other value is a general MissingValue or MissingValue of the
        same value spec.
    """
    if self is other:
      return True
    if isinstance(other, MissingValue):
      return self._value_spec == other.value_spec
    return isinstance(other, object_utils.MissingValue)

  def __hash__(self) -> int:
    """Overridden hashing to make all MissingValue return the same value."""
    return object_utils.MissingValue.__hash__(self)

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Format current object."""
    if compact:
      return 'MISSING_VALUE'
    else:
      spec_str = self._value_spec.format(
          compact=compact, verbose=verbose, root_indent=root_indent, **kwargs)
      return f'MISSING_VALUE({spec_str})'

  def __deepcopy__(self, memo):
    """Avoid deep copy by copying value_spec by reference."""
    return MissingValue(self.value_spec)


class Field(object_utils.Formattable):
  """Class that represents the definition of one or a group of attributes.

  ``Field`` is held by a :class:`pyglove.Schema` object for defining the
  name(s), the validation and transformation rules on its/their value(s) for a
  single symbolic attribute or a set of symbolic attributes.

  A ``Field`` is defined by a tuple of 4 items::

    (key specification, value specification, doc string, field metadata)

  For example::

    (pg.typing.StrKey('foo.*'),
     pg.typing.Int(),
     'Attributes with foo',
     {'user_data': 'bar'})

  The key specification (or ``KeySpec``, class :class:`pyglove.KeySpec`) and
  value specification (or ``ValueSpec``, class :class:`pyglove.ValueSpec`) are
  required, while the doc string and the field metadata are optional. The
  ``KeySpec`` defines acceptable identifiers for this field, and the
  ``ValueSpec`` defines the attribute's value type, its default value,
  validation rules and etc. The doc string serves as the description for the
  field, and the field metadata can be used for attribute-based code generation.

  ``Field`` supports extension, which allows the subclass to add more
  restrictions to a field inherited from the base class, or override its default
  value. A field can be frozen if subclasses can no longer extend it.

  See :class:`pyglove.KeySpec` and :class:`pyglove.ValueSpec` for details.
  """

  def __init__(
      self,
      key_spec: typing.Union[KeySpec, typing.Text],
      value_spec: ValueSpec,
      description: typing.Optional[typing.Text] = None,
      metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None):
    """Constructor.

    Args:
      key_spec: Key specification of the field. Can be a string or a KeySpec
        instance.
      value_spec: Value specification of the field.
      description: Description of the field.
      metadata: A dict of objects as metadata for the field.

    Raises:
      ValueError: metadata is not a dict.
    """
    if isinstance(key_spec, str):
      key_spec = ConstStrKey(key_spec)
    self._key = typing.cast(KeySpec, key_spec)
    self._value = value_spec
    self._description = description

    if metadata and not isinstance(metadata, dict):
      raise ValueError('metadata must be a dict.')
    self._metadata = metadata or {}

  @property
  def description(self) -> typing.Text:
    """Description of this field."""
    return self._description

  @property
  def key(self) -> KeySpec:
    """Key specification of this field."""
    return self._key

  @property
  def value(self) -> ValueSpec:
    """Value specification of this field."""
    return self._value

  @property
  def annotation(self) -> typing.Any:
    """Type annotation for this field."""
    return self._value.annotation

  @property
  def metadata(self) -> typing.Dict[typing.Text, typing.Any]:
    """Metadata of this field.

    Metadata is defined as a dict type, so we can add multiple annotations
    to a field.

      userdata = field.metadata.get('userdata', None):

    Returns:
      Metadata of this field as a dict.
    """
    return self._metadata

  def extend(self, base_field: 'Field') -> 'Field':
    """Extend current field based on a base field."""
    self.key.extend(base_field.key)
    self.value.extend(base_field.value)
    if not self._description:
      self._description = base_field.description
    if base_field.metadata:
      metadata = copy.copy(base_field.metadata)
      metadata.update(self.metadata)
      self._metadata = metadata
    return self

  def apply(
      self,
      value: typing.Any,
      allow_partial: bool = False,
      transform_fn: typing.Optional[typing.Callable[
          [object_utils.KeyPath, 'Field', typing.Any], typing.Any]] = None,
      root_path: typing.Optional[object_utils.KeyPath] = None) -> typing.Any:
    """Apply current field to a value, which validate and complete the value.

    Args:
      value: Value to validate against this spec.
      allow_partial: Whether partial value is allowed. This is for dict or
        nested dict values.
      transform_fn: Function to transform applied value into final value.
      root_path: Key path for root.

    Returns:
      final value.
      When allow_partial is set to False (default), only fully qualified value
      is acceptable. When allow_partial is set to True, missing fields will
      be placeheld using MISSING_VALUE.

    Raises:
      KeyError: if additional key is found in value, or required key is missing
        and allow_partial is set to False.
      TypeError: if type of value is not the same as spec required.
      ValueError: if value is not acceptable, or value is MISSING_VALUE while
        allow_partial is set to False.
    """
    value = self._value.apply(value, allow_partial, transform_fn, root_path)
    if transform_fn:
      value = transform_fn(root_path, self, value)
    return value

  @property
  def default_value(self) -> typing.Any:
    """Returns the default value."""
    return self._value.default

  @property
  def frozen(self) -> bool:
    """Returns True if current field's value is frozen."""
    return self._value.frozen

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Format this field into a string."""
    description = self._description
    if not verbose and self._description and len(self._description) > 20:
      description = self._description[:20] + '...'

    metadata = object_utils.format(
        self._metadata,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent + 1,
        **kwargs)
    if not verbose and len(metadata) > 24:
      metadata = '{...}'
    attr_str = object_utils.kvlist_str([
        ('key', self._key, None),
        ('value', self._value.format(
            compact=compact,
            verbose=verbose,
            root_indent=root_indent + 1,
            **kwargs), None),
        ('description', object_utils.quote_if_str(description), None),
        ('metadata', metadata, '{}')
    ])
    return f'Field({attr_str})'

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (isinstance(other, self.__class__) and self.key == other.key and
            self.value == other.value and
            self.description == other.description and
            self.metadata == other.metadata)

  def __ne__(self, other: typing.Any) -> bool:
    """Operator!=."""
    return not self.__eq__(other)


class Schema(object_utils.Formattable):
  """Class that represents a schema.

  PyGlove's runtime type system is based on the concept of ``Schema`` (
  class :class:`pyglove.Schema`), which defines what symbolic attributes are
  held by a symbolic type (e.g. a symbolic dict, a symbolic list or a symbolic
  class) and what values each attribute accepts. A ``Schema`` object consists of
  a list of ``Field`` (class :class:`pyglove.Field`), which define the
  acceptable keys and their values for these attributes. A ``Schema`` object is
  usually created automatically and associated with a symbolic type upon its
  declaration, through decorators such as :func:`pyglove.members`,
  :func:`pyglove.symbolize` or :func:`pyglove.functor`. For example::

    @pg.members([
        ('x', pg.typing.Int(default=1)),
        ('y', pg.typing.Float().noneable())
    ])
    class A(pg.Object):
      pass

    print(A.schema)

    @pg.symbolize([
        ('a', pg.typing.Int()),
        ('b', pg.typing.Float())
    ])
    def foo(a, b):
      return a + b

    print(foo.schema)

  Implementation-wise it holds an ordered dictionary of a field key
  (:class:`pyglove.KeySpec`) to its field definition (:class:`pyglove.Field`).
  The key specification describes what keys/attributes are acceptable for the
  field, and value specification within the ``Field`` describes the value type
  of the field and their validation rules, default values, and etc.

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

    assert B.schema.fields.keys() == ['x', 'y', 'z']

    @pg.members([
        # Raises: 'z' is frozen in class B and cannot be extended further.
        ('z', pg.typing.Str())
    ])
    class C(B):
      pass

  With a schema, an input dict can be validated and completed by the schema via
  :meth:`apply`. If required a field is missing from the schema, and the
  object's `allow_partial` is set to False, a ``KeyError`` will raise. Otherwise
  a partially validated/transformed dict will be returned. Missing values in the
  object will be placeheld by :const:`pyglove.MISSING_VALUE`.
  """

  def __init__(
      self,
      fields: typing.List[Field],
      name: typing.Optional[typing.Text] = None,
      base_schema_list: typing.Optional[typing.List['Schema']] = None,
      allow_nonconst_keys: bool = False,
      metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None):
    """Constructor.

    Args:
      fields: A list of Field as the definition of the schema. The order of the
        fields will be preserved.
      name: Optional name of this schema. Useful for debugging.
      base_schema_list: List of schema used as base. When present, fields
        from these schema will be copied to this schema. Fields from the
        latter schema will override those from the former ones.
      allow_nonconst_keys: Whether immediate fields can use non-const keys.
      metadata: Optional dict of user objects as schema-level metadata.

    Raises:
      TypeError: Argument `fields` is not a list.
      KeyError: If a field name contains characters ('.') which is not
        allowed, or a field name from `fields` already exists in parent
        schema.
      ValueError: When failed to create ValueSpec from `fields`.
        It could be an unsupported value type, default value doesn't conform
        with value specification, etc.
    """
    if not isinstance(fields, list):
      raise TypeError('Argument \'fields\' must be a list.')

    self._name = name
    self._allow_nonconst_keys = allow_nonconst_keys
    self._fields = {f.key: f for f in fields}
    self._metadata = metadata or {}

    if base_schema_list:
      # Extend base schema from the nearest ancestor to the farthest.
      for base in reversed(base_schema_list):
        self.extend(base)

    if not allow_nonconst_keys:
      for key in self._fields.keys():
        if isinstance(key, NonConstKey):
          raise ValueError(
              f'NonConstKey is not allowed in schema. Encountered \'{key}\'.')

  def extend(self, base: 'Schema') -> 'Schema':
    """Extend current schema based on a base schema."""

    def _merge_field(path, parent_field: Field, child_field: Field) -> Field:
      """Merge function on field with the same key."""
      if parent_field != MISSING_VALUE:
        if MISSING_VALUE == child_field:
          if (not self._allow_nonconst_keys and
              isinstance(parent_field.key, NonConstKey)):
            hints = object_utils.kvlist_str([
                ('base', object_utils.quote_if_str(base.name), None),
                ('path', path, None)
            ])
            raise ValueError(
                f'Non-const key {parent_field.key} is not allowed to be '
                f'added to the schema. ({hints})')
          return copy.deepcopy(parent_field)
        else:
          try:
            child_field.extend(parent_field)
          except Exception as e:  # pylint: disable=broad-except
            hints = object_utils.kvlist_str([
                ('base', object_utils.quote_if_str(base.name), None),
                ('path', path, None)
            ])
            raise e.__class__(f'{e} ({hints})').with_traceback(
                sys.exc_info()[2])
      return child_field

    self._fields = object_utils.merge([base.fields, self.fields], _merge_field)
    self._metadata = object_utils.merge([base.metadata, self.metadata])
    return self

  def is_compatible(self, other: 'Schema') -> bool:
    """Returns whether current schema is compatible with the other schema.

    NOTE(daiyip): schema A is compatible with schema B when:
    schema A and schema B have the same keys, with compatible values specs.

    Args:
      other: Other schema.

    Returns:
      True if values that is acceptable to the other schema is acceptable to
        current schema.
    Raises:
      TypeError: If `other` is not a schema object.
    """
    if not isinstance(other, Schema):
      raise TypeError(f'Argument \'other\' should be a Schema object. '
                      f'Encountered {other}.')

    for key_spec in other.keys():
      if key_spec not in self:
        return False

    for key_spec, field in self.items():
      if key_spec not in other:
        return False
      if not field.value.is_compatible(other[key_spec].value):
        return False
    return True

  def get_field(self, key: typing.Text) -> typing.Optional[Field]:
    """Get field definition (Field) for a key.

    Args:
      key: string as input key.

    Returns:
      Matched field. A field is considered a match when:
        * Its key spec is a ConstStrKey that equals to the input key.
        * Or it's the first field whose key spec is a NonConstKey
          which matches the input key.
    """
    # For const string key, we can directly retrieve from fields dict.
    if key in self._fields:
      return self._fields[key]

    if self._allow_nonconst_keys:
      for key_spec, field in self._fields.items():
        if key_spec.match(key):
          return field
    return None

  def resolve(
      self, keys: typing.Iterable[typing.Text]
  ) -> typing.Tuple[typing.Dict[KeySpec, typing.List[typing.Text]],
                    typing.List[typing.Text]]:
    """Resolve keys by grouping them by their matched fields.

    Args:
      keys: A list of string keys.

    Returns:
      A tuple of matched key results and unmatched keys.
        Matched key results are an ordered dict of KeySpec to matched keys,
        in field declaration order.
        Unmatched keys are strings from input.
    """
    keys = list(keys)
    input_keyset = set(keys)
    nonconst_key_specs = [
        k for k in self._fields.keys() if isinstance(k, NonConstKey)
    ]
    nonconst_keys = {k: [] for k in nonconst_key_specs}
    unmatched_keys = []
    keys_by_key_spec = dict()

    for key in keys:
      if key not in self._fields:
        matched_nonconst_keys = False
        for key_spec in nonconst_key_specs:
          if key_spec.match(key):
            nonconst_keys[key_spec].append(key)
            matched_nonconst_keys = True
            break
        if not matched_nonconst_keys:
          unmatched_keys.append(key)

    for key_spec in self._fields.keys():
      keys = []
      if isinstance(key_spec, NonConstKey):
        keys = nonconst_keys.get(key_spec, [])
      elif key_spec.text in input_keyset:
        keys.append(key_spec.text)
      keys_by_key_spec[key_spec] = keys

    return (keys_by_key_spec, unmatched_keys)

  def apply(
      self,
      dict_obj: typing.Dict[typing.Text, typing.Any],
      allow_partial: bool = False,
      child_transform: typing.Optional[typing.Callable[
          [object_utils.KeyPath, Field, typing.Any],
          typing.Any
      ]] = None,
      root_path: typing.Optional[object_utils.KeyPath] = None,
  ) -> typing.Dict[typing.Text, typing.Any]:  # pyformat: disable
    # pyformat: disable
    """Apply this schema to a dict object, validate and transform it.

    Args:
      dict_obj: JSON dict type that (maybe) conform to the schema.
      allow_partial: Whether allow partial object to be created.
      child_transform: Function to transform child node values in dict_obj into
        their final values. Transform function is called on leaf nodes first,
        then on their containers, recursively.
        The signature of transform_fn is: `(path, field, value) -> new_value`
        Argument `path` is a KeyPath object to the field. Argument `field` is
        on which Field the value should apply. Argument `value` is the value
        from input that matches a Field from the schema, with child fields
        already transformed by this function.
        There are possible values for these two arguments::

          ------------------------------------------------------------
                                  |   field       | value
          ------------------------------------------------------------
          The value with          |               |
          applicable Field is     |   Not None    | Not MISSING_VALUE
          found in schema.        |               |
          value.                  |               |
          ------------------------------------------------------------
          The value is            |               |
          not present for a       |   Not None    | MISSING_VALUE
          key defined in schema.  |               |
          ------------------------------------------------------------

        Return value will be inserted to the parent dict under path, unless
        return value is MISSING_VALUE.
      root_path: KeyPath of root element of dict_obj.

    Returns:
      A dict filled by the schema with transformed values.

    Raises:
      KeyError: Key is not allowed in schema.
      TypeError: Type of dict values are not aligned with schema.
      ValueError: Value of dict values are not aligned with schema.
    """
    # pyformat: enable
    matched_keys, unmatched_keys = self.resolve(dict_obj.keys())
    if unmatched_keys:
      raise KeyError(
          f'Keys {unmatched_keys} are not allowed in Schema. '
          f'(parent=\'{root_path}\')')

    for key_spec, keys in matched_keys.items():
      field = self._fields[key_spec]
      # For missing const keys, we add to keys collection to add missing value.
      if isinstance(key_spec, ConstStrKey) and key_spec.text not in keys:
        keys.append(key_spec.text)
      for key in keys:
        if dict_obj:
          value = dict_obj.get(key, MISSING_VALUE)
        else:
          value = MISSING_VALUE
        # NOTE(daiyip): field.default_value may be MISSING_VALUE too
        # or partial.
        if MISSING_VALUE == value:
          value = copy.deepcopy(field.default_value)

        child_path = object_utils.KeyPath(key, root_path)
        new_value = field.apply(
            value, allow_partial, child_transform, child_path)
        # NOTE(daiyip): minimize call to __setitem__ when possible.
        # Custom like symbolic dict may trigger additional logic
        # when __setitem__ is called.
        if key not in dict_obj or dict_obj[key] is not new_value:
          dict_obj[key] = new_value
    return dict_obj

  def validate(self,
               dict_obj: typing.Dict[typing.Text, typing.Any],
               allow_partial: bool = False,
               root_path: typing.Optional[object_utils.KeyPath] = None) -> None:
    """Validates whether dict object is conformed with the schema."""
    self.apply(
        copy.deepcopy(dict_obj),
        allow_partial=allow_partial,
        root_path=root_path)

  @property
  def name(self) -> typing.Text:
    """Name of this schema."""
    return self._name

  def set_name(self, name: typing.Text) -> None:
    """Sets the name of this schema."""
    self._name = name

  @property
  def allow_nonconst_keys(self) -> bool:
    """Returns whether to allow non-const keys."""
    return self._allow_nonconst_keys

  @property
  def fields(self) -> typing.Dict[typing.Union[typing.Text, KeySpec], Field]:
    """Returns fields of this schema."""
    return self._fields

  def __getitem__(self, key: typing.Union[typing.Text, KeySpec]) -> Field:
    """Returns field by key."""
    return self._fields[key]

  def __contains__(self, key: typing.Union[typing.Text, KeySpec]) -> bool:
    """Returns if a key or key spec exists in the schema."""
    return key in self._fields

  def get(self,
          key: typing.Union[typing.Text, KeySpec],
          default: typing.Optional[Field] = None) -> typing.Optional[Field]:
    """Returns field by key with default value if not found."""
    return self._fields.get(key, default)

  def keys(self) -> typing.Iterable[KeySpec]:
    """Return an iteratable of KeySpecs in declaration order."""
    return self._fields.keys()

  def values(self) -> typing.Iterable[Field]:
    """Returns an iterable of Field in declaration order."""
    return self._fields.values()

  def items(self) -> typing.Iterable[typing.Tuple[KeySpec, Field]]:
    """Returns an iterable of (KeySpec, Field) tuple in declaration order."""
    return self._fields.items()

  @property
  def metadata(self) -> typing.Dict[typing.Text, typing.Any]:
    """Returns metadata of this schema."""
    return self._metadata

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      cls_name: typing.Optional[typing.Text] = None,
      bracket_type: object_utils.BracketType = object_utils.BracketType.ROUND,
      **kwargs) -> typing.Text:
    """Format current Schema into nicely printed string."""
    if cls_name is None:
      cls_name = 'Schema'

    def _indent(text, indent):
      return ' ' * 2 * indent + text

    def _format_child(child):
      return child.format(
          compact=compact,
          verbose=verbose,
          root_indent=root_indent + 1,
          **kwargs)

    open_bracket, close_bracket = object_utils.bracket_chars(bracket_type)
    if compact:
      s = [f'{cls_name}{open_bracket}']
      s.append(', '.join([
          f'{f.key}={_format_child(f.value)}'
          for f in self.fields.values()
      ]))
      s.append(close_bracket)
    else:
      s = [f'{cls_name}{open_bracket}\n']
      last_field_show_description = False
      for i, f in enumerate(self.fields.values()):
        this_field_show_description = verbose and f.description
        if i != 0:
          s.append(',\n')
          if last_field_show_description or this_field_show_description:
            s.append('\n')
        if this_field_show_description:
          s.append(_indent(f'# {f.description}\n', root_indent + 1))
        last_field_show_description = this_field_show_description
        s.append(
            _indent(f'{f.key} = {_format_child(f.value)}', root_indent + 1))
      s.append('\n')
      s.append(_indent(close_bracket, root_indent))
    return ''.join(s)

  def get_signature(
      self,
      module_name: typing.Text,
      name: typing.Text,
      qualname: typing.Optional[typing.Text] = None,
      is_method: bool = True) -> 'Signature':
    """Gets signature of a function whose inputs are defined by the schema."""
    arg_names = list(self.metadata.get('init_arg_list', []))
    if arg_names and arg_names[-1].startswith('*'):
      vararg_name = arg_names[-1][1:]
      arg_names.pop(-1)
    else:
      vararg_name = None

    def get_arg_spec(arg_name):
      field = self.get_field(arg_name)
      if not field:
        raise ValueError(f'Argument {arg_name!r} is not a symbolic field.')
      return field.value

    args = []
    if is_method:
      args.append(Argument('self', Any()))

    # Prepare positional arguments.
    args.extend([Argument(n, get_arg_spec(n)) for n in arg_names])

    # Prepare varargs.
    varargs = None
    if vararg_name:
      vararg_spec = get_arg_spec(vararg_name)
      if not isinstance(vararg_spec, List):
        raise ValueError(
            f'Variable positional argument {vararg_name!r} should have a value '
            f'of `pg.typing.List` type. Encountered: {vararg_spec!r}.')
      varargs = Argument(vararg_name, vararg_spec.element.value)

    # Prepare keyword-only arguments.
    existing_names = set(arg_names)
    if vararg_name:
      existing_names.add(vararg_name)

    kwonlyargs = []
    varkw = None
    for key, field in self.fields.items():
      if key not in existing_names:
        if isinstance(key, ConstStrKey):
          kwonlyargs.append(Argument(str(key), field.value))
        else:
          varkw = Argument('kwargs', field.value)

    return Signature(
        callable_type=CallableType.FUNCTION,
        name=name,
        module_name=module_name,
        qualname=qualname,
        args=args,
        kwonlyargs=kwonlyargs,
        varargs=varargs,
        varkw=varkw,
        return_value=None)

  def __str__(self) -> typing.Text:
    return self.format(compact=False, verbose=True)

  def __eq__(self, other: typing.Any) -> bool:
    if self is other:
      return True
    return isinstance(other, Schema) and self._fields == other._fields

  def __ne__(self, other: typing.Any) -> bool:
    return not self.__eq__(other)


#
# Implementations of KeySpec.
#


class KeySpecBase(KeySpec):
  """Base class for key specification subclasses."""

  def extend(self, base: KeySpec) -> KeySpec:
    """Extend current key spec based on a base spec."""
    if self != base:
      raise KeyError(f'{self} cannot extend {base} for keys are different.')
    return self

  def __repr__(self) -> typing.Text:
    """Operator repr."""
    return self.__str__()

  def __ne__(self, other: typing.Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)


class ConstStrKey(KeySpecBase, object_utils.StrKey):
  """Class that represents a constant string key.

  Example::

      key = pg.typing.ConstStrKey('x')
      assert key == 'x'
      assert hash(key) == hash('x')
  """

  def __init__(self, text: typing.Text):
    """Constructor.

    Args:
      text: string value of this key.

    Raises:
      KeyError: If key contains dots ('.'), which is not allowed.
    """
    if '.' in text:
      raise KeyError(f'\'.\' cannot be used in key. Encountered: {text!r}.')
    super().__init__()
    self._text = text

  @property
  def text(self) -> typing.Text:
    """Text of this const string key."""
    return self._text

  def match(self, key: typing.Any) -> bool:
    """Whether can match against an input key."""
    return self._text == key

  def format(self, **kwargs) -> typing.Text:
    """Format this object."""
    return self._text

  def __hash__(self) -> int:
    """Hash function.

    NOTE(daiyip): ConstStrKey shares the same hash with its text, which
    makes it easy to lookup a dict of string by an ConstStrKey object, and
    vice versa.

    Returns:
      Hash code.
    """
    return self._text.__hash__()

  def __eq__(self, other: typing.Any) -> bool:
    """Operator ==."""
    if self is other:
      return True
    if isinstance(other, str):
      return self.text == other
    return isinstance(other, ConstStrKey) and other.text == self.text


class NonConstKey(KeySpecBase):
  """Base class of specification for non-const key.

  Subclasses: :class:`pyglove.typing.StrKey`.
  """
  pass


class StrKey(NonConstKey):
  """Class that represents a variable string key.

  Example::

      # Create a key spec that specifies all string keys started with 'foo'.
      key = pg.typing.StrKey('foo.*')

      assert key.match('foo')
      assert key.match('foo1')
      assert not key.match('bar')
  """

  def __init__(self, regex: typing.Optional[typing.Text] = None):
    """Constructor.

    Args:
      regex: An optional regular expression. If set to None, any string value is
        acceptable.
    """
    super().__init__()
    self._regex = re.compile(regex) if regex else None

  def match(self, key: typing.Any) -> bool:
    """Whether this key spec can match against input key."""
    if not isinstance(key, str):
      return False
    if self._regex:
      return self._regex.match(key) is not None
    return True

  @property
  def regex(self):
    """Returns regular expression of this key spec."""
    return self._regex

  def format(self, **kwargs):
    """Format this object."""
    regex_str = object_utils.kvlist_str([
        ('regex', object_utils.quote_if_str(
            self._regex.pattern if self._regex else None), None)
    ])
    return f'StrKey({regex_str})'

  def __hash__(self):
    """Hash function."""
    if self._regex:
      return self._regex.pattern.__hash__()
    else:
      return '.*'.__hash__()

  def __eq__(self, other):
    """Operator ==."""
    if self is other:
      return True
    return isinstance(other, StrKey) and self._regex == other.regex


class ListKey(KeySpecBase):
  """Class that represents key specification for a list.

  Example::

      # Create a key spec that specifies list items from 1 to 5 (zero-based).
      key = pg.typing.ListKey(min_value=1, max_value=5)

      assert key.match(1)
      assert key.match(5)
      assert not key.match(0)
  """

  def __init__(
      self, min_value: int = 0, max_value: typing.Optional[int] = None):
    """Constructor.

    Args:
      min_value: Min value that is acceptable for the list index.
      max_value: Max value that is acceptable for the list index. If None, there
        is no upper bound for list index.
    """
    super().__init__()
    self._min_value = min_value
    self._max_value = max_value

  def extend(self, base: KeySpec) -> 'ListKey':
    """Extend current key spec on top of base spec."""
    if not isinstance(base, ListKey):
      raise TypeError(f'{self} cannot extend {base}: incompatible type.')

    if self.min_value < base.min_value:
      raise TypeError(f'{self} cannot extend {base}: min_value is smaller.')
    if base.max_value is None:
      return self
    if self.max_value is None:
      self._max_value = base.max_value
    elif self.max_value > base.max_value:
      raise TypeError(f'{self} cannot extend {base}: max_value is greater.')
    return self

  @property
  def min_value(self) -> int:
    """Returns min value of acceptable list index value."""
    return self._min_value

  @property
  def max_value(self) -> typing.Optional[int]:
    """Returns max value of acceptable list index value."""
    return self._max_value

  def match(self, key: typing.Any) -> bool:
    """Returns whether this key spec can match against input key."""
    return (isinstance(key, int) and (self._min_value <= key) and
            (not self._max_value or self._max_value > key))

  def format(self, **kwargs):
    """Format this object."""
    return f'ListKey(min_value={self._min_value}, max_value={self._max_value})'

  def __eq__(self, other):
    """Operator==."""
    if self is other:
      return True
    return (isinstance(other, ListKey) and
            self._min_value == other.min_value and
            self._max_value == other.max_value)


class TupleKey(KeySpecBase):
  """Class that represents a key specification for tuple.

  Example::

      # Create a key spec that specifies item 0 of a tuple.
      key = pg.typing.TupleKey(0)

      assert key.match(0)
      assert not key.match(1)
  """

  def __init__(self, index: typing.Optional[int]):
    """Constructor.

    Args:
      index: index of the tuple field that this key spec applies to.
        If None, this tuple value spec applies to all elements of a
        variable-length tuple.
    """
    super().__init__()
    self._index = index

  def extend(self, base: KeySpec) -> 'TupleKey':
    """Extends this key spec on top of a base spec."""
    if not isinstance(base, TupleKey):
      raise TypeError(f'{self} cannot extend {base}: incompatible type.')
    if self._index is None:
      self._index = base.index
    elif base.index is not None and base.index != self.index:
      raise KeyError(f'{self} cannot extend {base}: unmatched index.')
    return self

  @property
  def index(self) -> typing.Optional[int]:
    """Returns the index of tuple field that the key applies to."""
    return self._index

  def match(self, key: typing.Any) -> bool:
    """Returns whether this key spec can match against input key."""
    return isinstance(key, int) and self._index == key

  def format(self, **kwargs):
    """Format this object."""
    return 'TupleKey(index={self._index})'

  def __eq__(self, other):
    """Operator==."""
    if self is other:
      return True
    return isinstance(other, TupleKey) and self._index == other.index


#
# Implementations for ValueSpec.
#


class ValueSpecBase(ValueSpec):
  """A convenient base class for ValueSpec subclasses.

  This base class provides common functionalities like None value check, type
  check, type conversion, and etc. leaving type specific validation to
  subclasses.
  """

  def __init__(
      self,
      value_type: typing.Optional[
          typing.Union[
              typing.Type[typing.Any],
              typing.Tuple[typing.Type[typing.Any], ...]
          ]],
      default: typing.Any = MISSING_VALUE,
      user_validator: typing.Optional[
          typing.Callable[[typing.Any], None]] = None,
      is_noneable: bool = False):  # pyformat: disable
    """Constructor of ValueSpecBase.

      This class provides common facilities for implementing ValueSpec,
      including type check, default value assignment, noneable handling,
      missing value handling, and etc. Subclasses only need to handle value
      specific logics in `apply`, `extend`, and `is_compatible`.

    Args:
      value_type: Type or tuples of type or None. When a not-none value_type is
        present, type check will be performed.
      default: (Optional) Default value. If not specified, it always require
        user to provide. Or it can be any value that can be accepted by this
        spec, or None, which automatically add Noneable property to the spec.
      user_validator: (Optional) user function or callable object for additional
        validation on applied value, which can reject a value by raising
        Exceptions. Please note that this validation is an addition to
        validation provided by built-in constraint, like `min_value` for
        `schema.Int`.
      is_noneable: (Optional) If True, None is acceptable for this spec.
    """
    super().__init__()
    self._value_type = value_type
    self._is_noneable = is_noneable
    self._frozen = False
    self._default = MISSING_VALUE
    self._user_validator = user_validator
    self.set_default(default)

  @property
  def is_noneable(self) -> bool:
    """Returns True if current value spec accepts None."""
    return self._is_noneable

  def noneable(self) -> 'ValueSpecBase':
    """Marks None is acceptable and returns `self`."""
    self._is_noneable = True
    if MISSING_VALUE == self._default:
      self._default = None
    return self

  @property
  def value_type(self) -> typing.Union[
      typing.Type[typing.Any],
      typing.Tuple[typing.Type[typing.Any], ...]]:  # pyformat: disable
    """Returns acceptable value type(s) for current value spec."""
    return self._value_type

  @property
  def default(self) -> typing.Any:
    """Returns the default value."""
    return self._default

  def set_default(self,
                  default: typing.Any,
                  use_default_apply: bool = True) -> ValueSpec:
    """Set default value and returns `self`."""
    # NOTE(daiyip): Default can be schema.MissingValue types, all are
    # normalized to MISSING_VALUE for consistency.
    if MISSING_VALUE == default:
      default = MISSING_VALUE
    if default != MISSING_VALUE and use_default_apply:
      default = self.apply(default, allow_partial=True)
    self._default = default
    return self

  def freeze(self,
             permanent_value: typing.Any = MISSING_VALUE,
             apply_before_use: bool = True) -> ValueSpec:
    """Sets the permanent value as default value and freeze the value spec."""
    if permanent_value != MISSING_VALUE:
      self.set_default(permanent_value, use_default_apply=apply_before_use)
    elif self._default == MISSING_VALUE:
      raise ValueError(f'Cannot freeze {self} without a default value.')
    self._frozen = True
    return self

  @property
  def frozen(self) -> bool:
    """Returns True if current value spec is frozen."""
    return self._frozen

  @property
  def user_validator(
      self) -> typing.Optional[typing.Callable[[typing.Any], None]]:
    """Returns user validator for custom validation logic."""
    return self._user_validator

  def extend(self, base: ValueSpec) -> ValueSpec:
    """Extend current value spec on top of a base spec."""
    if base.frozen:
      raise TypeError(f'Cannot extend a frozen value spec: {base}')

    if self._user_validator is None:
      self._user_validator = base.user_validator

    if isinstance(base, Any):
      return self

    if not isinstance(self, Union) and isinstance(base, Union):
      base_counterpart = base.get_candidate(self)
      if base_counterpart is None:
        raise TypeError(f'{self!r} cannot extend {self!r}: '
                        f'no compatible type found in Union.')
      base = base_counterpart

    if not isinstance(self, base.__class__):
      raise TypeError(f'{self!r} cannot extend {base!r}: incompatible type.')
    if not base.is_noneable and self._is_noneable:
      raise TypeError(f'{self!r} cannot extend {base!r}: '
                      f'None is not allowed in base spec.')
    self._extend(base)
    return self

  def _extend(self, base: ValueSpec) -> None:
    """Customized extension that each subclass can override."""
    pass

  def apply(
      self,
      value: typing.Any,
      allow_partial: bool = False,
      child_transform: typing.Optional[typing.Callable[
          [object_utils.KeyPath, Field, typing.Any],
          typing.Any
      ]] = None,
      root_path: typing.Optional[object_utils.KeyPath] = None) -> typing.Any:  # pyformat: disable pylint: disable=line-too-long
    """Apply spec to validate and complete value."""
    root_path = root_path or object_utils.KeyPath()

    if self.frozen:
      # Always return the default value if a field is frozen.
      if value != MISSING_VALUE and value != self.default:
        raise ValueError(
            f'Frozen field is not assignable. (Path='
            f'\'{root_path}\', ValueSpec={self!r}, AssignedValue={value!r})')
      return self.default

    if MISSING_VALUE == value:
      if not allow_partial:
        raise ValueError(
            f'Required value is not specified. '
            f'(Path=\'{root_path}\', ValueSpec={self!r})')
      return MissingValue(self)

    if value is None:
      if self.is_noneable:
        return
      raise ValueError(
          f'Value cannot be None. (Path=\'{root_path}\', ValueSpec={self!r})')

    # NOTE(daiyip): CustomTyping will take over the apply logic other than
    # standard apply process. This allows users to plugin complex types as
    # the inputs for Schema.apply and have full control on the transform.
    if isinstance(value, CustomTyping):
      should_continue, value = value.custom_apply(
          root_path,
          self,
          allow_partial=allow_partial,
          child_transform=child_transform)
      if not should_continue:
        return value

    if self._value_type is not None and not isinstance(value, self._value_type):
      converter = get_first_applicable_converter(type(value), self._value_type)
      if converter is None:
        raise TypeError(
            object_utils.message_on_path(
                f'Expect {self._value_type} '
                f'but encountered {type(value)!r}: {value}.', root_path))
      value = converter(value)

    # NOTE(daiyip): child nodes validation and transformation is done before
    # parent nodes, which makes sure when child_transform is called on current
    # node (which is in Field.apply), input are in good shape.
    # It also lets users to create complex object types from transform function
    # without downstream constraint.
    value = self._apply(value, allow_partial, child_transform, root_path)

    # Validation is applied after transformation.
    self._validate(root_path, value)
    if self._user_validator is not None:
      try:
        self._user_validator(value)
      except Exception as e:  # pylint: disable=broad-except
        raise e.__class__(
            object_utils.message_on_path(str(e), root_path)
            ).with_traceback(sys.exc_info()[2])
    return value

  def _validate(self, path: object_utils.KeyPath, value: typing.Any):
    """Validation on applied value. Child class can override."""

  def _apply(self,
             value: typing.Any,
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any], typing.Any],
             root_path: object_utils.KeyPath) -> typing.Any:
    """Customized apply so each subclass can override."""
    del allow_partial
    del child_transform
    del root_path
    return value

  def is_compatible(self, other: ValueSpec) -> bool:
    """Returns if current spec is compatible with the other value spec."""
    if self is other:
      return True
    if not isinstance(other, self.__class__):
      return False
    if not self.is_noneable and other.is_noneable:
      return False
    return self._is_compatible(other)

  def _is_compatible(self, other: ValueSpec) -> bool:
    """Customized compatibility check for child class to override."""
    return True

  @property
  def annotation(self) -> typing.Any:
    """Returns PyType annotation."""
    annotation = self._annotate()
    if annotation != MISSING_VALUE and self.is_noneable:
      return typing.Optional[annotation]
    return annotation

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return self._value_type

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    if not isinstance(other, self.__class__):
      return False
    if isinstance(self._value_type, tuple):
      self_value_types = list(self._value_type)
    else:
      self_value_types = [self._value_type]
    if isinstance(other.value_type, tuple):
      other_value_types = list(other.value_type)
    else:
      other_value_types = [self._value_type]
    return (set(self_value_types) == set(other_value_types)
            and self.default == other.default
            and self.is_noneable == other.is_noneable
            and self.frozen == other.frozen)

  def format(self, **kwargs) -> typing.Text:
    """Format this object."""
    details = object_utils.kvlist_str([
        ('default', object_utils.quote_if_str(self._default), MISSING_VALUE),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False)
    ])
    return f'{self.__class__.__name__}({details})'


class PrimitiveType(ValueSpecBase):
  """Base class of value specification for primitive types."""

  def __init__(self,
               value_type: typing.Union[typing.Type[typing.Any],
                                        typing.Tuple[typing.Type[typing.Any],
                                                     ...]],
               default: typing.Any = MISSING_VALUE,
               is_noneable: bool = False):
    """Constructor.

    Args:
      value_type: Acceptable value type(s).
      default: Default value.
      is_noneable: If True, None is acceptable.
    """
    super().__init__(
        value_type, default, is_noneable=is_noneable)


class Bool(PrimitiveType):
  """Value spec for boolean type.

  Examples::

    # A required bool value.
    pg.typing.Bool()

    # A bool value with the default value set to True.
    pg.typing.Bool(default=True)

    # An optional bool value with default value set to None.
    pg.typing.Bool().noneable()

    # An optional bool value with default value set to True.
    pg.typing.Bool(default=True).noneable()

    # A frozen bool with value set to True that is not modifiable by subclasses.
    pg.typing.Bool().freeze(True)
  """

  def __init__(self, default: typing.Optional[bool] = MISSING_VALUE):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: Default value for the value spec.
    """
    super().__init__(bool, default)


class Str(PrimitiveType):
  """Value spec for string type.

  Examples::

    # A required str value.
    pg.typing.Str()

    # A required str value which matches with a regular expression.
    pg.typing.Str(regex='foo.*'))

    # A str value with the default value set to 'foo'.
    pg.typing.Str(default='foo')

    # An optional str value with default value set to None.
    pg.typing.Str().noneable()

    # An optional str value with default value set to 'foo'.
    pg.typing.Str(default='foo').noneable()

    # A frozen str with value set to 'foo' that is not modifiable by subclasses.
    pg.typing.Str().freeze('foo')
  """

  def __init__(self,
               default: typing.Optional[typing.Text] = MISSING_VALUE,
               regex: typing.Optional[typing.Text] = None):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: Default value for this value spec.
      regex: Optional regular expression for acceptable value.
    """
    self._regex = re.compile(regex) if regex else None
    super().__init__(str, default)

  def _validate(self, path: object_utils.KeyPath, value: typing.Text) -> None:
    """Validates applied value."""
    if not self._regex:
      return
    if not self._regex.match(value):
      raise ValueError(
          object_utils.message_on_path(
              f'String {value!r} does not match '
              f'regular expression {self._regex.pattern!r}.', path))

  @property
  def regex(self):
    """Returns regular expression for acceptable values."""
    return self._regex

  def _extend(self, base: 'Str') -> None:
    """Str specific extend."""
    if not self._regex:
      # NOTE(daiyip): we may want to check if child regex
      # is a stricter form of base regex in future.
      self._regex = base.regex

  def _is_compatible(self, other: 'Str') -> bool:
    """Str specific compatibility check."""
    # NOTE(daiyip): loose the compatibility check for regular expressions,
    # since there is not an easy way for checking the compatibility of two
    # regular expressions. Returning True might lead to false-positives but
    # we cannot afford false-negatives.
    return True

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return typing.Text

  def format(self, **kwargs) -> typing.Text:
    """Format this object."""
    regex_pattern = self._regex.pattern if self._regex else None
    details = object_utils.kvlist_str([
        ('default', object_utils.quote_if_str(self._default), MISSING_VALUE),
        ('regex', object_utils.quote_if_str(regex_pattern), None),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False)
    ])
    return f'{self.__class__.__name__}({details})'

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return super().__eq__(other) and self.regex == other.regex


class Number(PrimitiveType):
  """Base class for value spec of numeric types."""

  def __init__(
      self,
      value_type,  # typing.Type[numbers.Number]
      default: typing.Optional[numbers.Number] = MISSING_VALUE,
      min_value: typing.Optional[numbers.Number] = None,
      max_value: typing.Optional[numbers.Number] = None):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      value_type: Type of number.
      default: Default value for this spec.
      min_value: (Optional) minimum value of acceptable values.
      max_value: (Optional) maximum value of acceptable values.
    """
    if (min_value is not None and max_value is not None and
        min_value > max_value):
      raise ValueError(
          f'"max_value" must be equal or greater than "min_value". '
          f'Encountered: min_value={min_value}, max_value={max_value}.')
    self._min_value = min_value
    self._max_value = max_value
    super().__init__(value_type, default)

  @property
  def min_value(self) -> typing.Optional[numbers.Number]:
    """Returns minimum value of acceptable values."""
    return self._min_value

  @property
  def max_value(self) -> numbers.Number:
    """Returns maximum value of acceptable values."""
    return self._max_value

  def _validate(self, path: object_utils.KeyPath,
                value: numbers.Number) -> None:
    """Validates applied value."""
    if ((self._min_value is not None and value < self._min_value) or
        (self._max_value is not None and value > self._max_value)):
      raise ValueError(
          object_utils.message_on_path(
              f'Value {value} is out of range '
              f'(min={self._min_value}, max={self._max_value}).', path))

  def _extend(self, base: 'Number') -> None:
    """Number specific extend."""
    min_value = self._min_value
    if base.min_value is not None:
      if min_value is None:
        min_value = base.min_value
      elif min_value < base.min_value:
        raise TypeError(f'{self} cannot extend {base}: min_value is smaller.')

    max_value = self._max_value
    if base.max_value is not None:
      if max_value is None:
        max_value = base.max_value
      elif max_value > base.max_value:
        raise TypeError(f'{self} cannot extend {base}: max_value is larger.')

    if (min_value is not None and max_value is not None and
        min_value > max_value):
      raise TypeError(
          f'{self} cannot extend {base}: '
          f'min_value ({min_value}) is greater than max_value ({max_value}) '
          'after extension.')
    self._min_value = min_value
    self._max_value = max_value

  def _is_compatible(self, other: 'Number') -> bool:
    """Number specific compatibility check."""
    if self._min_value is not None:
      if other.min_value is None or other.min_value < self._min_value:
        return False
    if self._max_value is not None:
      if other.max_value is None or other.max_value > self._max_value:
        return False
    return True

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (super().__eq__(other) and
            self.min_value == other.min_value and
            self.max_value == other.max_value)

  def format(self, **kwargs) -> typing.Text:
    """Format this object."""
    details = object_utils.kvlist_str([
        ('default', self._default, MISSING_VALUE),
        ('min', self._min_value, None),
        ('max', self._max_value, None),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False)
    ])
    return f'{self.__class__.__name__}({details})'


class Int(Number):
  """Value spec for int type.

  Examples::

    # A required int value.
    pg.typing.Int()

    # A required int value with min and max value (both inclusive.)
    pg.typing.Int(min_value=1, max_value=10)

    # A int value with the default value set to 1
    pg.typing.Int(default=1)

    # An optional int value with default value set to None.
    pg.typing.Int().noneable()

    # An optional int value with default value set to 1.
    pg.typing.Int(default=1).noneable()

    # A frozen int with value set to 1 that is not modifiable by subclasses.
    pg.typing.Int().freeze(1)
  """

  def __init__(self,
               default: typing.Optional[int] = MISSING_VALUE,
               min_value: typing.Optional[int] = None,
               max_value: typing.Optional[int] = None):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: (Optional) default value for this spec.
      min_value: (Optional) minimum value of acceptable values.
      max_value: (Optional) maximum value of acceptable values.
    """
    super().__init__(int, default, min_value, max_value)


class Float(Number):
  """Value spec for float type.

  Examples::

    # A required float value.
    pg.typing.Float()

    # A required float value with min and max value (both inclusive.)
    pg.typing.Float(min_value=1.0, max_value=10.0)

    # A float value with the default value set to 1
    pg.typing.Float(default=1)

    # An optional float value with default value set to None.
    pg.typing.Float().noneable()

    # An optional float value with default value set to 1.0.
    pg.typing.Float(default=1.0).noneable()

    # A frozen float with value set to 1.0 that is not modifiable by subclasses.
    pg.typing.Float().freeze(1)
  """

  def __init__(self,
               default: typing.Optional[float] = MISSING_VALUE,
               min_value: typing.Optional[float] = None,
               max_value: typing.Optional[float] = None):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: (Optional) default value for this spec.
      min_value: (Optional) minimum value of acceptable values.
      max_value: (Optional) maximum value of acceptable values.
    """
    super().__init__(float, default, min_value, max_value)


class Enum(PrimitiveType):
  """Value spec for enum type.

  Examples::

    # A str enum value with options 'a', 'b', 'c' and its default set to 'a'.
    pg.typing.Enum('a', ['a', 'b', 'c'])

    # A mixed-type enum value.
    pg.typing.Enum('a', ['a', 5, True])

    # An optional enum value with default value set to 'a'.
    pg.typing.Enum('a', ['a', 'b', 'c']).noneable()

   # A frozen enum with value set to 'a' that is not modifiable by subclasses.
    pg.typing.Enum('a', ['a', 'b', 'c']).freeze('a')
  """

  def __init__(self, default: typing.Any, values: typing.List[typing.Any]):
    """Constructor.

    Args:
      default: default value for this spec.
      values: all acceptable values.
    """
    if not isinstance(values, list) or not values:
      raise ValueError(
          f'Values for Enum should be a non-empty list. Found {values}')
    if default not in values:
      raise ValueError(
          f'Enum default value {default!r} is not in candidate list {values}.')

    value_type = None
    is_noneable = False
    for v in values:
      if v is None:
        is_noneable = True
        continue
      if value_type is None:
        value_type = type(v)
      else:
        next_type = type(v)
        if issubclass(value_type, next_type):
          value_type = next_type
        elif not issubclass(next_type, value_type):
          value_type = None
          break

    # NOTE(daiyip): When enum values are strings, we relax the `value_type`
    # to accept text types (unicode) as well. This allows enum value be read
    # from unicode JSON file.
    if value_type is not None and issubclass(value_type, str):
      value_type = str
    self._values = values
    super().__init__(value_type, default, is_noneable=is_noneable)

  def noneable(self) -> 'Enum':
    """Noneable is specially treated for Enum."""
    if None not in self._values:
      self._values.append(None)
    self._is_noneable = True
    return self

  @property
  def values(self) -> typing.List[typing.Any]:
    """Returns all acceptable values of this spec."""
    return self._values

  def _validate(self, path: object_utils.KeyPath, value: typing.Any) -> None:
    """Validates applied value."""
    if value not in self._values:
      raise ValueError(
          object_utils.message_on_path(
              f'Value {value!r} is not in candidate list {self._values}.',
              path))

  def _extend(self, base: 'Enum') -> None:
    """Enum specific extend."""
    if not set(base.values).issuperset(set(self._values)):
      raise TypeError(
          f'{self} cannot extend {base}: values in base should be super set.')

  def _is_compatible(self, other: 'Enum') -> bool:
    """Enum specific compatibility check."""
    for v in other.values:
      if v not in self.values:
        return False
    return True

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    if self._value_type == str:
      return typing.Text
    if self._value_type is None:
      return typing.Any
    return self._value_type

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (super().__eq__(other)
            and self._default == other._default  # pylint: disable=protected-access
            and self.values == other.values)

  def format(self, **kwargs) -> typing.Text:
    """Format this object."""
    details = object_utils.kvlist_str([
        ('default', object_utils.quote_if_str(self._default), MISSING_VALUE),
        ('values', self._values, None),
        ('frozen', self._frozen, False),
    ])
    return f'{self.__class__.__name__}({details})'


class List(ValueSpecBase):
  """Value spec for list type.

  Examples::

    # A required non-negative integer list.
    pg.typing.List(pg.typing.Int(min_value=0))

    # A optional str list with 2-5 items.
    pg.typing.List(pg.typing.Str(), min_size=2, max_size=5).noneable()

    # An size-2 list of arbitrary types
    pg.typing.List(pg.typing.Any(), size=2)

   # A frozen list that prevents subclass to extend/override.
    pg.typing.List(pg.typing.Int()).freeze([1])
  """

  def __init__(
      self,
      element_value: ValueSpec,
      default: typing.Optional[typing.List[typing.Any]] = MISSING_VALUE,
      min_size: typing.Optional[int] = None,
      max_size: typing.Optional[int] = None,
      size: typing.Optional[int] = None,
      user_validator: typing.Optional[
          typing.Callable[[typing.List[typing.Any]], None]] = None):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      element_value: Value spec for list element.
      default: (Optional) default value for this spec.
      min_size: (Optional) min size of list. If None, 0 will be used.
      max_size: (Optional) max size of list.
      size: (Optional) size of List. A shortcut to specify min_size and max_size
        at the same time. `size` and `min_size`/`max_size` are mutual exclusive.
      user_validator: (Optional) user function or callable object for additional
        validation on the applied list, which can reject a value by raising
        Exceptions. Please note that this validation is an addition to
        validating list size constraint.
    """
    if not isinstance(element_value, ValueSpec):
      raise ValueError('List element spec should be an ValueSpec object.')

    if size is not None and (min_size is not None or max_size is not None):
      raise ValueError(
          f'Either "size" or "min_size"/"max_size" pair can be specified. '
          f'Encountered: size={size}, min_size={min_size}, '
          f'max_size={max_size}.')
    if size is not None:
      min_size = size
      max_size = size

    if min_size is None:
      min_size = 0

    if min_size < 0:
      raise ValueError(
          f'"min_size" of List must be no less than 0. '
          f'Encountered: {min_size}.')
    if max_size is not None:
      if max_size < min_size:
        raise ValueError(
            f'"max_size" of List must be no less than "min_size". '
            f'Encountered: min_size={min_size}, max_size={max_size}')
    self._element = Field(
        ListKey(min_size, max_size), element_value, 'Field of list element')
    super().__init__(list, default, user_validator)

  @property
  def element(self) -> Field:
    """Returns Field specification of list element."""
    return self._element

  @property
  def min_size(self) -> int:
    """Returns max size of the list."""
    return self._element.key.min_value  # pytype: disable=attribute-error  # bind-properties

  @property
  def max_size(self) -> typing.Optional[int]:
    """Returns max size of the list."""
    return self._element.key.max_value  # pytype: disable=attribute-error  # bind-properties

  def _apply(self,
             value: typing.List[typing.Any],
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any], typing.Any],
             root_path: object_utils.KeyPath) -> typing.Any:
    """List specific apply."""
    # NOTE(daiyip): for symbolic List, write access using `__setitem__` will
    # trigger permission error when `accessor_writable` is set to False.
    # As a result, we always try `_set_item_without_permission_check` if it's
    # available.
    set_item = getattr(value, '_set_item_without_permission_check', None)
    if set_item is None:
      def _fn(i, v):
        value[i] = v
      set_item = _fn

    for i, v in enumerate(value):
      v = self._element.apply(v, allow_partial, child_transform,
                              object_utils.KeyPath(i, root_path))
      if value[i] is not v:
        set_item(i, v)
    return value

  def _validate(
      self, path: object_utils.KeyPath, value: typing.List[typing.Any]):
    """Validates applied value."""
    if len(value) < self.min_size:
      raise ValueError(
          object_utils.message_on_path(
              f'Length of list {value!r} is less than '
              f'min size ({self.min_size}).', path))

    if self.max_size is not None:
      if len(value) > self.max_size:
        raise ValueError(
            object_utils.message_on_path(
                f'Length of list {value!r} is greater than '
                f'max size ({self.max_size}).', path))

  def _extend(self, base: 'List') -> None:
    """List specific extend."""
    self._element.extend(base.element)

  def _is_compatible(self, other: 'List') -> bool:
    """List specific compatibility check."""
    if self.max_size is not None:
      if other.max_size is None or other.max_size > self.max_size:
        return False
    return self._element.value.is_compatible(other.element.value)

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return typing.List[_any_if_no_annotation(self._element.value.annotation)]

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return super().__eq__(other) and self.element == other.element

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             hide_default_values: bool = True,
             hide_missing_values: bool = True,
             **kwargs) -> typing.Text:
    """Format this object."""
    details = object_utils.kvlist_str([
        ('', self._element.value.format(
            compact=compact,
            verbose=verbose,
            root_indent=root_indent,
            **kwargs), None),
        ('min_size', self.min_size, 0),
        ('max_size', self.max_size, None),
        ('default', object_utils.format(
            self._default,
            compact=compact,
            verbose=verbose,
            root_indent=root_indent + 1,
            hide_default_values=hide_default_values,
            hide_missing_values=hide_missing_values,
            **kwargs), 'MISSING_VALUE'),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False),
    ])
    return f'{self.__class__.__name__}({details})'


class Tuple(ValueSpecBase):
  """Value spec for tuple type.

  Examples::

    # A required tuple with 2 items which are non-negative integers.
    pg.typing.Tuple([pg.typing.Int(min_value=0), pg.typing.Int(min_value=0)])

    # A optional int tuple of size 3 with None as its default value.
    pg.typing.Tuple(pg.typing.Int(), size=3).noneable()

    # A int tuple with minimal size 1 and maximal size 5.
    pg.typing.Tuple(pg.typing.Int(), min_size=1, max_size=5)

    # A (int, float) tuple with default value (1, 1.0).
    pg.typing.Tuple([pg.typing.Int(), pg.typing.Float()], default=(1, 1.0))

    # A frozen tuple that prevents subclass to extend/override.
    pg.typing.Tuple(pg.typing.Int()).freeze((1,))
  """

  def __init__(
      self,
      element_values: typing.Union[ValueSpec, typing.List[ValueSpec]],
      default: typing.Optional[typing.Tuple[typing.Any, ...]] = MISSING_VALUE,
      min_size: typing.Optional[int] = None,
      max_size: typing.Optional[int] = None,
      size: typing.Optional[int] = None,
      user_validator: typing.Optional[
          typing.Callable[[typing.Tuple[typing.Any, ...]], None]] = None):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      element_values: A ValueSpec as element spec for a variable-length tuple,
        or a list of ValueSpec as elements specs for a fixed-length tuple.
      default: (Optional) default value for this spec.
      min_size: (Optional) min size of tuple. If None, 0 will be used.
        Applicable only for variable-length tuple.
      max_size: (Optional) max size of list.
        Applicable only for variable-length tuple.
      size: (Optional) size of List. A shortcut to specify min_size and max_size
        at the same time. `size` and `min_size`/`max_size` are mutual exclusive.
      user_validator: (Optional) user function or callable object for additional
        validation on the applied tuple, which can reject a value by raising
        Exceptions. Please note that this validation is an addition to
        validating tuple size constraint.
    """
    if isinstance(element_values, ValueSpec):
      if size is not None and (min_size is not None or max_size is not None):
        raise ValueError(
            f'Either "size" or "min_size"/"max_size" pair can be specified. '
            f'Encountered: size={size}, min_size={min_size}, '
            f'max_size={max_size}.')
      if size is not None:
        min_size = size
        max_size = size
      if min_size is None:
        min_size = 0
      if min_size < 0:
        raise ValueError(
            f'"min_size" of List must be no less than 0. '
            f'Encountered: {min_size}.')
      if max_size is not None:
        if max_size < min_size:
          raise ValueError(
              f'"max_size" of List must be no less than "min_size". '
              f'Encountered: min_size={min_size}, max_size=max_size.')
      if min_size == max_size:
        element_values = [element_values] * min_size
    elif isinstance(element_values, list) and element_values:
      if size is not None or min_size is not None or max_size is not None:
        raise ValueError(
            f'"size", "min_size" and "max_size" are not applicable '
            f'for fixed-length Tuple with elements: {element_values!r}')
      min_size = len(element_values)
      max_size = min_size
    else:
      raise ValueError(
          f'Argument \'element_values\' must be a '
          f'non-empty list: {element_values!r}')

    if isinstance(element_values, ValueSpec):
      elements = [
          Field(TupleKey(None), element_values,
                'Field of variable-length tuple element')
      ]
    else:
      elements = []
      for i, element_value in enumerate(element_values):
        if not isinstance(element_value, ValueSpec):
          raise ValueError(
              f'Items in \'element_values\' must be ValueSpec objects.'
              f'Encountered: {element_value!r} at {i}.')
        elements.append(Field(TupleKey(i), element_value,
                              f'Field of tuple element at {i}'))

    self._min_size = min_size
    self._max_size = max_size
    self._elements = elements
    super().__init__(tuple, default, user_validator)

  @property
  def fixed_length(self) -> bool:
    """Returns True if current Tuple spec is fixed length."""
    return self._min_size == self._max_size

  @property
  def min_size(self) -> int:
    """Returns max size of this tuple."""
    return self._min_size

  @property
  def max_size(self) -> typing.Optional[int]:
    """Returns max size of this tuple."""
    return self._max_size

  @property
  def elements(self) -> typing.List[Field]:
    """Returns Field specification for tuple elements."""
    return self._elements

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    if self.fixed_length:
      return typing.Tuple[tuple([
          _any_if_no_annotation(elem.value.annotation)
          for elem in self._elements])]       # pytype: disable=invalid-annotation
    else:
      return typing.Tuple[self._elements[0].value.annotation, ...]  # pytype: disable=invalid-annotation

  def __len__(self) -> int:
    """Returns length of this tuple."""
    return len(self._elements) if self.fixed_length else 0

  def _apply(self,
             value: typing.Tuple[typing.Any, ...],
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any], typing.Any],
             root_path: object_utils.KeyPath) -> typing.Any:
    """Tuple specific apply."""
    if self.fixed_length:
      if len(value) != len(self.elements):
        raise ValueError(
            object_utils.message_on_path(
                f'Length of input tuple ({len(value)}) does not match the '
                f'length of spec ({len(self.elements)}). '
                f'Input: {value}, Spec: {self}', root_path))
    else:
      if len(value) < self.min_size:
        raise ValueError(
            object_utils.message_on_path(
                f'Length of tuple {value} is less than '
                f'min size ({self.min_size}).', root_path))
      if self.max_size is not None and len(value) > self.max_size:
        raise ValueError(
            object_utils.message_on_path(
                f'Length of tuple {value} is greater than '
                f'max size ({self.max_size}).', root_path))
    return tuple([
        self._elements[i if self.fixed_length else 0].apply(  # pylint: disable=g-complex-comprehension
            v, allow_partial, child_transform,
            object_utils.KeyPath(i, root_path))
        for i, v in enumerate(value)
    ])

  def _extend(self, base: 'Tuple') -> None:
    """Tuple specific extension."""
    if self.fixed_length and base.fixed_length:
      if len(self.elements) != len(base.elements):
        raise TypeError(
            f'{self} cannot extend {base}: unmatched number of elements.')
      for i, element in enumerate(self._elements):
        element.extend(base.elements[i])
    elif self.fixed_length and not base.fixed_length:
      if base.min_size > len(self):
        raise TypeError(
            f'{self} cannot extend {base} as it has '
            f'less elements than required.')
      if base.max_size is not None and base.max_size < len(self):
        raise TypeError(
            f'{self} cannot extend {base} as it has '
            f'more elements than required.')
      for i, element in enumerate(self._elements):
        element.extend(base.elements[0])
    elif not self.fixed_length and base.fixed_length:
      raise TypeError(
          f'{self} cannot extend {base}: a variable length tuple '
          f'cannot extend a fixed length tuple.')
    else:
      assert not self.fixed_length and not base.fixed_length
      if self.min_size != 0 and self.min_size < base.min_size:
        raise TypeError(
            f'{self} cannot extend {base} as it has smaller min size.')
      if (self.max_size is not None
          and base.max_size is not None
          and self.max_size > base.max_size):
        raise TypeError(
            f'{self} cannot extend {base} as it has greater max size.')
      if self._min_size == 0:
        self._min_size = base.min_size
      if self._max_size is None:
        self._max_size = base.max_size
      self.elements[0].extend(base.elements[0])

  def _is_compatible(self, other: 'Tuple') -> bool:
    """Tuple specific compatibility check."""
    if self.fixed_length and other.fixed_length:
      if len(self.elements) != len(other.elements):
        return False
      for i, element in enumerate(self._elements):
        if not element.value.is_compatible(other.elements[i].value):
          return False
      return True
    elif self.fixed_length and not other.fixed_length:
      return False
    elif not self.fixed_length and other.fixed_length:
      if self.min_size > len(other) or (
          self.max_size is not None and self.max_size < len(other)):
        return False
      for element in other.elements:
        if not self.elements[0].value.is_compatible(element.value):
          return False
      return True
    else:
      assert not self.fixed_length and not other.fixed_length
      if self.min_size > other.min_size:
        return False
      if self.max_size is not None and (
          other.max_size is None or self.max_size < other.max_size):
        return False
      return self.elements[0].value.is_compatible(other.elements[0].value)

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (super().__eq__(other)
            and self.elements == other.elements
            and self.min_size == other.min_size
            and self.max_size == other.max_size)

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             hide_default_values: bool = True,
             hide_missing_values: bool = True,
             **kwargs) -> typing.Text:
    """Format this object."""
    if self.fixed_length:
      element_values = [f.value for f in self._elements]
      details = object_utils.kvlist_str([
          ('', object_utils.format(
              element_values,
              compact=compact,
              verbose=verbose,
              root_indent=root_indent,
              **kwargs), None),
          ('default', object_utils.format(
              self._default,
              compact=compact,
              verbose=verbose,
              root_indent=root_indent + 1,
              hide_default_values=hide_default_values,
              hide_missing_values=hide_missing_values,
              **kwargs), 'MISSING_VALUE'),
          ('noneable', self._is_noneable, False),
          ('frozen', self._frozen, False),
      ])
      return f'{self.__class__.__name__}({details})'
    else:
      details = object_utils.kvlist_str([
          ('', object_utils.format(
              self._elements[0].value,
              compact=compact,
              verbose=verbose,
              root_indent=root_indent,
              **kwargs), None),
          ('default', object_utils.format(
              self._default,
              compact=compact,
              verbose=verbose,
              root_indent=root_indent + 1,
              hide_default_values=hide_default_values,
              hide_missing_values=hide_missing_values,
              **kwargs), 'MISSING_VALUE'),
          ('min_size', self._min_size, 0),
          ('max_size', self._max_size, None),
          ('noneable', self._is_noneable, False),
      ])
      return f'{self.__class__.__name__}({details})'


class Dict(ValueSpecBase):
  """Value spec for dict type.

  Examples::

    # A required symbolic dict of arbitrary keys and values.
    pg.typing.Dict()

    # An optional Dict of keys started with 'foo' and int values, with None as
    # the default value.
    pg.typing.Dict([
        (pg.typing.StrKey(), pg.typing.Int())
    ]).noneable()

    # A dict with two keys ('x' and 'y').
    pg.typing.Dict([
        ('x', pg.typing.Float()),
        ('y', pg.typing.Int(min_value=1))
    ])

    # A dict with a user validator.
    def validate_sum(d):
      if sum(d.values()) > 1.:
        raise ValueError('The sum of the dict values should be less than 1.')

    pg.typing.Dict([
        (pg.typing.StrKey(), pg.typing.Float())
    ], user_validator=validate_sum)

    # A frozen dict that prevents subclass to extend/override.
    pg.typing.Dict([
        ('x', 1),
        ('y, 2.0)
    ]).freeze()
  """

  def __init__(
      self,
      schema_or_field_list: typing.Optional[typing.Union[
          Schema, typing.List[typing.Union[Field, typing.Tuple]]  # pylint: disable=g-bare-generic
      ]] = None,  # pylint: disable=bad-whitespace
      user_validator: typing.Optional[
          typing.Callable[[typing.Dict[typing.Any, typing.Any]], None]] = None):
    """Constructor.

    Args:
      schema_or_field_list: (Optional) a Schema object for this Dict,
        or a list of Field or Field equivalents: tuple of (<key_spec>,
          <value_spec>, [description], [metadata]) When this field is empty, it
          specifies a schema-less Dict that may accept arbitrary key/value
          pairs.
      user_validator: (Optional) user function or callable object for additional
        validation on the applied dict, which can reject a value by raising
        Exceptions. Please note that this validation is an addition to
        validating nested members by their schema if present.
    """
    default_value = MISSING_VALUE
    schema = None
    if schema_or_field_list is not None:
      if isinstance(schema_or_field_list, Schema):
        schema = schema_or_field_list
      else:
        schema = create_schema(schema_or_field_list, allow_nonconst_keys=True)
      default_value = schema.apply({}, allow_partial=True)
    self._schema = schema
    super().__init__(dict, MISSING_VALUE, user_validator)
    self.set_default(default_value, use_default_apply=False)

  @property
  def schema(self) -> typing.Optional[Schema]:
    """Returns the schema of this dict spec."""
    return self._schema

  def noneable(self) -> 'Dict':
    """Override noneable in Dict to always set default value None."""
    self._is_noneable = True
    self._default = None
    return self

  def _apply(self,
             value: typing.Dict[typing.Any, typing.Any],
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any], typing.Any],
             root_path: object_utils.KeyPath) -> typing.Any:
    """Dict specific apply."""
    if not self._schema:
      return value
    return self._schema.apply(value, allow_partial, child_transform, root_path)

  def _extend(self, base: 'Dict') -> None:
    """Dict specific extension."""
    if base.schema:
      if not self._schema:
        self._schema = copy.deepcopy(base.schema)
        self._default = copy.deepcopy(base._default)  # pylint: disable=protected-access
      else:
        self._schema.extend(base.schema)
        self._default = self._schema.apply({}, allow_partial=True)

  def _is_compatible(self, other: 'Dict') -> bool:
    """Dict specific compatibility check."""
    if self._schema:
      if other.schema is None:
        return False
      return self._schema.is_compatible(other.schema)
    return True

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return typing.Dict[typing.Text, typing.Any]

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return super().__eq__(other) and self.schema == other.schema

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Format this object."""
    schema_details = ''
    if self._schema:
      schema_details = self._schema.format(
          compact,
          verbose,
          root_indent,
          cls_name='',
          bracket_type=object_utils.BracketType.CURLY,
          **kwargs)

    details = object_utils.kvlist_str([
        ('', schema_details, ''),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False),
    ])
    return f'{self.__class__.__name__}({details})'


class Object(ValueSpecBase):
  """Value spec for object type.

  Examples::

    # A required instance of class A and its subclasses.
    pg.typing.Object(A)

    # An optional instance of class A with None as its default value.
    pg.typing.Object(A).noneable()

    # An instance of class A with default value.
    pg.typing.Object(A, default=A())
  """

  def __init__(
      self,
      cls: typing.Type[typing.Any],
      default: typing.Any = MISSING_VALUE,
      user_validator: typing.Optional[
          typing.Callable[[typing.Any], None]] = None):
    """Constructor.

    Args:
      cls: Class of the object. Objects of subclass of this class is acceptable.
      default: (Optional) default value of this spec.
      user_validator: (Optional) user function or callable object for additional
        validation on the applied object, which can reject a value by raising
        Exceptions. Please note that this validation is an addition to
        validating object type constraint.
    """
    if cls is None:
      raise TypeError('"cls" for Object spec cannot be None.')
    if not isinstance(cls, type):
      raise TypeError('"cls" for Object spec should be a type.')
    if cls is object:
      raise TypeError('<class \'object\'> is too general for Object spec.')
    super().__init__(cls, default, user_validator)

  @property
  def cls(self) -> typing.Type[typing.Any]:
    """Returns the class of this object spec."""
    return self.value_type

  def _apply(self,
             value: typing.Any,
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any], typing.Any],
             root_path: object_utils.KeyPath) -> typing.Any:
    """Object specific apply."""
    del child_transform
    if isinstance(value, object_utils.MaybePartial):
      if not allow_partial and value.is_partial:
        raise ValueError(
            object_utils.message_on_path(
                f'Object {value} is not fully bound.', root_path))
    return value

  def extend(self, base: ValueSpec) -> ValueSpec:
    """Extend current value spec on top of a base spec."""
    if isinstance(base, Callable) and base.is_compatible(self):
      return self
    return super().extend(base)

  def _extend(self, base: 'Object') -> None:
    """Object specific extension."""
    if not issubclass(self.value_type, base.value_type):
      raise TypeError(f'{self!r} cannot extend {base!r}: incompatible class.')

  def _is_compatible(self, other: 'Object') -> bool:
    """Object specific compatiblity check."""
    return issubclass(other.cls, self.cls)

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return self.cls

  @property
  def schema(self) -> typing.Optional[Schema]:
    """Returns the schema of object class if available."""
    return getattr(self.value_type, 'schema', None)

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (super().__eq__(other) and
            self.value_type == other.value_type)

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             hide_default_values: bool = True,
             hide_missing_values: bool = True,
             **kwargs) -> typing.Text:
    """Format this object."""
    details = object_utils.kvlist_str([
        ('', self._value_type.__name__, None),
        ('default', object_utils.format(
            self._default,
            compact,
            verbose,
            root_indent,
            hide_default_values=hide_default_values,
            hide_missing_values=hide_missing_values,
            **kwargs), 'MISSING_VALUE'),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False),
    ])
    return f'{self.__class__.__name__}({details})'


class Callable(ValueSpecBase):
  """Value spec for callable.

  Examples::

    # A required callable object with any args.
    pg.typing.Callable()

    # An optional callable objects with the first argument as int, and the
    # second argument as float. The field has None as its default value.
    pg.typing.Callable([pg.typing.Int(), pg.typing.Float()]).noneable()

    # An callable object that has its first argument as int, and has keyword
    # arguments 'x' (any type), 'y' (a str) and return value as int.
    pg.typing.Callable(
        [pg.typing.Int()],
        kw=[
            ('x', pg.typing.Any()),
            ('y', pg.typing.Str())
        ],
        returns=pg.typing.Int())

  See also: :class:`pyglove.typing.Functor`.
  """

  def __init__(
      self,
      args: typing.Optional[typing.List[ValueSpec]] = None,
      kw: typing.Optional[
          typing.List[typing.Tuple[typing.Text, ValueSpec]]] = None,
      returns: typing.Optional[ValueSpec] = None,
      default: typing.Any = MISSING_VALUE,
      user_validator: typing.Optional[
          typing.Callable[[typing.Callable], None]] = None,  # pylint: disable=g-bare-generic
      callable_type: typing.Optional[typing.Type] = None):  # pylint: disable=g-bare-generic
    """Constructor."""
    args = args or []
    kw = kw or []
    if not isinstance(args, list):
      raise TypeError(
          f'\'args\' should be a list of ValueSpec objects. '
          f'Encountered: {args!r}.')
    for arg in args:
      if not isinstance(arg, ValueSpec):
        raise TypeError(
            f'\'args\' should be a list of ValueSpec objects. '
            f'Encountered: {args!r}.')

    if not isinstance(kw, list):
      raise TypeError(
          f'\'kw\' should be a list of (name, value_spec) tuples. '
          f'Encountered: {kw!r}.')
    for arg in kw:
      if (not isinstance(arg, tuple) or len(arg) != 2 or
          not isinstance(arg[0], str) or
          not isinstance(arg[1], ValueSpec)):
        raise TypeError(
            f'\'kw\' should be a list of (name, value_spec) tuples. '
            f'Encountered: {kw!r}.')

    if returns is not None and not isinstance(returns, ValueSpec):
      raise TypeError(
          f'\'returns\' should be a ValueSpec object. Encountered: {returns!r}')
    self._args = args
    self._kw = kw
    self._return_value = returns
    super().__init__(callable_type, default, user_validator)

  @property
  def args(self) -> typing.List[ValueSpec]:
    """Value specs for positional arguments."""
    return self._args

  @property
  def kw(self) -> typing.List[typing.Tuple[typing.Text, ValueSpec]]:
    """Names and value specs for keyword arguments."""
    return self._kw

  @property
  def return_value(self) -> typing.Optional[ValueSpec]:
    """Value spec for return value."""
    return self._return_value

  def _validate(self, path: object_utils.KeyPath, value: typing.Any) -> None:
    """Validate applied value."""
    if not callable(value):
      raise TypeError(
          object_utils.message_on_path(
              f'Value is not callable: {value!r}.', path))

    # Shortcircuit if there is no signature to check.
    if not (self._args or self._kw or self._return_value):
      return

    signature = get_signature(value)

    if len(self._args) > len(signature.args) and not signature.has_varargs:
      raise TypeError(
          object_utils.message_on_path(
              f'{signature.id} only take {len(signature.args)} positional '
              f'arguments, while {len(self._args)} is required by {self!r}.',
              path))

    # Check positional arguments.
    for i in range(min(len(self._args), len(signature.args))):
      src_spec = self._args[i]
      dest_spec = signature.args[i].value_spec
      if not dest_spec.is_compatible(src_spec):
        raise TypeError(
            object_utils.message_on_path(
                f'Value spec of positional argument {i} is not compatible. '
                f'Expected: {dest_spec!r}, Actual: {src_spec!r}.',
                path))
    if len(self._args) > len(signature.args):
      assert signature.has_varargs
      assert signature.varargs  # for pytype
      dest_spec = signature.varargs.value_spec
      for i in range(len(signature.args), len(self._args)):
        src_spec = self._args[i]
        if not dest_spec.is_compatible(src_spec):
          raise TypeError(
              object_utils.message_on_path(
                  f'Value spec of positional argument {i} is not compatible '
                  f'with the value spec of *{signature.varargs.name}. '
                  f'Expected: {dest_spec!r}, Actual: {src_spec!r}.', path))

    # Check keyword arguments.
    dest_args = signature.args + signature.kwonlyargs
    for arg_name, src_spec in self._kw:
      dest_spec = None
      for dest_arg in dest_args:
        if dest_arg.name == arg_name:
          dest_spec = dest_arg.value_spec
          break
      if dest_spec is not None:
        if not dest_spec.is_compatible(src_spec):
          raise TypeError(
              object_utils.message_on_path(
                  f'Value spec of keyword argument {arg_name!r} is not '
                  f'compatible. Expected: {src_spec!r}, Actual: {dest_spec!r}.',
                  path))
      elif signature.has_varkw:
        assert signature.varkw  # for pytype
        if not signature.varkw.value_spec.is_compatible(src_spec):
          raise TypeError(
              object_utils.message_on_path(
                  f'Value spec of keyword argument {arg_name!r} is not '
                  f'compatible with the value spec of '
                  f'**{signature.varkw.name}. '
                  f'Expected: {signature.varkw.value_spec!r}, '
                  f'Actual: {src_spec!r}.', path))
      else:
        raise TypeError(
            object_utils.message_on_path(
                f'Keyword argument {arg_name!r} does not exist in {value!r}.',
                path))

    # Check return value
    if (self._return_value and signature.return_value and
        not self._return_value.is_compatible(signature.return_value)):
      raise TypeError(
          object_utils.message_on_path(
              f'Value spec for return value is not compatible. '
              f'Expected: {self._return_value!r}, '
              f'Actual: {signature.return_value!r} ({value!r}).',
              path))

  def _extend(self, base: 'Callable') -> None:
    """Callable specific extension."""
    if not self._args:
      self._args = list(base.args)
    if not self._kw:
      self._kw = list(base.kw)
    if not self._return_value:
      self._return_value = base.return_value

  def is_compatible(self, other: typing.Any) -> bool:
    if isinstance(other, Object):
      call_method = getattr(other.value_type, '__call__', None)
      if call_method is None or not inspect.isfunction(call_method):
        return False
      try:
        self.apply(call_method)
      except:  # pylint: disable=bare-except
        return False
      return True
    return super().is_compatible(other)

  def _is_compatible(self, other: 'Callable') -> bool:
    """Callable specific compatible check."""
    if len(self._args) > len(other.args):
      return False
    if len(self._kw) > len(other.kw):
      return False
    for i, dest_spec in enumerate(self._args):
      src_spec = other.args[i]
      if not dest_spec.is_compatible(src_spec):
        return False
    src_kw = {k: v for k, v in other.kw}
    for arg_name, dest_spec in self._kw:
      if arg_name not in src_kw:
        return False
      if not dest_spec.is_compatible(src_kw[arg_name]):
        return False
    if self._return_value and other.return_value:
      return self._return_value.is_compatible(other.return_value)
    return True

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    if self._kw:
      args = ...
    elif self._args:
      args = [_any_if_no_annotation(arg.annotation) for arg in self._args]
    else:
      args = []

    if self._return_value:
      return_value = _any_if_no_annotation(self._return_value.annotation)
    else:
      return_value = None
    return typing.Callable[args, return_value]  # pytype: disable=invalid-annotation

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    return (super().__eq__(other)
            and self._args == other.args
            and self._kw == other.kw
            and self._return_value == other.return_value)

  def format(self, **kwargs) -> typing.Text:
    """Format this spec."""
    details = object_utils.kvlist_str([
        ('args', object_utils.format(self._args, **kwargs), '[]'),
        ('kw', object_utils.format(self._kw, **kwargs), '[]'),
        ('returns', object_utils.format(self._return_value, **kwargs), 'None'),
        ('default', object_utils.format(
            self._default, **kwargs), 'MISSING_VALUE'),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False)
    ])
    return f'{self.__class__.__name__}({details})'


class Functor(Callable):
  """Value spec for Functor.

  Examples::

    # A required PyGlove functor with any args.
    pg.typing.Functor()

    # An optional functor with the first argument as int, and the second
    # argument as float. The field has None as its default value.
    pg.typing.Functor([pg.typing.Int(), pg.typing.Float()]).noneable()

    # A functor that has its first argument as int, and has keyword
    # arguments 'x' (any type), 'y' (a str) and return value as int.
    pg.typing.Functor(
        [pg.typing.Int()],
        kw=[
            ('x', pg.typing.Any()),
            ('y', pg.typing.Str())
        ],
        returns=pg.typing.Int())

  See also: :class:`pyglove.typing.Callable`.
  """

  def __init__(
      self,
      args: typing.Optional[typing.List[ValueSpec]] = None,
      kw: typing.Optional[
          typing.List[typing.Tuple[typing.Text, ValueSpec]]] = None,
      returns: typing.Optional[ValueSpec] = None,
      default: typing.Any = MISSING_VALUE,
      user_validator: typing.Optional[
          typing.Callable[[typing.Callable], None]] = None):  # pylint: disable=g-bare-generic
    """Constructor."""
    super().__init__(
        args=args,
        kw=kw,
        returns=returns,
        default=default,
        user_validator=user_validator,
        callable_type=object_utils.Functor)

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return object_utils.Functor


class Type(ValueSpecBase):
  """Value spec for type.

  Examples::

    # A required type or subclass of A.
    pg.typing.Type(A)

    # An optional type or subclass of A.
    pg.typing.Type(A).noneable()

    # A required type or subclass of A with default value B
    # (B is a subclass of A).
    pg.typing.Type(A, default=B)

  """

  def __init__(
      self,
      t: typing.Type,  # pylint: disable=g-bare-generic
      default: typing.Type = MISSING_VALUE):  # pylint: disable=g-bare-generic  # pytype: disable=annotation-type-mismatch
    if not isinstance(t, type):
      raise TypeError(f'{t!r} is not a type.')
    self._expected_type = t
    super().__init__(type, default)

  @property
  def type(self):
    """Returns desired type."""
    return self._expected_type

  def _validate(self, path: object_utils.KeyPath, value: typing.Type) -> None:  # pylint: disable=g-bare-generic
    """Validate applied value."""
    if not issubclass(value, self.type):
      raise ValueError(
          object_utils.message_on_path(
              f'{value!r} is not a subclass of {self.type!r}', path))

  def _is_compatible(self, other: 'Type') -> bool:
    """Type specific compatiblity check."""
    return issubclass(other.type, self.type)

  def _extend(self, base: 'Type') -> None:
    """Type specific extension."""
    if not issubclass(self.type, base.type):
      raise TypeError(f'{self!r} cannot extend {base!r}: incompatible type.')

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return typing.Type[self._expected_type]

  def __eq__(self, other: 'Type') -> bool:
    """Equals."""
    return super().__eq__(other) and self.type == other.type

  def format(self, **kwargs):
    """Format this object."""
    details = object_utils.kvlist_str([
        ('', self.type, None),
        ('default', self._default, MISSING_VALUE),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False),
    ])
    return f'{self.__class__.__name__}({details})'


class Union(ValueSpecBase):
  """Value spec for Union.

  Examples::

    # A required int or float value.
    pg.typing.Union([pg.typing.Int(), pg.typing.Float()])

    # An optional int or float value with default set to None.
    pg.typing.Union([pg.typing.Int(), pg.typing.Float()]).noneable()

    # A dict of specific keys, instance of class A or B, with {x=1} as its
    # default value.
    pg.typing.Union([
        pg.typing.Dict([
            ('x', pg.typing.Int(min_value=1)),
        ]),
        pg.typing.Object(A),
        pg.typing.Object(B),
    ], default={'x': 1})
  """

  def __init__(self,
               candidates: typing.List[ValueSpec],
               default: typing.Any = MISSING_VALUE):
    """Constructor.

    Args:
      candidates: Value spec for candidate types.
      default: (Optional) default value of this spec.
    """
    if not isinstance(candidates, list) or len(candidates) < 2:
      raise ValueError(
          f'Argument \'candidates\' must be a list of at least 2 '
          f'elements. Encountered {candidates}.')
    candidates_by_type = {}
    has_noneable_candidate = False
    for i, c in enumerate(candidates):
      if not isinstance(c, ValueSpec):
        raise ValueError(
            f'Items in \'candidates\' must be ValueSpec objects.'
            f'Encountered {c} at {i}.')
      if c.is_noneable:
        has_noneable_candidate = True

      # pytype: disable=attribute-error
      spec_type = (c.__class__, c._value_type)  # pylint: disable=protected-access
      # pytype: enable=attribute-error
      if spec_type not in candidates_by_type:
        candidates_by_type[spec_type] = []
      candidates_by_type[spec_type].append(c)

    for spec_type, cs in candidates_by_type.items():
      if len(cs) > 1:
        # NOTE(daiyip): Now we simply reject union of multiple value spec of
        # the same type. We may consider support Union of different List, Tuple,
        # Dict and Object later.
        raise ValueError(
            f'Found {len(cs)} value specs of the same type {spec_type}.')

    self._candidates = candidates
    candidate_types = {c.value_type for c in candidates}
    union_value_type = (None
                        if None in candidate_types else tuple(candidate_types))
    super().__init__(union_value_type, default)
    if has_noneable_candidate:
      super().noneable()

  def noneable(self) -> 'Union':
    """Customized noneable for Union."""
    super().noneable()
    for c in self._candidates:
      c.noneable()
    return self

  @property
  def candidates(self) -> typing.List[ValueSpec]:
    """Returns candidate types of this union spec."""
    return self._candidates

  def get_candidate(
      self, dest_spec: ValueSpec) -> typing.Optional[ValueSpec]:
    """Get candidate by a destination value spec.

    Args:
      dest_spec: destination value spec which is a superset of the value spec
        to return. E.g. Any (dest_spec) is superset of Int (child spec).

    Returns:
      The first value spec under Union with which the destination value spec
        is compatible.
    """
    # NOTE(daiyip): we always try matching the candidate with the same
    # value spec type first, then different type but compatible specs.
    for c in self._candidates:
      if dest_spec.__class__ == c.__class__ and dest_spec.is_compatible(c):
        return c

    for c in self._candidates:
      if isinstance(c, Union):
        child = c.get_candidate(dest_spec)
        if child is not None:
          return child
      else:
        if dest_spec.is_compatible(c):
          return c
    return None

  def _apply(self,
             value: typing.Any,
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any],
                 typing.Any
             ],
             root_path: object_utils.KeyPath) -> typing.Any:
    """Union specific apply."""
    for c in self._candidates:
      if c.value_type is None or isinstance(value, c.value_type):
        return c.apply(value, allow_partial, child_transform, root_path)

    # NOTE(daiyip): This code is to support consider A as B scenario when there
    # is a converter from A to B (converter may return value that is not B). A
    # use case is that tf.Variable is not a tf.Tensor, but value spec of
    # tf.Tensor should be able to accept tf.Variable.
    for c in self._candidates:
      if get_first_applicable_converter(type(value), c.value_type) is not None:
        return c.apply(value, allow_partial, child_transform, root_path)

    raise ValueError('Should never happen: _apply is entered only when there '
                     'is a type match or convertible path.')

  def _extend(self, base: 'Union') -> None:
    """Union specific extension."""
    def _base_candidate(c, v):
      """Find a non-Union base spec from `v` for a input spec `c`."""
      if isinstance(v, Union):
        for vc in v.candidates:
          p = _base_candidate(c, vc)
          if p is not None:
            return p
      else:
        if (c.__class__ is v.__class__
            and (c.__class__ is not Object
                 or issubclass(c.value_type, v.value_type))):
          return v
      return None

    for sc in self.candidates:
      bc = _base_candidate(sc, base)
      if bc is None:
        raise TypeError(
            f'{self!r} cannot extend {base!r}: incompatible value spec {sc}.')
      sc.extend(bc)

  def is_compatible(self, other: ValueSpec) -> bool:
    """Union specific compatibility check."""
    if isinstance(other, Union):
      for oc in other.candidates:
        if not self.is_compatible(oc):
          return False
      return True
    else:
      for c in self._candidates:
        if c.is_compatible(other):
          return True
      return False

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    candidates = tuple([
        _any_if_no_annotation(c.annotation) for c in self._candidates
    ])
    return typing.Union[candidates]

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> typing.Text:
    """Format this object."""
    list_wrap_threshold = kwargs.pop('list_wrap_threshold', 20)
    details = object_utils.kvlist_str([
        ('', object_utils.format(
            self._candidates,
            compact,
            verbose,
            root_indent + 1,
            list_wrap_threshold=list_wrap_threshold,
            **kwargs), None),
        ('default', object_utils.quote_if_str(self._default), MISSING_VALUE),
        ('noneable', self._is_noneable, False),
        ('frozen', self._frozen, False),
    ])
    return f'{self.__class__.__name__}({details})'

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if not super().__eq__(other):
      return False
    if len(self.candidates) != len(other.candidates):
      return False
    for sc in self.candidates:
      oc = other.get_candidate(sc)
      if sc != oc:
        return False
    return True


class Any(ValueSpecBase):
  """Value spec for any type.

  Examples::

    # A required value of any type.
    pg.typing.Any()

    # An optional value of any type, with None as its default value.
    pg.typing.Any().noneable()

    # A required value of any type, with 1 as its default value.
    pg.typing.Any(default=1)

  .. note::

    While Any type is very flexible and useful to pass though data between
    components, we should minimize its usage since minimal validation is
    performed on this type.
  """

  def __init__(
      self,
      default: typing.Any = object_utils.MISSING_VALUE,
      annotation: typing.Any = object_utils.MISSING_VALUE,
      user_validator: typing.Optional[
          typing.Callable[[typing.Any], None]] = None):
    """Constructor.

    Args:
      default: (Optional) default value of this spec.
      annotation: (Optional) external provided type annotation.
      user_validator: (Optional) user function or callable object for additional
        validation on the applied value, which can reject a value by raising
        Exceptions.
    """
    super().__init__(object, default, user_validator, is_noneable=True)
    self._annotation = annotation

  def is_compatible(self, other: ValueSpec) -> bool:
    """Any is compatible with any ValueSpec."""
    return True

  def format(self, **kwargs) -> typing.Text:
    """Format this object."""
    details = object_utils.kvlist_str([
        ('default', object_utils.format(self._default, **kwargs),
         'MISSING_VALUE'),
        ('frozen', self._frozen, False),
        ('annotation', self._annotation, MISSING_VALUE)
    ])
    return f'{self.__class__.__name__}({details})'

  def annotate(self, annotation: typing.Any) -> 'Any':
    """Set external type annotation."""
    self._annotation = annotation
    return self

  @property
  def annotation(self) -> typing.Any:
    """Returns type annotation."""
    return self._annotation

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (super().__eq__(other)
            and self.annotation == other.annotation)


#
# Helper classes and functions.
#

Argument = collections.namedtuple('Argument', ['name', 'value_spec'])


class CallableType(enum.Enum):
  """Enum for Callable type."""
  # Regular function or lambdas without a subject bound.
  FUNCTION = 1

  # Function that is bound with subject. Like class methods or instance methods.
  METHOD = 2


class Signature(object_utils.Formattable):
  """PY3 function signature."""

  def __init__(self,
               callable_type: CallableType,
               name: typing.Text,
               module_name: typing.Text,
               args: typing.Optional[typing.List[Argument]] = None,
               kwonlyargs: typing.Optional[typing.List[Argument]] = None,
               varargs: typing.Optional[Argument] = None,
               varkw: typing.Optional[Argument] = None,
               return_value: typing.Optional[ValueSpec] = None,
               qualname: typing.Optional[typing.Text] = None):
    """Constructor.

    Args:
      callable_type: Type of callable.
      name: Function name.
      module_name: Module name.
      args: Specification for positional arguments
      kwonlyargs: Specification for keyword only arguments (PY3).
      varargs: Specification for wildcard list argument, e.g, 'args' is the name
        for `*args`.
      varkw: Specification for wildcard keyword argument, e.g, 'kwargs' is the
        name for `**kwargs`.
      return_value: Optional value spec for return value.
      qualname: Optional qualified name.
    """
    args = args or []
    self.callable_type = callable_type
    self.name = name
    self.module_name = module_name
    self.args = args or []
    self.kwonlyargs = kwonlyargs or []
    self.varargs = varargs
    self.varkw = varkw
    self.return_value = return_value
    self.qualname = qualname or name

  @property
  def named_args(self):
    """Returns all named arguments according to their declaration order."""
    return self.args + self.kwonlyargs

  @property
  def arg_names(self):
    """Returns names of all arguments according to their declaration order."""
    return [arg.name for arg in self.named_args]

  def get_value_spec(self, name: typing.Text) -> typing.Optional[ValueSpec]:
    """Returns Value spec for an argument name.

    Args:
      name: Argument name.

    Returns:
      ValueSpec for the requested argument. If name is not found, value spec of
      wildcard keyword argument will be used. None will be returned if name
      does not exist in signature and wildcard keyword is not accepted.
    """
    for arg in self.named_args:
      if arg.name == name:
        return arg.value_spec
    if self.has_varkw:
      return self.varkw.value_spec
    return None

  @property
  def id(self) -> typing.Text:
    """Returns ID of the function."""
    return f'{self.module_name}.{self.qualname}'

  @property
  def has_varargs(self) -> bool:
    """Returns whether wildcard positional argument is present."""
    return self.varargs is not None

  @property
  def has_varkw(self) -> bool:
    """Returns whether wildcard keyword argument is present."""
    return self.varkw is not None

  @property
  def has_wildcard_args(self) -> bool:
    """Returns whether any wildcard arguments are present."""
    return self.has_varargs or self.has_varkw

  def __ne__(self, other: typing.Any) -> bool:
    """Not equals."""
    return not self.__eq__(other)

  def __eq__(self, other: typing.Any) -> bool:
    """Equals."""
    if not isinstance(other, self.__class__):
      return False
    if self is other:
      return True
    return (self.callable_type == other.callable_type and
            self.name == other.name and
            self.qualname == other.qualname and
            self.module_name == other.module_name and
            self.args == other.args and self.kwonlyargs == other.kwonlyargs and
            self.varargs == other.varargs and self.varkw == other.varkw and
            self.return_value == other.return_value)

  def format(self, **kwargs) -> typing.Text:
    """Format current object."""
    details = object_utils.kvlist_str([
        ('', repr(self.id), ''),
        ('args', object_utils.format(self.args, **kwargs), '[]'),
        ('kwonlyargs', object_utils.format(self.kwonlyargs, **kwargs), '[]'),
        ('returns', object_utils.format(self.return_value, **kwargs), 'None'),
        ('varargs', object_utils.format(self.varargs, **kwargs), 'None'),
        ('varkw', object_utils.format(self.varkw, **kwargs), 'None'),
    ])
    return f'{self.__class__.__name__}({details})'

  @classmethod
  def from_callable(cls, callable_object: typing.Callable) -> 'Signature':  # pylint: disable=g-bare-generic
    """Creates Signature from a callable object."""
    callable_object = typing.cast(object, callable_object)
    if not callable(callable_object):
      raise TypeError(f'{callable_object!r} is not callable.')

    if isinstance(callable_object, object_utils.Functor):
      assert callable_object.signature is not None
      return callable_object.signature

    func = callable_object
    if not inspect.isroutine(func):
      if not inspect.isroutine(callable_object.__call__):
        raise TypeError(f'{callable_object!r}.__call__ is not a method.')
      func = callable_object.__call__

    def make_arg_spec(param: inspect.Parameter) -> Argument:
      value_spec = Any()
      if param.default != inspect.Parameter.empty:
        value_spec.set_default(param.default)
      if param.annotation != inspect.Parameter.empty:
        value_spec.annotate(param.annotation)
      return Argument(param.name, value_spec)

    sig = inspect.signature(func)
    args = []
    kwonly_args = []
    varargs = None
    varkw = None

    for param in sig.parameters.values():
      arg_spec = make_arg_spec(param)
      if (param.kind == inspect.Parameter.POSITIONAL_ONLY
          or param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
        args.append(arg_spec)
      elif param.kind == inspect.Parameter.KEYWORD_ONLY:
        kwonly_args.append(arg_spec)
      elif param.kind == inspect.Parameter.VAR_POSITIONAL:
        varargs = arg_spec
      else:
        assert param.kind == inspect.Parameter.VAR_KEYWORD, param.kind
        varkw = arg_spec

    if inspect.ismethod(func):
      callable_type = CallableType.METHOD
    else:
      callable_type = CallableType.FUNCTION

    return Signature(
        callable_type=callable_type,
        name=func.__name__,
        module_name=getattr(func, '__module__', 'wrapper'),
        qualname=func.__qualname__,
        args=args, kwonlyargs=kwonly_args, varargs=varargs, varkw=varkw)

  def make_function(
      self,
      body: typing.List[typing.Text],
      exec_globals: typing.Optional[
          typing.Dict[typing.Text, typing.Any]] = None,
      exec_locals: typing.Optional[
          typing.Dict[typing.Text, typing.Any]] = None):
    """Makes a function with current signature."""
    exec_globals = exec_globals or {}
    exec_locals = exec_locals or {}

    args = []
    def _append_arg(
        arg_name: typing.Text,
        arg_spec: ValueSpec,
        force_missing_as_default: bool = False,
        arg_prefix: typing.Text = ''):
      s = [f'{arg_prefix}{arg_name}']
      if arg_spec.annotation != MISSING_VALUE:
        s.append(f': _annotation_{arg_name}')
        exec_locals[f'_annotation_{arg_name}'] = arg_spec.annotation
      if not arg_prefix and (force_missing_as_default or arg_spec.has_default):
        s.append(f' = _default_{arg_name}')
        exec_locals[f'_default_{arg_name}'] = arg_spec.default
      args.append(''.join(s))

    has_previous_default = False
    # Build positional arguments.
    for arg in self.args:
      _append_arg(arg.name, arg.value_spec, has_previous_default)
      if arg.value_spec.has_default:
        has_previous_default = True

    # Build variable positional arguments.
    if self.varargs:
      _append_arg(self.varargs.name, self.varargs.value_spec, arg_prefix='*')
    elif self.kwonlyargs:
      args.append('*')

    # Build keyword-only arguments.
    for arg in self.kwonlyargs:
      _append_arg(arg.name, arg.value_spec)

    # Build variable keyword arguments.
    if self.varkw:
      _append_arg(self.varkw.name, self.varkw.value_spec, arg_prefix='**')

    # Generate function.
    fn = object_utils.make_function(
        self.name,
        args=args,
        body=body,
        exec_globals=exec_globals,
        exec_locals=exec_locals,
        return_type=self.return_value or MISSING_VALUE)
    fn.__module__ = self.module_name
    fn.__name__ = self.name
    fn.__qualname__ = self.qualname
    return fn


def _any_if_no_annotation(annotation: typing.Any):
  """Returns typing.Any if annotation is MISSING_VALUE."""
  return typing.Any if annotation == MISSING_VALUE else annotation


def get_signature(func: typing.Callable) -> Signature:  # pylint:disable=g-bare-generic
  """Gets signature from a python callable."""
  return Signature.from_callable(func)


def get_arg_fields(
    signature: Signature,
    args: typing.Optional[typing.List[typing.Union[typing.Tuple[
        typing.Tuple[typing.Text, KeySpec], ValueSpec,
        typing.Text], typing.Tuple[typing.Tuple[typing.Text, KeySpec],
                                   ValueSpec, typing.Text, typing.Any]]]] = None
) -> typing.List[Field]:
  """Get schema fields for the arguments from a function or method signature.

  Args:
    signature: A `Signature` object.
    args: (Optional) explicit value specifications for the arguments, which is a
      list of tuples in:
      (<argumment-name>, <value-spec>, [description], [metadata-objects]).

      * `argument-name` - a string or a `StrKey` object. This name must exist
        in the signature's argument list, unless the signature has a
        ``**kwargs``, where the argument name can be an acceptable key in the
        dict that is passed to the ``**kwargs``. If the argument name is a
        ``StrKey`` object, it specifies a field that matches any keys beyond
        the regular arguments for the ``**kwargs``.
      * `value-spec` - a ``ValueSpec`` object asssociate with the argument
        name.
      * `description` - an optional string as the description for the argument.
      * `metadata-objects` - an optional list of any type, which can be
        used to generate code according to the schema.

  Returns:
    `Field` objects for the arguments from the `signature` in declaration order.
    If an argument is not present in `args`, it will be considered an `Any`.
    Otherwise it will create a `Field` from the explicit specifications. Default
    values for the arguments will be automatially propagated from the signature
    to the fields.

  Raises:
    KeyError: If argument names defined in `args` does not match with the
      arguments from the signature.
    TypeError: The value spec defined in `args` is not compatible with the value
      spec inspected from the signature.
    ValueError: The value spec defined in `args` does not align with the default
      values from the signature.
  """
  arg_dict = dict()
  kwarg_spec = None
  varargs_spec = None

  func_arg_names = set(signature.arg_names)
  # Extra legal argument names that are out of function signature, it is not
  # empty only when function allow **kwargs.
  extra_arg_names = []
  for arg in args or []:
    if isinstance(arg[0], StrKey):
      if kwarg_spec is not None:
        raise KeyError(
            f'{signature.id}: multiple StrKey found in '
            f'symbolic arguments declaration.')
      kwarg_spec = arg
    else:
      assert isinstance(arg[0], (str, ConstStrKey))
      if arg[0] in arg_dict:
        raise KeyError(
            f'{signature.id}: multiple symbolic fields '
            f'found for argument {arg[0]!r}.')
      if signature.varargs and signature.varargs.name == arg[0]:
        varargs_spec = arg
      elif arg[0] not in func_arg_names:
        if signature.has_varkw:
          extra_arg_names.append(arg[0])
        else:
          raise KeyError(
              f'{signature.id}: found extra symbolic argument {arg[0]!r}.')
      arg_dict[arg[0]] = arg

  def get_arg_field(arg_spec):
    arg_name = arg_spec.name
    decl_spec = arg_spec.value_spec
    if arg_name not in arg_dict:
      # Automatic generate symbolic declaration for missing arguments.
      arg_field = (arg_name, decl_spec, f'Argument {arg_name!r}.')
    else:
      arg_field = arg_dict[arg_name]
      if not decl_spec.is_compatible(arg_field[1]):
        raise TypeError(
            f'{signature.id}: the value spec ({arg_field[1]!r}) of symbolic '
            f'argument {arg_name} is not compatible with the value spec '
            f'({decl_spec!r}) from function signature.')
      if arg_field[1].default in [MISSING_VALUE, None]:
        arg_field[1].extend(decl_spec).set_default(decl_spec.default)
      elif (decl_spec.default != arg_field[1].default
            and (not isinstance(arg_field[1], Dict)
                 or decl_spec.default != MISSING_VALUE)):
        raise ValueError(
            f'{signature.id}: the default value ({arg_field[1].default!r}) '
            f'of symbolic argument {arg_name!r} does not equal to the default '
            f'value ({decl_spec.default!r}) specified at function signature '
            f'declaration.')
    return arg_field

  arg_fields = []

  # Add positional named arguments.
  arg_fields.extend([get_arg_field(arg) for arg in signature.args])

  # Add positional wildcard arguments.
  if signature.varargs:
    if varargs_spec is None:
      varargs_spec = (
          ConstStrKey(signature.varargs.name),
          List(Any()), 'Wildcard positional arguments.')
    elif not isinstance(varargs_spec[1], List):
      raise ValueError(
          f'{signature.id}: the value spec for positional wildcard argument '
          f'{varargs_spec[0]!r} must be a `pg.typing.List` instance. '
          f'Encountered: {varargs_spec[1]!r}.')
    varargs_spec[1].set_default([])
    arg_fields.append(varargs_spec)

  # Add keyword-only arguments.
  arg_fields.extend([get_arg_field(arg) for arg in signature.kwonlyargs])

  # Add extra arguments that are keyword wildcard.
  for arg_name in extra_arg_names:
    arg_fields.append(arg_dict[arg_name])

  # Add keyword wildcard arguments.
  if signature.varkw:
    if kwarg_spec is None:
      kwarg_spec = (StrKey(), Any(), 'Wildcard keyword arguments.')
    arg_fields.append(kwarg_spec)
  return [Field(*arg_decl) for arg_decl in arg_fields]


def create_schema(
    maybe_field_list: typing.List[typing.Union[Field, typing.Tuple]],  # pylint: disable=g-bare-generic
    name: typing.Optional[typing.Text] = None,
    base_schema_list: typing.Optional[typing.List[Schema]] = None,
    allow_nonconst_keys: bool = False,
    metadata: typing.Optional[typing.Dict[typing.Text, typing.Any]] = None
) -> Schema:
  """Creates ``Schema`` from a list of ``Field``s or equivalences.

  Args:
    maybe_field_list: A list of field equivalent values. A Field equivalent
      value is either a Field object or a tuple of 2 - 4 elements:
      `(<key>, <value>, [description], [metadata])`.
      `key` can be a KeySpec subclass object or string. `value` can be a
      ValueSpec subclass object or equivalent value. (see create_value_spec
      method). `description` is the description of this field. It can be
      optional when this field overrides the default value of a field defined
      in parent schema. `metadata` is an optional field which is a dict of
      user objects.
    name: An optional name for the schema.
    base_schema_list: A list of schema objects as bases.
    allow_nonconst_keys: Whether to allow non const keys in schema.
    metadata: Optional dict of user objects as schema-level metadata.

  Returns:
    Schema object.

  Raises:
    TypeError: If input type is incorrect.
  """
  if not isinstance(maybe_field_list, list):
    raise TypeError('Schema definition should be a list of schema.Field or '
                    'a list of tuples of (key, value, description, metadata).')

  metadata = metadata or {}
  if not isinstance(metadata, dict):
    raise TypeError(f'Metadata of schema should be a dict. '
                    f'Encountered: {metadata}.')

  fields = []
  for maybe_field in maybe_field_list:
    if isinstance(maybe_field, Field):
      fields.append(maybe_field)
      continue
    if not isinstance(maybe_field, tuple):
      raise TypeError(
          f'Field definition should be tuples with 2 to 4 elements. '
          f'Encountered: {maybe_field}.')

    if len(maybe_field) == 4:
      (maybe_key_spec, maybe_value_spec, description,
       field_metadata) = maybe_field
    elif len(maybe_field) == 3:
      maybe_key_spec, maybe_value_spec, description = maybe_field
      field_metadata = {}
    elif len(maybe_field) == 2:
      maybe_key_spec, maybe_value_spec = maybe_field
      description = None
      field_metadata = {}
    else:
      raise TypeError(
          f'Field definition should be tuples with 2 to 4 elements. '
          f'Encountered: {maybe_field}.')
    key = None
    if isinstance(maybe_key_spec, (str, KeySpec)):
      key = maybe_key_spec
    else:
      raise TypeError(
          f'The 1st element of field definition should be of '
          f'<class \'str\'> or KeySpec. Encountered: {maybe_key_spec}.')
    value = create_value_spec(maybe_value_spec)
    if (description is not None and
        not isinstance(description, str)):
      raise TypeError('Description (the 3rd element) of field definition '
                      'should be text type.')
    if not isinstance(field_metadata, dict):
      raise TypeError('Metadata (the 4th element) of field definition '
                      'should be a dict of objects.')
    fields.append(Field(key, value, description, field_metadata))
  return Schema(
      fields=fields,
      name=name,
      base_schema_list=base_schema_list,
      allow_nonconst_keys=allow_nonconst_keys,
      metadata=metadata)


def create_value_spec(value: typing.Any) -> ValueSpec:
  """Create value spec from a value that will be used as default value."""
  if isinstance(value, ValueSpec):
    return value
  return value_spec_from_type(type(value)).set_default(value)


def value_spec_from_type(value_type: typing.Type[typing.Any]) -> ValueSpec:
  """Create value spec from a python type."""
  if value_type is bool:
    return Bool()
  elif value_type is int:
    return Int()
  elif value_type is float:
    return Float()
  elif issubclass(value_type, str):
    return Str()
  else:
    raise TypeError(f'Only primitive types (bool, int, float, str) are '
                    f'supported to create ValueSpec from a default value. '
                    f'Encountered {value_type}.'
                    f'Consider using schema.Enum, schema.Dict, schema.List '
                    f'and schema.Object for complex types.')


def ensure_value_spec(
    value_spec: ValueSpec,
    src_spec: ValueSpec,
    root_path: typing.Optional[object_utils.KeyPath] = None
) -> typing.Optional[ValueSpec]:
  """Extract counter part from value spec that matches dest spec type.

  Args:
    value_spec: Value spec.
    src_spec: Destination value spec.
    root_path: An optional path for the value to include in error message.

  Returns:
    value_spec of src_spec_type

  Raises:
    TypeError: When value_spec cannot match src_spec_type.
  """
  if isinstance(value_spec, Union):
    value_spec = value_spec.get_candidate(src_spec)
  if isinstance(value_spec, Any):
    return None
  if not src_spec.is_compatible(value_spec):
    raise TypeError(
        object_utils.message_on_path(
            f'Source spec {src_spec} is not compatible with destination '
            f'spec {value_spec}.', root_path))
  return value_spec


class CallableWithOptionalKeywordArgs:
  """Helper class for invoking callable objects with optional keyword args.

  Examples::

    f = pg.typing.CallableWithOptionalKeywordArgs(lambda x: x ** 2, 'y')
    # Returns 4. Keyword 'y' is ignored.
    f(2, y=3)
  """

  def __init__(self,
               func: typing.Callable[..., typing.Any],
               optional_keywords: typing.List[typing.Text]):
    sig = get_signature(func)
    absent_keywords = None

    # Check for variable keyword arguments.
    if sig.has_varkw:
      absent_keywords = []
    else:
      all_keywords = set(sig.arg_names)
      absent_keywords = [k for k in optional_keywords if k not in all_keywords]
    self._absent_keywords = absent_keywords
    self._func = func

  def __call__(self, *args, **kwargs):
    """Delegate the call to function."""
    for k in self._absent_keywords:
      kwargs.pop(k, None)
    return self._func(*args, **kwargs)


#
# Class and methods for convertor system.
#


class _TypeConverterRegistry:
  """Type converter registry."""

  def __init__(self):
    """Constructor."""
    self._converter_list = []
    self._json_value_types = set(
        [int, float, bool, type(None), list, tuple, dict, str])

  def register(
      self,
      src: typing.Union[
          typing.Type[typing.Any],
          typing.Tuple[typing.Type[typing.Any], ...]],
      dest: typing.Union[
          typing.Type[typing.Any],
          typing.Tuple[typing.Type[typing.Any], ...]],
      convert_fn: typing.Callable[[typing.Any], typing.Any]) -> None:  # pyformat: disable pylint: disable=line-too-long
    """Register a converter from src type to dest type."""
    if (not isinstance(src, (tuple, type)) or
        not isinstance(dest, (tuple, type))):
      raise TypeError('Argument \'src\' and \'dest\' must be a type or '
                      'tuple of types.')
    if isinstance(dest, tuple):
      json_value_convertible = False
      for d in dest:
        for dest_type in self._json_value_types:
          if issubclass(d, dest_type):
            json_value_convertible = True
            break
        if json_value_convertible:
          break
    else:
      json_value_convertible = False
      for dest_type in self._json_value_types:
        if issubclass(dest, dest_type):
          json_value_convertible = True
          break
    self._converter_list.append((src, dest, convert_fn, json_value_convertible))

  def get_converter(
      self, src: typing.Type[typing.Any], dest: typing.Type[typing.Any]
  ) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
    """Get converter from source type to destination type."""
    # TODO(daiyip): Right now we don't see the need of a large number of
    # converters, thus its affordable to iterate the list.
    # We may consider more efficient way to do lookup in future.
    # NOTE(daiyip): We do reverse lookup since usually subclass converter
    # is register after base class.
    for src_type, dest_type, converter, _ in reversed(self._converter_list):
      if issubclass(src, src_type) and issubclass(dest, dest_type):
        return converter
    return None

  def get_json_value_converter(
      self, src: typing.Type[typing.Any]
  ) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
    """Get converter from source type to a JSON simple type."""
    for src_type, _, converter, json_value_convertible in reversed(
        self._converter_list):
      if issubclass(src, src_type) and json_value_convertible:
        return converter
    return None


_TYPE_CONVERTER_REGISTRY = _TypeConverterRegistry()


def get_converter(
    src: typing.Type[typing.Any], dest: typing.Type[typing.Any]
) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
  """Get converter from source type to destination type."""
  return _TYPE_CONVERTER_REGISTRY.get_converter(src, dest)


def get_first_applicable_converter(
    src_type: typing.Type[typing.Any],
    dest_type_or_types: typing.Union[typing.Type[typing.Any],
                                     typing.Tuple[typing.Type[typing.Any],
                                                  ...]]):
  """Get first applicable converter."""
  if isinstance(dest_type_or_types, tuple):
    dest_types = list(dest_type_or_types)
  else:
    dest_types = [dest_type_or_types]
  for dest_type in dest_types:
    converter = get_converter(src_type, dest_type)
    if converter is not None:
      return converter
  return None


def get_json_value_converter(
    src: typing.Type[typing.Any]
) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
  """Get converter from source type to a JSON simple type."""
  return _TYPE_CONVERTER_REGISTRY.get_json_value_converter(src)


def register_converter(
    src_type: typing.Union[
        typing.Type[typing.Any],
        typing.Tuple[typing.Type[typing.Any], ...]],
    dest_type: typing.Union[
        typing.Type[typing.Any],
        typing.Tuple[typing.Type[typing.Any], ...]],
    convert_fn: typing.Callable[[typing.Any], typing.Any]) -> None:
  """Register converter from source type to destination type.

  Examples::

    # Add converter from int to float.
    pg.typing.register_converter(int, float, float)

    assert pg.typing.Float().apply(1) is 1.0

    # Add converter from a dict to class A.
    def from_dict(d):
      return A(**d)

    assert isinstance(pg.typing.Object(A).apply({'x': 1, 'y': 2}), A)

  Args:
      src_type: Source value type.
      dest_type: Target value type.
      convert_fn: Function that performs the conversion, in signature
        (src_type) -> dest_type.
  """
  _TYPE_CONVERTER_REGISTRY.register(src_type, dest_type, convert_fn)


def _register_builtin_converters():
  """Register built-in converters."""
  # int <=> datetime.datetime.
  register_converter(int, datetime.datetime, datetime.datetime.utcfromtimestamp)
  register_converter(datetime.datetime, int,
                     lambda x: calendar.timegm(x.timetuple()))

  # string <=> KeyPath.
  register_converter(str, object_utils.KeyPath,
                     object_utils.KeyPath.parse)
  register_converter(object_utils.KeyPath, str, lambda x: x.path)


_register_builtin_converters()


if typing.TYPE_CHECKING:

  _GenericCallable = typing.TypeVar('_GenericCallable')

  class Decorator(object):
    """A type annotation for decorators that do not change signatures.

    This is a stand-in for using `Callable[[T], T]` to represent a decorator.

    Given a decorator function, which takes in a callable and returns a callable
    with the same signature, apply this class as a decorator to that function.
    This can also be used for decorator factories.

    Examples:

    Plain decorator (decorator matches Callable[[T], T]):

    >>> @pg.typing.Decorator
    ... def my_decorator(func):
    ...   def wrapper(...):
    ...     ...
    ...   return wrapper

    Decorator factory (factory matches Callable[..., Callable[[T], T]]):

    >>> def my_decorator_factory(foo: int):
    ...
    ...   @py.typing.Decorator
    ...   def my_decorator(func):
    ...     ...
    ...   return my_decorator

    This class only exists at build time, for typechecking. At runtime, the
    'Decorator' member of this module is a simple identity function.
    """

    def __init__(
        self,
        decorator: typing.Callable[[_GenericCallable], _GenericCallable]):  # pylint: disable=unused-argument
      ...  # pylint: disable=pointless-statement

    def __call__(self, func: _GenericCallable) -> _GenericCallable:
      ...  # pytype: disable=bad-return-type  # pylint: disable=pointless-statement

else:
  Decorator = lambda d: d
