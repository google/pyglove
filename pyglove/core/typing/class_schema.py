# Copyright 2023 The PyGlove Authors
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
"""Schema definition for symbolic classes and lists/dicts."""

import abc
import copy
import inspect
import sys
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type, Union

from pyglove.core import object_utils


class KeySpec(object_utils.Formattable, object_utils.JSONConvertible):
  """Interface for key specifications.

  A key specification determines what keys are acceptable for a symbolic
  field (see :class:`pyglove.Field`). Usually, symbolic attributes have an 1:1
  relationship with symbolic fields. But in some cases (e.g. a dict with dynamic
  keys), a field can be used to describe a group of symbolic attributes::

      # A dictionary that accepts key 'x' with float value
      # or keys started with 'foo' with int values.
      d = pg.Dict(value_spec=pg.Dict([
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

  @property
  @abc.abstractmethod
  def is_const(self) -> bool:
    """Returns whether current key is const."""

  @abc.abstractmethod
  def match(self, key: Any) -> bool:
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

  @classmethod
  def from_str(cls, key: str) -> 'KeySpec':
    """Get a concrete ValueSpec from annotation."""
    del key
    assert False, 'Overridden in `key_specs.py`.'


class ForwardRef(object_utils.Formattable):
  """Forward type reference."""

  def __init__(self, module: types.ModuleType, name: str):
    self._module = module
    self._name = name

  @property
  def module(self) -> types.ModuleType:
    """Returns the module where the name is being referenced."""
    return self._module

  @property
  def name(self) -> str:
    """Returns the name of the type reference."""
    return self._name

  @property
  def qualname(self) -> str:
    """Returns the qualified name of the reference."""
    return f'{self.module.__name__}.{self.name}'

  def as_annotation(self) -> Union[Type[Any], str]:
    """Returns the forward reference as an annotation."""
    return self.cls if self.resolved else self.name

  @property
  def resolved(self) -> bool:
    """Returns True if the symbol for the name is resolved.."""
    return hasattr(self.module, self.name)

  @property
  def cls(self) -> Type[Any]:
    """Returns the resolved reference class.."""
    reference = getattr(self.module, self.name, None)
    if reference is None:
      raise TypeError(
          f'{self.name!r} does not exist in module {self.module.__name__!r}'
      )
    elif not inspect.isclass(reference):
      raise TypeError(
          f'{self.name!r} from module {self.module.__name__!r} is not a class.'
      )
    return reference

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs
  ) -> str:
    """Format this object."""
    return object_utils.kvlist_str(
        [
            ('module', self.module.__name__, None),
            ('name', self.name, None),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs,
    )

  def __eq__(self, other: Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    elif isinstance(other, ForwardRef):
      return self.module is other.module and self.name == other.name
    elif inspect.isclass(other):
      return self.resolved and self.cls is other

  def __ne__(self, other: Any) -> bool:
    """Operator!=."""
    return not self.__eq__(other)

  def __hash__(self) -> int:
    return hash((self.module, self.name))

  def __deepcopy__(self, memo) -> 'ForwardRef':
    """Override deep copy to avoid copying module."""
    return ForwardRef(self.module, self.name)


class ValueSpec(object_utils.Formattable, object_utils.JSONConvertible):
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
  | User transform        | :attr:`.transform`                              |
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
    | dict                      | :class:`pyglove.typing.Dict`
    |
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
          [transform=<transform>])

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

  ``ValueSpec`` object can also be derived from annotations.
  For example, annotation below

     @pg.members([
        ('a', pg.typing.List(pg.typing.Str)),
        ('b', pg.typing.Dict().set_default('key': 'value')),
        ('c', pg.typing.List(pg.typing.Any()).noneable()),
        ('x', pg.typing.Int()),
        ('y', pg.typing.Str().noneable()),
        ('z', pg.typing.Union(pg.typing.Int(), pg.typing.Float()))
    ])

  can be writen as

    @pg.members([
        ('a', list[str]),
        ('b', {'key': 'value}),
        ('c', Optional[list]),
        ('x', int),
        ('y', Optional[str]),
        ('z', Union[int, float])
    ])
  """
  # pylint: disable=invalid-name

  # List-type value spec class.
  ListType: Type['ValueSpec']

  # Dict-type value spec class.
  DictType: Type['ValueSpec']

  # Object-type value spec class.
  ObjectType: Type['ValueSpec']

  # pylint: enable=invalid-name

  @property
  @abc.abstractmethod
  def value_type(self) -> Union[
      Type[Any],
      Tuple[Type[Any], ...]]:  # pyformat: disable
    """Returns acceptable (resolved) value type(s)."""

  @property
  @abc.abstractmethod
  def forward_refs(self) -> Set[ForwardRef]:
    """Returns forward referenes used by the value spec."""

  @abc.abstractmethod
  def noneable(self) -> 'ValueSpec':
    """Marks none-able and returns `self`."""

  @property
  @abc.abstractmethod
  def is_noneable(self) -> bool:
    """Returns True if current value spec accepts None."""

  @abc.abstractmethod
  def set_default(
      self,
      default: Any,
      use_default_apply: bool = True,
      root_path: Optional[object_utils.KeyPath] = None
  ) -> 'ValueSpec':
    """Sets the default value and returns `self`.

    Args:
      default: Default value.
      use_default_apply: If True, invoke `apply` to the value, otherwise use
        default value as is.
      root_path: (Optional) The path of the field.

    Returns:
      ValueSpec itself.

    Raises:
      ValueError: If default value cannot be applied when use_default_apply
        is set to True.
    """

  @property
  @abc.abstractmethod
  def default(self) -> Any:
    """Returns the default value.

    If no default is provided, MISSING_VALUE will be returned for non-dict
    types. For Dict type, a dict that may contains nested MISSING_VALUE
    will be returned.
    """

  @property
  def has_default(self) -> bool:
    """Returns True if the default value is provided."""
    return self.default != object_utils.MISSING_VALUE

  @abc.abstractmethod
  def freeze(
      self,
      permanent_value: Any = object_utils.MISSING_VALUE,
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
  def annotation(self) -> Any:
    """Returns PyType annotation. MISSING_VALUE if annotation is absent."""

  @property
  @abc.abstractmethod
  def transform(self) -> Optional[Callable[[Any], Any]]:
    """Returns a transform that will be applied on the input before apply."""

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
      value: Any,
      allow_partial: bool = False,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, 'Field', Any], Any]] = None,
      root_path: Optional[object_utils.KeyPath] = None,
      ) -> Any:
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

  @property
  def type_resolved(self) -> bool:
    """Returns True if all forward references are resolved."""
    return not any(not ref.resolved for ref in self.forward_refs)

  def __ne__(self, other: Any) -> bool:
    """Operator !=."""
    return not self.__eq__(other)

  @classmethod
  def from_annotation(
      cls,
      annotation: Any,
      auto_typing: bool = False,
      accept_value_as_annotation: bool = False,
      parent_module: Optional[types.ModuleType] = None
      ) -> 'ValueSpec':
    """Gets a concrete ValueSpec from annotation."""
    del annotation, auto_typing, accept_value_as_annotation, parent_module
    assert False, 'Overridden in `annotation_conversion.py`.'


class Field(object_utils.Formattable, object_utils.JSONConvertible):
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

  __serialization_key__ = 'pyglove.typing.Field'

  def __init__(
      self,
      key_spec: Union[KeySpec, str],
      value_spec: ValueSpec,
      description: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None):
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
      key_spec = KeySpec.from_str(key_spec)
    assert isinstance(key_spec, KeySpec), key_spec
    self._key = key_spec
    self._value = value_spec
    self._description = description

    if metadata and not isinstance(metadata, dict):
      raise ValueError('metadata must be a dict.')
    self._metadata = metadata or {}

  @classmethod
  def from_annotation(
      cls,
      key: Union[str, KeySpec],
      annotation: Any,
      description: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None,
      auto_typing=True) -> 'Field':
    """Gets a Field from annotation."""
    del key, annotation, description, metadata, auto_typing
    assert False, 'Overridden in `annotation_conversion.py`.'

  @property
  def description(self) -> Optional[str]:
    """Description of this field."""
    return self._description

  def set_description(self, description: str) -> None:
    """Sets the description for this field."""
    self._description = description

  @property
  def key(self) -> KeySpec:
    """Key specification of this field."""
    return self._key

  @property
  def value(self) -> ValueSpec:
    """Value specification of this field."""
    return self._value

  @property
  def annotation(self) -> Any:
    """Type annotation for this field."""
    return self._value.annotation

  @property
  def metadata(self) -> Dict[str, Any]:
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
      value: Any,
      allow_partial: bool = False,
      transform_fn: Optional[Callable[
          [object_utils.KeyPath, 'Field', Any], Any]] = None,
      root_path: Optional[object_utils.KeyPath] = None) -> Any:
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
    value = self._value.apply(
        value,
        allow_partial=allow_partial,
        child_transform=transform_fn,
        root_path=root_path
    )

    if transform_fn:
      value = transform_fn(root_path, self, value)
    return value

  @property
  def default_value(self) -> Any:
    """Returns the default value."""
    return self._value.default

  @property
  def frozen(self) -> bool:
    """Returns True if current field's value is frozen."""
    return self._value.frozen

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this field into a string."""
    return object_utils.kvlist_str(
        [
            ('key', self._key, None),
            ('value', self._value, None),
            ('description', self._description, None),
            ('metadata', self._metadata, {}),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: Any) -> Dict[str, Any]:
    return self.to_json_dict(
        fields=dict(
            key_spec=(self._key, None),
            value_spec=(self._value, None),
            description=(self._description, None),
            metadata=(self._metadata, {}),
        ),
        exclude_default=True,
        **kwargs,
    )

  def __eq__(self, other: Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    return (isinstance(other, self.__class__) and self.key == other.key and
            self.value == other.value and
            self.description == other.description and
            self.metadata == other.metadata)

  def __ne__(self, other: Any) -> bool:
    """Operator!=."""
    return not self.__eq__(other)


class Schema(object_utils.Formattable, object_utils.JSONConvertible):
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

    print(A.__schema__)

    @pg.symbolize([
        ('a', pg.typing.Int()),
        ('b', pg.typing.Float())
    ])
    def foo(a, b):
      return a + b

    print(foo.__schema__)

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

  __serialization_key__ = 'pyglove.typing.Schema'

  def __init__(
      self,
      fields: List[Field],
      name: Optional[str] = None,
      base_schema_list: Optional[List['Schema']] = None,
      description: Optional[str] = None,
      *,
      allow_nonconst_keys: bool = False,
      metadata: Optional[Dict[str, Any]] = None):
    """Constructor.

    Args:
      fields: A list of Field as the definition of the schema. The order of the
        fields will be preserved.
      name: Optional name of this schema. Useful for debugging.
      base_schema_list: List of schema used as base. When present, fields
        from these schema will be copied to this schema. Fields from the
        latter schema will override those from the former ones.
      description: Optional str as the description for the schema.
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
      raise TypeError(
          f"Argument 'fields' must be a list. Encountered: {fields}."
      )

    self._name = name
    self._allow_nonconst_keys = allow_nonconst_keys
    self._fields = {f.key: f for f in fields}
    self._description = description
    self._metadata = metadata or {}

    self._dynamic_field = None
    for f in fields:
      if not f.key.is_const:
        self._dynamic_field = f
        break

    if base_schema_list:
      base = Schema.merge(base_schema_list)
      self.extend(base)

    if not allow_nonconst_keys and self._dynamic_field is not None:
      raise ValueError(
          f'NonConstKey is not allowed in schema. '
          f'Encountered \'{self._dynamic_field.key}\'.')

  @classmethod
  def merge(
      cls,
      schema_list: Sequence['Schema'],
      name: Optional[str] = None,
      description: Optional[str] = None
    ) -> 'Schema':
    """Merge multiple schemas into one.

    For fields shared by multiple schemas, the first appeared onces will be
    used in the merged schema.

    Args:
      schema_list: A list of schemas to merge.
      name: (Optional) name of the merged schema.
      description: (Optinoal) description of the schema.

    Returns:
      The merged schema.
    """
    field_names = set()
    fields = []
    kw_field = None
    for schema in schema_list:
      for key, field in schema.fields.items():
        if key.is_const and key not in field_names:
          fields.append(field)
          field_names.add(key)
        elif not key.is_const and kw_field is None:
          kw_field = field

    if kw_field is not None:
      fields.append(kw_field)
    return Schema(
        fields, name=name, description=description, allow_nonconst_keys=True
    )

  def extend(self, base: 'Schema') -> 'Schema':
    """Extend current schema based on a base schema."""

    def _merge_field(
        path,
        parent_field: Field,
        child_field: Field) -> Field:
      """Merge function on field with the same key."""
      if parent_field != object_utils.MISSING_VALUE:
        if object_utils.MISSING_VALUE == child_field:
          if (not self._allow_nonconst_keys and not parent_field.key.is_const):
            hints = object_utils.kvlist_str([
                ('base', base.name, None),
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
                ('base', base.name, None),
                ('path', path, None)
            ])
            raise e.__class__(f'{e} ({hints})').with_traceback(
                sys.exc_info()[2])
      return child_field

    self._fields = object_utils.merge([base.fields, self.fields], _merge_field)
    self._metadata = object_utils.merge([base.metadata, self.metadata])

    # Inherit dynamic field from base if it's not present in the child.
    if self._dynamic_field is None:
      for k, f in self._fields.items():
        if not k.is_const:
          self._dynamic_field = f
          break
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

  def get_field(self, key: str) -> Optional[Field]:
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

  @property
  def description(self) -> Optional[str]:
    """Returns the description for the schema."""
    return self._description

  def set_description(self, description: str) -> None:
    """Sets the description for the schema."""
    self._description = description

  @property
  def dynamic_field(self) -> Optional[Field]:
    """Returns the field that matches multiple keys if any."""
    return self._dynamic_field

  def resolve(
      self, keys: Iterable[str]
  ) -> Tuple[Dict[KeySpec, List[str]], List[str]]:
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
    nonconst_key_specs = [k for k in self._fields.keys() if not k.is_const]
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
      if not key_spec.is_const:
        keys = nonconst_keys.get(key_spec, [])
      elif key_spec in input_keyset:
        keys.append(str(key_spec))
      keys_by_key_spec[key_spec] = keys

    return (keys_by_key_spec, unmatched_keys)

  def apply(
      self,
      dict_obj: Dict[str, Any],
      allow_partial: bool = False,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, Field, Any], Any]] = None,
      root_path: Optional[object_utils.KeyPath] = None,
  ) -> Dict[str, Any]:  # pyformat: disable
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
    """  # pyformat: enable
    matched_keys, unmatched_keys = self.resolve(dict_obj.keys())
    if unmatched_keys:
      raise KeyError(
          f'Keys {unmatched_keys} are not allowed in Schema. '
          f'(parent=\'{root_path}\')'
      )

    for key_spec, keys in matched_keys.items():
      field = self._fields[key_spec]
      # For missing const keys, we add to keys collection to add missing value.
      if key_spec.is_const and key_spec not in keys:
        keys.append(str(key_spec))
      for key in keys:
        if dict_obj:
          value = dict_obj.get(key, object_utils.MISSING_VALUE)
        else:
          value = object_utils.MISSING_VALUE
        # NOTE(daiyip): field.default_value may be MISSING_VALUE too
        # or partial.
        if object_utils.MISSING_VALUE == value:
          value = copy.deepcopy(field.default_value)
        new_value = field.apply(
            value,
            allow_partial=allow_partial,
            transform_fn=child_transform,
            root_path=object_utils.KeyPath(key, root_path)
        )

        # NOTE(daiyip): `pg.Dict.__getitem__`` has special logics in handling
        # `pg.Contextual`` values. Therefore, we user `dict.__getitem__()`` to
        # avoid triggering side effect.
        if (key not in dict_obj
            or dict.__getitem__(dict_obj, key) is not new_value):
          # NOTE(daiyip): minimize call to __setitem__ when possible.
          # Custom like symbolic dict may trigger additional logic
          # when __setitem__ is called.
          dict_obj[key] = new_value
    return dict_obj

  def validate(self,
               dict_obj: Dict[str, Any],
               allow_partial: bool = False,
               root_path: Optional[object_utils.KeyPath] = None) -> None:
    """Validates whether dict object is conformed with the schema."""
    self.apply(
        copy.deepcopy(dict_obj),
        allow_partial=allow_partial,
        root_path=root_path)

  @property
  def name(self) -> Optional[str]:
    """Name of this schema."""
    return self._name

  def set_name(self, name: str) -> None:
    """Sets the name of this schema."""
    self._name = name

  @property
  def allow_nonconst_keys(self) -> bool:
    """Returns whether to allow non-const keys."""
    return self._allow_nonconst_keys

  @property
  def fields(self) -> Dict[KeySpec, Field]:
    """Returns fields of this schema."""
    return self._fields

  def __getitem__(self, key: Union[str, KeySpec]) -> Field:
    """Returns field by key."""
    return self._fields[key]

  def __contains__(self, key: Union[str, KeySpec]) -> bool:
    """Returns if a key or key spec exists in the schema."""
    return key in self._fields

  def get(self,
          key: Union[str, KeySpec],
          default: Optional[Field] = None
          ) -> Optional[Field]:
    """Returns field by key with default value if not found."""
    return self._fields.get(key, default)

  def keys(self) -> Iterable[KeySpec]:
    """Return an iteratable of KeySpecs in declaration order."""
    return self._fields.keys()

  def values(self) -> Iterable[Field]:
    """Returns an iterable of Field in declaration order."""
    return self._fields.values()

  def items(self) -> Iterable[Tuple[KeySpec, Field]]:
    """Returns an iterable of (KeySpec, Field) tuple in declaration order."""
    return self._fields.items()

  @property
  def metadata(self) -> Dict[str, Any]:
    """Returns metadata of this schema."""
    return self._metadata

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      cls_name: Optional[str] = None,
      bracket_type: object_utils.BracketType = object_utils.BracketType.ROUND,
      fields_only: bool = False,
      **kwargs,
  ) -> str:
    """Format current Schema into nicely printed string."""
    return object_utils.kvlist_str(
        [
            ('name', self.name, None),
            ('description', self.description, None),
            ('fields', list(self.fields.values()), []),
            ('allow_nonconst_keys', self.allow_nonconst_keys, True),
            ('metadata', self.metadata, {}),
        ],
        label=cls_name or self.__class__.__name__,
        bracket_type=bracket_type,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs,
    )

  def to_json(self, **kwargs) -> Dict[str, Any]:
    return self.to_json_dict(
        fields=dict(
            fields=(list(self._fields.values()), []),
            name=(self._name, None),
            description=(self._description, None),
            allow_nonconst_keys=(self._allow_nonconst_keys, False),
            metadata=(self._metadata, {}),
        ),
        exclude_default=True,
        **kwargs,
    )

  def __eq__(self, other: Any) -> bool:
    if self is other:
      return True
    return isinstance(other, Schema) and self._fields == other._fields

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)


FieldDef = Union[
    # Key, Value spec/annotation.
    Tuple[Union[str, KeySpec], Any],

    # Key, Value spec/annotation, field docstr.
    Tuple[Union[str, KeySpec], Any, str],

    # Key, Value spec/annotation, field docstr, field metadata.
    Tuple[Union[str, KeySpec], Any, str, Dict[str, Any]],
]

FieldKeyDef = Union[str, KeySpec]

FieldValueDef = Union[
    # Value spec/annotation.
    Any,

    # Value spec/annotation, field docstr.
    Tuple[Any, str],

    # Value spec/annotation, field docstr, field metadata.
    Tuple[Any, str, Dict[str, Any]]
]


def create_field(
    field_or_def: Union[Field, FieldDef],
    auto_typing: bool = True,
    accept_value_as_annotation: bool = True,
    parent_module: Optional[types.ModuleType] = None
) -> Field:
  """Creates ``Field`` from its equivalence.

  Args:
    field_or_def: a ``Field`` object or its equivalence, which is a tuple of
      2 - 4 elements:
      `(<key>, <value>, [description], [metadata])`.
      `key` can be a KeySpec subclass object or string. `value` can be a
      ValueSpec subclass object or equivalent value. (see
      ``ValueSpec.from_value`` method). `description` is the description of this
      field. It can be optional when this field overrides the default value of a
      field defined in parent schema. `metadata` is an optional field which is a
      dict of user objects.
    auto_typing: If True, infer value spec from Python annotations. Otherwise,
      ``pg.typing.Any()`` will be used.
    accept_value_as_annotation: If True, allow default values to be used as
      annotations when creating the value spec.
    parent_module: (Optional) parent module for defining this field, which will
      be used for forward reference lookup.

  Returns:
    A ``Field`` object.
  """
  if isinstance(field_or_def, Field):
    return field_or_def

  if not isinstance(field_or_def, tuple):
    raise TypeError(
        f'Field definition should be tuples with 2 to 4 elements. '
        f'Encountered: {field_or_def}.')

  field_def = list(field_or_def)
  if len(field_def) == 4:
    maybe_key_spec, maybe_value_spec, description, field_metadata = field_def
  elif len(field_def) == 3:
    maybe_key_spec, maybe_value_spec, description = field_def
    field_metadata = {}
  elif len(field_def) == 2:
    maybe_key_spec, maybe_value_spec = field_def
    description = None
    field_metadata = {}
  else:
    raise TypeError(
        f'Field definition should be tuples with 2 to 4 elements. '
        f'Encountered: {field_or_def}.')

  if isinstance(maybe_key_spec, (str, KeySpec)):
    key = maybe_key_spec
  else:
    raise TypeError(
        f'The 1st element of field definition should be of '
        f'<class \'str\'> or KeySpec. Encountered: {maybe_key_spec}.')

  value = ValueSpec.from_annotation(
      maybe_value_spec,
      auto_typing=auto_typing,
      accept_value_as_annotation=accept_value_as_annotation,
      parent_module=parent_module,
  )

  if (description is not None and
      not isinstance(description, str)):
    raise TypeError(f'Description (the 3rd element) of field definition '
                    f'should be text type. Encountered: {description}')
  if not isinstance(field_metadata, dict):
    raise TypeError(f'Metadata (the 4th element) of field definition '
                    f'should be a dict of objects. '
                    f'Encountered: {field_metadata}')
  return Field(key, value, description, field_metadata)


def create_schema(
    fields: Union[
        Dict[str, FieldValueDef],
        List[Union[Field, FieldDef]]   # pylint: disable=g-bare-generic
    ],
    name: Optional[str] = None,
    base_schema_list: Optional[List[Schema]] = None,
    allow_nonconst_keys: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    parent_module: Optional[types.ModuleType] = None
) -> Schema:
  """Creates ``Schema`` from a list of ``Field``s or equivalences.

  Examples:

    The following code examples are equivalent::

      schema = pg.typing.create_schema({
          'a': int,
          'b': (int, 'field b'),
          'c': (pg.typing.Str(regex='.*'), 'field c', dict(meta1=1))
      })

      schema = pg.typing.create_schema([
          ('a', int),
          ('b', int, 'field b'),
          ('c', pg.typing.Str(regex='.*'), 'field c', dict(meta1=1)),
      ])

      schema = pg.typing.create_schema([
          pg.typing.Field('a', pg.typing.Int()),
          pg.typing.Field('b', pg.typing.Int(), 'field b'),
          pg.typing.Field(
              'c', pg.typing.Str(regex='.*'), 'field c', dict(meta1=1)),
      ])

  Args:
    fields: A dict, a list of field or equivalent values. A Field equivalent
      value is either a Field object or a tuple of 2 - 4 elements: `(<key>,
      <value>, [description], [metadata])`. `key` can be a KeySpec subclass
      object or string. `value` can be a ValueSpec subclass object or equivalent
      value. (see ``ValueSpec.from_value`` method). `description` is the
      description of this field. It can be optional when this field overrides
      the default value of a field defined in parent schema. `metadata` is an
      optional field which is a dict of user objects.
    name: An optional name for the schema.
    base_schema_list: A list of schema objects as bases.
    allow_nonconst_keys: Whether to allow non const keys in schema.
    metadata: Optional dict of user objects as schema-level metadata.
    description: Optional description of the schema.
    parent_module: (Optional) parent module for defining this schema, which will
      be used for forward reference lookup.

  Returns:
    Schema object.

  Raises:
    TypeError: If input type is incorrect.
  """
  fields = _normalize_field_defs(fields)
  metadata = metadata or {}
  if not isinstance(metadata, dict):
    raise TypeError(f'Metadata of schema should be a dict. '
                    f'Encountered: {metadata}.')

  return Schema(
      fields=[create_field(field_or_def, parent_module=parent_module)
              for field_or_def in fields],
      name=name,
      base_schema_list=base_schema_list,
      allow_nonconst_keys=allow_nonconst_keys,
      metadata=metadata,
      description=description,
  )


def _normalize_field_defs(
    fields: Union[
        Dict[str, FieldValueDef],
        List[Union[Field, FieldDef]]   # pylint: disable=g-bare-generic
    ]
) -> List[Union[Field, FieldDef]]:
  """Normalizes field definitions."""
  if isinstance(fields, dict):
    normalized_fields = []
    for k, v in fields.items():
      if not isinstance(v, (tuple, list)):
        v = (v,)
      normalized_fields.append(tuple([k] + list(v)))
    return normalized_fields    # pytype: disable=bad-return-type
  elif not isinstance(fields, list):
    raise TypeError(
        'Schema definition should be a dict of field names to their '
        'definitions, a list of `pg.typing.Field` objects or a list of tuples '
        'in format (key, value, description, metadata). '
        f'Encountered: {fields}.'
    )
  return fields
