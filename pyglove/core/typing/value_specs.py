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
"""Concrete value specifications for field definition."""

import abc
import copy
import functools
import inspect
import numbers
import re
import sys
import typing
import __main__
from pyglove.core import object_utils
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import inspect as pg_inspect
from pyglove.core.typing import key_specs
from pyglove.core.typing import type_conversion
from pyglove.core.typing import typed_missing
from pyglove.core.typing.class_schema import Field
from pyglove.core.typing.class_schema import Schema
from pyglove.core.typing.class_schema import ValueSpec
from pyglove.core.typing.custom_typing import CustomTyping


MISSING_VALUE = object_utils.MISSING_VALUE

# Type alias for ValueSpec object or Python annotation that could be converted
# to ValueSpec via `pg.typing.ValueSpec.from_annotation()`. This type alias is
# just for better readability.
ValueSpecOrAnnotation = typing.Any


class GenericMeta(abc.ABCMeta):
  """Metaclass for generic value spec."""

  def __getitem__(cls, type_args: typing.Any) -> typing.Any:
    if not isinstance(type_args, tuple):
      type_args = (type_args,)
    generic = cls.with_type_args(type_args)  # pytype: disable=attribute-error
    if typing.TYPE_CHECKING and isinstance(generic, ValueSpec):
      return generic.annotation
    return generic


class Generic(metaclass=GenericMeta):
  """Generic value spec."""

  @classmethod
  def with_type_args(
      cls, type_args: typing.Tuple[typing.Any, ...]
  ) -> typing.Any:
    raise NotImplementedError()


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
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Any]] = None,
      is_noneable: bool = False,
      frozen: bool = False):  # pyformat: disable
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
      transform: (Optional) user-defined function to be called on the input
        of `apply`. It could be used as a type converter or a custom
        validator which may raise errors.
      is_noneable: (Optional) If True, None is acceptable for this spec.
      frozen: If True, values other than the default value is not accceptable.
    """
    super().__init__()
    self._value_type = value_type
    self._is_noneable = is_noneable
    self._frozen = False
    self._default = MISSING_VALUE
    self._transform = transform

    self.set_default(default)
    # Reset frozen after setting the default value.
    self._frozen = frozen

  @functools.cached_property
  def skip_user_transform(self) -> 'ValueSpec':
    """Returns a value spec of this without transform."""
    if self._transform is None:
      return self
    spec_without_transform = copy.copy(self)
    spec_without_transform._transform = None  # pylint: disable=protected-access
    return spec_without_transform

  @property
  def is_noneable(self) -> bool:
    """Returns True if current value spec accepts None."""
    return self._is_noneable

  def noneable(self) -> 'ValueSpecBase':
    """Marks None is acceptable and returns `self`."""
    self._is_noneable = True
    if MISSING_VALUE == self._default:  # pytype: disable=attribute-error
      self._default = None
    return self

  @property
  def value_type(self) -> typing.Union[
      None,
      typing.Type[typing.Any],
      typing.Tuple[typing.Type[typing.Any], ...]]:  # pyformat: disable
    """Returns acceptable value type(s) for current value spec."""
    return self._value_type

  @property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns no forward references by default, subclasses can override."""
    return set()

  @property
  def default(self) -> typing.Any:
    """Returns the default value."""
    return self._default

  def set_default(
      self,
      default: typing.Any,
      use_default_apply: bool = True,
      root_path: typing.Optional[object_utils.KeyPath] = None
  ) -> ValueSpec:
    """Set default value and returns `self`."""
    # NOTE(daiyip): Default can be schema.MissingValue types, all are
    # normalized to MISSING_VALUE for consistency.
    if MISSING_VALUE == default:
      default = MISSING_VALUE
    if MISSING_VALUE != default and use_default_apply:
      default = self.apply(default, allow_partial=True, root_path=root_path)
    self._default = default
    return self

  def freeze(self,
             permanent_value: typing.Any = MISSING_VALUE,
             apply_before_use: bool = True) -> ValueSpec:
    """Sets the permanent value as default value and freeze the value spec."""
    if MISSING_VALUE != permanent_value:
      self.set_default(permanent_value, use_default_apply=apply_before_use)
    elif MISSING_VALUE == self._default:
      raise ValueError(f'Cannot freeze {self!r} without a default value.')
    self._frozen = True
    return self

  @property
  def frozen(self) -> bool:
    """Returns True if current value spec is frozen."""
    return self._frozen

  @property
  def transform(
      self) -> typing.Optional[typing.Callable[[typing.Any], typing.Any]]:
    """Returns user validator for custom validation logic."""
    return self._transform

  def extend(self, base: ValueSpec) -> ValueSpec:
    """Extend current value spec on top of a base spec."""
    if base.frozen:
      raise TypeError(f'Cannot extend a frozen value spec: {base!r}')

    if self._transform is None:
      self._transform = base.transform

    if isinstance(base, Any):
      return self

    if not isinstance(self, Union) and isinstance(base, Union):
      base_counterpart = base.get_candidate(self)
      if base_counterpart is None:
        raise TypeError(f'{self!r} cannot extend {self!r}: '
                        f'no compatible type found in Union.')
      base = base_counterpart

    if not isinstance(self, (base.__class__, Enum)):
      raise TypeError(f'{self!r} cannot extend {base!r}: incompatible type.')
    if not base.is_noneable and self._is_noneable:
      raise TypeError(f'{self!r} cannot extend {base!r}: '
                      f'None is not allowed in base spec.')
    self._extend(base)  # pytype: disable=wrong-arg-types  # always-use-return-annotations
    return self

  def _extend(self, base: ValueSpec) -> None:
    """Customized extension that each subclass can override."""

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
      if MISSING_VALUE != value and self.default != value:
        raise ValueError(
            f'Frozen field is not assignable. (Path='
            f'\'{root_path}\', ValueSpec={self!r}, AssignedValue={value!r})')
      return self.default

    if MISSING_VALUE == value:
      if not allow_partial:
        raise ValueError(
            f'Required value is not specified. '
            f'(Path=\'{root_path}\', ValueSpec={self!r})')
      return typed_missing.MissingValue(self)

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

    if MISSING_VALUE != value and self._transform is not None:
      try:
        value = self._transform(value)
      except Exception as e:  # pylint: disable=broad-except
        raise e.__class__(
            object_utils.message_on_path(str(e), root_path)
            ).with_traceback(sys.exc_info()[2])

      return self.skip_user_transform.apply(
          value,
          allow_partial=allow_partial,
          child_transform=child_transform,
          root_path=root_path)

    if (
        self.type_resolved
        and self.value_type is not None
        and not pg_inspect.is_instance(value, self.value_type)
    ):
      converter = type_conversion.get_converter(type(value), self.value_type)
      if converter is None:
        raise TypeError(
            object_utils.message_on_path(
                f'Expect {self.value_type} '
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
    del other
    return True

  @property
  def annotation(self) -> typing.Any:
    """Returns PyType annotation."""
    annotation = self._annotate()
    if MISSING_VALUE != annotation and self.is_noneable:
      return typing.Optional[annotation]
    return annotation

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return self.value_type

  def __eq__(self, other: typing.Any) -> bool:
    """Operator==."""
    if self is other:
      return True
    if (
        not isinstance(other, self.__class__)
        # Default value might be callable.
        or not pg_inspect.callable_eq(self.default, other.default)
        or self.is_noneable != other.is_noneable
        or self.frozen != other.frozen
        or not pg_inspect.callable_eq(self.transform, other.transform)
    ):
      return False
    return self._eq(other)

  def _eq(self, other: 'ValueSpec') -> bool:
    """Subclasses can override."""
    del other
    return True

  def __or__(self, other: typing.Any) -> bool:
    return Union[self, other]

  def __ror__(self, other: typing.Any) -> bool:
    return Union[other, self]

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
            ('default', self._default, MISSING_VALUE),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False)
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )


class PrimitiveType(ValueSpecBase):
  """Base class of value specification for primitive types."""

  def __init__(
      self,
      value_type: typing.Union[
          typing.Type[typing.Any], typing.Tuple[typing.Type[typing.Any], ...]
      ],
      default: typing.Any = MISSING_VALUE,
      is_noneable: bool = False,
      frozen: bool = False,
  ):
    """Constructor.

    Args:
      value_type: Acceptable value type(s).
      default: Default value.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    super().__init__(
        value_type, default, is_noneable=is_noneable, frozen=frozen
    )


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

  __serialization_key__ = 'pyglove.typing.Bool'

  def __init__(
      self,
      default: typing.Optional[bool] = MISSING_VALUE,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: Default value for the value spec.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    super().__init__(bool, default, is_noneable=is_noneable, frozen=frozen)

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            default=(self.default, MISSING_VALUE),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )


class Str(Generic, PrimitiveType):
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

  __serialization_key__ = 'pyglove.typing.Str'

  def __init__(
      self,
      default: typing.Optional[str] = MISSING_VALUE,
      regex: typing.Optional[str] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: Default value for this value spec.
      regex: Optional regular expression for acceptable value.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    self._regex = re.compile(regex) if regex else None
    super().__init__(str, default, is_noneable=is_noneable, frozen=frozen)

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            default=(self.default, MISSING_VALUE),
            regex=(self._regex.pattern if self._regex else None, None),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  def _validate(self, path: object_utils.KeyPath, value: str) -> None:
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
    return str

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
    regex_pattern = self._regex.pattern if self._regex else None
    return object_utils.kvlist_str(
        [
            ('default', self._default, MISSING_VALUE),
            ('regex', regex_pattern, None),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def _eq(self, other: 'Str') -> bool:
    return self.regex == other.regex

  @classmethod
  def with_type_args(cls, type_args: typing.Tuple[typing.Any, ...]) -> 'Str':
    if len(type_args) != 1:
      raise TypeError(
          '`pg.typing.Str` requires 1 argument as the regular expression for '
          f'matching the str value. Encountered: {type_args!r}.'
      )
    return cls(regex=type_args[0])


class Number(Generic, PrimitiveType):
  """Base class for value spec of numeric types."""

  def __init__(
      self,
      value_type,  # typing.Type[numbers.Number]
      default: typing.Optional[numbers.Number] = MISSING_VALUE,
      min_value: typing.Optional[numbers.Number] = None,
      max_value: typing.Optional[numbers.Number] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      value_type: Type of number.
      default: Default value for this spec.
      min_value: (Optional) minimum value of acceptable values.
      max_value: (Optional) maximum value of acceptable values.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    if (min_value is not None and max_value is not None and
        min_value > max_value):
      raise ValueError(
          f'"max_value" must be equal or greater than "min_value". '
          f'Encountered: min_value={min_value}, max_value={max_value}.')
    self._min_value = min_value
    self._max_value = max_value
    super().__init__(
        value_type, default, is_noneable=is_noneable, frozen=frozen
    )

  @property
  def min_value(self) -> typing.Optional[numbers.Number]:
    """Returns minimum value of acceptable values."""
    return self._min_value

  @property
  def max_value(self) -> typing.Optional[numbers.Number]:
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
        raise TypeError(
            f'{self!r} cannot extend {base!r}: min_value is smaller.'
        )

    max_value = self._max_value
    if base.max_value is not None:
      if max_value is None:
        max_value = base.max_value
      elif max_value > base.max_value:
        raise TypeError(
            f'{self!r} cannot extend {base!r}: max_value is larger.'
        )

    if (min_value is not None and max_value is not None and
        min_value > max_value):
      raise TypeError(
          f'{self!r} cannot extend {base!r}: '
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

  def _eq(self, other: 'Number') -> bool:
    return (self.min_value == other.min_value
            and self.max_value == other.max_value)

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
            ('default', self._default, MISSING_VALUE),
            ('min', self._min_value, None),
            ('max', self._max_value, None),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False)
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            default=(self.default, MISSING_VALUE),
            min_value=(self._min_value, None),
            max_value=(self._max_value, None),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Tuple[typing.Any, ...]) -> 'Number':
    if len(type_args) != 2:
      raise TypeError(
          f'`pg.typing.{cls.__name__}` requires 2 arguments as the '
          'min value and the max value for the field. '
          f'Encountered: {type_args!r}'
      )
    # pylint: disable=no-value-for-parameter
    return cls(min_value=type_args[0], max_value=type_args[1])  # pytype: disable=missing-parameter
    # pylint: enable=no-value-for-parameter


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

  __serialization_key__ = 'pyglove.typing.Int'

  def __init__(
      self,
      default: typing.Optional[int] = MISSING_VALUE,
      min_value: typing.Optional[int] = None,
      max_value: typing.Optional[int] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: (Optional) default value for this spec.
      min_value: (Optional) minimum value of acceptable values.
      max_value: (Optional) maximum value of acceptable values.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    super().__init__(int, default, min_value, max_value, is_noneable, frozen)


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

  __serialization_key__ = 'pyglove.typing.Float'

  def __init__(
      self,
      default: typing.Optional[float] = MISSING_VALUE,
      min_value: typing.Optional[float] = None,
      max_value: typing.Optional[float] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: (Optional) default value for this spec.
      min_value: (Optional) minimum value of acceptable values.
      max_value: (Optional) maximum value of acceptable values.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    super().__init__(float, default, min_value, max_value, is_noneable, frozen)


class Enum(Generic, PrimitiveType):
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

  __serialization_key__ = 'pyglove.typing.Enum'

  def __init__(
      self,
      default: typing.Any,
      values: typing.List[typing.Any],
      frozen: bool = False,
  ):
    """Constructor.

    Args:
      default: default value for this spec.
      values: all acceptable values.
      frozen: If True, values other than the default value is not accceptable.
    """
    if not isinstance(values, list) or not values:
      raise ValueError(
          f'Values for Enum should be a non-empty list. Found {values!r}.')
    if MISSING_VALUE != default and default not in values:
      raise ValueError(
          f'Enum default value {default!r} is not in candidate list '
          f'{values!r}.')

    value_type = None
    for v in values:
      if v is None:
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

    is_noneable = any([v is None for v in values])

    # NOTE(daiyip): When enum values are strings, we relax the `value_type`
    # to accept text types (unicode) as well. This allows enum value be read
    # from unicode JSON file.
    if value_type is not None and issubclass(value_type, str):
      value_type = str
    self._values = values
    super().__init__(
        value_type, default, is_noneable=is_noneable, frozen=frozen
    )

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
    for v in self._values:
      try:
        _ = base.apply(v)
      except (TypeError, ValueError)as e:
        raise TypeError(
            f'{self!r} cannot extend {base!r}: '
            f'{repr(v)} is not an acceptable value.'
        ) from e

  def _is_compatible(self, other: 'Enum') -> bool:
    """Enum specific compatibility check."""
    for v in other.values:
      if v not in self.values:
        return False
    return True

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    if self.value_type == str:
      return str
    if self.value_type is None:
      return typing.Any
    return self.value_type

  def _eq(self, other: 'Enum') -> bool:
    return self.values == other.values

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
            ('default', self._default, MISSING_VALUE),
            ('values', self._values, None),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            default=(self.default, MISSING_VALUE),
            values=(self._values, None),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Tuple[typing.Any, ...]) -> 'Enum':
    if len(type_args) < 2:
      raise TypeError(
          '`pg.typing.Enum` requires at least 2 arguments as the candidate '
          f'values for the enum Encountered: {type_args!r}')
    return cls(default=MISSING_VALUE, values=list(type_args))


class List(Generic, ValueSpecBase):
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

  __serialization_key__ = 'pyglove.typing.List'

  def __init__(
      self,
      element_value: ValueSpecOrAnnotation,
      default: typing.Optional[typing.List[typing.Any]] = MISSING_VALUE,
      min_size: typing.Optional[int] = None,
      max_size: typing.Optional[int] = None,
      size: typing.Optional[int] = None,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.List[typing.Any]]
      ] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      element_value: A ``ValueSpec`` object or an equivalent annotation as the
        spec for the list element.
      default: (Optional) default value for this spec.
      min_size: (Optional) min size of list. If None, 0 will be used.
      max_size: (Optional) max size of list.
      size: (Optional) size of List. A shortcut to specify min_size and max_size
        at the same time. `size` and `min_size`/`max_size` are mutual exclusive.
      transform: (Optional) user-defined function to be called on the input
        of `apply`. It could be used as a type converter or a custom
        validator which may raise errors.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    element_value = ValueSpec.from_annotation(element_value, auto_typing=True)

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
        key_specs.ListKey(min_size, max_size),
        element_value, 'Field of list element')
    super().__init__(
        list, default, transform, is_noneable=is_noneable, frozen=frozen
    )

  @property
  def element(self) -> Field:
    """Returns Field specification of list element."""
    return self._element

  @property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    return self._element.value.forward_refs

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

    # NOTE(daiyip): list elements can be contextual values, thus we try
    # to get their symbolic form instead of the evaluated form.
    getitem = getattr(value, 'sym_getattr', value.__getitem__)
    for i in range(len(value)):
      v = self._element.apply(
          getitem(i), allow_partial=allow_partial, transform_fn=child_transform,
          root_path=object_utils.KeyPath(i, root_path))
      if getitem(i) is not v:
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

  def _eq(self, other: 'List') -> bool:
    return self.element == other.element

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    return object_utils.kvlist_str(
        [
            ('', self._element.value, None),
            ('min_size', self.min_size, 0),
            ('max_size', self.max_size, None),
            ('default', self._default, MISSING_VALUE),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            element_value=(self._element.value, None),
            default=(self._default, MISSING_VALUE),
            min_size=(self.min_size, None),
            max_size=(self.max_size, None),
            transform=(self._transform, None),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Tuple[typing.Any, ...]) -> 'List':
    if len(type_args) != 1:
      raise TypeError(
          '`pg.typing.List` requires 1 type argument. ',
          f'Encountered: {type_args}.',
      )
    return cls(type_args[0])


class Tuple(Generic, ValueSpecBase):
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

  __serialization_key__ = 'pyglove.typing.Tuple'

  def __init__(
      self,
      element_values: typing.Union[
          ValueSpecOrAnnotation, typing.Sequence[ValueSpecOrAnnotation]
      ],
      default: typing.Optional[typing.Tuple[typing.Any, ...]] = MISSING_VALUE,
      min_size: typing.Optional[int] = None,
      max_size: typing.Optional[int] = None,
      size: typing.Optional[int] = None,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Tuple[typing.Any, ...]]
      ] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      element_values: A ValueSpec object or its equivalence as the element spec
        for a variable-length tuple, or a sequence of ValueSpec objects or their
        equivalences as the elements specs for a fixed-length tuple.
      default: (Optional) default value for this spec.
      min_size: (Optional) min size of tuple. If None, 0 will be used.
        Applicable only for variable-length tuple.
      max_size: (Optional) max size of list. Applicable only for variable-length
        tuple.
      size: (Optional) size of List. A shortcut to specify min_size and max_size
        at the same time. `size` and `min_size`/`max_size` are mutual exclusive.
      transform: (Optional) user-defined function to be called on the input
        of `apply`. It could be used as a type converter or a custom
        validator which may raise errors.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    if isinstance(element_values, (tuple, list)):
      if not element_values:
        raise ValueError("Argument 'element_values' must be a non-empty list.")
      if size is not None or min_size is not None or max_size is not None:
        raise ValueError(
            '"size", "min_size" and "max_size" are not applicable '
            f'for fixed-length Tuple with elements: {element_values!r}'
        )
      element_values = [
          ValueSpec.from_annotation(v, auto_typing=True) for v in element_values
      ]
      min_size = len(element_values)
      max_size = min_size
    else:
      element_values = ValueSpec.from_annotation(
          element_values, auto_typing=True
      )
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

    if isinstance(element_values, ValueSpec):
      elements = [
          Field(key_specs.TupleKey(None), element_values,
                'Field of variable-length tuple element')
      ]
    else:
      elements = []
      for i, element_value in enumerate(element_values):
        if not isinstance(element_value, ValueSpec):
          raise ValueError(
              f'Items in \'element_values\' must be ValueSpec objects.'
              f'Encountered: {element_value!r} at {i}.')
        elements.append(Field(key_specs.TupleKey(i), element_value,
                              f'Field of tuple element at {i}'))

    self._min_size = min_size
    self._max_size = max_size
    self._elements = elements
    super().__init__(
        tuple, default, transform, is_noneable=is_noneable, frozen=frozen
    )

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

  @functools.cached_property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    forward_refs = set()
    for elem in self.elements:
      forward_refs.update(elem.value.forward_refs)
    return forward_refs

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
                f'Input: {value}, Spec: {self!r}', root_path))
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
            v, allow_partial=allow_partial, transform_fn=child_transform,
            root_path=object_utils.KeyPath(i, root_path))
        for i, v in enumerate(value)
    ])

  def _extend(self, base: 'Tuple') -> None:
    """Tuple specific extension."""
    if self.fixed_length and base.fixed_length:
      if len(self.elements) != len(base.elements):
        raise TypeError(
            f'{self!r} cannot extend {base!r}: unmatched number of elements.')
      for i, element in enumerate(self._elements):
        element.extend(base.elements[i])
    elif self.fixed_length and not base.fixed_length:
      if base.min_size > len(self):
        raise TypeError(
            f'{self!r} cannot extend {base!r} as it has '
            f'less elements than required.')
      if base.max_size is not None and base.max_size < len(self):
        raise TypeError(
            f'{self!r} cannot extend {base!r} as it has '
            f'more elements than required.')
      for i, element in enumerate(self._elements):
        element.extend(base.elements[0])
    elif not self.fixed_length and base.fixed_length:
      raise TypeError(
          f'{self!r} cannot extend {base!r}: a variable length tuple '
          f'cannot extend a fixed length tuple.')
    else:
      assert not self.fixed_length and not base.fixed_length
      if self.min_size != 0 and self.min_size < base.min_size:
        raise TypeError(
            f'{self!r} cannot extend {base!r} as it has smaller min size.')
      if (self.max_size is not None
          and base.max_size is not None
          and self.max_size > base.max_size):
        raise TypeError(
            f'{self!r} cannot extend {base!r} as it has greater max size.')
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

  def _eq(self, other: typing.Any) -> bool:
    return (self.elements == other.elements
            and self.min_size == other.min_size
            and self.max_size == other.max_size)

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    if self.fixed_length:
      value = [f.value for f in self._elements]
      default_min, default_max = self._min_size, self._max_size
    else:
      value = self._elements[0].value
      default_min, default_max = 0, None
    return object_utils.kvlist_str(
        [
            ('', value, None),
            ('default', self._default, MISSING_VALUE),
            ('min_size', self._min_size, default_min),
            ('max_size', self._max_size, default_max),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    if self.fixed_length:
      element_values = [e.value for e in self._elements]
      min_size, max_size = None, None
    else:
      assert len(self._elements) == 1, self
      element_values = self._elements[0].value
      min_size, max_size = self.min_size, self.max_size

    return self.to_json_dict(
        fields=dict(
            element_values=(element_values, None),
            default=(self._default, MISSING_VALUE),
            min_size=(min_size, None),
            max_size=(max_size, None),
            transform=(self._transform, None),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Any) -> 'Tuple':
    if len(type_args) == 2 and type_args[1] is ...:
      return cls(ValueSpec.from_annotation(type_args[0], auto_typing=True))
    return cls(type_args)


class Dict(Generic, ValueSpecBase):
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
    ], transform=validate_sum)

    # A frozen dict that prevents subclass to extend/override.
    pg.typing.Dict([
        ('x', 1),
        ('y, 2.0)
    ]).freeze()
  """

  __serialization_key__ = 'pyglove.typing.Dict'

  def __init__(
      self,
      schema: typing.Union[
          class_schema.ValueSpec,
          Schema,
          typing.Dict[class_schema.FieldKeyDef, class_schema.FieldValueDef],
          typing.List[typing.Union[Field, class_schema.FieldDef]],
          None,
      ] = None,  # pylint: disable=bad-whitespace
      default: typing.Any = MISSING_VALUE,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Dict[typing.Any, typing.Any]]
      ] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      schema: (Optional) a Schema object for this Dict, or a dict of field key
        to field value definition, or a list of field definitions, or a value
        spec.
        If None, it specifies a schema-less Dict that may accept arbitrary
        key/value pairs.
        If a value spec, it specifies a Dict that may accept arbitrary keys with
        values constrained by the value spec.
        A field definition is a tuple of
          (<key_spec>, <value_spec>, [description], [metadata]).
        A field value definition is a tuple of
          (<value_spec>, [description], [metadata]).
      default: Default value. If MISSING_VALUE, the default value will be
        computed according to the schema.
      transform: (Optional) user-defined function to be called on the input
        of `apply`. It could be used as a type converter or a custom
        validator which may raise errors.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    if schema is not None:
      if not isinstance(schema, (Schema, list, dict)):
        schema = [
            (
                key_specs.StrKey(),
                ValueSpec.from_annotation(schema, auto_typing=True)
            )
        ]
      if not isinstance(schema, Schema):
        schema = class_schema.create_schema(schema, allow_nonconst_keys=True)

    self._schema = typing.cast(typing.Optional[Schema], schema)
    super().__init__(
        dict, default, transform, is_noneable=is_noneable, frozen=frozen
    )

    # Automatically generate default based on schema if default is not present.
    if MISSING_VALUE == default:
      self.set_default(default)

  @property
  def schema(self) -> typing.Optional[Schema]:
    """Returns the schema of this dict spec."""
    return self._schema

  def noneable(self) -> 'Dict':
    """Override noneable in Dict to always set default value None."""
    self._is_noneable = True
    self.set_default(None, False)
    return self

  def set_default(
      self,
      default: typing.Any,
      use_default_apply: bool = True,
      root_path: typing.Optional[object_utils.KeyPath] = None,
  ) -> ValueSpec:
    if MISSING_VALUE == default and self._schema:
      self._use_generated_default = True
      default = self._schema.apply({}, allow_partial=True)
      return super().set_default(default, use_default_apply=False)
    else:
      self._use_generated_default = False
      return super().set_default(default, use_default_apply)

  @functools.cached_property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    if self.schema is None:
      return set()
    forward_refs = set()
    for field in self.schema.fields.values():
      forward_refs.update(field.value.forward_refs)
    return forward_refs

  def _apply(self,
             value: typing.Dict[typing.Any, typing.Any],
             allow_partial: bool,
             child_transform: typing.Callable[
                 [object_utils.KeyPath, Field, typing.Any], typing.Any],
             root_path: object_utils.KeyPath) -> typing.Any:
    """Dict specific apply."""
    if not self._schema:
      return value
    return self._schema.apply(
        value,
        allow_partial=allow_partial,
        child_transform=child_transform,
        root_path=root_path
    )

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
    return typing.Dict[str, typing.Any]

  def _eq(self, other: 'Dict') -> bool:
    return self.schema == other.schema

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    return object_utils.kvlist_str(
        [
            ('fields',
             list(self._schema.values()) if self._schema else None,
             None),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs,
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    fields = dict(
        schema=(self._schema, None),
        transform=(self._transform, None),
        is_noneable=(self._is_noneable, False),
        frozen=(self._frozen, False),
    )
    if not self._use_generated_default:
      fields['default'] = (self._default, MISSING_VALUE)
    return self.to_json_dict(fields=fields, exclude_default=True, **kwargs)

  @classmethod
  def with_type_args(cls, type_args: typing.Any) -> 'Dict':
    if len(type_args) == 1:
      if not isinstance(type_args[0], dict):
        raise TypeError(
            '`pg.typing.Dict` accepts 1 dict type argument as the schema, '
            'or 2 type arguments as the key type and value type. '
            f'Encountered: {type_args!r}.'
        )
      return cls(type_args[0])

    if len(type_args) != 2:
      raise TypeError(
          '`pg.typing.Dict` accepts 1 dict type argument as the schema, '
          'or 2 type arguments as the key type and value type. '
          f'Encountered: {type_args!r}.'
      )
    if type_args[0] is str:
      return cls([(key_specs.StrKey(), type_args[1])])
    # TODO(daiyip): support non-string type keys.
    return cls()


class Object(Generic, ValueSpecBase):
  """Value spec for object type.

  Examples::

    # A required instance of class A and its subclasses.
    pg.typing.Object(A)

    # A required instance of class A and its subclasses (forward declaration).
    pg.typing.Object('A')

    # An optional instance of class A with None as its default value.
    pg.typing.Object(A).noneable()

    # An instance of class A with default value.
    pg.typing.Object(A, default=A())
  """

  __serialization_key__ = 'pyglove.typing.Object'

  def __init__(
      self,
      t: typing.Union[
          typing.Type[typing.Any],
          class_schema.ForwardRef,
          str
      ],
      default: typing.Any = MISSING_VALUE,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Any]
      ] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      t: Class of the object. Objects of subclass of this class is acceptable.
      default: (Optional) default value of this spec.
      transform: (Optional) user-defined function to be called on the input
        of `apply`. It could be used as a type converter or a custom
        validator which may raise errors.
      is_noneable: If True, None is acceptable.
      frozen: If True, values other than the default value is not accceptable.
    """
    if t is None:
      raise TypeError('"cls" for Object spec cannot be None.')

    forward_ref = None
    type_args = []
    if isinstance(t, class_schema.ForwardRef):
      forward_ref = t
    elif isinstance(t, str):
      forward_ref = class_schema.ForwardRef(_get_spec_callsite_module(), t)
    elif isinstance(t, type):
      if t is object:
        raise TypeError('<class \'object\'> is too general for Object spec.')
    elif not pg_inspect.is_generic(t):
      raise TypeError('"cls" for Object spec should be a type or str.')

    self._forward_ref = forward_ref
    self._type_args = type_args
    super().__init__(
        t, default, transform, is_noneable=is_noneable, frozen=frozen
    )

  @property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    if self._forward_ref is None:
      return set()
    return set([self._forward_ref])

  @property
  def cls(self) -> typing.Type[typing.Any]:
    """Returns the class of this object spec."""
    if self._forward_ref is None:
      return typing.cast(typing.Type[typing.Any], self._value_type)
    return self._forward_ref.cls

  @property
  def value_type(self) -> typing.Type[typing.Any]:
    return self.cls

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
    if not base.is_compatible(self):
      raise TypeError(f'{self!r} cannot extend {base!r}: incompatible class.')

  def _is_compatible(self, other: 'Object') -> bool:
    """Object specific compatiblity check."""
    # NOTE(daiyip): When either the current spec or the other spec contains
    # unresolved forward declarations, we consider them compatible.
    if not self.type_resolved or not other.type_resolved:
      return True
    return pg_inspect.is_subclass(other.cls, self.cls)

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    if self._forward_ref is not None:
      return self._forward_ref.as_annotation()
    return self._value_type

  @property
  def schema(self) -> typing.Optional[Schema]:
    """Returns the schema of object class if available."""
    return getattr(self.value_type, '__schema__', None)

  def _eq(self, other: 'Object') -> bool:
    """Operator==."""
    if self.type_resolved:
      return self.value_type == other.value_type
    return self.forward_refs == other.forward_refs

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    if self._forward_ref is not None:
      name = self._forward_ref.name
    else:
      name = self._value_type.__name__
    return object_utils.kvlist_str(
        [
            ('', object_utils.RawText(name), None),
            ('default', self._default, MISSING_VALUE),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            t=(self.cls, None),
            default=(self._default, MISSING_VALUE),
            transform=(self._transform, None),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Any) -> 'Object':
    if len(type_args) != 1:
      raise TypeError(
          '`pg.typing.Object` requires 1 type argument. '
          f'Encountered: {type_args}'
      )
    return cls(type_args[0])


class Callable(Generic, ValueSpecBase):
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

  __serialization_key__ = 'pyglove.typing.Callable'

  def __init__(
      self,
      args: typing.Optional[typing.List[ValueSpecOrAnnotation]] = None,
      kw: typing.Optional[
          typing.List[typing.Tuple[str, ValueSpecOrAnnotation]]
      ] = None,
      returns: typing.Optional[ValueSpecOrAnnotation] = None,
      default: typing.Any = MISSING_VALUE,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Callable[..., typing.Any]]
      ] = None,
      callable_type: typing.Optional[typing.Type[typing.Any]] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor."""
    args = args or []
    kw = kw or []
    if not isinstance(args, list):
      raise TypeError(
          f'\'args\' should be a list of ValueSpec objects. '
          f'Encountered: {args!r}.')
    args = [ValueSpec.from_annotation(arg, auto_typing=True) for arg in args]

    if not isinstance(kw, list):
      raise TypeError(
          f'\'kw\' should be a list of (name, value_spec) tuples. '
          f'Encountered: {kw!r}.')
    for arg in kw:
      if (
          not isinstance(arg, tuple)
          or len(arg) != 2
          or not isinstance(arg[0], str)
      ):
        raise TypeError(
            f'\'kw\' should be a list of (name, value_spec) tuples. '
            f'Encountered: {kw!r}.')

    kw = [(k, ValueSpec.from_annotation(v, auto_typing=True)) for k, v in kw]

    if returns is not None:
      returns = ValueSpec.from_annotation(returns, auto_typing=True)
    self._args = args
    self._kw = kw
    self._return_value = returns
    super().__init__(
        callable_type,
        default,
        transform,
        is_noneable=is_noneable,
        frozen=frozen,
    )

  @functools.cached_property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    forward_refs = set()
    for arg in self.args:
      forward_refs.update(arg.forward_refs)
    for _, arg in self.kw:
      forward_refs.update(arg.forward_refs)
    if self.return_value:
      forward_refs.update(self.return_value.forward_refs)
    return forward_refs

  @property
  def args(self) -> typing.List[ValueSpec]:
    """Value specs for positional arguments."""
    return self._args

  @property
  def kw(self) -> typing.List[typing.Tuple[str, ValueSpec]]:
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

    signature = callable_signature.signature(
        value, auto_typing=False, auto_doc=False
    )

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
      assert signature.varargs
      assert isinstance(signature.varargs.value_spec, List), signature.varargs
      dest_spec = signature.varargs.value_spec.element.value
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
      elif signature.varkw:
        assert isinstance(signature.varkw.value_spec, Dict), signature.varkw
        varkw_value_spec = signature.varkw.value_spec.schema.dynamic_field.value   # pytype: disable=attribute-error
        if not varkw_value_spec.is_compatible(src_spec):
          raise TypeError(
              object_utils.message_on_path(
                  f'Value spec of keyword argument {arg_name!r} is not '
                  f'compatible with the value spec of '
                  f'**{signature.varkw.name}. '
                  f'Expected: {varkw_value_spec!r}, '
                  f'Actual: {src_spec!r}.', path))
      else:
        raise TypeError(
            object_utils.message_on_path(
                f'Keyword argument {arg_name!r} does not exist in {value!r}.',
                path))

    # Check return value
    if (self._return_value and signature.return_value
        and not isinstance(signature.return_value, Any)
        and not self._return_value.is_compatible(signature.return_value)):
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

  def _eq(self, other: 'Callable') -> bool:
    return (self._args == other.args
            and self._kw == other.kw
            and self._return_value == other.return_value)

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this spec."""
    return object_utils.kvlist_str(
        [
            ('args', self._args, []),
            ('kw', self._kw, []),
            ('returns', self._return_value, None),
            ('default', self._default, MISSING_VALUE),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False)
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs,
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            args=(self._args, []),
            kw=(self._kw, []),
            returns=(self._return_value, None),
            default=(self._default, MISSING_VALUE),
            transform=(self._transform, None),
            callable_type=(self._value_type, None),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Any) -> 'Callable':
    if len(type_args) != 2:
      raise TypeError(
          f'`pg.typing.{cls.__name__}` requires 2 type arguments. '
          f'Encountered: {type_args}'
      )
    if isinstance(type_args[0], (tuple, list)):
      args = type_args[0]
    elif type_args[0] is not ...:
      args = [type_args[0]]
    else:
      args = []
    returns = type_args[1]
    if returns is None:
      returns = type(None)
    return cls(args=args, returns=returns)


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

  __serialization_key__ = 'pyglove.typing.Functor'

  def __init__(
      self,
      args: typing.Optional[typing.List[ValueSpecOrAnnotation]] = None,
      kw: typing.Optional[
          typing.List[typing.Tuple[str, ValueSpecOrAnnotation]]
      ] = None,
      returns: typing.Optional[ValueSpecOrAnnotation] = None,
      default: typing.Any = MISSING_VALUE,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Callable[..., typing.Any]]
      ] = None,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor."""
    super().__init__(
        args=args,
        kw=kw,
        returns=returns,
        default=default,
        transform=transform,
        callable_type=object_utils.Functor,
        is_noneable=is_noneable,
        frozen=frozen,
    )

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    return object_utils.Functor

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    exclude_keys = kwargs.pop('exclude_keys', set())
    exclude_keys.add('callable_type')
    return super().to_json(exclude_keys=exclude_keys, **kwargs)


class Type(Generic, ValueSpecBase):
  """Value spec for type.

  Examples::

    # A required type or subclass of A.
    pg.typing.Type(A)

    # An required type or subclass of A (forward declaration).
    pg.typing.Type('A')

    # An optional type or subclass of A.
    pg.typing.Type(A).noneable()

    # A required type or subclass of A with default value B
    # (B is a subclass of A).
    pg.typing.Type(A, default=B)
  """

  __serialization_key__ = 'pyglove.typing.Type'

  def __init__(
      self,
      t: typing.Union[typing.Type[typing.Any], str],
      default: typing.Type[typing.Any] = MISSING_VALUE,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    forward_ref = None
    if isinstance(t, str):
      forward_ref = class_schema.ForwardRef(_get_spec_callsite_module(), t)
    elif not (
        isinstance(t, type) or pg_inspect.is_generic(t) or t is typing.Any
    ):
      raise TypeError(f'{t!r} is not a type.')
    self._expected_type = t
    self._forward_ref = forward_ref
    super().__init__(type, default, is_noneable=is_noneable, frozen=frozen)

  @property
  def type(self) -> typing.Type[typing.Any]:
    """Returns desired type."""
    if self._forward_ref is not None:
      return self._forward_ref.cls
    return self._expected_type

  @functools.cached_property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    if self._forward_ref is None:
      return set()
    return set([self._forward_ref])

  def _validate(self, path: object_utils.KeyPath, value: typing.Type) -> None:  # pylint: disable=g-bare-generic
    """Validate applied value."""
    if self.type_resolved and not pg_inspect.is_subclass(value, self.type):
      raise ValueError(
          object_utils.message_on_path(
              f'{value!r} is not a subclass of {self.type!r}', path))

  def _is_compatible(self, other: 'Type') -> bool:
    """Type specific compatiblity check."""
    # NOTE(daiyip): When either the current spec or the other spec contains
    # unresolved forward declarations, we consider them compatible.
    if not self.type_resolved or not other.type_resolved:
      return True
    return pg_inspect.is_subclass(other.type, self.type)

  def _extend(self, base: 'Type') -> None:
    """Type specific extension."""
    if not base.is_compatible(self):
      raise TypeError(f'{self!r} cannot extend {base!r}: incompatible type.')

  def _annotate(self) -> typing.Any:
    """Annotate with PyType annotation."""
    expected_type = self._expected_type
    if self._forward_ref is not None:
      expected_type = self._forward_ref.as_annotation()
    return typing.Type[expected_type]  # pytype: disable=invalid-annotation

  def _eq(self, other: 'Type') -> bool:
    """Equals."""
    if self.type_resolved:
      return self.type == other.type
    return self.forward_refs == other.forward_refs

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    return object_utils.kvlist_str(
        [
            ('', self._expected_type, None),
            ('default', self._default, MISSING_VALUE),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            t=(self._expected_type, None),
            default=(self._default, MISSING_VALUE),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  @classmethod
  def with_type_args(cls, type_args: typing.Any) -> 'Type':
    if len(type_args) != 1:
      raise TypeError(
          '`pg.typing.Type` requires 1 type argument. '
          f'Encountered: {type_args}.'
      )
    return cls(t=type_args[0])


# pytype: disable=attribute-error


class Union(Generic, ValueSpecBase):
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

  __serialization_key__ = 'pyglove.typing.Union'

  def __init__(
      self,
      candidates: typing.Sequence[ValueSpecOrAnnotation],
      default: typing.Any = MISSING_VALUE,
      is_noneable: bool = False,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      candidates: A sequence of value spec objects or their equivalence as the
        spec for candidate types.
      default: (Optional) default value of this spec.
      is_noneable: (Optional) If True, None is acceptable for this spec.
      frozen: If True, values other than the default value is not accceptable.
    """
    if not isinstance(candidates, (tuple, list)) or len(candidates) < 2:
      raise ValueError(
          f'Argument \'candidates\' must be a list of at least 2 '
          f'elements. Encountered {candidates}.')
    candidates = [
        ValueSpec.from_annotation(c, auto_typing=True) for c in candidates
    ]
    candidates_by_type = {}
    has_noneable_candidate = False
    for i, c in enumerate(candidates):
      if not isinstance(c, ValueSpec):
        raise ValueError(
            f'Items in \'candidates\' must be ValueSpec objects.'
            f'Encountered {c} at {i}.')
      if c.is_noneable:
        has_noneable_candidate = True

      # NOTE(daiyip): When forward declaration is present (e.g.
      # `pg.typing.Object('A')`), calling `value_type` will trigger the
      # resolution of the referred type. Therefore, we refer to `_value_type`
      # instead of `value_type` to access the class name string in such cases.
      spec_type = (c.__class__, getattr(c, '_value_type'))
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

    candidate_types = set()
    no_value_type_check = False
    for c in candidates:
      child_value_type = getattr(c, '_value_type')
      if child_value_type is None:
        no_value_type_check = True
      elif isinstance(child_value_type, tuple):
        candidate_types.update(child_value_type)
      else:
        candidate_types.add(child_value_type)

    self._candidates = candidates
    union_value_type = None if no_value_type_check else tuple(candidate_types)
    super().__init__(
        union_value_type,
        default,
        is_noneable=is_noneable or has_noneable_candidate,
        frozen=frozen,
    )

  @functools.cached_property
  def forward_refs(self) -> typing.Set[class_schema.ForwardRef]:
    """Returns forward references used in this spec."""
    forward_refs = set()
    for c in self._candidates:
      forward_refs.update(c.forward_refs)
    return forward_refs

  @functools.cached_property
  def value_type(
      self) -> typing.Optional[typing.Tuple[typing.Type[typing.Any]]]:
    # NOTE(daiyip): Override `value_type` property to delay the type resolution
    # when forward declaration is invovled.
    if self._value_type is None or not self.forward_refs:
      return self._value_type

    value_types = set()
    for c in self._candidates:
      child_value_type = c.value_type
      if not isinstance(child_value_type, tuple):
        child_value_type = (child_value_type,)
      value_types.update(child_value_type)
    return tuple(value_types)

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
      if (c.type_resolved
          and (c.value_type is None or isinstance(value, c.value_type))):
        return c.apply(
            value,
            allow_partial=allow_partial,
            child_transform=child_transform,
            root_path=root_path)

    # NOTE(daiyip): This code is to support consider A as B scenario when there
    # is a converter from A to B (converter may return value that is not B). A
    # use case is that tf.Variable is not a tf.Tensor, but value spec of
    # tf.Tensor should be able to accept tf.Variable.
    matched_candidate = None
    for c in self._candidates:
      if c.type_resolved and type_conversion.get_converter(
          type(value), c.value_type) is not None:
        matched_candidate = c
        break

    if self.type_resolved:
      # `_apply` is entered only when there is a type match or conversion path.
      assert matched_candidate is not None
      return matched_candidate.apply(
          value, allow_partial, child_transform, root_path)

    # Return value directly if the forward declaration of the current union
    # is unsolved.
    return value

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
        if c.__class__ is v.__class__ and (
            c.__class__ is not Object
            or pg_inspect.is_subclass(c.value_type, v.value_type)
        ):
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

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    return object_utils.kvlist_str(
        [
            ('', self._candidates, None),
            ('default', self._default, MISSING_VALUE),
            ('noneable', self._is_noneable, False),
            ('frozen', self._frozen, False),
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        list_wrap_threshold=kwargs.pop('list_wrap_threshold', 20),
        **kwargs
    )

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            candidates=(self._candidates, None),
            default=(self._default, MISSING_VALUE),
            is_noneable=(self._is_noneable, False),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )

  def _eq(self, other: 'Union') -> bool:
    if len(self.candidates) != len(other.candidates):
      return False
    for sc in self.candidates:
      oc = other.get_candidate(sc)
      if sc != oc:
        return False
    return True

  @classmethod
  def with_type_args(cls, type_args: typing.Any) -> ValueSpec:
    if len(type_args) < 2:
      raise TypeError(
          '`pg.typing.Union` requires at least 2 type arguments. '
          f'Encountered: {type_args}.'
      )
    type_args = list(type_args)
    noneable = None in type_args
    if noneable:
      type_args.remove(None)
    if len(type_args) == 1:
      v = ValueSpec.from_annotation(type_args[0], auto_typing=True)
    else:
      v = cls(type_args)
    if noneable:
      v = v.noneable()
    return v

# pytype: enable=attribute-error


class GenericTypeAlias(Generic):
  """Base class for generic type aliases."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    raise TypeError(
        f'Generic type alias {self.__class__.__name__} cannot be instantiated.')


class Sequence(GenericTypeAlias):
  """Sequence."""

  @classmethod
  def with_type_args(
      cls, type_args: typing.Tuple[typing.Any, ...]) -> Union:
    if len(type_args) != 1:
      raise TypeError(
          '`pg.typing.Sequence` requires 1 type argument. '
          f'Encountered: {type_args}.'
      )
    elem_spec = ValueSpec.from_annotation(type_args[0], auto_typing=True)
    return Union([List(elem_spec), Tuple(elem_spec)])


class Optional(GenericTypeAlias):
  """Optional."""

  @classmethod
  def with_type_args(
      cls, type_args: typing.Tuple[typing.Any, ...]) -> ValueSpec:
    if len(type_args) != 1:
      raise TypeError(
          '`pg.typing.Optional` requires 1 type argument. '
          f'Encountered: {type_args}.'
      )
    spec = ValueSpec.from_annotation(type_args[0], auto_typing=True)
    return spec.noneable()


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

  __serialization_key__ = 'pyglove.typing.Any'

  def __init__(
      self,
      default: typing.Any = MISSING_VALUE,
      annotation: typing.Any = MISSING_VALUE,
      transform: typing.Optional[
          typing.Callable[[typing.Any], typing.Any]
      ] = None,
      frozen: bool = False,
  ):  # pytype: disable=annotation-type-mismatch
    """Constructor.

    Args:
      default: (Optional) default value of this spec.
      annotation: (Optional) external provided type annotation.
      transform: (Optional) user-defined function to be called on the input
        of `apply`. It could be used as a type converter or a custom
        validator which may raise errors.
      frozen: If True, values other than the default value is not accceptable.
    """
    super().__init__(
        object, default, transform, is_noneable=True, frozen=frozen
    )
    self._annotation = annotation

  def is_compatible(self, other: ValueSpec) -> bool:
    """Any is compatible with any ValueSpec."""
    return True

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      **kwargs,
  ) -> str:
    """Format this object."""
    return object_utils.kvlist_str(
        [
            ('default', self._default, MISSING_VALUE),
            ('frozen', self._frozen, False),
            ('annotation', self._annotation, MISSING_VALUE)
        ],
        label=self.__class__.__name__,
        compact=compact,
        verbose=verbose,
        root_indent=root_indent,
        markdown=markdown,
        **kwargs
    )

  def annotate(self, annotation: typing.Any) -> 'Any':
    """Set external type annotation."""
    self._annotation = annotation
    return self

  @property
  def annotation(self) -> typing.Any:
    """Returns type annotation."""
    return self._annotation

  def _eq(self, other: 'Any') -> bool:
    return self.annotation == other.annotation

  def to_json(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    return self.to_json_dict(
        fields=dict(
            default=(self._default, MISSING_VALUE),
            annotation=(self._annotation, MISSING_VALUE),
            transform=(self._transform, None),
            frozen=(self._frozen, False),
        ),
        exclude_default=True,
        **kwargs,
    )


def _any_if_no_annotation(annotation: typing.Any):
  """Returns typing.Any if annotation is MISSING_VALUE."""
  return typing.Any if MISSING_VALUE == annotation else annotation


# We need to figure out the source file when a ForwardRef is created.
# This is where a `ValueSpec` is created. It could be from direct construction
# of the value spec object or calling from_annotation.
_VALUE_SPEC_CREATION_ROOT_SITES = [
    'pyglove/core/typing/',
    'pyglove/core/symbolic/',
    'abc.py',
]


def _get_spec_callsite_module():
  """Returns the module of the callsite where ValueSpec objects are created."""
  calling_module = None
  callstack = inspect.stack()
  for frame, file, *_ in callstack[1:]:
    # NOTE(daiyip): Dealing with Jupyter notebook. For spec callsite within
    # the notebook cell, the returned module shall be __main__.
    if file.startswith('<ipython-input'):
      break
    if (file.endswith('_test.py')
        or ('<' not in file   # Exclude callstack from Python builtin modules.
            and all(
                site not in file for site in _VALUE_SPEC_CREATION_ROOT_SITES))):
      calling_module = inspect.getmodule(frame)
      break
  return calling_module or __main__

# Set the forward declared values to avoid circular dependency.
ValueSpec.ListType = List
ValueSpec.DictType = Dict
ValueSpec.ObjectType = Object


def ensure_value_spec(
    value_spec: class_schema.ValueSpec,
    src_spec: class_schema.ValueSpec,
    root_path: typing.Optional[object_utils.KeyPath] = None
) -> typing.Optional[class_schema.ValueSpec]:
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
