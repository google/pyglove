# Copyright 2021 The PyGlove Authors
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
"""Symbolizing existing Python functions and classes.

While users can create symbolic classes by subclassing ``pg.Object``,
`pg.wrapping` module provides methods to create symbolic wrapper classes
based on existing classes. Besides, ``pg.symbolize`` is introduced as a
unified API for symbolizing functions and classes.
"""

import abc
import functools
import inspect
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from pyglove.core import detouring
from pyglove.core import object_utils
from pyglove.core import typing as pg_typing

from pyglove.core.symbolic import dict as pg_dict  # pylint: disable=unused-import
from pyglove.core.symbolic import list as pg_list  # pylint: disable=unused-import
from pyglove.core.symbolic import object as pg_object


class ClassWrapperMeta(pg_object.ObjectMeta):
  """Metaclass for class wrapper."""

  def __repr__(self) -> str:
    wrapped_cls = getattr(self, 'sym_wrapped_cls', None)
    if wrapped_cls is None:
      return f'<class {self.__type_name__!r}>'
    return f'Symbolic[{wrapped_cls!r}]'

  def __getattr__(self, name):
    """Pass through attribute requests to sym_wrapped_cls."""
    return getattr(object.__getattribute__(self, 'sym_wrapped_cls'), name)


class ClassWrapper(pg_object.Object, metaclass=ClassWrapperMeta):
  """Base class for symbolic class wrapper.

  Please see :func:`pyglove.wrap` for details.
  """

  @property
  @abc.abstractmethod
  def sym_wrapped(self):
    """Returns symbolically wrapped object."""


class _SubclassedWrapperBase(ClassWrapper):
  """Base for class wrappers using multi-inheritance."""

  # If True, fill the descriptions of schema fields automatically from the
  # class and __init__ docstrings.
  auto_doc = False

  # If True, PyGlove typing will be enabled based on type annotations from
  # the `__init__` method.
  auto_typing = False

  @object_utils.explicit_method_override
  def __init__(self, *args, **kwargs):
    """Overridden __init__ to construct symbolic wrapper only."""
    # NOTE(daiyip): We avoid `__init__` to be called multiple times.
    # This behavior is intended for a use case that detours a source class to
    # a function that returns an instance of the source class' symbolic
    # wrapper. The function may want to pass modified arguments to the
    # symbolic wrapper's `__init__`, but since the symbolic wrapper is a
    # subclass of the source class, the Python runtime will call its
    # `__init__` again with the original arguments.
    if not hasattr(self, '_wrapper_initialized'):
      # We set the '_wrapper_initialized' property first as within the
      # `ClassWrapper.__init__` it may trigger `Symbolic._on_bound` which
      # in turns trigger the user class `__init__`. If the user class
      # `__init__` calls `super().__init__`, this method will be reentered.
      object.__setattr__(self, '_wrapper_initialized', True)
      object.__setattr__(self, '_wrapped_cls_initialized', False)
      object.__setattr__(self, '_wrapped_cls_initializing', False)
      super().__init__(*args, explicit_init=True, **kwargs)  # pylint: disable=non-parent-init-called

  @property
  def wrapped_cls_initialized(self):
    """Returns True if wrapped class is initialized. Otherwise False."""
    return getattr(self, '_wrapped_cls_initialized', False)

  @classmethod
  def __init_subclass__(cls):
    # Class wrappers inherit `__init__` from the user class. Therefore, we mark
    # all of them as explicitly overridden.
    object_utils.explicit_method_override(cls.__init__)

    super().__init_subclass__()
    if cls.__init__ is _SubclassedWrapperBase.__init__:
      # Symbolized class inherits __init__ from ClassWrapper, meaning that
      # neither it defines its own __init__ nor inherits __init__ from any
      # non-symbolic bases. In such case we use object.__init__ as the original
      # __init__ method.
      cls.__init__ = object.__init__
      cls._call_init = cls.__post_init__

    # Determine if a wrapper is created from symbolizing a regular class
    # or subclassed from another wrapper class.
    if len(cls.__bases__) == 2 and cls.__bases__[1] is _SubclassedWrapperBase:
      wrapped_cls = cls.__bases__[0]
    else:
      wrapped_cls = None
    setattr(cls, 'sym_wrapped_cls', wrapped_cls)

    inherits_sym_init = getattr(cls.__init__, 'is_sym_init', False)
    if not inherits_sym_init:
      # This means the class is either wrapped from a user class
      # or subclassing a symbolic wrapper with its own __init__.
      # In both cases, we need to generate an __init__ wrapper for
      # calling the symbolic initialization.
      setattr(cls, '__orig_init__', cls.__init__)
      init_arg_list, arg_fields = _extract_init_signature(
          cls, auto_doc=cls.auto_doc, auto_typing=cls.auto_typing)

      @object_utils.explicit_method_override
      @functools.wraps(cls.__init__)
      def _sym_init(self, *args, **kwargs):
        _SubclassedWrapperBase.__init__(self, *args, **kwargs)

        # The following code is to deal with call to `super().__init__()`.
        # In such case, we need to invoke the original __init__ instead of
        # the symbolic init, which can be told by the
        # `self._wrapped_cls_initializing` property.
        if self._wrapped_cls_initializing:  # pylint: disable=protected-access
          getattr(cls, '__orig_init__')(self, *args, **kwargs)

      setattr(_sym_init, '__signature__', inspect.signature(cls.__init__))
      setattr(_sym_init, 'is_sym_init', True)
      setattr(cls, '__init__', _sym_init)

      # We do not extend existing schema which is inherited from the base
      # class.
      cls.update_schema(arg_fields, init_arg_list=init_arg_list, extend=False)
    else:
      assert hasattr(cls, '__orig_init__')

  def _init_user_cls(self, *args, **kwargs):
    """Init user class."""
    self.__orig_init__(*args, **kwargs)

  def _on_reset(self):
    """Reset object state. Subclass can override."""
    symbolic_attrs = [
        (k, self.__dict__[k])
        for k in getattr(self, '_wrapper_symbolic_attrs')
    ]
    self.__dict__.clear()
    for k, v in symbolic_attrs:
      self.__dict__[k] = v

  def _on_bound(self):
    """Overriden _on_bound to handle symbolic members update."""
    super()._on_bound()

    # NOTE(daiyip): store attribute names of class wrapper before
    # calling the `__init__` of the user class, so we can reset the state
    # by removing extra attributes from `self.__dict__` later.
    if not hasattr(self, '_wrapper_symbolic_attrs'):
      object.__setattr__(
          self,
          '_wrapper_symbolic_attrs',
          list(self.__dict__.keys()) + ['_wrapper_symbolic_attrs'])

    object.__setattr__(self, '_wrapped_cls_initialized', False)
    if not self.sym_abstract:
      # NOTE(daiyip): We delay the call to `__init__` of the user class
      # until current object is deterministic.
      self._on_reset()
      object.__setattr__(self, '_wrapped_cls_initializing', True)
      self._call_init()
      object.__setattr__(self, '_wrapped_cls_initializing', False)
      object.__setattr__(self, '_wrapped_cls_initialized', True)

  def _call_init(self):
    """Invoke the wrapped user class __init__."""
    init_arg_list = self.__class__.init_arg_list
    assert init_arg_list is not None, init_arg_list
    kwargs = dict(self.sym_init_args)
    list_args = []

    if init_arg_list and init_arg_list[-1].startswith('*'):
      vararg_name = init_arg_list[-1][1:]
      varargs = kwargs.pop(vararg_name)
      for arg_name in init_arg_list[:-1]:
        assert arg_name in kwargs
        list_args.append(kwargs.pop(arg_name))
      list_args.extend(varargs)
    self._init_user_cls(*list_args, **kwargs)

  def __post_init__(self):
    """Post initialization when class is being used as dataclass."""

  @property
  def sym_wrapped(self):
    """Returns wrapped object."""
    return self


def _subclassed_wrapper(
    user_cls,
    *,
    use_symbolic_repr: bool,
    use_symbolic_comp: bool,
    use_auto_doc: bool,
    use_auto_typing: bool,
    reset_state_fn: Optional[Callable[[Any], None]],
    class_name: Optional[str] = None,
    module_name: Optional[str] = None):
  """Class wrapper implementation by regular multi-inheritance."""
  # NOTE(daiyip): The user class may have a user-defined metaclass, which
  # conflicts with the metaclass of the symbolic base. Therefore, we detect
  # such case and create a common metaclass when a conflict is detected.
  wrapper_base_cls = _SubclassedWrapperBase
  if issubclass(wrapper_base_cls.__class__, user_cls.__class__):
    wrapper_metaclass = wrapper_base_cls.__class__
  else:
    wrapper_metaclass = type(
        'ClassWrapperMeta',
        (user_cls.__class__, wrapper_base_cls.__class__), {})

  class SubclassedWrapper(
      user_cls, wrapper_base_cls, metaclass=wrapper_metaclass):
    """Class wrapper bound to a user class."""
    sym_wrapped_cls = user_cls

    # Disable auto register so we can use function module and name
    # for registration later.
    auto_register = False

    # If True, set the descriptions of schema fields automatically from
    # class and __init__ docstrings.
    auto_doc = use_auto_doc

    # If True, PyGlove typing will be enabled based on type annotations from
    # the `__init__` method.
    auto_typing = use_auto_typing

    # NOTE(daiyip): For class wrappers, all symbolic properties are exposed from
    # `self.sym_init_args`. Therefore we do not allow symbolic members to be
    # accessed or changed via attributes using `self.<member_name>`.
    allow_symbolic_attribute = False

    # This class property lets `pg.Object` control whether to delegate the
    # implementation of `__setattr__` to `object.__setattr__`, when the flag is
    # set to True, `pg.Object.__setattr__` will behave as symbolically rebind.
    # Otherwise, `pg.Object.__setattr__` downgrades to `object.__setattr__`
    # which is more friendly to be used as a base class, this is especially
    # important for getting out of the `super().__setattr__` hell in
    # the user class under multi-inheritance scenarios.
    allow_symbolic_assignment = False

    # This class property lets `pg.Object` to control whether to allow `sym_*`
    # to be used for `__eq__`, `__ne__` and `__hash__`. If `user_cls` defines
    # these methods, even `use_symbolic_comp` is set to True, the methods from
    # `user_cls` will be used.
    use_symbolic_comparison = use_symbolic_comp

    # Do not infer symbolic fields from annotations. This is because that
    # symbolic fields are inspected from the `__init__`` signature.
    infer_symbolic_fields_from_annotations = False

    # ClassWrapper's schema will be based inspected based on __init__ instead
    # of annotations.
    auto_schema = False

  cls = SubclassedWrapper
  cls.__name__ = class_name or user_cls.__name__
  cls.__module__ = module_name or user_cls.__module__
  # NOTE(daiyip): for class wrapper, currently we don't not support nested
  # class definition.
  cls.__qualname__ = cls.__name__
  cls.__doc__ = user_cls.__doc__

  # Enable automatic registration for subclass.
  cls.auto_register = True

  if reset_state_fn:
    setattr(cls, '_on_reset', reset_state_fn)

  if use_symbolic_repr:
    cls.__repr__ = pg_object.Object.__repr__
    cls.__str__ = pg_object.Object.__str__
  else:
    cls.__repr__ = user_cls.__repr__
    cls.__str__ = user_cls.__str__
  return cls


def wrap(
    cls,
    init_args: Union[
        List[Union[pg_typing.Field, pg_typing.FieldDef]],
        Dict[pg_typing.FieldKeyDef, pg_typing.FieldValueDef],
        None
    ] = None,
    *,
    reset_state_fn: Optional[Callable[[Any], None]] = None,
    repr: bool = True,    # pylint: disable=redefined-builtin
    eq: bool = False,
    class_name: Optional[str] = None,
    module_name: Optional[str] = None,
    auto_doc: bool = False,
    auto_typing: bool = False,
    serialization_key: Optional[str] = None,
    additional_keys: Optional[List[str]] = None,
    override: Optional[Dict[str, Any]] = None
) -> Type['ClassWrapper']:
  """Makes a symbolic class wrapper from a regular Python class.

  ``pg.wrap`` is called by :func:`pyglove.symbolize` for symbolizing existing
  Python classes. For example::

    class A:
      def __init__(self, x):
        self.x = x

    # The following two lines are equivalent.
    A1 = pg.symbolize(A)
    A2 = pg.wrap(A)

  Besides passing the source class, ``pg.wrap`` allows the user to pass symbolic
  field definitions for the init arguments. For example::

    A3 = pg.wrap(A, [
      ('x', pg.typing.Int())
    ])

  Moreover, multiple flags are provided to determine whether or not to use the
  symbolic operations as the default behaviors. For example::

    A4 = pg.wrap(
      A,
      [],
      # Instead clearing out all internal states (default),
      # do not reset internal state.
      reset_state_fn=lambda self: None,
      # Use symbolic representation for __repr__ and __str__.
      repr=True,
      # use symbolic equality for __eq__, __ne__ and __hash__.
      eq=True,
      # Customize the class name obtained (the default behaivor
      # is to use the source class name).
      class_name='A4'
      # Customize the module name for created class (the default
      # behavior is to use the source module name).
      module_name='my_module')

  Args:
    cls: Class to wrap.
    init_args: An optional list of field definitions for the arguments of
      __init__. It can be a sparse value specifications for argument in the
      __init__ method of `cls`.
    reset_state_fn: An optional callable object to reset the internal state of
      the user class when rebind happens.
    repr: Options for generating `__repr__` and `__str__`. If True (default),
      use symbolic representation if the user class does not define its own.
      Otherwise use the user class' definition. If False, always use
      non-symbolic representations, which falls back to `object.__repr__` and
      `object.__str__` if the user class does not define them.
    eq: Options for generating `__eq__`, `__ne__` and `__hash__`. If True and
      the `user_cls` defines `__eq__`, `__ne__` and `__hash__`, use the
      definitions from the `user_cls`. If True and the `user_cls` does not
      define `__eq__`, `__ne__` and `__hash__`, use symbolic eq/hash. If False
      (default), use `user_cls`'s definition if present, or the definitions from
      the `object` class.
    class_name: An optional string used as class name for the wrapper class. If
      None, the wrapper class will use the class name of the wrapped class.
    module_name: An optional string used as module name for the wrapper class.
      If None, the wrapper class will use the module name of the wrapped class.
    auto_doc: If True, the descriptions for init argument fields will be
      extracted from docstring if present.
    auto_typing: If True, PyGlove typing (runtime-typing) will be enabled based
      on type annotations inspected from the `__init__` method.
    serialization_key: An optional string to be used as the serialization key
      for the class during `sym_jsonify`. If None, `cls.__type_name__` will be
      used. This is introduced for scenarios when we want to relocate a class,
      before the downstream can recognize the new location, we need the class to
      serialize it using previous key.
    additional_keys: An optional list of strings as additional keys to
      deserialize an object of the registered class. This can be useful when we
      need to relocate or rename the registered class while being able to load
      existing serialized JSON values.
    override: Additional class attributes to override.

  Returns:
    A subclass of `cls` and `ClassWrapper`.

  Raises:
    TypeError: input `cls` is not a class.
  """
  if not inspect.isclass(cls):
    raise TypeError(f'Class wrapper can only be created from classes. '
                    f'Encountered: {cls!r}.')

  if not issubclass(cls, ClassWrapper):
    cls = _subclassed_wrapper(
        cls,
        use_symbolic_repr=repr,
        use_symbolic_comp=eq,
        reset_state_fn=reset_state_fn,
        class_name=class_name,
        module_name=module_name,
        use_auto_doc=auto_doc,
        use_auto_typing=auto_typing)

  if issubclass(cls, ClassWrapper):
    # Update init argument specifications according to user specified specs.
    # Replace schema instead of extending it.
    init_arg_list, arg_fields = _extract_init_signature(
        cls, init_args, auto_doc=auto_doc, auto_typing=auto_typing)
    cls.update_schema(arg_fields, init_arg_list=init_arg_list, extend=False)
    cls.register_for_deserialization(serialization_key, additional_keys)

  if override:
    for k, v in override.items():
      setattr(cls, k, v)
  return cls


def wrap_module(
    module,
    names: Optional[Sequence[str]] = None,
    where: Optional[Callable[[Type['ClassWrapper']], bool]] = None,
    export_to: Optional[types.ModuleType] = None,
    **kwargs):
  """Wrap classes from a module.

  For example, users can wrap all subclasses of `xxx.Base` under module `xxx`::

    import xxx

    pg.wrap_module(
      xxx, where=lambda c: isinstance(c, xxx.Base))

  Args:
    module: A container that contains classes to wrap.
    names: An optional list of class names. If not provided, all classes under
      `module` will be considered candidates.
    where: An optional filter function in signature (user_class) -> bool.
      Only the classes under `module` with True return value will be wrapped.
    export_to: An optional module to export the wrapper classes.
    **kwargs: Keyword arguments passed to `wrap`

  Returns:
    Wrapper classes.
  """
  wrapper_classes = []
  module_name = export_to.__name__ if export_to else None
  origin_cls_to_wrap_cls = {}
  for symbol_name in (names or dir(module)):
    s = getattr(module, symbol_name)
    if inspect.isclass(s) and (not where or where(s)):
      # NOTE(daiyip): It's possible that a name under a module is an alias for
      # another class. In such case, we do not create duplicated wrappers but
      # shares the same wrapper classes with different names.
      if s in origin_cls_to_wrap_cls:
        wrapper_class = origin_cls_to_wrap_cls[s]
      else:
        wrapper_class = wrap(s, module_name=module_name, **kwargs)
        origin_cls_to_wrap_cls[s] = wrapper_class
        wrapper_classes.append(wrapper_class)
      if export_to:
        setattr(export_to, symbol_name, wrapper_class)
  return wrapper_classes


def apply_wrappers(
    wrapper_classes: Optional[Sequence[Type['ClassWrapper']]] = None,
    where: Optional[Callable[[Type['ClassWrapper']], bool]] = None):
  """Context manager for swapping user classes with their class wrappers.

  This helper method is a handy tool to swap user classes with their wrappers
  within a code block, without modifying exisiting code.

  For example::

    def foo():
      return A()

    APrime = pg.wrap(A)

    with pg.apply_wrappers([APrime]):
      # Direct creation of an instance of `A` will be detoured to `APrime`.
      assert isinstance(A(), APrime)

      Indirect creation of an instance of `A` will be detoured too.
      assert isinstance(foo(), APrime)

    # Out of the scope, direct/indirect creation `of` A will be restored.
    assert not isinstance(A(), APrime)
    assert not isinstance(foo(), APrime)

  ``pg.apply_wrappers`` can be nested, under which the inner context will apply
  the wrappers from the outter context. ``pg.apply_wrappers`` is NOT
  thread-safe.

  Args:
    wrapper_classes: Wrapper classes to use. If None, sets it to all registered
      wrapper classes.
    where: An optional filter function in signature (wrapper_class) -> bool.
      If not None, only filtered `wrapper_class` will be swapped.

  Returns:
    A context manager that detours the original classes to the wrapper classes.
  """
  if not wrapper_classes:
    wrapper_classes = []
    for _, c in object_utils.JSONConvertible.registered_types():
      if (issubclass(c, ClassWrapper)
          and c not in (ClassWrapper, _SubclassedWrapperBase)
          and (not where or where(c))
          and c.sym_wrapped_cls is not None):
        wrapper_classes.append(c)
  return detouring.detour([(c.sym_wrapped_cls, c) for c in wrapper_classes])


def _extract_init_signature(
    cls,
    arg_specs=None,
    auto_doc: bool = False,
    auto_typing: bool = False
) -> Tuple[List[str], List[pg_typing.Field]]:
  """Extract argument fields from class __init__ method."""
  init_method = getattr(cls, '__orig_init__', cls.__init__)
  docstr = None
  if auto_doc:
    # Read args docstr from both class doc string and __init__ doc string.
    args_docstr = dict()
    if cls.__doc__:
      cls_docstr = object_utils.DocStr.parse(cls.__doc__)
      args_docstr = cls_docstr.args
    if init_method.__doc__:
      init_docstr = object_utils.DocStr.parse(init_method.__doc__)
      args_docstr.update(init_docstr.args)
    docstr = object_utils.DocStr(
        object_utils.DocStrStyle.GOOGLE,
        short_description=None,
        long_description=None,
        examples=[],
        args=args_docstr,
        returns=None,
        raises=[],
        blank_after_short_description=True,
    )
  if init_method is object.__init__:
    if arg_specs:
      raise ValueError(
          f'{cls.__name__}.__init__ takes no argument while non-empty `args` '
          f'is provided: {arg_specs}')
    init_arg_list = []
    arg_fields = []
  else:
    signature = pg_typing.Signature.from_signature(
        inspect.signature(init_method),
        name=cls.__name__,
        callable_type=pg_typing.CallableType.METHOD,
        module_name=cls.__module__,
        qualname=cls.__qualname__,
        auto_typing=auto_typing,
        docstr=docstr,
    ).annotate(arg_specs)
    if not signature.args or signature.args[0].name != 'self':
      raise ValueError(
          f'{cls.__name__}.__init__ must have `self` as the first argument.')
    # Remove field for 'self'.
    arg_fields = signature.fields(remove_self=True)
    init_arg_list = [arg.name for arg in signature.args[1:]]
    if signature.varargs is not None:
      init_arg_list.append(f'*{signature.varargs.name}')
  return (init_arg_list, arg_fields)
