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
"""Tests for pyglove.symbolic.class_wrapper."""

import copy
import dataclasses
import inspect
import unittest

from pyglove.core import detouring as pg_detouring
from pyglove.core import typing as pg_typing

from pyglove.core.symbolic.base import eq as pg_eq
from pyglove.core.symbolic.base import ne as pg_ne
from pyglove.core.symbolic.base import query as pg_query
from pyglove.core.symbolic.base import sym_hash as pg_hash
from pyglove.core.symbolic.class_wrapper import apply_wrappers as pg_apply_wrappers
from pyglove.core.symbolic.class_wrapper import ClassWrapper
from pyglove.core.symbolic.class_wrapper import wrap as pg_wrap
from pyglove.core.symbolic.class_wrapper import wrap_module as pg_wrap_module
from pyglove.core.symbolic.pure_symbolic import NonDeterministic


class WrapTest(unittest.TestCase):
  """Tests for `pg.wrap`."""

  def test_wrap_with_super_init(self):

    class A:
      def __init__(self):
        self.y = 0

    class B(A):

      def __init__(self, x):
        super().__init__()
        self.x = x

    B1 = pg_wrap(B)  # pylint: disable=invalid-name
    b = B1(1)
    self.assertIsInstance(b, B)
    self.assertIsInstance(b, ClassWrapper)
    self.assertEqual(b.x, 1)
    self.assertEqual(b.y, 0)

  def test_wrap_with_varargs_and_kwargs(self):

    class A:

      def __init__(self, x, *args, y=1, **kwargs):
        self.x = x
        self.varargs = args
        self.y = y
        self.kw = kwargs

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    a = A1(0, 1, 2, y=3, p=4, q=5)
    self.assertEqual(a.x, 0)
    self.assertEqual(a.varargs, (1, 2))
    self.assertEqual(a.y, 3)
    self.assertEqual(a.kw, {'p': 4, 'q': 5})

    self.assertEqual(a.rebind({'args[0]': 0}).varargs, (0, 2))
    self.assertEqual(
        a.rebind({'p': 5, 'q': 6, 'r': 7}).kw,
        {'p': 5, 'q': 6, 'r': 7})

  def test_wrap_with_no_typing(self):

    class A:

      def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = x + y

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    a = A1(x=1, y=1)
    self.assertIsInstance(a, ClassWrapper)
    self.assertIsInstance(a, A)
    self.assertEqual(inspect.getfullargspec(A).args, ['self', 'x', 'y'])

    # Test equality semantics doesn't change.
    self.assertNotEqual(a, A1(1, 1))
    self.assertTrue(a, a.sym_eq(A1(1, 1)))
    self.assertNotEqual(hash(a), hash(A1(1, 1)))
    self.assertEqual(pg_hash(a), pg_hash(A1(1, 1)))

    self.assertEqual(a.x, 1)
    self.assertEqual(a.z, 2)
    a.rebind(y=2)
    self.assertEqual(a.z, 3)

    class B:    # pylint: disable=unused-variable

      def __init__(unused_x):  # pylint: disable=no-self-argument
        pass

    with self.assertRaisesRegex(
        ValueError, '.*__init__ must have `self` as the first argument'):
      _ = pg_wrap(B)

  def test_wrap_with_auto_typing(self):

    class A:

      def __init__(self, x: int, y: int = 1, **kwargs):
        """Class A.

        Args:
          x: The first integer.
          y: The second integer.
          **kwargs: Other arguments.
        """
        self.z = x + y + sum(kwargs.values())

    A1 = pg_wrap(A, auto_typing=True)  # pylint: disable=invalid-name
    self.assertEqual(A1.__schema__.get_field('x').value, pg_typing.Int())
    self.assertEqual(
        A1.__schema__.get_field('y').value, pg_typing.Int(default=1)
    )

  def test_wrap_with_user_typing(self):

    class B:

      def __init__(self, x, y):
        self.z = x + y

    class C(B):
      pass

    C1 = pg_wrap(C, [('x', pg_typing.Int())])  # pylint: disable=invalid-name
    self.assertEqual(inspect.getfullargspec(C1).args, ['self', 'x', 'y'])
    self.assertIn('x', C1.__schema__)
    self.assertIn('y', C1.__schema__)
    self.assertNotIn('z', C1.__schema__)

    c = C1(1, 2)
    self.assertIsInstance(c, ClassWrapper)
    self.assertIsInstance(c, B)
    self.assertFalse(hasattr(c, 'x'))
    self.assertFalse(hasattr(c, 'y'))

    with self.assertRaisesRegex(KeyError, 'Key .* is not allowed'):
      c.rebind(z=4)

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      C1(0.5, 1)

  def test_wrap_with_user_typing_for_kwargs(self):
    # When `__init__` accepts `**kwargs`, additional arugment names (e.g: `p`
    # and `q`) can be defined as symbolic arguments.

    class A:

      def __init__(self, x, y=1, **kwargs):
        self.z = x + y + sum(kwargs.values())

    A1 = pg_wrap(A, [   # pylint: disable=invalid-name
        ('p', pg_typing.Int()),
        ('q', pg_typing.Int()),
    ])
    a = A1(1, p=2, q=3)
    self.assertEqual(a.z, 7)   # 1 + 1 + 2 + 3
    self.assertNotIn('z', a.sym_init_args)

  def test_wrap_with_auto_doc(self):

    class A:
      """A test class."""

      def __init__(self, x, y):
        """Init method.

        Args:
          x: Argument x.
          y: Argument y.
        """
        self.x = x
        self.y = y

    A1 = pg_wrap(A, auto_doc=True)  # pylint: disable=invalid-name
    self.assertEqual(A1.__schema__.description, 'A test class.')
    self.assertEqual(
        list(A1.__schema__.fields.values()),
        [
            pg_typing.Field('x', pg_typing.Any(), 'Argument x.'),
            pg_typing.Field('y', pg_typing.Any(), 'Argument y.'),
        ],
    )

  def test_wrap_dataclass_with_auto_doc(self):

    @dataclasses.dataclass
    class A:
      """A test class.

      Attributes:
        x: Argument x.
        y: Argument y.
      """
      x: int
      y: str

    A1 = pg_wrap(A, auto_doc=True)  # pylint: disable=invalid-name
    self.assertEqual(A1.__schema__.description, 'A test class.')
    self.assertEqual(
        list(A1.__schema__.fields.values()),
        [
            pg_typing.Field('x', pg_typing.Any(annotation=int), 'Argument x.'),
            pg_typing.Field('y', pg_typing.Any(annotation=str), 'Argument y.'),
        ],
    )

  def test_automatic_reset_state(self):

    class A:

      def __init__(self, v):
        if hasattr(self, 'x'):
          self.y = v
        else:
          self.x = v

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    a = A1(1)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.sym_init_args.v, 1)
    self.assertFalse(hasattr(a, 'y'))
    a.rebind(v=2)
    self.assertEqual(a.x, 2)
    self.assertEqual(a.sym_init_args.v, 2)
    self.assertFalse(hasattr(a, 'y'))

  def test_wrap_with_reset_fn(self):
    context = dict(id=0)

    class A:

      def __init__(self, x):
        self.state = {context['id']: x}
        context['id'] += 1

    def reset_state_fn(unused_obj):
      context['id'] = 0

    a1 = A(1)
    self.assertEqual(a1.state, {0: 1})
    a2 = A(2)
    self.assertEqual(a2.state, {1: 2})

    A1 = pg_wrap(A, reset_state_fn=reset_state_fn)  # pylint: disable=invalid-name
    a1 = A1(1)
    self.assertEqual(a1.state, {0: 1})
    a2 = A1(2)
    self.assertEqual(a2.state, {0: 2})

  def test_wrap_with_symbolic_repr(self):

    class A:
      pass

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    self.assertEqual(repr(A1()), 'A()')

    # When the source class does not have __repr__, symbolic
    # repr will be used even when `repr` is set to False.
    A2 = pg_wrap(A, repr=False)  # pylint: disable=invalid-name
    self.assertRegex(repr(A2()), '.* object at 0x.*')

    class B:
      def __repr__(self):
        return 'custom_repr'

    # When the source class has __repr__, symbolic
    # repr will not be used even when `repr` is set to True.
    B1 = pg_wrap(B)  # pylint: disable=invalid-name
    self.assertEqual(repr(B1()), 'B()')

    B2 = pg_wrap(B, repr=False)  # pylint: disable=invalid-name
    self.assertEqual(repr(B2()), 'custom_repr')

  def test_wrap_with_symbolic_eq(self):

    class A:
      pass

    # By default argument `eq` equals False.
    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    self.assertNotEqual(A1(), A1())
    self.assertTrue(pg_eq(A1(), A1()))

    A2 = pg_wrap(A, eq=True)  # pylint: disable=invalid-name
    self.assertEqual(A2(), A2())
    self.assertTrue(pg_eq(A2(), A2()))

    class B:

      def __eq__(self, other):
        return False

    B1 = pg_wrap(B, eq=True)  # pylint: disable=invalid-name
    self.assertNotEqual(B1(), B1())

  def test_bad_wrap(self):
    with self.assertRaisesRegex(
        TypeError, 'Class wrapper can only be created from classes'):
      pg_wrap('abc')

    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):

      class A:

        def __init__(self, x, y=1):
          self.z = x + y

      _ = pg_wrap(A, [('y', pg_typing.Bool())])

    with self.assertRaisesRegex(
        KeyError, 'found extra symbolic argument \'z\''):

      class B:

        def __init__(self, x, y=1):
          self.z = x + y

      _ = pg_wrap(B, [('z', pg_typing.Int())])


class WrapModuleTest(unittest.TestCase):
  """Tests for `pg.wrap_module`."""

  def setUp(self):
    super().setUp()

    class A:
      pass

    class B:
      pass

    class D(A):
      pass

    class Module:
      """Fake module."""

      def __init__(self, name, **kwargs):
        self.__name__ = name
        self._syms = kwargs

      def __dir__(self):
        return sorted(self._syms.keys())

      def __getattr__(self, name):
        return self._syms[name]

    self._A = A  # pylint: disable=invalid-name
    self._B = B  # pylint: disable=invalid-name
    self._D = D  # pylint: disable=invalid-name
    self._Module = Module  # pylint: disable=invalid-name
    self._src_module = Module('source_module', A=A, B=B, C=A, D=D)

  def test_wrap_all(self):
    wrapper_classes = pg_wrap_module(self._src_module)
    self.assertEqual(len(wrapper_classes), 3)
    self.assertIsInstance(wrapper_classes[0](), self._A)
    self.assertIsInstance(wrapper_classes[0](), ClassWrapper)
    self.assertIsInstance(wrapper_classes[1](), self._B)
    self.assertIsInstance(wrapper_classes[1](), ClassWrapper)
    self.assertIsInstance(wrapper_classes[2](), self._D)
    self.assertIsInstance(wrapper_classes[2](), ClassWrapper)

  def test_manual_selected_classes(self):
    wrapper_classes = pg_wrap_module(self._src_module, ['B'])
    self.assertEqual(len(wrapper_classes), 1)
    self.assertIsInstance(wrapper_classes[0](), self._B)
    self.assertIsInstance(wrapper_classes[0](), ClassWrapper)
    self.assertEqual(wrapper_classes[0].__module__, 'pyglove.core.symbolic.class_wrapper_test')

  def test_where(self):
    wrapper_classes = pg_wrap_module(
        self._src_module, where=lambda c: issubclass(c, self._A))

    self.assertEqual(len(wrapper_classes), 2)
    self.assertIsInstance(wrapper_classes[0](), self._A)
    self.assertIsInstance(wrapper_classes[0](), ClassWrapper)
    self.assertIsInstance(wrapper_classes[1](), self._D)
    self.assertIsInstance(wrapper_classes[1](), ClassWrapper)

  def test_export_to_module(self):
    target_module = self._Module('target_module')
    wrapper_classes = pg_wrap_module(
        self._src_module, ['A', 'B'], export_to=target_module)
    self.assertIs(target_module.A, wrapper_classes[0])
    self.assertIs(target_module.B, wrapper_classes[1])
    self.assertEqual(target_module.A.__module__, 'target_module')


class ApplyWrappersTest(unittest.TestCase):
  """Tests for `apply_wrappers` context manager."""

  def test_apply_wrappers(self):
    """Test `apply_wrappers`."""

    class Base:
      pass

    class A:

      def __init__(self, x):
        self.x = x

    class B(Base):
      pass

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    B1 = pg_wrap(B)  # pylint: disable=invalid-name

    # Test apply wrappers that are explicitly passed in.
    with pg_apply_wrappers([A1]):
      self.assertIsInstance(A(1), A1)
      self.assertNotIsInstance(B(), B1)
    self.assertNotIsInstance(A(1), A1)
    self.assertEqual(A(1).x, 1)

    # Test apply wrappers on all registered wrapper classes with where clause.
    with pg_apply_wrappers(where=lambda c: issubclass(c, Base)):
      self.assertNotIsInstance(A(1), A1)
      self.assertIsInstance(B(), B1)
    self.assertNotIsInstance(B(), B1)

    # Nested apply_wrappers.
    with pg_apply_wrappers([A1]):
      self.assertIsInstance(A(1), A1)
      self.assertNotIsInstance(B(), B1)
      with pg_apply_wrappers([B1]):
        self.assertIsInstance(A(1), A1)
        self.assertIsInstance(B(), B1)
      self.assertNotIsInstance(B(), B1)
    self.assertNotIsInstance(A(1), A1)

    # Test for user class with custom __new__.
    class C:

      def __new__(cls, *args, **kwargs):
        return super(C, cls).__new__(cls, *args, **kwargs)

    C1 = pg_wrap(C)  # pylint: disable=invalid-name
    with pg_apply_wrappers([C1]):
      c = C()
      self.assertIsInstance(c, C1)


class ClassWrapperTest(unittest.TestCase):
  """Tests for class wrapper."""

  def test_wrapper_for_baseless_class(self):

    class A:
      pass

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    self.assertTrue(pg_eq(A1(), A1()))
    self.assertEqual(repr(A1), f'Symbolic[{A1.sym_wrapped_cls!r}]')
    a = A1()
    self.assertIs(a.sym_wrapped, a)
    with self.assertRaisesRegex(
        ValueError, '.* takes no argument while non-empty `args` is provided'):
      _ = pg_wrap(A, [('x', pg_typing.Int())])

    class B:

      def __init__(self, x):
        self.x = x
        self.y = x + 1

    B1 = pg_wrap(B)  # pylint: disable=invalid-name
    b = B1(1)
    self.assertTrue(pg_eq(b, B1(1)))
    self.assertEqual(b.y, 2)

  def test_wrapper_for_class_with_single_inheritance(self):
    call_count = dict(a=0, c=0)

    class A:

      def __init__(self):
        self.a = 0
        call_count['a'] += 1

    class B(A):
      pass

    B1 = pg_wrap(B)  # pylint: disable=invalid-name
    b = B1()
    self.assertTrue(pg_eq(b, B1()))
    self.assertEqual(b.a, 0)
    self.assertEqual(call_count['a'], 2)

    class C(B1):

      def __init__(self, x):
        super().__init__()
        self.c = x
        call_count['c'] += 1

    c = C(1)
    self.assertTrue(pg_eq(c, C(1)))
    # Make sure parent's __init__ is invoked.
    self.assertEqual(c.a, 0)
    # Make sure self's __init__ is invoked.
    self.assertEqual(c.c, 1)
    self.assertEqual(call_count['a'], 4)
    self.assertEqual(call_count['c'], 2)

    class D(C):
      pass

    d = D(1)
    self.assertTrue(pg_eq(d, D(1)))
    self.assertTrue(pg_ne(d, C(1)))
    # Make sure parent's __init__ is invoked.
    self.assertEqual(d.a, 0)
    # Make sure parent's __init__ is invoked.
    self.assertEqual(d.c, 1)
    self.assertEqual(call_count['a'], 7)
    self.assertEqual(call_count['c'], 5)

  def test_wrapper_for_class_with_multi_inheritance(self):

    class A:

      def __init__(self, x):
        super().__init__(x)
        self.a = x

    A1 = pg_wrap(A)  # pylint: disable=invalid-name

    class B:

      def __init__(self, x):
        super().__init__(x)
        self.b = x

    B1 = pg_wrap(B)  # pylint: disable=invalid-name

    class C(A1, B1):

      def __init__(self, x):
        super().__init__(x)
        self.c = x

    c = C(1)
    self.assertEqual(c.a, 1)
    self.assertEqual(c.b, 1)
    self.assertEqual(c.c, 1)

  def test_automatic_wrap_from_inheritance(self):

    class A:

      def __init__(self, x):
        self.x = x

    A1 = pg_wrap(A)  # pylint: disable=invalid-name

    # Automatic wrapped by inheriting a class wrapper.
    class C(A1):

      def __init__(self, x, y):
        super().__init__(x)
        self.y = y

    self.assertIsInstance(C(1, 2), ClassWrapper)
    self.assertTrue(pg_eq(C(1, 2), C(1, 2)))
    self.assertEqual(list(C.__schema__.fields.keys()), ['x', 'y'])
    self.assertEqual(repr(C), f'<class {C.type_name!r}>')

  def test_custom_metaclass(self):

    class CustomMeta(type):

      @property
      def foo(cls):
        return 'foo'

    class A(metaclass=CustomMeta):
      pass

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    self.assertTrue(issubclass(A1, ClassWrapper))
    self.assertTrue(issubclass(A1, A))
    self.assertEqual(A1.type_name, 'pyglove.core.symbolic.class_wrapper_test.A')
    self.assertEqual(A1.__schema__, pg_typing.Schema([]))
    self.assertEqual(A1.foo, 'foo')
    self.assertRegex(repr(A1), r'Symbolic\[.*\]')

  def test_custom_setattr(self):

    class A:

      def __init__(self, x):
        self.x = x

      def __setattr__(self, name, value):
        super().__setattr__(f'shadow_{name}', value)

    A1 = pg_wrap(A)  # pylint: disable=invalid-name
    a = A1(1)
    self.assertEqual(a.shadow_x, 1)
    a.rebind(x=2)
    self.assertEqual(a.shadow_x, 2)

  def test_method_clashes(self):

    class A:

      def __init__(self, x):
        self.y = x

      def get_result(self):
        return self.y_y

      def __contains__(self, key):
        return key == 'y'

      def __iter__(self):
        yield 'y'

      def keys(self):
        return ['y']

      def __getattr__(self, key):
        raise AttributeError('No extra attribute allowed.')

      def __setattr__(self, key, value):
        super().__setattr__('y_' + key, value)

      def __eq__(self, other):
        return isinstance(other, int) and other == 1

      def __ne__(self, other):
        return not self.__eq__(other)

      def __hash__(self):
        return 0

      def __copy__(self):
        return 1

      def __deepcopy__(self, memo):
        return 2

      def __repr__(self):
        return 'foo'

      def __str__(self):
        return 'bar'

    A1 = pg_wrap(A, eq=True)  # pylint: disable=invalid-name
    a = A1({'p': 1})

    # Test A1's non-symbolic members are inherited from user class.
    self.assertEqual(a.get_result(), {'p': 1})
    a.m = 0
    self.assertEqual(a.y_m, 0)
    self.assertNotIn('x', a)
    self.assertEqual(a.keys(), ['y'])
    self.assertEqual(list(iter(a)), ['y'])
    with self.assertRaisesRegex(
        AttributeError, 'No extra attribute allowed'):
      _ = a.x

    # Test A1's symbolic members are inherited from ClassWrapper.
    # Since class A has defined its own __eq__, __ne__ and __hash__,
    # Even `eq` is set to True, these methods from the
    # the user class will be used.
    self.assertNotEqual(a, A1({'p': 1}))
    self.assertTrue(a.sym_eq(A1({'p': 1})))
    self.assertFalse(a.sym_eq(A1({'p': 2})))
    self.assertEqual(hash(a), 0)
    self.assertNotEqual(pg_hash(a), 0)
    self.assertEqual(copy.copy(a), 1)
    self.assertEqual(copy.deepcopy(a), 2)
    self.assertTrue(a.sym_eq(a.clone(deep=True)))
    self.assertTrue(a.sym_eq(a.clone(deep=False)))
    self.assertEqual(a.sym_init_args, dict(x=dict(p=1)))
    self.assertEqual(pg_query(A1({'p': 2}), where=lambda v: v == 2),
                     {'x.p': 2})
    self.assertEqual(str(a), inspect.cleandoc('''
        A(
          x = {
            p = 1
          }
        )
        '''))
    self.assertEqual(repr(a), 'A(x={p=1})')

    A2 = pg_wrap(  # pylint: disable=invalid-name
        A,
        repr=False,
        override=dict(get_result=lambda self: 'overridden results'))
    a = A2({'p': 1})

    # Test get_results is overriden.
    self.assertEqual(a.get_result(), 'overridden results')

    # Test A2's symbolic members are not inherited from ClassWrapper.
    self.assertEqual(a, 1)
    self.assertFalse(pg_eq(a, 1))
    self.assertTrue(pg_eq(a, A2({'p': 1})))
    self.assertNotEqual(a, 2)
    self.assertTrue(pg_ne(a, None))
    self.assertFalse(pg_ne(a, A2({'p': 1})))
    self.assertEqual(hash(a), 0)
    self.assertEqual(repr(a), 'foo')
    self.assertEqual(str(a), 'bar')
    self.assertEqual(copy.copy(a), 1)
    self.assertEqual(copy.deepcopy(a), 2)
    self.assertTrue(pg_eq(a, a.clone(deep=True)))
    self.assertTrue(pg_eq(a, a.clone(deep=False)))

  def test_delayed_init_if_nondeterministic(self):

    class A(NonDeterministic):
      pass

    class B:

      def __init__(self, a):
        self._a = a

    B1 = pg_wrap(B)  # pylint: disable=invalid-name
    # `B.__init__` shall not be called when it is partial.
    b = B1.partial()
    self.assertFalse(b.wrapped_cls_initialized)

    # `B.__init__` shall not be called when any of its
    # arguments has non-deterministic values.
    b = B1(A())
    self.assertFalse(b.wrapped_cls_initialized)

    # After rebinding `b.a` to a fixed value, `B.__init__`
    # will be automatically triggered.
    b.rebind(a=1)
    self.assertTrue(b.wrapped_cls_initialized)

    # When `b.a` is rebound to a non-deterministic value again,
    # `b` should be regarded as uninitialized again.
    b.rebind(a=A())
    self.assertFalse(b.wrapped_cls_initialized)

  def test_detour_with_classwrapper(self):

    class A:

      def __init__(self, x):
        self.x = x

    A1 = pg_wrap(A)    # pylint: disable=invalid-name

    def fun(unused_cls, x):
      return A1(x + 1)

    with pg_detouring.detour([(A, fun)]):
      a = A(1)
    self.assertIsInstance(a, A1)

    # Since A1 is a subclass of A, A1.__init__ will be called with 1 by
    # the Python runtime after `fun` is called, however it will not override
    # existing value which is 2.
    self.assertEqual(a.x, 2)


if __name__ == '__main__':
  unittest.main()
