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
"""Tests for pyglove.symbolic."""

import copy
import inspect
import unittest

from pyglove.core import symbolic
from pyglove.core import typing
from pyglove.core import wrapping

symbolic.allow_empty_field_description()
symbolic.allow_repeated_class_registration()


class WrappingTest(unittest.TestCase):
  """Tests for common wrapping methods."""

  def testInvalidWrap(self):
    """Test class_wrapper basics."""
    with self.assertRaisesRegex(
        TypeError, 'Class wrapper can only be created from classes'):
      wrapping.wrap('abc')

  def testWrapWithSuperInit(self):
    """Test symbolizing class which calls super.__init__."""

    class A:
      pass

    @wrapping.symbolize
    class B(A):

      def __init__(self, x):
        super().__init__()
        self.x = x

    _ = B(1)

  def testWrapAutomaticSchematization(self):
    """Test automatic schematization."""

    @wrapping.symbolize
    class A:

      def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = x + y

    a = A(x=1, y=1)
    self.assertIsInstance(a, symbolic.Object)
    self.assertEqual(inspect.getfullargspec(A).args, ['self', 'x', 'y'])

    # Test equality semantics doesn't change.
    self.assertNotEqual(a, A(1, 1))
    self.assertTrue(a, a.sym_eq(A(1, 1)))
    self.assertNotEqual(hash(a), hash(A(1, 1)))
    self.assertEqual(symbolic.sym_hash(a), symbolic.sym_hash(A(1, 1)))

    self.assertEqual(a.x, 1)
    self.assertEqual(a.z, 2)
    a.rebind(y=2)
    self.assertEqual(a.z, 3)

    with self.assertRaisesRegex(
        ValueError, '.*__init__ must have `self` as the first argument'):

      @wrapping.symbolize
      class B:    # pylint: disable=unused-variable

        def __init__(unused_x):  # pylint: disable=no-self-argument
          pass

  def testWrapWithCustomSchematization(self):
    """Test custom schematization."""

    class B:

      def __init__(self, x, y):
        self.z = x + y

    @wrapping.symbolize([
        ('x', typing.Int()),
    ])
    class C(B):
      pass

    self.assertEqual(inspect.getfullargspec(C).args, ['self', 'x', 'y'])
    self.assertIn('x', C.schema)
    self.assertIn('y', C.schema)
    self.assertNotIn('z', C.schema)

    c = C(1, 2)
    self.assertIsInstance(c, wrapping.ClassWrapper)
    self.assertIsInstance(c, B)
    self.assertFalse(hasattr(c, 'x'))
    self.assertFalse(hasattr(c, 'y'))

    with self.assertRaisesRegex(KeyError, 'Key .* is not allowed'):
      c.rebind(z=4)

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      C(0.5, 1)

    # `symbolize` on a symbolic class will return itself.
    self.assertIs(wrapping.symbolize(C), C)

    # When `__init__` accepts `**kwargs`, additional arugment names (e.g: `p`
    # and `q`) can be defined as symbolic arguments.
    @wrapping.symbolize([
        ('p', typing.Int()),
        ('q', typing.Int()),
    ])
    class C1(C):

      def __init__(self, x, y=1, **kwargs):
        super().__init__(x, y)
        self.z += sum(kwargs.values())

    c = C1(1, p=2, q=3)
    self.assertEqual(c.z, 7)   # 1 + 1 + 2 + 3
    self.assertNotIn('z', c.sym_init_args)

    # Bad case 1: the default value and value specification is incompatible.
    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):

      @wrapping.symbolize([
          ('y', typing.Bool()),
      ])
      class C2:  # pylint: disable=unused-variable

        def __init__(self, x, y=1):
          self.z = x + y

    # Bad case 2: additional key defined as symbolic members but not feedable
    # to `__init__` method.
    with self.assertRaisesRegex(
        KeyError, 'found extra symbolic argument \'z\''):

      @wrapping.symbolize([
          ('z', typing.Int()),
      ])
      class C3:  # pylint: disable=unused-variable

        def __init__(self, x, y=1):
          self.z = x + y

  def testWrapWithWildcardArgs(self):
    """Test wrap with wildcard arguments."""

    @wrapping.symbolize
    class C5:

      def __init__(self, x, *args, y=1, **kwargs):
        self.x = x
        self.varargs = args
        self.y = y
        self.kw = kwargs

    c = C5(0, 1, 2, y=3, p=4, q=5)
    self.assertEqual(c.x, 0)
    self.assertEqual(c.varargs, (1, 2))
    self.assertEqual(c.y, 3)
    self.assertEqual(c.kw, {'p': 4, 'q': 5})

    self.assertEqual(c.rebind({'args[0]': 0}).varargs, (0, 2))
    self.assertEqual(
        c.rebind({'p': 5, 'q': 6, 'r': 7}).kw,
        {'p': 5, 'q': 6, 'r': 7})

  def testSymbolize(self):
    """Test symbolize."""

    class C:
      pass

    C1 = wrapping.symbolize(C)  # pylint: disable=invalid-name
    self.assertIsInstance(C1(), C)
    self.assertIsInstance(C1(), symbolic.Object)

    self.assertIs(wrapping.symbolize(dict), symbolic.Dict)
    self.assertIs(wrapping.symbolize(list), symbolic.List)

    with self.assertRaisesRegex(
        ValueError, 'Constraints are not supported in symbolic .*'):
      wrapping.symbolize(dict, [(typing.StrKey(), typing.Int())])

    with self.assertRaisesRegex(
        ValueError, 'Constraints are not supported in symbolic .*'):
      wrapping.symbolize(list, [(typing.Int())])

    with self.assertRaisesRegex(
        TypeError, '.* cannot be symbolized.'):
      wrapping.symbolize(dict())

    with self.assertRaisesRegex(
        ValueError,
        'Only `constraint` is supported as positional arguments.*'):
      wrapping.symbolize(C, 1, 2)

    class D(symbolic.Object):
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot symbolize .*'):
      wrapping.symbolize(D)

  def testWrapModule(self):
    """Test wrap_module."""

    class A:
      pass

    class B:
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

    src_module = Module('source_module', A=A, B=B, C=A)
    target_module = Module('target_module')

    # Wrap all classe from module.
    wrapper_classes = wrapping.wrap_module(src_module)
    self.assertEqual(len(wrapper_classes), 2)
    self.assertIsInstance(wrapper_classes[0](), A)
    self.assertIsInstance(wrapper_classes[0](), wrapping.ClassWrapper)
    self.assertIsInstance(wrapper_classes[1](), B)
    self.assertIsInstance(wrapper_classes[1](), wrapping.ClassWrapper)

    # Wrap selected classes from module via names.
    wrapper_classes = wrapping.wrap_module(src_module, ['B'])
    self.assertEqual(len(wrapper_classes), 1)
    self.assertIsInstance(wrapper_classes[0](), B)
    self.assertIsInstance(wrapper_classes[0](), wrapping.ClassWrapper)
    self.assertEqual(wrapper_classes[0].__module__, 'pyglove.core.wrapping_test')

    # Wrap selected classes from module via `where` argument and export it
    # to the target module.
    wrapper_classes = wrapping.wrap_module(
        src_module, where=lambda c: issubclass(c, A), export_to=target_module)

    self.assertEqual(len(wrapper_classes), 1)
    self.assertIsInstance(wrapper_classes[0](), A)
    self.assertIsInstance(wrapper_classes[0](), wrapping.ClassWrapper)
    self.assertIs(target_module.A, wrapper_classes[0])
    self.assertIs(target_module.C, wrapper_classes[0])
    self.assertEqual(wrapper_classes[0].__module__, 'target_module')

  def testApplyWrappers(self):
    """Test `apply_wrappers`."""

    class Base:
      pass

    class A:

      def __init__(self, x):
        self.x = x

    class B(Base):
      pass

    A1 = wrapping.wrap(A)  # pylint: disable=invalid-name
    B1 = wrapping.wrap(B)  # pylint: disable=invalid-name

    # Test apply wrappers that are explicitly passed in.
    with wrapping.apply_wrappers([A1]):
      self.assertIsInstance(A(1), A1)
      self.assertNotIsInstance(B(), B1)
    self.assertNotIsInstance(A(1), A1)
    self.assertEqual(A(1).x, 1)

    # Test apply wrappers on all registered wrapper classes with where clause.
    with wrapping.apply_wrappers(where=lambda c: issubclass(c, Base)):
      self.assertNotIsInstance(A(1), A1)
      self.assertIsInstance(B(), B1)
    self.assertNotIsInstance(B(), B1)

    # Nested apply_wrappers.
    with wrapping.apply_wrappers([A1]):
      self.assertIsInstance(A(1), A1)
      self.assertNotIsInstance(B(), B1)
      with wrapping.apply_wrappers([B1]):
        self.assertIsInstance(A(1), A1)
        self.assertIsInstance(B(), B1)
      self.assertNotIsInstance(B(), B1)
    self.assertNotIsInstance(A(1), A1)

    # Test for user class with custom __new__.
    class C:

      def __new__(cls, *args, **kwargs):
        return super(C, cls).__new__(cls, *args, **kwargs)

    C1 = wrapping.wrap(C)  # pylint: disable=invalid-name
    with wrapping.apply_wrappers([C1]):
      c = C()
      self.assertIsInstance(c, C1)

  def testSymbolizeFunction(self):
    """Test `symbolize` on function."""

    # Use `symbolize` and function.
    def foo(a, b):
      del a, b
    foo1 = wrapping.symbolize(foo)
    self.assertIsInstance(foo1(1, 2), symbolic.Functor)

    # Use 'symbolize' as functor.
    @wrapping.symbolize
    def bar(a, b):
      del a, b
    self.assertIsInstance(bar(1, 2), symbolic.Functor)

    # use `symbolize with constraints.
    @wrapping.symbolize([
        ('a', typing.Int())
    ], returns=typing.Int())
    def bar2(a):
      return a

    self.assertEqual(bar2(1)(), 1)   # pylint: disable=not-callable
    with self.assertRaisesRegex(
        TypeError, 'Expect .* but encountered .*'):
      bar2('a string value')

  def testSerialization(self):
    """Test serialization."""

    @wrapping.symbolize
    class A:

      def __init__(self, x):
        self.x = x

    a = A(1)
    v = a.to_json()
    self.assertEqual(v['_type'], A.type_name)
    self.assertTrue(symbolic.eq(symbolic.from_json(v), a))

    @wrapping.symbolize(serialization_key='BB', additional_keys=['BBB'])
    class B:

      def __init__(self, x):
        self.x = x

    b = B(1)
    v = b.to_json()
    self.assertEqual(v['_type'], 'BB')

    # Deserilaize with serialization key: OK
    self.assertTrue(symbolic.eq(symbolic.from_json(v), b))

    # Deserialize with type name: OK
    v['_type'] = B.type_name
    self.assertTrue(symbolic.eq(symbolic.from_json(v), b))

    # Deserialize with additional key: OK
    v['_type'] = 'BBB'
    self.assertTrue(symbolic.eq(symbolic.from_json(v), b))

    # Use `symbolize` and function.
    @wrapping.symbolize
    def foo(x, y):
      del x, y

    f = foo(1, 2)
    v = f.to_json()
    self.assertEqual(v['_type'], foo.type_name)
    self.assertEqual(symbolic.from_json(v), f)

    @wrapping.symbolize(serialization_key='BAR', additional_keys=['RRR'])
    def bar(a, b):
      del a, b

    b = bar(3, 4)
    v = b.to_json()
    self.assertEqual(v['_type'], 'BAR')

    # Deserialize with type name: OK
    v['_type'] = bar.type_name
    self.assertTrue(symbolic.eq(symbolic.from_json(v), b))

    # Deserialize with additional key: OK
    v['_type'] = 'RRR'
    self.assertTrue(symbolic.eq(symbolic.from_json(v), b))


class SubclassWrapperTest(unittest.TestCase):
  """Tests for Subclass wrapper."""

  def testNoBaseClass(self):
    """Test symbolize class without base classes."""

    @wrapping.symbolize
    class A:
      pass

    self.assertTrue(symbolic.eq(A(), A()))
    self.assertEqual(repr(A), f'Symbolic[{A.sym_wrapped_cls!r}]')

    @wrapping.symbolize
    class B:

      def __init__(self, x):
        self.x = x
        self.y = x + 1

    b = B(1)
    self.assertTrue(symbolic.eq(b, B(1)))
    self.assertEqual(b.y, 2)

    class C(A):
      pass

    self.assertTrue(symbolic.eq(C(), C()))
    self.assertEqual(repr(C), f'<class {C.type_name!r}>')

  def testSingleInheritance(self):
    """Test single inheritance on symbolized class."""

    call_count = dict(a=0, c=0)

    class A:

      def __init__(self):
        self.a = 0
        call_count['a'] += 1

    @wrapping.symbolize
    class B(A):
      pass

    b = B()
    self.assertTrue(symbolic.eq(b, B()))
    self.assertEqual(b.a, 0)
    self.assertEqual(call_count['a'], 2)

    class C(B):

      def __init__(self, x):
        super().__init__()
        self.c = x
        call_count['c'] += 1

    c = C(1)
    self.assertTrue(symbolic.eq(c, C(1)))
    # Make sure parent's __init__ is invoked.
    self.assertEqual(c.a, 0)
    # Make sure self's __init__ is invoked.
    self.assertEqual(c.c, 1)
    self.assertEqual(call_count['a'], 4)
    self.assertEqual(call_count['c'], 2)

    class D(C):
      pass

    d = D(1)
    self.assertTrue(symbolic.eq(d, D(1)))
    self.assertTrue(symbolic.ne(d, C(1)))
    # Make sure parent's __init__ is invoked.
    self.assertEqual(d.a, 0)
    # Make sure parent's __init__ is invoked.
    self.assertEqual(d.c, 1)
    self.assertEqual(call_count['a'], 7)
    self.assertEqual(call_count['c'], 5)

  def testMultiInheritance(self):
    """Test multi-inheritance on symbolized class."""

    @wrapping.symbolize
    class A:

      def __init__(self, x):
        super().__init__(x)
        self.a = x

    @wrapping.symbolize
    class B:

      def __init__(self, x):
        super().__init__(x)
        self.b = x

    class C(A, B):

      def __init__(self, x):
        super().__init__(x)
        self.c = x

    c = C(1)
    self.assertEqual(c.a, 1)
    self.assertEqual(c.b, 1)
    self.assertEqual(c.c, 1)

  def testCustomMetaclass(self):
    """Test custom metaclass."""

    class CustomMeta(type):

      @property
      def foo(cls):
        return 'foo'

    class A(metaclass=CustomMeta):
      pass

    A1 = wrapping.wrap(A)  # pylint: disable=invalid-name
    self.assertTrue(issubclass(A1, wrapping.ClassWrapper))
    self.assertTrue(issubclass(A1, A))
    self.assertEqual(A1.type_name, 'pyglove.core.wrapping_test.A')
    self.assertEqual(A1.schema, typing.Schema([]))
    self.assertEqual(A1.foo, 'foo')
    self.assertRegex(repr(A1), r'Symbolic\[.*\]')

  def testCustomSetAttr(self):
    """Test custom __setattr__ from user class."""

    class A:

      def __init__(self, x):
        self.x = x

      def __setattr__(self, name, value):
        super().__setattr__(f'shadow_{name}', value)

    A1 = wrapping.wrap(A)  # pylint: disable=invalid-name
    a = A1(1)
    self.assertEqual(a.shadow_x, 1)
    a.rebind(x=2)
    self.assertEqual(a.shadow_x, 2)

  def testCustomization(self):
    """Test methods from user class clash with pg.Object methods."""

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

    A1 = wrapping.wrap(     # pylint: disable=invalid-name
        A, eq=True, copy=True)
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
    self.assertNotEqual(symbolic.sym_hash(a), 0)
    self.assertEqual(copy.copy(a), 1)
    self.assertEqual(copy.deepcopy(a), 2)
    self.assertTrue(a.sym_eq(a.clone(deep=True)))
    self.assertTrue(a.sym_eq(a.clone(deep=False)))
    self.assertEqual(a.sym_init_args, dict(x=dict(p=1)))
    self.assertEqual(symbolic.query(A1({'p': 2}), where=lambda v: v == 2),
                     {'x.p': 2})
    self.assertEqual(str(a), inspect.cleandoc('''
        A(
          x = {
            p = 1
          }
        )
        '''))
    self.assertEqual(repr(a), 'A(x={p=1})')

    A2 = wrapping.wrap(  # pylint: disable=invalid-name
        A,
        repr=False,
        override=dict(get_result=lambda self: 'overridden results'))
    a = A2({'p': 1})

    # Test get_results is overriden.
    self.assertEqual(a.get_result(), 'overridden results')

    # Test A2's symbolic members are not inherited from ClassWrapper.
    self.assertEqual(a, 1)
    self.assertTrue(symbolic.eq(a, 1))
    self.assertTrue(symbolic.eq(a, A2({'p': 1})))
    self.assertNotEqual(a, 2)
    self.assertTrue(symbolic.ne(a, None))
    self.assertFalse(symbolic.ne(a, A2({'p': 1})))
    self.assertEqual(hash(a), 0)
    self.assertEqual(repr(a), 'foo')
    self.assertEqual(str(a), 'bar')
    self.assertEqual(copy.copy(a), 1)
    self.assertEqual(copy.deepcopy(a), 2)
    self.assertTrue(symbolic.eq(a, a.clone(deep=True)))
    self.assertTrue(symbolic.eq(a, a.clone(deep=False)))

  def testAutomaticReset(self):
    """Test automatic reset."""

    class A:

      def __init__(self, v):
        if hasattr(self, 'x'):
          self.y = v
        else:
          self.x = v

    A1 = wrapping.wrap(A)  # pylint: disable=invalid-name
    a = A1(1)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.sym_init_args.v, 1)
    self.assertFalse(hasattr(a, 'y'))
    a.rebind(v=2)
    self.assertEqual(a.x, 2)
    self.assertEqual(a.sym_init_args.v, 2)
    self.assertFalse(hasattr(a, 'y'))

  def testNonDeterministicMembers(self):
    """Test non-deterministic members."""

    class A(symbolic.NonDeterministic):
      pass

    @wrapping.symbolize
    class B:

      def __init__(self, a):
        self._a = a

    # `B.__init__` shall not be called when it is partial.
    b = B.partial()
    self.assertFalse(b.wrapped_cls_initialized)

    # `B.__init__` shall not be called when any of its
    # arguments has non-deterministic values.
    b = B(A())
    self.assertFalse(b.wrapped_cls_initialized)

    # After rebinding `b.a` to a fixed value, `B.__init__`
    # will be automatically triggered.
    b.rebind(a=1)
    self.assertTrue(b.wrapped_cls_initialized)

    # When `b.a` is rebound to a non-deterministic value again,
    # `b` should be regarded as uninitialized again.
    b.rebind(a=A())
    self.assertFalse(b.wrapped_cls_initialized)

  def testResetStateFn(self):
    """Test custom reset state fn."""
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

    A1 = wrapping.wrap(A, reset_state_fn=reset_state_fn)  # pylint: disable=invalid-name
    a1 = A1(1)
    self.assertEqual(a1.state, {0: 1})
    a2 = A1(2)
    self.assertEqual(a2.state, {0: 2})


if __name__ == '__main__':
  unittest.main()
