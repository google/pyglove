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
"""Tests for pyglove.Object."""

import copy
import inspect
import io
import os
import pickle
import tempfile
import typing
import unittest

from pyglove.core import object_utils
from pyglove.core import typing as pg_typing
from pyglove.core.symbolic import base
from pyglove.core.symbolic import flags
from pyglove.core.symbolic import inferred
from pyglove.core.symbolic.base import query as pg_query
from pyglove.core.symbolic.base import traverse as pg_traverse
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.functor import functor as pg_functor
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.object import members as pg_members
from pyglove.core.symbolic.object import Object
from pyglove.core.symbolic.object import use_init_args as pg_use_init_args
from pyglove.core.symbolic.origin import Origin
from pyglove.core.symbolic.pure_symbolic import NonDeterministic
from pyglove.core.symbolic.pure_symbolic import PureSymbolic


MISSING_VALUE = object_utils.MISSING_VALUE


class ObjectMetaTest(unittest.TestCase):
  """Test `pg.symbolic.ObjectMeta` class."""

  def setUp(self):
    super().setUp()

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Dict()),
        ('z', pg_typing.List(pg_typing.Any())),
        ('p', pg_typing.Bool())
    ])
    class A(Object):
      pass

    @pg_members([
        ('x', pg_typing.Int()),                             # Override 'x'.
        ('p', pg_typing.Bool().freeze(True)),               # Freeze `p`.
        ('q', pg_typing.Bool(default=True)),                # Add new field `q`.
        ('z', pg_typing.List(pg_typing.Int(min_value=1))),  # Override `z`.
    ], serialization_key='B', additional_keys=['ClassB'])
    class B(A):
      pass

    @pg_members([
        ('args', pg_typing.List(pg_typing.Str())),
    ], init_arg_list=['x', 'y', 'z', '*args'])
    class C(B):
      pass

    self._A = A    # pylint: disable=invalid-name
    self._B = B    # pylint: disable=invalid-name
    self._C = C    # pylint: disable=invalid-name

  def test_schema(self):
    self.assertEqual(
        self._C.__schema__,
        pg_typing.create_schema([
            ('x', pg_typing.Int()),
            ('y', pg_typing.Dict()),
            ('z', pg_typing.List(pg_typing.Int(min_value=1))),
            ('p', pg_typing.Bool().freeze(True)),
            ('q', pg_typing.Bool(default=True)),
            ('args', pg_typing.List(pg_typing.Str())),
        ]),
    )

  def test_sym_fields(self):
    self.assertEqual(
        self._C.sym_fields,
        pg_typing.Dict([
            ('x', pg_typing.Int()),
            ('y', pg_typing.Dict()),
            ('z', pg_typing.List(pg_typing.Int(min_value=1))),
            ('p', pg_typing.Bool().freeze(True)),
            ('q', pg_typing.Bool(default=True)),
            ('args', pg_typing.List(pg_typing.Str())),
        ]))

  def test_init_arg_list(self):
    self.assertEqual(
        self._A.init_arg_list, ['x', 'y', 'z', 'p'])
    self.assertEqual(
        self._B.init_arg_list, ['x', 'y', 'z', 'p', 'q'])
    self.assertEqual(
        self._C.init_arg_list, ['x', 'y', 'z', '*args'])

  def test_serialization_key(self):
    self.assertEqual(self._A.__serialization_key__, self._A.__type_name__)
    self.assertEqual(self._B.__serialization_key__, 'B')
    self.assertEqual(self._C.__serialization_key__, self._C.__type_name__)

  def test_type_name(self):
    self.assertEqual(
        self._A.__type_name__, f'{self._A.__module__}.{self._A.__qualname__}')


class ObjectTest(unittest.TestCase):
  """Tests for `pg.Dict`."""

  def test_init(self):
    # Refer to `MembersTest` for detailed tests on various `@pg.members`
    # options.

    @pg_members([
        ('c', pg_typing.Dict([
            ('d', pg_typing.Enum('foo', ['foo', 'bar']))
        ])),
        ('a', pg_typing.Int()),
        ('b', pg_typing.Str().noneable())
    ])
    class A(Object):
      pass

    # Bad init.
    self.assertEqual(A(dict(d='bar'), 1), A(a=1, c=dict(d='bar'), b=None))
    with self.assertRaisesRegex(
        TypeError, '.* missing 1 required argument: \'a\''):
      A()

    with self.assertRaisesRegex(
        TypeError, 'Expect bool type for argument \'allow_partial\''):
      A(a=1, allow_partial='no')

    with self.assertRaisesRegex(
        TypeError, '.* got unexpected keyword arguments'):
      A(x=1, y=2)

    with self.assertRaisesRegex(
        TypeError, '.* takes 3 positional arguments but 4 were given'):
      A(1, 2, 3, 4)

    with self.assertRaisesRegex(
        TypeError, '.* got multiple values for argument \'c\''):
      A(1, c=2)

    class B(Object):
      pass

    with self.assertRaisesRegex(TypeError, '.* takes no arguments.'):
      B(1)

  def test_partial(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Str().noneable())
    ])
    class A(Object):
      pass

    a = A.partial()
    self.assertTrue(a.is_partial)

    a.rebind(x=1)
    self.assertFalse(a.is_partial)

    with self.assertRaisesRegex(TypeError, 'missing 1 required argument'):
      _ = A()

    with flags.allow_partial(True):
      self.assertTrue(A().is_partial)

  def test_empty_field_description(self):
    flags.allow_empty_field_description(False)

    with self.assertRaisesRegex(
        ValueError, 'Field description must not be empty'):

      @pg_members([('x', pg_typing.Int())])
      class A(Object):  # pylint: disable=unused-variable
        pass

    flags.allow_empty_field_description(True)

  def test_override_init(self):

    @pg_members([
        ('x', pg_typing.Int())
    ])
    class A(Object):

      @object_utils.explicit_method_override
      def __init__(self, x):
        super().__init__(int(x))

    a = A(1.1)
    self.assertEqual(a.x, 1)

    class B(A):

      @object_utils.explicit_method_override
      def __init__(self, x):  # pylint: disable=super-init-not-called
        # Forgot to call super().__init__ will trigger error.
        self.x = x

    with self.assertRaisesRegex(
        ValueError, '.* should call `super.*__init__`'):
      _ = B(1)

    with self.assertRaisesRegex(
        TypeError, '.* is a PyGlove managed method.'
    ):
      class C(A):  # pylint: disable=unused-variable
        def __init__(self, x):
          super().__init__(x + 1)

  def test_symbolic_fields_from_annotations(self):

    class X(Object):
      pass

    self.assertEqual(X.init_arg_list, [])

    class A(X):
      x: int
      y: typing.Annotated[float, 'field y'] = 0.0
      z = 2
      # P is upper-case, thus will not be treated as symbolic field.
      P: int = 1
      # _q starts with _, which will not be treated as symbolic field either.
      _q: int = 2

    self.assertEqual(A.init_arg_list, ['x', 'y'])
    self.assertEqual(list(A.__schema__.fields.keys()), ['x', 'y'])

    a = A(1)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 0.0)
    self.assertEqual(A.__schema__.get_field('y').description, 'field y')

    a = A(2, y=1.0)
    self.assertEqual(a.x, 2)
    self.assertEqual(a.y, 1.0)

    class B(A):
      p: str = 'foo'
      q: typing.Any = None

    self.assertEqual(B.init_arg_list, ['x', 'y', 'p', 'q'])
    self.assertEqual(
        list(B.__schema__.fields.keys()),
        ['x', 'y', 'p', 'q'],
    )
    b = B(1, q=2)
    self.assertEqual(b.x, 1)
    self.assertEqual(b.y, 0.0)
    self.assertEqual(b.p, 'foo')
    self.assertEqual(b.q, 2)

    @pg_members([
        ('k', pg_typing.Int())
    ])
    class C(B):
      # Override the default value of 'y'.
      y: float = 1.0

    self.assertEqual(
        list(C.__schema__.fields.keys()),
        ['x', 'y', 'p', 'q', 'k'],
    )
    self.assertEqual(C.init_arg_list, ['x', 'y', 'p', 'q', 'k'])

    c = C(1, q=2, k=3)
    self.assertEqual(c.x, 1)
    self.assertEqual(c.y, 1.0)
    self.assertEqual(c.q, 2)
    self.assertEqual(c.k, 3)

    @pg_members([
        ('e', pg_typing.Int())
    ])
    class D(C):
      f: int = 5

    self.assertEqual(D.init_arg_list, ['x', 'y', 'p', 'q', 'k', 'f', 'e'])
    self.assertEqual(
        list(D.__schema__.fields.keys()),
        ['x', 'y', 'p', 'q', 'k', 'f', 'e'],
    )
    d = D(1, q=2, k=3, e=4)
    self.assertEqual(d.x, 1)
    self.assertEqual(d.y, 1.0)
    self.assertEqual(d.q, 2)
    self.assertEqual(d.k, 3)
    self.assertEqual(d.e, 4)
    self.assertEqual(d.f, 5)

    class E(Object):
      __kwargs__: typing.Any
      x: int

    self.assertEqual(E.init_arg_list, ['x'])
    self.assertEqual(
        list(E.__schema__.fields.keys()),
        [pg_typing.StrKey(), 'x'],
    )
    e = E(1, y=3)
    self.assertEqual(e.x, 1)
    self.assertEqual(e.y, 3)

  def test_forward_reference(self):
    self.assertIs(Foo.schema.get_field('p').value.cls, Foo)

  def test_update_of_default_values(self):

    class A(Object):
      x: int

    class B(A):
      x = 1
    self.assertEqual(B().x, 1)

    class C(B):
      x = 2
    self.assertEqual(C().x, 2)

    @pg_members([
        ('x', pg_typing.Int(default=3))
    ])
    class D(C):
      pass
    self.assertEqual(D().x, 3)

    class F(D):
      def x(self):
        pass
    self.assertEqual(F().sym_init_args.x, 3)

    @pg_members([
        ('x', pg_typing.Callable([pg_typing.Int()])),
        ('y', pg_typing.Int())
    ])
    class G(Object):
      pass

    class H(G):

      # Member method as the default value for callable symbolic attribute.
      def x(self, v):
        return self.sym_init_args.y + v * 2

      # Member method will not override non-callable symbolic attribute.
      def y(self):
        return self.sym_init_args.y * 2

    h = H(y=1)
    self.assertEqual(h.x(1), 3)
    self.assertEqual(h.y(), 2)
    self.assertEqual(h.sym_init_args.x(h, 1), 3)

  def test_override_symbolic_attribute_with_property(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int()),
        ('z', pg_typing.Int()),
    ])
    class A(Object):

      @property
      def x(self):
        return self.sym_init_args.x + 1

      def z(self):
        return self.sym_init_args.z + 2

    a = A(1, 2, 3)
    self.assertEqual(a.x, 2)
    self.assertEqual(a.sym_init_args.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z(), 5)
    self.assertEqual(a.sym_init_args.z, 3)

  def test_runtime_type_check(self):

    @pg_members([
        ('x', pg_typing.Int(min_value=0)),
        ('y', pg_typing.Dict([
            (pg_typing.StrKey(), pg_typing.Bool())
        ]))
    ])
    class A(Object):
      pass

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      A(x=1.0)

    with self.assertRaisesRegex(ValueError, '.* out of range'):
      A(x=-1)

    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      A(x=1, y=dict(a=1))

  def test_symbolization_on_nested_containers(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    a = A(x=dict(y=list()))
    self.assertIsInstance(a.x, Dict)
    self.assertIsInstance(a.x.y, List)

  def test_implicit_copy_during_assignment(self):

    class X:
      pass

    @pg_members([
        ('x', pg_typing.Object(X)),
        ('y', pg_typing.Dict()),
        ('z', pg_typing.List(pg_typing.Any())),
    ])
    class A(Object):
      pass

    # There is no impliit copy when assigning a root symbolic object to
    # another tree.
    x = X()
    a = A(x, dict(), [])
    sd = Dict(a=a)
    self.assertIs(a, sd.a)

    # There is an implicit copy when assigning a symbolic object with
    # a parent to another tree.
    sd2 = Dict(a=a)
    self.assertEqual(a, sd2.a)
    self.assertIsNot(a, sd2.a)
    self.assertIsNot(a.y, sd2.a.y)
    self.assertIsNot(a.z, sd2.a.z)
    # Non-symbolic member is copied by reference.
    self.assertIs(a.x, sd2.a.x)

  def test_inspect(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    s = io.StringIO()
    a = A([A(1), A(2)])
    a.inspect(file=s, compact=True)
    self.assertEqual(s.getvalue(), 'A(x=[0: A(x=1), 1: A(x=2)])\n')

    s = io.StringIO()
    a.inspect(where=lambda v: v == 1, file=s)
    self.assertEqual(s.getvalue(), '{\n  \'x[0].x\': 1\n}\n')

  def test_copy(self):

    class X:
      pass

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A([dict(), A(X())])
    a2 = copy.copy(a)
    self.assertEqual(a, a2)
    # Symbolic containers are deeply copied.
    self.assertIsNot(a.x, a2.x)
    self.assertIsNot(a.x[0], a2.x[0])
    self.assertIsNot(a.x[1], a2.x[1])
    # Regualr objects are shallowly copied.
    self.assertIs(a.x[1].x, a2.x[1].x)

  def test_deepcopy(self):

    class X:
      def __init__(self, v):
        self.v = v

      def __eq__(self, other):
        return isinstance(other, X) and self.v == other.v

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A([dict(), A(X(1))])
    a2 = copy.deepcopy(a)
    self.assertEqual(a, a2)
    # Symbolic containers are deeply copied.
    self.assertIsNot(a.x, a2.x)
    self.assertIsNot(a.x[0], a2.x[0])
    self.assertIsNot(a.x[1], a2.x[1])
    # Regualr objects are also deeply copied.
    self.assertIsNot(a.x[1].x, a2.x[1].x)

  def test_sym_init_args(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Bool(default=True)),
    ])
    class A(Object):
      pass

    a = A(1)
    self.assertEqual(a.sym_init_args, {
        'x': 1,
        'y': True
    })
    a = A.partial()
    self.assertEqual(a.sym_init_args, {
        'x': MISSING_VALUE,
        'y': True
    })

  def test_setattr(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Bool(default=True)),
    ])
    class A(Object):
      pass

    a = A(1)
    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot set attribute .*allow_symbolic_assignment` is set to False'):
      a.x = 1

    class B(A):
      allow_symbolic_assignment = True

      def _on_bound(self):
        super()._on_bound()
        self.z = self.x + int(self.y)

    b = B(1)
    self.assertEqual(b.z, 2)
    b.x = 5
    b.y = False
    self.assertEqual(b.z, 5)

    # Reset b.y to default.
    b.y = MISSING_VALUE
    self.assertTrue(b.y)
    self.assertEqual(b.z, 6)

    # Class with dynamic field.
    @pg_members([
        (pg_typing.StrKey('p.*'), pg_typing.Int())
    ])
    class C(B):
      pass

    c = C(1, p1=12)
    c.p2 = 24
    c.q1 = 36

    self.assertEqual(c.p1, 12)
    self.assertEqual(c.p2, 24)
    self.assertEqual(c.q1, 36)
    # c.q1 can be set as a class attribute but not a symbolic attribute.
    self.assertEqual(c.sym_init_args, dict(x=1, y=True, p1=12, p2=24))

  def test_getattr(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Bool(default=True)),
    ])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.z = self.x + int(self.y)

    a = A(1)
    self.assertEqual(a.x, 1)
    self.assertTrue(a.y)      # Use default value from the field definition.
    self.assertEqual(a.z, 2)  # Non-symbolic field.

    class B(Object):
      x: int

    b = B(inferred.ValueFromParentChain())
    with self.assertRaises(AttributeError):
      _ = b.x

    sd = Dict(x=1, b=b)
    self.assertEqual(sd.b.x, 1)

  def test_non_default(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Bool(default=True)),
        ('z', pg_typing.Dict([
            ('p', pg_typing.Float(default=1.0))
        ]))
    ])
    class A(Object):
      pass

    a = A(x=1, y=True, z=dict(p=2.0))
    self.assertEqual(a.non_default_values(), {'x': 1, 'z.p': 2.0})
    self.assertEqual(
        a.non_default_values(flatten=False), {'x': 1, 'z': {'p': 2.0}})
    a.rebind({'z.p': 1.0}, y=False, x=2)
    self.assertEqual(a.non_default_values(), {'x': 2, 'y': False})

  def test_missing_values(self):

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Bool(default=True)),
        ('z', pg_typing.Dict([
            ('p', pg_typing.Float())
        ]))
    ])
    class A(Object):
      pass

    a = A.partial()
    self.assertEqual(
        a.missing_values(), {'x': MISSING_VALUE, 'z.p': MISSING_VALUE})
    self.assertEqual(
        a.missing_values(flatten=False),
        {'x': MISSING_VALUE, 'z': {'p': MISSING_VALUE}})

    # After rebind, the non_default_values are updated.
    a.rebind({'x': 1, 'y': MISSING_VALUE, 'z.p': 1.0})
    self.assertEqual(a.missing_values(), {})

    # Test inferred value as the default value.
    class B(Object):
      x: int

    b = B(inferred.ValueFromParentChain())
    self.assertEqual(b.sym_missing(), {})

  def test_sym_has(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.y = 1

    a = A(A(dict(y=A(1))))
    self.assertTrue(a.sym_has('x'))
    self.assertTrue(a.sym_has('x.x'))
    self.assertTrue(a.sym_has(object_utils.KeyPath.parse('x.x.y')))
    self.assertTrue(a.sym_has(object_utils.KeyPath.parse('x.x.y.x')))
    self.assertFalse(a.sym_has('y'))   # `y` is not a symbolic field.

  def test_sym_get(self):

    @pg_members([('x', pg_typing.Any()), ('p', pg_typing.Any().noneable())])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.y = 1

    a = A(
        A(dict(y=A(1, p=inferred.ValueFromParentChain()))),
        p=inferred.ValueFromParentChain(),
    )

    self.assertIs(a.sym_get('x'), a.x)
    self.assertIs(a.sym_get('p'), a.sym_getattr('p'))
    self.assertIs(a.sym_get('x.x'), a.x.x)
    self.assertIs(a.sym_get(object_utils.KeyPath.parse('x.x.y')), a.x.x.y)
    self.assertIs(a.sym_get(object_utils.KeyPath.parse('x.x.y.x')), a.x.x.y.x)
    self.assertIs(
        a.sym_get(object_utils.KeyPath.parse('x.x.y.p')),
        a.x.x.y.sym_getattr('p'),
    )

    with self.assertRaisesRegex(
        KeyError, 'Path y does not exist.'):  # `y` is not a symbolic field.
      a.sym_get('y')

  def test_sym_hasattr(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.y = 1

    a = A(1)
    self.assertTrue(a.sym_hasattr('x'))
    self.assertFalse(a.sym_hasattr('y'))

    @pg_members([
        (pg_typing.StrKey('x.*'), pg_typing.Int())
    ])
    class B(Object):
      allow_symbolic_assignment = True

    b = B(x1=1)
    b.x2 = 2
    b.y = 1

    self.assertTrue(b.sym_hasattr('x1'))
    self.assertTrue(b.sym_hasattr('x2'))
    self.assertFalse(b.sym_hasattr('y'))

  def test_sym_getattr(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.y = 1

    a = A(1)
    self.assertEqual(a.sym_getattr('x'), 1)

    a = A(x=inferred.ValueFromParentChain())
    self.assertEqual(a.sym_getattr('x'), inferred.ValueFromParentChain())

    with self.assertRaisesRegex(
        AttributeError, 'has no symbolic attribute \'y\''):
      _ = a.sym_getattr('y')

    @pg_members([
        (pg_typing.StrKey('x.*'), pg_typing.Int())
    ])
    class B(Object):
      allow_symbolic_assignment = True

    b = B(x1=1)
    b.x2 = 2
    b.y = 1

    self.assertEqual(b.sym_getattr('x1'), 1)
    self.assertEqual(b.sym_getattr('x2'), 2)
    with self.assertRaisesRegex(
        AttributeError, 'has no symbolic attribute \'y\''):
      _ = b.sym_getattr('y')

  def test_sym_inferred(self):
    class StaticValue(inferred.InferredValue):
      v: typing.Any

      def infer(self):
        return self.v

    class A(Object):
      x: int = 1
      y: int = StaticValue(v=0)

    a = A()
    self.assertEqual(a.sym_inferred('x'), 1)
    self.assertEqual(a.sym_inferred('y'), 0)
    self.assertEqual(a.sym_inferred('y', None), 0)
    with self.assertRaisesRegex(AttributeError, 'z'):
      _ = a.sym_inferred('z')
    self.assertIsNone(a.sym_inferred('z', None))

  def test_sym_inferrable(self):
    class A(Object):
      x: int
      y: int = 1
      z: int = inferred.ValueFromParentChain()

    a = A.partial()
    _ = Dict(p=Dict(a=a, b=3), z=2)
    self.assertFalse(a.sym_inferrable('x'))
    self.assertTrue(a.sym_inferrable('y'))
    self.assertTrue(a.sym_inferrable('z'))

  def test_sym_field(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    a = A(A(dict(y=A(1))))
    self.assertIsNone(a.sym_field)
    self.assertIs(a.x.sym_field, A.__schema__.get_field('x'))
    self.assertIs(a.x.x.sym_field, A.__schema__.get_field('x'))
    self.assertIsNone(a.x.x.y.sym_field)   # The dict is not schematized.

  def test_sym_attr_field(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A(A(1, 2), A(3, 4))
    self.assertIs(a.sym_attr_field('x'), A.__schema__.get_field('x'))
    self.assertIs(a.sym_attr_field('y'), A.__schema__.get_field('y'))
    self.assertIs(a.x.sym_attr_field('x'), A.__schema__.get_field('x'))
    self.assertIs(a.y.sym_attr_field('y'), A.__schema__.get_field('y'))

  def test_sym_keys(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Any()),
        (pg_typing.StrKey(), pg_typing.Str()),
    ])
    class A(Object):
      pass

    @pg_members([
        ('x', pg_typing.Int()),
        ('z', pg_typing.Any()),
    ])
    class B(A):
      pass

    b = B(a='foo', b='bar', y=1, x=2, z=True)
    self.assertEqual(next(b.sym_keys()), 'x')
    # Dynamic fields always go at last.
    self.assertEqual(list(b.sym_keys()), ['x', 'y', 'z', 'a', 'b'])

  def test_sym_values(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Any()),
        (pg_typing.StrKey(), pg_typing.Str()),
    ])
    class A(Object):
      pass

    @pg_members([
        ('x', pg_typing.Int()),
        ('z', pg_typing.Any()),
        (pg_typing.StrKey(), pg_typing.Str()),
    ])
    class B(A):
      pass

    b = B(a='foo', b='bar', y=1, x=2, z=True)
    self.assertEqual(next(b.sym_values()), 2)
    # Dynamic fields always go at last.
    self.assertEqual(list(b.sym_values()), [2, 1, True, 'foo', 'bar'])

  def test_sym_items(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Any()),
        (pg_typing.StrKey(), pg_typing.Str()),
    ])
    class A(Object):
      pass

    @pg_members([
        ('x', pg_typing.Int()),
        ('z', pg_typing.Any()),
        (pg_typing.StrKey(), pg_typing.Str()),
    ])
    class B(A):
      pass

    b = B(a='foo', b='bar', y=1, x=2, z=True)
    self.assertEqual(next(b.sym_items()), ('x', 2))
    # Dynamic fields always go at last.
    self.assertEqual(
        list(b.sym_items()),
        [('x', 2), ('y', 1), ('z', True), ('a', 'foo'), ('b', 'bar')])

  def test_sym_jsonify(self):
    # Refer to SerializationTest for more detailed tests.

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Any()),
        (pg_typing.StrKey(), pg_typing.Str()),
    ])
    class A(Object):
      pass

    a = A(1, 2, a='foo')
    json_dict = a.sym_jsonify()
    self.assertEqual(
        json_dict, {'_type': A.__type_name__, 'x': 1, 'y': 2, 'a': 'foo'}
    )
    self.assertEqual(
        base.from_json(json_dict), a)

  def test_sym_rebind(self):
    # Refer to RebindTest for more detailed tests.

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int()),
    ])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.z = self.x + self.y

    a = A(1, 2)
    self.assertEqual(a.z, 3)
    a_prime = a.sym_rebind(x=3, y=4)
    self.assertIs(a, a_prime)
    self.assertEqual(a.z, 7)

  def test_sym_clone(self):

    class X:
      pass

    @pg_members([
        ('x', pg_typing.List(pg_typing.Int())),
        ('y', pg_typing.Dict([
            ('a', pg_typing.List(pg_typing.Any())),
            ('b', pg_typing.Dict()),
            ('c', pg_typing.Object(X)),
        ])),
        ('z', pg_typing.Object(X)),
    ])
    class A(Object):
      pass

    x = X()
    a = A([], dict(a=[], b={}, c=x), x)
    a2 = a.clone()
    self.assertEqual(a, a2)
    self.assertIsNot(a, a2)

    # Symbolic members are always copied by value.
    self.assertIsNot(a.x, a2.x)
    self.assertIsNot(a.y, a2.y)
    self.assertIsNot(a.y.a, a2.y.a)
    self.assertIsNot(a.y.b, a2.y.b)

    # Non-symbolic members are copied by reference.
    self.assertIs(a.z, a2.z)
    self.assertIs(a.y.c, a2.y.c)

  def test_sym_origin(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    @pg_functor
    def builder(x):
      return A(x)

    @pg_functor
    def builder_of_builder(x):
      return builder(x)

    # Origin is not tracked by default.
    a = builder_of_builder(1)
    a1 = a()       # a1 is a `builder`.
    a2 = a()       # a2 is an `A`.
    a3 = a.clone()
    a4 = a3.clone(deep=True)
    self.assertIsNone(a4.sym_origin)

    # Test automatic origin tracking.
    with flags.track_origin():
      a = builder_of_builder(1)
      a1 = a()       # a1 is a `builder`.
      a2 = a1()       # a2 is an `A`.
      a3 = a2.clone()
      a4 = a3.clone(deep=True)
      self.assertIsNotNone(a4.sym_origin)
      self.assertEqual(a4.sym_origin.chain(), [
          Origin(a3, tag='deepclone'),
          Origin(a2, tag='clone'),
          Origin(a1, tag='return'),
          Origin(a, tag='return'),
          Origin(None, tag='__init__')
      ])
      self.assertEqual(a3.sym_origin.chain('clone'), [
          Origin(a2, tag='clone'),
      ])

    # Test not to track origin.
    with flags.track_origin(False):
      a = A(1)
      with flags.track_origin(True):
        a2 = a.clone()
        a3 = a2.clone(deep=True)
        self.assertEqual(a3.sym_origin.chain(), [
            Origin(a2, tag='deepclone'),
            Origin(a, tag='clone'),
        ])

    # Test setting origin by the user.
    a = A(1)
    a.sym_setorigin(None, '__init__')
    self.assertIsNone(a.sym_origin.source)
    self.assertEqual(a.sym_origin.tag, '__init__')
    self.assertIsNone(a.sym_origin.stack)
    self.assertIsNone(a.sym_origin.stacktrace)

    # Set origin with a different description.
    a1 = A(1)
    a.sym_setorigin(a1, 'producer', stacktrace=True)
    self.assertIs(a.sym_origin.source, a1)
    self.assertEqual(a.sym_origin.tag, 'producer')
    self.assertIsNotNone(a.sym_origin.stack)
    self.assertIsNotNone(a.sym_origin.stacktrace)

    # Once origin is set, cannot change the source object.
    a2 = A(1)
    with self.assertRaisesRegex(
        ValueError, 'Cannot set the origin with a different source value'):
      a.sym_setorigin(a2, 'builder3')

  def test_sym_partial(self):
    # Refer to `test_partial` for more details.

    @pg_members([
        ('x', pg_typing.Int())
    ])
    class A(Object):
      pass

    a = A.partial()
    self.assertTrue(a.sym_partial)
    a.rebind(x=1)
    self.assertFalse(a.sym_partial)

  def test_sym_missing(self):

    @pg_members([
        ('x', pg_typing.Int())
    ])
    class A(Object):
      pass

    a = A.partial()
    self.assertEqual(a.sym_missing(), {'x': MISSING_VALUE})
    a.rebind(x=1)
    self.assertEqual(len(a.sym_missing()), 0)

  def test_sym_nondefault(self):
    # Refer to `test_non_default_values` for more details.

    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Int(default=2)),
        ('z', pg_typing.Dict([
            ('a', pg_typing.Float(default=0.0)),
            ('b', pg_typing.Float(default=1.0))
        ]))
    ])
    class A(Object):
      pass

    a = A(1, 1, dict(a=0.2))
    self.assertEqual(a.sym_nondefault(), {'x': 1, 'y': 1, 'z.a': 0.2})
    a.rebind({'z.b': 0.0, 'z.a': 0.0}, y=2)
    self.assertEqual(a.sym_nondefault(), {'x': 1, 'z.b': 0.0})

  def test_sym_puresymbolic(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    class X(PureSymbolic):
      pass

    self.assertFalse(A(x=1).sym_puresymbolic)
    self.assertFalse(A.partial().sym_puresymbolic)
    self.assertTrue(A(x=X()).sym_puresymbolic)

    a = A(1)
    self.assertFalse(a.sym_puresymbolic)
    a.rebind(x=X())
    self.assertTrue(a.sym_puresymbolic)

  def test_sym_abstract(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    class X(PureSymbolic):
      pass

    self.assertFalse(A(x=1).sym_abstract)
    self.assertFalse(base.is_abstract(A(x=1)))
    self.assertTrue(A.partial().sym_abstract)
    self.assertTrue(base.is_abstract(A.partial()))
    self.assertTrue(A(x=X()).sym_abstract)
    self.assertTrue(base.is_abstract(A(x=X())))

    a = A(1)
    self.assertFalse(a.sym_abstract)
    a.rebind(x=X())
    self.assertTrue(a.sym_abstract)
    with flags.allow_partial():
      a.rebind(x=MISSING_VALUE)
    self.assertTrue(a.sym_abstract)

  def test_is_deterministic(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    class X(NonDeterministic):
      pass

    self.assertTrue(A(1).is_deterministic)
    self.assertFalse(A(X()).is_deterministic)
    self.assertFalse(A([dict(y=X())]).is_deterministic)

  def test_sym_contains(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    a = A(dict(y=[dict(z=1)]))
    self.assertTrue(a.sym_contains(value=1))
    self.assertFalse(a.sym_contains(value=2))
    self.assertTrue(a.sym_contains(type=int))
    self.assertFalse(a.sym_contains(type=str))

  def test_sym_eq(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    # Use cases that `__eq__` and `sym_eq` have the same results.
    self.assertEqual(A(1), A(1))
    self.assertTrue(base.eq(A(1), A(1)))
    self.assertTrue(
        base.eq(
            A(inferred.ValueFromParentChain()),
            A(inferred.ValueFromParentChain()),
        )
    )

    self.assertEqual(A.partial(), A.partial())
    self.assertTrue(base.eq(A.partial(), A.partial()))

    # Use case that `__eq__` rules both Python equality and `pg.eq`.
    class X:

      def __init__(self, value):
        self.value = value

      def __eq__(self, other):
        return ((isinstance(other, X) and self.value == other.value)
                or self.value == other)

    self.assertEqual(A(X(1)), A(1))
    self.assertTrue(base.eq(A(X(1)), A(1)))

    # Use case that `sym_eq` only rule `pg.eq` but not Python equality.
    class Y:

      def __init__(self, value):
        self.value = value

      def sym_eq(self, other):
        return ((isinstance(other, Y) and self.value == other.value)
                or self.value == other)

    self.assertTrue(base.eq(A(Y(1)), A(1)))
    self.assertNotEqual(Y(1), 1)
    # NOTICE!! This returns True since __eq__ is delegated to sym_eq
    # when `use_symbolic_comparison` is True (default).
    self.assertEqual(A(Y(1)), A(1))

    class B(A):
      use_symbolic_comparison = False

    self.assertNotEqual(B(1), B(1))
    self.assertTrue(base.eq(B(1), B(1)))
    self.assertNotEqual(B(Y(1)), B(1))
    self.assertTrue(base.eq(B(Y(1)), B(1)))

  def test_sym_ne(self):
    # Refer test_sym_eq for more details.

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    self.assertNotEqual(A(1), 1)
    self.assertTrue(base.ne(A(1), 1))
    self.assertNotEqual(A(1), A(2))
    self.assertTrue(base.ne(A(1), A(2)))
    self.assertNotEqual(A(A(1)), A(1))
    self.assertTrue(base.ne(A(A(1)), A(1)))

  def test_sym_lt(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Int().noneable())
    ])
    class A(Object):
      pass

    class X:

      def __init__(self, value):
        self.value = value

      def __lt__(self, other):
        if isinstance(other, X):
          return self.value < other.value
        return False

    self.assertFalse(A(1).sym_lt(MISSING_VALUE))
    self.assertFalse(A(1).sym_lt(None))
    self.assertFalse(A(1).sym_lt(True))
    self.assertFalse(A(1).sym_lt(1))
    self.assertFalse(A(1).sym_lt(2.0))
    self.assertFalse(A(1).sym_lt('abc'))
    self.assertFalse(A(1).sym_lt([]))
    self.assertFalse(A(1).sym_lt(tuple()))
    self.assertFalse(A(1).sym_lt(set()))
    self.assertFalse(A(1).sym_lt(dict()))
    self.assertFalse(A(1).sym_lt(A(1)))

    self.assertTrue(A(None).sym_lt(A(0)))
    self.assertTrue(A(0).sym_lt(A(1)))
    self.assertTrue(A(0).sym_lt(A(0, 1)))
    self.assertTrue(A(0, 1).sym_lt(A(1)))
    self.assertTrue(A(X(0)).sym_lt(A(X(1))))
    self.assertFalse(A(0).sym_lt(A(0)))
    self.assertFalse(A(1).sym_lt(A(0)))
    self.assertFalse(A(0, 1).sym_lt(A(0)))
    self.assertFalse(A(1).sym_lt(A(0, 1)))
    self.assertFalse(A(X(1)).sym_lt(A(X(0))))

    class Y:
      pass

    self.assertTrue(A(1).sym_lt(Y()))

  def test_sym_gt(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Int().noneable())
    ])
    class A(Object):
      pass

    class X:

      def __init__(self, value):
        self.value = value

      def __lt__(self, other):
        if isinstance(other, X):
          return self.value < other.value
        return False

    self.assertTrue(A(1).sym_gt(MISSING_VALUE))
    self.assertTrue(A(1).sym_gt(None))
    self.assertTrue(A(1).sym_gt(True))
    self.assertTrue(A(1).sym_gt(1))
    self.assertTrue(A(1).sym_gt(2.0))
    self.assertTrue(A(1).sym_gt('abc'))
    self.assertTrue(A(1).sym_gt([]))
    self.assertTrue(A(1).sym_gt((1,)))
    self.assertTrue(A(1).sym_gt(set()))
    self.assertTrue(A(1).sym_gt(dict()))

    self.assertTrue(A(0).sym_gt(A(None)))
    self.assertTrue(A(1).sym_gt(A(0)))
    self.assertTrue(A(0, 1).sym_gt(A(0)))
    self.assertTrue(A(1).sym_gt(A(0)))
    self.assertTrue(A(X(1)).sym_gt(A(X(0))))
    self.assertFalse(A(0).sym_gt(A(0)))
    self.assertFalse(A(0).sym_gt(A(1)))
    self.assertFalse(A(0).sym_gt(A(0, 1)))
    self.assertFalse(A(0, 1).sym_gt(A(1)))
    self.assertFalse(A(0).sym_gt(A(0)))
    self.assertFalse(A(0).sym_gt(A(1)))
    self.assertFalse(A(0).sym_gt(A(0, 1)))
    self.assertFalse(A(X(0)).sym_gt(A(X(1))))

    class Y:
      pass

    self.assertFalse(A(0).sym_gt(Y()))

  def test_sym_hash(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Int().noneable())
    ])
    class A(Object):

      def result(self):
        return self.x + self.y

    self.assertEqual(hash(A(0)), hash(A(0)))
    self.assertEqual(hash(A(1, None)), hash(A(1, None)))
    self.assertEqual(hash(A(A([A({})]))), hash(A(A([A({})]))))
    self.assertNotEqual(hash(A(0)), hash(A(1)))
    self.assertNotEqual(hash(A(0, 1)), hash(A(0, 2)))

    class X:
      pass

    a = X()
    b = X()
    self.assertNotEqual(hash(A(a)), hash(A(b)))

    class Y:

      def __init__(self, value):
        self.value = value

      def __hash__(self):
        return hash((Y, self.value))

    a = Y(1)
    b = Y(1)
    self.assertEqual(hash(a), hash(b))
    self.assertEqual(hash(A(a)), hash(A(b)))
    self.assertNotEqual(hash(A(Y(1))), hash(A(Y(2))))

    # Test symbolic hashing for functions and methods.
    a = lambda x: x
    b = base.from_json_str(base.to_json_str(a))
    self.assertNotEqual(hash(a), hash(b))
    self.assertEqual(base.sym_hash(a), base.sym_hash(b))
    self.assertEqual(
        base.sym_hash(A(1, 2).result), base.sym_hash(A(1, 2).result))
    self.assertNotEqual(
        base.sym_hash(A(1, 2).result), base.sym_hash(A(2, 3).result))

  def test_sym_parent(self):

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A(dict(x=A([A(1)])))
    self.assertIsNone(a.sym_parent)

    self.assertIs(a.x.sym_parent, a)
    self.assertIs(a.x.x.sym_parent, a.x)
    self.assertIs(a.x.x.x.sym_parent, a.x.x)
    self.assertIs(a.x.x.x[0].sym_parent, a.x.x.x)

    pa = A(a)
    self.assertIs(a.sym_parent, pa)

  def test_sym_root(self):

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A(dict(x=A([A(1)])))
    self.assertIs(a.sym_root, a)

    self.assertIs(a.x.sym_root, a)
    self.assertIs(a.x.x.sym_root, a)
    self.assertIs(a.x.x.x.sym_root, a)
    self.assertIs(a.x.x.x[0].sym_root, a)

    pa = A(a)
    self.assertIs(a.sym_root, pa)
    self.assertIs(a.x.sym_root, pa)
    self.assertIs(a.x.x.sym_root, pa)
    self.assertIs(a.x.x.x.sym_root, pa)
    self.assertIs(a.x.x.x[0].sym_root, pa)

  def test_sym_ancestor(self):

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A(dict(x=A([A(1)])))
    self.assertIs(a.x.sym_ancestor(), a)
    self.assertIs(a.x.sym_ancestor(lambda x: isinstance(x, A)), a)
    self.assertIsNone(a.x.sym_ancestor(lambda x: isinstance(x, int)), a)
    self.assertIs(a.x.x.sym_ancestor(lambda x: isinstance(x, A)), a)
    self.assertIs(a.x.x.x[0].sym_ancestor(lambda x: isinstance(x, A)), a.x.x)
    self.assertIs(a.x.x.x[0].sym_ancestor(lambda x: isinstance(x, list)),
                  a.x.x.x)
    self.assertIs(a.x.x.x[0].sym_ancestor(lambda x: isinstance(x, dict)), a.x)

  def test_sym_path(self):

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A(dict(x=A([A(1)])))
    self.assertEqual(a.sym_path, '')
    self.assertEqual(a.x.sym_path, 'x')
    self.assertEqual(a.x.x.sym_path, 'x.x')
    self.assertEqual(a.x.x.x.sym_path, 'x.x.x')
    self.assertEqual(a.x.x.x[0].sym_path, 'x.x.x[0]')

    a.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(a.sym_path, 'a')
    self.assertEqual(a.x.sym_path, 'a.x')
    self.assertEqual(a.x.x.sym_path, 'a.x.x')
    self.assertEqual(a.x.x.x.sym_path, 'a.x.x.x')
    self.assertEqual(a.x.x.x[0].sym_path, 'a.x.x.x[0]')

  def test_accessor_writable(self):

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    a = A(0)
    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot set attribute of .* while .*allow_symbolic_assignment` '
        'is set to False.'):
      a.x = 2
    self.assertEqual(a.x, 0)

    with flags.allow_writable_accessors(True):
      a.x = 2
    self.assertEqual(a.x, 2)

    a.rebind(x=3)
    self.assertEqual(a.x, 3)

    a.set_accessor_writable(True)
    a.x = 1
    self.assertEqual(a.x, 1)
    with flags.allow_writable_accessors(False):
      with self.assertRaisesRegex(
          base.WritePermissionError, 'Cannot set attribute of .*.'):
        a.x = 2
    a.x = 2
    self.assertEqual(a.x, 2)

    class B(Object):
      allow_symbolic_assignment = True
      y: int

    a = A(x=B(1))
    with self.assertRaisesRegex(
        base.WritePermissionError,
        'Cannot set attribute of .* while .*allow_symbolic_assignment` '
        'is set to False.',
    ):
      a.x = 1
    a.x.y = 2
    self.assertEqual(a.x.y, 2)

  def test_seal(self):

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      allow_symbolic_assignment = True

    a = A(0).seal()
    self.assertTrue(a.is_sealed)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot set attribute .* object is sealed'):
      a.x = 1

    with flags.as_sealed(False):
      a.x = 2
      # Object-level is_sealed flag is not modified.
      self.assertTrue(a.is_sealed)
    self.assertEqual(a.x, 2)

    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot rebind a sealed A'):
      a.rebind(x=1)

    with flags.as_sealed(False):
      a.rebind(x=3)
    self.assertEqual(a.x, 3)

    # Unseal.
    a.seal(False)
    self.assertFalse(a.is_sealed)

    # Test repeated seal has no side effect.
    a.seal(False)
    self.assertFalse(a.is_sealed)

    a.x = 2
    self.assertEqual(a.x, 2)

    with flags.as_sealed(True):
      with self.assertRaisesRegex(
          base.WritePermissionError,
          'Cannot set attribute .* object is sealed'):
        a.x = 1
      self.assertEqual(a.x, 2)
      # Object-level sealed state is not changed,
      self.assertFalse(a.is_sealed)

    # Seal again.
    a.seal()
    with self.assertRaisesRegex(
        base.WritePermissionError, 'Cannot rebind a sealed A.'):
      a.rebind(x=0)

    # Test nested sealed object.
    a = A([A(dict(x=A(1)))])
    self.assertFalse(a.is_sealed)
    self.assertFalse(a.x.is_sealed)
    self.assertFalse(a.x[0].is_sealed)
    self.assertFalse(a.x[0].x.is_sealed)
    self.assertFalse(a.x[0].x.x.is_sealed)

    a.seal()
    self.assertTrue(a.is_sealed)
    self.assertTrue(a.x.is_sealed)
    self.assertTrue(a.x[0].is_sealed)
    self.assertTrue(a.x[0].x.is_sealed)
    self.assertTrue(a.x[0].x.x.is_sealed)


class MembersTest(unittest.TestCase):
  """Tests for `pg.members`."""

  def test_class_with_varargs(self):

    @pg_members([
        ('x', pg_typing.Int()),

        # `args` is a field for varargs, for two reasons:
        # 1) it must be described by a `pg_typing.List` value spec.
        # 2) it must appear in `init_arg_list` with a prefix "*".
        ('args', pg_typing.List(pg_typing.Int())),

        # `y` is a keyword-only argument.
        ('y', pg_typing.Str())
    ], init_arg_list=['x', '*args'])
    class A(Object):
      pass

    a = A(1, 2, 3, 4, y='foo')
    self.assertEqual(a.x, 1)
    self.assertEqual(a.args, [2, 3, 4])
    self.assertEqual(a.y, 'foo')
    self.assertEqual(a.sym_init_args, dict(x=1, args=[2, 3, 4], y='foo'))

    with self.assertRaisesRegex(
        TypeError, '.* missing 1 required argument: \'y\''):
      _ = A(1, 2, 'foo')

  def test_class_with_kwonly_args(self):

    # Setting `init_arg_list` to an empty list allows all fields to be
    # keyword only.
    @pg_members([
        ('x', pg_typing.Int()),
        ('y', pg_typing.Str(default='foo'))
    ], init_arg_list=[])
    class A(Object):
      pass

    with self.assertRaisesRegex(
        TypeError, '.* takes 0 positional arguments but 2 were given'):
      _ = A(1, 'foo')

    a = A(x=1, y='foo')
    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 'foo')

  def test_class_with_varkw(self):

    # Adding field with `pg_typing.StrKey()` key will enable variable length
    # keyword arguments.
    @pg_members([
        ('x', pg_typing.Int()),
        (pg_typing.StrKey(), pg_typing.Int())
    ])
    class A(Object):
      pass

    a = A(1, y=2, z=3)
    self.assertEqual(a.x, 1)
    self.assertEqual(a.y, 2)
    self.assertEqual(a.z, 3)

    # Moreover, users can use regex expression to whitelist what keyword
    # arguments are accessible.

    @pg_members([
        ('x', pg_typing.Int()),
        (pg_typing.StrKey('y.*'), pg_typing.Int())
    ])
    class B(Object):
      pass

    # Okay.
    b = B(1, y1=2, y2=3, y3=4)
    self.assertEqual(b.sym_init_args, dict(x=1, y1=2, y2=3, y3=4))

    # Not okay: `z` does not conform to regex `y.*`.
    with self.assertRaisesRegex(
        TypeError, 'got unexpected keyword argument: \'z\''):
      _ = B(1, z=2)

  def test_use_init_args(self):

    @pg_use_init_args(['x', 'y', '*z'])
    class A(Object):
      y: int
      x: str
      z: list[str]
      p: str
      q: int

    a = A('foo', 1, 'a', 'b', p='bar', q=2)
    self.assertEqual(a.x, 'foo')
    self.assertEqual(a.y, 1)
    self.assertEqual(a.z, ['a', 'b'])
    self.assertEqual(a.p, 'bar')
    self.assertEqual(a.q, 2)

  def test_serialization_key(self):

    @pg_members([
        ('x', pg_typing.Int())
    ], serialization_key='ClassA')
    class A(Object):
      pass

    json_dict = A(1).to_json()
    self.assertEqual(json_dict['_type'], 'ClassA')
    self.assertEqual(base.from_json(json_dict), A(1))

    # Despite of serialization key, `_type` with type name also works.
    json_dict['_type'] = A.__type_name__
    self.assertEqual(base.from_json(json_dict), A(1))

  def test_additional_keys(self):

    additional_deserialization_keys = ['moduleA.A', 'moduleB.A']

    @pg_members([
        ('x', pg_typing.Int())
    ], additional_keys=additional_deserialization_keys)
    class A(Object):
      pass

    json_dict = A(1).to_json()
    self.assertEqual(json_dict['_type'], A.__type_name__)
    self.assertEqual(base.from_json(json_dict), A(1))

    for key in additional_deserialization_keys:
      json_dict['_type'] = key
      self.assertEqual(base.from_json(json_dict), A(1))

  def test_bad_cases(self):

    with self.assertRaisesRegex(TypeError, 'Unsupported keyword arguments'):

      @pg_members([], unsupported_arg=1)
      class A(Object):  # pylint: disable=unused-variable
        pass


class InitSignatureTest(unittest.TestCase):
  """Tests for `pg.Object.__init__` signature."""

  def test_auto_init_arg_list(self):

    @pg_members([
        ('x', pg_typing.Int(default=1)),
        ('y', pg_typing.Any()),
        ('z', pg_typing.List(pg_typing.Int())),
        (pg_typing.StrKey(), pg_typing.Str())
    ])
    class A(Object):
      pass

    signature = inspect.signature(A.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'x', 'y', 'z', 'kwargs'])

    self.assertEqual(
        signature.parameters['self'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['self'].annotation,
        inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['self'].default,
        inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['x'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['x'].annotation, int)
    self.assertEqual(signature.parameters['x'].default, 1)

    self.assertEqual(
        signature.parameters['y'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].default, pg_typing.MISSING_VALUE)

    self.assertEqual(
        signature.parameters['z'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(signature.parameters['z'].annotation, typing.List[int])
    self.assertEqual(signature.parameters['z'].default, pg_typing.MISSING_VALUE)

    self.assertEqual(
        signature.parameters['kwargs'].kind,
        inspect.Parameter.VAR_KEYWORD)
    self.assertEqual(signature.parameters['kwargs'].annotation, str)

  def test_user_specified_init_arg_list(self):

    @pg_members([
        ('x', pg_typing.Int(default=1)),
        ('y', pg_typing.Any()),
        ('z', pg_typing.List(pg_typing.Int())),
    ], init_arg_list=['y', '*z'])
    class B(Object):
      pass

    signature = inspect.signature(B.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'y', 'z', 'x'])

    self.assertEqual(
        signature.parameters['self'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['self'].annotation,
        inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['self'].default,
        inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['y'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].default, inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['z'].kind,
        inspect.Parameter.VAR_POSITIONAL)
    self.assertEqual(
        signature.parameters['z'].annotation, int)
    self.assertEqual(
        signature.parameters['z'].default, inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['x'].kind,
        inspect.Parameter.KEYWORD_ONLY)
    self.assertEqual(signature.parameters['x'].annotation, int)
    self.assertEqual(signature.parameters['x'].default, 1)

  def test_user_specified_init_arg_list_with_metadata(self):

    @pg_members([
        ('x', pg_typing.Int(default=1)),
        ('y', pg_typing.Any()),
        ('z', pg_typing.List(pg_typing.Int())),
    ], metadata=dict(init_arg_list=['y', '*z']))
    class B(Object):
      pass

    signature = inspect.signature(B.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'y', 'z', 'x'])

    self.assertEqual(
        signature.parameters['self'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['self'].annotation,
        inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['self'].default,
        inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['y'].kind,
        inspect.Parameter.POSITIONAL_OR_KEYWORD)
    self.assertEqual(
        signature.parameters['y'].annotation, inspect.Signature.empty)
    self.assertEqual(
        signature.parameters['y'].default, inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['z'].kind,
        inspect.Parameter.VAR_POSITIONAL)
    self.assertEqual(
        signature.parameters['z'].annotation, int)
    self.assertEqual(
        signature.parameters['z'].default, inspect.Signature.empty)

    self.assertEqual(
        signature.parameters['x'].kind,
        inspect.Parameter.KEYWORD_ONLY)
    self.assertEqual(signature.parameters['x'].annotation, int)
    self.assertEqual(signature.parameters['x'].default, 1)

  def test_custom_init(self):

    @pg_members([
        ('x', pg_typing.Int(default=1)),
        ('y', pg_typing.Any()),
        ('z', pg_typing.List(pg_typing.Int())),
    ], init_arg_list=['y', '*z'])
    class B(Object):
      pass

    class C(B):
      """Custom __init__."""

      @object_utils.explicit_method_override
      def __init__(self, a, b):
        super().__init__(b, x=a)

    signature = inspect.signature(C.__init__)
    self.assertEqual(
        list(signature.parameters.keys()), ['self', 'a', 'b'])

  def test_bad_init_arg_list(self):

    @pg_members([
        ('x', pg_typing.Int(default=1)),
        ('y', pg_typing.Any()),
        ('z', pg_typing.List(pg_typing.Int())),
    ], init_arg_list=['y', '*z'])
    class B(Object):
      pass

    with self.assertRaisesRegex(
        TypeError, 'Argument .* from `init_arg_list` is not defined.'):

      @pg_members([], init_arg_list=['a'])
      class D(B):  # pylint: disable=unused-variable
        pass

    with self.assertRaisesRegex(
        TypeError,
        'Variable positional argument .* should be declared with '
        '`pg.typing.List.*`'):

      @pg_members([], init_arg_list=['*y'])
      class E(B):  # pylint: disable=unused-variable
        pass

  def test_inferred(self):
    class A(Object):
      x: int
      y: str = inferred.ValueFromParentChain()

    # Okay: `A.y` is an inferred value.
    a = A(1)

    # Not okay: `A.y` is not yet available in its context.
    with self.assertRaisesRegex(
        AttributeError, '`y` is not found under its context'
    ):
      _ = a.y

    sd = Dict(x=a, y=Dict(z=1))
    self.assertIs(a.y, sd.y)

    # Clear context by reset a's parent.
    a.sym_setparent(None)
    with self.assertRaisesRegex(
        AttributeError, '`y` is not found under its context'
    ):
      _ = a.y

    # Test a custom inferred value.
    class ValueFromRedirectedKey(inferred.ValueFromParentChain):
      key: str

      @property
      def inference_key(self):
        return self.key

    sd = Dict(a='bar', b=Dict(x=a, y=ValueFromRedirectedKey('a')))

    # a.y is redirected to sd.a.
    self.assertEqual(a.y, 'bar')

    class B(Object):
      x: int = inferred.ValueFromParentChain()  # pylint: disable=no-value-for-parameter

    b = B()
    _ = Dict(a='bar', b=Dict(b=b, x=ValueFromRedirectedKey('a')))

    # b.x -> parent.x -> parent.parent.a
    self.assertEqual(b.x, 'bar')


class RebindTest(unittest.TestCase):
  """Dedicated tests for `pg.Dict.rebind`."""

  def test_rebind_with_kwargs(self):

    @pg_members([
        ('x', pg_typing.Any()),
        ('y', pg_typing.Any()),
    ])
    class A(Object):

      def _on_bound(self):
        super()._on_bound()
        self.z = self.x + self.y

    # Rebind using only kwargs.
    a = A(1, 2)
    self.assertEqual(a.z, 3)

    a.rebind(x=2, y=2)
    self.assertEqual(a, A(2, 2))
    self.assertEqual(a.z, 4)

    # Rebind using both update dict and kwargs.
    a = A(1, 2)
    a.rebind(Dict(x=2), x=3, y=3)
    self.assertEqual(a, A(3, 3))
    self.assertEqual(a.z, 6)

  def test_rebind_with_typing(self):

    @pg_members([
        ('x', pg_typing.Int(min_value=0)),
        ('y', pg_typing.Str(regex='foo.*')),
    ])
    class A(Object):
      pass

    a = A(0, 'foo')
    a.rebind(x=1, y='foo1')
    self.assertEqual(a, A(1, 'foo1'))

    with self.assertRaisesRegex(
        ValueError, 'does not match regular expression'):
      a.rebind(y='bar')

    with self.assertRaisesRegex(ValueError, '.* is out of range'):
      a.rebind(x=-1)

  def test_rebind_with_reset_default(self):

    @pg_members([
        ('a', pg_typing.Int(default=0)),
        ('b', pg_typing.Str()),
        ('c', pg_typing.Dict([
            ('x', pg_typing.Int(default=1)),
            ('y', pg_typing.Bool(default=False)),
        ]))
    ])
    class A(Object):
      pass

    a = A(1, 'foo', dict(x=0, y=True))

    # Reset the default value of `a` and `c`, and update `b`.
    a.rebind({
        'a': MISSING_VALUE,
        'b': 'bar',
        'c': MISSING_VALUE,
    })
    self.assertEqual(a, A(0, 'bar', dict(x=1, y=False)))

  def test_rebind_with_no_updates(self):

    @pg_members([
        ('x', pg_typing.Int(min_value=0)),
        ('y', pg_typing.Str(regex='foo.*')),
    ])
    class A(Object):

      def _on_change(self, field_updates):
        super()._on_change(field_updates)
        assert False

    a = A(1, 'foo')
    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind'):
      a.rebind()
    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind'):
      a.rebind(lambda k, v, p: v)
    a.rebind(x=1, y='foo', raise_on_no_change=False)

  def test_rebind_with_skipping_notification(self):

    @pg_members([
        ('x', pg_typing.Int(min_value=0)),
        ('y', pg_typing.Str(regex='foo.*')),
    ])
    class A(Object):

      def _on_change(self, field_updates):
        super()._on_change(field_updates)
        assert False

    a = A(1, 'foo')
    a.rebind(x=2, skip_notification=True)
    self.assertEqual(a, A(2, 'foo'))

  def test_rebind_without_notifying_parents(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):

      def _on_init(self):
        super()._on_init()
        self.num_updates = 0

      def _on_change(self, field_updates):
        super()._on_change(field_updates)
        self.num_updates += 1

    c = A(x=1)
    b = A(x=c)
    a = A(x=b)
    y = A(x=a)

    a.rebind({'x.x.x': 2}, notify_parents=False)
    self.assertEqual(c.num_updates, 1)
    self.assertEqual(b.num_updates, 1)
    self.assertEqual(a.num_updates, 1)
    self.assertEqual(y.num_updates, 0)

    a.rebind({'x.x.x': 3}, notify_parents=True)
    self.assertEqual(c.num_updates, 2)
    self.assertEqual(b.num_updates, 2)
    self.assertEqual(a.num_updates, 2)
    self.assertEqual(y.num_updates, 1)

    a.rebind(x=1, notify_parents=False)
    self.assertEqual(c.num_updates, 2)
    self.assertEqual(b.num_updates, 2)
    self.assertEqual(a.num_updates, 3)
    self.assertEqual(y.num_updates, 1)

  def test_rebind_with_fn(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    a = A([0, 1, A(2), A(A(3)), dict(x=A(4))])
    def increment(k, v, p):
      del k, p
      if isinstance(v, int):
        return v + 1
      return v
    a.rebind(increment)
    self.assertEqual(a, A([1, 2, A(3), A(A(4)), dict(x=A(5))]))

  def test_notify_on_change(self):

    @pg_members([
        ('x', pg_typing.Int())
    ])
    class A(Object):

      def _on_init(self):
        super()._on_init()
        self.num_changes = 0

      def _on_change(self, unused_updates):
        self.num_changes += 1

    a = A(x=1)
    with flags.notify_on_change(False):
      a.rebind(x=5)
      with flags.notify_on_change(True):
        a.rebind(x=6)
      a.rebind(x=7)
    self.assertEqual(a.num_changes, 1)

  def test_bad_rebind(self):

    @pg_members([
        ('x', pg_typing.Any())
    ])
    class A(Object):
      pass

    # Rebind is invalid on root object.
    with self.assertRaisesRegex(
        KeyError, 'Root key .* cannot be used in .*rebind.'):
      A(1).rebind({'': 1})

    # Rebind is invalid on non-symbolic object.
    with self.assertRaisesRegex(
        KeyError, 'Cannot rebind key .* is not a symbolic type.'):
      A(1).rebind({'x.y': 2})

    with self.assertRaisesRegex(
        ValueError, 'Argument \'path_value_pairs\' should be a dict.'):
      A(1).rebind(1)

    with self.assertRaisesRegex(
        ValueError, 'There are no values to rebind.'):
      A(1).rebind({})

    with self.assertRaisesRegex(
        KeyError, 'Key must be string type. Encountered 1'):
      A(1).rebind({1: 1})

    with self.assertRaisesRegex(
        ValueError, 'Required value is not specified.'):
      A(1).rebind(x=MISSING_VALUE)

    with self.assertRaisesRegex(
        TypeError, 'Rebinder function .* should accept 2 or 3 arguments'):
      A(1).rebind(lambda x: 1)


class EventsTest(unittest.TestCase):
  """Tests for symbolic events."""

  def test_on_change(self):
    object_updates = []

    @pg_members([
        ('x', pg_typing.Int(1)),
        ('y', pg_typing.Bool().noneable()),
        ('z', pg_typing.Str())
    ])
    class A(Object):

      def _on_change(self, field_updates):
        object_updates.append(field_updates)

    value_spec = pg_typing.Dict([
        ('a1', pg_typing.Int()),
        ('a2', pg_typing.Dict([
            ('b1', pg_typing.Dict([
                ('c1', pg_typing.List(pg_typing.Dict([
                    ('d1', pg_typing.Str('foo')),
                    ('d2', pg_typing.Bool(True)),
                    ('d3', pg_typing.Object(A))
                ])))
            ]))
        ]))
    ])
    root_updates = []

    def _onchange(field_updates):
      root_updates.append(field_updates)

    list_updates = []

    def _onchange_list(field_updates):
      list_updates.append(field_updates)

    child_dict_updates = []

    def _onchange_child(field_updates):
      child_dict_updates.append(field_updates)

    sd = Dict.partial(
        {
            'a2': {
                'b1': {
                    'c1':
                        List([
                            Dict(
                                d3=A.partial(),
                                allow_partial=True,
                                onchange_callback=_onchange_child)
                        ], allow_partial=True, onchange_callback=_onchange_list)
                }
            }
        },
        value_spec=value_spec,
        onchange_callback=_onchange)

    # There are no updates in object A.
    self.assertEqual(object_updates, [])

    # innermost Dict get updated after bind with List.
    self.assertEqual(
        child_dict_updates,
        [
            # Set default value from outer space (parent List) for field d1.
            {
                'd1':
                    base.FieldUpdate(
                        path=object_utils.KeyPath.parse('a2.b1.c1[0].d1'),
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                        old_value=MISSING_VALUE,
                        new_value='foo')
            },
            # Set default value from outer space (parent List) for field d2.
            {
                'd2':
                    base.FieldUpdate(
                        path=object_utils.KeyPath.parse('a2.b1.c1[0].d2'),
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                        old_value=MISSING_VALUE,
                        new_value=True)
            }
        ])

    # list get updated after bind with parent structures.
    self.assertEqual(list_updates, [{
        '[0].d1':
            base.FieldUpdate(
                path=object_utils.KeyPath.parse('a2.b1.c1[0].d1'),
                target=sd.a2.b1.c1[0],
                field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                old_value=MISSING_VALUE,
                new_value='foo')
    }, {
        '[0].d2':
            base.FieldUpdate(
                path=object_utils.KeyPath.parse('a2.b1.c1[0].d2'),
                target=sd.a2.b1.c1[0],
                field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                old_value=MISSING_VALUE,
                new_value=True)
    }])

    # There are no updates in root.
    self.assertEqual(root_updates, [])

    child_dict_updates = []
    list_updates = []

    sd.rebind({
        'a1': 1,
        'a2.b1.c1[0].d1': 'bar',
        'a2.b1.c1[0].d2': False,
        'a2.b1.c1[0].d3.z': 'foo',
    })

    # Inspect root object changes.
    self.assertEqual(len(root_updates), 1)
    self.assertEqual(len(root_updates[0]), 4)

    self.assertEqual(
        root_updates[0],
        {
            'a1': base.FieldUpdate(
                path=object_utils.KeyPath.parse('a1'),
                target=sd,
                field=sd.value_spec.schema['a1'],
                old_value=MISSING_VALUE,
                new_value=1,
            ),
            'a2.b1.c1[0].d1': base.FieldUpdate(
                path=object_utils.KeyPath.parse('a2.b1.c1[0].d1'),
                target=sd.a2.b1.c1[0],
                field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                old_value='foo',
                new_value='bar',
            ),
            'a2.b1.c1[0].d2': base.FieldUpdate(
                path=object_utils.KeyPath.parse('a2.b1.c1[0].d2'),
                target=sd.a2.b1.c1[0],
                field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                old_value=True,
                new_value=False,
            ),
            'a2.b1.c1[0].d3.z': base.FieldUpdate(
                path=object_utils.KeyPath.parse('a2.b1.c1[0].d3.z'),
                target=sd.a2.b1.c1[0].d3,
                field=sd.a2.b1.c1[0].d3.__class__.__schema__['z'],
                old_value=MISSING_VALUE,
                new_value='foo',
            ),
        },
    )

    # Inspect list node changes.
    self.assertEqual(
        list_updates,
        [
            # Root object rebind.
            {
                '[0].d1': base.FieldUpdate(
                    path=object_utils.KeyPath.parse('a2.b1.c1[0].d1'),
                    target=sd.a2.b1.c1[0],
                    field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                    old_value='foo',
                    new_value='bar',
                ),
                '[0].d2': base.FieldUpdate(
                    path=object_utils.KeyPath.parse('a2.b1.c1[0].d2'),
                    target=sd.a2.b1.c1[0],
                    field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                    old_value=True,
                    new_value=False,
                ),
                '[0].d3.z': base.FieldUpdate(
                    path=object_utils.KeyPath.parse('a2.b1.c1[0].d3.z'),
                    target=sd.a2.b1.c1[0].d3,
                    field=sd.a2.b1.c1[0].d3.__class__.__schema__['z'],
                    old_value=MISSING_VALUE,
                    new_value='foo',
                ),
            }
        ],
    )

    # Inspect leaf node changes.
    self.assertEqual(
        child_dict_updates,
        [
            # Root object rebind.
            {
                'd1':
                    base.FieldUpdate(
                        path=object_utils.KeyPath.parse('a2.b1.c1[0].d1'),
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d1'],
                        old_value='foo',
                        new_value='bar'),
                'd2':
                    base.FieldUpdate(
                        path=object_utils.KeyPath.parse('a2.b1.c1[0].d2'),
                        target=sd.a2.b1.c1[0],
                        field=sd.a2.b1.c1[0].value_spec.schema['d2'],
                        old_value=True,
                        new_value=False),
                'd3.z':
                    base.FieldUpdate(
                        path=object_utils.KeyPath.parse('a2.b1.c1[0].d3.z'),
                        target=sd.a2.b1.c1[0].d3,
                        field=sd.a2.b1.c1[0].d3.__class__.schema['z'],
                        old_value=MISSING_VALUE,
                        new_value='foo')
            }
        ])

  def test_on_change_notification_order(self):
    change_order = []

    @pg_members([
        (pg_typing.StrKey(), pg_typing.Any())
    ])
    class Node(Object):

      def _on_change(self, field_updates):
        change_order.append(self.sym_path)

    node = Node(
        b=Node(z=1, x=1),
        a=Node(y=1, x=Node(p=1), z=1),
        c=Node(z=Node(q=1), y=1),
        d=Node())
    node.rebind({
        'c.y': 2,
        'a.z': 2,
        'b.z': 2,
        'a.x.p': 2,
        'c.z.q': 2
    })
    self.assertEqual(change_order, [
        'c.z', 'c', 'b', 'a.x', 'a', ''
    ])

  def test_on_parent_change(self):

    class A(Object):

      def _on_parent_change(self, old_parent, new_parent):
        self.old_parent = old_parent
        self.new_parent = new_parent

    x = A()
    y = Dict(x=x)
    self.assertIsNone(x.old_parent)
    self.assertIs(x.new_parent, y)
    self.assertEqual(x.sym_path, 'x')

    y.x = A()
    self.assertIs(x.old_parent, y)
    self.assertIsNone(x.new_parent)
    self.assertEqual(x.sym_path, object_utils.KeyPath())

  def test_on_path_change(self):

    class A(Object):

      def _on_path_change(self, old_path, new_path):
        self.old_path = old_path
        self.new_path = new_path

    x = A()
    x.sym_setpath(object_utils.KeyPath('a'))
    self.assertEqual(x.old_path, object_utils.KeyPath())
    self.assertEqual(x.new_path, 'a')

    y = Dict(x=x)
    self.assertEqual(x.old_path, 'a')
    self.assertEqual(x.new_path, 'x')

    _ = Dict(y=y)
    self.assertEqual(x.old_path, 'x')
    self.assertEqual(x.new_path, 'y.x')


class TraverseTest(unittest.TestCase):
  """Tests for `pg.traverse` on symbolic Object."""

  def setUp(self):
    super().setUp()

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    self._v = [A(x={'y': A(x=0), 'z': 'foo'}), 1, 'bar']

  def visit_all(self, visited_keys):
    visited_keys[:] = []
    def _fn(k, v, p):
      del v, p
      visited_keys.append(str(k))
      return base.TraverseAction.ENTER
    return _fn

  def visit_all_implicit(self, visited_keys):
    visited_keys[:] = []
    def _fn(k, v, p):
      del v, p
      visited_keys.append(str(k))
    return _fn

  def stop_after(self, path, visited_keys):
    visited_keys[:] = []
    def _fn(k, v, p):
      del v, p
      visited_keys.append(str(k))
      if k == path:
        return base.TraverseAction.STOP
      return base.TraverseAction.ENTER
    return _fn

  def enter_if_shallower_than(self, depth, visited_keys):
    visited_keys[:] = []
    def _fn(k, v, p):
      del v, p
      visited_keys.append(str(k))
      if len(k) < depth:
        return base.TraverseAction.ENTER
      return base.TraverseAction.CONTINUE
    return _fn

  def test_visit_all(self):
    preorder_paths, postorder_paths = [], []
    ret = pg_traverse(
        self._v,
        self.visit_all(preorder_paths),
        self.visit_all(postorder_paths)
    )
    self.assertTrue(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[0].x.y',
        '[0].x.y.x',
        '[0].x.z',
        '[1]',
        '[2]',
    ])
    self.assertEqual(postorder_paths, [
        '[0].x.y.x',
        '[0].x.y',
        '[0].x.z',
        '[0].x',
        '[0]',
        '[1]',
        '[2]',
        '',
    ])

  def test_visit_all_implicit(self):
    preorder_paths, postorder_paths = [], []
    ret = pg_traverse(
        self._v,
        self.visit_all_implicit(preorder_paths),
        self.visit_all_implicit(postorder_paths)
    )
    self.assertTrue(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[0].x.y',
        '[0].x.y.x',
        '[0].x.z',
        '[1]',
        '[2]',
    ])
    self.assertEqual(postorder_paths, [
        '[0].x.y.x',
        '[0].x.y',
        '[0].x.z',
        '[0].x',
        '[0]',
        '[1]',
        '[2]',
        '',
    ])

  def test_stop_after_preorder(self):
    preorder_paths, postorder_paths = [], []
    ret = pg_traverse(
        self._v,
        self.stop_after('[0].x.y', preorder_paths),
        self.visit_all(postorder_paths))
    self.assertFalse(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[0].x.y',
    ])
    self.assertEqual(postorder_paths, ['[0].x.y', '[0].x', '[0]', ''])

  def test_enter_if_shallower_preorder(self):
    preorder_paths, postorder_paths = [], []
    ret = pg_traverse(
        self._v,
        self.enter_if_shallower_than(2, preorder_paths),
        self.visit_all(postorder_paths))
    self.assertTrue(ret)
    self.assertEqual(preorder_paths, [
        '',
        '[0]',
        '[0].x',
        '[1]',
        '[2]',
    ])
    self.assertEqual(postorder_paths, [
        '[0].x',
        '[0]',
        '[1]',
        '[2]',
        '',
    ])

  def test_default_preorder_and_postorder_visit_fn(self):
    ret = pg_traverse(self._v)
    self.assertTrue(ret)


class QueryTest(unittest.TestCase):
  """Tests for `pg.query` on symbolic objects."""

  def setUp(self):
    super().setUp()

    @pg_members([('x', pg_typing.Int())])
    class A(Object):
      pass

    @pg_members([
        ('a', pg_typing.Object(A)),
        ('y', pg_typing.Str()),
        ('z', pg_typing.Int())
    ])
    class B(Object):
      pass

    self._A = A   # pylint: disable=invalid-name
    self._B = B   # pylint: disable=invalid-name
    self._v = List([Dict(a=A(x=0), b=B(a=A(x=1), y='foo', z=2))])

  def test_query_without_constraint(self):
    self.assertEqual(pg_query(self._v), {'': self._v})

  def test_query_by_path(self):
    self.assertEqual(pg_query(self._v, r'.*y'), {'[0].b.y': 'foo'})

  def test_query_by_value(self):
    self.assertEqual(
        pg_query(self._v, where=lambda v: isinstance(v, int)), {
            '[0].a.x': 0,
            '[0].b.a.x': 1,
            '[0].b.z': 2
        })

  def test_query_by_path_and_value(self):
    self.assertEqual(
        pg_query(self._v, r'.*a', where=lambda v: isinstance(v, int)), {
            '[0].a.x': 0,
            '[0].b.a.x': 1,
        })

  def test_query_by_value_and_parent(self):
    where = lambda v, p: isinstance(v, int) and not isinstance(p, self._A)
    self.assertEqual(
        pg_query(self._v, where=where),
        {
            '[0].b.z': 2,
        })

  def test_query_with_enter_selected_flag(self):
    self.assertEqual(
        pg_query(self._v,
                 where=lambda v: isinstance(v, Object), enter_selected=True),
        {
            '[0].a': self._A(x=0),
            '[0].b': self._B(a=self._A(x=1), y='foo', z=2),
            '[0].b.a': self._A(x=1),
        })

  def test_query_with_k_v_selector(self):
    selector = lambda k, v: len(k) == 2 and isinstance(v, self._A)
    self.assertEqual(
        pg_query(self._v, custom_selector=selector),
        {'[0].a': self._A(x=0)})

  def test_query_with_k_v_p_selector(self):
    selector = lambda k, v, p: len(k) > 2 and isinstance(p, self._A) and v > 0
    self.assertEqual(
        pg_query(self._v, custom_selector=selector),
        {'[0].b.a.x': 1})

  def test_query_with_no_match(self):
    self.assertEqual(0, len(pg_query(self._v, r'xx')))

  def test_bad_query(self):
    with self.assertRaisesRegex(
        TypeError, 'Where function .* should accept 1 or 2 arguments'):
      pg_query(self._v, where=lambda: True)

    with self.assertRaisesRegex(
        TypeError, 'Custom selector .* should accept 2 or 3 arguments'):
      pg_query(self._v, custom_selector=lambda: True)

    with self.assertRaisesRegex(
        ValueError, '\'path_regex\' and \'where\' must be None when '
        '\'custom_selector\' is provided'):
      pg_query(self._v, path_regex=r'x', custom_selector=lambda: True)


class SymDescendantsTests(unittest.TestCase):
  """Tests for `sym_descendants`."""

  def setUp(self):
    super().setUp()

    @pg_members([
        ('x', pg_typing.Any()),
    ])
    class A(Object):
      pass

    self._a = A(dict(x=A([A(1)]), y=[A(2)]))

  def test_descendants_with_no_filter(self):
    a = self._a
    self.assertEqual(
        a.sym_descendants(),
        [
            a.x,
            a.x.x,
            a.x.x.x,
            a.x.x.x[0],
            a.x.x.x[0].x,
            a.x.y,
            a.x.y[0],
            a.x.y[0].x,
        ])

    self.assertEqual(
        a.sym_descendants(option=base.DescendantQueryOption.IMMEDIATE),
        [a.x])

    self.assertEqual(
        a.sym_descendants(option=base.DescendantQueryOption.LEAF),
        [
            a.x.x.x[0].x,
            a.x.y[0].x,
        ])

  def test_descendants_with_filter(self):
    a = self._a
    where = lambda x: isinstance(x, a.__class__)
    self.assertEqual(
        a.sym_descendants(where),
        [
            a.x.x,
            a.x.x.x[0],
            a.x.y[0],
        ])

    self.assertEqual(
        a.sym_descendants(where, base.DescendantQueryOption.IMMEDIATE),
        [
            a.x.x,
            a.x.y[0],
        ])

    self.assertEqual(
        a.sym_descendants(where, base.DescendantQueryOption.LEAF),
        [
            a.x.x.x[0],
            a.x.y[0],
        ])

    self.assertEqual(
        a.sym_descendants(
            where, base.DescendantQueryOption.IMMEDIATE, include_self=True),
        [a])

  def test_descendants_with_including_self(self):
    a = self._a
    self.assertEqual(
        a.sym_descendants(include_self=True),
        [
            a,
            a.x,
            a.x.x,
            a.x.x.x,
            a.x.x.x[0],
            a.x.x.x[0].x,
            a.x.y,
            a.x.y[0],
            a.x.y[0].x,
        ])

    self.assertEqual(
        a.sym_descendants(
            option=base.DescendantQueryOption.IMMEDIATE, include_self=True),
        [a])

    self.assertEqual(
        a.sym_descendants(
            option=base.DescendantQueryOption.LEAF, include_self=True),
        [
            a.x.x.x[0].x,
            a.x.y[0].x,
        ])


class SerializationTest(unittest.TestCase):
  """Dedicated tests for `pg.Object` serialization."""

  def setUp(self):
    super().setUp()

    @pg_members([
        ('w', pg_typing.Any()),
        ('x', pg_typing.Int(default=1)),
        ('y', pg_typing.Bool()),
    ])
    class A(Object):
      pass

    @pg_members([
        ('w', pg_typing.Str()),
        # Frozen field shall not be written.
        ('y', pg_typing.Bool().freeze(True)),
        ('z', pg_typing.Str().noneable()),
    ])
    class B(A):
      pass

    class X:

      def __init__(self, value: int):
        self.value = value

      def __eq__(self, other):
        return isinstance(other, X) and other.value == self.value

    @pg_members([
        ('w', pg_typing.Object(X)),
    ])
    class C(A):
      pass

    self._A = A   # pylint: disable=invalid-name
    self._B = B   # pylint: disable=invalid-name
    self._C = C   # pylint: disable=invalid-name
    self._X = X   # pylint: disable=invalid-name

  def test_standard_serialization(self):
    b = self._B('foo', 1)
    self.assertEqual(
        b.to_json_str(),
        '{"_type": "%s", "w": "foo", "x": 1, "z": null}'
        % self._B.__type_name__,
    )

  def test_serialization_with_json_convertible(self):

    class Y(object_utils.JSONConvertible):

      TYPE_NAME = 'Y'

      def __init__(self, value: int):
        self.value = value

      def __eq__(self, other):
        return isinstance(other, Y) and other.value == self.value

      def to_json(self, *args, **kwargs):
        return {
            '_type': self.TYPE_NAME,
            'value': self.value,
        }

      @classmethod
      def from_json(cls, json_dict, *args, **kwargs):
        return cls(json_dict.pop('value'))

    object_utils.JSONConvertible.register(Y.TYPE_NAME, Y)

    a = self._A(Y(1), y=True)
    self.assertEqual(base.from_json_str(a.to_json_str()), a)

  def test_serialization_with_partial_object(self):

    class P(Object):
      x: int

    class Q(Object):
      p: P
      y: str

    self.assertEqual(
        base.from_json_str(
            Q.partial(P.partial()).to_json_str(), allow_partial=True),
        Q.partial(P.partial()))

  def test_serialization_with_force_dict(self):

    class P(Object):
      x: int

    class Q(Object):
      p: P
      y: str

    self.assertEqual(
        base.from_json_str(Q(P(1), y='foo').to_json_str(), force_dict=True),
        {'p': {'x': 1}, 'y': 'foo'}
    )

  def test_serialization_with_converter(self):

    c = self._C(self._X(1), y=True)
    with self.assertRaisesRegex(
        ValueError, 'Cannot encode opaque object .* with pickle'):
      c.to_json_str()

    pg_typing.register_converter(self._X, int, convert_fn=lambda x: x.value)
    pg_typing.register_converter(int, self._X, convert_fn=self._X)

    self.assertEqual(
        c.to_json_str(),
        '{"_type": "%s", "w": 1, "x": 1, "y": true}' % self._C.__type_name__,
    )
    self.assertEqual(base.from_json_str(c.to_json_str()), c)

  def test_hide_default_values(self):
    b = self._B('foo', 1)
    self.assertEqual(
        b.to_json_str(hide_default_values=True),
        '{"_type": "%s", "w": "foo"}' % self._B.__type_name__,
    )

  def test_from_json(self):
    b = self._B('foo', 1)
    self.assertEqual(base.from_json(b.to_json()), b)
    self.assertEqual(base.from_json_str(b.to_json_str()), b)

  def test_non_serializable(self):

    class Z:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot encode opaque object .* with pickle'):
      base.to_json(self._A(w=Z(), y=True))

    with self.assertRaisesRegex(
        TypeError,
        'Type name \'.*\' is not registered with a .* subclass'):
      base.from_json_str('{"_type": "pyglove.core.symbolic.object_test.NotExisted", "a": 1}')

  def test_default_load_save_handler(self):
    flags.set_load_handler(None)
    flags.set_save_handler(None)

    @pg_members([
        ('a', pg_typing.Int()),
        ('b', pg_typing.List(pg_typing.Int()))
    ])
    class A(Object):
      pass

    tmp_dir = tempfile.gettempdir()

    # Test save/load in JSON.
    path = os.path.join(tmp_dir, 'subdir/a.json')
    base.save(A(a=1, b=[0, 1]), path)
    with open(path) as f:
      content = f.read()
    self.assertEqual(
        content, '{"_type": "%s", "a": 1, "b": [0, 1]}' % A.__type_name__
    )
    # Test save/load in TXT.
    path2 = os.path.join(tmp_dir, 'subdir/a.txt')
    base.save(A(a=1, b=[0, 1]), path2, file_format='txt')
    content = base.load(path2, file_format='txt')
    self.assertEqual(
        content, 'A(\n  a = 1,\n  b = [\n    0 : 0,\n    1 : 1\n  ]\n)')

    path3 = os.path.join(tmp_dir, 'subdir/b.txt')
    base.save('foo', path3, file_format='txt')
    self.assertEqual(base.load(path3, file_format='txt'), 'foo')

    # Test save/load in unsupported format.
    with self.assertRaisesRegex(ValueError, 'Unsupported `file_format`'):
      base.save(A(a=1, b=[0]), path2, file_format='bin')

    with self.assertRaisesRegex(ValueError, 'Unsupported `file_format`'):
      base.load(path2, file_format='bin')

    # Test tracking origin.
    with flags.track_origin():
      a = base.load(path)
    self.assertEqual(a, A(a=1, b=[0, 1]))
    self.assertEqual(a.sym_origin.source, path)
    self.assertEqual(a.sym_origin.tag, 'load')

  def test_custom_load_save_handler(self):
    repo = {}
    def _load(name):
      return repo[name]
    old_loader = flags.set_load_handler(_load)
    self.assertIsNone(old_loader)
    self.assertIs(flags.get_load_handler(), _load)

    def _save(value, name):
      repo[name] = value
    old_saver = flags.set_save_handler(_save)
    self.assertIsNone(old_saver)
    self.assertIs(flags.get_save_handler(), _save)

    # Test 'base.save/load'.
    base.save([1, 2, 3], 'foo')
    self.assertEqual(base.load('foo'), [1, 2, 3])

    # Test 'base.save/load'.
    @pg_members([
        ('x', pg_typing.Int(1)),
        ('y', pg_typing.Str())
    ])
    class A(Object):
      pass

    A(y='abc').save('bar')
    self.assertEqual(A.load('bar'), A(y='abc'))
    with self.assertRaisesRegex(
        TypeError, 'Value is not of type .*'):
      Dict.load('bar')

    # Test 'save/load' with empty save/load handler.
    with self.assertRaisesRegex(
        ValueError, '`load_handler` must be callable.'):
      flags.set_load_handler(1)

    with self.assertRaisesRegex(
        ValueError, '`save_handler` must be callable.'):
      flags.set_save_handler(1)

    flags.set_load_handler(None)
    flags.set_save_handler(None)


class FormatTest(unittest.TestCase):
  """Dedicated tests for `pg.Object.format`."""

  def setUp(self):
    super().setUp()

    @pg_members([
        ('x', pg_typing.Any(default=1), 'Field `x`.'),
        ('y', pg_typing.Any(), 'Field `y`.\nAny type.')
    ])
    class A(Object):
      pass

    self._a = A.partial([A(1, None), A('foo', dict(a=A(True, 1.0)))])

  def test_compact(self):
    self.assertEqual(
        self._a.format(compact=True),
        'A(x=[0: A(x=1, y=None), 1: A(x=\'foo\', y={a=A(x=True, y=1.0)})], '
        'y=MISSING_VALUE)')

  def test_compact_python_format(self):
    self.assertEqual(
        self._a.format(compact=True, python_format=True),
        'A(x=[A(x=1, y=None), A(x=\'foo\', y={\'a\': A(x=True, y=1.0)})], '
        'y=MISSING_VALUE)')

  def test_noncompact_with_inferred_value(self):

    class A(Object):
      x: typing.Any
      y: int

    self.assertEqual(
        A(x=1, y=inferred.ValueFromParentChain()).format(compact=False),
        inspect.cleandoc("""A(
            x = 1,
            y = ValueFromParentChain()
          )
        """),
    )
    # Use inferred but values are not inferrable yet.
    self.assertEqual(
        A(x=1, y=inferred.ValueFromParentChain()).format(
            compact=False, use_inferred=True),
        inspect.cleandoc("""A(
            x = 1,
            y = ValueFromParentChain()
          )
        """),
    )
    # Use inferred and values are inferrable.
    self.assertEqual(
        A(y=2, x=A(x=1, y=inferred.ValueFromParentChain())).format(
            compact=False, use_inferred=True),
        inspect.cleandoc("""A(
            x = A(
              x = 1,
              y = 2
            ),
            y = 2
          )
        """),
    )

  def test_noncompact_python_format(self):
    self.assertEqual(
        self._a.format(compact=False, verbose=False, python_format=True),
        inspect.cleandoc("""A(
          x=[
            A(
              x=1,
              y=None
            ),
            A(
              x='foo',
              y={
                'a': A(
                  x=True,
                  y=1.0
                )
              }
            )
          ],
          y=MISSING_VALUE(Any())
        )"""))

  def test_noncompact_nonverbose(self):
    self.assertEqual(
        self._a.format(compact=False, verbose=False),
        inspect.cleandoc("""A(
          x = [
            0 : A(
              x = 1,
              y = None
            ),
            1 : A(
              x = 'foo',
              y = {
                a = A(
                  x = True,
                  y = 1.0
                )
              }
            )
          ],
          y = MISSING_VALUE(Any())
        )"""))

  def test_noncompact_verbose(self):
    self.assertEqual(
        self._a.format(compact=False, verbose=True),
        inspect.cleandoc("""A(
          # Field `x`.
          x = [
            0 : A(
              # Field `x`.
              x = 1,
              # Field `y`.
              # Any type.
              y = None
            ),
            1 : A(
              # Field `x`.
              x = 'foo',
              # Field `y`.
              # Any type.
              y = {
                a = A(
                  # Field `x`.
                  x = True,
                  # Field `y`.
                  # Any type.
                  y = 1.0
                )
              }
            )
          ],
          # Field `y`.
          # Any type.
          y = MISSING_VALUE(Any())
        )"""))

  def test_noncompact_verbose_with_extra_blankline_for_field_docstr(self):
    self.assertEqual(
        self._a.format(
            compact=False, verbose=True, extra_blankline_for_field_docstr=True),
        inspect.cleandoc("""A(
          # Field `x`.
          x = [
            0 : A(
              # Field `x`.
              x = 1,

              # Field `y`.
              # Any type.
              y = None
            ),
            1 : A(
              # Field `x`.
              x = 'foo',

              # Field `y`.
              # Any type.
              y = {
                a = A(
                  # Field `x`.
                  x = True,

                  # Field `y`.
                  # Any type.
                  y = 1.0
                )
              }
            )
          ],

          # Field `y`.
          # Any type.
          y = MISSING_VALUE(Any())
        )"""))

  def test_noncompact_verbose_hide_default_and_missing_values(self):
    self.assertEqual(
        self._a.format(
            compact=False,
            verbose=True,
            hide_default_values=True,
            hide_missing_values=True),
        inspect.cleandoc("""A(
          # Field `x`.
          x = [
            0 : A(
              # Field `y`.
              # Any type.
              y = None
            ),
            1 : A(
              # Field `x`.
              x = 'foo',
              # Field `y`.
              # Any type.
              y = {
                a = A(
                  # Field `y`.
                  # Any type.
                  y = 1.0
                )
              }
            )
          ]
        )"""))


class Foo(Object):
  x: typing.List[typing.Dict[str, int]] = [dict(x=1)]
  y: typing.Dict[str, int]
  z: bool = True
  # Test forward reference.
  p: typing.Optional['Foo'] = None


class PickleTest(unittest.TestCase):

  def assert_pickle_correctness(self, v: Object) -> Object:
    payload = pickle.dumps(v)
    v2 = pickle.loads(payload)
    self.assertEqual(v, v2)
    self.assertEqual(v.sym_sealed, v2.sym_sealed)
    self.assertEqual(v.sym_partial, v2.sym_partial)
    self.assertEqual(v.accessor_writable, v2.accessor_writable)
    return v2

  def test_basic(self):
    self.assert_pickle_correctness(
        Foo([dict(x=2)], dict(x=1), p=Foo([], dict(a=2))))

  def test_sealed(self):
    self.assert_pickle_correctness(Foo([dict(x=2)], dict(x=1)).seal())

  def test_partial(self):
    self.assert_pickle_correctness(Foo.partial())


if __name__ == '__main__':
  unittest.main()
