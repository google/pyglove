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
import abc
import base64
import copy
import pickle
import typing
import unittest
from pyglove.core.symbolic import unknown_symbols
from pyglove.core.typing import inspect as pg_inspect
from pyglove.core.utils import json_conversion


class X:
  """Classes with nested classes for testing purpose."""

  class Y:

    class Z:

      @classmethod
      def class_method(cls):
        return cls.__name__

      @staticmethod
      def static_method():
        return 1

      def instance_method(self):
        return str(self)

  def __init__(self, x):
    self.x = x

  def __eq__(self, other):
    return isinstance(other, X) and self.x == other.x

  def __ne__(self, other):
    return not self.__eq__(other)


def bar():
  pass

T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')
T3 = typing.TypeVar('T3')
T4 = typing.TypeVar('T4')
T5 = typing.TypeVar('T5')


class G1(typing.Generic[T1]):
  pass


class G2(typing.Generic[T1, T2]):
  pass


class G3(typing.Generic[T1, T2, T3]):
  pass


class G4(typing.Generic[T1, T2, T3, T4]):
  pass


class G5(typing.Generic[T1, T2, T3, T4, T5]):
  pass


class JSONConvertibleTest(unittest.TestCase):
  """Tests for JSONConvertible type registry."""

  def test_registry(self):

    class A(json_conversion.JSONConvertible):

      @abc.abstractmethod
      def value(self):
        pass

    class B(A):

      def __init__(self, x):
        super().__init__()
        self.x = x

      def to_json(self):
        return self.__class__.to_json_dict({
            'x': self.x
        })

      def value(self):
        return self.x

    typename = lambda cls: f'{cls.__module__}.{cls.__qualname__}'

    # A is abstract.
    self.assertFalse(json_conversion.JSONConvertible.is_registered(typename(A)))
    self.assertTrue(json_conversion.JSONConvertible.is_registered(typename(B)))
    self.assertIs(
        json_conversion.JSONConvertible.class_from_typename(typename(B)), B)
    self.assertIn(
        (typename(B), B),
        list(json_conversion.JSONConvertible.registered_types()))

    class C(B):
      auto_register = False

    # Auto-register is off.
    self.assertFalse(json_conversion.JSONConvertible.is_registered(typename(C)))

    with self.assertRaisesRegex(
        KeyError, 'Type .* has already been registered with class .*'):
      json_conversion.JSONConvertible.register(typename(B), C)

    json_conversion.JSONConvertible.register(
        typename(B), C, override_existing=True)
    self.assertIn(
        (typename(B), C),
        list(json_conversion.JSONConvertible.registered_types()))

    # Test load_types_for_deserialization.
    class D(C):
      auto_register = False

    with self.assertRaisesRegex(TypeError, 'Cannot load class .*'):
      json_conversion.from_json(D(1).to_json())

    self.assertEqual(
        json_conversion.JSONConvertible._TYPE_REGISTRY._ondemand_registry_stack,
        []
    )
    with json_conversion.JSONConvertible.load_types_for_deserialization(A):
      with json_conversion.JSONConvertible.load_types_for_deserialization(
          D) as ondemand_registry:
        self.assertEqual(ondemand_registry, {'A': A, 'D': D})
        self.assertIsNotNone(json_conversion.from_json(D(1).to_json()))
    self.assertEqual(
        json_conversion.JSONConvertible._TYPE_REGISTRY._ondemand_registry_stack,
        []
    )

  def test_json_conversion(self):

    class T(json_conversion.JSONConvertible):

      def __init__(self, x=None):
        self.x = x

      def to_json(self, **kwargs):
        return T.to_json_dict(dict(x=(self.x, None)), exclude_default=True)

      def __eq__(self, other):
        return isinstance(other, T) and self.x == other.x

      def __ne__(self, other):
        return not self.__eq__(other)

    typename = lambda cls: f'{cls.__module__}.{cls.__qualname__}'
    json_value = json_conversion.to_json([(T(1), 2), {'y': T(3)}])
    self.assertEqual(json_value, [
        ['__tuple__', {'_type': typename(T), 'x': 1}, 2],
        {'y': {'_type': typename(T), 'x': 3}}
    ])
    self.assertEqual(json_conversion.from_json(json_value),
                     [(T(1), 2), {'y': T(3)}])

    # Omitting default values.
    json_value = json_conversion.to_json([(T(), 2), {'y': T(3)}])
    self.assertEqual(json_value, [
        ['__tuple__', {'_type': typename(T)}, 2],
        {'y': {'_type': typename(T), 'x': 3}}
    ])
    self.assertEqual(json_conversion.from_json(json_value),
                     [(T(), 2), {'y': T(3)}])

    # Test module alias.
    json_conversion.JSONConvertible.add_module_alias(T.__module__, 'mymodule')
    self.assertEqual(
        json_conversion.from_json({'_type': f'mymodule.{T.__qualname__}'}),
        T()
    )

  def assert_conversion_is(self, v):
    self.assertIs(json_conversion.from_json(json_conversion.to_json(v)), v)

  def assert_conversion_equal(self, v):
    self.assertEqual(json_conversion.from_json(json_conversion.to_json(v)), v)

  class CustomJsonConvertible(json_conversion.JSONConvertible):
    auto_register = False

    def __init__(self, x=None):
      self.x = x

    def to_json(self, **kwargs):
      return self.to_json_dict(
          dict(x=(self.x, None)), exclude_default=True
      )

    def __eq__(self, other):
      return isinstance(other, self.__class__) and self.x == other.x

    def __ne__(self, other):
      return not self.__eq__(other)

  def test_json_conversion_with_auto_import(self):
    json_dict = json_conversion.to_json(self.CustomJsonConvertible(1))
    with self.assertRaisesRegex(
        TypeError, 'Type name .* is not registered'):
      json_conversion.from_json(json_dict, auto_import=False)

    self.assertEqual(
        json_conversion.from_json(json_dict),
        self.CustomJsonConvertible(1)
    )

  def test_json_conversion_for_types(self):
    # Built-in types.
    self.assert_conversion_is(int)
    self.assert_conversion_is(bool)
    self.assert_conversion_is(float)
    self.assert_conversion_is(str)
    self.assert_conversion_is(list)
    self.assert_conversion_equal(list[int])
    self.assert_conversion_is(tuple)
    self.assert_conversion_equal(tuple[int, int])
    self.assert_conversion_is(dict)
    self.assert_conversion_equal(dict[str, int])
    self.assert_conversion_is(Exception)
    self.assert_conversion_is(type(None))
    self.assert_conversion_is(...)
    self.assert_conversion_equal(typing.Callable[[int, int], None])
    self.assert_conversion_equal(typing.Callable[..., None])

    # User types.
    self.assert_conversion_is(X)
    self.assert_conversion_is(X.Y)
    self.assert_conversion_is(X.Y.Z)

    # Local type.
    class B:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert local class .* to JSON.'):
      json_conversion.to_json(B)

    # Generic types.
    self.assert_conversion_is(G1[int])
    self.assert_conversion_is(G2[int, int])
    self.assert_conversion_is(G3[int, int, int])
    self.assert_conversion_is(G4[int, int, int, int])
    with self.assertRaisesRegex(
        NotImplementedError,
        'Cannot convert generic type with more than 4 type arguments'):
      json_conversion.to_json(G5[int, int, int, int, int])

  def test_json_conversion_for_annotations(self):
    self.assert_conversion_is(typing.Any)
    self.assert_conversion_is(typing.List)
    self.assert_conversion_is(typing.List[typing.List[int]])
    self.assert_conversion_is(typing.Annotated[int, 'abc'])
    self.assert_conversion_is(typing.Dict[str, typing.Any])
    # From Python 3.14, union no longer preserves `is` identity.
    self.assert_conversion_equal(typing.Union[int, str])
    self.assert_conversion_is(typing.Sequence[int])
    self.assert_conversion_is(typing.Set[int])
    self.assert_conversion_is(typing.FrozenSet[int])
    self.assert_conversion_is(typing.Mapping[int, str])
    self.assert_conversion_is(typing.MutableMapping[int, str])
    # Optional will be converted to Union[int, None]
    self.assert_conversion_equal(typing.Optional[int])

    with self.assertRaisesRegex(ValueError, 'Annotation cannot be converted'):
      json_conversion.to_json(typing.Literal)

  def test_json_conversion_for_functions(self):
    # Built-in functions.
    self.assert_conversion_is(print)
    self.assert_conversion_is(zip)
    self.assert_conversion_is(input)

    # User-defined functions.
    self.assert_conversion_is(bar)

    # Lambda function.
    s = lambda x, y=1: x + y
    s1 = json_conversion.from_json(json_conversion.to_json(s))
    self.assertTrue(pg_inspect.callable_eq(s1, s))
    self.assertEqual(s1(1), 2)
    self.assertEqual(s1(1, 2), 3)

    # Locally defined function.
    def baz(x, y=1):
      return x + y
    baz1 = json_conversion.from_json(json_conversion.to_json(baz))
    self.assertTrue(pg_inspect.callable_eq(baz1, baz))
    self.assertEqual(baz1(1), 2)
    self.assertEqual(baz1(1, 2), 3)

    with self.assertRaisesRegex(
        TypeError, 'Cannot load function .*'):
      json_conversion.from_json(
          {'_type': 'function', 'name': 'non_existent_function'}
      )
    self.assertEqual(
        json_conversion.from_json(
            {'_type': 'function', 'name': 'non_existent_function'},
            convert_unknown=True
        ),
        unknown_symbols.UnknownFunction('non_existent_function')
    )

  def test_json_conversion_for_methods(self):
    # Test class-level method.
    f = json_conversion.from_json(json_conversion.to_json(X.Y.Z.class_method))
    self.assertEqual(f(), 'Z')

    # Test static method.
    self.assert_conversion_is(X.Y.Z.static_method)

    with self.assertRaisesRegex(
        ValueError, 'Cannot convert instance method .* to JSON.'):
      json_conversion.to_json(X.Y.Z().instance_method)

    with self.assertRaisesRegex(
        TypeError, 'Cannot load method .*'):
      json_conversion.from_json(
          {'_type': 'method', 'name': 'non_existent_method'}
      )
    self.assertEqual(
        json_conversion.from_json(
            {'_type': 'method', 'name': 'non_existent_method'},
            convert_unknown=True
        ),
        unknown_symbols.UnknownMethod('non_existent_method')
    )

  def test_json_conversion_for_opaque_objects(self):
    # Default is enabled, round-trip should work.
    self.assert_conversion_equal(X(1))

    class LocalX:
      pass

    with self.assertRaisesRegex(
        ValueError, 'Cannot encode opaque object .* with pickle.'):
      json_conversion.to_json(LocalX())

    json_dict = json_conversion.to_json(X(1))
    json_dict['value'] = 'abc'
    with self.assertRaisesRegex(
        ValueError, 'Cannot decode opaque object with pickle.'
    ):
      json_conversion.from_json(json_dict)

  def test_json_conversion_convert_unknown(self):
    self.assertEqual(
        json_conversion.from_json([
            '__tuple__',
            1,
            {
                '_type': 'type',
                'name': 'Unknown type',
            },
            {
                '_type': 'Unknown type',
                'x': [{
                    '_type': 'Unknown type',
                }, {
                    '_type': 'function',
                    'name': 'builtins.print'
                }]
            }
        ], convert_unknown=True),
        (
            1,
            unknown_symbols.UnknownType('Unknown type'),
            unknown_symbols.UnknownTypedObject(
                type_name='Unknown type',
                x=[
                    unknown_symbols.UnknownTypedObject('Unknown type'),
                    print
                ]
            )
        )
    )

  def test_json_conversion_with_bad_types(self):
    # Bad tuple.
    with self.assertRaisesRegex(
        ValueError, 'Tuple should have at least one element besides .*'):
      json_conversion.from_json(['__tuple__'])

    # Unregistered type without auto_import.
    with self.assertRaisesRegex(
        TypeError, 'Type name .* is not registered with'
    ):
      json_conversion.from_json(
          {
              '_type': 'Unknown type',
              'x': [{
                  '_type': 'Unknown type',
              }]
          }, auto_import=False
      )

    # Type does not exist.
    with self.assertRaisesRegex(
        TypeError, 'Cannot load class .*'):
      json_conversion.from_json({'_type': '__main__.ABC'})

    with self.assertRaisesRegex(
        TypeError, 'Cannot load type .*'):
      json_conversion.from_json({'_type': 'type', 'name': '__main__.ABC'})

    # Type exist but not a JSONConvertible subclass.
    class A:
      pass

    json_conversion.JSONConvertible.register('__main__.A', A)
    with self.assertRaisesRegex(
        TypeError, '.* is not a `pg.JSONConvertible` subclass'):
      json_conversion.from_json({'_type': '__main__.A'})

  def test_json_conversion_with_sharing(self):

    class T(json_conversion.JSONConvertible):

      def __init__(self, x=None):
        self.x = x

      def to_json(self, **kwargs):
        return T.to_json_dict(dict(x=(self.x, None)), exclude_default=True)

    t = T(1)
    x = X(1)
    u = {'x': x}
    v = [u, t]
    y = dict(t=t, x=x, u=u, v=v)
    y_json = json_conversion.to_json(y)
    x_serialized = json_conversion._OpaqueObject(x).to_json()
    self.assertEqual(
        y_json,
        {
            '__context__': {
                'shared_objects': [
                    {
                        '_type': json_conversion._type_name(T),
                        'x': 1
                    },
                    x_serialized,
                    {
                        'x': {
                            '__ref__': 1
                        }
                    }
                ]
            },
            '__root__': {
                't': {
                    '__ref__': 0
                },
                'x': {
                    '__ref__': 1
                },
                'u': {
                    '__ref__': 2
                },
                'v': [
                    {
                        '__ref__': 2
                    },
                    {
                        '__ref__': 0
                    }
                ]
            }
        }
    )
    # Default is enabled, round-trip should work.
    y_prime = json_conversion.from_json(y_json)
    self.assertIs(y_prime['t'], y_prime['v'][1])
    self.assertIs(y_prime['u'], y_prime['v'][0])

  def test_json_conversion_with_sharing_convert_unknown(self):
    self.assertEqual(
        json_conversion.from_json(
            {
                '__context__': {
                    'shared_objects': [
                        {
                            '_type': 'type',
                            'name': '__main__.ABC',
                        },
                        {
                            '_type': '__main__.ABC',
                            'x': 1
                        }
                    ]
                },
                '__root__': [
                    {
                        '__ref__': 0
                    },
                    {
                        '__ref__': 1
                    },
                ]
            },
            convert_unknown=True
        ),
        [
            unknown_symbols.UnknownType('__main__.ABC'),
            unknown_symbols.UnknownTypedObject(
                type_name='__main__.ABC',
                x=1
            )
        ]
    )

  def test_opaque_object_not_in_registry(self):
    """_OpaqueObject must not be reachable via type registry."""
    # _OpaqueObject should NOT be auto-registered, preventing attackers from
    # crafting a JSON payload that triggers pickle.loads on untrusted data.
    opaque_typename = json_conversion._type_name(json_conversion._OpaqueObject)
    self.assertFalse(
        json_conversion.JSONConvertible.is_registered(opaque_typename),
        f'_OpaqueObject is registered under {opaque_typename!r}. '
        'This allows RCE via pickle deserialization from untrusted JSON.',
    )

  def test_opaque_object_rce_blocked_when_disabled(self):
    """Malicious JSON targeting _OpaqueObject must be rejected when disabled."""

    # Simulate an attacker's payload: a pickle bomb inside _OpaqueObject JSON.
    class _Canary:
      triggered = False

      def __reduce__(self):
        # If this runs, the attacker wins.
        _Canary.triggered = True
        return (int, (0,))

    malicious_payload = {
        '_type': 'pyglove.core.utils.json_conversion._OpaqueObject',
        'value': base64.encodebytes(pickle.dumps(_Canary())).decode('utf-8'),
    }
    # pickle.dumps calls __reduce__ during serialization, so reset the flag
    # to only detect execution during the deserialization (attack) path.
    _Canary.triggered = False
    # When opaque pickle is disabled, deserialization must be rejected.
    with json_conversion.enable_opaque_pickle(False):
      with self.assertRaises(TypeError):
        json_conversion.from_json(malicious_payload)
    self.assertFalse(
        _Canary.triggered,
        'Pickle payload was executed — RCE vulnerability is still present!',
    )

  def test_opaque_from_json_gate_when_disabled(self):
    """Direct _OpaqueObject.from_json must be gated when disabled."""
    x = X(1)
    opaque = json_conversion._OpaqueObject(x)
    json_value = opaque.to_json()
    with json_conversion.enable_opaque_pickle(False):
      with self.assertRaisesRegex(TypeError, 'disabled'):
        json_conversion._OpaqueObject.from_json(json_value)

  def test_opaque_from_json_works_by_default(self):
    """from_json works by default (backward compatible)."""
    x = X(1)
    opaque = json_conversion._OpaqueObject(x)
    json_value = opaque.to_json()
    result = json_conversion._OpaqueObject.from_json(json_value)
    self.assertEqual(result, x)

  def test_opaque_from_json_works_inside_allow_context(self):
    """from_json works when re-enabled inside a disallow scope."""
    x = X(1)
    opaque = json_conversion._OpaqueObject(x)
    json_value = opaque.to_json()
    with json_conversion.enable_opaque_pickle(False):
      with json_conversion.enable_opaque_pickle(True):
        result = json_conversion._OpaqueObject.from_json(json_value)
    self.assertEqual(result, x)

  def test_function_code_deserialization_gated_when_disabled(self):
    """Inline-code (marshal) function deser must be gated when disabled."""
    # A lambda serializes via the inline-`code` (marshal.dumps) path.
    fn = lambda x: x + 1
    json_value = json_conversion.to_json(fn)
    self.assertIn('code', json_value)
    # When opaque pickle is disabled, untrusted JSON must not be turned into an
    # executable function (the marshal.loads -> FunctionType RCE path).
    with json_conversion.enable_opaque_pickle(False):
      with self.assertRaisesRegex(TypeError, 'disabled'):
        json_conversion.from_json(json_value)

  def test_function_code_deserialization_works_by_default(self):
    """Inline-code function deser works by default (backward compatible)."""
    fn = lambda x: x + 1
    restored = json_conversion.from_json(json_conversion.to_json(fn))
    self.assertEqual(restored(3), 4)
    # Re-enabling inside a disabled scope works too.
    with json_conversion.enable_opaque_pickle(False):
      with json_conversion.enable_opaque_pickle(True):
        restored2 = json_conversion.from_json(json_conversion.to_json(fn))
    self.assertEqual(restored2(3), 4)

  def test_enable_opaque_pickle_restores_on_exception(self):
    """Flag must be restored even if the body raises."""
    self.assertTrue(json_conversion._opaque_pickle_enabled)
    try:
      with json_conversion.enable_opaque_pickle(False):
        self.assertFalse(json_conversion._opaque_pickle_enabled)
        raise RuntimeError('simulated crash')
    except RuntimeError:
      pass
    # Flag must be restored to True after exception.
    self.assertTrue(json_conversion._opaque_pickle_enabled)

  def test_enable_opaque_pickle_nested(self):
    """Nested context managers must restore correctly."""
    self.assertTrue(json_conversion._opaque_pickle_enabled)
    with json_conversion.enable_opaque_pickle(False):
      self.assertFalse(json_conversion._opaque_pickle_enabled)
      with json_conversion.enable_opaque_pickle(True):
        self.assertTrue(json_conversion._opaque_pickle_enabled)
      # After inner allow exits, flag should be False (outer disallow active).
      self.assertFalse(json_conversion._opaque_pickle_enabled)
    # After outer disallow exits, flag must be True.
    self.assertTrue(json_conversion._opaque_pickle_enabled)

  def test_opaque_rce_blocked_via_auto_import_when_disabled(self):
    """auto_import must NOT bypass the pickle gate when disabled."""
    x = X(1)
    json_dict = json_conversion.to_json(x)
    # from_json mutates the dict in-place (pops _type), so use copies.
    # When disabled, even with auto_import=True (default), must fail.
    with json_conversion.enable_opaque_pickle(False):
      with self.assertRaises(TypeError):
        json_conversion.from_json(copy.deepcopy(json_dict))
    # By default (enabled), it works.
    result = json_conversion.from_json(copy.deepcopy(json_dict))
    self.assertEqual(result, x)

  def test_opaque_to_json_always_works(self):
    """Serialization must never be gated."""
    # to_json must work even when deserialization is disabled.
    with json_conversion.enable_opaque_pickle(False):
      x = X(1)
      json_dict = json_conversion.to_json(x)
      self.assertIn('_type', json_dict)
      opaque_typename = json_conversion._type_name(
          json_conversion._OpaqueObject
      )
      self.assertEqual(json_dict['_type'], opaque_typename)
      self.assertIn('value', json_dict)

  def test_opaque_rce_blocked_with_nested_payload(self):
    """Nested malicious _OpaqueObject in a list must be blocked when disabled."""
    nested_payload = [
        1,
        'safe',
        {
            '_type': 'pyglove.core.utils.json_conversion._OpaqueObject',
            'value': base64.encodebytes(pickle.dumps(42)).decode('utf-8'),
        },
    ]
    with json_conversion.enable_opaque_pickle(False):
      with self.assertRaises(TypeError):
        json_conversion.from_json(nested_payload)

  def test_opaque_rce_blocked_with_dict_payload(self):
    """_OpaqueObject inside a dict value must be blocked when disabled."""
    dict_payload = {
        'safe_key': 'safe_value',
        'malicious': {
            '_type': 'pyglove.core.utils.json_conversion._OpaqueObject',
            'value': base64.encodebytes(pickle.dumps(42)).decode('utf-8'),
        },
    }
    with json_conversion.enable_opaque_pickle(False):
      with self.assertRaises(TypeError):
        json_conversion.from_json(dict_payload)


if __name__ == '__main__':
  unittest.main()
