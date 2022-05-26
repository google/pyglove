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
"""Tests for pyglove.patching."""

import unittest
from pyglove.core import patching
from pyglove.core import symbolic
from pyglove.core import typing as schema


class ConditionalPatchingTest(unittest.TestCase):
  """Conditional patching test."""

  def testPatchOnKey(self):
    """Test `patching.patch_on_key`."""
    d = symbolic.Dict(a=1, b={'a': 2, 'b': 1})
    patching.patch_on_key(d, 'a', 3)
    self.assertEqual(d, {'a': 3, 'b': {'a': 3, 'b': 1}})

    patching.patch_on_key(d, 'a', value_fn=lambda v: v + 1)
    self.assertEqual(d, {'a': 4, 'b': {'a': 4, 'b': 1}})

    with self.assertRaisesRegex(
        ValueError, 'Either `value` or `value_fn` should be specified'):
      patching.patch_on_key(d, 'a', value=1, value_fn=lambda v: v + 1)

  def testPatchOnPath(self):
    """Test `patching.patch_on_path`."""
    d = symbolic.Dict(a=1, b={'a': 2, 'b': 1})
    patching.patch_on_path(d, '.+b', 3)
    self.assertEqual(d, {'a': 1, 'b': {'a': 2, 'b': 3}})

    patching.patch_on_path(d, '.*a', value_fn=lambda v: v + 1)
    self.assertEqual(d, {'a': 2, 'b': {'a': 3, 'b': 3}})

  def testPatchOnValue(self):
    """Test `patching.patch_on_value`."""
    d = symbolic.Dict(a=1, b={'a': 2, 'b': 1})
    patching.patch_on_value(d, 1, 3)
    self.assertEqual(d, {'a': 3, 'b': {'a': 2, 'b': 3}})

    patching.patch_on_value(d, 2, value_fn=lambda v: v * 2)
    self.assertEqual(d, {'a': 3, 'b': {'a': 4, 'b': 3}})

  def testPatchOnType(self):
    """Test `patching.patch_on_type`."""
    d = symbolic.Dict(a='abc', b={'a': 2, 'b': 'def'})
    patching.patch_on_type(d, str, 'foo')
    self.assertEqual(d, {'a': 'foo', 'b': {'a': 2, 'b': 'foo'}})

    patching.patch_on_type(d, int, value_fn=lambda v: v * 2)
    self.assertEqual(d, {'a': 'foo', 'b': {'a': 4, 'b': 'foo'}})

  def testPatchOnMember(self):
    """Test `patching.patch_on_member`."""

    @symbolic.members([
        ('x', schema.Int()),
        ('y', schema.Int()),
    ])
    class A(symbolic.Object):
      pass

    d = symbolic.Dict(a=A(x=1, y=2), x=1)
    patching.patch_on_member(d, A, 'x', 2)
    self.assertEqual(d, {'a': A(x=2, y=2), 'x': 1})

    patching.patch_on_member(d, A, 'y', value_fn=lambda v: v * 2)
    self.assertEqual(d, {'a': A(x=2, y=4), 'x': 1})


class PatcherTest(unittest.TestCase):
  """Patcher test."""

  def testPatch(self):
    """Test patch."""
    @patching.patcher([
        ('v', schema.Int())
    ])
    def p1(unused_src, k, v):
      return {k: v}

    @patching.patcher()
    def p2(unused_src):  # pylint: disable=unused-variable
      return {'': symbolic.Dict()}

    @patching.patcher()
    def p3(unused_src):
      def validate(x):
        if 'a' not in x:
          raise ValueError('Key `a` does not exist.')
      return {}, validate

    # Patch with a single rule.
    self.assertEqual(
        patching.patch(symbolic.Dict(x=1, y=2), {'x': 0, 'z': 3}),
        symbolic.Dict(x=0, y=2, z=3))

    # Patch a list of rules without replacement.
    self.assertEqual(
        patching.patch(symbolic.Dict(), [
            p1(k='a', v=1),  # pylint: disable=no-value-for-parameter
            'p1?b&2',
            'p1?c&v=3',
            'p1?k=d&v=4',
            'p1?a&2'
        ]),
        {
            'a': 2,
            'b': 2,
            'c': 3,
            'd': 4
        })

    # Patch a list of rules with replacement.
    self.assertEqual(
        patching.patch(symbolic.Dict(), [
            p1(k='a', v=1),  # pylint: disable=no-value-for-parameter
            'p1?b&2',
            'p1?c&v=3',
            'p2',
            'p1?k=d&v=4',
            'p1?a&2'
        ]),
        {
            'a': 2,
            'd': 4
        })

    # Patch with a rebind dict as the patching rule.
    self.assertEqual(
        patching.patch(symbolic.Dict(a=1, b=2), {'a': 0, 'c': 3}),
        symbolic.Dict(a=0, b=2, c=3))

    # Patch with a rebind function as the patching rule.
    self.assertEqual(
        patching.patch(
            symbolic.Dict(a=1),
            lambda k, v, p: (v + 1) if isinstance(v, int) else v),
        symbolic.Dict(a=2))

    self.assertEqual(
        patching.patch(symbolic.Dict(a=1), [
            lambda k, v: v
        ]),
        symbolic.Dict(a=1))

    # Patch with a patcher with validator.
    self.assertEqual(
        patching.patch(symbolic.Dict(), [
            p1(k='a', v=1),  # pylint: disable=no-value-for-parameter
            'p1?b&2',
            'p3'
        ]),
        {
            'a': 1,
            'b': 2,
        })

    with self.assertRaisesRegex(
        ValueError, 'Key `a` does not exist.'):
      _ = patching.patch(symbolic.Dict(), [
          'p1?b&2',
          p3()  # pylint: disable=no-value-for-parameter
      ])

    # Nothing to patch.
    self.assertEqual(
        patching.patch(symbolic.Dict(), []), {})

    # Bad patch input.
    with self.assertRaisesRegex(
        TypeError, 'Patching rule .* should be a dict of path to values'):
      patching.patch(symbolic.Dict(), [1])

  def testPatcher(self):
    """Test patcher."""
    # Test patcher without symbolic arguments.
    def assert_patch_equal(src, patcher_or_uri, expected_value):
      if isinstance(patcher_or_uri, patching.Patcher):
        patcher = patcher_or_uri
      else:
        patcher = patching.from_uri(patcher_or_uri)
      self.assertIsInstance(patcher, patching.Patcher)
      dst = patcher.patch(src)
      self.assertIs(src, dst)
      self.assertEqual(dst, expected_value)

    @patching.patcher()
    def set_value1(unused_src, k, v='0'):
      return {
          k: v
      }
    assert_patch_equal(symbolic.Dict(), set_value1(k='a'), {'a': '0'})  # pylint: disable=not-callable, no-value-for-parameter
    assert_patch_equal(symbolic.Dict(), 'set_value1?a&1', {'a': '1'})
    assert_patch_equal(symbolic.Dict(), 'set_value1?a&v=1', {'a': '1'})
    assert_patch_equal(symbolic.Dict(), 'set_value1?k=a&v=1', {'a': '1'})
    self.assertIn('set_value1', patching.patcher_names())

    with self.assertRaisesRegex(
        TypeError, 'The 1st argument of .* must be a symbolic type'):
      set_value1(k='a').patch(1)  # pylint: disable=not-callable, no-value-for-parameter

    # Test patcher with symbolic arguments.
    @patching.patcher([
        ('k', schema.Str()),
        ('v', schema.Int())
    ])
    def set_value2(unused_src, k, v):
      return {
          k: v
      }
    assert_patch_equal(symbolic.Dict(), set_value2(k='a', v=1), {'a': 1})  # pylint: disable=no-value-for-parameter
    assert_patch_equal(symbolic.Dict(), 'set_value2?a&1', {'a': 1})
    assert_patch_equal(symbolic.Dict(), 'set_value2?a&v=1', {'a': 1})
    assert_patch_equal(symbolic.Dict(), 'set_value2?k=a&v=1', {'a': 1})
    self.assertIn('set_value2', patching.patcher_names())

    # Test patcher with replacing source value:
    @patching.patcher([
        ('v', schema.Int())
    ])
    def set_value3(unused_src, v):
      return {'': v}
    self.assertEqual(
        set_value3(v=1).patch(symbolic.Dict()), 1)   # pylint: disable=not-callable, no-value-for-parameter
    self.assertEqual(
        patching.from_uri('set_value3?1').patch(symbolic.Dict()), 1)
    self.assertIn('set_value3', patching.patcher_names())

    # Test patcher with returning rebind fn.
    @patching.patcher()
    def increment(unused_src):
      return lambda k, v: (v + 1) if isinstance(v, int) else v

    self.assertEqual(
        increment().patch(symbolic.Dict(a=1, b=2)), symbolic.Dict(a=2, b=3))  # pylint: disable=not-callable, no-value-for-parameter

    @patching.patcher()
    def no_change(unused_src):
      return lambda k, v: v

    self.assertEqual(
        no_change().patch(symbolic.Dict(a=1, b=2)), symbolic.Dict(a=1, b=2))  # pylint: disable=not-callable, no-value-for-parameter

    # Test patcher with composition of rules.
    @patching.patcher()
    def compound(unused_src):
      return [
          set_value2(k='a', v=1),   # pylint: disable=no-value-for-parameter
          'set_value2?k=b&v=2',
          {
              'c': 3
          }
      ]
    self.assertEqual(
        compound().patch(symbolic.Dict(x=1)), symbolic.Dict(x=1, a=1, b=2, c=3))  # pylint: disable=no-value-for-parameter

    # Test patcher with validation logic.
    @patching.patcher()
    def set_value_with_validate(unused_src):
      def validate(x):
        if 'a' not in x:
          raise ValueError('Key `a` does not exist.')
      return {'a': 1}, validate

    p = set_value_with_validate()  # pylint: disable=not-callable, no-value-for-parameter
    x = p.patch(symbolic.Dict())
    p.validate(x)
    del x['a']
    with self.assertRaisesRegex(
        ValueError, 'Key `a` does not exist.'):
      p.validate(x)

    # Bad patcher function.
    @patching.patcher()
    def set_value4(unused_src):
      return 1

    with self.assertRaisesRegex(
        TypeError,
        'Patching rule .* should be a dict of path to values'):
      set_value4().patch(symbolic.Dict())  # pylint: disable=not-callable, no-value-for-parameter

    # Bad patcher argument spec.
    with self.assertRaisesRegex(
        TypeError,
        '.* cannot be used for constraining Patcher target.'):
      @patching.patcher([
          ('unused_src', schema.Int()),
      ])   # pylint: disable=unused-variable
      def set_value5(unused_src, x):
        del x

    # Bad patcher argument spec.
    with self.assertRaisesRegex(
        TypeError,
        '.* cannot be used for constraining Patcher argument'):
      @patching.patcher([
          ('x', schema.Enum(1, [1, 2]))
      ])   # pylint: disable=unused-variable
      def set_value6(unused_src, x=1):
        del x

    # Bad patcher without target argument.
    with self.assertRaisesRegex(
        TypeError,
        r'Patcher function should have at least 1 argument as patching target'):
      @patching.patcher()
      def set_value7():  # pylint: disable=unused-variable
        return {'': 1}

    # Bad patcher with wrong return type for validator.
    @patching.patcher()
    def set_value8(unused_src):
      return {'': 1}, 1

    with self.assertRaisesRegex(
        TypeError,
        r'The validator returned from patcher .* is not callable'):
      set_value8().patch(symbolic.Dict())  # pylint: disable=no-value-for-parameter

  def testIsPatcherParameterSpec(self):
    """Test for `patching.is_patcher_parameter_spec`."""
    def assert_argument_spec(value_spec):
      self.assertTrue(patching._is_patcher_parameter_spec(value_spec))

    def assert_not_argument_spec(value_spec):
      self.assertFalse(patching._is_patcher_parameter_spec(value_spec))

    assert_argument_spec(schema.Any())
    assert_argument_spec(schema.Int())
    assert_argument_spec(schema.Float())
    assert_argument_spec(schema.Bool())
    assert_argument_spec(schema.Str())
    assert_argument_spec(schema.Enum('a', ['a', 'b']))
    assert_argument_spec(schema.Enum('a', ['a', 'b']))
    assert_argument_spec(schema.List(schema.Int()))
    assert_argument_spec(schema.List(schema.Enum('a', ['a', 'b'])))

    assert_not_argument_spec(schema.Dict())
    assert_not_argument_spec(schema.Type(Exception))
    assert_not_argument_spec(schema.Object(ValueError))
    assert_not_argument_spec(schema.Enum(1, [0, 1]))
    assert_not_argument_spec(schema.List(schema.List(schema.Int())))
    assert_not_argument_spec(schema.Union([schema.Int(), schema.Float()]))

  def testFromUri(self):
    """Test `patching.from_uri`."""
    @patching.patcher([
        ('x', schema.Int())
    ], name='my_patcher')
    def foo(src, x, y='bar'):
      del src, x, y

    self.assertEqual(patching.from_uri('my_patcher?1'), foo(x=1))    # pylint: disable=not-callable, no-value-for-parameter
    self.assertEqual(patching.from_uri('my_patcher?x=1'), foo(x=1))  # pylint: disable=not-callable, no-value-for-parameter

    with self.assertRaisesRegex(
        KeyError, 'Patcher .* is not registered'):
      patching.from_uri('foo')

    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to int'
        r'.*Patcher=.*foo\', Argument=\'x\''):
      patching.from_uri('my_patcher?x=a').patch(symbolic.Dict())

    with self.assertRaisesRegex(
        KeyError,
        r'Too many positional arguments are provided.'):
      patching.from_uri('my_patcher?1&2&3').patch(symbolic.Dict())

    with self.assertRaisesRegex(
        TypeError,
        r'foo\(\) (missing 1 required positional argument|'
        'takes at least 2 arguments)'):
      patching.from_uri('my_patcher').patch(symbolic.Dict())

  def testParseUri(self):
    """Test `patching.parse_uri`."""
    def test_parse(uri, expected_name, expected_args, expected_kwargs):
      name, args, kwargs = patching.parse_uri(uri)
      self.assertEqual(name, expected_name)
      self.assertEqual(args, expected_args)
      self.assertEqual(kwargs, expected_kwargs)

    test_parse('foo', 'foo', [], {})
    test_parse('foo?', 'foo', [''], {})
    test_parse('foo?1', 'foo', ['1'], {})
    test_parse('foo?1&a:b&c', 'foo', ['1', 'a:b', 'c'], {})
    test_parse('foo?a=1', 'foo', [], {'a': '1'})
    test_parse('foo?a=1&b=2', 'foo', [], {'a': '1', 'b': '2'})
    test_parse('foo?1:2:3&b=2', 'foo', ['1:2:3'], {'b': '2'})

    with self.assertRaisesRegex(
        ValueError,
        '.* is not a valid Patcher name.'):
      patching.parse_uri('1foo')

    with self.assertRaisesRegex(
        ValueError,
        'Invalid argument specification: a-1=2'):
      patching.parse_uri('foo?a-1=2')

    with self.assertRaisesRegex(
        ValueError,
        'Positional argument should be provided before keyword arguments'):
      patching.parse_uri('foo?a=1&2')

  def testParseArgs(self):
    """Test `patching.parse_args`."""
    signature = schema.Signature(
        schema.CallableType.FUNCTION, 'foo', '__main__',
        [
            schema.Argument('src', schema.Any()),
            schema.Argument('x', schema.Int()),
            schema.Argument('y', schema.List(
                schema.Float(min_value=0.0, max_value=1.0)))
        ])
    args, kwargs = patching.parse_args(signature, ['0'], {'y': '0.1:0.5'})
    self.assertEqual(args, [0])
    self.assertEqual(kwargs, {'y': [0.1, 0.5]})

    with self.assertRaisesRegex(
        KeyError, 'Unexpected argument'):
      patching.parse_args(signature, [], {'z': '0.1'})

  def testParseArg(self):
    """Test `patching.parse_arg`."""
     # Test float.
    def parse_arg(value_spec, arg_str):
      return patching.parse_arg('__main__.patcher_a', 'x', value_spec, arg_str)
    self.assertTrue(parse_arg(schema.Bool(), 'True'))
    self.assertTrue(parse_arg(schema.Bool(), 'true'))
    self.assertTrue(parse_arg(schema.Bool(), 'yes'))
    self.assertTrue(parse_arg(schema.Bool(), '1'))
    self.assertFalse(parse_arg(schema.Bool(), 'False'))
    self.assertFalse(parse_arg(schema.Bool(), 'false'))
    self.assertFalse(parse_arg(schema.Bool(), 'no'))
    self.assertFalse(parse_arg(schema.Bool(), '0'))
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to bool. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(schema.Bool(), 'a')

    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to float. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(schema.Float(), 'a')

    with self.assertRaisesRegex(
        ValueError,
        'Value -1.0 is out of range .*path=__main__.patcher_a.x'):
      parse_arg(schema.Float(min_value=0.), '-1.0')

    # Test int.
    self.assertEqual(parse_arg(schema.Int(), '1'), 1)
    self.assertIsNone(parse_arg(schema.Int().noneable(), 'None'))
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to int. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(schema.Int(), 'a')

    with self.assertRaisesRegex(
        ValueError,
        'Value -1 is out of range .*path=__main__.patcher_a.x'):
      parse_arg(schema.Int(min_value=0), '-1')

    # Test float.
    self.assertEqual(parse_arg(schema.Float(), '1'), 1.0)
    self.assertIsNone(parse_arg(schema.Float().noneable(), 'None'))
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to float. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(schema.Float(), 'a')
    with self.assertRaisesRegex(
        ValueError, 'Value -1.0 is out of range'):
      parse_arg(schema.Float(min_value=0.), '-1.0')

    # Test string.
    self.assertEqual(parse_arg(schema.Str(), '1'), '1')
    self.assertEqual(parse_arg(schema.Str(), '"1"'), '1')
    self.assertIsNone(parse_arg(schema.Str().noneable(), 'none'))
    self.assertEqual(parse_arg(schema.Str().noneable(), '"none"'), 'none')

    with self.assertRaisesRegex(
        ValueError,
        'String \'bar\' does not match regular expression '
        '.*path=__main__.patcher_a.x'):
      parse_arg(schema.Str(regex='foo.*'), 'bar')

    with self.assertRaisesRegex(
        ValueError,
        'Unmatched quote for string value: \'"1\'.*'):
      parse_arg(schema.Str(), '"1')

    with self.assertRaisesRegex(
        ValueError,
        'Unmatched quote for string value: \'1"\'.*'):
      parse_arg(schema.Str(), '1"')

    # Test enum.
    self.assertEqual(parse_arg(schema.Enum('a', ['a', 'b']), 'b'), 'b')
    with self.assertRaisesRegex(
        ValueError,
        r'.* cannot be used for Patcher argument. '
        r'Only Enum of string type can be used.'):
      parse_arg(schema.Enum(1, [1, 2]), '1')

    # Test list.
    self.assertEqual(parse_arg(schema.List(schema.Int()), ''), [])
    self.assertEqual(parse_arg(schema.List(schema.Int()), '1'), [1])
    self.assertEqual(parse_arg(schema.List(schema.Int()), '1:2'),
                     [1, 2])
    self.assertEqual(parse_arg(schema.List(schema.Bool()), 'yes:no'),
                     [True, False])
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to int. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\[0\]\'\)'):
      parse_arg(schema.List(schema.Int()), 'a:b')

    # Test other types.
    with self.assertRaisesRegex(
        ValueError,
        r'Dict\(\) cannot be used for Patcher argument'):
      parse_arg(schema.Dict(), '{}')

  def testAllowRepeatedPatcherRegistration(self):
    """Test `patching.allow_repeated_patcher_registration`."""
    patching.patcher()

    patching.allow_repeated_patcher_registration()


@patching.patcher([
    ('value', schema.Int())
])
def update_a(unused_src, value):
  return {'a': value}


class ObjectFactoryTest(unittest.TestCase):
  """ObjectFactory test."""

  def testBaseValueOnly(self):
    """Test creating trainer from base experiment."""
    # Create factory using base value in object form.
    value1 = patching.object_factory(symbolic.Dict, symbolic.Dict(a=1))()
    self.assertEqual(value1, {'a': 1})

    # Create factory using base value in callable form.
    value2 = patching.object_factory(
        symbolic.Dict, lambda: symbolic.Dict(a=1))()
    self.assertEqual(value2, value1)

    # Create factory using base value in file form.
    file_db = {}
    def save_handler(v, filepath):
      file_db[filepath] = v

    def load_handler(filepath):
      return file_db[filepath]

    old_save_handler = symbolic.set_save_handler(save_handler)
    old_load_handler = symbolic.set_load_handler(load_handler)

    filepath = 'myfile.json'
    value1.save(filepath)
    value3 = patching.object_factory(symbolic.Dict, filepath)()
    self.assertEqual(value3, value1)

    with self.assertRaisesRegex(
        TypeError,
        '.* is neither an instance of .*, nor a factory or a path '
        'of JSON file that produces an instance of .*'):
      patching.object_factory(symbolic.Dict, symbolic.List())()

    symbolic.set_save_handler(old_save_handler)
    symbolic.set_load_handler(old_load_handler)

  def testFactoryWithPatchers(self):
    """Test base value + patchers."""
    value = patching.object_factory(
        symbolic.Dict, symbolic.Dict(a=1),
        ['update_a?value=2'])()
    self.assertEqual(value, {'a': 2})

  def testFactoryWithParamsOverride(self):
    """Test base value + params override."""
    # Using dict for `params_override`.
    value = patching.object_factory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override={
            'a': {
                'x': 2
            }
        })()
    self.assertEqual(value, {'a': {'x': 2, 'y': 2}})

    # Using flattened dict for `params_override`.
    value = patching.object_factory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override={
            'a.x': 2
        })()
    self.assertEqual(value, {'a': {'x': 2, 'y': 2}})

    # Using string for `params_override`.
    value = patching.object_factory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override='{"a": {"x": 2}}')()
    self.assertEqual(value, {'a': {'x': 2, 'y': 2}})

    # Using flattened dict string for `params_override`.
    value = patching.object_factory(
        symbolic.Dict, symbolic.Dict(a={'x': 1, 'y': 2}),
        params_override='{"a.x": 2}')()
    self.assertEqual(value, {'a': {'x': 2, 'y': 2}})

    with self.assertRaisesRegex(
        TypeError,
        'Loaded value .* is not an instance of .*'):
      patching.object_factory(
          symbolic.Dict, symbolic.Dict(a=1),
          params_override='1')()

  def testFactoryWithPatchersAndParamsOverride(self):
    """Test base value + patchers + params override."""
    value = patching.object_factory(
        symbolic.Dict, symbolic.Dict(a=1, b=2),
        ['update_a?value=2'],
        params_override={'b': 3, 'c': 0})()
    self.assertEqual(value, {
        'a': 2,
        'b': 3,
        'c': 0
    })


if __name__ == '__main__':
  unittest.main()
