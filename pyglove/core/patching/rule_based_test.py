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
"""Tests for conditional conditional."""

import unittest
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.patching import rule_based


class PatcherTest(unittest.TestCase):
  """Patcher test."""

  def assert_patch_equal(self, src, patcher_or_uri, expected_value):
    if isinstance(patcher_or_uri, rule_based.Patcher):
      patcher = patcher_or_uri
    else:
      patcher = rule_based.from_uri(patcher_or_uri)
    self.assertIsInstance(patcher, rule_based.Patcher)
    dst = patcher.patch(src)
    self.assertIs(src, dst)
    self.assertEqual(dst, expected_value)

  def test_patcher_without_typing(self):
    @rule_based.patcher()
    def set_value1(unused_src, k, v='0'):
      return {
          k: v
      }
    self.assert_patch_equal(symbolic.Dict(), set_value1(k='a'), {'a': '0'})  # pylint: disable=not-callable, no-value-for-parameter
    self.assert_patch_equal(symbolic.Dict(), 'set_value1?a&1', {'a': '1'})
    self.assert_patch_equal(symbolic.Dict(), 'set_value1?a&v=1', {'a': '1'})
    self.assert_patch_equal(symbolic.Dict(), 'set_value1?k=a&v=1', {'a': '1'})
    self.assertIn('set_value1', rule_based.patcher_names())

    with self.assertRaisesRegex(
        TypeError, 'The 1st argument of .* must be a symbolic type'):
      set_value1(k='a').patch(1)  # pylint: disable=not-callable, no-value-for-parameter

  def test_patcher_with_auto_typing(self):
    @rule_based.patcher(auto_typing=True)
    def set_value2(unused_src, k: str, v: int):
      return {
          k: v
      }
    self.assert_patch_equal(symbolic.Dict(), set_value2(k='a', v=1), {'a': 1})  # pylint: disable=no-value-for-parameter
    self.assert_patch_equal(symbolic.Dict(), 'set_value2?a&1', {'a': 1})
    self.assert_patch_equal(symbolic.Dict(), 'set_value2?a&v=1', {'a': 1})
    self.assert_patch_equal(symbolic.Dict(), 'set_value2?k=a&v=1', {'a': 1})
    self.assertIn('set_value2', rule_based.patcher_names())

  def test_patcher_with_typing(self):
    @rule_based.patcher([
        ('k', pg_typing.Str()),
        ('v', pg_typing.Int())
    ])
    def set_value2(unused_src, k, v):
      return {
          k: v
      }
    self.assert_patch_equal(symbolic.Dict(), set_value2(k='a', v=1), {'a': 1})  # pylint: disable=no-value-for-parameter
    self.assert_patch_equal(symbolic.Dict(), 'set_value2?a&1', {'a': 1})
    self.assert_patch_equal(symbolic.Dict(), 'set_value2?a&v=1', {'a': 1})
    self.assert_patch_equal(symbolic.Dict(), 'set_value2?k=a&v=1', {'a': 1})
    self.assertIn('set_value2', rule_based.patcher_names())

  def test_patcher_with_replacing_source_value(self):
    @rule_based.patcher([
        ('v', pg_typing.Int())
    ])
    def set_value3(unused_src, v):
      return {'': v}
    self.assertEqual(
        set_value3(v=1).patch(symbolic.Dict()), 1)   # pylint: disable=not-callable, no-value-for-parameter
    self.assertEqual(
        rule_based.from_uri('set_value3?1').patch(symbolic.Dict()), 1)
    self.assertIn('set_value3', rule_based.patcher_names())

  def test_patcher_with_returning_rebind_fn(self):
    @rule_based.patcher()
    def increment(unused_src):
      return lambda k, v: (v + 1) if isinstance(v, int) else v

    self.assertEqual(
        increment().patch(symbolic.Dict(a=1, b=2)), symbolic.Dict(a=2, b=3))  # pylint: disable=not-callable, no-value-for-parameter

    @rule_based.patcher()
    def no_change(unused_src):
      return lambda k, v: v

    self.assertEqual(
        no_change().patch(symbolic.Dict(a=1, b=2)), symbolic.Dict(a=1, b=2))  # pylint: disable=not-callable, no-value-for-parameter

  def test_patcher_with_composition(self):
    @rule_based.patcher([
        ('k', pg_typing.Str()),
        ('v', pg_typing.Int())
    ])
    def set_value2(unused_src, k, v):
      return {
          k: v
      }

    @rule_based.patcher()
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

  def test_patcher_with_validation(self):
    @rule_based.patcher()
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

  def test_bad_patcher(self):
    @rule_based.patcher()
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
      @rule_based.patcher([
          ('unused_src', pg_typing.Int()),
      ])   # pylint: disable=unused-variable
      def set_value5(unused_src, x):
        del x

    # Bad patcher argument spec.
    with self.assertRaisesRegex(
        TypeError,
        '.* cannot be used for constraining Patcher argument'):
      @rule_based.patcher([
          ('x', pg_typing.Enum(1, [1, 2]))
      ])   # pylint: disable=unused-variable
      def set_value6(unused_src, x=1):
        del x

    # Bad patcher without target argument.
    with self.assertRaisesRegex(
        TypeError,
        r'Patcher function should have at least 1 argument as patching target'):
      @rule_based.patcher()
      def set_value7():  # pylint: disable=unused-variable
        return {'': 1}

    # Bad patcher with wrong return type for validator.
    @rule_based.patcher()
    def set_value8(unused_src):
      return {'': 1}, 1

    with self.assertRaisesRegex(
        TypeError,
        r'The validator returned from patcher .* is not callable'):
      set_value8().patch(symbolic.Dict())  # pylint: disable=no-value-for-parameter


@rule_based.patcher([
    ('v', pg_typing.Int())
])
def p1(unused_src, k, v):
  return {k: v}


@rule_based.patcher()
def p2(unused_src):  # pylint: disable=unused-variable
  return {'': symbolic.Dict()}


@rule_based.patcher()
def p3(unused_src):
  def validate(x):
    if 'a' not in x:
      raise ValueError('Key `a` does not exist.')
  return {}, validate


class PatchTest(unittest.TestCase):
  """Tests for `pg.patch`."""

  def test_patch_with_a_single_rule(self):
    self.assertEqual(
        rule_based.patch(symbolic.Dict(x=1, y=2), {'x': 0, 'z': 3}),
        symbolic.Dict(x=0, y=2, z=3))

  def test_patch_with_multiple_rules_without_replacement(self):
    self.assertEqual(
        rule_based.patch(symbolic.Dict(), [
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

  def test_patch_with_multiple_rules_with_replacement(self):
    self.assertEqual(
        rule_based.patch(symbolic.Dict(), [
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

  def test_patch_with_a_rebind_dict(self):
    self.assertEqual(
        rule_based.patch(symbolic.Dict(a=1, b=2), {'a': 0, 'c': 3}),
        symbolic.Dict(a=0, b=2, c=3))

  def test_patch_with_a_rebind_function(self):
    self.assertEqual(
        rule_based.patch(
            symbolic.Dict(a=1),
            lambda k, v, p: (v + 1) if isinstance(v, int) else v),
        symbolic.Dict(a=2))

    self.assertEqual(
        rule_based.patch(symbolic.Dict(a=1), [
            lambda k, v: v
        ]),
        symbolic.Dict(a=1))

  def test_patch_with_validator(self):
    self.assertEqual(
        rule_based.patch(symbolic.Dict(), [
            p1(k='a', v=1),  # pylint: disable=no-value-for-parameter
            'p1?b&2',
            'p3'
        ]),
        {
            'a': 1,
            'b': 2,
        })

  def test_nothing_to_patch(self):
    self.assertEqual(rule_based.patch(symbolic.Dict(), []), {})

  def test_bad_patch(self):
    with self.assertRaisesRegex(ValueError, 'Key `a` does not exist.'):
      _ = rule_based.patch(symbolic.Dict(), [
          'p1?b&2',
          p3()  # pylint: disable=no-value-for-parameter
      ])

    # Bad patch input.
    with self.assertRaisesRegex(
        TypeError, 'Patching rule .* should be a dict of path to values'):
      rule_based.patch(symbolic.Dict(), [1])


class PatcherHelpersTest(unittest.TestCase):
  """Tests for patcher helpers."""

  def test_is_patcher_parameter_spec(self):
    def assert_argument_spec(value_spec):
      self.assertTrue(rule_based._is_patcher_parameter_spec(value_spec))

    def assert_not_argument_spec(value_spec):
      self.assertFalse(rule_based._is_patcher_parameter_spec(value_spec))

    assert_argument_spec(pg_typing.Any())
    assert_argument_spec(pg_typing.Int())
    assert_argument_spec(pg_typing.Float())
    assert_argument_spec(pg_typing.Bool())
    assert_argument_spec(pg_typing.Str())
    assert_argument_spec(pg_typing.Enum('a', ['a', 'b']))
    assert_argument_spec(pg_typing.Enum('a', ['a', 'b']))
    assert_argument_spec(pg_typing.List(pg_typing.Int()))
    assert_argument_spec(pg_typing.List(pg_typing.Enum('a', ['a', 'b'])))

    assert_not_argument_spec(pg_typing.Dict())
    assert_not_argument_spec(pg_typing.Type(Exception))
    assert_not_argument_spec(pg_typing.Object(ValueError))
    assert_not_argument_spec(pg_typing.Enum(1, [0, 1]))
    assert_not_argument_spec(pg_typing.List(pg_typing.List(pg_typing.Int())))
    assert_not_argument_spec(
        pg_typing.Union([pg_typing.Int(), pg_typing.Float()]))

  def test_from_uri(self):
    @rule_based.patcher([
        ('x', pg_typing.Int())
    ], name='my_patcher')
    def foo(src, x, y='bar'):
      del src, x, y

    self.assertEqual(rule_based.from_uri('my_patcher?1'), foo(x=1))    # pylint: disable=not-callable, no-value-for-parameter
    self.assertEqual(rule_based.from_uri('my_patcher?x=1'), foo(x=1))  # pylint: disable=not-callable, no-value-for-parameter

    with self.assertRaisesRegex(
        KeyError, 'Patcher .* is not registered'):
      rule_based.from_uri('foo')

    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to int'
        r'.*Patcher=.*foo\', Argument=\'x\''):
      rule_based.from_uri('my_patcher?x=a').patch(symbolic.Dict())

    with self.assertRaisesRegex(
        KeyError,
        r'Too many positional arguments are provided.'):
      rule_based.from_uri('my_patcher?1&2&3').patch(symbolic.Dict())

    with self.assertRaisesRegex(
        TypeError,
        r'foo\(\) (missing 1 required positional argument|'
        'takes at least 2 arguments)'):
      rule_based.from_uri('my_patcher').patch(symbolic.Dict())

  def test_parse_uri(self):
    def test_parse(uri, expected_name, expected_args, expected_kwargs):
      name, args, kwargs = rule_based.parse_uri(uri)
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
      rule_based.parse_uri('1foo')

    with self.assertRaisesRegex(
        ValueError,
        'Invalid argument specification: a-1=2'):
      rule_based.parse_uri('foo?a-1=2')

    with self.assertRaisesRegex(
        ValueError,
        'Positional argument should be provided before keyword arguments'):
      rule_based.parse_uri('foo?a=1&2')

  def test_parse_args(self):
    signature = pg_typing.Signature(
        pg_typing.CallableType.FUNCTION, 'foo', '__main__',
        [
            pg_typing.Argument(
                'src',
                pg_typing.Argument.Kind.POSITIONAL_OR_KEYWORD,
                pg_typing.Any()
            ),
            pg_typing.Argument(
                'x',
                pg_typing.Argument.Kind.POSITIONAL_OR_KEYWORD,
                pg_typing.Int()
            ),
            pg_typing.Argument(
                'y',
                pg_typing.Argument.Kind.POSITIONAL_OR_KEYWORD,
                pg_typing.List(pg_typing.Float(min_value=0.0, max_value=1.0))
            )
        ])
    args, kwargs = rule_based.parse_args(signature, ['0'], {'y': '0.1:0.5'})
    self.assertEqual(args, [0])
    self.assertEqual(kwargs, {'y': [0.1, 0.5]})

    with self.assertRaisesRegex(
        KeyError, 'Unexpected argument'):
      rule_based.parse_args(signature, [], {'z': '0.1'})

  def test_allow_repeated_patcher_registration(self):
    """Test `rule_based.allow_repeated_patcher_registration`."""
    @rule_based.patcher(name='my_patch')
    def patcher1(unused_src):
      return {}

    @rule_based.patcher(name='my_patch')
    def patcher2(unused_src):
      return {}

    rule_based.allow_repeated_patcher_registration(False)
    with self.assertRaisesRegex(
        KeyError, 'Patcher .* already registered.'):
      @rule_based.patcher(name='my_patch')
      def patcher3(unused_src):
        return {}
    rule_based.allow_repeated_patcher_registration(True)


def parse_arg(value_spec, arg_str):
  return rule_based.parse_arg('__main__.patcher_a', 'x', value_spec, arg_str)


class ParseArgTest(unittest.TestCase):

  def test_parse_float(self):
    self.assertTrue(parse_arg(pg_typing.Bool(), 'True'))
    self.assertTrue(parse_arg(pg_typing.Bool(), 'true'))
    self.assertTrue(parse_arg(pg_typing.Bool(), 'yes'))
    self.assertTrue(parse_arg(pg_typing.Bool(), '1'))
    self.assertFalse(parse_arg(pg_typing.Bool(), 'False'))
    self.assertFalse(parse_arg(pg_typing.Bool(), 'false'))
    self.assertFalse(parse_arg(pg_typing.Bool(), 'no'))
    self.assertFalse(parse_arg(pg_typing.Bool(), '0'))
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to bool. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(pg_typing.Bool(), 'a')

    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to float. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(pg_typing.Float(), 'a')

    with self.assertRaisesRegex(
        ValueError,
        'Value -1.0 is out of range .*path=__main__.patcher_a.x'):
      parse_arg(pg_typing.Float(min_value=0.), '-1.0')

  def test_parse_int(self):
    self.assertEqual(parse_arg(pg_typing.Int(), '1'), 1)
    self.assertIsNone(parse_arg(pg_typing.Int().noneable(), 'None'))
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to int. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(pg_typing.Int(), 'a')

    with self.assertRaisesRegex(
        ValueError,
        'Value -1 is out of range .*path=__main__.patcher_a.x'):
      parse_arg(pg_typing.Int(min_value=0), '-1')

    # Test float.
    self.assertEqual(parse_arg(pg_typing.Float(), '1'), 1.0)
    self.assertIsNone(parse_arg(pg_typing.Float().noneable(), 'None'))
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to float. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\'\)'):
      parse_arg(pg_typing.Float(), 'a')
    with self.assertRaisesRegex(
        ValueError, 'Value -1.0 is out of range'):
      parse_arg(pg_typing.Float(min_value=0.), '-1.0')

  def test_parse_str(self):
    self.assertEqual(parse_arg(pg_typing.Str(), '1'), '1')
    self.assertEqual(parse_arg(pg_typing.Str(), '"1"'), '1')
    self.assertIsNone(parse_arg(pg_typing.Str().noneable(), 'none'))
    self.assertEqual(parse_arg(pg_typing.Str().noneable(), '"none"'), 'none')

    with self.assertRaisesRegex(
        ValueError,
        'String \'bar\' does not match regular expression '
        '.*path=__main__.patcher_a.x'):
      parse_arg(pg_typing.Str(regex='foo.*'), 'bar')

    with self.assertRaisesRegex(
        ValueError,
        'Unmatched quote for string value: \'"1\'.*'):
      parse_arg(pg_typing.Str(), '"1')

    with self.assertRaisesRegex(
        ValueError,
        'Unmatched quote for string value: \'1"\'.*'):
      parse_arg(pg_typing.Str(), '1"')

  def test_parse_enum(self):
    self.assertEqual(parse_arg(pg_typing.Enum('a', ['a', 'b']), 'b'), 'b')
    with self.assertRaisesRegex(
        ValueError,
        r'.* cannot be used for Patcher argument. '
        r'Only Enum of string type can be used.'):
      parse_arg(pg_typing.Enum(1, [1, 2]), '1')

  def test_parse_list(self):
    self.assertEqual(parse_arg(pg_typing.List(pg_typing.Int()), ''), [])
    self.assertEqual(parse_arg(pg_typing.List(pg_typing.Int()), '1'), [1])
    self.assertEqual(parse_arg(pg_typing.List(pg_typing.Int()), '1:2'),
                     [1, 2])
    self.assertEqual(parse_arg(pg_typing.List(pg_typing.Bool()), 'yes:no'),
                     [True, False])
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot convert \'a\' to int. '
        r'\(Patcher=\'__main__.patcher_a\', Argument=\'x\[0\]\'\)'):
      parse_arg(pg_typing.List(pg_typing.Int()), 'a:b')

  def test_parse_unsupported_types(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Dict\(\) cannot be used for Patcher argument'):
      parse_arg(pg_typing.Dict(), '{}')


if __name__ == '__main__':
  unittest.main()
