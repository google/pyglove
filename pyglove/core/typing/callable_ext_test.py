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
"""Tests for pyglove.core.typing.callable_ext."""

import unittest
from pyglove.core.typing import annotation_conversion   # pylint: disable=unused-import
from pyglove.core.typing import callable_ext


class PresetArgValueTest(unittest.TestCase):
  """Tests for typing.PresetArgValue."""

  def test_basics(self):
    v = callable_ext.PresetArgValue()
    self.assertFalse(v.has_default)
    v = callable_ext.PresetArgValue(default=1)
    self.assertTrue(v.has_default)
    self.assertEqual(v.default, 1)
    self.assertEqual(repr(v), 'PresetArgValue(default=1)')
    self.assertEqual(str(v), 'PresetArgValue(\n  default=1\n)')
    self.assertEqual(
        callable_ext.PresetArgValue(), callable_ext.PresetArgValue()
    )
    self.assertEqual(
        callable_ext.PresetArgValue(1), callable_ext.PresetArgValue(1)
    )
    self.assertNotEqual(
        callable_ext.PresetArgValue(), callable_ext.PresetArgValue(default=1)
    )

  def test_inspect(self):

    def foo(
        x=callable_ext.PresetArgValue(),
        y=1,
        *,
        z=callable_ext.PresetArgValue(default=2)
    ):
      return x + y + z

    self.assertEqual(
        callable_ext.PresetArgValue.inspect(foo),
        dict(
            x=callable_ext.PresetArgValue(),
            z=callable_ext.PresetArgValue(default=2)
        )
    )

    def bar(x, y=1):
      return x + y

    self.assertEqual(
        callable_ext.PresetArgValue.inspect(bar),
        {}
    )

  def test_resolve_args(self):
    # Both positional and keyword arguments are from preset.
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[],
            call_kwargs=dict(),
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(),
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict(x=1, y=2, z=3, w=4)
        ),
        ([1], dict(z=3))
    )
    # Both positional and keyword arguments are absent and use the default.
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[],
            call_kwargs=dict(),
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(default=1),
                'y': 0,
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict()
        ),
        ([1, 0], dict(z=2))
    )
    # Positional args from preset, keyword argument use preset default.
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[],
            call_kwargs=dict(),
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(),
                'y': 0,
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict(x=1, y=2)
        ),
        ([1, 0], dict(z=2))
    )
    # Postional argument provided by user, which should take precedence over
    # the preset value.
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[2],
            call_kwargs={},
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(),
                'y': 0,
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict(x=1, y=2, z=3, w=4)
        ),
        ([2, 0], dict(z=3))
    )
    # Postional argument provided in keyword.
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[],
            call_kwargs=dict(x=2),
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(),
                'y': 0,
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict(x=1, y=2, z=3, w=4)
        ),
        ([2, 0], dict(z=3))
    )
    # Postional argument provided in keyword, and there are more args
    # (due to varargs)
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[1, 2, 3],
            call_kwargs=dict(x=2),
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(),
                'y': 0,
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict(x=1, y=2, z=3, w=4)
        ),
        ([1, 2, 3], dict(z=3))
    )
    # Required preset argument is not provided.
    with self.assertRaisesRegex(ValueError, 'Argument .* is not present.'):
      callable_ext.PresetArgValue.resolve_args(
          call_args=[],
          call_kwargs=dict(),
          positional_arg_names=['x'],
          arg_defaults={
              'x': callable_ext.PresetArgValue(),
          },
          preset_kwargs=dict()
      )

    # Include all preset kwargs.
    self.assertEqual(
        callable_ext.PresetArgValue.resolve_args(
            call_args=[],
            call_kwargs=dict(),
            positional_arg_names=['x', 'y'],
            arg_defaults={
                'x': callable_ext.PresetArgValue(),
                'y': 0,
                'z': callable_ext.PresetArgValue(default=2),
            },
            preset_kwargs=dict(x=1, y=2, z=3, w=4),
            include_all_preset_kwargs=True,
        ),
        ([1, 0], dict(z=3, w=4))
    )

  def test_preset_args(self):
    @callable_ext.enable_preset_args()
    def foo(
        x=callable_ext.PresetArgValue(),
        y=1,
        *args,
        z=callable_ext.PresetArgValue(default=2)
    ):
      del args
      return x + y + z

    with self.assertRaisesRegex(ValueError, 'Argument \'x\' is not present.'):
      foo()

    with callable_ext.preset_args(dict(x=1)):
      self.assertEqual(foo(), 1 + 1 + 2)

    # `y`` should not take precedence over the non-preset default.
    with callable_ext.preset_args(dict(x=1, y=2)):
      self.assertEqual(foo(), 1 + 1 + 2)

    with callable_ext.preset_args(dict(x=1, y=2, z=3)):
      self.assertEqual(foo(3), 3 + 1 + 3)

    with callable_ext.preset_args(dict(x=1, y=2, z=3)):
      self.assertEqual(foo(3, 3, z=4), 3 + 3 + 4)

  def test_enable_preset_args(self):

    # No side-effect if function does not have PresetArgValue.
    def bar(x, y):
      return x + y
    self.assertIs(bar, callable_ext.enable_preset_args()(bar))

    # `include_all_preset_kwargs` sets to False.
    @callable_ext.enable_preset_args()
    def baz(x, y=callable_ext.PresetArgValue(default=1), **kwargs):
      return x + y + sum(kwargs.values())

    with callable_ext.preset_args(dict(z=3, p=4)):
      self.assertEqual(baz(1), 1 + 1)

    # `include_all_prset_kwargs` is effective only when there is varkw.
    @callable_ext.enable_preset_args(include_all_preset_kwargs=True)
    def foo(x, y=callable_ext.PresetArgValue(default=1), **kwargs):
      return x + y + sum(kwargs.values())

    with callable_ext.preset_args(dict(z=3, p=4)):
      self.assertEqual(foo(1), 1 + 1 + 3 + 4)

    # `include_all_preset_kwargs` should be ignored if there is no varkw.
    @callable_ext.enable_preset_args(include_all_preset_kwargs=True)
    def fuz(x, y=callable_ext.PresetArgValue(default=1)):
      return x + y

    with callable_ext.preset_args(dict(y=2, z=3)):
      self.assertEqual(fuz(1), 1 + 2)

  def test_preset_args_nesting(self):
    @callable_ext.enable_preset_args()
    def foo(
        x=callable_ext.PresetArgValue(),
        y=1,
        *,
        z=callable_ext.PresetArgValue(default=2)
    ):
      return x + y + z

    def bar(inherit_preset: bool = False, **kwargs):
      with callable_ext.preset_args(
          {k: v + 1 for k, v in kwargs.items()},
          inherit_preset=inherit_preset
      ):
        return foo()

    with callable_ext.preset_args(dict(x=1)):
      self.assertEqual(foo(), 1 + 1 + 2)

      self.assertEqual(bar(x=1), 2 + 1 + 2)
      self.assertEqual(bar(x=1, z=2), 2 + 1 + 3)
      self.assertEqual(bar(x=1, z=3), 2 + 1 + 4)

      with self.assertRaisesRegex(ValueError, 'Argument \'x\' is not present.'):
        bar()
      self.assertEqual(bar(inherit_preset=True), 1 + 1 + 2)


class CallWithOptionalKeywordArgsTest(unittest.TestCase):
  """Tests for typing.CallWithOptionalKeywordArgs."""

  def test_function(self):
    """Test call with function."""

    def foo(a, b):
      return a + b

    f = callable_ext.CallableWithOptionalKeywordArgs(foo, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)

    def bar(a, **kwargs):
      return sum([a] + list(kwargs.values()))

    f = callable_ext.CallableWithOptionalKeywordArgs(bar, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 6)

  def testMethod(self):
    """Test call with method."""

    class A:

      def __call__(self, a, b):
        return a + b

    f = callable_ext.CallableWithOptionalKeywordArgs(A(), ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)

    class B:

      def __call__(self, a, **kwargs):
        return sum([a] + list(kwargs.values()))

    f = callable_ext.CallableWithOptionalKeywordArgs(B(), ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 6)

  def test_classmethod(self):
    """Test call with class method."""

    class A:

      @classmethod
      def foo(cls, a, b):
        return a + b

    f = callable_ext.CallableWithOptionalKeywordArgs(A.foo, ['b', 'c'])
    self.assertEqual(f(1, b=2, c=3), 3)


if __name__ == '__main__':
  unittest.main()
