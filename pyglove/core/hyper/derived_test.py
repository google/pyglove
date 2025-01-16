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
import unittest

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core import utils
from pyglove.core.hyper.derived import ValueReference


class ValueReferenceTest(unittest.TestCase):
  """Tests for pg.hyper.ValueReference."""

  def test_resolve(self):
    sd = symbolic.Dict({'c': [
        {
            'x': [{
                'z': 0
            }],
        },
        {
            'x': [{
                'z': 1
            }]
        },
    ]})
    sd.a = ValueReference(reference_paths=['c[0].x[0].z'])
    self.assertEqual(sd.a.resolve(), [(sd, 'c[0].x[0].z')])

    # References refer to the same relative path under different parent.
    ref = ValueReference(reference_paths=['x[0].z'])
    sd.c[0].y = ref
    sd.c[1].y = ref
    self.assertEqual(sd.c[0].y.resolve(), [(sd.c[0], 'c[0].x[0].z')])
    self.assertEqual(sd.c[1].y.resolve(), [(sd.c[1], 'c[1].x[0].z')])
    # Resolve references from this point.
    self.assertEqual(sd.c[0].y.resolve(utils.KeyPath(0)), (sd.c, 'c[0]'))
    self.assertEqual(sd.c[0].y.resolve('[0]'), (sd.c, 'c[0]'))
    self.assertEqual(
        sd.c[0].y.resolve(['[0]', '[1]']), [(sd.c, 'c[0]'), (sd.c, 'c[1]')])

    # Bad inputs.
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'reference_path_or_paths\' must be None, a string, KeyPath '
        'object, a list of strings, or a list of KeyPath objects.'):
      sd.c[0].y.resolve([1])

    with self.assertRaisesRegex(
        ValueError,
        'Argument \'reference_path_or_paths\' must be None, a string, KeyPath '
        'object, a list of strings, or a list of KeyPath objects.'):
      sd.c[0].y.resolve(1)

    with self.assertRaisesRegex(
        ValueError, 'Cannot resolve .*: parent not found.'):
      ValueReference(reference_paths=['x[0].z']).resolve()

  def test_call(self):

    @symbolic.members([('a', pg_typing.Int(), 'Field a.')])
    class A(symbolic.Object):
      pass

    sd = symbolic.Dict({'c': [
        {
            'x': [{
                'z': 0
            }],
        },
        {
            'x': [{
                'z': A(a=1)
            }]
        },
    ]})
    sd.a = ValueReference(reference_paths=['c[0].x[0].z'])
    self.assertEqual(sd.a(), 0)

    # References refer to the same relative path under different parent.
    ref = ValueReference(reference_paths=['x[0]'])
    sd.c[0].y = ref
    sd.c[1].y = ref
    self.assertEqual(sd.c[0].y(), {'z': 0})
    self.assertEqual(sd.c[1].y(), {'z': A(a=1)})

    # References to another reference is not supported.
    sd.c[1].z = ValueReference(reference_paths=['y'])
    with self.assertRaisesRegex(
        ValueError,
        'Derived value .* should not reference derived values'):
      sd.c[1].z()

    sd.c[1].z = ValueReference(reference_paths=['c'])
    with self.assertRaisesRegex(
        ValueError,
        'Derived value .* should not reference derived values'):
      sd.c[1].z()

  def test_assignment_compatibility(self):
    sd = symbolic.Dict.partial(
        x=0,
        value_spec=pg_typing.Dict([
            ('x', pg_typing.Int()),
            ('y', pg_typing.Int()),
            ('z', pg_typing.Str())
        ]))

    sd.y = ValueReference(['x'])
    # TODO(daiyip): Enable this test once static analysis is done
    # on derived values.
    # with self.assertRaisesRegex(
    #     TypeError, ''):
    #   sd.z = ValueReference(['x'])

  def test_bad_init(self):
    with self.assertRaisesRegex(
        ValueError,
        'Argument \'reference_paths\' should have exact 1 item'):
      ValueReference([])


if __name__ == '__main__':
  unittest.main()
