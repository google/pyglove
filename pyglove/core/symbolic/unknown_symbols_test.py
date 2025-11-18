# Copyright 2025 The PyGlove Authors
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
from pyglove.core import utils
from pyglove.core.symbolic import unknown_symbols


class UnknownTypeTest(unittest.TestCase):

  def test_basics(self):
    t = unknown_symbols.UnknownType(name='__main__.ABC', args=[int, str])
    self.assertEqual(t.name, '__main__.ABC')
    self.assertEqual(t.args, [int, str])
    self.assertEqual(
        repr(t),
        '<unknown-type __main__.ABC>[<class \'int\'>, <class \'str\'>]'
    )
    self.assertEqual(
        t.to_json(),
        {
            '_type': 'type',
            'name': '__main__.ABC',
            'args': [
                {'_type': 'type', 'name': 'builtins.int'},
                {'_type': 'type', 'name': 'builtins.str'},
            ]
        }
    )
    self.assertEqual(utils.from_json(t.to_json(), convert_unknown=True), t)
    self.assertEqual(
        t(x=1, y=2),
        unknown_symbols.UnknownTypedObject(type_name='__main__.ABC', x=1, y=2)
    )


class UnknownFunctionTest(unittest.TestCase):

  def test_basics(self):
    t = unknown_symbols.UnknownFunction(name='__main__.foo')
    self.assertEqual(t.name, '__main__.foo')
    self.assertEqual(repr(t), '<unknown-function __main__.foo>')
    self.assertEqual(
        t.to_json(),
        {
            '_type': 'function',
            'name': '__main__.foo',
        }
    )
    self.assertEqual(utils.from_json(t.to_json(), convert_unknown=True), t)


class UnknownMethodTest(unittest.TestCase):

  def test_basics(self):
    t = unknown_symbols.UnknownMethod(name='__main__.ABC.bar')
    self.assertEqual(t.name, '__main__.ABC.bar')
    self.assertEqual(repr(t), '<unknown-method __main__.ABC.bar>')
    self.assertEqual(
        t.to_json(),
        {
            '_type': 'method',
            'name': '__main__.ABC.bar',
        }
    )
    self.assertEqual(utils.from_json(t.to_json(), convert_unknown=True), t)


class UnknownObjectTest(unittest.TestCase):

  def test_basics(self):
    v = unknown_symbols.UnknownTypedObject(type_name='__main__.ABC', x=1)
    self.assertEqual(v.type_name, '__main__.ABC')
    self.assertEqual(v.x, 1)
    self.assertEqual(repr(v), '<unknown-type __main__.ABC>(x=1)')
    self.assertEqual(
        str(v), '<unknown-type __main__.ABC>(\n  x = 1\n)')
    self.assertEqual(
        v.to_json(),
        {
            '_type': '__main__.ABC',
            'x': 1,
        }
    )
    self.assertEqual(utils.from_json(v.to_json(), convert_unknown=True), v)


if __name__ == '__main__':
  unittest.main()
