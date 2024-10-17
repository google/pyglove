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
"""Tests for pyglove.diff."""

import inspect
from typing import Any
import unittest

from pyglove.core import typing as pg_typing
from pyglove.core.symbolic.dict import Dict
from pyglove.core.symbolic.diff import Diff
from pyglove.core.symbolic.diff import diff as pg_diff
from pyglove.core.symbolic.list import List
from pyglove.core.symbolic.object import members as pg_members
from pyglove.core.symbolic.object import Object


class DiffTest(unittest.TestCase):
  """Tests for `pg.diff`."""

  def test_diff_class(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    self.assertTrue(bool(Diff(1, 2)))
    self.assertTrue(bool(Diff(A, B)))
    self.assertTrue(bool(Diff(A(1), A(2))))
    self.assertTrue(bool(Diff(A, A, children={'x': Diff(1, 2)})))
    self.assertFalse(bool(Diff()))
    self.assertFalse(bool(Diff(1, 1)))
    self.assertFalse(bool(Diff(A, A)))
    self.assertFalse(bool(Diff(A(1), A(1))))

    self.assertTrue(Diff(1, 2).is_leaf)
    self.assertFalse(
        Diff(int, int, children={'x': Diff(1, 2)}).is_leaf)

    self.assertEqual(Diff(1, 1), 1)
    self.assertEqual(Diff(1, 1), Diff(1, 1))
    self.assertNotEqual(Diff(1, 1), 2)
    self.assertNotEqual(Diff(1, 1), Diff(1, 2))

    self.assertEqual(
        repr(Diff()), 'No diff')
    self.assertEqual(
        repr(Diff(A(1), A(1))), 'A(x=1)')
    self.assertEqual(
        repr(Dict(a=Diff(A(1), A(1)))), '{a=A(x=1)}')
    self.assertEqual(
        repr(Diff(A(1), A(2))), 'Diff(left=A(x=1), right=A(x=2))')
    self.assertEqual(
        repr(Diff(A, A, children={'x': Diff(1, 2)})),
        'A(x=Diff(left=1, right=2))')
    self.assertEqual(
        repr(Diff(A, B, children={
            'x': Diff(1, 2),
            'y': Diff(Diff.MISSING, 3)
        })),
        'A|B(x=Diff(left=1, right=2), y=Diff(left=MISSING, right=3))')
    self.assertEqual(
        repr(Diff(List, List, children={
            '0': Diff(1, 2),
            '1': Diff(Diff.MISSING, 3)
        })),
        '[0=Diff(left=1, right=2), 1=Diff(left=MISSING, right=3)]')

    with self.assertRaisesRegex(
        ValueError,
        '\'left\' must be a type when \'children\' is specified.'):
      Diff(1, int, children={'x': Diff(3, 4)})

    with self.assertRaisesRegex(
        ValueError,
        '\'right\' must be a type when \'children\' is specified.'):
      Diff(int, 2, children={'x': Diff(3, 4)})

    with self.assertRaisesRegex(
        ValueError,
        '\'value\' cannot be accessed when \'left\' and \'right\' '
        'are not the same.'):
      _ = Diff(1, 2).value

  def test_diff_on_simple_types(self):
    self.assertEqual(pg_diff(1, 1), Diff())
    self.assertEqual(pg_diff(1, 1, mode='same'), Diff(1, 1))
    self.assertEqual(pg_diff(1, 1, flatten=True, mode='same'), Diff(1, 1))
    self.assertEqual(pg_diff(1, 2), Diff(1, 2))
    self.assertEqual(pg_diff(1, 2, mode='same'), Diff(1, 2))
    self.assertEqual(pg_diff(1, 2, flatten=True, mode='same'), Diff(1, 2))

  def test_diff_on_list(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    # List vs. list.
    self.assertEqual(pg_diff([A(1)], [A(1)]), Diff())
    self.assertEqual(pg_diff([A(1)], [A(1)], mode='same'), Diff([A(1)], [A(1)]))
    self.assertEqual(
        pg_diff([A(1)], [A(0)]),
        Diff(left=List, right=List, children={
            '0': Diff(A, A, children={
                'x': Diff(1, 0)
            })}))

    # List vs. other types.
    self.assertEqual(pg_diff([A(1)], 1), Diff([A(1)], 1))

  def test_diff_on_dict(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    # Dict vs. dict.
    self.assertEqual(
        pg_diff({'a': A(1)}, {'a': A(1)}),
        Diff())
    self.assertEqual(
        pg_diff({'a': A(1)}, {'a': A(1)}, mode='same'),
        Diff({'a': A(1)}, {'a': A(1)}))
    self.assertEqual(
        pg_diff({'a': A(1), 'b': A(2), 'c': A(3)},
                {'a': A(1), 'b': A(3), 'd': A(4)}),
        Diff(dict, dict, children={
            'b': Diff(A, A, children={
                'x': Diff(2, 3)
            }),
            'c': Diff(A(3)),
            'd': Diff(right=A(4)),
        }))

    # Dict vs. symbolic object.
    self.assertEqual(
        pg_diff(A(1), {'x': 1}),
        Diff(A(1), {'x': 1}))

    self.assertEqual(
        pg_diff(A(1), {'x': 1}, collapse=True),
        Diff(A, dict))

    self.assertEqual(
        pg_diff(A(1), {'x': 2}, collapse=True),
        Diff(A, dict, children={
            'x': Diff(1, 2)
        }))

    # Dict vs. other types.
    self.assertEqual(pg_diff({'x': 1}, 1), Diff({'x': 1}, 1))

  def test_diff_on_symbolic_object(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    # Same types.
    self.assertEqual(pg_diff(A(1), A(1)), Diff())
    self.assertEqual(pg_diff(A(1), A(1), mode='same'), Diff(A(1), A(1)))
    self.assertEqual(
        pg_diff(B(1, 2), B(1, 3)),
        Diff(B, B, children={'y': Diff(2, 3)}))

    # Different types without collapse.
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2)),
        Diff(B(1, 2), C(1, 2)))

    # Different types with always collapse.
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True),
        Diff(B, C))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3)),
        pg_diff(B(1, 2), C(1, 3)))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True),
        Diff(B, C, children={
            'y': Diff(2, 3)
        }))

  def test_diff_between_symbolic_and_dict(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    a = A(A(1))
    d = Dict(x=Dict(x=1))
    self.assertEqual(pg_diff(a, d), Diff(left=a, right=d))
    self.assertEqual(
        pg_diff(a, d, collapse=True),
        Diff(A, Dict, children=dict(x=Diff(A, Dict))))

  def test_collapse(self):

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class D(Object):
      pass

    def collapse_subclass(x, y):
      return issubclass(type(x), type(y)) or issubclass(type(y), type(x))

    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=collapse_subclass),
        Diff(B, C))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=collapse_subclass),
        Diff(B, C, children={
            'y': Diff(2, 3)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), D(1, 3), collapse=collapse_subclass),
        Diff(B(1, 2), D(1, 3)))

    # Test bad collapse option.
    with self.assertRaisesRegex(
        ValueError, 'Unsupported `collapse` value'):
      pg_diff(B(1, 2), C(1, 2), collapse='unsupported_option')

  def test_mode(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    self.assertEqual(pg_diff(A(1), A(1), mode='diff'), Diff())
    self.assertEqual(pg_diff(A(1), A(1), mode='same'), Diff(A(1), A(1)))
    self.assertEqual(pg_diff(A(1), A(1), mode='both'), Diff(A(1), A(1)))

    self.assertEqual(
        pg_diff(B(1, 2), B(1, 3), mode='diff'),
        Diff(B, B, children={'y': Diff(2, 3)}))
    self.assertEqual(
        pg_diff(B(1, 2), B(1, 3), mode='same'),
        Diff(B, B, children={'x': Diff(1, 1)}))
    self.assertEqual(
        pg_diff(B(1, 2), B(2, 2), mode='both'),
        Diff(B, B, children={
            'x': Diff(1, 2),
            'y': Diff(2, 2)
        }))

    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True, mode='diff'), Diff(B, C))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True, mode='same'),
        Diff(B, C, children={
            'x': Diff(1, 1),
            'y': Diff(2, 2)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 2), collapse=True, mode='both'),
        Diff(B, C, children={
            'x': Diff(1, 1),
            'y': Diff(2, 2)
        }))

    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, mode='diff'),
        Diff(B, C, children={
            'y': Diff(2, 3)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, mode='same'),
        Diff(B, C, children={
            'x': Diff(1, 1)
        }))
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, mode='both'),
        Diff(B, C, children={
            'x': Diff(1, 1),
            'y': Diff(2, 3)
        }))

  def test_flatten(self):

    @pg_members([('x', pg_typing.Any())])
    class A(Object):
      pass

    @pg_members([('x', pg_typing.Any()), ('y', pg_typing.Any())])
    class B(Object):
      pass

    class C(B):
      pass

    self.assertEqual(
        pg_diff(A(Dict(a=1, b=2, c=3)), A(Dict(a=1, b=3, d=4)), flatten=True),
        {
            'x.b': Diff(2, 3),
            'x.c': Diff(3, Diff.MISSING),
            'x.d': Diff(Diff.MISSING, 4)
        })
    self.assertEqual(
        pg_diff(B(1, 2), C(1, 3), collapse=True, flatten=True),
        {
            'y': Diff(2, 3),
            '_type': Diff(B, C),
        })

  def test_to_html(self):

    class Foo(Object):
      x: Any
      y: Any

    class Bar(Foo):
      pass

    def assert_style(html, expected):
      expected = inspect.cleandoc(expected).strip()
      actual = html.style_section.strip()
      if expected not in actual:
        print(actual)
      self.assertIn(expected, actual)

    def assert_content(html, expected):
      expected = inspect.cleandoc(expected).strip()
      actual = html.content.strip()
      if actual != expected:
        print(actual)
      self.assertEqual(actual.strip(), expected)

    assert_style(
        pg_diff(1, 1).to_html(),
        """
        /* Diff styles. */
        .has-diff.summary-title::after {
          content: ' (diff)';
          color: #aaa;
        }
        .has-diff.summary-title {
          background-color: yellow;
        }
        .no-diff.summary-title::after {
          content: ' (no diff)';
          font-style: italic;
          font-weight: normal;
          color: #aaa;
        }
        .no-diff {
          opacity: 0.6;
        }
        .no-diff.simple_value::after {
          content: '(no diff)';
          margin-left: 0.5em;
          color: #aaa;
          font-style: italic;
        }
        .diff-empty::before {
          content: '(empty)';
          font-style: italic;
          margin-left: 0.5em;
          color: #aaa;
        }
        .diff-left.summary-title::before, .diff-left.simple-value::before{
          content: '🇱';
        }
        .diff-left {
          background-color: #ffcccc;
        }
        .diff-right.summary-title::before, .diff-right.simple-value::before{
          content: '🇷';
        }
        .diff-right {
          background-color: #ccffcc;
        }
        """
    )

    # No diff leaf value (diff only)
    assert_content(
        pg_diff(1, 1).to_html(),
        """
        <details open class="pyglove diff no-diff"><summary><div class="summary-title no-diff">Diff</div><span class="tooltip no-diff">No diff</span></summary><span class="diff-empty"></span></details>
        """
    )

    # No diff leaf value (keep both)
    assert_content(
        pg_diff(1, 1, mode='both').to_html(),
        """
        <span class="simple-value int no-diff">1</span>
        """
    )
    # No diff complex value (diff only)
    assert_content(
        pg_diff(
            Foo(x=1, y=[Foo(x=2, y=3)]),
            Foo(x=1, y=[Foo(x=2, y=3)])
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff no-diff"><summary><div class="summary-title no-diff">Diff</div></summary><span class="diff-empty"></span></details>
        """
    )
    # No diff complex value.
    assert_content(
        pg_diff(
            Foo(x=1, y=[Foo(x=2, y=3)]),
            Foo(x=1, y=[Foo(x=2, y=3)]), mode='both'
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff no-diff"><summary><div class="summary-title no-diff">Foo(...)</div></summary><div class="complex-value foo no-diff"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details class="pyglove list"><summary><div class="summary-name">y</div><div class="summary-title">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove foo"><summary><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details></div></details></td></tr></table></div></details></div></details>
        """
    )

    # Diff on simple value.
    assert_content(
        pg_diff(1, 2).to_html(),
        """
        <div class="diff-value"><span class="simple-value int diff-left">1</span><span class="simple-value int diff-right">2</span></div>
        """
    )

    # Diff on list.
    assert_content(
        pg_diff([0, 1, 2], [0, 2], mode='both').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">List</div></summary><div class="complex-value list-class"><table><tr><td><span class="object-key int no-diff">0</span><span class="tooltip">[0]</span></td><td><span class="simple-value int no-diff">0</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><div class="diff-value"><span class="simple-value int diff-left">1</span><span class="simple-value int diff-right">2</span></div></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><div class="diff-value"><span class="simple-value int diff-left">2</span></div></td></tr></table></div></details>
        """
    )

    # Diff on dict.
    assert_content(
        pg_diff(dict(x=1, y=2, z=3), dict(x=1, y=3, w=4), mode='both').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">dict</div></summary><div class="complex-value dict-class"><table><tr><td><span class="object-key str no-diff">x</span><span class="tooltip">x</span></td><td><span class="simple-value int no-diff">1</span></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">y</span></td><td><div class="diff-value"><span class="simple-value int diff-left">2</span><span class="simple-value int diff-right">3</span></div></td></tr><tr><td><span class="object-key str">z</span><span class="tooltip">z</span></td><td><div class="diff-value"><span class="simple-value int diff-left">3</span></div></td></tr><tr><td><span class="object-key str">w</span><span class="tooltip">w</span></td><td><div class="diff-value"><span class="simple-value int diff-right">4</span></div></td></tr></table></div></details>
        """
    )

    # Diff on symbolic objects of the same type.
    assert_content(
        pg_diff(
            Foo(x=2, y=Foo(x=3, y=3)),
            Foo(x=1, y=Foo(x=2, y=3)),
            mode='both',
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">Foo</div></summary><div class="complex-value foo-class"><table><tr><td><span class="object-key str">x</span><span class="tooltip">x</span></td><td><div class="diff-value"><span class="simple-value int diff-left">2</span><span class="simple-value int diff-right">1</span></div></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">y</span></td><td><details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">Foo</div></summary><div class="complex-value foo-class"><table><tr><td><span class="object-key str">x</span><span class="tooltip">y.x</span></td><td><div class="diff-value"><span class="simple-value int diff-left">3</span><span class="simple-value int diff-right">2</span></div></td></tr><tr><td><span class="object-key str no-diff">y</span><span class="tooltip">y.y</span></td><td><span class="simple-value int no-diff">3</span></td></tr></table></div></details></td></tr></table></div></details>
        """
    )

    # Diff on symbolic objects of different types.
    assert_content(
        pg_diff(
            Foo(x=2, y=Foo(x=3, y=3)),
            Bar(x=2, y=Foo(x=3, y=3)),
            mode='both',
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <div class="diff-value"><details open class="pyglove foo diff-left"><summary><div class="summary-title diff-left">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details class="pyglove foo"><summary><div class="summary-name">y</div><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details></div></details></div></details><details open class="pyglove bar diff-right"><summary><div class="summary-title diff-right">Bar(...)</div></summary><div class="complex-value bar"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details class="pyglove foo"><summary><div class="summary-name">y</div><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details></div></details></div></details></div>
        """
    )

    # Different types but same values, with value collapsing.
    assert_content(
        pg_diff(
            Foo(x=2, y=Foo(x=3, y=3)),
            Bar(x=2, y=Bar(x=3, y=3)),
            mode='both', collapse=True,
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">Foo | Bar</div></summary><div class="complex-value foo-class"><table><tr><td><span class="object-key str no-diff">x</span><span class="tooltip">x</span></td><td><span class="simple-value int no-diff">2</span></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">y</span></td><td><details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">Foo | Bar</div></summary><div class="complex-value foo-class"><table><tr><td><span class="object-key str no-diff">x</span><span class="tooltip">y.x</span></td><td><span class="simple-value int no-diff">3</span></td></tr><tr><td><span class="object-key str no-diff">y</span><span class="tooltip">y.y</span></td><td><span class="simple-value int no-diff">3</span></td></tr></table></div></details></td></tr></table></div></details>
        """
    )

    # Different types and different values, with value collapsing.
    assert_content(
        pg_diff(
            Foo(x=2, y=Foo(x=3, y=3)),
            Bar(x=3, y=Bar(x=2, y=3)),
            mode='both', collapse=True,
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">Foo | Bar</div></summary><div class="complex-value foo-class"><table><tr><td><span class="object-key str">x</span><span class="tooltip">x</span></td><td><div class="diff-value"><span class="simple-value int diff-left">2</span><span class="simple-value int diff-right">3</span></div></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">y</span></td><td><details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">Foo | Bar</div></summary><div class="complex-value foo-class"><table><tr><td><span class="object-key str">x</span><span class="tooltip">y.x</span></td><td><div class="diff-value"><span class="simple-value int diff-left">3</span><span class="simple-value int diff-right">2</span></div></td></tr><tr><td><span class="object-key str no-diff">y</span><span class="tooltip">y.y</span></td><td><span class="simple-value int no-diff">3</span></td></tr></table></div></details></td></tr></table></div></details>
        """
    )

    # Diff with uncollapsing UI.
    assert_content(
        pg_diff(
            [
                Foo(1, 2), Foo(1, 2), None,
                Foo(x=1, y=Foo(2, 3)), [dict(x=1, y=2)]
            ],
            [
                Foo(1, 2), Bar(1, 2), dict(x=1),
                [1, 2], Foo(x=2, y=Foo(2, 4)), [dict(x=3)]
            ],
            mode='both'
        ).to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            uncollapse=['[0]', '[1].right', '[3].left.y'],
        ),
        """
        <details open class="pyglove diff has-diff"><summary><div class="summary-title has-diff">List</div></summary><div class="complex-value list-class"><table><tr><td><span class="object-key int no-diff">0</span><span class="tooltip">[0]</span></td><td><details open class="pyglove diff no-diff"><summary><div class="summary-title no-diff">Foo(...)</div></summary><div class="complex-value foo no-diff"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div></details></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">[1]</span></td><td><div class="diff-value"><details open class="pyglove foo diff-left"><summary><div class="summary-title diff-left">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div></details><details open class="pyglove bar diff-right"><summary><div class="summary-title diff-right">Bar(...)</div></summary><div class="complex-value bar"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div></details></div></td></tr><tr><td><span class="object-key int">2</span><span class="tooltip">[2]</span></td><td><div class="diff-value"><span class="simple-value none-type diff-left">None</span><details open class="pyglove dict diff-right"><summary><div class="summary-title diff-right">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details></div></details></div></td></tr><tr><td><span class="object-key int">3</span><span class="tooltip">[3]</span></td><td><div class="diff-value"><details open class="pyglove foo diff-left"><summary><div class="summary-title diff-left">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove foo"><summary><div class="summary-name">y</div><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details></div></details></div></details><details open class="pyglove list diff-right"><summary><div class="summary-title diff-right">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key int">1</span></td><td><span class="simple-value int">2</span></td></tr></table></div></details></div></td></tr><tr><td><span class="object-key int">4</span><span class="tooltip">[4]</span></td><td><div class="diff-value"><details open class="pyglove list diff-left"><summary><div class="summary-title diff-left">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove dict"><summary><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div></details></td></tr></table></div></details><details open class="pyglove foo diff-right"><summary><div class="summary-title diff-right">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details class="pyglove foo"><summary><div class="summary-name">y</div><div class="summary-title">Foo(...)</div></summary><div class="complex-value foo"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details open class="pyglove int"><summary><div class="summary-name">y</div><div class="summary-title">int</div></summary><span class="simple-value int">4</span></details></div></details></div></details></div></td></tr><tr><td><span class="object-key int">5</span><span class="tooltip">[5]</span></td><td><div class="diff-value"><details open class="pyglove list diff-right"><summary><div class="summary-title diff-right">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span></td><td><details class="pyglove dict"><summary><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">x</div><div class="summary-title">int</div></summary><span class="simple-value int">3</span></details></div></details></td></tr></table></div></details></div></td></tr></table></div></details>
        """
    )


if __name__ == '__main__':
  unittest.main()
