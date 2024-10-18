# Copyright 2024 The Langfun Authors
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

import copy
import functools
import os
import tempfile
from typing import Any, Callable, Iterable, Optional, Union
import unittest

from pyglove.core import io as pg_io
from pyglove.core.views import base

Content = base.Content
View = base.View


class ReferenceLinks(Content.SharedParts):

  @functools.cached_property
  def content(self) -> str:
    return '\n'.join(
        [
            f'link={url}' for url in self.parts.keys()
        ]
    )


class SharedPartsTest(unittest.TestCase):

  def test_basics(self):
    links = ReferenceLinks()
    self.assertFalse(links)
    self.assertEqual(links.content, '')
    self.assertEqual(list(links), [])

    links = ReferenceLinks('https://x/y.css')
    self.assertEqual(links.content, 'link=https://x/y.css')
    self.assertTrue(links)
    self.assertIn('https://x/y.css', links)
    self.assertNotIn('https://x/z.css', links)
    self.assertEqual(list(links), ['https://x/y.css'])

    links = ReferenceLinks('https://x/y.css', 'https://x/z.css')
    self.assertEqual(
        links.content,
        'link=https://x/y.css\nlink=https://x/z.css'
    )
    self.assertTrue(links)
    self.assertIn('https://x/y.css', links)
    self.assertIn('https://x/z.css', links)
    self.assertEqual(list(links), ['https://x/y.css', 'https://x/z.css'])

  def test_add(self):
    links = ReferenceLinks('https://x/y.css')
    self.assertEqual(links.content, 'link=https://x/y.css')

    # Updated.
    self.assertTrue(links.add('https://x/z.css'))
    self.assertEqual(links.parts, {'https://x/y.css': 1, 'https://x/z.css': 1})
    # Make sure the cached property `content` is updated.
    self.assertEqual(
        links.content,
        'link=https://x/y.css\nlink=https://x/z.css'
    )
    # Not updated but the number of reference gets updated.
    self.assertFalse(links.add('https://x/z.css'))
    self.assertEqual(links.parts, {'https://x/y.css': 1, 'https://x/z.css': 2})

    # Adding None is ignored.
    self.assertFalse(links.add(None))
    self.assertEqual(links.parts, {'https://x/y.css': 1, 'https://x/z.css': 2})

    # Add another instance of `ReferenceLinks`.
    self.assertTrue(
        links.add(ReferenceLinks('https://x/y.css', 'https://x/w.css'))
    )
    self.assertEqual(
        links.parts,
        {
            'https://x/y.css': 2,
            'https://x/z.css': 2,
            'https://x/w.css': 1
        }
    )

  def test_copy(self):
    links = ReferenceLinks('https://x/y.css')
    links2 = copy.copy(links)
    self.assertEqual(links, links2)
    self.assertEqual(links.parts, links2.parts)

  def test_eq(self):
    links1 = ReferenceLinks('https://x/y.css')
    links2 = ReferenceLinks('https://x/y.css')
    self.assertEqual(links1, links2)
    self.assertNotEqual(links1, 'abc')
    self.assertNotEqual(links1, ReferenceLinks('https://x/z.css'))

  def test_format(self):
    links = ReferenceLinks('https://x/y.css')
    self.assertEqual(
        repr(links),
        """ReferenceLinks(parts={'https://x/y.css': 1})"""
    )
    self.assertEqual(
        str(links),
        """link=https://x/y.css"""
    )


class Document(Content):

  WritableTypes = Union[    # pylint: disable=invalid-name
      str, 'Document', None, Callable[[], Union[str, 'Document', None]]
  ]

  def __init__(   # pylint: disable=useless-super-delegation
      self,
      *content: WritableTypes, ref_links: Optional[Iterable[str]] = None):
    super().__init__(*content, ref_links=ReferenceLinks(*(ref_links or [])))

  @property
  def ref_links(self) -> ReferenceLinks:
    return self._shared_parts['ref_links']

  def to_str(self, *, content_only: bool = False, **kwargs):
    if content_only:
      return self.content
    return self.ref_links.content + '\n' + self.content


class ContentTest(unittest.TestCase):

  def test_basics(self):
    document = Document()
    self.assertEqual(document.content, '')

    document = Document('abc', 'def', ref_links=['https://x/y.css'])
    self.assertEqual(document.content, 'abcdef')
    self.assertEqual(document.to_str(content_only=True), 'abcdef')
    self.assertEqual(document.to_str(), 'link=https://x/y.css\nabcdef')
    self.assertEqual(document.shared_parts, {'ref_links': document.ref_links})
    self.assertEqual(
        Document('abc', ref_links=['https://x/z.css']),
        Document('abc', ref_links=['https://x/z.css'])
    )
    self.assertNotEqual(
        Document('abc', ref_links=['https://x/y.css']),
        'abc'
    )
    self.assertNotEqual(
        Document('abc', ref_links=['https://x/y.css']),
        Document('abc', ref_links=['https://x/z.css'])
    )
    self.assertEqual(
        hash(Document('abc', ref_links=['https://x/y.css'])),
        hash(Document('abc', ref_links=['https://x/y.css']))
    )
    self.assertNotEqual(
        hash(Document('abc', ref_links=['https://x/y.css'])),
        hash(Document('abc', ref_links=['https://x/z.css']))
    )

  def test_write(self):
    document = Document('abc', ref_links=['https://x/y.css'])
    self.assertEqual(document.content, 'abc')

    self.assertIs(document.write('def'), document)
    self.assertEqual(document.content, 'abcdef')

    # No op.
    document.write(None)
    self.assertEqual(document.content, 'abcdef')

    # Write another instance of `Document`.
    document.write(
        Document('ghi', ref_links=['https://x/y.css', 'https://x/z.css'])
    )
    self.assertEqual(document.content, 'abcdefghi')
    self.assertEqual(
        document.to_str(),
        'link=https://x/y.css\nlink=https://x/z.css\nabcdefghi'
    )

    # Write a lambda.
    document.write(lambda: 'jkl')
    self.assertEqual(document.content, 'abcdefghijkl')
    self.assertEqual(
        document.to_str(),
        'link=https://x/y.css\nlink=https://x/z.css\nabcdefghijkl'
    )

    document.write(lambda: None)
    self.assertEqual(document.content, 'abcdefghijkl')

    document.write(lambda: Document('mno'))
    self.assertEqual(document.content, 'abcdefghijklmno')

  def test_save(self):
    filename = os.path.join(tempfile.gettempdir(), '1', 'test_doc.txt')
    Document('abc', ref_links=['https://x/y.css']).save(filename)
    self.assertTrue(pg_io.path_exists(filename))
    self.assertEqual(pg_io.readfile(filename), 'link=https://x/y.css\nabc')

  def test_add_and_radd(self):
    self.assertEqual(
        (Document('abc', ref_links=['https://x/y.css']) + 'def').to_str(),
        'link=https://x/y.css\nabcdef'
    )
    self.assertEqual(
        ('def' + Document('abc', ref_links=['https://x/y.css'])).to_str(),
        'link=https://x/y.css\ndefabc'
    )
    self.assertEqual(
        (Document('abc', ref_links=['https://x/y.css']) + None).to_str(),
        'link=https://x/y.css\nabc'
    )
    self.assertEqual(
        (None + Document('abc', ref_links=['https://x/y.css'])).to_str(),
        'link=https://x/y.css\nabc'
    )
    self.assertEqual(
        (Document('abc', ref_links=['https://x/y.css'])
         + (lambda: 'def')).to_str(),
        'link=https://x/y.css\nabcdef'
    )
    self.assertEqual(
        ((lambda: 'def')
         + Document('abc', ref_links=['https://x/y.css'])).to_str(),
        'link=https://x/y.css\ndefabc'
    )

  def test_format(self):
    document = Document('abc', ref_links=['https://x/y.css'])
    self.assertEqual(
        repr(document),
        (
            'Document(content=\'abc\', '
            'ref_links=ReferenceLinks(parts={\'https://x/y.css\': 1}))'
        )
    )
    self.assertEqual(
        str(document),
        'link=https://x/y.css\nabc'
    )

  def test_from_value(self):
    document = Document('abc', ref_links=['https://x/y.css'])
    self.assertIs(Document.from_value(document), document)
    document2 = Document.from_value(document, copy=True)
    self.assertIsNot(document2, document)
    self.assertEqual(document2, document)

    self.assertIsNone(Document.from_value(None))
    self.assertEqual(Document.from_value('abc'), Document('abc'))
    self.assertEqual(
        Document.from_value(lambda: 'abc'), Document('abc')
    )


class TestView(View):

  VIEW_ID = 'test'

  class Extension(View.Extension):

    def __init__(self, x: int):
      super().__init__()
      self.x = x

    def __str__(self):
      return f'{self.__class__.__name__}(x={self.x})'

    def _test_view_render(
        self,
        *,
        view: 'TestView',
        use_summary: bool = False,
        use_content: bool = False,
        **kwargs):
      if use_summary:
        return view.summary(self, **kwargs)
      if use_content:
        return view.content(self, **kwargs)
      return Document(f'[Custom Document] {self}')

    def _test_view_summary(
        self,
        view: 'TestView',
        default_behavior: bool = False,
        **kwargs):
      if default_behavior:
        return view.summary(self, **kwargs)
      return Document(f'[Custom Summary] {self}')

    def _test_view_content(self, **kwargs):
      del kwargs
      return Document(f'[Custom Content] {self}')

    def bar(self, x: int = 2):
      return x

  @View.extension_method('_test_view_render')
  def render(
      self,
      value: Any,
      *,
      greeting: str = 'hi',
      **kwargs):
    return Document(f'{greeting} {value}')

  @View.extension_method('_test_view_summary')
  def summary(
      self,
      value: Any,
      title: str = 'abc',
      **kwargs
  ):
    del kwargs
    return Document(f'[Default Summary] {title}: {value}')

  @View.extension_method('_test_view_content')
  def content(self, value: Any, **kwargs):
    del kwargs
    return Document(f'[Default Content] {value}')

  def render_key(self, key: str, *, value: Optional[Any] = None, **kwargs):
    del kwargs
    return Document(f'[Default Key] {key}: {value}')

  def foo(self, x: int = 1):
    return x


class TestView2(View):

  VIEW_ID = 'test2'

  class Extension(View.Extension):

    def _test_view2_render(self, value: Any, **kwargs):
      del kwargs
      return Document(f'<h2>{value}</h2>')

    def _test_view2_tooltip(
        self,
        **kwargs,
    ):
      del kwargs
      return Document('<h2>Tooltip</h2>')

  @View.extension_method('_test_view2_render')
  def render(self, value: Any, **kwargs):
    return Document(f'<h1>{value}</h1>')

  @View.extension_method('_test_view2_tooltip')
  def tooltip(self, *, value: Optional[Any] = None, **kwargs):
    del kwargs
    return Document(f'<tooltip>{value}</tooltip>')


class MyObject(TestView.Extension, TestView2.Extension):

  def __init__(self, x: int, y: str):
    super().__init__(x)
    self.y = y

  def _test_view_content(
      self, *, text: str = 'MyObject', **kwargs
  ):
    return Document(f'[{text}] {self.x} {self.y}')

  def _test_view2_render(self, **kwargs):
    return Document(f'<span>{self.x}</span>')


class MyObject2(MyObject):

  def _doc_content(self, **kwargs):
    del kwargs
    return Document('Nothing')


class ViewTest(unittest.TestCase):

  def test_bad_views(self):
    # VIEW ID must be present for concrete View classes.
    with self.assertRaisesRegex(
        ValueError, '`VIEW_ID` must be set'
    ):
      class NoTypeNonAbstractView(View):  # pylint: disable=unused-variable

        def render(self, *args, **kwargs):
          return Document('foo')

    with self.assertRaisesRegex(
        TypeError, 'must have a `value` argument'
    ):
      class NodeMethodWithoutValue(View):  # pylint: disable=unused-variable

        def render(self, *args, **kwargs):
          return Document('foo')

        @View.extension_method('_some_method')
        def bad_method(self, text: str):
          pass

    with self.assertRaisesRegex(
        TypeError, 'must not have variable positional argument.'
    ):
      class NodeMethodWithVarargs(View):  # pylint: disable=unused-variable

        def render(self, *args, **kwargs):
          return Document('foo')

        @View.extension_method('_some_method')
        def bad_method(self, value, *args):
          pass

    class AbstractView(View):
      VIEW_ID = 'abstract'

    with self.assertRaisesRegex(
        ValueError, 'The `VIEW_ID` .* is the same as the base class'
    ):
      class AbstractViewWithNodeMethod(AbstractView):  # pylint: disable=unused-variable

        def render(self, *args, **kwargs):
          return Document('foo')

  def test_create(self):
    view = View.create('test')
    self.assertIsInstance(view, TestView)
    with self.assertRaisesRegex(ValueError, 'No view class found'):
      View.create('nonexisted_view')

  def test_dir(self):
    self.assertIn('test', View.dir())

  def test_supported_view_classes(self):
    self.assertEqual(
        TestView.Extension.supported_view_classes(),
        {TestView}
    )
    self.assertEqual(
        MyObject.supported_view_classes(),
        {TestView, TestView2}
    )

  def test_view_options(self):
    view = View.create('test')
    self.assertEqual(view.foo(), 1)
    with base.view_options(text='hello', use_content=True):
      self.assertEqual(
          base.view(MyObject(1, 'a'), view_id='test').content,
          '[hello] 1 a'
      )

  def test_view_default_behavior(self):
    view = View.create('test')
    self.assertEqual(
        view.render(1).content,
        'hi 1'
    )
    self.assertEqual(
        view.summary(value=1).content,
        '[Default Summary] abc: 1'
    )
    self.assertEqual(
        view.content(1).content,
        '[Default Content] 1'
    )
    self.assertEqual(
        view.render_key(1).content,
        '[Default Key] 1: None'
    )
    with self.assertRaisesRegex(
        ValueError, 'No value is provided for the `value` argument'
    ):
      view.render()

  def test_custom_behavior(self):
    view = View.create('test')
    self.assertEqual(
        view.render(MyObject(1, 'foo'), use_content=True).content,
        '[MyObject] 1 foo'
    )

    self.assertEqual(
        view.render(TestView.Extension(1)).content,
        '[Custom Document] Extension(x=1)'
    )
    self.assertEqual(
        view.summary(TestView.Extension(1)).content,
        '[Custom Summary] Extension(x=1)'
    )
    self.assertEqual(
        view.content(TestView.Extension(1)).content,
        '[Custom Content] Extension(x=1)'
    )
    self.assertEqual(
        view.render_key('foo', value=TestView.Extension(1)).content,
        '[Default Key] foo: Extension(x=1)'
    )

    view2 = View.create('test2')
    self.assertEqual(
        view2.render(TestView.Extension(1)).content,
        '<h1>Extension(x=1)</h1>'
    )
    self.assertEqual(
        view2.render(MyObject(1, 'foo')).content,
        '<span>1</span>'
    )
    self.assertEqual(
        view2.tooltip(value=MyObject(1, 'foo')).content,
        '<h2>Tooltip</h2>'
    )

  def test_advanced_routing_with_arg_passing(self):
    view = View.create('test')
    self.assertEqual(
        view.render(
            TestView.Extension(1), use_summary=True).content,
        '[Custom Summary] Extension(x=1)'
    )
    self.assertEqual(
        view.render(
            TestView.Extension(1), use_content=True).content,
        '[Custom Content] Extension(x=1)'
    )
    self.assertEqual(
        view.render(
            TestView.Extension(1),
            use_summary=True,
            default_behavior=True
        ).content,
        '[Default Summary] abc: Extension(x=1)'
    )
    self.assertEqual(
        base.view(
            TestView.Extension(1),
            view_id='test',
            use_summary=True,
            title='def',
            default_behavior=True,
            content_only=True,
        ).content,
        '[Default Summary] def: Extension(x=1)'
    )

  def test_view_selection_with_multi_inheritance(self):
    self.assertEqual(
        base.view(MyObject(1, 'foo'), view_id='test').content,
        '[Custom Document] MyObject(x=1)'
    )
    self.assertEqual(
        base.view(MyObject(1, 'foo'), view_id='test2').content,
        '<span>1</span>'
    )

  def test_view_with_content_input(self):
    doc = Document('abc')
    self.assertIs(base.view(doc), doc)


if __name__ == '__main__':
  unittest.main()
