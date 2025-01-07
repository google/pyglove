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
import inspect
import os
import unittest
from pyglove.core.utils import text_color


class TextColorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ.pop('ANSI_COLORS_DISABLED', None)
    os.environ.pop('NO_COLOR', None)
    os.environ['FORCE_COLOR'] = '1'

  def test_colored_block_without_colors_and_styles(self):
    self.assertEqual(text_color.colored_block('foo', '{{', '}}'), 'foo')

  def test_colored_block(self):
    original_text = inspect.cleandoc("""
        Hi << foo >>
        <# print x if x is present #>
        <% if x %>
        << x >>
        <% endif %>
        """)

    colored_text = text_color.colored_block(
        text_color.colored(original_text, color='blue'),
        '<<', '>>',
        color='white',
        background='blue',
    )
    origin_color = '\x1b[34m'
    reset = '\x1b[0m'
    block_color = text_color.colored(
        'TEXT', color='white', background='blue'
    ).split('TEXT')[0]
    self.assertEqual(
        colored_text,
        f'{origin_color}Hi {block_color}<< foo >>{reset}{origin_color}\n'
        '<# print x if x is present #>\n<% if x %>\n'
        f'{block_color}<< x >>{reset}{origin_color}\n'
        f'<% endif %>{reset}'
    )
    self.assertEqual(text_color.decolor(colored_text), original_text)

  def test_colored_block_without_full_match(self):
    self.assertEqual(
        text_color.colored_block(
            'Hi {{ foo',
            '{{', '}}',
            color='white',
            background='blue',
        ),
        'Hi {{ foo'
    )

  def test_colored_block_without_termcolor(self):
    termcolor = text_color.termcolor
    text_color.termcolor = None
    original_text = inspect.cleandoc("""
        Hi {{ foo }}
        {# print x if x is present #}
        {% if x %}
        {{ x }}
        {% endif %}
        """)

    colored_text = text_color.colored_block(
        text_color.colored(original_text, color='blue'),
        '{{', '}}',
        color='white',
        background='blue',
    )
    self.assertEqual(colored_text, original_text)
    self.assertEqual(text_color.decolor(colored_text), original_text)
    text_color.termcolor = termcolor


if __name__ == '__main__':
  unittest.main()
