# Copyright 2024 The PyGlove Authors
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
"""PyGlove views."""

from pyglove.core.views import base
from pyglove.core.views import html

View = base.View
view = base.view
view_options = base.view_options

# Pytype annotation.
NodeFilter = base.NodeFilter

Html = html.Html
HtmlView = html.HtmlView
HtmlTreeView = html.HtmlTreeView

to_html = html.to_html
to_html_str = html.to_html_str
