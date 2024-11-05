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
"""Common HTML controls."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order

from pyglove.core.views.html.controls.base import HtmlControl


from pyglove.core.views.html.controls.label import Label
from pyglove.core.views.html.controls.label import LabelGroup
from pyglove.core.views.html.controls.label import Badge

from pyglove.core.views.html.controls.tooltip import Tooltip

from pyglove.core.views.html.controls.tab import Tab
from pyglove.core.views.html.controls.tab import TabControl

from pyglove.core.views.html.controls.progress_bar import ProgressBar
from pyglove.core.views.html.controls.progress_bar import SubProgress

# pylint: enable=g-bad-import-order
# pylint: enable=g-importing-member
