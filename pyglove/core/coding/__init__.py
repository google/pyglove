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
# pylint: disable=line-too-long
"""Code generation utilities."""

# pylint: enable=line-too-long
# pylint: disable=g-bad-import-order
# pylint: disable=g-importing-member

from pyglove.core.coding.errors import CodeError
from pyglove.core.coding.errors import SerializationError

from pyglove.core.coding.permissions import CodePermission
from pyglove.core.coding.permissions import permission
from pyglove.core.coding.permissions import get_permission

from pyglove.core.coding.parsing import parse

from pyglove.core.coding.execution import context
from pyglove.core.coding.execution import get_context
from pyglove.core.coding.execution import evaluate
from pyglove.core.coding.execution import sandbox_call
from pyglove.core.coding.execution import maybe_sandbox_call
from pyglove.core.coding.execution import run

from pyglove.core.coding.function_generation import NO_TYPE_ANNOTATION
from pyglove.core.coding.function_generation import make_function

# pylint: disable=line-too-long
# pylint: enable=g-bad-import-order
# pylint: enable=g-importing-member
