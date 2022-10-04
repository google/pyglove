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
"""PyGlove's built-in early stopping policies."""

# pylint: disable=g-bad-import-order

from pyglove.generators.early_stopping.base import EarlyStopingPolicyBase
from pyglove.generators.early_stopping.base import And
from pyglove.generators.early_stopping.base import Or
from pyglove.generators.early_stopping.base import Not

from pyglove.generators.early_stopping.step_wise import early_stop_by_rank
from pyglove.generators.early_stopping.step_wise import early_stop_by_value
from pyglove.generators.early_stopping.step_wise import StepWise

# pylint: enable=g-bad-import-order
