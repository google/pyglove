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
"""Systematic patching on symbolic values.

As :meth:`pyglove.Symbolic.rebind` provides a flexible programming
interface for modifying symbolic values, why bother to have this module?
Here are the  motivations:

  * Provide user friendly methods for addressing the most common patching
    patterns.

  * Provide a systematic solution for

    * Patch semantic groups.
    * Enable combination of these groups.
    * Provide an interface that patching can be invoked from the command line.
"""

# pylint: disable=g-bad-import-order

# Pattern-based patching.

from pyglove.core.patching.pattern_based import patch_on_key
from pyglove.core.patching.pattern_based import patch_on_path
from pyglove.core.patching.pattern_based import patch_on_type
from pyglove.core.patching.pattern_based import patch_on_value
from pyglove.core.patching.pattern_based import patch_on_member

# Patcher: modular rule-based patching.
from pyglove.core.patching.rule_based import patcher
from pyglove.core.patching.rule_based import patch

from pyglove.core.patching.rule_based import Patcher
from pyglove.core.patching.rule_based import from_uri

from pyglove.core.patching.rule_based import patcher_names
from pyglove.core.patching.rule_based import allow_repeated_patcher_registration

# Object factory based on patchers.
from pyglove.core.patching.object_factory import object_factory


# pylint: enable=g-bad-import-order
