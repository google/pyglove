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
"""Distributed tuning with pluggable backends.

:func:`pyglove.iter` provides an interface for sampling examples from a search
space within a process. To support distributed tuning, PyGlove introduces
:func:`pyglove.sample`, which is almost identical but with more features:

 * Allow multiple worker processes (aka. workers) to collaborate on a search
   with failover handling.
 * Each worker can process different trials, or can cowork on the same trials
   via work groups.
 * Provide APIs for communicating between the co-workers.
 * Provide API for retrieving the search results.
 * Provide a pluggable backend system for supporting user infrastructures.

"""

# pylint: disable=g-bad-import-order

# User facing APIs for tuning.
from pyglove.core.tuning.sample import sample
from pyglove.core.tuning.backend import poll_result

from pyglove.core.tuning.backend import default_backend
from pyglove.core.tuning.backend import set_default_backend

# Tuning protocols.
from pyglove.core.tuning.protocols import Measurement
from pyglove.core.tuning.protocols import Trial
from pyglove.core.tuning.protocols import Result
from pyglove.core.tuning.protocols import Feedback
from pyglove.core.tuning.protocols import RaceConditionError

# Interface for early stopping.
from pyglove.core.tuning.early_stopping import EarlyStoppingPolicy

# Interfaces for tuning backend developers.
from pyglove.core.tuning.backend import Backend
from pyglove.core.tuning.backend import BackendFactory
from pyglove.core.tuning.backend import add_backend
from pyglove.core.tuning.backend import available_backends

# Importing local backend.
import pyglove.core.tuning.local_backend

# pylint: enable=g-bad-import-order
