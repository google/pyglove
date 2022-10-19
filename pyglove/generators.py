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
"""Sub-module for exposing PyGlove DNA generators."""

# NOTE(daiyip): We disable bad-import-order to preserve the relation of
# imported symbols
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order

from pyglove import core
from pyglove import ext

# NOTE(daiyip): For backward compatibility.
evolution = ext.evolution
RegularizedEvolution = ext.evolution.RegularizedEvolution
evolution_mutators = ext.evolution.mutators

geno = core.geno
Sweeping = geno.Sweeping
Random = geno.Random
Deduping = geno.Deduping

# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order
