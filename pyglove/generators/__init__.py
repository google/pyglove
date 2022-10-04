# Copyright 2019 The PyGlove Authors
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
"""Package pyglove.generators."""

# NOTE(daiyip): We disable bad-import-order to preserve the relation of
# imported symbols
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order

from pyglove import core
from pyglove.generators import early_stopping
from pyglove.generators import evolution

geno = core.geno
Sweeping = geno.Sweeping
Random = geno.Random
Deduping = geno.Deduping

# TODO(daiyip): The following exports are for backward compatibility only
# Remove once the dependencies are removed.

DNA = geno.DNA

# Symbols for hyper values.
# Remove them once legacy dependencies are cleared.
Encoder = core.hyper.HyperValue
Template = core.hyper.ObjectTemplate
ChoiceValue = core.hyper.OneOf
ChoiceList = core.hyper.ManyOf
Float = core.hyper.Float

template = core.template
oneof = core.oneof
one_of = core.oneof
manyof = core.manyof
sublist_of = core.manyof
floatv = core.floatv
float_value = core.floatv
permutate = core.permutate

dna_spec = core.dna_spec
iterate = core.iter

# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order
