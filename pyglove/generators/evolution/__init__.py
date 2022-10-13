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
"""PyGlove's genetic computing framework."""

from pyglove.generators import scalars
from pyglove.generators.evolution import base
from pyglove.generators.evolution import hill_climb as hill_climb_lib
from pyglove.generators.evolution import mutators
from pyglove.generators.evolution import neat as neat_lib
from pyglove.generators.evolution import nsga2 as nsga2_lib
from pyglove.generators.evolution import recombinators
from pyglove.generators.evolution import regularized_evolution as regularized_evolution_lib
from pyglove.generators.evolution import selectors
from pyglove.generators.evolution.hill_climb import hill_climb
from pyglove.generators.evolution.neat import neat
from pyglove.generators.evolution.nsga2 import nsga2
from pyglove.generators.evolution.regularized_evolution import regularized_evolution

# Alias for backward compatibility.
# Remove once dependencies are fixed.
RegularizedEvolution = regularized_evolution

# Interfaces.
Operation = base.Operation
DNAOperation = base.DNAOperation
Selector = base.Selector
Recombinator = base.Recombinator
Mutator = base.Mutator
Scalar = scalars.Scalar

# The compositional evolution class.
Evolution = base.Evolution

# Compositional operators.
# Common operations that are not associated with an operator.
Lambda = base.Lambda

Pipeline = base.Pipeline                         # operator >>
Power = base.Power                               # operator **

Concatenation = base.Concatenation                # operator +
Slice = base.Slice                                # operator []
Repeat = base.Repeat                              # operator *

Identity = base.Identity
Union = base.Union                                # operator |
Intersection = base.Intersection                  # operator &
Difference = base.Difference                      # operator -
SymmetricDifference = base.SymmetricDifference    # operator ^
Inversion = base.Inversion                        # operator ~

Choice = base.Choice                             # .with_prob
Conditional = base.Conditional                   # .if_true/.if_false
ElementWise = base.ElementWise                   # .for_each
Flatten = base.Flatten                           # .flatten
UntilChange = base.UntilChange                   # .until_change


GlobalStateGetter = base.GlobalStateGetter       # .global_state
GlobalStateSetter = base.GlobalStateSetter       # .as_global_state

# Helper method.
scalar_spec = scalars.scalar_spec
scalar_value = scalars.scalar_value

set_fitness = base.set_fitness
get_fitness = base.get_fitness

get_generation_id = base.get_generation_id
get_feedback_sequence_number = base.get_feedback_sequence_number
is_initial_population = base.is_initial_population


