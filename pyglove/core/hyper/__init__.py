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
"""Hyper objects: representing template-based object space.

In PyGlove, an object space is represented by a hyper object, which is an
symbolic object that is placeheld by hyper primitives
(:class:`pyglove.hyper.HyperPrimitive`). Through hyper objects, object templates
(:class:`pyglove.hyper.ObjectTemplate`) can be obtained to generate objects
based on program genomes (:class:`pyglove.DNA`).

 .. graphviz::
    :align: center

    digraph hypers {
      node [shape="box"];
      edge [arrowtail="empty" arrowhead="none" dir="back" style="dashed"];
      hyper [label="HyperValue" href="hyper_value.html"];
      template [label="ObjectTemplate" href="object_template.html"];
      primitive [label="HyperPrimitive" href="hyper_primitive.html"];
      choices [label="Choices" href="choices.html"];
      oneof [label="OneOf" href="oneof_class.html"];
      manyof [label="ManyOf" href="manyof_class.html"];
      float [label="Float" href="float.html"];
      custom [label="CustomHyper" href="custom_hyper.html"];
      hyper -> template;
      hyper -> primitive;
      primitive -> choices;
      choices -> oneof;
      choices -> manyof;
      primitive -> float;
      primitive -> custom
    }

Hyper values map 1:1 to genotypes as the following:

+-------------------------------------+----------------------------------------+
| Hyper class                         | Genotype class                         |
+=====================================+========================================+
|:class:`pyglove.hyper.HyperValue`    |:class:`pyglove.DNASpec`                |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.ObjectTemplate`|:class:`pyglove.geno.Space`             |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.HyperPrimitive`|:class:`pyglove.geno.DecisionPoint`     |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.Choices`       |:class:`pyglove.geno.Choices`           |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.Float`         |:class:`pyglove.geno.Float`             |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.CustomHyper` :class:`pyglove.geno.CustomDecisionPoint`  |
+------------------------------------------------------------------------------+
"""

# pylint: disable=g-bad-import-order

# The hyper value interface and hyper primitives.
from pyglove.core.hyper.base import HyperValue
from pyglove.core.hyper.base import HyperPrimitive

from pyglove.core.hyper.categorical import Choices
from pyglove.core.hyper.categorical import OneOf
from pyglove.core.hyper.categorical import ManyOf
from pyglove.core.hyper.numerical import Float
from pyglove.core.hyper.custom import CustomHyper

from pyglove.core.hyper.evolvable import Evolvable
from pyglove.core.hyper.evolvable import MutationType
from pyglove.core.hyper.evolvable import MutationPoint

# Helper functions for creating hyper values.
from pyglove.core.hyper.categorical import oneof
from pyglove.core.hyper.categorical import manyof
from pyglove.core.hyper.categorical import permutate
from pyglove.core.hyper.numerical import floatv
from pyglove.core.hyper.evolvable import evolve

# Object template and helper functions.
from pyglove.core.hyper.object_template import ObjectTemplate
from pyglove.core.hyper.object_template import template
from pyglove.core.hyper.object_template import materialize
from pyglove.core.hyper.object_template import dna_spec

from pyglove.core.hyper.derived import DerivedValue
from pyglove.core.hyper.derived import ValueReference
from pyglove.core.hyper.derived import reference

# Classes and functions for dynamic evaluation.
from pyglove.core.hyper.dynamic_evaluation import dynamic_evaluate
from pyglove.core.hyper.dynamic_evaluation import DynamicEvaluationContext
from pyglove.core.hyper.dynamic_evaluation import trace


# Helper functions for iterating examples from the search space.
from pyglove.core.hyper.iter import iterate
from pyglove.core.hyper.iter import random_sample


# Alias for backward compatibility:
ChoiceList = ManyOf
ChoiceValue = OneOf
Template = ObjectTemplate
one_of = oneof
sublist_of = manyof
float_value = floatv
search_space = dna_spec


# pylint: enable=g-bad-import-order
