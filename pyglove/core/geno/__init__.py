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
"""Program genome and genotypes.

Program genome (DNA) is a representation for encoding actions for manipulating
symbolic objects. Genotypes (:class:`pyglove.DNASpec`) are the specification on
how to generate them. Genotypes are separated from their corresponding hyper
values (:class:`pyglove.HyperValue`) which generate client-side objects, in the
aim to decouple the algorithms that generate genomes from the ones that consume
them. As a result, the algorithms can be applied on different user programs
for optimization.

.. graphviz::
   :align: center

    digraph genotypes {
      node [shape="box"];
      edge [arrowtail="empty" arrowhead="none" dir="back" style="dashed"];
      dna_spec [label="DNASpec" href="dna_spec.html"]
      space [label="Space" href="space_class.html"];
      dp [label="DecisionPoint" href="decision_point.html"];
      choices [label="Choices" href="choices.html"];
      float [label="Float" href="float.html"];
      custom [label="CustomDecisionPoint" href="custom_decision_point.html"];
      dna [label="DNA", href="dna.html"]
      dna_spec -> space;
      dna_spec -> dp;
      space -> dp [arrowtail="diamond" style="none" label="elements"];
      dp -> choices;
      choices -> space [arrowtail="diamond" style="none" label="candidates"];
      dp -> float;
      dp -> custom;
      dna -> dna [arrowtail="diamond" style="none" label="children"];
      dna -> dna_spec [arrowhead="normal" dir="forward" style="none"
                       label="spec"];
    }

Genotypes map 1:1 to hyper primitives as the following:

+-------------------------------------+--------------------------------------+
| Genotype class                      | Hyper class                          |
+=====================================+======================================+
|:class:`pg.DNASpec`                  | :class:`pg.hyper.HyperValue`         |
+-------------------------------------+--------------------------------------+
|:class:`pg.geno.Space`               | :class:`pg.hyper.ObjectTemplate`     |
+-------------------------------------+--------------------------------------+
|:class:`pg.geno.DecisionPoint`       | :class:`pg.hyper.HyperPrimitive`     |
+-------------------------------------+--------------------------------------+
|:class:`pg.geno.Choices`             | :class:`pg.hyper.Choices`            |
|                                     | (:func:`pg.oneof`, :func:`pg.manyof`)|
+-------------------------------------+--------------------------------------+
|:class:`pg.geno.Float`               | :class:`pg.floatv`                   |
+-------------------------------------+--------------------------------------+
|:class:`pg.geno.CustomDecisionPoint` | :class:`pg.hyper.CustomHyper`        |
|                                     | (:func:`pg.evolve`)                  |
+-------------------------------------+--------------------------------------+
"""

# pylint: disable=g-bad-import-order

from pyglove.core.geno.base import AttributeDict
from pyglove.core.geno.base import DNA

# DNA specifications
from pyglove.core.geno.base import DNASpec
from pyglove.core.geno.base import DecisionPoint
from pyglove.core.geno.space import Space
from pyglove.core.geno.categorical import Choices
from pyglove.core.geno.numerical import Float
from pyglove.core.geno.custom import CustomDecisionPoint

# Helper functions for creating DNA specifications.
from pyglove.core.geno.space import constant
from pyglove.core.geno.space import space
from pyglove.core.geno.categorical import oneof
from pyglove.core.geno.categorical import manyof
from pyglove.core.geno.numerical import floatv
from pyglove.core.geno.custom import custom

from pyglove.core.geno.numerical import float_scale_spec

# DNA generators.
from pyglove.core.geno.dna_generator import DNAGenerator
from pyglove.core.geno.dna_generator import dna_generator

from pyglove.core.geno.random import Random
from pyglove.core.geno.sweeping import Sweeping
from pyglove.core.geno.deduping import Deduping

from pyglove.core.geno.random import random_dna

# Helper classes and functions.
from pyglove.core.geno.base import ConditionalKey




# pylint: enable=g-bad-import-order
