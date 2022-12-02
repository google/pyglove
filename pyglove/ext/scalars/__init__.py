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
"""Step-based scalars used as hyper-parameters for PyGlove algorithms."""

# pylint: disable=g-bad-import-order

# Helper methods for creating and retrieving scalars.
from pyglove.ext.scalars.base import scalar_spec
from pyglove.ext.scalars.base import scalar_value
from pyglove.ext.scalars.base import make_scalar

# Interface and common scalars.
from pyglove.ext.scalars.base import Scalar

from pyglove.ext.scalars.base import Lambda
from pyglove.ext.scalars.base import Constant
from pyglove.ext.scalars.base import STEP

from pyglove.ext.scalars.base import UnaryOp
from pyglove.ext.scalars.base import Negation
from pyglove.ext.scalars.base import Absolute
from pyglove.ext.scalars.base import Floor
from pyglove.ext.scalars.base import Ceiling

from pyglove.ext.scalars.base import BinaryOp
from pyglove.ext.scalars.base import Addition
from pyglove.ext.scalars.base import Substraction
from pyglove.ext.scalars.base import Multiplication
from pyglove.ext.scalars.base import Division
from pyglove.ext.scalars.base import Mod
from pyglove.ext.scalars.base import Power

# Common math functions.
from pyglove.ext.scalars.maths import linear
from pyglove.ext.scalars.maths import cosine_decay
from pyglove.ext.scalars.maths import exponential_decay
from pyglove.ext.scalars.maths import cyclic
from pyglove.ext.scalars.maths import sqrt
from pyglove.ext.scalars.maths import exp
from pyglove.ext.scalars.maths import log
from pyglove.ext.scalars.maths import cos
from pyglove.ext.scalars.maths import sin

from pyglove.ext.scalars.maths import SquareRoot
from pyglove.ext.scalars.maths import Exp
from pyglove.ext.scalars.maths import Log
from pyglove.ext.scalars.maths import Cosine
from pyglove.ext.scalars.maths import Sine

# Common random scalars.
from pyglove.ext.scalars.randoms import RandomScalar
from pyglove.ext.scalars.randoms import Uniform
from pyglove.ext.scalars.randoms import Triangular
from pyglove.ext.scalars.randoms import Gaussian
from pyglove.ext.scalars.randoms import Normal
from pyglove.ext.scalars.randoms import LogNormal

# Step-wise scalar.
from pyglove.ext.scalars.step_wise import StepWise

# pylint: enable=g-bad-import-order
