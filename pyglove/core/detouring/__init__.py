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
"""Symbolic detour.

It is straighforward to symbolize existing classes and functions, but in order
to use them, we need to replace the classes that were used in existing code
with the symbolic ones. Sometimes, we just cannot modify existing source code.
Or in some other cases, objects created within a function or a class method are
not exposed to the external, therefore we cannot manipulate them as a part of
the symbolic tree. For example::

  @pg.symbolize
  def foo():
    # Object `a` is not a part of `foo`'s interface,
    # therefore it cannot be seen from the symbolic tree
    # that contains a `foo` object.
    a = A(1)
    return a.do_something()

Symbolic detour is introduced to address these use cases, which redirects
the ``__new__`` method of a class to another class or function when it’s
evaluated under a context manager. Symbolic detour is not dependent on
symbolization, so in theory it can be used for detouring any classes.
Therefore, it does not require the presence of symbolic objects for mutating
the program.
"""

# pylint: disable=g-bad-import-order

from pyglove.core.detouring.class_detour import detour
from pyglove.core.detouring.class_detour import current_mappings
from pyglove.core.detouring.class_detour import undetoured_new

# pylint: enable=g-bad-import-order
