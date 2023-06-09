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
"""Base of symbolic hyper values for representing client-side search spaces."""


import abc
from typing import Any, Callable, Optional

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing


class HyperValue(symbolic.NonDeterministic):  # pytype: disable=ignored-metaclass
  """Base class for a hyper value.

  Hyper value represents a space of objects, which is essential for
  programmatically generating objects. It can encode a concrete object into a
  DNA, or decode a DNA into a concrete object.

  DNA is a nestable numeric interface we use to generate object (see `geno.py`).
  Each position in the DNA represents either the index of a choice, or a value
  itself is numeric. There could be multiple choices standing side-by-side,
  representing knobs on different parts of an object, or choices being chained,
  forming conditional choice spaces, which can be described by a tree structure.

  Hyper values form a tree as the following:

  .. graphviz::

    digraph relationship {
      template [label="ObjectTemplate" href="object_template.html"];
      primitive [label="HyperPrimitive" href="hyper_primitive.html"];
      choices [label="OneOf/ManyOf" href="choices.html"];
      float [label="Float" href="float_class.html"];
      custom [label="CustomHyper" href="custom_hyper.html"];
      template -> primitive [label="elements (1:*)"];
      primitive -> choices [dir="back" arrowtail="empty" style="dashed"];
      primitive -> float [dir="back" arrowtail="empty" style="dashed"];
      primitive -> custom [dir="back" arrowtail="empty" style="dashed"];
      choices -> template [label="candidates (1:*)"];
    }
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self):
    # DNA and decoded value are states for __call__.
    # Though `decode` and `encode` methods are stateless.
    self._dna = None
    self._decoded_value = None

  def set_dna(self, dna: geno.DNA) -> None:
    """Use this DNA to generate value.

    NOTE(daiyip): self._dna is only used in __call__.
    Thus 'set_dna' can be called multiple times to generate different values.

    Args:
      dna: DNA to use to decode the value.
    """
    self._dna = dna
    # Invalidate decoded value when DNA is refreshed.
    self._decoded_value = None

  @property
  def dna(self) -> Optional[geno.DNA]:
    """Returns the DNA that is being used by this hyper value."""
    return self._dna

  def __call__(self) -> Any:
    """Generate value from DNA provided by set_dna."""
    if self._decoded_value is None:
      if self._dna is None:
        raise ValueError(
            '\'set_dna\' should be called to set a DNA before \'__call__\'.')
      self._decoded_value = self.decode(self._dna)
    return self._decoded_value

  def decode(self, dna: geno.DNA) -> Any:
    """Decode a value from a DNA."""
    self.set_dna(dna)
    return self._decode()

  @abc.abstractmethod
  def _decode(self) -> Any:
    """Decode using self.dna."""

  @abc.abstractmethod
  def encode(self, value: Any) -> geno.DNA:
    """Encode a value into a DNA.

    Args:
      value: A value that conforms to the hyper value definition.

    Returns:
      DNA for the value.
    """

  @abc.abstractmethod
  def dna_spec(self,
               location: Optional[object_utils.KeyPath] = None) -> geno.DNASpec:
    """Get DNA spec of DNA that is decodable/encodable by this hyper value."""


@symbolic.members([
    ('name', pg_typing.Str().noneable(),
     ('Name of the hyper primitive. Useful in define-by-run mode to identify a'
      'decision point in the search space - that is - different instances with '
      'the same name will refer to the same decision point in the search space '
      'under define-by-run mode. '
      'Please refer to `pg.hyper.trace` for details.')),
    ('hints', pg_typing.Any(default=None), 'Generator hints')
])
class HyperPrimitive(symbolic.Object, HyperValue):
  """Base class for hyper primitives.

  A hyper primitive is a pure symbolic object which represents an object
  generation rule. It correspond to a decision point
  (:class:`pyglove.geno.DecisionPoint`) in the algorithm's view.

  Child classes:

    * :class:`pyglove.hyper.Choices`

      * :class:`pyglove.hyper.OneOf`
      * :class:`pyglove.hyper.ManyOf`
    * :class:`pyglove.hyper.Float`
    * :class:`pyglove.hyper.CustomHyper`
  """

  def __new__(cls, *args, **kwargs) -> Any:
    """Overrides __new__ for supporting dynamic evaluation mode.

    Args:
      *args: Positional arguments passed to init the custom hyper.
      **kwargs: Keyword arguments passed to init the custom hyper.

    Returns:
      A dynamic evaluated value according to current `dynamic_evaluate` context.
    """
    dynamic_evaluate_fn = get_dynamic_evaluate_fn()
    if dynamic_evaluate_fn is None:
      return super().__new__(cls)    # pylint: disable=no-value-for-parameter
    else:
      hyper_value = object.__new__(cls)
      cls.__init__(hyper_value, *args, **kwargs)
      return dynamic_evaluate_fn(hyper_value)  # pylint: disable=not-callable

  def _sym_clone(self, deep: bool, memo=None) -> 'HyperPrimitive':
    """Overrides _sym_clone to force no dynamic evaluation."""
    kwargs = dict()
    for k, v in self._sym_attributes.items():
      if deep or isinstance(v, symbolic.Symbolic):
        v = symbolic.clone(v, deep, memo)
      kwargs[k] = v

    # NOTE(daiyip): instead of calling self.__class__(...),
    # we manually create a new instance without invoking dynamic
    # evaluation.
    new_value = object.__new__(self.__class__)
    new_value.__init__(   # pylint: disable=unexpected-keyword-arg
        allow_partial=self._allow_partial, sealed=self._sealed, **kwargs)
    return new_value


_TLS_KEY_DYNAMIC_EVALUATE_FN = 'dynamic_evaluate_fn'
_global_dynamic_evaluate_fn = None


def set_dynamic_evaluate_fn(
    fn: Optional[Callable[[HyperValue], Any]], per_thread: bool) -> None:
  """Set current dynamic evaluate function."""
  global _global_dynamic_evaluate_fn
  if per_thread:
    assert _global_dynamic_evaluate_fn is None, _global_dynamic_evaluate_fn
    object_utils.thread_local_set(_TLS_KEY_DYNAMIC_EVALUATE_FN, fn)
  else:
    _global_dynamic_evaluate_fn = fn


def get_dynamic_evaluate_fn() -> Optional[Callable[[HyperValue], Any]]:
  """Gets current dynamic evaluate function."""
  return object_utils.thread_local_get(
      _TLS_KEY_DYNAMIC_EVALUATE_FN, _global_dynamic_evaluate_fn)
