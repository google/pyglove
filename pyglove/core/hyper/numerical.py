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
"""Numerical hyper primitives."""

import typing
from typing import Any, Callable, Optional, Tuple

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper import base


@symbolic.members(
    [
        ('min_value', pg_typing.Float(), 'Minimum acceptable value.'),
        ('max_value', pg_typing.Float(), 'Maximum acceptable value.'),
        geno.float_scale_spec('scale'),
    ],
    init_arg_list=['min_value', 'max_value', 'scale', 'name', 'hints'],
    serialization_key='hyper.Float',
    additional_keys=['pyglove.generators.genetic.Float']
)
class Float(base.HyperPrimitive):
  """A continuous value within a range.

  Example::

    # A float value between between 0.0 and 1.0.
    v = pg.floatv(0.0, 1.0)

  See also:

    * :func:`pyglove.floatv`
    * :class:`pyglove.hyper.Choices`
    * :class:`pyglove.hyper.OneOf`
    * :class:`pyglove.hyper.ManyOf`
    * :class:`pyglove.hyper.CustomHyper`
  """

  def _on_bound(self):
    """Constructor."""
    super()._on_bound()
    if self.min_value > self.max_value:
      raise ValueError(
          f'\'min_value\' ({self.min_value}) is greater than \'max_value\' '
          f'({self.max_value}).')
    if self.scale in ['log', 'rlog'] and self.min_value <= 0:
      raise ValueError(
          f'\'min_value\' must be positive when `scale` is {self.scale!r}. '
          f'encountered: {self.min_value}.')

  def dna_spec(self,
               location: Optional[object_utils.KeyPath] = None) -> geno.Float:
    """Returns corresponding DNASpec."""
    return geno.Float(
        min_value=self.min_value,
        max_value=self.max_value,
        scale=self.scale,
        hints=self.hints,
        name=self.name,
        location=location or object_utils.KeyPath())

  def _decode(self) -> float:
    """Decode a DNA into a float value."""
    dna = self._dna
    if not isinstance(dna.value, float):
      raise ValueError(
          object_utils.message_on_path(
              f'Expect float value. Encountered: {dna.value}.', self.sym_path))
    if dna.value < self.min_value:
      raise ValueError(
          object_utils.message_on_path(
              f'DNA value should be no less than {self.min_value}. '
              f'Encountered {dna.value}.', self.sym_path))

    if dna.value > self.max_value:
      raise ValueError(
          object_utils.message_on_path(
              f'DNA value should be no greater than {self.max_value}. '
              f'Encountered {dna.value}.', self.sym_path))
    return dna.value

  def encode(self, value: float) -> geno.DNA:
    """Encode a float value into a DNA."""
    if not isinstance(value, float):
      raise ValueError(
          object_utils.message_on_path(
              f'Value should be float to be encoded for {self!r}. '
              f'Encountered {value}.', self.sym_path))
    if value < self.min_value:
      raise ValueError(
          object_utils.message_on_path(
              f'Value should be no less than {self.min_value}. '
              f'Encountered {value}.', self.sym_path))
    if value > self.max_value:
      raise ValueError(
          object_utils.message_on_path(
              f'Value should be no greater than {self.max_value}. '
              f'Encountered {value}.', self.sym_path))
    return geno.DNA(value)

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool = False,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
      ) -> Tuple[bool, 'Float']:
    """Validate candidates during value_spec binding time."""
    del allow_partial
    del child_transform
    # Check if value_spec directly accepts `self`.
    if value_spec.value_type and isinstance(self, value_spec.value_type):
      return (False, self)

    float_spec = typing.cast(
        pg_typing.Float, pg_typing.ensure_value_spec(
            value_spec, pg_typing.Float(), path))
    if float_spec:
      if (float_spec.min_value is not None
          and self.min_value < float_spec.min_value):
        raise ValueError(
            object_utils.message_on_path(
                f'Float.min_value ({self.min_value}) should be no less than '
                f'the min value ({float_spec.min_value}) of value spec: '
                f'{float_spec}.', path))
      if (float_spec.max_value is not None
          and self.max_value > float_spec.max_value):
        raise ValueError(
            object_utils.message_on_path(
                f'Float.max_value ({self.max_value}) should be no greater than '
                f'the max value ({float_spec.max_value}) of value spec: '
                f'{float_spec}.', path))
    return (False, self)

  def is_leaf(self) -> bool:
    """Returns whether this is a leaf node."""
    return True


def floatv(min_value: float,
           max_value: float,
           scale: Optional[str] = None,
           *,
           name: Optional[str] = None,
           hints: Optional[Any] = None) -> Any:
  """A continuous value within a range.

  Example::

    # A continuous value within [0.0, 1.0]
    v = pg.floatv(0.0, 1.0)

  See also:

    * :class:`pyglove.hyper.Float`
    * :func:`pyglove.oneof`
    * :func:`pyglove.manyof`
    * :func:`pyglove.permutate`
    * :func:`pyglove.evolve`

  .. note::

    Under symbolic mode (by default), `pg.floatv` returns a ``pg.hyper.Float``
    object. Under dynamic evaluate mode, which is called under the context of
    :meth:`pyglove.hyper.DynamicEvaluationContext.collect` or
    :meth:`pyglove.hyper.DynamicEvaluationContext.apply`, it evaluates to
    a concrete candidate value.

  Args:
    min_value: Minimum acceptable value (inclusive).
    max_value: Maximum acceptable value (inclusive).
    scale: An optional string as the scale of the range. Supported values
      are None, 'linear', 'log', and 'rlog'.
      If None, the feasible space is unscaled.
      If `linear`, the feasible space is mapped to [0, 1] linearly.
      If `log`, the feasible space is mapped to [0, 1] logarithmically with
        formula `x -> log(x / min) / log(max / min)`.
      If `rlog`, the feasible space is mapped to [0, 1] "reverse"
        logarithmically, resulting in values close to `max_value` spread
        out more than the points near the `min_value`, with formula:
        x -> 1.0 - log((max + min - x) / min) / log (max / min).
      `min_value` must be positive if `scale` is not None.
      Also, it depends on the search algorithm to decide whether this
      information is used or not.
    name: A name that can be used to identify a decision point in the search
      space. This is needed when the code to instantiate the same hyper
      primitive may be called multiple times under a
      `pg.DynamicEvaluationContext.collect` context or a
      `pg.DynamicEvaluationContext.apply` context.
    hints: An optional value which acts as a hint for the controller.

  Returns:
    In symbolic mode, this function returns a `Float`.
    In dynamic evaluate mode, this function returns a float value that is no
    less than the `min_value` and no greater than the `max_value`.
    If evaluated under an `pg.DynamicEvaluationContext.apply` scope,
    this function will return a chosen float value from the controller
    decisions.
    If evaluated under a `pg.DynamicEvaluationContext.collect`
    scope, it will return `min_value`.
  """
  return Float(
      min_value=min_value, max_value=max_value,
      scale=scale, name=name, hints=hints)
