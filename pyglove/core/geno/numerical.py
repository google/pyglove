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
"""Genotype for numerical decisions."""

import random
import types
from typing import Any, List, Optional, Union

from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing

from pyglove.core.geno.base import DecisionPoint
from pyglove.core.geno.base import DNA


def float_scale_spec(field_name):
  """Returns value spec for the scale of a continuous range."""
  return (field_name, pg_typing.Enum(None, [None, 'linear', 'log', 'rlog']),
          ('The scale of values within the range for the search algorithm '
           'to explore. '
           'If None, the feasible space is unscaled;'
           'If `linear`, the feasible space is mapped to [0, 1] linearly.'
           'If `log`, the feasible space is mapped to [0, 1] logarithmically '
           'with formula: x -> log(x / min) / log(max / min). '
           'if `rlog`, the feasible space is mapped to [0, 1] "reverse" '
           'logarithmically, resulting in values close to `max_value` spread '
           'out more than the points near the `min_value`, with formula: '
           'x -> 1.0 - log((max + min - x) / min) / log (max / min). '
           '`min_value` must be positive if `scale` is not None. '
           'Also, it depends on the search algorithm to decide whether this '
           'information is used.'))


@symbolic.members(
    [
        ('min_value', pg_typing.Float(), 'Minimum value.'),
        ('max_value', pg_typing.Float(), 'Maximum value.'),
        float_scale_spec('scale')
    ],
    init_arg_list=[
        'min_value', 'max_value', 'scale', 'hints', 'location', 'name'],
    # TODO(daiyip): For backward compatibility.
    # Move this to additional keys later.
    serialization_key='pyglove.generators.geno.Float',
    additional_keys=['geno.Float'])
class Float(DecisionPoint):
  """Represents the genotype for a float-value genome.

  Example::

    # Create a float decision point within range [0.1, 1.0].
    decision_point = pg.geno.floatv(0.1, 1.0)

  See also: :func:`pyglove.geno.floatv`.
  """

  def _on_bound(self):
    """Custom logics to validate value."""
    super()._on_bound()
    if self.min_value > self.max_value:
      raise ValueError(
          f'Argument \'min_value\' ({self.min_value}) should be no greater '
          f'than \'max_value\' ({self.max_value}).')
    if self.scale in ['log', 'rlog'] and self.min_value <= 0:
      raise ValueError(
          f'\'min_value\' must be positive when `scale` is {self.scale!r}. '
          f'encountered: {self.min_value}.')

  @property
  def is_categorical(self) -> bool:
    """Returns True if current node is a categorical choice."""
    return False

  @property
  def is_subchoice(self) -> bool:
    """Returns True if current node is a subchoice of a multi-choice."""
    return False

  @property
  def is_numerical(self) -> bool:
    """Returns True if current node is numerical decision."""
    return True

  @property
  def is_custom_decision_point(self) -> bool:
    """Returns True if current node is a custom decision point."""
    return False

  @property
  def decision_points(self) -> List[DecisionPoint]:
    """Returns all decision points in their declaration order."""
    return [self]

  @property
  def space_size(self) -> int:
    """Returns the size of the search space. Use -1 for infinity."""
    return -1

  def _next_dna(self, dna: Optional['DNA'] = None) -> Optional['DNA']:
    """Returns the next DNA in the space represented by this spec.

    Args:
      dna: The DNA whose next will be returned. If None, `next_dna` will return
        the first DNA.

    Returns:
      The next DNA or None if there is no next DNA.
    """
    if dna is None:
      return DNA(self.min_value)
    # TODO(daiyip): Use hint for implementing stateful `next_dna`.
    raise NotImplementedError('`next_dna` is not supported on `Float` yet.')

  def _random_dna(
      self,
      random_generator: Union[random.Random, types.ModuleType],
      previous_dna: Optional[DNA]) -> DNA:
    """Returns a random DNA based on current spec."""
    del previous_dna
    return DNA(value=random_generator.uniform(self.min_value, self.max_value))

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    return 1

  def validate(self, dna: DNA) -> None:
    """Validate whether a DNA value conforms to this spec."""
    if not isinstance(dna.value, float):
      raise ValueError(
          f'Expect float value. Encountered: {dna.value}, '
          f'Location: {self.location.path}.')
    if dna.value < self.min_value:
      raise ValueError(
          f'DNA value should be no less than {self.min_value}. '
          f'Encountered {dna.value}, Location: {self.location.path}.')
    if dna.value > self.max_value:
      raise ValueError(
          f'DNA value should be no greater than {self.max_value}. '
          f'Encountered {dna.value}, Location: {self.location.path}.')
    if dna.children:
      raise ValueError(
          f'Float DNA should have no children. '
          f'Encountered: {dna.children!r}, Location: {self.location.path}.')

  def format(self,
             compact: bool = True,
             verbose: bool = True,
             root_indent: int = 0,
             show_id: bool = True,
             **kwargs):
    """Format this object."""
    if not compact:
      return super().format(compact, verbose, root_indent, **kwargs)
    if show_id:
      kvlist = [('id', object_utils.quote_if_str(str(self.id)), '\'\'')]
    else:
      kvlist = []
    details = object_utils.kvlist_str(kvlist + [
        ('name', object_utils.quote_if_str(self.name), None),
        ('min_value', self.min_value, None),
        ('max_value', self.max_value, None),
        ('scale', self.scale, None),
        ('hints', object_utils.quote_if_str(self.hints), None),
    ])
    return f'{self.__class__.__name__}({details})'


def floatv(min_value: float,
           max_value: float,
           scale: Optional[str] = None,
           hints: Any = None,
           location: object_utils.KeyPath = object_utils.KeyPath(),
           name: Optional[str] = None) -> Float:
  """Returns a Float specification.

  It creates the genotype for :func:`pyglove.floatv`.

  Example::

    spec = pg.geno.floatv(0.0, 1.0)

  Args:
    min_value: The lower bound of decision.
    max_value: The upper bound of decision.
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
    hints: An optional hint object.
    location: A ``pg.KeyPath`` object that indicates the location of the
      decision point.
    name: An optional global unique name for identifying this decision
      point.

  Returns:
    A ``pg.geno.Float`` object.

  See also:

    * :func:`pyglove.geno.constant`
    * :func:`pyglove.geno.space`
    * :func:`pyglove.geno.oneof`
    * :func:`pyglove.geno.manyof`
    * :func:`pyglove.geno.custom`
  """
  return Float(min_value, max_value, scale,
               hints=hints, location=location, name=name)

