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
"""Genotype for user-defined decisions."""

import random
import types
from typing import Any, Callable, List, Optional, Union

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core import utils
from pyglove.core.geno.base import DecisionPoint
from pyglove.core.geno.base import DNA


@symbolic.members(
    [
        ('hyper_type', pg_typing.Str().noneable(),
         'The display type for the decision point.'),
        ('next_dna_fn', pg_typing.Callable(
            [pg_typing.Object(DNA).noneable()],
            returns=pg_typing.Object(DNA).noneable()).noneable(),
         'An optional callable object to get the next DNA for current point.'),
        ('random_dna_fn', pg_typing.Callable(
            [
                pg_typing.Any(),                   # Random module or object.
                pg_typing.Object(DNA).noneable()   # Previous DNA.
            ],
            returns=pg_typing.Object(DNA)).noneable(),
         'An optional callable object to get a random DNA for current point.'),
    ],
    init_arg_list=[
        'hyper_type', 'next_dna_fn', 'random_dna_fn',
        'hints', 'location', 'name'],
    # TODO(daiyip): For backward compatibility.
    # Move this to additional keys later.
    serialization_key='pyglove.generators.geno.CustomDecisionPoint',
    additional_keys=['geno.CustomDecisionPoint'])
class CustomDecisionPoint(DecisionPoint):
  """Represents a user-defined decision point.

  Example::

    decision_point = pg.geno.custom()

  See also: :func:`pyglove.geno.custom`.
  """

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
    return False

  @property
  def is_custom_decision_point(self) -> bool:
    """Returns True if current node is a custom decision point."""
    return True

  @property
  def decision_points(self) -> List[DecisionPoint]:
    """Returns all decision points in their declaration order."""
    return [self]

  @property
  def space_size(self) -> int:
    """Returns the size of the search space. Use -1 for infinity."""
    return -1

  def _next_dna(self, dna: Optional[DNA] = None) -> Optional[DNA]:
    """Returns the next DNA in the space represented by this spec.

    Args:
      dna: The DNA whose next will be returned. If None, `next_dna` will return
        the first DNA.

    Returns:
      The next DNA or None if there is no next DNA.
    """
    if self.next_dna_fn is None:
      cls_name = self.hyper_type or self.__class__.__name__
      raise NotImplementedError(f'`next_dna` is not supported on {cls_name!r}.')
    return self.next_dna_fn(dna)

  def _random_dna(
      self,
      random_generator: Union[types.ModuleType, random.Random],
      previous_dna: Optional[DNA]) -> DNA:
    """Returns a random DNA based on current spec."""
    if self.random_dna_fn is None:
      cls_name = self.hyper_type or self.__class__.__name__
      raise NotImplementedError(
          f'`random_dna` is not supported on {cls_name!r}.')
    return self.random_dna_fn(random_generator, previous_dna)

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    return 1

  def validate(self, dna: DNA) -> None:
    """Validate whether a DNA value conforms to this spec."""
    if not isinstance(dna.value, str):
      raise ValueError(
          f'CustomDecisionPoint expects string type DNA. '
          f'Encountered: {dna!r}, Location: {self.location.path}.')

  def sym_jsonify(self, **kwargs: Any) -> utils.JSONValueType:
    """Overrides sym_jsonify to exclude non-serializable fields."""
    exclude_keys = kwargs.pop('exclude_keys', [])
    exclude_keys.extend(['random_dna_fn', 'next_dna_fn'])
    return super().sym_jsonify(exclude_keys=exclude_keys, **kwargs)

  def sym_eq(self, other: Any) -> bool:
    """Overrides sym_eq to exclude non-serializable fields."""
    if not isinstance(other, CustomDecisionPoint):
      return False
    return (self.hyper_type == other.hyper_type
            and self.name == other.name
            and self.location == other.location
            and self.hints == other.hints)

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
      kvlist = [('id', str(self.id), '\'\'')]
    else:
      kvlist = []
    details = utils.kvlist_str(
        kvlist
        + [
            ('hyper_type', self.hyper_type, None),
            ('name', self.name, None),
            ('hints', self.hints, None),
        ]
    )
    return f'{self.__class__.__name__}({details})'


def custom(
    hyper_type: Optional[str] = None,
    next_dna_fn: Optional[Callable[[Optional[DNA]], Optional[DNA]]] = None,
    random_dna_fn: Optional[Callable[[Any], DNA]] = None,
    hints: Any = None,
    location: utils.KeyPath = utils.KeyPath(),
    name: Optional[str] = None,
) -> CustomDecisionPoint:
  """Returns a custom decision point.

  It creates the genotype for subclasses of :func:`pyglove.hyper.CustomHyper`.

  Example::

    spec = pg.geno.custom('my_hyper', hints='some hints')

  Args:
    hyper_type: An optional display type for the custom decision point.
    next_dna_fn: An optional callable for computing the next DNA for current
      decision point.
    random_dna_fn: An optional callable for computing a random DNA for current
      decision point.
    hints: An optional hint object.
    location: A ``pg.KeyPath`` object that indicates the location of the
      decision point.
    name: An optional global unique name for identifying this decision
      point.

  Returns:
    A ``pg.geno.CustomDecisionPoint`` object.

  See also:

    * :func:`pyglove.geno.constant`
    * :func:`pyglove.geno.space`
    * :func:`pyglove.geno.oneof`
    * :func:`pyglove.geno.manyof`
    * :func:`pyglove.geno.floatv`
  """
  return CustomDecisionPoint(
      hyper_type, next_dna_fn=next_dna_fn, random_dna_fn=random_dna_fn,
      hints=hints, location=location, name=name)
