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
"""Genotype for a space of decisions."""


import random
import types
from typing import List, Optional, Union

from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing

from pyglove.core.geno.base import DecisionPoint
from pyglove.core.geno.base import DNA
from pyglove.core.geno.base import DNASpec


@symbolic.members(
    [
        ('elements', pg_typing.List(
            pg_typing.Object(DecisionPoint), default=[]),
         'Elements of current composition.'),
        ('index', pg_typing.Int(min_value=0).noneable(),
         ('Index of the template among the candidates of a parent Choice. '
          'If None, the template is the root template.'))
    ],
    init_arg_list=['elements', 'index'],
    # TODO(daiyip): For backward compatibility.
    # Move this to additional keys later.
    serialization_key='pyglove.generators.geno.Template',
    additional_keys=['geno.Space'])
class Space(DNASpec):
  """Represents a search space that consists of a list of decision points.

  Example::

    # Create a constant space.
    space = pg.geno.Space([])

    # Create a space with one categorical decision point
    # and a float decision point
    space = pg.geno.space([
        pg.geno.oneof([
            pg.geno.constant(),
            pg.geno.constant(),
       ]),
       pg.geno.floatv(0.0, 1.0)
    ])

  See also: :func:`pyglove.geno.space`, :func:`pyglove.geno.constant`.
  """

  def _on_bound(self):
    super()._on_bound()

    # Fields that will be lazily computed.
    self._space_size = None
    self._decision_points = []
    for elem in self.elements:
      self._decision_points.extend(elem.decision_points)

    # Validate the space upon creation.
    self._validate_space()

  @property
  def is_space(self) -> bool:
    """Returns True if current node is a sub-space."""
    return True

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
    return False

  def _validate_space(self) -> None:
    """Validate the search space."""
    names = {}

    # Ensure the names of decision points do not clash.
    def _ensure_unique_names(spec: DNASpec):
      if isinstance(spec, Space):
        for child in spec.elements:
          _ensure_unique_names(child)
      else:
        if spec.name is not None:
          if spec.name in names:
            raise ValueError(
                f'Found 2 decision point definitions clash on name '
                f'{spec.name!r}. Definition1={spec!r}, '
                f'Definition2={names[spec.name]!r}.')
          names[spec.name] = spec
        if spec.is_categorical:
          for child in spec.candidates:
            _ensure_unique_names(child)

    _ensure_unique_names(self)

    # Ensure the names and ids of decision points do not clash.
    for dp in self.decision_points:
      spec = names.get(dp.id)
      if spec is not None and spec is not dp:
        raise ValueError(
            f'Found 2 decision point definitions clash between name {dp.id!r} '
            f'and id {dp.id!r}. Definition1={spec!r}, Definition2={dp!r}.')

  @property
  def is_constant(self) -> bool:
    """Returns whether this template is constant.

    A constant Space does not have any genetic encoders.
    """
    return not self.elements

  def validate(self, dna: DNA) -> None:
    """Validate whether a DNA value conforms to this spec."""
    if not self.elements and (dna.value is not None or dna.children):
      raise ValueError(
          f'Extra DNA values encountered: {dna!r}, '
          f'Location: {self.location.path}.')

    if len(self.elements) == 1:
      self.elements[0].validate(dna)
    else:
      if len(dna.children) != len(self.elements):
        raise ValueError(
            f'Number of child values in DNA ({len(dna.children)}) does not '
            f'match the number of elements ({len(self.elements)}). Child '
            f'values: {dna.children!r}, Location: {self.location.path}.')
      for i, elem in enumerate(self.elements):
        elem.validate(dna[i])

  @property
  def decision_points(self) -> List['DecisionPoint']:
    """Returns all decision points in their declaration order."""
    return self._decision_points

  @property
  def space_size(self) -> int:
    """Returns the size of the search space. Use -1 for infinity."""
    if self._space_size is None:
      self._space_size = 1
      for e in self.elements:
        sub_space_size = e.space_size
        if sub_space_size == -1:
          self._space_size = -1
          break
        self._space_size *= sub_space_size
    return self._space_size

  def _next_dna(self, dna: Optional['DNA'] = None) -> Optional['DNA']:
    """Returns the next DNA."""
    if dna is None:
      return DNA(None, [e.first_dna(attach_spec=False) for e in self.elements])

    new_children = []
    increment_next_element = True

    def element_dna_at(i):
      if len(self.elements) == 1:
        return dna
      return dna.children[i]

    for i in reversed(range(len(self.elements))):
      child_dna = element_dna_at(i)
      child_spec = self.elements[i]
      # Keep incrementing the next element in the right-to-left direction
      # until an element can be incremented.
      if increment_next_element:
        child_dna = child_spec.next_dna(child_dna, attach_spec=False)
        if child_dna is None:
          child_dna = child_spec.first_dna(attach_spec=False)
        else:
          increment_next_element = False
      new_children.append(child_dna)

    if increment_next_element:
      return None
    else:
      return DNA(None, list(reversed(new_children)))

  def _random_dna(
      self,
      random_generator: Union[random.Random, types.ModuleType],
      previous_dna: Optional[DNA]) -> DNA:
    """Returns a random DNA based on current spec."""
    if previous_dna is None:
      child_dnas = [None] * len(self.elements)
    else:
      # Previous DNA could be in three forms.
      # DNA(None, dna_for_elements)
      # dna_for_element_1 (where elements == 1)
      if len(self.elements) == 1:
        child_dnas = [previous_dna]
      else:
        assert previous_dna.value is None, previous_dna
        child_dnas = previous_dna.children
      assert len(child_dnas) == len(self.elements), (self, child_dnas)
    return DNA(value=None, children=[
        self.elements[i].random_dna(random_generator, False, child_dnas[i])
        for i in range(len(self.elements))])

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    return sum([len(elem) for elem in self.elements])

  def __getitem__(
      self, index: Union[int, slice, str, object_utils.KeyPath]
      ) -> Union[DecisionPoint, List[DecisionPoint]]:
    """Operator [] to return element by index or sub-DNASpec by name."""
    if isinstance(index, (int, slice)):
      return self.elements[index]
    return super().__getitem__(index)  # pytype:disable=unsupported-operands

  def __iter__(self):
    """Operator iter."""
    return self.elements.__iter__()

  def format(self,
             compact: bool = True,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs):
    """Format this object."""
    if not compact:
      return super().format(compact, verbose, root_indent, **kwargs)

    if not self.elements:
      return 'Space()'

    def _indent(text, indent):
      return ' ' * 2 * indent + text

    s = ['Space({\n']
    for i, elem in enumerate(self.elements):
      elem_str = elem.format(
          compact, verbose, root_indent + 1, show_id=False, **kwargs)
      s.append(_indent(
          f'{i} = \'{elem.location.path}\': {elem_str}\n', root_indent + 1))
    s.append(_indent('}', root_indent))
    s.append(')')
    return ''.join(s)


# Alias for backward compatibility.
Template = Space


def constant() -> Space:
  """Returns an constant candidate of Choices.

  Example::

    spec = pg.geno.constant()

  Returns:
    a constant ``pg.geno.Space`` object.

  See also:

    * :func:`pyglove.geno.space`
    * :func:`pyglove.geno.oneof`
    * :func:`pyglove.geno.manyof`
    * :func:`pyglove.geno.floatv`
    * :func:`pyglove.geno.custom`
  """
  return Space()


# Alias for Space class.
space = Space   # pylint: disable=invalid-name
