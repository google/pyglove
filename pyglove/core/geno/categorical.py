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
"""Genotype for categorical decisions."""

import random
import re
import types
from typing import Any, List, Optional, Union

from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core import utils
from pyglove.core.geno.base import DecisionPoint
from pyglove.core.geno.base import DNA
from pyglove.core.geno.base import DNASpec
from pyglove.core.geno.space import Space


_CHOICE_REGEX = re.compile(r'^(\d+)/(\d+)$')
_CHOICE_AND_LITERAL_REGEX = re.compile(r'^(\d+)/(\d+) \((.*)\)$', re.S)


@symbolic.members(
    [
        ('num_choices', pg_typing.Int(min_value=1, default=1),
         'Number of choices to make.'),
        ('candidates', pg_typing.List(pg_typing.Object(Space), min_size=1),
         ('A list of Spaces for every candidate, which may contain '
          'child nodes to form sub (conditional) search spaces.')),
        ('literal_values', pg_typing.List(
            pg_typing.Union([
                pg_typing.Str(), pg_typing.Int(), pg_typing.Float()]),
            min_size=1).noneable(),
         'Optional literal value for each candidate. Used for debugging.'),
        ('distinct', pg_typing.Bool(True), 'Whether choices are distinct.'),
        ('sorted', pg_typing.Bool(False),
         ('Whether choices are sorted. The order key is the index of the '
          'candidate in the `candidates` field instead of the candidate\'s '
          'value.')),
        ('subchoice_index', pg_typing.Int(min_value=0).noneable(),
         ('Index of current choice as a subchoice of a parent multi-choice. '
          'If None, current choice is either a single choice or a root spec '
          'for a multi-choice.'))
    ],
    init_arg_list=[
        'num_choices', 'candidates', 'literal_values',
        'distinct', 'sorted', 'subchoice_index',
        'hints', 'location', 'name'],
    # TODO(daiyip): For backward compatibility.
    # Move this to additional keys later.
    serialization_key='pyglove.generators.geno.Choices',
    additional_keys=['geno.Choices']
)
class Choices(DecisionPoint):
  """Represents a single or multiple choices from a list of candidates.

  Example::

    # Create a single choice with a nested subspace for candidate 2 (0-based).
    pg.geno.oneof([
        pg.geno.constant(),
        pg.geno.constant(),
        pg.geno.space([
            pg.geno.floatv(0.1, 1.0)
            pg.geno.manyof(2, [
                pg.geno.constant(),
                pg.geno.constant(),
                pg.geno.constant()
            ])
        ])
    ])

  See also: :func:`pyglove.geno.oneof`, :func:`pyglove.geno.manyof`.
  """

  def _on_bound(self):
    super()._on_bound()
    if self.distinct and self.num_choices > len(self.candidates):
      raise ValueError(
          f'There are not enough candidates ({len(self.candidates)}) to make '
          f'{self.num_choices} distinct choices.')
    if self.num_choices > 1 and self.subchoice_index is not None:
      raise ValueError(
          f'Multi-choice spec cannot be a subchoice. '
          f'Encountered: {self!r}.')
    if (self.literal_values is not None and
        len(self.literal_values) != len(self.candidates)):
      raise ValueError(
          f'The length of \'candidates\' ({len(self.candidates)}) '
          f'should be equal to the length of \'literal_values\' '
          f'({len(self.literal_values)}).')

    if self.literal_values is not None:
      # NOTE(daiyip): old PyGlove client will generate literal value in
      # format of 'choice_and_literal'. For backward compatibility, we will
      # convert the old format to the new format. We can deprecate this logic
      # in future.
      literal_values = []
      updated = False
      for i, literal in enumerate(self.literal_values):
        if isinstance(literal, str):
          r = _CHOICE_AND_LITERAL_REGEX.match(literal)
          if (r
              and int(r.group(1)) == i
              and int(r.group(2)) == len(self.candidates)):
            literal = r.group(3)
            updated = True
        literal_values.append(literal)
      if updated:
        self.rebind(literal_values=literal_values, skip_notification=True)

      # Store the literal to index mapping for quick candidate index lookup.
      self._literal_index = {v: i for i, v in enumerate(self.literal_values)}
    else:
      self._literal_index = {}

    self._space_size = None

    # Automatically set the candidate index for template.
    for i, c in enumerate(self.candidates):
      c.rebind(index=i, skip_notification=True, raise_on_no_change=False)

    # Create sub choice specs and index decision points.
    if self.num_choices > 1 and not self.is_subchoice:
      subchoice_specs = []
      self._decision_points = []
      for i in range(self.num_choices):
        subchoice_spec = Choices(
            subchoice_index=i,
            location=utils.KeyPath(i),
            num_choices=1,
            candidates=self.candidates,
            literal_values=self.literal_values,
            distinct=self.distinct,
            sorted=self.sorted,
            name=self.name,
            hints=self.hints,
        )
        self._decision_points.extend(subchoice_spec.decision_points)
        subchoice_specs.append(subchoice_spec)
      self._subchoice_specs = symbolic.List(subchoice_specs)
      self._subchoice_specs.sym_setparent(self)
    else:
      self._subchoice_specs = None
      self._decision_points = [self]
      for c in self.candidates:
        self._decision_points.extend(c.decision_points)

  def _update_children_paths(
      self, old_path: utils.KeyPath, new_path: utils.KeyPath
  ):
    """Trigger path change for subchoices so their IDs can be invalidated."""
    super()._update_children_paths(old_path, new_path)
    if self._subchoice_specs:
      for i, spec in enumerate(self._subchoice_specs):
        spec.sym_setpath(new_path + i)

  def subchoice(self, index: int) -> 'Choices':
    """Returns spec for choice i."""
    if self.is_subchoice:
      raise ValueError(
          f'\'subchoice\' should not be called on a subchoice of a '
          f'multi-choice. Encountered: {self!r}.')
    if self.num_choices == 1:
      return self
    return self._subchoice_specs[index]

  @property
  def choice_specs(self) -> List['Choices']:
    """Returns all choice specs."""
    return [self.subchoice(i) for i in range(self.num_choices)]

  @property
  def is_categorical(self) -> bool:
    """Returns True if current node is a categorical choice."""
    return True

  @property
  def is_subchoice(self) -> bool:
    """Returns if current choice is a subchoice of a multi-choice."""
    return self.subchoice_index is not None

  @property
  def is_numerical(self) -> bool:
    """Returns True if current node is numerical decision."""
    return False

  @property
  def is_custom_decision_point(self) -> bool:
    """Returns True if current node is a custom decision point."""
    return False

  def format_candidate(
      self,
      index: int,
      display_format: str = 'choice_and_literal') -> Union[str, int, float]:
    """Get a formatted candidate value by index.

    Args:
      index: The index of the candidate to format.
      display_format: One of 'choice', 'literal' and 'choice_and_literal' as
        the output format for human consumption.

    Returns:
      A int, float or string that represent the candidate based on the
        display format.
    """
    if display_format not in ['choice', 'literal', 'choice_and_literal']:
      raise ValueError(
          f'`display_format` must be either \'choice\', \'literal\', or '
          f'\'choice_and_literal\'. Encountered: {display_format!r}.')
    if self.literal_values:
      if display_format == 'literal':
        return self.literal_values[index]
      elif display_format == 'choice_and_literal':
        return f'{index}/{len(self.candidates)} ({self.literal_values[index]})'
    return f'{index}/{len(self.candidates)}'

  def candidate_index(self, choice_value: Union[str, int, float]) -> int:
    """Returns the candidate index of a choice value.

    Args:
      choice_value: Choice value can be:
        a (integer, float or string) as a candidate's literal value,
        or a text in the format "<index>/<num_candidates>"
        or a text in the format "<index>/<num_candidates> (<literal>)".

    Returns:
      The index of chosen candidate.

    Raises:
      ValueError: `choice_value` is not a valid index or it does not matches
      any candidate's literal value.
    """
    def _index_from_literal(value: Union[str, int, float]) -> int:
      index = self._literal_index.get(value)
      if index is None:
        raise ValueError(
            f'There is no candidate in {self!r} with literal value {value!r}.')
      return index

    index = None
    literal = None
    num_candidates = len(self.candidates)
    if isinstance(choice_value, str):
      r = _CHOICE_AND_LITERAL_REGEX.match(choice_value)
      if r:
        index = int(r.group(1))
        num_candidates = int(r.group(2))
        literal = r.group(3)
      else:
        r = _CHOICE_REGEX.match(choice_value)
        if r:
          index = int(r.group(1))
          num_candidates = int(r.group(2))
        else:
          index = _index_from_literal(choice_value)
    elif isinstance(choice_value, (int, float)):
      index = _index_from_literal(choice_value)
    else:
      raise ValueError(
          f'The value for Choice \'{self.id}\' should be either an integer, '
          f'a float or a string. Encountered: {choice_value!r}.')
    if index < -len(self.candidates) or index >= len(self.candidates):
      raise ValueError(
          f'Candidate index out of range at choice \'{self.name or self.id}\'. '
          f'Index={index}, Number of candidates={len(self.candidates)}.')
    if num_candidates != len(self.candidates):
      raise ValueError(
          f'Number of candidates ({num_candidates}) for Choice '
          f'\'{self.name or self.id}\' does not match with '
          f'DNASpec ({len(self.candidates)}).')
    if (literal is not None
        and (not self.literal_values
             or literal != str(self.literal_values[index]))):
      raise ValueError(
          f'The literal value from the input ({literal!r}) does not match '
          f'with the literal value ({self.literal_values[index]!r}) from the '
          f'chosen candidate (index={index}).')
    return index

  def validate(self, dna: DNA) -> None:
    """Validate whether a DNA value conforms to this spec."""
    if self.num_choices == 1:
      # For single choice, choice is decoded from dna.value.
      # children should be decoded by chosen candidate.
      if dna.value is None and dna.children:
        raise ValueError(
            f'Expect an integer for a single choice, but encountered a list '
            f'({dna.to_numbers(flatten=False)}). '
            f'Location: {self.location.path}.')
      if not isinstance(dna.value, int):
        raise ValueError(
            f'Expect an integer for a single choice, but encountered: '
            f'{dna.value!r}, Location: {self.location.path}.')
      if dna.value >= len(self.candidates):
        raise ValueError(
            f'Choice out of range. Value: {dna.value}, '
            f'Candidates: {len(self.candidates)}, '
            f'Location: {self.location.path}.')
      chosen = self.candidates[dna.value]
      if chosen.is_constant and dna.children:
        raise ValueError(
            f'Child DNA ({dna.children!r}) provided while there is no child '
            f'decision point.')
      elif not chosen.is_constant and not dna.children:
        raise ValueError(
            f'No child DNA provided for child space {chosen!r}.')
      chosen.validate(DNA(None, dna.children))
    else:
      # For multiple choices, choice is encoded in DNA.children.
      # dna.value could be None (Choices as the only encoder in root template)
      # or int (parent choice).
      if len(dna.children) != self.num_choices:
        raise ValueError(
            f'Number of DNA child values does not match the number of choices. '
            f'Child values: {dna.children!r}, Choices: {self.num_choices}, '
            f'Location: {self.location.path}.')
      if self.distinct or self.sorted:
        sub_dna_values = [s.value for s in dna]
        if self.distinct and len(set(sub_dna_values)) != len(dna.children):
          raise ValueError(
              f'DNA child values should be distinct. '
              f'Encountered: {sub_dna_values}, Location: {self.location.path}.')
        if self.sorted and sorted(sub_dna_values) != sub_dna_values:
          raise ValueError(
              f'DNA child values should be sorted. '
              f'Encountered: {sub_dna_values}, Location: {self.location.path}.')
      for i, sub_dna in enumerate(dna):
        sub_location = utils.KeyPath(i, self.location)
        if not isinstance(sub_dna.value, int):
          raise ValueError(
              f'Choice value should be int. Encountered: {sub_dna.value}, '
              f'Location: {sub_location.path}.')
        if sub_dna.value >= len(self.candidates):
          raise ValueError(
              f'Choice out of range. Value: {sub_dna.value}, '
              f'Candidates: {len(self.candidates)}, '
              f'Location: {sub_location.path}.')
        self.candidates[sub_dna.value].validate(DNA(None, sub_dna.children))

  @property
  def decision_points(self) -> List[DecisionPoint]:
    """Returns all decision points in their declaration order.

    Returns:
      All decision points in current space. For multi-choices, the sub-choice
      objects will be returned. Users can call `spec.parent_choice` to access
      the parent multi-choice node.
    """
    return self._decision_points

  @property
  def space_size(self) -> int:
    """Returns the search space size."""
    if self._space_size is None:
      sub_space_sizes = [c.space_size for c in self.candidates]
      if any([v == -1 for v in sub_space_sizes]):
        self._space_size = -1
      else:
        def _space_size(s: List[int], k: int) -> int:
          """Returns space size of k choices from a list of sub-space sizes."""
          if k == 0:
            return 1
          elif k == 1:
            return sum(s)
          elif k > len(s) and self.distinct:
            return 0
          elif len(s) == 1:
            assert not self.distinct
            return s[0] ** k  # pytype: disable=bad-return-type
          elif self.distinct and self.sorted:
            # When choice is distinct and sorted, current chosen item
            # must appear at the front.
            return s[0] * _space_size(s[1:], k - 1) + _space_size(s[1:], k)
          elif self.distinct:
            # When choice is distinct but not sorted, current chosen item
            # could appear at all `k` positions.
            return s[0] * k * _space_size(s[1:], k - 1) + _space_size(s[1:], k)
          elif self.sorted:
            # When choice is sorted but not distinct, the space size is the sum
            # of k + 1 terms, where the i'th term (0 <= i <= k) is the space
            # size of selecting s[0] for i times multiplying the space size of
            # choosing k - i items from s[1:].
            size = 0
            for i in range(k + 1):
              size += (s[0] ** i) * _space_size(s[1:], k - i)
            return size  # pytype: disable=bad-return-type  # always-use-return-annotations
          else:
            # When choice is neither distinct nor sorted,
            return _space_size(s, 1) ** k  # pytype: disable=bad-return-type  # always-use-return-annotations
        self._space_size = _space_size(sub_space_sizes, self.num_choices)
    return self._space_size

  def _next_dna(self, dna: Optional[DNA] = None) -> Optional[DNA]:
    """Returns the next DNA."""
    if dna is None:
      choice_dna_list = []
      for i in range(self.num_choices):
        choice = i if self.distinct else 0
        dna = self.candidates[choice].first_dna(attach_spec=False)
        choice_dna_list.append(DNA(choice, [dna]))
      return DNA(None, choice_dna_list)

    if self.num_choices == 1:
      # NOTE(daiyip): the DNA represenation of Choices should be
      # DNA(value=None, children=[dna_for_choice1, dna_for_choice2, ...]
      # However, `DNA(value=None, children=[choice1])` will be collapsed
      # into `choice1` for a more compact representation when `num_choices`
      # is equal to 1.
      parent_choice_value = None
      choice_dna_list = [dna]
    else:
      if len(dna.children) != self.num_choices:
        raise ValueError(
            f'Expect {self.num_choices} choices but encountered '
            f'{len(dna.children)} sub-DNA ({dna.children!r}).')
      parent_choice_value = dna.value
      choice_dna_list = dna.children

    def next_value_for_choice(
        prior_choices: List[int], current_choice: int) -> Optional[int]:
      """Get the next value conditioned by prior choices and current value."""
      n = len(self.candidates)
      next_value = current_choice + 1
      if self.distinct:
        possible_choices = set(range(next_value, n))
        possible_choices -= set(prior_choices)
        next_value = min(possible_choices) if possible_choices else n
      return next_value if next_value < n else None

    def min_remaining_choices(prior_choices: List[int]) -> Optional[List[int]]:
      """Get the minimal remaining choices conditioned by prior choices.."""
      if self.sorted and prior_choices:
        possible_choices = set(range(prior_choices[-1], len(self.candidates)))
      else:
        possible_choices = set(range(len(self.candidates)))
      if self.distinct:
        possible_choices -= set(prior_choices)

      remaining_choices = []
      for _ in range(self.num_choices - len(prior_choices)):
        if not possible_choices:
          return None
        next_choice = min(possible_choices)
        if self.distinct:
          possible_choices.remove(next_choice)
        remaining_choices.append(next_choice)
      return remaining_choices

    # Now we want to increment the DNA from right to left (last choice first),
    # which means we will increment the sub-DNA if possible, or we will have
    # to increment the choice value with its first DNA. If we can increment
    # neither the sub-DNA nor the choice value, we have reached the end of the
    # space represented by the DNASpec.
    for choice_id in reversed(range(self.num_choices)):
      choice_dna = choice_dna_list[choice_id]

      if (not isinstance(choice_dna.value, int)
          or choice_dna.value < 0 or choice_dna.value >= len(self.candidates)):
        raise ValueError(
            f'Choice value ({choice_dna.value}) is out of range '
            f'([0, {len(self.candidates)}]). DNASpec: {self!r}.')

      subspace_dna = DNA(None, choice_dna.children)
      subspace_next_dna = self.candidates[choice_dna.value].next_dna(
          subspace_dna, attach_spec=False)

      updated_current_choice = False
      prior_choices = [choice_dna_list[i].value for i in range(choice_id)]
      if subspace_next_dna is not None:
        # The sub-DNA is incremented, thus we can keep using the same choice
        # value.
        new_choice_dna = DNA(choice_dna.value, [subspace_next_dna])
        updated_current_choice = True
      else:
        # We need to find the next valid value for current choice.
        new_choice_value = next_value_for_choice(
            prior_choices, choice_dna.value)
        if new_choice_value is not None:
          new_choice_dna = DNA(
              new_choice_value,
              [self.candidates[new_choice_value].first_dna(
                  attach_spec=False)])
          updated_current_choice = True

      if updated_current_choice:
        # We found next sub-DNA at choice i, therefore we can reuse sub-DNA
        # from choice [0...i-1], and find remaining choices for choices
        # [i+1...k].
        remaining_choices = min_remaining_choices(
            prior_choices + [new_choice_dna.value])

        if remaining_choices is not None:
          subdna_list = choice_dna_list[:choice_id] + [new_choice_dna] + [
              DNA(v, [self.candidates[v].first_dna(attach_spec=False)])
              for v in remaining_choices
          ]
          return DNA(parent_choice_value, subdna_list)
    return None

  def _random_dna(
      self,
      random_generator: Union[random.Random, types.ModuleType],
      previous_dna: Optional[DNA]) -> DNA:
    """Returns a random DNA based on current spec."""
    if self.distinct:
      choices = random_generator.sample(
          list(range(len(self.candidates))), self.num_choices)
    else:
      choices = [random_generator.randint(0, len(self.candidates) - 1)
                 for _ in range(self.num_choices)]
    if self.sorted:
      choices = sorted(choices)

    # Figure out previous DNAs.
    if previous_dna is None:
      child_dnas = [None] * len(choices)
    else:
      if self.num_choices == 1:
        choices_dnas = [previous_dna]
      else:
        choices_dnas = list(previous_dna.children)
      assert len(choices_dnas) == self.num_choices, (self, choices_dnas)
      child_dnas = []
      for i, (choice_dna, choice) in enumerate(zip(choices_dnas, choices)):
        if choice_dna.value != choice:
          child_dna = None
        else:
          child_dna = DNA(
              None,
              children=choice_dna.children,
              spec=self.candidates[choice])
        child_dnas.append(child_dna)

    children = []
    for i, c in enumerate(choices):
      children.append(DNA(value=c, children=[
          self.candidates[c].random_dna(
              random_generator, False, child_dnas[i])]))
    return DNA(value=None, children=children)

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    sub_length = sum([len(c) for c in self.candidates])
    return self.num_choices * (1 + sub_length)

  def __getitem__(
      self, index: Union[int, slice, str]
      ) -> Union[DecisionPoint, List[DecisionPoint]]:
    """Operator [] to return the sub choice(s) if index is int."""
    if isinstance(index, (int, slice)):
      if self.num_choices == 1:
        sub_choices = [self]
      else:
        sub_choices = self._subchoice_specs
      return sub_choices[index]
    return super().__getitem__(index)

  def format(self,
             compact: bool = True,
             verbose: bool = True,
             root_indent: int = 0,
             show_id: bool = True,
             **kwargs):
    """Format this object."""
    if not compact:
      return super().format(compact, verbose, root_indent, **kwargs)

    def _indent(text, indent):
      return ' ' * 2 * indent + text

    s = [f'Choices(num_choices={self.num_choices}, [\n']
    if not self.literal_values:
      for i, candidate in enumerate(self.candidates):
        candidate_str = candidate.format(
            compact, verbose, root_indent + 1, **kwargs)
        s.append(_indent(f'({i}): {candidate_str}\n', root_indent + 1))
    else:
      assert len(self.literal_values) == len(self.candidates)
      for i, (candidate, literal_value) in enumerate(
          zip(self.candidates, self.literal_values)):
        if not candidate.is_constant:
          value_str = candidate.format(compact, verbose, root_indent + 1,
                                       **kwargs)
        else:
          value_str = literal_value
        s.append(_indent(f'({i}): {value_str}\n', root_indent + 1))
    s.append(_indent(']', root_indent))
    if show_id:
      kvlist = [('id', str(self.id), '\'\'')]
    else:
      kvlist = []
    additionl_properties = utils.kvlist_str(
        kvlist
        + [
            ('name', self.name, None),
            ('distinct', self.distinct, True),
            ('sorted', self.sorted, False),
            ('hints', self.hints, None),
            ('subchoice_index', self.subchoice_index, None),
        ],
        compact=False,
        root_indent=root_indent,
    )
    if additionl_properties:
      s.append(', ')
      s.append(additionl_properties)
    s.append(')')
    return ''.join(s)


def manyof(
    num_choices: int,
    candidates: List[DNASpec],
    distinct: bool = True,
    sorted: bool = False,  # pylint: disable=redefined-builtin
    literal_values: Optional[List[Union[str, int, float]]] = None,
    hints: Any = None,
    location: Union[str, utils.KeyPath] = utils.KeyPath(),
    name: Optional[str] = None,
) -> Choices:
  """Returns a multi-choice specification.

  It creates the genotype for :func:`pyglove.manyof`.

  Example::

    spec = pg.geno.manyof(2, [
        pg.geno.constant(),
        pg.geno.constant(),
        pg.geno.oneof([
            pg.geno.constant(),
            pg.geno.constant()
        ])
    ])

  Args:
    num_choices: Number of choices.
    candidates: A list of ``pg.geno.Space`` objects as the candidates.
    distinct: If True, the decisions for the multiple choices should be
      distinct from each other (based on the value of selected indices).
    sorted: If Ture, the decisions returned should be sorted by the
      values of selected indices.
    literal_values: An optional list of string, integer, or float as the
      literal values for the candidates for display purpose.
    hints: An optional hint object.
    location: A ``pg.KeyPath`` object that indicates the location of the
      decision point.
    name: An optional global unique name for identifying this decision
      point.

  Returns:
    A ``pg.geno.Choices`` object.

  See also:

    * :func:`pyglove.geno.constant`
    * :func:`pyglove.geno.space`
    * :func:`pyglove.geno.oneof`
    * :func:`pyglove.geno.floatv`
    * :func:`pyglove.geno.custom`
  """
  normalized_candidates = []
  for c in candidates:
    if isinstance(c, DecisionPoint):
      c = Space([c])
    normalized_candidates.append(c)
  return Choices(num_choices, normalized_candidates, distinct=distinct,
                 sorted=sorted, literal_values=literal_values,
                 hints=hints, location=location, name=name)


def oneof(
    candidates: List[DNASpec],
    literal_values: Optional[List[Union[str, int, float]]] = None,
    hints: Any = None,
    location: Union[str, utils.KeyPath] = utils.KeyPath(),
    name: Optional[str] = None,
) -> Choices:
  """Returns a single choice specification.

  It creates the genotype for :func:`pyglove.oneof`.

  Example::

    spec = pg.geno.oneof([
        pg.geno.constant(),
        pg.geno.oneof([
            pg.geno.constant(),
            pg.geno.constant()
        ])
    ])

  Args:
    candidates: A list of ``pg.geno.Space`` objects as the candidates.
    literal_values: An optional list of string, integer, or float as the
      literal values for the candidates for display purpose.
    hints: An optional hint object.
    location: A ``pg.KeyPath`` object that indicates the location of the
      decision point.
    name: An optional global unique name for identifying this decision
      point.

  Returns:
    A ``pg.geno.Choices`` object.

  See also:

    * :func:`pyglove.geno.constant`
    * :func:`pyglove.geno.space`
    * :func:`pyglove.geno.manyof`
    * :func:`pyglove.geno.floatv`
    * :func:`pyglove.geno.custom`
  """
  return manyof(1, candidates, literal_values=literal_values,
                hints=hints, location=location, name=name)
