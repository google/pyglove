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
"""Categorical hyper primitives."""

import numbers
import typing
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper import base
from pyglove.core.hyper import object_template


@symbolic.members([
    ('num_choices', pg_typing.Int(min_value=0).noneable(),
     ('Number of choices to make. If set to None, any number of choices is '
      'acceptable.')),
    ('candidates', pg_typing.List(pg_typing.Any()),
     ('Candidate values, which may contain nested hyper values.'
      'Candidate can customize its display value (literal) by implementing the '
      '`pg.Formattable` interface.')),
    ('choices_distinct', pg_typing.Bool(True), 'Whether choices are distinct.'),
    ('choices_sorted', pg_typing.Bool(False), 'Whether choices are sorted.'),
    ('where', pg_typing.Callable([pg_typing.Object(base.HyperPrimitive)],
                                 returns=pg_typing.Bool()).noneable(),
     ('Callable object to filter nested hyper values. If None, all nested '
      'hyper value will be included in the encoding/decoding process. '
      'Otherwise only the hyper values on which `where` returns True will be '
      'included. `where` can be useful to partition a search space into '
      'separate optimization processes. '
      'Please see `ObjectTemplate` docstr for details.'))
])
class Choices(base.HyperPrimitive):
  """Categorical choices from a list of candidates.

  Example::

    # A single categorical choice:
    v = pg.oneof([1, 2, 3])

    # A multiple categorical choice as a list:
    vs = pg.manyof(2, [1, 2, 3])

    # A hierarchical categorical choice:
    v2 = pg.oneof([
        'foo',
        'bar',
        pg.manyof(2, [1, 2, 3])
    ])

  See also:

    * :class:`pyglove.hyper.OneOf`
    * :class:`pyglove.hyper.ManyOf`
    * :func:`pyglove.oneof`
    * :func:`pyglove.manyof`
    * :func:`pyglove.permutate`
  """

  def _on_bound(self):
    """On members are bound."""
    super()._on_bound()
    if self.num_choices > len(self.candidates) and self.choices_distinct:
      raise ValueError(
          f'{len(self.candidates)} candidates cannot produce '
          f'{self.num_choices} distinct choices.')
    self._candidate_templates = [
        object_template.ObjectTemplate(c, where=self.where)
        for c in self.candidates]
    # ValueSpec for candidate.
    self._value_spec = None

  def _update_children_paths(
      self, old_path: object_utils.KeyPath, new_path: object_utils.KeyPath):
    """Customized logic to update children paths."""
    super()._update_children_paths(old_path, new_path)
    for t in self._candidate_templates:
      t.root_path = self.sym_path

  @property
  def candidate_templates(self):
    """Returns candidate templates."""
    return self._candidate_templates

  @property
  def is_leaf(self) -> bool:
    """Returns whether this is a leaf node."""
    for t in self._candidate_templates:
      if not t.is_constant:
        return False
    return True

  def dna_spec(self,
               location: Optional[object_utils.KeyPath] = None) -> geno.Choices:
    """Returns corresponding DNASpec."""
    return geno.Choices(
        num_choices=self.num_choices,
        candidates=[ct.dna_spec() for ct in self._candidate_templates],
        literal_values=[self._literal_value(c)
                        for i, c in enumerate(self.candidates)],
        distinct=self.choices_distinct,
        sorted=self.choices_sorted,
        hints=self.hints,
        name=self.name,
        location=location or object_utils.KeyPath())

  def _literal_value(
      self, candidate: Any, max_len: int = 120) -> Union[int, float, str]:
    """Returns literal value for candidate."""
    if isinstance(candidate, numbers.Number):
      return candidate

    literal = object_utils.format(candidate, compact=True,
                                  hide_default_values=True,
                                  hide_missing_values=True,
                                  strip_object_id=True)
    if len(literal) > max_len:
      literal = literal[:max_len - 3] + '...'
    return literal

  def _decode(self) -> List[Any]:
    """Decode a DNA into a list of object."""
    dna = self._dna
    if self.num_choices == 1:
      # Single choice.
      if not isinstance(dna.value, int):
        raise ValueError(
            object_utils.message_on_path(
                f'Did you forget to specify values for conditional choices?\n'
                f'Expect integer for {self.__class__.__name__}. '
                f'Encountered: {dna!r}.', self.sym_path))
      if dna.value >= len(self.candidates):
        raise ValueError(
            object_utils.message_on_path(
                f'Choice out of range. Value: {dna.value!r}, '
                f'Candidates: {len(self.candidates)}.', self.sym_path))
      choices = [self._candidate_templates[dna.value].decode(
          geno.DNA(None, dna.children))]
    else:
      # Multi choices.
      if len(dna.children) != self.num_choices:
        raise ValueError(
            object_utils.message_on_path(
                f'Number of DNA child values does not match the number of '
                f'choices. Child values: {dna.children!r}, '
                f'Choices: {self.num_choices}.', self.sym_path))
      if self.choices_distinct or self.choices_sorted:
        sub_dna_values = [s.value for s in dna]
        if (self.choices_distinct
            and len(set(sub_dna_values)) != len(dna.children)):
          raise ValueError(
              object_utils.message_on_path(
                  f'DNA child values should be distinct. '
                  f'Encountered: {sub_dna_values}.', self.sym_path))
        if self.choices_sorted and sorted(sub_dna_values) != sub_dna_values:
          raise ValueError(
              object_utils.message_on_path(
                  f'DNA child values should be sorted. '
                  f'Encountered: {sub_dna_values}.', self.sym_path))
      choices = []
      for i, sub_dna in enumerate(dna):
        if not isinstance(sub_dna.value, int):
          raise ValueError(
              object_utils.message_on_path(
                  f'Choice value should be int. '
                  f'Encountered: {sub_dna.value}.',
                  object_utils.KeyPath(i, self.sym_path)))
        if sub_dna.value >= len(self.candidates):
          raise ValueError(
              object_utils.message_on_path(
                  f'Choice out of range. Value: {sub_dna.value}, '
                  f'Candidates: {len(self.candidates)}.',
                  object_utils.KeyPath(i, self.sym_path)))
        choices.append(self._candidate_templates[sub_dna.value].decode(
            geno.DNA(None, sub_dna.children)))
    return choices

  def encode(self, value: List[Any]) -> geno.DNA:
    """Encode a list of values into DNA.

    Example::

        # DNA of an object containing a single OneOf.
        # {'a': 1} => DNA(0)
        {
           'a': one_of([1, 2])
        }


        # DNA of an object containing multiple OneOfs.
        # {'b': 1, 'c': bar} => DNA([0, 1])
        {
           'b': pg.oneof([1, 2]),
           'c': pg.oneof(['foo', 'bar'])
        }

        # DNA of an object containing conditional space.
        # {'a': {'b': 1} => DNA(0, 0, 0)])
        # {'a': {'b': [4, 7]} => DNA(1, [(0, 1), 2])
        # {'a': {'b': 'bar'} => DNA(2)
        {
           'a': {
              'b': pg.oneof([
                pg.oneof([
                  pg.oneof([1, 2]),
                  pg.oneof(3, 4)]),
                  pg.manyof(2, [
                    pg.oneof([4, 5]),
                    6,
                    7
                  ]),
                ]),
                'bar',
              ])
           }
        }

    Args:
      value: A list of value that can match choice candidates.

    Returns:
      Encoded DNA.

    Raises:
      ValueError if value cannot be encoded.
    """
    if not isinstance(value, list):
      raise ValueError(
          object_utils.message_on_path(
              f'Cannot encode value: value should be a list type. '
              f'Encountered: {value!r}.', self.sym_path))
    choices = []
    if self.num_choices is not None and len(value) != self.num_choices:
      raise ValueError(
          object_utils.message_on_path(
              f'Length of input list is different from the number of choices '
              f'({self.num_choices}). Encountered: {value}.', self.sym_path))
    for v in value:
      choice_id = None
      child_dna = None
      for i, b in enumerate(self._candidate_templates):
        succeeded, child_dna = b.try_encode(v)
        if succeeded:
          choice_id = i
          break
      if child_dna is None:
        raise ValueError(
            object_utils.message_on_path(
                f'Cannot encode value: no candidates matches with '
                f'the value. Value: {v!r}, Candidates: {self.candidates}.',
                self.sym_path))
      choices.append(geno.DNA(choice_id, [child_dna]))
    return geno.DNA(None, choices)


@symbolic.members(
    [],
    init_arg_list=[
        'num_choices', 'candidates', 'choices_distinct',
        'choices_sorted', 'hints'
    ],
    # TODO(daiyip): Change to 'ManyOf' once existing code migrates to ManyOf.
    serialization_key='hyper.ManyOf',
    additional_keys=['pyglove.generators.genetic.ChoiceList']
)
class ManyOf(Choices):
  """N Choose K.

  Example::

    # Chooses 2 distinct candidates.
    v = pg.manyof(2, [1, 2, 3])

    # Chooses 2 non-distinct candidates.
    v = pg.manyof(2, [1, 2, 3], distinct=False)

    # Chooses 2 distinct candidates sorted by their indices.
    v = pg.manyof(2, [1, 2, 3], sorted=True)

    # Permutates the candidates.
    v = pg.permutate([1, 2, 3])

    # A hierarchical categorical choice:
    v2 = pg.manyof(2, [
        'foo',
        'bar',
        pg.oneof([1, 2, 3])
    ])

  See also:

    * :func:`pyglove.manyof`
    * :func:`pyglove.permutate`
    * :class:`pyglove.hyper.Choices`
    * :class:`pyglove.hyper.OneOf`
    * :class:`pyglove.hyper.Float`
    * :class:`pyglove.hyper.CustomHyper`
  """

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
      ) -> Tuple[bool, 'Choices']:
    """Validate candidates during value_spec binding time."""
    # Check if value_spec directly accepts `self`.
    if value_spec.value_type and isinstance(self, value_spec.value_type):
      return (False, self)

    if self._value_spec:
      src_spec = self._value_spec
      dest_spec = value_spec
      if not dest_spec.is_compatible(src_spec):
        raise TypeError(
            object_utils.message_on_path(
                f'Cannot bind an incompatible value spec {dest_spec} '
                f'to {self.__class__.__name__} with bound spec {src_spec}.',
                path))
      return (False, self)

    list_spec = typing.cast(
        pg_typing.List,
        pg_typing.ensure_value_spec(
            value_spec, pg_typing.List(pg_typing.Any()), path))
    if list_spec:
      for i, c in enumerate(self.candidates):
        list_spec.element.value.apply(
            c,
            self._allow_partial,
            root_path=path + f'candidates[{i}]')
    self._value_spec = list_spec
    return (False, self)


@symbolic.members(
    [
        ('num_choices', 1)
    ],
    init_arg_list=['candidates', 'hints', 'where'],
    serialization_key='hyper.OneOf',
    additional_keys=['pyglove.generators.genetic.ChoiceValue']
)
class OneOf(Choices):
  """N Choose 1.

  Example::

    # A single categorical choice:
    v = pg.oneof([1, 2, 3])

    # A hierarchical categorical choice:
    v2 = pg.oneof([
        'foo',
        'bar',
        pg.oneof([1, 2, 3])
    ])

  See also:

    * :func:`pyglove.oneof`
    * :class:`pyglove.hyper.Choices`
    * :class:`pyglove.hyper.ManyOf`
    * :class:`pyglove.hyper.Float`
    * :class:`pyglove.hyper.CustomHyper`
  """

  def _on_bound(self):
    """Event triggered when members are bound."""
    super()._on_bound()
    assert self.num_choices == 1

  def _decode(self) -> Any:
    """Decode a DNA into an object."""
    return super()._decode()[0]

  def encode(self, value: Any) -> geno.DNA:
    """Encode a value into a DNA."""
    # NOTE(daiyip): Single choice DNA will automatically be pulled
    # up from children to current node. Thus we simply returns
    # encoded DNA from parent node.
    return super().encode([value])

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
      ) -> Tuple[bool, 'OneOf']:
    """Validate candidates during value_spec binding time."""
    # Check if value_spec directly accepts `self`.
    if value_spec.value_type and isinstance(self, value_spec.value_type):
      return (False, self)

    if self._value_spec:
      if not value_spec.is_compatible(self._value_spec):
        raise TypeError(
            object_utils.message_on_path(
                f'Cannot bind an incompatible value spec {value_spec} '
                f'to {self.__class__.__name__} with bound '
                f'spec {self._value_spec}.', path))
      return (False, self)

    for i, c in enumerate(self.candidates):
      value_spec.apply(
          c,
          self._allow_partial,
          root_path=path + f'candidates[{i}]')
    self._value_spec = value_spec
    return (False, self)

#
# Helper methods for creating hyper values.
#


def oneof(candidates: Iterable[Any],
          *,
          name: Optional[str] = None,
          hints: Optional[Any] = None) -> Any:
  """N choose 1.

  Example::

    @pg.members([
      ('x', pg.typing.Int())
    ])
    class A(pg.Object):
      pass

    # A single categorical choice:
    v = pg.oneof([1, 2, 3])

    # A complex type as candidate.
    v1 = pg.oneof(['a', {'x': 1}, A(1)])

    # A hierarchical categorical choice:
    v2 = pg.oneof([
        'foo',
        'bar',
        A(pg.oneof([1, 2, 3]))
    ])

  See also:

    * :class:`pyglove.hyper.OneOf`
    * :func:`pyglove.manyof`
    * :func:`pyglove.floatv`
    * :func:`pyglove.permutate`
    * :func:`pyglove.evolve`

  .. note::

    Under symbolic mode (by default), `pg.oneof` returns a ``pg.hyper.OneOf``
    object. Under dynamic evaluation mode, which is called under the context of
    :meth:`pyglove.hyper.DynamicEvaluationContext.collect` or
    :meth:`pyglove.hyper.DynamicEvaluationContext.apply`, it evaluates to
    a concrete candidate value.

    To use conditional search space in dynamic evaluation mode, the candidate
    should be wrapped with a `lambda` function, which is not necessary under
    symbolic mode. For example::

      pg.oneof([lambda: pg.oneof([0, 1], name='sub'), 2], name='root')

  Args:
    candidates: Candidates to select from. Items of candidate can be any type,
      therefore it can have nested hyper primitives, which forms a hierarchical
      search space.
    name: A name that can be used to identify a decision point in the search
      space. This is needed when the code to instantiate the same hyper
      primitive may be called multiple times under a
      `pg.DynamicEvaluationContext.collect` context or under a
      `pg.DynamicEvaluationContext.apply` context.
    hints: An optional value which acts as a hint for the controller.

  Returns:
    In symbolic mode, this function returns a `ChoiceValue`.
    In dynamic evaluation mode, this function returns one of the items in
    `candidates`.
    If evaluated under a `pg.DynamicEvaluationContext.apply` scope,
    this function will return the selected candidate.
    If evaluated under a `pg.DynamicEvaluationContext.collect`
    scope, it will return the first candidate.
  """
  return OneOf(candidates=list(candidates), name=name, hints=hints)


def manyof(k: int,
           candidates: Iterable[Any],
           distinct: bool = True,
           sorted: bool = False,    # pylint: disable=redefined-builtin
           *,
           name: Optional[str] = None,
           hints: Optional[Any] = None,
           **kwargs) -> Any:
  """N choose K.

  Example::

    @pg.members([
      ('x', pg.typing.Int())
    ])
    class A(pg.Object):
      pass

    # Chooses 2 distinct candidates.
    v = pg.manyof(2, [1, 2, 3])

    # Chooses 2 non-distinct candidates.
    v = pg.manyof(2, [1, 2, 3], distinct=False)

    # Chooses 2 distinct candidates sorted by their indices.
    v = pg.manyof(2, [1, 2, 3], sorted=True)

    # A complex type as candidate.
    v1 = pg.manyof(2, ['a', {'x': 1}, A(1)])

    # A hierarchical categorical choice:
    v2 = pg.manyof(2, [
        'foo',
        'bar',
        A(pg.oneof([1, 2, 3]))
    ])

  .. note::

    Under symbolic mode (by default), `pg.manyof` returns a ``pg.hyper.ManyOf``
    object. Under dynamic evaluation mode, which is called under the context of
    :meth:`pyglove.hyper.DynamicEvaluationContext.collect` or
    :meth:`pyglove.hyper.DynamicEvaluationContext.apply`, it evaluates to
    a concrete candidate value.

    To use conditional search space in dynamic evaluate mode, the candidate
    should be wrapped with a `lambda` function, which is not necessary under
    symbolic mode. For example::

        pg.manyof(2, [
           lambda: pg.oneof([0, 1], name='sub_a'),
           lambda: pg.floatv(0.0, 1.0, name='sub_b'),
           lambda: pg.manyof(2, ['a', 'b', 'c'], name='sub_c')
        ], name='root')

  See also:

    * :class:`pyglove.hyper.ManyOf`
    * :func:`pyglove.manyof`
    * :func:`pyglove.floatv`
    * :func:`pyglove.permutate`
    * :func:`pyglove.evolve`

  Args:
    k: number of choices to make. Should be no larger than the length of
      `candidates` unless `choice_distinct` is set to False,
    candidates: Candidates to select from. Items of candidate can be any type,
      therefore it can have nested hyper primitives, which forms a hierarchical
      search space.
    distinct: If True, each choice needs to be unique.
    sorted: If True, choices are sorted by their indices in the
      candidates.
    name: A name that can be used to identify a decision point in the search
      space. This is needed when the code to instantiate the same hyper
      primitive may be called multiple times under a
      `pg.DynamicEvaluationContext.collect` context or a
      `pg.DynamicEvaluationContext.apply` context.
    hints: An optional value which acts as a hint for the controller.
    **kwargs: Keyword arguments for backward compatibility.
      `choices_distinct`: Old name for `distinct`.
      `choices_sorted`: Old name for `sorted`.

  Returns:
    In symbolic mode, this function returns a `Choices`.
    In dynamic evaluate mode, this function returns a list of items in
    `candidates`.
    If evaluated under a `pg.DynamicEvaluationContext.apply` scope,
    this function will return a list of selected candidates.
    If evaluated under a `pg.DynamicEvaluationContext.collect`
    scope, it will return a list of the first valid combination from the
    `candidates`. For example::

        # Evaluates to [0, 1, 2].
        manyof(3, range(5))

        # Evaluates to [0, 0, 0].
        manyof(3, range(5), distinct=False)
  """
  choices_distinct = kwargs.pop('choices_distinct', distinct)
  choices_sorted = kwargs.pop('choices_sorted', sorted)
  return ManyOf(
      num_choices=k,
      candidates=list(candidates),
      choices_distinct=choices_distinct,
      choices_sorted=choices_sorted,
      name=name,
      hints=hints)


def permutate(candidates: Iterable[Any],
              name: Optional[str] = None,
              hints: Optional[Any] = None) -> Any:
  """Permuatation of candidates.

  Example::

    @pg.members([
      ('x', pg.typing.Int())
    ])
    class A(pg.Object):
      pass

    # Permutates the candidates.
    v = pg.permutate([1, 2, 3])

    # A complex type as candidate.
    v1 = pg.permutate(['a', {'x': 1}, A(1)])

    # A hierarchical categorical choice:
    v2 = pg.permutate([
        'foo',
        'bar',
        A(pg.oneof([1, 2, 3]))
    ])

  .. note::

    Under symbolic mode (by default), `pg.manyof` returns a ``pg.hyper.ManyOf``
    object. Under dynamic evaluate mode, which is called under the context of
    :meth:`pyglove.hyper.DynamicEvaluationContext.collect` or
    :meth:`pyglove.hyper.DynamicEvaluationContext.apply`, it evaluates to
    a concrete candidate value.

    To use conditional search space in dynamic evaluate mode, the candidate
    should be wrapped with a `lambda` function, which is not necessary under
    symbolic mode. For example::

      pg.permutate([
         lambda: pg.oneof([0, 1], name='sub_a'),
         lambda: pg.floatv(0.0, 1.0, name='sub_b'),
         lambda: pg.manyof(2, ['a', 'b', 'c'], name='sub_c')
      ], name='root')

  See also:

    * :class:`pyglove.hyper.ManyOf`
    * :func:`pyglove.oneof`
    * :func:`pyglove.manyof`
    * :func:`pyglove.floatv`
    * :func:`pyglove.evolve`

  Args:
    candidates: Candidates to select from. Items of candidate can be any type,
      therefore it can have nested hyper primitives, which forms a hierarchical
      search space.
    name: A name that can be used to identify a decision point in the search
      space. This is needed when the code to instantiate the same hyper
      primitive may be called multiple times under a
      `pg.DynamicEvaluationContext.collect` context or a
      `pg.DynamicEvaluationContext.apply` context.
    hints: An optional value which acts as a hint for the controller.

  Returns:
    In symbolic mode, this function returns a `Choices`.
    In dynamic evaluate mode, this function returns a permutation from
    `candidates`.
    If evaluated under an `pg.DynamicEvaluationContext.apply` scope,
    this function will return a permutation of candidates based on controller
    decisions.
    If evaluated under a `pg.DynamicEvaluationContext.collect`
    scope, it will return the first valid permutation.
    For example::

      # Evaluates to [0, 1, 2, 3, 4].
      permutate(range(5), name='numbers')
  """
  candidates = list(candidates)
  return manyof(
      len(candidates), candidates,
      choices_distinct=True, choices_sorted=False, name=name, hints=hints)
