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
"""Hyper objects: representing template-based object space.

In PyGlove, an object space is represented by a hyper object, which is an
symbolic object that is placeheld by hyper primitives
(:class:`pyglove.hyper.HyperPrimitive`). Through hyper objects, object templates
(:class:`pyglove.hyper.ObjectTemplate`) can be obtained to generate objects
based on program genomes (:class:`pyglove.DNA`).

 .. graphviz::
    :align: center

    digraph hypers {
      node [shape="box"];
      edge [arrowtail="empty" arrowhead="none" dir="back" style="dashed"];
      hyper [label="HyperValue" href="hyper_value.html"];
      template [label="ObjectTemplate" href="object_template.html"];
      primitive [label="HyperPrimitive" href="hyper_primitive.html"];
      choices [label="Choices" href="choices.html"];
      oneof [label="OneOf" href="oneof_class.html"];
      manyof [label="ManyOf" href="manyof_class.html"];
      float [label="Float" href="float.html"];
      custom [label="CustomHyper" href="custom_hyper.html"];
      hyper -> template;
      hyper -> primitive;
      primitive -> choices;
      choices -> oneof;
      choices -> manyof;
      primitive -> float;
      primitive -> custom
    }

Hyper values map 1:1 to genotypes as the following:

+-------------------------------------+----------------------------------------+
| Hyper class                         | Genotype class                         |
+=====================================+========================================+
|:class:`pyglove.hyper.HyperValue`    |:class:`pyglove.DNASpec`                |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.ObjectTemplate`|:class:`pyglove.geno.Space`             |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.HyperPrimitive`|:class:`pyglove.geno.DecisionPoint`     |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.Choices`       |:class:`pyglove.geno.Choices`           |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.Float`         |:class:`pyglove.geno.Float`             |
+-------------------------------------+----------------------------------------+
|:class:`pyglove.hyper.CustomHyper` :class:`pyglove.geno.CustomDecisionPoint`  |
+------------------------------------------------------------------------------+
"""

import abc
import contextlib
import copy
import numbers
import threading
import types
import typing
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as schema


# Disable implicit str concat in Tuple as it's used for multi-line docstr for
# symbolic members.
# pylint: disable=implicit-str-concat


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
  def dna(self) -> geno.DNA:
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
    ('name', schema.Str().noneable(),
     'Name of the hyper primitive. Useful in define-by-run mode to identify a'
     'decision point in the search space - that is - different instances with '
     'the same name will refer to the same decision point in the search space '
     'under define-by-run mode. '
     'Please refer to `pg.hyper.trace` for details.'),
    ('hints', schema.Any(default=None), 'Generator hints')
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
    dynamic_evaluate_fn = getattr(
        _thread_local_state,
        _TLS_KEY_DYNAMIC_EVALUATE_FN,
        _global_dynamic_evaluate_fn)

    if dynamic_evaluate_fn is None:
      return super().__new__(cls)
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


@symbolic.members([
    ('num_choices', schema.Int(min_value=0).noneable(),
     'Number of choices to make. If set to None, any number of choices is '
     'acceptable.'),
    ('candidates', schema.List(schema.Any()),
     'Candidate values, which may contain nested hyper values.'
     'Candidate can customize its display value (literal) by implementing the '
     '`pg.Formattable` interface.'),
    ('choices_distinct', schema.Bool(True), 'Whether choices are distinct.'),
    ('choices_sorted', schema.Bool(False), 'Whether choices are sorted.'),
    ('where', schema.Callable([schema.Object(HyperPrimitive)],
                              returns=schema.Bool()).noneable(),
     'Callable object to filter nested hyper values. If None, all nested hyper '
     'value will be included in the encoding/decoding process. Otherwise only '
     'the hyper values on which `where` returns True will be included. `where` '
     'can be useful to partition a search space into separate optimization '
     'processes. Please see `ObjectTemplate` docstr for details.')
])
class Choices(HyperPrimitive):
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
        ObjectTemplate(c, where=self.where) for c in self.candidates
    ]
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
      value_spec: schema.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, schema.Field, Any], Any]] = None
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

    list_spec = typing.cast(schema.List,
                            schema.ensure_value_spec(
                                value_spec, schema.List(schema.Any()), path))
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
      value_spec: schema.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, schema.Field, Any], Any]] = None
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


@symbolic.members(
    [
        ('min_value', schema.Float(), 'Minimum acceptable value.'),
        ('max_value', schema.Float(), 'Maximum acceptable value.'),
        geno.float_scale_spec('scale'),
    ],
    init_arg_list=['min_value', 'max_value', 'scale', 'name', 'hints'],
    serialization_key='hyper.Float',
    additional_keys=['pyglove.generators.genetic.Float']
)
class Float(HyperPrimitive):
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
      value_spec: schema.ValueSpec,
      allow_partial: bool = False,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, schema.Field, Any], Any]] = None
      ) -> Tuple[bool, 'Float']:
    """Validate candidates during value_spec binding time."""
    del allow_partial
    del child_transform
    # Check if value_spec directly accepts `self`.
    if value_spec.value_type and isinstance(self, value_spec.value_type):
      return (False, self)

    float_spec = typing.cast(
        schema.Float, schema.ensure_value_spec(
            value_spec, schema.Float(), path))
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


class CustomHyper(HyperPrimitive):
  """User-defined hyper primitive.

  User-defined hyper primitive is useful when users want to have full control
  on the semantics and genome encoding of the search space. For example, the
  decision points are of variable length, which is not yet supported by
  built-in hyper primitives.

  To use user-defined hyper primitive is simple, the user should:


  1) Subclass `CustomHyper` and implements the
     :meth:`pyglove.hyper.CustomHyper.custom_decode` method.
     It's optional to implement the
     :meth:`pyglove.hyper.CustomHyper.custom_encode` method, which is only
     necessary when the user want to encoder a material object into a DNA.
  2) Introduce a DNAGenerator that can generate DNA for the
     :class:`pyglove.geno.CustomDecisionPoint`.

  For example, the following code tries to find an optimal sub-sequence of an
  integer sequence by their sums::

    import random

    class IntSequence(pg.hyper.CustomHyper):

      def custom_decode(self, dna):
        return [int(v) for v in dna.value.split(',') if v != '']

    class SubSequence(pg.evolution.Mutator):

      def mutate(self, dna):
        genome = dna.value
        items = genome.split(',')
        start = random.randint(0, len(items))
        end = random.randint(start, len(items))
        new_genome = ','.join(items[start:end])
        return pg.DNA(new_genome, spec=dna.spec)

    @pg.geno.dna_generator
    def initial_population():
      yield pg.DNA('12,-34,56,-2,100,98', spec=dna_spec)

    algo = pg.evolution.Evolution(
        (pg.evolution.selectors.Random(10)
         >> pg.evolution.selectors.Top(1)
         >> SubSequence()),
        population_init=initial_population(),
        population_update=pg.evolution.selectors.Last(20))

    best_reward, best_example = None, None
    for int_seq, feedback in pg.sample(IntSequence(), algo, num_examples=100):
      reward = sum(int_seq)
      if best_reward is None or best_reward < reward:
        best_reward, best_example = reward, int_seq
      feedback(reward)

    print(best_reward, best_example)

  Please note that user-defined hyper value can be used together with PyGlove's
  built-in hyper primitives, for example::

    pg.oneof([IntSequence(), None])

  Therefore it's also a mechanism to extend PyGlove's search space definitions.
  """

  def _decode(self):
    if not isinstance(self.dna.value, str):
      raise ValueError(
          f'{self.__class__} expects string type DNA. '
          f'Encountered {self.dna!r}.')
    return self.custom_decode(self.dna)

  @abc.abstractmethod
  def custom_decode(self, dna: geno.DNA) -> Any:
    """Decode a DNA whose value is a string of user-defined genome."""

  def encode(self, value: Any) -> geno.DNA:
    """Encode value into DNA with user-defined genome."""
    return self.custom_encode(value)

  def custom_encode(self, value: Any) -> geno.DNA:
    """Encode value to user defined genome."""
    raise NotImplementedError(
        f'\'custom_encode\' is not supported by {self.__class__.__name__!r}.')

  def dna_spec(
      self, location: Optional[object_utils.KeyPath] = None) -> geno.DNASpec:
    """Always returns CustomDecisionPoint for CustomHyper."""
    return geno.CustomDecisionPoint(
        hints=self.hints, name=self.name, location=location)

  def first_dna(self) -> geno.DNA:
    """Returns the first DNA of current sub-space.

    Returns:
      A string-valued DNA.
    """
    raise NotImplementedError(
        f'{self.__class__!r} must implement method `first_dna` to be used in '
        f'dynamic evaluation mode.')

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: schema.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, schema.Field, Any], Any]] = None
      ) -> Tuple[bool, 'CustomHyper']:
    """Validate candidates during value_spec binding time."""
    del path, value_spec, allow_partial, child_transform
    # Allow custom hyper to be assigned to any type.
    return (False, self)


@symbolic.members([
    ('reference_paths', schema.List(schema.Object(object_utils.KeyPath)),
     'Paths of referenced values, which are relative paths searched from '
     'current node to root.')
])
class DerivedValue(symbolic.Object, schema.CustomTyping):
  """Base class of value that references to other values in object tree."""

  @abc.abstractmethod
  def derive(self, *args: Any) -> Any:
    """Derive the value from referenced values."""

  def resolve(self,
              reference_path_or_paths: Optional[Union[Text, List[Text]]] = None
             ) -> Union[
                 Tuple[symbolic.Symbolic, object_utils.KeyPath],
                 List[Tuple[symbolic.Symbolic, object_utils.KeyPath]]]:
    """Resolve reference paths based on the location of this node.

    Args:
      reference_path_or_paths: (Optional) a string or KeyPath as a reference
        path or a list of strings or KeyPath objects as a list of
        reference paths.
        If this argument is not provided, prebound reference paths of this
        object will be used.

    Returns:
      A tuple (or list of tuple) of (resolved parent, resolved full path)
    """
    single_input = False
    if reference_path_or_paths is None:
      reference_paths = self.reference_paths
    elif isinstance(reference_path_or_paths, str):
      reference_paths = [object_utils.KeyPath.parse(reference_path_or_paths)]
      single_input = True
    elif isinstance(reference_path_or_paths, object_utils.KeyPath):
      reference_paths = [reference_path_or_paths]
      single_input = True
    elif isinstance(reference_path_or_paths, list):
      paths = []
      for path in reference_path_or_paths:
        if isinstance(path, str):
          path = object_utils.KeyPath.parse(path)
        elif not isinstance(path, object_utils.KeyPath):
          raise ValueError('Argument \'reference_path_or_paths\' must be None, '
                           'a string, KeyPath object, a list of strings, or a '
                           'list of KeyPath objects.')
        paths.append(path)
      reference_paths = paths
    else:
      raise ValueError('Argument \'reference_path_or_paths\' must be None, '
                       'a string, KeyPath object, a list of strings, or a '
                       'list of KeyPath objects.')

    resolved_paths = []
    for reference_path in reference_paths:
      parent = self.sym_parent
      while parent is not None and not reference_path.exists(parent):
        parent = getattr(parent, 'sym_parent', None)
      if parent is None:
        raise ValueError(
            f'Cannot resolve \'{reference_path}\': parent not found.')
      resolved_paths.append((parent, parent.sym_path + reference_path))
    return resolved_paths if not single_input else resolved_paths[0]

  def __call__(self):
    """Generate value by deriving values from reference paths."""
    referenced_values = []
    for reference_path, (parent, _) in zip(
        self.reference_paths, self.resolve()):
      referenced_value = reference_path.query(parent)

      # Make sure referenced value does not have referenced value.
      # NOTE(daiyip): We can support dependencies between derived values
      # in future if needed.
      if not object_utils.traverse(
          referenced_value, self._contains_not_derived_value):
        raise ValueError(
            f'Derived value (path={referenced_value.sym_path}) should not '
            f'reference derived values. '
            f'Encountered: {referenced_value}, '
            f'Referenced at path {self.sym_path}.')
      referenced_values.append(referenced_value)
    return self.derive(*referenced_values)

  def _contains_not_derived_value(
      self, path: object_utils.KeyPath, value: Any) -> bool:
    """Returns whether a value contains derived value."""
    if isinstance(value, DerivedValue):
      return False
    elif isinstance(value, symbolic.Object):
      for k, v in value.sym_items():
        if not object_utils.traverse(
            v, self._contains_not_derived_value,
            root_path=object_utils.KeyPath(k, path)):
          return False
    return True


class ValueReference(DerivedValue):
  """Class that represents a value referencing another value."""

  def _on_bound(self):
    """Custom init."""
    super()._on_bound()
    if len(self.reference_paths) != 1:
      raise ValueError(
          f'Argument \'reference_paths\' should have exact 1 '
          f'item. Encountered: {self.reference_paths}')

  def derive(self, referenced_value: Any) -> Any:
    """Derive value by return a copy of the referenced value."""
    return copy.copy(referenced_value)

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: schema.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, schema.Field, Any], Any]] = None
      ) -> Tuple[bool, 'DerivedValue']:
    """Implement schema.CustomTyping interface."""
    # TODO(daiyip): perform possible static analysis on referenced paths.
    del path, value_spec, allow_partial, child_transform
    return (False, self)


def reference(reference_path: Text) -> ValueReference:
  """Create a referenced value from a referenced path."""
  return ValueReference(reference_paths=[reference_path])


class ObjectTemplate(HyperValue, object_utils.Formattable):
  """Object template that encodes and decodes symbolic values.

  An object template can be created from a hyper value, which is a symbolic
  object with some parts placeheld by hyper primitives. For example::

    x = A(a=0,
      b=pg.oneof(['foo', 'bar']),
      c=pg.manyof(2, [1, 2, 3, 4, 5, 6]),
      d=pg.floatv(0.1, 0.5),
      e=pg.oneof([
        {
            'f': pg.oneof([True, False]),
        }
        {
            'g': pg.manyof(2, [B(), C(), D()], distinct=False),
            'h': pg.manyof(2, [0, 1, 2], sorted=True),
        }
      ])
    })
    t = pg.template(x)

  In this example, the root template have 4 children hyper primitives associated
  with keys 'b', 'c', 'd' and 'e', while the hyper primitive 'e' have 3 children
  associated with keys 'f', 'g' and 'h', creating a conditional search space.

  Thus the DNA shape is determined by the definition of template, described
  by geno.DNASpec. In this case, the DNA spec of this template looks like::

    pg.geno.space([
        pg.geno.oneof([            # Spec for 'b'.
            pg.geno.constant(),    # A constant template for 'foo'.
            pg.geno.constant(),    # A constant template for 'bar'.
        ]),
        pg.geno.manyof([           # Spec for 'c'.
            pg.geno.constant(),    # A constant template for 1.
            pg.geno.constant(),    # A constant template for 2.
            pg.geno.constant(),    # A constant template for 3.
            pg.geno.constant(),    # A constant template for 4.
            pg.geno.constant(),    # A constant template for 5.
            pg.geno.constant(),    # A constant template for 6.
        ]),
        pg.geno.floatv(0.1, 0.5),  # Spec for 'd'.
        pg.geno.oneof([            # Spec for 'e'.
            pg.geno.space([
                pg.geno.oneof([          # Spec for 'f'.
                    pg.geno.constant(),  # A constant template for True.
                    pg.geno.constant(),  # A constant template for False.
                ])
            ]),
            pg.geno.space([
                pg.geno.manyof(2, [         # Spec for 'g'.
                    pg.geno.constant(),     # A constant template for B().
                    pg.geno.constant(),     # A constant template for C().
                    pg.geno.constant(),     # A constant template for D().
                ], distinct=False)    # choices of the same value can
                                      # be selected multiple times.
                pg.geno.manyof(2, [         # Spec for 'h'.
                    pg.geno.constant(),     # A constant template for 0.
                    pg.geno.constant(),     # A constant template for 1.
                    pg.geno.constant(),     # A constant template for 2.
                ], sorted=True)       # acceptable choices needs to be sorted,
                                      # which enables using choices as set (of
                                      # possibly repeated values).
            ])
        ])

  It may generate DNA as the following:
    DNA([0, [0, 2], 0.1, (0, 0)])

  A template can also work only on a subset of hyper primitives from the input
  value through the `where` function. This is useful to partition a search space
  into parts for separate optimization.

  For example::

    t = pg.hyper.ObjectTemplate(
      A(a=pg.oneof([1, 2]), b=pg.oneof([3, 4])),
      where=lambda e: e.root_path == 'a')
    assert t.dna_spec() == pg.geno.space([
        pg.geno.oneof(location='a', candidates=[
            pg.geno.constant(),   # For a=1
            pg.geno.constant(),   # For a=2
        ], literal_values=['(0/2) 1', '(1/2) 2'])
    ])
    assert t.decode(pg.DNA(0)) == A(a=1, b=pg.oneof([3, 4]))
  """

  def __init__(self,
               value: Any,
               compute_derived: bool = False,
               where: Optional[Callable[[HyperPrimitive], bool]] = None):
    """Constructor.

    Args:
      value: Value (maybe) annotated with generators to use as template.
      compute_derived: Whether to compute derived value at this level.
        We only want to compute derived value at root level since reference path
        may go out of scope of a non-root ObjectTemplate.
      where: Function to filter hyper primitives. If None, all hyper primitives
        from `value` will be included in the encoding/decoding process.
        Otherwise only the hyper primitives on which 'where' returns True will
        be included. `where` can be useful to partition a search space into
        separate optimization processes.
        Please see 'ObjectTemplate' docstr for details.
    """
    super().__init__()
    self._value = value
    self._root_path = object_utils.KeyPath()
    self._compute_derived = compute_derived
    self._where = where
    self._parse_generators()

  @property
  def root_path(self) -> object_utils.KeyPath:
    """Returns root path."""
    return self._root_path

  @root_path.setter
  def root_path(self, path: object_utils.KeyPath):
    """Set root path."""
    self._root_path = path

  def _parse_generators(self) -> None:
    """Parse generators from its templated value."""
    hyper_primitives = []
    def _extract_immediate_child_hyper_primitives(
        path: object_utils.KeyPath, value: Any) -> bool:
      """Extract top-level hyper primitives."""
      if (isinstance(value, HyperValue)
          and (not self._where or self._where(value))):
        # Apply where clause to child choices.
        if isinstance(value, Choices) and self._where:
          value = value.clone().rebind(where=self._where)
        hyper_primitives.append((path, value))
      elif isinstance(value, symbolic.Object):
        for k, v in value.sym_items():
          object_utils.traverse(
              v, _extract_immediate_child_hyper_primitives,
              root_path=object_utils.KeyPath(k, path))
      return True

    object_utils.traverse(
        self._value, _extract_immediate_child_hyper_primitives)
    self._hyper_primitives = hyper_primitives

  @property
  def value(self) -> Any:
    """Returns templated value."""
    return self._value

  @property
  def hyper_primitives(self) -> List[Tuple[Text, HyperValue]]:
    """Returns hyper primitives in tuple (relative path, hyper primitive)."""
    return self._hyper_primitives

  @property
  def is_constant(self) -> bool:
    """Returns whether current template is constant value."""
    return not self._hyper_primitives

  def dna_spec(
      self, location: Optional[object_utils.KeyPath] = None) -> geno.Space:
    """Return DNA spec (geno.Space) from this template."""
    return geno.Space(
        elements=[
            primitive.dna_spec(primitive_location)
            for primitive_location, primitive in self._hyper_primitives
        ],
        location=location or object_utils.KeyPath())

  def _decode(self) -> Any:
    """Decode DNA into a value."""
    dna = self._dna
    if not self._hyper_primitives and (dna.value is not None or dna.children):
      raise ValueError(
          object_utils.message_on_path(
              f'Encountered extra DNA value to decode: {dna!r}',
              self._root_path))

    # Compute hyper primitive values first.
    rebind_dict = {}
    if len(self._hyper_primitives) == 1:
      primitive_location, primitive = self._hyper_primitives[0]
      rebind_dict[primitive_location.path] = primitive.decode(dna)
    else:
      if len(dna.children) != len(self._hyper_primitives):
        raise ValueError(
            object_utils.message_on_path(
                f'The length of child values ({len(dna.children)}) is '
                f'different from the number of hyper primitives '
                f'({len(self._hyper_primitives)}) in ObjectTemplate. '
                f'DNA={dna!r}, ObjectTemplate={self!r}.', self._root_path))
      for i, (primitive_location, primitive) in enumerate(
          self._hyper_primitives):
        rebind_dict[primitive_location.path] = (
            primitive.decode(dna.children[i]))

    if rebind_dict:
      if len(rebind_dict) == 1 and '' in rebind_dict:
        # NOTE(daiyip): Special handle the case when the root value needs to be
        # replaced. For example: `template(oneof([0, 1])).decode(geno.DNA(0))`
        # should return 0 instead of rebinding the root `OneOf` object.
        value = rebind_dict['']
      else:
        # NOTE(daiyip): Instead of deep copying the whole object (with hyper
        # primitives), we can cherry-pick only non-hyper parts. Unless we saw
        # performance issues it's not worthy to optimize this.
        value = symbolic.clone(self._value, deep=True)
        value.rebind(rebind_dict)
      copied = True
    else:
      assert self.is_constant
      value = self._value
      copied = False

    # Compute derived values if needed.
    if self._compute_derived:
      # TODO(daiyip): Currently derived value parsing is done at decode time,
      # which can be optimized by moving to template creation time.
      derived_values = []
      def _extract_derived_values(
          path: object_utils.KeyPath, value: Any) -> bool:
        """Extract top-level primitives."""
        if isinstance(value, DerivedValue):
          derived_values.append((path, value))
        elif isinstance(value, symbolic.Object):
          for k, v in value.sym_items():
            object_utils.traverse(
                v, _extract_derived_values,
                root_path=object_utils.KeyPath(k, path))
        return True
      object_utils.traverse(value, _extract_derived_values)

      if derived_values:
        if not copied:
          value = symbolic.clone(value, deep=True)
        rebind_dict = {}
        for path, derived_value in derived_values:
          rebind_dict[path.path] = derived_value()
        assert rebind_dict
        value.rebind(rebind_dict)
    return value

  def encode(self, value: Any) -> geno.DNA:
    """Encode a value into a DNA.

    Example::

      # DNA of a constant template:
      template = pg.hyper.ObjectTemplate({'a': 0})
      assert template.encode({'a': 0}) == pg.DNA(None)
      # Raises: Unmatched value between template and input.
      template.encode({'a': 1})

      # DNA of a template containing only one pg.oneof.
      template = pg.hyper.ObjectTemplate({'a': pg.oneof([1, 2])})
      assert template.encode({'a': 1}) == pg.DNA(0)

      # DNA of a template containing only one pg.oneof.
      template = pg.hyper.ObjectTemplate({'a': pg.floatv(0.1, 1.0)})
      assert template.encode({'a': 0.5}) == pg.DNA(0.5)

    Args:
      value: Value to encode.

    Returns:
      Encoded DNA.

    Raises:
      ValueError if value cannot be encoded by this template.
    """
    children = []
    def _encode(path: object_utils.KeyPath,
                template_value: Any,
                input_value: Any) -> Any:
      """Encode input value according to template value."""
      if (schema.MISSING_VALUE == input_value
          and schema.MISSING_VALUE != template_value):
        raise ValueError(
            f'Value is missing from input. Path=\'{path}\'.')
      if (isinstance(template_value, HyperValue)
          and (not self._where or self._where(template_value))):
        children.append(template_value.encode(input_value))
      elif isinstance(template_value, DerivedValue):
        if self._compute_derived:
          referenced_values = [
              reference_path.query(value)
              for _, reference_path in template_value.resolve()
          ]
          derived_value = template_value.derive(*referenced_values)
          if derived_value != input_value:
            raise ValueError(
                f'Unmatched derived value between template and input. '
                f'(Path=\'{path}\', Template={template_value!r}, '
                f'ComputedValue={derived_value!r}, Input={input_value!r})')
        # For template that doesn't compute derived value, it get passed over
        # to parent template who may be able to handle.
      elif isinstance(template_value, symbolic.Object):
        if type(input_value) is not type(template_value):
          raise ValueError(
              f'Unmatched Object type between template and input: '
              f'(Path=\'{path}\', Template={template_value!r}, '
              f'Input={input_value!r})')
        template_keys = set(template_value.sym_keys())
        value_keys = set(input_value.sym_keys())
        if template_keys != value_keys:
          raise ValueError(
              f'Unmatched Object keys between template value and input '
              f'value. (Path=\'{path}\', '
              f'TemplateOnlyKeys={template_keys - value_keys}, '
              f'InputOnlyKeys={value_keys - template_keys})')
        for key in template_value.sym_keys():
          object_utils.merge_tree(
              template_value.sym_getattr(key),
              input_value.sym_getattr(key),
              _encode, root_path=object_utils.KeyPath(key, path))
      elif isinstance(template_value, symbolic.Dict):
        # Do nothing since merge will iterate all elements in dict and list.
        if not isinstance(input_value, dict):
          raise ValueError(
              f'Unmatched dict between template value and input '
              f'value. (Path=\'{path}\', Template={template_value!r}, '
              f'Input={input_value!r})')
      elif isinstance(template_value, symbolic.List):
        if (not isinstance(input_value, list)
            or len(input_value) != len(template_value)):
          raise ValueError(
              f'Unmatched list between template value and input '
              f'value. (Path=\'{path}\', Template={template_value!r}, '
              f'Input={input_value!r})')
        for i, template_item in enumerate(template_value):
          object_utils.merge_tree(
              template_item, input_value[i], _encode,
              root_path=object_utils.KeyPath(i, path))
      else:
        if template_value != input_value:
          raise ValueError(
              f'Unmatched value between template and input. '
              f'(Path=\'{path}\', '
              f'Template={object_utils.quote_if_str(template_value)}, '
              f'Input={object_utils.quote_if_str(input_value)})')
      return template_value
    object_utils.merge_tree(
        self._value, value, _encode, root_path=self._root_path)
    return geno.DNA(None, children)

  def try_encode(self, value: Any) -> Tuple[bool, geno.DNA]:
    """Try to encode a value without raise Exception."""
    try:
      dna = self.encode(value)
      return (True, dna)
    except ValueError:
      return (False, None)  # pytype: disable=bad-return-type
    except KeyError:
      return (False, None)  # pytype: disable=bad-return-type

  def __eq__(self, other):
    """Operator ==."""
    if not isinstance(other, self.__class__):
      return False
    return self.value == other.value

  def __ne__(self, other):
    """Operator !=."""
    return not self.__eq__(other)

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs) -> Text:
    """Format this object."""
    details = object_utils.format(
        self._value, compact, verbose, root_indent, **kwargs)
    return f'{self.__class__.__name__}(value={details})'

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: schema.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, schema.Field, Any], Any]] = None
      ) -> Tuple[bool, 'ObjectTemplate']:
    """Validate candidates during value_spec binding time."""
    # Check if value_spec directly accepts `self`.
    if not value_spec.value_type or not isinstance(self, value_spec.value_type):
      value_spec.apply(
          self._value,
          allow_partial,
          root_path=self.root_path)
    return (False, self)


# TODO(daiyip): For backward compatibility, remove after legacy dependencies
# are updated.
ChoiceList = ManyOf
ChoiceValue = OneOf
Template = ObjectTemplate


#
# Helper methods for creating hyper values.
#


def oneof(candidates: Iterable[Any],
          *,
          name: Optional[Text] = None,
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


one_of = oneof


def manyof(k: int,
           candidates: Iterable[Any],
           distinct: bool = True,
           sorted: bool = False,    # pylint: disable=redefined-builtin
           *,
           name: Optional[Text] = None,
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


sublist_of = manyof


def permutate(candidates: Iterable[Any],
              name: Optional[Text] = None,
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


def floatv(min_value: float,
           max_value: float,
           scale: Optional[Text] = None,
           *,
           name: Optional[Text] = None,
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


# For backward compatibility
float_value = floatv


def template(
    value: Any,
    where: Optional[Callable[[HyperPrimitive], bool]] = None) -> ObjectTemplate:
  """Creates an object template from the input.

  Example::

    d = pg.Dict(x=pg.oneof(['a', 'b', 'c'], y=pg.manyof(2, range(4))))
    t = pg.template(d)

    assert t.dna_spec() == pg.geno.space([
        pg.geno.oneof([
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant(),
        ], location='x'),
        pg.geno.manyof([
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant(),
        ], location='y')
    ])

    assert t.encode(pg.Dict(x='a', y=0)) == pg.DNA([0, 0])
    assert t.decode(pg.DNA([0, 0])) == pg.Dict(x='a', y=0)

    t = pg.template(d, where=lambda x: isinstance(x, pg.hyper.ManyOf))
     assert t.dna_spec() == pg.geno.space([
        pg.geno.manyof([
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant(),
        ], location='y')
    ])
    assert t.encode(pg.Dict(x=pg.oneof(['a', 'b', 'c']), y=0)) == pg.DNA(0)
    assert t.decode(pg.DNA(0)) == pg.Dict(x=pg.oneof(['a', 'b', 'c']), y=0)

  Args:
    value: A value based on which the template is created.
    where: Function to filter hyper values. If None, all hyper primitives from
      `value` will be included in the encoding/decoding process. Otherwise
      only the hyper values on which 'where' returns True will be included.
      `where` can be useful to partition a search space into separate
      optimization processes. Please see 'ObjectTemplate' docstr for details.

  Returns:
    A template object.
  """
  return ObjectTemplate(value, compute_derived=True, where=where)


#
# Helper methods for operating on hyper values.
#


def dna_spec(
    value: Any,
    where: Optional[Callable[[HyperPrimitive], bool]] = None) -> geno.DNASpec:
  """Returns the DNASpec from a (maybe) hyper value.

  Example::

    hyper = pg.Dict(x=pg.oneof([1, 2, 3]), y=pg.oneof(['a', 'b']))
    spec = pg.dna_spec(hyper)

    assert spec.space_size == 6
    assert len(spec.decision_points) == 2
    print(spec.decision_points)

    # Select a partial space with `where` argument.
    spec = pg.dna_spec(hyper, where=lambda x: len(x.candidates) == 2)

    assert spec.space_size == 2
    assert len(spec.decision_points) == 1

  See also:

    * :class:`pyglove.DNASpec`
    * :class:`pyglove.DNA`

  Args:
    value: A (maybe) hyper value.
    where: Function to filter hyper primitives. If None, all hyper primitives
      from `value` will be included in the encoding/decoding process. Otherwise
      only the hyper primitives on which 'where' returns True will be included.
      `where` can be very useful to partition a search space into separate
      optimization processes. Please see 'Template' docstr for details.

  Returns:
    A DNASpec object, which represents the search space from algorithm's view.
  """
  return template(value, where).dna_spec()


# NOTE(daiyip): For backward compatibility, we use `search_space` as an alias
# for `dna_spec`. Once downstream users are updated to call `dna_spec`, we will
# remove this method.
search_space = dna_spec


def materialize(
    value: Any,
    parameters: Union[geno.DNA, Dict[Text, Any]],
    use_literal_values: bool = True,
    where: Optional[Callable[[HyperPrimitive], bool]] = None) -> Any:
  """Materialize a (maybe) hyper value using a DNA or parameter dict.

  Example::

    hyper_dict = pg.Dict(x=pg.oneof(['a', 'b']), y=pg.floatv(0.0, 1.0))

    # Materialize using DNA.
    assert pg.materialize(
      hyper_dict, pg.DNA([0, 0.5])) == pg.Dict(x='a', y=0.5)

    # Materialize usign key value pairs.
    # See `pg.DNA.from_dict` for more details.
    assert pg.materialize(
      hyper_dict, {'x': 0, 'y': 0.5}) == pg.Dict(x='a', y=0.5)

    # Partially materialize.
    v = pg.materialize(
      hyper_dict, pg.DNA(0), where=lambda x: isinstance(x, pg.hyper.OneOf))
    assert v == pg.Dict(x='a', y=pg.floatv(0.0, 1.0))

  Args:
    value: A (maybe) hyper value
    parameters: A DNA object or a dict of string (key path) to a
      string (in format of '<selected_index>/<num_choices>' for
      `geno.Choices`, or '<float_value>' for `geno.Float`), or their literal
      values when `use_literal_values` is set to True.
    use_literal_values: Applicable when `parameters` is a dict. If True, the
      values in the dict will be from `geno.Choices.literal_values` for
      `geno.Choices`.
    where: Function to filter hyper primitives. If None, all hyper primitives
      from `value` will be included in the encoding/decoding process. Otherwise
      only the hyper primitives on which 'where' returns True will be included.
      `where` can be useful to partition a search space into separate
      optimization processes. Please see 'Template' docstr for details.

  Returns:
    A materialized value.

  Raises:
    TypeError: if parameters is not a DNA or dict.
    ValueError: if parameters cannot be decoded.
  """
  t = template(value, where)
  if isinstance(parameters, dict):
    dna = geno.DNA.from_parameters(
        parameters=parameters,
        dna_spec=t.dna_spec(),
        use_literal_values=use_literal_values)
  else:
    dna = parameters

  if not isinstance(dna, geno.DNA):
    raise TypeError(
        f'\'parameters\' must be a DNA or a dict of string to DNA values. '
        f'Encountered: {dna!r}.')
  return t.decode(dna)


def iterate(hyper_value: Any,
            num_examples: Optional[int] = None,
            algorithm: Optional[geno.DNAGenerator] = None,
            where: Optional[Callable[[HyperPrimitive], bool]] = None,
            force_feedback: bool = False):
  """Iterate a hyper value based on an algorithm.

  Example::

    hyper_dict = pg.Dict(x=pg.oneof([1, 2, 3]), y=pg.oneof(['a', 'b']))

    # Get all examples from the hyper_dict.
    assert list(pg.iter(hyper_dict)) == [
        pg.Dict(x=1, y='a'),
        pg.Dict(x=1, y='b'),
        pg.Dict(x=2, y='a'),
        pg.Dict(x=2, y='b'),
        pg.Dict(x=3, y='a'),
        pg.Dict(x=3, y='b'),
    ]

    # Get the first two examples.
    assert list(pg.iter(hyper_dict, 2)) == [
        pg.Dict(x=1, y='a'),
        pg.Dict(x=1, y='b'),
    ]

    # Random sample examples, which is equivalent to `pg.random_sample`.
    list(pg.iter(hyper_dict, 2, pg.geno.Random()))

    # Iterate examples with feedback loop.
    for d, feedback in pg.iter(
        hyper_dict, 10,
        pg.evolution.regularized_evolution(pg.evolution.mutators.Uniform())):
      feedback(d.x)

    # Only materialize selected parts.
    assert list(
        pg.iter(hyper_dict, where=lambda x: len(x.candidates) == 2)) == [
            pg.Dict(x=pg.oneof([1, 2, 3]), y='a'),
            pg.Dict(x=pg.oneof([1, 2, 3]), y='b'),
        ]

  ``pg.iter`` distinguishes from `pg.sample` in that it's designed
  for simple in-process iteration, which is handy for quickly generating
  examples from algorithms without maintaining trail states. On the contrary,
  `pg.sample` is designed for distributed sampling, with parallel workers and
  failover handling.

  Args:
    hyper_value: A hyper value that represents a space of instances.
    num_examples: An optional integer as the max number of examples to
        propose. If None, propose will return an iterator of infinite examples.
    algorithm: An optional DNA generator. If None, Sweeping will be used, which
        iterates examples in order.
    where: Function to filter hyper primitives. If None, all hyper primitives
      from `value` will be included in the encoding/decoding process. Otherwise
      only the hyper primitives on which 'where' returns True will be included.
      `where` can be useful to partition a search space into separate
      optimization processes. Please see 'Template' docstr for details.
    force_feedback: If True, always return the Feedback object together
      with the example, this is useful when the user want to pass different
      DNAGenerators to `pg.iter` and want to handle them uniformly.

  Yields:
    A tuple of (example, feedback_fn) if the algorithm needs a feedback or
    `force_feedback` is True, otherwise the example.

  Raises:
    ValueError: when `hyper_value` is a constant value.
  """
  if isinstance(hyper_value, DynamicEvaluationContext):
    dynamic_evaluation_context = hyper_value
    spec = hyper_value.dna_spec
    t = None
  else:
    t = template(hyper_value, where)
    if t.is_constant:
      raise ValueError(
          f'\'hyper_value\' is a constant value: {hyper_value!r}.')
    dynamic_evaluation_context = None
    spec = t.dna_spec()

  if algorithm is None:
    algorithm = geno.Sweeping()

  # NOTE(daiyip): algorithm can continue if it's already set up with the same
  # DNASpec, or we will setup the algorithm with the DNASpec from the template.
  if algorithm.dna_spec is None:
    algorithm.setup(spec)
  elif symbolic.ne(spec, algorithm.dna_spec):
    raise ValueError(
        f'{algorithm!r} has been set up with a different DNASpec. '
        f'Existing: {algorithm.dna_spec!r}, New: {spec!r}.')

  count = 0
  while num_examples is None or count < num_examples:
    try:
      count += 1
      dna = algorithm.propose()
      if t is not None:
        example = t.decode(dna)
      else:
        assert dynamic_evaluation_context is not None
        example = lambda: dynamic_evaluation_context.apply(dna)
      if force_feedback or algorithm.needs_feedback:
        yield example, Feedback(algorithm, dna)
      else:
        yield example
    except StopIteration:
      return


class Feedback:
  """Feedback object."""

  def __init__(self, algorithm: geno.DNAGenerator, dna: geno.DNA):
    """Creates a feedback object."""
    self._algorithm = algorithm
    self._dna = dna

  def __call__(self, reward: Union[float, Tuple[float, ...]]):
    """Call to feedback reward."""
    self._algorithm.feedback(self._dna, reward)

  @property
  def dna(self) -> geno.DNA:
    """Returns DNA."""
    return self._dna


def random_sample(
    value: Any,
    num_examples: Optional[int] = None,
    where: Optional[Callable[[HyperPrimitive], bool]] = None,
    seed: Optional[int] = None):
  """Returns an iterator of random sampled examples.

  Example::

    hyper_dict = pg.Dict(x=pg.oneof(range(3)), y=pg.floatv(0.0, 1.0))

    # Generate one random example from the hyper_dict.
    d = next(pg.random_sample(hyper_dict))

    # Generate 5 random examples with random seed.
    ds = list(pg.random_sample(hyper_dict, 5, seed=1))

    # Generate 3 random examples of `x` with `y` intact.
    ds = list(pg.random_sample(hyper_dict, 3,
        where=lambda x: isinstance(x, pg.hyper.OneOf)))


  Args:
    value: A (maybe) hyper value.
    num_examples: An optional integer as number of examples to propose. If None,
      propose will return an iterator that iterates forever.
    where: Function to filter hyper primitives. If None, all hyper primitives in
      `value` will be included in the encoding/decoding process. Otherwise only
      the hyper primitives on which 'where' returns True will be included.
      `where` can be useful to partition a search space into separate
      optimization processes. Please see 'Template' docstr for details.
    seed: An optional integer as random seed.

  Returns:
    Iterator of random examples.
  """
  return iterate(
      value, num_examples, geno.Random(seed), where=where)

#
# Methods for dynamically evaluting hyper values.
#


_thread_local_state = threading.local()
_TLS_KEY_DYNAMIC_EVALUATE_FN = 'dynamic_evaluate_fn'
_global_dynamic_evaluate_fn = None


@contextlib.contextmanager
def dynamic_evaluate(evaluate_fn: Optional[Callable[[HyperValue], Any]],
                     yield_value: Optional[Any] = None,
                     exit_fn: Optional[Callable[[], None]] = None,
                     per_thread: bool = True):
  """Eagerly evaluate hyper primitives within current scope.

  Example::

    global_indices = [0]
    def evaluate_fn(x: pg.hyper.HyperPrimitive):
      if isinstance(x, pg.hyper.OneOf):
        return x.candidates[global_indices[0]]
      raise NotImplementedError()

    with pg.hyper.dynamic_evaluate(evaluate_fn):
      assert 0 = pg.oneof([0, 1, 2])

  Please see :meth:`pyglove.DynamicEvaluationContext.apply` as an example
  for using this method.

  Args:
    evaluate_fn: A callable object that evaluates a hyper value such as
      oneof, manyof, floatv, and etc. into a concrete value.
    yield_value: Value to yield return.
    exit_fn: A callable object to be called when exiting the context scope.
    per_thread: If True, the context manager will be applied to current thread
      only. Otherwise, it will be applied on current process.

  Yields:
    `yield_value` from the argument.
  """
  global _global_dynamic_evaluate_fn
  if evaluate_fn is not None and not callable(evaluate_fn):
    raise ValueError(
        f'\'evaluate_fn\' must be either None or a callable object. '
        f'Encountered: {evaluate_fn!r}.')
  if exit_fn is not None and not callable(exit_fn):
    raise ValueError(
        f'\'exit_fn\' must be a callable object. Encountered: {exit_fn!r}.')
  if per_thread:
    old_evaluate_fn = getattr(
        _thread_local_state, _TLS_KEY_DYNAMIC_EVALUATE_FN, None)
  else:
    old_evaluate_fn = _global_dynamic_evaluate_fn

  has_errors = False
  try:
    if per_thread:
      setattr(_thread_local_state, _TLS_KEY_DYNAMIC_EVALUATE_FN, evaluate_fn)
    else:
      _global_dynamic_evaluate_fn = evaluate_fn
    yield yield_value
  except Exception:
    has_errors = True
    raise
  finally:
    if per_thread:
      setattr(
          _thread_local_state, _TLS_KEY_DYNAMIC_EVALUATE_FN, old_evaluate_fn)
    else:
      _global_dynamic_evaluate_fn = old_evaluate_fn
    if not has_errors and exit_fn is not None:
      exit_fn()


class DynamicEvaluationContext:
  """Context for dynamic evaluation of hyper primitives.

  Example::

    import pyglove as pg

    # Define a function that implicitly declares a search space.
    def foo():
      return pg.oneof(range(-10, 10)) ** 2 + pg.oneof(range(-10, 10)) ** 2

    # Define the search space by running the `foo` once.
    search_space = pg.hyper.DynamicEvaluationContext()
    with search_space.collect():
      _ = foo()

    # Create a search algorithm.
    search_algorithm = pg.evolution.regularized_evolution(
        pg.evolution.mutators.Uniform(), population_size=32, tournament_size=16)

    # Define the feedback loop.
    best_foo, best_reward = None, None
    for example, feedback in pg.sample(
        search_space, search_algorithm, num_examples=100):
      # Call to `example` returns a context manager
      # under which the `program` is connected with
      # current search algorithm decisions.
      with example():
        reward = foo()
      feedback(reward)
      if best_reward is None or best_reward < reward:
        best_foo, best_reward = example, reward
  """

  def __init__(self,
               where: Optional[Callable[[HyperPrimitive], bool]] = None,
               require_hyper_name: bool = False,
               per_thread: bool = True,
               dna_spec: Optional[geno.DNASpec] = None) -> None:  # pylint: disable=redefined-outer-name
    """Create a dynamic evaluation context.

    Args:
      where: A callable object that decide whether a hyper primitive should be
        included when being instantiated under `collect`.
        If None, all hyper primitives under `collect` will be
        included.
      require_hyper_name: If True, all hyper primitives (e.g. pg.oneof) must
        come with a `name`. This option helps to eliminate errors when a
        function that contains hyper primitive definition may be called multiple
        times. Since hyper primitives sharing the same name will be registered
        to the same decision point, repeated call to the hyper primitive
        definition will not matter.
      per_thread: If True, the context manager will be applied to current thread
        only. Otherwise, it will be applied on current process.
      dna_spec: External provided search space. If None, the dynamic evaluation
        context can be used to create new search space via `colelct` context
        manager. Otherwise, current context will use the provided DNASpec to
        apply decisions.
    """
    self._where = where
    self._require_hyper_name: bool = require_hyper_name
    self._name_to_hyper: Dict[Text, HyperPrimitive] = dict()
    self._annoymous_hyper_name_accumulator: int = 0
    self._hyper_dict = symbolic.Dict() if dna_spec is None else None
    self._dna_spec: Optional[geno.DNASpec] = dna_spec
    self._per_thread = per_thread

  @property
  def dna_spec(self) -> geno.DNASpec:
    """Returns the DNASpec of the search space defined so far."""
    if self._dna_spec is None:
      assert self._hyper_dict is not None
      self._dna_spec = dna_spec(self._hyper_dict)
    return self._dna_spec

  def _decision_name(self, hyper_primitive: HyperPrimitive) -> Text:
    """Get the name for a decision point."""
    name = hyper_primitive.name
    if name is None:
      if self._require_hyper_name:
        raise ValueError(
            f'\'name\' must be specified for hyper '
            f'primitive {hyper_primitive!r}.')
      name = f'decision_{self._annoymous_hyper_name_accumulator}'
      self._annoymous_hyper_name_accumulator += 1
    return name

  @property
  def is_external(self) -> bool:
    """Returns True if the search space is defined by an external DNASpec."""
    return self._hyper_dict is None

  @property
  def hyper_dict(self) -> Optional[symbolic.Dict]:
    """Returns collected hyper primitives as a dict.

    None if current context is controlled by an external DNASpec.
    """
    return self._hyper_dict

  @contextlib.contextmanager
  def collect(self):
    """A context manager for collecting hyper primitives within this context.

    Example::

      context = DynamicEvaluationContext()
      with context.collect():
        x = pg.oneof([1, 2, 3]) + pg.oneof([4, 5, 6])

      # Will print 1 + 4 = 5. Meanwhile 2 hyper primitives will be registered
      # in the search space represented by the context.
      print(x)

    Yields:
      The hyper dict representing the search space.
    """
    if self.is_external:
      raise ValueError(
          f'`collect` cannot be called on a dynamic evaluation context that is '
          f'using an external DNASpec: {self._dna_spec}.')

    with self._collect() as sub_space:
      try:
        yield self._hyper_dict
      finally:
        # NOTE(daiyip): when registering new hyper primitives in the sub-space,
        # the keys are already ensured not to conflict with the keys in current
        # search space. Therefore it's safe to update current space.
        self._hyper_dict.update(sub_space)

        # Invalidate DNASpec.
        self._dna_spec = None

  def _collect(self):
    """A context manager for collecting hyper primitive within the scope."""
    hyper_dict = symbolic.Dict()

    def _register_child(c):
      if isinstance(c, types.LambdaType):
        s = schema.get_signature(c)
        if not s.args and not s.has_wildcard_args:
          with self._collect() as child_hyper:
            v = c()
          return (v, child_hyper)
      return (c, c)

    def _register_hyper_primitive(hyper_primitive):
      """Registers a decision point from an hyper_primitive."""
      if self._where and not self._where(hyper_primitive):
        # Skip hyper primitives that do not pass the `where` predicate.
        return hyper_primitive

      if isinstance(hyper_primitive, Template):
        return hyper_primitive.value

      assert isinstance(hyper_primitive, HyperPrimitive), hyper_primitive
      name = self._decision_name(hyper_primitive)
      if isinstance(hyper_primitive, Choices):
        candidate_values, candidates = zip(
            *[_register_child(c) for c in hyper_primitive.candidates])
        if hyper_primitive.choices_distinct:
          assert hyper_primitive.num_choices <= len(hyper_primitive.candidates)
          v = [candidate_values[i] for i in range(hyper_primitive.num_choices)]
        else:
          v = [candidate_values[0]] * hyper_primitive.num_choices
        hyper_primitive = hyper_primitive.clone(deep=True, override={
            'candidates': list(candidates)
        })
        first_value = v[0] if isinstance(hyper_primitive, ChoiceValue) else v
      elif isinstance(hyper_primitive, Float):
        first_value = hyper_primitive.min_value
      else:
        assert isinstance(hyper_primitive, CustomHyper), hyper_primitive
        first_value = hyper_primitive.decode(hyper_primitive.first_dna())

      if (name in self._name_to_hyper
          and hyper_primitive != self._name_to_hyper[name]):
        raise ValueError(
            f'Found different hyper primitives under the same name {name!r}: '
            f'Instance1={self._name_to_hyper[name]!r}, '
            f'Instance2={hyper_primitive!r}.')
      hyper_dict[name] = hyper_primitive
      self._name_to_hyper[name] = hyper_primitive
      return first_value
    return dynamic_evaluate(
        _register_hyper_primitive, hyper_dict, per_thread=self._per_thread)

  def _decision_getter_and_evaluation_finalizer(
      self, decisions: Union[geno.DNA, List[Union[int, float, str]]]):
    """Returns decision getter based on input decisions."""
    # NOTE(daiyip): when hyper primitives are required to carry names, we do
    # decision lookup from the DNA dict. This allows the decision points
    # to appear in any order other than strictly following the order of their
    # appearences during the search space inspection.
    if self._require_hyper_name:
      if isinstance(decisions, list):
        dna = geno.DNA.from_numbers(decisions, self.dna_spec)
      else:
        dna = decisions
        dna.use_spec(self.dna_spec)
      decision_dict = dna.to_dict(
          key_type='name_or_id', multi_choice_key='parent')

      used_decision_names = set()
      def get_decision_from_dict(
          hyper_primitive, sub_index: Optional[int] = None
          ) -> Union[int, float, str]:
        name = hyper_primitive.name
        assert name is not None, hyper_primitive
        if name not in decision_dict:
          raise ValueError(
              f'Hyper primitive {hyper_primitive!r} is not defined during '
              f'search space inspection (pg.hyper.DynamicEvaluationContext.'
              f'collect()). Please make sure `collect` and `apply` are applied '
              f'to the same function.')

        # We use assertion here since DNA is validated with `self.dna_spec`.
        # User errors should be caught by `dna.use_spec`.
        decision = decision_dict[name]
        used_decision_names.add(name)
        if (not isinstance(hyper_primitive, Choices)
            or hyper_primitive.num_choices == 1):
          return decision
        assert isinstance(decision, list), (hyper_primitive, decision)
        assert len(decision) == hyper_primitive.num_choices, (
            hyper_primitive, decision)
        return decision[sub_index]

      def err_on_unused_decisions():
        if len(used_decision_names) != len(decision_dict):
          remaining = {k: v for k, v in decision_dict.items()
                       if k not in used_decision_names}
          raise ValueError(
              f'Found extra decision values that are not used. {remaining!r}')
      return get_decision_from_dict, err_on_unused_decisions
    else:
      if isinstance(decisions, geno.DNA):
        decision_list = decisions.to_numbers()
      else:
        decision_list = decisions
      value_context = dict(pos=0, value_cache={})

      def get_decision_by_position(
          hyper_primitive, sub_index: Optional[int] = None
          ) -> Union[int, float, str]:
        if sub_index is None or hyper_primitive.name is None:
          name = hyper_primitive.name
        else:
          name = f'{hyper_primitive.name}:{sub_index}'
        if name is None or name not in value_context['value_cache']:
          if value_context['pos'] >= len(decision_list):
            raise ValueError(
                f'No decision is provided for {hyper_primitive!r}.')
          decision = decision_list[value_context['pos']]
          value_context['pos'] += 1
          if name is not None:
            value_context['value_cache'][name] = decision
        else:
          decision = value_context['value_cache'][name]

        if (isinstance(hyper_primitive, Float)
            and not isinstance(decision, float)):
          raise ValueError(
              f'Expect float-type decision for {hyper_primitive!r}, '
              f'encoutered {decision!r}.')
        if (isinstance(hyper_primitive, CustomHyper)
            and not isinstance(decision, str)):
          raise ValueError(
              f'Expect string-type decision for {hyper_primitive!r}, '
              f'encountered {decision!r}.')
        if (isinstance(hyper_primitive, Choices)
            and not (isinstance(decision, int)
                     and decision < len(hyper_primitive.candidates))):
          raise ValueError(
              f'Expect int-type decision in range '
              f'[0, {len(hyper_primitive.candidates)}) for choice {sub_index} '
              f'of {hyper_primitive!r}, encountered {decision!r}.')
        return decision

      def err_on_unused_decisions():
        if value_context['pos'] != len(decision_list):
          remaining = decision_list[value_context['pos']:]
          raise ValueError(
              f'Found extra decision values that are not used: {remaining!r}')
      return get_decision_by_position, err_on_unused_decisions

  def apply(
      self, decisions: Union[geno.DNA, List[Union[int, float, str]]]):
    """Context manager for applying decisions.

      Example::

        def fun():
          return pg.oneof([1, 2, 3]) + pg.oneof([4, 5, 6])

        context = DynamicEvaluationContext()
        with context.collect():
          fun()

        with context.apply([0, 1]):
          # Will print 6 (1 + 5).
          print(fun())

    Args:
      decisions: A DNA or a list of numbers or strings as decisions for currrent
        search space.

    Returns:
      Context manager for applying decisions to the function that defines the
      search space.
    """
    if not isinstance(decisions, (geno.DNA, list)):
      raise ValueError('`decisions` should be a DNA or a list of numbers.')

    get_decision, evaluation_finalizer = (
        self._decision_getter_and_evaluation_finalizer(decisions))

    def _apply_child(c):
      if isinstance(c, types.LambdaType):
        s = schema.get_signature(c)
        if not s.args and not s.has_wildcard_args:
          return c()
      return c

    def _apply_decision(hyper_primitive: HyperPrimitive):
      """Apply a decision value to an hyper_primitive object."""
      if self._where and not self._where(hyper_primitive):
        # Skip hyper primitives that do not pass the `where` predicate.
        return hyper_primitive

      if isinstance(hyper_primitive, Float):
        return get_decision(hyper_primitive)

      if isinstance(hyper_primitive, CustomHyper):
        return hyper_primitive.decode(geno.DNA(get_decision(hyper_primitive)))

      assert isinstance(hyper_primitive, Choices)
      value = symbolic.List()
      for i in range(hyper_primitive.num_choices):
        # NOTE(daiyip): during registering the hyper primitives when
        # constructing the search space, we will need to evaluate every
        # candidate in order to pick up sub search spaces correctly, which is
        # not necessary for `pg.DynamicEvaluationContext.apply`.
        value.append(_apply_child(
            hyper_primitive.candidates[get_decision(hyper_primitive, i)]))
      if isinstance(hyper_primitive, ChoiceValue):
        assert len(value) == 1
        value = value[0]
      return value
    return dynamic_evaluate(
        _apply_decision,
        exit_fn=evaluation_finalizer,
        per_thread=self._per_thread)


def trace(
    fun: Callable[[], Any],
    require_hyper_name: bool = False,
    per_thread: bool = True
    ) -> DynamicEvaluationContext:
  """Trace the hyper primitives called within a function by executing it.

  See examples in :class:`pyglove.hyper.DynamicEvaluationContext`.

  Args:
    fun: Function in which the search space is defined.
    require_hyper_name: If True, all hyper primitives defined in this scope
      will need to carry their names, which is usually a good idea when the
      function that instantiates the hyper primtives need to be called multiple
      times.
    per_thread: If True, the context manager will be applied to current thread
      only. Otherwise, it will be applied on current process.

  Returns:
      An DynamicEvaluationContext that can be passed to `pg.sample`.
  """
  context = DynamicEvaluationContext(
      require_hyper_name=require_hyper_name, per_thread=per_thread)
  with context.collect():
    fun()
  return context

