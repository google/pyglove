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
"""Program genome and genotypes.

Program genome (DNA) is a representation for encoding actions for manipulating
symbolic objects. Genotypes (:class:`pyglove.DNASpec`) are the specification on
how to generate them. Genotypes are separated from their corresponding hyper
values (:class:`pyglove.HyperValue`) which generate client-side objects, in the
aim to decouple the algorithms that generate genomes from the ones that consume
them. As a result, the algorithms can be applied on different user programs
for optimization.

.. graphviz::
   :align: center

    digraph genotypes {
      node [shape="box"];
      edge [arrowtail="empty" arrowhead="none" dir="back" style="dashed"];
      dna_spec [label="DNASpec" href="dna_spec.html"]
      space [label="Space" href="space_class.html"];
      dp [label="DecisionPoint" href="decision_point.html"];
      choices [label="Choices" href="choices.html"];
      float [label="Float" href="float.html"];
      custom [label="CustomDecisionPoint" href="custom_decision_point.html"];
      dna [label="DNA", href="dna.html"]
      dna_spec -> space;
      dna_spec -> dp;
      space -> dp [arrowtail="diamond" style="none" label="elements"];
      dp -> choices;
      choices -> space [arrowtail="diamond" style="none" label="candidates"];
      dp -> float;
      dp -> custom;
      dna -> dna [arrowtail="diamond" style="none" label="children"];
      dna -> dna_spec [arrowhead="normal" dir="forward" style="none"
                       label="spec"];
    }

Genotypes map 1:1 to hyper primitives as the following:

+----------------------------------------+-------------------------------------+
| Genotype class                         | Hyper class                         |
+========================================+=====================================+
|:class:`pyglove.DNASpec`                |:class:`pyglove.hyper.HyperValue`    |
+----------------------------------------+-------------------------------------+
|:class:`pyglove.geno.Space`             |:class:`pyglove.hyper.ObjectTemplate`|
+----------------------------------------+-------------------------------------+
|:class:`pyglove.geno.DecisionPoint`     |:class:`pyglove.hyper.HyperPrimitive`|
+----------------------------------------+-------------------------------------+
|:class:`pyglove.geno.Choices`           |:class:`pyglove.hyper.Choices`       |
+----------------------------------------+-------------------------------------+
|:class:`pyglove.geno.Float`             |:class:`pyglove.hyper.Float`         |
+----------------------------------------+-------------------------------------+
|:class:`pyglove.geno.CustomDecisionPoint` :class:`pyglove.hyper.CustomHyper`  |
+-----------------------------------------+------------------------------------+

"""

import abc
import functools
import random
import re
import types
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Text, Tuple, Union

from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as schema


# Disable implicit str concat in Tuple as it's used for multi-line docstr for
# symbolic members.
# pylint: disable=implicit-str-concat


_CHOICE_REGEX = re.compile(r'^(\d+)/(\d+)$')
_CHOICE_AND_LITERAL_REGEX = re.compile(r'^(\d+)/(\d+) \((.*)\)$', re.S)


class AttributeDict(dict):
  """A dictionary that allows attribute access."""

  def __getattr__(self, key):
    """Enable read access on attribute."""
    return self[key]

  def __setattr__(self, key: Any, value: Any) -> None:
    """Enable write access on attribute."""
    self[key] = value


@symbolic.members([
    ('location',
     schema.Object(object_utils.KeyPath, default=object_utils.KeyPath()),
     'KeyPath of associated genetic encoder relative to parent object template.'
     'This allows DNA generator to apply rule based on locations.'),
    ('hints', schema.Any(default=None), 'Hints for DNA generator to consume.')
])
class DNASpec(symbolic.Object):
  """Base class for DNA specifications (genotypes).

  A DNASpec object describes the rules and tips for generating a DNA.

    * :class:`pyglove.geno.Space`: Represents a space or sub-space, which
      contains a list of decision points.
    * :class:`pyglove.geno.DecisionPoint`: Represents a concrete decision
      point.

  Concrete decision points are the following:

    * :class:`pyglove.geno.Choices`: Represents a single categorical
      choice or multiple related categorical choices. Each candidate
      is a sub-space.
    * :class:`pyglove.geno.Float`: Represents a continuous float value
      within a range.
    * :class:`pyglove.geno.CustomDecisionPoint`: Represents the
      genotype for a :class:`pyglove.hyper.CustomHyper`.

  All `DecisionPoints` provide `hints` for DNA generator as a reference
  when generating values, it needs to be serializable to pass between client
  and servers.

  All DNASpec types allow user to attach their data via `set_userdata`
  method, it's aimed to be used within the same process, thus not required to
  be serializable.
  """
  # NOTE(daiyip): we disable the symbolic comparison to allow hashing DNASpec
  # by object ID, therefore we can use DNASpec objects as the keys for a dict.
  # This is helpful when we want to align decision points using DNASpec as
  # dictionary key. Users can use `pg.eq`/`pg.ne` for symbolic comparisons
  # and `pg.hash` for symbolic hashing.
  allow_symbolic_comparison = False

  def _on_bound(self):
    """Event that is triggered when object is modified."""
    super()._on_bound()
    self._id = None
    self._named_decision_points_cache = None
    self._decision_point_by_id_cache = None
    self._userdata = AttributeDict()

  def _on_path_change(self, old_path, new_path):
    """Event that is triggered when path changes."""
    super()._on_path_change(old_path, new_path)
    # We invalidate the ID cache and decision_point_by_id cache
    # when the ancestor hierarchy changes, which will force the ID and
    # the cache to be recomputed upon usage.
    self._id = None
    self._decision_point_by_id_cache = None

  @property
  def _decision_point_by_id(self):
    """Returns lazy-loaded ID to decision point mapping."""
    if self._decision_point_by_id_cache is None:
      cache = {}
      for dp in self.decision_points:
        if isinstance(dp, Choices) and dp.is_subchoice:
          parent_key = dp.parent_spec.id
          parent_value = cache.get(parent_key, None)
          if parent_value is None:
            parent_value = []
            cache[parent_key] = parent_value
          parent_value.append(dp)
        else:
          cache[dp.id] = dp
      self._decision_point_by_id_cache = cache
    return self._decision_point_by_id_cache

  @property
  def _named_decision_points(self):
    """Return lazy-loaded named decision points."""
    if self._named_decision_points_cache is None:
      named_decision_points = {}
      for dp in self.decision_points:
        if dp.name is not None:
          v = named_decision_points.get(dp.name, None)
          if v is None:
            named_decision_points[dp.name] = dp
          elif isinstance(v, list):
            v.append(dp)
          else:
            named_decision_points[dp.name] = [v, dp]
      self._named_decision_points_cache = named_decision_points
    return self._named_decision_points_cache

  @abc.abstractmethod
  def validate(self, dna: 'DNA') -> bool:
    """Validate whether a DNA value conforms to this spec."""

  @property
  @abc.abstractmethod
  def decision_points(self) -> List['DecisionPoint']:
    """Returns all decision points in their declaration order."""

  @property
  def decision_ids(self) -> List[object_utils.KeyPath]:
    """Returns decision IDs."""
    return list(self._decision_point_by_id.keys())

  @property
  def named_decision_points(
      self) -> Dict[Text, Union['DecisionPoint', List['DecisionPoint']]]:
    """Returns all named decision points in their declaration order."""
    return self._named_decision_points

  @property
  @abc.abstractmethod
  def space_size(self) -> int:
    """Returns the size of the search space. Use -1 for infinity."""

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the number of decision points."""

  def first_dna(self, attach_spec: bool = True) -> 'DNA':
    """Returns the first DNA in the spec."""
    return self.next_dna(None, attach_spec)

  @abc.abstractmethod
  def next_dna(self,
               dna: Optional['DNA'] = None,
               attach_spec: bool = True) -> Optional['DNA']:
    """Returns the next DNA in the space represented by this spec.

    Args:
      dna: The DNA whose next will be returned. If None, `next_dna` will return
        the first DNA.
      attach_spec: If True, current spec will be attached to the returned DNA.

    Returns:
      The next DNA or None if there is no next DNA.
    """

  def iter_dna(self, dna: Optional['DNA'] = None, attach_spec: bool = True):
    """Iterate the DNA in the space represented by this spec.

    Args:
      dna: An optional DNA as the start point (exclusive) for iteration.
      attach_spec: If True, the DNASpec will be attached to the DNA returned.

    Yields:
      The next DNA according to the spec.
    """
    while True:
      dna = self.next_dna(dna, attach_spec)
      if dna is None:
        break
      yield dna

  @property
  def parent_spec(self) -> Optional['DNASpec']:
    """Returns parent spec. None if spec is root."""
    if self.sym_parent is None:
      return None
    # NOTE(daiyip):
    # For child specs of Space, `self.sym_parent` points to `Space.elements`.
    # For child specs of Choices, `self.sym_parent` points to
    #   `Choices.candidates` or `Choices._subchoice_specs`.
    assert self.sym_parent.sym_parent is not None
    return self.sym_parent.sym_parent  # pytype: disable=bad-return-type

  @property
  def parent_choice(self) -> Optional['Choices']:
    """Returns the parent choice of current space."""
    if isinstance(self, Space):
      return self.parent_spec
    return self.parent_spec.parent_choice

  @property
  def id(self) -> object_utils.KeyPath:
    """Returns a path of locations from the root as the ID for current node."""
    if self._id is None:
      parent = self.parent_spec
      if parent is None:
        self._id = self.location
      elif isinstance(self, Space):
        assert isinstance(parent, Choices)
        assert self.index is not None
        self._id = object_utils.KeyPath(
            ConditionalKey(self.index, len(parent.candidates)),
            parent.id) + self.location
      else:
        # Float() or a multi-choice spec of a parent Choice.
        self._id = parent.id + self.location
    return self._id

  def get(self,
          name_or_id: Union[object_utils.KeyPath, Text],
          default: Any = None
          ) -> Union['DecisionPoint', List['DecisionPoint']]:
    """Get decision point(s) by name or ID."""
    try:
      return self[name_or_id]
    except KeyError:
      return default

  def __getitem__(
      self,
      name_or_id: Union[object_utils.KeyPath, Text]
      ) -> Union['DecisionPoint', List['DecisionPoint']]:
    """Get decision point(s) by name or ID ."""
    v = self._named_decision_points.get(name_or_id, None)
    if v is None:
      v = self._decision_point_by_id[name_or_id]
    return v

  def set_userdata(self, key: Text, value: Any) -> None:
    """Sets user data.

    User data can be used for storing state associated with the DNASpec, and
    is not persisted across processes or during serialization. Use `hints` to
    carry persistent objects for the DNASpec.

    Args:
      key: Key of the user data.
      value: Value of the user data.
    """
    self._userdata[key] = value

  @property
  def userdata(self) -> AttributeDict:
    """Gets user data."""
    return self._userdata

  def __str__(self):
    """Operator str."""
    return self.format(
        compact=True,
        verbose=False,
        hide_default_values=True,
        hide_missing_values=True)

  @classmethod
  def from_json(cls, json_value, *args, **kwargs) -> symbolic.Object:
    """Override from_json for backward compatibility with serialized data."""
    assert isinstance(json_value, dict)
    json_value.pop('userdata', None)
    return super().from_json(json_value, *args, **kwargs)


@functools.total_ordering
class DNA(symbolic.Object):
  """A tree of numbers that encodes an symbolic object.

  Each DNA object represents a node in a tree, which has a value and a list of
  DNA as its children. DNA value can be None, int or float, with valid form as
  following:

    +-----------------------+-----------------+-----------------------------+
    |  Encoder type         | Possible values |   Child nodes               |
    |  (DNASpec type)       |                 |                             |
    +=======================+=================+=============================+
    |hyper.ObjectTemplate   |None             |DNA of child decision points |
    |(geno.Space)           |(elements > 1)   |(Choices/Float) in the       |
    |                       |                 |template.                    |
    +-----------------------+-----------------+-----------------------------+
    |                       |None             |Children of elements[0]      |
    |                       |(elements == 1   |                             |
    |                       |and elements[0]. |                             |
    |                       |num_choices > 1) |                             |
    +-----------------------+-----------------+-----------------------------+
    |                       |int              |Children of:                 |
    |                       |(elements == 1   |elements[0][0]               |
    |                       |and elements[0]. |                             |
    |                       |num_choices ==1) |                             |
    +-----------------------+-----------------+-----------------------------+
    |                       |float            |Empty                        |
    |                       |(elements == 1   |                             |
    |                       |and elements[0]  |                             |
    |                       |is geno.Float    |                             |
    +-----------------------+-----------------+-----------------------------+
    |hyper.OneOf            |int              |Children of Space            |
    |(geno.Choices)         |(candidate index |for the chosen candidate     |
    |                       |as choice)       |                             |
    +-----------------------+-----------------+-----------------------------+
    |hyper.ManyOf           |None             |DNA of each chosen candidate |
    |(geno.Choices)         |(num_choices > 1 |                             |
    +-----------------------+-----------------+-----------------------------+
    |                       |int              |Children of chosen candidate |
    |                       |(num_choices==1) |                             |
    +-----------------------+-----------------+-----------------------------+
    |hyper.Float            |float            |Empty                        |
    |(geno.Float)           |                 |                             |
    +-----------------------+-----------------+-----------------------------+
    |hyper.CustomHyper      |string           |User defined.                |
    |(geno.CustomDecision   |                 |                             |
    |Point)                 |                 |                             |
    +-----------------------+-----------------+-----------------------------+

  DNA can also be represented as a mix of JSON number, list and tuples for a
  more intuitive illustration, formally defined as::

    <dna> := empty | <decision>
    <decision>: = <single-decision>
               | <multi-decisions>
               | <conditional-choice>
               | <custom-decision>
    <single-decision> := <categorical-decision>
                      | <float-decision>
                      | <custom-decision>
    <categorical-decision> := int
    <float-decision> := float
    <custom-decision> := str
    <multiple-decisions> := [<decision>, <decision>, ...]
    <conditional-choice> := (<categorical-decision>,
                             <categorical-decision>,
                             ...
                             <decision>)

  Thus DNA can be constructed by nested structures of list, tuple and numbers.
  The numeric value for DNA can be integer (as index of choice) or float (the
  value itself will used as decoded value).

  Examples::

    # Empty DNA. This may be generated by an empty template.
    DNA()

    # A DNA of a nested choice of depth 3.
    DNA(0, 0, 0)

    # A DNA of three choices at the same level,
    # positioned at 0, 1, 2, each choice has value 0.
    DNA([0, 0, 0])

    # A DNA of two choices (one two-level conditional,
    # one non-conditional), position 0 is with choice path: 0 -> [0, 1],
    # while [0, 1] means it's a multi-choice, decodable by Sublist or Subset.
    # position 1 is a single non-conditional choice: 0.
    DNA([(0, [0, 1]), 0])

    # A DNA with custom decision point whose encoding
    # is defined by the user.
    DNA('abc')
  """

  # Allow assignment on symbolic attributes.
  allow_symbolic_assignment = True

  def __init__(
      self,
      value: Union[None, int, float, Text, List[Any], Tuple[Any]] = None,
      # Set MISSING_VALUE to use default from schema.
      children: Optional[List['DNA']] = None,
      spec: Optional[DNASpec] = None,
      metadata: Optional[Dict[Text, Any]] = None,
      *,
      allow_partial: bool = False,
      **kwargs):
    """Constructor.

    Args:
      value: Value for current node.
      children: Child DNA(s).
      spec: DNA spec that constraint current node.
      metadata: Optional dict as controller metadata for the DNA.
      allow_partial: If True, allow the object to be partial.
      **kwargs: keyword arguments that will be passed through to
        symbolic.Object.
    """
    value, children, metadata = self._parse_value_and_children(
        value, children, metadata, spec)
    super().__init__(
        value=value,
        children=children,
        metadata=metadata or symbolic.Dict(),
        allow_partial=allow_partial,
        **kwargs)

    self._decision_by_id_cache = None
    self._named_decisions = None
    self._userdata = AttributeDict()
    self._cloneable_metadata_keys = set()
    self._cloneable_userdata_keys = set()
    self._spec = None
    if spec:
      self.use_spec(spec)

  def _on_bound(self):
    """Event that is triggered when any symbolic member changes."""
    super()._on_bound()
    self._decision_by_id_cache = None
    self._named_decisions = None

  def _parse_value_and_children(
      self,
      value: Union[
          int,         # As a single chosen index.
          float,       # As a single chosen value.
          Text,        # As a custom genome.
          List[Any],   # As multi-choice. (coexisting)
          Tuple[Any],  # As a conditional choice.
          None],
      children: Optional[List['DNA']],
      metadata: Optional[Dict[Text, Any]],
      dna_spec: Optional[DNASpec]
      ) -> Tuple[Union[int, None, float], Optional[List['DNA']]]:
    """Parse value (leaf) and children from maybe compositional value."""
    if isinstance(value, (list, tuple)):
      # The value is compositional, therefore we need to parse the decision
      # for current node and construct the children.
      if children is not None:
        raise ValueError(
            f'\'children\' ({children!r}) must be None when '
            f'\'value\' ({value!r}) is compositional.')
      new_value = None
      children = []
      if isinstance(value, list):
        # Space or multi-choices
        children = [DNA(v) for v in value]
      else:
        # Conditional choices.
        if len(value) < 2:
          raise ValueError(
              f'Tuple as conditional choices must have at least 2 '
              f'items. Encountered {value}.')
        if isinstance(value[0], (float, int)):
          new_value = value[0]
          if len(value) == 2:
            if isinstance(value[1], list):
              # NOTE(daiyip): Multi-choice is allowed only as a leaf choice.
              children = [DNA(v) for v in value[1]]
            elif isinstance(value[1], (int, float, str)):
              children = [DNA(value[1])]
          else:
            children.append(DNA(value[1:]))
        else:
          raise ValueError(
              f'Tuple as conditional choices only allow multiple '
              f'choices to be used at the last position. '
              f'Encountered: {value}')
    else:
      # Normalize DNA by removing trivial intermediate nodes,
      # which is DNA with empty value and only one child.
      # NOTE(daiyip): during deserialization (from_json) of nested DNA,
      # the elements in children might be dicts that are not yet converted
      # to DNA. Therefore, we always call `symbolic.from_json` on children,
      # which is a no-op for already initialized child DNA.
      new_value = value
      children = symbolic.from_json(children) or []
      if len(children) == 1 and children[0].value is None:
        children = children[0].children

    if new_value is None and len(children) == 1:
      c = children[0]
      new_value, children, metadata = c.value, c.children, c.metadata
    return new_value, children, metadata

  def set_metadata(
      self, key: Text, value: Any, cloneable: bool = False) -> 'DNA':
    """Set metadata associated with a key.

    Metadata associated with the DNA will be persisted and carried over across
    processes, which is different the `userdata`. (See `set_userdata` for more
    details.)

    Args:
      key: Key for the metadata.
      value: Value for the metadata.
      cloneable: If True, the key/value will be propagated during clone.

    Returns:
      Self.
    """
    self.metadata.rebind(
        {key: value}, raise_on_no_change=False, skip_notification=True)
    if cloneable:
      self._cloneable_metadata_keys.add(key)
    return self

  def set_userdata(
      self, key: Text, value: Any, cloneable: bool = False) -> 'DNA':
    """Sets user data associated with a key.

    User data associated with the DNA will live only within current process,
    and is not carried over during serialization/deserialization, which is
    different from DNA metadata. (See `set_metadata` for more details.)

    Args:
      key: Key of the user data.
      value: Value of the user data.
      cloneable: If True, the key/value will be carry over to the cloned DNA.

    Returns:
      Self.
    """
    self._userdata[key] = value
    if cloneable:
      self._cloneable_userdata_keys.add(key)
    return self

  @property
  def userdata(self) -> AttributeDict:
    """Gets user data."""
    return self._userdata

  def _ensure_dna_spec(self) -> None:
    """Raises error if current DNA is not bound with a DNASpec."""
    if self._spec is None:
      raise ValueError(f'{self!r} is not bound with a DNASpec.')

  @property
  def spec(self) -> Optional[DNASpec]:
    """Returns DNA spec of current DNA."""
    return self._spec

  @property
  def parent_dna(self) -> Optional['DNA']:
    """Returns parent DNA."""
    if self.sym_parent is None:
      return None
    # NOTE(daiyip): `self.sym_parent` is the `children` field of parent DNA,
    # its presence should always align with parent DNA.
    parent = self.sym_parent.sym_parent
    assert parent is not None
    return parent

  @property
  def root(self) -> 'DNA':
    """Returns the DNA root."""
    current = self
    parent = current.parent_dna
    while parent is not None:
      current = parent
      parent = parent.parent_dna
    return current

  @property
  def is_subchoice(self) -> bool:
    """Returns True if current DNA is a subchoice of a multi-choice."""
    self._ensure_dna_spec()
    return isinstance(self._spec, Choices) and self._spec.is_subchoice

  @property
  def multi_choice_spec(self) -> Optional['Choices']:
    """Returns the multi-choice spec for child DNAs.

    Returns:
      If the children of this DNA are decisions of a multi-choice's subchoices,
      return the multi-choice spec (`pg.geno.Choices`). Otherwise returns None.
    """
    self._ensure_dna_spec()

    multi_choice_spec = None
    if self.children:
      child_spec = self.children[0].spec
      if isinstance(child_spec, Choices) and child_spec.is_subchoice:
        multi_choice_spec = child_spec.parent_spec
    return multi_choice_spec

  @property
  def is_multi_choice_container(self) -> bool:
    """Returns True if the children of this DNA are multi-choice subchoices."""
    return self.multi_choice_spec is not None

  @property
  def literal_value(self) -> Union[Text, int, float,
                                   List[Union[Text, int, float]]]:
    """Returns the literal value represented by current DNA."""
    self._ensure_dna_spec()
    def _literal_value(dna, prefix):
      if dna.children:
        if dna.value is not None:
          prefix += f'{dna.value}/{len(dna.spec.candidates)} -> '
        v = [_literal_value(c, prefix) for c in dna.children]
        return v[0] if len(v) == 1 else v
      if isinstance(dna.spec, Float):
        value = str(dna.value)
      elif isinstance(dna.spec, CustomDecisionPoint):
        value = dna.value
      elif dna.spec.literal_values:
        value = dna.spec.literal_values[dna.value]
      else:
        value = f'{dna.value}/{len(dna.spec.candidates)}'
      if not prefix:
        return value
      return prefix + str(value)
    return _literal_value(self, '')

  @property
  def _decision_by_id(self):
    """Lazy loaded decision by ID dict."""
    if self._decision_by_id_cache is None:
      self._decision_by_id_cache = self.to_dict(
          key_type='id', value_type='dna',
          include_inactive_decisions=True,
          multi_choice_key='both')
    return self._decision_by_id_cache

  @property
  def decision_ids(self) -> List[object_utils.KeyPath]:
    """Returns decision IDs."""
    self._ensure_dna_spec()
    return self._spec.decision_ids

  @property
  def named_decisions(self) -> Dict[Text, Union['DNA', List['DNA']]]:
    """Returns a dict of name to the named DNA in the sub-tree."""
    if self._named_decisions is None:
      named_decisions = {}
      for spec, dna in self.to_dict(
          key_type='dna_spec', value_type='dna',
          multi_choice_key='parent',
          include_inactive_decisions=True).items():
        if spec.name is not None:
          v = named_decisions.get(spec.name, None)
          if v is None:
            v = dna
          else:
            if not isinstance(dna, list):
              dna = [dna]
            if isinstance(v, list):
              v.extend(dna)
            else:
              v = [v] + dna
          named_decisions[spec.name] = v
      self._named_decisions = named_decisions
    return self._named_decisions

  def use_spec(self, spec: DNASpec) -> 'DNA':
    """Use a DNA spec for this node and children recursively.

    Args:
      spec: DNA spec.

    Returns:
      Self.

    Raises:
      ValueError: current DNA tree does not conform to the DNA spec.
    """
    if spec is None:
      raise ValueError('Argument \'spec\' must not be None.')

    if self._spec is spec:
      return self

    def _use_spec_for_child_choices(spec, children):
      """Use spec for child choices."""
      assert isinstance(spec, Choices)
      if spec.num_choices != len(children):
        raise ValueError(
            f'Number of choices ({spec.num_choices}) does not match with '
            f'the number of child values (len(children)). '
            f'Spec: {spec!r}, Children: {children!r}.')

      for i, child in enumerate(children):
        choice_spec = spec.choice_spec(i)
        child.use_spec(choice_spec)

      child_values = [c.value for c in children]
      if spec.sorted and sorted(child_values) != child_values:
        raise ValueError(
            f'Child values {child_values!r} are not sorted. Spec: {spec!r}.')
      if spec.distinct and len(set(child_values)) != len(child_values):
        raise ValueError(
            f'Child values {child_values!r} are not distinct. Spec: {spec!r}.')

    # Skip dummy DNA specs.
    while isinstance(spec, Space) and len(spec.elements) == 1:
      spec = spec.elements[0]

    if isinstance(spec, Space):
      # Multiple value composition.
      if self.value is not None:
        raise ValueError(
            f'DNA value type mismatch. Value: {self.value}, Spec: {spec!r}.')
      if len(spec.elements) != len(self.children):
        raise ValueError(
            f'Length of DNA child values ({len(self.children)}) is different '
            f'from the number of elements ({len(spec.elements)}) '
            f'in Spec: {spec!r}.')
      for i, elem_spec in enumerate(spec.elements):
        self.children[i].use_spec(elem_spec)
    elif isinstance(spec, Choices):
      if spec.num_choices == 1:
        # Single choice.
        if not isinstance(self.value, int):
          raise ValueError(
              f'DNA value type mismatch. Value: {self.value}, Spec: {spec!r}.')
        if self.value >= len(spec.candidates):
          raise ValueError(
              f'Value of DNA is out of range according to the DNA spec. '
              f'Value: {self.value}, Spec: {spec!r}.')
        chosen_candidate = spec.candidates[self.value]
        assert isinstance(chosen_candidate, Space)

        # Empty template in chosen candidate.
        if not chosen_candidate.elements and self.children:
          raise ValueError(
              f'There is no DNA spec for child DNA values. '
              f'Child values: {self.children}.')

        # None-empty template in chosen candidate.
        if len(chosen_candidate.elements) > 1:
          # Children are DNA of multiple encoders in chosen composition.
          if len(chosen_candidate.elements) != len(self.children):
            raise ValueError(
                f'Number of elements in child templates '
                f'({len(chosen_candidate.elements)}) does not match with '
                f'the length of children ({len(self.children)}) from DNA: '
                f'{self!r}, Spec: {chosen_candidate}.')
          for i, elem_spec in enumerate(chosen_candidate.elements):
            self.children[i].use_spec(elem_spec)
        elif len(chosen_candidate.elements) == 1:
          # Children are multiple choices of the only encoder
          # in chosen composition.
          sub_spec = chosen_candidate
          while (isinstance(sub_spec, Space) and
                 len(sub_spec.elements) == 1):
            sub_spec = sub_spec.elements[0]

          if isinstance(sub_spec, (Float, CustomDecisionPoint)):
            if len(self.children) != 1:
              raise ValueError(
                  f'Encountered more than 1 value.'
                  f'Child value: {self.children}, Spec: {sub_spec}.')
            self.children[0].use_spec(sub_spec)
          else:
            assert isinstance(sub_spec, Choices)
            _use_spec_for_child_choices(sub_spec, self.children)
      else:
        # Multiple choices.
        if self.value is not None:
          raise ValueError(
              f'Cannot apply multi-choice DNA spec on '
              f'value {self.value}: {spec!r}.')
        _use_spec_for_child_choices(spec, self.children)
    elif isinstance(spec, Float):
      if not isinstance(self.value, float):
        raise ValueError(
            f'DNA value type mismatch. Value: {self.value}, '
            f'Spec: {spec!r}.')
      if self.value < spec.min_value:
        raise ValueError(
            f'DNA value should be no less than {spec.min_value}. '
            f'Encountered {self.value}, Spec: {spec!r}.')
      if self.value > spec.max_value:
        raise ValueError(
            f'DNA value should be no greater than {spec.max_value}. '
            f'Encountered {self.value}, Spec: {spec!r}.')
    elif isinstance(spec, CustomDecisionPoint):
      if not isinstance(self.value, str):
        raise ValueError(
            f'DNA value type mismatch, Value: {self.value!r}, Spec: {spec!r}.')
    else:
      raise ValueError(f'Unsupported spec: {spec!r}.')
    self._spec = spec
    return self

  @classmethod
  def parse(
      cls,
      json_value: Union[int,         # As a single chosen index.
                        float,       # As a single chosen value.
                        str,         # As a custom genome.
                        List[Any],   # As multi-choice. (coexisting)
                        Tuple[Any],  # As a conditional choice.
                        None         # An empty DNA.
                       ],
      spec: Optional[DNASpec] = None) -> 'DNA':
    """Parse DNA from a nested structure of numbers.

    Deprecated: use `DNA.__init__` instead.

    Args:
      json_value: A nested structure of numbers.
      spec: DNA spec that will be applied to current DNA tree.

    Returns:
      an instance of DNA object.

    Raises:
      ValueError: Bad format for json_value or parsed DNA does not conform to
        the DNA spec.
    """
    return DNA(json_value, spec=spec)

  @classmethod
  def from_dict(
      cls,
      dict_repr: Dict[Union['DecisionPoint', Text],
                      Union[None, 'DNA', float, int, Text]],
      dna_spec: DNASpec,
      use_ints_as_literals: bool = False) -> 'DNA':
    """Create a DNA from its dictionary representation.

    Args:
      dict_repr: The dictionary representation of the DNA.
        The keys should be either strings as the decision point ID
        or DNASpec objects. The values should be either numeric or literal
        values for the decisions.
        For inactive decisions, their ID/spec should either be absent from the
        dictionary, or use None as their values.
      dna_spec: The DNASpec that applies to the DNA.
      use_ints_as_literals: If True, when an integer is encountered for
        a dictinonary value, treat it as the literal value.
        Otherwise, always treat it as a candidate index.

    Returns:
      A DNA object.
    """
    def _get_decision(spec: DNASpec):
      """Gets the decision for DNASpec."""
      decision = dict_repr.get(spec.id, None)
      if decision is None:
        decision = dict_repr.get(spec, None)

      if decision is None and spec.name:
        decision = dict_repr.get(spec.name, None)
        # A spec can result in multiple decision points (e.g. multi-choices)
        # therefore we always pop up the next single decision if a name
        # is associated with multiple decisions.
        if isinstance(decision, list):
          dict_repr[spec.name] = decision[1:]
          decision = decision[0] if decision else None
      return decision

    def _choice_index(choice_spec: Choices,
                      value: Union[int, float, Text]) -> int:
      """Gets the index of choice value based on its spec."""
      if isinstance(value, int) and not use_ints_as_literals:
        index = value
        if index < 0 or index > len(choice_spec.candidates):
          identifier = choice_spec.name or choice_spec.id
          raise ValueError(
              f'Candidate index out of range at choice \'{identifier}\'. Index='
              f'{index}, Number of candidates={len(choice_spec.candidates)}.')
      else:
        index = choice_spec.candidate_index(value)
      return index

    def _make_dna(spec: DNASpec) -> DNA:
      """Lookup DNA value from parameter values according to DNA spec."""
      if isinstance(spec, Space):
        children = []
        for elem in spec.elements:
          child = _make_dna(elem)
          if child is not None:
            children.append(child)
        return DNA(None, children)
      else:
        if isinstance(spec, Choices):
          children = []
          for choice_idx in range(spec.num_choices):
            choice_spec = spec.choice_spec(choice_idx)
            value = _get_decision(choice_spec)

            # It's possible that the decisions for multiple choices are
            # collapsed as a single entry in the dictionary. e.g: {'x': [0, 1]}.
            # In this case, we will make another attempt to get the decision
            # from the parent spec entry.
            if value is None and choice_spec.is_subchoice:
              parent_decisions = _get_decision(spec)
              if parent_decisions is not None:
                assert len(parent_decisions) == spec.num_choices, (
                    parent_decisions, spec)
                value = parent_decisions[choice_idx]

            if value is None:
              identifier = choice_spec.name or choice_spec.id
              raise ValueError(
                  f'Value for \'{identifier}\' is not found in '
                  f'the dictionary {dict_repr!r}.')

            if isinstance(value, DNA):
              children.append(value)
            else:
              choice_index = _choice_index(choice_spec, value)
              subspace_dna = _make_dna(choice_spec.candidates[choice_index])
              children.append(
                  DNA(choice_index, [subspace_dna] if subspace_dna else []))
          return DNA(None, children)
        elif isinstance(spec, (Float, CustomDecisionPoint)):
          value = _get_decision(spec)
          if value is None:
            raise ValueError(
                f'Value for \'{spec.name or spec.id}\' is not found '
                f'in the dictionary {dict_repr!r}.')
          if isinstance(value, DNA):
            value = value.value
          if isinstance(spec, Float):
            if value < spec.min_value:
              raise ValueError(
                  f'The decision for \'{spec.name or spec.id}\' should '
                  f'be no less than {spec.min_value}. Encountered {value}.')
            if value > spec.max_value:
              raise ValueError(
                  f'The decision for \'{spec.name or spec.id}\' should '
                  f'be no greater than {spec.max_value}. Encountered {value}.')
          else:
            if not isinstance(value, str):
              raise ValueError(
                  f'The decision for \'{spec.name or spec.id}\' should '
                  f'be a string. Encountered {value}.')
          return DNA(value, None)
        else:
          raise NotImplementedError('Should never happen.')

    dna = _make_dna(dna_spec)
    return dna.use_spec(dna_spec)

  def to_dict(
      self,
      key_type='id',
      value_type='value',
      multi_choice_key='subchoice',
      include_inactive_decisions=False,
      filter_fn: Optional[Callable[['DecisionPoint'], bool]] = None
      ) -> Dict[Union['DecisionPoint', Text],
                Union[None, 'DNA', float, int, Text,
                      List['DNA'], List[int], List[Text]]]:
    """Returns the dict representation of current DNA.

    Args:
      key_type: Key type in returned dictionary. Acceptable values are:

        * 'id': Use the ID (canonical location) of each decision point as key.
          This is the default behavior.
        * 'name_or_id': Use the name of each decision point as key if it's
          present, otherwise use ID as key. When the name of a decision
          point is presented, it is guaranteed not to clash with other
          decision points' names or IDs.
        * 'dna_spec': Use the DNASpec object of each decision point as key.
      value_type: Value type for choices in returned dictionary.
        Acceptable values are:

        * 'value': Use the index of the chosen candidate for `Choices`, and
          use the float number for `Float`. This is the default behavior.
        * 'dna': Use `DNA` for all decision points.
        * 'choice': Use '{index}/{num_candidates}' for the chosen candidate
          for `Choices`, and the chosen float number for `Float`.
        * 'literal': Use the literal value for the chosen candidate for
          `Choices`, and the chosen float number for `Float`. If the literal
          value for the `Choices` decision point is not present, fall back
          to the '{index}/{num_candidates}' format.
        * 'choice_and_literal': Use '{index}/{num_candidates} ({literal})'
          for the chosen candidate for `Choices` and then chosen float number
          for `Float`. If the literal value for the `Choices` decision point
          is not present, fall back to the '{index}/{num_candidates}' format.

      multi_choice_key: 'subchoice', 'parent', or 'both'. If 'subchoice', each
        subchoice will insert a key into the dict. If 'parent', subchoices of
        a multi-choice will share the parent spec as key, its value will be
        a list of decisions from the subchoices. If 'both', the dict will
        contain both the keys for subchoices and the key for the parent
        multi-choice.
      include_inactive_decisions: If True, inactive decisions from the search
        space will be added to the dict with value None. Otherwise they will
        be absent in the dict.
      filter_fn: Decision point filter. If None, all the decision points will be
        included in the dict. Otherwise only the decision points that pass
        the filter (returns True) will be included.

    Returns:
      A dictionary of requested key type to value type mapped from the DNA.

    Raises:
      ValueError: argument `key_type` or `value_type` is not valid.
      RuntimeError: If DNA is not associated with a DNASpec.
    """
    if key_type not in ['id', 'name_or_id', 'dna_spec']:
      raise ValueError(
          f'\'key_type\' must be either \'id\', \'name_or_id\' or \'dna_spec\'. '
          f'Encountered: {key_type!r}.')

    if value_type not in ['dna', 'value', 'choice',
                          'literal', 'choice_and_literal']:
      raise ValueError(
          f'\'value_type\' must be either \'dna\', \'value\', \'choice\' '
          f'\'literal\' or \'choice_and_literal\'. '
          f'Encountered: {value_type!r}.')

    if multi_choice_key not in ['subchoice', 'parent', 'both']:
      raise ValueError(
          f'\'multi_choice_key\' must be either \'subchoice\', \'parent\', or '
          f'\'both\'. Encountered: {multi_choice_key!r}.')

    multi_choice_use_parent_as_key = multi_choice_key != 'subchoice'
    multi_choice_use_subchoice_as_key = multi_choice_key != 'parent'
    filter_fn = filter_fn or (lambda x: True)

    self._ensure_dna_spec()
    dict_repr = dict()

    def _needs_subchoice_key(choice_spec):
      return (multi_choice_use_subchoice_as_key
              and (not multi_choice_use_parent_as_key
                   or (key_type != 'name_or_id' or choice_spec.name is None)))

    def _key(spec: 'DecisionPoint'):
      if key_type == 'id':
        return spec.id.path
      elif key_type == 'name_or_id':
        return spec.name if spec.name else spec.id.path
      else:
        return spec

    def _put(key, value):
      if key in dict_repr:
        accumulated = dict_repr[key]
        if not isinstance(accumulated, list):
          accumulated = [accumulated]
        accumulated.append(value)
        value = accumulated
      dict_repr[key] = value
      return value

    def _dump_node(dna: DNA):
      """Dump node value to dict representation."""
      if isinstance(dna.spec, DecisionPoint) and filter_fn(dna.spec):
        key = _key(dna.spec)
        value = None
        if isinstance(dna.spec, Choices) and dna.value is not None:
          if value_type == 'dna':
            value = dna
          elif value_type == 'value':
            value = dna.value
          else:
            value = dna.spec.format_candidate(
                dna.value, display_format=value_type)

          if dna.spec.is_subchoice:
            # Append multi-choice values into parent's key.
            if multi_choice_use_parent_as_key:
              _put(_key(dna.spec.parent_spec), value)

            # Insert subchoice in its own key.
            if _needs_subchoice_key(dna.spec):
              _put(key, value)
          else:
            _put(key, value)
        elif isinstance(dna.spec, (Float, CustomDecisionPoint)):
          if value_type == 'dna':
            value = dna
          else:
            value = dna.value
          _put(key, value)

      for child_dna in dna.children:
        _dump_node(child_dna)

    _dump_node(self)
    if not include_inactive_decisions:
      return dict_repr

    result = dict()
    for dp in self.spec.decision_points:
      if not filter_fn(dp):
        continue
      if isinstance(dp, Choices) and dp.is_subchoice:
        if multi_choice_use_parent_as_key:
          if dp.subchoice_index == 0:
            k = _key(dp.parent_spec)
            result[k] = dict_repr.get(k, None)
        if _needs_subchoice_key(dp):
          k = _key(dp)
          result[k] = dict_repr.get(k, None)
      else:
        k = _key(dp)
        result[k] = dict_repr.get(k, None)
    return result

  @classmethod
  def from_numbers(
      cls,
      dna_values: List[Union[int, float, Text]],
      dna_spec: DNASpec) -> 'DNA':
    """Create a DNA from a flattened list of dna values.

    Args:
      dna_values: A list of DNA values.
      dna_spec: DNASpec that interprets the dna values.

    Returns:
      A DNA object.
    """
    context = dict(index=0)
    def _next_decision():
      if context['index'] >= len(dna_values):
        raise ValueError(
            f'The input {dna_values!r} is too short for {dna_spec!r}.')
      decision = dna_values[context['index']]
      context['index'] += 1
      return decision

    def _bind_decisions(dna_spec):
      value = None
      children = None
      if isinstance(dna_spec, Space):
        children = [
            _bind_decisions(elem)
            for elem in dna_spec.elements]
      elif isinstance(dna_spec, Choices):
        if dna_spec.num_choices == 1:
          value = _next_decision()
          if value < 0 or value >= len(dna_spec.candidates):
            raise ValueError(
                f'Candidate index out of range at choice '
                f'\'{dna_spec.name or dna_spec.id}\'. Index={value}, '
                f'Number of candidates={len(dna_spec.candidates)}.')
          children = [_bind_decisions(dna_spec.candidates[value])]
        else:
          children = [_bind_decisions(spec) for spec in dna_spec.choice_specs]
      else:
        value = _next_decision()
      return DNA(value, children, spec=dna_spec)
    dna = _bind_decisions(dna_spec)
    if context['index'] != len(dna_values):
      end_pos = context['index']
      raise ValueError(
          f'The input {dna_values!r} is too long for {dna_spec!r}. '
          f'Remaining: {dna_values[end_pos:]!r}.')
    return dna

  def to_numbers(
      self, flatten: bool = True,
      ) -> Union[List[Union[int, float, Text]],
                 object_utils.Nestable[Union[int, float, Text]]]:
    """Returns a (maybe) nested structure of numbers as decisions.

    Args:
      flatten: If True, the hierarchy of the numbers will not be preserved.
        Decisions will be returned as a flat list in DFS order. Otherwise, a
        nestable structure of numbers will be returned.

    Returns:
      A flat list or a hierarchical structure of numbers as the decisions made
        for each decision point.
    """
    if flatten:
      decisions = [self.value] if self.value is not None else []
      for c in self.children:
        decisions.extend(c.to_numbers(flatten))
      return decisions
    else:
      if self.value is None:
        return [c.to_numbers(flatten) for c in self.children]
      elif not self.children:
        return self.value
      elif len(self.children) == 1:
        child = self.children[0].to_numbers(flatten)
        if isinstance(child, tuple):
          return tuple([self.value, list(child)])
        else:
          return (self.value, child)
      else:
        assert len(self.children) > 1
        return (self.value, [c.to_numbers(flatten) for c in self.children])

  @classmethod
  def from_fn(
      cls,
      dna_spec: DNASpec,
      generator_fn: Callable[['DecisionPoint'],
                             Union[List[int], float, str, 'DNA']]
      ) -> 'DNA':
    """Generate a DNA with user generator function.

    Args:
      dna_spec: The DNASpec for the DNA.
      generator_fn: A callable object with signature:

        `(decision_point) -> decision`

        The decision_point is a `Choices` object or a `Float` object.
        The returned decision should be:

         * a list of integer or a DNA object for a `Choices` decision point.
           When a DNA is returned, it will be used as the DNA for the entire
           sub-tree, hence `generate_fn` will not be called on sub-decision
           points.
         * a float or a DNA object for a Float decision point.
         * a string or a DNA object for a CustomDecisionPoint.

    Returns:
      A DNA generated from the user function.
    """
    if isinstance(dna_spec, Space):
      # Generate values for Space.
      children = []
      for child_spec in dna_spec.elements:
        children.append(DNA.from_fn(child_spec, generator_fn))
      if len(children) == 1:
        return children[0]
      dna = DNA(None, children)
    elif isinstance(dna_spec, Choices):
      # Generate values for Choices.
      decision = generator_fn(dna_spec)
      if isinstance(decision, DNA):
        dna = decision
      else:
        if len(decision) != dna_spec.num_choices:
          raise ValueError(
              f'Number of DNA child values does not match the number of '
              f'choices. Child values: {decision!r}, '
              f'Choices: {dna_spec.num_choices}, '
              f'Location: {dna_spec.location.path}.')
        children = []
        for i, choice in enumerate(decision):
          choice_location = object_utils.KeyPath(i, dna_spec.location)
          if not isinstance(choice, int):
            raise ValueError(
                f'Choice value should be int. Encountered: {choice}, '
                f'Location: {choice_location.path}.')
          if choice >= len(dna_spec.candidates):
            raise ValueError(
                f'Choice out of range. Value: {choice}, '
                f'Candidates: {len(dna_spec.candidates)}, '
                f'Location: {choice_location.path}.')
          child_dna = DNA.from_fn(dna_spec.candidates[choice], generator_fn)
          children.append(DNA(choice, [child_dna]))
        dna = DNA(None, children)
    elif isinstance(dna_spec, (Float, CustomDecisionPoint)):
      decision = generator_fn(dna_spec)
      if isinstance(decision, DNA):
        dna = decision
      else:
        dna = DNA(decision)
    else:
      raise TypeError(
          f'Argument \'dna_spec\' should be DNASpec type. '
          f'Encountered {dna_spec}.')
    dna_spec.validate(dna)
    return dna

  def sym_jsonify(
      self,
      compact: bool = True,
      type_info: bool = True,
      **kwargs) -> Any:
    """Convert DNA to JSON object.

    Args:
      compact: Whether use compact form. If compact, the nested number structure
        in DNA.parse will be used, otherwise members will be rendered out as
        regular symbolic Object.
      type_info: If True, type information will be included in output, otherwise
        type information will not be included. Applicable when compact is set
        to True.
      **kwargs: Keyword arguments that will be passed to symbolic.Object if
        compact is False.

    Returns:
      JSON representation of DNA.
    """
    if not compact:
      json_value = super().sym_jsonify(**kwargs)
      assert isinstance(json_value, dict), json_value
      if self._cloneable_metadata_keys:
        json_value['_cloneable_metadata_keys'] = list(
            self._cloneable_metadata_keys)
      return json_value

    if self.children:
      child_nodes = [c.sym_jsonify(compact, type_info=False, **kwargs)
                     for c in self.children]
      if self.value is not None:
        if len(child_nodes) == 1:
          # Chain single choices into one tuple.
          single_choice = child_nodes[0]
          if isinstance(single_choice, tuple):
            value = (self.value,) + single_choice
          else:
            value = (self.value, single_choice)
        else:
          # Put multiple choice as sub-nodes.
          value = (self.value, child_nodes)
      else:
        value = child_nodes
    else:
      value = self.value

    if type_info:
      json_value = {
          symbolic._TYPE_NAME_KEY: self.__class__.type_name,  # pylint: disable=protected-access
          'format': 'compact',
          'value': symbolic.to_json(value),
      }
      # NOTE(daiyip): For now, we only attach metadata from the root node for
      # faster serialization/deserialization speed. This should be revised if
      # metadata for child DNA is used.
      if self.metadata:
        json_value['metadata'] = symbolic.to_json(self.metadata)

      if self._cloneable_metadata_keys:
        json_value['_cloneable_metadata_keys'] = list(
            self._cloneable_metadata_keys)
      return json_value
    else:
      return value

  @classmethod
  def from_json(
      cls,
      json_value: Dict[Text, Any],
      *,
      allow_partial: bool = False,
      root_path: Optional[object_utils.KeyPath] = None) -> 'DNA':
    """Class method that load a DNA from a JSON value.

    Args:
      json_value: Input JSON value, only JSON dict is acceptable.
      allow_partial: Whether to allow elements of the list to be partial.
      root_path: KeyPath of loaded object in its object tree.

    Returns:
      A DNA object.
    """
    cloneable_metadata_keys = json_value.pop('_cloneable_metadata_keys', None)
    if json_value.get('format', None) == 'compact':
      # NOTE(daiyip): DNA.parse will validate the input. Therefore, we can
      # disable runtime type check during constructing the DNA objects.
      with symbolic.enable_type_check(False):
        dna = DNA.parse(symbolic.from_json(json_value.get('value')))
        if 'metadata' in json_value:
          dna.rebind(
              metadata=symbolic.from_json(json_value.get('metadata')),
              raise_on_no_change=False, skip_notification=True)
    else:
      dna = super(DNA, cls).from_json(
          json_value,
          allow_partial=allow_partial,
          root_path=root_path)  # pytype: disable=bad-return-type
      assert isinstance(dna, DNA)
    if cloneable_metadata_keys:
      dna._cloneable_metadata_keys = set(cloneable_metadata_keys)  # pylint: disable=protected-access
    return dna

  @property
  def is_leaf(self) -> bool:
    """Returns whether the current node is a leaf node."""
    return not self.children

  def __getitem__(
      self, key: Union[int, slice, Text, object_utils.KeyPath, 'DecisionPoint']
      ) -> Union[None, 'DNA', List[Optional['DNA']]]:
    """Get an immediate child DNA or DNA in the sub-tree.

    Args:
      key: The key for retrieving the sub-DNA or sub-DNA list. The key should
        be one of:
          1) An integer as the index of an immediate child DNA.
          2) A name (string) for named decisions whose DNASpec has a not-None
             `name` argument.
          3) An ID (string or KeyPath) for the decision point to retrieve.
             See `DNASpec.id` for details.
          4) A DecisionPoint object whose decision value will be retrived.

    Returns:
      The return value should be one of the following:
      1) A DNA object if the key only maps to a single DNA object.
      2) None if the decision point exists but it's inactive.
      3) A list of DNA or None if there are multiple decision points associated
         with the key.
    """
    if isinstance(key, (int, slice)):
      return self.children[key]
    if isinstance(key, DNASpec):
      key = key.id
      return self._decision_by_id[key]
    else:
      v = self.named_decisions.get(key, None)
      if v is None:
        v = self._decision_by_id[key]
      return v

  def get(self,
          key: Union[int, slice, Text, object_utils.KeyPath, 'DecisionPoint'],
          default: Any = None
          ) -> Union[Any, None, 'DNA', List[Optional['DNA']]]:
    """Get an immediate child DNA or DNA in the sub-tree."""
    try:
      return self[key]
    except KeyError:
      return default

  def __iter__(self):
    """Iterate child DNA(s)."""
    return self.children.__iter__()

  def __contains__(self, dna_or_value: Union[int, 'DNA']) -> bool:
    """Returns whether child DNA(s) contains a value."""
    for child in self.children:
      if isinstance(dna_or_value, (int, float, str)):
        if child.value == dna_or_value:
          return True
      elif isinstance(dna_or_value, DNA):
        if child == dna_or_value:
          return True
      else:
        raise ValueError(
            f'DNA.__contains__ does not accept '
            f'{object_utils.quote_if_str(dna_or_value)!r}.')
    return False

  def __hash__(self):
    """Hash code."""
    return hash((self.value, tuple(self.children)))

  def __cmp__(self, other: 'DNA') -> int:
    """DNA comparison."""
    if other is None:
      return 1

    def compare_dna_value(x, y):
      if x == y:
        return 0
      if x is None:
        return -1
      if y is None:
        return 1
      return -1 if x < y else 1

    result = compare_dna_value(self.value, other.value)
    if result != 0:
      return result

    if len(self.children) != len(other.children):
      raise ValueError(
          f'The two input DNA have different number of children. '
          f'(Left={self!r}, Right={other!r})')

    for i, c in enumerate(self.children):
      result = c.__cmp__(other.children[i])
      if result != 0:
        return result
    return 0

  def __eq__(self, other: 'DNA') -> bool:
    if not isinstance(other, DNA):
      return False
    return not self.__cmp__(other)

  def __ne__(self, other: 'DNA') -> bool:
    return not self == other

  def __lt__(self, other: 'DNA') -> bool:
    if not isinstance(other, DNA):
      raise TypeError(f'unorderable types: DNA & {type(other)}')
    return self.__cmp__(other) == -1

  def next_dna(self) -> Optional['DNA']:
    """Get the next DNA in the spec."""
    self._ensure_dna_spec()
    return self.spec.next_dna(self)

  def iter_dna(self):
    """Iterate DNA of the space starting from self."""
    self._ensure_dna_spec()
    return self.spec.iter_dna(self)

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             list_wrap_threshold: int = 80,
             as_dict: bool = False,
             **kwargs):
    """Customize format method for DNA for more compact representation."""
    if as_dict and self.spec:
      details = object_utils.format(
          self.to_dict(value_type='choice_and_literal'),
          False,
          verbose,
          root_indent,
          **kwargs)
      return f'DNA({details})'

    if 'list_wrap_threshold' not in kwargs:
      kwargs['list_wrap_threshold'] = list_wrap_threshold

    if not verbose:
      return super().format(False, verbose, root_indent, **kwargs)

    if self.is_leaf:
      return f'DNA({self.value!r})'

    rep = object_utils.format(
        self.to_json(compact=True, type_info=False),
        compact, verbose, root_indent, **kwargs)
    if rep and rep[0] == '(':
      # NOTE(daiyip): for conditional choice from the root,
      # we don't want to keep duplicate round bracket.
      return f'DNA{rep}'
    return f'DNA({rep})'

  def parameters(
      self, use_literal_values: bool = False) -> Dict[Text, Text]:
    """Returns parameters for this DNA to emit based on its spec.

    Deprecated: use `to_dict` instead.

    Args:
      use_literal_values: If True, use literal values from DNASpec for Choices,
        otherwise returns '{choice}/{num_candidates} ({literal})'. Otherwise
        returns '{choice}/{num_candidates}'.

    Returns:
      Dict of parameter names to their values mapped from this DNA.

    Raises:
      RuntimeError: If DNA is not associated with a DNASpec.
    """
    value_type = 'choice_and_literal' if use_literal_values else 'choice'
    return self.to_dict(value_type=value_type)

  def _sym_clone(self, deep: bool, memo: Any) -> 'DNA':
    """Override to copy DNASpec."""
    other = super()._sym_clone(deep, memo)
    other._spec = self._spec      # pylint: disable=protected-access
    for k, v in self._userdata.items():
      if k in self._cloneable_userdata_keys:
        other._userdata[k] = v    # pylint: disable=protected-access
    other._cloneable_userdata_keys = set(self._cloneable_userdata_keys)  # pylint: disable=protected-access

    # Remove none-clonable meta-data.
    metadata = {}
    for k, v in self.metadata.items():
      if k in self._cloneable_metadata_keys:
        metadata[k] = v
    other.rebind(metadata=metadata)
    other._cloneable_metadata_keys = set(self._cloneable_metadata_keys)  # pylint: disable=protected-access
    return other

  @classmethod
  def from_parameters(cls,
                      parameters: Dict[Text, Any],
                      dna_spec: DNASpec,
                      use_literal_values: bool = False) -> 'DNA':
    """Create DNA from parameters based on DNASpec.

    Deprecated: use `from_dict` instead.

    Args:
      parameters: A 1-depth dict of parameter names to parameter values.
      dna_spec: DNASpec to interpret the parameters.
      use_literal_values: If True, parameter values are literal values from
        DNASpec.

    Returns:
      DNA instance bound with the DNASpec.

    Raises:
      ValueError: If parameters are not aligned with DNA spec.
    """
    del use_literal_values
    return cls.from_dict(parameters, dna_spec)

  def __str__(self) -> Text:
    """Use compact form as string representation."""
    return self.format(compact=True)


# NOTE(daiyip): members is declared separately as a function call since
# decorator requires DNA to be defined before used as a part of value spec
# of 'children' field.
symbolic.members([
    ('value', schema.Union([
        schema.Int(), schema.Float(), schema.Str()]).noneable(),
     'Value of DNA node.'),
    ('children', schema.List(schema.Object(DNA), default=[]),
     'DNA list as child nodes for template members or '
     'chosen candidates of choices.'),
    ('metadata', schema.Dict([
        (schema.StrKey(), schema.Any(), 'Key-value pairs.')
    ]), 'Metadata assigned to the DNA.')
])(DNA)


#
# Implementation of DNASpecs according to to different encoders.
#


@symbolic.members([
    ('name', schema.Str().noneable(),
     'Name of current node. If present, it should be unique in the search '
     'space. We can use `root_spec[name]` to access named DNASpec.'),
])
class DecisionPoint(DNASpec):
  """Base class for decision points.

  Child classes:

    * :class:`pyglove.geno.Choices`
    * :class:`pyglove.geno.Float`
    * :class:`pyglove.geno.CustomDecisionPoint`
  """


@symbolic.members(
    [
        ('elements', schema.List(schema.Object(DecisionPoint), default=[]),
         'Elements of current composition.'),
        ('index', schema.Int(min_value=0).noneable(),
         'Index of the template among the candidates of a parent Choice. '
         'If None, the template is the root template.')
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
        if isinstance(spec, Choices):
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
    if isinstance(dna.value, float):
      raise ValueError(
          f'Unexpected float value in {dna!r}, Location: {self.location.path}.')
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

  def next_dna(self,
               dna: Optional['DNA'] = None,
               attach_spec: bool = True) -> Optional['DNA']:
    """Returns the next DNA."""
    if dna is None:
      return DNA(None, [e.first_dna(attach_spec=False) for e in self.elements],
                 spec=self if attach_spec else None)

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
      return DNA(None, list(reversed(new_children)),
                 spec=self if attach_spec else None)

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    return sum([len(elem) for elem in self.elements])

  def __getitem__(
      self, index: Union[int, slice, Text, object_utils.KeyPath]
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


@symbolic.members(
    [
        ('num_choices', schema.Int(min_value=1, default=1),
         'Number of choices to make.'),
        ('candidates', schema.List(schema.Object(Space), min_size=1),
         'A list of Spaces for every candidate, which may contain '
         'child nodes to form sub (conditional) search spaces.'),
        ('literal_values', schema.List(
            schema.Union([schema.Str(), schema.Int(), schema.Float()]),
            min_size=1).noneable(),
         'Optional literal value for each candidate. Used for debugging.'),
        ('distinct', schema.Bool(True), 'Whether choices are distinct.'),
        ('sorted', schema.Bool(False),
         'Whether choices are sorted. The order key is the index of the '
         'candidate in the `candidates` field instead of the candidate\'s '
         'value.'),
        ('subchoice_index', schema.Int(min_value=0).noneable(),
         'Index of current choice as a subchoice of a parent multi-choice. '
         'If None, current choice is either a single choice or a root spec for '
         'a multi-choice.')
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
      c.rebind(index=i, skip_notification=True)

    # Create sub choice specs and index decision points.
    if self.num_choices > 1 and not self.is_subchoice:
      subchoice_specs = []
      self._decision_points = []
      for i in range(self.num_choices):
        subchoice_spec = Choices(
            subchoice_index=i,
            location=object_utils.KeyPath(i),
            num_choices=1,
            candidates=self.candidates,
            literal_values=self.literal_values,
            distinct=self.distinct,
            sorted=self.sorted,
            name=self.name,
            hints=self.hints)
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
      self, old_path: object_utils.KeyPath, new_path: object_utils.KeyPath):
    """Trigger path change for subchoices so their IDs can be invalidated."""
    super()._update_children_paths(old_path, new_path)
    if self._subchoice_specs:
      for i, spec in enumerate(self._subchoice_specs):
        spec.sym_setpath(new_path + i)

  def choice_spec(self, index: int) -> 'Choices':
    """Returns spec for choice i."""
    if self.is_subchoice:
      raise ValueError(
          f'\'choice_spec\' should not be called on a subchoice of a '
          f'multi-choice. Encountered: {self!r}.')
    if self.num_choices == 1:
      return self
    return self._subchoice_specs[index]

  @property
  def choice_specs(self) -> List['Choices']:
    """Returns all choice specs."""
    return [self.choice_spec(i) for i in range(self.num_choices)]

  @property
  def is_subchoice(self) -> bool:
    """Returns if current choice is a subchoice of a multi-choice."""
    return self.subchoice_index is not None

  def format_candidate(
      self,
      index: int,
      display_format: Text = 'choice_and_literal') -> Union[Text, int, float]:
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

  def candidate_index(self, choice_value: Union[Text, int, float]) -> int:
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
    def _index_from_literal(value: Union[Text, int, float]) -> int:
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
      if not isinstance(dna.value, int):
        raise ValueError(
            f'Expect integer for Choices. Encountered: {dna!r}, '
            f'Location: {self.location.path}.\n'
            f'Did you forget to specify values for conditional choices?')
      if dna.value >= len(self.candidates):
        raise ValueError(
            f'Choice out of range. Value: {dna.value}, '
            f'Candidates: {len(self.candidates)}, '
            f'Location: {self.location.path}.')
      self.candidates[dna.value].validate(DNA(None, dna.children))
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
        sub_location = object_utils.KeyPath(i, self.location)
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
            return size
          else:
            # When choice is neither distinct nor sorted,
            return _space_size(s, 1) ** k
        self._space_size = _space_size(sub_space_sizes, self.num_choices)
    return self._space_size

  def next_dna(self,
               dna: Optional['DNA'] = None,
               attach_spec: bool = True) -> Optional['DNA']:
    """Returns the next DNA."""
    if dna is None:
      choice_dna_list = []
      for i in range(self.num_choices):
        choice = i if self.distinct else 0
        dna = self.candidates[choice].first_dna(attach_spec=False)
        choice_dna_list.append(DNA(choice, [dna]))
      return DNA(None, choice_dna_list, spec=self if attach_spec else None)

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
      if choice_dna.value < 0 or choice_dna.value >= len(self.candidates):
        raise ValueError(
            f'Choice value (choice_dna.value) is out of range '
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
          return DNA(parent_choice_value,
                     subdna_list, spec=self if attach_spec else None)
    return None

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    sub_length = sum([len(c) for c in self.candidates])
    return self.num_choices * (1 + sub_length)

  def __getitem__(
      self, index: Union[int, slice]
      ) -> Union['Choices', List['Choices']]:
    """Operator [] to return the sub choice(s) if index is int."""
    if self.num_choices == 1:
      sub_choices = [self]
    else:
      sub_choices = self._subchoice_specs
    return sub_choices[index]

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
      kvlist = [('id', object_utils.quote_if_str(str(self.id)), '\'\'')]
    else:
      kvlist = []
    additionl_properties = object_utils.kvlist_str(kvlist + [
        ('name', object_utils.quote_if_str(self.name), None),
        ('distinct', self.distinct, True),
        ('sorted', self.sorted, False),
        ('hints', object_utils.quote_if_str(self.hints), None),
        ('subchoice_index', self.subchoice_index, None)
    ], compact=False, root_indent=root_indent)
    if additionl_properties:
      s.append(', ')
      s.append(additionl_properties)
    s.append(')')
    return ''.join(s)


def float_scale_spec(field_name):
  """Returns value spec for the scale of a continuous range."""
  return (field_name, schema.Enum(None, [None, 'linear', 'log', 'rlog']),
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
        ('min_value', schema.Float(), 'Minimum value.'),
        ('max_value', schema.Float(), 'Maximum value.'),
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
  def is_leaf(self) -> bool:
    """Returns whether the current node is a leaf node."""
    return True

  @property
  def decision_points(self) -> List[DecisionPoint]:
    """Returns all decision points in their declaration order."""
    return [self]

  @property
  def space_size(self) -> int:
    """Returns the size of the search space. Use -1 for infinity."""
    return -1

  def next_dna(self,
               dna: Optional['DNA'] = None,
               attach_spec: bool = True) -> Optional['DNA']:
    """Returns the next DNA in the space represented by this spec.

    Args:
      dna: The DNA whose next will be returned. If None, `next_dna` will return
        the first DNA.
      attach_spec: If True, current spec will be attached to the returned DNA.

    Returns:
      The next DNA or None if there is no next DNA.
    """
    if dna is None:
      return DNA(self.min_value, spec=self if attach_spec else None)
    # TODO(daiyip): Use hint for implementing stateful `next_dna`.
    raise NotImplementedError(
        '`next_dna` is not supported on `Float` yet.')

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


@symbolic.members(
    [],
    init_arg_list=['hints', 'location', 'name'],
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
  def is_leaf(self) -> bool:
    """Returns whether the current node is a leaf node."""
    return True

  @property
  def decision_points(self) -> List[DecisionPoint]:
    """Returns all decision points in their declaration order."""
    return [self]

  @property
  def space_size(self) -> int:
    """Returns the size of the search space. Use -1 for infinity."""
    return -1

  def next_dna(self,
               dna: Optional['DNA'] = None,
               attach_spec: bool = True) -> Optional['DNA']:
    """Returns the next DNA in the space represented by this spec.

    Args:
      dna: The DNA whose next will be returned. If None, `next_dna` will return
        the first DNA.
      attach_spec: If True, current spec will be attached to the returned DNA.

    Returns:
      The next DNA or None if there is no next DNA.
    """
    raise NotImplementedError(
        '`next_dna` is not supported on `CustomDecisionPoint`.')

  def __len__(self) -> int:
    """Returns number of decision points in current space."""
    return 1

  def validate(self, dna: DNA) -> None:
    """Validate whether a DNA value conforms to this spec."""
    if not isinstance(dna.value, str):
      raise ValueError(
          f'CustomDecisionPoint expects string type DNA. '
          f'Encountered: {dna!r}, Location: {self.location.path}.')

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
        ('hints', object_utils.quote_if_str(self.hints), None),
    ])
    return f'{self.__class__.__name__}({details})'


class ConditionalKey:
  """Key used in :class:`pyglove.KeyPath` to represent conditional element.

  For example, `a[=1].b` means `a.b` when `a == 1`.
  `a[0][=0][0]` means `a[0][0]` when `a[0] == 0`.
  """

  def __init__(self, index: int, num_choices: int):
    self._index = index
    self._num_choices = num_choices

  @property
  def index(self) -> int:
    """Return selected index of current condition."""
    return self._index

  @property
  def num_choices(self) -> int:
    """Returns number of choices of current condition."""
    return self._num_choices

  def __str__(self) -> Text:
    return f'={self._index}/{self._num_choices}'


#
# Helper methods for creating DNASpec manually.
#


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


def manyof(num_choices: int,
           candidates: List[DNASpec],
           distinct: bool = True,
           sorted: bool = False,      # pylint: disable=redefined-builtin
           literal_values: Optional[List[Union[Text, int, float]]] = None,
           hints: Any = None,
           location: object_utils.KeyPath = object_utils.KeyPath(),
           name: Optional[Text] = None) -> Choices:
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


def oneof(candidates: List[DNASpec],
          literal_values: Optional[List[Union[Text, int, float]]] = None,
          hints: Any = None,
          location: object_utils.KeyPath = object_utils.KeyPath(),
          name: Optional[Text] = None) -> Choices:
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


def floatv(min_value: float,
           max_value: float,
           scale: Optional[Text] = None,
           hints: Any = None,
           location: object_utils.KeyPath = object_utils.KeyPath(),
           name: Optional[Text] = None) -> Float:
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


def custom(hints: Any = None,
           location: object_utils.KeyPath = object_utils.KeyPath(),
           name: Optional[Text] = None) -> CustomDecisionPoint:
  """Returns a custom decision point.

  It creates the genotype for subclasses of :func:`pyglove.hyper.CustomHyper`.

  Example::

    spec = pg.geno.custom(hints='some hints')

  Args:
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
  return CustomDecisionPoint(hints=hints, location=location, name=name)


#
# Interface for DNAGenerator.
#


class DNAGenerator(symbolic.Object):
  """Base class for genome generator.

  A DNA generator is an object that produces :class:`pyglove.DNA`, and
  optionally takes feedback from the caller to improve its future proposals.

  To implement a DNA generator, the user must implement the `_propose` method,
  and can optionally override the `_setup`, `_feedback` and `_replay` methods.

   * Making proposals (Required): This method defines what to return as the
     next DNA from the generator, users MUST override the `_propose` method to
     implement this logic. `_propose` can raise `StopIteration` when no more
     DNA can be produced.

   * Custom setup (Optional): Usually a DNAGenerator subclass has its internal
     state, which can be initialized when the search space definition is
     attached to the DNAGenerator. To do so, the user can override the `_setup`
     method, in which we can access the search space definition (DNASpec object)
     via `self.dna_spec`.

   * Taking feedback (Optional): A DNAGenerator may take feedbacks from the
     caller on the fitness of proposed DNA to improve future proposals. The
     fitness is measured by a reward (a float number as the measure of a single
     objective, or a tuple of float numbers as the measure for multiple
     objectives). The user should override the `_feedback` method to implement
     such logics. If the reward is for multiple objectives. The user should
     override the `multi_objective` property to return True.

   * State recovery (Optional): DNAGenerator was designed with distributed
     computing in mind, in which a process can be preempted or killed
     unexpectedly. Therefore, a DNAGenerator should be able to recover its
     state from historical proposals and rewards. The `recover` method was
     introduced for such purpose, whose default implementation is to replay the
     history through the `_feedback` method. If the user has a custom replay
     logic other than `_feedback`, they should override the `_replay` method.
     In some use cases, the user may want to implement their own checkpointing
     logic. In such cases, the user can override the `recover` method as a
     no-op. As aside note, the `recover` method will be called by the tuning
     backend (see `tuning.py`) after `setup` but before `propose`.

  See also:

   * :class:`pyglove.geno.Sweeping`
   * :class:`pyglove.geno.Random`
   * :func:`pyglove.geno.dna_generator`
  """

  def setup(self, dna_spec: DNASpec) -> None:
    """Setup DNA spec."""
    self._dna_spec = dna_spec
    self._num_proposals = 0
    self._num_feedbacks = 0
    self._setup()

  def _setup(self) -> None:
    """Subclass should override this for adding additional setup logics."""

  @property
  def multi_objective(self) -> bool:
    """If True, current DNA generator supports multi-objective optimization."""
    return False

  @property
  def needs_feedback(self) -> bool:
    """Returns True if the DNAGenerator needs feedback."""
    return self._feedback.__code__ is not DNAGenerator._feedback.__code__  # pytype: disable=attribute-error

  @property
  def dna_spec(self) -> Optional[DNASpec]:
    return getattr(self, '_dna_spec', None)

  @property
  def num_proposals(self):
    """Get number of proposals that are already produced."""
    return self._num_proposals

  @property
  def num_feedbacks(self):
    """Get number of proposals whose feedback are provided."""
    return self._num_feedbacks

  def propose(self) -> DNA:
    """Propose a DNA to evaluate."""
    dna = self._propose()
    self._num_proposals += 1
    return dna

  def _propose(self) -> DNA:
    """Actual propose method which should be implemented by the child class."""
    raise NotImplementedError()

  def feedback(self, dna: DNA, reward: Union[float, Tuple[float]]) -> None:
    """Feedback a completed trial to the algorithm.

    Args:
      dna: a DNA object.
      reward: reward for the DNA. It is a float if `self.multi_objective`
        returns False, otherwise it's a tuple of floats.
    """
    if self.needs_feedback:
      if self.multi_objective and isinstance(reward, float):
        reward = (reward,)
      elif not self.multi_objective and isinstance(reward, tuple):
        if len(reward) != 1:
          raise ValueError(
              f'{self!r} is single objective, but the reward {reward!r} '
              f'contains multiple objectives.')
        reward = reward[0]
      self._feedback(dna, reward)
    self._num_feedbacks += 1

  def _feedback(self, dna: DNA, reward: Union[float, Tuple[float]]) -> None:
    """Actual feedback method which should be implemented by the child class.

    The default implementation is no-op.

    Args:
      dna: a DNA object.
      reward: reward for the DNA. It is a float if `self.multi_objective`
        returns False, otherwise it's a tuple of floats.
    """

  def recover(
      self,
      history: Iterable[Tuple[DNA, Union[None, float, Tuple[float]]]]
      ) -> None:
    """Recover states by replaying the proposal history.

    NOTE: `recover` will always be called before first `propose` and could be
    called multiple times if there are multiple source of history, e.g: trials
    from a previous study and existing trials from current study.

    Args:
      history: An iterable object that consists of historically proposed DNA
        with its reward. The reward will be None if it is not yet provided
        (via feedback).
    """
    for i, (dna, reward) in enumerate(history):
      self._replay(i, dna, reward)
      self._num_proposals += 1
      if reward is not None:
        self._num_feedbacks += 1

  def _replay(
      self,
      trial_id: int,
      dna: DNA,
      reward: Union[None, float, Tuple[float]]):
    """Replay a single DNA from the history for state recovery.

    The default implementation to call `DNAGenerator._feedback`. Subclasses that
    have states and can be recovered from replaying the history should override
    this method. See class `Sweeping` as an example.

    Args:
      trial_id: A zero-based integer as the trial ID for the DNA.
      dna: A historically proposed DNA.
      reward: The reward for the DNA. If None, the reward is not yet fed back
        to the optimizer.
    """
    del trial_id
    if reward is not None:
      self._feedback(dna, reward)

  def __iter__(self) -> Iterator[
      Union[DNA,
            Tuple[DNA, Callable[[Union[float, Tuple[float]]], None]]]]:
    """Iterates DNA generated from current DNAGenerator.

    NOTE(daiyip): `setup` needs to be called first before a DNAGenerator can
    be iterated.

    Yields:
      A tuple of (DNA, feedback) if current DNAGenerator requires feedback,
        otherwise DNA.
    """
    while True:
      try:
        dna = self.propose()
        if self.needs_feedback:
          feedback = lambda r: self.feedback(dna, r)
          yield (dna, feedback)
        else:
          yield dna
      except StopIteration:
        break


class Sweeping(DNAGenerator):
  """Sweeping (Grid Search) DNA generator."""

  def _setup(self):
    """Setup DNA spec."""
    self._last_proposed_dna = None

  def _propose(self) -> DNA:
    """Propose a random DNA."""
    next_dna = self.dna_spec.next_dna(self._last_proposed_dna)
    if next_dna is None:
      raise StopIteration()
    self._last_proposed_dna = next_dna
    return next_dna

  def _replay(self, trial_id: int, dna: DNA, reward: Any) -> None:
    """Replay the history to recover the last proposed DNA."""
    del trial_id, reward
    self._last_proposed_dna = dna


@symbolic.members([
    ('seed', schema.Int().noneable(), 'Random seed.')
])
class Random(DNAGenerator):
  """Random DNA generator."""

  def _setup(self):
    """Setup DNA spec."""
    if self.seed is None:
      self._random = random
    else:
      self._random = random.Random(self.seed)

  def _propose(self) -> DNA:
    """Propose a random DNA."""
    return random_dna(self._dna_spec, self._random)

  def _replay(self, trial_id: int, dna: DNA, reward: Any) -> None:
    """Replay the history to recover the last proposed DNA."""
    # NOTE(daiyip): If the seed is fixed, we want to reproduce the same
    # sequence of random examples, we can do this simply by repeating previous
    # generation process.
    if self.seed is not None:
      random_dna(self._dna_spec, self._random)


@symbolic.members([
    ('generator', schema.Object(DNAGenerator),
     'Inner generator, whose proposal will be deduped.'),
    ('hash_fn', schema.Callable(
        [schema.Object(DNA)], returns=schema.Int()).noneable(),
     'Hashing function. If None, the hash will be based on DNA values.'),
    ('auto_reward_fn', schema.Callable(
        [schema.List(
            schema.Union([schema.Float(), schema.Tuple(schema.Float())]))],
        returns=schema.Union(
            [schema.Float(), schema.Tuple(schema.Float())])).noneable(),
     ('If None, duplicated proposals above the `max_duplicates` limit will be '
      'dropped. Otherwise, its reward will be automatically computed from the '
      'rewards of previous duplicates, without the client to evaluate it.')),
    ('max_duplicates', schema.Int(min_value=1, default=1),
     'Max number of duplicates allowed per entry.'),
    ('max_proposal_attempts', schema.Int(min_value=1, default=100),
     'Max number of attempts if duplicated entries are encountered.')
])
class Deduping(DNAGenerator):
  """Deduping generator.

  A deduping generator can be applied on another generator to dedup its
  proposed DNAs.

  **Hash function**

  By default, the hash function is the symbolic hash of the DNA, which returns
  the same hash when the decisions from the DNA are the same.

  For example::

    pg.geno.Deduping(pg.geno.Random())

  will only generate unique DNAs. When `hash_fn` is specified, it allows the
  user to compute the hash for a DNA.

  For example::

    pg.geno.Deduping(pg.geno.Random(),
                     hash_fn=lambda dna: sum(dna.to_numbers()))

  will dedup based on the sum of all decision values.

  **Number of duplicates**

  An optional `max_duplicates` can be provided by the user to allow a few
  duplicates.

  For example::

    pg.geno.Deduping(pg.geno.Random(), max_duplicates=5)

  Note: for inner DNAGenerators that requires user feedback, duplication
  accounting is based on DNAs that are fed back to the DNAGenerator, not
  proposed ones.

  **Automatic reward computation**

  Automatic reward computation will be enabled when `auto_reward_fn` is
  provided AND when the inner generator takes feedback. It allows users to
  compute the reward for new duplicates (which exceed the `max_duplicates`
  limit) by aggregating rewards from previous duplicates. Such DNAs will be
  fed back to the DNAGenerator without client's evaluation (supported by
  `pg.sample` through the 'reward' metadata).

  For example::

    pg.geno.Deduping(pg.evolution.regularized_evolution(),
                     auto_reward_fn=lambda rs: sum(rs) / len(rs))
  """

  @property
  def needs_feedback(self) -> bool:
    return self.generator.needs_feedback

  def _setup(self):
    self.generator.setup(self.dna_spec)
    self._hash_fn = self.hash_fn or symbolic.sym_hash
    self._cache = {}
    self._enables_auto_reward = (
        self.needs_feedback and self.auto_reward_fn is not None)

  def _propose(self):
    """Proposes a deduped DNA."""
    attempts = 0
    while attempts < self.max_proposal_attempts:
      dna = self.generator.propose()
      hash_key = self._hash_fn(dna)
      history = self._cache.get(hash_key, [])
      dna.set_metadata('dedup_key', hash_key)
      if len(history) < self.max_duplicates:
        break
      elif self._enables_auto_reward:
        reward = self.auto_reward_fn(history)
        dna.set_metadata('reward', reward)
        break
      attempts += 1
    if attempts == self.max_proposal_attempts:
      raise StopIteration()
    if not self.needs_feedback:
      self._add_dna_to_cache(dna, None)
    return dna

  def _feedback(self, dna: DNA, reward: Union[float, Tuple[float]]) -> None:
    self.generator.feedback(dna, reward)
    self._add_dna_to_cache(dna, reward)

  def _replay(self, trial_id: int, dna: DNA, reward: Any) -> None:
    self.generator._replay(trial_id, dna, reward)  # pylint: disable=protected-access
    self._add_dna_to_cache(dna, reward)

  def _add_dna_to_cache(
      self, dna: DNA, reward: Union[None, float, Tuple[float]]) -> None:
    hash_key = dna.metadata.get('dedup_key', None)
    assert hash_key is not None, dna
    if hash_key not in self._cache:
      self._cache[hash_key] = []
    self._cache[hash_key].append(reward)


def random_dna(
    dna_spec: DNASpec,
    random_generator: Union[None, types.ModuleType, random.Random] = None
    ) -> DNA:
  """Generates a random DNA from a DNASpec.

  Example::

    spec = pg.geno.space([
        pg.geno.oneof([
            pg.geno.constant(),
            pg.geno.constant(),
            pg.geno.constant()
        ]),
        pg.geno.floatv(0.1, 0.2)
    ])

    print(pg.random_dna(spec))
    # DNA([2, 0.1123])

  Args:
    dna_spec: a DNASpec object.
    random_generator: a Python random generator.

  Returns:
    A DNA object.
  """
  r = random_generator or random
  def generator_fn(dna_spec: DNASpec):
    """DNA generation function."""
    if isinstance(dna_spec, Choices):
      if dna_spec.distinct:
        choices = r.sample(
            list(range(len(dna_spec.candidates))), dna_spec.num_choices)
      else:
        choices = [r.randint(0, len(dna_spec.candidates) - 1)
                   for _ in range(dna_spec.num_choices)]
      if dna_spec.sorted:
        choices = sorted(choices)
      return choices
    elif isinstance(dna_spec, Float):
      return r.uniform(dna_spec.min_value, dna_spec.max_value)
    else:
      raise ValueError(f'\'random_dna\' for {dna_spec!r} is not supported.')
  return DNA.from_fn(dna_spec, generator_fn).use_spec(dna_spec)


def dna_generator(func: Callable[[DNASpec], Iterator[DNA]]):
  """Decorator that converts a generation function to a DNAGenerator class.

  Example::

    # A DNA generator that reads DNA from file.

    def from_file(filepath):
      @pg.geno.dna_generator
      def file_based_dna_generator(dna_spec):
        dna_list = pg.load(filepath)
        for dna in dna_list:
          dna.use_spec(dna_spec)
          yield dna

      return file_based_dna_generator

  See also: :class:`pyglove.DNAGenerator`

  Args:
    func: the generation function in signature:
      `(DNASpec) -> Iterator[DNA]`

  Returns:
    A DNAGenerator class.
  """

  class SimpleDNAGenerator(DNAGenerator):
    """Simple DNA generator."""

    def _setup(self):
      self._iterator = func(self.dna_spec)
      self._error = None

    def _propose(self) -> DNA:
      if self._error is not None:
        raise ValueError(
            f'Error happened earlier: {self._error}') from self._error
      try:
        return next(self._iterator)
      except Exception as e:
        if not isinstance(e, StopIteration):
          self._error = e
        raise

  return SimpleDNAGenerator
