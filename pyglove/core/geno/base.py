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
"""Program genotype (base) and genome."""

import abc
import functools
import random
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing


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
     pg_typing.Object(object_utils.KeyPath, default=object_utils.KeyPath()),
     ('KeyPath of associated genetic encoder relative to parent object '
      'template. This allows DNA generator to apply rule based on locations.')),
    ('hints',
     pg_typing.Any(default=None), 'Hints for DNA generator to consume.')
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

  # Override format kwargs for __str__.
  __str_format_kwargs__ = dict(
      compact=True,
      verbose=False,
      hide_default_values=True,
      hide_missing_values=True
  )

  # NOTE(daiyip): we disable the symbolic comparison to allow hashing DNASpec
  # by object ID, therefore we can use DNASpec objects as the keys for a dict.
  # This is helpful when we want to align decision points using DNASpec as
  # dictionary key. Users can use `pg.eq`/`pg.ne` for symbolic comparisons
  # and `pg.hash` for symbolic hashing.
  use_symbolic_comparison = False

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
        if dp.is_categorical and dp.is_subchoice:
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

  @property
  @abc.abstractmethod
  def is_space(self) -> bool:
    """Returns True if current node is a sub-space."""

  @property
  @abc.abstractmethod
  def is_categorical(self) -> bool:
    """Returns True if current node is a categorical choice."""

  @property
  @abc.abstractmethod
  def is_subchoice(self) -> bool:
    """Returns True if current node is a subchoice of a multi-choice."""

  @property
  @abc.abstractmethod
  def is_numerical(self) -> bool:
    """Returns True if current node is numerical decision."""

  @property
  @abc.abstractmethod
  def is_custom_decision_point(self) -> bool:
    """Returns True if current node is a custom decision point."""

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
      self) -> Dict[str, Union['DecisionPoint', List['DecisionPoint']]]:
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
    dna = self._next_dna(dna)
    if attach_spec and dna is not None:
      dna.use_spec(self)
    return dna

  @abc.abstractmethod
  def _next_dna(self, dna: Optional['DNA'] = None) -> Optional['DNA']:
    """Next DNA generation logic that should be overridden by subclasses."""

  def random_dna(self,
                 random_generator: Union[types.ModuleType,
                                         random.Random,
                                         None] = None,
                 attach_spec: bool = True,
                 previous_dna: Optional['DNA'] = None) -> 'DNA':
    """Returns a random DNA based on current spec.

    Args:
      random_generator: An optional Random object. If None, the global random
        module will be used.
      attach_spec: If True, current spec will be attached to the returned DNA.
      previous_dna: An optional DNA representing previous DNA. This field might
        be useful for generating stateful random DNAs.

    Returns:
      A random DNA based on current spec.
    """
    random_generator = random_generator or random
    dna = self._random_dna(random_generator, previous_dna)
    if attach_spec:
      dna.use_spec(self)
    return dna

  @abc.abstractmethod
  def _random_dna(self,
                  random_generator: Union[types.ModuleType, random.Random],
                  previous_dna: Optional['DNA']) -> 'DNA':
    """Random DNA generation logic that should be overridden by subclasses."""

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
  def parent_choice(self) -> Optional['DecisionPoint']:
    """Returns the parent choice of current space."""
    if self.parent_spec is None:
      return None
    return self.parent_spec if self.is_space else self.parent_spec.parent_choice

  @property
  def id(self) -> object_utils.KeyPath:
    """Returns a path of locations from the root as the ID for current node."""
    if self._id is None:
      parent = self.parent_spec
      if parent is None:
        self._id = self.location
      elif self.is_space:
        assert parent.is_categorical, parent
        assert self.index is not None
        self._id = object_utils.KeyPath(
            ConditionalKey(self.index, len(parent.candidates)),
            parent.id) + self.location
      else:
        # Float() or a multi-choice spec of a parent Choice.
        self._id = parent.id + self.location
    return self._id

  def get(self,
          name_or_id: Union[object_utils.KeyPath, str],
          default: Any = None
          ) -> Union['DecisionPoint', List['DecisionPoint']]:
    """Get decision point(s) by name or ID."""
    try:
      return self[name_or_id]
    except KeyError:
      return default

  def __getitem__(
      self,
      name_or_id: Union[object_utils.KeyPath, str]
      ) -> Union['DecisionPoint', List['DecisionPoint']]:
    """Get decision point(s) by name or ID ."""
    v = self._named_decision_points.get(name_or_id, None)
    if v is None:
      v = self._decision_point_by_id[name_or_id]
    return v

  def set_userdata(self, key: str, value: Any) -> None:
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

  @classmethod
  def from_json(cls, json_value, *args, **kwargs) -> symbolic.Object:
    """Override from_json for backward compatibility with serialized data."""
    assert isinstance(json_value, dict)
    json_value.pop('userdata', None)
    return super().from_json(json_value, *args, **kwargs)


@symbolic.members([
    ('name', pg_typing.Str().noneable(),
     ('Name of current node. If present, it should be unique in the search '
      'space. We can use `root_spec[name]` to access named DNASpec.')),
])
class DecisionPoint(DNASpec):
  """Base class for decision points.

  Child classes:

    * :class:`pyglove.geno.Choices`
    * :class:`pyglove.geno.Float`
    * :class:`pyglove.geno.CustomDecisionPoint`
  """

  @property
  def is_space(self) -> bool:
    return False


# pylint: disable=line-too-long
@functools.total_ordering
class DNA(symbolic.Object):
  """The genome of a symbolic object relative to its search space.

  DNA is a hierarchical structure - each DNA node has a value, and a list of
  child DNA nodes. The root node represents the genome that encodes an entire
  object relative to its space. The value of a DNA node could be None, an
  integer, a float number or a string, dependening on its specification
  (:class:`pg.DNASpec`). A valid DNA has a form of the following.

  +--------------------------------------+-----------------+-----------------------------+
  |  Hyper value type                    | Possible values |   Child nodes               |
  |  (DNASpec type)                      |                 |                             |
  +======================================+=================+=============================+
  |:class:`pg.hyper.ObjectTemplate`      | None            |DNA of child decision points |
  |(:class:`pg.geno.Space`)              |(elements > 1)   |(Choices/Float) in the       |
  |                                      |                 |template.                    |
  +--------------------------------------+-----------------+-----------------------------+
  |                                      |None             |Children of elements[0]      |
  |                                      |(elements == 1   |                             |
  |                                      |and elements[0]. |                             |
  |                                      |num_choices > 1) |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |                                      |int              |Children of:                 |
  |                                      |(elements == 1   |elements[0][0]               |
  |                                      |and elements[0]. |                             |
  |                                      |num_choices ==1) |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |                                      |float            |Empty                        |
  |                                      |(elements == 1   |                             |
  |                                      |and elements[0]  |                             |
  |                                      |is geno.Float)   |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |:func:`pg.oneof`                      |int              |Children of Space            |
  |(:class:`pg.geno.Choices`)            |(candidate index |for the chosen candidate     |
  |                                      |as choice)       |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |:func:`pg.manyof`                     |None             |DNA of each chosen candidate |
  |(:class:`pg.geno.Choices)             |(num_choices > 1 |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |                                      |int              |Children of chosen candidate |
  |                                      |(num_choices==1) |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |:func:`pg.floatv`                     |float            |Empty                        |
  |(:class:`pg.geno.Float` )             |                 |                             |
  +--------------------------------------+-----------------+-----------------------------+
  |:class:`pg.hyper.CustomHyper`         |string           |User defined.                |
  |(:class:`pg.geno.CustomDecisionPoint`)|(serialized      |                             |
  |                                      | object)         |                             |
  +--------------------------------------+-----------------+-----------------------------+

  DNA can also be represented in a compact form - a tree of numbers/strs,
  formally defined as::

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
  # pylint: enable=line-too-long

  # Use compact format for __str__ output.
  __str_format_kwargs__ = dict(compact=True)

  # Allow assignment on symbolic attributes.
  allow_symbolic_assignment = True

  @object_utils.explicit_method_override
  def __init__(
      self,
      value: Union[None, int, float, str, List[Any], Tuple[Any]] = None,
      # Set MISSING_VALUE to use default from pg_typing.
      children: Optional[List['DNA']] = None,
      spec: Optional[DNASpec] = None,
      metadata: Optional[Dict[str, Any]] = None,
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
          str,        # As a custom genome.
          List[Any],   # As multi-choice. (coexisting)
          Tuple[Any],  # As a conditional choice.
          None],
      children: Optional[List['DNA']],
      metadata: Optional[Dict[str, Any]],
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
      self, key: str, value: Any, cloneable: bool = False) -> 'DNA':
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
      self, key: str, value: Any, cloneable: bool = False) -> 'DNA':
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
    return self._spec.is_subchoice

  @property
  def multi_choice_spec(self) -> Optional['DecisionPoint']:
    """Returns the multi-choice spec for child DNAs.

    Returns:
      If the children of this DNA are decisions of a multi-choice's subchoices,
      return the multi-choice spec (`pg.geno.Choices`). Otherwise returns None.
    """
    self._ensure_dna_spec()
    multi_choice_spec = None
    if self.children:
      child_spec = self.children[0].spec
      if child_spec.is_subchoice:
        multi_choice_spec = child_spec.parent_spec
    return multi_choice_spec

  @property
  def is_multi_choice_container(self) -> bool:
    """Returns True if the children of this DNA are multi-choice subchoices."""
    return self.multi_choice_spec is not None

  @property
  def literal_value(self) -> Union[str, int, float,
                                   List[Union[str, int, float]]]:
    """Returns the literal value represented by current DNA."""
    self._ensure_dna_spec()
    def _literal_value(dna, prefix):
      if dna.children:
        if dna.value is not None:
          assert dna.spec.is_categorical, dna.spec
          prefix += f'{dna.value}/{len(dna.spec.candidates)} -> '
        v = [_literal_value(c, prefix) for c in dna.children]
        return v[0] if len(v) == 1 else v
      if dna.spec.is_numerical:
        value = str(dna.value)
      elif dna.spec.is_custom_decision_point:
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
  def named_decisions(self) -> Dict[str, Union['DNA', List['DNA']]]:
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
    if not isinstance(spec, DNASpec):
      raise ValueError(
          f'Argument \'spec\' must be a `pg.DNASpec` object. '
          f'Encountered: {spec!r}.')

    if self._spec is spec:
      return self

    def _use_spec_for_child_choices(spec: DNASpec, children: List[DNA]):
      """Use spec for child choices."""
      assert spec.is_categorical, spec
      if spec.num_choices != len(children):
        raise ValueError(
            f'Number of choices ({spec.num_choices}) does not match with '
            f'the number of child values (len(children)). '
            f'Spec: {spec!r}, Children: {children!r}.')

      for i, child in enumerate(children):
        subchoice = spec.subchoice(i)
        child.use_spec(subchoice)

      child_values = [c.value for c in children]
      if spec.sorted and sorted(child_values) != child_values:
        raise ValueError(
            f'Child values {child_values!r} are not sorted. Spec: {spec!r}.')
      if spec.distinct and len(set(child_values)) != len(child_values):
        raise ValueError(
            f'Child values {child_values!r} are not distinct. Spec: {spec!r}.')

    # Skip dummy DNA specs.
    while spec.is_space and len(spec.elements) == 1:
      spec = spec.elements[0]

    if spec.is_space:
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
    elif spec.is_categorical:
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
        assert chosen_candidate.is_space, chosen_candidate

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
          while sub_spec.is_space and len(sub_spec.elements) == 1:
            sub_spec = sub_spec.elements[0]
          if sub_spec.is_numerical or sub_spec.is_custom_decision_point:
            if len(self.children) != 1:
              raise ValueError(
                  f'Encountered more than 1 value.'
                  f'Child value: {self.children}, Spec: {sub_spec}.')
            self.children[0].use_spec(sub_spec)
          else:
            assert sub_spec.is_categorical, sub_spec
            _use_spec_for_child_choices(sub_spec, self.children)
      else:
        # Multiple choices.
        if self.value is not None:
          raise ValueError(
              f'Cannot apply multi-choice DNA spec on '
              f'value {self.value}: {spec!r}.')
        _use_spec_for_child_choices(spec, self.children)
    elif spec.is_numerical:
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
    else:
      assert spec.is_custom_decision_point, spec
      if not isinstance(self.value, str):
        raise ValueError(
            f'DNA value type mismatch, Value: {self.value!r}, Spec: {spec!r}.')
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
      dict_repr: Dict[Union['DecisionPoint', str],
                      Union[None, 'DNA', float, int, str]],
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

    def _choice_index(subchoice, value: Union[int, float, str]) -> int:
      """Gets the index of choice value based on its spec."""
      if isinstance(value, int) and not use_ints_as_literals:
        index = value
        if index < 0 or index >= len(subchoice.candidates):
          identifier = subchoice.name or subchoice.id
          raise ValueError(
              f'Candidate index out of range at choice \'{identifier}\'. Index='
              f'{index}, Number of candidates={len(subchoice.candidates)}.')
      else:
        index = subchoice.candidate_index(value)
      return index

    def _make_dna(spec: DNASpec) -> DNA:
      """Lookup DNA value from parameter values according to DNA spec."""
      if spec.is_space:
        children = []
        for elem in spec.elements:
          child = _make_dna(elem)
          if child is not None:
            children.append(child)
        return DNA(None, children)
      else:
        if spec.is_categorical:
          children = []
          for choice_idx in range(spec.num_choices):
            subchoice = spec.subchoice(choice_idx)
            value = _get_decision(subchoice)

            # It's possible that the decisions for multiple choices are
            # collapsed as a single entry in the dictionary. e.g: {'x': [0, 1]}.
            # In this case, we will make another attempt to get the decision
            # from the parent spec entry.
            if value is None and subchoice.is_subchoice:
              parent_decisions = _get_decision(spec)
              if parent_decisions is not None:
                assert len(parent_decisions) == spec.num_choices, (
                    parent_decisions, spec)
                value = parent_decisions[choice_idx]

            if value is None:
              identifier = subchoice.name or subchoice.id
              raise ValueError(
                  f'Value for \'{identifier}\' is not found in '
                  f'the dictionary {dict_repr!r}.')

            if isinstance(value, DNA):
              children.append(value)
            else:
              choice_index = _choice_index(subchoice, value)
              subspace_dna = _make_dna(subchoice.candidates[choice_index])
              children.append(
                  DNA(choice_index, [subspace_dna] if subspace_dna else []))
          return DNA(None, children)
        elif spec.is_numerical or spec.is_custom_decision_point:
          value = _get_decision(spec)
          if value is None:
            raise ValueError(
                f'Value for \'{spec.name or spec.id}\' is not found '
                f'in the dictionary {dict_repr!r}.')
          if isinstance(value, DNA):
            value = value.value
          if spec.is_numerical:
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
      filter_fn: Optional[Callable[[DecisionPoint], bool]] = None
      ) -> Dict[Union[DecisionPoint, str],
                Union[None, 'DNA', float, int, str,
                      List['DNA'], List[int], List[str]]]:
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
          f'\'key_type\' must be either \'id\', \'name_or_id\' '
          f'or \'dna_spec\'. Encountered: {key_type!r}.')

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

    def _needs_subchoice_key(subchoice):
      return (multi_choice_use_subchoice_as_key
              and (not multi_choice_use_parent_as_key
                   or (key_type != 'name_or_id' or subchoice.name is None)))

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
        if dna.spec.is_categorical and dna.value is not None:
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
        elif dna.spec.is_numerical or dna.spec.is_custom_decision_point:
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
      if dp.is_categorical and dp.is_subchoice:
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
      dna_values: List[Union[int, float, str]],
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
      if dna_spec.is_space:
        children = [_bind_decisions(elem) for elem in dna_spec.elements]
      elif dna_spec.is_categorical:
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
      ) -> Union[List[Union[int, float, str]],
                 object_utils.Nestable[Union[int, float, str]]]:
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
    if not isinstance(dna_spec, DNASpec):
      raise TypeError(
          f'Argument \'dna_spec\' should be DNASpec type. '
          f'Encountered {dna_spec}.')

    if dna_spec.is_space:
      # Generate values for Space.
      children = []
      for child_spec in dna_spec.elements:
        children.append(DNA.from_fn(child_spec, generator_fn))
      if len(children) == 1:
        return children[0]
      dna = DNA(None, children)
    elif dna_spec.is_categorical:
      assert isinstance(dna_spec, DecisionPoint), dna_spec
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
    else:
      assert isinstance(dna_spec, DecisionPoint), dna_spec
      decision = generator_fn(dna_spec)
      if isinstance(decision, DNA):
        dna = decision
      else:
        dna = DNA(decision)
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
          object_utils.JSONConvertible.TYPE_NAME_KEY: (
              self.__class__.__serialization_key__
          ),
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
      json_value: Dict[str, Any],
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
      self, key: Union[int, slice, str, object_utils.KeyPath, 'DecisionPoint']
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
          key: Union[int, slice, str, object_utils.KeyPath, 'DecisionPoint'],
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
      if isinstance(x, (int, float)) and isinstance(y, str):
        return -1
      if isinstance(x, str) and isinstance(y, (int, float)):
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

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      *,
      markdown: bool = False,
      list_wrap_threshold: int = 80,
      as_dict: bool = False,
      **kwargs,
  ):
    """Customize format method for DNA for more compact representation."""
    if as_dict and self.spec:
      details = object_utils.format(
          self.to_dict(value_type='choice_and_literal'),
          False,
          verbose,
          root_indent,
          **kwargs)
      s = f'DNA({details})'
      compact = False
    else:
      if 'list_wrap_threshold' not in kwargs:
        kwargs['list_wrap_threshold'] = list_wrap_threshold

      if not verbose:
        s = super().format(False, verbose, root_indent, **kwargs)
      elif self.is_leaf:
        s = f'DNA({self.value!r})'
      else:
        rep = object_utils.format(
            self.to_json(compact=True, type_info=False),
            compact,
            verbose,
            root_indent,
            **kwargs,
        )
        if rep and rep[0] == '(':
          # NOTE(daiyip): for conditional choice from the root,
          # we don't want to keep duplicate round bracket.
          s = f'DNA{rep}'
        else:
          s = f'DNA({rep})'
    return object_utils.maybe_markdown_quote(s, markdown)

  def parameters(
      self, use_literal_values: bool = False) -> Dict[str, str]:
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

  def _sym_clone(self, deep: bool, memo: Any = None) -> 'DNA':
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
                      parameters: Dict[str, Any],
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


symbolic.members([
    (
        'value',
        pg_typing.Union(
            [pg_typing.Int(), pg_typing.Float(), pg_typing.Str()]
        ).noneable(),
        'Value of DNA node.',
    ),
    (
        'children',
        pg_typing.List(pg_typing.Object(DNA), default=[]),
        (
            'DNA list as child nodes for template members or '
            'chosen candidates of choices.'
        ),
    ),
    (
        'metadata',
        pg_typing.Dict(
            [(pg_typing.StrKey(), pg_typing.Any(), 'Key-value pairs.')]
        ),
        'Metadata assigned to the DNA.',
    ),
])(DNA)


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

  def __str__(self) -> str:
    return f'={self._index}/{self._num_choices}'
