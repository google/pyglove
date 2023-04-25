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
"""Object template using hyper primitives."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper import base
from pyglove.core.hyper import derived


class ObjectTemplate(base.HyperValue, object_utils.Formattable):
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
               where: Optional[Callable[[base.HyperPrimitive], bool]] = None):
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
      if (isinstance(value, base.HyperValue)
          and (not self._where or self._where(value))):
        # Apply where clause to child choices.
        if (self._where
            and isinstance(value, base.HyperPrimitive)
            and hasattr(value, 'where')):
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
  def hyper_primitives(self) -> List[Tuple[str, base.HyperValue]]:
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
    assert dna is not None
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
        if isinstance(value, derived.DerivedValue):
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
      if (pg_typing.MISSING_VALUE == input_value
          and pg_typing.MISSING_VALUE != template_value):
        raise ValueError(
            f'Value is missing from input. Path=\'{path}\'.')
      if (isinstance(template_value, base.HyperValue)
          and (not self._where or self._where(template_value))):
        children.append(template_value.encode(input_value))
      elif isinstance(template_value, derived.DerivedValue):
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
             **kwargs) -> str:
    """Format this object."""
    details = object_utils.format(
        self._value, compact, verbose, root_indent, **kwargs)
    return f'{self.__class__.__name__}(value={details})'

  def custom_apply(
      self,
      path: object_utils.KeyPath,
      value_spec: pg_typing.ValueSpec,
      allow_partial: bool,
      child_transform: Optional[Callable[
          [object_utils.KeyPath, pg_typing.Field, Any], Any]] = None
      ) -> Tuple[bool, 'ObjectTemplate']:
    """Validate candidates during value_spec binding time."""
    # Check if value_spec directly accepts `self`.
    if not value_spec.value_type or not isinstance(self, value_spec.value_type):
      value_spec.apply(
          self._value,
          allow_partial,
          root_path=self.root_path)
    return (False, self)


def template(
    value: Any,
    where: Optional[Callable[[base.HyperPrimitive], bool]] = None
    ) -> ObjectTemplate:
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


def dna_spec(
    value: Any,
    where: Optional[Callable[[base.HyperPrimitive], bool]] = None
    ) -> geno.DNASpec:
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


def materialize(
    value: Any,
    parameters: Union[geno.DNA, Dict[str, Any]],
    use_literal_values: bool = True,
    where: Optional[Callable[[base.HyperPrimitive], bool]] = None) -> Any:
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
