# Copyright 2025 The PyGlove Authors
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
"""PyGlove typing to JSON schema conversion."""

import dataclasses
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

from pyglove.core import utils
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


@dataclasses.dataclass
class _DefEntry:
  definition: Optional[Dict[str, Any]]
  num_references: int = 0


def _json_schema_from_schema(
    schema: class_schema.Schema,
    *,
    defs: Dict[str, _DefEntry],
    type_name: Optional[str] = None,
    include_type_name: bool = True,
    include_subclasses: bool = False,
) -> Dict[str, Any]:
  """Converts a PyGlove schema to JSON schema."""
  title = schema.name.split('.')[-1] if schema.name else None
  entry = defs.get(title)
  if entry is None:
    # NOTE(daiyip): Add a forward reference to the entry so recursive
    # references can be resolved.
    entry = _DefEntry(definition=None, num_references=0)
    if title:
      defs[title] = entry

    additional_properties = False
    properties = {}
    required = []

    if type_name and include_type_name:
      properties['_type'] = {'const': type_name}
      required.append('_type')

    for key, field in schema.items():
      if isinstance(key, ks.ConstStrKey):
        prop = _json_schema_from_value_spec(
            field.value, defs=defs
        )
        if field.description:
          prop['description'] = field.description
        properties[key.text] = prop
        if not field.value.has_default:
          required.append(key.text)
      else:
        if isinstance(field.value, vs.Any):
          additional_properties = True
        else:
          additional_properties = _json_schema_from_value_spec(
              field.value, defs=defs,
              include_type_name=include_type_name,
              include_subclasses=include_subclasses,
          )
    entry.definition = _dict_with_optional_fields([
        ('type', 'object', None),
        ('title', title, None),
        ('description', schema.description, None),
        ('properties', properties, {}),
        ('required', required, []),
        ('additionalProperties', additional_properties, None),
    ])

  entry.num_references += 1
  if title:
    return {'$ref': f'#/$defs/{title}'}
  else:
    assert entry.definition is not None
    return entry.definition   # pytype: disable=bad-return-type


def _json_schema_from_value_spec(
    value_spec: vs.ValueSpec,
    *,
    defs: Dict[str, _DefEntry],
    include_type_name: bool = True,
    include_subclasses: bool = False,
    ignore_nonable: bool = False,
) -> Dict[str, Any]:
  """Converts a value spec to JSON schema."""
  def _child_json_schema(v: vs.ValueSpec, ignore_nonable: bool = False):
    return _json_schema_from_value_spec(
        v, defs=defs,
        include_type_name=include_type_name,
        include_subclasses=include_subclasses,
        ignore_nonable=ignore_nonable
    )

  if isinstance(value_spec, vs.Bool):
    definition = {
        'type': 'boolean'
    }
  elif isinstance(value_spec, vs.Int):
    definition = _dict_with_optional_fields([
        ('type', 'integer', None),
        ('minimum', value_spec.min_value, None),
        ('maximum', value_spec.max_value, None),
    ])
  elif isinstance(value_spec, vs.Float):
    definition = _dict_with_optional_fields([
        ('type', 'number', None),
        ('minimum', value_spec.min_value, None),
        ('maximum', value_spec.max_value, None),
    ])
  elif isinstance(value_spec, vs.Str):
    definition = _dict_with_optional_fields([
        ('type', 'string', None),
        ('pattern', getattr(value_spec.regex, 'pattern', None), None)
    ])
  elif isinstance(value_spec, vs.Enum):
    for v in value_spec.values:
      if not isinstance(v, (str, bool, int, float, type(None))):
        raise ValueError(
            f'Enum candidate {v!r} is not supported for JSON schema generation.'
        )
    definition = {
        'enum': [v for v in value_spec.values if v is not None],
    }
  elif isinstance(value_spec, vs.List):
    definition = {
        'type': 'array',
    }
    if not isinstance(value_spec.element.value, vs.Any):
      definition['items'] = _child_json_schema(value_spec.element.value)
    return definition
  elif isinstance(value_spec, vs.Dict):
    if value_spec.schema is None:
      definition = {
          'type': 'object',
          'additionalProperties': True
      }
    else:
      definition = _json_schema_from_schema(
          value_spec.schema, defs=defs, include_type_name=include_type_name
      )
  elif isinstance(value_spec, vs.Object):
    def _json_schema_from_cls(cls: type[Any]):
      schema = getattr(cls, '__schema__', None)
      if schema is None:
        schema = callable_signature.signature(cls).to_schema()
      return _json_schema_from_schema(
          schema,
          defs=defs,
          type_name=getattr(cls, '__type_name__', None),
          include_type_name=include_type_name,
          include_subclasses=include_subclasses,
      )
    definitions = [
        _json_schema_from_cls(value_spec.cls)
    ]

    if include_subclasses:
      for subclass in value_spec.cls.__subclasses__():
        if not inspect.isabstract(subclass):
          definitions.append(_json_schema_from_cls(subclass))

    if len(definitions) == 1:
      definition = definitions[0]
    else:
      definition = {'anyOf': _normalize_anyofs(definitions)}
  elif isinstance(value_spec, vs.Union):
    definition = {
        'anyOf': _normalize_anyofs([
            _child_json_schema(v, ignore_nonable=True)
            for v in value_spec.candidates
        ])
    }
  elif isinstance(value_spec, vs.Any):
    # Consider using return {}
    definition = {
        'anyOf': [
            _child_json_schema(v) for v in [
                vs.Bool(),
                vs.Float(),
                vs.Str(),
                vs.List(vs.Any()),
                vs.Dict(),
            ]
        ]
    }
  else:
    raise TypeError(
        f'Value spec {value_spec!r} cannot be converted to JSON schema.'
    )

  if not ignore_nonable and value_spec.is_noneable:
    nullable = {'type': 'null'}
    if 'anyOf' in definition:
      definition['anyOf'].append(nullable)
    else:
      definition = {'anyOf': [definition, nullable]}
  return definition


def _normalize_anyofs(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  flattened = []
  for candidate in candidates:
    if 'anyOf' in candidate:
      flattened.extend(_normalize_anyofs(candidate['anyOf']))
    else:
      flattened.append(candidate)
  return flattened


def _dict_with_optional_fields(
    pairs: List[Tuple[str, Any, Any]]
) -> Dict[str, Any]:
  return {
      k: v for k, v, default in pairs if v != default
  }


def to_json_schema(
    value: Union[vs.ValueSpec, class_schema.Schema, Any],
    *,
    include_type_name: bool = True,
    include_subclasses: bool = False,
    inline_nested_refs: bool = False,
) -> Dict[str, Any]:
  """Converts a value spec to JSON schema."""
  defs = dict()
  if isinstance(value, class_schema.Schema):
    root = _json_schema_from_schema(
        value, defs=defs,
        type_name=value.name,
        include_type_name=include_type_name,
        include_subclasses=include_subclasses,
    )
  else:
    value = vs.ValueSpec.from_annotation(value, auto_typing=True)
    root = _json_schema_from_value_spec(
        value,
        defs=defs,
        include_type_name=include_type_name,
        include_subclasses=include_subclasses,
    )
  return _canonicalize_schema(root, defs, inline_nested_refs=inline_nested_refs)


def _canonicalize_schema(
    root: Dict[str, Any],
    defs: Dict[str, _DefEntry],
    *,
    inline_nested_refs: bool = True,
    include_defs: bool = True,
) -> Dict[str, Any]:
  """Canonicalizes a JSON schema."""
  if not defs:
    return root

  def _maybe_inline_ref(k: utils.KeyPath, v: Any):
    del k
    if isinstance(v, dict) and '$ref' in v:
      ref_key = v['$ref'].split('/')[-1]
      if defs[ref_key].num_references == 1:
        defs[ref_key].num_references -= 1
        if not inline_nested_refs:
          return defs[ref_key].definition
        else:
          return _canonicalize_schema(
              defs[ref_key].definition, defs=defs, include_defs=False,
              inline_nested_refs=True
          )
    return v

  new_root = utils.transform(root, _maybe_inline_ref)
  if not include_defs:
    return new_root
  referenced_defs = {
      k: v.definition for k, v in defs.items() if v.num_references > 0
  }
  canonical_form = {'$defs': referenced_defs} if referenced_defs else {}
  canonical_form.update(new_root)
  return canonical_form

#
# Override to_json_schema() method for ValueSpec and Schema.
#


def _value_spec_to_json_schema(
    self: Union[vs.ValueSpec, class_schema.Schema, Any],
    *,
    include_type_name: bool = True,
    include_subclasses: bool = False,
    inline_nested_refs: bool = False,
    **kwargs
) -> Dict[str, Any]:
  """Converts a value spec to JSON schema."""
  return to_json_schema(
      self,
      include_type_name=include_type_name,
      include_subclasses=include_subclasses,
      inline_nested_refs=inline_nested_refs,
      **kwargs
  )


def _schema_to_json_schema(
    self: class_schema.Schema,
    *,
    include_type_name: bool = True,
    include_subclasses: bool = False,
    inline_nested_refs: bool = False,
    **kwargs
) -> Dict[str, Any]:
  """Converts a schema to JSON schema."""
  return to_json_schema(
      self,
      include_type_name=include_type_name,
      include_subclasses=include_subclasses,
      inline_nested_refs=inline_nested_refs,
      **kwargs
  )

vs.ValueSpec.to_json_schema = _value_spec_to_json_schema
class_schema.Schema.to_json_schema = _schema_to_json_schema
