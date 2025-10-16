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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
      properties[utils.JSONConvertible.TYPE_NAME_KEY] = {'const': type_name}
      required.append(utils.JSONConvertible.TYPE_NAME_KEY)

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
    definitions = [_json_schema_from_cls(value_spec.cls)]

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

  if (value_spec.has_default
      and value_spec.default is not None
      and not isinstance(value_spec, vs.Dict)):
    default = utils.to_json(value_spec.default)
    if not include_type_name:
      def _remove_type_name(_, v: Dict[str, Any]):
        if isinstance(v, dict):
          v.pop(utils.JSONConvertible.TYPE_NAME_KEY, None)
        return v
      default = utils.transform(default, _remove_type_name)
    definition['default'] = default

  if not ignore_nonable and value_spec.is_noneable:
    nullable = {'type': 'null'}
    if 'anyOf' in definition:
      definition['anyOf'].append(nullable)
    else:
      definition = {'anyOf': [definition, nullable]}
    if value_spec.default is None:
      definition['default'] = None
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
  if 'default' in root:
    canonical_form['default'] = root['default']
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

class_schema.ValueSpec.to_json_schema = _value_spec_to_json_schema
class_schema.Schema.to_json_schema = _schema_to_json_schema

#
# from JSON schema to PyGlove value spec.
#


def _json_schema_to_value_spec(
    json_schema: Dict[str, Any],
    defs: Dict[str, Type[Any]],
    class_fn: Optional[Callable[[str, class_schema.Schema], Type[Any]]],
    add_json_schema_as_metadata: bool,
) -> class_schema.ValueSpec:
  """Converts a JSON schema to a value spec."""
  # Generate code to convert JSON schema to value spec.
  def _value_spec(value_schema: Dict[str, Any]):
    return _json_schema_to_value_spec(
        value_schema, defs, class_fn, add_json_schema_as_metadata
    )

  if '$ref' in json_schema:
    # TODO(daiyip): Support circular references.
    ref_key = json_schema['$ref'].split('/')[-1]
    type_ref = defs.get(ref_key)
    if type_ref is None:
      raise ValueError(
          f'Reference {ref_key!r} not defined in defs. '
          'Please make sure classes being referenced are defined '
          'before the referencing classes. '
      )
    return type_ref
  type_str = json_schema.get('type')
  default = json_schema.get('default', utils.MISSING_VALUE)
  if type_str is None:
    if 'enum' in json_schema:
      for v in json_schema['enum']:
        if not isinstance(v, (str, int, float, bool)):
          raise ValueError(
              f'Enum candidate {v!r} is not supported for JSON schema '
              'conversion.'
          )
      return vs.Enum(
          default,
          [v for v in json_schema['enum'] if v is not None]
      )
    elif 'anyOf' in json_schema:
      candidates = []
      accepts_none = False
      for v in json_schema['anyOf']:
        candidate = _value_spec(v)
        if candidate.frozen and candidate.default is None:
          accepts_none = True
          continue
        candidates.append(candidate)

      if len(candidates) == 1:
        spec = candidates[0]
      else:
        spec = vs.Union(candidates)
      if accepts_none:
        spec = spec.noneable()
      return spec
  elif type_str == 'null':
    return vs.Any().freeze(None)
  elif type_str == 'boolean':
    return vs.Bool(default=default)
  elif type_str == 'integer':
    minimum = json_schema.get('minimum')
    maximum = json_schema.get('maximum')
    return vs.Int(min_value=minimum, max_value=maximum, default=default)
  elif type_str == 'number':
    minimum = json_schema.get('minimum')
    maximum = json_schema.get('maximum')
    return vs.Float(min_value=minimum, max_value=maximum, default=default)
  elif type_str == 'string':
    pattern = json_schema.get('pattern')
    return vs.Str(regex=pattern, default=default)
  elif type_str == 'array':
    items = json_schema.get('items')
    return vs.List(_value_spec(items) if items else vs.Any(), default=default)
  elif type_str == 'object':
    schema = _json_schema_to_schema(
        json_schema, defs, class_fn, add_json_schema_as_metadata
    )
    if class_fn is not None and 'title' in json_schema:
      return vs.Object(class_fn(json_schema['title'], schema))
    return vs.Dict(schema=schema if schema.fields else None)
  raise ValueError(f'Unsupported type {type_str!r} in JSON schema.')


def _json_schema_to_schema(
    json_schema: Dict[str, Any],
    defs: Dict[str, Type[Any]],
    class_fn: Optional[Callable[[str, class_schema.Schema], Type[Any]]],
    add_json_schema_as_metadata: bool,
) -> class_schema.Schema:
  """Converts a JSON schema to a schema."""
  title = json_schema.get('title')
  properties = json_schema.get('properties', {})
  fields = []
  required = set(json_schema.get('required', []))
  for name, property_schema in properties.items():
    value_spec = _json_schema_to_value_spec(
        property_schema, defs, class_fn, add_json_schema_as_metadata
    )
    if name not in required and not value_spec.has_default:
      value_spec = value_spec.noneable()
    fields.append(
        class_schema.Field(
            name,
            value_spec,
            description=property_schema.get('description'),
            metadata=(
                dict(json_schema=property_schema)
                if add_json_schema_as_metadata else None
            )
        )
    )
  additional_properties = json_schema.get('additionalProperties')
  if additional_properties:
    if isinstance(additional_properties, dict):
      value_spec = _json_schema_to_value_spec(
          additional_properties, defs, class_fn, add_json_schema_as_metadata
      )
    else:
      value_spec = vs.Any()
    fields.append(class_schema.Field(ks.StrKey(), value_spec))
  return class_schema.Schema(
      name=title,
      description=json_schema.get('description'),
      fields=fields,
      allow_nonconst_keys=True,
  )


@classmethod
def _value_spec_from_json_schema(
    cls,
    json_schema: Dict[str, Any],
    class_fn: Optional[Callable[[str, class_schema.Schema], Type[Any]]] = None,
    add_json_schema_as_metadata: bool = False,
) -> class_schema.ValueSpec:
  """Creates a PyGlove value spec from a JSON schema.

  Args:
    json_schema: The JSON schema for a value spec.
    class_fn: A function that creates a PyGlove class from a class name and a
      schema. If None, all "object" type properties will be converted to
      `pg.typing.Dict`. Otherwise, "object" type properties will be converted to
      a class.
    add_json_schema_as_metadata: Whether to add the JSON schema as field
      metadata.

  Returns:
    A PyGlove value spec.
  """
  del cls
  defs = {}
  if '$defs' in json_schema:
    for key, def_entry in json_schema['$defs'].items():
      defs[key] = _json_schema_to_value_spec(
          def_entry, defs, class_fn, add_json_schema_as_metadata
      )
  return _json_schema_to_value_spec(
      json_schema, defs, class_fn, add_json_schema_as_metadata
  )


@classmethod
def _schema_from_json_schema(
    cls,
    json_schema: Dict[str, Any],
    class_fn: Optional[Callable[[str, class_schema.Schema], Type[Any]]] = None,
    add_json_schema_as_metadata: bool = False,
) -> class_schema.Schema:
  """Creates a PyGlove schema from a JSON schema.

  Args:
    json_schema: The JSON schema to convert.
    class_fn: A function that creates a PyGlove class from a class name and a
      schema. If None, all "object" type properties will be converted to
      `pg.typing.Dict`. Otherwise, "object" type properties will be converted to
      a class.
    add_json_schema_as_metadata: Whether to add the JSON schema as field
      metadata.

  Returns:
    A PyGlove schema.
  """
  del cls
  if json_schema.get('type') != 'object':
    raise ValueError(
        f'JSON schema is not an object type: {json_schema!r}'
    )
  return _json_schema_to_schema(
      json_schema, {}, class_fn, add_json_schema_as_metadata
  )


class_schema.ValueSpec.from_json_schema = _value_spec_from_json_schema
class_schema.Schema.from_json_schema = _schema_from_json_schema
