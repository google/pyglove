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
"""Typing helpers."""

from typing import Any, List, Optional, Tuple, Union

from pyglove.core import object_utils
from pyglove.core.typing import callable_signature
from pyglove.core.typing import class_schema
from pyglove.core.typing import key_specs as ks
from pyglove.core.typing import value_specs as vs


def get_arg_fields(
    signature: callable_signature.Signature,
    args: Optional[List[Union[
        Tuple[Union[str, class_schema.KeySpec], class_schema.ValueSpec, str],
        Tuple[Union[str, class_schema.KeySpec],
              class_schema.ValueSpec, str, Any]]]
    ] = None   # pylint: disable=bad-continuation
) -> List[class_schema.Field]:
  """Get schema fields for the arguments from a function or method signature.

  Args:
    signature: A `Signature` object.
    args: (Optional) explicit value specifications for the arguments, which is a
      list of tuples in:
      (<argumment-name>, <value-spec>, [description], [metadata-objects]).

      * `argument-name` - a string or a `StrKey` object. This name must exist
        in the signature's argument list, unless the signature has a
        ``**kwargs``, where the argument name can be an acceptable key in the
        dict that is passed to the ``**kwargs``. If the argument name is a
        ``StrKey`` object, it specifies a field that matches any keys beyond
        the regular arguments for the ``**kwargs``.
      * `value-spec` - a ``ValueSpec`` object asssociate with the argument
        name.
      * `description` - an optional string as the description for the argument.
      * `metadata-objects` - an optional list of any type, which can be
        used to generate code according to the schema.

  Returns:
    `Field` objects for the arguments from the `signature` in declaration order.
    If an argument is not present in `args`, it will be considered an `Any`.
    Otherwise it will create a `Field` from the explicit specifications. Default
    values for the arguments will be automatially propagated from the signature
    to the fields.

  Raises:
    KeyError: If argument names defined in `args` does not match with the
      arguments from the signature.
    TypeError: The value spec defined in `args` is not compatible with the value
      spec inspected from the signature.
    ValueError: The value spec defined in `args` does not align with the default
      values from the signature.
  """
  arg_dict = dict()
  kwarg_spec = None
  varargs_spec = None

  func_arg_names = set(signature.arg_names)
  # Extra legal argument names that are out of function signature, it is not
  # empty only when function allow **kwargs.
  extra_arg_names = []
  for arg in args or []:
    if isinstance(arg[0], ks.StrKey):
      if kwarg_spec is not None:
        raise KeyError(
            f'{signature.id}: multiple StrKey found in '
            f'symbolic arguments declaration.')
      kwarg_spec = arg
    else:
      assert isinstance(arg[0], (str, ks.ConstStrKey))
      if arg[0] in arg_dict:
        raise KeyError(
            f'{signature.id}: multiple symbolic fields '
            f'found for argument {arg[0]!r}.')
      if signature.varargs and signature.varargs.name == arg[0]:
        varargs_spec = arg
      elif arg[0] not in func_arg_names:
        if signature.has_varkw:
          extra_arg_names.append(arg[0])
        else:
          raise KeyError(
              f'{signature.id}: found extra symbolic argument {arg[0]!r}.')
      arg_dict[arg[0]] = arg

  def get_arg_field(arg_spec):
    arg_name = arg_spec.name
    decl_spec = arg_spec.value_spec
    if arg_name not in arg_dict:
      # Automatic generate symbolic declaration for missing arguments.
      arg_field = (arg_name, decl_spec, f'Argument {arg_name!r}.')
    else:
      arg_field = arg_dict[arg_name]
      if not decl_spec.is_compatible(arg_field[1]):
        raise TypeError(
            f'{signature.id}: the value spec ({arg_field[1]!r}) of symbolic '
            f'argument {arg_name} is not compatible with the value spec '
            f'({decl_spec!r}) from function signature.')
      if arg_field[1].default in [object_utils.MISSING_VALUE, None]:
        arg_field[1].extend(decl_spec).set_default(decl_spec.default)
      elif (decl_spec.default != arg_field[1].default
            and (not isinstance(arg_field[1], vs.Dict)
                 or decl_spec.default != object_utils.MISSING_VALUE)):
        raise ValueError(
            f'{signature.id}: the default value ({arg_field[1].default!r}) '
            f'of symbolic argument {arg_name!r} does not equal to the default '
            f'value ({decl_spec.default!r}) specified at function signature '
            f'declaration.')
    return arg_field

  arg_fields = []

  # Add positional named arguments.
  arg_fields.extend([get_arg_field(arg) for arg in signature.args])

  # Add positional wildcard arguments.
  if signature.varargs:
    if varargs_spec is None:
      varargs_spec = (
          ks.ConstStrKey(signature.varargs.name),
          vs.List(vs.Any()),
          'Wildcard positional arguments.')
    elif not isinstance(varargs_spec[1], vs.List):
      raise ValueError(
          f'{signature.id}: the value spec for positional wildcard argument '
          f'{varargs_spec[0]!r} must be a `pg.typing.List` instance. '
          f'Encountered: {varargs_spec[1]!r}.')
    varargs_spec[1].set_default([])
    arg_fields.append(varargs_spec)

  # Add keyword-only arguments.
  arg_fields.extend([get_arg_field(arg) for arg in signature.kwonlyargs])

  # Add extra arguments that are keyword wildcard.
  for arg_name in extra_arg_names:
    arg_fields.append(arg_dict[arg_name])

  # Add keyword wildcard arguments.
  if signature.varkw:
    if kwarg_spec is None:
      kwarg_spec = (ks.StrKey(), vs.Any(),
                    'Wildcard keyword arguments.')
    arg_fields.append(kwarg_spec)
  return [class_schema.Field(*arg_decl) for arg_decl in arg_fields]


def get_init_signature(
    schema: class_schema.Schema,
    module_name: str,
    name: str,
    qualname: Optional[str] = None,
    is_method: bool = True) -> callable_signature.Signature:
  """Get __init__ signature from schema."""
  arg_names = list(schema.metadata.get('init_arg_list', []))
  if arg_names and arg_names[-1].startswith('*'):
    vararg_name = arg_names[-1][1:]
    arg_names.pop(-1)
  else:
    vararg_name = None

  def get_arg_spec(arg_name):
    field = schema.get_field(arg_name)
    if not field:
      raise ValueError(f'Argument {arg_name!r} is not a symbolic field.')
    return field.value

  args = []
  if is_method:
    args.append(callable_signature.Argument.from_annotation('self'))

  # Prepare positional arguments.
  args.extend([callable_signature.Argument(n, get_arg_spec(n))
               for n in arg_names])

  # Prepare varargs.
  varargs = None
  if vararg_name:
    vararg_spec = get_arg_spec(vararg_name)
    if not isinstance(vararg_spec, vs.List):
      raise ValueError(
          f'Variable positional argument {vararg_name!r} should have a value '
          f'of `pg.typing.List` type. Encountered: {vararg_spec!r}.')
    varargs = callable_signature.Argument(
        vararg_name, vararg_spec.element.value)

  # Prepare keyword-only arguments.
  existing_names = set(arg_names)
  if vararg_name:
    existing_names.add(vararg_name)

  kwonlyargs = []
  varkw = None
  for key, field in schema.fields.items():
    if key not in existing_names:
      if key.is_const:
        kwonlyargs.append(callable_signature.Argument(str(key), field.value))
      else:
        varkw = callable_signature.Argument('kwargs', field.value)

  return callable_signature.Signature(
      callable_type=callable_signature.CallableType.FUNCTION,
      name=name,
      module_name=module_name,
      qualname=qualname,
      args=args,
      kwonlyargs=kwonlyargs,
      varargs=varargs,
      varkw=varkw,
      return_value=None)


def ensure_value_spec(
    value_spec: class_schema.ValueSpec,
    src_spec: class_schema.ValueSpec,
    root_path: Optional[object_utils.KeyPath] = None
) -> Optional[class_schema.ValueSpec]:
  """Extract counter part from value spec that matches dest spec type.

  Args:
    value_spec: Value spec.
    src_spec: Destination value spec.
    root_path: An optional path for the value to include in error message.

  Returns:
    value_spec of src_spec_type

  Raises:
    TypeError: When value_spec cannot match src_spec_type.
  """
  if isinstance(value_spec, vs.Union):
    value_spec = value_spec.get_candidate(src_spec)
  if isinstance(value_spec, vs.Any):
    return None
  if not src_spec.is_compatible(value_spec):
    raise TypeError(
        object_utils.message_on_path(
            f'Source spec {src_spec} is not compatible with destination '
            f'spec {value_spec}.', root_path))
  return value_spec
