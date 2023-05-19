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

from typing import Any, Dict, List, Optional, Tuple, Union

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
    ] = None,   # pylint: disable=bad-continuation
    args_docstr: Optional[Dict[str, object_utils.DocStrArgument]] = None
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
    args_docstr: (Optional) a dict of argument names to
      :class:`pg.object_utils.DocStrArgument` object. If present, they will
      be used as the description for the ``Field`` objects.

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

  def maybe_add_description(arg_name, field):
    if args_docstr and not field.description:
      arg_docstr = args_docstr.get(arg_name, None)
      if arg_docstr is not None:
        field.set_description(arg_docstr.description)
    return field

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
      arg_spec = (arg_name, decl_spec)
    else:
      arg_spec = arg_dict[arg_name]
      if not decl_spec.is_compatible(arg_spec[1]):
        raise TypeError(
            f'{signature.id}: the value spec ({arg_spec[1]!r}) of symbolic '
            f'argument {arg_name} is not compatible with the value spec '
            f'({decl_spec!r}) from function signature.')
      if arg_spec[1].default in [object_utils.MISSING_VALUE, None]:
        arg_spec[1].extend(decl_spec).set_default(decl_spec.default)
      elif (decl_spec.default != arg_spec[1].default
            and (not isinstance(arg_spec[1], vs.Dict)
                 or decl_spec.default != object_utils.MISSING_VALUE)):
        raise ValueError(
            f'{signature.id}: the default value ({arg_spec[1].default!r}) '
            f'of symbolic argument {arg_name!r} does not equal to the default '
            f'value ({decl_spec.default!r}) specified at function signature '
            f'declaration.')
    return maybe_add_description(arg_name, class_schema.Field(*arg_spec))
  # Add positional named arguments.
  arg_fields: List[class_schema.Field] = [
      get_arg_field(arg) for arg in signature.args]

  # Add positional wildcard arguments.
  if signature.varargs:
    if varargs_spec is None:
      varargs_spec = (
          ks.ConstStrKey(signature.varargs.name),
          vs.List(vs.Any()))
    elif not isinstance(varargs_spec[1], vs.List):
      raise ValueError(
          f'{signature.id}: the value spec for positional wildcard argument '
          f'{varargs_spec[0]!r} must be a `pg.typing.List` instance. '
          f'Encountered: {varargs_spec[1]!r}.')
    varargs_spec[1].set_default([])
    vararg_field = maybe_add_description(
        f'*{signature.varargs.name}', class_schema.Field(*varargs_spec))
    arg_fields.append(vararg_field)

  # Add keyword-only arguments.
  arg_fields.extend([get_arg_field(arg) for arg in signature.kwonlyargs])

  # Add extra arguments that are keyword wildcard.
  for arg_name in extra_arg_names:
    arg_field = maybe_add_description(
        arg_name,
        class_schema.Field(*arg_dict[arg_name]))
    arg_fields.append(arg_field)

  # Add keyword wildcard arguments.
  if signature.varkw:
    if kwarg_spec is None:
      kwarg_spec = (ks.StrKey(), vs.Any())
    varkw_field = maybe_add_description(
        f'**{signature.varkw.name}', class_schema.Field(*kwarg_spec))
    arg_fields.append(varkw_field)
  return arg_fields


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
