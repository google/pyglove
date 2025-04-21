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
"""Python code execution."""

import ast
import contextlib
import io
import multiprocessing
import pickle
import queue
from typing import Any, Callable, Dict, Optional, Union

from pyglove.core import utils
from pyglove.core.coding import errors
from pyglove.core.coding import parsing
from pyglove.core.coding import permissions


# Key in returned dict that captures stdout.
STDOUT_KEY = '__stdout__'

# Key in the returned dict that represents the final result.
RESULT_KEY = '__result__'
_TLS_CODE_RUN_CONTEXT = '__code_run_context__'


@contextlib.contextmanager
def context(**kwargs):
  """Context manager to inject symbols for code execution."""
  ctx = get_context()
  ctx.update(kwargs)
  utils.thread_local_push(_TLS_CODE_RUN_CONTEXT, ctx)

  try:
    yield ctx
  finally:
    utils.thread_local_pop(_TLS_CODE_RUN_CONTEXT)


def get_context() -> Dict[str, Any]:
  """Gets the current context for code execution."""
  context_stack = utils.thread_local_get(_TLS_CODE_RUN_CONTEXT, None)
  return dict(context_stack[-1]) if context_stack else {}


def evaluate(
    code: str,
    *,
    global_vars: Optional[Dict[str, Any]] = None,
    permission: Optional[permissions.CodePermission] = None,
    returns_stdout: bool = False,
    outputs_intermediate: bool = False,
) -> Union[Any, Dict[str, Any]]:
  """Executes Python code.

  Features:
    * Fine-grained execution policy for limiting what APIs could be executed.
      This eliminates the need for sandboxing.
    * It exposes both the final results and intermediate results (variables).

  Args:
    code: Python code to run.
    global_vars: An optional dict as the globals that could be referenced by the
      code.
    permission: Permission for the Python code to run.
    returns_stdout: If True, the stdout (a str) will be returned.
    outputs_intermediate: Applicable when returns_stdout is False. If True,
      intermediate output will be outputted as a dict, with the last line's
      value accessible by key '__result__' and the std output accessible by
      key '__stdout__'. Otherwise the value of the last line will be returned.

  Returns:
    The value of the last line of the code block. Or a dict of variable
    names of all locals to their evaluated values as the output of the code to
    run. The value for the last line can be accessed by key '__result__'. Or the
    stdout as a str.
  """
  # Set up the permission and context.
  permission = permission or permissions.get_permission()
  ctx = dict(get_context())
  if global_vars:
    ctx.update(global_vars)

  # Parse the code str.
  code_block = parsing.parse(code, permission)
  global_vars, orig_global_vars = ctx, ctx.copy()

  # No code.
  if not code_block.body:   # pytype: disable=attribute-error
    return {} if outputs_intermediate else None

  stdout = io.StringIO()
  with contextlib.redirect_stdout(stdout):
    if hasattr(code_block.body[-1], 'value'):   # pytype: disable=attribute-error
      last_expr = code_block.body.pop()  # pytype: disable=attribute-error
      result_vars = [RESULT_KEY]

      if isinstance(last_expr, ast.Assign):
        for name_node in last_expr.targets:
          if isinstance(name_node, ast.Name):
            result_vars.append(name_node.id)

      last_expr = ast.Expression(last_expr.value)  # pytype: disable=attribute-error

      try:
        # Execute the lines before the last expression.
        # NOTE(daiyip): Only a `globals` dict is specified here, which will also
        # be used to output intermediate values by `exec`. We do not specify a
        # separate `locals` dict here, for - "If exec gets two separate objects
        # as globals and locals, the code will be executed as if it were
        # embedded in a class definition." - as the Python document explains.
        # The outcome is that new functions defined in the code block could not
        # be called by other newly defined functions.
        # Refer to https://stackoverflow.com/questions/
        # 73940751/why-cant-i-call-a-function-from-another-function-using-exec
        # for more details.
        exec(compile(code_block, '', mode='exec'), global_vars)  # pylint: disable=exec-used

        # Evaluate the last expression.
        result = eval(  # pylint: disable=eval-used
            compile(last_expr, '', mode='eval'), global_vars
        )
      except BaseException as e:
        raise errors.CodeError(code, e) from e

      for result_var in result_vars:
        global_vars[result_var] = result
    else:
      try:
        exec(compile(code_block, '', mode='exec'), global_vars)  # pylint: disable=exec-used
      except BaseException as e:
        raise errors.CodeError(code, e) from e
      global_vars[RESULT_KEY] = list(global_vars.values())[-1]

  if returns_stdout:
    return stdout.getvalue()
  if outputs_intermediate:
    outputs = {}
    for k, v in global_vars.items():
      if k == '__builtins__':
        continue
      if k not in orig_global_vars or v is not orig_global_vars[k]:
        outputs[k] = v
    # Add stdout to outputs.
    outputs[STDOUT_KEY] = stdout.getvalue()
    return outputs
  return global_vars[RESULT_KEY]


def sandbox_call(
    func: Callable[..., Any],
    *args,
    timeout: Optional[float] = None,
    **kwargs) -> Any:
  """Calls a function with sandboxing.

  Args:
    func: Function to call.
    *args: Positional arguments for `func`
    timeout: Execution timeout in seconds. If None, wait `func` to complete.
    **kwargs: Keyword arguments for `func`.

  Returns:
    Return value from `func`.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception raised from `func`.
  """
  def _call(q, *args, **kwargs):
    # NOTE(daiyip): if `q` is closed by the main process when `q.put` is called
    # on a subprocess, ValueError will be raised. This is okay since the main
    # process is no longer waiting for the result, and the subprocess could
    # recycled with non-zero error code, which does not affect the main
    # process.
    def _run():
      r = func(*args, **kwargs)
      try:
        return pickle.dumps(r)
      except BaseException as e:
        raise errors.SerializationError(
            f'Cannot serialize sandbox result: {r}', e
        ) from e

    try:
      q.put(_run())
    except Exception as e:  # pylint: disable=broad-exception-caught
      q.put(e)

  q = multiprocessing.Queue()
  p = multiprocessing.Process(
      target=_call, args=tuple([q] + list(args)), kwargs=kwargs
  )
  try:
    p.start()
    x = q.get(timeout=timeout)
  except queue.Empty as e:
    if p.is_alive():
      # We use `kill` instead of `terminate` to release process resources
      # right away.
      p.kill()
    raise TimeoutError(f'Execution time exceed {timeout} seconds.') from e
  finally:
    q.close()

  if isinstance(x, Exception):
    raise x
  try:
    return pickle.loads(x)
  except Exception as e:
    raise errors.SerializationError(
        'Cannot deserialize the output from sandbox.', e
    ) from e


def maybe_sandbox_call(
    func: Callable[..., Any],
    *args,
    sandbox: Optional[bool] = None,
    timeout: Optional[float] = None,
    **kwargs
) -> Any:
  """Maybe calls a function with sandboxing.

  Args:
    func: Function to call.
    *args: Postional args that will be passed to `func`.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: Execution timeout in seconds. If None, wait the code the complete.
    **kwargs: Keyword args that will be passed to `func`.

  Returns:
    The return value of `func`.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception  that are raised from `func`.
  """
  if sandbox is None:
    try:
      return sandbox_call(func, *args, timeout=timeout, **kwargs)
    # NOTE(daiyip): output could be serialized across processes, giving it
    # already finishes on sandbox, so it should be much safer to run under
    # current process.
    except errors.SerializationError:
      return func(*args, **kwargs)
  elif sandbox:
    return sandbox_call(func, *args, timeout=timeout, **kwargs)
  else:
    return func(*args, **kwargs)


def run(
    code: str,
    *,
    global_vars: Optional[Dict[str, Any]] = None,
    permission: Optional[permissions.CodePermission] = None,
    returns_stdout: bool = False,
    outputs_intermediate: bool = False,
    sandbox: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> Union[Any, Dict[str, Any]]:
  """Executes Python code.

  Features:
    * Fine-grained execution policy for limiting what APIs could be executed.
      This eliminates the need for sandboxing.
    * It exposes both the final results and intermediate results (variables).

  Args:
    code: Python code to run.
    global_vars: An optional dict of
    permission: Permission for the Python code to run.
    returns_stdout: If True, the stdout (a str) will be returned.
    outputs_intermediate: Applicable when returns_stdout is False. If True,
      intermediate output will be outputted as a dict, with the last line's
      value accessible by key '__result__' and the std output accessible by
      key '__stdout__'. Otherwise the value of the last line will be returned.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: Execution timeout in seconds. If None, wait the code the complete.

  Returns:
    The value of the last line of the code block. Or a dict of variable
    names of all locals to their evaluated values as the output of the code to
    run. The value for the last line can be accessed by key '__result__'. Or the
    stdout as a str.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception  that are raised from the code.
  """
  return maybe_sandbox_call(
      evaluate, code=code, global_vars=global_vars, permission=permission,
      returns_stdout=returns_stdout, outputs_intermediate=outputs_intermediate,
      sandbox=sandbox, timeout=timeout
  )
