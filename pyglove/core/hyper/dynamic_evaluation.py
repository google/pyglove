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
"""Dynamic evaluation for hyper primitives."""

import contextlib
import types
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing
from pyglove.core.hyper import base
from pyglove.core.hyper import categorical
from pyglove.core.hyper import custom
from pyglove.core.hyper import numerical
from pyglove.core.hyper import object_template


@contextlib.contextmanager
def dynamic_evaluate(evaluate_fn: Optional[Callable[[base.HyperValue], Any]],
                     yield_value: Any = None,
                     exit_fn: Optional[Callable[[], None]] = None,
                     per_thread: bool = True) -> Iterator[Any]:
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
  if evaluate_fn is not None and not callable(evaluate_fn):
    raise ValueError(
        f'\'evaluate_fn\' must be either None or a callable object. '
        f'Encountered: {evaluate_fn!r}.')
  if exit_fn is not None and not callable(exit_fn):
    raise ValueError(
        f'\'exit_fn\' must be a callable object. Encountered: {exit_fn!r}.')
  old_evaluate_fn = base.get_dynamic_evaluate_fn()
  has_errors = False
  try:
    base.set_dynamic_evaluate_fn(evaluate_fn, per_thread)
    yield yield_value
  except Exception:
    has_errors = True
    raise
  finally:
    base.set_dynamic_evaluate_fn(old_evaluate_fn, per_thread)
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

  class _AnnoymousHyperNameAccumulator:
    """Name accumulator for annoymous hyper primitives."""

    def __init__(self):
      self.index = 0

    def next_name(self):
      name = f'decision_{self.index}'
      self.index += 1
      return name

  def __init__(self,
               where: Optional[Callable[[base.HyperPrimitive], bool]] = None,
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
    self._name_to_hyper: Dict[str, base.HyperPrimitive] = dict()
    self._annoymous_hyper_name_accumulator = (
        DynamicEvaluationContext._AnnoymousHyperNameAccumulator())
    self._hyper_dict = symbolic.Dict() if dna_spec is None else None
    self._dna_spec: Optional[geno.DNASpec] = dna_spec
    self._per_thread = per_thread
    self._decision_getter = None

  @property
  def per_thread(self) -> bool:
    """Returns True if current context collects/applies decisions per thread."""
    return self._per_thread

  @property
  def dna_spec(self) -> geno.DNASpec:
    """Returns the DNASpec of the search space defined so far."""
    if self._dna_spec is None:
      assert self._hyper_dict is not None
      self._dna_spec = object_template.dna_spec(self._hyper_dict)
    return self._dna_spec

  def _decision_name(self, hyper_primitive: base.HyperPrimitive) -> str:
    """Get the name for a decision point."""
    name = hyper_primitive.name
    if name is None:
      if self._require_hyper_name:
        raise ValueError(
            f'\'name\' must be specified for hyper '
            f'primitive {hyper_primitive!r}.')
      name = self._annoymous_hyper_name_accumulator.next_name()
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

    # Ensure per-thread dynamic evaluation context will not be used
    # together with process-level dynamic evaluation context.
    _dynamic_evaluation_stack.ensure_thread_safety(self)

    self._hyper_dict = {}
    with dynamic_evaluate(self.add_decision_point, per_thread=self._per_thread):
      try:
        # Push current context to dynamic evaluatoin stack so nested context
        # can defer unresolved hyper primitive to current context.
        _dynamic_evaluation_stack.push(self)
        yield self._hyper_dict

      finally:
        # Invalidate DNASpec.
        self._dna_spec = None

        # Pop current context from dynamic evaluatoin stack.
        _dynamic_evaluation_stack.pop(self)

  def add_decision_point(self, hyper_primitive: base.HyperPrimitive):
    """Registers a parameter with current context and return its first value."""
    def _add_child_decision_point(c):
      if isinstance(c, types.LambdaType):
        s = pg_typing.get_signature(c)
        if not s.args and not s.has_wildcard_args:
          sub_context = DynamicEvaluationContext(
              where=self._where, per_thread=self._per_thread)
          sub_context._annoymous_hyper_name_accumulator = (  # pylint: disable=protected-access
              self._annoymous_hyper_name_accumulator)
          with sub_context.collect() as hyper_dict:
            v = c()
          return (v, hyper_dict)
      return (c, c)

    if self._where and not self._where(hyper_primitive):
      # Delegate the resolution of hyper primitives that do not pass
      # the `where` predicate to its parent context.
      parent_context = _dynamic_evaluation_stack.get_parent(self)
      if parent_context is not None:
        return parent_context.add_decision_point(hyper_primitive)
      return hyper_primitive

    if isinstance(hyper_primitive, object_template.ObjectTemplate):
      return hyper_primitive.value

    assert isinstance(hyper_primitive, base.HyperPrimitive), hyper_primitive
    name = self._decision_name(hyper_primitive)
    if isinstance(hyper_primitive, categorical.Choices):
      candidate_values, candidates = zip(
          *[_add_child_decision_point(c) for c in hyper_primitive.candidates])
      if hyper_primitive.choices_distinct:
        assert hyper_primitive.num_choices <= len(hyper_primitive.candidates)
        v = [candidate_values[i] for i in range(hyper_primitive.num_choices)]
      else:
        v = [candidate_values[0]] * hyper_primitive.num_choices
      hyper_primitive = hyper_primitive.clone(deep=True, override={
          'candidates': list(candidates)
      })
      first_value = v[0] if isinstance(
          hyper_primitive, categorical.OneOf) else v
    elif isinstance(hyper_primitive, numerical.Float):
      first_value = hyper_primitive.min_value
    else:
      assert isinstance(hyper_primitive, custom.CustomHyper), hyper_primitive
      first_value = hyper_primitive.decode(hyper_primitive.first_dna())

    if (name in self._name_to_hyper
        and hyper_primitive != self._name_to_hyper[name]):
      raise ValueError(
          f'Found different hyper primitives under the same name {name!r}: '
          f'Instance1={self._name_to_hyper[name]!r}, '
          f'Instance2={hyper_primitive!r}.')
    self._hyper_dict[name] = hyper_primitive
    self._name_to_hyper[name] = hyper_primitive
    return first_value

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
        if (not isinstance(hyper_primitive, categorical.Choices)
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

        if (isinstance(hyper_primitive, numerical.Float)
            and not isinstance(decision, float)):
          raise ValueError(
              f'Expect float-type decision for {hyper_primitive!r}, '
              f'encoutered {decision!r}.')
        if (isinstance(hyper_primitive, custom.CustomHyper)
            and not isinstance(decision, str)):
          raise ValueError(
              f'Expect string-type decision for {hyper_primitive!r}, '
              f'encountered {decision!r}.')
        if (isinstance(hyper_primitive, categorical.Choices)
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

  @contextlib.contextmanager
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

    Yields:
      None
    """
    if not isinstance(decisions, (geno.DNA, list)):
      raise ValueError('`decisions` should be a DNA or a list of numbers.')

    # Ensure per-thread dynamic evaluation context will not be used
    # together with process-level dynamic evaluation context.
    _dynamic_evaluation_stack.ensure_thread_safety(self)

    get_current_decision, evaluation_finalizer = (
        self._decision_getter_and_evaluation_finalizer(decisions))

    has_errors = False
    with dynamic_evaluate(self.evaluate, per_thread=self._per_thread):
      try:
        # Set decision getter for current decision.
        self._decision_getter = get_current_decision

        # Push current context to dynamic evaluation stack so nested context
        # can delegate evaluate to current context.
        _dynamic_evaluation_stack.push(self)

        yield
      except Exception:
        has_errors = True
        raise
      finally:
        # Pop current context from dynamic evaluatoin stack.
        _dynamic_evaluation_stack.pop(self)

        # Reset decisions.
        self._decision_getter = None

        # Call evaluation finalizer to make sure all decisions are used.
        if not has_errors:
          evaluation_finalizer()

  def evaluate(self, hyper_primitive: base.HyperPrimitive):
    """Evaluates a hyper primitive based on current decisions."""
    if self._decision_getter is None:
      raise ValueError(
          '`evaluate` needs to be called under the `apply` context.')

    get_current_decision = self._decision_getter
    def _apply_child(c):
      if isinstance(c, types.LambdaType):
        s = pg_typing.get_signature(c)
        if not s.args and not s.has_wildcard_args:
          return c()
      return c

    if self._where and not self._where(hyper_primitive):
      # Delegate the resolution of hyper primitives that do not pass
      # the `where` predicate to its parent context.
      parent_context = _dynamic_evaluation_stack.get_parent(self)
      if parent_context is not None:
        return parent_context.evaluate(hyper_primitive)
      return hyper_primitive

    if isinstance(hyper_primitive, numerical.Float):
      return get_current_decision(hyper_primitive)

    if isinstance(hyper_primitive, custom.CustomHyper):
      return hyper_primitive.decode(
          geno.DNA(get_current_decision(hyper_primitive)))

    assert isinstance(hyper_primitive, categorical.Choices), hyper_primitive
    value = symbolic.List()
    for i in range(hyper_primitive.num_choices):
      # NOTE(daiyip): during registering the hyper primitives when
      # constructing the search space, we will need to evaluate every
      # candidate in order to pick up sub search spaces correctly, which is
      # not necessary for `pg.DynamicEvaluationContext.apply`.
      value.append(_apply_child(
          hyper_primitive.candidates[get_current_decision(hyper_primitive, i)]))
    if isinstance(hyper_primitive, categorical.OneOf):
      assert len(value) == 1
      value = value[0]
    return value


# We maintain a stack of dynamic evaluation context for support search space
# combination
class _DynamicEvaluationStack:
  """Dynamic evaluation stack used for dealing with nested evaluation."""

  _TLS_KEY = 'dynamic_evaluation_stack'

  def __init__(self):
    self._global_stack = []

  def ensure_thread_safety(self, context: DynamicEvaluationContext):
    if ((context.per_thread and self._global_stack)
        or (not context.per_thread and self._local_stack)):
      raise ValueError(
          'Nested dynamic evaluation contexts must be either all per-thread '
          'or all process-wise. Please check the `per_thread` argument of '
          'the `pg.hyper.DynamicEvaluationContext` objects being used.')

  @property
  def _local_stack(self):
    """Returns thread-local stack."""
    stack = object_utils.thread_local_get(self._TLS_KEY, None)
    if stack is None:
      stack = []
      object_utils.thread_local_set(self._TLS_KEY, stack)
    return stack

  def push(self, context: DynamicEvaluationContext):
    """Pushes the context to the stack."""
    stack = self._local_stack if context.per_thread else self._global_stack
    stack.append(context)

  def pop(self, context: DynamicEvaluationContext):
    """Pops the context from the stack."""
    stack = self._local_stack if context.per_thread else self._global_stack
    assert stack
    stack_top = stack.pop(-1)
    assert stack_top is context, (stack_top, context)

  def get_parent(
      self,
      context: DynamicEvaluationContext) -> Optional[DynamicEvaluationContext]:
    """Returns the parent context of the input context."""
    stack = self._local_stack if context.per_thread else self._global_stack
    parent = None
    for i in reversed(range(1, len(stack))):
      if context is stack[i]:
        parent = stack[i - 1]
        break
    return parent


# System-wise dynamic evaluation stack.
_dynamic_evaluation_stack = _DynamicEvaluationStack()


def trace(
    fun: Callable[[], Any],
    *,
    where: Optional[Callable[[base.HyperPrimitive], bool]] = None,
    require_hyper_name: bool = False,
    per_thread: bool = True) -> DynamicEvaluationContext:
  """Trace the hyper primitives called within a function by executing it.

  See examples in :class:`pyglove.hyper.DynamicEvaluationContext`.

  Args:
    fun: Function in which the search space is defined.
    where: A callable object that decide whether a hyper primitive should be
      included when being instantiated under `collect`.
      If None, all hyper primitives under `collect` will be included.
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
      where=where, require_hyper_name=require_hyper_name, per_thread=per_thread)
  with context.collect():
    fun()
  return context

