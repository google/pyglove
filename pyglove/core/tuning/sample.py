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
"""Distributed sampling."""

from typing import Any, Callable, Optional, Sequence, Union

from pyglove.core import geno
from pyglove.core import hyper
from pyglove.core import symbolic
from pyglove.core.tuning import backend as backend_lib
from pyglove.core.tuning.early_stopping import EarlyStoppingPolicy


# A hyper value is a symbolic value that contains objects of
# `pg.hyper.HyperValue`. Since it is a composition constraint instead of
# a type constraint, we use `Any` as its pytype annotation for now.
HyperValue = Any


def sample(space: Union[HyperValue,
                        hyper.DynamicEvaluationContext,
                        geno.DNASpec],
           algorithm: geno.DNAGenerator,
           num_examples: Optional[int] = None,
           early_stopping_policy: Optional[EarlyStoppingPolicy] = None,
           where: Optional[Callable[[hyper.HyperPrimitive], bool]] = None,
           name: Optional[str] = None,
           group: Union[None, int, str] = None,
           backend: Optional[str] = None,
           metrics_to_optimize: Optional[Sequence[str]] = None,
           **kwargs):
  """Yields an example and its feedback sampled from a hyper value.

  Example 1: sample a search space defined by a symbolic hyper value::

    for example, feedback in pg.sample(
        pg.Dict(x=pg.floatv(-1, 1))),
        pg.geno.Random(),
        num_examples=10,
        name='my_search'):

      # We can access trial ID (staring from 1) and DNA from the feedback.
      print(feedback.id, feedback.dna)

      # We can report the reward computed on the example using
      # `feedback.add_measurement`, which can be called
      # multiple times to report the rewards incrementally.
      # Once a trial is done, we call `feedback.done` to mark evaluation on
      # current example as completed, or use `feedback.skip` to move to the
      # next sample without passing any feedback to the algorithm.
      # Without `feedback.done` or `feedback.skip`, the same trial will be
      # iterated over and over again for failover handling purpose.
      # Besides the reward and the step, metrics and checkpoint can be added
      # to each measurement. Additional meta-data and related links (URLs) can
      # be passed to `feedback.done` which can be retrieved via
      # `pg.poll_result` later.
      if example.x >= 0:
        # If we only want to add one measurement for each example, a
        # shortcut expression for the next two lines can be written as
        # follows:
        #   `feedback(reward=math.sqrt(example.x), step=1)`
        feedback.add_measurement(reward=math.sqrt(example.x), step=1)
        feedback.done()
      else:
        feedback.skip()

      # IMPORTANT: to stop the loop among all workers, we can call
      # `feedback.end_loop`. As a result, each worker will quit their loop
      # after current iteration, while using `break` in the for-loop is
      # only effective for the local process rather than remotely.
      if session.id > 1000:
        feedback.end_loop()

    # At any time, we can poll the search result via `pg.poll_result`, please
    # see `pg.tuning.Result` for more details.
    result = pg.poll_result('my_search')
    print(result.best_trial)

  Example 2: sample a search space defined by `pg.hyper.trace`::

    def fun():
      return pg.oneof([
        lambda: pg.oneof([1, 2, 3]),
        lambda: pg.float(0.1, 1.0),
        3]) + sum(pg.manyof(2, [1, 2, 3]))

    for example, feedback in pg.sample(
        pg.hyper.trace(fun),
        pg.geno.Random(),
        num_examples=10,
        name='my_search'):
      # When space is a `pg.hyper.DynamicEvaluationContext` object,
      # the `example` yielded at each iteration is a context manager under which
      # the hyper primitives (e.g. pg.oneof) will be materialized into concrete
      # values according to the controller decision.
      with example():
        reward = fun()
      feedback(reward)

  Example 3: sample DNAs from an abstract search space represented by
  `pg.DNASpec`::

    for dna, feedback in pg.sample(
        pg.List([pg.oneof(range(3))] * 5).dna_spec(),
        pg.geno.Random(),
        num_examples=10,
        name='my_search'):
      reward = evaluate_dna(dna)
      feedback(reward)

  **Using `pg.sample` in distributed environment**

  `pg.sample` is designed with distributed sampling in mind, in which multiple
  processes can work on the trials of the same sampling loop. While the default
  'in-memory' backend works only within a single process without failover
  handling, other backends may support distributed computing environment with
  persistent state. Nevertheless, the `pg.sample` API is the same
  among different backends, users can switch the backend easily by passing a
  different value to the `backend` argument, or set the default value globally
  via `pg.tuning.set_default_backend`.


  **Identifying a sampling**

  To identify a distributed loop, a unique `name` is introduced, which will
  also be used to poll the latest sampling result via `pg.poll_result`.

  **Failover handling**

  In a distributed setup, worker processes may incidentally die and restart.
  Unless a trial is explicitly marked as done (via `feedback(reward)` or
  `feedback.done()`) or skipped (via feedback.skip()), a worker will try to
  resume its work on the trial from where it left off.

  **Workroup**

  In a distributed setup, worker processes may or may not work on the same
  trials. Worker group is introduced to serve this purpose, which is identified
  by an integer or a string named `group`. If `group` is not specified or
  having different values among the workers, every worker will work on
  different trials of the loop. On the contrary, workers having the same
  `group` will co-work on the same trials. Group is useful when evaluation on
  one example can be parallelized - for example - an example in the outer loop
  of a nested search. However, feedback on the same example should be fed back
  to the search algorithm only once. Therefore, workers in the same group need
  to communicate with each other to avoid duplicated evaluation and feedbacks.
  To facilitate such communication, per-trial metadata is supported, and can be
  accessed via `feedback.set_metadata/get_metadata` methods. The consistency in
  reading and writing the metadata is defined by the backend used. For the
  'in-memory' backend, all the trials and their metadata is stored in memory,
  thus will be lost if the process get restarted. On the contrary, the backends
  built on distributed computing environment may store both the trials and
  metadata in a persistent storage with varing read/write QPS and read/write
  consistency guarentees.

  **Switch between backends**

  The `backend` argument of `pg.sample` lets users choose a backend used in
  current sampling loop. Users can use different backends in the same process
  to achieve a best performance trade-off, e.g., using `in-memory` backend when
  the communication cost overweighs the redudant evaluation cost upon worker
  failure.

  Helper function :func:`pyglove.tuning.set_default_backend` is introduced to
  set the default tuning backend for the entire process.

  Args:
    space: One of (a hyper value, a `pg.hyper.DynamicEvaluationContext`,
      a `pg.DNASpec`) to sample from.
      A hyper value is an object with to-be-determined values specified by
      `pg.oneof`, `pg.manyof`, `pg.floatv` and etc, representing a search space.
      A `pg.hyper.DynamicEvaluationContext` object represents a search space
      that is traced via dynamic evaluation.
      A `pg.DNASpec` represents an abstract search space that emits DNAs.
    algorithm: The search algorithm that samples the search space. For example:
      `pg.geno.Random()`, `pg.evolution.regularized_evolution(...)`, and
      etc.
    num_examples: An optional integer as the max number of examples to
      sample. If None, sample will return an iterator of infinite examples.
    early_stopping_policy: An optional early stopping policy for user to tell
      if incremental evaluation (which reports multiple measurements) on each
      example can be early short circuited.
      After each call to `feedback.add_measurement`, users can use method
      `feedback.should_stop_early` to check whether current example worth
      further evaluation or not.
    where: Function to filter the hyper values. If None, all decision points
      from the `space` will be included for the algorithm to make
      decisions. Otherwise only the decision points on which 'where' returns
      True will be included. The rest decision points will be passed through
      in the example, intact, which is a sub-space of the search space
      represented by the `space`. `where` is usually used in nested search
      flows. Please see 'hyper.Template' docstr for details.
    name: A string as a unique identifier for current sampling. Two separate
      calls to `pg.sample` with the same `name` (also the same algorithm) will
      share the same sampling queue, whose examples are proposed by the same
      search algorithm.
    group: An string or integer as the group ID of current process in
      distributed sampling, which will be used to group different workers into
      co-worker groups. Workers with the same group id will work on the same
      trial. On the contrary, workers in different groups will always be working
      with different trials. If not specified, each worker in current sampling
      will be in different groups. `group` is usually used in the outer
      loops of nested search, in order to allow workers to work on the same
      higher-order item.
    backend: An optional string to specify the backend name for sampling.
      if None, the default backend set by `pg.tuning.set_default_backend`
      will be used.
    metrics_to_optimize: A sequence of string as the names of the metrics
      to be optimized by the algorithm, which is ['reward'] by default.
      When specified, it should have only 1 item for single-objective algorithm
      and can have multiple items for algorithms that support multi-objective
      optimization.
    **kwargs: Arguments passed to the `BackendFactory` subclass registered with
      the requested backend.

  Yields:
    An iterator of tuples (example, feedback) as examples sampled from the
    search space defined by the `space` through `algorithm`.

  Raises:
    ValueError: `space` is a fixed value, or requested `backend` is not
    available.
  """
    # Placeholder for Google-internal usage instrumentation.

  # Create template based on the hyper value.
  if isinstance(space, hyper.DynamicEvaluationContext):
    dynamic_evaluation_context = space
    dna_spec = space.dna_spec
    template = None
  elif isinstance(space, geno.DNASpec):
    dynamic_evaluation_context = None
    dna_spec = space
    template = None
  else:
    if symbolic.is_deterministic(space):
      raise ValueError(f'\'space\' is a constant value: {space!r}.')
    template = hyper.template(space, where)
    dna_spec = template.dna_spec()
    dynamic_evaluation_context = None

  # Create and set up the backend.
  metrics_to_optimize = metrics_to_optimize or ['reward']
  backend = backend_lib.get_backend_cls(backend).create(
      name, group, dna_spec, algorithm, metrics_to_optimize,
      early_stopping_policy, num_examples, **kwargs)

  while True:
    try:
      feedback = backend.next()
      dna = feedback.dna
      reward = dna.metadata.get('reward')
      if reward is None:
        # Decode and return current example to client code for evaluation.
        if template is not None:
          value = template.decode(dna)
        elif dynamic_evaluation_context is not None:
          value = lambda: dynamic_evaluation_context.apply(dna)
        else:
          value = dna
        yield (value, feedback)
      else:
        # Reward may be computed at the controller side, we can
        # short circuit the client-side evaluation for the current item.
        # Also, there can be multiple co-workers working on the same trial,
        # we ignore errors triggered by racing feedbacks, which will be
        # considered as no-op.
        with feedback.ignore_race_condition():
          feedback(reward, metadata=dict(client_evaluation_skipped=True))
    except StopIteration:
      return
