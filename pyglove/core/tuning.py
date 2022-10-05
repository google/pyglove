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
"""Distributed tuning with pluggable backends.

:func:`pyglove.iter` provides an interface for sampling examples from a search
space within a process. To support distributed tuning, PyGlove introduces
:func:`pyglove.sample`, which is almost identical but with more features:

 * Allow multiple worker processes (aka. workers) to collaborate on a search
   with failover handling.
 * Each worker can process different trials, or can cowork on the same trials
   via work groups.
 * Provide APIs for communicating between the co-workers.
 * Provide API for retrieving the search results.
 * Provide a pluggable backend system for supporting user infrastructures.

"""

import abc
import collections
import contextlib
import datetime
import inspect
import re
import threading
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Text, Tuple, Type, Union

from pyglove.core import geno
from pyglove.core import hyper
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as schema


# Disable implicit str concat in Tuple as it's used for multi-line docstr for
# symbolic members.
# pylint: disable=implicit-str-concat


#
# Interfaces and classes for tuning results.
#


class _DataEntity(symbolic.Object):
  """Base class for object that is used as data entity."""

  # Allow assignment on symbolic attributes.
  allow_symbolic_assignment = True

  def __hash__(self):
    """Hash code."""
    return hash(repr(self))

  def __str__(self):
    """Overrides __str__ to use non-verbose format."""
    return self.format(compact=False, verbose=False)


@symbolic.members([
    ('step', schema.Int(), 'At which step the result is reported.'),
    ('elapse_secs', schema.Float(), 'Elapse in seconds since trial start.'),
    ('reward', schema.Float().noneable(),
     'Reward of reported tunable target. Can be None if multi-objective '
     'optimization is used.'),
    ('metrics', schema.Dict([
        (schema.StrKey(), schema.Float(), 'Metric item.')
    ]).noneable(), 'Metric in key/value pairs (optional).'),
    ('checkpoint_path', schema.Str().noneable(),
     'Path to the checkpoint of this specific measurement.')
])
class Measurement(_DataEntity):
  """Measurement of a trial at certain step."""


@symbolic.members([
    ('id', schema.Int(), 'Identifier of the trial.'),
    ('description', schema.Str().noneable(), 'Description of the trial.'),
    ('dna', schema.Object(geno.DNA), 'Proposed DNA for the trial.'),
    ('status',
     schema.Enum('PENDING', [
         'UNKNOWN',
         'REQUESTED',
         'PENDING',
         'COMPLETED',
         'DELETED',
         'STOPPING',
     ]), 'Trial status.'),
    ('final_measurement', schema.Object(Measurement).noneable(),
     'Reported final results.'),
    ('infeasible', schema.Bool(False), 'Whether trial is infeasible.'),
    ('measurements', schema.List(schema.Object(Measurement),
                                 default=[]), 'All reported measurements.'),
    ('metadata',
     schema.Dict(
         [(schema.StrKey(), schema.Any(),
           'Serializable key value pairs as metadata.')]),
     'Trial metadata.'),
    ('related_links',
     schema.Dict([
         (schema.StrKey(), schema.Str(), 'Related link.')]),
     'Related links'),
    # TODO(daiyip): consider change time from timestamp to datetime.
    # Need to introduce a mechanism in symbolic to cherry pick serialization
    # for individual fields.
    ('created_time', schema.Int(), 'Created time in Unix timestamp.'),
    ('completed_time', schema.Int().noneable(),
     'Completed time in Unix timestamp.'),
])
class Trial(_DataEntity):
  """Metadata of a trial."""

  def get_reward_for_feedback(
      self, metric_names: Optional[Sequence[Text]] = None
      ) -> Union[None, float, Tuple[float]]:
    """Get reward for feedback."""
    if self.status != 'COMPLETED' or self.infeasible:
      return None
    assert self.final_measurement is not None
    measurement = self.final_measurement
    if metric_names is None:
      return measurement.reward
    assert metric_names, metric_names
    metric_values = []
    for metric_name in metric_names:
      if metric_name == 'reward':
        v = measurement.reward
      else:
        v = measurement.metrics.get(metric_name, None)
      if v is None:
        raise ValueError(
            f'Metric {metric_name!r} does not exist in final '
            f'measurement {measurement!r} in trial {self.id}.')
      metric_values.append(v)
    return tuple(metric_values) if len(metric_values) > 1 else metric_values[0]


class Result(object_utils.Formattable):
  """Interface for tuning result."""

  @property
  @abc.abstractmethod
  def metadata(self) -> Dict[Text, Any]:
    """Returns the metadata of current sampling."""

  @property
  @abc.abstractmethod
  def is_active(self) -> bool:
    """Returns whether the tuner task is active."""

  @property
  @abc.abstractmethod
  def last_updated(self) -> datetime.datetime:
    """Last updated time."""

  @property
  @abc.abstractmethod
  def trials(self) -> List[Trial]:
    """Retrieve all trials."""

  @property
  @abc.abstractmethod
  def best_trial(self) -> Optional[Trial]:
    """Get best trial so far."""


#
# Interface for early stopping policy.
#


class EarlyStoppingPolicy(symbolic.Object):
  """Interface for early stopping policy."""

  def setup(self, dna_spec: geno.DNASpec) -> None:
    """Setup states of an early stopping policy based on dna_spec.

    Args:
      dna_spec: DNASpec for DNA to propose.

    Raises:
      RuntimeError: if dna_spec is not supported.
    """
    self._dna_spec = dna_spec

  @property
  def dna_spec(self) -> Optional[geno.DNASpec]:
    return getattr(self, '_dna_spec', None)

  @abc.abstractmethod
  def should_stop_early(self, trial: Trial) -> bool:
    """Should stop the input trial early based on its measurements."""

  def recover(self, history: Iterable[Trial]) -> None:
    """Recover states by replaying the trial history. Subclass can override.

    NOTE: `recover` will always be called before the first `should_stop_early`
    is called. It could be called multiple times if there are multiple source
    of history, e.g: trials from a previous study and existing trials from
    current study.

    The default behavior is to replay `should_stop_early` on all intermediate
    measurements on all trials.

    Args:
      history: An iterable object of trials.
    """
    for trial in history:
      if trial.status in ['COMPLETED', 'PENDING', 'STOPPING']:
        self.should_stop_early(trial)


#
# Interfaces for tuning backends.
#


class Feedback(metaclass=abc.ABCMeta):
  """Interface for the feedback object for a trial.

  Feedback object is an agent to communicate to the search algorithm and other
  workers based on current trial, which includes:

  * Information about current example:

    * :attr:`id`: The ID of current example, started from 1.
    * :attr:`dna`: The DNA for current example.

  * Methods to communicate with the search algorithm:

    * :meth:`add_measurement`: Add a measurement for current example.
      Multiple measurements can be added as progressive evaluation of the
      example, which can be used by the early stopping policy to suggest
      whether current evaluation can be stopped early.
    * :meth:`done`: Mark evaluation on current example as done, use the
      reward from the latest measurement to feedback to the algorithm, and
      move to the next example.
    * :meth:`__call__`: A shortcut method that calls :meth:`add_measurement`
      and :meth:`done` in sequence.
    * :meth:`skip`: Mark evaluation on current example as done, and move to
      the next example without providing feedback to the algorithm.
    * :meth:`should_stop_early`: Tell if progressive evaluation on current
      example can be stopped early.
    * :meth:`end_loop`: Mark the loop as done. All workers will get out of
      the loop after they finish evaluating their current examples.

  * Methods to publish information associated with current trial:

    * :meth:`set_metadata`: Set persistent metadata by key.
    * :meth:`get_metadata`: Get persistent metadata by key.
    * :meth:`add_link`: Add a related link by key.
  """

  def __init__(self, metrics_to_optimize: Sequence[Text]):
    super().__init__()
    self._metrics_to_optimize = metrics_to_optimize
    self._sample_time = time.time()

  @property
  @abc.abstractmethod
  def id(self) -> int:
    """Gets the ID of current trial."""

  @property
  @abc.abstractmethod
  def dna(self) -> geno.DNA:
    """Gets the DNA of the example used in current trial."""

  @property
  @abc.abstractmethod
  def checkpoint_to_warm_start_from(self) -> Optional[Text]:
    """Gets checkpoint path to warm start from."""

  def add_measurement(
      self,
      reward: Union[None, float, Sequence[float]] = None,
      metrics: Optional[Dict[Text, float]] = None,
      step: int = 0,
      checkpoint_path: Optional[Text] = None,
      elapse_secs: Optional[float] = None) -> None:
    """Add a measurement for current trial.

    This method can be called multiple times on the same trial, e.g::

      for model, feedback in pg.sample(...):
        accuracy = train_and_evaluate(model, step=10)
        feedback.add_measurement(accuracy, step=10)
        accuracy = train_and_evaluate(model, step=15)
        feedback.add_measurement(accuracy, step=25)
        feedback.done()

    Args:
      reward: An optional float value as the reward for single-objective
        optimization, or a sequence of float values for multiple objectives
        optimization. In multiple-objective scenario, the float sequence will
        be paired up with the `metrics_to_optimize` argument of `pg.sample`,
        thus their length must be equal.
        Another way for providing reward for multiple-objective reward is
        through the `metrics` argument, which is a dict using metric name as key
        and its measure as value (the key should match with an element of the
        `metrics_to_optimize` argument). When multi-objective reward is provided
        from both the `reward` argument (via a sequence of float) and the
        `metrics` argument, their value should agree with each other.
      metrics: An optional dictionary of string to float as metrics. It can
        be used to provide metrics for multi-objective optimization, and/or
        carry additional metrics for study analysis.
      step: An optional integer as the step (e.g. step for model training),
        at which the measurement applies. When a trial is completed, the
        measurement at the largest step will be chosen as the final measurement
        to feed back to the controller.
      checkpoint_path: An optional string as the checkpoint path produced
        from the evaluation (e.g. training a model), which can be used in
        transfer learning.
      elapse_secs: Time spent on evaluating current example so far. If None,
        it will be automatically computed by the backend.
    """
    metrics_to_optimize = self._metrics_to_optimize
    metrics = metrics or {}

    if isinstance(reward, (list, tuple)):
      rewards = reward
      if len(rewards) != len(metrics_to_optimize):
        raise ValueError(
            f'The number of items in the reward ({rewards!r}) computed by the '
            f'controller does not match with the number of metrics to '
            f'optimize ({metrics_to_optimize!r}).')
      for k, v in zip(metrics_to_optimize, rewards):
        if k in metrics and metrics[k] != v:
          raise ValueError(
              f'The value for metric {k} is provided from both the \'reward\' '
              f'and the \'metrics\' arguments with different values: '
              f'{[v, metrics[k]]!r}.')
        metrics[k] = v
      reward = metrics.pop('reward', None)
    elif reward is not None:
      reward = float(reward)

    for metric_name in metrics_to_optimize:
      if metric_name == 'reward':
        if reward is None:
          raise ValueError(
              '\'reward\' must be provided as it is a goal to optimize.')
      elif metric_name in metrics:
        metrics[metric_name] = float(metrics[metric_name])
      else:
        raise ValueError(
            f'Metric {metric_name!r} must be provided '
            f'as it is a goal to optimize.')

    if len(metrics_to_optimize) == 1 and metrics_to_optimize[0] != 'reward':
      if reward is None:
        reward = metrics[metrics_to_optimize[0]]
      else:
        raise ValueError(
            f'\'reward\' {reward!r} is provided while it is '
            f'not a goal to optimize.')

    if elapse_secs is None:
      elapse_secs = time.time() - self._sample_time

    self._add_measurement(
        reward, metrics, step, checkpoint_path, elapse_secs)

  def _add_measurement(
      self,
      reward: Optional[float],
      metrics: Dict[Text, float],
      step: int,
      checkpoint_path: Optional[Text],
      elapse_secs: float) -> None:
    """Child class should implement."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_trial(self) -> Trial:
    """Gets current Trial.

    Returns:
      An up-to-date `Trial` object. A distributed tuning backend should make
      sure the return value is up-to-date not only locally, but among different
      workers.
    """

  @abc.abstractmethod
  def set_metadata(self, key: Text, value: Any, per_trial: bool = True) -> None:
    """Sets metadata for current trial or current sampling.

    Metadata can be used in two use cases:

     * Worker processes that co-work on the same trial can use meta-data to
       communicate with each other.
     * Worker use metadata as a persistent store to save information for
       current trial, which can be retrieved via `poll_result` method later.

    Args:
      key: A string as key to metadata.
      value: A value that can be serialized by `pg.to_json_str`.
      per_trial: If True, the key is set per current trial. Otherwise, it
        is set per current sampling loop.
    """

  @abc.abstractmethod
  def get_metadata(self, key: Text, per_trial: bool = True) -> Optional[Any]:
    """Gets metadata for current trial or current sampling.

    Args:
      key: A string as key to metadata.
      per_trial: If True, the key is retrieved per curent trial. Otherwise, it
        is retrieved per current sampling.

    Returns:
      A value that can be deserialized by `pg.from_json_str`.
    """

  @abc.abstractmethod
  def add_link(self, name: Text, url: Text) -> None:
    """Adds a related link to current trial.

    Added links can be retrieved from the `Trial.related_links` property via
    `pg.poll_result`.

    Args:
      name: Name for the related link.
      url: URL for this link.
    """

  @abc.abstractmethod
  def done(self,
           metadata: Optional[Dict[Text, Any]] = None,
           related_links: Optional[Dict[Text, Text]] = None) -> None:
    """Marks current trial as done.

    Args:
      metadata: Additional metadata to add to current trial.
      related_links: Additional links to add to current trial.
    """

  @abc.abstractmethod
  def skip(self, reason: Optional[Text] = None) -> None:
    """Move to next example without providing the feedback to the algorithm."""

  @abc.abstractmethod
  def should_stop_early(self) -> bool:
    """Whether progressive evaluation can be stopped early on current trial."""

  @abc.abstractmethod
  def end_loop(self) -> None:
    """Ends current sapling loop."""

  def __call__(
      self,
      reward: Union[None, float, Sequence[float]] = None,
      metrics: Optional[Dict[Text, float]] = None,
      checkpoint_path: Optional[Text] = None,
      metadata: Optional[Dict[Text, Any]] = None,
      related_links: Optional[Dict[Text, Text]] = None,
      step: int = 0) -> None:
    """Adds a measurement and marks the trial as done."""
    self.add_measurement(
        reward, metrics, step=step, checkpoint_path=checkpoint_path)
    self.done(metadata=metadata, related_links=related_links)

  @contextlib.contextmanager
  def skip_on_exceptions(
      self, exceptions: Sequence[
          Union[Type[Exception], Tuple[Exception, str]]]):
    """Yield skip on exceptions.

    Usages::

      with feedback.skip_on_exceptions((ValueError, KeyError)):
        ...

      with feedback.skip_on_exceptions(((ValueError, 'bad value for .*'),
                                        (ValueError, '.* invalid range'),
                                        TypeError)):
        ...

    Args:
      exceptions: A sequence of (exception type, or exception type plus regular
        expression for error message).

    Yields:
      None.
    """
    error_mapping: Dict[Type[Exception], List[str]] = {}
    for error_type in exceptions:
      regex = None
      if isinstance(error_type, tuple):
        assert len(error_type) == 2, error_type
        error_type, regex = error_type
      if not (inspect.isclass(error_type)
              and issubclass(error_type, Exception)):
        raise TypeError(f'Exception contains non-except types: {error_type!r}.')
      if error_type not in error_mapping:
        error_mapping[error_type] = []
      if regex is not None:
        error_mapping[error_type].append(regex)

    try:
      yield
    except tuple(error_mapping.keys()) as e:
      error_message = str(e)
      found_match = False
      for error_type, error_regexes in error_mapping.items():
        if isinstance(e, error_type):
          if not error_regexes:
            found_match = True
          else:
            for regex in error_regexes:
              if re.match(regex, error_message):
                found_match = True
                break
      if found_match:
        self.skip(traceback.format_exc())
      else:
        raise

  @contextlib.contextmanager
  def ignore_race_condition(self):
    """Context manager for ignoring RaceConditionError within the scope.

    Race condition may happen when multiple workers are working on the same
    trial (e.g. paired train/eval processes). Assuming there are two co-workers
    (X and Y), common race conditions are:

    1) Both X and Y call `feedback.done` or `feedback.skip` to the same trial.
    2) X calls `feedback.done`/`feedback.skip`, then B calls
     `feedback.add_measurement`.

    Users can use this context manager to simplify the code for handling
    multiple co-workers. (See the `group` argument of `pg.sample`)

    Usages::
      feedback = ...
      def thread_fun():
        with feedback.ignore_race_condition():
          feedback.add_measurement(0.1)

          # Multiple workers working on the same trial might trigger this code
          # from different processes.
          feedback.done()

      x = threading.Thread(target=thread_fun)
      x.start()
      y = threading.Thread(target=thread_fun)
      y.start()

    Yields:
      None.
    """
    try:
      yield
    except RaceConditionError:
      pass


class RaceConditionError(RuntimeError):
  """Race condition error.

  This error will be raisen when the operations made to `Feedback` indicates
  a race condition. There are possible scenarios that may lead to such race
  conditions, which happen among multiple co-workers (taking X and Y for
  example) on the same trial:

  * X calls `feedback.done`/`feedback.skip`, then B calls
   `feedback.add_measurement`.
  """


class Backend(metaclass=abc.ABCMeta):
  """Interface for the tuning backend."""

  @abc.abstractmethod
  def setup(self,
            name: Optional[Text],
            group_id: Optional[Text],
            dna_spec: geno.DNASpec,
            algorithm: geno.DNAGenerator,
            metrics_to_optimize: Sequence[Text],
            early_stopping_policy: Optional[EarlyStoppingPolicy] = None,
            num_examples: Optional[int] = None) -> None:
    """Setup current backend for an existing or a new sampling.

    Args:
      name: An unique string as the identifier for the sampling instance.
      group_id: An optional group id for current process.
      dna_spec: DNASpec for current sampling.
      algorithm: Search algorithm used for current sampling.
      metrics_to_optimize: metric names to optimize.
      early_stopping_policy: An optional early stopping policy.
      num_examples: Max number of examples to sample. Infinite if None.
    """

  @abc.abstractmethod
  def next(self) -> Feedback:
    """Get the feedback object for the next sample."""


class BackendFactory(metaclass=abc.ABCMeta):
  """Interface for tuning backend factory."""

  @abc.abstractmethod
  def create(self, **kwargs) -> Backend:
    """Creates a tuning backend for an existing or a new sampling.

    Args:
      **kwargs: Backend-specific keyword arguments passed from `pg.sample`.
    """

  @abc.abstractmethod
  def poll_result(self, name: Text) -> Result:
    """Gets tuning result by a unique tuning identifier."""


_backend_registry = dict()
_default_backend_name = 'in-memory'


def add_backend(backend_name: Text):
  """Decorator to register a backend factory with name."""
  def _decorator(factory_cls):
    if not issubclass(factory_cls, BackendFactory):
      raise TypeError(f'{factory_cls!r} is not a BackendFactory subclass.')
    _backend_registry[backend_name] = factory_cls
    return factory_cls
  return _decorator


def available_backends() -> List[Text]:
  """Gets available backend names."""
  return list(_backend_registry.keys())


def set_default_backend(backend_name: Text):
  """Sets the default tuning backend name."""
  if backend_name not in _backend_registry:
    raise ValueError(f'Backend {backend_name!r} does not exist.')
  global _default_backend_name
  _default_backend_name = backend_name


def default_backend() -> Text:
  """Gets the default tuning backend name."""
  return _default_backend_name


def _create_backend_factory(backend_name: Text) -> BackendFactory:
  """Get backend by name."""
  backend_name = backend_name or default_backend()
  if backend_name not in _backend_registry:
    raise ValueError(f'Backend {backend_name!r} does not exist.')
  return _backend_registry[backend_name]()


def sample(hyper_value: Any,
           algorithm: geno.DNAGenerator,
           num_examples: Optional[int] = None,
           early_stopping_policy: Optional[EarlyStoppingPolicy] = None,
           where: Optional[Callable[[hyper.HyperPrimitive], bool]] = None,
           name: Optional[Text] = None,
           group: Union[None, int, Text] = None,
           backend: Optional[Text] = None,
           metrics_to_optimize: Optional[Sequence[Text]] = None,
           **kwargs):
  """Yields an example and its feedback sampled from a hyper value.

  Example 1: sample a search space defined by a symbolic hyper value::

    for example, feedback in pg.sample(
        hyper_value=pg.Dict(x=pg.floatv(-1, 1))),
        algorithm=pg.generators.Random(),
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
        hyper_value=pg.hyper.trace(fun),
        algorithm=pg.generators.Random(),
        num_examples=10,
        name='my_search'):
      # When hyper_value is a `pg.hyper.DynamicEvaluationContext` object,
      # the `example` yielded at each iteration is a context manager under which
      # the hyper primitives (e.g. pg.oneof) will be materialized into concrete
      # values according to the controller decision.
      with example():
        reward = fun()
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

  In a distribured setup, worker processes may or may not work on the same
  trials. Worker group is introduced to serve this purpose, which is identified
  by an integer or a string named `group`. If `group` is not specified or
  having different values among the workers, every worker will work on
  different trials of the loop. On the contrary, workers having the same
  `group` will co-work on the same trials. Group is useful when evaluation on
  one example can be parallelized - for example - an example in the outer loop
  of a nested search. However, feedback on the same example should be fed back
  to the search algorithm only once. Therefore, workers in the same group need
  to communicate with each other to avoid duplicated evaluation and feedbacks.
  To faciliate such communication, per-trial metadata is supported, and can be
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
    hyper_value: A hyper value to sample from. A hyper value is an object with
      to-be-determined values specified by `pg.oneof`, `pg.manyof`, `pg.floatv`
      and etc, representing a search space, or a
      `pg.hyper.DynamicEvaluationContext` object.
    algorithm: The search algorithm that samples the search space. For example:
      `pg.generators.Random()`, `pg.evolution.regularized_evolution(...)`, and
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
      from the `hyper_value` will be included for the algorithm to make
      decisions. Otherwise only the decision points on which 'where' returns
      True will be included. The rest decision points will be passed through
      in the example, intact, which is a sub-space of the search space
      represented by the `hyper_value`. `where` is usually used in nested search
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
    search space defined by the `hyper_value` through `algorithm`.

  Raises:
    ValueError: `hyper_value` is a fixed value, or requested `backend` is not
    available.
  """
    # Placeholder for Google-internal usage instrumentation.

  # Create template based on the hyper value.
  if isinstance(hyper_value, hyper.DynamicEvaluationContext):
    dynamic_evaluation_context = hyper_value
    dna_spec = hyper_value.dna_spec
    template = None
  else:
    if symbolic.is_deterministic(hyper_value):
      raise ValueError(f'\'hyper_value\' is a constant value: {hyper_value!r}.')
    template = hyper.template(hyper_value, where)
    dna_spec = template.dna_spec()
    dynamic_evaluation_context = None

  # Create and set up the backend.
  metrics_to_optimize = metrics_to_optimize or ['reward']
  backend = _create_backend_factory(backend).create(**kwargs)
  backend.setup(
      name, group, dna_spec, algorithm, metrics_to_optimize,
      early_stopping_policy, num_examples)

  while True:
    try:
      feedback = backend.next()
      dna = feedback.dna
      reward = dna.metadata.get('reward')
      if reward is None:
        # Decode and return current example to client code for evaluation.
        if template is not None:
          value = template.decode(dna)
        else:
          assert dynamic_evaluation_context is not None
          value = lambda: dynamic_evaluation_context.apply(dna)
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


def poll_result(
    name: Text,
    backend: Optional[Text] = None,
    **kwargs) -> Result:
  """Gets tuning result by name."""
  return _create_backend_factory(backend).poll_result(name, **kwargs)


#
# A built-in in-memory tuning backend.
#


class _InMemoryFeedback(Feedback):
  """An in-memory tuning feedback."""

  def __init__(self,
               study: '_InMemoryResult',
               trial: Trial,
               feedback_fn: Callable[[geno.DNA, Trial], None],
               should_stop_early_fn: Callable[[Trial], bool],
               metrics_to_optimize: Sequence[Text]):
    super().__init__(metrics_to_optimize)
    self._study = study
    self._trial = trial
    self._feedback_fn = feedback_fn
    self._should_stop_early_fn = should_stop_early_fn
    self._sample_time = time.time()

  @property
  def id(self) -> int:
    """Gets Trial ID starting from 1."""
    return self._trial.id

  @property
  def dna(self) -> geno.DNA:
    """Gets DNA of tuning object for current trial."""
    return self._trial.dna

  def get_trial(self) -> Trial:
    """Gets current trial."""
    return self._trial

  @property
  def checkpoint_to_warm_start_from(self) -> Optional[Text]:
    """Gets checkpoint path to warm start from."""
    return None

  @property
  def status(self) -> Text:
    """Gets status of current trial."""
    return self._trial.status

  def set_metadata(self, key: Text, value: Any, per_trial: bool = True) -> None:
    """Sets metadata for current trial, which can be get by other co-workers."""
    # Verify value is serializable.
    _ = symbolic.to_json_str(value)
    if per_trial:
      self._trial.metadata[key] = value
    else:
      self._study._set_metadata(key, value)  # pylint: disable=protected-access

  def get_metadata(self, key: Text, per_trial: bool = True) -> Optional[Any]:
    """Gets metadata for current trial."""
    if per_trial:
      metadata = self._trial.metadata
    else:
      metadata = self._study.metadata
    return metadata.get(key, None)

  def add_link(self, name: Text, url: Text) -> None:
    """Adds a related link to current trial."""
    self._trial.related_links[name] = url

  def _add_measurement(
      self,
      reward: Optional[float],
      metrics: Dict[Text, float],
      step: int,
      checkpoint_path: Optional[Text],
      elapse_secs: float) -> None:
    """Adds a measurement to current trial."""
    if self._trial.status != 'PENDING':
      raise RaceConditionError(
          f'Measurements can only be added to PENDING trials. '
          f'Encountered: {self._trial}')
    self._trial.measurements.append(Measurement(
        step=step,
        reward=reward,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        elapse_secs=elapse_secs))

  def done(self,
           metadata: Optional[Dict[Text, Any]] = None,
           related_links: Optional[Dict[Text, Text]] = None) -> None:
    """Marks current tuning trial as done, and export final object."""
    del related_links
    if self._trial.status == 'PENDING':
      if not self._trial.measurements:
        raise ValueError(
            f'At least one measurement should be added for trial {self.id}.')
      self._trial.status = 'COMPLETED'
      self._trial.final_measurement = self._trial.measurements[-1]
      self._feedback_fn(self.dna, self._trial)
      self._trial.metadata.update(metadata or {})
      self._study._complete_trial(self._trial)  # pylint: disable=protected-access

  def skip(self, reason: Optional[Text] = None) -> None:
    """Skips current trial without providing feedback to the controller."""
    del reason
    if self._trial.status == 'PENDING':
      self._trial.status = 'COMPLETED'
      self._trial.infeasible = True
      self._trial.final_measurement = Measurement(
          reward=0.0, step=0, elapse_secs=0.0)
      self._study._complete_trial(self._trial)  # pylint: disable=protected-access

  def should_stop_early(self) -> bool:
    """Tells whether current trial should be stopped early.

    In `pg.sample`, an optional `EarlyStoppingPolicy` can be provided, which is
    useful for terminating trials which are progressive evaluated. Progressive
    evaluation on examples can be achieved by calling `feedback.add_measurement`
    multiple times at different steps. In-between these steps, users can call
    this method to determine if current trial is considered less competitive by
    the early stopping policy, and thus can be abandoned. In that case, users
    should call `feedback.skip()` to abandon current trial without feeding back
    the reward to the search algorithm.

    Returns:
      If current trial can be stopped early.
    """
    if not self._trial.measurements:
      return False
    return self._should_stop_early_fn(self._trial)

  def end_loop(self) -> None:
    """Ends current search loop."""
    self._study._set_active(False)  # pylint: disable=protected-access


class _InMemoryResult(Result):
  """An in-memory tuning result."""

  def __init__(self, name: Text, max_num_trials: Optional[int] = None):
    super().__init__()
    self._name = name
    self._max_num_trials = max_num_trials
    self._last_update_time = None
    self._trials = []
    self._is_active = True
    self._metadata = dict()
    self._num_trials_by_status = {
        'PENDING': 0,
        'COMPLETED': 0
    }
    self._num_infeasible = 0
    self._best_trial = None
    self._latest_trial_per_group = {}
    self._lock = threading.Lock()

  def create_trial(
      self, dna_fn: Callable[[], geno.DNA], group_id: Text) -> Trial:
    """Appends a trial to the result."""
    with self._lock:
      if (self._max_num_trials is not None
          and self.next_trial_id() > self._max_num_trials):
        raise StopIteration()
      trial = Trial(id=self.next_trial_id(), dna=dna_fn(), status='PENDING',
                    created_time=int(time.time()), metadata=dict())
      self._trials.append(trial)
      self._num_trials_by_status['PENDING'] += 1
      self._latest_trial_per_group[group_id] = trial
    return trial

  def _complete_trial(self, trial: Trial) -> None:
    """Status change callback."""
    with self._lock:
      self._num_trials_by_status['COMPLETED'] += 1
      self._num_trials_by_status['PENDING'] -= 1
      if trial.infeasible:
        self._num_infeasible += 1
      else:
        best = self._best_trial
        if (best is None or (
            trial.final_measurement.reward is not None
            and best.final_measurement.reward
            < trial.final_measurement.reward)):
          self._best_trial = trial
      self._last_update_time = datetime.datetime.utcnow()

  @property
  def metadata(self) -> Dict[Text, Any]:
    """Gets metadata for current sampling."""
    return self._metadata

  def _set_metadata(self, key: Text, value: Any) -> Any:
    """Sets metadata for current sampling."""
    self._metadata[key] = value

  @property
  def last_updated(self) -> datetime.datetime:
    """Last update time."""
    return self._last_update_time

  @property
  def is_active(self) -> bool:
    """Returns if current in-memory study is active or not."""
    return self._is_active

  def _set_active(self, active: bool) -> None:
    """Set in-memory study is active or not."""
    self._is_active = active

  def get_latest_trial(self, group_id: Text) -> Optional[Trial]:
    """Get latest trial per group id."""
    return self._latest_trial_per_group.get(group_id, None)

  def next_trial_id(self) -> int:
    """Gets the next trial ID."""
    return len(self._trials) + 1

  @property
  def best_trial(self) -> Optional[Trial]:
    """Gets the best trial."""
    return self._best_trial

  @property
  def trials(self) -> List[Trial]:
    """Gets all trials."""
    return self._trials

  def format(self,
             compact: bool = False,
             verbose: bool = True,
             root_indent: int = 0,
             **kwargs):
    """Return summary."""
    possible_status = ['PENDING', 'COMPLETED']
    json_repr = collections.OrderedDict([
        ('name', self._name),
        ('status',
         collections.OrderedDict([
             (s, f'{self._num_trials_by_status[s]}/{len(self._trials)}')
             for s in possible_status if (s in self._num_trials_by_status
                                          and self._num_trials_by_status[s])
         ])),
    ])
    if self._num_infeasible:
      json_repr['infeasible'] = f'{self._num_infeasible}/{len(self._trials)}'

    if self._best_trial:
      json_repr['best_trial'] = collections.OrderedDict([
          ('id', self._best_trial.id),
          ('reward', self._best_trial.final_measurement.reward),
          ('step', self._best_trial.final_measurement.step),
          ('dna', self._best_trial.dna.format(compact=True))
      ])
    return object_utils.format(json_repr, compact, False, root_indent, **kwargs)


class _InMemoryBackend(Backend):
  """In-memory tuning backend."""

  def __init__(self):
    """Constructor."""
    super().__init__()
    self._study = None
    self._group_id = None
    self._dna_spec = None
    self._algorithm = None
    self._early_stopping_policy = None
    self._num_examples = None
    self._metrics_to_optimize = None

  def setup(self,
            name: Optional[Text],
            group_id: Optional[Text],
            dna_spec: geno.DNASpec,
            algorithm: geno.DNAGenerator,
            metrics_to_optimize: Sequence[Text],
            early_stopping_policy: Optional[EarlyStoppingPolicy] = None,
            num_examples: Optional[int] = None,
            ) -> None:
    """Sets up the backend for an existing or a new sampling."""
    # Lookup or create a new in-memory result.
    if name is None or name not in _in_memory_results:
      study = _InMemoryResult(name, num_examples)
      if name is not None:
        _in_memory_results[name] = study
    else:
      study = _in_memory_results[name]

    if group_id is None:
      group_id = str(threading.get_ident())
    self._group_id = group_id

    if not algorithm.multi_objective and len(metrics_to_optimize) > 1:
      raise ValueError(
          f'\'metrics_to_optimize\' should include only 1 metric as '
          f'multi-objective optimization is not supported by {algorithm!r}.')

    # NOTE(daiyip): algorithm can continue if it's already set up with the same
    # DNASpec, or we will setup the algorithm with input DNASpec.
    if algorithm.dna_spec is None:
      algorithm.setup(dna_spec)
    elif symbolic.ne(algorithm.dna_spec, dna_spec):
      raise ValueError(
          f'{algorithm!r} has been set up with a different DNASpec. '
          f'Existing: {algorithm.dna_spec!r}, New: {dna_spec!r}.')

    if early_stopping_policy:
      if early_stopping_policy.dna_spec is None:
        early_stopping_policy.setup(dna_spec)
      elif early_stopping_policy.dna_spec != dna_spec:
        raise ValueError(
            f'{early_stopping_policy!r} has been set up with a different '
            f'DNASpec. Existing: {early_stopping_policy.dna_spec!r}, '
            f'New: {dna_spec!r}.')

    self._study = study
    self._dna_spec = dna_spec
    self._algorithm = algorithm
    self._early_stopping_policy = early_stopping_policy
    self._num_examples = num_examples
    self._metrics_to_optimize = metrics_to_optimize

  def _create_feedback(
      self,
      study: _InMemoryResult,
      trial: Trial) -> _InMemoryFeedback:
    """Creates a feedback object for input trial."""
    return _InMemoryFeedback(
        study, trial, self._feedback,
        self._should_stop_early,
        self._metrics_to_optimize)

  def _feedback(self, dna: geno.DNA, trial: Trial):
    """Feedback callback for a trial."""
    reward = trial.get_reward_for_feedback(self._metrics_to_optimize)
    if reward is not None:
      self._algorithm.feedback(dna, reward)

  def _should_stop_early(self, trial: Trial) -> bool:
    if self._early_stopping_policy is not None:
      assert trial.measurements
      return self._early_stopping_policy.should_stop_early(
          trial, trial.measurements[-1])
    return False

  def next(self) -> Feedback:
    """Get the feedback object for the next trial."""
    def next_dna():
      return self._algorithm.propose()

    if not self._study.is_active:
      raise StopIteration()

    # If current session is pending, always return current session.
    trial = self._study.get_latest_trial(self._group_id)
    if trial is None or trial.status != 'PENDING':
      trial = self._study.create_trial(next_dna, self._group_id)
    return self._create_feedback(self._study, trial)


# Global dictionary for locally sampled in-memory results by name.
_in_memory_results = {}


@add_backend('in-memory')
class _InMemoryBackendFactory(BackendFactory):
  """In-memory backend factory."""

  def create(
      self,
      # NOTE(daiyip): passing through other keyword arguments to allow
      # swapping of the default backend.
      **kwargs) -> Backend:
    """Creates a tuning backend for an existing or a new sampling."""
    return _InMemoryBackend()

  def poll_result(self, name: Text) -> Result:
    """Gets tuning result by a unique tuning identifier."""
    if name not in _in_memory_results:
      raise ValueError(f'Result {name!r} does not exist.')
    return _in_memory_results[name]
