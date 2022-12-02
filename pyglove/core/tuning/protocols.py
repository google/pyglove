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
"""Tuning protocols."""

import abc
import contextlib
import datetime
import inspect
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from pyglove.core import geno
from pyglove.core import object_utils
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing


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
    ('step', pg_typing.Int(), 'At which step the result is reported.'),
    ('elapse_secs', pg_typing.Float(), 'Elapse in seconds since trial start.'),
    ('reward', pg_typing.Float().noneable(),
     ('Reward of reported tunable target. Can be None if multi-objective '
      'optimization is used.')),
    ('metrics', pg_typing.Dict([
        (pg_typing.StrKey(), pg_typing.Float(), 'Metric item.')
    ]).noneable(), 'Metric in key/value pairs (optional).'),
    ('checkpoint_path', pg_typing.Str().noneable(),
     'Path to the checkpoint of this specific measurement.')
])
class Measurement(_DataEntity):
  """Measurement of a trial at certain step."""


@symbolic.members([
    ('id', pg_typing.Int(), 'Identifier of the trial.'),
    ('description', pg_typing.Str().noneable(), 'Description of the trial.'),
    ('dna', pg_typing.Object(geno.DNA), 'Proposed DNA for the trial.'),
    ('status',
     pg_typing.Enum('PENDING', [
         'UNKNOWN',
         'REQUESTED',
         'PENDING',
         'COMPLETED',
         'DELETED',
         'STOPPING',
     ]), 'Trial status.'),
    ('final_measurement', pg_typing.Object(Measurement).noneable(),
     'Reported final results.'),
    ('infeasible', pg_typing.Bool(False), 'Whether trial is infeasible.'),
    ('measurements', pg_typing.List(pg_typing.Object(Measurement), default=[]),
     'All reported measurements.'),
    ('metadata',
     pg_typing.Dict(
         [(pg_typing.StrKey(), pg_typing.Any(),
           'Serializable key value pairs as metadata.')]),
     'Trial metadata.'),
    ('related_links',
     pg_typing.Dict([
         (pg_typing.StrKey(), pg_typing.Str(), 'Related link.')]),
     'Related links'),
    # TODO(daiyip): consider change time from timestamp to datetime.
    # Need to introduce a mechanism in symbolic to cherry pick serialization
    # for individual fields.
    ('created_time', pg_typing.Int(), 'Created time in Unix timestamp.'),
    ('completed_time', pg_typing.Int().noneable(),
     'Completed time in Unix timestamp.'),
])
class Trial(_DataEntity):
  """Metadata of a trial."""

  def get_reward_for_feedback(
      self, metric_names: Optional[Sequence[str]] = None
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
  def metadata(self) -> Dict[str, Any]:
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

  def __init__(self, metrics_to_optimize: Sequence[str]):
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
  def checkpoint_to_warm_start_from(self) -> Optional[str]:
    """Gets checkpoint path to warm start from."""

  def add_measurement(
      self,
      reward: Union[None, float, Sequence[float]] = None,
      metrics: Optional[Dict[str, float]] = None,
      step: int = 0,
      checkpoint_path: Optional[str] = None,
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
      metrics: Dict[str, float],
      step: int,
      checkpoint_path: Optional[str],
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
  def set_metadata(self, key: str, value: Any, per_trial: bool = True) -> None:
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
  def get_metadata(self, key: str, per_trial: bool = True) -> Optional[Any]:
    """Gets metadata for current trial or current sampling.

    Args:
      key: A string as key to metadata.
      per_trial: If True, the key is retrieved per curent trial. Otherwise, it
        is retrieved per current sampling.

    Returns:
      A value that can be deserialized by `pg.from_json_str`.
    """

  @abc.abstractmethod
  def add_link(self, name: str, url: str) -> None:
    """Adds a related link to current trial.

    Added links can be retrieved from the `Trial.related_links` property via
    `pg.poll_result`.

    Args:
      name: Name for the related link.
      url: URL for this link.
    """

  @abc.abstractmethod
  def done(self,
           metadata: Optional[Dict[str, Any]] = None,
           related_links: Optional[Dict[str, str]] = None) -> None:
    """Marks current trial as done.

    Args:
      metadata: Additional metadata to add to current trial.
      related_links: Additional links to add to current trial.
    """

  @abc.abstractmethod
  def skip(self, reason: Optional[str] = None) -> None:
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
      metrics: Optional[Dict[str, float]] = None,
      checkpoint_path: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None,
      related_links: Optional[Dict[str, str]] = None,
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
