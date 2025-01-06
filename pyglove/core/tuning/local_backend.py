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
"""A local (in-memory) tuning backend."""

import collections
import datetime
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from pyglove.core import geno
from pyglove.core import logging
from pyglove.core import symbolic
from pyglove.core import utils
from pyglove.core.tuning import backend
from pyglove.core.tuning.early_stopping import EarlyStoppingPolicy
from pyglove.core.tuning.protocols import Feedback
from pyglove.core.tuning.protocols import Measurement
from pyglove.core.tuning.protocols import RaceConditionError
from pyglove.core.tuning.protocols import Result
from pyglove.core.tuning.protocols import Trial


class _InMemoryFeedback(Feedback):
  """An in-memory tuning feedback."""

  def __init__(self,
               study: '_InMemoryResult',
               trial: Trial,
               feedback_fn: Callable[[geno.DNA, Trial], None],
               should_stop_early_fn: Callable[[Trial], bool],
               metrics_to_optimize: Sequence[str]):
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
  def checkpoint_to_warm_start_from(self) -> Optional[str]:
    """Gets checkpoint path to warm start from."""
    return None

  @property
  def status(self) -> str:
    """Gets status of current trial."""
    return self._trial.status

  def set_metadata(self, key: str, value: Any, per_trial: bool = True) -> None:
    """Sets metadata for current trial, which can be get by other co-workers."""
    # Verify value is serializable.
    _ = symbolic.to_json_str(value)
    if per_trial:
      self._trial.metadata[key] = value
    else:
      self._study._set_metadata(key, value)  # pylint: disable=protected-access

  def get_metadata(self, key: str, per_trial: bool = True) -> Optional[Any]:
    """Gets metadata for current trial."""
    if per_trial:
      metadata = self._trial.metadata
    else:
      metadata = self._study.metadata
    return metadata.get(key, None)

  def add_link(self, name: str, url: str) -> None:
    """Adds a related link to current trial."""
    self._trial.related_links[name] = url

  def _add_measurement(
      self,
      reward: Optional[float],
      metrics: Dict[str, float],
      step: int,
      checkpoint_path: Optional[str],
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
           metadata: Optional[Dict[str, Any]] = None,
           related_links: Optional[Dict[str, str]] = None) -> None:
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

  def skip(self, reason: Optional[str] = None) -> None:
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

  def __init__(self, name: str, max_num_trials: Optional[int] = None):
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
      self, dna_fn: Callable[[], geno.DNA], group_id: str) -> Trial:
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
      self._last_update_time = datetime.datetime.now(tz=datetime.timezone.utc)

  @property
  def metadata(self) -> Dict[str, Any]:
    """Gets metadata for current sampling."""
    return self._metadata

  def _set_metadata(self, key: str, value: Any) -> Any:
    """Sets metadata for current sampling."""
    self._metadata[key] = value

  @property
  def last_updated(self) -> Optional[datetime.datetime]:
    """Last update time."""
    return self._last_update_time

  @property
  def is_active(self) -> bool:
    """Returns if current in-memory study is active or not."""
    return self._is_active

  def _set_active(self, active: bool) -> None:
    """Set in-memory study is active or not."""
    self._is_active = active

  def get_latest_trial(self, group_id: str) -> Optional[Trial]:
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
    return utils.format(json_repr, compact, False, root_indent, **kwargs)


@backend.add_backend('in-memory')
class _InMemoryBackend(backend.Backend):
  """In-memory tuning backend."""

  def __init__(self,
               name: Optional[str],
               group: Optional[str],
               dna_spec: geno.DNASpec,
               algorithm: geno.DNAGenerator,
               metrics_to_optimize: Sequence[str],
               early_stopping_policy: Optional[EarlyStoppingPolicy] = None,
               num_examples: Optional[int] = None,
               **kwargs):
    """Constructor."""
    super().__init__()

    if name is None or name not in _in_memory_results:
      study = _InMemoryResult(name, num_examples)
      if name is not None:
        _in_memory_results[name] = study
    else:
      study = _in_memory_results[name]

    if group is None:
      group = str(threading.get_ident())

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

    if kwargs:
      logging.warning(
          f'Ignoring keyword arguments that are not supported by \'in-memory\' '
          f'backend: {kwargs}')

    self._study = study
    self._group_id = group
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
      return self._early_stopping_policy.should_stop_early(trial)
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

  @classmethod
  def poll_result(cls, name: str) -> Result:
    """Gets tuning result by a unique tuning identifier."""
    if name not in _in_memory_results:
      raise ValueError(f'Result {name!r} does not exist.')
    return _in_memory_results[name]


# Global dictionary for locally sampled in-memory results by name.
_in_memory_results: Dict[str, _InMemoryResult] = {}
