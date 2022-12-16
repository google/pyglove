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
"""Interface for tuning backend and backend factory."""

import abc
from typing import List, Optional, Sequence, Type

from pyglove.core import geno
from pyglove.core.tuning.early_stopping import EarlyStoppingPolicy
from pyglove.core.tuning.protocols import Feedback
from pyglove.core.tuning.protocols import Result


class Backend(metaclass=abc.ABCMeta):
  """Interface for the tuning backend."""

  @classmethod
  def create(cls,
             name: Optional[str],
             group: Optional[str],
             dna_spec: geno.DNASpec,
             algorithm: geno.DNAGenerator,
             metrics_to_optimize: Sequence[str],
             early_stopping_policy: Optional[EarlyStoppingPolicy] = None,
             num_examples: Optional[int] = None,
             **kwargs) -> 'Backend':
    """Create an instance of `Backend` based on `pg.sample` arguments.

    The default implementation is to pass through all the arguments to
    ``__init__`` for creating an instance of the backend. Users can override.

    Args:
      name: A string as a unique identifier for current sampling. Two separate
        calls to `pg.sample` with the same `name` (also the same algorithm) will
        share the same sampling queue, whose examples are proposed by the same
        search algorithm.
      group: An string or integer as the group ID of current process in
        distributed sampling, which will be used to group different workers into
        co-worker groups. Workers with the same group id will work on the same
        trial. On the contrary, workers in different groups will always be
        working with different trials. If not specified, each worker in current
        sampling will be in different groups. `group` is usually used in the
        outer loops of nested search, in order to allow workers to work on the
        same higher-order item.
      dna_spec: An `pg.DNASpec` object representing the search space.
      algorithm: The search algorithm that samples the search space.
      metrics_to_optimize: A sequence of string as the names of the metrics
        to be optimized by the algorithm, which is ['reward'] by default.
        When specified, it should have only 1 item for single-objective
        algorithm and can have multiple items for algorithms that support
        multi-objective optimization.
      early_stopping_policy: An optional early stopping policy for user to tell
        if incremental evaluation (which reports multiple measurements) on each
        example can be early short circuited.
        After each call to `feedback.add_measurement`, users can use method
        `feedback.should_stop_early` to check whether current example worth
        further evaluation or not.
      num_examples: An optional integer as the max number of examples to
        sample. If None, sample will return an iterator of infinite examples.
      **kwargs: Arguments passed to the `BackendFactory` subclass registered
        with the requested backend.

    Returns:
      A `pg.tuning.Backend` object.
    """
    return cls(     # pytype: disable=wrong-keyword-args
        name=name,
        group=group,
        dna_spec=dna_spec,
        algorithm=algorithm,
        metrics_to_optimize=metrics_to_optimize,
        early_stopping_policy=early_stopping_policy,
        num_examples=num_examples,
        **kwargs)

  @classmethod
  @abc.abstractmethod
  def poll_result(cls, name: str, **kwargs) -> Result:
    """Gets tuning result by a unique tuning identifier."""

  @abc.abstractmethod
  def next(self) -> Feedback:
    """Get the feedback object for the next sample."""


_backend_registry = dict()
_default_backend_name = 'in-memory'


def add_backend(backend_name: str):
  """Decorator to register a backend factory with name."""
  def _decorator(backend_cls):
    if not issubclass(backend_cls, Backend):
      raise TypeError(f'{backend_cls!r} is not a `pg.tuning.Backend` subclass.')
    _backend_registry[backend_name] = backend_cls
    return backend_cls
  return _decorator


def available_backends() -> List[str]:
  """Gets available backend names."""
  return list(_backend_registry.keys())


def set_default_backend(backend_name: str):
  """Sets the default tuning backend name."""
  if backend_name not in _backend_registry:
    raise ValueError(f'Backend {backend_name!r} does not exist.')
  global _default_backend_name
  _default_backend_name = backend_name


def default_backend() -> str:
  """Gets the default tuning backend name."""
  return _default_backend_name


def get_backend_cls(backend_name: str) -> Type[Backend]:
  """Get backend by name."""
  backend_name = backend_name or default_backend()
  if backend_name not in _backend_registry:
    raise ValueError(f'Backend {backend_name!r} does not exist.')
  return _backend_registry[backend_name]


def poll_result(
    name: str,
    backend: Optional[str] = None,
    **kwargs) -> Result:
  """Gets tuning result by name."""
  return get_backend_cls(backend).poll_result(name, **kwargs)
