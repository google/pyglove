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
from typing import List, Optional, Sequence

from pyglove.core import geno
from pyglove.core.tuning.early_stopping import EarlyStoppingPolicy
from pyglove.core.tuning.protocols import Feedback
from pyglove.core.tuning.protocols import Result


class Backend(metaclass=abc.ABCMeta):
  """Interface for the tuning backend."""

  @abc.abstractmethod
  def setup(self,
            name: Optional[str],
            group_id: Optional[str],
            dna_spec: geno.DNASpec,
            algorithm: geno.DNAGenerator,
            metrics_to_optimize: Sequence[str],
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
  def poll_result(self, name: str) -> Result:
    """Gets tuning result by a unique tuning identifier."""


_backend_registry = dict()
_default_backend_name = 'in-memory'


def add_backend(backend_name: str):
  """Decorator to register a backend factory with name."""
  def _decorator(factory_cls):
    if not issubclass(factory_cls, BackendFactory):
      raise TypeError(f'{factory_cls!r} is not a BackendFactory subclass.')
    _backend_registry[backend_name] = factory_cls
    return factory_cls
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


def create_backend_factory(backend_name: str) -> BackendFactory:
  """Get backend by name."""
  backend_name = backend_name or default_backend()
  if backend_name not in _backend_registry:
    raise ValueError(f'Backend {backend_name!r} does not exist.')
  return _backend_registry[backend_name]()


def poll_result(
    name: str,
    backend: Optional[str] = None,
    **kwargs) -> Result:
  """Gets tuning result by name."""
  return create_backend_factory(backend).poll_result(name, **kwargs)
