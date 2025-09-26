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
"""Pluggable metric systems for monitoring.

This module allows PyGlove to plugin metrics to monitor the execution of
programs.
"""

import abc
import collections
import math
import threading
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


try:
  import numpy  # pylint: disable=g-import-not-at-top
except ImportError:
  numpy = None


class Metric(metaclass=abc.ABCMeta):
  """Base class for metrics."""

  def __init__(
      self,
      namespace: str,
      name: str,
      description: str,
      parameter_definitions: Dict[str, Type[Union[int, str, bool]]]
  ) -> None:
    self._namespace = namespace
    self._name = name
    self._description = description
    self._parameter_definitions = parameter_definitions

  @property
  def namespace(self) -> str:
    """Returns the namespace of the metric."""
    return self._namespace

  @property
  def name(self) -> str:
    """Returns the name of the metric."""
    return self._name

  @property
  def full_name(self) -> str:
    """Returns the full name of the metric."""
    return f'{self.namespace}/{self.name}'

  @property
  def description(self) -> str:
    """Returns the description of the metric."""
    return self._description

  @property
  def parameter_definitions(self) -> Dict[str, Type[Union[int, str, bool]]]:
    """Returns the parameter definitions of the metric."""
    return self._parameter_definitions

  def _parameters_key(self, **parameters) -> Tuple[Any, ...]:
    """Returns the parameters tuple for the metric."""
    for k, t in self._parameter_definitions.items():
      v = parameters.get(k)
      if v is None:
        raise KeyError(
            f'Metric {self.full_name!r}: Parameter {k!r} is required but not '
            f'given.'
        )
      if not isinstance(v, t):
        raise TypeError(
            f'Metric {self.full_name!r}: Parameter {k!r} has type '
            f'{type(v)} but expected type {t}.'
        )
    for k in parameters:
      if k not in self._parameter_definitions:
        raise KeyError(
            f'Metric {self.full_name!r}: Parameter {k!r} is not defined but '
            f'provided. Available parameters: '
            f'{list(self._parameter_definitions.keys())}'
        )
    return tuple(parameters[k] for k in self._parameter_definitions)


class Counter(Metric):
  """Base class for counters."""

  @abc.abstractmethod
  def increment(self, delta: int = 1, **parameters) -> int:
    """Increments the counter by delta and returns the new value.

    Args:
      delta: The amount to increment the counter by.
      **parameters: Parameters for parameterized counters.

    Returns:
      The new value of the counter.
    """

  @abc.abstractmethod
  def value(self, **parameters) -> int:
    """Returns the value of the counter for the given parameters.

    Args:
      **parameters: Parameters for parameterized counters.

    Returns:
      The value of the counter.
    """


class Distribution(metaclass=abc.ABCMeta):
  """Distribution of scalar values."""

  @property
  @abc.abstractmethod
  def count(self) -> int:
    """Returns the number of samples in the distribution."""

  @property
  @abc.abstractmethod
  def sum(self) -> float:
    """Returns the sum of the distribution."""

  @property
  @abc.abstractmethod
  def mean(self) -> float:
    """Returns the mean of the distribution."""

  @property
  @abc.abstractmethod
  def stddev(self) -> float:
    """Returns the standard deviation of the distribution."""

  @property
  def median(self) -> float:
    """Returns the standard deviation of the distribution."""
    return self.percentile(50)

  @property
  @abc.abstractmethod
  def variance(self) -> float:
    """Returns the variance of the distribution."""

  @abc.abstractmethod
  def percentile(self, n: float) -> float:
    """Returns the median of the distribution.

    Args:
      n: The percentile to return. Should be in the range [0, 100].

    Returns:
      The n-th percentile of the distribution.
    """

  @abc.abstractmethod
  def fraction_less_than(self, value: float) -> float:
    """Returns the fraction of values in the distribution less than value."""


class Scalar(Metric):
  """Base class for scalar values."""

  @abc.abstractmethod
  def record(self, value: int, **parameters) -> None:
    """Records a value to the scalar.

    Args:
      value: The value to record.
      **parameters: Parameters for parameterized scalars.
    """

  @abc.abstractmethod
  def distribution(self, **parameters) -> Distribution:
    """Returns the distribution of the scalar.

    Args:
      **parameters: Parameters for parameterized scalars.

    Returns:
      The distribution of the scalar.
    """


class MetricCollection(metaclass=abc.ABCMeta):
  """Base class for counter collections."""

  def __init__(
      self,
      namespace: str,
      default_parameters: Optional[
          Dict[str, Type[Union[int, str, bool]]]
      ] = None
  ):
    """Initializes the metric collection.

    Args:
      namespace: The namespace of the metric collection.
      default_parameters: The default parameters used to create metrics
        if not specified.
    """
    self._namespace = namespace
    self._default_parameter_definitions = default_parameters or {}
    self._metrics = self._metric_container()

  @property
  def namespace(self) -> str:
    """Returns the namespace of the metric collection."""
    return self._namespace

  def metrics(self) -> List[Metric]:
    """Returns the names of the metrics."""
    return [m for m in self._metrics.values() if m.namespace == self._namespace]

  def _metric_container(self) -> dict[str, Metric]:
    """Returns the container for metrics."""
    return {}

  def _get_or_create_metric(
      self,
      metric_cls: Type[Metric],
      create_metric_fn: Callable[
          [str, str, Dict[str, Type[Union[int, str, bool]]]],
          Metric
      ],
      name: str,
      description: str,
      parameter_definitions: Dict[str, Type[Union[int, str, bool]]]
  ) -> Metric:
    """Gets or creates a metric with the given name."""
    full_name = f'{self._namespace}/{name}'
    metric = self._metrics.get(full_name)
    if metric is not None:
      if not isinstance(metric, metric_cls):
        raise ValueError(
            f'Metric {full_name!r} already exists with a different type '
            f'({type(metric)}).'
        )
      if description != metric.description:
        raise ValueError(
            f'Metric {full_name!r} already exists with a different description '
            f'({metric.description!r}).'
        )
      if parameter_definitions != metric.parameter_definitions:
        raise ValueError(
            f'Metric {full_name!r} already exists with different parameter '
            f'definitions ({metric.parameter_definitions!r}).'
        )
    else:
      metric = create_metric_fn(name, description, parameter_definitions)
      self._metrics[full_name] = metric
    return metric

  def get_counter(
      self,
      name: str,
      description: str,
      parameters: Optional[Dict[str, Type[Union[int, str, bool]]]] = None,
      **kwargs
  ) -> Counter:
    """Gets or creates a counter with the given name.

    Args:
      name: The name of the counter.
      description: The description of the counter.
      parameters: The definitions of the parameters for the counter.
        `default_parameters` from the collection will be used if not specified.
      **kwargs: Additional arguments for creating the counter.

    Returns:
      The counter with the given name.
    """
    if parameters is None:
      parameters = self._default_parameter_definitions
    return typing.cast(
        Counter, self._get_or_create_metric(
            Counter, self._create_counter, name, description, parameters,
            **kwargs
        )
    )

  @abc.abstractmethod
  def _create_counter(
      self,
      name: str,
      description: str,
      parameter_definitions: Dict[str, Type[Union[int, str, bool]]],
      **kwargs
  ) -> Counter:
    """Creates a counter with the given name."""

  def get_scalar(
      self,
      name: str,
      description: str,
      parameters: Optional[Dict[str, Type[Union[int, str, bool]]]] = None,
      **kwargs
  ) -> Scalar:
    """Gets or creates a scalar with the given name.

    Args:
      name: The name of the counter.
      description: The description of the counter.
      parameters: The definitions of the parameters for the counter.
        `default_parameters` from the collection will be used if not specified.
      **kwargs: Additional arguments for creating the scalar.

    Returns:
      The counter with the given name.
    """
    if parameters is None:
      parameters = self._default_parameter_definitions
    return typing.cast(
        Scalar,
        self._get_or_create_metric(
            Scalar, self._create_scalar, name, description, parameters, **kwargs
        )
    )

  @abc.abstractmethod
  def _create_scalar(
      self,
      name: str,
      description: str,
      parameter_definitions: Dict[str, Type[Union[int, str, bool]]],
      **kwargs
  ) -> Scalar:
    """Creates a counter with the given name."""

#
# InMemoryMetricCollection.
#


class _InMemoryCounter(Counter):
  """In-memory counter."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._counter = collections.defaultdict(int)
    self._lock = threading.Lock()

  def increment(self, delta: int = 1, **parameters) -> int:
    """Increments the counter by delta and returns the new value."""
    parameters_key = self._parameters_key(**parameters)
    with self._lock:
      value = self._counter[parameters_key]
      value += delta
      self._counter[parameters_key] = value
      return value

  def value(self, **parameters) -> int:
    """Returns the value of the counter based on the given parameters."""
    return self._counter[self._parameters_key(**parameters)]


class _InMemoryScalar(Scalar):
  """In-memory scalar."""

  def __init__(self, *args, window_size: int = 1024 * 1024, **kwargs):
    super().__init__(*args, **kwargs)
    self._window_size = window_size
    self._distributions = collections.defaultdict(
        lambda: _InMemoryDistribution(self._window_size)
    )
    self._lock = threading.Lock()

  def record(self, value: Any, **parameters) -> None:
    """Records a value to the scalar."""
    parameters_key = self._parameters_key(**parameters)
    self._distributions[parameters_key].add(value)

  def distribution(self, **parameters) -> Distribution:
    """Returns the distribution of the scalar."""
    parameters_key = self._parameters_key(**parameters)
    return self._distributions[parameters_key]


class _InMemoryDistribution(Distribution):
  """In memory distribution of scalar values."""

  def __init__(self, window_size: int = 1024 * 1024):
    self._window_size = window_size
    self._data = []
    self._sum = 0.0
    self._square_sum = 0.0
    self._count = 0
    self._lock = threading.Lock()

  def add(self, value: int):
    """Adds a value to the distribution."""
    with self._lock:
      if len(self._data) == self._window_size:
        x = self._data.pop(0)
        self._sum -= x
        self._count -= 1
        self._square_sum -= x ** 2

      self._data.append(value)
      self._count += 1
      self._sum += value
      self._square_sum += value ** 2

  @property
  def count(self) -> int:
    """Returns the number of samples in the distribution."""
    return self._count

  @property
  def sum(self) -> float:
    """Returns the sum of the distribution."""
    return self._sum

  @property
  def mean(self) -> float:
    """Returns the mean of the distribution."""
    return self._sum / self._count if self._count else 0.0

  @property
  def stddev(self) -> float:
    """Returns the standard deviation of the distribution."""
    return math.sqrt(self.variance)

  @property
  def variance(self) -> float:
    """Returns the variance of the distribution."""
    if self._count < 2:
      return 0.0
    return self._square_sum / self._count - self.mean ** 2

  def percentile(self, n: float) -> float:
    """Returns the median of the distribution."""
    if n < 0 or n > 100:
      raise ValueError(f'Percentile {n} is not in the range [0, 100].')

    if self._count == 0:
      return 0.0

    if numpy is not None:
      return numpy.percentile(self._data, n)  # pytype: disable=attribute-error

    sorted_data = sorted(self._data)
    index = (n / 100) * (len(sorted_data) - 1)
    if index % 1 == 0:
      return sorted_data[int(index)]
    else:
      # Interpolate the value at the given percentile.
      lower_index = int(index)
      fraction = index - lower_index

      # Get the values at the two surrounding integer indices
      lower_value = sorted_data[lower_index]
      upper_value = sorted_data[lower_index + 1]
      return lower_value + fraction * (upper_value - lower_value)

  def fraction_less_than(self, value: float) -> float:
    """Returns the fraction of values in the distribution less than value."""
    if self._count == 0:
      return 0.0
    with self._lock:
      return len([x for x in self._data if x < value]) / self._count


class InMemoryMetricCollection(MetricCollection):
  """In-memory counter."""

  def _create_counter(
      self,
      name: str,
      description: str,
      parameter_definitions: Dict[str, Type[Union[int, str, bool]]],
      **kwargs
  ) -> Counter:
    return _InMemoryCounter(
        self._namespace, name, description, parameter_definitions
    )

  def _create_scalar(
      self,
      name: str,
      description: str,
      parameter_definitions: Dict[str, Type[Union[int, str, bool]]],
      *,
      window_size: int = 1024 * 1024,
      **kwargs
  ) -> Scalar:
    return _InMemoryScalar(
        self._namespace, name, description, parameter_definitions,
        window_size=window_size, **kwargs
    )


_METRIC_COLLECTION_CLS = InMemoryMetricCollection  # pylint: disable=invalid-name


def metric_collection(namespace: str, **kwargs) -> MetricCollection:
  """Creates a metric collection."""
  return _METRIC_COLLECTION_CLS(namespace, **kwargs)


def set_default_metric_collection_cls(cls: Type[MetricCollection]) -> None:
  """Sets the default metric collection class."""
  global _METRIC_COLLECTION_CLS
  _METRIC_COLLECTION_CLS = cls


def default_metric_collection_cls() -> Type[MetricCollection]:
  """Returns the default metric collection class."""
  return _METRIC_COLLECTION_CLS
