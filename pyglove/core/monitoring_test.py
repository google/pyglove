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

import time
import unittest
from pyglove.core import monitoring
from pyglove.core.symbolic import error_info  # pylint: disable=unused-import


class MetricCollectionTest(unittest.TestCase):
  """Tests for metric collection."""

  def test_default_metric_collection_cls(self):
    self.assertIs(
        monitoring.default_metric_collection_cls(),
        monitoring.InMemoryMetricCollection
    )

    class TestMetricCollection(monitoring.MetricCollection):
      pass

    monitoring.set_default_metric_collection_cls(TestMetricCollection)
    self.assertIs(
        monitoring.default_metric_collection_cls(),
        TestMetricCollection
    )
    monitoring.set_default_metric_collection_cls(
        monitoring.InMemoryMetricCollection
    )
    self.assertIs(
        monitoring.default_metric_collection_cls(),
        monitoring.InMemoryMetricCollection
    )

  def test_metric_collection(self):
    collection = monitoring.metric_collection('/test')
    self.assertEqual(collection.namespace, '/test')
    self.assertIsInstance(collection, monitoring.InMemoryMetricCollection)

  def test_creation_failures(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    counter = collection.get_counter('counter', 'counter description')
    self.assertIsInstance(counter, monitoring.Counter)
    with self.assertRaisesRegex(
        ValueError, 'Metric .* already exists with a different type'
    ):
      collection.get_distribution('counter', 'counter description')

    with self.assertRaisesRegex(
        ValueError, 'Metric .* already exists with a different description'
    ):
      collection.get_counter('counter', 'different description')

    with self.assertRaisesRegex(
        ValueError,
        'Metric .* already exists with different parameter definitions'
    ):
      collection.get_counter(
          'counter', 'counter description', parameters={'field1': str}
      )


class InMemoryDistributionValueTest(unittest.TestCase):
  """Tests for in memory distribution value."""

  def test_empty_distribution(self):
    dist = monitoring._InMemoryDistributionValue()
    self.assertEqual(dist.count, 0)
    self.assertEqual(dist.sum, 0.0)
    self.assertEqual(dist.mean, 0.0)
    self.assertEqual(dist.stddev, 0.0)
    self.assertEqual(dist.variance, 0.0)
    self.assertEqual(dist.median, 0.0)
    self.assertEqual(dist.percentile(50), 0.0)
    self.assertEqual(dist.fraction_less_than(100), 0.0)

  def test_add_value(self):
    dist = monitoring._InMemoryDistributionValue()
    dist.add(1)
    dist.add(3)
    dist.add(10)
    dist.add(2)
    self.assertEqual(dist.count, 4)
    self.assertEqual(dist.sum, 16.0)
    self.assertEqual(dist.mean, 4.0)
    self.assertEqual(dist.stddev, 3.5355339059327378)
    self.assertEqual(dist.variance, 12.5)
    self.assertEqual(dist.median, 2.5)
    self.assertEqual(dist.percentile(50), 2.5)
    self.assertEqual(dist.percentile(10), 1.3)
    self.assertEqual(dist.fraction_less_than(100), 1.0)
    self.assertEqual(dist.fraction_less_than(1), 0.0)
    self.assertEqual(dist.fraction_less_than(10), 0.75)

  def test_add_value_no_numpy(self):
    numpy = monitoring.numpy
    monitoring.numpy = None
    dist = monitoring._InMemoryDistributionValue()
    dist.add(1)
    dist.add(3)
    dist.add(10)
    dist.add(2)
    self.assertEqual(dist.count, 4)
    self.assertEqual(dist.sum, 16.0)
    self.assertEqual(dist.mean, 4.0)
    self.assertEqual(dist.stddev, 3.5355339059327378)
    self.assertEqual(dist.variance, 12.5)
    self.assertEqual(dist.median, 2.5)
    self.assertEqual(dist.percentile(50), 2.5)
    self.assertEqual(dist.percentile(10), 1.3)
    self.assertEqual(dist.fraction_less_than(100), 1.0)
    self.assertEqual(dist.fraction_less_than(1), 0.0)
    self.assertEqual(dist.fraction_less_than(10), 0.75)
    monitoring.numpy = numpy

  def test_window_size(self):
    dist = monitoring._InMemoryDistributionValue(window_size=3)
    dist.add(1)
    dist.add(3)
    dist.add(10)
    dist.add(2)
    self.assertEqual(dist.count, 3)
    self.assertEqual(dist.sum, 15.0)
    self.assertEqual(dist.mean, 5.0)
    self.assertEqual(dist.stddev, 3.5590260840104366)
    self.assertEqual(dist.variance, 12.666666666666664)
    self.assertEqual(dist.median, 3.0)
    self.assertEqual(dist.percentile(50), 3.0)
    self.assertEqual(dist.percentile(10), 2.2)
    self.assertEqual(dist.fraction_less_than(100), 1.0)
    self.assertEqual(dist.fraction_less_than(1), 0.0)
    self.assertEqual(dist.fraction_less_than(10), 0.6666666666666666)


class InMemoryCounterTest(unittest.TestCase):
  """Tests for in memory counter."""

  def test_counter_without_parameters(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    counter = collection.get_counter('counter', 'counter description')
    self.assertEqual(counter.namespace, '/test')
    self.assertEqual(counter.name, 'counter')
    self.assertEqual(counter.description, 'counter description')
    self.assertEqual(counter.parameter_definitions, {})
    self.assertEqual(counter.full_name, '/test/counter')
    self.assertEqual(counter.value(), 0)
    self.assertEqual(counter.increment(), 1)
    self.assertEqual(counter.value(), 1)
    self.assertEqual(counter.increment(2), 3)
    self.assertEqual(counter.value(), 3)
    self.assertIs(collection.metrics()[0], counter)

  def test_counter_with_parameters(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    counter = collection.get_counter(
        'counter', 'counter description', {'field1': str}
    )
    self.assertEqual(counter.namespace, '/test')
    self.assertEqual(counter.name, 'counter')
    self.assertEqual(counter.description, 'counter description')
    self.assertEqual(counter.parameter_definitions, {'field1': str})
    self.assertEqual(counter.full_name, '/test/counter')
    self.assertEqual(counter.value(field1='foo'), 0)
    self.assertEqual(counter.increment(field1='foo'), 1)
    self.assertEqual(counter.value(field1='bar'), 0)
    self.assertEqual(counter.increment(field1='bar'), 1)
    self.assertEqual(counter.increment(field1='foo', delta=2), 3)
    self.assertEqual(counter.value(field1='foo'), 3)

    with self.assertRaisesRegex(TypeError, '.* has type .* but expected type'):
      counter.increment(field1=1)

    with self.assertRaisesRegex(KeyError, '.* is required but not given'):
      counter.increment()

    with self.assertRaisesRegex(KeyError, '.* is not defined but provided'):
      counter.increment(field1='foo', field2='a')


class InMemoryScalarTest(unittest.TestCase):
  """Tests for in memory scalar."""

  def test_scalar_without_parameters(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    scalar = collection.get_scalar('scalar', 'scalar description')
    self.assertEqual(scalar.namespace, '/test')
    self.assertEqual(scalar.name, 'scalar')
    self.assertEqual(scalar.description, 'scalar description')
    self.assertEqual(scalar.parameter_definitions, {})
    self.assertEqual(scalar.full_name, '/test/scalar')
    self.assertEqual(scalar.value(), 0)
    self.assertEqual(scalar.increment(), 1)
    self.assertEqual(scalar.value(), 1)
    scalar.set(3)
    self.assertEqual(scalar.increment(2), 5)
    self.assertEqual(scalar.value(), 5)

  def test_scalar_with_parameters(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    scalar = collection.get_scalar(
        'scalar', 'scalar description', {'field1': str}, float
    )
    self.assertEqual(scalar.namespace, '/test')
    self.assertEqual(scalar.name, 'scalar')
    self.assertEqual(scalar.description, 'scalar description')
    self.assertEqual(scalar.parameter_definitions, {'field1': str})
    self.assertEqual(scalar.full_name, '/test/scalar')
    self.assertEqual(scalar.value(field1='foo'), 0.0)
    scalar.set(2.5, field1='bar')
    self.assertEqual(scalar.value(field1='bar'), 2.5)
    self.assertEqual(scalar.increment(1.1, field1='bar'), 3.6)
    self.assertEqual(scalar.value(field1='bar'), 3.6)
    self.assertEqual(scalar.value(field1='foo'), 0.0)


class InMemoryDistributionTest(unittest.TestCase):
  """Tests for in memory distribution."""

  def test_distribution_without_parameters(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    dist = collection.get_distribution(
        'distribution', 'distribution description'
    )
    self.assertEqual(dist.namespace, '/test')
    self.assertEqual(dist.name, 'distribution')
    self.assertEqual(dist.description, 'distribution description')
    self.assertEqual(dist.parameter_definitions, {})
    self.assertEqual(dist.full_name, '/test/distribution')
    v = dist.value()
    self.assertEqual(v.count, 0)
    dist.record(1)
    dist.record(2)
    dist.record(3)
    v = dist.value()
    self.assertEqual(v.count, 3)

    dist = collection.get_distribution(
        'distribution2', 'distribution description'
    )
    with dist.record_duration():
      time.sleep(0.1)
    self.assertGreaterEqual(dist.value().mean, 100)

  def test_distribution_with_parameters(self):
    collection = monitoring.InMemoryMetricCollection('/test')
    dist = collection.get_distribution(
        'distribution', 'distribution description', {'field1': str}
    )
    self.assertEqual(dist.namespace, '/test')
    self.assertEqual(dist.name, 'distribution')
    self.assertEqual(dist.description, 'distribution description')
    self.assertEqual(dist.parameter_definitions, {'field1': str})
    self.assertEqual(dist.full_name, '/test/distribution')
    value = dist.value(field1='foo')
    self.assertEqual(value.count, 0)
    dist.record(1, field1='foo')
    dist.record(2, field1='foo')
    dist.record(3, field1='bar')
    value = dist.value(field1='foo')
    self.assertEqual(value.count, 2)
    value = dist.value(field1='bar')
    self.assertEqual(value.count, 1)

    dist = collection.get_distribution(
        'distribution2', 'distribution description', {'error': str}
    )
    with self.assertRaises(ValueError):
      with dist.record_duration():
        time.sleep(0.1)
        raise ValueError()
    self.assertGreaterEqual(dist.value(error='ValueError').mean, 100)
    with dist.record_duration():
      time.sleep(0.1)
    self.assertGreaterEqual(dist.value(error='').mean, 100)


if __name__ == '__main__':
  unittest.main()
