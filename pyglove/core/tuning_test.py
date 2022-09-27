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
"""Tests for pyglove.tuning."""

import inspect
import threading
import time
import unittest

from pyglove.core import geno
from pyglove.core import hyper
from pyglove.core import symbolic
from pyglove.core import tuning
from pyglove.core import typing


class DummyEarlyStoppingPolicy(tuning.EarlyStoppingPolicy):
  """Early stopping policy for testing."""

  def should_stop_early(self, trial, measurement):
    # NOTE(daiyip): stop trial 2, 4, 8 at step 1.
    if trial.id in [2, 4, 8]:
      if trial.measurements and trial.measurements[-1].step > 0:
        return True
    return False


class DummySingleObjectiveAlgorithm(geno.DNAGenerator):
  """Dummy single-object algorithm for testing."""

  def _setup(self):
    self.rewards = []

  def _propose(self):
    return geno.DNA(1)

  def _feedback(self, dna, reward):
    self.rewards.append(reward)


class DummyMultiObjectiveAlgorithm(DummySingleObjectiveAlgorithm):
  """Dummy multi-objective algorithm for testing."""

  @property
  def multi_objective(self):
    return True


class TrialTest(unittest.TestCase):
  """Test for Trial class."""

  def testGetRewardForFeedback(self):
    """Test `Trial.get_reward`."""
    t = tuning.Trial(
        id=0, dna=geno.DNA(0),
        status='PENDING',
        created_time=int(time.time()))
    self.assertIsNone(t.get_reward_for_feedback())

    t = tuning.Trial(
        id=0, dna=geno.DNA(0),
        status='COMPLETED',
        infeasible=True,
        created_time=int(time.time()))
    self.assertIsNone(t.get_reward_for_feedback())

    t = tuning.Trial(
        id=0, dna=geno.DNA(0),
        status='COMPLETED',
        infeasible=False,
        final_measurement=tuning.Measurement(
            step=1, elapse_secs=0.1, reward=1.0, metrics=dict(
                accuracy=0.9, latency=750.0)),
        created_time=int(time.time()))
    self.assertEqual(t.get_reward_for_feedback(), 1.0)
    self.assertEqual(t.get_reward_for_feedback(['accuracy', 'latency']),
                     (0.9, 750.0))
    with self.assertRaisesRegex(
        ValueError, 'Metric \'foo\' does not exist'):
      t.get_reward_for_feedback(['foo'])


class SamplingTest(unittest.TestCase):
  """Test `sample` with in-memory tuning backend."""

  def testSample(self):
    """Test `sample`."""
    feedbacks = []
    algo = geno.Random(seed=1)
    for example, f in tuning.sample(
        hyper_value=symbolic.Dict(x=hyper.oneof([5, 6, 7])),
        algorithm=algo,
        num_examples=10,
        name='my_search',):
      self.assertIsNone(f.checkpoint_to_warm_start_from)

      f.set_metadata('example', example)
      self.assertEqual(f.get_metadata('example'), example)

      f.set_metadata('global_key', 1, per_trial=False)
      self.assertEqual(f.get_metadata('global_key', per_trial=False), 1)

      f.add_link('filepath', f'https://path/to/file_{example.x}')
      with f.skip_on_exceptions([ValueError]):
        if f.id == 5:
          raise ValueError('bad trial')
        f(example.x)
      feedbacks.append(f)

    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(len(feedbacks), 10)
    self.assertEqual([c.id for c in feedbacks], list(range(1, 11)))

    # Test `poll_result`.
    result = tuning.poll_result('my_search')
    self.assertTrue(result.is_active)
    self.assertIsNotNone(result.best_trial)
    self.assertEqual(result.best_trial.final_measurement.reward, 7.0)
    self.assertEqual(result.best_trial.dna, geno.DNA.parse([2]))
    self.assertEqual(result.best_trial.metadata.example, symbolic.Dict(x=7))
    self.assertEqual(
        result.best_trial.related_links.filepath, 'https://path/to/file_7')

    self.assertEqual(result.metadata['global_key'], 1)
    self.assertEqual(len(result.trials), 10)
    self.assertEqual(
        str(result),
        inspect.cleandoc('''{
          'name': 'my_search',
          'status': {
            'COMPLETED': '10/10'
          },
          'infeasible': '1/10',
          'best_trial': {
            'id': 2,
            'reward': 7.0,
            'step': 0,
            'dna': 'DNA(2)'
          }
        }'''))

  def testSkipOnExceptions(self):
    """Test various forms of `pg.tuning.skip_on_exceptions`."""
    search_space = symbolic.Dict(x=hyper.oneof(range(10)))
    algo = geno.Random(seed=1)
    sample = tuning.sample(search_space, algo)
    _, f = next(sample)

    with f.skip_on_exceptions((ValueError,)):
      # should succeed.
      f(0)
    self.assertEqual(algo.num_proposals, 1)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions((ValueError,)):
      # should skip.
      raise ValueError
    self.assertEqual(algo.num_proposals, 2)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions((ValueError,)):
      # should skip.
      raise ValueError('abc')
    self.assertEqual(algo.num_proposals, 3)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions((Exception,)):
      # should skip.
      raise ValueError('abc')
    self.assertEqual(algo.num_proposals, 4)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions(((ValueError, '.*a'),)):
      # should skip.
      raise ValueError('abc')
    self.assertEqual(algo.num_proposals, 5)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions(((Exception, '.*a'),)):
      # should skip.
      raise ValueError('abc')
    self.assertEqual(algo.num_proposals, 6)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with self.assertRaisesRegex(
        ValueError, 'bcd'):
      with f.skip_on_exceptions(((Exception, '.*a'),)):
        # should skip.
        raise ValueError('bcd')
    self.assertEqual(algo.num_proposals, 7)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions(((ValueError, '.*a'),
                               (ValueError, '.*b'),
                               KeyError)):
      # should skip.
      raise ValueError('bcd')
    self.assertEqual(algo.num_proposals, 7)
    self.assertEqual(algo.num_feedbacks, 1)

    _, f = next(sample)
    with f.skip_on_exceptions(((ValueError, '.*a'),
                               (ValueError, '.*b'),
                               KeyError)):
      # should skip.
      raise KeyError
    self.assertEqual(algo.num_proposals, 8)
    self.assertEqual(algo.num_feedbacks, 1)

  def testSampleWithRaceCondition(self):
    """Test `pg.sample` with race condition among co-workers."""
    _, f = next(tuning.sample(hyper.oneof([1, 2, 3]), geno.Random(seed=1)))

    f(1)
    with self.assertRaisesRegex(
        tuning.RaceConditionError,
        '.*Measurements can only be added to PENDING trials.*'):
      f.add_measurement(0.1)

    with f.ignore_race_condition():
      f.add_measurement(0.1)

  def testSampleWithDefineByRunSearchSpace(self):
    """Test `pg.sample` with define-by-run search space definition."""
    def fun():
      return hyper.oneof([1, 2, 3]) + hyper.oneof([3, 4, 5])

    for example, f in tuning.sample(
        hyper.trace(fun),
        geno.Sweeping(), num_examples=6, name='define-by-run-search'):
      with example():
        f(fun())

    # Test `poll_result`.
    result = tuning.poll_result('define-by-run-search')
    rewards = [t.final_measurement.reward for t in result.trials]
    self.assertEqual(rewards, [4., 5., 6., 5., 6., 7.])

  def testSampleWithContinuationAndEndLoop(self):
    """Test `pg.sample` with continuation and `end_loop`."""
    hyper_value = symbolic.Dict(x=hyper.oneof([1, 2, 3]))
    for _, feedback in tuning.sample(
        hyper_value=hyper_value,
        algorithm=geno.Random(seed=1),
        name='my_search2'):
      # Always invoke the feedback function in order to advance
      # to the next trail.
      feedback(0.)
      if feedback.id == 2:
        # We break without ending the loop
        break

    result = tuning.poll_result('my_search2')
    self.assertTrue(result.is_active)
    self.assertEqual(len(result.trials), 2)

    sample1 = tuning.sample(
        name='my_search2',
        hyper_value=hyper_value,
        algorithm=geno.Random(seed=1))

    # Make sure sampling within the same worker get the same trial IDs before
    # feedback.
    _, c1 = next(sample1)
    self.assertEqual(c1.id, 3)
    self.assertEqual(c1.get_trial().id, 3)

    _, c1b = next(sample1)
    self.assertEqual(c1b.id, 3)
    c1(0.)

    # Make sure sampling within the same worker get different trial IDs after
    # previous trial is done.
    _, c1c = next(sample1)
    self.assertEqual(c1c.id, 4)

    # Make sure after `end_loop`, sampling will raise StopIteration error.
    # Also the in-memory study is no longer active.
    c1c.end_loop()
    with self.assertRaises(StopIteration):
      next(sample1)
    self.assertFalse(result.is_active)

  def testSamplingWithMetricsToOptimize(self):
    """Test sampling with argument 'metrics_to_optimize'."""
    # Test 'metrics_to_optimize' with a single objective algorithm.
    algo = DummySingleObjectiveAlgorithm()
    _, f = next(tuning.sample(
        hyper.oneof([1, 2]), algo,
        metrics_to_optimize=['reward']))

    with self.assertRaisesRegex(
        ValueError,
        '\'reward\' must be provided as it is a goal to optimize'):
      f()
    f(1.0)
    self.assertEqual(algo.rewards, [1.0])

    algo = DummySingleObjectiveAlgorithm()
    _, f = next(tuning.sample(
        hyper.oneof([1, 2]), algo,
        metrics_to_optimize=['accuracy']))

    with self.assertRaisesRegex(
        ValueError,
        'Metric .* must be provided as it is a goal to optimize.'):
      f.add_measurement(0.0)

    with self.assertRaisesRegex(
        ValueError,
        '\'reward\' .* is provided while it is not a goal to optimize'):
      f.add_measurement(0.0, metrics={'accuracy': 2.0})

    f(metrics={'accuracy': 2.0})
    self.assertEqual(algo.rewards, [2.0])

    with self.assertRaisesRegex(
        ValueError,
        '\'metrics_to_optimize\' should include only 1 metric as '
        'multi-objective optimization is not supported'):
      next(tuning.sample(
          hyper.oneof([1, 2]), DummySingleObjectiveAlgorithm(),
          metrics_to_optimize=['reward', 'accuracy', 'latency']))

    # Test 'metrics_to_optimize' with a multi-objective algorithm.
    algo = DummyMultiObjectiveAlgorithm()
    it = tuning.sample(
        hyper.oneof([1, 2]), algo,
        metrics_to_optimize=['reward', 'accuracy', 'latency'])
    _, f = next(it)
    with self.assertRaisesRegex(
        ValueError,
        '\'reward\' must be provided as it is a goal to optimize'):
      f.add_measurement(metrics={'accuracy': 0.9, 'latency': 0.5})
    f(0., metrics={'accuracy': 0.9, 'latency': 0.5})

    self.assertEqual(algo.rewards, [(0., 0.9, 0.5)])
    _, f = next(it)

    f((0.1, 0.2, 0.3))
    self.assertEqual(algo.rewards, [(0., 0.9, 0.5), (0.1, 0.2, 0.3)])
    self.assertEqual(f.get_trial().final_measurement.metrics, {
        'accuracy': 0.2,
        'latency': 0.3
    })

    _, f = next(it)
    with self.assertRaisesRegex(
        ValueError,
        'The number of items in the reward .* does not match '):
      f((0.1, 0.2))

    with self.assertRaisesRegex(
        ValueError,
        'The value for metric .* is provided from both .* different values'):
      f((0.1, 0.2, 0.3), metrics={'accuracy': 0.5})

    algo = DummyMultiObjectiveAlgorithm()
    _, f = next(tuning.sample(hyper.oneof([1, 2]), algo))
    f(1.)
    self.assertEqual(algo.rewards, [(1.,)])

  def testSampleWithControlleredEvaluatedRewards(self):
    """Test scenario when controller provided the reward for a DNA."""

    @symbolic.members([
        ('num_objectives', typing.Int(min_value=1))
    ])
    class MaybeControllerEvaluated(geno.DNAGenerator):

      def _setup(self):
        self._dna = None

      @property
      def multi_objective(self):
        return self.num_objectives > 1

      def _propose(self):
        self._dna = self.dna_spec.next_dna(self._dna)
        if self.num_proposals % 2 == 1:
          if self.multi_objective:
            reward = tuple([0.] * self.num_objectives)
          else:
            reward = 0.
          self._dna.set_metadata('reward', reward)
        return self._dna

    algo = MaybeControllerEvaluated(1)
    client_evaluated = []
    for x, f in tuning.sample(hyper.oneof(range(100)), algo, 10):
      f(x)
      client_evaluated.append(f.dna)

    self.assertEqual(
        client_evaluated,
        [geno.DNA(0), geno.DNA(2), geno.DNA(4), geno.DNA(6), geno.DNA(8)])
    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(algo.num_feedbacks, 10)

    algo = MaybeControllerEvaluated(2)
    client_evaluated = []
    for x, f in tuning.sample(
        hyper.oneof(range(100)), algo, 10, metrics_to_optimize=['a', 'b']):
      f(None, metrics=dict(a=0., b=0.))
      client_evaluated.append(f.dna)

    self.assertEqual(
        client_evaluated,
        [geno.DNA(0), geno.DNA(2), geno.DNA(4), geno.DNA(6), geno.DNA(8)])
    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(algo.num_feedbacks, 10)

    it = tuning.sample(hyper.oneof(range(100)), algo, 10)
    # The first call will get a DNA to be evaluated at the client side.
    # It should pass
    x, f = next(it)
    f(0.0)

    # The second call will poll a DNA evaluated at the controller side.
    # Since the number of reward items is 2 while metrics_to_optimize is 1,
    # there will be an error.
    with self.assertRaisesRegex(
        ValueError, 'The number of items in the reward .* does not match'):
      next(it)

  def testBadSampling(self):
    """Test bad sampling."""
    _, f = next(tuning.sample(
        symbolic.Dict(x=hyper.oneof([1, 2])),
        geno.Random(seed=1), 1))
    with self.assertRaisesRegex(
        ValueError, 'At least one measurement should be added for trial'):
      f.done()

    with self.assertRaisesRegex(
        ValueError, '\'hyper_value\' is a constant value'):
      next(tuning.sample(1, geno.Random(seed=1)))

    with self.assertRaisesRegex(
        ValueError, 'Backend .* does not exist.'):
      next(tuning.sample(
          hyper.oneof([1, 2]), geno.Random(seed=1),
          backend='non-exist'))

    # Using the sample algorithm to optimize different search spaces will
    # trigger a value error.
    algo = geno.Random(seed=1)
    early_stopping_policy = DummyEarlyStoppingPolicy()
    next(tuning.sample(
        hyper.oneof([1, 2]), algo, 1, early_stopping_policy))
    with self.assertRaisesRegex(
        ValueError, '.* has been set up with a different DNASpec'):
      next(tuning.sample(symbolic.Dict(x=hyper.oneof([3, 4])), algo))

    algo = geno.Random(seed=1)
    with self.assertRaisesRegex(
        ValueError, '.* has been set up with a different DNASpec'):
      next(tuning.sample(
          symbolic.Dict(x=hyper.oneof([1, 2])),
          algo, 1, early_stopping_policy))

    with self.assertRaisesRegex(
        ValueError, 'Result .* does not exist.'):
      tuning.poll_result('non-exist-search')

  def testEarlyStoppingPolicy(self):
    """Test `tuning.EarlyStoppingPolicy`."""
    stopped_trial_steps = []
    early_stopping_policy = DummyEarlyStoppingPolicy()
    for _, f in tuning.sample(
        symbolic.Dict(x=hyper.oneof([1, 2])),
        geno.Random(seed=1), 10,
        early_stopping_policy,
        name='early_stopping'):
      skipped = False
      for step in [0, 1, 2]:
        if f.should_stop_early():
          stopped_trial_steps.append((f.id, step))
          skipped = True
          break
        else:
          f.add_measurement(0., step=step)
      if skipped:
        f.skip()
      else:
        f.done()

    self.assertEqual(
        stopped_trial_steps, [(2, 2), (4, 2), (8, 2)])

    result = tuning.poll_result('early_stopping')
    for t in result.trials:
      if t.id in [2, 4, 8]:
        self.assertTrue(t.infeasible)
        self.assertEqual(len(t.measurements), 2)
        self.assertEqual(
            t.final_measurement,
            tuning.Measurement(step=0, reward=0.0, elapse_secs=0.0))
      else:
        self.assertFalse(t.infeasible)
        self.assertEqual(t.final_measurement.step, 2)
        self.assertEqual(len(t.measurements), 3)

  def testLocalBackendWithMultiThreadSampling(self):
    """Test local backend with multi-thread sampling."""
    threads_trial_ids = []
    def create_worker_func(study_name, num_examples=None, group_id=None):
      hyper_value = symbolic.Dict(x=hyper.oneof([1, 2, 3]))
      def worker_func():
        trial_ids = []
        threads_trial_ids.append(trial_ids)
        for _, feedback in tuning.sample(
            hyper_value=hyper_value,
            algorithm=geno.Random(seed=1),
            group_id=group_id,
            num_examples=num_examples,
            name=study_name):
          trial_ids.append(feedback.id)
          feedback(0.)
          if feedback.id == 7:
            feedback.end_loop()
          time.sleep(0.1)
      return worker_func

    # Test multiple worker thread with different group ID.
    num_workers = 3
    worker_func = create_worker_func('mt_search_different_group', 10)
    threads = [threading.Thread(target=worker_func) for _ in range(num_workers)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    self.assertEqual(len(threads_trial_ids), 3)
    all_trial_ids = []
    for ids_per_thread in threads_trial_ids:
      all_trial_ids.extend(ids_per_thread)

    # Make sure different threads get different trials.
    self.assertCountEqual(all_trial_ids, set(all_trial_ids))
    # NOTE(daiyip): when worker with trial#7 trigger end_loop, other 2 threads
    # may already move forward to #8 and #9. But it should not go beyond that
    # point due to every iteration sleeps for 100 ms.
    self.assertIn(len(all_trial_ids), list(range(7, 7 + num_workers)))

  def testPluggableBackend(self):
    """Test pluggable backend."""

    @tuning.add_backend('test')
    class TestBackendFactory(tuning._InMemoryBackendFactory):  # pylint: disable=unused-variable
      """A backend factory for testing."""

    self.assertEqual(tuning.available_backends(), ['in-memory', 'test'])
    self.assertEqual(tuning.default_backend(), 'in-memory')
    tuning.set_default_backend('test')
    self.assertEqual(tuning.default_backend(), 'test')
    with self.assertRaisesRegex(
        ValueError, 'Backend .* does not exist'):
      tuning.set_default_backend('non-exist-backend')

    with self.assertRaisesRegex(
        TypeError, '.* is not a BackendFactory subclass'):

      @tuning.add_backend('bad')
      class BadBackendFactory:  # pylint: disable=unused-variable
        pass

if __name__ == '__main__':
  unittest.main()
