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
"""Tests for `pg.sample`."""

import inspect
import threading
import time
import unittest

from pyglove.core import geno
from pyglove.core import hyper
from pyglove.core import symbolic
from pyglove.core import typing as pg_typing

# Import 'in-memory' backend as the default backend.
from pyglove.core.tuning import local_backend  # pylint: disable=unused-import

from pyglove.core.tuning import protocols
from pyglove.core.tuning.backend import poll_result as pg_poll_result
from pyglove.core.tuning.early_stopping import EarlyStoppingPolicy
from pyglove.core.tuning.sample import sample as pg_sample


class DummyEarlyStoppingPolicy(EarlyStoppingPolicy):
  """Early stopping policy for testing."""

  def should_stop_early(self, trial):
    # NOTE(daiyip): stop trial 2, 4, 8 at step 1.
    if trial.id in [2, 4, 8]:
      if trial.measurements and trial.measurements[-1].step > 0:
        return True
    return False


class DummySingleObjectiveAlgorithm(geno.DNAGenerator):
  """Single-object algorithm for testing."""

  def _setup(self):
    self.rewards = []

  def _propose(self):
    return geno.DNA(1)

  def _feedback(self, dna, reward):
    self.rewards.append(reward)


class DummyMultiObjectiveAlgorithm(DummySingleObjectiveAlgorithm):
  """Multi-objective algorithm for testing."""

  @property
  def multi_objective(self):
    return True


class SamplingTest(unittest.TestCase):
  """Test `pg.sample` with the default tuning backend."""

  def test_sample_with_set_metadata(self):
    feedbacks = []
    algo = geno.Random(seed=1)
    for example, f in pg_sample(
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
    result = pg_poll_result('my_search')
    self.assertTrue(result.is_active)
    self.assertIsNotNone(result.best_trial)
    self.assertEqual(result.best_trial.final_measurement.reward, 7.0)
    self.assertEqual(result.best_trial.dna, geno.DNA.parse([2]))
    self.assertEqual(result.best_trial.metadata.example, symbolic.Dict(x=7))
    self.assertEqual(
        result.best_trial.related_links.filepath, 'https://path/to/file_7')

    self.assertEqual(result.metadata['global_key'], 1)
    self.assertEqual(len(result.trials), 10)
    # TODO(daiyip): Move this test to 'local_backend_test.py'
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

  def test_sample_with_skip_on_exceptions(self):
    search_space = symbolic.Dict(x=hyper.oneof(range(10)))
    algo = geno.Random(seed=1)
    sample = pg_sample(search_space, algo)
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

  def test_sample_with_race_condition(self):
    _, f = next(pg_sample(hyper.oneof([1, 2, 3]), geno.Random(seed=1)))

    f(1)
    with self.assertRaisesRegex(
        protocols.RaceConditionError,
        '.*Measurements can only be added to PENDING trials.*'):
      f.add_measurement(0.1)

    with f.ignore_race_condition():
      f.add_measurement(0.1)

  def test_sample_with_dynamic_evaluation(self):
    def fun():
      return hyper.oneof([1, 2, 3]) + hyper.oneof([3, 4, 5])

    for example, f in pg_sample(
        hyper.trace(fun),
        geno.Sweeping(), num_examples=6, name='define-by-run-search'):
      with example():
        f(fun())

    # Test `poll_result`.
    result = pg_poll_result('define-by-run-search')
    rewards = [t.final_measurement.reward for t in result.trials]
    self.assertEqual(rewards, [4., 5., 6., 5., 6., 7.])

  def test_sample_with_continuation_and_end_loop(self):
    hyper_value = symbolic.Dict(x=hyper.oneof([1, 2, 3]))
    for _, feedback in pg_sample(
        hyper_value=hyper_value,
        algorithm=geno.Random(seed=1),
        name='my_search2'):
      # Always invoke the feedback function in order to advance
      # to the next trail.
      feedback(0.)
      if feedback.id == 2:
        # We break without ending the loop
        break

    result = pg_poll_result('my_search2')
    self.assertTrue(result.is_active)
    self.assertEqual(len(result.trials), 2)

    sample1 = pg_sample(
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
    # Also the study is no longer active.
    c1c.end_loop()
    with self.assertRaises(StopIteration):
      next(sample1)
    self.assertFalse(result.is_active)

  def test_sample_with_single_objective(self):
    algo = DummySingleObjectiveAlgorithm()
    _, f = next(pg_sample(
        hyper.oneof([1, 2]), algo,
        metrics_to_optimize=['reward']))

    with self.assertRaisesRegex(
        ValueError,
        '\'reward\' must be provided as it is a goal to optimize'):
      f()
    f(1.0)
    self.assertEqual(algo.rewards, [1.0])

    algo = DummySingleObjectiveAlgorithm()
    _, f = next(pg_sample(
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
      next(pg_sample(
          hyper.oneof([1, 2]), DummySingleObjectiveAlgorithm(),
          metrics_to_optimize=['reward', 'accuracy', 'latency']))

  def test_sample_with_multi_objective(self):
    algo = DummyMultiObjectiveAlgorithm()
    it = pg_sample(
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
    _, f = next(pg_sample(hyper.oneof([1, 2]), algo))
    f(1.)
    self.assertEqual(algo.rewards, [(1.,)])

  def test_sample_with_controller_evaluated_rewards(self):

    @symbolic.members([
        ('num_objectives', pg_typing.Int(min_value=1))
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
    for x, f in pg_sample(hyper.oneof(range(100)), algo, 10):
      f(x)
      client_evaluated.append(f.dna)

    self.assertEqual(
        client_evaluated,
        [geno.DNA(0), geno.DNA(2), geno.DNA(4), geno.DNA(6), geno.DNA(8)])
    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(algo.num_feedbacks, 10)

    algo = MaybeControllerEvaluated(2)
    client_evaluated = []
    for x, f in pg_sample(
        hyper.oneof(range(100)), algo, 10, metrics_to_optimize=['a', 'b']):
      f(None, metrics=dict(a=0., b=0.))
      client_evaluated.append(f.dna)

    self.assertEqual(
        client_evaluated,
        [geno.DNA(0), geno.DNA(2), geno.DNA(4), geno.DNA(6), geno.DNA(8)])
    self.assertEqual(algo.num_proposals, 10)
    self.assertEqual(algo.num_feedbacks, 10)

    it = pg_sample(hyper.oneof(range(100)), algo, 10)
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

  def test_sample_with_early_stopping(self):
    stopped_trial_steps = []
    early_stopping_policy = DummyEarlyStoppingPolicy()
    for _, f in pg_sample(
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

    self.assertEqual(stopped_trial_steps, [(2, 2), (4, 2), (8, 2)])

    result = pg_poll_result('early_stopping')
    for t in result.trials:
      if t.id in [2, 4, 8]:
        self.assertTrue(t.infeasible)
        self.assertEqual(len(t.measurements), 2)
        self.assertEqual(
            t.final_measurement,
            protocols.Measurement(step=0, reward=0.0, elapse_secs=0.0))
      else:
        self.assertFalse(t.infeasible)
        self.assertEqual(t.final_measurement.step, 2)
        self.assertEqual(len(t.measurements), 3)

  def test_sample_with_concurrent_workers(self):
    threads_trial_ids = []
    def create_worker_func(study_name, num_examples=None, group_id=None):
      hyper_value = symbolic.Dict(x=hyper.oneof([1, 2, 3]))
      def worker_func():
        trial_ids = []
        threads_trial_ids.append(trial_ids)
        for _, feedback in pg_sample(
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

  def test_bad_sampling(self):
    _, f = next(pg_sample(
        symbolic.Dict(x=hyper.oneof([1, 2])),
        geno.Random(seed=1), 1))
    with self.assertRaisesRegex(
        ValueError, 'At least one measurement should be added for trial'):
      f.done()

    with self.assertRaisesRegex(
        ValueError, '\'hyper_value\' is a constant value'):
      next(pg_sample(1, geno.Random(seed=1)))

    with self.assertRaisesRegex(
        ValueError, 'Backend .* does not exist.'):
      next(pg_sample(
          hyper.oneof([1, 2]), geno.Random(seed=1),
          backend='non-exist'))

    # Using the sample algorithm to optimize different search spaces will
    # trigger a value error.
    algo = geno.Random(seed=1)
    early_stopping_policy = DummyEarlyStoppingPolicy()
    next(pg_sample(
        hyper.oneof([1, 2]), algo, 1, early_stopping_policy))
    with self.assertRaisesRegex(
        ValueError, '.* has been set up with a different DNASpec'):
      next(pg_sample(symbolic.Dict(x=hyper.oneof([3, 4])), algo))

    algo = geno.Random(seed=1)
    with self.assertRaisesRegex(
        ValueError, '.* has been set up with a different DNASpec'):
      next(pg_sample(
          symbolic.Dict(x=hyper.oneof([1, 2])),
          algo, 1, early_stopping_policy))

    with self.assertRaisesRegex(
        ValueError, 'Result .* does not exist.'):
      pg_poll_result('non-exist-search')


if __name__ == '__main__':
  unittest.main()
