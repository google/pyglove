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
"""NATS-Bench with PyGlove.

https://arxiv.org/abs/2009.00437
NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size

Please refer the original code via:
https://github.com/D-X-Y/AutoDL-Projects
"""

import os
import time
from absl import app
from absl import flags

import nats_bench
import pyglove as pg


flags.DEFINE_string(
    'output_dir', None, 'An optional directory to dump results during search.')

flags.DEFINE_integer(
    'repeat_start', 0,
    'Start index for repeating the search. This is used for running multiple '
    'search in parallel while outputting to the same output directory.')

flags.DEFINE_integer(
    'repeat_end', 1,
    'End index for repeating the search. This is used for running multiple '
    'search in parallel while outputting to the same output directory.')

flags.DEFINE_integer(
    'max_train_hours', int(2e4), 'Max training hours to simulate in a search.')

flags.DEFINE_string(
    'dataset', 'cifar10', 'cifar10, cifar100, or ImageNet16-120.'
    'This flag indicates the dataset used in NATS-Bench.')

flags.DEFINE_string(
    'search_space', 'tss', 'tss or sss, indicating the search space in NATS.')

flags.DEFINE_string(
    'algorithm', 'random',
    '"random", "evolution", "ppo" or path to a JSON file for '
    'serialized algorithm.')

flags.DEFINE_enum('file_system', 'default', ['default', 'google'],
                  'Which fileystem to use. See nats_bench.api_utils.')

# Placeholder for Google-internal tuning backend flags.


FLAGS = flags.FLAGS

DEFAULT_NATS_FILEs = dict(tss=None, sss=None)

# Results in the paper use reporting epochs $H^1$ and $H^2$ for the topology
# and size search spaces respectively. See section 3.3 of the paper.
DEFAULT_REPORTING_EPOCH = dict(tss=200, sss=90)
VALIDATION_SET_REPORTING_EPOCH = 12


@pg.functor([
    ('ops', pg.typing.List(pg.typing.Str())),
    ('num_nodes', pg.typing.Int())
])
def model_tss_spc(ops, num_nodes):
  """The architecture in the topology search space of NATS-Bench."""
  nodes, k = [], 0
  for i in range(1, num_nodes):
    xstrs = []
    for j in range(i):
      xstrs.append('{:}~{:}'.format(ops[k], j))
      k += 1
    nodes.append('|' + '|'.join(xstrs) + '|')
  return '+'.join(nodes)


@pg.functor([
    ('channels', pg.typing.List(pg.typing.Int()))
])
def model_sss_spc(channels):
  """The architecture in the size search space of NATS-Bench."""
  return ':'.join(str(x) for x in channels)


def get_search_space(ss_indicator):
  """The default search space in NATS-Bench.

  Args:
    ss_indicator: tss or sss, indicating the topology or size search space.

  Returns:
    A hyper model object that repesents a search space.
  """
  info = nats_bench.search_space_info('nats-bench', ss_indicator)
  if ss_indicator == 'tss':
    total = info['num_nodes'] * (info['num_nodes'] - 1) // 2
    return model_tss_spc(
        pg.sublist_of(total, info['op_names'], choices_distinct=False),
        info['num_nodes'])
  elif ss_indicator == 'sss':
    return model_sss_spc(
        pg.sublist_of(
            info['num_layers'], info['candidates'], choices_distinct=False))


def search(nats_api,
           search_model,
           algo,
           dataset='cifar10',
           reporting_epoch=12,
           max_train_hours=2e4):
  """Define the search procedure.

  Args:
    nats_api: the NATS-Bench object.
    search_model: which is a `model` object annotated with `one_of`.
    algo: algorithm for search.
    dataset: the target dataset
    reporting_epoch: Use test set results for models trained for this
      many epochs.
    max_train_hours: max time budget to train the models, which is the sum
      of training time queried from NAS-Bench.

  Returns:
    A tuple of (total time spent at step i for all steps,
                best validation accuracy at step i for all steps,
                best test accuracy at step i for all steps)
  """
  nats_api.reset_time()
  times, best_valids, best_tests = [0.0], [0.0], [0.0]
  valid_models = 0
  time_spent = 0
  start_time = time.time()
  last_report_time = start_time
  for model, feedback in pg.sample(search_model, algo):
    spec = model()
    (validation_accuracy, _, _, _) = nats_api.simulate_train_eval(
        spec, dataset=dataset, hp=VALIDATION_SET_REPORTING_EPOCH)
    time_spent = nats_api.used_time
    more_info = nats_api.get_more_info(spec, dataset, hp=reporting_epoch)
    valid_models += 1
    feedback(validation_accuracy)
    if validation_accuracy > best_valids[-1]:
      best_valids.append(validation_accuracy)
      best_tests.append(more_info['test-accuracy'])
    else:
      best_valids.append(best_valids[-1])
      best_tests.append(best_tests[-1])

    times.append(time_spent)
    time_spent_in_hours = time_spent / (60 * 60)
    if time_spent_in_hours > max_train_hours:
      # Break the first time we exceed the budget.
      break
    if feedback.id % 100 == 0:
      now = time.time()
      print(f'Tried {feedback.id} models, valid {valid_models}, '
            f'time_spent {time_spent}, elapse since last report: '
            f'{now - last_report_time} seconds.')
      last_report_time = now
  print(f'Total time elapse: {time.time() - start_time} seconds.')
  # Remove the first element of each list because these are placeholders
  # used for computing the current max. They don't correspond to
  # actual results from nats_api.
  return times[1:], best_valids[1:], best_tests[1:]


def get_algorithm(algorithm_str):
  """Creates algorithm."""
  if algorithm_str == 'random':
    return pg.generators.Random()
  elif algorithm_str == 'evolution':
    return pg.evolution.regularized_evolution(
        mutator=pg.evolution.mutators.Uniform(),
        population_size=50, tournament_size=10)  # pytype: disable=wrong-arg-types  # gen-stub-imports
  elif algorithm_str == 'ppo':
    return pg.generators.PPO(  # pytype: disable=module-attr  # gen-stub-imports
        train_batch_size=20, update_batch_size=10, num_updates_per_feedback=10)
  else:
    return pg.load(algorithm_str)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.repeat_end <= FLAGS.repeat_start:
    raise ValueError(
        f'Flag `repeat_end` must be greater than `repeat_start`. '
        f'Encountered: {FLAGS.repeat_end}, {FLAGS.repeat_start}.')

  # Placeholder for Google-internal tuning backend setup.

  # Load the dataset.
  nats_bench.api_utils.reset_file_system(FLAGS.file_system)
  nats_api = nats_bench.create(
      DEFAULT_NATS_FILEs[FLAGS.search_space],
      FLAGS.search_space, fast_mode=False, verbose=False)

  # Create search space.
  search_model = get_search_space(FLAGS.search_space)
  reporting_epoch = DEFAULT_REPORTING_EPOCH[FLAGS.search_space]

  # Start search.
  for i in range(FLAGS.repeat_start, FLAGS.repeat_end):
    print(f'Repeat #{i}')
    algorithm = get_algorithm(FLAGS.algorithm)
    times, best_valid, best_test = search(nats_api, search_model, algorithm,
                                          FLAGS.dataset, reporting_epoch,
                                          FLAGS.max_train_hours)

    print('%15s %15s %15s %15s' % ('# trials',
                                   'best valid',
                                   'best test',
                                   'simulated train hours'))
    print('%15d %15.4f %15.4f %15d' % (len(times),
                                       best_valid[-1],
                                       best_test[-1],
                                       times[-1]))

    if FLAGS.output_dir:
      pg.Dict(times=times, best_valid=best_valid, best_test=best_test).save(
          os.path.join(FLAGS.output_dir, f'repeat_{i}.json'))


if __name__ == '__main__':
  app.run(main)
