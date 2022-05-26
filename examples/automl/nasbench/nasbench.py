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
"""NAS-Bench with PyGlove.

https://arxiv.org/abs/1902.09635
Ying, etc. NAS-Bench-101: Towards Reproducible Neural Architecture Search

Please refer the original Colab via:
https://colab.sandbox.google.com/github/google-research/nasbench/blob/master/NASBench.ipynb
"""

import os
import time
from absl import app
from absl import flags

from nasbench.nasbench import api
import numpy as np
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
    'max_train_hours', int(5e6), 'Max training hours to simulate in a search.')

flags.DEFINE_string(
    'search_space', None, 'JSON file for serialized search space.')

flags.DEFINE_string(
    'algorithm', 'random',
    '"random", "evolution" or path to a JSON file for serialized algorithm.')

# Placeholder for Google-internal tuning backend flags.


FLAGS = flags.FLAGS


# !curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
# Use nasbench_full.tfrecord for full dataset (run download command above).
DEFAULT_NAS_BENCH_108_EPOCHS_FILE = 'nasbench_only108.tfrecord'


# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix


@pg.symbolize([
    ('ops', pg.typing.List(pg.typing.Str())),
    ('matrix', pg.typing.List(pg.typing.List(pg.typing.Int()))),
])
def model_spec(ops, matrix):
  """NASBench model spec that is parameterized by ops and their connections.

  Args:
    ops: a list of allowed ops except the INPUT and OUTPUT layer.
    matrix: the adjacency matrix for the connectivity of each layers, which
      should be an upper triangle matrix.

  Returns:
    A NASBench spec.
  """
  return api.ModelSpec(matrix=np.array(matrix), ops=[INPUT] + ops + [OUTPUT])


# We introduce hints so controller can deal with different knobs differently.
OP_HINTS = 0
EDGE_HINTS = 1


def default_search_space():
  """The default search space in NAS-Bench.

  This equals to the default search space of NAS-Bench, which mutate candidate
  ops and their connections.

  Returns:
    A hyper model object that repesents a search space.
  """
  matrix = [
      [pg.oneof([0, 1], hints=EDGE_HINTS) if y > x else 0
       for y in range(NUM_VERTICES)]
      for x in range(NUM_VERTICES)
  ]
  return model_spec(
      pg.manyof(NUM_VERTICES - 2, ALLOWED_OPS,
                choices_distinct=False, hints=OP_HINTS),
      matrix)


def search(nasbench, search_model, algo, repeat_id, max_train_hours=5e6):
  """Define the search procedure.

  Args:
    nasbench: NASBench object.
    search_model: which is a `model` object annotated with `oneof`.
    algo: algorithm for search.
    repeat_id: identifier of current repeat.
    max_train_hours: max time budget to train the models, which is the sum
      of training time queried from NAS-Bench.

  Returns:
    A tuple of (total time spent at step i for all steps,
                best validation accuracy at step i for all steps,
                best test accuracy at step i for all steps)
  """
  nasbench.reset_budget_counters()
  times, best_valids, best_tests = [0.0], [0.0], [0.0]
  valid_models = 0
  time_spent = 0
  start_time = time.time()
  last_report_time = start_time
  for model, feedback in pg.sample(search_model, algo, name=str(repeat_id)):
    spec = model()
    if nasbench.is_valid(spec):
      results = nasbench.query(spec)
      valid_models += 1
      feedback(results['validation_accuracy'])
      if results['validation_accuracy'] > best_valids[-1]:
        best_valids.append(results['validation_accuracy'])
        best_tests.append(results['test_accuracy'])
      else:
        best_valids.append(best_valids[-1])
        best_tests.append(best_tests[-1])
      time_spent, _ = nasbench.get_budget_counters()
      times.append(time_spent)
      if time_spent > max_train_hours:
        # Break the first time we exceed the budget.
        feedback.end_loop()
        break
    else:
      feedback.skip()

    if feedback.id % 100 == 0:
      now = time.time()
      print(f'Tried {feedback.id} models, valid {valid_models}, '
            f'time_spent {time_spent}, elapse since last report: '
            f'{now - last_report_time} seconds.')
      last_report_time = now
  print(f'Total time elapse: {time.time() - start_time} seconds.')
  return times, best_valids, best_tests


@pg.symbolize
def node_selector(x, hints):
  """A functor to select node based on hints."""
  return x.spec.hints == hints


def create_search_algorithm(flag_value):
  """Create search algorithm from flag."""
  if flag_value == 'random':
    return pg.generators.Random()
  elif flag_value == 'evolution':
    return pg.evolution.regularized_evolution(
        mutator=(
            pg.evolution.mutators.Uniform(
                where=node_selector(hints=OP_HINTS))         # pylint: disable=no-value-for-parameter
            >> pg.evolution.mutators.Uniform(
                where=node_selector(hints=EDGE_HINTS)) ** 3  # pylint: disable=no-value-for-parameter
        ),
        population_size=50,
        tournament_size=10)  # pytype: disable=wrong-arg-types  # gen-stub-imports
  else:
    return pg.load(flag_value)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.repeat_end <= FLAGS.repeat_start:
    raise ValueError(
        f'Flag `repeat_end` must be greater than `repeat_start`. '
        f'Encountered: {FLAGS.repeat_end}, {FLAGS.repeat_start}.')

  # Placeholder for Google-internal tuning backend setup.

  # Load the dataset.
  nasbench = api.NASBench(DEFAULT_NAS_BENCH_108_EPOCHS_FILE)

  # Create search space.
  if FLAGS.search_space:
    search_model = pg.load(FLAGS.search_space)
  else:
    search_model = default_search_space()

  # Create algorithm.
  algorithm = create_search_algorithm(FLAGS.algorithm)

  # Start search.
  for i in range(FLAGS.repeat_start, FLAGS.repeat_end):
    print(f'Repeat #{i}')
    times, best_valid, best_test = search(
        nasbench, search_model, algorithm, i, FLAGS.max_train_hours)

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
