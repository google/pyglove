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
"""NAS on MNIST with define-by-run search space definition.

This is a basic working ML program which trains MNIST.
The code is modified from the tf.keras tutorial here:
https://www.tensorflow.org/tutorials/keras/classification
"""

from typing import Tuple

from absl import app
from absl import flags
import numpy as np
import pyglove as pg
import tensorflow as tf


flags.DEFINE_integer(
    'max_trials', 10, 'Number of max trials for tuning.')

flags.DEFINE_integer(
    'num_epochs', 10, 'Number of epochs to train for each trail.')

# Placeholder for Google-internal tuning backend flags.


FLAGS = flags.FLAGS


def download_and_prep_data() -> Tuple[np.ndarray,
                                      np.ndarray,
                                      np.ndarray,
                                      np.ndarray]:
  """Download dataset and scale to [0, 1].

  Returns:
    tr_x: Training data.
    tr_y: Training labels.
    te_x: Testing data.
    te_y: Testing labels.
  """
  mnist_dataset = tf.keras.datasets.mnist
  (tr_x, tr_y), (te_x, te_y) = mnist_dataset.load_data()
  tr_x = tr_x / 255.0
  te_x = te_x / 255.0
  return tr_x, tr_y, te_x, te_y


def create_model() -> tf.keras.Model:
  """Create model for training.

  Create a simple tf.keras model for training.

  Returns:
    The model to use for training.
  """
  return tf.keras.Sequential(pg.oneof([
      # NOTE(daiyip): We use a zero-argument `lambda` function to
      # wrap each candidate in order to construct a conditional search
      # space.
      lambda: [  # pylint: disable=g-long-lambda
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(pg.oneof([64, 128]),
                                pg.oneof(['relu', 'sigmoid']))
      ],
      lambda: [  # pylint: disable=g-long-lambda
          tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 1))),
          tf.keras.layers.Conv2D(pg.oneof([64, 128]),
                                 pg.oneof([(3, 3), (5, 5)]),
                                 activation=pg.oneof(['relu', 'sigmoid'])),
          tf.keras.layers.Flatten()
      ]
  ]) + [
      tf.keras.layers.Dense(10, activation='softmax')
  ])


def train_and_eval(input_data, num_epochs) -> None:
  """Run training and evaluation."""
  tr_x, tr_y, te_x, te_y = input_data
  model = create_model()
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(tr_x, tr_y, epochs=num_epochs)
  test_loss, test_acc = model.evaluate(te_x, te_y, verbose=2)
  print('Test loss: {}, accuracy: {}'.format(test_loss, test_acc))
  return test_acc


def tune(max_trials, num_epochs):
  """Tune MNIST model via random search."""
  results = []
  input_data = download_and_prep_data()
  # NOTE(daiyip): `pg.sample` returns an iterator of (example, feedback_fn)
  # from a hyper object (the search space) and a DNAGenerator (the search
  # algorithm), with an optional flag to set the max examples to sample.
  # Different from defining a search space from symbolized classes or functors,
  # `pg.hyper.trace` allows users to define a search space
  # without symbolizing user classes, by eagerly executing the user function to
  # collect decision points in the search space. Each point in the search space
  # is a list of numbers materialized from the decision points in the search
  # space. Instead of using these numbers to construct the user program, we can
  # directly apply `feedback.dna` to the user function called in the loop via
  # `with example():`, which materializes the hyper values used
  # within the user function using the decisions made by the search algorithm
  # for current trial. As a result, each call to the user function yields
  # different hyper-parameters implicitly bound with current trial.
  for example, feedback in pg.sample(
      pg.hyper.trace(create_model),
      pg.generators.Random(), max_trials):  # pytype: disable=wrong-arg-types  # gen-stub-imports
    print('{}: DNA: {}'.format(feedback.id, feedback.dna))
    with example():
      test_acc = train_and_eval(input_data, num_epochs)
    results.append((feedback.id, feedback.dna, test_acc))
    # NOTE: for random generator, following call to `feedback` is a no-op.
    # We keep it here in case we want to change algorithm.
    feedback(test_acc)

  # Print best results.
  top_results = sorted(results, key=lambda x: x[2], reverse=True)
  print('Top 10 results.')
  for i, (trial_id, dna, test_acc) in enumerate(top_results[:10]):
    print('#{0:2d} - trial {1:2d} ({2:.3f}): {3}'.format(
        i + 1, trial_id, test_acc, dna))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Placeholder for Google-internal tuning backend setup.

  tune(FLAGS.max_trials, FLAGS.num_epochs)


if __name__ == '__main__':
  app.run(main)
