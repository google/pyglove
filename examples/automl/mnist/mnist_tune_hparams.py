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
"""NAS on MNIST with passing a search space of hyper-parameters.

This is a basic working ML program which does NAS on MNIST.
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


# NOTE(daiyip): decorator 'pg.symbolize' turns a function into a Functor, which
# is a class with a `__call__` method. An instance of a functor represents a
# function whose arguments are fully or partially bound.
#
# For example:
#   ```
#   # Create a `build_model` instance whose `hparams` is bound at construction
#   # time.
#   model_builder = build_model(hparams={...})
#
#   # Invoke the functor with late-bound `inputs` argument.
#   model = model_builder(inputs)
#   ```
@pg.symbolize
def build_model(inputs, hparams):
  """Build model from hyper-parameters."""
  if hparams.use_conv_net:
    x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 1)))(inputs)
    x = tf.keras.layers.Conv2D(filters=hparams.filters,
                               kernel_size=hparams.kernel_size,
                               activation=hparams.activation)(x)
    x = tf.keras.layers.Flatten()(x)
  else:
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(hparams.units, activation=hparams.activation)(x)
  return tf.keras.layers.Dense(10, activation='softmax')(x)


def train_and_eval(model_builder, input_data, num_epochs=10) -> float:
  """Returns model accuracy after train and evaluation."""
  tr_x, tr_y, te_x, te_y = input_data
  inputs = tf.keras.Input(shape=tr_x.shape[1:])
  model = tf.keras.Model(inputs, model_builder(inputs))
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(tr_x, tr_y, epochs=num_epochs)
  _, test_acc = model.evaluate(te_x, te_y, verbose=2)
  return test_acc


def tune(max_trials, num_epochs):
  """Tune MNIST model via random search."""
  results = []
  input_data = download_and_prep_data()

  # NOTE(daiyip): Being decorated by `pg.symbolize`, `build_model` is now a
  # functor, which is a class whose arguments can be partially or late-bound.
  # Here we create a `build_model` instance by partially binding the `hparams`
  # argument, whose `inputs` argument will be bound during the `__call__` time
  # within the definition of `train_and_eval`.
  hyper_model_builder = build_model(hparams=pg.oneof([  # pylint: disable=no-value-for-parameter
      # For dense layer as the backbone.
      pg.Dict(use_conv_net=False,
              units=pg.oneof([64, 128]),
              activation=pg.oneof(['relu', 'sigmoid'])),
      # For conv net as the backbone.
      pg.Dict(use_conv_net=True,
              filters=pg.oneof([64, 128]),
              kernel_size=pg.oneof([(3, 3), (5, 5)]),
              activation=pg.oneof(['relu', 'sigmoid']))
  ]))

  # NOTE(daiyip): `pg.sample` returns an iterator of (example, feedback_fn)
  # from a hyper object (the search space) and a DNAGenerator (the search
  # algorithm), with an optional flag to set the max examples to sample.
  # `example` is a materialized object of the search space, and `feedback_fn`
  # is a callable object that we can send back a float reward to the
  # controller. `feedback_fn` also has a property `dna` to access the DNA value
  # of current example.
  for builder, feedback in pg.sample(
      hyper_model_builder, pg.generators.Random(), max_trials):  # pytype: disable=wrong-arg-types  # gen-stub-imports
    print('{}: DNA: {}'.format(feedback.id, feedback.dna))
    test_acc = train_and_eval(builder, input_data, num_epochs)
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
