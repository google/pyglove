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
"""Train MNIST.

This is a basic working ML program which trains MNIST.
The code is modified from the tf.keras tutorial here:
https://www.tensorflow.org/tutorials/keras/classification

(The tutorial uses Fashion-MNIST,
but we just use "regular" MNIST for these tutorials.)

"""

from typing import Tuple

from absl import app
import numpy as np
import tensorflow.google.compat.v2 as tf


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
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  return model


def train_and_eval() -> None:
  """Run training and evaluation.

  Code to run all of the prep, training, and evaluation.
  """
  tr_x, tr_y, te_x, te_y = download_and_prep_data()
  model = create_model()
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(tr_x, tr_y, epochs=10)
  test_loss, test_acc = model.evaluate(te_x, te_y, verbose=2)
  print('Test loss: {}, accuracy: {}'.format(test_loss, test_acc))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  train_and_eval()


if __name__ == '__main__':
  app.run(main)
