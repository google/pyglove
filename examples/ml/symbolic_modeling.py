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
"""Symbolic neural modeling with PyGlove.

For more details, see:
https://colab.research.google.com/github/google/pyglove/blob/main/docs/notebooks/ml/neural_modeling.ipynb
"""
import pyglove as pg
import tensorflow as tf


# Symbolizing Keras layers so their instances can be symbolically manipulated.

Sequential = pg.symbolize(tf.keras.Sequential)
Conv2D = pg.symbolize(tf.keras.layers.Conv2D)
Dense = pg.symbolize(tf.keras.layers.Dense)
Flatten = pg.symbolize(tf.keras.layers.Flatten)
ReLU = pg.symbolize(tf.keras.layers.ReLU)


def create_model() -> tf.keras.layers.Layer:
  return Sequential([
      Conv2D(16, (5, 5)),
      ReLU(),
      Conv2D(32, (3, 3)),
      ReLU(),
      Flatten(),
      Dense(10)
  ])


def scale_model(model) -> None:
  """Scale the model up by doubling the filters of Conv2D layers."""
  def double_width(k, v, p):
    """A rebind rule for doubling the filters for Conv2D layers.

    Args:
      k: A `pg.KeyPath` object representing the location of current node.
      v: The value of current node.
      p: The parent of current node.

    Returns:
      The output value for current node.
    """
    if isinstance(p, Conv2D) and k.key == 'filters':
      return 2 * v
    return v

  # Rebind allows the users to manipulate a symbolic object by
  # rules.
  model.rebind(double_width)


def remove_relus(model) -> None:
  """Remove ReLU layers from the model."""
  def remove_activations(k, v, p):
    del k, p
    if isinstance(v, ReLU):
      # `pg.MISSING_VALUE` is a placeholder for deleting a value from
      # a container.
      return pg.MISSING_VALUE
    return v
  model.rebind(remove_activations)


def change_classification_head_width(model, width: int) -> None:
  """Update classification head width."""
  result = pg.query(model, where=lambda v: isinstance(v, Dense))
  classification_head_location = list(result.keys())[-1]
  model.rebind({
      f'{classification_head_location}.units': width
  })


def main() -> None:
  model = create_model()
  # The symbolized Keras layers can be printed in human readable form.
  # For clarity, we hide the default values of the layers.
  print('Original model.')
  print(model.format(hide_default_values=True))

  scale_model(model)
  print('After doubling the width.')
  print(model.format(hide_default_values=True))

  remove_relus(model)
  print('After removing the ReLUs.')
  print(model.format(hide_default_values=True))

  print('After changing the classification head width.')
  change_classification_head_width(model, 100)
  print(model.format(hide_default_values=True))


if __name__ == '__main__':
  main()
