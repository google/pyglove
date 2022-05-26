# PyGlove Tutorial: Neural Architecture Search on MNIST

## Overview
PyGlove provides 3 options for creating a search space out of a
regular Python program. Each may find its users among different codebases and
scenarios:

* **Passing a search space of hyper-parameters**:
This is the option for most ML practioners who already have a codebase that
trains a single model by passing the hyper-parameters from the top.
By symbolizing the top-level function using functor, search can be enabled on
an existing codebase with a few lines of code. This option is illustrated in [mnist_tune_hparams.py](https://github.com/google/pyglove/examples/automl/mnist/mnist_tune_hparams.py)

* **Define-by-run search space definition**:
This option is for codebases whose hyper-parameters are not centrally managed
(e.g. specified within a function without passing from the function argument).
Therefore, users want to directly modify the code (e.g. function definitions)
to convert it into a search space. This option is the least flexibile but has
the smallest cost to get started with, illustrated in [mnist_tune_eagerly.py](https://github.com/google/pyglove/examples/automl/mnist/mnist_tune_eagerly.py)

* **Search space as a composition of symbolic objects**:
This is the most flexible and powerful option, which works well for software
systems that are already compositional (e.g. Keras layers).
Within a hierarchical composition, nodes are not only searchable, but also
rewrittable into different components or search spaces. This option is
recommended for new systems built for maximum flexiblity. This option is illustrated in [mnist_tune.py](https://github.com/google/pyglove/examples/automl/mnist/mnist_tune.py)


## Option 1: Passing a search space of hyper-parameters

[Builder pattern](https://en.wikipedia.org/wiki/Builder_pattern) is commonly
used in Machine Learning pipelines. Different from the option 1 which symbolizes
the user classes directly and creates the search space by composition of their
hyper values, this option symbolizes the builder function, which makes it easy
to drop search into an existing code base whose hyper-parameters are passed down
from a top-level function.

For example, given a user function:

  ```python
  def foo(a, b):
    return a + b
  ```

We can symbolize the function by applying a `pg.symbolize` decorator:

  ```python
  @pg.symbolize
  def foo(a, b):
    return a + b
  ```
Or we can call `pg.symbolize` as a function, without modifying the source file where `foo` is defined:

  ```python
  foo = pg.symbolize(foo)
  ```

As a result, `foo` is turned into a functor, which is a class with a
`__call__` method, and we can create its instance with hyper values:

  ```python
  hyper_foo = foo(pg.oneof([1, 2]), pg.floatv(0.1, 0.5))
  ```

It can be used as a search space and optimized by `pg.sample` as follows:

 ```python
 for foo, feedback in pg.sample(hyper_foo, search_algorithm, max_trials):
   reward = foo()  # Call the function with prebound `a` and `b`.
   feedback(reward)
 ```

See example on MNIST in [mnist_tune_hparams.py](https://github.com/google/pyglove/examples/automl/mnist/mnist_tune_hparams.py)

## Option 2: Define-by-run search space definition

Define-by-run style search is advocated by the [Optuna paper](https://arxiv.org/abs/1907.10902), whose pros and cons are distinctive.

*Pros*:

- The user program can be immediately made searchable by replacing fixed
values in the code with hyper values (e.g. `pg.oneof`), without symbolizing the
user classes or builder functions.
- It does not require to pass the hyper values from the top.

*Cons*:

- Only one search space is preserved unless the code
is copied.
- The search space definition is a bit scattered within the
function and class definitions called by the function which is to be optimized.

Different from former options which represent a search space using a symbolic
hyper value (symbolic object or functor) and materialize the concrete objects
through late-binding, this example eagerly evaluates the hyper values involved
in the execution of user function. E.g:

  ```python
  def foo():
    return pg.oneof([1, 2]) + pg.floatv(0.1, 0.5)
  ```

When `foo` is executed, `pg.oneof` and `pg.floatv` here will be evaluated to
an integer and a float.

`foo` will be called once for inspecting the decision points in the search
space via `pg.hyper.trace`, then it should be called multiple
times within the `pg.sample` loop under the context of
the `example` yield in each iteration. The code pattern of tuning a program eagerly is
shown as follows:

  ```python
    for example, feedback in pg.sample(
        pg.hyper.trace(foo),
        search_algorithm,
        max_number_examples):
      with example():
        reward = foo()
        feedback(reward)
  ```

When calling `foo` during defining the search space, its output will be
discarded, and later runs within the loop will produce a number as the feedback
to the search algorithm.

See example on MNIST in [mnist_tune_eagerly.py](https://github.com/google/pyglove/examples/automl/mnist/mnist_tune_eagerly.py)


## Option 3: Search space as a composition of symbolic objects

Assume the user has an existing class `Foo`:

  ```python
  class Foo(object):
    def __init__(self, a, b):
      self.a = a
      self.b = b

    def compute(self):
      return self.a + self.b
  ```

The user can make a symbolic wrapper class out of it by:

  ```python
  SymbolicFoo = pg.symbolize(Foo)
  ```

With the symbolic wrapper, we can create a search space from its instance and
optimize it using `pg.sample`:

  ```python
  # Create a search space by tuning `a` and `b`.
  hyper_foo = SymbolicFoo(a=pg.oneof([1, 2]), b=pg.floatv(0.1, 0.5))

  # Search is a sampling process with feeding back the reward
  # computed from each example to the search algorithm.
  for foo, feedback in pg.sample(hyper_foo, search_algorithm, max_trials):
    reward = foo.compute()
    feedback(reward)
  ```

See example on MNIST in [mnist_tune.py](https://github.com/google/pyglove/examples/automl/mnist/mnist_tune.py)
