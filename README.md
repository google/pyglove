<div align="center">
<img src="https://raw.githubusercontent.com/google/pyglove/main/docs/_static/logo_light.svg#gh-light-mode-only" width="320px" alt="logo"></img>
<img src="https://raw.githubusercontent.com/google/pyglove/main/docs/_static/logo_dark.svg#gh-dark-mode-only" width="320px" alt="logo"></img>
</div>

# PyGlove: Manipulating Python Programs

[![PyPI version](https://badge.fury.io/py/pyglove.svg)](https://badge.fury.io/py/pyglove)
[![codecov](https://codecov.io/gh/google/pyglove/branch/main/graph/badge.svg)](https://codecov.io/gh/google/pyglove)
![pytest](https://github.com/google/pyglove/actions/workflows/ci.yaml/badge.svg)


PyGlove is a general-purpose library for Python object manipulation.
It introduces symbolic object-oriented programming to Python, allowing
direct manipulation of objects that makes meta-programs much easier to write.
It has been used to handle complex machine learning scenarios, such as AutoML,
as well as facilitating daily programming tasks with extra flexibility.

PyGlove is lightweight and has no dependencies beyond the Python interpreter.
It provides:

* A mutable symbolic object model for Python;
* A rich set of operations for Python object manipulation;
* A solution for automatic search of better Python programs, including:
  * An easy-to-use API for dropping search into an arbitrary pre-existing Python
    program;
  * A set of powerful search primitives for defining the search space;
  * A library of search algorithms ready to use, and a framework for developing
    new search algorithms;
  * An API to interface with any distributed infrastructure for such search.

It's commonly used in:

* Automated machine learning (AutoML);
* Evolutionary computing;
* Machine learning for large teams (evolving and sharing ML code, reusing
  ML techniques, etc.);
* Daily programming tasks in Python (advanced binding capabilities, mutability,
  etc.).

PyGlove has been [published](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf)
at NeurIPS 2020. It is widely used within [Alphabet](https://abc.xyz/), including Google Research, Google Cloud, Youtube and Waymo.

PyGlove is developed by Daiyi Peng and colleagues in [Google Brain Team](https://research.google/teams/brain/).


## Install

```
pip install pyglove
```

## Hello PyGlove

```python
import pyglove as pg

@pg.symbolize
class Hello:
  def __init__(self, subject):
    self._greeting = f'Hello, {subject}!'

  def greet(self):
    print(self._greeting)


hello = Hello('World')
hello.greet()
```
> Hello, World!

```python
hello.rebind(subject='PyGlove')
hello.greet()
```
> Hello, PyGlove!

```python
hello.rebind(subject=pg.oneof(['World', 'PyGlove']))
for h in pg.iter(hello):
  h.greet()
```
> Hello, World!<br>
> Hello, PyGlove!


## Examples

* AutoML
  * [Neural Architecture Search on MNIST](https://github.com/google/pyglove/tree/main/examples/automl/mnist)
  * [NAS-Bench-101](https://github.com/google/pyglove/tree/main/examples/automl/nasbench)
  * [NATS-Bench](https://github.com/google/pyglove/tree/main/examples/automl/natsbench)
  * [Evolving Reinforcement Learning Algorithms](https://github.com/google/brain_autorl/tree/main/evolving_rl)
* Evolution
  * Framework: [[Algorithm](https://github.com/google/pyglove/blob/main/docs/notebooks/intro/search/evolution_algorithm.ipynb)]
    [[Ops](https://github.com/google/pyglove/blob/main/docs/notebooks/intro/search/evolution_ops.ipynb)]
    [[Fine Control](https://github.com/google/pyglove/blob/main/docs/notebooks/intro/search/evolution_scheduling.ipynb)]
  * [Travelling Salesman Problem](https://github.com/google/pyglove/blob/main/docs/notebooks/evolution/tsp.ipynb)
  * [One-Max Problem](https://github.com/google/pyglove/blob/main/docs/notebooks/evolution/onemax.ipynb)

* Machine Learning
  * [Symbolic Machine Learning](https://github.com/google/pyglove/blob/main/docs/notebooks/ml/symbolic_ml.ipynb)
  * [Symbolic Neural Modeling](https://github.com/google/pyglove/blob/main/docs/notebooks/ml/neural_modeling.ipynb)

* Advanced Python Programming
  * [Sticky Notes: A mini Domain-specific Language](https://github.com/google/pyglove/blob/main/docs/notebooks/python/sticky_notes.ipynb)
  * [Interactive SVG: Components for Direct Manipulation](https://github.com/google/pyglove/blob/main/docs/notebooks/python/interactive_svg.ipynb)
  * [Where is the Duck: Developing Context-aware Component](https://github.com/google/pyglove/blob/main/docs/notebooks/python/where_is_the_duck.ipynb)

## Citing PyGlove

```
@inproceedings{peng2020pyglove,
  title={PyGlove: Symbolic programming for automated machine learning},
  author={Peng, Daiyi and Dong, Xuanyi and Real, Esteban and Tan, Mingxing and Lu, Yifeng and Bender, Gabriel and Liu, Hanxiao and Kraft, Adam and Liang, Chen and Le, Quoc},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={96--108},
  year={2020}
}
```

*Disclaimer: this is not an officially supported Google product.*
