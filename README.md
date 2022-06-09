# PyGlove

[![codecov](https://codecov.io/gh/google/pyglove/branch/main/graph/badge.svg)](https://codecov.io/gh/google/pyglove)
![pytest](https://github.com/google/pyglove/actions/workflows/ci.yaml/badge.svg)

PyGlove is a general-purpose library for Python object manipulation. 
It is mainly used to handle complex machine learning scenarios such as AutoML. It also has been used in daily programming tasks, with no dependencies beyond the Python interpreter.

It provides:

* A mutable symbolic object model for Python;
* A rich set of operations for Python object manipulation;
* An easy-to-use API to automatically search for better Python programs;
* A framework for developing search algorithms;
* An API to interface with any distributed infrastructure.

It has been used in:

* Machine learning for large teams (code sharing, hparam management);
* Automated machine learning (AutoML);
* Evolutionary computing;
* Enhancing productivity and flexibility for general-purpose programs in Python.

PyGlove's method for AutoML was [published](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf) at NeurIPS 2020. It has been widely used within [Alphabet](https://abc.xyz/) since 2019, including Google Research, Google Cloud, Youtube, Waymo and etc.

PyGlove is developed by Daiyi Peng and his colleagues in [Google Brain](https://research.google/teams/brain/).


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


## Citing PyGlove

```
@article{peng2020pyglove,
  title={PyGlove: Symbolic programming for automated machine learning},
  author={Peng, Daiyi and Dong, Xuanyi and Real, Esteban and Tan, Mingxing and Lu, Yifeng and Bender, Gabriel and Liu, Hanxiao and Kraft, Adam and Liang, Chen and Le, Quoc},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={96--108},
  year={2020}
}
```

*Disclaimer: this is not an officially supported Google product.*
