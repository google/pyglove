# PyGlove

PyGlove is a library for building Python programs that are easily evolvable,
whether it be manually or automatically. For example, to find an optimal neural
architecture for an image classifier, the architecture needs to be evolved by
humans or by machines. With adding a single line of code, PyGlove upgrades a
regular Python class into a symbolic class, whose objects can be manipulated
safely at run time as if their source code were modified. On top of that,
meta-programs can thrive, which not only add new aspects to such classes with
ease (e.g. serialization, visualization), but also invites algorithms to
participate in the modifications of the program automatically, enabling
applications such as program search.

PyGlove was created for scaling [automated machine learning](https://en.wikipedia.org/wiki/Automated_machine_learning) (AutoML) in
research and application. Its method was [published](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf)
in 2020 and has also been used widely in Google since 2019: through PyGlove, a
better algorithm has been found on top of DQN by
[evolving reinforcement learning algorithms](https://ai.googleblog.com/2021/04/evolving-reinforcement-learning.html); multi-shot and one-shot neural architecture search were
[unified under the same programming interface](https://proceedings.neurips.cc/paper/2020/file/012a91467f210472fab4e11359bbfef6-Paper.pdf);
the [co-design of neural architectures and hardware accelerators](https://proceedings.mlsys.org/paper/2022/file/31fefc0e570cb3860f2a6d4b38c6490d-Paper.pdf)
was possible for maximizing the model performance while meeting the hardware
constraints. PyGlove is also being used in key products under Alphabet, such as
Waymo's self-driving cars, Youtube's recommendation systems, Google Cloud's
[AI solutions](https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-nas-makes-the-most--advanced-ml-modeling-possible), and etc.

Behind the scene, PyGlove introduces symbolic object-oriented programming, a new
programming paradigm that adds symbolic programmability to objects, making it a
useful tool for programs that need mutable representations, such as neural
modeling, or a machine learning program as a whole. Moreover, PyGlove enhances
the programmability for daily programming tasks. It is carefully implemented to
be a general-purpose programming toolkit, and has no dependencies beyond the
Python interpreter. PyGlove is developed by Daiyi Peng and his colleagues in
[Google Research Brain team](https://research.google/teams/brain/).


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
