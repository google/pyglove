Learning PyGlove
################

.. PyGlove is organized into three layers: 

.. * At the bottom, the layer of symbolic object-oriented programming, enables a mutable
..   programming model with objects, which allows unknown program parts to be expressed
..   side-by-side with the expression of known program parts, and enables dynamic interpretations
..   on them.
.. * At the middle, the layer of intelligent programs, provides the representation
..   and operations to convert between symbols and numbers. It also introduces the expression for
..   feedback loops, as well as a framework for building algorithms that evolve the program.
.. * At the top, the layer of distributed symbolic computing, introduces API to allow
..   feedback loops to be distributed so that intelligent programs can run at scale. This layer
..   also provide the interfaces for the user to plug in their own infrastructures into PyGlove.

PyGlove is a Python library for manipulating Python programs. It is built on the
concept of symbolic object-oriented programming (SOOP), which forms its core foundation. 
On top of that, PyGlove includes multiple layers of components that enhance its capabilities and
enable the development of intelligent systems.


.. toctree::
   :maxdepth: 1

   SOOP <soop/index>
   evolution/index

.. .. image:: how_pyglove_works.svg

