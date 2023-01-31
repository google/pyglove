Symbolic Object-Oriented Programming
####################################

PyGlove offers two key capabilities for symbolic object-oriented programming: the
*Symbolic Object Model (SOM)* and *Symbolic Detour (SD)*. SOM implements *dynamic representation*,
allowing for flexible manipulation of symbolic objects. SD enables *dynamic interpretation*,
allowing symbols to be interpreted in different ways at runtime.


Symbolic Object Model
*********************

The Symbolic Object Model (SOM) is the core of symbolic object-oriented programming, 
providing dynamic representation through symbolic objects. SOM stores initialization
arguments as symbolic attributes and allows inspection and manipulation of them. 
It also includes a symbolic schema system for validation and a messaging system for
handling mutations. Symbolic placeholders are also supported for representing unknown
program parts.

.. .. _`dynamic representation`: ../../overview/what_and_why.html#programming-the-unknown


.. toctree::
   :maxdepth: 1

   som/definition
   som/types
   som/operations
   som/events
   som/validation
   som/placeholding

Symbolic Detour
***************

Symbolic Detour (SD) is independent of the SOM, allowing users to alter
class mapping without modifying the source code that instantiate the classes. This is
particularly useful when the source code cannot be symbolized in various reasons.
SD complements SOM.

.. toctree::
   :maxdepth: 1

   detour


   