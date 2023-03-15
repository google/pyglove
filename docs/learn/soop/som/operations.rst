Symbolic Operations
###################

Symbolic operations are operations that can be applied on any node in a
symbolic tree. They are organized into 5 categories:

  * Topological_: Symbolic tree access, traversal and query.
  * Semantical_: Common semantics for symbols, such as partial, pure symbolic, and etc.
  * Replication_: Symbolic tree cloning and serialization.
  * Mutation_: Mutating nodes in a symbolic tree in various ways.
  * Tracking_: Tracking creation of objects for debugging.

To make it easy to demostrate symbolic operations, we uses a symbolic
object ``zoo`` as the operation target::

    zoo = Zoo(
        name='San Diego Zoo',
        city='San Diego',
        exhibits=[
            Cage(animal=Python('Bob', color='black')),
            Pool(animal=Shark('Jack'))
        ]
    )

When we talk about *"manipulating the symbolic tree"* under the context of a
zoo-based program, it means to change one or more sub-nodes within ``zoo``.

Topological
***********

Starting from an arbitrary node in a symbolic tree (e.g. ``Cage``, ``Python``,
``Shark`` from the code above), PyGlove allows the user to access any other
nodes in the tree. To do so, PyGlove maintains bi-directional links between
containing/contained symbolic objects, allowing update notifications to
propagate through them.

Root
====

Users can access the root of current symbolic tree via property
:attr:`sym_root <pyglove.symbolic.Symbolic.sym_root>`. For example::

  assert zoo.sym_root is zoo
  assert zoo.exhibits[0].sym_root is zoo
  assert zoo.exhibits[0].animal.sym_root is zoo


Parent Node
===========

Similarly, users can access the immediate parent (the containing node) of a
symbolic value via property
:attr:`sym_parent <pyglove.symbolic.Symbolic.sym_parent>`. For example, the
``Cage`` object in the `zoo` has ``exhibits`` (a symbolic list) as its parent::

    assert zoo.exhibits[0].sym_parent is zoo.exhibits[0]


.. tip::

    PyGlove automatically sets the parent of a symbolic object when assignment
    happens. For example::

         shark = Shark('Jack')
         assert shark.sym_parent is None
         assert shark.sym_path == ''

         pool  = Pool(animal=shark)
         assert shark.sym_parent is pool
         assert shark.sym_path == 'x'


Ancestor
========

:meth:`sym_ancestor <pyglove.symbolic.Symbolic.sym_ancestor>` can be useful when
users require an ancestor in the containing chain that meets specific criteria
instead of the root or immediate parent. For instance, the following code
illustrates how to retrieve the nearest ``Zoo`` object from an ``Animal`` object
located in a zoo::

  assert zoo.exhibts[0].animal.sym_ancestor(lambda x: isinstance(x, Zoo)) is zoo

Child Nodes
===========

Child nodes (aka. symbolic attributes) mean different things for different symbolic types:

  * For **symbolic classes**, the child nodes are the arguments of
    ``__init__`` method, which can be accessed through
    :attr:`sym_init_args <pyglove.symbolic.Object.sym_init_args>`.
    For example::

      @pg.symbolize
      class Foo:
        
        def __init__(self, x):
          self._x = x

      f = Foo(1)
      assert f.sym_init_args['x'] == 1

    .. tip::

        For symbolic classes that are created from subclassing ``pg.Object``,
        the symbolic attributes can also be accessed through object properties with the
        argument names::

          @pg.members([
            ('x', pg.typing.Int())
          ])
          class Foo(pg.Object):
            pass
          
          f = Foo(1)
          assert f.x == f.sym_init_args.x

  * For **symbolic functions**, the child nodes are the bound arguments of the
    function, which can be directly accessed through its properties as well as
    :attr:`sym_init_args <pyglove.symbolic.Object.sym_init_args>`::

      @pg.symbolize
      def foo(x):
        return x ** 2

      f = foo(1)
      assert f.x == 1
      assert f.sym_init_args['x'] == 1

  * For **symbolic lists**, the child nodes are the items in the list, which can be
    directedly acccessed via the ``[]`` operator with their indices::

      l = pg.List([1, 2, 3])
      assert l[0] == 1

  * For **symbolic dicts**, the child nodes are the key/value pairs stored in the
    dict, which can be accessed via either the ``[]`` operator, or the dict
    attributes::

      d = pg.Dict(x=1, y=2)
      assert d.x == 1
      assert d['x'] == 1
  
The following table illustrates the uniform APIs to test and access symbolic attributes
across symbolic types:

.. list-table::
   :header-rows: 1
   :align: center

   * - Method
     - Description

   * - :meth:`sym_hasattr <pyglove.symbolic.Symbolic.sym_hasattr>`
     - Test if a child key exists 

   * - :meth:`sym_getattr <pyglove.symbolic.Symbolic.sym_getattr>`
     - Get the value of a child by key.

   * - :meth:`sym_keys <pyglove.symbolic.Symbolic.sym_keys>`
     - Iterate the child keys
   
   * - :meth:`sym_values <pyglove.symbolic.Symbolic.sym_values>`
     - Iterate the child values

   * - :meth:`sym_items <pyglove.symbolic.Symbolic.sym_items>`
     - Iterate child key/value pairs


For example::

    list(zoo.sym_keys()) == ['name', 'city', 'exhibits']
    list(zoo.sym_values())[0] == 'San Diego Zoo'
    list(zoo.sym_items())[0] == ('name', 'San Diego Zoo')

    zoo.sym_hasattr('name') == True
    zoo.sym_getattr('name') == 'San Diego Zoo'


Descendants
===========

In addition to accessing immediate child nodes, 
:meth:`sym_descendants <pyglove.symbolic.Symbolic.sym_descendants>`
is a handy tool to retrieve all nodes in the sub-tree. Users can also specify a filter
function (using the argument "where") and choose whether to include
intermediate nodes, leaves, or both in the returned nodes (using the argument
"option"). For instance, consider the following code, which demonstrates how to
select all animals from a zoo::

  assert zoo.sym_descendants(lambda x: isinstance(x, Animal)) == [
      Python('Bob', color='black'), Shark('Jack')]

Location
========

Each symbolic object has a unique location within a symbolic tree, represented a key path
(:class:`pg.KeyPath <pyglove.object_utils.KeyPath>`), which is a path consists of the keys
from the root node to the current node. 

For example, ``a.b[0].c`` is a path with height 4:

  * Level 0: a symbolic object or dict as the root node, bearing an empty key;
  * Level 1: a symbolic object or dict assigned to attribute ``a`` of the root node;
  * Level 2: a symbolic list assigned to attribute ``b`` of the level-1 node;
  * Level 3: a symbolic object or dict assigned to the first item of the level-2 list;
  * Level 4: a value assigned to argument ``c`` of the level-3 node.
  
Property :attr:`sym_path <pg.symbolic.Symbolic.sym_path>` is the API to access the symbolic
location, which is set when a symbolic object is added into a symbolic tree, and will be
updated when the hierarchy of the tree changes.

Relational
==========

`IS-A` and `HAS-A` are two common relationships among symbolic representations. Symbolic objects
are the instances of their symbolic classes, therefore `IS-A` relation can be easily tested
through :func:`isinstance` operator in Python. For `HAS-A` relation, :func:`pg.contains <pyglove.symbolic.contains>`
does the job. For example::

  @pg.symbolize
  def foo(x, y):
    pass
  
  @pg.symbolize
  def bar(a, b):
    pass
  
  f = foo(1, 2)
  b = bar(f, 3)
  # `f` has a `IS-A` relation with class `foo`.
  assert isinstance(f, foo)
  assert isinstance(b, bar)

  # `f` has a `HAS-A` relation with integer 2.
  assert pg.contains(f, 2)
  # `HAS-A` relation is transitive.
  assert pg.contains(b, 2)

  # `HAS-A` can be tested on types as well.
  # The following code is to query whether `b` contains any sub-node of type `foo`.
  assert pg.contains(b, type=foo)


Traversal
=========

:func:`pg.traverse <pyglove.symbolic.traverse>` is the API for facilitating symbolic tree traversal:

  * Users provide either a pre-order visitor function or a post-order
    visitor function, or both to perform the traversal;
  * Each visitor function takes a tuple of (``key_path``, ``value``, ``parent``)
    as the input and returns an action
    (see :class:`pg.TraverseAction <pyglove.symbolic.TraverseAction>`) to indicate whether to
    continue the traversal, stop or just skip current branch.

For example::

  def print_integers(key_path, value, parent):
    if isintance(value, int) and isinstance(parent, Foo):
      print(key_path, value)
    return pg.symbolic.TraverseAction.ENTER

  # Print all integer arguments of `Foo` objects in the
  # symbolic tree.
  pg.traverse(tree, print_integers)

Query
=====

:func:`pg.query <pyglove.symbolic.query>` is the helper when the user needs to
query a symbolic tree, which selects nodes from the tree based on user defined predicates:

  * A regular expression can be provided to perform path-based filtering;
  * A value selector can be provided to perform value-based filtering;
  * OR a custom selector can be provided to perform more complex filtering
    based on a node's path, value and parent node.

For example::

  @symbolic.members([
      ('x', schema.Int()),
      ('y', schema.Int())
  ])
  class A(symbolic.Object):
    pass

  value = {
    'a1': A(x=0, y=1),
    'a2': [A(x=1, y=1), A(x=1, y=2)],
    'a3': {
      'p': A(x=2, y=1),
      'q': A(x=2, y=2)
    }
  }

  # Query by path regex.
  print(symbolic.query(value, r'.*p'))
  # {'a3.p': A(x=2, y=1)}

  # Query by value.
  print(symbolic.query(value, where=lambda v: v==2))
  # {
  #    'a2[1].y': 2,
  #    'a3.p.x': 2,
  #    'a3.q.x': 2,
  #    'a3.q.y': 2,
  # }

  # Query by path, value and parent.
  print(symbolic.query(
      value, r'.*y',
      where=lambda v, p: v > 1 and isinstance(p, A) and p.x == 1))
  # {
  #    'a2[1].y': 2,
  # }

On top of ``pg.query``, :func:`pg.inspect <pyglove.symbolic.inspect>` provides a shortcut
to query nodes from a symbolic tree and print them to the standard output.


Formatting
==========

A symbolic tree can be presented nicely for human consumption.
By default, all symbolic types override ``__repr__`` and ``__str__`` so a
human-readable format can be shown during debugging:

  * ``__repr__`` formats a symbolic tree into a single-line string
    representation, which is usually used in error messages;
  * ``__str__``  formats a symbolic tree into a multi-line string
    representation, which is usually used in debugging purposes.

Both of these methods are based on :func:`pg.format <pyglove.object_utils.format>`, which provides a
rich set of features for formatting symbolic trees. For example, exclude
the keys that have the default values from the string representation::

  @pg.members([
     ('x', pg.typing.Int()),
     ('y', pg.typing.Int(default=2)),
  ])
  class Foo(pg.Object):
    pass

  foo = Foo(1, 2)
  print(foo.format(compact=False))
  # Foo(
  #   x=1,
  #   y=2 
  # )

  print(foo.format(compact=False, hide_default_values=True))
  # Foo(
  #   x=1
  # )


Semantical
**********

..  * **Partiality**: a symbolic object can be created without specifying all required arguments,
..    representing an partial object which can be filled later.
..  * **Pure symbolic**: a symbolic object that can placehold any node in a symbolic tree, for
..    representing an abstract concept. It needs to be replaced with the value type required by
..    its parent node before the program can be evaluated.
..  * **Abstract**: An abstract symbolic object is either partial or pure symbolic.
..  * **Missing values**: retrieve the missing values from a partial symbolic object.
..  * **Non-default values**: inspect the arguments of a symbolic object which are not the default
..    values.
  
In software development, oftentimes developers need to work with object representations
rather than their states. This poses a requirement such as comparing the equality of two
representations, hashing objects using their representations, and cloning objects through
their representations instead of duplicating their entire state. The APIs necessary for
achieving these objectives are discussed in this section.

Equality
========

Symbolic equality is determined by matching types and equal symbolic attributes, regardless
of the internal states being identical or not. For example::

  @pg.symbolize
  class File:

    def __init__(self, file_path):
      self._file_path = file_path
      self._file_handle = None
    
    def read(self, bytes):
      self._file_handle = open(self._file_path)
      ...
  
  path = 'a.json'
  f1 = File(path)
  # `f1.read()` triggers the creation of `f1._file_handle`.
  f1.read(10)

  f2 = File(path)
  assert pg.eq(f1, f2)


``f1`` and ``f2`` are considered equal as they have the same ``file_path``,
even their ``_file_handle`` are different. 

Symbolic equality can be tested via :func:`pg.eq <pyglove.symbolic.eq>` and
:func:`pg.ne <pyglove.symbolic.ne>`:

  * For symbolic objects, member methods :meth:`sym_eq <pyglove.symbolic.Symbolic.sym_eq>`
    and :meth:`sym_eq <pyglove.symbolic.Symbolic.eq>` will be called to determine whether
    they are symbolically equal or not.
  * For non-symbolic objects, the comparison will be delegated to :meth:`object.__eq__`
    and :meth:`object.__ne__`.

.. tip::

  For symbolic classes which subclass :class:`pg.Object <pyglove.symbolic.Object>`, whether to use
  symbolic equality as the default ``__eq__``/``__ne__``/``__hash__``
  behavior can be customized  by class variable
  :attr:`use_symbolic_comparison <pyglove.symbolic.Object.use_symbolic_comparison>`,
  which is set to ``True`` by default. For symbolized classes via :func:`pg.symbolize <pyglove.symbolic.symbolize>`,
  this can be achieved by specifying the ``eq`` argument to ``pg.symbolize``, which is set to ``False`` by default.

Less-Than/Greater-Than
======================

Two symbolic objects can be compared not only for equality, but also for ordering. 
A symbolic object ``x`` is considered less than another symbolic object ``y`` when:

* If ``x`` and ``y`` are comparable by their values, the operator ``__lt__`` is used for comparison.
  (e.g. :class:`bool`, :class:`int`, :class:`float`, :class:`str`)
* If ``x`` and ``y`` are of the same type and are symbolic containers
  (e.g. :class:`list`, :class:`dict`, :class:`pg.Symbolic <pyglove.symbolic.Symbolic>`), 
  the order is determined by the order of their first differing sub-nodes. 
  For example, ``['b']`` is greater than ``['a', 'b']``.
* If ``x`` and ``y`` are not directly comparable and have different types, they are compared based on
  their types. The order of different types is as follows:
  :data:`pg.MISSING_VALUE <pyglove.typing.MISSING_VALUE>`, NoneType, bool, int, float, str, list,
  tuple, set, dict, functions/classes. 
  If different functions or classes are compared, their order is determined by their qualified name.
* Non-symbolic classes can define the method ``sym_lt`` to enable symbolic comparison.

Here are some examples::

  assert pg.lt(False, True) == Flase < True
  assert pg.lt(0.1, 1) == 0.1 < 1
  assert pg.lt('a', 'ab') == 'a' < 'ab'
  
  assert pg.lt(['a'], ['a', 'b'])
  assert pg.lt(['a', 'b', 'c'], ['b'])
  assert pg.lt({'x': 1}, {'x': 2})
  assert pg.lt({'x': 1}, {'y': 1})
  assert pg.lt(A(x=1), A(x=2))

  assert pg.lt(pg.MISSING_VALUE, None)
  assert pg.lt(None, 1)
  assert pg.lt(1, 'abc')
  assert pg.lt('abc', [])
  assert pg.lt([], {})
  assert pg.lt([], A(x=1))

Similarly, :func:`pg.gt <pyglove.symbolic.gt>` determines if a symbolic object is greater than another
symbolic object by its representation.

Hashing
=======

The semantics of symbolic hashing is aligned with equality: two symbolically equal
objects should produce the same symbolic hash value.

In PyGlove, symbolic hash can be computed via ``pg.hash``:

  * For symbolic objects, member method ``sym_hash`` will be called for
    computing the symbolic hash value.
  * For non-symbolic objects, PyGlove depends on their original hash
    semantics.

.. warning::

  Always override ``sym_hash``  when ``sym_eq`` is overriden.


Difference
==========

Besides, the symbolic differences between two objects can be obtained by :func:`pg.diff <pyglove.symbolic.diff>`.
``pg.diff`` is a handy tool for figuring out which parts from the objects are different. 

TODO(daiyip): add examples

Special Symbolic Forms
======================

PyGlove supports abstract objects through symbolic placeholding (see :doc:`placeholding`), which allows creating and manipulating symbolic
objects that are merely representations. Here is a summary of operations that detects the forms of symbolic objects.

.. list-table::
   :header-rows: 1
   :align: center

   * - API
     - Method
     - Description

   * - :func:`pg.is_abstract <pyglove.symbolic.is_abstract>`
     - :meth:`~pyglove.symbolic.Symbolic.sym_abstract`
     - Test whether an object is abstract or not.

   * - :func:`pg.is_partial <pyglove.object_utils.is_partial>`
     - :meth:`~pyglove.symbolic.Symbolic.sym_partial`
     - Test whether an object is partial or not.
     
   * - :func:`pg.is_pure_symbolic <pyglove.symbolic.is_pure_symbolic>`
     - :meth:`~pyglove.symbolic.Symbolic.sym_puresymbolic`
     - Test whether an object is pure symbolic or not.
   
   * - :func:`pg.is_deterministic <pyglove.symbolic.is_deterministic>`
     - N/A
     - Test whether an object contains objects of :class:`pg.symbolic.NonDeterministic <pyglove.symbolic.NonDeterministic>`.


Besides, the following APIs offers capabilities to query the parts of special interests:

.. list-table::
   :header-rows: 1
   :align: center

   * - Method
     - Description

   * - :meth:`~pyglove.symbolic.Symbolic.sym_missing` or
       :meth:`~pyglove.symbolic.Symbolic.missing_values`
     - Query the missing values from the object.
     
   * - :meth:`~pyglove.symbolic.Symbolic.sym_nondefault` or
       :meth:`~pyglove.symbolic.Symbolic.non_default_values`
     - Query the default values from the object.

Replication
***********

Symbolic objects can be replicated in process or across processes. In-process replication is achieved by cloning, and
inter-process replication is achieved by serialization/deserialization. 

.. warning::

  By default, symbolic replication does not deal with replication of internal states, which means a replicated
  symbolic object is equivalent to a freshly constructed object with the same binding parameters. But the user
  can optionally handle internal state replication by override the ``sym_clone`` and ``sym_jsonify`` methods.

Clone
=====

Users can clone a symboic object via the ``pg.clone`` function or call the ``clone`` member method of the symbolic
objects. The semantics of symbolic clone are the following:

  * For symbolic types, ``sym_clone`` will be called when cloning the object.
  * For non-symbolic types, ``__copy__`` / ``__deepcopy__`` will be called when cloning the object. The ``deep`` argument
    of ``pg.clone`` determines which function to use.

It is common that the user clones a symbolic object with overrides, this can be done with the ``overrides`` argument,
which accepts a dictionary of path to values to override in the cloned object.

For example::

  TODO(daiyip): add examples.


Serialization
=============

The automatic serialization/deserialization capability for symbolic objects is
provided by member method ``sym_jsonify`` and class method ``from_json``. 
``sym_jsonify`` converts current symbolic object into a Python dict mapped from
strings to basic python values, while ``from_json`` converts them back. 

Based on the two methods, PyGlove provides a few helper methods for serialization
and deserialization.

.. list-table::
   :header-rows: 1
   :widths: 20 50
   :align: center

   * - Method
     - Description

   * - :func:`pg.to_json <pyglove.symbolic.to_json>`
     - Converts a symbolic object into a plain Python dict.

   * - :func:`pg.from_json <pyglove.symbolic.from_json>`
     - Converts a plain Python dict into a symbolic object.

   * - :func:`pg.to_json_str <pyglove.symbolic.to_json_str>`
     - Converts a symbolic object into a JSON string.

   * - :func:`pg.from_json_str <pyglove.symbolic.from_json_str>`
     - Creates a symbolic object from a JSON string.

   * - :func:`pg.save <pyglove.symbolic.save>`
     - Saves a symbolic object into a file.
      
   * - :func:`pg.load <pyglove.symbolic.load>`
     - Loads a symbolic object from a file.

.. tip::

  For deserialization to work, the user class definition needs to be imported first.

The save and load hook
----------------------

:func:`pg.set_save_handler <pyglove.symbolic.set_save_handler>` and
:func:`pg.set_load_handler <pyglove.symbolic.set_load_handler>` are introduced
for user to plug in custom IO operations when calling
:func:`pg.save <pyglove.symbolic.save>` and :func:`pg.load <pyglove.symbolic.load>`.
Through this, the user are able to load/save symbolic objects in cloud-based
storages without changing the client code.

Mutation
********

Symbolic mutation is the core of symbolic programming. PyGlove provides a rich set of APIs for mutating
symbolic objects.

Location-based mutations
============================

Location-based mutation is a basic form of symbolic mutation. This can be achieved by the ``Symbolic.rebind`` interface, which takes a dict object. The keys in the dict are the
key paths of the nodes whose values are to be replaced, and the values are their new values.

For example::

  TODO: daiyip, add an example here.

Pattern-based mutations
===========================


Oftentimes, the user mutates a symbolic object by rules. Many of these rules can be described as patterns, for example:
change the ``name`` property of all objects; or change the ``filters`` property if the object type is a ``Conv2D``.

Built on top of ``Symbolic.rebind``, ``pg.patching`` is a sub-module of PyGlove for pattern-based object patching. Common
patterns are supported such as:

.. list-table::
   :header-rows: 1
   :widths: 20 50
   :align: center

   * - Method
     - Description
   * - :func:`pg.patching.patch_on_key <pyglove.patching.patch_on_key>`
     - Replaces objects assigned to certain keys (described by a regular
       expression) in the tree;
   * - :func:`pg.patching.patch_on_path <pyglove.patching.patch_on_path>`
     - Replaces objects with certain paths (described by a regular expression)
       in the tree;
   * - :func:`pg.patching.patch_on_value <pyglove.patching.patch_on_value>`
     - Replaces objects whose values match with the condition;
   * - :func:`pg.patching.patch_on_type <pyglove.patching.patch_on_type>`
     - Replaces objects of specific types in the tree;
   * - :func:`pg.patching.patch_on_member <pyglove.patching.patch_on_member>`
     - Replaces objects which are the members of a given type.

Rule-based mutations
====================


More complex symbolic mutations is achievable by using a transform function, which can be passed to ``Symbolic.rebind``
as rebinding rules. The function takes 3 inputs: the ``location``, ``value`` and ``parent`` of a node to transform from 
the tree. The function returns the new value for that node.

For example::

  TODO(daiyip): add examples.


Command-based mutations
===========================

* Manipuate object with user commands
* introduce ``pg.patcher``.


Sealing an Object
=================




Tracking
********

Since a symbolic object can be created and modified at runtime, at times we want to track the origin of symbolic objects
for the purposes of debugging. PyGlove introduces an ``Origin`` class, whose instance can be associated with a symbolic object
during its creation. The ``Origin`` object contains stack information and the source form of the symbolic object, whether it's
a file path string, or an object from where current object is cloned. The user can also add origin information to objects using
``Symbolic.sym_setorigin`` and access it using ``Symbolic.sym_origin`` property.
