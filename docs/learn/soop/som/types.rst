Symbolic Types
##############

Symbolic types are the types of :doc:`symbolic objects <definition>`. This section provides an overview of the most commonly
used symbolic types, and explains how they can be derived from regular types.

  * `Symbolic Class`_
  * `Symbolic Function (Functor)`_
  * `Symbolic Container Types`_

    * `Symbolic List`_
    * `Symbolic Dict`_

Symbolic Class
**************

Classes are the basic units of modern computer programs. 
PyGlove makes it easy to create symbolic classes from regular Python classes using two methods:

  * Developing a `dataclass-like`_ symbolic class by
    :ref:`subclassing<Defining a dataclass-like symbolic class>` :class:`pg.Object <pyglove.symbolic.Object>`.
  * Developing a class as usual and :ref:`decorate<Symbolizing a regular class>`
    it using :func:`pg.symbolize <pyglove.symbolic.symbolize>`. This also work with existing classes.

.. warning::
   
  ``pg.symbolize`` **on existing classes can fail**: The flexibility of Python
  allows a user class to do a wide range of things. For example, the Python
  classes generated from `Protocol Buffers`_ do not allow themselves to be
  subclassed, while ``pg.symbolize`` requires inheritance to create symbolic
  types from existing ones. Another example is neural modeling library Flax_,
  which keeps track of objects in the callstack of ``__init__``, in order to
  figure out the containing layer for current layer. However, the generated
  symbolic class will change the ``__init__`` callstack, which breaks the
  premise. In such cases, the user class may need to make some adjustments
  in order to make peace with PyGlove's implementation on symbolization.


.. _`dataclass-like`: https://docs.python.org/3/library/dataclasses.html
.. _`Protocol Buffers`: https://developers.google.com/protocol-buffers
.. _Flax: https://github.com/google/flax


Defining a Dataclass-like Symbolic Class
========================================

This is the simplest method for creating a symbolic class from scratch, which
increases productivity by automatically generating the ``__init__`` method and
allowing access to symbolic attributes through object properties. To do this,
users can extend :class:`pg.Object <pyglove.symbolic.Object>` or a subclass
and declare symbolic fields using :func:`pg.members <pyglove.symbolic.members>`.

For example::

    @pg.members([
        # Each tuple in the list defines a symbolic field for `__init__`.
        ('name', pg.typing.Str().noneable(), 'Name to greet'),
        ('time_of_day', 
         pg.typing.Enum('morning', ['morning', 'afternnon', 'evening']),
         'Time of the day.')
    ])
    class Greeting(pg.Object):

      def __call__(self):
        # Values for symbolic fields can be accessed
        # as public data members of the symbolic object.
        print('Good %s, %s' % (self.time_of_day, self.name))

    # Create an object of Greeting and invoke it,
    # which shall print 'Good morning, Bob'.
    Greeting('Bob')()

Understanding Symbolic Fields
-----------------------------

*Symbolic fields* define the names and acceptable values for a symbolic class' `__init__` method, thus
defining its symbolic attributes. For a symbolic field `x`, users can access its corresponding symbolic attribute 
at runtime through the :attr:`sym_init_args <pyglove.Object.sym_init_args>` property, and also through object
properties if the symbolic class is created by subclassing :class:`pg.Object <pyglove.symbolic.Object>`.

Symbolic fields can be organized hierarchically, which is useful when there are many of them and can be
grouped together for better organization. For example::

    @pg.members([
        ('training', pg.typing.Dict([
            ('dataset', pg.typing.Object(Dataset)),
            ('total_steps', pg.typing.Int())
            ...
        ])),
        ('evaluation', pg.typing.Dict([
            ('dataset', pg.typing.Object(Dataset)),
            ('steps', pg.typing.Int())
            ...
        ]))
    ])
    class Trainer(pg.Object):
      pass

    trainer = Trainer(
        training=dict(
            dataset=Dataset(...),
            total_steps=100,
            ...
        ),
        evaluation=dict(
            dataset=Dataset(...),
            steps=20
        ))

See :doc:`validation` for more details on symbolic field declaration.

Field Inheritance
-----------------

PyGlove allows for field inheritance for classes created by subclassing
:class:`pg.Object <pyglove.symbolic.Object>` or its subclasses. Fields from
the base class will be inherited by the subclass in their order of declaration,
and the subclass can override the inherited fields with stricter validation rules
or different default values. For example::

    @pg.members([
        ('x', pg.typing.Int(max_value=10)),
        ('y', pg.typing.Float(min_value=0))
    ])
    class Foo(pg.Object)
      pass
    
    @pg.members([
        ('x', pg.typing.Int(min_value=1, default=1)),
        ('z', pg.typing.Str().noneable())
    ])
    class Bar(Foo)
      pass

    # Printing Bar's schema will show that there are 3 parameters defined:
    # x : pg.typing.Int(min_value=1, max_value=10, default=1))
    # y : pg.typing.Float(min_value=0)
    # z : pg.typing.Str().noneable()
    print(Bar.schema)


Symbolizing a Regular Class
===========================

There are several scenarios that you may want to use :func:`pg.symbolize <pyglove.symbolic.symbolize>`
to create symbolic classes:

 * You need to make an existing class symbolic;
 * You want to develop a class as usual and make it symbolic with minimal change;
 * You encounter a use case that needs to multi-inherit ``pg.Object`` and another
   class;
 * You need to subclass an already symbolized class.

Here is how ``pg.symbolize`` works: it generates a class by multi-inheriting 
:class:`pg.ClassWrapper <pyglove.symbolic.ClassWrapper>` (a ``pg.Object`` subclass) and
your (regular) class. As a result, functionalities from both worlds can be combined.

``pg.symbolize`` can be used as a decorator to make symbolic class developement simple::

    @pg.symbolize
    class Foo:

      def __init__(self, x):
        self.x = x

Or it can be used as a function to symbolize a class without modifying
the source code of the original classes::

    class Foo:

      def __init__(self, x):
        self.x = x

    SymbolicFoo = pg.symbolize(Foo)

To avoid name clash on object attributes, symbolic fields are only accessible 
via the `sym_init_args` property for symbolized classes.

Custom Behaviors
----------------

There are a few behaviors you can customize during ``pg.symbolize`` via its
arguments:

 * **repr**: default set to `True``, whether to generate ``__repr__`` and
   ``__str__`` based on the symbolic representation of the object.
 * **eq**: default set to `False`, whether to generate ``__eq__``, ``__ne__``
   and ``__hash__`` based on the symbolic equality of objects.
 * **class_name**: class name used for the symbolized class. By default it
   uses the same name as the source class.
 * **module_name**: module name used for the symbolized class. By default it
   uses the same module name as the source class.
 * **override**: an optional dict that contains key value pairs to override
   the symbolized class' attributes.

Enable Symbolic Validation
--------------------------

Users can enable symbolic validation on class arguments by providing value
specifications during :func:`pg.symbolize <pyglove.symbolic.symbolize>`,
similar to how it's done with :func:`pg.members <pyglove.symbolic.members>`. 
This allows for automatic validation of the argument values on a symbolic
object at the time of its creation and any subsequent manipulation::


    SymbolicFoo = pg.symbolize(Foo, [
        ('x', pg.typing.Int())
    ])

    # Raises: `x` should be an integer.
    SymbolicFoo('abc')

Class Inheritance
-----------------

A symbolized class can be subclassed, which automatically makes the subclass
symbolic. For example, ``Bar`` is also a symbolized class since it subclasses
``Foo``::
 
   @pg.symbolize
   class Foo:
     def __init__(self, x):
       self._x = x

   class Bar(Foo):
      def __init__(self, y):
        super().__init__(y ** 2)

.. tip::

    There is a subtle difference between symbolic classes created by subclassing
    ``pg.Object`` and those created using ``pg.symbolize``. While the former inherit
    symbolic fields from their base classes (like :class:`dataclasses.dataclass``),
    the latter do not. Instead, a symbolized class always has the same number of fields
    aligned with its ``__init__`` signature. The field definitions passed to ``pg.symbolize``
    can specify the validation rules or add metadata to the arguments, but cannot add
    new fields whose keys are absent from the ``__init__`` signature. If default values
    are present in the signature, they will be checked against the fields when they are
    present and will be carried over to the fields if they are not specified.


Symbolic Function (Functor)
***************************

A *symbolic function* (or *functor*) represents a symbolized Python function.
Symbolic functions are subclasses of :class:`pg.Functor <pyglove.symbolic.Functor>`, which
is a symbolic class with a ``__call__`` method. Therefore, their instances are also symbolic
objects, representing functions with bound arguments.

Functors vs. Regular Functions
==============================

In Python, this is no language construct for representing a bound function.
When a function is bound with values, it is immediatelly evaluated, leaving
no runtime entity that captures the binding itself. For example::

    def foo(x, y):
      return x + y
    
    # Binding is evaluated immediately,
    # and there is no long living object for a bound function.
    assert foo(1, 2) == 3

.. note::

    :func:`functools.partial` is commonly used to create partially bound
    functions that can be passed around, but it is not yet widely used to
    make bound functions and objects interchangeable and equal throughout
    a software system.

PyGlove introduces the concept of symbolic functions, which allows bound
functions to be treated on par with objects. This means that bound functions
can be created and manipulated using the same API as symbolic objects. 
Instead of invoking the function immediately at binding time, a symbolic
function returns an object representing the binding. The user must then call
the object separately to invoke the function's body. This allows for greater
flexibility and consistency in the way functions and objects are handled
throughout a software system. For example::

    @pg.symbolize
    def foo(x, y):
      return x + y
    
    # `f` is a bound `foo` with (1, 2).
    f = foo(1, 2)

    # `f` needs to be explicitly called.
    f()

Creating Symbolic Functions
===========================

Creating a symbolic function is simply to annotate it with
:func:`pg.symbolize <pyglove.symbolic.symbolize>` decorator, for example::

    @pg.symbolize
    def foo(x, y, z):
      return x + y + z

If the function is defined in a source file that can be modified, you can also do::

    foo = pg.symbolize(another_module.foo)


Defining Validation Rules
=========================

Similar as symbolic classes, users can also provide an optional specification for the
validation rules for its arguments::

    @pg.symbolize([
        ('x', pg.typing.Int(min_value=1)),
        ('z', pg.typing.Int(min_value=1))
    ])
    def foo(x, y, z):
      pass

The specification is not required to cover all argument names. For ommited arguments,
PyGlove's runtime validation system treats them as :class:`pg.typing.Any <pyglove.typing.Any>`().

Handling Return Value
---------------------

Symbolic validation can be used not only to check the values of arguments, but also to validate
the return value of a function or method. This allows for increased type safety and ensures that
the function or method is returning the expected output. To validate the return value, we can do::

    @pg.symbolize([], returns=pg.typing.Int(min_value=0, max_value=10))
    def foo(x, y, z):
      pass

Handling ``*args``
------------------

We can add validation rule for variable positional argument by defining
a field whose key is the name of the variable positional argument, and its
value a :class:`pg.typing.List <pyglove.typing.List>`::

    @pg.symbolize([
        ('args', pg.typing.List(pg.typing.Int(min_value=1)))
    ])
    def bar(x, *args):
      pass

    # Okay.
    bar(1, 2, 3)
    assert bar.sym_init_args.args == [2, 3]

    # Not okay: 'abc' is not an integer.
    bar(1, 'abc')

Handling ``**kwargs``
---------------------

Similarly, we can add validation rules for variable keyword arguments.
If we want to use a uniform rule for all keyword arguments, we can do
the following::

    @pg.symbolize([
        (pg.typing.StrKey('foo.*'), pg.typing.Int())
    ])
    def bar(x, y, **kwargs):
      pass

    # Okay: `foo1` can match with regular expression 'foo.*' and 3 is an integer.
    bar(1, 2, foo1=3)

    # Not okay: `s` is neither an argument nor acceptable
    # by the regular expression 'foo.*'.
    bar(1, 2, s=3)

    # Not okay: 'abc' is not an integer.
    bar(1, 2, foo2='abc')

Furthermore, if we want to specify validation rules separately
based on the keyword, we can add multiple fields in the definition.
For example::

    @pg.symbolize([
        ('p', pg.typing.Int()),
        ('q', pg.typing.Str()),
        (pg.typing.StrKey(), pg.typing.Bool())
    ])
    def bar(x, y, **kwargs)
      pass

    # Okay: `p`, `q` are applied with separate validation rules
    # instead of using the general keyword argument rules.
    bar(1, 2, p=3, q='abc', r=True)


Advanced Binding
================

Symbolic function supports a set of advanced binding capabilities.

Regular Binding
---------------

Create a symbolic function instance with all arguments bound::

    @pg.symbolize
    def foo(x, y, z):
      return x + y + z

    f = foo(1, 2, 3)


Partial Binding
---------------

Partially bind a symbolic function on some arguments::

    # `f` is partially bound on `y`.
    f = foo(y=1)

Incremental Binding
-------------------

Incremental binding can be done via attribute assignment::

    f.x = 2

Rebinding
---------

We can also override an existing bound argument::

    f.x = 3

    # Or:

    f.rebind(x=3)


Binding at Invocation Time
--------------------------

A functor can be invoked via its ``__call__`` method, with arguments that are
not yet provided, or new values to override exisitng bound ones::

    # Invoke functor with x=2 (incrementally bound), y=1 (early bound)
    # and z=2.
    f(z=2)

    # Invoke functor with x=1 (override existing value 2), y=1 (early bound)
    # and z=2.
    f(z=2, x=1, override_args=True)

    # Raises: x is already bound.
    f(z=2, x=1)

.. tip::

    When `f` is called with arguments that is not yet bound, it only use the provided value
    for calling the function, without binding it. For example::

        f(x=1, y=2)

        # Call `f` with argument `z` which is not bound yet.
        f(z=3)

        # Raises: `z` is required but not provided.
        f()


Other Operations
================

The same as symbolic classes, symbolic operations can be applied to symbolic functions too.
See :doc:`operations` for details.


Symbolic Container Types
************************

PyGlove provides :class:`pg.List <pyglove.symbolic.List>` and :class:`pg.Dict <pyglove.symbolic.Dict>`
to address the symbolic needs for :class:`list` and :class:`dict`.

Symbolic List
=============

:class:`pg.List <pyglove.symbolic.List>` implements a list type whose instances are
symbolically programmable. ``pg.List`` is

  * a subclass of the standard Python :class:`list`.
  * a subclass of class :class:`pg.Symbolic <pyglove.symbolic.Symbolic>`.

Instantiation
-------------


``pg.List`` can be used as a regular list::

    # Construct a symbolic list from an iterable object.
    l = pg.List(range(10))

Symbolic Validation
-------------------

``pg.List`` supports symbolic validation through the ``value_spec`` argument::

    l = pg.List([1, 2, 3], value_spec=pg.typing.List(
        pg.typing.Int(min_value=1),
        max_size=10
    ))

    # Raises: 0 is not in acceptable range.
    l.append(0)

See :doc:`validation` for more details.


Subscription to Changes
-----------------------

Users can subscribe to subtree updates within ``pg.List``::

    def on_change(updates):
      print(updates)

    l = pg.List([{'foo': 1}], onchange_callaback=on_change)

    # `on_change` will be triggered on item insertion.
    l.append({'bar': 2})

    # `on_change` will be triggered on item removal.
    l.pop(0)

    # `on_change` will also be triggered on subtree change.
    l.rebind({'[0].bar': 3})


Operations
----------

See :doc:`operations` for details.

Caveats
-------


Recursive Symbolic Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pg.List`` converts a regular list into its symbolic representation. Therefore,
if the input list contains nested ``list`` or ``dict``, they will be converted to
instances of ``pg.List`` and ``pg.Dict`` respectively. For example::

    regular_list = [
        [1, 2, 3],
        {'a': 1, 'b': 2}
    ]
    symbolic_list = pg.List(regular_list)

    # Nested lists and dicts are converted into symbolic ones.
    assert isinstance(symbolic_list[0], pg.List)
    assert isinstance(symbolic_list[1], pg.Dict)

Symbolic Hashing
^^^^^^^^^^^^^^^^

A regular list is not hashable, for example::

    # Raises: a list is not hashable.
    hash([1, 2, 3])

However, a symbolic list is hashable, whose hash value is computed based on the
symbolic representations of its items. Therefore, two bindings with the same
type and parameters will end up with the same hash value::

    @pg.members([
        ('x', pg.typing.Int())
    ])
    class Foo(pg.Object):
      pass

    assert hash(pg.List([Foo(1), Foo(2)])) == hash(pg.List([Foo(1), Foo(2)]))


Symbolic Dict
=============


Class :class:`pg.Dict <pyglove.symbolic.Dict>` implements a dict type whose instances are
symbolically programmable. ``pg.Dict`` is

  * a subclass of the standard Python ``dict``.
  * a subclass of class :class:`pg.Symbolic <pyglove.symbolic.Symbolic>`.

Instantiation
-------------

``pg.Dict`` can be used as a regular dict with string keys::

    # Construct a symbolic dict from key value pairs.
    d = pg.Dict(x=1, y=2)

or::

    # Construct a symbolic dict from a mapping object.
    d = pg.Dict({'x': 1, 'y': 2})

.. warning::

    ``pg.Dict`` does not support non-string keys.


Attribute Access
^^^^^^^^^^^^^^^^

Besides regular items access using ``[]``, ``pg.Dict`` allows attribute access
to its keys::

    # Read access to key `x`.
    assert d.x == 1

    # Write access to key 'y'.
    d.y = 1

Creating Hyper Dict
^^^^^^^^^^^^^^^^^^^

``pg.Dict`` is oftentimes used for constructing hyper values during
prototyping, without introducing symbolic classes or functions::

    space = pg.Dict(x=pg.oneof(range(10)), y=pg.floatv(0.1, 1.0))
    example = next(pg.random_sample(space))


Symbolic Validation
^^^^^^^^^^^^^^^^^^^

``pg.Dict`` supports symbolic validation when the ``value_spec`` argument is
provided::

    d = pg.Dict(x=1, y=2, value_spec=pg.typing.Dict([
        ('x', pg.typing.Int(min_value=1)),
        ('y', pg.typing.Int(min_value=1)),
        (pg.typing.StrKey('foo.*'), pg.typing.Str())
    ])
    
    # Okay: all keys started with 'foo' is acceptable and are strings.
    d.foo1 = 'abc'

    # Raises: 'bar' is not acceptable as keys in the dict.
    d.bar = 'abc'

See :doc:`validation` for more details.


Subscription to Changes
^^^^^^^^^^^^^^^^^^^^^^^


Users can subscribe to subtree updates within ``pg.Dict``::

    def on_change(updates):
      print(updates)

    d = pg.Dict(x=1, onchange_callaback=on_change)

    # `on_change` will be triggered on item insertion.
    d['y'] = {'z': 1}

    # `on_change` will be triggered on item removal.
    del d.x

    # `on_change` will also be triggered on subtree change.
    d.rebind({'y.z': 2})


Operations
^^^^^^^^^^

See :doc:`operations` for details.

Caveats
-------

Recursive Symbolic Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


``pg.Dict`` converts a regular dict into its symbolic representation. Therefore,
if the input dict contains nested ``list`` or ``dict``, they will be converted to
instances of ``pg.List`` and ``pg.Dict`` respectively. For example::

    regular_dict = {
        'a': [1, 2, 3],
        'b': {
            'x': 1,
            'y': 2
        }
    }
    symbolic_dict = pg.Dict(regular_dict)

    # Nested lists and dicts are converted into symbolic ones.
    assert isinstance(symbolic_dict.a, pg.List)
    assert isinstance(symbolic_dict.b, pg.Dict)

Symbolic Hashing
^^^^^^^^^^^^^^^^

A regular dict  is not hashable, for example::

    # Raises: a dict is not hashable.
    hash({'x': 1, 'y': 2}})

However, a symbolic dict is hashable, whose hash value is computed based on the
symbolic representations of its items. Therefore, two bindings with the same
type and parameters will end up with the same hash value::

    @pg.members([
        ('x', pg.typing.Int())
    ])
    class Foo(pg.Object):
      pass

    assert hash(pg.Dict(x=Foo(1))) == hash(pg.Dict(x=Foo(1)))
