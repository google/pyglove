Symbolic Detour
***************

PyGlove offers the :func:`pg.detour <pyglove.detouring.detour>` feature to redirect a class
to another class or function, allowing the object creation behavior of that class to be changed
dynamically at runtime.

Motivation
==========

Symbolizing existing classes and functions is straightforward, but in order to use them,
the original classes used in existing code must be replaced with the symbolic versions.
However, modifying the source code may not always be possible or objects created within
a function or class method may not be accessible externally, making it impossible to
manipulate them as part of the symbolic tree.

For example::

    @pg.symbolize
    def foo():
      # Object `a` is not a part of `foo`'s interface,
      # therefore it cannot be seen from the symbolic tree
      # that contains a `foo` object.
      a = A(1)
      return a.do_something()


Symbolic Detour (SD) is a solution for these scenarios, it redirects the ``__new__`` method
of a class to another class or function when it's evaluated under a context manager.
It is not dependent on symbolization and can be used to detour any classes, it does not
require the presence of symbolic objects to modify the program.

Usage
=====

Redirecting Classes to Classes
------------------------------

The code below illustrates class ``Foo`` is detoured to ``Bar`` under the context manager
of :func:`pg.detour <pyglove.detouring.detour>`::

    class Foo:

      def __init__(self, x, y):
        self.x = x
        self.y = y
    
      def __call__(self):
        return self.x + self.y

    class Bar:

      def __init__(self, a, b):
        self.a = a
        self.b = b

      def __call__(self):
        return self.a * self.b


    def my_fun():
      # Parameters of `Foo` is not exposed as any argument of `my_fun`.
      return Foo(1, 2)() + 2

    # Symbolically detour `Foo` to `Bar` under the context manager,
    # which changes the behavior inside `my_fun` while not requiring
    # to modify its source code.
    with pg.detour([(Foo, Bar)]):
      v = my_fun()

    assert v == (1 * 2) + 2

    # Execute `my_fun` outside the context manager will result in
    # the creation of original `Foo` object.
    v2 = my_fun()
    assert v2 == (1 + 2) + 2

Redirecting Classes to Functions
------------------------------

Symbolic detour can redirect classes to functions, but it has a limitation:
if the function returns an object of the same type (or a subtype) as the original
class, the object's ``__init__`` method will be called again with the original
arguments. This means that using detour to change argument values won't work. 
For example::

    def foo_with_incremented_x(cls, x, y):
      return cls(x + 1, y)

    with pg.detour([(Foo, foo_with_incremented_x)]):
      v= my_fun(1, 2)
    
    # Fails: though argument `x` is incremented by the function,
    # but Python calls the `__init__` again with the original value 1,
    # thus the Foo's value remains 
    assert v == (2 * 2) + 2

A simple solution is to create an instance of the symbolized class instead of the
original class. Symbolic classes have built-in handling for re-initialization,
which allows them to do nothing when ``__init__`` is called after an object is
already initialized. For example::

    SymbolicFoo = pg.symbolize(Foo)

    def foo_with_incremented_x(cls, x, y):
      return SymbolicFoo(x + 1, y)

    with pg.detour([(Foo, foo_with_incremented_x)]):
      v= my_fun(1, 2)
    
    # Okay now!
    assert v == (2 * 2) + 2


The Nesting Rules
-----------------

Symbolic detour can be nested, with outer scope mappings taking precedence
over inner mappings, allowing users to change object creation behaviors from
the outside. For example, the following code will detour class A to class C::


    with pg.detour([(A, C)]):
      with pg.detour([A, B]):
        v = A()   # v is a C object.


Detour is transitive across the inner and outer scope. For example::


    with pg.detour([(B, C)]):
      v1 = A()     # v1 is an A object.
      with pg.detour([A, B]):
        v2 = A()    # v2 is a C object. (A -> B -> C)


For more details about symbolic detour, see :func:`pg.detour <pyglove.detouring.detour>`.
