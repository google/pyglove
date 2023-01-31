Symbolic Placeholding
#####################

A regular Python object serves as a program state after its creation. Therefore,
it must be constructed with all required arguments fully specified and meet the type definition.
On the other hand, symbolic objects can serve as pure representations and can
exist before being fully specified. This enables developers to start with an unfinished
representation and gradually make it concrete. This is achieved through *symbolic placeholding*, which
results in *abstract objects*.

Abstract Objects
****************

Abstract objects are symbolic objects that are not concrete, meaning they are not yet ready
for triggering the ``__init__`` logic upon creation. For example, ``Add(x=TBD(), y=1)`` represents an
addition between a to-be-determined value and 1. Abstract objects can be *partial objects* or
*pure symbolic objects*.


Partial Objects
===============

Partial objects are objects that have missing parts, which can be instantiated through class method
:meth:`partial <pyglove.symbolic.Object.partial>`. Under the hood, the missing
parts in the symbolic object are placeheld with :data:`pg.MISSING_VALUE <pyglove.typing.MISSING_VALUE>`.
Such placeholding can occur at the immediate children level or deeper into sub-trees. For example::

  @pg.symbolize
  class Exp:

    def __init__(self, x, y):
      self.x = x
      self.y = y
      print('`__init__` is called.')


  # `a` is a partial object as `a.y` is not specified.
  a = Exp.partial(x=1)
  assert pg.is_partial(a)
  assert a.sym_init_args.x == pg.MISSING_VALUE

  # `b` is also a partial object as it contains partial object `a` as its sub-node.
  b = Exp.partial(x=a, b=2)
  assert pg.is_partial(b)

  # Making partial objects concrete.
  # Till this point, we shall see message "`__init__` is called" printed out.
  a.rebind(x=2)
  assert not pg.is_partial(a)
  assert not pg.is_partial(b)


More on Partial Objects Creation
--------------------------------

Partial objects need to be explicitly created with :meth:`partial <pyglove.symbolic.Object.partial>`. For example::

  # Raises: `y` is not provided.
  Exp(x=1)

  # Raises: `y` is partial.
  Exp(x=1, y=Exp.partial(x=1))

This means that when users need to create a hierarchy of partial objects, it requires every containing
class to call ``parital`` explicitly. This prevents human errors, but is also inconvenient. 
PyGlove offers context manager :func:`pg.allow_partial <pyglove.symbolic.allow_partial>` to address
this scenario, allowing partial objects to be created using standard class constructors::

  with pg.allow_partial():
    a = Exp(x=1, y=Exp(1))
  assert pg.is_partial(a)

For a partial object, the missing values in the object hierarchy can be queried via
:meth:`sym_missing <pyglove.symbolic.Symbolic.sym_missing>`::

  # Shall print {'y.y': pg.MISSING_VALUE}
  a.sym_missing()

Partial Functions
-----------------

For functions, there is a distinction between a *partially bound function* and a *partial function object*:

A partially bound function is a :class:`pg.Functor <pyglove.symbolic.Functor>` object whose arguments are
partially specified, but each of the specified argument is concrete. For example::

  @pg.symbolize
  def foo(x, y)
    return x + y
  
  @pg.symbolize
  def bar(a, b):
     return a() + b()

  # `f` is partially bound, but not partial.
  f = foo(1)
  assert not f.fully_bound
  assert not pg.is_partial(f)
  # `f` can be evaluated by providing the missing argument at call time.  
  assert f(y=2) == 3

  # `g` is not partial since `f` is not partial.
  g = bar(f)
  assert not pg.is_partial(g)

  # Raises: calling `a()` within `bar` will fail since `f` is partially bound.
  # However, it's the user's responsibility to make sure
  # a partially bound function may be used as an argument.
  g(b=foo(1, 2))

On the other hand, a partial function object is ``pg.Functor`` object whose bound arguments
contain partial values. For example::

  @pg.symbolize
  class Foo:
    def __init__(self, v):
      self.v = v
    
    def __call__(self):
      return self.v ** 2

  # `f` is now partial since `Foo()` is partial.
  f = bar(Foo.partial())


Pure Symbolic Objects
=====================

PyGlove introduces the concept of pure symbolic objects, for describing a program whose details will be decided later.
Leaf pure symbolic objects are the instances of :class:`pg.PureSymbolic <pyglove.symbolic.PureSymbolic>` subclasses.
Symbolic objects that contain pure symbolic objects as its sub-nodes are also pure symbolic::

  @pg.symbolize
  class Foo:
    def __init__(self, x, y):
      self.x = x
      self.y = y
      self.z = x + y

  @pg.symbolize
  class Bar:
    def __init__(self, foo):
      self.foo = foo
    
    def __call__(self):
      return self.foo.x * self.foo.y

  # `bar1` is a concrete object since all its sub-nodes are concrete.
  bar1 = Bar(Foo(1, 2))
  assert not pg.is_pure_symbolic(bar1)

  class TBD(pg.PureSymbolic):
    pass

  # `bar2` is pure symbolic since its `foo` argument is pure symbolic, which
  # contains an object of `TBD` which is a subclass of `PureSymbolic`.`
  bar2 = Bar(Foo(TBD(), 2))
  assert pg.is_pure_symbolic(bar2)

Delayed Evaluation
------------------

A pure symbolic object cannot be evaluated until it becomes concrete, meaning that
the behavior of calling any non-symbolic method of a pure symbolic object is undetermined. 
As a result, the ``__init__`` method of a pure symbolic object will also be delayed. For example::

  # Raises: `bar2.__init__` has not been evaluated yet since it's pure symbolic.
  bar2.foo

  # Raises: `bar2.__call__` cannot be called since it's pure symbolic.
  bar2()

  # Manipulate `bar2` into a concrete object by replacing all `TBD` with integer 1.
  # which triggers its `__init__`.
  bar2.rebind(lambda k, v, p: 1 if isinstance(v, TBD) else v)

  # Okay: `bar2.__init__` is called by the end of `bar2.rebind` since it's then concrete.
  assert bar2.sym_init_args.foo.z == 3

  # Okay: `bar2.__call__` can be called now since it's concrete.
  assert bar2() == 2

Placeholding Targets
--------------------

Besides, the ``PureSymbolic`` subclass developer can control what symbolic fields can be
placeheld by the current pure symbolic class. For example, hyper primitive :func:`pg.oneof <pyglove.hyper.oneof>`
will make sure all candidate values are acceptable to the target field when it is used
as a placeholder. This can be done via implementing the :meth:`~pyglove.symbolic.PureSymbolic.custom_apply`
method, which is inherited from the :class:`pg.typing.CustomTyping <pyglove.typing.CustomTyping>` interface. 


Caveats
=======

As we have shown above, for symbolic classes created with :func:`pg.symbolize <pyglove.symbolic.symbolize>`,
the ``__init__`` method will be delayed until the object becomes concrete. For symbolic classes created by
subclassing :class:`pg.Object <pyglove.symbolic.Object>`, special care needs to be taken care of when handling
:ref:`_on_bound`, :ref:`_on_init` and :ref:`_on_change` events. These methods are always triggered when the
object is first created or later mutated, even when the object is abstract. So always check `self.sym_abstract`
to carry on the logic that requires a concrete `self`. For example::

  @pg.members([
    ('x', pg.typing.Int()),
    ('y', pg.typing.Int())
  ])
  class MyObject:
    def _on_bound(self):
      super()._on_bound()
      # This check ensures all symbolic attributes are concrete.
      if not self.sym_abstract:
        self._z = self.x + self.y