Symbolic Validation
###################

PyGlove uses a runtime type system (module :mod:`pg.typing <pyglove.typing>`)
to prevent errors in symbolic object manipulation. Without it, bugs can arise easily,
such as a mistakenly modified int attribute. PyGlove's type system automatically
validates symbolic objects on creation and modification, reducing the need for manual
input validation, allowing the developer to focus on the main logic.

.. Symbolic objects are intended to be manipulated after creation. Without a
.. runtime  typing system, things can go wrong easily. For instance, an ``int``
.. attribute which was mistakenly modified at early program stages can be very
.. difficut to debug at later stages. PyGlove introduces a runtime type system
.. (module :module:`pg.typing <pyglove.typing>`) that automatically validates
.. symbolic objects upon creation and modification, minimizing boilerplated code
.. for input validation, so the developer can focus on the main business logic.

Runtime Typing
**************

The runtime type system of PyGlove is based on *schemas*
(class :class:`pg.typing.Schema <pyglove.typing.Schema>`), which define the symbolic
attributes of a type (e.g. dict, list, class, function).

A schema consists of *symbolic fields* (class :class:`pg.typing.Field <pyglove.typing.Field>`)
that specify the keys and acceptable values for the attributes. Schemas are created
and associated with a symbolic type through decorators like
:func:`pg.members <pyglove.symbolic.members>` and :func:`pg.symbolize <pyglove.symbolic.symbolize>`
during the declaration. For example::

  @pg.members([
      ('x', pg.typing.Int(default=1)),
      ('y', pg.typing.Float().noneable())
  ])
  class A(pg.Object):
    pass

  print(A.schema)

  @pg.symbolize([
      ('a', pg.typing.Int()),
      ('b', pg.typing.Float())
  ])
  def foo(a, b):
    return a + b

  print(foo.schema)


Key and Value Specifications
============================

The first argument of `pg.members` and `pg.symbolize` takes a list of ``Field`` as the definitions
for the symbolic attributes. It's usually described by a tuple of four items::

    (Key specification, Value specification, Doc string, Field metadata)

The key specification (or ``KeySpec``, described by class :class:`pg.typing.KeySpec <pyglove.typing.KeySpec>`) and
value specification (or ``ValueSpec``, described by class :class:`pg.typing.ValueSpec <pyglove.typing.ValueSpec>`) are
required, while the doc string and the field metadata are optional.
``KeySpec`` defines acceptable identifiers for this field, and ``ValueSpec``
defines the attribute's type, default value and validation rules. The doc string provides additional
description for the field, and the field metadata can be used for code generation.

The following code snippet illustrates common ``KeySpec`` and
``ValueSpec`` subclasses and their usage with a manually created schema::

    schema = pg.typing.create_schema([
        # Primitive types.
        ('a', pg.typing.Bool(default=True).noneable()),
        ('b', True),       # Equivalent to ('b', pg.typing.Bool(default=True)).
        ('c', pg.typing.Int()),
        ('d', 0),          # Equivalent to ('d', pg.typing.Int(default=0)).
        ('e', pg.typing.Int(
            min_value=0,
            max_value=10).noneable()),
        ('f', pg.typing.Float()),
        ('g', 1.0),        # Equivalent to ('g', pg.typing.Float(default=1.0)).
        ('h', pg.typing.Str()),
        ('i', 'foo'),      # Equivalent to ('i', pg.typing.Str(default='foo').
        ('j', pg.typing.Str(regex='foo.*')),

        # Enum type.
        ('l', pg.typing.Enum('foo', ['foo', 'bar', 0, 1]))

        # List type.
        ('m', pg.typing.List(pg.typing.Int(), size=2, default=[])),
        ('n', pg.typing.List(pg.typing.Dict([
            ('n1', pg.typing.List(pg.typing.Int())),
            ('n2', pg.typing.Str().noneable())
        ]), min_size=1, max_size=10, default=[])),

        # Dict type.
        ('o', pg.typing.Dict([
            ('o1', pg.typing.Int()),
            ('o2', pg.typing.List(pg.typing.Dict([
                ('o21', 1),
                ('o22', 1.0),
            ]))),
            ('o3', pg.typing.Dict([
                # Use of regex key,
                (pg.typing.StrKey('n3.*'), pg.typing.Int())
            ]))
        ]))

        # Tuple type.
        ('p', pg.typing.Tuple([
            ('p1', pg.typing.Int()),
            ('p2', pg.typing.Str())
        ]))

        # Object type.
        ('q', pg.typing.Object(A, default=A()))

        # Type type.
        ('r', pg.typing.Type(int))

        # Callable type.
        ('s', pg.typing.Callable([pg.typing.Int(), pg.typing.Int()],
                                  kw=[('a', pg.typing.Str())])),

        # Functor type (same as Callable, but only for symbolic.Functor).
        ('t', pg.typing.Functor([pg.typing.Str()],
                                 kwargs=[('a', pg.typing.Str())]))

        # Union type.
        ('u', pg.typing.Union([
            pg.typing.Int(),
            pg.typing.Str()
        ], default=1),

        # Any type.
        ('v', pg.typing.Any(default=1))
    ])


Schema inheritance
==================

In PyGlove, symbolic attributes and their defining schemas can be inherited during subclassing.
The base class's schema is carried over to the subclass and can be overridden by redefining a
field with the same key. The subclass cannot arbitrarily change the base class's field but must
use a more restrictive validation rule of the same type or change the default value. See
:meth:`ValueSpec.extend <pyglove.typing.ValueSpec.extend>` for details.

The code snippet below illustrates schema inheritance during subclassing::

  @pg.members([
      ('x', pg.typing.Int(min_value=1)),
      ('y', pg.typing.Float()),
  ])
  class A(pg.Object):
    pass

  @pg.members([
      # Further restrict inherited 'x' by specifying the max value, as well
      # as providing a default value.
      ('x', pg.typing.Int(max_value=5, default=2)),
      ('z', pg.typing.Str('foo').freeze())
  ])
  class B(A):
    pass

  assert B.schema.fields.keys() == ['x', 'y', 'z']

  @pg.members([
      # Raises: 'z' is frozen in class B and cannot be extended further.
      ('z', pg.typing.Str())
  ])
  class C(B):
    pass


Automatic type conversions
**************************

PyGlove's typing system can be extended through type conversion, which allows
for registering type conversions. If a value being assigned to an attribute
does not match its type defined by the ``ValueSpec``, a conversion will occur
automatically when a converter from the input type to the target type exists.

Type converter
==============

Type converter is a callable object that converts a source value into a target
value. For example::

  class A:
    def __init__(self, str):
      self._str = str

    def __str__(self):
      return self._str

    def __eq__(self, other):
      return isinstance(other, self.__class__) and self._str == other._str

  pg.typing.register_converter(A, str, str)
  pg.typing.register_converter(str, A, A)

  assert pg.typing.Str().accept(A('abc')) == 'abc'
  assert pg.typing.Object(A).accept('abc') == A('abc')

See :func:`pyglove.typing.register_converter` for more details.

Built-in converters
===================

By default, PyGlove registered converters between the following pairs:

.. list-table::
   :header-rows: 1
   :widths: 20 50
   :align: center

   * - Source
     - Target

   * - :class:`int`
     - :class:`datetime.datetime`

   * - :class:`str`
     - :class:`pg.KeyPath <pyglove.object_utils.KeyPath>`
