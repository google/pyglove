# Copyright 2021 The PyGlove Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common interfaces for evolutionary algorithms."""
import abc
import collections
import random
import typing
from typing import Any, Callable, Iterable, List, Text, Optional, Tuple, Type

import pyglove.core as pg
from pyglove.generators.evolution import scalars

# We disable implicit str concat as it is commonly used class schema docstr.
# pylint: disable=implicit-str-concat


#
# Common keys used in DNA meta-data for evolution.
#

# Proposal ID is an 1-based integer that is incremented for every proposal.
DNA_METADATA_PROPOSAL_ID = 'proposal_id'

# Feedback sequence number is an 1-based integer that is incremented for
# every feedback.
DNA_METADATA_FEEDBACK_SEQUENCE_NUMBER = 'feedback_sequence_number'

# Fitness for a DNA, can be a float number for single-objective or
# a tuple of float numbers for multiple objectives.
DNA_METADATA_FITNESS = 'reward'

# Whether a DNA belongs to the initial population.
DNA_METADATA_INITIAL_POPULATION = 'initial_population'

# DNA generation ID, which is incremented for each reproduction.
DNA_METADATA_GENERATION_ID = 'generation_id'


def set_proposal_id(dna: pg.DNA, proposal_id: int) -> pg.DNA:
  """Set the proposal ID to a DNA and return itself."""
  return dna.set_metadata(DNA_METADATA_PROPOSAL_ID, proposal_id)


def get_proposal_id(dna: pg.DNA) -> int:
  """Returns the 1-based proposal ID of the DNA."""
  return dna.metadata[DNA_METADATA_PROPOSAL_ID]


def set_generation_id(dna: pg.DNA, generation_id: int) -> pg.DNA:
  """Set the generation ID to a DNA and return itself."""
  return dna.set_metadata(DNA_METADATA_GENERATION_ID, generation_id)


def get_generation_id(dna: pg.DNA) -> int:
  """Returns the 1-based generation ID."""
  return dna.metadata[DNA_METADATA_GENERATION_ID]


def set_feedback_sequence_number(
    dna: pg.DNA, sequence_number: int) -> pg.DNA:
  """Set the feedback sequence number to a DNA and return itself."""
  return dna.set_metadata(
      DNA_METADATA_FEEDBACK_SEQUENCE_NUMBER, sequence_number)


def get_feedback_sequence_number(dna: pg.DNA) -> Optional[int]:
  """Returns the 1-based feedback sequence number of the DNA."""
  return dna.metadata.get(DNA_METADATA_FEEDBACK_SEQUENCE_NUMBER)


def set_fitness(dna: pg.DNA,
                fitness: typing.Union[float, Tuple[float]]) -> pg.DNA:
  """Set the evaluated fitness to a DNA and return itself."""
  return dna.set_metadata(DNA_METADATA_FITNESS, fitness)


def get_fitness(dna: pg.DNA) -> typing.Union[float, Tuple[float]]:
  """Returns the fitness associated with a DNA."""
  return dna.metadata[DNA_METADATA_FITNESS]


def _set_initial_population(dna: pg.DNA, is_initial: bool) -> pg.DNA:
  """Set whether the DNA belongs to the initial population."""
  return dna.set_metadata(DNA_METADATA_INITIAL_POPULATION, is_initial)


def is_initial_population(dna: pg.DNA) -> bool:
  """Returns True if the DNA belongs to the initial population."""
  return dna.metadata[DNA_METADATA_INITIAL_POPULATION]


#
# Interfaces for evolutionary operations.
#


def operation_spec(
    input_element_spec: Optional[pg.typing.ValueSpec] = None,
    output_element_spec: Optional[pg.typing.ValueSpec] = None
    ) -> pg.typing.ValueSpec:
  """Returns the value spec (PyGlove typing) for an evolutionary operation.

  We use `pg.typing.Callable` instead of `pg.typing.Object(Operation)`
  to make it more flexible to plugin lambdas.

  Args:
    input_element_spec: The value spec for input element.
    output_element_spec: The value spec for output element.

  Returns:
    A value spec for Callable[[List[DNA]], List[DNA]].
  """
  if input_element_spec is None:
    input_element_spec = pg.typing.Object(pg.DNA)
  if output_element_spec is None:
    output_element_spec = pg.typing.Object(pg.DNA)
  return pg.typing.Callable(
      [pg.typing.List(input_element_spec)],
      returns=pg.typing.List(output_element_spec))


class Operation(pg.Object):
  """Base class for evolutionary operations.

  An evolutionary operation transforms a DNA list into another DNA list.
  This abstraction provides the flexibility to describe common evolutionary
  operations including but not limited to:

    * Selection:  Select M parents from a population of size N.
    * Recombination: Generate M child candidates from N parents.
    * Mutation: Mutate the N child candidates one by one, each can
      produce one or K DNA, resulting in N * K child DNA.

  Operations are compositional via the following operators::

    x >> y:    Pipeline x and y by passing the output of x to y as input.
    x + y:     Concatenate the outputs of x and y, based on the same input.
               Resulted list could contain duplicates.
    x[m:n:p]   Slice x's output from m (inclusive) to n (exclusive) with step p.
    x | y:     Union the outputs of x and y based on the same input.
               Resulted list will not contain duplicated items.
    x & y:     Intersect the outputs of x and y, based on the same input.
               Resulted list will not contain duplicated items.
    x - y:     Remove elements from x's output which also appears in y's output,
               based on the same input.
    x ^ y:     Keep items that are either in x's output or in y's output, but
               not both. Duplicated items in x's output or in y's output will
               be preserved.
    x * K:     Repeat operation x for K times based on the same input,
               concatenate their outputs.
    x ** K:    Pipeline operation x for K times, equivalent to `x >> .. >> x`
               (`x` appears K times in the formula).
    -x or ~x:  Compute the inversion of x's output based on the input.
               Equivalent to `Identity() - x`

    x.if_true:   Conditionally apply x if predicate returns True.
    x.if_false:  Conditionally apply x if predicate returns False.

  Besides, the following operations are commonly used during composition::

    Identity()           :   Returns the input DNA list.

    Lambda(              :   Creates an Operation using lambda.
      lambda x: x[1:])

    Choice([             :   Perform x, y based on probabilities.
      (x, prob1),
      (y, prob2)])

    Conditional(         :   Perform x if input length > 5, otherwise y.
      lambda x: len(x) > 5,
      x, y)

    UntilChange(x)       :   Repeat x until its output changes from the input.

    ElementWise(         :   Perform element-wise operation and concatenate.
      lambda x: sorted(x))

    Flatten()            :   Flatten nested lists into a flat list.

    GlobalStateGetter(   :   Fetch global state by key 'species'.
      'species')

    GlobalStateSetter(   :   Set global state by key 'species' from input.
      'species')

  The operand in compositional operations supports callable objects in signature
  `(List[DNA], int) -> List[DNA]`. For example::

      pg.evolution.selectors.Last(50) >> (lambda x, s: x[:2])
  """

  def __call__(self,
               inputs: List[Any],
               global_state: Optional[pg.geno.AttributeDict] = None,
               step: int = 0) -> List[Any]:
    """Transform a list of input values to a list of output values.

    Args:
      inputs: A list of values as inputs.
      global_state: An `AttributeDict` object (dictionary that provides
        attribute access) as the global state container, which is
        readable/writable during the operation.
      step: Number of examples historically proposed, which can be used for
        determining a cross over schedule.

    Returns:
      A list of values as output of current operation.
    """
    if self.input_element_type is not None:
      elem_type = self.input_element_type
      for i, elem in enumerate(inputs):
        if not isinstance(elem, elem_type):
          raise TypeError(
              f'The input is expected to be a list of {elem_type!r} '
              f'but {elem!r} is encountered at position {i}.')

    # NOTE(daiyip): `global_state` will always be provided by `Evolution`.
    # For testing purpose, we allow it to be None to make the caller more
    # convenient.
    if global_state is None:
      global_state = pg.geno.AttributeDict()

    self._on_input(inputs)

    outputs = self._operate(inputs, global_state=global_state, step=step)
    if self.output_element_type is not None:
      elem_type = self.output_element_type
      for i, elem in enumerate(outputs):
        if not isinstance(elem, elem_type):
          raise TypeError(
              f'The output is expected to be a list of {elem_type!r} but '
              f'{elem!r} is encountered at position {i}.')
    return outputs

  def _on_bound(self):
    super()._on_bound()
    self._operate = make_operation_compatible(self.operation_method)

  def _on_input(self, inputs: List[Any]) -> None:
    """Event that is triggered when an input is received.."""

  @property
  def input_element_type(
      self) -> typing.Union[None, Type[Any], Tuple[Type[Any]]]:
    """Retuns the input element type. Subclasses can override."""
    return None

  @property
  def output_element_type(
      self) -> typing.Union[None, Type[Any], Tuple[Type[Any]]]:
    """Retuns the output element type. Subclasses can override."""
    return None

  @property
  def operation_method(self) -> Callable[..., List[Any]]:
    """Returns a member method as operation."""
    return self.call

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    """Subclasses should override this method.

    The `global_state` and `step` are optional for the subclasses' call
    signature.

    Args:
      inputs: A list of values as inputs.
      global_state: An `AttributeDict` object as the global state container,
        which is readable/writable during the operation.
      step: Number of examples historically proposed, which can be used for
        determining a cross over schedule.

    Returns:
      A list of values as output of current operation.
    """
    raise NotImplementedError()

  def __rshift__(self, x):
    """The pipeline operator (>>)."""
    if x is None:
      return self
    return Pipeline([self, x])

  def __rrshift__(self, x):
    """The rhs pipeline operator (>>)."""
    if x is None:
      return self
    return Pipeline([x, self])

  def __getitem__(self, index):
    """The slie operator ([])."""
    return Slice(self, index)

  def __or__(self, x):
    """The union operator (|)."""
    if x is None:
      return self
    return Union([self, x])

  def __ror__(self, x):
    """The union operator (|)."""
    if x is None:
      return self
    return Union([x, self])

  def __and__(self, x):
    """The intersection operator (&)."""
    if x is None:
      return self
    return Intersection([self, x])

  def __rand__(self, x):
    """The intersection operator (&)."""
    if x is None:
      return self
    return Intersection([x, self])

  def __add__(self, x):
    """The concatenate operator (+)."""
    if x is None:
      return self
    return Concatenation([self, x])

  def __radd__(self, x):
    """The concatenate operator (+)."""
    if x is None:
      return self
    return Concatenation([x, self])

  def __sub__(self, x):
    """The difference operator (-)."""
    if x is None:
      return self
    return Difference([self, x])

  def __rsub__(self, x):
    """The difference operator (-)."""
    if x is None:
      return Inversion(self)
    return Difference([x, self])

  def __xor__(self, x):
    """The symmetric difference operator (^)."""
    if x is None:
      return self
    return SymmetricDifference([self, x])

  def __rxor__(self, x):
    """The symmetric difference operator (^)."""
    if x is None:
      return self
    return SymmetricDifference([x, self])

  def __mul__(self, k: typing.Union[int, Callable[[int], int]]):
    """The repeat operator (*)."""
    return Repeat(self, k)

  def __pow__(self, k: typing.Union[int, Callable[[int], int]]):
    """The power operator (**)."""
    return Power(self, k)

  def __neg__(self):
    """The negative operator (-)."""
    return Inversion(self)

  def __invert__(self):
    """The inversion operator (~)."""
    return Inversion(self)

  def with_prob(self,
                probability: typing.Union[float, Callable[[int], float]],
                seed: Optional[int] = None):
    """With probability."""
    return Choice([(self, probability)], seed=seed)

  def if_true(self, predicate: Callable[..., bool]):
    """Conditionally applies current operation when predicate returns True.

    Args:
      predicate: The predicate that takes the outputs from the previous
        operation as input, with optional keyword arguments `global_state` and
        `step`. Returns True if current operation needs to be enabled.
        Otherwise no operation will be performed.

    Returns:
       A conditional operation.
    """
    return Conditional(predicate, self, None)

  def if_false(self, predicate: Callable[..., bool]):
    """Conditionally applies current operation when predicate returns False.

    Args:
      predicate: The predicate that takes the outputs from the previous
        operation as input, with optional keyword arguments `global_state` and
        `step`. Returns False if current operation needs to be enabled.
        Otherwise no operation will be performed.

    Returns:
       A conditional operation.
    """
    return Conditional(predicate, None, self)

  def for_each(self, op):
    """For each element in the output perform the operation."""
    return self >> ElementWise(op)

  def until_change(self, max_attempts: Optional[int] = None):
    """Ensure current operation will change the inputs."""
    return UntilChange(self, max_attempts)

  def flatten(self, max_level: Optional[int] = None):
    """Flatten output with a max level."""
    return self >> Flatten(max_level)

  def global_state(self, key: Text):
    """Returns a global state getter based on the key."""
    return GlobalStateGetter(key)

  def as_global_state(self, key: Text):
    """Returns a global state setter based on the key."""
    return Pipeline([self, GlobalStateSetter(key)])

  def set_global_state(self, key: Text, value: Any):
    """Set global state and return current output."""
    return self + GlobalStateSetter(key, value)


class Selector(Operation):
  """Base class for selectors."""

  @abc.abstractmethod
  def select(
      self,
      inputs: List[Any],
      global_state: pg.geno.AttributeDict,
      step: int) -> List[Any]:
    """Select a list of outputs from the inputs.

    A selector has two use cases:

    * Used as parents selector, which selects individuals from the population
      as parents for recombination. It will be called before the recombination
      step within the :meth:`pyglove.evolution.Evolution.propose` method.

    * Used as a population updater, which selects individuals from previous
      population as a new population. It will be called everytime the
      population is updated, triggered by the
      :meth:`pyglove.evolution.Evolution.feedback` method.

    Args:
      inputs: a list of objects as input.
      global_state: An `AttributeDict` object as the global state container,
        which is readable/writable during the operation.
      step: Number of examples historically proposed, which can be used for
        determining a cross over schedule.
    """

  @property
  def operation_method(self):
    return self.select


class DNAOperation(Operation):
  """Operation that takes a list of DNA as both input and output."""

  @property
  def input_element_type(self) -> Type[pg.DNA]:
    """Retuns the input element type."""
    return pg.DNA

  @property
  def output_element_type(self) -> Type[pg.DNA]:
    """Retuns the output element type."""
    return pg.DNA


class Recombinator(DNAOperation):
  """Base class for recombinators."""

  # Number of parents allowed for recombination. If None, there is no limit.
  NUM_PARENTS = None

  @abc.abstractmethod
  def recombine(self,
                parents: List[pg.DNA],
                global_state: pg.geno.AttributeDict,
                step: int) -> List[pg.DNA]:
    """Generate a list of child DNA based on the list of parents given.

    User should override this method with optional keyword arguments
    'global_state' and 'step'.

    The parents DNA contains a metadata field 'generation', which is the
    generation of the parent DNA. If the Recombinator does not assign this
    field for the new child DNA, the child DNA will have the maximum generation
    from the parents plus 1.

    Args:
      parents: Parent trials.
      global_state: An `AttributeDict` object as the global state container,
        which is readable/writable during the operation.
      step: Number of examples historically proposed, which can be used for
        determining a cross over schedule.

    Returns:
      A list of generated child DNA.
    """

  def _on_input(self, inputs: List[Any]) -> None:
    """Override to check number of parents."""
    if self.NUM_PARENTS is not None and len(inputs) != self.NUM_PARENTS:
      raise ValueError(
          f'{self.__class__.__name__} supports recombination on exact '
          f'{self.NUM_PARENTS} parents. Encountered: {inputs!r}.')

  @property
  def operation_method(self):
    return self.recombine


class Mutator(DNAOperation):
  """Base class for mutators.

  A mutator performs a mutation, i.e. a random transformation that converts
  a parent DNA to a child DNA. Mutations should reach the full search space
  through composition and should prefer local transformations individually.
  """

  def _on_bound(self):
    super()._on_bound()
    self._mutate = make_operation_compatible(self.mutate)

  def mutate(
      self,
      dna: pg.DNA,
      global_state: pg.geno.AttributeDict,
      step: int = 0) -> typing.Union[pg.DNA, List[pg.DNA]]:
    """Mutates the DNA at a given step.

    User should override this method or `mutate_list` method with optional
    keyword arguments 'global_state' and 'step'.

    Args:
      dna: DNA to mutate.
      global_state: An `AttributeDict` object as the container of global states.
      step: Number of examples historically proposed, which can be used for
        determining a mutation schedule.

    Returns:
      A new DNA or a DNA list as the result of the mutation.
    """
    raise NotImplementedError()

  def mutate_list(self,
                  dna_list: List[pg.DNA],
                  global_state: pg.geno.AttributeDict,
                  step: int = 0) -> List[pg.DNA]:
    """Mutate the DNA in the input one by one and concatenate their outputs.

    User should override this method instead of `mutate` if mutation depends on
    the list-wise information. Keyword arguments `global_state` and `step` are
    optional when override.

    Args:
      dna_list: a list of DNA to mutate.
      global_state: An `AttributeDict` object as the container of global states.
      step: Number of examples historically proposed, which can be used for
        determining a mutation schedule.

    Returns:
      a list of DNA as the result of the mutation.
    """
    results = []
    for dna in dna_list:
      output = self._mutate(dna, global_state=global_state, step=step)
      if isinstance(output, list):
        results.extend(output)
      else:
        results.append(output)
    return results

  @property
  def operation_method(self):
    return self.mutate_list


@pg.members([
    ('reproduction', operation_spec(),
     'Operation for performing selection and reproduction.'),
    ('population_init',
     pg.typing.Union([
         pg.typing.Object(pg.DNAGenerator),
         pg.typing.Tuple([pg.typing.Object(pg.DNAGenerator), pg.typing.Int()])
     ], default=(pg.geno.Random(), 50)),
     'A DNAGenerator or a tuple of (initial_population_generator, '
     'initial_population_size) as the population initializer for bootstrapping '
     'the initial population. If the initial_population_size is not provided, '
     'it will delegate all `propose` calls to the initial population generator '
     'untill the generator raises a StopIteration error.'),
    ('population_update', operation_spec().noneable(),
     'Operation for population update, which is called every time when the '
     'fitness of a proposed DNA is fed back to the algorithm. It passes the '
     'previous population with the newly added DNA as the input to this '
     'operation, whose return value will be used as the new population. '
     'If None, the population will be accumulated without a limit.'),
    ('multi_objective', pg.typing.Bool(default=False),
     'If True, the fitness is a tuple of float numbers as the metrics of '
     'multiple objectives. Otherwise the fitness is a float value as the '
     'metric for a single objective.')
], init_arg_list=[
    'reproduction', 'population_init', 'population_update'
])
class Evolution(pg.DNAGenerator):
  """An evolutionary algorithm based on compositional operations.

  Common evolutionary algorithms can be abstracted with 3 operations:

   * Select parent(s) from the population.
   * Recombine the DNA from the parents into a list of child DNA.
   * Mutate each child DNA.

  Plus 2 population related operations:

   * Initialize the population.
   * Update the population when the fitness is determined for the child DNA.

  All the operations above except population initialization can be described
  by the `Operation` interface, which transforms a list of DNA into another list
  of DNA. Moreover, `Operation` allows complex algorithms to be represented as
  a composition of elementary operations. For example, Regularized Evolution
  can be described as::

      pg.evolution.Evolution(
          op=(pg.evolution.selectors.Random(10)
              >> pg.evolution.selectors.Top(1)
              >> pg.evolution.mutators.Uniform()),
          population_init=(pg.generators.Random(), 50),
          population_update=pg.evolution.selectors.Last(50))
  """

  def _setup(self) -> None:
    """Setup the algorithm."""
    if isinstance(self.population_init, tuple):
      (self._init_population_generator,
       self._init_population_size) = self.population_init
    else:
      self._init_population_generator = self.population_init
      self._init_population_size = None

    self._init_population_generator.setup(self.dna_spec)
    self._reproduction = make_operation_compatible(self.reproduction)
    self._population_update = make_operation_compatible(
        self.population_update)

    # NOTE(daiyip): global state can be accessed as attributes, and new keys
    # can be inserted during operations.
    self._global_state = pg.geno.AttributeDict(num_generations=0)

    self._population_initialized = False
    self._population = []
    self._pending_proposals = collections.deque()

  @property
  def multi_objective(self) -> bool:
    """Returns True if fitness is a tuple of float numbers."""
    return self.sym_getattr('multi_objective')

  @property
  def population(self) -> List[pg.DNA]:
    """Returns current population."""
    return self._population

  @property
  def global_state(self) -> pg.geno.AttributeDict:
    """Returns the global state."""
    return self._global_state

  @property
  def num_generations(self) -> int:
    return self._global_state.num_generations

  def _propose(self) -> pg.DNA:
    """Implementation of DNA proposal."""
    if not self._pending_proposals:
      if self._population_initialized:
        # Propose new individuals using evolution.
        self._pending_proposals.extend(self._evolve())
      else:
        # Propose initial population.
        try:
          dna = self._init_population_generator.propose()
          set_proposal_id(dna, self.num_proposals + 1)
          set_generation_id(dna, self.num_generations + 1)
          _set_initial_population(dna, True)
          self._pending_proposals.append(dna)
        except StopIteration:
          self._population_initialized = True
          self._global_state.num_generations = 1
          self._pending_proposals.extend(self._evolve())
    return self._pending_proposals.popleft()

  def _evolve(self) -> List[pg.DNA]:
    """Performs a single round of evolution process."""
    # Step 1: Select parents from the population.
    current_step = self.num_proposals
    children = self._reproduction(
        self._population, global_state=self._global_state, step=current_step)

    for i, child in enumerate(children):
      # NOTE(daiyip): If a child's feedback sequence number exists, it's
      # an existing DNA from the population, in such case, we should clone
      # to avoid the existing population get polluted.
      if get_feedback_sequence_number(child) is not None:
        child = child.clone(deep=True)
        children[i] = child

      # Update the 1-based ID and generation information for the DNA.
      set_proposal_id(child, current_step + 1 + i)
      set_generation_id(child, self.num_generations + 1)
      _set_initial_population(child, False)

    # Update number of generations in global state.
    self._global_state.num_generations += 1
    return children

  def _feedback(
      self, dna: pg.DNA, reward: typing.Union[float, Tuple[float]]) -> None:
    """Feedback a DNA with its reward."""
    set_feedback_sequence_number(dna, self._num_feedbacks + 1)
    set_fitness(dna, reward)
    assert get_fitness(dna) is not None

    # Feedback generation-zero DNA to the population initializer.
    if is_initial_population(dna):
      self._init_population_generator.feedback(dna, reward)

    # We use `num_feedbacks` instead of `num_proposals` to determine
    # initial population size to avoid starting the evolution process
    # too early without enough feedbacks.
    # `self.num_feedbacks` will be incremented after _feedback returns.
    # Therefore, we compare it with initial population size - 1.
    if (not self._population_initialized
        and self._init_population_size is not None
        and self.num_feedbacks >= self._init_population_size - 1):
      self._population_initialized = True
      self._global_state.num_generations = 1

    # Update the population if needed.
    self._population.append(dna)
    if self._population_update:
      self._population = self._population_update(
          self._population,
          global_state=self._global_state,
          step=self.num_feedbacks)

  def recover(
      self,
      history: Iterable[
          Tuple[pg.DNA, typing.Union[None, float, Tuple[float]]]]
      ) -> None:
    """Recover states by replaying the proposal history."""
    # Recover the state of the population.
    init_population = []

    for dna, reward in history:
      self._num_proposals += 1
      dna.use_spec(self.dna_spec)
      if reward is not None:
        # NOTE(daiyip): There is a possibility that the client has provided the
        # reward to the controller, but the controller process restarted before
        # calling the `feedback` method. Such cases can be identified by
        # checking the `feedback_sequence_number` metadata.
        if get_feedback_sequence_number(dna) is None:
          self.feedback(dna, reward)
        else:
          assert get_fitness(dna) == reward, (dna, reward)
          self._population.append(dna)
          if self._population_update:
            self._population = self._population_update(
                self._population,
                global_state=self._global_state,
                step=self._num_feedbacks)
          self._num_feedbacks += 1
      if is_initial_population(dna):
        init_population.append((dna, reward))

      # Recover `self.num_generations`.
      generation_id = get_generation_id(dna)
      if generation_id > self.num_generations:
        self._global_state.num_generations = generation_id

    # Recover the state of the population initializer.
    if (self._init_population_size is not None
        and len(init_population) >= self._init_population_size):
      self._population_initialized = True
    self._init_population_generator.recover(init_population)


#
#  Implementation of compositional operations.
#


class Identity(Operation):
  """Returns the input itself."""

  def call(self, inputs: List[Any]) -> List[Any]:
    return inputs


@pg.members([
    ('fn', operation_spec(pg.typing.Any(), pg.typing.Any()),
     'A callable object that performs the operation.')
])
class Lambda(Operation):
  """A lambda operation."""

  def _on_bound(self):
    super()._on_bound()
    self._fn = make_operation_compatible(self.fn)

  def call(
      self,
      inputs: List[Any],
      global_state: pg.geno.AttributeDict,
      step: int = 0) -> List[Any]:
    return self._fn(inputs, global_state=global_state, step=step)


@pg.members([
    ('op', operation_spec(pg.typing.Any(), pg.typing.Any()),
     'Operation to repeat if the output is the same as the input.'),
    ('max_attempts', scalars.scalar_spec(pg.typing.Int(min_value=1)).noneable(),
     'Maximum attempts to make if the output is the same as the input.')
])
class UntilChange(Operation):
  """Repeat an opertion until its output is different from the input."""

  def _on_bound(self):
    super()._on_bound()
    self._op = make_operation_compatible(self.op)

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    max_attempts = scalars.scalar_value(self.max_attempts, step)
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
      output = self._op(inputs, global_state=global_state, step=step)
      if output != inputs:
        break
      attempts += 1
    return output


@pg.members([
    ('ops', pg.typing.List(pg.typing.Tuple([
        # An operation candidate.
        operation_spec(pg.typing.Any(), pg.typing.Any()),
        # The probability of applying the operation.
        scalars.scalar_spec(pg.typing.Float(min_value=0.0, max_value=1.0))])),
     'A list of (operation, probability) tuples, denoting the probability of '
     'applying each operation. '
     'Example: [(mutator1, 0.25), (mutator2, 0.5)] will '
     'result in applying mutator1 to 25% of the DNAs that are mtuated and '
     'mutator2 to 50% of the DNAs. The decision to apply each mutator is '
     'independent of the application of the others; in particular, both '
     'mutator1 and mutator2 will be applied to 12.5% of the examples. The '
     'probability can be a callable object that takes an integer step as input '
     'and returns a float value as the probability for the step.'),
    ('limit', scalars.scalar_spec(pg.typing.Int(min_value=0)).noneable(),
     'The maximum number of operations that are allowed to perform. '
     'If None, there is limit. This is useful when we want to control '
     'the max difference between the input and the output.'),
    ('seed', pg.typing.Int().noneable(), 'Random seed for choosing ops.')
])
class Choice(Operation):
  """Applying a list of operations with probabilities in a pipeline.

  Example::

    pg.evolution.Choice([
      (pg.evolution.mutators.Uniform() ** 3, 0.5)
      (pg.evolution.mutators.Swap(), 0.4)
    ], limit=1)

  The code above apply 3 pipelined uniform mutation with probability 0.5, and
  apply 1 swap between two multiple choices with probability 0.4. At most 1
  operation can be applied.
  """

  def _on_bound(self):
    super()._on_bound()
    self._random = random if self.seed is None else random.Random(self.seed)
    self._ops = [(make_operation_compatible(op), prob)
                 for op, prob in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    num_performed_ops = 0
    for op, prob in self._ops:
      prob = scalars.scalar_value(prob, step)
      if self._random.random() < prob:
        inputs = op(inputs, global_state=global_state, step=step)
        num_performed_ops += 1
        if self.limit is not None and num_performed_ops == self.limit:
          break
    return inputs


@pg.members([
    ('predicate', pg.typing.Callable(
        [pg.typing.List(pg.typing.Any())], returns=pg.typing.Bool()),
     'A callable object that takes the output from previous operation as '
     'input, with optional `global_step` and `step` arguments, to tell which '
     'branch should be enabled.'),
    ('true_op', operation_spec(pg.typing.Any(), pg.typing.Any()).noneable(),
     'An operation that will be applied when predicate returns True.'),
    ('false_op', operation_spec(pg.typing.Any(), pg.typing.Any()).noneable(),
     'An operation that will be applied when predicate returns False. '
     'If None, an empty operation will be applied.')
])
class Conditional(Operation):
  """Conditional operation.

  A conditional operation allows sub-operations to be applied only when a
  condition is met.

  Example::

    pg.evolution.Conditional(
      lambda x: len(x) > 10,
      true_op=Top(10))

  which is equivalent to::

    Top(10).if_true(lambda x: len(x) > 10)
  """

  def _on_bound(self):
    super()._on_bound()
    self._predicate = make_operation_compatible(self.predicate)
    self._true_op = make_operation_compatible(self.true_op)
    self._false_op = make_operation_compatible(self.false_op)

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    if self._predicate(inputs, global_state=global_state, step=step):
      branch = self._true_op
    else:
      branch = self._false_op
    if branch is not None:
      return branch(inputs, global_state=global_state, step=step)
    else:
      return inputs


@pg.members([
    ('ops', pg.typing.List(
        operation_spec(pg.typing.Any(), pg.typing.Any()), min_size=2),
     'Child operations.')
])
class Pipeline(Operation):
  """Chaining operations into a pipeline.

  A pipeline operation chains multiple operations by passing the output
  of previous operation to the next operations.

  Example::

    pg.evolution.Pipeline([
      pg.evolution.selectors.Random(10)
      pg.evolution.selectors.Top(1),
      pg.evolution.mutators.Uniform()
    ])

  which is equivalent to::

    (pg.evolution.selectors.Random(10)
      >> pg.evolution.selectors.Top(1)
      >> pg.evolution.mutators.Uniform())

  The code above returns a uniformly mutated DNA from the winner of 10
  randomly selected DNA from the input.
  """

  def _on_bound(self):
    super()._on_bound()
    self._ops = [make_operation_compatible(op) for op in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    for op in self._ops:
      inputs = op(inputs, global_state=global_state, step=step)
    return inputs


@pg.members([
    ('op', operation_spec(pg.typing.Any(), pg.typing.Any()),
     'Operation to pipeline multiple times.'),
    ('k', scalars.scalar_spec(pg.typing.Int()), 'Number of repeats.')
])
class Power(Operation):
  """Pipeline a repeated operation multiple times.

  Power(x, k) (or x ** k) is equivalent to x >> x >> ... >> x (k items).

  Example::

    pg.evolution.Power(pg.evolution.mutators.Uniform, 3)

  which is equivalent to::

    pg.evolution.mutators.Uniform() ** 3

  The code above creates a DNA with 3 mutated positions for each DNA in
  the input,
  """

  def _on_bound(self):
    super()._on_bound()
    self._op = make_operation_compatible(self.op)

  def call(self,
           inputs: List[pg.DNA],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[pg.DNA]:
    for _ in range(scalars.scalar_value(self.k, step)):
      inputs = self._op(inputs, global_state=global_state, step=step)
    return inputs


@pg.members([
    ('ops', pg.typing.List(
        operation_spec(pg.typing.Any(), pg.typing.Any()), min_size=2),
     'Child operations.')
])
class Concatenation(Operation):
  """Concatenating the operations' outputs based on the same input.

  A concatenate operation passes the input to all its child operations, and
  concatenate their outputs. Different from `Union`, the duplicated items will
  be kept.

  Example::

    pg.evolution.Concatenation([
      pg.evolution.selectors.Top(10),
      pg.evolution.selectors.First(10)
    ])

  which is equivalent to::

    pg.evolution.selectors.Top(10) + pg.evolution.selectors.First(10)

  The code above returns top 10 DNA concatenated with the first 10 DNAs. The
  result may contain duplicated items.
  """

  def _on_bound(self):
    super()._on_bound()
    self._ops = [make_operation_compatible(op) for op in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    results = []
    for op in self._ops:
      results.extend(op(inputs, global_state=global_state, step=step))
    return results


@pg.members([
    ('op', operation_spec(pg.typing.Any(), pg.typing.Any()),
     'Operation whose output will be sliced'),
    ('index', scalars.scalar_spec(
        pg.typing.Union([pg.typing.Int(), pg.typing.Object(slice)])),
     'An integer index/a slice object or a function that computes them '
     'based on a step.')
])
class Slice(Operation):
  """Slice the operation's output.

  A slice operation passes the input the child operation and slice its output
  based on the index.

  Example::

    Slice(pg.evolution.Identity(), slice(0, 4, 2))

  which is equivalent to::

    pg.evolution.Identity()[:4:2]
  """

  def _on_bound(self):
    super()._on_bound()
    self._op = make_operation_compatible(self.op)

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    index = scalars.scalar_value(self.index, step)
    output = self._op(inputs, global_state=global_state, step=step)[index]
    if not isinstance(output, list):
      output = [output]
    return output


@pg.members([
    ('op', operation_spec(pg.typing.Any(), pg.typing.Any()),
     'Operation to repeat.'),
    ('k', scalars.scalar_spec(pg.typing.Int()), 'Number of repeats.')
])
class Repeat(Operation):
  """Repeat an operation multiple times based on the same input.

  A repeat operation passes the input to the child operation for K times, and
  concatenate their outputs.

  Example::

    Repeat(pg.evolution.mutators.Uniform(), 3)

  which is equivalent to::

    pg.evolution.mutators.Uniform() * 3

  The code above creates 3 mutated DNA for each DNA from the input.

  It's important to differentiate ``*`` and ``**``. For instance,
  `Uniform() ** 3` creates one DNA (mutated on 3 positions) for each DNA
  from the input.
  """

  def _on_bound(self):
    super()._on_bound()
    self._op = make_operation_compatible(self.op)

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    results = []
    for _ in range(scalars.scalar_value(self.k, step)):
      results.extend(self.op(inputs, global_state=global_state, step=step))
    return results


@pg.members([
    ('ops', pg.typing.List(
        operation_spec(pg.typing.Any(), pg.typing.Any()), min_size=1),
     'Child operations.')
])
class Union(Operation):
  """Unioning operations' outputs based on the same input.

  A union operation passes the input to all its child operations, and
  union their output. The items in the output will follow the order of
  their presence in the input with duplicated items removed.

  Example::

    pg.evolution.Union([
      pg.evolution.selectors.Random(10)
      pg.evolution.selectors.Top(1),
    ])

  which is equivalent to::

    pg.evolution.selectors.Random(10) | pg.evolution.selectors.Top(1)

  The code above returns the top 1 DNA and a random DNA. If both are the same
  DNA, only one occurence will be returned.
  """

  def _on_bound(self):
    super()._on_bound()
    self._ops = [make_operation_compatible(op) for op in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    ids = set()
    results = []
    for op in self._ops:
      for dna in op(inputs, global_state=global_state, step=step):
        dna_id = id(dna)
        if dna_id not in ids:
          results.append(dna)
          ids.add(dna_id)
    return results


@pg.members([
    ('ops', pg.typing.List(
        operation_spec(pg.typing.Any(), pg.typing.Any()), min_size=2),
     'Child operations.')
])
class Intersection(Operation):
  """Intersecting operations's outputs based on the same input.

  An intersect operation pass the input to all its child operations, and
  intersect their output as the result. The items in the output will follow
  the order of their presences in the input with duplicated items removed.

  Example::

    pg.evolution.Intersection([
      pg.evolution.selectors.Last(10)
      pg.evolution.selectors.Top(1),
    ])

  which is equivalent to::

    pg.evolution.selectors.Last(10) & pg.evolution.selectors.Top(1)

  The code above returns top 1 DNA only when it's among the last 10 items of
  the input.
  """

  def _on_bound(self):
    super()._on_bound()
    self._ops = [make_operation_compatible(op) for op in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    id_count = {}
    for op in self._ops[1:]:
      for dna in op(inputs, global_state=global_state, step=step):
        dna_id = id(dna)
        if dna_id not in id_count:
          id_count[dna_id] = 0
        id_count[dna_id] += 1

    candidates = self._ops[0](inputs, global_state=global_state, step=step)
    n = len(self._ops) - 1
    return [dna for dna in candidates if id_count.get(id(dna), 0) == n]


@pg.members([
    ('ops', pg.typing.List(
        operation_spec(pg.typing.Any(), pg.typing.Any()), min_size=2),
     'Child operations.')
])
class Difference(Operation):
  """Compute the difference of operation outputs based on the same input.

  A difference operation passes the input to all its child selectors, and
  substract the output from the rest operations from the output from the
  first operation. The items in the output will follow the order of their
  presences in the input.

  Example::


    pg.evolution.Difference([
      pg.evolution.selectors.Top(10),
      pg.evolution.selectors.First(10)
    ])

  which is equivalent to::

    pg.evolution.selectors.Top(10) - pg.evolution.selectors.First(10)

  The code above returns top 10 DNA only when they are not among the first 10
  elements of the input.
  """

  def _on_bound(self):
    super()._on_bound()
    self._ops = [make_operation_compatible(op) for op in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    excluded_ids = set()
    for op in self._ops[1:]:
      for dna in op(inputs, global_state=global_state, step=step):
        excluded_ids.add(id(dna))
    results = []
    for dna in self._ops[0](inputs, global_state=global_state, step=step):
      if id(dna) not in excluded_ids:
        results.append(dna)
    return results


@pg.members([
    ('ops', pg.typing.List(
        operation_spec(pg.typing.Any(), pg.typing.Any()), min_size=2),
     'Child operations.')
])
class SymmetricDifference(Operation):
  """Compute the symmetric difference of operation outputs based on the input.

  A symmetric difference operation passes the input to all its child selectors,
  and include the items that only appear in a single operation output.
  The items in the output will follow the order of their presences in the input.

  Example::

    pg.evolution.SymmetricDifference([
      pg.evolution.selectors.Top(10),
      pg.evolution.selectors.First(10)
    ])

  which is equivalent to::

    pg.evolution.selectors.Top(10) ^ pg.evolution.selectors.First(10)

  The code above returns items that are either among the top 10 or among the
  first 10 but not both.
  """

  def _on_bound(self):
    super()._on_bound()
    self._ops = [make_operation_compatible(op) for op in self.ops]

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    output_set_map = {}
    all_items = []
    for i, op in enumerate(self._ops):
      for dna in op(inputs, global_state=global_state, step=step):
        all_items.append(dna)
        output_set = output_set_map.get(id(dna), None)
        if output_set is None:
          output_set = set()
          output_set_map[id(dna)] = output_set
        output_set.add(i)
    return [item for item in all_items if len(output_set_map[id(item)]) == 1]


@pg.members([
    ('op', operation_spec(pg.typing.Any()),
     'Operation to invert.'),
])
class Inversion(Operation):
  """Computing the inversion of operation's output against the input."""

  def _on_bound(self):
    super()._on_bound()
    self._invert_op = Difference([Identity(), self.op])

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    return self._invert_op(inputs, global_state=global_state, step=step)


@pg.members([
    ('op', operation_spec(pg.typing.Any()),
     'Operation to apply to each element.'),
])
class ElementWise(Operation):
  """Apply the operation on each input item and aggregate them."""

  def _on_bound(self):
    super()._on_bound()
    self._op = make_operation_compatible(self.op)

  def call(self,
           inputs: List[Any],
           global_state: pg.geno.AttributeDict,
           step: int = 0) -> List[Any]:
    results = []
    for elem in inputs:
      results.append(self._op(elem, global_state=global_state, step=step))
    return results


@pg.members([
    ('max_level', pg.typing.Int(min_value=1).noneable(),
     'To max level of depth to flatten. If None, a nested list of any depth '
     'will be converted to a flat list.')
])
class Flatten(Operation):
  """Flatten elements from the input."""

  def call(self, inputs: List[Any]) -> List[Any]:
    return self._flatten_list(inputs, 0, [])

  def _flatten_list(
      self, input_list: List[Any], level: int, output: List[Any]) -> List[Any]:
    if self.max_level is not None and level > self.max_level:
      output.append(input_list)
    else:
      for elem in input_list:
        if isinstance(elem, list):
          self._flatten_list(elem, level + 1, output)
        else:
          output.append(elem)
    return output


@pg.members([
    ('key', pg.typing.Str(), 'Key in the global state.'),
    ('default', pg.typing.List(pg.typing.Any()).noneable(),
     'The default value to return when key does not exist. '
     'If None, KeyError will be raised.')
])
class GlobalStateGetter(Operation):
  """Returns a key in the `global_state` as output."""

  def call(self, inputs: List[Any], global_state: pg.geno.AttributeDict):
    del inputs
    if self.default is None:
      return global_state[self.key]
    return global_state.get(self.key, self.default)


@pg.members([
    ('key', pg.typing.Str(), 'Key in the global state.'),
    ('value', pg.typing.Any().set_default((pg.MISSING_VALUE,)),
     'A constant value to set. By default the input will be used as the value '
     'to set the global state.')
])
class GlobalStateSetter(Operation):
  """Set the value for a key in `global_state` and return an empty list."""

  def call(self, inputs: List[Any], global_state: pg.geno.AttributeDict):
    value = self.value
    if value == (pg.MISSING_VALUE,):
      value = inputs
    global_state[self.key] = value
    # Return empty list as output.
    return []


def make_operation_compatible(callable_object):
  """Returns an `Operation` compatible callable from an user callable object."""
  if callable_object is None:
    return None
  elif isinstance(callable_object, Operation):
    return callable_object
  else:
    return pg.typing.CallableWithOptionalKeywordArgs(
        callable_object, ['global_state', 'step'])
