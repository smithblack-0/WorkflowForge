"""
Scope nodes are used to make the zcp graph. They form a natural
nested hierarchy that captures the blocks we use to make up code.
Scope nodes upon running their exit routine should attach.

Generally, a GraphBuilderNode is maintained and extended as
we travel along, which ends up maintaining and extending the
actual graph as well. That graph builder is responsible for
tracking the forward connections as we go.

You should review the Builder folder in ZCP or the graph
builder documentation before editing the node construction
system. Of note, the ZCP nodes perform the actual lowering
process; this just invokes those functions and links
the RZCP nodes together.
"""
import copy
import textwrap
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List, Type, Callable

from workflow_forge.resources import AbstractResource, StaticStringResource
from workflow_forge.zcp.nodes import ZCPNode, RZCPNode
from workflow_forge.zcp.workflow import Workflow
from workflow_forge.frontend.parsing.config_parsing import Config
from workflow_forge.zcp.builder import GraphBuilderNode


class ScopeException(Exception):
    """Exception raised when something else fails."""
    pass

class ProgramException(Exception):
    """Exception raised when a program fails.."""
    pass

### I HATE tightly coupled systems
#
# The main classes here are all accessed, when referencing
# each other, through a central factory dataclass.
# This dataclass is usually it's default, but can be
# redone for testing or patching purposes.

@dataclass
class FCFactories:
    """
    The holder for the constructors the various
    classes used, which can decouple this
    complex system.
    """
    while_context: Type['WhileContext']
    condition_context: Type['ConditionalContext']
    scope: Type['Scope']
    program: Type['Program']
    graph_builder: Type['GraphBuilderNode']
    workflow: Type['Workflow']
    str_resource: Type['StaticStringResource']


def make_default_factories()->FCFactories:
    return FCFactories(
        while_context=WhileContext,
        condition_context=ConditionalContext,
        scope=Scope,
        program=Program,
        graph_builder=GraphBuilderNode,
        workflow=Workflow,
        str_resource=StaticStringResource,
    )

### Builder Contexts
#
# Builder contexts resolve around context managers for setting up,
# tearing down, and otherwise updating graph builder nodes
# to complete the project.

class AbstractScopeContext:
    """
    A ScopeContext promises to maintain, manage,
    fuse, and update the parent scope upon completion
    of the relevant underlying tasks.

    The parent scope contains a builder node,
    which will ultimately be connected
    back to to continue the graph chain.

    It is very important to keep in mind that
    builder nodes are functional. As such, we
    generally need to retrieve the most current
    node on a scope.
    """
    def update_parent(self, node: GraphBuilderNode):
        self.parent_scope.builder = node

    def __init__(self,
                 parent_scope: 'Scope',
                 intake_node: GraphBuilderNode
                 ):
        self.parent_scope = parent_scope
        self.intake_node = intake_node

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()


class ConditionalContext(AbstractScopeContext):
    """
    A context manager specifically designed for handling
    conditional if/else statements.
    
    Mechanically, this class is storing the next node
    of the sequence we need to attach our forward references
    onto, and then on enter invoking and thus forking the 
    intake node from the parent scope. This indeed attaches
    those forward references, and results in a forked set of
    GraphBuilder nodes. These nodes are then used to construct subcontexts, 
    which are then returned. 
    
    During exit, the scope, which always has the most up-to-date
    builders possible stored on them, are retrieved and their nodes
    merged. This can then be used to replace the builder on the parent
    node, completing the cycle.
    """
    def __init__(self,
                 parent_scope: 'Scope',
                 intake_node: GraphBuilderNode,
                 sequence: RZCPNode,
                 ):
        super().__init__(parent_scope, intake_node)
        self.sequence = sequence
        self.if_scope: Optional['Scope'] = None
        self.else_scope: Optional['Scope'] = None

    def __enter__(self)->Tuple['Scope', 'Scope']:
        if_branch, else_branch = self.intake_node.fork(self.sequence)
        self.if_scope = self.parent_scope.fork(if_branch)
        self.else_scope = self.parent_scope.fork(else_branch)
        return self.if_scope, self.else_scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        if_scope, else_scope = self.if_scope, self.else_scope
        final_builder = GraphBuilderNode.merge(if_scope.builder, else_scope.builder)
        self.parent_scope.replace_builder(final_builder)

class WhileContext(AbstractScopeContext):
    """
    Responsible for running the while loop contexts,
    which loop while the python system is not yet emitting
    its jump statement.
    
    As is usually the case, the manager begins by causing
    a fork. This fork is driven by the loop context response.
    We then make a subscope based on the nominal branch, corrosponding
     to not emitting a jump.

    It is important to keep in mind that intake_node will capture the
    loop control prompt here.
    """
    def __init__(self,
                 parent_scope: 'Scope',
                 intake_node: GraphBuilderNode,
                 sequence: RZCPNode
                 ):
        super().__init__(parent_scope, intake_node)
        self.sequence = sequence
        self.loop_scope: Optional['Scope'] = None
        self.loop_branch: Optional[GraphBuilderNode] = None
        self.escape_branch: Optional[GraphBuilderNode] = None
    def __enter__(self)->'Scope':
        loop_branch, escape_branch = self.intake_node.fork(self.sequence)
        self.escape_branch = escape_branch
        self.loop_scope = self.parent_scope.fork(loop_branch)
        return self.loop_scope
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Complete the loopback
        loop_branch = self.loop_scope.builder
        self.intake_node.attach(loop_branch)

        # The escape branch becomes the new parent scope.
        self.parent_scope.replace_builder(self.escape_branch)


### Scope functionality
#
# The scope the user interacts with, along with the
# main program class, are indeed defined and manipulated
# right here.

class Scope:
    """
    The scope class is the main user class for building
    flow control. It provides access to the various
    userspace methods that can be used to build a flow
    control graph and, later, compile it.

    This is the primary control class, and indeed
    program only calls into it.

    """
    def __init__(self,
                 factories: FCFactories,
                 config: Config,
                 builder: GraphBuilderNode,
                 head: RZCPNode,  # The head of the entire shebang.
                 program: 'Program',
                 resources: Dict[str, AbstractResource],
                 sequences: Dict[str, ZCPNode],
                 ):
        """
        Primary
        :param factories: The container for the factories that can make classes.
            Dependency injection at it's best and worst.
        :param config: The overall configuration. From the UDPL parser
        :param builder: The builder node in it's current status.
        :param head: The head node of the graph. Always a mock node.
        :param program: The program this is part of.
        :param resources:
            The available resources for the particular process under consideration. This is
            excluding custom dynamic resources.
        :param sequences: The sequences of zcp nodes we can draw upon, and their names
        """
        self.factories = factories
        self.config = config
        self.head = head
        self.builder = builder
        self.program = program
        self.resources = resources
        self.sequences = sequences
        self.builder = builder

    def _fetch_resources(self,
                         resources: Dict[str, Any]
                         )->Dict[str, AbstractResource]:
        """
        Resources may be added during SFCS time, and will then be integrated
        into the interally defined dictionary after processing
        :param resources: The extra resources that may have been defined
        :return: The resources available during this step.
        """
        final_resources = self.resources.copy()
        for name, resource in resources.items():
            if name in final_resources:
                warnings.warn("Resource of name {name} is being overwritten. Make sure this is intended")
            if not isinstance(resource, AbstractResource):
                try:
                    resource = str(resource)
                    resource = self.factories.str_resource(resource)
                except TypeError as err:
                    raise ScopeException("Cound not convert python resource to text resource") from err
            final_resources[name] = resource
        return final_resources

    def _load_sequence(self,
                       sequence_name: str,
                       resources: Dict[str, AbstractResource]
                       ) -> RZCPNode:
        """
        Loads a sequence name, converts everything we can.
        :param sequence_name: Loads a particular sequence name from the stored sequences
        :param resources: The resolved resources to use for this sequence
        :return: The loaded RZCP nodes, sans of course any flow control integration yet.
        """
        if sequence_name not in self.sequences:
            raise ScopeException(f"Sequence '{sequence_name}' not found in available sequences")

        zcp_head = self.sequences[sequence_name]
        return zcp_head.lower(resources, self.config)

    def replace_builder(self,
                        builder: GraphBuilderNode
                        ):
        """Replaces the graph builder node, used in context management. Not userspace function"""
        self.builder = builder

    def fork(self,
             builder: GraphBuilderNode
             )-> 'Scope':
        """Creates a copy of the scope, with the new builder inserted instead. not userspace function"""
        return self.factories.scope(self.factories,
                                    self.config,
                                    builder,
                                    self.head,
                                    self.program,
                                    self.resources,
                                    self.sequences,
                                    )

    ### Commands.

    def run(self,
            sequence_name: str,
            **extra_resources: Dict[str, Any]
            ):
        """
        Attaches a run statement using the indicated sequence to the graph.
        :param sequence_name: The sequence to load and add.
        :param extra_resources: Any extra resources to process.
        """
        resources = self._fetch_resources(extra_resources)
        sequence = self._load_sequence(sequence_name, resources)
        self.builder = self.builder.extend(sequence)

    def when(self,
             sequence_name: str,
             **extra_resources: Dict[str, Any]
             )->ConditionalContext:
        """
        If-based flow control execution. Must be provided by the decision
        sequence for flow jump compiling. Keep in mind the if branch goes off
        by default and the model must emit a flow control token to prevent this.
        :param sequence_name: The name to use for the flow control question section.
        :param extra_resources: Any extra resources to process.
        :return: The if, else scope statements.
        """
        resources = self._fetch_resources(extra_resources)
        sequence = self._load_sequence(sequence_name, resources)
        return self.factories.condition_context(self, self.builder, sequence)

    def loop(self,
             sequence_name: str,
             **extra_resources: Dict[str, Any]
             )->WhileContext:
        """
        Sets up a loop context, which can be used to perform a
        loop scope. Loop scopes keep looping unless the jump token is
        issued.
        :param sequence_name: The sequence to load to run the flow control entry.
        :param extra_resources: Any extra resources to run flow control with.
        :return: The loops scope. When it exists the loop finishes building.
        """
        resources = self._fetch_resources(extra_resources)
        sequence = self._load_sequence(sequence_name, resources)
        return self.factories.while_context(self, self.builder, sequence)

    def subroutine(self,
                   subroutine: 'Program'
                   ):
        """
        Merges another program containing a graph
        or subroutine with the existing program, and
        extends the graph from this point forward
        to run it.
        """
        head = subroutine.head
        sequence = copy.deepcopy(head)
        self.builder = self.builder.extend(sequence)
        self.program.merge(subroutine)

    def capture(self,
                sequence_name: str,
                tool_name: str,
                **extra_resources: Dict[str, Any]
                ):
        """
        Sets up last zone in sequence as a capture zone, in
        which all contents will be stored and fed to tool
        :param sequence_name: Sequence to load and add.
        :param tool_name: Tool to call back into
        :param extra_resources: Any extra resources sequence should be setup with
        """
        if tool_name not in self.config.tools:
            raise ScopeException(f"Tool '{tool_name}' not found in available tools")

        resources = self._fetch_resources(extra_resources)
        sequence = self._load_sequence(sequence_name, resources)
        tail = sequence.get_last_node()
        tail.output = True
        tail.tool_name = tool_name
        self.builder = self.builder.extend(sequence)

    def feed(self,
             sequence_name: str,
             **extra_resources: Dict[str, Any]
             ):
        """
        Sets up last zone in sequence as a feed zone, which will feed
        in captured tokens from
        :param sequence_name: Sequence to load and add.
        :param extra_resources: Any extra resources sequence should be setup with
        """
        resources = self._fetch_resources(extra_resources)
        sequence = self._load_sequence(sequence_name, resources)
        tail = sequence.get_last_node()
        tail.input = True
        self.builder = self.builder.extend(sequence)


### Program
#
#

def make_placeholder_node() -> RZCPNode:
    """
    The placeholder node exists at the start of the graph, and will
    later be removed after the graph resolves.
    """
    return RZCPNode(
        sequence="Placeholder",
        block = 0,
        zone_advance_str="",
        tags=[],
        timeout = 0,
        sampling_callback=lambda _: "",
        escape_strs= ("", ""),
    )



class Program:
    """
    Main program class for building and compiling AI workflows.

    Acts as both a program orchestrator and provides scope-like interface
    for building workflows with flow control.
    """

    def __init__(self,
                 factories: FCFactories,
                 sequences: Dict[str, ZCPNode],
                 resources: Dict[str, AbstractResource],
                 config: Config,
                 ):
        """Initialize a new program."""
        self.factories = factories
        self.sequences = sequences
        self.resources = resources
        self.config = config

        # Create mock head node for graph construction
        self.head = make_placeholder_node()
        initial_builder = self.factories.graph_builder(self.config.control_pattern,
                                                       [self.head])

        self.scope = self.factories.scope(
            factories=self.factories,
            config=self.config,
            builder=initial_builder,
            head=self.head,
            program=self,
            resources=self.resources,
            sequences=self.sequences,
        )

        # Program-level state
        self.extractions: Dict[str, List[str]] = {}

    def extract(self,
                name: str,
                tags: List[str]
                ):
        """
        Configures the extraction mechanism to, when invoked, find all tokens
        tagged with the union of the tags, extract that, and concatenate them
        together, then detokenize. A dictionary with that entry, under name,
        will be returned. Other tags with different names can be defined as
        well. It should be kept in mind that this will result in a list of
        these dictionaries, one per batch.
        :param name: The name to return the sequence under
        :param tags: The tags to union together and then extract.
        """
        # Validate tags exist in config
        for tag in tags:
            if tag not in self.config.valid_tags:
                raise ProgramException(f"Invalid tag '{tag}' not in config.valid_tags")
        if name in self.extractions:
            raise ProgramException("Already specified an extract to that name")
        self.extractions[name] = tags

    def merge(self,
              other_program: 'Program'
              ) -> None:
        """Merge another program's state into this program."""
        self.extractions.update(other_program.extractions)

    # Compilation

    def _convert_resources(self, resources: Dict[str, Any])-> Dict[str, AbstractResource]:
        """
        Converts anything passed in into string abstract resources... if possible.
        :param resources: The resources, usually from python directly
        :return: Converted resources
        """
        try:
            output = {}
            for name, resource in resources.items():
                resource = str(resource)
                resource = self.factories.str_resource(resource)
                output[name] = resource
            return output
        except Exception as e:
            msg = f"""
            Conversion of python into resources failed
            This usually means you are passing something that 
            cannot be converted into a string.
            """
            msg = textwrap.dedent(msg)
            raise ProgramException(msg) from e


    def compile(self)->Callable[[Dict[str, AbstractResource]], Workflow]:
        """
        Compile this to a usable specification, that can then be dispatched by a client
        or lowered to tensors.
        :param self:
        :return:
        """
        # Get the final RZCP graph from the scope's builder
        final_graph = self.scope.head.next_zone
        if final_graph is None:
            raise ProgramException("Cannot compile empty program - no sequences were added")

        # Define and return the workflow factory
        def workflow_factory(argument_resources: Optional[Dict[str, Any]] = None)->Workflow:
            if argument_resources is None:
                argument_resources = {}
            argument_resources = self._convert_resources(argument_resources)
            szcp = final_graph.lower(argument_resources)
            return self.factories.workflow(
                     config=self.config,
                     nodes=szcp,
                     extractions=self.extractions
                     )
        return workflow_factory

    ### Scope Passthroughs.
    #
    # The program needs to also act like a scope. This
    # makes sure this happens

    def run(self,
            sequence_name: str,
            **extra_resources: Dict[str, Any]
            ):
        """
        Attaches a run statement using the indicated sequence to the graph.
        :param sequence_name: The sequence to load and add.
        :param extra_resources: Any extra resources to process.
        """
        self.scope.run(sequence_name, **extra_resources)

    def when(self,
             sequence_name: str,
             **extra_resources: Dict[str, Any]
             )->ConditionalContext:
        """
        If-based flow control execution. Must be provided by the decision
        sequence for flow jump compiling. Keep in mind the if branch goes off
        by default and the model must emit a flow control token to prevent this.
        :param sequence_name: The name to use for the flow control question section.
        :param extra_resources: Any extra resources to process.
        :return: The if, else scope statements.
        """
        return self.scope.when(sequence_name, **extra_resources)

    def loop(self,
             sequence_name: str,
             **extra_resources: Dict[str, Any]
             )->WhileContext:
        """
        Sets up a loop context, which can be used to perform a
        loop scope. Loop scopes keep looping unless the jump token is
        issued.
        :param sequence_name: The sequence to load to run the flow control entry.
        :param extra_resources: Any extra resources to run flow control with.
        :return: The loops scope. When it exists the loop finishes building.
        """
        return self.scope.loop(sequence_name, **extra_resources)

    def subroutine(self,
                   subroutine: 'Program'
                   ):
        """
        Merges another program containing a graph
        or subroutine with the existing program, and
        extends the graph from this point forward
        to run it.
        """
        return self.scope.subroutine(subroutine)

    def capture(self,
                sequence_name: str,
                tool_name: str,
                **extra_resources: Dict[str, Any]
                ):
        """
        Sets up last zone in sequence as a capture zone, in
        which all contents will be stored and fed to tool
        :param sequence_name: Sequence to load and add.
        :param tool_name: Tool to call back into
        :param extra_resources: Any extra resources sequence should be setup with
        """
        return self.scope.capture(sequence_name, tool_name, **extra_resources)

    def feed(self,
             sequence_name: str,
             **extra_resources: Dict[str, Any]
             ):
        """
        Sets up last zone in sequence as a feed zone, which will feed
        in captured tokens from
        :param sequence_name: Sequence to load and add.
        :param extra_resources: Any extra resources sequence should be setup with
        """
        return self.scope.feed(sequence_name, **extra_resources)

def new_program(
        sequences: Dict[str, ZCPNode],
        resources: Dict[str, AbstractResource],
        config: Config,
    )->Program:
    """
    Create a new program to make a workflow
    with.

    :param sequences: The sequences from UDPL parsing
    :param resources: Any resources we wish to use
    :param config: The config
    :return: The program
    """
    factories = make_default_factories()
    return Program(factories, sequences, resources, config)



