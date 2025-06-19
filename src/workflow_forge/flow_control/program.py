"""
Scope nodes are used to make the ZCP graph. They form a natural
nested hierarchy that captures the blocks we use to make up code.
Scope nodes upon running their exit routine should attach.
"""
import numpy as np
import warnings
import copy
from typing import Dict, Tuple, Any, Optional, Callable, List, Type

from src.workflow_forge.ZCP.builder import GraphBuilderNode
from ..parsing.config_parsing import Config
from ..resources import AbstractResource
from ..ZCP.nodes import ZCPNode, RZCPNode, SamplerFactoryFactory, SamplerFactory, GraphLoweringErrorFactory
from .tag_converter import TagConverter
from ..tokenizer_interface import TokenizerInterface
from .tools import Toolbox, Tool
from dataclasses import dataclass

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
    toolbox: Type['Toolbox']

def make_default_factories()->FCFactories:
    return FCFactories(
        while_context=WhileContext,
        condition_context=ConditionalContext,
        scope=Scope,
        program=Program,
        graph_builder=GraphBuilderNode,
        toolbox=Toolbox,
    )
factories = make_default_factories()


### Builder Contexts
#
# Builder contexts resolve around context managers for setting up,
# tearing down, and otherwise updating graph builder nodes
# to complete the project.

class AbstractScopeContext:
    """
    A ScopeContext promises to maintain, manage,
    fuse, and update the parent scope upon completion
    of the relevant underlying tasks
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
    """
    def __init__(self,
                 parent_scope: 'Scope',
                 intake_node: GraphBuilderNode,
                 sequence: RZCPNode
                 ):
        super().__init__(parent_scope, intake_node)
        self.sequence = sequence
        self.loop_branch: Optional[GraphBuilderNode] = None
        self.escape_branch: Optional[GraphBuilderNode] = None
    def __enter__(self)->'Scope':
        loop_branch, escape_branch = self.intake_node.fork(self.sequence)
        self.loop_branch = loop_branch
        self.escape_branch = escape_branch
        return self.parent_scope.fork(loop_branch)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop_branch.attach(self.parent_scope.builder)
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
    """
    def __init__(self,
                 factories: FCFactories,
                 config: Config,
                 builder: GraphBuilderNode,
                 head: RZCPNode,  # The head of the entire shebang.
                 program: 'Program',

                 # Required operational parameters
                 resources: Dict[str, AbstractResource],
                 sequences: Dict[str, ZCPNode],
                 tokenizer: TokenizerInterface,
                 tag_converter: TagConverter,
                 ):
        """
        Primary
        :param factories: The container for the factories that can make classes.
            Dependency injection at it's best and worst.
        :param config: The overall configuration. From the UDPL parser
        :param builder: The builder node in it's current status.
        :param head: The head node of the graph. Always a mock node.
        :param program: The program this is part of.

        Support
        :param resources:
            The available resources for the particular process under consideration. This is
            excluding custom dynamic resources.
        :param sequences: The sequences of ZCP nodes we can draw upon, and their names
        :param tokenizer: The tokenizer we have to draw upon
        :param tag_converter: The tail converter instance
        """
        self.factories = factories
        self.config = config
        self.head = head
        self.builder = builder
        self.program = program
        self.resources = resources
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tag_converter = tag_converter
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
                except TypeError as err:
                    raise ScopeException("Cound not convert python resource to text") from err
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

        # Get the ZCP chain head
        zcp_head = self.sequences[sequence_name]

        # Create callback factory that validates and captures resources
        def callback_factory(raw_text: str,
                             resource_specs: Dict[str, Dict[str, Any]],
                             error_callback: GraphLoweringErrorFactory
                             ) -> Callable[[], np.ndarray]:
            try:
                # Validate that all required resources exist
                for placeholder, spec in resource_specs.items():
                    resource_name = spec['name']
                    if resource_name not in resources:
                        raise ScopeException(f"Resource '{resource_name}' not found for placeholder '{placeholder}'")

                # Create the deferred construction callback
                def construction_callback():
                    # Resolve placeholders using resources
                    try:
                        resolved_values = {}
                        for placeholder, spec in resource_specs.items():
                            resource_name = spec['name']
                            arguments = spec.get('arguments')

                            resource = resources[resource_name]
                            if arguments is not None:
                                resolved_values[placeholder] = resource(**arguments)
                            else:
                                resolved_values[placeholder] = resource()

                        # Format the text with resolved values
                        resolved_text = raw_text.format(**resolved_values)

                        # Tokenize and return
                        return self.tokenizer.tokenize(resolved_text)
                    except Exception as err:
                        raise error_callback("Failed to run sampling process") from err

                return construction_callback
            except Exception as err:
                raise error_callback("Failed to successfully make the sampler callback") from err

        # Lower the ZCP chain to RZCP using our callback factory
        return zcp_head.lower(callback_factory, self.tokenizer, self.tag_converter)

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
                                    self.tokenizer,
                                    self.tag_converter
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
                tool: Tool,
                **extra_resources: Dict[str, Any]
                ):
        """
        Sets up last zone in sequence as a capture zone, in
        which all contents will be stored and fed to tool
        :param sequence_name: Sequence to load and add.
        :param tool: Tool to call back into
        :param extra_resources: Any extra resources sequence should be setup with
        """

        resources = self._fetch_resources(extra_resources)
        sequence = self._load_sequence(sequence_name, resources)
        tail = sequence.get_last_node()
        tail.output = True
        tail.tool_callback = tool
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

def make_placeholder_node(tag_converter: TagConverter) -> RZCPNode:
    """Create a placeholder node for graph construction."""
    return RZCPNode(
        zone_advance_token=0,  # Dummy token
        tags=tag_converter.tensorize([]),  # No tags
        timeout=0,  # No timeout
        construction_callback=lambda: np.array([]),  # Empty tokens
        input=False,
        output=False
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
                 tokenizer: TokenizerInterface,
                 tag_converter: TagConverter):
        """Initialize a new program."""
        self.factories = factories
        self.sequences = sequences
        self.resources = resources
        self.config = config
        self.tokenizer = tokenizer
        self.tag_converter = tag_converter

        # Create mock head node for graph construction
        self.head = make_placeholder_node(tag_converter)

        # Create initial builder and scope
        jump_tokens = np.array(tokenizer.tokenize(config.control_token))
        initial_builder = self.factories.graph_builder(jump_tokens, [self.head])

        self.scope = Scope(
            factories=self.factories,
            config=self.config,
            builder=initial_builder,
            head=self.head,
            program=self,
            resources=self.resources,
            sequences=self.sequences,
            tokenizer=self.tokenizer,
            tag_converter=self.tag_converter
        )

        # Program-level state
        self.toolboxes: List[Toolbox] = []
        self.extractions: Dict[str, np.ndarray] = {}
    ### Primary program responsibilites
    #
    # Extract configuration, toolbox manufactoring, and
    # other similer details.

    def new_toolbox(self,
                   input_buffer_size: int,
                   output_buffer_size: int
                   ) -> Toolbox:
        """
        Create a new toolbox that will be backed by
        an input buffer and output buffer of the appropriate size.
        :param input_buffer_size: The size of the buffer which can contain callback results
        :param output_buffer_size: The size of the buffer which can contain tokens to feed into callbacks
        :return: The new toolbox
        """
        toolbox = self.factories.toolbox(
            tokenizer=self.tokenizer.tokenize,
            detokenizer=self.tokenizer.detokenize,
            input_buffer_size=input_buffer_size,
            output_buffer_size=output_buffer_size
        )
        self.toolboxes.append(toolbox)
        return toolbox

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

        # Convert to boolean mask and store
        tag_mask = self.tag_converter.tensorize(tags)
        if name in self.extractions:
            raise ProgramException("Already specified an extract to that name")
        self.extractions[name] = tag_mask

    def merge(self,
              other_program: 'Program'
              ) -> None:
        """Merge another program's state into this program."""
        self.toolboxes.extend(other_program.toolboxes)
        self.extractions.update(other_program.extractions)

    # Compilation
    def compile(self,
                backend: str = "default"
                ) -> 'ControllerFactory':
        """Compile the program to a controller factory using specified backend."""
        # Get the final RZCP graph from the scope's builder
        final_graph = self.scope.builder.head.next_zone
        if final_graph is None:
            raise ProgramException("Cannot compile empty program - no sequences were added")

        raise NotImplementedError()
        return compile_program(
            graph=final_graph,
            toolboxes=self.toolboxes,
            extractions=self.extractions,
            backend=backend
        )

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
                tool: Tool,
                **extra_resources: Dict[str, Any]
                ):
        """
        Sets up last zone in sequence as a capture zone, in
        which all contents will be stored and fed to tool
        :param sequence_name: Sequence to load and add.
        :param tool: Tool to call back into
        :param extra_resources: Any extra resources sequence should be setup with
        """
        return self.scope.capture(sequence_name, tool, **extra_resources)

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








