"""
Scope nodes are used to make the ZCP graph. They form a natural
nested hierarchy that captures the blocks we use to make up code.
Scope nodes upon running their exit routine should attach.
"""
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from ..parsing.config_parsing import Config
from ..resources import AbstractResource, StaticStringResource
from ..ZCP.nodes import ZCPNode, RZCPNode
from .tag_converter import TagConverter


class ScopeException(Exception):
    """Exception raised when something else fails."""
    pass

class GraphBuilderException(Exception):
    """Exception raised when graph building fails"""
    pass

class GraphBuilderNode:
    """
    Graph builder node for constructing RZCP flow control graphs with
    forward reference resolution. It delays attachment of the most recent
    set of nodes until the forward resolution is found, then catches the new
    node set.

    Manages the construction of RZCP node chains while handling forward references elegantly.
    Each GraphBuilderNode maintains a list of RZCP node tails that need to be wired to
    whatever sequence comes next. New graph builder nodes are returned functionally, but
    note the underlying constructed graph is not functional.

    Attributes:
        forward_refs: List of RZCP node tails waiting to be wired to the next sequence
    """

    def __init__(self,
                 jump_token: int,
                 nominal_refs: Optional[List[RZCPNode]] = None,
                 flow_control_refs: Optional[List[RZCPNode]] = None,
                 ):
        """
        Initialize a GraphBuilderNode.

        Args:
            nominal_refs: No more or less than the forward references that need to be wired
                up from the next_zone of the reference to the resolved forward reference
            flow_control_refs: No more or less than the forward references that need to be wired up
                up from their jump_zone references to the resolved forward reference
        """
        self.nominal_refs = nominal_refs or []
        self.flow_control_refs = flow_control_refs or []
        self.head: Optional[RZCPNode] = None
        self.jump_token = jump_token

    def _resolve_forward_references(self, sequence):
        """Internal utility to wire up all the forward references"""
        # Wire all pending forward refs to this sequence head
        for tail in self.nominal_refs:
            if tail.next_zone is not None:
                raise GraphBuilderException("Attempted to replace graph nominal link")
            tail.next_zone = sequence

        for tail in self.flow_control_refs:
            if tail.jump_zone is not None:
                raise GraphBuilderException("Attempted to replace graph flow control link")
            tail.jump_zone = sequence
            tail.jump_token = self.jump_token

    def extend(self, sequence: RZCPNode) -> 'GraphBuilderNode':
        """
        Extend the graph with a normal sequence, resolving forward references.

        Wires all pending forward references to the head of the provided sequence,
        then returns a new GraphBuilderNode with this sequence's tail as the
        new forward reference.

        Args:
            sequence: RZCP node chain to attach to the graph

        Returns:
            New GraphBuilderNode with this sequence's tail as forward reference
        """
        self._resolve_forward_references(sequence)
        sequence_tail = sequence.get_last_node()
        self.head = sequence
        return GraphBuilderNode(self.jump_token, [sequence_tail])

    def fork(self, sequence: RZCPNode) -> Tuple['GraphBuilderNode', 'GraphBuilderNode']:
        """
        Create a fork in the graph for flow control branching.

        Wires all pending forward references to the provided sequence, then creates
        two new GraphBuilderNodes representing the normal path (next_zone) and
        the jump path (jump_zone) from this sequence.

        Args:
            sequence: RZCP node chain that implements the flow control logic

        Returns:
            Tuple of (main_path_builder, jump_path_builder) where:
            - main_path_builder continues the normal next_zone path
            - jump_path_builder represents the jump_zone path
        """
        self._resolve_forward_references(sequence)

        # Get new tail

        sequence_tail = sequence.get_last_node()

        # Create builders for both paths
        main_path = GraphBuilderNode(self.jump_token, [sequence_tail], None)
        jump_path = GraphBuilderNode(self.jump_token, None, [sequence_tail])  # Jump wiring handled separately

        return main_path, jump_path

    @classmethod
    def merge(cls, *builders: 'GraphBuilderNode') -> 'GraphBuilderNode':
        """
        Merge multiple GraphBuilderNodes into a single convergence point.

        Collects all forward references from the provided builders and creates
        a new GraphBuilderNode that will wire all of them to whatever sequence
        comes next.

        Args:
            *builders: GraphBuilderNode instances to merge

        Returns:
            New GraphBuilderNode containing all forward references from input builders
        """
        nominal_refs = []
        flow_control_refs = []
        if len(builders) == 0:
            raise GraphBuilderException("No builders provided")
        for builder in builders:
            nominal_refs.extend(builder.nominal_refs)
            flow_control_refs.extend(builder.flow_control_refs)

        return cls(builders[0].jump_token, nominal_refs, flow_control_refs)

    def attach(self, node: 'GraphBuilderNode'):
        """
        Attach a node to this graph builder.
        :param node: The target node to attach myself to
        """

        if node.head is None:
            raise GraphBuilderException("Attempted to attach to a node that was never extended")
        self._resolve_forward_references(sequence=node.head)


### Builder Contexts
#
# Builder contexts resolve around context managers for setting up,
# tearing down, and otherwise updating graph builder nodes
# to complete the project.
#
# They

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
                 sequence: RZCPNode
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

class



class Scope:
    """
    The scope class is the main user class for building
    flow control. It provides access to the various
    userspace methods that can be used to build a flow
    control graph and, later, compile it.
    """
    def __init__(self,
                 # Scope resolution locations
                 config: Config,
                 builder: GraphBuilderNode,

                 # Required operational parameters
                 resources: Dict[str, AbstractResource],
                 sequences: Dict[str, ZCPNode],
                 tokenizer: Callable[[str], np.ndarray],
                 tag_converter: TagConverter,

                 # Parent! We merge back with this once
                 # our context is done.


                 ):
        """
        :param Config: The overall configuration. From the UDPL parser
        :param resources:
            The available resources for the particular process under consideration. This is
            excluding custom dynamic resources.
        :param sequences: The sequences of ZCP nodes we can draw upon, and their names
        :param tokenizer: The tokenizer we have to draw upon
        :param tag_converter: The tail converter instance
        :param tail: The node to continue from.
        """
        self.config = config
        self.resources = resources
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.tag_converter = tag_converter
        self.builder = builder

    def __enter__(self):
        self.head = make_placeholder_node()
        self.tail = self.head
        return self

    def __exit__(self,
                 exc_type,
                 exc_val,
                 exc_tb
                 ):
        sequence = self.head.next_zone



    def process_resources(self, resources: Dict[str, Any])->Dict[str, AbstractResource]:
        """
        Resources may be added during SCFS time, and will then be integrated
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
                    raise ScopeRunException("Cound not convert python resource to text") from err
            final_resources[name] = resource
        return final_resources

    def load_sequence(self, sequence_name: str, resources: Dict[str, AbstractResource]) -> RZCPNode:
        """
        Loads a sequence name, converts everything we can.
        :param sequence_name: Loads a particular sequence name from the stored sequences
        :param resources: The resolved resources to use for this sequence
        :return: The loaded RZCP nodes, sans of course any flow control integration yet.
        """
        if sequence_name not in self.sequences:
            raise ScopeRunException(f"Sequence '{sequence_name}' not found in available sequences")

        # Get the ZCP chain head
        zcp_head = self.sequences[sequence_name]

        # Create callback factory that validates and captures resources
        def callback_factory(raw_text: str,
                             resource_specs: Dict[str, Dict[str, Any]]
                             ) -> Callable[[], np.ndarray]:

            # Validate that all required resources exist
            for placeholder, spec in resource_specs.items():
                resource_name = spec['name']
                if resource_name not in resources:
                    raise ScopeRunException(f"Resource '{resource_name}' not found for placeholder '{placeholder}'")

            # Create the deferred construction callback
            def construction_callback():
                # Resolve placeholders using resources
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
                return self.tokenizer(resolved_text)

            return construction_callback

        # Lower the ZCP chain to RZCP using our callback factory
        return zcp_head.lower(callback_factory, self.tokenizer, self.tag_converter)

    def run(self, sequence_name: str, **extra_resources: Dict[str, Any]):
        """
        Attaches a run statement using the indicated sequence to the graph.
        :param sequence_name: The sequence to load and add.
        :param extra_resources: Any extra resources to process.
        """
        resources = self.process_resources(extra_resources)
        sequence = self.load_sequence(sequence_name, resources)
        if self.tail is not None:
            self.tail.next_zone = sequence
            self.tail = sequence.get_last_node()
        else:
            self.tail = sequence
    def if_(self, sequence_name: str, **extra_resources: Dict[str, Any])->Tuple['Scope', 'Scope']:
        """
        If-based flow control execution. Must be provided by the decision
        sequence for flow jump compiling.
        :param sequence_name: The name to use for the flow control question section
        :param extra_resources: Any extra resources to process
        :return: The if, else scope statements
        """


    def loop(self, sequence_name: str, **extra_resources: Dict[str, Any])->'Scope':
        """
        Sets up a loop context, which can be used to perform a
        loop scope. Loop scopes keep looping unless the jump is issued
        :param sequence_name: The sequence to load to run the flow control entry
        :param extra_resources: Any extra resources to run flow control with
        :return: The loops scope. When it exists the loop finishes building
        """
        resources = self.process_resources(extra_resources)





