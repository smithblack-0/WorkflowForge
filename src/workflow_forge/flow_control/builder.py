"""
The builder is responsible for committing and monitoring the right
build actions given the current state. The
"""
from typing import Optional, List, Tuple
from src.workflow_forge.ZCP.nodes import RZCPNode

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

    It also, later, captures whatever the forward references were wired to under the
    head field

    Attributes:
        head: the captured forward reference head.
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
        self.head = sequence

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
