"""
The builder is responsible for committing and monitoring the right
build actions given the current state. The
"""
import numpy as np
from typing import Optional, List, Tuple
from src.workflow_forge.ZCP.nodes import RZCPNode

class GraphBuilderException(Exception):
    """Exception raised when graph building fails"""
    pass

class GraphBuilderNode:
    """
    Graph builder for RZCP flow control compilation with forward reference
    resolution.

    When building a flow control graph, we encounter situations where we know
    certain edges must connect to some future vertex, but that vertex hasn't
    been created yet. This is the mathematical concept of "dangling edges" -
    edges that have a source but no destination. This class exists to resolve
    this situation.

    In graph theory terms: this represents a set of directed edges E that will be terminated
    at V, and then later stores V itself and connects the edges. Each edge can have
    one of two 'colors' determined by the color site the edge is emitted from on the
    originating node.


    A GraphBuilderNode is constructed with:
    - A set of dangling edges waiting for their target vertex.
    - Speaking formally these edges have one of two 'colors' which are determined
      by the emission site at the source node.

    After construction, it can be manipulated and eventually one of these
    manipulations will resolve the question of what V is. When this happens:
    1. The connections are resolved. Originating at the right 'color', which technically is
       one of the nominal advance or jump advance links
    2. Store the node.


    Methods:
    - extend(): Resolves all dangling edges to a new vertex V, returns new builder
      with edges from V as the new dangling set
    - fork(): Creates two builders representing different edge types from the same vertex
    - merge(): Takes the union of dangling edge sets from multiple builders

    The key insight: by treating this as an edge-centric problem rather than
    vertex-centric, we can build graphs incrementally without knowing all vertices
    in advance.

    Attributes:
        nominal_refs: Set of edges of type "sequential flow" waiting for target vertex
        flow_control_refs: Set of edges of type "conditional jump" waiting for target vertex
        head: The target vertex (None until edges are resolved)
        jump_tokens: The tokens representing the "execute jump" instruction
    """

    def __init__(self,
                 jump_tokens: np.ndarray,
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
        self.jump_tokens = jump_tokens

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
            tail.jump_tokens = self.jump_tokens
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
        return GraphBuilderNode(self.jump_tokens, [sequence_tail])

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
        main_path = GraphBuilderNode(self.jump_tokens, [sequence_tail], None)
        jump_path = GraphBuilderNode(self.jump_tokens, None, [sequence_tail])  # Jump wiring handled separately

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

        return cls(builders[0].jump_tokens, nominal_refs, flow_control_refs)

    def attach(self, node: 'GraphBuilderNode'):
        """
        Attach a node to this graph builder.
        :param node: The target node to attach myself to
        """

        if node.head is None:
            raise GraphBuilderException("Attempted to attach to a node that was never extended")
        self._resolve_forward_references(sequence=node.head)
