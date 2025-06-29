"""
Zone Control Protocol (zcp) Node Definitions

This module defines the data structures for zcp nodes, which represent zones of text
in the intermediate representation used by Workflow Forge before compilation to
the Token Triggered Finite Automata (TTFA) backend.

zcp supports Directed Cyclic IO Graphs (DCG-IO), which are directed graphs that:
- Allow cycles (unlike DAGs)
- Have exactly one source and one sink vertex
- Guarantee all vertices are reachable from source and can reach sink
- Maintain computational tractability for workflow analysis
"""
import numpy as np
import textwrap
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any, Tuple
from src.workflow_forge.zcp.tag_converter import TagConverter
from ..tokenizer_interface import TokenizerInterface
from ..resources import AbstractResource
from ..parsing.config_parsing import Config
try:
    # Most users do not need this. But if you are
    # trying to visualize the nodes for debugging your
    # workflow, this is involved in showing what it
    # compiles to.
    import networkx as nx
    import igraph as ig
    from .rendering import GraphNode, GraphData, create_plotly_graph
    VISUALIZATION_ENABLED = True
except ImportError as vis_err:
    VISUALIZATION_ENABLED = False
    reason = vis_err

## Setup the error banks

class GraphLoweringError(Exception):
    """
    This flavor of exception is invoked only when a graph
    lowering stage ends up malformed or does not perform
    correctly. This can be due to a variety of things,
    """
    def __init__(self,
                 message: str,
                 block: int,
                 sequence: str
                 ):
        """
        :param message: The main message to display
        :param block: The block it occurred associated with
        :param sequence: The sequence it is associated with
        """

        msg = f"""\
        An issue occurred while attempting to lower the graph
        This occurred in sequence "{sequence}", in block "{block}"
        The error is:
        """
        msg= textwrap.dedent(msg) + "\n" + message
        super().__init__(msg)
        self.sequence = sequence
        self.block = block


class GraphError(Exception):
    """
    This flavor of exception is invoked only when a graph
    lowering stage ends up malformed or does not perform
    correctly. This can be due to a variety of things,
    """
    def __init__(self,
                 message: str,
                 block: int,
                 sequence: str
                 ):
        """
        :param message: The main message to display
        :param block: The block it occurred associated with
        :param sequence: The sequence it is associated with
        """

        msg = f"""\
        An issue occurred while manipulating the graph
        This occurred in sequence "{sequence}", in block "{block}"
        The error is:
        """
        msg= textwrap.dedent(msg) + "\n" + message
        super().__init__(msg)
        self.sequence = sequence
        self.block = block


@dataclass
class ZCPNode:
    """
    Simple Zone Control Protocol node from UDPL parsing.

    Represents a basic linked list of prompt zones with resource metadata.
    No flow control - just what UDPL can express.

    Attributes:
        sequence: Name of UDPL sequence this zone belongs to
        block: Block number within the sequence
        construction_callback: When given a mapping of placeholders to their replacements, returns text.
        resource_specs: Placeholder → resource mapping (unresolved)
        raw_text: Template text with {placeholder} syntax
        zone_advance_str: String that triggers advancement to next zone
        tags: String tags for extraction
        timeout: Maximum tokens to generate before forcing advancement
        next_zone: Next zone in the linked list
    """
    sequence: str
    block: int
    construction_callback: Callable[[Dict[str, AbstractResource]], str]
    resource_specs: Dict[str, Dict[str, Any]]
    raw_text: str
    zone_advance_str: str
    tags: List[str]
    timeout: int
    next_zone: Optional['ZCPNode'] = None

    def get_last_node(self) -> 'ZCPNode':
        """Follow the linked list to get the tail node."""
        node = self
        while node.next_zone is not None:
            node = node.next_zone
        return node

    def _make_sampling_factory(self,
                               resources: Dict[str, AbstractResource]
                               ) -> Callable[[Dict[str, AbstractResource]], str]:
        """
        Factory for making the main sampling factory, which invokes the construction callback.

        Args:
            resources: The resources dictionary

        Returns:
            A callback which runs a sampling call and returns a string.
        """

        def sample(dynamic_resources: Dict[str, AbstractResource]) -> str:
            """
            Main sampling function, draws from the resources, fills
            in placeholders, gets text, returns the resolved string.
            Args:
                dynamic_resources: Runtime resources dictionary.
            Returns:
                The resolved text string
            """
            final_resource = {**dynamic_resources, **resources}
            for placeholder, spec in self.resource_specs.items():
                resource_name = spec["name"]
                if resource_name not in final_resource:
                    if spec["type"] == "argument":
                        msg = f"""
                        It appears an argument resource of name '{resource_name}' was
                        not provided when invoking the factory. This is a sign you
                        likely forgot to provide it. This concerns placeholder
                        of name '{placeholder}'
                        """
                        msg = textwrap.dedent(msg)
                        raise GraphError(message=msg, block=self.block, sequence=self.sequence)
                    else:
                        msg = f"""
                        An impossible condition was reached. Resource named '{resource_name}' of 
                        type '{spec["type"]}' is not available on factory invocation. However, this
                        should have been caught earlier. Please contact the maintainer.
                        """
                        msg = textwrap.dedent(msg)
                        raise GraphError(message=msg, block=self.block, sequence=self.sequence)

            try:
                text = self.construction_callback(final_resource)
                return text
            except Exception as err:
                msg = f"""
                An issue was encountered when invoking resource. This is 
                despite the fact the resources were resolved. This is usually
                a sign that you are not providing arguments in the right pattern,
                but may indicate a malformed resource as well.
                """
                msg = textwrap.dedent(msg)
                raise GraphError(message=msg, block=self.block, sequence=self.sequence) from err

        return sample

    def _lower_node(self,
                    resources: Dict[str, AbstractResource],
                    config: Config
                    ) -> 'RZCPNode':
        """
        Convert this zcp node to RZCP representation.

        Args:
            resources: The resources dictionary
            config: The config.

        Returns:
            RZCPNode with callback for sampling text.
        """
        for placeholder, spec in self.resource_specs.items():
            resource_name = spec["name"]
            resource_type = spec["type"]
            if resource_name not in resources and resource_type != "argument":
                msg = f"""
                A resource could not be resolved at compile time. Resource of
                name '{resource_name}', type '{resource_type}', associated with
                placeholder {placeholder} was not present in compile resources.
                """
                raise GraphLoweringError(message=msg, block=self.block, sequence=self.sequence)

        sampling_callback = self._make_sampling_factory(resources)

        return RZCPNode(
            sequence=self.sequence,
            block=self.block,
            zone_advance_str=self.zone_advance_str,
            escape_strs=config.escape_patterns,
            tags=self.tags,
            timeout=self.timeout,
            sampling_callback=sampling_callback,
            input=False,
            output=False,
            next_zone=None,
            jump_advance_str=None,
            jump_zone=None
        )

    def lower(self,
              resources: Dict[str, AbstractResource],
              config: Config,
              ) -> 'RZCPNode':
        """
        Walk through entire graph and lower all nodes to RZCP, maintaining linkage.
        Handles cycles and complex flow control graphs correctly.

        Args:
            resources: The resolved resources that can now be used to lower this further.
            config: The config.
        Returns:
            Head of the lowered RZCP graph
        """
        lowered_self = self._lower_node(resources, config)
        if self.next_zone is not None:
            next_lowered = self.next_zone.lower(resources, config)
            lowered_self.next_zone = next_lowered
        return lowered_self

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class RZCPNode:
    """
    Resolved Zone Control Protocol node from SFCS construction.

    Has resolved resources and flow control markers. Contains sampling callbacks
    that return resolved text strings (not tokenized). This stage comes after
    resource resolution but before tokenization in the pipeline.

    Attributes:
        sequence: String sequence, used only for error reporting
        block: Block number within the sequence. Only for error reporting.
        zone_advance_str: String that triggers advancement to next zone
        jump_advance_str: Optional string that triggers jump flow control
        tags: List of string tags that apply to this zone
        timeout: Maximum tokens to generate before forcing advancement
        sampling_callback: Function that returns resolved text string (no tokenization)
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        tool_name: Optional tool name for tool integration (serializable reference)
        next_zone: Next zone in the execution chain
        jump_zone: Optional target node for jump flow control
    """
    sequence: str
    block: int
    zone_advance_str: str
    tags: List[str]
    timeout: int
    sampling_callback: Callable[[Dict[str, AbstractResource]], str]
    escape_strs: Tuple[str, str]
    input: bool = False
    output: bool = False
    jump_advance_str: Optional[str] = None
    tool_name: Optional[str] = None
    next_zone: Optional['RZCPNode'] = None
    jump_zone: Optional['RZCPNode'] = None

    def __post_init__(self):
        """Validate node consistency after initialization."""
        # Jump advance string and jump zone must be both present or both absent
        if (self.jump_advance_str is None) != (self.jump_zone is None):
            raise GraphError("jump_advance_str and jump_zone must both be present or both be None",
                             sequence=self.sequence,
                             block=self.block)

    def has_jump(self) -> bool:
        """Check if this node supports jump flow control."""
        return self.jump_advance_str is not None and self.jump_zone is not None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (sink in the DCG-IO)."""
        return self.next_zone is None and not self.has_jump()

    def is_input_zone(self) -> bool:
        """Check if this zone feeds from input buffer."""
        return self.input

    def is_output_zone(self) -> bool:
        """Check if this zone's content should be captured."""
        return self.output

    def has_tool(self) -> bool:
        """Check if this zone has an associated tool."""
        return self.tool_name is not None

    def get_last_node(self) -> 'RZCPNode':
        """
        Get the last node of the chain by following the linked list structure.

        Returns:
            The last node of the chain we could find
        """
        node = self
        while node.next_zone is not None:
            node = node.next_zone
        return node

    def _lower_node(self,
                    resources: Dict[str, AbstractResource]
                    ) -> Tuple['SZCPNode', Optional[str]]:
        """
        Convert this RZCP node to SZCP representation.

        Executes the sampling callback to resolve all placeholders to final text,
        creating a fully serializable node ready for network transmission.
        Args:
            resources: Dictionary of resources to use. They are the dynamic ones.
        Returns:
            SZCPNode with fully resolved text content
        """
        try:
            # Execute sampling callback to get resolved text
            resolved_text = self.sampling_callback(resources)

            node = SZCPNode(
                sequence=self.sequence,
                block=self.block,
                text=resolved_text,
                zone_advance_str=self.zone_advance_str,
                escape_strs=self.escape_strs,
                tags=self.tags,
                timeout=self.timeout,
                input=self.input,
                output=self.output,
                jump_advance_str=None,
                tool_name=self.tool_name,
                next_zone=None,
                jump_zone=None
            )

            return node, self.jump_advance_str
        except Exception as err:
            raise GraphLoweringError("Failed to lower RZCP to SZCP",
                                     sequence=self.sequence, block=self.block) from err

    def lower(self,
              resources: Dict[str, AbstractResource],
              lowered_map: Optional[Dict['RZCPNode', 'SZCPNode']] = None
              ) -> 'SZCPNode':
        """
        Walk through entire graph and lower all nodes to SZCP, maintaining linkage.
        Handles cycles and complex flow control graphs correctly.

        Args:
            resources: Any dynamic resources of relevance.
            lowered_map: Optional mapping to handle cycles during graph traversal

        Returns:
            Head of the lowered SZCP graph
        """
        if lowered_map is None:
            lowered_map = {}
        if self in lowered_map:
            return lowered_map[self]

        lowered_self, jump_advance_str = self._lower_node(resources)
        lowered_map[self] = lowered_self

        if self.next_zone is not None:
            lowered_self.next_zone = self.next_zone.lower(resources, lowered_map)
        if self.jump_zone is not None:
            lowered_self.jump_zone = self.jump_zone.lower(resources, lowered_map)
            lowered_self.jump_advance_str = jump_advance_str

        return lowered_self

    def attach(self, sources: List['RZCPNode']) -> 'RZCPNode':
        """
        Attaches other RZCP nodes so their nominal advancement
        stages point at myself.

        Args:
            sources: The sources to attach to me

        Returns:
            Myself for method chaining
        """
        for source in sources:
            source.next_zone = self
        return self

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class SZCPNode:
    """
    Serializable Zone Control Protocol node with fully resolved content.

    Represents a zone where all placeholders have been resolved to final text,
    but tokenization has not yet occurred. This is the serialization boundary
    in the compilation pipeline - SZCP nodes can be serialized and sent over
    the network to backend execution engines.

    Key characteristics:
    - All resource placeholders resolved to actual text
    - Tool callbacks referenced by name (serializable)
    - Zone advance triggers still as strings (not tokenized)
    - Tags remain as string lists (not boolean arrays)
    - Graph structure preserved with proper references
    - Supports full DCG-IO flow control including jumps

    Attributes:
        sequence: Name of UDPL sequence this zone belongs to
        block: Block number within the sequence
        text: Fully resolved text with all placeholders filled in
        zone_advance_str: String that triggers advancement to next zone
        tags: String tags for selective extraction
        timeout: Maximum tokens to generate before forcing advancement
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        next_zone: Next zone in the execution chain
        jump_advance_str: Optional string that triggers jump flow control
        jump_zone: Optional target node for jump flow control
        tool_name: Optional tool name for tool integration (serializable reference)
    """
    sequence: str
    block: int
    text: str
    zone_advance_str: str
    escape_strs: Tuple[str, str]
    tags: List[str]
    timeout: int
    input: bool
    output: bool
    next_zone: Optional['SZCPNode'] = None
    jump_advance_str: Optional[str] = None
    jump_zone: Optional['SZCPNode'] = None
    tool_name: Optional[str] = None

    def has_jump(self) -> bool:
        """Check if this node supports jump flow control."""
        return self.jump_advance_str is not None and self.jump_zone is not None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (sink in the DCG-IO)."""
        return self.next_zone is None and not self.has_jump()

    def is_input_zone(self) -> bool:
        """Check if this zone feeds from input buffer."""
        return self.input

    def is_output_zone(self) -> bool:
        """Check if this zone's content should be captured."""
        return self.output

    def has_tool(self) -> bool:
        """Check if this zone has an associated tool."""
        return self.tool_name is not None

    def get_last_node(self) -> 'SZCPNode':
        """
        Get the last node of the chain by following the linked list structure.

        Returns:
            The last node in the next_zone chain
        """
        node = self
        while node.next_zone is not None:
            node = node.next_zone
        return node

    def _lower_node(self,
                    tokenizer: TokenizerInterface,
                    tag_converter: TagConverter,
                    tool_registry: Dict[str, Callable[[np.ndarray], np.ndarray]]
                    )->Tuple['LZCPNode', np.ndarray]:
        """
        Utility responsible for lowing this particular node, but
        not responsible for linking up flow control in general
        :param tokenizer: The tokenizer to use to convert strings to tokens
        :param tag_converter: The tag converter to convert the tags to their boolean arrays
        :param tool_registry: The tool registry to convert tools to their callbacks
        :return:
        - The LZCPNode, with no attachments
        - The jump tokens, if they exist.
        """
        try:

            # Basic tokenization
            tokens = np.array(tokenizer.tokenize(self.text))
            zone_advance_tokens = np.array(tokenizer.tokenize(self.zone_advance_str))

            # Jump advance tokenization
            if self.jump_advance_str is not None:
                jump_tokens = np.array(tokenizer.tokenize(self.jump_advance_str))
            else:
                jump_tokens = None

            # Tool resolution
            if self.has_tool():
                if self.tool_name not in tool_registry:
                    raise RuntimeError(f"Tool {self.tool_name} not found in available tools")
                tool = tool_registry[self.tool_name]
            else:
                tool = None

            tags = tag_converter.tensorize(self.tags)
            escape_tokens = tuple(tokenizer.tokenize(item) for item in self.escape_strs)

            node = LZCPNode(
                sequence= self.sequence,
                block = self.block,
                tokens = tokens,
                zone_advance_tokens = zone_advance_tokens,
                escape_tokens = escape_tokens,
                tags = tags,
                timeout = self.timeout,
                input = self.input,
                output = self.output,
                tool_callback=tool,

            )

            return node, jump_tokens
        except Exception as err:
            raise GraphLoweringError("Could not lower SZCP to LZCP", self.block, self.sequence) from err

    def lower(self,
            tokenizer: TokenizerInterface,
            tag_converter: TagConverter,
            tool_registry: Dict[str, Callable[[str], str]],
            lowered_map: Optional[Dict['SZCPNode', 'LZCPNode']] = None
            ) -> 'LZCPNode':
        """
        Walk through entire graph and lower all nodes to LZCP, maintaining linkage.
        Handles cycles and complex flow control graphs correctly.

        Args:
            tokenizer: Function to convert strings to token arrays
            tag_converter: Converts tag names to boolean arrays
            tool_registry: Maps tool names to actual callback functions

        Returns:
            LZCPNode with tokenized content and resolved tool callbacks

        Raises:
            GraphLoweringError: If tokenization fails or tool not found in registry
        """
        if lowered_map is None:
            lowered_map = {}
        if self in lowered_map:
            return lowered_map[self]

        lowered_self, jump_tokens = self._lower_node(tokenizer, tag_converter, tool_registry)
        lowered_map[self] = lowered_self
        if self.next_zone is not None:
            lowered_self.next_zone = self.next_zone.lower(tokenizer, tag_converter, tool_registry, lowered_map)
        if self.jump_zone is not None:
            lowered_self.jump_zone = self.jump_zone.lower(tokenizer, tag_converter, tool_registry, lowered_map)
            lowered_self.jump_tokens = jump_tokens
        return lowered_self

    def serialize(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """
        Serialize this SZCP graph to a dictionary representation for network transmission.

        Performs a complete traversal of the DCG-IO graph starting from this node,
        discovers all reachable nodes, and converts object references to integer indices
        to create a serializable format suitable for client/server communication.

        Returns:
            Dict[int, Dict[str, Dict[str, Any]]] Dictionary mapping node indices to serialized node data.
            Index 0 represents this root node. Node references (next_zone, jump_zone) are
            converted to integer indices ("next_zone_index", "jump_zone_index").
            None references become None values in the serialized format.
        """
        # Phase 1: Discover all reachable nodes and assign indices
        nodes = self._discover_all_nodes()

        # Phase 2: Serialize each node's data
        serialized_nodes = {}
        for node, index in nodes.items():
            serialized_nodes[index] = node._serialize_node(nodes)

        return serialized_nodes

    def _discover_all_nodes(self, visited: Optional[Dict['SZCPNode', int]] = None) -> Dict['SZCPNode', int]:
        """
        Phase 1: Walk the entire graph and assign a unique index to each node.

        Uses depth-first traversal with cycle detection. Returns a complete
        mapping of all reachable nodes to their assigned indices.

        Args:
            visited: Internal parameter for recursion, tracks already-discovered nodes

        Returns:
            Dict mapping each discovered node to its unique index (starting from 0)
        """
        if visited is None:
            visited = {}

        if self in visited:
            return visited

        visited[self] = len(visited)

        if self.next_zone is not None:
            self.next_zone._discover_all_nodes(visited)
        if self.jump_zone is not None:
            self.jump_zone._discover_all_nodes(visited)

        return visited
    def _serialize_node(self, nodes: Dict['SZCPNode', int]) -> Dict[str, Dict[str, Any]]:
        """
        Serialize this node's data, replacing object references with indices.

        Args:
            nodes: Mapping from nodes to their assigned indices

        Returns:
            Dictionary representation of this node with index-based references
        """
        internal_data = {
            "sequence": self.sequence,
            "block": self.block,
            "text": self.text,
            "zone_advance_str": self.zone_advance_str,
            "tags": self.tags,
            "timeout": self.timeout,
            "input": self.input,
            "output": self.output,
            "jump_advance_str": self.jump_advance_str,
            "escape_strs" : self.escape_strs,
            "tool_name" : self.tool_name,
        }
        links = {
            "next_zone_index": nodes.get(self.next_zone),
            "jump_zone_index": nodes.get(self.jump_zone),
        }
        representation = {"data" : internal_data, "links" : links}
        return representation

    @classmethod
    def deserialize(cls, data: Dict[int, Dict[str, Dict[str, Any]]]) -> 'SZCPNode':
        """
        Deserialize a dictionary representation back to an SZCP graph.

        Reconstructs the full DCG-IO graph structure from serialized data,
        properly restoring all node references including next_zone and jump_zone
        relationships.

        Args:
            data: Dictionary mapping node indices to {"data": {...}, "links": {...}}
                  format. Index 0 represents the root node of the graph.

        Returns:
            Root SZCPNode of the reconstructed graph

        Raises:
            KeyError: If index 0 does not exist or if link indices reference non-existent nodes
            TypeError: If data structure does not match expected format
        """
        nodes, links = cls._create_unlinked_nodes(data)
        cls._resolve_references(nodes, links)
        return nodes[0]

    @classmethod
    def _create_unlinked_nodes(cls,
                               data: Dict[int, Dict[str, Dict[str, Any]]]
                               ) -> Tuple[Dict[int, 'SZCPNode'], Dict[int, Dict[str, Any]]]:
        """
        Create SZCPNode objects from serialized data and extract link information.

        Constructs all nodes with their basic data using **kwargs from the "data" section,
        while extracting link information for later reference resolution.

        Args:
            data: Serialized graph data with "data" and "links" sections per node

        Returns:
            Tuple of:
            - Dict mapping indices to created SZCPNode objects (with None references)
            - Dict mapping indices to link data for reference resolution

        Raises:
            TypeError: If node data is missing required fields or has wrong types
        """
        nodes = {}
        links = {}

        for index, node_data in data.items():
            nodes[index] = cls(**node_data["data"])
            links[index] = node_data["links"]

        return nodes, links

    @classmethod
    def _resolve_references(cls,
                            nodes: Dict[int, 'SZCPNode'],
                            links: Dict[int, Dict[str, Any]]
                            ) -> None:
        """
        Wire up next_zone and jump_zone references using extracted link data.
        It will, by looking up nodes, be able to resolve the links data onto
        each given node.

        Args:
            nodes: Dictionary of created SZCPNode objects indexed by their position
            links: Dictionary of link data containing next_zone_index and jump_zone_index

        Raises:
            KeyError: If link indices reference nodes that don't exist in the nodes dict
        """
        for index, link_data in links.items():
            node = nodes[index]

            if link_data["next_zone_index"] is not None:
                node.next_zone = nodes[link_data["next_zone_index"]]

            if link_data["jump_zone_index"] is not None:
                node.jump_zone = nodes[link_data["jump_zone_index"]]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def visualize(self, save_to_file: Optional[str] = None) -> None:
        """
        Create an interactive Plotly visualization of this SZCP workflow graph.

        Uses igraph's Sugiyama hierarchical layout algorithm to automatically
        arrange nodes, then converts to GraphData format for rendering.

        Args:
            save_to_file: Optional filename to save the figure. If provided, saves to file
                         instead of displaying. Supports .html format.
        """
        # Check if visualization was enabled
        if not VISUALIZATION_ENABLED:
            msg = """
           The visualization libraries could not be successfully imported.
           Most likely one of igraph, networkx, or plotly is not installed.

           Run:

           pip install python-igraph
           pip install plotly
           pip install networkx

           To ensure they are all installed
           """
            msg = textwrap.dedent(msg)
            raise NotImplementedError(msg) from reason

        # Extract all nodes from SZCP graph
        all_nodes = self._discover_all_nodes()

        # Build NetworkX graph for layout algorithm
        G = nx.DiGraph()
        for szcp_node, index in all_nodes.items():
            G.add_node(index)

            if szcp_node.next_zone:
                target_index = all_nodes[szcp_node.next_zone]
                G.add_edge(index, target_index)

            if szcp_node.jump_zone:
                target_index = all_nodes[szcp_node.jump_zone]
                G.add_edge(index, target_index)

        # Apply Sugiyama hierarchical layout (left-to-right orientation)
        ig_graph = ig.Graph.from_networkx(G)
        layout = ig_graph.layout("sugiyama")

        # Map layout positions back to original NetworkX node IDs
        # The layout includes positions for both original nodes and dummy nodes,
        # but the first N positions correspond to our original N nodes
        nx_node_list = list(G.nodes())
        positions = {}
        for i in range(len(nx_node_list)):
            original_node_id = nx_node_list[i]
            positions[original_node_id] = (layout[i][1], -layout[i][0])

        # Simple color cycling - change color when sequence changes
        colors = ['#FF6B6B', '#45B7D1', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
        current_sequence = None
        color_index = -1

        # Convert to GraphNode objects for rendering
        graph_nodes = []
        index_to_node = {index: szcp_node for szcp_node, index in all_nodes.items()}

        for index in sorted(positions.keys()):
            szcp_node = index_to_node[index]
            x, y = positions[index]

            # Assign new color when sequence changes
            if szcp_node.sequence != current_sequence:
                current_sequence = szcp_node.sequence
                color_index = (color_index + 1) % len(colors)

            # Build node data for hover information - ALL available data for debugging
            node_data = {
                "sequence": szcp_node.sequence,
                "block": szcp_node.block,
                "tags": szcp_node.tags,
                "timeout": szcp_node.timeout,
                "zone_advance_str": szcp_node.zone_advance_str,
                "jump_advance_str": szcp_node.jump_advance_str,
                "escape_strs": szcp_node.escape_strs,
                "input": szcp_node.input,
                "output": szcp_node.output,
                "tool_name": szcp_node.tool_name,
                "text": szcp_node.text  # Full text at bottom since it's long
            }

            # Create GraphNode
            graph_node = GraphNode(
                id=str(index),
                name=szcp_node.sequence,
                color=colors[color_index],
                x=x,
                y=y,
                nominal=str(all_nodes[szcp_node.next_zone]) if szcp_node.next_zone else None,
                jump=str(all_nodes[szcp_node.jump_zone]) if szcp_node.jump_zone else None,
                node_data=node_data
            )

            graph_nodes.append(graph_node)

        # Create GraphData and use existing rendering system
        graph_data = GraphData(graph_nodes, "SZCP Workflow Visualization")
        fig = create_plotly_graph(graph_data)

        if save_to_file is None:
            fig.show()
        else:
            # Save as HTML - preserves full interactivity
            if not save_to_file.endswith('.html'):
                save_to_file = save_to_file + '.html'
            fig.write_html(save_to_file)
            print(f"Workflow visualization saved to: {save_to_file}")

@dataclass
class LZCPNode:
    """
    Lowered Zone Control Protocol node with resolved tokens and tensor-ready data.

    This is the low-level representation used after sampling callbacks have been
    resolved and string references converted to tensor indices. LZCP nodes are
    ready for compilation to the TTFA backend.

    Attributes:
        sequence: String sequence, used only for error reporting
        block: Block number within the sequence
        tokens: Actual token sequence to feed as input (from resolved sampling)
        next_zone: Pointer to the next zone in the execution chain
        zone_advance_tokens: Token ID that triggers advancement to next_zone
        tags: Boolean array indicating which tags apply to this zone
        timeout: Maximum tokens to generate before advancing (prevents infinite loops)
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        jump_token: Optional token ID that triggers jump flow control
        jump_zone: Optional target node for jump flow control
    """
    sequence: str
    block: int
    tokens: np.ndarray  # Shape: (sequence_length,) - token IDs
    zone_advance_tokens: np.ndarray
    escape_tokens: Tuple[np.ndarray, np.ndarray]
    tags: np.ndarray  # Shape: (num_tags,) - boolean array for tag membership
    timeout: int
    input: bool
    output: bool
    next_zone: Optional['LZCPNode'] = None
    jump_tokens: Optional[np.ndarray] = None
    jump_zone: Optional['LZCPNode'] = None
    tool_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self):
        """Validate node consistency and array shapes after initialization."""
        try:
            # Jump token and jump node must be both present or both absent
            if (self.jump_tokens is None) != (self.jump_zone is None):
                raise ValueError("jump_tokens and jump_zone must both be present or both be None")

            # Validate zone_advance_tokens
            if not isinstance(self.zone_advance_tokens, np.ndarray):
                raise TypeError("zone_advance_tokens must be a numpy array")
            if self.zone_advance_tokens.ndim != 1:
                raise ValueError("zone_advance_tokens must be a 1D array")
            if self.zone_advance_tokens.dtype not in [np.int32, np.int64]:
                raise ValueError("zone_advance_tokens must have integer dtype")

            # Validate jump_tokens if present
            if self.jump_tokens is not None:
                if not isinstance(self.jump_tokens, np.ndarray):
                    raise TypeError("jump_tokens must be a numpy array")
                if self.jump_tokens.ndim != 1:
                    raise ValueError("jump_tokens must be a 1D array")
                if self.jump_tokens.dtype not in [np.int32, np.int64]:
                    raise ValueError("jump_tokens must have integer dtype")

            # Validate tags array
            if not isinstance(self.tags, np.ndarray):
                raise TypeError("tags must be a numpy array")
            if self.tags.ndim != 1:
                raise ValueError("tags must be a 1D array")
            if self.tags.dtype != np.bool_:
                raise ValueError("tags must have boolean dtype")

            # Validate tokens array
            if not isinstance(self.tokens, np.ndarray):
                raise TypeError(f"tokens must be a numpy array, got {type(self.tokens)}")
            if self.tokens.ndim != 1:
                raise ValueError("tokens must be a 1D array")
            if self.tokens.dtype not in [np.int32, np.int64]:
                raise ValueError("tokens must have integer dtype")

            # Validate escape patterns
            if not isinstance(self.escape_tokens, tuple):
                raise TypeError("escape_tokens must be a tuple")
            if len(self.escape_tokens) != 2:
                raise TypeError("escape_tokens must have length 2")
            for item in self.escape_tokens:
                if not isinstance(item, np.ndarray):
                    raise TypeError("escape_tokens must have numpy arrays")

        except Exception as err:
            raise GraphError(f"LZCP node validation failed",
                             sequence=self.sequence, block=self.block) from err
    def has_jump(self) -> bool:
        """Check if this node supports jump flow control."""
        return self.jump_tokens is not None and self.jump_zone is not None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (sink in the DCG-IO)."""
        return self.next_zone is None and not self.has_jump()

    def is_input_zone(self) -> bool:
        """Check if this zone feeds from input buffer."""
        return self.input

    def is_output_zone(self) -> bool:
        """Check if this zone's content should be captured."""
        return self.output

    def num_tokens(self) -> int:
        """Get the number of tokens in this zone."""
        return len(self.tokens)

    def get_active_tags(self, tag_names: List[str]) -> List[str]:
        """
        Get the names of tags that are active for this zone.

        Args:
            tag_names: List of tag names corresponding to the boolean array indices

        Returns:
            List of active tag names
        """
        if len(tag_names) != len(self.tags):
            raise ValueError("tag_names length must match tags array length")

        return [name for name, active in zip(tag_names, self.tags) if active]

    def get_last_node(self)->'LZCPNode':
        """
        Get the last node of the chain by just following
        the linked list structure
        :return: The last node of the chain we could find
        """
        node = self
        while node.next_zone is not None:
            node = node.next_zone
        return node

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other