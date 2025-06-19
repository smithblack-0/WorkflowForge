"""
Zone Control Protocol (ZCP) Node Definitions

This module defines the data structures for ZCP nodes, which represent zones of text
in the intermediate representation used by Workflow Forge before compilation to
the Token Triggered Finite Automata (TTFA) backend.

ZCP supports Directed Cyclic IO Graphs (DCG-IO), which are directed graphs that:
- Allow cycles (unlike DAGs)
- Have exactly one source and one sink vertex
- Guarantee all vertices are reachable from source and can reach sink
- Maintain computational tractability for workflow analysis

The module provides two levels of representation:
1. ZCP nodes - High-level with string references and sampling callbacks
2. LZCP nodes - Lowered with actual tokens and tensor-ready data structures
"""
from functools import total_ordering

from ..resources import AbstractResource
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any
from ..flow_control.tag_converter import TagConverter
import numpy as np
from ..tokenizer_interface import TokenizerInterface

@dataclass
class ZCPNode:
    """
    Simple Zone Control Protocol node from UDPL parsing.

    Represents a basic linked list of prompt zones with resource metadata.
    No flow control - just what UDPL can express.

    Attributes:
        sequence: Name of UDPL sequence this zone belongs to
        block: Block number within the sequence
        resource_specs: Placeholder → resource mapping (unresolved)
        raw_text: Template text with {placeholder} syntax
        zone_advance_token: Token that triggers advancement to next zone
        tags: String tags for extraction (or np.ndarray if converted)
        timeout: Maximum tokens to generate before forcing advancement
        next_zone: Next zone in the linked list
    """
    sequence: str
    block: int
    resource_specs: Dict[str, Dict[str, Any]]
    raw_text: str
    zone_advance_token: str
    tags: List[str]
    timeout: int
    next_zone: Optional['ZCPNode'] = None

    def get_last_node(self) -> 'ZCPNode':
        """Follow the linked list to get the tail node."""
        node = self
        while node.next_zone is not None:
            node = node.next_zone
        return node

    def _lower_node(self,
                    callback_factory: Callable[[str, Dict[str, Dict[str, Any]]], Callable[[], np.ndarray]],
                    tokenizer: TokenizerInterface,
                    tag_converter: 'TagConverter'
                    ) -> 'RZCPNode':
        """
        Convert this ZCP node to RZCP representation.

        Args:
            callback_factory: Function that takes raw_text and resource_specs, returns construction callback
            tokenizer: Function to convert strings to token arrays
            tag_converter: Converts tag names to boolean arrays

        Returns:
            RZCPNode with resolved tokenization and construction callback
        """
        # Create construction callback using factory
        construction_callback = callback_factory(self.raw_text, self.resource_specs)

        # Tokenize zone advance token
        zone_advance_tokens = tokenizer.tokenize(self.zone_advance_token)
        if len(zone_advance_tokens) != 1:
            raise ValueError(f"Zone advance token '{self.zone_advance_token}' must tokenize to single token")
        zone_advance_token_id = zone_advance_tokens[0]

        # Convert tags to bool array
        tags_array = tag_converter.tensorize(self.tags)

        return RZCPNode(
            zone_advance_token=int(zone_advance_token_id),
            tags=tags_array,
            timeout=self.timeout,
            construction_callback=construction_callback,
            input=False,  # Will be set by flow control if needed
            output=False,  # Will be set by flow control if needed
            next_zone=None,  # Will be wired in chain conversion
            jump_token=None,  # No flow control at this stage
            jump_zone=None
        )

    def lower(self,
              callback_factory: Callable[[str, Dict[str, Dict[str, Any]]], Callable[[], np.ndarray]],
              tokenizer: TokenizerInterface,
              tag_converter: 'TagConverter',
              lowered_map: Optional[Dict['ZCPNode', 'RZCPNode']] = None) -> 'RZCPNode':
        """
        Walk through entire graph and lower all nodes to RZCP, maintaining linkage.
        Handles cycles and complex flow control graphs correctly.

        Args:
            callback_factory: Function that takes raw_text and resource_specs, returns construction callback
            tokenizer: Function to convert strings to token arrays
            tag_converter: Converts tag names to boolean arrays
            lowered_map: Map of already lowered nodes to avoid cycles

        Returns:
            Head of the lowered RZCP graph
        """
        if lowered_map is None:
            lowered_map = {}
        if self in lowered_map:
            return lowered_map[self]

        lowered_self = self._lower_node(callback_factory, tokenizer, tag_converter)
        lowered_map[self] = lowered_self

        if self.next_zone is not None:
            lowered_self.next_zone = self.next_zone.lower(callback_factory, tokenizer, tag_converter, lowered_map)

        return lowered_self

@dataclass
class RZCPNode:
    """
    Resolved Zone Control Protocol node from SFCS construction.

    Has resolved resources and flow control markers. Designed for sampling -
    can walk through itself and lower to equivalent LZCP sequence when invoked.
    Sampling is performed by invoking the construction callback,
    and which will then tokenize and return the results. This happens once
    at the start of the batch, and is then applied in many locations.

    Attributes:
        zone_advance_token: Token ID that triggers advancement to next zone
        tags: Boolean array indicating which tags apply to this zone
        timeout: Maximum tokens to generate before forcing advancement
        construction_callback: Function that returns tokenized prompt (no params, everything resolved)
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        next_zone: Next zone in the execution chain
        jump_token: Optional token ID that triggers jump flow control
        jump_node: Optional target node for jump flow control
    """
    zone_advance_token: int
    tags: np.ndarray
    timeout: int
    construction_callback: Callable[[], np.ndarray]
    input: bool = False
    output: bool = False
    jump_token: Optional[int] = None
    next_zone: Optional['RZCPNode'] = None
    jump_zone: Optional['RZCPNode'] = None
    tool_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self):
        """Validate node consistency after initialization."""
        # Jump token and jump node must be both present or both absent
        if (self.jump_token is None) != (self.jump_zone is None):
            raise ValueError("jump_token and jump_node must both be present or both be None")

    def has_jump(self) -> bool:
        """Check if this node supports jump flow control."""
        return self.jump_token is not None and self.jump_zone is not None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (sink in the DCG-IO)."""
        return self.next_zone is None and not self.has_jump()

    def is_input_zone(self) -> bool:
        """Check if this zone feeds from input buffer."""
        return self.input

    def is_output_zone(self) -> bool:
        """Check if this zone's content should be captured."""
        return self.output

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

    def _lower_node(self) -> 'LZCPNode':
        """
        Convert this RZCP node to LZCP representation.

        Since all tokenization and resolution is already done,
        this is mostly just copying data over.

        Returns:
            LZCPNode with pre-resolved tokens and data
        """
        # Get the pre-tokenized prompt
        tokens = self.construction_callback()

        return LZCPNode(
            tokens=tokens,
            zone_advance_token=self.zone_advance_token,
            tags=self.tags,
            timeout=self.timeout,
            input=self.input,
            output=self.output,
            next_zone=None,  # Will be wired later
            jump_token=self.jump_token,
            jump_zone=None,  # Will be wired later
            tool_callback = self.tool_callback
        )

    def lower(self,
            lowered_map: Optional[Dict['RZCPNode', 'LZCPNode']] = None
            ) -> 'LZCPNode':
        """
        Walk through entire graph and lower all nodes to LZCP, maintaining linkage.
        Handles cycles and complex flow control graphs correctly.

        Returns:
            Head of the lowered LZCP graph
        """
        if lowered_map is None:
            lowered_map = {}
        if self in lowered_map:
            return lowered_map[self]

        lowered_self = self._lower_node()
        lowered_map[self] = lowered_self
        if self.next_zone is not None:
            lowered_self.next_zone = self.next_zone.lower(lowered_map)
        if self.jump_zone is not None:
            lowered_self.jump_zone = self.jump_zone.lower(lowered_map)
        return lowered_self

    def attach(self, sources: List['RZCPNode'])->'RZCPNode':
        """
        Attaches other RZCP nodes so their nominal advancement
        stages point at myself.
        :param sources: The sources to attach to me
        :return: Myself
        """
        for source in sources:
            source.next_zone = self
        return self

@dataclass
class LZCPNode:
    """
    Lowered Zone Control Protocol node with resolved tokens and tensor-ready data.

    This is the low-level representation used after sampling callbacks have been
    resolved and string references converted to tensor indices. LZCP nodes are
    ready for compilation to the TTFA backend.

    Attributes:
        tokens: Actual token sequence to feed as input (from resolved sampling)
        next_zone: Pointer to the next zone in the execution chain
        zone_advance_token: Token ID that triggers advancement to next_zone
        tags: Boolean array indicating which tags apply to this zone
        timeout: Maximum tokens to generate before advancing (prevents infinite loops)
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        jump_token: Optional token ID that triggers jump flow control
        jump_zone: Optional target node for jump flow control
    """
    tokens: np.ndarray  # Shape: (sequence_length,) - token IDs
    zone_advance_token: int  # Token ID from tokenizer
    tags: np.ndarray  # Shape: (num_tags,) - boolean array for tag membership
    timeout: int
    input: bool
    output: bool
    next_zone: Optional['LZCPNode'] = None
    jump_token: Optional[int] = None
    jump_zone: Optional['LZCPNode'] = None
    tool_callback: Optional[Callable[np.ndarray], np.ndarray] = None

    def __post_init__(self):
        """Validate node consistency and array shapes after initialization."""
        # Jump token and jump node must be both present or both absent
        if (self.jump_token is None) != (self.jump_zone is None):
            raise ValueError("jump_token and jump_node must both be present or both be None")

        # Validate array shapes and types
        if not isinstance(self.tokens, np.ndarray):
            raise TypeError("tokens must be a numpy array")
        if self.tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")
        if self.tokens.dtype not in [np.int32, np.int64]:
            raise ValueError("tokens must have integer dtype")

        if not isinstance(self.tags, np.ndarray):
            raise TypeError("tags must be a numpy array")
        if self.tags.ndim != 1:
            raise ValueError("tags must be a 1D array")
        if self.tags.dtype != np.bool_:
            raise ValueError("tags must have boolean dtype")

    def has_jump(self) -> bool:
        """Check if this node supports jump flow control."""
        return self.jump_token is not None and self.jump_zone is not None

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
        while self.next_zone is not None:
            node = self.next_zone
        return node