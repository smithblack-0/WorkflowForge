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
from ..resources import AbstractResource
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict
import numpy as np


@dataclass
class ZCPNode:
    """
    Zone Control Protocol node representing a zone of text in the workflow.

    This is the high-level representation used during initial compilation from UDPL.
    ZCP nodes contain string references and callbacks that need to be resolved
    before lowering to LZCP.

    Attributes:
        sequence: Name of the sequence this zone belongs to (used during construction)
        block: Block number within the sequence (used during construction)
        sampling_callbacks: Function that resolves all placeholders to their value when invoked
         with the resource collection.
        next_zone: Pointer to the next zone in the execution chain
        zone_advance_token: Token that triggers advancement to next_zone
        tags: List of string tags for this zone (used for extraction)
        timeout: Maximum tokens to generate before advancing (prevents infinite loops)
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        jump_token: Optional token that triggers jump flow control
        jump_node: Optional target node for jump flow control
    """
    sequence: str
    block: int
    sampling_callbacks: Callable[[Dict[str, AbstractResource]], Dict[str, str]]
    zone_advance_token: str
    tags: List[str]
    timeout: int
    input: bool
    output: bool
    next_zone: Optional['ZCPNode'] = None
    jump_token: Optional[str] = None
    jump_node: Optional['ZCPNode'] = None

    def __post_init__(self):
        """Validate node consistency after initialization."""
        # Jump token and jump node must be both present or both absent
        if (self.jump_token is None) != (self.jump_node is None):
            raise ValueError("jump_token and jump_node must both be present or both be None")

    def has_jump(self) -> bool:
        """Check if this node supports jump flow control."""
        return self.jump_token is not None and self.jump_node is not None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (sink in the DCG-IO)."""
        return self.next_zone is None and not self.has_jump()

    def is_input_zone(self) -> bool:
        """Check if this zone feeds from input buffer."""
        return self.input

    def is_output_zone(self) -> bool:
        """Check if this zone's content should be captured."""
        return self.output
    
    def get_last_node(self)->'ZCPNode':
        """
        Get the last node of the chain by just following
        the linked list structure
        :return: The last node of the chain we could find
        """
        node = self
        while self.next_zone is not None:
            node = self.next_zone
        return node


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
        jump_node: Optional target node for jump flow control
    """
    tokens: np.ndarray  # Shape: (sequence_length,) - token IDs
    zone_advance_token: int  # Token ID from tokenizer
    tags: np.ndarray  # Shape: (num_tags,) - boolean array for tag membership
    timeout: int
    input: bool
    output: bool
    next_zone: Optional['LZCPNode'] = None
    jump_token: Optional[int] = None  # Token ID from tokenizer
    jump_node: Optional['LZCPNode'] = None

    def __post_init__(self):
        """Validate node consistency and array shapes after initialization."""
        # Jump token and jump node must be both present or both absent
        if (self.jump_token is None) != (self.jump_node is None):
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
        return self.jump_token is not None and self.jump_node is not None

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