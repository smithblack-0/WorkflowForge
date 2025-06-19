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

The module provides two levels of representation:
1. zcp nodes - High-level with string references and sampling callbacks
2. LZCP nodes - Lowered with actual tokens and tensor-ready data structures
"""
import numpy as np
import textwrap
import functools
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any, Type
from ..flow_control.tag_converter import TagConverter
from ..tokenizer_interface import TokenizerInterface
from ..resources import AbstractResource

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

## Setup the various aliases

GraphLoweringErrorFactory = Callable[[str], GraphLoweringError]
SamplerFactory = Callable[[], np.ndarray]
SamplerFactoryFactory = Callable[[str, Dict[str, Dict[str, Any]],
                                  GraphLoweringErrorFactory],
                                 SamplerFactory]


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
        tags: String tags for extraction (or np.ndarray if converted)
        timeout: Maximum tokens to generate before forcing advancement
        next_zone: Next zone in the linked list
    """
    sequence: str
    block: int
    construction_callback: Callable[[Dict[str, AbstractResource]],str]
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
                               tokenizer: TokenizerInterface,
                               resources: Dict[str, AbstractResource]
                               ):
        """
        Factory for making the main sampling factory, which invokes the

        :param tokenizer:
        :param resources:
        :return:
        """

        def sample()->np.ndarray:
            """
            Main sampling function, draws from the resources, fills
            in placeholders, gets text, tokenizes, returns an ndarray.
            :return: The tokenized text
            """
            try:
                text = self.construction_callback(resources)
                tokens = tokenizer.tokenize(text)
                return np.array(tokens)
            except Exception as err:

                raise GraphLoweringError("Issue occurred while resolving resources or tokenizing texts",
                                   self.block, self.sequence) from err
        return sample

    def _lower_node(self,
                    resources: Dict[str, AbstractResource],
                    tokenizer: TokenizerInterface,
                    tag_converter: TagConverter
                    ) -> 'RZCPNode':
        """
        Convert this zcp node to RZCP representation.

        Args:
            callback_factory: Function that takes raw_text and resource_specs, returns construction callback
            tokenizer: Function to convert strings to token arrays
            tag_converter: Converts tag names to boolean arrays

        Returns:
            RZCPNode with resolved tokenization and construction callback
        """
        try:
            for placeholder, spec in self.resource_specs.items():
                resource_name = spec['name']
                if resource_name not in resources:
                    raise RuntimeError(f"Resource '{resource_name}' not found for placeholder '{placeholder}'")

            sampling_callback = self._make_sampling_factory(tokenizer, resources)
            zone_advance_tokens = np.array(tokenizer.tokenize(self.zone_advance_str))
            tags_array = tag_converter.tensorize(self.tags)

            return RZCPNode(
                sequence = self.sequence,
                block = self.block,
                zone_advance_tokens=zone_advance_tokens,
                tags=tags_array,
                timeout=self.timeout,
                sampling_callback=sampling_callback,
                input=False,
                output=False,
                next_zone=None,
                jump_tokens=None,
                jump_zone=None
            )
        except Exception as err:
            raise GraphLoweringError("Failed to lower zcp to RZCP",
                                     block=self.block, sequence=self.sequence) from err

    def lower(self,
              resources: Dict[str, AbstractResource],
              tokenizer: TokenizerInterface,
              tag_converter: TagConverter,
              ) -> 'RZCPNode':
        """
        Walk through entire graph and lower all nodes to RZCP, maintaining linkage.
        Handles cycles and complex flow control graphs correctly.

        Args:
            resources: The resolved resources that can now be used to lower this further.
            tokenizer: Function to convert strings to token arrays
            tag_converter: Converts tag names to boolean arrays

        Returns:
            Head of the lowered RZCP graph
        """
        lowered_self = self._lower_node(resources, tokenizer, tag_converter)
        if self.next_zone is not None:
            next_lowered = self.next_zone.lower(resources, tokenizer, tag_converter)
            lowered_self.next_zone = next_lowered
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
        sequence: String sequence, used only for error reporting
        block: Block number within the sequence. Only for error reporting.
        zone_advance_tokens: Token List ID that triggers advancement to next zone
        tags: Boolean array indicating which tags apply to this zone
        timeout: Maximum tokens to generate before forcing advancement
        sampling_callback: Function that returns tokenized prompt (no params, everything resolved)
        input: If True, zone feeds from input buffer when prompt tokens exhausted
        output: If True, zone content is captured for extraction
        next_zone: Next zone in the execution chain
        jump_tokens: Optional token ID List that triggers jump flow control
        jump_zone: Optional target node for jump flow control
    """
    sequence: str
    block: int
    zone_advance_tokens: np.ndarray
    tags: np.ndarray
    timeout: int
    sampling_callback: Callable[[], np.ndarray]
    input: bool = False
    output: bool = False
    jump_tokens: Optional[np.ndarray] = None
    next_zone: Optional['RZCPNode'] = None
    jump_zone: Optional['RZCPNode'] = None
    tool_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self):
        """Validate node consistency after initialization."""
        # Jump token and jump node must be both present or both absent
        if (self.jump_tokens is None) != (self.jump_zone is None):
            raise GraphError("jump_token and jump_node must both be present or both be None",
                             sequence=self.sequence,
                             block=self.block)

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
        try:
            tokens = self.sampling_callback()

            return LZCPNode(
                sequence=self.sequence,
                block=self.block,
                tokens=tokens,
                zone_advance_tokens=self.zone_advance_tokens,
                tags=self.tags,
                timeout=self.timeout,
                input=self.input,
                output=self.output,
                next_zone=None,  # Will be wired later
                jump_tokens=self.jump_tokens,
                jump_zone=None,  # Will be wired later
                tool_callback = self.tool_callback
            )
        except Exception as err:
            raise GraphLoweringError("Failed to lower RZCP to LZCP",
                                     sequence=self.sequence, block=self.block) from err

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
                raise TypeError("tokens must be a numpy array")
            if self.tokens.ndim != 1:
                raise ValueError("tokens must be a 1D array")
            if self.tokens.dtype not in [np.int32, np.int64]:
                raise ValueError("tokens must have integer dtype")

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
        while self.next_zone is not None:
            node = self.next_zone
        return node