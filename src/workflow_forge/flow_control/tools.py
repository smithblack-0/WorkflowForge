"""
The tools and toolbox package are used to contain,
interact with, and resolve tool usage.
"""
import numpy as np
from typing import Callable, List

class Toolbox:
    """
    Manages a collection of tools with shared buffer configuration.
    Represents a SIFA/SOCA pair specification for the backend.
    """

    def __init__(self,
                 tokenizer: Callable[[str], np.ndarray],
                 detokenizer: Callable[[np.ndarray], str],
                 input_buffer_size: int,
                 output_buffer_size: int):
        """
        Initialize toolbox with buffer configuration.

        Args:
            tokenizer: Function to convert strings to token arrays
            detokenizer: Function to convert token arrays to strings
            input_buffer_size: Size of input buffer for SIFA
            output_buffer_size: Size of output buffer for SOCA
        """
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.input_buffer_size = input_buffer_size
        self.output_buffer_size = output_buffer_size
        self.tools: List[Tool] = []
        self._next_tool_index = 0

    def new_tool(self, callback: Callable[[str], str]) -> 'Tool':
        """
        Create a new tool with a string-based callback.

        Args:
            callback: Function that accepts a string and returns a string

        Returns:
            Tool instance with unique index
        """
        tool = Tool(
            index=self._next_tool_index,
            callback=callback,
            tokenizer=self.tokenizer,
            detokenizer=self.detokenizer
        )
        self.tools.append(tool)
        self._next_tool_index += 1
        return tool


class Tool:
    """
    Individual tool that wraps a string callback with tokenization.
    """

    def __init__(self,
                 index: int,
                 callback: Callable[[str], str],
                 tokenizer: Callable[[str], np.ndarray],
                 detokenizer: Callable[[np.ndarray], str]):
        """
        Initialize tool with callback and tokenization functions.

        Args:
            index: Unique index for this tool within its toolbox
            callback: String-to-string user callback
            tokenizer: Function to convert strings to tokens
            detokenizer: Function to convert tokens to strings
        """
        self.index = index
        self.callback = callback
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

    def __call__(self, tokens: np.ndarray) -> np.ndarray:
        """
        Process tokens through the callback and return tokenized result.

        Args:
            tokens: Input token array from captured zone

        Returns:
            Tokenized result from callback
        """
        # Detokenize input
        text = self.detokenizer(tokens)

        # Call user callback
        result_text = self.callback(text)

        # Retokenize result
        return self.tokenizer(result_text)