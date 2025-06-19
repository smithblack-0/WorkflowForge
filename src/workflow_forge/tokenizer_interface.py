"""
Tokenizer Interface and Adapter System

This module provides a unified interface for different tokenizer implementations
and an adapter system for registering new tokenizer types.
"""

from typing import Callable, Dict, Any, Type, List, Union
import numpy as np


class TokenizerInterface:
    """
    Unified interface for tokenization operations.
    Wraps different tokenizer implementations with consistent API.
    """

    def __init__(self,
                 tokenize_fn: Callable[[str], np.ndarray],
                 detokenize_fn: Callable[[np.ndarray], str],
                 get_special_tokens_fn: Callable[[], List[str]]):
        """
        Initialize with tokenization callbacks.

        Args:
            tokenize_fn: Function that converts string to token array
            detokenize_fn: Function that converts token array to string
            get_special_tokens_fn: Function that returns list of special tokens
        """
        self.tokenize = tokenize_fn
        self.detokenize = detokenize_fn
        self.get_special_tokens = get_special_tokens_fn


# Global registry mapping types to constructor functions
_TOKENIZER_CONSTRUCTORS: Dict[Type, Callable[[Any], TokenizerInterface]] = {}


def register_tokenizer_constructor(tokenizer_type: Type,
                                   constructor: Callable[[Any], TokenizerInterface]) -> None:
    """
    Register a constructor function for a tokenizer type.

    Args:
        tokenizer_type: The type of tokenizer this constructor handles
        constructor: Function that creates TokenizerInterface from tokenizer instance
    """
    _TOKENIZER_CONSTRUCTORS[tokenizer_type] = constructor


def load_tokenizer(tokenizer: Any) -> TokenizerInterface:
    """
    Load tokenizer by finding registered constructor for its type.

    Args:
        tokenizer: Tokenizer object to wrap

    Returns:
        TokenizerInterface wrapping the tokenizer

    Raises:
        ValueError: If no constructor registered for this tokenizer type
    """
    tokenizer_type = type(tokenizer)
    if tokenizer_type in _TOKENIZER_CONSTRUCTORS:
        return _TOKENIZER_CONSTRUCTORS[tokenizer_type](tokenizer)

    raise ValueError(f"No constructor registered for tokenizer type: {tokenizer_type}")


# Register HuggingFace tokenizers
try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


    def create_huggingface_interface(
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> TokenizerInterface:
        """Create TokenizerInterface for HuggingFace tokenizer."""

        def tokenize_fn(text: str) -> np.ndarray:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return np.array(tokens, dtype=np.int32)

        def detokenize_fn(tokens: np.ndarray) -> str:
            return tokenizer.decode(tokens.tolist(), skip_special_tokens=False)

        def get_special_tokens_fn() -> List[str]:
            return tokenizer.all_special_tokens

        return TokenizerInterface(tokenize_fn, detokenize_fn, get_special_tokens_fn)


    # Register both HuggingFace tokenizer types
    register_tokenizer_constructor(PreTrainedTokenizer, create_huggingface_interface)
    register_tokenizer_constructor(PreTrainedTokenizerFast, create_huggingface_interface)

except ImportError:
    # HuggingFace transformers not available
    pass