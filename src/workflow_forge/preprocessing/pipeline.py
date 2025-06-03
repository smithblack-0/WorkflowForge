"""
The pipeline unit is responsible for primary pipeline construction, standardization,
and defaults deployment.
"""
import os
import parsing
import torch
from prompt_logic import PromptSequenceFactory
from resources import AbstractResource
from typing import Dict, Optional, Callable, Tuple, Any
from src.CE.SUPS.preprocessing.tags import TagConverter

# tokenization framework support.
try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def standardize_tokenizer(tokenizer: Any)->Callable[[str], torch.Tensor]:
    """
    Converts a tokenizer into a format we expect.
    Supported architectures are, at the moment
    - Huggingface

    This is deliberately designed to be extended by framework
    maintainers.

    :param tokenizer: The tokenizer to standardize.
    :return: The callback that applies the tokenizer in code context.
    """
    if HUGGINGFACE_AVAILABLE:
        # Handle the two huggingface varieties
        if isinstance(tokenizer, PreTrainedTokenizer):
            def huggingface_tokenizer_wrapper(text: str)->torch.Tensor:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens = torch.tensor(tokens)
                return tokens
            return huggingface_tokenizer_wrapper
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            def huggingface_tokenizer_wrapper(text: str)->torch.Tensor:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens = torch.tensor(tokens)
                return tokens
            return huggingface_tokenizer_wrapper
    raise NotImplementedError("No matching standard")

def deploy_defaults(folder_path: str,
                    force_overwrite: bool = False):
    """
    Deploy default constitutions and prompting toml
    into a folder named folder_name
    :param folder_path: The name of the folder to dump into
    """
    # Needs to get the default collection from
    # CE.preprocessing.defaults and move it to under that
    # folder.


def construct_pipeline(tokenizer: Any,
                       folder_path: str,
                       extra_resources: Optional[Dict[str, AbstractResource]] = None,
                       raw_tokenizer: bool = False
                       )->Tuple[PromptSequenceFactory, TagConverter]:
    """
    Constructs a complete generative pipeline from the config folder
    :param folder_path: The way to the config folder
    :param tokenizer: The tokenizer in use.
    :param extra_resources: Any extra resources to resolve with
    :param raw_tokenizer: If true, tokenizer is a raw  Callable[[str], torch.Tensor] closure. Tokenizers
           can always be made compatible with the framework this way. Otherwise, the passed tokenizer
           must be a supported framework standard and will be automatically converted into this
           closure type.
    :return:
    - The prompt sequence factory
    - The loaded tag converter.
    """

    if extra_resources is None:
        extra_resources = {}
    if not raw_tokenizer:
        tokenizer = standardize_tokenizer(tokenizer)

    # Parse the prompting config
    prompting_path = os.path.join(folder_path, "prompting.toml")
    zone_stubs, tag_converter = parsing.parse_prompts_file(prompting_path)

    # Parse the resources
    constitution_path = os.path.join(folder_path, "constitutions/")
    resources = parsing.parse_constitutions(constitution_path)
    resources.update(extra_resources)

    # Construct the factory
    return PromptSequenceFactory(zone_stubs, tokenizer, resources), tag_converter

