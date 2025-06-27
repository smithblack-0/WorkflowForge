"""
Specifications are the actual directives that are intended to move
between the frontend and the backend, containing the capture directives,
szcp nodes, and such considerations. They can serialize themselves for
transport, compare themselves, and generally exist to transport program-level
data in addition to the zone level data.
"""

import json
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable

from .nodes import SZCPNode, LZCPNode
from .tag_converter import TagConverter
from ..parsing.config_parsing import Config
from ..tokenizer_interface import TokenizerInterface
@dataclass
class Workflow:
    """
    The main transport specification, designed to convey SZCP,
    tagging information, and other such details to the remote
    backend. It is also capable of deserializing or lowering itself.

    attributes:
    - config: The configuration file
    - nodes: The SZCP nodes that have been parsed
    - extractions: The list of tags to extract.
    """
    config: Config
    nodes: SZCPNode
    extractions: Dict[str, List[str]]

    def serialize(self)->str:
        """
        Serializes this dataclass into a JSON string
        for transport over the interwebs.
        :return: The serialized JSON string
        """
        stub = {
            "config" : dataclasses.asdict(self.config),
            "nodes" : self.nodes.serialize(),
            "extractions" : self.extractions,
        }
        return json.dumps(stub)

    @classmethod
    def deserialize(cls,
                    json_str:str
                    )-> "Workflow":
        """
        Attempts to deserialize the given JSON string into the
        original specification. While we could place config checks
        here, they really belong as an additional check in the
        backend.

        :param json_str: The json payload.
        :return: The specification.
        """
        stub = json.loads(json_str)
        config = Config(**stub["config"])
        nodes = SZCPNode.deserialize(stub["nodes"])
        extractions = stub["extractions"]

        return Workflow(config=config, nodes=nodes, extractions=extractions)

    def lower(self,
              tokenizer: TokenizerInterface,
              tools: Dict[str, Callable[[str],str]]
              )->'LoweredWorkflow':
        """
        Lowers the workflow down to tensors, though does
        not convert to a specific framework.
        :param tokenizer: The tokenizer system
        :param tools: The tools callback repository
        :return: The lowered workflow.
        """
        tag_converter = TagConverter(self.config.valid_tags)
        nodes = self.nodes.lower(tokenizer, tag_converter, tools)
        extractions = {key : tag_converter.tensorize(tags) for key, tags in self.extractions.items()}
        return LoweredWorkflow(
            tag_converter=tag_converter,
            tokenizer=tokenizer,
            nodes=nodes,
            extractions= extractions
        )

@dataclass
class LoweredWorkflow:
    """
    A fully lowered workflow,
    which has had tokenization done,
    the tag converter built, callbacks resolved,
    etc. The only thing left, really, is to
    convert it to tensors and feed it to
    the TTFA system,
    """
    tag_converter: TagConverter
    tokenizer: TokenizerInterface
    nodes: LZCPNode
    extractions: Dict[str, np.ndarray]
