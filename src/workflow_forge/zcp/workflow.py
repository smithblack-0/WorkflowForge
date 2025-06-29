"""
Specifications are the actual directives that are intended to move
between the frontend and the backend, containing the capture directives,
szcp nodes, and such considerations. They can serialize themselves for
transport, compare themselves, and generally exist to transport program-level
data in addition to the zone level data.
"""

import msgpack
import base64
import dataclasses
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Type, Optional

from .nodes import SZCPNode, LZCPNode
from .tag_converter import TagConverter
from ..parsing.config_parsing import Config
from ..tokenizer_interface import TokenizerInterface

# Let me tell you I have been developing awhile without actually
# telling you: I HATE strong coupling.
#
# This is a fairly common trick of mine - I put
# classes in a factory so that I can easily swap it out
# for unit testing with mocking.

@dataclass
class WFFactories:
    """Construction factories for the various class dependencies"""
    tag_converter: Type[TagConverter]
    SZCP_node: Type[SZCPNode]
    Config: Type[Config]

def make_default_factories()->WFFactories:
    return WFFactories(TagConverter,
                       SZCPNode,
                       Config,
                       )
default_factories = make_default_factories()


@dataclass
class Workflow:
    """
    The main transport specification, designed to convey SZCP,
    tagging information, and other such details to the remote
    backend. It is also capable of deserializing or lowering itself
    to the final tensor set.

    attributes:
    - config: The configuration file
    - nodes: The SZCP nodes that have been parsed
    - extractions: The list of tags to extract.
    """
    config: Config
    nodes: SZCPNode
    extractions: Dict[str, List[str]]
    factories: WFFactories = default_factories

    def serialize(self)->str:
        """
        Serializes this dataclass into a JSON string
        for transport over the interwebs.
        :return: The serialized JSON string
        """
        stub = {
            "config" : self.config.serialize(),
            "nodes" : self.nodes.serialize(),
            "extractions" : self.extractions,
        }
        msg = msgpack.packb(stub)
        return base64.b64encode(msg).decode('utf-8')

    @classmethod
    def deserialize(cls,
                    msg_str:str,
                    factories: WFFactories = default_factories
                    )-> "Workflow":
        """
        Attempts to deserialize the given JSON string into the
        original specification. While we could place config checks
        here, they really belong as an additional check in the
        backend.

        :param msg_str: The payload.
        :return: The specification.
        """
        msg = base64.b64decode(msg_str.encode('utf-8'))
        stub = msgpack.unpackb(msg, strict_map_key=False)
        config = factories.Config.deserialize(stub["config"])
        nodes = factories.SZCP_node.deserialize(stub["nodes"])
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
        tag_converter = self.factories.tag_converter(self.config.valid_tags)
        nodes = self.nodes.lower(tokenizer, tag_converter, tools)
        extractions = {key : tag_converter.tensorize(tags) for key, tags in self.extractions.items()}
        return LoweredWorkflow(
            tag_converter=tag_converter,
            tokenizer=tokenizer,
            nodes=nodes,
            extractions= extractions
        )

    def visualize(self, file_name: Optional[str]=None)->None:
        """
        Produces a visualization of the workflow
        from the underlying SZCP nodes. If a filename is
        provided, it shows up there; else it shows in
        the console or cell/
        :param file_name: Location to save workflow html viewer at
        """
        self.nodes.visualize(save_to_file=file_name)

@dataclass
class LoweredWorkflow:
    """
    A fully lowered workflow,
    which has had tokenization done,
    the tag converter built, callbacks resolved,
    etc. The only thing left, really, is to
    convert it to tensors and feed it to
    the TTFA system.
    """
    tag_converter: TagConverter
    tokenizer: TokenizerInterface
    nodes: LZCPNode
    extractions: Dict[str, np.ndarray]
