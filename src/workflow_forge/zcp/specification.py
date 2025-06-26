"""
Specifications are the actual directives that are intended to move
between the frontend and the backend, containing the capture directives,
szcp nodes, and such considerations. They can serialize themselves for
transport, compare themselves, and generally exist to transport program-level
data in addition to the zone level data.
"""

import json
import dataclasses
from dataclasses import dataclass
from typing import Tuple, List, Dict

from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.zcp.nodes import SZCPNode


@dataclass
class Specification:
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
    extractions: List[str]

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
                    config: Config,
                    json_str:str
                    )->"Specification":
        """
        Attempts to deserialize the given JSON string into the
        original specification. Also checks for compatibility
        issues here.

        :param config: The backend config, in all its glory
        :param json_str: The json payload.
        :return: The specification.
        """


