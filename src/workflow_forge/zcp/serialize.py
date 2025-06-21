"""
ZCP Boundary Serialization Functions

Functions to serialize/deserialize Config + SZCP for client/server boundary.
"""

import json
from dataclasses import asdict
from typing import Tuple

from src.workflow_forge.parsing.config_parsing import Config
from src.workflow_forge.zcp.nodes import SZCPNode


def serialize(config: Config, szcp: SZCPNode) -> str:
    """
    Serialize Config + SZCP to JSON string for HTTP transport.

    Args:
        config: The UDPL configuration
        szcp: The serializable ZCP graph head node

    Returns:
        JSON string ready for HTTP POST
    """
    payload = {
        "config": asdict(config),
        "szcp_graph": szcp.serialize()
    }

    return json.dumps(payload)


def deserialize(json_str: str) -> Tuple[Config, SZCPNode]:
    """
    Deserialize JSON string back to Config + SZCP.

    Args:
        json_str: JSON string from HTTP request

    Returns:
        Tuple of (config, szcp_head_node)
    """
    payload = json.loads(json_str)

    # Reconstruct Config from dict
    config = Config(**payload["config"])

    # Reconstruct SZCP graph from serialized data
    szcp = SZCPNode.deserialize(payload["szcp_graph"])

    return config, szcp