"""
Workflow Forge Graph Rendering Module

Provides interactive visualization of SZCP workflow graphs using Plotly.
Supports hover details, node coloring, proper z-ordering, and purple loopback edges.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    pass


@dataclass
class GraphNode:
    """Represents a single node in the visualization graph."""
    id: str  # Unique identifier
    name: str  # Short display name for the node
    color: str  # Node color (hex or named color)
    x: float
    y: float
    nominal: Optional[str] = None  # Node ID for nominal (next_zone) connection
    jump: Optional[str] = None  # Node ID for jump (jump_zone) connection
    node_data: Dict[str, Any] = None  # Pytree of primitive types (lists, bools, ints, etc.)

    def __post_init__(self):
        if self.node_data is None:
            self.node_data = {}


@dataclass
class GraphData:
    """Complete graph representation for visualization."""
    nodes: List[GraphNode]
    title: str = "Workflow Graph"


def _create_edge_trace(connections: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                      color: str, dash: str, name: str) -> go.Scatter:
    """Create a scatter trace for edge lines."""
    edge_x = []
    edge_y = []

    for start_pos, end_pos in connections:
        edge_x.extend([start_pos[0], end_pos[0], None])
        edge_y.extend([start_pos[1], end_pos[1], None])

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=2, color=color, dash=dash),
        hoverinfo='none',
        name=name,
        showlegend=True
    )


def _collect_connections(graph_data: GraphData) -> Tuple[
    List[Tuple[Tuple[float, float], Tuple[float, float]]],
    List[Tuple[Tuple[float, float], Tuple[float, float]]],
    List[Tuple[Tuple[float, float], Tuple[float, float]]]
]:
    """Collect nominal, jump, and loopback connections from graph data."""
    node_positions = {node.id: (node.x, node.y) for node in graph_data.nodes}

    nominal_connections = []
    jump_connections = []
    loopback_connections = []

    for node in graph_data.nodes:
        source_pos = node_positions[node.id]

        # Collect nominal connections
        if node.nominal and node.nominal in node_positions:
            target_pos = node_positions[node.nominal]

            # Check if this is a loopback (target x coordinate is lower than source)
            if target_pos[0] < source_pos[0]:
                loopback_connections.append((source_pos, target_pos))
            else:
                nominal_connections.append((source_pos, target_pos))

        # Collect jump connections
        if node.jump and node.jump in node_positions:
            target_pos = node_positions[node.jump]

            # Check if this is a loopback (target x coordinate is lower than source)
            if target_pos[0] < source_pos[0]:
                loopback_connections.append((source_pos, target_pos))
            else:
                jump_connections.append((source_pos, target_pos))

    return nominal_connections, jump_connections, loopback_connections


def _plot_edges(graph_data: GraphData) -> List[go.Scatter]:
    """Create edge traces with proper z-ordering (edges only, no arrows)."""
    nominal_connections, jump_connections, loopback_connections = _collect_connections(graph_data)

    traces = []

    # Create edge traces
    nominal_edge_trace = _create_edge_trace(nominal_connections, 'blue', 'solid', 'Nominal Flow')
    jump_edge_trace = _create_edge_trace(jump_connections, 'red', 'dash', 'Jump Flow')
    loopback_edge_trace = _create_edge_trace(loopback_connections, 'purple', 'dot', 'Loopback Flow')

    traces.extend([nominal_edge_trace, jump_edge_trace, loopback_edge_trace])

    return traces


def _plot_nodes(graph_data: GraphData) -> go.Scatter:
    """Create scatter trace for all nodes in the graph (top layer)."""
    x_coords = [node.x for node in graph_data.nodes]
    y_coords = [node.y for node in graph_data.nodes]
    colors = [node.color for node in graph_data.nodes]
    names = [node.name for node in graph_data.nodes]

    # Create hover text from node_data
    hover_texts = []
    for node in graph_data.nodes:
        hover_lines = [f"<b>{node.name}</b>", f"ID: {node.id}"]

        # Add connections info
        if node.nominal:
            hover_lines.append(f"Nominal → {node.nominal}")
        if node.jump:
            hover_lines.append(f"Jump → {node.jump}")

        # Add node_data information
        for key, value in node.node_data.items():
            hover_lines.append(f"{key}: {value}")

        hover_texts.append("<br>".join(hover_lines))

    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(
            size=25,
            color=colors,
            line=dict(width=2, color='black')
        ),
        text=names,
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        hovertext=hover_texts,
        hoverinfo='text',
        name='Nodes'
    )


def create_plotly_graph(graph_data: GraphData) -> go.Figure:
    """
    Create interactive Plotly graph from GraphData with proper z-ordering.

    Z-order (bottom to top):
    1. Edge lines (blue=nominal, red=jump, purple=loopback)
    2. Nodes

    Args:
        graph_data: Complete graph data with nodes and connections

    Returns:
        Plotly Figure ready for display
    """
    # Create traces in proper z-order
    edge_traces = _plot_edges(graph_data)  # Bottom layer
    node_trace = _plot_nodes(graph_data)  # Top layer

    # Combine traces in correct order (edges first, nodes last)
    all_traces = edge_traces + [node_trace]

    # Create figure
    fig = go.Figure(data=all_traces)

    # Configure layout
    fig.update_layout(
        title=graph_data.title,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="This is interactive! Hover over a node for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color="gray")
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    return fig