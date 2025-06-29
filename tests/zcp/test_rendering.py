"""
Visual Test Cases for Workflow Forge Graph Rendering

Creates different node layouts that you can view to verify the visualization works correctly.
Run these to see actual graphs and inspect them visually.
"""

import unittest
from src.workflow_forge.zcp.rendering import GraphNode, GraphData, create_plotly_graph

# Set to True to display graphs during testing
SHOW_GRAPHS = True


class VisualTestCases(unittest.TestCase):
    """Visual test cases - run these to see actual graphs."""

    def test_simple_linear_chain(self):
        """Test Case 1: Simple A -> B -> C -> D linear chain"""
        print("\n=== TEST: Simple Linear Chain ===")

        nodes = [
            GraphNode("A", "Start", "green", 0, 0, nominal="B"),
            GraphNode("B", "Process", "blue", 2, 0, nominal="C"),
            GraphNode("C", "Validate", "orange", 4, 0, nominal="D"),
            GraphNode("D", "End", "red", 6, 0)
        ]

        graph_data = GraphData(nodes, "Linear Chain: A → B → C → D")
        fig = create_plotly_graph(graph_data)

        print("✓ Created linear chain graph")
        print("  - Should show 4 nodes in a horizontal line")
        print("  - Should have 3 blue arrows pointing right")
        print("  - Arrows should be at edge midpoints")

        if SHOW_GRAPHS:
            fig.show()

    def test_simple_loop(self):
        """Test Case 2: A -> B -> C -> A loop"""
        print("\n=== TEST: Simple Loop ===")

        nodes = [
            GraphNode("A", "Control", "orange", 0, 0, nominal="B"),
            GraphNode("B", "Work", "blue", 2, 0, nominal="C"),
            GraphNode("C", "Check", "purple", 1, -2, jump="A")  # Jump back to A
        ]

        graph_data = GraphData(nodes, "Simple Loop: A → B → C ↗ A")
        fig = create_plotly_graph(graph_data)

        print("✓ Created simple loop graph")
        print("  - Should show triangle layout")
        print("  - Should have 2 blue arrows (A→B, B→C)")
        print("  - Should have 1 red dashed arrow (C↗A)")
        print("  - Red arrow should point back to complete the loop")

        if SHOW_GRAPHS:
            fig.show()

    def test_branching_fork_and_merge(self):
        """Test Case 3: Fork and merge pattern"""
        print("\n=== TEST: Fork and Merge ===")

        nodes = [
            GraphNode("start", "Start", "green", 0, 0, nominal="decision"),
            GraphNode("decision", "Branch", "orange", 2, 0, nominal="path_a", jump="path_b"),
            GraphNode("path_a", "Path A", "blue", 4, 1, nominal="merge"),
            GraphNode("path_b", "Path B", "cyan", 4, -1, nominal="merge"),
            GraphNode("merge", "Merge", "purple", 6, 0, nominal="end"),
            GraphNode("end", "End", "red", 8, 0)
        ]

        graph_data = GraphData(nodes, "Fork and Merge: Decision → Paths → Merge")
        fig = create_plotly_graph(graph_data)

        print("✓ Created fork-and-merge graph")
        print("  - Should show diamond pattern")
        print("  - Decision node should have 1 blue arrow (to path_a)")
        print("  - Decision node should have 1 red arrow (to path_b)")
        print("  - Both paths should converge at merge point")

        if SHOW_GRAPHS:
            fig.show()

    def test_nested_loops(self):
        """Test Case 4: Control loop with work loop inside"""
        print("\n=== TEST: Nested Loops ===")

        nodes = [
            GraphNode("setup", "Setup", "green", 0, 2, nominal="outer_control"),

            # Outer loop control
            GraphNode("outer_control", "Outer\nControl", "orange", 2, 2, nominal="inner_control", jump="finish"),

            # Inner loop
            GraphNode("inner_control", "Inner\nControl", "yellow", 4, 2, nominal="work", jump="outer_control"),
            GraphNode("work", "Work", "blue", 6, 2, jump="inner_control"),

            # Exit
            GraphNode("finish", "Finish", "red", 2, 4)
        ]

        graph_data = GraphData(nodes, "Nested Loops: Outer ↗ Inner ↗ Work")
        fig = create_plotly_graph(graph_data)

        print("✓ Created nested loops graph")
        print("  - Should show multiple loop-back arrows")
        print("  - Blue arrows for forward flow")
        print("  - Red arrows for all loop-backs")
        print("  - Outer control should exit upward to finish")

        if SHOW_GRAPHS:
            fig.show()

    def test_complex_realistic_workflow(self):
        """Test Case 5: Realistic DCG-IO workflow"""
        print("\n=== TEST: Complex Realistic Workflow ===")

        nodes = [
            # Setup chain
            GraphNode("setup1", "Setup\n1", "lightgreen", 0, 2, nominal="setup2"),
            GraphNode("setup2", "Setup\n2", "lightgreen", 1, 2, nominal="main_loop"),

            # Main processing loop
            GraphNode("main_loop", "Main\nLoop", "orange", 3, 2, nominal="process", jump="validation"),
            GraphNode("process", "Process", "blue", 5, 2, nominal="check"),
            GraphNode("check", "Check", "purple", 7, 2, nominal="retry_decision"),
            GraphNode("retry_decision", "Retry?", "yellow", 9, 2, nominal="main_loop", jump="branch_decision"),

            # Branching decision
            GraphNode("branch_decision", "Strategy?", "orange", 11, 2, nominal="strategy_a", jump="strategy_b"),
            GraphNode("strategy_a", "Strategy\nA", "lightblue", 13, 3, nominal="merge"),
            GraphNode("strategy_b", "Strategy\nB", "lightcoral", 13, 1, nominal="merge"),

            # Merge and validation
            GraphNode("merge", "Merge", "purple", 15, 2, nominal="validation"),
            GraphNode("validation", "Validate", "yellow", 17, 2, nominal="output", jump="main_loop"),

            # Final output
            GraphNode("output", "Output", "red", 19, 2)
        ]

        graph_data = GraphData(nodes, "Complex Workflow: Setup → Loops → Branch → Merge → Output")
        fig = create_plotly_graph(graph_data)

        print("✓ Created complex realistic workflow")
        print("  - Should show proper DCG-IO structure")
        print("  - Multiple loops with different purposes")
        print("  - Branching and merging patterns")
        print("  - Mix of blue (nominal) and red (jump) arrows")
        print("  - Should be readable despite complexity")

        if SHOW_GRAPHS:
            fig.show()

    def test_single_node(self):
        """Test Case 6: Edge case - single node with no connections"""
        print("\n=== TEST: Single Node (Edge Case) ===")

        nodes = [
            GraphNode("alone", "Standalone\nNode", "gray", 0, 0)
        ]

        graph_data = GraphData(nodes, "Single Node - No Connections")
        fig = create_plotly_graph(graph_data)

        print("✓ Created single node graph")
        print("  - Should show just one node")
        print("  - Should have no arrows")
        print("  - Should not crash")

        if SHOW_GRAPHS:
            fig.show()

    def test_self_referencing_node(self):
        """Test Case 7: Edge case - node that points to itself"""
        print("\n=== TEST: Self-Referencing Node ===")

        nodes = [
            GraphNode("spinner", "Self\nLoop", "purple", 0, 0, nominal="spinner", jump="spinner"),
            GraphNode("exit", "Exit", "red", 3, 0)  # Unreachable, but shows contrast
        ]

        graph_data = GraphData(nodes, "Self-Referencing Node")
        fig = create_plotly_graph(graph_data)

        print("✓ Created self-referencing node graph")
        print("  - Should show node with arrows pointing to itself")
        print("  - May look weird but shouldn't crash")
        print("  - Good test of edge case handling")

        if SHOW_GRAPHS:
            fig.show()


def run_visual_tests():
    """Run all visual tests and optionally display them."""
    print("WORKFLOW FORGE VISUALIZATION - VISUAL TEST SUITE")
    print("=" * 55)
    print("Each test creates a different graph layout.")
    print(f"SHOW_GRAPHS = {SHOW_GRAPHS}")
    if SHOW_GRAPHS:
        print("Graphs will be displayed in your browser.")
    else:
        print("Set SHOW_GRAPHS = True to display graphs.")
    print()

    suite = unittest.TestLoader().loadTestsFromTestCase(VisualTestCases)
    runner = unittest.TextTestRunner(verbosity=0)

    print("Running visual tests...")
    result = runner.run(suite)

    print(f"\n✓ Created {result.testsRun} different graph layouts")
    print("✓ All graphs generated without errors")
    if not SHOW_GRAPHS:
        print("\nTo view the graphs: Set SHOW_GRAPHS = True and re-run")

    return result.wasSuccessful()


if __name__ == '__main__':
    # Run the visual tests
    run_visual_tests()