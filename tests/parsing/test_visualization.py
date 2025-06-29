"""
Test script to create SZCP nodes matching the demo graph structure
and test the visualization system.

This creates a realistic DCG-IO workflow similar to the demo but using
actual SZCP node objects, then calls visualize() to test the layout.
"""


def create_test_szcp_workflow():
    """
    Create a test SZCP workflow that matches the demo graph structure:
    - Linear setup chain (8 nodes)
    - Control decision point with loop
    - Loop work nodes (12 nodes)
    - Conditional branch (strategy A vs B)
    - Merge point
    - Validation loop
    - Final output chain
    """
    from src.workflow_forge.zcp.nodes import SZCPNode

    nodes = {}  # Will store all nodes by their logical name

    # 1. LINEAR SETUP CHAIN (source of DCG-IO)
    setup_nodes = []
    for i in range(8):
        node = SZCPNode(
            sequence="setup",
            block=i,
            text=f"Setup step {i}: Initialize system components and prepare workflow execution.",
            zone_advance_str="[EOS]",
            escape_strs=("[Escape]", "[EndEscape]"),
            tags=["Training"],
            timeout=1000,
            input=False,
            output=False
        )
        setup_nodes.append(node)
        nodes[f"setup_{i}"] = node

        # Link setup chain
        if i > 0:
            setup_nodes[i - 1].next_zone = node

    # 2. CONTROL DECISION POINT
    control_node = SZCPNode(
        sequence="control",
        block=0,
        text="Control: Check if should continue loop or exit to next phase.",
        zone_advance_str="[EOS]",
        escape_strs=("[Escape]", "[EndEscape]"),
        tags=["Decision"],
        timeout=1000,
        input=False,
        output=False
    )
    nodes["control"] = control_node
    setup_nodes[-1].next_zone = control_node

    # 3. LOOP BODY (work that gets repeated)
    loop_work_nodes = []
    for i in range(12):
        node = SZCPNode(
            sequence="loop_work",
            block=i,
            text=f"Loop iteration work step {i}: Process data and perform iterative calculations.",
            zone_advance_str="[EOS]",
            escape_strs=("[Escape]", "[EndEscape]"),
            tags=["Training", "Iteration"],
            timeout=1000,
            input=False,
            output=False
        )
        loop_work_nodes.append(node)
        nodes[f"loop_work_{i}"] = node

        # Link loop work chain
        if i > 0:
            loop_work_nodes[i - 1].next_zone = node

    # Connect control to loop body (enter loop)
    control_node.next_zone = loop_work_nodes[0]

    # Loop back: last work node goes back to control (the actual loop)
    loop_work_nodes[-1].jump_zone = control_node
    loop_work_nodes[-1].jump_advance_str = "[Jump]"

    # 4. CONDITIONAL BRANCH (if/else pattern)
    condition_node = SZCPNode(
        sequence="condition",
        block=0,
        text="Condition: Choose strategy A or B based on analysis results.",
        zone_advance_str="[EOS]",
        escape_strs=("[Escape]", "[EndEscape]"),
        tags=["Decision"],
        timeout=1000,
        input=False,
        output=False
    )
    nodes["condition"] = condition_node

    # Control can jump to condition (exit loop)
    control_node.jump_zone = condition_node
    control_node.jump_advance_str = "[Jump]"

    # Branch A (strategy 1)
    branch_a_nodes = []
    for i in range(6):
        node = SZCPNode(
            sequence="strategy_a",
            block=i,
            text=f"Strategy A step {i}: Conservative approach with careful validation.",
            zone_advance_str="[EOS]",
            escape_strs=("[Escape]", "[EndEscape]"),
            tags=["Correct"],
            timeout=1000,
            input=False,
            output=False
        )
        branch_a_nodes.append(node)
        nodes[f"strategy_a_{i}"] = node

        if i > 0:
            branch_a_nodes[i - 1].next_zone = node

    # Branch B (strategy 2)
    branch_b_nodes = []
    for i in range(8):
        node = SZCPNode(
            sequence="strategy_b",
            block=i,
            text=f"Strategy B step {i}: Aggressive approach with rapid iteration.",
            zone_advance_str="[EOS]",
            escape_strs=("[Escape]", "[EndEscape]"),
            tags=["Feedback"],
            timeout=1000,
            input=False,
            output=False
        )
        branch_b_nodes.append(node)
        nodes[f"strategy_b_{i}"] = node

        if i > 0:
            branch_b_nodes[i - 1].next_zone = node

    # Condition branches to A or B
    condition_node.next_zone = branch_a_nodes[0]  # Nominal flow to strategy A
    condition_node.jump_zone = branch_b_nodes[0]  # Jump to strategy B
    condition_node.jump_advance_str = "[Jump]"

    # 5. MERGE POINT (branches reconverge)
    merge_node = SZCPNode(
        sequence="merge",
        block=0,
        text="Merge: Combine results from chosen strategy and prepare for validation.",
        zone_advance_str="[EOS]",
        escape_strs=("[Escape]", "[EndEscape]"),
        tags=["Training"],
        timeout=1000,
        input=False,
        output=False
    )
    nodes["merge"] = merge_node

    # Both branches flow to merge
    branch_a_nodes[-1].next_zone = merge_node
    branch_b_nodes[-1].next_zone = merge_node

    # 6. VALIDATION LOOP
    validation_control = SZCPNode(
        sequence="validation_control",
        block=0,
        text="Validation: Check if results meet quality standards.",
        zone_advance_str="[EOS]",
        escape_strs=("[Escape]", "[EndEscape]"),
        tags=["Decision"],
        timeout=1000,
        input=False,
        output=False
    )
    nodes["validation_control"] = validation_control
    merge_node.next_zone = validation_control

    # Validation work nodes
    validation_nodes = []
    for i in range(5):
        node = SZCPNode(
            sequence="validation",
            block=i,
            text=f"Validation step {i}: Verify quality and correctness of results.",
            zone_advance_str="[EOS]",
            escape_strs=("[Escape]", "[EndEscape]"),
            tags=["Feedback"],
            timeout=1000,
            input=False,
            output=False
        )
        validation_nodes.append(node)
        nodes[f"validation_{i}"] = node

        if i > 0:
            validation_nodes[i - 1].next_zone = node

    # Validation loop connections
    validation_control.next_zone = validation_nodes[0]
    validation_nodes[-1].jump_zone = validation_control  # Loop back
    validation_nodes[-1].jump_advance_str = "[Jump]"

    # 7. FINAL OUTPUT CHAIN (sink of DCG-IO)
    output_nodes = []
    for i in range(6):
        node = SZCPNode(
            sequence="output",
            block=i,
            text=f"Output step {i}: Generate final result and prepare for delivery.",
            zone_advance_str="[EOS]",
            escape_strs=("[Escape]", "[EndEscape]"),
            tags=["Correct"],
            timeout=1000,
            input=False,
            output=False
        )
        output_nodes.append(node)
        nodes[f"output_{i}"] = node

        if i > 0:
            output_nodes[i - 1].next_zone = node

    # Validation exits to output
    validation_control.jump_zone = output_nodes[0]
    validation_control.jump_advance_str = "[Jump]"

    print(f"Created SZCP workflow with {len(nodes)} nodes")
    print("Structure:")
    print("  - Setup chain: 8 nodes")
    print("  - Control + Loop work: 1 + 12 nodes")
    print("  - Condition + Strategy A: 1 + 6 nodes")
    print("  - Strategy B: 8 nodes")
    print("  - Merge + Validation: 1 + 1 + 5 nodes")
    print("  - Output chain: 6 nodes")

    # Return the root node (first setup node)
    return setup_nodes[0]


def test_szcp_visualization():
    """Test the SZCP visualization system with a realistic workflow."""
    print("Creating test SZCP workflow...")
    root_node = create_test_szcp_workflow()

    print("\nTesting visualization...")
    try:
        # Test live display
        print("Showing live visualization...")
        root_node.visualize()

        # Test file saving
        print("Saving to file...")
        root_node.visualize("test_workflow_visualization")

        print("✓ Visualization test completed successfully!")

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        raise


if __name__ == "__main__":
    test_szcp_visualization()