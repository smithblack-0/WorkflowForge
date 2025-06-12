# Zone Control Protocol

The Zone Control Protocol is the Intermediate Representation used to encode the flow control before it is converted to bytecode for the TTFA backend

## Formal Innovations.

A ZCP graph is governed by constructing more specialized  Control Flow Graph. I deem it a Directed Cyclic IO Graph (DCG-IO). This is a more restricted subset of the standard CFG which:

1) allows only one sink node, and like a CFG has only one source node.
2) Has a 'natural' representation that flattens directly to program counters and instructions, with occasional jump commands.

### Formal Definition

A Directed Cyclic IO Graph G = (V, E, s, t) is a directed graph where:

* V is a finite set of vertices
* E ⊆ V × V is a set of directed edges
* s ∈ V is a unique source vertex (in-degree 0)
* t ∈ V is a unique sink vertex (out-degree 0)
* Every vertex v ∈ V is reachable from s
* Every vertex v ∈ V can reach t

Key Properties:

* Single Entry/Exit: Exactly one source s and one sink t
* Connectivity: All vertices lie on some path from s to t
* Cycle Admissible: Unlike DAGs, cycles are permitted
* Workflow Complete: Every execution path begins at s and terminates at t

Distinguishing Characteristics:

* More restrictive than general directed graphs (single IO constraint)
* More permissive than DAGs (cycles allowed)
* Maintains computational tractability for workflow analysis
* Guarantees termination properties despite cycles

Computational Properties:

* Compilation to linear instruction sequences is decidable
* Reachability analysis remains polynomial
* Cycle detection and analysis feasible
* Suitable for deterministic execution models

## The ZCP node

ZCP nodes contain the following information. They are corrolated to a zone of text.

* Sequence: string. Used mainly while constructing
* Block: Int. Used mainly while constructing.
* Sampling callback. Will return a sampled bit of text for further processing and tokenization.
* Next Zone. The next zone in the chain.
* Zone Advance. When this token is observed, the next zone is moved to.
* Tags. The tags this zone possesses. 
* Timeout. How many tokens to generate before timing out and going to the next zone.
* Input (bool): Whether this is an input zone that should feed from the input buffer once out of prompt tokens, or whether it should freewheel. If fed, it will automatically include the next zone token when exhausted.
* Output (bool): Whether this is an output zone. Output zones are compiled into the capturing chain, and all content in the zone is captured.
* (Optional) Jump token. 
* (Optional) Jump node. Where to jump to on jump token triggering.

## The Lowered ZCP node (LZCP)

Lowered ZCP is one stage below ZCP, but it has had sampling done, and also been converted into the local tensor system.

* tokens: The token sequence to feed in.
* Next Zone. The next zone in the chain.
* Zone Advance. When this token is observed, the next zone is moved to. Now an int.
* Tags. The tags this zone possesses. Now bool array
* Timeout. How many tokens to generate before timing out and going to the next zone. Still an int.
* Input (bool): Whether this is an input zone that should feed from the input buffer once out of prompt tokens, or whether it should freewheel. If fed, it will automatically include the next zone token when exhausted.
* Output (bool): Whether this is an output zone. Output zones are compiled into the capturing chain, and all content in the zone is captured.
* (Optional) Jump token. Int.
* (Optional) Jump node. Where to jump to on jump token triggering.