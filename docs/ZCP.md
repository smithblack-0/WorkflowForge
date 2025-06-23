# Zone Control Protocol

The Zone Control Protocol (ZCP) is the Intermediate Representation used to encode flow control graphs before compilation to bytecode for the TTFA backend. ZCP represents AI workflows as mathematically well-defined directed graphs with specific properties that enable efficient compilation and execution.

## Graph Theory Foundation

### Directed Cyclic IO Graphs (DCG-IO)

A ZCP graph is formally defined as a Directed Cyclic IO Graph G = (V, E, s, t), which is a specialized Control Flow Graph with the following constraints:

**Formal Definition:**
* V is a finite set of vertices (ZCP nodes)
* E ⊆ V × V is a set of directed edges  
* s ∈ V is a unique source vertex (in-degree 0)
* t ∈ V is a unique sink vertex (out-degree 0)
* Every vertex v ∈ V is reachable from s
* Every vertex v ∈ V can reach t

**Key Properties:**
* **Single Entry/Exit**: Exactly one source s and one sink t
* **Connectivity**: All vertices lie on some path from s to t
* **Cycle Admissible**: Unlike DAGs, cycles are permitted for loops
* **Workflow Complete**: Every execution path begins at s and terminates at t

**Computational Advantages:**
* Compilation to linear instruction sequences is decidable
* Reachability analysis remains polynomial time
* Cycle detection and analysis feasible
* Suitable for deterministic execution models

### Edge-Colored Graph Model

ZCP uses an **edge-colored graph model** where each vertex has exactly two **emission sites**:

1. **Nominal emission site** (next_zone) - for sequential flow
2. **Control emission site** (jump_zone) - for conditional flow control

**Edge Properties:**

Each edge has a **color** determined by its emission site on the source vertex. Any number of edges (of either color) may terminate at a single target vertex, however the color is very important for the SFCS flow control programming process. **Forward jumps are not currently supported** - only backward jumps for loops. However, forward references to just the next node are supported; the color then determines what kind of emission site we launched from and thus how to link the graph nodes.

**Forward Reference Resolution:**

During compilation, we encounter **dangling edges** - edges with a source vertex but no target vertex yet. The GraphBuilderNode class implements edge-centric forward reference resolution, treating compilation as incremental edge connection rather than vertex manipulation. They are both a closure of edges we wish to point at the upcoming node, and later on a capture of that node itself.

## ZCP Node Pipeline Overview

The ZCP intermediate representation consists of four sequential stages, each resolving different types of abstractions:

```
ZCP → RZCP → SZCP → LZCP
```

**Frontend/Backend Boundary:**
The pipeline splits between frontend (client-side) and backend (server-side) execution:

* **Frontend**: ZCP → RZCP → SZCP (with serialization)
* **Backend**: SZCP (deserialization) → LZCP → TTFA Bytecode

**Stage Responsibilities:**
1. **ZCP → RZCP**: Resolves flow control and creates sampling callbacks
2. **RZCP → SZCP**: Executes sampling callbacks to resolve all placeholders to final text
3. **SZCP → LZCP**: Tokenizes content and resolves tool callbacks for execution

SZCP serves as the **serialization boundary**, enabling "compile locally, execute remotely" workflows by containing fully resolved text content that can be transmitted over networks.

## Node Specifications

### General Node Properties

All ZCP nodes share fundamental characteristics that support the DCG-IO graph model:

* **Graph Structure**: Each node supports the edge-colored graph model with nominal emission (next_zone) and control emission (jump_zone) sites
* **Provenance Tracking**: Sequence and block identifiers for error reporting and debugging
* **Zone Advancement**: Trigger strings or tokens that cause transitions between zones
* **Tag System**: Metadata for selective extraction of generated content
* **Timeout Protection**: Maximum token limits to prevent infinite generation
* **Lowering Pipeline**: Each node type can transform to the next stage in the compilation process

### ZCP Nodes

**Purpose**: Template representation with unresolved resource placeholders.

ZCP nodes are the direct output of UDPL parsing, containing string templates with placeholder syntax. They represent the user's intent before any resolution has occurred.

**Key Characteristics:**
* **Unresolved placeholders**: Contains `{resource_name}` template syntax
* **Resource specifications**: Mappings from placeholders to resource definitions
* **Construction callbacks**: Functions that can resolve templates when given actual resources
* **Simple linking**: Only next_zone pointers, no flow control yet

### RZCP Nodes  

**Purpose**: Flow control resolution with executable sampling callbacks.

RZCP nodes represent the stage where flow control graphs have been constructed and resource resolution mechanisms are in place, but actual sampling has not yet occurred.

**Key Characteristics:**
* **Flow control resolution**: Full next_zone and jump_zone graph connectivity
* **Sampling callbacks**: Functions that return resolved text strings (not tokens)
* **Input/output flags**: Control data flow behavior for extraction
* **Tool integration**: Tool names for serializable tool references

### SZCP Nodes

**Purpose**: Serialization boundary with fully resolved text content.

SZCP nodes mark the frontend/backend boundary. All resource placeholders have been resolved to final text, making the nodes ready for network transmission.

**Key Characteristics:**
* **Resolved text content**: All placeholders replaced with actual text
* **Serialization support**: Can be converted to/from network-transmissible format
* **Tool references**: Tool names (not callbacks) for backend resolution
* **Complete graph preservation**: Maintains all DCG-IO properties across serialization

### LZCP Nodes

**Purpose**: Execution-ready representation with tokenized content.

LZCP nodes are the final stage before TTFA compilation, with all text converted to token arrays and all tools resolved to executable callbacks.

**Key Characteristics:**
* **Tokenized content**: All text converted to numpy token arrays
* **Resolved tools**: Tool names converted to executable callback functions
* **Boolean tag arrays**: Tag membership as boolean vectors for efficient processing
* **Execution validation**: Extensive consistency checking for backend execution

## Implementation Notes

**Edge-Centric Compilation**: The GraphBuilderNode class treats graph construction as edge management rather than vertex manipulation, enabling clean forward reference resolution without global state.

**Mathematical Constraints**: The DCG-IO constraints ensure that every compiled workflow has guaranteed termination properties and predictable execution flow, making the TTFA backend implementation tractable.

**Lowering Pipeline**: Each lowering stage resolves one type of abstraction:
* ZCP→RZCP resolves flow control and creates callbacks
* RZCP→SZCP resolves sampling by executing callbacks to get final text
* SZCP→LZCP resolves tokenization and tool callbacks for execution

**Cycle Handling**: All lowering methods include cycle detection via `lowered_map` parameters to handle complex flow control graphs with loops correctly.

**Frontend/Backend Split**: SZCP serves as the clean boundary between client compilation and server execution, enabling distributed workflows while maintaining mathematical guarantees.