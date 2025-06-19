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

## ZCP Node Hierarchy

### High-Level ZCP Nodes
ZCP nodes represent zones of text with unresolved string references:

* **Sequence/Block**: Construction metadata for provenance tracking
* **Resource specs**: Placeholder → resource mapping (unresolved)
* **Raw text**: Template with placeholders like `{resource_name}`
* **Zone advance str**: String that triggers advancement (e.g., "[Answer]")
* **Tags**: List of string tags for selective extraction
* **Timeout**: Maximum tokens before forced advancement
* **Next zone**: Simple linked list pointer (no flow control)

### Resolved ZCP (RZCP) Nodes  
RZCP represents the stage with resolved flow control and construction callbacks:

* **Construction callback**: Function that returns tokenized prompt when called
* **Integer tokens**: Zone advance and jump tokens as token IDs NumPy array
* **Graph connectivity**: Full next_zone and jump_zone references resolved
* **Input/Output flags**: Control data flow behavior
* **Tool callbacks**: Optional tool integration functions
* **Tags**: Boolean arrays for tag membership

### Lowered ZCP (LZCP) Nodes

LZCP is the final stage ready for TTFA compilation, with full sampling resolution:

* **Tokens**: Actual token sequence as numpy array (from construction callback)
* **Zone advance tokens**: Token triggering np.array  
* **Jump tokens**: Integer token IDs as np.array
* **Tags**: Boolean array for tag membership
* **Timeout**: Integer token limit
* **Input/Output flags**: Boolean values
* **Graph connectivity**: Resolved to LZCP node references
* **Tool callbacks**: Resolved tool integration

## Compilation Pipeline

The compilation process follows this transformation:

```
[front end..] [IR]  [backend...........]
UDPL → ZCP →  RZCP → LZCP → TTFA Bytecode
```

1. **UDPL parsing** produces ZCP nodes with string templates
2. **SFCS flow control** converts ZCP to RZCP with flow control graph and construction callbacks
3. **Batch sampling** converts RZCP to LZCP by calling construction callbacks to get actual tokens.
4. **Graph flattening** converts LZCP to TTFA instruction sequences

## Implementation Notes

**Edge-Centric Compilation**: The GraphBuilderNode class treats graph construction as edge management rather than vertex manipulation, enabling clean forward reference resolution without global state.

**Mathematical Constraints**: The DCG-IO constraints ensure that every compiled workflow has guaranteed termination properties and predictable execution flow, making the TTFA backend implementation tractable.

**Lowering Pipeline**: Each lowering stage resolves one type of abstraction - ZCP→RZCP resolves flow control and creates callbacks, RZCP→LZCP resolves sampling by executing callbacks.

**Cycle Handling**: All lowering methods include cycle detection via `lowered_map` parameters to handle complex flow control graphs with loops correctly.