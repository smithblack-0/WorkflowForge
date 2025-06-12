# Workflow Forge

**AI models that control their own prompt feeds, and the tools to interpret this**

Workflow Forge intercepts and replaces tokens in the generation stream, allowing models to execute complex workflows by generating control tokens that instantly redirect their own execution path - all without ever leaving the GPU.

## The Core Innovation: Token Stream Interception

Instead of calling the model multiple times with different prompts, Workflow Forge lets the model **control its own execution flow** during a single generation:

```
Model generates: "I need to think more... [Jump]"
                                         ↑
Workflow Forge intercepts this token ────┘
                 ↓
Instantly replaces stream with: "Let me reconsider the problem..."
                 ↓  
Model continues generating from new context
```

**This happens entirely in tensor operations** - no Python roundtrips, no generation restarts, no external orchestration. The model literally loads precompiled and tokenized prompts by generating special control tokens, then reads through them and generates its own responses. This happens in parallel, during evaluation, across batches, and with flow control. It is designed for evaluation, and is particularly suitable for mass sampling of a particular workflow, as in synthetic training data generation. And once deployed using it is as simple as passing the token stream through a function.

**What this enables:**
- **Self-Modifying Workflows**: Models dynamically switch between reasoning strategies mid-generation by declaring its preferred flow prompt or context.
- **Autonomous Decision Trees**: Each model instance follows different paths based on what it generates. The same workflow is applied, but the choices made can differ. This is Single Workflow Multiple Streams (SWMS), a close cousin of SIMD.
- **Massive Parallelism**: hundreds of independent decision-making streams per batch. Great for mass sampling.
- **Zero-Latency Flow Control**: Instant transitions between workflow states.
- 
### The Architecture

Workflow Forge uses a 4-stage compilation pipeline inspired by traditional compilers:

```
UDPL (Config) → SFCS (Programming) → ZCP (Bytecode) → TTFA (Execution)
     ↓               ↓                  ↓               ↓
TOML Files → Python Flow Control → IR Graph → GPU Tensors
```

Each stage is modular and extensible - you can write your own frontend languages or target different backends.

## How It Works

### 1. Declarative Configuration (UDPL)
Define your prompts and data extraction rules in human-readable TOML:

```toml
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
sequences = ["control", "reasoning", "conclusion"]
valid_tags = ["Training", "Correct", "Feedback"]
control_token = "[Jump]"
escape_token = "[Escape]"



[[control]]
text = """
[Prompt] If satisfied with your answer, emit [Escape] "[Jump]".
Otherwise continue thinking. If this is the first time you have seen this,
say nothing.
[Answer]
"""
tags = [[], []]


[[reasoning]]
text = """
[Prompt]
Answer the following question, thinking it
through from a first person perspective {user_question}
[Answer]"""
[reasoning.user_question]
type="control"
name="question"


[[conclusion]]
text = "[Prompt] State your final conclusion. Only tokens after the answer count. [Answer]"
tags = [[], ["answer"]]
```

### 2. Flow Control Programming (SFCS)
Write Python-like code that compiles to GPU execution:

```python
# Parse configuration
sequences, config = forge.parse_udpl_file("workflow.toml")

# Program the flow control
program = forge.new_program(sequences, resources, config, tokenizer)
with program.loop("control") as loop:
    loop.run("reasoning",  question="Are you alive?")  # Model decides when to break via [Jump] token
program.run("conclusion")
program.extract("training_data", tags=["answer"])

# Compile to autonomous agents
controller_factory = program.compile()
```

### 3. Autonomous Execution
Deploy hundreds of independent agents:

```python
# Each batch creates autonomous agents
training_data = []
for batch in range(1000):
    
    # Note that this sets up the FSM backend. No further dynamic injection is possible. 
    flow_manager = controller_factory(batch_size=32, starting_token="[BOS]")
    
    # Agent runs autonomously on GPU
    while not flow_manager.done():
        tokens = flow_manager.next()      # Get next tokens to feed
        tokens = model.generate(tokens)   # Model generates response
        flow_manager.advance(tokens)      # Agent processes output
    
    # Extract results
    results = flow_manager.extract()
    training_data.extend(results["training_data"])
```

**Key insight**: The model's own token outputs (like `[Jump]`) control the agent's execution flow. No Python logic runs during generation.

## Use Cases

**Synthetic Data Generation**: Generate thousands of reasoning chains with different complexity levels and extract training pairs automatically. Draw all samples from a configuration in one pass.

**Constitutional AI**: Self-improving systems where agents evaluate their own outputs and provide feedback for future iterations.

**Adaptive Evaluation**: Testing pipelines that adjust difficulty based on model performance, with automatic data collection. 

## Core Concepts

- **Zones**: Regions of text between special tokens (like `[Prompt]` to `[Answer]`)
- **Tags**: Metadata for selective extraction of generated content. We can configure to extract and concatenate all tokens with these tags painted on them.
- **Sequences**: Named chains of prompts that can be combined with flow control
- **Resources**: Dynamic or static content injection (placeholders, sampling, feedback loops)
- **Flow Control**: Model-commanded transitions using special tokens. The model commands the FSM backend, and never leaves python.

## Documentation

- 📚 [Complete Documentation](docs/) - Architecture and API reference
- 📝 [UDPL Specification](docs/UDPL.md) - Prompting language reference  
- 🔄 [SFCS Guide](docs/SFCS.md) - Flow control programming
- ⚙️ [ZCP Reference](docs/ZCP.md) - Intermediate representation details
- 🚀 [TTFA Architecture](docs/Autonoma/) - GPU execution engine

## Installation

Pending.

## Status

🚧 **Early Development**
- ✅ Architecting and scoping done.
- ✅ UDPL parsing pipeline complete
- 🚧 SFCS flow control system in development  
- 🚧 TTFA execution engine in development
- 🚧 **Tools** coming up, but will be blocking. 

This is a production-grade tool designed for advanced AI workflow automation. The compilation approach enables capabilities impossible with traditional prompt orchestration.

## Philosophy

Current prompting tools force you to choose between simple chains or complex orchestration. Workflow Forge applies compiler design principles to AI workflows - write high-level logic once, compile to efficient execution, run autonomously at scale.

The goal is **C++-level power with configuration-level simplicity** - sophisticated control flow that compiles down to fast, autonomous execution.

## Contributing

Workflow Forge is designed with the capability of becoming a community standard for AI workflow automation:

- 🐛 [Report Issues](issues/new) 
- 💡 [Feature Requests](issues/new)
- 🔧 [Pull Requests](CONTRIBUTING.md)
- 📖 [Documentation](docs/)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*For researchers who need complex workflows and speed*
