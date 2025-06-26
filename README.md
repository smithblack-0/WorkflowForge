# Workflow Forge

Workflow forge is a project to allow compiled, remotely executed AI workflows, suitable for batch processing, mass sampling, flow control, and, if exceptionally successful, possibly even real-time processing.

## Project Status

**Pre-Alpha**

Not yet operation, but prove of concepts for the difficult issues have been executed. This is not a pet project, but intended as serious, foundational research and production infrastructure and the code base, tests, documentation, and interface is held to this standard. 

## Project reason

Prompt engineering with most current prompt libraries is 
somewhat like being forced to choose between BASIC or 
assembly: You can choose to do simple tasks easily,
or more complex tasks with a lot of work, but you cannot
do both.Simple single purpose libraries can get common jobs done
quite easily, but are not very powerful on variations. Meanwhile, the lower level libraries such as Microsoft's 'Guidance' are powerful but require excessive manual and often brittle loading of code resources and segments, making pivots during research or tuning unnecessarily difficult. 

This is unnecessary. C++ is a very powerful language that nonetheless can be compiled down to something small and very fast; this is the approach taken here. UDPL allows you to specify chains of prompts to feed with 'tagging' for automatic extraction of texts later, and SFCS is a simple flow control system that captures the flow control graphs and their requirements in a pythonic manner that can then be lowered into
ZCP. Every effort has been taken to ensure all portions of the pipeline are human readable and easy to grok at a glance - for instance, activating flow control means making an indented flow region in python like normal if you are following standard workflow forge linting protocols.

## How It Works

To make a working WF project, you setup the frontend, the backend, and move information
between them. A forge project needs a frontend and a backend to run, but there is no reason
whatsoever you cannot have the backend on the same machine as the frontend.

### 1. Setup System

In this case we will use a simple direct interface
```python
import workflow_forge as forge


# We are configuring this for local execution
backend = forge.make_backend(model,
                             tokenizer,
                             type="default"
                             )
server = forge.make_server(backend, address=None, ident=None)
session = forge.make_session(server)
```

Note until someone competent can show me how to implement auth safely, I am not touching it; api hooks are going to be available, but I do NOT trust myself to not fuck up crypto work somehow. So auth-type will be a registry of capabilities, and auth-payload whatever that ultimately expects.

### 2. Declarative Configuration (UDPL)

Define your prompts and data extraction rules in human-readable TOML:

```toml
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]"]
sequences = ["control", "reasoning", "conclusion"]
valid_tags = ["Training", "Correct", "Feedback"]
control_token = "[Jump]"
escape_token = "[Escape]"
tokenizer_override = {"eos_token" : "[EOS]"}

[[setup]]
text = """
[Prompt] From now on you will frequently see [Escape] "[Prompt]" tokens, which indicate you are receiving a prompt from the user or system, and [Escape] "[Answer]" tokens which means you are being given room to answer yourself. Keep that in mind moving forward. After the final [Escape] "[Answer]" token you would need to produce your normal [Escape] "[EOS]" tokens. As a final note, a [Escape] "[Escape]" token should be ignored and just influences external flow control. You should never generate one of these tokens unless you intend to advance zones, unless you first put an escape token in place.
[Answer] 
I understand. The [Escape] "[Prompt]" token means the user has something to say, the [Escape] "[Answer]" token is where I talk, and the [Escape] "[EOS]" token ends a section. I use and see [Escape] "[Escape]" tokens to skip causing actions for the upcoming control token.
[EOS]
"""
tags = [[], []]

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

Parse them into python.
```python
# Parse configuration
sequences, config = forge.parse_udpl_file("workflow.toml")
```

### 3. Flow Control Programming (SFCS)
Write Python-like code that compiles to GPU execution. Resources, seen as an empty dictionary in this case, also allow replacement between batches of strings based on placeholders. To keep it simple we do not include one, but invoking the workflow factory has the effect of sampling the resources.

```python

# Program the flow control
program = forge.new_program(sequences, {}, config)
program.run("setup")
with program.loop("control") as loop:
    loop.run("reasoning",  question="Are you alive?")  # Model decides when to break via [Jump] token
program.run("conclusion")
program.extract("training_data", tags=["answer"])

# Compile to a workflow factory
workflow_factory = program.compile()
```

### 4 Execution.

Naturally, remote execution is about the same thing. Note that with the direct connection used here the we are directly calling into the server. Loopback, or even remote connections, will eventually be possible.

```python
model = ...
tokenizer = ...
tools = ...
backend = forge.make_backend(model, tokenizer, config, tools)
client = forge.make_session(backend, config)

samples = []
for batch in range(1000):
    workflow = workflow_factory() #<- This is sampling resources if they existing. This allows feedback between batches.
    results = client.request(config, workflow, batch_size=500)
    samples.extend(results["training_data"])
```
## The Core Innovation: Token Stream Interception

Instead of calling the model multiple times with different prompts, Workflow Forge lets the model **control its own execution flow** during a single generation.:

```
Model generates: "I need to think more... [Jump]"
                                         ↑
Workflow Forge intercepts this pattern ────┘
                 ↓
Instantly replaces stream with: "Let me reconsider the problem..."
                 ↓  
Model continues generating from new context
```

**This happens entirely in tensor operations** - no Python roundtrips, no generation restarts, no external orchestration. The model literally loads precompiled and tokenized prompts by generating special control tokens, then reads through them and generates its own responses. This happens in parallel, during evaluation, across batches, and with flow control. It is designed for evaluation, and is particularly suitable for mass sampling of a particular workflow, as in synthetic training data generation. And once deployed using it is as simple as passing the token stream through a function.

Do not make the mistake of misunderstanding, however; this is ultimately intended to work with a wrapper schema like you are used to. Additionally, some special finite state machines ensure you do not have to add special tokens to your vocabulary either, just prompt the model to emit a sequence of text. A series of additional novel innovations in FSM in the backend completes the picture and allows fully vectorized substitution.

**What this enables:**
- **Self-Modifying Workflows**: Models dynamically switch between reasoning strategies mid-generation by declaring its preferred flow prompt or context.
- **Autonomous Decision Trees**: Each model instance follows different paths based on what it generates. The same workflow is applied, but the choices made can differ. This is Single Workflow Multiple Streams (SWMS), a close cousin of SIMD.
- **Massive Parallelism**: hundreds of independent decision-making streams per batch. Great for mass sampling.
- **Zero-Latency Flow Control**: Instant transitions between workflow states.


## The Architecture

Workflow Forge uses a multistage compilation pipeline with a clean location to
serialize for frontend/backend separation. Naturally, it is perfectly possible
to run the backend on the same machine as the frontend, and even possible to
skip serialization alltogether; nonetheless, web API hooks are provided.

### Frontend (Client-Side Compilation)

```
[Frontend userspace.............]   [Compiling chain (front)...............]
UDPL (Config) → SFCS (Programming) → ZCP      → RZCP      → SZCP
    ↓               ↓                ↓          ↓             ↓
TOML Files → Python Flow Control → Blocks → Sampling → SerializableIR
```

SZCP can be serialized at any point, and sends raw
data rather than pickle. Note that as far as the user needs
to be concerned they only ever interact with the frontend
userspace.

### Backend (Server-Side Execution)  

```
[Compiling chain (backend)]  [Backend Execution...........]
SZCP        →     LZCP →     ByteTTFA   → GPU Execution
 ↓                  ↓              ↓           ↓
SerializableIR → Tokenized → Instructions → Results
```

### Key Stages:
- **ZCP**: Sequence linked lists. Think a 'scope'
- **RZCP**: Graph with resolved flow control and resource callbacks. 
            Lowering this samples the resources
- **SZCP**: Fully resolved, serializable workflow.
- **LZCP**: Tokenized, ready for compiling to bytecode.
- **ByteTTFA**: Compiled bytecode running on GPU finite automata

### Deployment Flexibility:
- **Local**: Full pipeline in one process
- **Distributed**: Frontend compiles to SZCP → HTTP → Backend executes  
- **Hybrid**: Teams can build workflows locally, deploy remotely

The **SZCP serialization boundary** enables "compile locally, execute remotely" workflows while keeping each stage modular and extensible.
    
## Use Cases

**Synthetic Data Generation**: Generate thousands of reasoning chains with different complexity levels and extract training pairs automatically. Draw all samples from a configuration in one pass.

**Mass Sampling**: Sample the same question 500 times. Keep the best result. Repeat.

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
- [Techical Status](TECHNICAL_STATUS.md)
- 📝 [UDPL Specification](docs/UserGuide/UDPL.md) - Prompting language reference  
- 🔄 [SFCS Guide](docs/UserGuide/SFCS.md) - Flow control programming
- ⚙️ [ZCP Reference](docs/ZCP/ZCP.md) - Intermediate representation details
- 🚀 [TTFA Architecture](docs/Autonoma/) - GPU execution engine

## Philosophy

Current prompting tools force you to choose between simple chains or complex orchestration. Workflow Forge applies compiler design principles to AI workflows - write high-level logic once, compile to efficient execution, run autonomously at scale.

The goal is **C++-level power with configuration-level simplicity** - sophisticated control flow that compiles down to fast, autonomous execution.

## Contributing

Workflow Forge is designed with the capability of becoming a community standard for AI workflow automation.

- 🐛 [Report Issues](issues/new) 
- 💡 [Feature Requests](issues/new)
- 🔧 [Pull Requests](CONTRIBUTING.md)
- 📖 [Documentation](docs/)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*For researchers who need complex workflows and speed*