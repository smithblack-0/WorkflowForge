# Workflow Forge: Open Source Launch - GPU-Native Flow Control for AI Workflows

A few weeks ago, I posted an RFC about a system I'd built for my research needs - a way to let AI models control their own prompt feeds through GPU-native token stream interception. The response was encouraging, so I've spent the time since then getting everything production-ready for open source. That response chain was located here: [SUPS](https://discuss.huggingface.co/t/request-for-comments-simple-universal-prompting-system/157648/4) (Why did no one tell me the repo was private?)

I have gone through with developing this into open source status, as I had enough interest. I think the broader community will find it very useful.

Please note this is still under development, and I want to have an initial version done within a month or so to compete in ARC. Nonetheless, since the core architecture is planned out and proof-of-concepted, and I have a UDPL parser and the entire Token Triggered Finite Autonoma backend mapped out, I think I am going to open source it and release the broader picture, the repo, and the current status. It is currently released as an open source github project [here](https://github.com/smithblack-0/WorkflowForge?tab=readme-ov-file). This is also the primary coordination point atm; if anyone wants to volunteer to get a website set up I am all in on the idea.

The intention is to have something community and production grade that may serve as a new, better standard for complex workflow. The staged design means existing bespoke prompting config languages can still use the backend if they can compile to ZCP, or you can write your own backend with extended functionality.

## The Core Problem 

Complex AI workflows force you to choose between "simple but inflexible" tools and "powerful but brittle" frameworks. Want multi-step reasoning with flow control? You're writing Python glue code with constant CPU-GPU round trips. It's like being stuck with BASIC or Assembly when you need C++.

## What Workflow Forge Does
Instead of calling the model multiple times with different prompts, Workflow Forge lets the model **control its own execution flow** during a single generation:

```
Model generates: "I need to think more... [Jump]"
                                            ^
Workflow Forge intercepts this token -------|
                 
Instantly begin to replace stream with: "Let me reconsider the problem..."

One token is replaced per generative step until all tokens in the prompt are used, and model then continues generating from new context.
```

**This happens entirely in tensor operations** - no Python roundtrips, no generation restarts, no external orchestration. The model literally loads precompiled and tokenized prompts by generating special control tokens, then reads through them and generates its own responses. This happens in parallel, during evaluation, across batches, and with flow control.
## How Workflow Forge Compares to Existing Tools

| Tool | Easy to Start | Scale & Change | Serial Prompts | Flow Control | GPU-Native | Batched | Tool Use | Online Interaction | Production Ready |
|------|:-------------:|:--------------:|:--------------:|:------------:|:----------:|:-------:|:--------:|:-----------------:|:----------------:|
| **LangChain** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |        ✅         |
| **AutoGen** | ❌ | ⚠️ | ✅ | ⚠️ | ❌ | ❌ | ✅ | ✅ |        ✅         |
| **Guidance** | ⚠️ | ⚠️ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ✅ |        ✅         |
| **OpenAI API** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |        ✅         |
| **Workflow Forge** | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ |      Soon!       |

### Legend
- ✅ **Yes** - Strong capability
- ⚠️ **Somewhat** - Limited or requires workarounds  
- ❌ **No** - Not supported or very difficult

### What Each Column Means

- **Easy to Start:** Can you get a basic workflow running quickly?
- **Scale & Change:** How easy is it to modify complex workflows or scale to thousands of samples?
- **Serial Prompts:** Support for chaining multiple prompts together?
- **Flow Control:** Loops, conditionals, branching based on model outputs?
- **GPU-Native:** Execution stays on GPU without Python roundtrips?
- **Batched:** Run many independent workflow instances in parallel?
- **Tool Use:** Function calling, external API integration, tool orchestration?
- **Online Interaction:** Real-time chat, user input, interactive workflows?
- **Production Ready:** Stable and ready for production use today?

### The Trade-offs

**Workflow Forge** trades immediate usability for powerful capabilities - it's harder to get started but offers unique features like GPU-native batched execution with model-driven flow control that other tools can't match. 

**Choose Workflow Forge when:** You need mass synthetic data generation, complex reasoning chains with flow control, or research requiring hundreds/thousands of parallel reasoning paths.

**Choose existing tools when:** You need production-ready systems today, simple prompt chains, are just getting started with AI workflows, or do not feel comfortable extending your tokenizer and embeddings. 

I will mention that it is designed to be a LangChain replacement eventually if interest materializes; It can be extended for single-user sessions across threads that even include tool usage. Nonetheless, while I plan on using it this way it is not the immediate point.

## Intended trajectory and commitments.
 
Ultimately, I would be thrilled to see this eventually be taken up under the umbrella of one of the existing mature open source organizations. Lets get the core release done first, though, and I will maintain it for at least two years myself.

## The Architecture (4-Stage Compilation Pipeline)
Workflow Forge uses a 4-stage compilation pipeline inspired by traditional compilers. An extra stage exists as it turns out flow control is easier to capture in python rather than TOML:

```
UDPL (Config) → SFCS (Programming) → ZCP (Bytecode) → TTFA (Execution)
     ↓               ↓                  ↓               ↓
TOML Files → Python Flow Control → IR Graph → GPU Tensors
```

Each stage is modular and extensible - you can write your own frontend languages or target different backends.

**Key technical insight:** Vector indexing in tensors = pointer dereferencing. Add a program counter and you have a complete computer running batched flow control entirely on the GPU.

## What's Actually Built and Working

**Architecture Planning and Modules**
- This is not general purpose computation. A restricted subset of control, smart TOML, and a new kind of graph subset eliminates entire catagories of errors.
- Intuitive flow control and prompt structuring.
- Main dependencies between stages, though I no doublt missed a few.

**Complete UDPL Parsing Pipeline**
- Human-readable TOML configuration for complex prompt sequences.
- Comprehensive error checking across 5 functional stages. If there is an issue, it will tell you how and where.
- Full unit test coverage with proper CI/CD

**Infrastructure** 
- MIT licensed and properly open sourced
- GitHub Actions CI pipeline
- Initial Python packaging with pyproject.toml. No experimental release yet, though, we will wait until the core system is operational unless pressured otherwise
- Contributions guidelines.
- Living documentation and examples

### In Active Development; near future

**In Active Development**
- SFCS flow control compilation 
- TTFA execution engine with modular "OS kernel" design

### In active development; more remote future beyond one to two months.

- Tool usage.
- User stream tool

### Under Consideration

- Automatic special token extension tools
- Automatic special token fine-tuning data synthesis tools.
- "Hello World" guide and quickstart abbreviations to make this a LangChain replacement tool too.

## Real Example - Philosophical Self-Play

Here's what a complete workflow looks like using the actual UDPL syntax. While this cannot currently run, the architecture is planned out to this point. These are the commands, and the way to use them, you use in a real workflow.

**UDPL Configuration (TOML):**
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

**SFCS Programming (Python-like):**
```python
# Parse configuration
sequences, config = forge.parse_udpl_file("workflow.toml")

# Program the flow control
program = forge.new_program(sequences, resources, config, tokenizer)
with program.loop("control") as loop:
    loop.run("reasoning", question="Are you alive?")  # Model decides when to break via [Jump] token
program.run("conclusion")
program.extract("training_data", tags=["answer"])

# Compile to execution engine
controller_factory = program.compile()
```

**Deployment (Batched Sampling):**
```python
# Each batch creates independent workflow streams
samples = []
for batch in range(1000):
    
    # Note that this sets up the FSM backend. No further dynamic injection is possible. 
    flow_manager = controller_factory(batch_size=32, starting_token="[BOS]")
    
    # Workflow runs autonomously on GPU
    while not flow_manager.done():
        tokens = flow_manager.next()      # Get next tokens to feed
        tokens = model.generate(tokens)   # Model generates response
        flow_manager.advance(tokens)      # System processes output
    
    # Extract results
    results = flow_manager.extract()
    for batch in results:
        samples.append(results["training_data"])
```

## Why This Matters

**Single Workflow, Multiple Streams (SWMS):** One workflow definition, but each batch element makes different decisions based on what it generates. Perfect for massive synthetic data generation where you want variety in the reasoning paths.

**Zero-Latency Flow Control:** Model decisions instantly change execution path without dropping back to Python.

**Mass Sampling:** Designed for evaluation and particularly suitable for mass sampling of a particular workflow, as in synthetic training data generation.

**Simple Deployment:** Once compiled, using it is as simple as passing the token stream through a function.
