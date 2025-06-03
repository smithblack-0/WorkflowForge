# Workflow Forge: GPU-Native Flow Control for AI Workflows

Hey HuggingFace community!

First time poster here - I've been working on something that could solve a major pain point in complex AI workflows, and I'm wondering if there's enough community interest to make 
it worth open sourcing.

## The Problem

Anyone doing complex AI prompting workflows knows this pain: you're stuck choosing between "simple but inflexible" tools and "powerful but brittle" frameworks. Want basic prompt chaining? Easy. Want complex multi-step reasoning with flow control? Time to write brittle Python glue code with constant CPU-GPU round trips. It's like being forced to choose between BASIC and Assembly when what you really need is C++.

## Why I Built This

I had a unique research task that required something significantly more capable: Single Workflow Multiple Stream generation with automatic prompt feeding, generation periods, and flow control without ever leaving the GPU. I need massive sampling to produce synthetic training data for a self-play application. Additionally, depending on results, I needed massive, unpredictable reconfiguration of prompts and flow control without having to rewrite the pipeline from scratch each time.

The existing tools just wouldn't work, so I developed something extremely flexible that would. But in the process, I developed a formal language and four-stage compiler pipeline with an IR. Oops.  I'm moving beyond prototyping into production development, and I'm curious about wider interest.

**My question to the community**: I've already developed tooling for my research needs, and it is flexible enough to fit all my use cases, but my use case was so large I appear to have designed a new standard and reference implementation in the process. Now that I am moving into getting my tooling production ready, this works for my research needs, but could this solve workflow problems you're facing too? Is it worth the effort to open source and extend it? What use cases are in high enough demand to be worth adding?

## What I Built

**Workflow Forge** is a complete stack for AI workflow automation with a multistage compilation process and dead-simple configuration and coding. It, like the "C++-> LLVM -> machine code" compilation process, is also built to be extendable and swappable at any stage, in most part because I did not feel I could handle the complexity without the full compilation package:

- **Universal Declarative Prompting Language (UDPL)**: Human-readable TOML configs that define complex 
 prompt sequences. A formal programming language, a DSL, with parse
 rules and hence error conditions
- **Straightforward Agentic Control System (SACS)**: Python-like flow control that compiles to efficient execution. A formal programming language, a DSL, with parse rules and hence error conditions. Also stages forward from the UDPL and builds the 
 flow control factory.
- **Zone Control Protocol (ZCP)**: Intermediate representation for prompt workflows. Internal to the system. Same purpose as LLVM.
- **GPU-Native Token Triggered Finite Automata**: The secret sauce - a computer implemented entirely in tensors that supports Single Workflow, Multiple Streams. Since tokens can be matched to produce bool tensor masks, we can trigger using vector logic.

The key technical breakthrough for the last, which may raise some eyebrows: **vector indexing in tensors is pointer dereferencing**. Add a program counter, and you can implement a complete computer that runs batched flow control entirely on GPU, never dropping back to Python. This enables Single Workflow, Multiple Stream generation where a single prompt with marked flow control runs across multiple batch elements, each making different decisions along the way.

The key design insight that lead me to pursue: **C++ made hard tasks easy when previously we only had BASIC and Assembly.** With careful language design, we can perform a similar transformation for prompting workflows. The UDPL and SACS DSLs are designed to be so simple that a one-year ML intern could read them and roughly follow along with no prior exposure, then bring the questions back to their seniors. Who said the history of computing had no practical applications?

## The Technical Summary

**Current approaches:**
- Simple tools: Limited to basic prompt chaining
- Complex tools: Brittle Python callbacks, CPU-GPU round trips
- Research frameworks: Not production-ready

**Workflow Forge approach:**
- **Declarative configuration** compiles to efficient bytecode
- **GPU-native execution** using tensor operations as computer primitives  
- **Batched autonomy** - hundreds of independent workflow streams per batch
- **Modular design** - use my frontend with your backend, or vice versa

The finite state machine uses Harvard architecture (separate instruction/data streams) and is simple enough for formal verification, but nonetheless powerful enough for complex workflows by exporting compilation into python. Think compiling C++ into an arduino, or the ideas behind RISC.

## Current Status

**✅ Proof-of-concept and partially built:**
- Full formal description of UDPL
- UDPL parser (v0.1)
- ZCP intermediate representation (v0.1) 
- GPU FSM proof-of-concept (zone advancement across batches)
- Core architecture proven for my research needs

**🚧 Ready to develop:**
- Full formal description of SACS
- Full flow control integration
- SACS compiler
- Framework integrations
- Documentation and examples
- Open sourcing? :)

## What I'm Looking For

Before I commit to full development and open sourcing:

- **Is this interesting?** Worth the effort to open source?
- **Framework priorities?** PyTorch first, then what?
- **Missing use cases?** What workflows would you build with this?
- **Interface feedback?** Are these abstractions intuitive?
- **Want to collaborate?** Especially on testing - Multiple Workflow, Multiple Streams is theoretically possible but much more complex to test properly
- **Help!** I have never open sourced a project before. What do I need to do? I have published to PyPi but it has been a LONG time since I setup a CI too.
- **Names**: Should we keep SUPS, or try to rename it on the "why wasn't this obvious" angle I am feeling - Maybe Facepalm or something?

Full transparency: I've never had a formal ML job in academia or industry (I do have a physics degree though, and coded my way out of the womb), so if someone more experienced wants to help with best practices and community standards, that would be incredible! Also, if this is not written to normal standards please tell me - I got some advice from the LLMs but it may not be perfect.


## Possible Extensions

These extensions are architecturally feasible and I've already planned out the token flow, control signals, and TTFA modifications needed - mostly because I might need them for my own research down the line:

- **Multi-agent workflows** - Multiple agents communicating through shared memory spaces. Would need UDPL extensions for agent transitions and an additional independent TTFA - only works with stateful models like RWKV/Mamba, unfortunately, and there is nothing I can do about it as I would cache and load the correct agent by indexing into a stack of states. 

- **Tool usage simulation** - Agents simulating tool responses in-workflow rather than making external calls. Natural extension once multi-agent support exists - one agent becomes the "tool" that responds to another agent's queries.

- **Evaluation-time tool integration** - Real tool calls during workflow execution. Tricky because it breaks the GPU-native execution, but possible with either separate backend for eval workflows or very clever 'nonblocking' tool usage logic with a results buffer on the TTFA. Idle time is spent producing padding tokens.

- **Custom backend support** - The modular design supports any framework with numpy-style indexing. Currently focusing on PyTorch, but the ZCP intermediate representation could target other frameworks.

Prioritization depends entirely on what the community actually needs - I'm curious which of these would be most valuable for your use cases! Note that I plan on getting the core system working first, for obvious reasons.

## Why Is Significant.

The most important effect, by far, of this tooling suite is providing a massive reduction in prompt control difficulty, analogous to stepping into C++ when you previously just had BASIC and Assembly. This outweighs even the batched backend system, which while clever and needed for my purposes is likely not needed as immediately by broader industry.

Here's what this looks like in practice - automated philosophical self-play. It generates synthetic training data, automatically slicing apart the token stream, uses flow control, uses dynamic template filling and includes feedback. It would be a nightmare to implement using today's open prompting technology, and with a lot of extensions is in fact one of my relevant research applications.

**The UDPL configuration:**
```toml
# All UDPL instances have to have a config
#
# These specify a variety of important features
# including certain special tokens that must
# be available. The FSM backend matches to tokens
# per speed, so the tokenizer must be made to support
# them
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"] 
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags =["Training", "Correct", "Incorrect", "Feedback"]
default_max_token_length = 20000
sequences = ["setup", "loop", "solving", "concluding"]
control_token = "[Jump]"
escape_token = "[Escape]"

# This will setup the scenario.
# Notice the tagging on the second
# entry
[[setup]]
text = """
[Prompt] 
Think of an interesting philosophical scenario with unclear
ethical solutions. Prefer to come up with scenarios
where ethical principles can come into conflict.
[Answer]
"""
tags = [[], ['Correct']]

# This controls looping.
[[loop]]
text = """
[Prompt]
You are in flow control right now. If you are satisfied
with your current answer emit [Escape] "[Jump]". Otherwise
answer "try again". If this is your first time seeing this,
just say "proceed". Repeat at least {min} times and at most
{max} times.
[Answer]
"""
tags = [[],[]]

[loop.min]
name = "min"
type = "control"
[loop.max]
name = "max"
type = "control"

# This will be repeated again and again as needed.

[[solving]]
text ="""
[Prompt]
Reason through an answer to the scenario.
You may also want to keep in mind this
feedback you previous created. You 
may change your answer or stick with your
current one if one exists.

{feedback}
[Answer]
"""
tags=[[], []]
[solving.feedback]
name = "feedback_backend"
arguments = {"num_samples" : 3}

# This sequence generates output and feedback
[[concluding]]
text ="""
[Prompt]
State the scenario and the best way to resolve it directly;
[Answer]
"""
tags =[[],["Correct"]]

[[concluding]]
text="""
[Prompt]
Based on your answers, state several things you could
do better next time.
[Answer]
"""
tags = [[],["Feedback"]]
```

**The Python control flow looks like normal programming**: Note we also are declaring what tags to extract, which extracts and concatenates all zones with the union of those tags.

```python
from CE import workflows

# Resources return strings when a callback
# was placed in the UDPL toml file. This one 
# was concretely implemented to track feedback
resources = {}
resources["feedback"] = sups.FeedbackSamplerResource(buffer_size=300)

sequences, config= sups.parse_udpl_file("prompts.toml")

# Tokenizer setup

tokenizer = make_tokenizer()
tokenizer = add_special_tokens(tokenizer, config.special_tokens)

# Programming the actual control
program = sups.new_program(sequences, resources, config, tokenizer)
program.run(sequence="setup") # This runs the sequence called setup
with program.while(sequence="loop", min=2, max=6) as loop:
   # Loop, recall, can sometimes emit a 
   # [Jump] token when run. This brings us 
   # OUT of the loop. Control sequences
   # should have prompting telling the model
   # when to emit the control token.
   loop.run("solving")
program.run("concluding")

# Programming the tag extraction to automatically
# extract relevant zones.
program.extract(name="synthetic", 
                tags = ["Correct"])
program.extract(name="feedback",
                tags = ["Feedback"])

# Compile the program. 
controller_factory = program.compile(backend="PFA")
```

**Once compiled, running hundreds of autonomous workflows is trivial** 

Note this is currently feeding one token group at a time, assuming some sort of stateful model like RWKV or Mamba, as that has to do with my broader research, but you could adapt this to feed in the token stream for a normal transformer too.

```python
batch_width = 32
num_batches = 1000
training_data = []
for batch in range(num_batches):
    sups_manager = controller_factory()
    tokens = initialize_tokens(batch_width)
    state = initialize_state(batch_width)
    sequence = []
    tags = []
    while not sups_manager.done():
        tokens, state = model.predict(tokens, sequence, state)
        tokens, tag = sups_manager.inject(tokens)
        sequence.append(tokens)
        tags.append(tag)
    output = sups_manager.extract(sequence, tags)
    for batch in output:
        training_data.append(batch["synthetic"])
        resources["feedback"].insert(batch["feedback"])

save_to_disk(training_data)
```

Each batch follows its own path through the workflow - some might loop 2 times, others 6, some might generate different constitutional scenarios. All running in parallel, all handled by the GPU-native TTFA. Technical details and documentation are, of course, found in the repository.

## Where to go for more.

Consider going to https://github.com/smithblack-0/CognitionEngineering for more details. Open the 
documentations folder, and read the files under Simple Universal Prompting System. Alternatively, go into Src/CE/SUPS for current code, what is available. The other documents and files have to do with the bigger picture --- please feel free to DM me if interested, or follow the kerfluffle that should shortly be occurring on LessWrong.