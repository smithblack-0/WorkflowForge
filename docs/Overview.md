# Overview

 Workflow Forge (WF) is a system consisting of a prompting frontend and configuration language, a flow control language, an IR compiler, and a final backend consisting of a FSM designed to operate with flow control on the GPU.It compiles an automatic structure to run a process using flow control on a single model. It runs without exiting to python for flow control in most situations.

# Project introduction
## What does this do?

Capabilities explicitly supported by the project are:

* Stupid straightforward prompting configuration at all levels of complexity from beginner to expert.
* Batched generation with flow control and different decisions under the Single Workflow, Multiple Streams (SWMS) principle. Muliple Workflow, Multiple Streams is not currently supported. This is explicitly designed for mass generation of synthetic training data, the original use case.
* Python interaction with tool usage. This, however, blocks batched execution so run it in your own thread and mock tool usage during training. This could be extended in future versions to eject and move onto another batch while the tool resolves.
* Automatic extracting of synthetic data or audit data by generative zones post-generation.
* Straightforward integration into existing training loops: The controller intercepts and processes your tokens, and returns your next tokens for the batch.

An additional possible extension, but only really possible for 
stateful models, is:

* Agentic flow control. 

## Why does Workflow Forge exist?

The WF system is designed to greatly lower the 
workload required to get complex prompting jobs 
done and be adaptable enough to form a core for future 
extension. It has, overall, three different conceptual
levels of components.

1) Universal Declarative Prompting Language (UDPL) and Straightforward Flow Control System (SFCS) make up a set of modern specification languages that amplify developer effort, much like C++ was when all your previous options were BASIC or Assembly. An explicit goal is that a one-year ML intern could follow what is going on with no language training at all then get up to speed in a week. *As far as users should be concerned, this is all that matters unless they need to debug the language itself, in which case they should see below.* 
2) Zone Control Protocol (ZCP), and the associated parsing backend systems, are a specialized IR designed to expose individual Prompt/Generate blocks and token-driven flow control declaration. It is in essence a prompting bytecode specification consisting of a graph of instructions. The combination of UDPL parsing and then SFCS programming compiles to this language. 
3) The Main Orchestration Autonoma is a Token Triggered Finite Autonoma backend that ZCP bytecode can then be parsed into. It is, in effect, a very primitive purpose-built computer with a program counter implemented entirely using tensors on the GPU itself. As such, it can support batched flow control involving on the GPU without ever breaking to python. Users should almost never have to deal with this.

Learning from previous generations of compiler and language mistakes, these are deliberately interchangable; you can write your own prompting parser if it compiles down to ZCP, or your own TTFA if you need more capabilities. Additionally, while support is only planned for Torch at first, it is possible to develop backends for other languages as well.

### Why does UDPL and SFCS exist

UDPL and SFCS are together two domain specific languages 
intended to make the process of defining prompt workflows 
much more straightforward. It is a set of innovations that,
together, make up a language that turns prompting programming into something more akin to configuration.

Prompt engineering with most current prompt libraries is 
somewhat like being forced to choose between BASIC or 
assembly: You can choose to do simple tasks easily,
or more complex tasks with a lot of work, but you cannot
do both.Simple single purpose libraries can get common jobs done
quite easily, but are not very powerful on variations. Meanwhile, the lower level libraries such as Microsoft's 'Guidance' are powerful but require excessive manual and often brittle loading of code resources and segments, making pivots during research or tuning unnecessarily difficult. 

This is unnecessary. C++ is a very powerful language that nonetheless can be compiled down to something small and very fast; this is the approach taken here. UDPL allows you to specify chains of prompts to feed with 'tagging' for automatic extraction of texts later, and SFCS is a simple flow control system that captures the flow control graphs and their requirements in a pythonic manner that can then be lowered into
ZCP. Every effort has been taken to ensure all portions of the pipeline are human readable and easy to grok at a glance - for instance, activating flow control means making an indented flow region in python like normal if you are following standard workflow forge linting protocols.

It is worth checking their respective files in the documentation if you wish to know more about each of the language, or how they fit together. 

### What is ZCP?

It should be emphasized, first, that as far as the user is concerned they never have to see ZCP.

The Zone Control Protocol is an intermediate stage that is at the core of the SFCS system. The smallest unit of instruction is the Zone. A Zone has an optional sequence of tokens to feed while teacher forcing, a conceptual token to listen for to move onto the next zone, and sometimes a conceptual token to listen for to engage 'jump' flow control logic, along with some
other details. We use the word conceptual as we prevent the need to extend the tokenizer by actually matching what the sequence tokenizes into; this, however, is not a userspace detail. Anytime hereafter tokens are referred to, they are conceptual.

The Zone Control Protocol is a graph of these zones that 
walks us through the flow control, as originally defined
in the SFCS system. If anyone wants to worry about visualizing
this, it is the best low-level place to look when debugging.

### What is the Prompt Feeding Automata

The Main Orchestration Autonoma (MOA) is a simple, turning-incomplete computer operating entirely on the GPU using vector indexing logic. It is the first of what I am calling a Token Triggered Finite Automata (TTFA). It is divided into a subset of token injector autonoma that in essence act as its kernel modules.

The basic strategy is exactly what Torchscript did; decide on a restricted subset of flow control supported, and support that. Then we put a really smart compiler strategy in front of it. The program counter can advance zones or jump on flow control, and all underlying complex token manipulation is performed by listening to the program counter and intercepting signals within a particular zone.

It is worth briefly discussing the insight that makes this possible in a batch-parallelized format.

1) It is the case that vector indexing, known as advanced indexing, can be performed in Numpy-derivative languages quite efficiently using tensors of indexes.
2) Grabbing data using indexes is exactly the same thing as dereferencing a pointer. But these tensors can dereference multiple pointers across all batches.
3) Throw in a Program Counter and you can implement a computer that runs entirely on the GPU, never leaving it, and even supports flow control logic all while being batched.

The computer is a very simple Harvard architecture
which behaves as a finite state machine that can move between 
instruction 'Zones', and transitions are triggered by listening
to instructions. Sequence has a program counter which is advanced one zone at a time under normal conditions, and can also trigger jumps to addresses; these jumps are statically defined at compile time, making two options - jump to the static destination or continue to the next zone. See the Autonoma file in the documentation for technical details, starting at MOA. The other files define the kernel extensions.

# Clarifying Key Concepts

Here are some essential clarifications that may not be obvious from reading their individual sections:

## Sequences, Blocks, and Zones

These terms frequently appear and may cause confusion if not clearly distinguished:

- **Sequence**: A named stage or phase within your program logic (e.g., `setup`, `loop`, `solving`).
- **Block**: A single prompt-response entry within a sequence, defined in the TOML configuration.
- **Zone**: A clearly delimited portion of text within a block, marked by special tokens like `[Prompt]` and `[Answer]`. Zones are the units tagged for selective extraction later.

## Error Handling

The UDPL and SFCS language come with a formal specification of what they can and cannot do, and under what conditions.
This makes it straightforward to write parsing logic that
does rigorous error checking; writing a config that has the
wrong configuration will tell you what is wrong, and where.

## **Tags and Their Purpose**

Tags like `Training`, `Correct`, or `Feedback` do 
**not directly control runtime behavior**. Instead,
they’re metadata for selectively extracting parts 
of the generated text afterward.

## Flow Control via Prompts and Tokens

WF supports genuine flow control—though 
differently than traditional programming languages.
You define logical control structures 
(loops, conditionals, etc.) using SFCS in 
a familiar programming style, and these compile
down to an FSM (Finite State Machine).

During execution, the model is prompted
to decide whether to emit special tokens 
(such as `[Jump]`) or treat them as no-operations.
When emitted, these tokens trigger transitions within 
the FSM according to the paths pre-defined by your
SFCS structure.

To support this, it must be noted that you should
be, when defining your flow control, selecting
a prompt to use that tells the model to emit the
branch token. 

Finally, the escape token defined in config
can skip the next flow control instruction, such
as advancing zones or not jumping.

## The Model as Commander

At runtime, the language model acts 
as the "commander," prompted to control 
transitions by emitting or withholding 
special tokens. WF listens for these 
commands and transitions accordingly, 
loading new prompts or sequences.
This enables efficient flow control 
entirely on the GPU, without Python-level
branching during generation.

# Quick Examples

## Universal Declarative Prompting Language (UDPL)

A simple minimal UDPL declaration might be the
following, which is a valid file. This is being
configured to perform philosophical self-play.

```toml
# All UDPL instances have to have a config
#
# These specify a variety of important features
# including certain special tokens that must
# be available. The MOA backend matches to tokens
# per speed, so the tokenizer must be made to support
# them
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags =["Training", "Correct", "Incorrect", "Feedback"]
default_max_token_length = 20000
sequences = ["setup", "loop", "solving", "solution"]
control_token = "[Jump]"
escape_token = "[Escape]"

# This will setup the scenario.
# Notice the tagging on the second
# entry
[[setup]]
text = """
[Prompt] 
Think of an interesting philosophical scenario with unclear
ethical solutions. Reason it out
[Answer]
"""
tags = [['Training'], []]

[[setup]]
text = """
[Prompt] 
Clearly state the philosophical scenario, omiting the 
reasoning, like you are going to show this to another
person and ask them to solve it.
[Answer]
"""
tags = [[], ["Training"]]

# This controls 
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
name = "min_control_resource"
type = "control"
[loop.max]
name = "max_control_resource"
type = "control"

# This will be repeated again and again as needed.

[[solving]]
text ="""
[Prompt]
Reason through an answer to the scenario. First
select a small subset of the following principles.
Revise your previous answer if desired and it 
exists.

{placeholder}

You may also want to keep in mind this
feedback you previous created

{feedback}
[Answer]
"""
tags=[[], []]
[solving.placeholder]
name = "constitution"
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
text ="""
[Prompt]
State the scenario and a subtly incorrect way to resolve the scenario;
[Answer]
"""
tags =[[],["Incorrect"]]

[[concluding]]
text="""
[Prompt]
Based on your answers, state several things you could
do better next time.
[Answer]
"""
tags = [[],["Feedback"]]
```

This sequence has been designed to have hooks for flow
control in the jump sequencing, and is tagged for zone
extraction. Extracting the union of the (Training, Incorrect) tagged zones, for instance, will produce outputs that are as though the model just went straight to the right answer. Extracting the (Training, Incorrect) responses that may be subtly wrong. And just the feedback gets the feedback response. More on extracting later.

### Resources

These contain the functions which can be called upon in 
order to provide the needed dynamic dependencies requested
by a UPDL config. They can sometimes be parsed from the 
UPDL file, but other times also have to be created by
the user or automated processes. They support, for instance,
sampling from a list of points to consider.

For example, the previous example would have needed to 
resolve a resource named "constitution_overview". You
might have performed

```python
import workflow_forge as forge
from forge import StringResource

constitution = """
... whatever
"""

resource = StringResource(constitution)
```

Much more complex sampling strategies, such as from a list
of strings or using a resource that can be changed between runs 
to incorporate feedback are also possible. See the Resources
documentation for more details on how to use this and how
they work.

### Straightforward Flow Control System

The Straightforward Agentic Control System (SFCS)
is designed to both define flow control in terms of prompts 
in as simple and straightforward pythonic way as possible while at the same time supporting flow control ideas such as loops, conditions etc. This is all intended to be accomplished using a pseudoprogrammic system where steps are invoked and 
the main object captures a graph of the actions. It
is, in essence, a way of making a program that can be compiled
down to the Zone Control Protocol intermediate byte language.

Notice in the example program below the core power of WF;
this feels like python flow control, not another language.

```python


# Using, setting up, dependencies for the 
# example.
constitution = ...

resources = {}
resources["constitution"] = forge.StringResource(constitution)
resources["feedback"] = forge.FeedbackSamplerResource()(buffer_size=300)

sequences, config = forge.parse_udpl_file("prompts.toml")

# Tokenizer setup

tokenizer = make_tokenizer()
tokenizer = add_special_tokens(tokenizer, config.special_tokens)

# Programming the actual control 

program = forge.new_program(sequences, resources, config, tokenizer)
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

program.extract(name="good_synthetic_training_data", 
                tags = ["Training", "Correct"]
                )
program.extract(name="bad_synthetic_training_data",
                tags = ["Training", "Incorrect"])
program.extract(name="feedback",
                tags = ["Feedback"])

# Compile the program. 
controller_factory = program.compile(backend="default")
```

### Deployment by Backend

Once programs are compiled, the factory can be called to
make a TTFA machine that runs the program. This TTFA machine
is designed to complete a single process entirely autonomously
by following the prompts and responding to them, and execute
flow control, task judgement, and other such utilities on the 
GPU by simply replacing tokens as needed using vector indexing.
This allows for autonomous exploration and reasoning processes
to occur in a very fast and batched manner. Continuing our program from before might look like this for completing a thousand
separate batches.

```python
training_data = []
for batch in range(1000):
    flow_manager = controller_factory()
    while not flow_manager.done():
        tokens = flow_manager.next()
        tokens = model.predict(tokens, sequence)
        flow_manager.advance(tokens)
    output = flow_manager.extract()
    for batch in output:
      case = {"correct" : batch["good_synthetic_training_data"],
              "incorrect" : batch["bad_synthetic_training_data"]}
      training_data.append(case)
      resources["feedback"].insert(batch["feedback"])

save_to_disk(training_data)
```
