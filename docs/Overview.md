# Overview

The Simple Universal Prompting System (SUPS) is a proposed system consisting 
of a prompting frontend and configuration language, a flow
control language, an IR compiler, and a final backend consisting
of a FSM designed to operate without python flow control on the GPU,
but nonetheless achieve real flow control by means of model prompting
and token triggering. It compiles an automatic structure to 
run a process using flow control on a single model.

The general idea of the structure is to, after all programming 
is done, provide something that can be setup per batch for the
programmer to use which simply intercepts flowing tokens and
either replaces them with a teacher-forced form, or lets the model
continue to generate freely.

# Project introduction

## Why does SUPS exist?

The SUPS system is designed to greatly lower the 
workload required to get complex prompting jobs 
done and be adaptable enough to form a core for future 
extension. It has, overall, three different conceptual
levels of components.

1) Universal Declarative Prompting Language (UDPL) and 
   Straightforward Agentic Control System (SACS) make up
   a set of modern specification languages that amplify
   developer effort, much like C was when all your previous
   options were BASIC or Assembly. An explicit goal is that
   a one-year ML intern could follow what is going on with
   no language training at all.
2) Zone Control Protocol (ZCP), and the associated parsing
   backend systems, are a specialized IR designed to expose 
   individual Prompt/Generate blocks and token-driven flow
   control declaration. It is in essence a prompting bytecode
   specification consisting of a graph of instructions. The 
   combination of UDPL parsing and then SACS programming should
   compile to this language. 
3) The Prompt Feeding Automata is a backend that
   ZCP bytecode can then be parsed into. It is, in effect, a
   very primitive computer with a program counter implemented
   entirely using tensors on the GPU itself. As such, it can
   support batched flow control involving only self-play tasks
   on the GPU. Before you scoff, yes, it is prototyped; it does
   work. Everything is generated in one smooth run.

Learning from previous generations of mistakes, these are deliberately
interchangable; you can write your own prompting parser if it 
compiles down to ZCP, or your own TTFA if you need more capabilities.
Additionally, while support is only planned for Torch at first,
you may easily write your own backend systems in other frameworks;
any framework with numpy indexing behavior should be compatible.

Extraction technology is also included with the package, with
the ability to tag zones then extract a sequence of all tags
in order planned. Note that tool usage is NOT explicitly
supported at the moment, as the backend is designed for rapid
bulk generation of samples and waiting in python for tool
responses would stall this process. Extensions which have been mapped out as
feasable by additional modules, without modification to the 
above, include

* Multiagentic support, though this only works in stateful models
  like Mambda and RWKV; no transformers. That restriction does not 
  apply to everything else, however. Extensions are required in
  UDPL, which is minor, and an additional independent TTFA is needed
  which stores and returns the active agent state for each step
* Tool usage training. Once we have multiagent support, it is easy
  to simulate tool results by having another agent be prompted to 
  simulate a response.
* Eval time tool usage. At the moment, I do not need it, but
  it would be perfectly feasable to include a flag in the language
  that set tool usage as allowed. However, this will likely require
  a separate backend, as though UDPL, SACS, and ZCP are perfectly
  capable of expressing these requests, deployment-state tool usage
  does not 

Implemenetation of these will depend on community interest, as
they are not immediately needed to complete my research.

### Why does UDPL and SACS exist

UDPL and SACS are together two domain specific languages 
intended to make the process of defining prompt workflows 
much more straightforward. It is a set of innovations that,
together, make up a compiler for prompting that is far more
capable than anything openly available at the moment.

Prompt engineering with most current prompt libraries is 
somewhat like being forced to choose between BASIC or 
assembly: You can choose to do simple tasks easily,
or more complex tasks with a lot of work, but you cannot
do both.Simple single purpose libraries can get common jobs done
quite easily, but are not very powerful on variations. Meanwhile,
the lower level libraries such as Microsoft's 'Guidance' are 
powerful but require excessive manual and often brittle
loading of code resources and segments, making pivots during
research or tuning unnecessarily difficult. 

This is unnecessary. C++ is a very powerful language that nonetheless
can be compiled down to something small and very fast; this is the 
approach taken here. UDPL allows you to specify chains of prompts
to feed with 'tagging' for automatic extraction of texts later, and
SACS is a simple flow control system that captures the flow control 
graphs and their requirements in a manner that can then be lowered into
ZCP. Every effort has been taken to ensure all portions of the pipeline
are human readable and easy to grok at a glance - for instance, activating
flow control means making an indented flow region in python like normal 
if you are following standard SUPS linting protocols.

The **Universal Prompting Declarative Language** is a human-readable 
TOML file that defines sequences of prompts to feed and then generate
in response to, and also defines tagging information that lets you 
assign tags to text regions; these tags can then be declared as 
required for extraction later on in SACS. The 
**Straightforward Agentic Control System** is intended to allow
display of python-like flow control, and configuration that develops
a static flow control graph and invokes UPDL sequences in each flow
block. Tag extraction is also specified as part of the process. 
Maximal effort is placed on easy of reading, writing, and reasoning
through these languages.

These two frontend languages together are intended to provide a new
foundation for defining flow control and prompts. They are, I 
hope, useful to others as well.

### What is ZCP?

The Zone Control Protocol is an intermediate stage that is
at the core of the SACS system. The smallest unit of 
instruction is the Zone. A Zone has an optional sequence
of tokens to feed while teacher forcing, a token to listen
for to move onto the next zone, and sometimes a token to listen
for to engage 'jump' flow control logic, along with some
other details. 

The Zone Control Protocol is a graph of these zones that 
walks us through the flow control, as originally defined
in the SACS system. If anyone wants to worry about visualizing
this, it is the best low-level place to look when debugging.

### What is the Prompt Feeding Automata

The Prompt Feeding Automata (PFA) is a simple, turning-incomplete
computer operating entirely on the GPU using vector indexing logic.
It is the first of what I am calling a Token Triggered Finite Automata (TTFA).

The basic strategy is exactly what Torchscript did; decide on a restricted subset of flow control supported, in fact consisting of only advance The program counter or jump to a location determinedly statically by something associated with the current counter. Despite this simplicity, this is turning-complete in the same way an arduino is; if you accept you cannot change your instructions after encoding, but can watch the data stream, you can write
a compiler to handle arbitrary flow control. Indeed, this is actually what happens in some C or C++ backends. The Harvard architecture is used for simplicity.

It is worth briefly discussing the insight that makes this possible
in a batch-parallelized format.

1) It is the case that vector indexing,
   known as advanced indexing, can be performed in Numpy-derivative
   languages quite efficiently using tensors of indexes.
2) Grabbing data using indexes is exactly the same thing as dereferencing
   a pointer. But these tensors can dereference multiple pointers across all
   batches.
3) Throw in a Program Counter and you can implement a computer that runs entirely on the GPU, never leaving it, and even supports flow control logic all while being batched.

The computer is a very simple Harvard architecture
which behaves as a finite state machine that can move between 
instruction 'Zones', and transitions are triggered by listening
to instructions. Sequence has a program counter which is advanced one zone at a time under normal conditions, and can also trigger jumps to addresses; these jumps are statically defined at compile time, making two options - jump to the static destination or continue to the next zone. The intention is to have an automata
so simple it is possible to formally verify. 

The overall idea is as follows:

- **Zones as Instructions**  
  Each zone contains prompt tokens, tags,
  and optional jump info. The ZCP compiler
  assigns control tokens and jump targets.
- **Program Counter (PC)**  
  A per-batch counter tracks the current
  zone. It normally moves forward one zone
  at a time unless a jump is triggered.
- **Zone Execution**  
  1. Prompt tokens override model input.
  2. After prompts, tokens flow freely
     until:
     - **Advance token** → next zone  
     - **Jump token** → jump target  
     - **Timeout** → inject advance token
- **Tag painting**: The tags that are active for this zone
    are painted onto the generated tokens. This manifests as
    a separate tags output from the autonoma.
- **Vectorized GPU Execution**  
  All operations—token matching, PC updates,
  prompt injection—are parallel and GPU-local.
  No Python or CPU sync is needed.

This simple, efficient structure allows
massively batched flow control and prompting
with minimal overhead and full GPU locality.
We should also discuss how to store token sequences,
which may be of varying length and which, in a list, would 
require going back to python. We can flatten
the prompt tokens to all zones, concatenate those 
together, then store the offset to start and end
feeding from with the Instruction. This completes
the picture.

# Technical Status

Basically, already in progress or ready to get quite serious.

* UDPL parser - V0.1 done, but needs to be reconfigured after 
  adding flow control.
* SACS - Outlined, graph theory corrolation between inline flow
  and flow control mapped
* ZCP - V0.1 done, but again needs to be reconfigured after
  adding flow control.
* PFA - Proof-of-concept on the computation mechanism completed,
  showing we can feed a sequence of prompts across different
  batches advancing to the next prompt at different times.


# Clarifying Key Concepts

Here are some essential clarifications that may not be obvious from
reading their individual sections:

## Sequences, Blocks, and Zones

These terms frequently appear and may cause confusion if not clearly
distinguished:

- **Sequence**: A named stage or phase within your program logic (e.g., `setup`, `loop`, `solving`).
- **Block**: A single prompt-response entry within a sequence, defined in the TOML configuration.
- **Zone**: A clearly delimited portion of text within a block, marked by special tokens like `[Prompt]` and `[Answer]`. Zones are the units tagged for selective extraction later.

## Error Handling

The UDPL and SACS language come with a formal specification of what 
they can and cannot do, and under what conditions.
This makes it straightforward to write parsing logic that
does rigorous error checking; writing a config that has the
wrong configuration will tell you what is wrong, and where.

## **Tags and Their Purpose**

Tags like `Training`, `Correct`, or `Feedback` do 
**not directly control runtime behavior**. Instead,
they’re metadata for selectively extracting parts 
of the generated text afterward.

## Flow Control via Prompts and Tokens

SUPS supports genuine flow control—though 
differently than traditional programming languages.
You define logical control structures 
(loops, conditionals, etc.) using SACS in 
a familiar programming style, and these compile
down to an FSM (Finite State Machine).

During execution, the model is prompted
to decide whether to emit special tokens 
(such as `[Jump]`) or treat them as no-operations.
When emitted, these tokens trigger transitions within 
the FSM according to the paths pre-defined by your
SACS structure.

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
special tokens. SUPS listens for these 
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
# be available. The FSM backend matches to tokens
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
extraction. Extracting the union of the 
(Training, Incorrect) tagged zones,
for instance, will produce outputs that are as though
the model just went straight to the right answer. Extracting
the (Training, Incorrect) responses that may be subtly wrong.
And just the feedback gets the feedback response. More on extracting
later.

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
from CE import sups
from sups import StringResource

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

### Straightforward Agentic Control System

The Straightforward Agentic Control System, SACS, or 'sacks'
is designed to both define flow control in terms of prompts 
in as simple and straightforward pythonic way as possible while at the same 
time supporting flow control ideas such as loops, conditions
etc. This is all intended to be accomplished using a
pseudoprogrammic system where steps are invoked and 
the main object captures a graph of the actions. It
is, in essence, a way of making a program that can be compiled
down to the Zone Control Protocol intermediate byte language.

Notice in the example program below the core power of SUPS;
this feels like python flow control, not another language.

```python

from CE import sups

# Using, setting up, dependencies for the 
# example.
constitution = ...

resources = {}
resources["constitution"] = sups.StringResource(constitution)
resources["feedback"] = sups.FeedbackSamplerResource(buffer_size=300)

sequences, config = sups.parse_udpl_file("prompts.toml")

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

program.extract(name="good_synthetic_training_data", 
                tags = ["Training", "Correct"]
                )
program.extract(name="bad_synthetic_training_data",
                tags = ["Training", "Incorrect"])
program.extract(name="feedback",
                tags = ["Feedback"])

# Compile the program. 
controller_factory = program.compile(backend="PFA")
```

### Deployment by Backend

Once programs are compiled, the factory can be called to
make a FSM machine that runs the program. This FSM machine
is designed to complete a single process entirely autonomously
by following the prompts and responding to them, and execute
flow control, task judgement, and other such utilities on the 
GPU by simply replacing tokens as needed using vector indexing.
This allows for autonomous exploration and reasoning processes
to occur in a very fast and batched manner. Continuing our program
from before might look like this for completing a thousand
separate batches.

```python
training_data = []
for batch in range(1000):
    sups_manager = controller_factory()
    tokens = ... #default
    sequence = []
    tags = []
    while not sups_manager.done():
        tokens = model.predict(tokens, sequence)
        tokens, tag = sups_manager.inject(tokens)
        sequence.append(tokens)
        tags.append(tag)
    output = sups_manager.extract(sequence, tags)
    for batch in output:
      case = {"correct" : batch["good_synthetic_training_data"],
              "incorrect" : batch["bad_synthetic_training_data"]}
      training_data.append(case)
      resources["feedback"].insert(batch["feedback"])

save_to_disk(training_data)
```
