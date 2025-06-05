# Simple Flow Control System

## What is this?

This is the official specification for v.01 of the 
SFCS flow control language.

## Overview

The Simple Flow Control System (SFCS) is a python flow control language designed to provide a pythonic environment to commit a reduced set of flow control operations that can then be linked with the zone specification, built into a flow control graph, and parsed into ZCP.

ZCP, in turn, is a language describing an idealized finite autonoma which exposes transitions, and in which a certain action occurs in terms of feeding information after the transition. As such, the primary purpose of SCFS boils down to forming isomorphisms between python flow control and ZCP transitions. 

Most of the time, this means making nice long linked lists, and indeed the sequences being fed in are already in this format. However, sometimes it means handling more complex cases like flow control.


## Starting point.

The SFCS system starts by initializing a new program. This should be thought of representing, one to one, a literal program that runs the indicated prompt. 

This program must be initialized with:

1) A sequences dictionary, developed by parsing the UDPL file or folder. 
2) A resources dictionary, containing named dictionary resources.
3) A config object, containing important parsed config information from the UDPL pass
4) A tokenizer resource, to allow conversion.

Once the resources requirements vs provded are crosschecked, it is possible to start building a program. This is what is returned when creating a new program. The terminology is

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources, 
                            config, tokenizer)
```

## Programs: Under the hood.

A program is a sequence of linked lists which is being constructed in ZCP, and which ultimately will be compiled into bytecode for the TTFA. 

Each node is in itself an instruction to feed a certain sequence of tokens in then let the model generate until a certain zone advance token is seen. This is the standard linked list feed mechanism the majority of the model operates under.

Flow control, meanwhile, is handled carefully by SFCS by means of having special flow control tokens that can be specified, and then a destination node. As far as the ZCP IR is concerned this means perform this transition when seeing this token.

The running process is, ultimately then, a finite state machine that either steps to the next state in the linked list under nominal processing when the zone advance token is needed, then runs the instruction. Or, it may be triggered by flow control and jump to a specified node. This is more or less the entire ZCP intermediate language.

It should be clarified that, in practice, blocks and sequences decompose into a linked list of individual zones. Stitching these together based on flow control occurs here.

### Compiling

Compiling takes the ZCP language, samples the researchers, tokenizes the results, converts them into tokens, and loads them into the backend as a sequence of instructions, and a sequence of token sources. The instructions contain begin and end offsets to stream tokens from.

## Commands

Commands tell the model to do something during generation, whether it be flow control or otherwise. Most commands interact with prompts, and Prompt Command commands share certain properties. 

1) They must have a 'sequences' feature, which is always the first argument. This must match a sequences defined in UDPL to resolve validly.
2) They may have additional arguments, if the control type was selected as a placeholder in the UDPL. If this is true, those arguments are passed along as kwarg arguments.

### The .run command

The .run command is the most straightforward of commands, and simply has the effect of running that sequence. An example is shown below. This runs setup and then question. The model will literally be fed those prompts in order. If the "Sequence" and "Setup" sequences are not defined in the UDPL, this will fail.

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

# Show both ways to invoke it.
program.run(sequence = "Setup")
program.run("Question")
```

Note that as currently written, this program is useless, as it requires extract postprocessing to get information back out. More on this later.

**Transitions**

This is analogous to a single link forward, or an advancement of the program counter.

### The .loop command

The loop is a model-commanded flow control action, and unlocks the flow control jump. 

Loops invoke sequences just like any other Command, and as such rely on good prompting to make the model behave as desired. It is HIGHLY recommended to use your escape token together with your jump token to tell the model to jump when it is time to exit the loop. For example, by specifying in UDPL, with reserved "[Escape]" and "[Jump]" as tokens. 

**Warning**: Flow control defaults to loop. Only emitting a jump token will exit the loop.
**Warning**: The model backend, as envisioned as of writing this, will IMMEDIATELY jump upon seeing that token. Think GOTO, not blocks.

```toml
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
```

This can now be interacted with, and you may even feed in controls from python. This ability is now also shown.

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

program.run(sequence = "Setup")
with program.loop("loop", min=3, max=6) as loop:
    loop.run("Think")
```

Observe the pythonic flow control using the with statement. This will, naturally, automatically exit when the context closes.

**Transitions**

Two exist. One is to proceed to the next link like normal. The second is to instead point directly into a zone.

### The .if command

"If" is also supported. However, some very important notes are needed:

* **Danger** the "IF" case is executed by *default*. You must manually skip it by emitting a jump if you do not want it. This is to simplify the TTFA. 
* **Warning**: In the reference implementation the model will IMMEDIATELY skip the if state the moment the jump token is emitted.

An example might be the following

```toml
[[if_rethink]]
text = """
[Prompt]
If the number of times you have tried since rethinking your approach is less than {max}, emit token [Escape] "[Jump]" to skip rethinking. Else emit nothing.
[Answer]
"""
tags = [[],[]]

[loop.max]
name = "max"
type = "control"
```

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

program.run(sequence = "Setup")
with program.loop("loop", min=3, max=6) as loop:
    with loop.if("if_needs_rethink", max=3) as if_branch:
        if_branch.run("Rethink")
    loop.run("Think")
```

**Transitions**

When the if jump command triggers, it skips the rest of the sequence chain. Otherwise, we proceed like normal. Either way, we end up in the same place. Note that before the if context closes, an else context may be invoked and this changes the skip behavior to point to the else block instead. 

### The .else command 

The else statement, naturally, is also supported. It is very straightforward to use, based on continuing the if context.

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

program.run(sequence = "Setup")
with program.loop("loop", min=3, max=6) as loop:
    with loop.if("if_rethink", max=3) as if_branch:
        if_branch.run("Rethink")
    with if_branch.else() as else_branch:
        else_branch.run("Think")
program.run("Summarize")
```

**Transition**

Else behavior is nicely produced by having the end of the if chain jump to the outer program source, and the skip jump into the else chain which then jumps into the main program loop when the Linked List finishes.

### The .subroutine command.

It is possible to use another program as a function, though it should be warned this does not resample. What this is actually doing is inlining the function and then compiling like normal, so there is no dynamic flow control and absolutely no stack to speak of- the model is still thinking through everything and remembering it's thought processes.

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

# Setup the subroutine
subroutine = forge.new_program(sequences, resources,
                               config, tokenizer)
with subroutine.loop("loop", min=3, max=6) as loop:
    with loop.if("if_rethink", max=3) as if_branch:
        if_branch.run("Rethink")
    with if_branch.else() as else_branch:
        else_branch.run("Think")

# Setup the main program
program.run(sequence = "Setup")
program.subroutine(subroutine)
program.run("Summarize")
```

**Transitions**

This has no real transitions. Instead, it just inserts the associated linked list/graph at that location.

### The .extract command

The .extract statement is used to configure and extract text. It controls what the extraction system later on will return.

Lets suppose "Summarize" in the above marked "output" features. Lets also suppose we have an "auditing" tag on everything:

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

program.run(sequence = "Setup")
with program.loop("loop", min=3, max=6) as loop:
    with loop.if("if_rethink", max=3) as if_branch:
        if_branch.run("Rethink")
    with if_branch.else() as else_branch:
        else_branch.run("Think")
program.run("Summarize")
program.extract(name="output", tags =["output"])
controller_factory = program.compile(backend="PFA.md")
```

Later on in the main training loop we can do:

```python
flow_manager = controller_factory(num_batches=1000, 
                                  batch_size=64)
while not flow_manager.done():
    tokens = flow_manager.next()
    tokens, state = model.predict(tokens, state)
    flow_manager.advance(tokens)
results = flow_manager.extract()
```

with the automated feature produced after compiling in order to 
automatically extract the tags. Note as well that unions of tags
also work. The following for instance extracts anything named output OR tags

```python
program.extract(name="audit_trail", tags=["audit", "output"])
```

**Transitions**

Tags are actually specialized auxiliary information indicated in each ZCP node. Compiling turns them into tensor representations, and every TTFA timestep returns as well what tags are active in the zone. This naturally lets us build up a collection of bool arrays that can be vector-manipulated later into something we can index to get the tagged regions out.

## Tool usage

Tool usage is handled by an auxiliary structure in which you define a tool in terms of it's callback that accepts a string, and then can execute some special commands to integrate with it. One important limitations exists however.

**Input buffer**: There is one statically sized input buffer for the state machine backend, per batch, to hold the results of the tool calls in. Calling new tools will overwrite this result. 
**Capture**: A procedural capture mechanism exists to capture tool calls, where the last zone of an entire sequence is captured from. This capture is sent onto a tool, that dumps into the input buffer. 

### Setup

To setup a tool, you declare its existance in the main program. Note that you have to setup the tools buffer size first as well.

```python
import workflow_forge as forge
resources = make_my_resources()
sequences, config = forge.parse_udpl_file('my_file.toml')
tokenizer = make_my_tokenizer(config)
program = forge.new_program(sequences, resources,
                            config, tokenizer)

toolbox = program.new_toolbox(input_buffer_size=100000, 
                              output_buffer_size=10000)
```

Using the tool is then a matter of using the .capture and .feed commands

```python
tool = toolbox.setup_tool(user_gui_callback)

program.run(sequence = "Setup")
with program.loop("loop", min=3, max=6) as loop:
    loop.capture("Ask", tool)
    loop.feed("Answer")
program.run("Summarize")
program.extract(name="output", tags =["output"])
controller_factory = program.compile(backend="PFA.md")
```

**Transitions**

Under the hood, input and output states are being marked on the final zones of particular sequences in ZCP. Eventually, that gets compiled into the finite state machine backends in the ways that matter.