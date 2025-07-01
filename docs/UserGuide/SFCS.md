# Simple Flow Control System

## Overview

SFCS provides Python-like flow control that compiles to GPU-executable workflows. It translates familiar programming constructs (loops, conditionals) into ZCP finite state machines that models can control during generation.

This document covers the commands and syntax for building complex workflows. For setup and basic concepts, see the [Frontend Overview](Overview.md).

## Core Concepts

**Prompt Loading**: You do not specify your prompts here, but load prompt sequences by name that were defined in UDPL. This tends to keep the flow control far more concise and clear. You may pass extra information to resolve placeholders as well. You cannot define what to say in this system, only indicate to run an existing configured prompt.

**Model-Commanded Flow Control**: Unlike traditional programming, the language model decides when to branch or loop by emitting special control text sequences during generation. The prompting for this must have been implemented already in the UDPL spec. This means you are loading flow control prompts you defined in UDPL, not specifying them here.

**Sequences and Zones** Sequences are composed of a number of zones. A lot of the flow control has an effect on the last zone of a sequence.

## Hazard warning

**EXTREME HAZARD WARNING**

The backend is quite dumb in some ways. It does not care where a string came from, it will execute it if it matches a trigger and is not escaped. Additionally it executes as soon as it notices a pattern match, meaning that if you have text:

"This will be fed [Jump] This will not" 

The model will never reach "This will not" as the TTFA will trigger immediately on seeing the end of the jump pattern and immediately change zones.

## Program vs Scope

There are conceptually two objects of great user interest. These are the program and scope.

**Program**: The top-level orchestrator that manages compilation, extraction, and toolboxes. Acts as the main entry point for building workflows. Has all upcoming commands available.

**Scope**: The execution context within flow control blocks. Created automatically by context managers (`with program.loop()`, `with loop.when()`). Scopes handle the actual graph building while Programs manage the overall workflow lifecycle. They cannot perform all commands.

## Resources

Resources can in fact be resolved during construction time, or even compile time, with the right flags.

* **custom**: If the type of a placeholder is defined as custom, that provides permission to resolve the resource as late as when you reference the sequence.
* **argument** if the type of the placeholder is defined as **argument**, it can be delayed until factory invokation.

As always, this language is made for serious workflows and strongly typed. This means you need to release permission in UDPL to use the indicated features. 

## Setup

Assume standard setup from the Frontend Overview:

```python
import workflow_forge as forge
sequences, config = forge.parse_udpl_file('my_file.toml')
program = forge.new_program(sequences, resources, config)
```

Note that resources can be used to resolve placeholders on a placeholder match.

# Commands Reference

All commands that interact with prompts require a `sequence` parameter matching a sequence defined in your UDPL configuration. They also all accept a kwargs argument that can accept a resource dictionary if desired. 


## Universal commands

All commands in this subset are available whether working with a scope or a program.

### .run - Sequential Execution

The most straightforward command - simply runs the specified sequence.

```python
program.run("setup")
program.run("question") 
```

The model will be fed those prompts in order. Note that without extraction commands, this program produces no output.

**Deeper Understanding**: This just attaches one more node in the RZCP chain, on the nominal flow control pathway.

### .loop - Model-Controlled Iteration  

Loops continue until the model emits the jump text to exit. Recall that we specified you need to program your own flow control. For now, it is suggested to have a flow_control.toml file in a folder you keep this in separately, and consider it tightly coupled. Eventually, we will likely provide our own folder.

**Important**: Flow control defaults to loop. Only emitting a jump pattern will exit the loop.
**Warning**: The model will IMMEDIATELY jump upon seeing the control pattern.

```toml
[[loop]]
text = """
[Prompt]
You are in flow control right now. If you are satisfied
with your current answer emit [Escape] "[Jump]" [EndEscape]. Otherwise
answer "try again". If this is your first time seeing this,
just say "proceed". Repeat at least {min} times and at most
{max} times.
[Answer]
"""
tags = [[],[]]

[loop.min]
name = "min"
type = "custom"
[loop.max]
name = "max"
type = "custom"
```

```python
with program.loop("loop", min=3, max=6) as loop_scope:
    loop_scope.run("think")
```

**Deeper Understanding**: At a technical level, this defines flow control in terms of RZCP nodes that loop back on the nominal path, or move to the next sequence on the flow control path.

### .when - Conditional Branching

**Critical**: The "if" branch executes by default. The model must emit a jump pattern to skip it.

```python
with program.loop("loop", min=3, max=6) as loop_scope:
    with loop_scope.when("if_rethink", max=3) as (if_branch, else_branch):
        if_branch.run("rethink")
        else_branch.run("think")
```

**Deeper Understanding**: The nominal color is connnected to the if branch, and the else color to the else branch. These both come to be pointed at the upcoming node. This is done in RZCP.


### .subroutine - Workflow Composition

Inline another program's workflow into the current program.

- **Important**: This is inlining, not function calling. No dynamic flow control or stack - the model remembers all previous context.
- **Important**: While you can invoke subroutine from a scope or a program, the thing being attached must be a program.

```python
# Create subroutine
subroutine = forge.new_program(sequences, resources, config)
with subroutine.loop("loop", min=3, max=6) as loop_scope:
    loop_scope.run("think")

# Use in main program  
program.run("setup")
program.subroutine(subroutine)
program.run("summarize")
```

**Deeper Understanding**: Direct graph insertion - no special transition logic.

## Tool Commands

Tool commands are still universal, but have some additional details that should be clarified. Tool usage is maintained in a mostly nonblocking manner by having a capture buffer and a input buffer in the backend. 

This means that what is actual available for tool usage is the .capture command, which captures on the last zone until generation finishes then immediately invokes the named tool, and the input command, which feeds the input buffer from the last tool result into the last zone of the target sequence in place of generated content.

Buffers are maintained per batch, which can make them quite expensive. 

### Setup and Usage

```python
# Use in workflow
with program.loop("loop", min=3, max=6) as loop_scope:
    loop_scope.capture("ask", tool="search")    # Capture output to tool
    loop_scope.feed("results")          # Feed tool result back into the model.
program.run("answer")
```

### .capture

Invoke a request to use a tool, using the content of the last zone of the sequence. Flags movement into capture buffer, and triggers tool invokation on zone change.

**Deeper Understanding**: Sets the output RZCP flag to something nonzero.


### .feed

Feeds, for the batch, the last results of the capture buffer in place of the generative context. Note this literally feeds the last thing that was run, and will be overwritten by subsequent tool calls.

**Deeper Understanding**: Sets the input RZCP flag to something nonzero.

## Program Only.

### .extract - Tag-Based Output Collection

Configures what data to extract from generated content based on zone tags.

```python
program.run("summarize")
program.extract(name="output", tags=["output"])
program.extract(name="audit_trail", tags=["audit", "output"])  # Union of tags
```

### .compile

Produces a workflow factory object that can be invoked to create individual workflow instances.

```python
# After building your program
workflow_factory = program.compile()

# Create workflow instances with optional runtime parameters
workflow1 = workflow_factory(user_input="What is consciousness?")
workflow2 = workflow_factory(user_input="Explain quantum mechanics")
```

Naturally, this presumes you left a user_input placeholder defined with the argument type back in your UDPL files.

The workflow factory captures the compiled workflow structure while allowing runtime customization through kwargs. Any placeholders not resolved by resources at compile time can be provided when calling the factory.