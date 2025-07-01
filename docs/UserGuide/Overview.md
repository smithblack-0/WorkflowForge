# Frontend Overview

This is the file for the frontend overview.

The frontend, for this definition, is everything that interacts with mechanisms before the SZCP layer, where serialization can happen and be paired with the backend. This includes the formal language specification, the required zcp stages and conceptual architecture stages, and more.

```
[Frontend userspace.............]   [Compiling chain (front)...............]
UDPL (Config) → SFCS (Programming) → ZCP      → RZCP      → SZCP
    ↓               ↓                ↓          ↓             ↓
TOML Files → Python Flow Control → Blocks → Sampling → SerializableIR
```

## Goal

The frontend system has three primary goals:

* Have a very straightforward configuration language and flow control system.
* Handle prompting changes between runs in a straightforward manner. 
* Have a client system that can connect to a backend server system in order to connect the frontend to the backend.

## Terminology

Going forward it is important to keep the following terms in mind:

- **Zone**: A region of tokens delimited by two special text patterns defined in the config (e.g., a [Prompt] to [Answer] span). Zones are the unit of tagging and extraction. Every block is made up of one or more zones. Note that the internal parser actually adds one more zone, but users should not need to worry about that.

- **Zone Pattern**: Special text patterns such as [Prompt] or [Answer] used to indicate the edge of a zone. These are not guaranteed to be single tokens.

- **Block**: A full prompt sequence consisting of one or more adjacent zones (e.g., [Prompt]...[Answer]...[EOS]). All zones in a block are resolved before the block completes. The model may take over any time all original prompt text is exhausted, at which point it completes the remaining zones.

- **Prompt**: A partially completed input string in a block that includes one or more defined zones. Prompts may include placeholder fields to be filled by resources during preprocessing.

- **Tags**: Labels applied to individual zones within a block to enable selective extraction of generated tokens for training, evaluation, or filtering.

- **Flow Control Pattern**: A special text pattern such as '[Jump]' that can be defined in order to make manipulations happen in SFCS. **DANGER** unless escaped, flow control in teacher-forcing prompts are executed immediately, breaking the prompt.

- **Escape Pattern**: A special text pattern such as '[Escape]' which can be defined to make the MOA skip the next transition instruction. Very useful for telling the model what text to emit next.

- **Sequence**: A named chain of blocks that can be invoked by SFCS commands.

## Philosophy

Workflow Forge is designed to be **strongly-typed and fail-fast**. The system will catch errors as early as possible and provide clear diagnostics. Confusing error messages should be considered bugs.

The frontend separates concerns cleanly:
- **UDPL**: Declares what to say (prompt content and structure)
- **SFCS**: Orchestrates how to say it (flow control and execution order)  
- **Resources**: Provides dynamic content injection

## User Interface

The frontend input system consists of the Universal Declarative Prompting Language specifications, config objects, and of course the Simple Flow Control System language as well. These, together, make up the user interface.

**UDPL**

The UDPL system subset was created based on the observation that existing systems, such as LangChain, often mix python flow control and prompt generation together in a manner which is difficult to adjust using automated systems and difficult to use to grasp the overall flow.

Instead, in UDPL the prompts can be defined in terms of blocks, tagging keys can be attached, configs can be defined, and much of the information can be declared in terms of sequences of text to feed under the right condition.

**Block-Based Structure**: UDPL is organized around blocks, which are complete prompt sequences. By including or excluding zone edge patterns in your prompt text, you control exactly where the model takes over generation.

**Tag-Based Extraction**: Zones can be tagged for selective extraction, enabling automatic collection of reasoning chains, final answers, feedback, etc.

For extremely simple programs, one hardly needs to use SFCS, as one can just run a single sequence that feeds the prompt in order and then extract the right strings.

You should see [UDPL](UDPL.md) for more details. 

Once the UDPL toml has been defined, the config and sequences can be produced using `parse_udpl_file` or even `parse_udpl_folder`. This yields the sequence dictionary, then the config.

**SFCS**

Flow control was judged to be much easier to program when you separate what you need to do with how to say it. As such, SFCS invokes from a parsed UDPL specification in terms of the sequences to perform various actions. The core idea is defining a 'program' which, using context managers, can add on natural additional parts to your flow control in a very pythonic manner. Under the hood, you are building a control flow graph that is later lowered.

**Model-Commanded Flow Control**: Unlike traditional programming, the language model decides when to branch or loop by emitting control patterns. Your UDPL prompts must instruct the model when to emit these patterns.

**Context-Based Building**: Flow control uses Python context managers (`with program.loop()`) that create scopes for building workflow graphs.

See [SFCS](SFCS.md) for more details.

**Resources**

The last thing users need to worry about are resources. Resources are the dynamic injection system we provide, and should be used for input, feedback, constitution sampling, or other purposes. Ultimately, a resource just accepts kwargs defined in the UDPL file, or provided in python, and returns a string. That string then resolves a placeholder.

Resources can be resolved at different times based on configuration:
- **Standard resources**: Resolved at compile time (default)
- **Custom type**: Resolved when sequence is referenced  
- **Argument type**: Resolved when workflow factory is called

See [UDPL](UDPL.md) for details on how to specify resource types in your configuration.

This was specifically designed to support sampling of constitutions, the original application, but will be extended to include input caches and other behavior. 

Resources must always be resolved for all placeholders defined in UDPL before the system can execute. However, this resolution can be significantly delayed. In particular, resolution is checked when invoking workflow_factory, and additional information such as text can be passed in to resolve the missing kwargs and will automatically be constructed and resolved.

## Architecture

Ultimately, the point of the entire frontend system is to turn the input directives into a SZCP graph. If this specialized graph is correctly generated, it can be serialized, sent to the backend, deserialized, compiled, and executed. For those more interested in this process, you may be feel free to refer to the [ZCP system](../ZCP/ZCP.md)

For now, it is safe to say the following stages occur:

* Client creation: Bind the client to the backend.
* UDPL compiling: We end up with a dictionary of ZCP sequences, which are just linked lists of zones
* SFCS construction: Lowers sequences to RZCP, integrates flow control, and marks extraction directions.
* Compiling: Results in workflow_factory, which can produce the serializable SZCP sequence
* Factory invocation: With any remaining kwargs, and this then yields a SZCP chain that we can pass into the client to send over to the server.

## Simple End-to-End Example

UNFINISHED - needs to be revised once the client system is finalized.

```toml
[config]
zone_patterns = ["[Prompt]", "[Answer]", "[EOS]"]
required_patterns = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Final"]
default_max_token_length = 20000
sequences = ["blocks"]
control_pattern = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]

[[blocks]]
text="""[Prompt]Consider and resolve a philosophical dilemma
according to the following principles: {placeholder} 
[Answer]
Okay, I should think this through. 
"""
tags=[["Training"], []]
[blocks.placeholder]
name = "constitution_overview"

[[blocks]]
text="""[Prompt]Revise your previous answer after
considering the following additional details.
Make sure to also pretend you are directly answering
the prior response: {details}
[Answer]
Okay, I should begin by thinking through the new point, then
consider if I should revise my answer. Then I give my final 
answer.
[EOS]
"""
tags= [[], []]
repeats = 3
[blocks.details]
name = "constitution_details"
arguments = {"num_samples" : 3}

[[blocks]]
text = """[Prompt]
Consider your reflections so far. State a perfect answer
as though you jumped straight to the right answer.
[Answer]"""
tags = [[], ["Final"]]
```

In python, we will manually define a few principles we wish to follow using the backend resources. Then we will parse the UDPL.

```python
import workflow_forge as forge

# User specifications for the philosophy.
my_philosophy_overview= """
... whatever
"""
my_details = ["...whatever", "...whatever", ...]

# Create resources
resources = {}
resources["constitution_overview"] = forge.StaticStringResource(my_philosophy_overview)
resources["constitution_details"] = forge.StringListSampler(my_details)

# Parse the UDPL
sequences, config = forge.parse_udpl_file('prompts.toml')

# Build workflow with SFCS
program = forge.new_program(sequences, resources, config)
program.run(sequence="blocks")
program.extract(name="synthetic_answer", tags=["Training", "Final"])
factory = program.compile()
```

This runs a simple philosophical reflection exercise that has the model produce a more refined answer at the end; a union between the training and final tags can refine this for synthetic training data purposes. The reflection step runs three times.