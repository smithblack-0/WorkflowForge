# Universal Declarative Prompting Language

## What is this?

This is the official specification for v1 of the UDPL prompting language. Linters and the ZCP compilers should conform to this spec.

## Overview 

The Universal Declarative Prompting Language (UDPL)\is a TOML-based configuration format for defining structured prompting sequences used in large language model workflows. It allows developers to declaratively specify sequences of prompt blocks, organize them into zones (like [Prompt]...[Answer]), attach semantic tags, and inject dynamic content using named placeholders.

Each UDPL file defines a set of named sequences, where each sequence contains a series of prompt blocks. A block consists of a predictable sequence of a multi-zone regions squeezed between special tokens, and may specify up to all zones that were defined in the UDPL config section. This can be used to teacher-force the prompt.

Alternatively, by leaving the special tokens indicating zone transitions outside the prompt string you may let the model generate this region instead. This allows fine-grained control over when prompting ends and generation begins — entirely from the config file.

In addition to prompt structure, UDPL supports:

- Zone tagging, which enables selective extraction of generated outputs (e.g., for training, evaluation, or feedback)
- Placeholder resolution, using resource-backed dynamic fill-in at ZCP.
- Repeat and tagset patterns, for efficient contrastive or curriculum-style data generation

The outcome of parsing a valid UDPL file or folder is a specification indicating what sequences exist, but lacking any indications of what to do when flow control is encountered. This must later be programmed in using SFCS to produce an actual ZCP IR graph. A dictionary of Zone Sequences is what is
ultimately constructed and returned when parsing.

While this language is intended to work with the WorkflowForge systems, others are encouraged to perform pull requests to make their own extensions for their particular use cases; we hope UDPL can become an industry standard for better prompting configuration.

UDPL itself does not define control flow — it is consumed by downstream systems that run prompts linearly or with flow control. Its design supports both use cases equally, and serves as a flexible frontend for declaratively configuring prompt-based generation.

## Terminology

Going forward it is important to keep the following
terms in mind:

- Zone: A region of tokens delimited by two special
  tokens defined in the config (e.g., a [Prompt] to
  [Reason] span). Zones are the unit of tagging and
  extraction. Every block is made up of one or more
  zones.

- Zone Edge Token: Special tokens such as [Prompt] or
  [Reason] used to indicate the edge of a zone.

- Block: A full prompt sequence consisting of one or
  more adjacent zones (e.g., [Prompt]...[Reason]...
  [Answer]...[EOS]). All zones in a block are resolved
  before the block completes. The model may take over
  any time all original prompt tokens are exhausted,
  at which point it completes the remaining zones.

- Prompt: A partially completed input string in a block
  that includes one or more defined zones. Prompts may
  include placeholder fields to be filled by resources
  during preprocessing.

- Tags: Labels applied to individual zones within a
  block to enable selective extraction of generated
  tokens for training, evaluation, or filtering.

- Flow Control Token: A special token such as '[Jump]' that can be defined in order to make manipulations happen in SFCS. **DANGER** unless escaped, flow control in teacher-forcing prompts are executed immediately, breaking the prompt.

- Escape Token: A special token such as '[Escape]' which can be defined to make the MOA skip the next transition instruction. Very useful for telling the model what token to emit next.

## Config

All UDPL files must include a single [config] section.
If parsing a folder, one file in the folder must
contain this section. The config defines the set of
zone boundaries, required zones, valid tags, and
generation limits. These options are global and
govern how all sequences and blocks are interpreted.

A typical config looks like this:

```toml
[config]
zone_tokens = ["[Prompt]",
               "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Reasoning]"]
valid_tags = ["Training", "Correct", "Incorrect1",
              "Incorrect2", "Incorrect3"]
default_max_token_length = 20000
sequences = ["setup", "loop", "solving", "concluding"]
control_token = "[Jump]"
escape_token = "[Escape]"
```

- `zone_tokens` defines the edge of each zone.
  Zones are the spans between each pair of tokens. A fully generated block must contain these tokens in order to be valid.
- `required_tokens` lists tokens that must be present
  in every blocks text feature as part of the prompt.
- `valid_tags` are the labels that can be attached
  to zones for later extraction.
- `default_max_token_length` sets the maximum token
  budget per zone, unless overridden in a block.
- `sequences` is the list of named sequences. Each
  corresponds to a section of blocks in the file.
- `control_token` is a special token used by the model
  to trigger flow control transitions.
- `escape_token` is a special token that disables a
  control transition when generated. This can skip
  zone advancement or flow control.

This configuration governs the interpretation of all
blocks defined in the file. Note you will see errors 
later on if you do not provide singular special tokens
in your tokenizer corresponding to this configuration.

If desired, additional entries can be parsed into 
you config. All entries in the config are returned as named by the Config object parsing generates. Additionally, the misc folder contains the raw toml parse if you have extra terms you want to place into your config.

### Linting/Parsing conditions

* Is zone_tokens present? Is it a list of strings? Is it nonempty? Is it of length at least two?
* Is required tokens present? Is it a list of strings? Is it nonempty? Are all required tokens in zone tokens?
* Is valid_tags present? Is it a list? Warn if empty
* Is default_max_token_length present? Is it an integer > 0?
* Is sequences present? Is it a list? Does the list contain strings and is nonempty? 
* Is control_token present? Does it contain a string? Is that string nonempty?
* Is escape_token present? Does it contain a string? Is that string nonempty?

### Config Object

The config object has fields which have been covered:

- zone_tokens
- required_tokens
- valid_tags
- default_max_token_length
- sequences
- control_token
- escape_token

It also has the additional features

- special_tokens: A list of all special tokens needed based on the config. 
- misc: The raw parsed toml. Use it to get your custom values for setup purposes and centralize your config.

## Sequences

A Sequence is a list of blocks - more on that shortly - which together makes up a feed of prompt instructions that we are declaring we want to follow. Sequences have the restriction that for every sequence declared in the config, a sequence with blocks attached must actually exist. When a sequence is parsed, it produces an unlowered ZCP zone chain as a linked list with no flow control. What is actually returned by the UDPL parser is the aforementioned config, and a dictionary of sequences to these chains.

### TOML

To declare a sequence in toml, first make sure there is a valid sequence entry in the config. Then use the [[list]]toml indicator in order to indicate what to put the list on and in what order. As a simple example, the following would make a toml file with a dictionary called example filled with {item : 1} and then {item : 2}.

```toml

[[example]]
item = 1

[[example]]
item=2
```
Naturally, the above is not valid UDPL. You would have to specify example as a sequence in the config, and then define valid blocks. Lets talk about blocks now.

### Linting/Parser

For each sequence in the config

* Does the sequence exist at the TOML top level data? 
* Does the sequence resolve to a list?
* For each element in the list invoke the block parser/linter and check validity
* If constructing ZCP by parsing, merge the linked list between steps.

## Blocks

### What Is a Block?

A block is a single structured unit within a UDPL
sequence. Each block defines a multi-zone prompt
template composed of one or more adjacent zones, such
as [Prompt]...[Answer]. Zones may be fully specified
(teacher-forced) or left incomplete to be generated by
the model.

Blocks are grouped under named sequences. Each block
must appear under a header of the form:

[[<sequence_name>]]

The sequence name must be one of the names listed in
the `[config]` section under `sequences`.

Each block is processed in the order it appears. All
zones in the block must eventually be resolved before
the next block begins, either by prompting or by model
generation.

### Block Fields

Each block must define a minimum of two fields:
- `text`: The prompt template containing zone tokens
- `tags`: A list of tag lists, one per zone span

Optional fields may also be included to control how
blocks are repeated, tagged, or filled dynamically.

---

#### Required Fields

- `text`: A multiline string containing the prompt.
  It must include any tokens listed in the
  `required_tokens` field of the config. These tokens
  mark zone **boundaries**, not zones themselves.
  Each zone is defined as the span between two
  adjacent zone tokens. Zones not included in the
  `text` will be generated by the model.

- `tags`: A list of sublists, one for each zone span.
  Each sublist contains zero or more tags to apply to
  the corresponding zone. The number of sublists must
  be exactly one less than the number of `zone_tokens`
  declared in the config. This is due again to the fact
  that zones are the text between the zone edges. This may sometimes be replaced with tagsets, see below.

#### Optional Fields

- `repeats`: An integer. If present, the block will be
  repeated this number of times. All resource fills
  will be independently resampled for each repetition.
  The same tags are used for each repetition.

- `tagset`: A list of tag lists for each repetition.
  If provided, this replaces both `tags` and `repeats`.
  Each entry in the tagset is a full tag list (same
  structure as `tags`), and defines tags for one
  repetition of the block. This is used to produce
  multiple examples with the same text, but different 
  resource fills and tagging metadata.

- `max_gen_tokens`: An integer specifying the maximum
  number of tokens the model is allowed to generate
  per zone in this block. **Warning**: Some backends
  may enforce this limit even during teacher-forcing,
  which can cause the zone to be truncated. Use with
  care when manually specifying long zones.

- `[<sequence>.<placeholder>]`: One entry per
  placeholder used in `text`. Each must define the
  `name` of a resource to call, and may include
  `arguments` if needed. These interact with the 
  optional but recommended resource system.

### Text Rules

the `text` required field has certain flow control responsibilities that also need to be formally checked. In particular, it does not make sense to include a flow control token --- in our examples "[Jump]" but exact details depend on configuration --- in the text stream without escaping the jump first; otherwise, the TTFA will happily follow flow control due to teacher-forced prompts. This is checked for.

### Placeholders & Resource Bindings

Blocks may include placeholders in their `text` field
using `{placeholder_name}` syntax. These placeholders
are resolved at runtime using external resources.

Each placeholder must have a corresponding TOML table
entry under the same sequence. The structure is:

```toml 
[<sequence>.<placeholder_name>]
name = "resource_name"
arguments = { ... }      # optional
```

#### Basic Resource Example

This example uses a static resource named `principles`.

```toml
[[solving]]
text = """
[Prompt] Consider this: {philosophy}
[Answer]
"""
tags = [["Training"], ["Correct"]]

[solving.philosophy]
name = "principles"
```

#### Resource with Arguments

This example uses a sampler that returns multiple
points, controlled by `arguments`. This is useful
when you are sampling from consistituions of points,
for example.

```toml

[[solving]]
text = """
[Prompt] Consider: {feedback}
[Answer]
"""
tags = [["Training"], ["Feedback"]]

[solving.feedback]
name = "feedback_sampler"
arguments = { num_samples = 3 }
```

Each resource must return a string when invoked. If a
block is repeated, the resource is resampled each time. Resources will be sampled when the ZCP is compiled at the start of each batch. This means no dynamic sampling during the batch, but also means dynamic changes and feedback between batches are possible.

#### Advanced: Resource Types

You may optionally specify a `type` field for a
placeholder binding. This allows extensions to UDPL
that change how resources are interpreted, but removes
the general cross-usage of resources from being shared
among all sequences to being a sequence speficific dynamic
dependendency.

One built-in extension is `type = "flow_control"`.
These resources inject structured values into the
prompt, such as loop bounds or flow-triggering tokens.

```toml
[[loop]]
text = """
[Prompt] Repeat {min} to {max} times. Emit the
[Escape] "[Jump]" token when you are ready to break.
Otherwise just say continue. 
[Answer]
"""
tags = [[], []]

[loop.min]
name = "min_value"
type = "flow_control"

[loop.max]
name = "max_value"
type = "flow_control"
```

**Note**: When `type` is used, the resource is bound
to the sequence in which it appears. All such
resources must be resolved before that sequence can
be compiled. This allows more specific or dynamic
control, but also means a compiled sequence must
provide those resources. This is not true when 
type is not used, in which case this resource
can be resolved from the general pool.


### Repeats & Tagsets

Blocks can be repeated using either the `repeats`
field or the `tagset` field. These mechanisms allow
you to generate multiple examples from a single
prompt structure.

#### Using `repeats` with `tags`

The `repeats` field is an integer. It causes the block
to be repeated N times using the same `text` and `tags`.

Each repetition will independently resample any
resources used in the block.

```toml
[[training]]
text = """
[Prompt] Analyze the following principle: {point}
[Answer]
"""
tags = [["Training"], ["Correct"]]
repeats = 3

[training.point]
name = "principle_sampler"
arguments = { num_samples = 1 }
```

This produces three blocks with the same tagging and
text structure, but different sampled content from
the resource.

#### Using `tagset`

The `tagset` field defines a list of tag lists, one
per repetition. It replaces both `tags` and `repeats`.

Each entry in the tagset must match the structure of
a valid `tags` field: one list per zone.

```toml
[[contrast]]
text = """
[Prompt] Create a flawed argument based on: {claim}
[Answer]
"""
tagset = [
  [[], ["Incorrect1"]],
  [[], ["Incorrect2"]]
]

[contrast.claim]
name = "claim_sampler"
arguments = { num_samples = 1 }
```

This creates two blocks with the same structure but
different tagging metadata.

#### When do I use tags vs tagset?

Exactly one of `tags` or `tagset` must be used per
block. If `tagset` is used, `repeats` is not allowed. 
Use `repeats` to create multiple examples with the
same tags and varying content. Use `tagset` to apply
different tag sets to each repetition.



## Blocks

After a config, a sequence of [[blocks]] are defined. 
These each will be returned in the sequence they are
defined, and are a complete collection of all intentions
for this region. They contain

- A required `text` field: a string representing the
  full prompt, optionally containing `{formatting_fill}`
  placeholders. This must use any required tokens in 
  the string.
- A required `tags` field: a list of  sublists,
  specifying tags for each of the three logical spans:
  `[Prompt]`, `[Reasoning]`, and `[Answer]`.
  Each sublist must be present, though it may be empty.
  This is one minus the number of zone tokens in length
- If placeholders are defined then `[blocks.<fill>]`
  entries corresponding to each `{fill}` in `text`: 
  these define how to invoke callbacks to 
  resolve each placeholder using external resources.

#### Simple Examples

An example of a simple valid block compatible with the
earlier config would be as follows. Notice how all four
prompt tokens are provided. This also means all three
zones are specified, and as such the entire example
would be teacher-forced. Notice as well we tag each of the 
three zones as train.

```toml
[[blocks]]
text ="""
[Prompt] 
Think about what it would take to generate good philosophical scenarios. Just tell me when you are done
[Reasoning]
Okay, it looks like the user wants...(omitted)
[Answer]
I am done
[EOS]
"""
tags = [["Train"],["Train"],["Train"]]
```

The model can be made to generate content by not providing
all these zones. Generation will pick up immediately after
the last token, and the generation logic would then be expected
to advance to the next zone when the next zone boundary token
is emitted. If this followed the last block, it would have then
been loaded.

```toml
[[blocks]]
text ="""
[Prompt] 
Now, actually make an interesting philosophical scenario
[Reasoning]
Okay, let me think this out.
"""
tags = [[],[],["Answer"]]
```

Notice that the model takes over right after the end of
the string. Additionally, since we are only tagging the 
answer we can later very easily only extract those tokens.

### Per-Zone Token Limits

Each block may optionally define a `max_gen_tokens`
field. This sets a hard limit on the number of tokens
that may be emitted per zone — whether the zone is
teacher-forced or generated.

The limit applies **per zone**, not per block.

Once the limit is reached, the default backend advances to 
the next zone, even if more prompt tokens remain or
no zone boundary has been emitted by the model.

```toml
[[revision]]
text = """
[Prompt] Revise your prior answer using this hint:
{hint}
[Answer]
"""
tags = [["Training"], ["Training"]]
max_gen_tokens = 128

[revision.hint]
name = "hint_sampler"
arguments = { num_samples = 1 }
```

**Warning:** The default backend enforces this limit
strictly, even for teacher-forced zones. If a zone's
content exceeds `max_gen_tokens`, the system will
truncate the zone and move on. This applies even when
the zone is provided entirely by the prompt, and is 
a trade-off made to make the backend simple enough to
exist purely using vectorized tensor logic.

Other backends may handle this differently, or upgraded
versions may eventually include memory protection.

To avoid unintended cutoff, either omit the field
or ensure the limit exceeds the longest expected zone.
It is intended that the compiler will eventually emit an 
error when this happens. 

# Parsing Details

When invoked, one ends up with a Config, and a Sequences dictionary. The sequences dictionary, as you would expect, maps to the individual sequences. These sequences are arrays of Zones, which are initial primitives stating, in untokenized form, what they need, when, and how. They will eventually become ZCP naturally.

# Using the library.

## Straightforward example.

### UDPL file.

Suppose we have a straightforward UDPL file. This is 
a simple straightthrough self-play example.

```toml
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Final"]
default_max_token_length = 20000
sequences = ["blocks"]
control_token = "[Jump]"
escape_token = "[Escape]"

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
"""
tags= [[], []]
[blocks.details]
name = "constitution_details"
arguments = {"num_samples" : 3}
repeats = 3

[[blocks]]
text = """[Prompt]
Consider your reflections so far. State a perfect answer
as though you jumped straight to the right answer.
[Answer]"""
tags = [[], ["Final"]]
```

This runs a simple philosophical reflection exercises that has the model produce a more refined
answer at the end; a union between the training and correct tags can refine this for synthetic training data
purposes. The reflection step runs three times.



### Code 

In python, we will manually define a few principles we wish to follow using the backend resources.
Then we will parse the UDPL.

```python
import workflow_forge as forge
from mycustomcode import parse_constitutions

# User specifications for the philosophy.
my_philosophy_overview= """
... whatever
"""
my_details = ["...whatever", "...whatever", ...]

# Create resources
resources = {}
resources["constitution_overview"] = forge.StaticStringResource(my_philosophy_overview)
resources["constitution_details"] = forge.StringListSampler(my_details)

# Parse the UDPl. Sequences now contains a 'block' factory that makes a sequence factory
sequences, config, tag_converter = forge.parse_udpl_file('prompts.toml', resources)
```

If you are using the SFCS flow control system, this could then be continued into a tagging and extraction program that compiles to the ZCP IR. This is straightforward

```python
program = forge.new_program(sequences, resources, config, tokenizer)
program.run(sequence="blocks")
program.extract(name="synthetic_answer", tags=["Training", "Final"])
factory = program.compile()
```

