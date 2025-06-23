# Universal Declarative Prompting Language

## What is this?

This is the official specification for v1 of the UDPL prompting language. Linters and the ZCP compilers should conform to this spec. This is a frontend/client responsibility.

## Overview 

The Universal Declarative Prompting Language (UDPL) is a TOML-based configuration format for defining structured prompting sequences used in large language model workflows. It allows developers to declaratively specify sequences of prompt blocks, organize them into zones, attach semantic tags, and inject dynamic content using named placeholders.

Each UDPL file defines a set of named sequences, where each sequence contains a series of prompt blocks. A block consists of multiple zone regions between special text patterns, and may specify up to all zones defined in the UDPL config section for teacher-forcing.

Alternatively, by excluding zone edge patterns from the prompt text you let the model generate those regions instead. This allows fine-grained control over when prompting ends and generation begins.

UDPL supports:
- Zone tagging for selective extraction of generated outputs
- Placeholder resolution using resource-backed dynamic content
- Repeat and tagset patterns for efficient data generation
- Hooks for flow control

The outcome of parsing a valid UDPL file or folder is a specification of sequences and a config file, which must later be programmed using SFCS to produce executable workflows.

UDPL itself does not define control flow — it is consumed by downstream systems that orchestrate execution, and prompts the model to use flow control in the way it feels is most relevant.

## Parsing API

```python
import workflow_forge as forge

# Parse single UDPL file (must contain config section)
sequences, config = forge.parse_udpl_file('prompts.toml')

# Parse UDPL folder (exactly one file must contain config section)  
sequences, config = forge.parse_udpl_folder('prompts/')
```

**Single File**: The file must contain a `[config]` section.

**Folder**: Exactly one TOML file in the folder must contain the `[config]` section. All other files contribute sequences but cannot have config sections.

Returns:
- `sequences`: Dictionary mapping sequence names to ZCP node chains
- `config`: Config object with all parsed settings

## Config

All UDPL files must include a single [config] section. The config defines zone boundaries, required patterns, valid tags, and generation limits that govern how all sequences and blocks are interpreted.

```toml
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]
valid_tags = ["Training", "Correct", "Incorrect"]
default_max_token_length = 20000
sequences = ["setup", "reasoning", "conclusion"]
control_token = "[Jump]"
escape_patterns = ["[Escape]", "[EndEscape]"]
tools = ["search", "calculator"]
```

**Note**: For historical reasons, these config fields use "token" naming, though they represent text patterns that may span multiple tokens.

### Config Fields

- `zone_tokens`: Text patterns that define zone boundaries. Zones are the spans between each pair of patterns.
- `required_tokens`: Patterns that must be present in every block's text field.
- `valid_tags`: Labels that can be attached to zones for later extraction.
- `default_max_token_length`: Maximum generation length per zone, unless overridden.
- `sequences`: List of named sequences. Each corresponds to a section of blocks.
- `control_token`: Pattern used by models to trigger flow control transitions.
- `escape_patterns`: Patterns that disable control transitions when generated. Two of them.
- `tools`: List of valid tool names that can be referenced in SFCS.

### Linting Conditions

- `zone_tokens`: Must be list of strings, nonempty, length ≥ 2
- `required_tokens`: Must be list of strings, nonempty, all must be in zone_tokens
- `valid_tags`: Must be list, warn if empty
- `default_max_token_length`: Must be integer > 0
- `sequences`: Must be nonempty list of strings
- `control_token`: Must be nonempty string
- `escape_patterns`: Must be list of exactly two strings.
- `tools`: Must be list of strings (optional)

## Flow Control and Escaping Control Patterns

UDPL is designed to work with SFCS flow control, where models emit special control patterns to trigger workflow transitions. However, this creates a critical issue: if control patterns appear in your prompt text, they will trigger immediately during execution, breaking your intended workflow.

**The Problem**: When teaching a model about flow control or showing examples, you might write:

```toml
# DANGEROUS - will trigger flow control immediately
text = """[Prompt] When ready to continue, emit [Jump] to proceed [Answer]"""
```

The moment the system encounters `[Jump]` during execution, it will trigger a workflow transition, interrupting the prompt before the model can respond.

**The Solution**: Use escape patterns to disable control matching:

```toml
# SAFE - model sees the literal text without triggering control
text = """[Prompt] When ready to continue, emit [Escape] "[Jump]" [EndEscape] to proceed [Answer]"""
```

The escape patterns (`escape_token` in config) create a "safe zone" where control patterns are treated as literal text. This allows you to instruct models about control flow without accidentally triggering it.

**Best Practice**: Always escape control patterns when they appear in prompt text, unless you specifically intend to trigger immediate flow control (which is rarely desired in UDPL blocks).

### Linting and Behavior Conditions

* The number of open escape tokens must equal the number of closing escape tokens.
* Zone splits will not happen inside escaped regions.

## Sequences

A sequence is a list of blocks that together form a coherent prompting flow. Sequences represent the logical stages of your workflow - for example, you might have sequences for "setup", "reasoning", and "conclusion".

Every sequence declared in the config must have corresponding blocks defined in the TOML file. When parsed, sequences produce unlowered ZCP zone chains as linked lists with no flow control - the flow control is added later by SFCS.

### TOML Structure

Use TOML's array-of-tables syntax (`[[sequence_name]]`) to define blocks within sequences:

```toml
[[reasoning]]
text = "First reasoning block"
tags = [[], ["Training"]]

[[reasoning]]  
text = "Second reasoning block" 
tags = [["Context"], []]
```

Each `[[reasoning]]` entry creates a new block in the "reasoning" sequence. Blocks are processed in the order they appear.

### Linting

For each sequence declared in config:
- The sequence must exist at the TOML top level
- The sequence must resolve to a list of valid blocks
- Each block in the sequence is validated according to block rules

## Blocks

### What Is a Block?

A block is a single structured unit within a UDPL sequence. Each block defines a multi-zone prompt template composed of one or more adjacent zones, such as `[Prompt]...[Answer]`. Zones may be fully specified (teacher-forced) or left incomplete to be generated by the model.

Blocks are grouped under named sequences using the `[[<sequence_name>]]` header syntax. The sequence name must match one of the names listed in the config `sequences` field.

Each block is processed in the order it appears. All zones in the block must eventually be resolved before the next block begins, either by prompting or by model generation.

### Block Structure

Every block defines a sequence of zones between special text patterns. For example, with `zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]`, you get two zones:
- Zone 1: Between `[Prompt]` and `[Answer]` 
- Zone 2: Between `[Answer]` and `[EOS]`

By including or excluding these patterns in your prompt text, you control exactly where the model takes over generation.

### Required Fields

**text**: A multiline string containing the prompt. Must include any patterns from `required_tokens`. These patterns mark zone boundaries - zones are the spans between adjacent patterns. Zones not included in the text will be generated by the model.

**tags**: A list of sublists, one for each zone span. Each sublist contains zero or more tags to apply to the corresponding zone. The number of sublists must be exactly `len(zone_tokens) - 1` since zones are the text between the zone edge patterns.

### Optional Fields

**repeats**: An integer. If present, the block will be repeated this number of times. All resource fills will be independently resampled for each repetition. The same tags are used for each repetition.

**tagset**: A list of tag lists for each repetition. If provided, this replaces both `tags` and `repeats`. Each entry in the tagset is a full tag list (same structure as `tags`), and defines tags for one repetition of the block. This is used to produce multiple examples with the same text, but different resource fills and tagging metadata.

**max_gen_tokens**: An integer specifying the maximum number of tokens the model is allowed to generate per zone in this block. **Warning**: Some backends may enforce this limit even during teacher-forcing, which can cause the zone to be truncated. Use with care when manually specifying long zones.

### Linting Conditions

- `text` must exist and contain a string
- `text` may contain placeholders using `{placeholder_name}` syntax
- Placeholders must correspond to resource binding entries
- `tags` must exist and be a list of length `len(zone_tokens) - 1`
- Each entry in `tags` must be a list containing only tags from `valid_tags`
- Exactly one of `tags` or `tagset` must be present
- If `tagset` is used, `repeats` is not allowed
- `max_gen_tokens` must be a positive integer if present

### Placeholders & Resource Bindings

Blocks may include placeholders in their `text` field using `{placeholder_name}` syntax. These placeholders are resolved using external resources at various stages of the compilation process.

Each placeholder must have a corresponding TOML table entry under the same sequence:

```toml 
[<sequence>.<placeholder_name>]
name = "resource_name"
type = "standard"  # optional
arguments = { ... }  # optional
```

#### Basic Resource Example

```toml
[[solving]]
text = """
[Prompt] Consider this principle: {philosophy}
[Answer]
"""
tags = [["Training"], ["Correct"]]

[solving.philosophy]
name = "constitution_sampler"
```

#### Resource with Arguments

```toml
[[solving]]
text = """
[Prompt] Consider this feedback: {feedback}
[Answer]
"""
tags = [["Training"], ["Feedback"]]

[solving.feedback]
name = "feedback_sampler"
arguments = { num_samples = 3 }
```

### Resource Types

Resources can be resolved at different times based on their `type` field:

**type = "standard"** (default): Resolved at compile time. The resource must be provided when creating the program with `forge.new_program()`.

**type = "custom"**: Resolved when the sequence is referenced in SFCS. Allows late binding of resources during workflow construction.

**type = "argument"**: Resolved when the workflow factory is called. Enables runtime parameterization where different values can be provided for each workflow instance.

In line with fail early, we default to standard, which has the most restrictive rules. 

#### Resource Linting

- All `{placeholder}` patterns in text must have corresponding `[sequence.placeholder]` entries
- `name` field is required for all placeholder entries  
- `type` field must be one of: "standard", "custom", "argument" (if present)
- `arguments` field must be a valid TOML table (if present)


### Repeats vs Tagsets

Use `repeats` for multiple examples with same tags:
```toml
[[training]]
text = "Analyze: {point}"
tags = [[], ["Correct"]]
repeats = 3
```

Use `tagset` for different tags per repetition:
```toml
[[contrast]]  
text = "Create argument: {claim}"
tagset = [
  [[], ["Correct"]],
  [[], ["Incorrect"]]
]
```

## Example

```toml
[config]
zone_tokens = ["[Prompt]", "[Answer]", "[EOS]"]
required_tokens = ["[Prompt]", "[Answer]"]  
valid_tags = ["Training", "Final"]
sequences = ["reflect"]
control_token = "[Jump]"
escape_token = ["[Escape]", "[EndEscape]"]

[[reflect]]
text = """[Prompt]Consider this dilemma: {scenario}
[Answer]Let me think through this carefully."""
tags = [["Training"], []]

[reflect.scenario]
name = "dilemma_sampler"
type = "standard"

[[reflect]]
text = """[Prompt]State your final answer.
[Answer]"""  
tags = [[], ["Final"]]
```

This defines a two-block reflection sequence with tagged zones for extraction.