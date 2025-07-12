# Introduction


Recall the core idea that makes this work is that the computer listens for tokens and transparently intercepts then replaces them:

```
Model generates: "I need to think more... [Jump]"
                                           ↑
Workflow Forge intercepts this pattern ────┘
                 ↓
Instantly replaces stream with: "Let me reconsider the problem..."
                 ↓  
Model continues generating from new context
```

Fortuitously, this process of listening and replacing tokens on demand is isomorphic to a vectorized finite state machine with recurrent state. This means it can implemented in a parallelized manner with separate program counters and a pipeline architecture.  Information flows through the transforms in terms of certain streams, and instruction memory can be programmed once on computer initialization. This, in it's instanced form as deployed here, is referred to as the VM Core

The VM Core is a full fledged computer implemented entirely in terms of tensor operations. This is a literal statement of fact - it has its own instruction memory, decoding pipeline, and subsequent actions it can take in response. This document covers the core of it. External logic would collect the token stream for extraction, user return, and processing the tool callback and is not shown.

This document in particular focuses on the flow of information after installation. Setting up the objects with their instruction data, and indeed compiling a LZCP chain into instructions, is a construction activity that is performed in other logic and documented in other files.

## Program Interface

The VM Core is not intended to be user-facing, and is intended to be wrapped in user collection and interfacing logic. Ultimately, using the class revolves around providing the model tokens. Generation in the NLP field produces a sequence of tokens, one for each step. These tokens are generated across all batches. This sequence is fed into the computer. It looks something like

```python
outputs, state = core(tokens, state)
tokens = outputs["tokens"]
tags = outputs["tags"]
tools_status = outputs["tools_status"]
```

In this example

* **tokens**: Shape (Batch), and integer tensor. The input, and later on the substituted output. Most of the flow control behavior is governed by this token stream, and the computer will both listen to it and overwrite it to prompt the model, or listen to its responses.
* **tags**: Shape (Batch x L), bool, where L is the number of tag options. Used by higher level logic, but always the same within the same given zone. It is a boolean array that can be used to extract information later
* **tools_ready** Integer. Indicates cases with tool callbacks that now need to be run. A -1 means run no tool callback. Requires checking in python. 
* **state**: Hidden state used by this mechanism. 

If you have a core class, and a set of tokens, you may engage the computer

# Design


T brielfy discuss the overall design of the system, as observed by token and state information flowing through the model. This can be adequetely visualized in terms of the stages of operation as a datastream performing transforms on the token stream and hidden states. This is visualized as follows:

```text

state_in ───[Setup]──→autonoma_states ─────┬────────────────────┬────────────────────┐          ┌→[Packing]──→state_out
                │ └──→hidden_states ─────┬─│────┬─────────────┬─│─────────────────┬──│───┬──────│──↑ ↑
                └────→named_states ────┬─│─│────↑──┬────────┬─│─│──────────────┬──│──│───↑──┬───│────┘
                                       │ │ │    │  ↑        │ │ │              │  │  │   │  ↑   │
                                       ↓ ↓ ↓    │  │     ┌──↓─↓─↓──────────┐   ↓  ↓  ↓   │  │   │   
tokens_input →token_stream────────→[Token Manipulation]──┴→[Transitions]─┐ └─→[Post_Processing]─│────────────→outputs
                                     ↑                      ↑            ↓            ↑         │
                                    aux                    aux    new_autonoma_state──┴─────────┘
```

The primary stages of operation can be clearly observed here. It should be noted that arrows pointing into a t indicate a dictionary join operation, and that all of the above are typically datastructure streams of some sort. The **Setup** and **Packing** stages have to do with default state, resetting to defaults, and packaging hidden state for passing between rounds. The three other sections called **Token Manipulation**, **Transitions**, and **Post Processing** are stages that can be loaded with modules which go off in order to accomplish whatever task is desired.

## Initialization and instructions.

It is intended that the GPU VM Core be intialized with a list of modules in the order that they are needed. This will consist of three lists, one for each stage (those will be defined shortly). Getting this list in the right order and with all needed dependencies cross-injected is not the point of this class. We also should note that instruction memory 


## Streams and IO.

To understand how information flows through the computer, it is useful to discuss what a stream is. A **stream** is a channel of information that passes through subsequent stages that can be modified by modules as it passes through, or used as auxiliary information to make changes in the other stream. The Stage of the module controls what streams can be modified. There are three primary streams in 

* Token stream: A dictionary with an entry of "tokens"; additional information can be placed in it as well to survive between stages.
* Auxiliary stream: Helper information can be placed here to be consumed by a later module. Is created empty at the beginning of a stage and vanishes at the end. Think "locals". These are basically empty dictionaries that get discarded (usually)
* Autonoma state: A batched int tensor tensor consisting of the program counter, and the number of tokens processed within this zone. This is sufficient to uniquely retrieve instruction memory. Depending on stage there may be only initial state, or initial and final state available. The autonoma state governs most behavior in the computer. Depending on stage you may have access to both the old and the new autonoma state.

There is, it should be noted, one final "stream" of sorts used for IO, which is the state object that travels between invocations. You can see it being unpacked in setup, and packed in packing. This can be None if we just started out. This is a dataclass containing three pieces of information

* autonoma_state: This is exactly what you think it is
* named_states: A dictionary of Dict[str, Any] mapping onto states that were defined with names; set and get methods are provided to get or set to these states.
* hidden_states: A List[List[Any]] encoding of the hidden states for the three subsequence stages, and the modules inside. Notably, while Any is usually a pytree with tensor leaves if it is a string we retrieve the information from named_states instead, and insert back into it.

## Stages

The process is broken up into four stages. these stages are:

* **Setup**: Has a special purpose, which is setting up and resetting hidden state for individual modules.
* **Token Manipulation**: All primary token manipulation should occur here, including replacement
* **Transitions**: Makes changes to the finite state.
* **Post Processing**: Can make final token changes and ready the output. Can look ahead to the next state.
* **Packing**: Puts everything together to produce the next state object.

The *Token Manipulation*, *Transitions*, and *Post Processing* stage can each be loaded with the associated module type, which is a subclass of the module abstract type, in order to get a specific job done. When modules are loaded into a stage in this manner, they will go off in the loaded order. The **Setup** stage has an additional purpose related to managing the hidden state. In particular, the setup stage sets up default states, and also resets states to default upon bound triggers resolving.

## Modules
Modules are the entities that are actually loaded into stages. The main class for modules is the AbstractModule. Every module has it's own private hidden state bound to it and fed recurrently, and as such AbstractModules require implementation with information that tells us how to configure this. In particular, an AbstractModule must specify:

* field state_name: Optional[str]: Tells us whether the module posssesses a named state we wish to display in the named state system. Useful for tools.
* field triggers: List[str]: Triggers to reset state to default on. Usually "zone_change"
* method make_default_state(batch_size): Produces a pytree with tensor leaves that makes a default state for the indicated batch size.

These have the exclusive effect of changing how the setup stage performs, and how information is stored in the state object. In particular:

* make_default_state is used to construct the default state, * triggers lets us know when to reset to defaults and will cause that state to be reset on that batch when the trigger condition is met
* if state_name is not none we can get and set the name by state in the state object. This makes updating and retrieving tool states easy.

## Specific Stages and Modules 

Each stage is dedicated to a particular purpose, with particular rules for what belongs there and particular requirements on their input/output patterns. All stages beyond setup have a specific abstract module that provides the contract for them. We discuss these stages, their abstract modules, and the planned pieces here

### Token Manipulation

#### Stage Details

The token manipulation stage has one purpose and one purpose alone: **Replace and overwrite tokens as required by the current state**. 

The modules are effectively a sequence of transforms that transform the token_stream into a final output. Failing to put token modifications here will prevent the VM from listening to teacher-forced transitions; while technically possible replacing or modifying tokens should never be done in post-processing.

Once this stage has run, all tokens that need to be replaced should have been and downstream units can listen to them to perform state updates and manipulate various actions.

#### Module Details

The AbstractTokenManipulatorModule object specifies the contract the module must use, and also gives us a good idea of what information will be passing in and out. It has the input/output format of:

* **Input**: hidden_state, autonoma_state, token_stream
* **Output**: hidden_state, token_stream

The updates will be integrated into the associated streams, while the hidden states are held privately. The modules that need to be implemented, in order and currently, are

* **Aux_Setup**: Sets up a feature called "has_been_replaced" that is of size tokens and false
* **Prompt Feeder**: Feeds a prompt into the token stream until it runs out of tokens. This means replacing tokens in a prompt feeding replacement state based on autonoma state until exhausted, then passing through unchanged. Releases an update to "tokens". Also releases an update to "has_been_replaced" based on whether or not the prompt was overwritten.
* **Tool Feeder**: Feeds the inputs from a tool buffer into the stream. This listens to has_been_replaced and will only start feeding AFTER all tokens in the prompt feeder have been exhausted.  
* **Timeout**: The timeout module does NOT respect has_been_replaced. It offsets when it starts feeding based on the length in tokens of the zone advance pattern to ensure sequences never exceed the token length. Sets "has_been_replaced" to true.
* **WasTeacherForced**: Moves a copy of "has_been_replaced" onto the token stream, under the name "teacher_forced".

### Transitions

#### Stage Details

The transitions stage performs everything which has to do with updating the hidden state. It listens to the token stream, uses hidden state to track token strings, matches those against patterns, and uses that to make flow control decisions. Ultimately, the stage has one purpose: Turn the auxiliary_stream into a new autonoma_state.

* This stage feeds the token_stream into all its modules. It does not pass a token_stream back out.
* The auxiliary_stream is expected to contain a feature called "autonoma_state" which is a batch x 2 int tensor by the end. This will indeed become the new autonoma state.

#### Module Details

Modules in this system are dedicated entirely to modifying the autonoma state. The contract for a AbstractTransitionModule is the following:

* **Inputs**: hidden_state, autonoma_state, auxiliary_stream, token_stream
* **Outputs**: hidden_state, auxiliary_stream

As you can observe, there is no intention whatsoever to allow token stream modifications. The modules which are needed at the time of this writing are:

* Patterns: Uses hidden state to track the token stream. Emits a bool tensor that can be interpreted through a PatternMatch object by downstream modules to tell if a particular pattern was observed. Emits a "patterns" bool array into the auxiliary stream. Hidden state resets on zone transition.
* Escape: A stack-based escape mechanism that counts how many escape vs endescape patterns have been seeing. Emits a bool tensor in the "escaped" channel. True means it is escaped, and we should not take transition actions.
* Transition: Nullified by Escape pattern. Responds to jump and advance commands from the model, as detected in the pattern unit. Emits an "autonoma_state" feature. Under most conditions without control this advances token_number by one. Else, it sets token number to zero and changes the program counter to either the next zone or the jump destination.

### Post Processing

#### Stage Details

A variety of final bits of logic are taken care of in this stage. This includes capturing tool usage streams, token painting, and other mechanisms. These should all be mostly decoupled, and so an auxiliary stream is not provided. Instead, it is expected you turn the token stream into the final output format.

* This stage feeds the token stream through all the modules
* Modules get to see the next autonoma_state, and the named_states dictionary as additional context.

#### Module Details

Modules in this section have the contract:

* **Input**: hidden_state, autonoma_state, next_autonoma_state, token_stream, named_states
* **Output**: token_stream

As you can observe, token stream is passed between modules. The modules which are planned include

* TagPainter: Provides the tag array for the zone in the "tags" dictionary feature.
* ToolCapture: Views the "token", "was_teacher_forced", and the autonoma state. Uses this to decide if we are in an output capture zone, and we are not teacher forcing. If so, we capture. 
* ToolResolution: Sets up tool resolution objects. These are containers containing the capture buffer, the associated callback ids, a bool for each indicating if a callback is ready to run, and an integer for each buffer indicating the length of the stored content. They can show using .is_tool_ready() if there is a tool that is ready to be used, and .break_apart() can then break it into sections in python. It detects and sets the bool is ready flag by seeing when we are in a capture zone, but the next iteration will not be. It sets the "tools_status" feature.

### Packing/Unpacking

Technically, the master object has one additional "stage" involving packing away a state object, or unpacking it for usage.