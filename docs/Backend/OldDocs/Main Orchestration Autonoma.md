# Overview

The GPU "Main Orchestration Autonoma" Core VM is a full fledged computer implemented entirely in terms of tensor operations. This is a literal statement of fact - it has its own instruction memory, decoding pipeline, and subsequent actions it can take in response. Nonetheless, it is implemented entirely using CUDA tensor operations in a manner that can execute in parallel across batches on the GPU. It's clock is the token feeds, it listens to input tokens, interprets the static instruction memory, and inserts new tokens when appropriate.


### Underlying Breakthroughs

The primary theoretical innovations underlying this computer has been produced by bringing together a collection of observations that, collectively, mean you can syncronize a clock to the token stream and use instructions to then make decisions and perform actions based on that stream. These are, in no particular order

1) Token generation can be used as a clock, and vectorized boolean responses performed when matching token stream elements. This provides the input ability to the computer, and the clock cycle.
2) Vectorized numpy "Advanced Indexing" using integers is isomorphic to dereferencing pointers. This then allows us to specify a set of tensors of instructions which can be dereferenced by program count in terms of raw tensor operations, so long as we can then decode the retrieved instructions. In practice, this means storing many parallel tensors that sometimes consist of padding to encode, for instance, jump destinations even if one will never actually be used
3) Overriding the token stream by inserting a precompiled prompt in order is tremendously powerful. This technique can be used to arbitrarily switch between teacher-forcing and generative freewheeling if a triggering mechanism to control the feed and vectorized logic can be found. 
4) The proposed instruction set would be unwieldy if this token stream was directly compiled into an instruction stream. However, by having the instructions point into a data array with a start and end pointer, it is possible to encode information of arbitrary length and bind it to instructions.


Put together, these effects mean it is possible to design a batch-parallel computer which can accept in sampled tokens, replace them if needed, and used the sampling stream itself to make prompting flow decisions. This puts the model in the seat of the commander, and enables the logic of the system. 

### Restrictions

Several restrictions should be noted with regards to the computer before proceeding into the primary architectural layout. These include:

* The **Single Workflow Multiple Streams** (SWMS) restriction: While in theory it is perfectly viable to have multiple instruction streams operating over multiple batches, it significantly complicates the underlying systems, namely by requiring instruction padding and enormously more sophisticated unit tests. For the initial build, we are sticking with SWMS only, which will adequately serve the analysis, mass sampling, and constitutional workflow purposes this system was designed for in the first place. The extension is, however, possible.
* **No General Computing** The MOA is not a general purpose computer. Enough exploration has been performed to confirm that such a computer is in fact possible using this technology; nonetheless, it is not needed and a purpose-built computer is easier to implement. The most notable missing element for general computation is memory access and pointer indirection. This would be an interesting direction to extend the research in.
* **No Agentic Support**: Eventually, this is planned, but at the moment there is no agentic support. Notably, however, stateful models are extremely viable for integration with this system. While stateless models are in theory also possible, they are significantly less efficient.
* **No C++ Backend/Custom CUDA**: While this should eventually be implemented, the first version is going to be written in python, if for no other reason than I do not currently know C++. This should still be fantastically fast, and prove the point for when it is time to do the full system, as most of the computation is being performed through CUDA intermediaries. Presumably, a rewrite in terms of a custom CUDA general purpose GPU will be possible at some time.
* **No Dynamic Instructions**: Since the instructions are compiled ahead of time, there is no ability to interpret new instructions, and the computer is not general enough to support that anyhow. A consequence of this is that there is no such thing as dynamic flow control; while the model can make different stochastic sampling decisions, the prompts these can lead to have fixed flow control decisions and fixed prompting patterns.
* **Only PyTorch Support**: The backend can be implemented in any language with advanced indexing and scatter operations, but for the moment will be implemented only in PyTorch. I will do my best to use functional notation to make it easy on the Jax and TensorFlow community to port it, but I am not doing it myself in this initial release. I will also ensure there are API hooks for this too.


# Breakdown

The overall system is designed to flow through a sequence of steps, repeatedly, accepting the last finite state as an input and producing the next finite state as an output. We visualize it below. Note that as debugging information is drawn from many places, these traces have been omitted, as has the pattern state due to room. The complexity here is making functional event processes.


## Intake Stage


If we were going to define the intake stage, it would look something like this. The intake stage accepts various collections of recurrent information, performs decoding and resetting of hidden states under some conditions, and then outputs a set of three streams which will be critically important shortly. We now discuss the entire intake system. Note that in reality we never actually separate the dictionaries into tool_buffer, pattern_buffer, and triggers_state. 

```text
autonoma_state ───┬───────┬────────────────(state stream)─────────→
                  │       ↓           ┌────(recurrent stream)─────→
tool_buffer ──────│─→┬→[Reset/Default]┘ ┌──(instruction stream)───→
pattern_buffer────│─→┤                  │                        
triggers_state────│─→┘                  │                        
                  │                     │ 
                  └───────┐             │                                                                          
                          ↓             │                                                                          
instruction_memory────→[Decode]─────────┘                                                                          
```

### Inputs

The inputs for this stage are

* **autonoma_state**: Consists of a 2xbatched tensor of (int program_counter, int token_number). Indicates all autonoma information needed to dereference the hidden state and retrieve the instruction state for each batch element. Token number is the token number relative to the start of the zone, and becomes zero on a zone transition.
* **tool_buffer**: A datastructure used to store per batch input/output buffers, with the Tool class handling the exact implementation and typing. It is safe to say, however, that on merging into the recurrent stream this can be accessed through the "tool_buffer" stream. It needs to be reset on zone transitions
* **pattern_buffer**: A special buffer used in the patterns stage later on in transitions. It can be used to detect when we have seen one of our control patterns. It needs to be reset on zone transitions,
* **triggers_state**: Used to keep track of certain transition information, most notably escape stack height. It needs to be reset on zone transitions.

* **instruction_memory**: The compiled instruction memory object. It can be dereferenced when we know our program counter value to get the actual instructions.

### Outputs

The outputs from the stage are three bundled collections of information, which we call the *autonoma stream*, the *state stream*, the *token_stream*, and the *instruction_stream*. 

* **autonoma_stream**: It is just the same autonoma_state input we started with
* **state_stream**: Persistent state information not related to the main autonoma process go here. It includes in a dictionary "tool_buffer", "pattern_buffer", and "triggers_state". 
* **instruction_stream**: An accessable form of the instructions, which have now been derefernced by program counter value. The actual instruction we are working with, in terms of all the relevant zone information.

### Modules

#### Reset/Default

The reset/default system is responsible for ensuring we can get sane default states into the system's stream. It will produce and return a default state 
## Token Stage

The token stage consists of manipulations of tokens, believe it or not! It is where stream interception and overwriting happens. Notably, transitions occur after this stage, which allows for teacher-forcing our own transitions with appropriate logic. Generally the stages involved go off in sequence, so keep this in mind.

```text
(state stream)──────────────┬─────────┬───────────────┬─────────────────────────────────────┐
(recurrent stream)──────────│─────────│──┬─┬──────────│──────────────┬──┬───────┬──┬────────│
(instruction stream)────┬───│───────┬─│──│─↑───────┬──│────────┬─────│──↑───────│──↑─────┬──│
                        │   │       │ │  │ │       │  │        │     │  │       │  │     │  │
                        ↓   ↓       ↓ ↓  ↓ ↑       ↓  ↓        ↓     ↓  │       ↓  │     │  ↓
tokens────→[prompt_feed]─→[tool_insert]─→[timeout]─→[tool_capture]┐
│                                                                          │
│                                                                          │
┤                                                                          └─────────────────────────────────→token_outputs
↑
[NotIsReplaced]

---------- token stage ---------------
```

### Graphical Representation

If we were to represent the stage graphically 


```text
                                                                                                                                          ┌────→tool_needs_callback
                                                                                                                                          │
autonoma_state ───┬───────┬────────(autonoma stream)─────────────┬─────────┬───────────────┬─────────────────────────────────────┐[tool_ready]←┬─→autonoma_state
                  │       ↓   ┌────(state stream)─────────│─────────│──┬─┬──────────│──────────────┬──┬───────┬──┬────────│──↑─┬──┬┐         │   
tool_buffer ──────│─→┬→[Reset]┘ ┌──(instruction stream)──┬────│───────┬─│──│─↑───────┬──│────────┬─────│──↑───────│──↑─────┬──│──┤ │  ↑└─────│┬→tool_buffer
pattern_buffer────│─→┤          │                        │    │       │ │  │ │       │  │        │     │  │       │  │     │  │  │ │  │      │├→pattern_buffer
triggers_state────│─→┘          │                        ↓    ↓       ↓ ↓  ↓ ↑       ↓  ↓        ↓     ↓  │       ↓  │     │  ↓  ↓ ↓  │      │└→triggers_state
                  │             │ ┌(token stream)────→[prompt_feed]─→[tool_insert]─→[timeout]─→[tool_capture]┐ ┌[Patterns]─│→[Transitions]───┘   
                  └───────┐     │ │                                                                          │ ↑           │                 
                          ↓     │ │                                                                          ├─┘           └→[Tag Painting]────→tags
instruction_memory────→[Decode]─┘ │                                                                          │       
token inputs───────────┳─────────→┤                                                                          └─────────────────────────────────→token_outputs
                       ↓          ↑
                      [NotIsReplaced]                                                           
                      
---------- intake stage----------------|----------------------replacement/capture stage ----------------------|---------transitions stage-------  
```

## Inputs and Outputs

The inputs for each step of the finite state machine are the following. Much of this is recursive, with only tokens ultimately not being so.


* trigger_state: Se**tup once, and then maintained by the trigger unit, it can keep track of various hidden information needed by the triggers. This is where, for instance, the stack depth of the escape unit is stored. How triggers are defined and kept modular, and more generally how construction works, will be discussed in some detail later.**
* autonoma_state: The batched program_counter, and token_number. Token number is the number of tokens generated since this zone started.
* instruction_memory: The compiled instruction memories. Tensors of length equal to the number of zones, and addressable by program counters.
* token_stream: The generated tokens for each step.
* tool_buffers: An internal, interpretable tool token capture region which also will specify what tool we are trying to invoke.
* pattern_buffer: A buffer used to cache the last few generated tokens, and thus used to identify when we have achieved a pattern match.

The outputs, meanwhile, are:

* trigger_state: Loops back. Internal.
* autonoma_state: Loops back. The program counter and token number.
* tags: The tag array for each token. Collected and used for token extraction later.
* token_stream: The token stream, with possibly some elements replaced.
* tool_ready: There is a tool ready to run at a particular batch dim. This will use
* debug (not shown): Certain debug information, such as triggers, autonoma state, etc.
* tool_buffers: An internal, interpretable tool token capture region which also will specify what tool we are trying to invoke.
* pattern_buffer: A buffer used to cache the last few generated tokens, and thus used to identify when we have achieved a pattern match.

It should be noted a mostly functional approach is preferred, even if the underlying framework is torch, due to easing any attempt to port this to other frameworks like tensorflow or jax. 

## Streams

There are several primary streams of note

* **Token_Stream** The stream of tokens. Literally that. Not compound
* **Trigger_Stream** The stream of trigger information. Triggers can yield more details, so it consists of both boolean triggered arrays and additional information the trigger listens to and yields for downstream usage. A good example of the latter would be listening to the token_number so the token replacer can feed in the right part of the prompt.

## Stages

There are, roughly, three stages of operation the system goes through.

* **Decoding**: Pattern reading and instruction decoding. We get the instruction of specific relevance to our particular program counters, passing that onto later stages. We also get an array of bools which can be read in order to figure out if a particular pattern has been detected last iteration.
* **Triggering**: Both the triggering process, and the subsequent transition process, occur using this. The downstream logic will respond primarily to triggers, and also sometimes draw from instructions or data. The state transitions for the next step occur here. This stage also acts as an orchestrator and data dispatcher.
* **Substitution and Tools**: Tool usage is captured as needed, tags are painted, and stream tokens are substituted if needed.

## Decoupling and Stream Dispatch.

Triggers and Transitions collectively make up the Stream Dispatch system. 

The Stream Dispatch system packages information into the trigger stream by drawing from an input dictionary of common sources such as decoded instructions, 

## Modules

We now also briefly discuss each model as well

* **[Decode]** Primarily takes in the parallel instruction data and dereferences it, for each batch, into the actual relevant instructions. 
* **[Pattern]** This has been loaded with and listens for patterns. It can be questioned to get a PatternListener, and will produce an array of bools indicating if the pattern was seen. This will raise a true on the clock cycle it sees the last token in the given pattern. It has the pattern.read, which tells us if we are triggering patterns based on the buffer, and pattern.step mechanism, which updates the buffer.
* **[Triggers]** The main trigger unit. During initialization, every subunit in Token Replacement, Tools, or Transition defined TriggerSpecifications and got back TriggerListeners. This then implements the logic in the specification - namely, by feeding in the required keywords and concatenating all trigger results into a boolean array of the expected shape and order.
* **[Transitions]** Responsible for transition connections, naturally. It must emit transition events into the trigger stream, indicating this is going to transition to a new zone.
* **[Token Replacements]** Most of the token replacement activity occurs here, whether it has to do with timeouts or otherwise. The TriggerListeners are able to decode exactly the features we said we need.
* **[Tools]** Tools will, depending on trigger mode, capture input token streams, or possibly just replace input tokens completely with their own content. They have an additional token buffer and tool callback mechanism, which can be accessed through external methods when it is known a tool callback is waiting to be run. 
* **[Transitions]** As you might imagine, controls state transitions. Transitions move us to a new program counter, and usually reset the token_count counter. The default behavior, however, just advances the token_count

