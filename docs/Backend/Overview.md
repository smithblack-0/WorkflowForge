# Overview


## Background

The GPU "Main Orchestration Autonoma" VM is a full fledged computer implemented entirely in terms of tensor operations. This is a literal statement of fact - it has its own instruction memory, decoding pipeline, and subsequent actions it can take in response. Nonetheless, it is implemented entirely using CUDA tensor operations in a manner that can execute in parallel across batches on the GPU. This is achieved through several key innovations which have all been brought together.

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

## Design

### Programming, Inputs, Outputs, and Methods

#### Programming and Setup

The central object involved in the system is the MOA_VM. It is programmed using a harvard architecture by loading in an Instruction object - under the hood this object is a dataclass that can be dereferenced in parallel using a batched progam counter. The instructions, and data backing it, is statically compiled from a workflow once, and then applied across all batches. A consequences of this is it is not possible to alter the prompt data while running, though tools provide a narrow exception to this rule.

#### Inputs and Outputs

The MOA_VM is operated under normal circumstances by providing it with tokens, which it then uses to make all relevant decisions. One batch of tokens, corrosponding to the tokens generated during this step, should be provided; this should not be the collection of all tokens up to this point, but only the tokens for the current generative step.

The return, meanwhile, is then simply the tokens to be utilized. This is sometimes the input tokens that have been passed through unchanged, or sometimes replaced to perform prompting and teacher-forcing. This is the primary mode of generative operation, and these tokens can now be inserted into the model in whatever way is compatible to produce the next set of predictions.

#### Methods and Properties

Several crucial properties exist on the MOA_VM

* **.done**: Tells us whether all zones have advanced until in their ending position. False otherwise. 
* **.callbacks_ready**: Tells us if a callback is queued and needs to be resolved

Also several crucial methods exist

* **.extract()**: When generation is done, extracts and detokenizes a list of dictionaries containing the extraction commands for each individual stream that were programmed in UDPL and SFCS earlier.
* **.generative_sequence()**: Extracts and detokenizes the raw process stream, and returns a list of strings correlated with each batch sample. What the model actually saw.

The debugging ones include information that is informative about how the state machine works:
* **.decisions()**: Returns a list of the zones visited in the order that is relevent. It is highly recommended to cross reference it with workflow.visualize() if you have any errors. zcp.nodes.discover_all_nodes provides a map that is synronous with visualize.
* **.raw_stream**: The untokenized raw stream the model used.
* **.status**: The status, in terms of zone, token_num_in_zone, triggers, for each step of the state machine corrolated with raw_stream.

### Overall State Machine

There are three features that determine, exactly and statefully, precisely what will be done with a given input stream. 

* **program_counter**: The zone, or instruction, being dereferenced and decoded.
* **token_num**: The number of tokens that have been generated within this zone. Used for teacher forcing, and switching to free generation.
* **triggers**: A bool array per batch, representing if we have noticed any special patterns. Used for instruction transitions primarily. The trigger can be said to be in a Trigger State, and masks can be retrieved from the trigger abstraction to respond to particular trigger statuses.

Injection logic goes off after all three of these are known, and interprets behavior according to the instructions. The injection itself depends only on the program_counter and token_num variable, making it quite robust. Triggers, meanwhile, largely controls state transitions.

### Instruction Set: Control

Instructions, often referred to as 'zones', are collections of tensors that can be integer or boolean which all have the same length; this length then correlates with the number of zones. These often interact with data storage in some way. Each instruction possesses the following information. They can be dereferenced, in parallel, using a batch of program counters.

This should likely be corrolated around one main abstraction, with several subroles. 

#### zone advancement

* **zone_advance_active**: (Bool) A boolean flag that indicates whether a zone advance, which just increments the program counter by one, is a valid exit state from this instruction
* **zone_advance_mask** (Bool) The mask corrolating to the trigger state that performs a zone advance.

 ### jump advancement

* **jump_advance_active**: (Bool) A boolean flag that indicates whether a jump advance, which dereferences the jump address, is valid for this zone
* **jump_advance_mask** (Bool) The boolean mask indicating a trigger state that calls for a zone advance.
* **jump_address** (Int) The program counter 

#### input and output

* **input**: (Bool) indicates this is an input instruction that should feed from the inputs buffer within this zone instead of allowing free generation
* **output** (Bool) indicates the contents of the zone should be captured into the output buffer for resolution with a tool callback
* **output_callback** (Int) names the integer associated with the callback for the zone. Can be decoded elsewhere.

### Instruction Set: Data Feeding

Tokens are fed based on the state as well, with a state machine taking care of this with independent instruction sets. Additionally, tags need to be painted as well.

* **token_start_pointer**: (Int) The location in the prompts data tensor to start loading tokens from when transferring into this zone. Corrolated with the number of zones
* **token_end_pointer**: (Int) The location in the prompts data tensor to stop loading tokens from and start free generating. Corrolated with the number of zones
* **tags** (Bool) A bool array, representing the tagging state to be painted onto the tokens. Corrolated with the number of zones.

Finally, there is the ragged array used to store variable-length token streams when the instructions all need to be a fixed length:

* **token_data**: (Int) A flattened, concatenated ragged array. The compilation process will have determined what part token start_pointer and token_end_pointer should point at.

One abstraction focused around data can control these roles.

### MOA CORE: Data storage, Memory, reconstruction, pipeline

The GPU VM has almost no memory associated with it. The only place memory is required is in the trigger abstraction to keep track of the last N tokens in case the model was producing a control patterns. Otherwise, the MOA_Core only needs to accept tokens, and produces:

- tokens: Replacements or passthroughs. Combined together externally
- tags: The tag bool for this sequence of tokens. Combined together externally
- status: The zone, token_num, trigger arrays that control the state machine. 

The external MOA_VM wrapper then wraps this and provides the gathering, extracting, transferring, and similar functionality. It does, however, display the .done feature.
Internally, the system has a triggers detection and configuration system, and then a sequence
of triggered injection kernel and listening mechanisms. The core is intended to be very extendable, and is where we keep the instruction program itself.

### MOA_VM

The main MOA_VM handles the extraction logic, and also keeps track of the tokenizer, the tokens as we go, and handles extraction and debug collection logic. 