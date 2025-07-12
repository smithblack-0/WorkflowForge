# The Initialization Problem statement

VM initialization concerns the problem of taking a workflow, getting LZCP out of it, and using that to create a GPU VM core and the subsequent GPU VM wrapper system in order to setup a GPU VM core and extraction system appropriate for the problem. This is, in essence, another stage of compilation that compiles for the particular bytecode of the backend and particular system which is present.

## GPU VM 

The GPU VM system will need to be initialized with a GPU VM core, and with the extract directives. It can then perform the needed extract information and operations, and use the core to collect and pass on information. The GPU VM can be dependency injected with the GPU VM core in order to simplify setup logic.

## Instructions and Instruction Memory

The LZCP graph is flattened into a tensor array - the DCG-IO format helps ensure usually we can just advance to the next zone. The behavior of the default advancement is configurable by an extra array 


## GPU VM Core and Setup.

The GPU VM core is dependency injected with the Token Manipulation, Transitions, and Post Processing stages. It creates a Setup Stage to manage hidden state for each of the stages provided, based on the stage requirements they present when asked. Conceptually, Setup and GPU VM Core are part of the same responsibility package, a decision made because all stages need setup producing a coupling. The packaging 'stage' also occurs here, though does not need to be a distinct class in and of itself - just creating a state object is more than enough. The State object also lives in this collection.

## Token Manipulation Stage

The token manipulation stage is loaded with several modules. These modules include:


* **Aux_Setup**: Sets up a feature called "has_been_replaced" that is of size tokens and false
* **Prompt Feeder**: Feeds a prompt into the token stream until it runs out of tokens. This means replacing tokens in a prompt feeding replacement state based on autonoma state until exhausted, then passing through unchanged. Releases an update to "tokens". Also releases an update to "has_been_replaced" based on whether or not the prompt was overwritten.
* **Tool Feeder**: Feeds the inputs from a tool buffer into the stream. This listens to has_been_replaced and will only start feeding AFTER all tokens in the prompt feeder have been exhausted.  
* **Timeout**: The timeout module does NOT respect has_been_replaced. It offsets when it starts feeding based on the length in tokens of the zone advance pattern to ensure sequences never exceed the token length. Sets "has_been_replaced" to true.
* **WasTeacherForced**: Moves a copy of "has_been_replaced" onto the token stream, under the name "teacher_forced".

Some of these have instruc
