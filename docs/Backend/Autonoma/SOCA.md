# Stream Output Capturing Autonoma

The Stream Output Capturing Autonoma is designed to capture tokens being produced by the stream and automatically block, resolve, then release tool usage. As it is blocking, it is recommended to either not use it in evaluation, or simulate the usage of it.

## Internals

The mechanism contains

- Output Buffer: batch_size x length: Contains the tokens being captured to feed into a particular tool.
- Mode (int): batch_size. Target tool we are capturing for. If -1, no capture event is occurring. 
- Callbacks: Dict[int, Callable[[str], str]]
- Tokenizer: For tokenization and detokenization. 
- SIFA: To handle the input feedback.

It also has 

- Output_Indirections: Shape Program_Counter_Length+1. Maps the Program Counter values onto tools. If value is -1, no tool is being captured. +1 is due to the fact that the PFA resolves between zones as -1
- Index: batch_size (int). The index we are writing the buffer into.

## Interface

The output capture is fed with the result after using the PFA to inject tokens. It is in fact fed with

* Tokens (int): shape batch. The tokens just generated
* PC (int): The program counters for each interface. Batch
* Claims (bool): Shape batch. The existing claims if any.

It passes the tokens on unchanged, as does it the claims, but may sometimes block execution to run a tool.

## Feeding Pattern

Tokens are always accepted, and then returned unchanged.

## Transitions

All transitions are defined per batch, using vectorized logic.

* If no claims.
  * Mode is checked and updated at every single step
  * When mode changes
    * If mode is changing to -1, capture ends and current tool is run.
    * If mode is changing from -1, reset index.
  * While mode is not -1
    * Feed token into current index
    * Advance index by one.
  * Return token

## Running tools

When a tool run situation is detected, we break to python, loop over and slice apart the buffer, then detokenize the results and invoke the tool. Once we get the answer back, we tokenize it, use the SIFA and set the appropriate batch with the feedback.

Thus the vast majority of time everything is run in vectorize mode. 
