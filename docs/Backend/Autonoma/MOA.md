# Main Orchestration Autonoma

The Main Orchestration Autonoma (MOA) is the main Token Triggered Finite Autonoma. This is, in essence, a small Turing-complete computer implemented on the GPU that can operate in parallel but still do general computation by means of pointer dereferencing. 

## Data Structures

Internally, the data of the PTA consists of the instructions tensors, the data tensors, and then the state tracking.

### Other

One general feature that is not correlated is:
* max_genned_per_zone (int). How many tokens can be in one zone before forcing advancement.
* padding token (int), a token to make when the program is done.

### State

There are only a few state values of any concern

* Program_Counter: Per batch, what the program counter through the instructions are
* Escape: Per batch flag. If an escape token was seen, it negates the next instruction then resets.

### Feeders

A sequence of special autonoma are designed to trigger based on program counter value through subsequent transitions. These will occur following the chain of responsibility pattern; when one stage is claiming responsibility, later stages will not go off. If no stage claims responsibility, token is not changed. Expected order as of now is:

* Token Timeout Autonoma
* Prompt Feeding Autonoma
* Stream Input Feeding Autonoma
* Stream Output Capture Autonoma.

### Instructions

A set of same length instruction tensors that encode various features. They are always of length L and have no batch dimension

* Jump_Enable: Bool array. Whether Flow Control Signal is allowed
* Jump_Location: Int tensor. Where to jump to when jump is triggered.
* Tags: A bool array of shape LxN, with N being the number of tags. Indicates the tag pattern to return when in this zone
* Step_Trigger: A L array of int tokens. When the token at the current PC is seen, we advance to the next zone. 

Each instruction only activates the set of values relevant to their current Program Counter.

### Token Data

All tokenized information was flattened into a single large array and concatenated. Offsets to load from were stored in the instruction. This is the same technology used to store ragged tensors.

### Program

A program then consists of loading a MOA with all of this information. All other complexity is exported to the compiler chain, and fortunately ZCP is isomorphic with flatted jump flow control.

## Usage.

When used, the MOA accepts only

* Tokens: A "B" shaped batch of tokens from each batch predicted by the model

It then returns

* Tokens: A "B" shaped batch of tokens. Some tokens may, or may not, be replaced depending on the state of the FSM
* Tags: A "BxN" array of tags, indicating the tags active in the zone. It is always the same each time, in the zone, based on looking up the stored tag based on the program counters.
* Resolving Autonoma: Which autonoma if any resolved the call, by order. -1 means the model's tokens were passed through freely.

## Feeding pattern.

Feeding occurs by the following mechanism:

1) A B bool array filled with false is created, to indicate nothing has claimed responsiblity. 
2) The token, PC, and claims array are passed successively into feeder transforms, and token, claims arrays passed back out. 
3) The final token values are returned.

Claims array being true will mean that token was already claimed by something earlier up the chain, preventing the later autonoma from using up their influence triggering it.

## Transitions

On transitioning to a new zone, the feeding pattern transition mentioned above is setup. These transitions themselves need discussion. Transitions to a new zone can be triggered by one of the following.

* If a transition token is noticed and esape is set, skip that instruction and set escape to false.
* Zone transition. The Step_Trigger token we were listening for was noticed. 
* Flow Control. The Jump token is noticed, causing us to dereference the jump destination and change state to load that location.

Finally, when paging off of the end of the program counter the program is done and will only produce padding tokens. Once all
program counters are done, the .done() call will resolve to true.

## Debugging

When set in debugging mode, the above will also emit a resolution map indicating, by index, which of the kernel extensions is resolving each individual token. Count starts at zero, and falling all the way through ends up at the length of the number of kernel extensions. Otherwise, this is not emitted. 