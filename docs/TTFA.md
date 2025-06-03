# Token Triggered Finite Autonoma

The Token Triggered Finite Autonoma is, in essence, a small Turing-complete computer implemented on the GPU that can operate in parallel but still do general computation by means of pointer dereferencing. 

Each TTFA is compiled for the individual batch that is being processed, and they support single workflow multiple stream generation across different batches.

## Data Structures

Internally, the data of the TTFA consists of the instructions tensors, the data tensors, and then the state tracking.

### Other

One general feature that is not correlated is:

* int: max_genned_per_zone. How many tokens can be in one zone before forcing advancement.
* int: padding token, a token to make when the program is done.

### State

There are only three state values of any concern

* Program_Counter: Per batch, what the program counter through the instructions are
* Token_offset: Per batch, where the token pointer is currently pointing.
* Genned_tokens: Per batch, reset on moving zones, how many tokens have been genned in this zone.

### Instructions

A set of same length instruction tensors that encode various features. They are always of length L and have no batch dimension

* Jump_Enable: Bool array. Whether Flow Control Signal is allowed
* Jump_Location: Int tensor. Where to jump to when jump is triggered.
* Starting_Offset: What index to start feeding tensors from in the tokens data
* Ending_Offset: What index to stop feeding tensors from in the tokens data. Same as start = do not feed anything
* Tags: A bool array of shape LxN, with N being the number of tags. Indicates the tag pattern to return when in this zone
* Step_Trigger: A L array of int tokens. When the token at the current PC is seen, we advance to the next zone.

Each instruction only activates the set of values relevant to their current Program Counter.

### Token Data

All tokenized information was flattened into a single large array and concatenated. Offsets to load from were stored in the instruction. This is the same technology used to store ragged tensors.

### Program

A program then consists of loading a TTFA with all of this information. All other complexity is exported to the compiler chain, and fortunately ZCP is isomorphic with flatted jump flow control...

## Usage.

When used, the TTFA accepts only

* Tokens: A "B" shaped batch of tokens from each batch predicted by the model

It then returns

* Tokens: A "B" shaped batch of tokens. Some tokens may, or may not, be replaced depending on the state of the FSM
* Tags: A "BxN" array of tags, indicating the tags active in the zone. It is always the same each time, in the zone, based on looking up the stored tag based on the program counters.

## Transitions

### Feeding pattern.

Lets assume no zone transition is called for, and we just entered
a zone. This is what happened

* On transition in, the PC was set to the right value, the feed pointer was set to the starting offset, and the genned_tokens set to zer
* Every subsequent invokation reads feed from the pointer, advances it by one, and increases the gen count by one.
* If we reach the ending offset, we stop replacing tokens and let the model's natural answers come through. We do, however, keep advancing gen tokens.

### Transitions

Transitions to a new zone can be triggered by three things.

* Zone transition. The Step_Trigger token we were listening for was noticed. The feeding pattern is loaded as described above, after advancing the program counter by one.
* Flow Control. The Jump token is noticed, causing us to dereference the jump destination and change state to load that location
* Timeout. If gen count > max_genned_per_zone, we immediately emit the zone transition trigger by force. we then respond to it, and move onto the next zone.

Finally, when paging off of the end of the program counter the program is done and will only produce padding tokens. Once all
program counters are done, the .done() call will resolve to true.