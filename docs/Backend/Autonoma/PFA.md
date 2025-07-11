# Prompt Feeding Autonoma

The Prompt Feeding Autonoma is responsible for the behavior that actually stores and feeds
from the prompts involved in individual zones, then switch over to relaxing once done

## Internals

There is a data storage mechanism:

* token_data: A flattened collection of all tokenized prompts stored as a 1d array

There are some instructions with program counter associations

* Start_Offset: (int) Shape PC. Where to start getting prompts out of the token_data based on the Program Counter
* End_Offset: (int) Shape PC. When to end getting prompts out of the token_data based on the prompts counter.

Finally, there are some states too of course

* pointer: (int) shape Batch. The pointer per batch indicating what part of the 1d array to reference out of. 


## Transitions

Transitions occur first. They occur in this order

* If unclaimed
  * If PC changed:
    * Set pointer to start offset
  * copy feed pointer directly from state pointer (use tensor.clone())
  * Advance state pointer

## Feeding

While we have not yet reached the end_offset, we:

* If unclaimed
  * Claim only if end_offset > feed_pointer
    * Feed token_data[feed_pointer] 
  * Pass through unchanged otherwise.
