# Timeout Triggering Autonoma

The Timeout Triggering Autonoma automatically replaces the incoming token stream with the advancement token when the timeout state is reached.

## Internals

The mechanism contains

### Instructions

Per PC:

- Timeout_Values: When this many tokens have passed, it is time to timeout
- Timeout_Tokens: What token to replace with to indicate we need to advance

- Counter: Shape (batch), type int. The current count.

## Token Feeding

The input is always

* Tokens: The token stream at this moment. Shape (batch)
* PC: The program counter at this moment Shape (batch)
* Claims: Existing claims, if any. This will usually pass through the claim while advancing the counter.

The output is then 

* Tokens: Either the original token, or the timeout token. If counter > timeout_values[PC], we replace and claim. Since the TTA is always first, there is no competition.
* Claims: Usually just what passed in, unless staking a claim.

## Transitions

Transitions occur any time we move zone, in the following order

* On zone change:
  * Reset counter. 
* Increment counter.

