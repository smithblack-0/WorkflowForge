# Stream Input Feeding Autonoma

The Stream Input Feeding Autonoma (SIFA) has one job. Feed any unused input tokens into the stream until all input tokens are exhausted.

## Internals

There is an input feed buffer.

* Input_Buffer: Contains a BxL buffer that can be used to store input values per batch
* Input Index: Contains a B index into the input buffer
* Input_End: The location to stop feeding input buffer values from.
* Input_Feed: Bool array of length Total_Program_Counters+1 indicating whether or not we feed from the input feed if needed. Plus one is needed as between zones is -1.
* Trigger_Advance: Bool_Array, indicating for each token what token to emit to force advancing when input_end is reached

## Interface

The SIFA accepts

* Tokens (int): Shape batch (B). Current tokens in the stream.
* Program Counter batch (B). Program counter at the moment.
* Claims array (Bool) shape (B). Whether something has already claimed it.

It then returns

* tokens (int): shape B. Possibly just passed through, possibly replaced.
* Claims (bool) shape B. 

It is, in other words, a transform.


## Transitions

Transitioning always goes off first. 

* If new zone, reset the input index to zero
* Make a copy of the claims tensor called the original claims
* If in a zone that feeds, set the claims.

## Feeding Behavior

The rest of the feeding logic then goes off. Claims are handled here

* When in a zone that feeds and unclaimed, we replace the token then advance the index.
* When in a zone that feeds and at end, replace token with trigger advance.

finally, return tokens, claims.

## Method

The .set_feedback method accepts a int batch number and a tensor 1d array. It sets the feedback for that tensor. 