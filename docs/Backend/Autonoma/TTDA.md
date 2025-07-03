# Token Trigger Detection Autonoma

The Token Trigger Detection Autonoma is a TTFA used extensively in order to detect the token triggers the internal logic needs to respond to. It contains an internal repository of token sequences which, if observed in the token stream, can be interpreted as representing a triggering event. It returns when one of these triggering events has been detected for each configured pattern match and emits bool arrays that are marked as true when a pattern has been matched. 

## Construction

The TTDA is constructed by asking it to maintain a buffer of a given size. This size should be equal to or greater than the length of all token patterns we may wish to match. It then contains certain pattern-matching logic that evaluates whether this sequence has been seen

## Usage

The TTDA is fairly simple to use. 

1) It should always be invoked with the output of the token stream, regardless of if it is claimed or now.
2) A tensor pattern to match, in terms of the sequence in the specified order, can be passed into the .match method, and returns a bool array per batch indicating if the pattern is matched on this iteration.

## Internals

A buffer is maintained of the last N tokens which have been seen. On match request, this buffer is stacked, the relevant section is sliced out, and it is compared to the match target. The result is then returned as a bool array.
