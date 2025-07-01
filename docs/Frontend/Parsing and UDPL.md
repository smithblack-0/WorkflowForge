# UDPL

The philosophy behind UDPL is to be human readable, interpretable, powerful, and concise. Mostly linear tasks can be handled almost entirely in UDPL. It is the original language of the project. Please note you should have read [UDPL](../UserGuide/UDPL.md) as this is implemented to support that specification.

## Validation

Validation is extremely heavy, and extremely helpful. This is in line with the fail early philosophy brought up earlier, and also in part because I do not trust myself. This occurs in the config, and then when cross referencing the config to the sequences. Generally if it is not allowed in the userguide, it is forbidden.

Errors are raised based on the parsing stage they are at, with very detailed messages.

## Parsing

The parsing module is designed to parse UDPL into something usable. Namely,

* Into a config object.
* Into a dictionary of sequence ZCP nodes

These can then be consumed downstream. The primary objective, and indeed the primary difficulty, is ending up with the ZCP linked lists.

## Parsing layers

There are several primary parsing layers. These are intended to correspond
roughly with various functions, and exist to meet our modular breakdown targets.
They also accept helper functions as parameters, making unit testing trivial.

* Main: Parse from file or folder
* Config: The very first thing, and then used to establish strong typing
* Sequence: Checking the sequence is valid, then checking blocks
* Block: Checking the block is valid, breaking up into zones
* Zone: Checking zones are valid, ensuring we have ZCP

Once all are done, we will have our results.

## ZCP and Zones

The ZCP nodes correspond, almost one-to-one, with the zones in the sequence. However, there is an exception. Because it makes coding the backend so much easier, one extra zone without any text is added at the start of blocks corresponding to just the initial input - such as "[Prompt]"