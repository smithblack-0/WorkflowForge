# Introduction

This is tracking of the underlying technical details of the Simple Flow Control System and how it works under the hood in python.


## Overview

There are three primary stages to the system. These are

* Intake: Perform basic validation on the parsed ZCP sequences, and ensure resources are either resolved or custom. 
* Constructing: Uses scopes and methods to combine the ZCP sequences together into a central chain
* Compiling: Construction of parsed ZCP. This calls into the backend compilation callbacks. 

## Intake

* Validate noncustom resources are all available
* Validate all special tokens are defined in the tokenizer
* Construct the Tag Converter.

## Construction

Construction is built around building, extending, and resolving scopes. A **Scope** is a region in which we are building one or more ZCP chains with possible flow control with subscopes.  



### Scope

Building the ZCP graph with its flow control is all about resolving resources and assigning scopes. 

