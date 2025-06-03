# Simple Agentic Control System

## What is this?

This is the official specification for v.01 of the 
SACS flow control language.

## Overview

The Simple Agentic Control System (SACS) is a python flow control language designed to provide a pythonic environment to commit a reduced set of flow control operations that can then be linked with the zone specification, built into a flow control graph, and parsed into ZCP. 

In actually, the initialization of the system also takes responsibility for invoking and compiling into backend during the compilation stage, but conceptually the language itself is simply building a graph of zones with triggers that can be parsed. However, that is not the focus of this document.

The output of the system is the ZCP specification

## Starting point.

The SACS system starts by initializing a new program. 
This program must consist of:

1) A sequences dictionary, developed by parsing the UDPL file or folder. 
2) A resources dictionary, containing named dictionary resources.
3) A config object, containing important parsed config information from the UDPL pass
4) A tokenizer resource, to allow conversion.

Once the resources requirements vs provded are crosschecked, it is possible to start building a program. This is what is returned when creating a new program

```python
from CE import sups

program = sups.
```



## Program

The program is the core object of the SACS system. It is a sequence of triggered UDPL sequences to apply in the given order. A program can be linear order. The simplest form of program is created using the ".run"


## ZCP

