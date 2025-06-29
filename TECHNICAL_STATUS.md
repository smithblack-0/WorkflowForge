# What is this

This is the technical status document being used to track current operational status, stages, and plans for the current system.

**Last Updated**

6-21-2025

# Current Status

## Column Meanings
The status of various components is given in terms of architected, programmed, tested, integrated, and integration tested. 

* **Architecture**: There is a clear idea about how to accomplish the task, and proof-of-concepts have been run when relevant
* **Programming**: It is programmed as a nice DRY module.
* **Tested**: Unit tests are in place
* **Integrated**: Cross checks with the other modules have been performed, and the necessary changes in both of them made to merge the components.
* **Integration Testing**: Tests that verify the components now work together correctly have also been done.

The status can be indicated as:

* ✅: Done
* 🚧: In progress/needs work
* ❌: Not started/needs rebuild

When something has all green checkmarks, it is completely done to initial release quality. It should be undestood architecting means the modules themselves are pretty much designed, but minor dependencies may yet be missed.

## Status

| Component                      |Architecture|Programming|Tested|Integrated|Integration Testing|
|--------------------------------|--|--|-|-|-|
| State machine PoC              |✅|✅|✅|✅|✅|
| UDPL                           |✅|✅|✅|✅|✅|
| Resources                      |✅|🚧|🚧|❌|❌|
| SFCS-System                    |✅|✅|🚧|❌|❌|
| ZCP-Architecture               |✅|✅|✅|✅|✅|
| ZCP-Nodes                      |✅|✅|✅|✅|✅|
| ZCP-GraphBuilder               |✅|✅|✅|❌|❌|
| ZCP-Serialization              |✅|✅|✅|✅|✅|
| ZCP-visualization              |✅|✅|✅|✅|✅|
| Backend - Compiling/Flattening |✅|❌|❌|❌|❌|
| Backend - MOA                  |✅|❌|❌|❌|❌|
| Backend - Support              |✅|❌|❌|❌|❌|
| Backend - Kernel modules       |✅|❌|❌|❌|❌|
| Backend - Tools                |✅|❌|❌|❌|❌|
| Server-Client tools            |🚧|❌|❌|❌|❌|
| Wrapper Utils                  |🚧|❌|❌|❌|❌|

# Current issues and priorities

* ZCP-Nodes needs another pass now that we have escape open AND close tokens.
* Tools needs a rebuild, and once done the documentation will need to be updated.
* Resources needs actual documentation, and to be fleshed out more fully in terms of classes.
* Graph Builder may be out of sync with frontend
* The backend needs another pass. I need to add a Token Trigger Autonoma that just detects and emits trigger states, and update the kernel modules to read if triggers are occuring from this directly. 
* Also, the backend does not have documentation for the tags painting logic right now, or the TTDA integrations done/well thought through.
* The backend could probably use a section in and of itself
* The TTFA section needs to have a philosophy section discussing what to do as pieces interact and what the protocol is.
* Various pieces of documentation are out
* The decision has been made to use a [Escape] ... [EndEscape] system instead. The ZCP, documentation, and frontend system needs to evolve to reflect it.
## Next steps

Mainly getting serialization and server systems in place. A few minor changes appear to be needed.

STEP: Serialization Tokenization Support:
* Rebuild config parsing, other systems, slightly in order to ensure users can set tokenizer overrides in their configs.
* Ensure this config ends up in the serialized object that can be passed around, so the server could parse and load it.

STEP: Tool Serialization Support
* Rebuilt tooling frontend system to request tool usage by name, not by passing callbacks directly.
* Rebuilt system to include this information in config.

STEP: Fix flow control
* Rebuild flow control/parsing/config to expect to see tools declared by name, rather than by function.
* Rebuild system to properly interact with the new SZCP node system.
* 

## Milestones/Brag list

- All ZCP nodes complete with integration testing 6-21-2025.

## Proposals/actions

# UDPL Parser Update Specification

## Change Request: Update UDPL Parser Implementation to Match UDPL v1 Specification

### What We're Changing
The UDPL parser implementation is out of sync with the documented UDPL v1 specification. The parser was written for an earlier version of the spec and needs to be updated to handle the current escape token design and validation rules.

### Where We're Changing
1. **Config parsing module** (`config_parsing.py`) - escape token validation and missing tools field
2. **Block parsing module** (`block_parsing.py`) - escape region handling in text parsing  
3. **Zone parsing module** (`zone_parsing.py`) - flow control validation and resource type validation

### Why We're Changing
**Root Cause**: The parser implementation predates the current UDPL specification. Specifically:

- **Escape tokens were redesigned** from single tokens to start/end pairs, but the parser still expects single strings
- **Tools field was added** to the spec but never implemented in the parser
- **Resource type validation** was tightened in the spec but the parser uses outdated defaults and no validation

**Impact**: The parser cannot correctly parse valid UDPL v1 files and will accept invalid configurations that should be rejected per the specification.

**Documentation-Driven Development Goal**: Bring the parser implementation into full compliance with the documented UDPL v1 specification so that the system works as documented.