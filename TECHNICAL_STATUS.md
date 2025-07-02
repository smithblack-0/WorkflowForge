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
| Resources                      |✅|✅|✅|✅|✅|
| SFCS-System                    |✅|✅|✅|✅|✅|
| ZCP-Architecture               |✅|✅|✅|✅|✅|
| ZCP-Nodes                      |✅|✅|✅|✅|✅|
| ZCP-GraphBuilder               |✅|✅|✅|✅|✅|
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



## Milestones/Brag list

- All ZCP nodes complete with integration testing 6-21-2025.
- Entire frontend done, besides for client-server integration 7-1-2025
- 