# Changelog

All notable changes to Workflow Forge will be documented in this file.

# Commitments

*Note: This is an early development release. SFCS flow control and TTFA execution engine are still in development. Backend engine is not yet functional. Language is still vulnerable to change, and backwards compatibility not yet guaranteed*

## [0.1.5]

6-23-2025

- Added new escape strategy. "[Escape]....[EndEscape]"
- Pivoted a bunch of documentation to comply
- Ensured documention was up to date before beginning final integration rebuild up to LZCP.

## [0.1.4]

6-21-2025

- Documentation modifications, and added a much clearer technical status tracker
- Adopting serialization mechanism into ZCP nodes... finished. Unit testing... finished.


## [0.1.3] 

2025-06-20

- I can see a clear way to serialize the workflow, and then deserialize it later. Making it work means a few tweaks in the tool-using system, but is probably worth the trouble.
- I think I will go ahead and implement the changes. There will be an extra SZCP stage, and a workflow factory mechanism that does the actual resource sampling to produce a workflow. Resources stay on the client. The workflow's themselves are what can be serialized and sent to the other backend system.
- Went through with the pivot. We are going to have a server/client system. I believe this is as wide as the scope is ever going to get for the initial release.


## [0.1.2] 2025-06-19

(2025-06-19)
- We are pivoting to using entire strings of tokens, rather than requiring the user to setup special tokens in their tokenizer and embeddings. This adds complexity, but it is not unmanagable.
- This involved adding the Token Trigger Detection Autonoma, to detect when a special command has been issued, and the Forced Token Feeding Autonoma, which can be triggered by a bool state and automatically claim the stream and feed tokens to force, for instance, '[NextZone]' into the stream on timeout.
- Fixed all the various typing references to refer to an array of tokens rather than an int. userspace document still talks about a 'token' usually as that is what the user may think of as specifying.
- Renamed folder ZCP to zcp, to comply with the style standards. Come to think of it, I need to write down the style standards don't I?
- Added much stronger error coupling between zcp nodes and factory construction in program, ensuring that 

## [0.1.1] - 2025-06-18

- Spent about 3 days mulling over compiler design to figure out the forward reference situation, but finally found a graph theory approach to make it tractable.
- Added a GraphBuilderNode helper mechanism in the ZCP module
- ZCP has been significantly extended to include ZCP (original), ResolvedZCP (IR, and SFCS emission) then still has LZCP (right before machine code compilation). 
- The tokenizer interface has been formally declared and made a file. 
- The core SFCS construction basically done, but still needs to be tested and debugged. The graph approach made this very simple, with most commands mapping 1-1 onto graph actions. Combined with the graph helper under the ZCP module, I could get away with only a few classes:
  - Context classes; run the context managers, only ConditionalContext and WhileContext right now
  - Program class: top level class users interact with, including the various commands
  - Scope class: What is interacted with and returned by context managers.
  - Factory class: Contains all the various classes, and is dependency injected for easy testing.
  - Tool and Toolbox class: For tool usage. 
  - TagConverter: For converting tag strings into their boolean mask format. 
- NOT done are the intake validation for the SFCS, the connection to the backend, and the testing.

## [0.1.0] - 2025-06-11

### Legal
- Open source release! Documentation and licensing added.

### Added
- Complete UDPL (Universal Declarative Prompting Language) parsing pipeline and unit tests. Wow, that is a lot of error conditions, but it is thorough enough Linus would be impressed.
- Comprehensive test suite for UDPL parsing components
- GitHub Actions CI pipeline for automated testing
- Project documentation and architecture overview
- Branch protection and pull request workflow

### Infrastructure
- Project structure with proper Python packaging
- MIT license and contributor guidelines

## [Unreleased]

- Overall architecting and design.
- Initial throwaway parsing code.
- Full Tensor Triggered Finite Autonoma breakdown. Though I notice I missed a case now.
