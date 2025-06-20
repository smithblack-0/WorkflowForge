# Changelog

All notable changes to Workflow Forge will be documented in this file.

# Commitments

*Note: This is an early development release. SFCS flow control and TTFA execution engine are still in development. Backend engine is not yet functional. Language is still vulnerable to change, and backwards compatibility not yet guaranteed*



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
