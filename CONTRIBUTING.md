# Contributing

Workflow Forge is designed to become a community standard for AI workflow automation. We welcome contributions from researchers, engineers, and AI practitioners. 

## Development Philosophy

### Document driven development.
Workflow forge core, which at the moment is workflow forge itself, is being developed using a documentation-driven development strategy. This means we declare in the documentation what should happen, and then update the codebase to match. 

Documentation must exist at least to the extent of "containerizing" issues such that modules can be swapped out if a core technology is found to be impossible. API specs need to be quite strong. Technical sections can go a bit deeper but may be considered more optional; However, interaction APIs should always be up to date.

Pull requests that add api details without accompanying documentation changes will be rejected, particularly if they modify the core. This is also true if something has been majorly rescoped without documentation changes.

### Code Quality

Core section code quality must be extremely high to be accepted; however, we would be happy to help you get to that level. Type hints, docstrings, and maintainable software practices are required. Note that if you want to add something in a separate folder that is not core, the standards may become considerably lower.

#### Design patterns

The design patterns criteria is quite strict. **Maintainability** is the number one goal, and we frontload coding effort in pursuit of it. This is because the underlying logic of the core is extremely complex to verify unless broken up well.

* The single responsibility principle is heavily weighted. Modularization is required. 
* Dependency injection is required. This is for both testing and maintenance purposes. Utility functions to construct classes can then be defined with the "make" pattern.
* Except in very linear processes, functions or methods over 100 lines long should be broken apart. 
* Functions which invoke other functions should consider passing the function in as a parameter for unit testing and modularization.
* Large centralized classes that do everything are discouraged. Instead, break responsibilities into "processing" classes  which do something, and then an organization class which organizes and orchastrates them.
* "processing" classes should generally never exceed 300 lines. If they do, you usually need to rethink your abstractions. This does not apply to organization classes that just contain and provide passthroughs into a bunch of doer classes.

### Commentary and typing

The commentary standard is thorough, and should focus on
why and what, over how. The person reading the code can
understand how. Nonetheless, avoid explicitly linking comments
to other classes where possible to avoid stale comments.

* Type hints are mandatory in all functions, methods, and classes.
* Docstrings are mandatory in all classes. 
* Method strings are mandatory in all public methods.

### Testing

Test quality must be extremely high in the core

* The unittest library is used in the core.
* Unit test coverage must be 100%, and focus on each main responsibility.
* Integration test coverage must cover major cases. 
* Each architecture zone must have an integration test.
* The architecture itself must have an end-to-end test.
* LLM assistance writing unit tests is acceptable.

Additionally:

- Unit tests should be performed using mock. 
- One test suite should exist per tested feature.
- Error raising should also be tested.

### Getting Started
1. Read the [Architecture Overview](docs/Overview.md) to understand the system. 
2. Read the [UDPL Spec](docs/UserGuide/UDPL.md) to understand how UDPL and the declaritive syntax works.
3. Read the [SFCS Spec](docs/UserGuide/SFCS.md) to understand how flow control works and the common mistakes.
4. Check [open issues](../../issues) for specific tasks
5. Join discussions in [GitHub Discussions](../../discussions) for design questions



