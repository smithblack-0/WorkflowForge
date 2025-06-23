## Contributing

Workflow Forge is designed to become a community standard for AI workflow automation. We welcome contributions from researchers, engineers, and AI practitioners

### Current Priority Areas

Under Control by Chris until first working release, though suggestions welcome:

-  **SFCS Implementation** - Flow control compilation logic
-  **TTFA Backend** - GPU tensor operations for autonomous execution

Excellent places for improvement.
- **Community Direction** - What do we want?
-  **Testing & Validation cases** - Particularly in tool usage, about which I know only a bit.
-  **Documentation** - Examples, tutorials, and API docs
- **Research Applications** - Novel use cases and benchmark applications.

### Getting Started
1. Read the [Architecture Overview](docs/Overview.md) to understand the system. 
2. Read the [UDPL Spec](docs/Frontend/UDPL.md) to understand how UDPL and the declaritive syntax works.
3. Read the [SFCS Spec](docs/Frontend/SFCS.md) to understand how flow control works and the common mistakes.
4. Check [open issues](../../issues) for specific tasks
5. Join discussions in [GitHub Discussions](../../discussions) for design questions

### Philosophy

These are the rules I tend to code by when doing large, complex projects. It is neither waterfall nor agile, but has a frontloaded architecture phase with clear termination.   

1) Poc/Fucking around. Figure out the problem scope, and make the unknown unknowns into known unknowns. around 5% of the effort
2) Artitect Interfaces. Figure out what the major modules are, and how information will need to flow between them. This is not final as in waterfall. Figure out what each major module needs to do. 20% of effort. You are done when failures are containerized; you could swap out one failing module for another design. That is my anti-perfect measure. This is Document Driven DEvelopment
3) PoC2/Fucking around 2. Take each module, and make sure I am not asking something impossible from it. 10%. This prevents "OCRAP it cannot do that" issues. It also tells us where we did not abstract enough, or overabstracted.
4) Architect System. Come up with a final plan. Get everyone on the same page if in a team. Make sure responsibilities and objectives are clear. Modify architecture to account for the hidden dependencies, and rebuild units. This, again, ends when failures are containerized. It is about fixing our architecture plan and addressing critical technical details, not waterfall. 15%
5) Primary coding. 20%
6) Pivoting interfaces and propogating due to missed dependencies. 30%


