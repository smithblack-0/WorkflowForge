# Intro

The SFCS DSL is the flow control language viewed by this system. The frontend flow control module is what implements it

## Design

There is one primary purpose the SFCS system performs. That purpose is creating a flow control graph, and particularly the DCG-IO graph our system needs. This will be the RZCP graph when finished.

To build such a graph, several things are needed:

1): A way to map user directives onto the graph
2): A way to resolve forward references

Number one was nicely satisied by making the graph a control flow graph with emission control mechanisms and multiple launching node sites. Numper two, largerly, was resolved by just forbidding forward references in the first place. The special GraphBuilder class handles forward references, but the only kind of forward reference that is allowed is to attach onto the next valid node. This is still plenty, with reverse references, to support loops, if statements, and even else statements.

With almost all complexity dealt with by selection of the underlying mathematical structures, all the SFCS system must then do is attach graph pieces in the right order, and ensure the mapping is one-to-one. Context managers for when forks occur then round out the needed behavior.

