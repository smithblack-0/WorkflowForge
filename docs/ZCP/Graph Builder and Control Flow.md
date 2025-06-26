# Control Flow

ZCP control flow is handled as a constructive process involving attaching graph nodes in the proper order working with RZCP nodes. These nodes can be lowered from ZCP, then combined together, in order to perform control actions

## Mathematical simplifications

Control flow allows jumps to previously defined nodes, or zone advancements to the next one. However, as a mathematical simplification, jumps to nodes which have not yet been defined are not allowed.

## Graph Builder Node

The graph builder node class is a primary class designed to support this mechanism. It handles forward references. The graph builder can be thought to be maintaining a list of colored edges, and resolving to attach those edges to the next concrete node it sees. It thus allows us to define forward references we desire to attach. In an abstract sense, it represents the next edges, and then later on the node those edges were attached to.

### Methods

* .extend: Extends the current RZCP chain, as in a .run command, and returns a new graphbuilder containing the new forward references
* .fork: Used to create graph builder nodes that are bound to the nominal or control forks of flow control. Both types of nodes are returned. The difference is based on which color of edge the forward reference is now placed on.
* .merge: The only class method. Merges multiple GraphBuilderNodes together so they will all attach to the next reference.
* .attach: Used to attach a graph builder to the captured node. This can be used to complete jumps, for example.

## Control Flow Actions

* run: Simply attaches a forward reference 
* jumps: Jump is supported by means of attach