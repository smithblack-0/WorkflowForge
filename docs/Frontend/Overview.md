# What is this

This is technical documentation on how the workflow forge frontend portion is structured, in terms of underlying code and ideas. This is not how to use it, which is contained over in the user guide

## Terminology

Going forward it is important to keep the following terms in mind:

- **Zone**: A region of tokens delimited by two special text patterns defined in the config (e.g., a [Prompt] to [Answer] span). Zones are the unit of tagging and extraction. Every block is made up of one or more zones.

- **Zone Edge Pattern**: Special text patterns such as [Prompt] or [Answer] used to indicate the edge of a zone. These are not guaranteed to be single tokens.

- **Block**: A full prompt sequence consisting of one or more adjacent zones (e.g., [Prompt]...[Answer]...[EOS]). All zones in a block are resolved before the block completes. The model may take over any time all original prompt text is exhausted, at which point it completes the remaining zones.

- **Prompt**: A partially completed input string in a block that includes one or more defined zones. Prompts may include placeholder fields to be filled by resources during preprocessing.

- **Tags**: Labels applied to individual zones within a block to enable selective extraction of generated tokens for training, evaluation, or filtering.

- **Flow Control Pattern**: A special text pattern such as '[Jump]' that can be defined in order to make manipulations happen in SFCS. **DANGER** unless escaped, flow control in teacher-forcing prompts are executed immediately, breaking the prompt.

- **Escape Pattern**: A special text pattern such as '[Escape]' which can be defined to make the MOA skip the next transition instruction. Very useful for telling the model what text to emit next.

- **Sequence**: A named chain of blocks that can be invoked by SFCS commands.

## Philosophy

The frontend, basically, is intended to be heavily validation-focused and is intended to hide graph logic difficulty from the user. The philosophy of the frontend includes

* Strongly typed: The user guide contains the linting spec we match. Match it exactly
* Fail early: We want to throw an error as early as possible
* Very descriptive errors: Errors should say exactly when and where something went wrong, and even possibly suggest alternatives.

## Purpose

The frontends purpose is ultimately broken into the following responsibilities

* **Client** contain the client code, and be able to establish a connection to an associated server.
* **Accept input** Accept input in the form of udpl files and sfcs directives. 
* **Lower to SZCP** provide factories that produce lowered SZCP. this can be dispatched by the client.

## Concepts

* **Config** A strongly typed entity which will be compared to the backend config, as well as specifies valid tags and other features. It is defined in the udpl, and serves as the strong typing constraint for the system.
* **Sequences** See the user guide, but sequences of text to feed basically.

### Modules

#### Parsing

The parsing module is heavily validation-focused, and is built around parsing the config, then the various sequences of the provided udpl, and ensuring the compiled results are sane. 

Code in this segment is constructed in terms of a sequence of transforms that go off in order. In order to make unit testing trivial, function dependencies are provided as parameters so mocking becomes trivial. 

Testing includes both unit testing for each function along with integration testing for the entire system.

The result of finishing parsing is a Config object, and a Sequences dictionary of sequence names to ZCP linked list heads.

See [Parsing and UDPL](Parsing%20and%20UDPL.md) for more details

#### Flow control

Flow control interacts with the ZCP lowering system defined in the zcp module. This walks the process from the ZCP linked lists through forming valid RZCP DCG-IO graphs, into finally producing a factory that produces SZCP when invoked. 

Additional validation, based around resource resolution, must also occur at this stage, though dynamic factories make error messages a little bit more tricky. 

The main outer classes are scope, which is used during flow control, and program, which is the main class the user interacts with. These manage graph building using the underlying zcp system. Conceptually, each call enqueues additional edges to be placed on the next graph known that is resolved.

See [SFCS and UDPL](SFCS%20and%20flow%20control.md) for more details.

#### Client

The client system is modular, intended to allow different authentication mechanisms to be hooked up to it. It can execute a szcp request when hooked up to a server. A client can either be hooked up to a server directly, or hooked up remotely by address. When hooking up a client, a valid config is needed, as it will be compared to the server. Errors may be raised when the server config cannot be made to match the client config. Clients are functional, and running a client request returns a references to the previous client state. Clients can be continued, with an option of a previous run being selected to continue generation from.


