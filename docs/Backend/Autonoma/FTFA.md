# Forced Token Feeding Autonoma

The Forced Token Feeding Autonoma (FTFA) is a simple helper class used by other autonoma to inject predetermined token sequences into the token stream when triggered. It is mainly used for injecting forced advancement sequences into the stream, and injected into other classes. It operates as a per-batch state machine, replacing generated tokens with a pre-loaded sequence when activated by a boolean condition until all tokens are used, then going back to inactive mode.

## Construction

The FTFA is constructed with a pre-loaded injection sequence (e.g., `[512, 212, 513]` representing a tokenized trigger pattern like `"[Jump]"`). This sequence is stored and used for all injection events.

## Usage

The FTFA is owned and invoked by other autonoma that need token injection capabilities:

1. **Input**: Receives tokens, trigger mask (bool array), and claims matrix
2. **Triggering**: When trigger mask is `True` for a batch, begins injection
3. **Injection**: Replaces tokens with injection sequence until complete
4. **Completion**: Returns to inactive state when sequence fully injected

## Internals

**State Variables:**
- `injection_sequence`: Pre-loaded token array to inject
- `active`: Per-batch bool array indicating currently injecting batches
- `index`: Per-batch int array tracking current position in injection sequence
