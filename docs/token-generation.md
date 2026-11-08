# Stop generation on an end token or a token budget

`TokenGenerator` connects a session, a selection policy and explicit stop rules.
You need .NET 10 and the built repository. Start with the
[session walkthrough](generation-session.md) and [sampling](sampling.md) if
prefill, logits or seeded selection are unfamiliar.

## 1. Try both stop conditions without a model download

```sh
dotnet build Marmotte.slnx
dotnet fsi examples/Generation.fsx
dotnet fsi examples/Generation.fsx -- --eos
```

The first command generates three IDs with greedy on an invented model:

```text
Prompt: 3 tokens (batch size 2)
Generated token 1: 2 (cached: 3)
Generated token 2: 2 (cached: 4)
Generated token 3: 2 (cached: 5)
Stopped: TokenLimit; returned: 3; cached: 5
```

With `--eos`, the script explicitly declares ID 2 an end token. This is only a
synthetic fixture convention, not a default for real models:

```text
Prompt: 3 tokens (batch size 2)
Stopped: EndToken 2; returned: 0; cached: 3
```

Add `--sample` to use the seeded probabilistic sampler. Neither policy owns the
stopping rules. With real model weights, supply end IDs from the tokenizer/model
configuration; no ID or spelling is inferred by this executor.

## 2. Construct a generator over an existing session

```fsharp
let selectToken (logits: ReadOnlyMemory<single>) =
    TokenSampling.greedy logits.Span

let generator =
    TokenGenerator(sequence, maximumNewTokens = 128,
                   endTokenIds = ReadOnlyMemory<int>(endIds), selectToken = selectToken)
let result = generator.Generate(ReadOnlyMemory<int>(promptIds), onToken = displayToken)
```

For probabilistic selection, pass a function that calls `sampler.Sample(logits.Span)`.
Keep the sampler alive across selections. The generator borrows the session and
selector and does not own disposable resources. The session must be new or reset
before each `Generate`; no reset occurs automatically.

The limit is non-negative and counts only newly emitted tokens, not the prompt.
End IDs are copied at construction and validated against the session vocabulary.
Several end IDs are allowed, duplicates are harmless, and an empty list disables
end-token stopping. The selector must return an ID from the same vocabulary.
Invalid settings or selected IDs raise argument exceptions. No implicit BOS/EOS
insertion or text tokenization is performed.

## 3. Follow the stopping order

The non-empty prompt is validated and prefilled first. A zero generation budget
then returns `TokenLimit` without invoking the selector. Otherwise, the generator
selects one token from the current logits and follows this order:

| Condition | Action | Stop reason |
| --- | --- | --- |
| Selection returns `Error` | Return any already emitted output | `SamplingFailed error` |
| Selected ID is an end token | Do not emit or decode it | `EndToken id` |
| Ordinary token reaches the budget | Emit it; do not decode it | `TokenLimit` |
| Ordinary token cannot be decoded because the cache is full | Emit the prediction from the last available logits | `ContextLimit` |
| Otherwise | Emit, decode it, then repeat with the next logits | Continue |

EOS takes precedence when selected; token budget takes precedence over context
capacity if both apply to an ordinary token. An EOS already present in the prompt
does not stop the loop. EOS is excluded from both returned IDs and callbacks, and
is not put in the cache. No selection happens beyond the requested budget just
to check whether another token would have been EOS.

A selected token can be displayed before it enters the cache. Consequently the
last emitted token remains uncached on a budget/context stop. A prompt that fills
the cache can still produce one continuation ID from its final logits. This is
not context extension: that ID cannot be fed back without a new session/prefix.
The token budget is not pre-reserved in the cache; `ContextLimit` may end a large
request earlier. On EOS or sampling failure after ordinary tokens, those earlier
tokens have already been decoded in order to produce the terminating logits.

## 4. Stream IDs and inspect the outcome

`onToken` runs synchronously once per ordinary selected ID, before its decode.
Use it to feed `TokenTextDecoder.Append` and display the returned text, then call
`Complete` when the loop has ended. See [UTF-8 decoding](token-decoding.md).
The callback never receives EOS. It must not reenter the generator or mutate/use
the session concurrently.

`GenerationResult.Tokens` is a caller-owned array containing exactly the IDs
sent to the callback, or the same sequence if no callback was supplied.
`StopReason` distinguishes successful stopping from a typed sampling failure;
do not interpret `SamplingFailed` as successful completion. The loop retains
O(emitted tokens) host IDs in addition to the session's bounded logits/cache
storage. It does not retain prompt text or decode it automatically.

## 5. Cancellation, exceptions and reuse

`Generate` accepts an optional `CancellationToken`. Checks occur before prefill,
selection and decode; prefill also checks its own batch boundaries. Running
kernels, selectors and callbacks are not forcibly interrupted. A cancellation
requested during the final callback can arrive too late to prevent normal
completion. Cancellation throws `OperationCanceledException` rather than returning
a result. Previously delivered IDs remain delivered.

Callback/selector exceptions also propagate without a result. Invalid selected
IDs are rejected before callback or decode. Session execution failures keep the
session's existing fault semantics. Selection errors or callback exceptions do
not by themselves mark a valid cache faulted, but a new `Generate` still requires
`Reset()` because the prompt has already been processed. Resetting the session
does not reset a seeded sampler; recreate that sampler to restart its stream.
The generator never disposes or resets a borrowed session on success or failure.

## Validation and remaining work

Synthetic tests compare the logits passed to selection with the independent
forward reference and cover EOS in prompts, copied/multiple end IDs, empty output,
zero/one/multiple-token budgets, context limits, stop precedence, exact callback
counts, sampling failures, invalid IDs and cancellation/exception paths. The
absence of extra decoding is checked through the processed cache count.

This completes I.4. A script can now connect a real GGUF model, tokenizer and
prompt with the full generation loop. A user-facing generation CLI remains J.1;
real-model correctness and text quality still require separate validation.
