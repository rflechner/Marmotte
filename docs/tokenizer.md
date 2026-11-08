# D.1: turn vocabulary metadata into BPE token IDs

This tutorial explains the BPE tokenizer in [roadmap item D.1](../ROADMAP.md#d-tokenizer).
By the end, you should be able to trace a small input through ranked merges,
map the result to vocabulary IDs, and explain where BOS/EOS markers belong.
No model download is needed for the worked example.

**Status:** D.1 is implemented as `BpeTokenizer` in Core and
`GgufTokenizer.read` in the GGUF library. Input must already be split into
pieces; automatic Tekken pre-tokenization (D.2), streaming text decoding (D.3)
and chat templates (D.4) remain unimplemented. The examples below use invented
vocabularies, not measured Ministral IDs.

## 1. Find the tokenizer's ingredients

If you have a GGUF file and the .NET 10 SDK, run this from the repository root,
replacing the path with your file:

```sh
dotnet run --project src/Marmotte.Cli -- info "/path/to/model.gguf" --metadata --tensor-limit 0
```

Look for `tokenizer.ggml.*`. The output previews arrays rather than printing
the whole vocabulary; the complete values remain available in `model.Metadata`
after `Gguf.readFile`. Inspection reads the index, without loading weights.

| Metadata suffix after `tokenizer.ggml.` | Type | Meaning |
| --- | --- | --- |
| `model` | string | Tokenizer family; `gpt2` denotes a BPE family |
| `tokens` | string array | Array position is the model's token ID |
| `token_type` | int32 array | Parallel token categories, when present |
| `merges` | string array | Ordered pair-merge rules |
| `bos_token_id`, `eos_token_id` | uint32 | Beginning/end markers |
| `unknown_token_id`, `separator_token_id`, `padding_token_id` | uint32 | Other special roles, when supplied |
| `pre` | string | Pre-tokenizer profile, such as `tekken` |
| `add_bos_token`, `add_eos_token` | bool | Automatic marker insertion policy |

Token types distinguish normal (1), unknown (2), control (3), user-defined (4),
unused (5), and byte (6) entries. A present type array must align with the
vocabulary. Token scores, when supplied, are separate from merge ranks.
Vocabulary and special-ID fields are described in the
[GGUF tokenizer specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#tokenizer);
profile and insertion keys are listed in the
[metadata constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py).

The loader preserves absent flags separately from their effective defaults. It
rejects unsupported profiles, wrong types, mismatched type-array lengths, invalid
IDs, duplicate spellings/pairs, and merges whose operands or result are absent
from the ordinary vocabulary.

## 2. Follow one small example

Imagine one already separated input piece, `catcat`. Our invented vocabulary is:

| ID | Stored spelling | Role in this example |
| --- | --- | --- |
| 0 | `<bos>` | Control marker |
| 1 | `<eos>` | Control marker |
| 2 | `c` | Ordinary token |
| 3 | `a` | Ordinary token |
| 4 | `t` | Ordinary token |
| 5 | `ca` | Ordinary token |
| 6 | `cat` | Ordinary token |

Its entire merge list is `c a`, then `ca t`: their ranks are 0 and 1.
These are illustrative IDs and rules, not values from Ministral.

![Four merge steps turn c a t c a t into cat cat, then vocabulary lookup produces IDs 6 and 6; optional BOS insertion produces 0, 6, 6.](tokenizer-bpe.svg)

Start with `c | a | t | c | a | t`. Select the available adjacent pair with the
lowest rank, breaking ties from left to right, and repeat after each merge:

| Step | Symbols | Reason |
| --- | --- | --- |
| 0 | `c · a · t · c · a · t` | Initial symbols |
| 1 | `ca · t · c · a · t` | Leftmost rank-0 pair |
| 2 | `ca · t · ca · t` | Remaining rank-0 pair beats rank 1 |
| 3 | `cat · ca · t` | Leftmost rank-1 pair |
| 4 | `cat · cat` | Remaining rank-1 pair |

There is no `cat cat` rule, so merging stops. Vocabulary lookup gives `[6; 6]`.
A merge rank chooses an operation; a token ID selects a vocabulary entry.
They are different numbers with different purposes. The ranked-pair ordering
and tie rule follow the
[reference BPE implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp).

This example uses ASCII to keep the symbols readable. A byte-level tokenizer
must handle UTF-8 bytes and its profile's reversible byte-to-symbol mapping;
splitting a .NET string into UTF-16 `char` values is not an equivalent algorithm.
Do not add normalization, trimming, or whitespace splitting as a convenience:
those choices can change the IDs presented to the model.

### Try a sentence: `Hello, how are you ?`

Use a **second, invented tokenizer** for this walkthrough. Its vocabulary and
merge rules are independent of the `catcat` example. These are not measured
Ministral or Tekken results; real token IDs require the model's exact tokenizer.

Assume this teaching pre-tokenizer produces the following pieces. Here `␠`
is a visible notation for one ordinary space, not a literal character inserted
into the input or a prescribed byte-to-symbol encoding:

```text
Hello, how are you ?
          ↓ assumed piece boundaries
[Hello] [,] [␠how] [␠are] [␠you] [␠?]
```

For this example, the complete ordered merge list is:

| Rank | Pair → merged symbol | Rank | Pair → merged symbol |
| --- | --- | --- | --- |
| 0 | `H + e → He` | 6 | `a + r → ar` |
| 1 | `He + l → Hel` | 7 | `ar + e → are` |
| 2 | `Hel + l → Hell` | 8 | `␠ + are → ␠are` |
| 3 | `Hell + o → Hello` | 9 | `y + o → yo` |
| 4 | `h + o → ho` | 10 | `yo + u → you` |
| 5 | `ho + w → how` | 11 | `␠ + you → ␠you` |

The vocabulary contains every initial and merged symbol. Its final entries
have the illustrative IDs below; intermediate entries have other distinct IDs.
Apply the rules within each piece, then concatenate the resulting ID sequences:

| Piece | Merge progression | Final IDs |
| --- | --- | --- |
| `Hello` | `H · e · l · l · o → He · l · l · o → Hel · l · o → Hell · o → Hello` | `[20]` |
| `,` | No applicable rule | `[21]` |
| `␠how` | `␠ · h · o · w → ␠ · ho · w → ␠ · how` | `[22; 23]` |
| `␠are` | `␠ · a · r · e → ␠ · ar · e → ␠ · are → ␠are` | `[24]` |
| `␠you` | `␠ · y · o · u → ␠ · yo · u → ␠ · you → ␠you` | `[25]` |
| `␠?` | No applicable rule | `[22; 26]` |

Notice that one piece can produce several tokens: `␠how` stays as a space
(ID 22) followed by `how` (ID 23), because there is no merge for that pair.
Conversely, `␠are` and `␠you` each include their leading space in one token.

```text
Tokens:  [Hello] [,] [␠] [how] [␠are] [␠you] [␠] [?]
IDs:     [ 20;    21; 22;  23;    24;     25;   22; 26 ]
```

Joining those token spellings, with `␠` restored to a space, gives exactly
`Hello, how are you ?`, including the space before `?`. With an invented BOS
ID of 0, BOS insertion enabled and EOS insertion disabled, the model input
would be `[0; 20; 21; 22; 23; 24; 25; 22; 26]`.

## 3. Apply the sequence policy

For the invented vocabulary, explicitly choose `add_bos_token = true` and
`add_eos_token = false`. Wrapping the payload once gives `[0; 6; 6]`.
With both flags enabled it would give `[0; 6; 6; 1]`. A defined EOS ID alone
does not imply that every prompt ends with EOS.

The API distinguishes automatic boundary insertion from explicit
`TokenizerSegment.SpecialToken` IDs. `TextPiece` never recognizes special
spellings automatically. Control, user-defined and unknown tokens can be inserted
explicitly; unused and ordinary entries cannot. No duplicate BOS suppression
occurs: callers must choose one owner for boundary insertion. Chat rendering
belongs to D.4.

The resulting IDs will select rows in `token_embd.weight` through
[`EmbeddingLookup`](kernels.md). In our invented example, rows 0, 6 and 6 would
be selected in that order; tokenization itself does not read those weight rows.

## 4. Keep D.1's boundary visible

D.1 covers vocabulary loading, BPE merging and special-token policy. D.2 supplies
Tekken pre-tokenization; merges must stay within its pieces. D.3 reconstructs
UTF-8 text, including sequences split between tokens. D.4 formats conversations.
Implementing the worked merge loop alone will not make the target model usable.

There is another compatibility detail to check with D.2: the reference Tekken
profile enables a whole-piece vocabulary shortcut before ordinary merging
(`ignore_merges`). A generic merge-only example does not capture that behavior.
The profile also supplies defaults which explicit metadata may override. See
[the profile selection and BPE session logic](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp).

The format-neutral tokenizer and algorithm live in Core; GGUF interpretation
lives in `Marmotte.Gguf`. Construction uses metadata only, without reading
weights or changing the index reader's allocation limits.

## 5. Turn the example into acceptance cases

The Core and GGUF tests use small independently specified fixtures:

- Expect `[6; 6]` for the toy piece and `[0; 6; 6]` with the stated wrapping policy.
- Exercise competing ranks, repeated pairs, ties, and a piece with no applicable merge.
- Reject invalid IDs, mismatched arrays and malformed merge entries; explicitly
  define how absent optional fields and unsupported token categories are handled.
- Check whitespace, non-ASCII bytes, empty input and literal special-token text.
- Once D.2 exists, compare exact token IDs for a fixed corpus against the target
  tokenizer, including pieces already present in the vocabulary. A text round
  trip alone cannot prove that the segmentation matches.

## Run the implementation

From the repository root, without a local model:

```sh
dotnet fsi examples/Tokenizer.fsx
```

This runs the sentence example with its exact invented IDs and checks the result:
`[0; 20; 21; 22; 23; 24; 25; 22; 26]`. The actual stored spelling for the space
byte is `Ġ`; the earlier tables use `␠` only for display.

To construct the same engine from an already read GGUF index:

```fsharp
let loaded = GgufTokenizer.read model.Metadata
let segments = [| TokenizerSegment.TextPiece "Hello" |]
let ids = loaded.Tokenizer.Encode(segments, true)
```

Open `Marmotte.Core` and `Marmotte.Gguf` for that snippet. Supplying a whole
sentence as one `TextPiece` bypasses pre-tokenization and is not a substitute
for D.2. `Encode` wraps the complete segment sequence once when its second
argument is true; false disables boundary insertion.

### Supported contract and limits

- `model` must be `gpt2`. Missing `pre`, `default`, and `gpt-2` use ranked merges
  with BOS/EOS off by default. `tekken` enables whole-piece lookup and defaults
  BOS on, EOS off. Explicit flags override these defaults; enabled insertion
  requires a corresponding ID. No conventional special IDs are guessed.
- Missing `merges` means no merges. Missing token types mean Normal. Scores
  do not participate in BPE ranking. Unknown categories and profiles are rejected.
- Normal and Byte entries participate in byte-alphabet lookup and merging.
  Alternate byte spellings such as hexadecimal placeholders are not interpreted.
  Missing input-byte coverage raises `InvalidDataException`; no unknown-token
  replacement silently changes the text.
- Metadata errors raise `InvalidDataException`; invalid Core constructor or
  special-ID arguments raise `ArgumentException`. Null arguments are rejected.
  Malformed UTF-16 raises `EncoderFallbackException` instead of replacement.
- Construction snapshots vocabulary arrays and merge ranks. Results are fresh
  arrays; encoding calls share no mutable working state. Token IDs are Int32.
- Each merge rescans adjacent pairs with leftmost tie-breaking. This correctness
  baseline can be quadratic in piece length, with additional string allocation.
  No performance claim or target-model compatibility claim is made yet.

The byte alphabet preserves bytes 33–126, 161–172 and 174–255 as their same-valued
Unicode characters. Other bytes, in ascending order, map to characters starting
at U+0100. Input first becomes strict UTF-8 bytes, then these symbols. This was
checked against the [byte mapping reference](https://github.com/ggml-org/llama.cpp/blob/master/src/unicode.cpp).
The upstream sources linked above were consulted on 2026-11-08. They are moving
branches; pin a revision when recording target-model compatibility fixtures.
