# v0 Model Representation

How the learned world model `M_θ` (SPEC-2 §5, milestone M4) serializes states,
actions, and deltas to tokens, and the decisions that resolve the SPEC-2 §17 open
questions #1 (content modeling) and #3 (path/tree tokenization). The executable
truth is [`src/verisim/model/`](../src/verisim/model/) (`vocab.py`, `tokenizer.py`,
`grammar.py`).

## The task

`M_θ` maps `<bos> serialize(state) serialize(action) <gen>` → `serialize(Δ) <eos>`
(SPEC-2 §5.2): it predicts the **structured delta**, not a regenerated state. A
small from-scratch decoder-only transformer is trained teacher-forced on
oracle-generated `(prompt, target)` pairs (Stage 1, SPEC-2 §5.3).

## Closed vocabulary

v0 fixes finite content/mode/env/name vocabularies (SPEC-2 §2.2), so the whole DSL
maps to a small **closed** token set (~64 tokens) built deterministically from the
`EnvConfig`: specials (`<pad> <bos> <eos>`), section/structure markers, delta-op
tokens, one command token per §2.2 command, exit codes, and the four leaf
vocabularies (path-name segments, content tokens, modes, env keys). A from-scratch
tiny transformer trains over exactly these ids.

## Decision: content (resolves §17.1 #1)

File content is a **string formed by concatenating the content vocabulary**
(`write` sets one token; `append` concatenates). The content vocabulary is
**prefix-free** (no token is a prefix of another), so a content string decomposes
**uniquely by greedy longest-match** into content tokens. This keeps the M0–M3
state representation unchanged (content stays an inline `str`) while making it
cleanly tokenizable. Boundary noted for later phases: real byte content would
replace the fixed vocabulary (SPEC-2 §17.1).

## Decision: paths (resolves §17.1 #3)

Paths are **absolute** in v0 (the drivers only ever emit absolute paths), encoded
as `<p>` + one name token per segment + `</p>` and reconstructed as `"/" +
"/".join(segments)`. This is the "serialized-DSL + segment tokenizer" option of
§17.1 #3; a structural/graph encoding of the tree is left for later if structure
proves hard to learn.

## Decision: stdout and the result hash

The state stores `last.stdout_hash` (a hash), which is not tokenizable. Two moves:

- The model **input** serializes `last.exit_code` only and **omits**
  `stdout_hash` — no command reads prior stdout, so it is not needed to predict the
  next transition.
- The model **output** predicts `stdout` as tokens (`<o> ... </o>`, from the
  recorded observation), and the parser reconstructs the edit as
  `SetResult(exit_code, content_hash(stdout))`. cat-stdout (content words) and
  ls-stdout (single-char names + `<nl>`) use **disjoint** token languages, so the
  reconstruction is unambiguous.

## Constrained decoding (grammar-validity by construction)

Decoding is constrained by an LL(1) automaton over the delta grammar
([`grammar.py`](../src/verisim/model/grammar.py)): at each step the model's logits
are masked to the grammar's allowed next tokens, so the decoded sequence is always
parseable into a valid `Delta` **regardless of the model's weights**. Termination
is guaranteed by capping the top-level edit count (forcing `<eos>`) and each
repeating leaf run (forcing its closing token); both forced tokens are always
grammar-valid in their state.

This keeps the spec's distinction crisp (SPEC-2 §5.2): **grammar-validity is the
decoder's job; semantic faithfulness is the oracle's.** A syntactically valid delta
can still be *wrong* — catching that is the propose-verify-correct loop's purpose,
not the decoder's.
