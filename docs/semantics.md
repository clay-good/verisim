# v0 Shell / Filesystem Semantics

This is the **normative English description** of the v0 environment's transition
semantics (SPEC-2 §2.3). The **executable truth** is the reference oracle
([`src/verisim/oracle/reference.py`](../src/verisim/oracle/reference.py)). Any
disagreement between this document and the code is a bug, resolved by the golden
tests in [`tests/test_goldens.py`](../tests/test_goldens.py) (SPEC-2 §16).

## State

A state is `(fs, cwd, env, last)`:

- `fs`: a map from normalized absolute path to a node. Root `/` is always present
  as a directory.
- A node is either a **file** `{content, mode}` or a **directory** `{mode}`. A
  file's `content_hash` and `size` derive from its content.
- `cwd`: the current working directory (a path).
- `env`: a map of environment variables (small fixed keyset).
- `last`: the observation of the last action — `{exit_code, stdout_hash}`.

### v0 content storage

SPEC-2 §2.1 describes a content-addressable blob store (hash → bytes). v0 stores
file content **inline** as a string over the fixed content-token vocabulary
(`append` concatenates), and derives `content_hash` from it. File equality is
still hash equality, states are still canonical, and a state round-trips from a
single JSONL record without a side store. The blob store is a pure optimization,
deferred until it pays for itself. This is the only deviation from §2.1.

## Paths

Paths normalize to canonical absolute form: `.` is dropped, `..` pops a segment
(clamped at root, so `/..` is `/`), redundant slashes collapse. Relative paths
resolve against `cwd`. The result always begins with `/` and has no trailing
slash except the root itself.

## Exit codes

v0 uses a coarse split: **0** = success, **1** = failure. The specific failure
*condition* per command is listed below; the coarse code is sufficient for the
divergence metric (which compares the exit code as a scalar fact) and keeps the
grammar small. A finer code set may be introduced later with cause.

## Observations (stdout)

Only `cat` and `ls` produce stdout. `cat <file>` emits the file content. `ls
<dir>` emits the newline-joined, sorted basenames of the directory's **direct**
children; `ls <file>` emits the file's basename. All other commands emit empty
stdout. `last.stdout_hash` is the content hash of stdout.

## Commands

Every command sets `last`. A failing command leaves `fs`, `cwd`, and `env`
unchanged and sets `last = {exit_code: 1, stdout_hash: hash("")}`.

| Command | Effect on success | Fails when |
|---------|-------------------|------------|
| `mkdir <path>` | Create an empty directory at `path`. | `path` exists; or parent is missing / not a directory. |
| `rmdir <path>` | Remove the (empty) directory. | `path` is not a directory; or it has children. |
| `touch <path>` | Create an empty file. If `path` already exists (file **or** dir), succeed as a no-op. | Parent is missing / not a directory (only when creating). |
| `rm <path>` | Remove a file. | `path` is missing; or `path` is a directory (use `rm -r`). |
| `rm -r <path>` | Remove `path` and its entire subtree. | `path` is missing. |
| `write <path> <token>` | Set the file's content to `token` (creating it if absent). | `path` is a directory; or parent is missing / not a directory (when creating). |
| `append <path> <token>` | Append `token` to the file's content (creating it with `token` if absent). | `path` is a directory; or parent is missing / not a directory (when creating). |
| `mv <src> <dst>` | Move `src` (and, if a directory, its whole subtree) to `dst`. If `dst` is an existing directory, move *into* it as `dst/basename(src)`. | `src` is missing; the resolved target already exists; the resolved target is inside `src`'s own subtree; or (non-dir-into case) the target's parent is missing / not a directory. `mv a a` is a no-op success. |
| `cp <src> <dst>` | Copy a file. If `dst` is an existing directory, copy *into* it. If the resolved target is an existing file, overwrite its content. | `src` is missing; `src` is a directory (use `cp -r`); or (new target) parent is missing / not a directory. |
| `cp -r <src> <dst>` | Copy `src` and its subtree. Dir-into rule as `cp`. | `src` is missing; the resolved target already exists; or the resolved target is inside `src`'s own subtree. |
| `chmod <mode> <path>` | Set the node's mode (octal). | `path` is missing. |
| `cd <path>` | Set `cwd` to `path`. | `path` is not a directory. |
| `cat <path>` | Emit file content to stdout. No state change. | `path` is not a file. |
| `ls <path>` | Emit directory listing (or file basename) to stdout. No state change. | `path` is missing. |
| `export <KEY>=<token>` | Set `env[KEY] = token`. | (Total — always succeeds.) |

### Notes on chosen simplifications

- **`mv`/`cp -r` reject overwriting an existing target.** Real shells have
  intricate overwrite/merge rules; rejecting overwrite keeps the transition
  total and unambiguous for v0, where the point is *structure and consequence*,
  not faithfulness to every `coreutils` corner. Documented here so it is a
  deliberate choice, not an accident.
- **`mv` of a directory moves the whole subtree** (a single `Move` edit relocates
  all descendants). This is exactly the long-range-dependency structure SPEC-2
  §2.4 wants, so the model must track structure it saw many steps ago.
- **Root `/` is protected.** `rm`/`rm -r`/`rmdir` of `/`, and `mv`/`cp` with `/`
  as the source, all fail. This preserves the invariant that root always exists
  (cf. real `rm --preserve-root`) and keeps every transition total.
- **A directory cannot be moved or copied into its own subtree.** `mv /a /a/b` (or
  `cp -r /a /a/b`) fails — exactly as real `mv`/GNU `cp` reject it. This guard was
  added after the SPEC-11 system-oracle differential harness caught the reference
  interpreter producing an *invalid* state here (the moved subtree was orphaned, its
  parent path gone). It is the canonical example of the H28 loop: the real kernel acting
  as a debugger for the from-scratch model — surface, localize, fix, re-run to agreement.

### Validated against reality (SPEC-11)

The reference interpreter above is a *from-scratch model* of POSIX. SPEC-11's system oracle
runs the **same grammar against a real `/bin/sh` on a real kernel**, inside a hermetic
sandbox, and compares bit-for-bit. The result: on the **structure-building** regime these
semantics are designed to model, the reference oracle is **bit-exactly faithful to a real
computer (agreement = 1.000)**. Every disagreement outside that regime is one of a small set
of *named, intentional modeling boundaries*, enumerated by SY2 and listed here so each is a
deliberate choice, not an accident:

- **root protection** — the `/`-is-protected rule above (a real kernel operates on `/`);
- **overwrite policy** — the mv/cp no-clobber rule above (a real kernel clobbers);
- **permission enforcement** — v0 models `mode` as *data*, not access control: `chmod` records
  a mode but no later command checks it, whereas a real kernel denies traversal/access through
  a directory lacking owner-execute. v0 studies structure and consequence, not access control;
- **self-subtree** — the move/copy-into-own-subtree guard above; GNU `cp` and v0 reject it,
  BSD `cp` (the macOS dev host) does not detect it — a *coreutils*-portability boundary that
  agrees on Linux and is platform-stamped on every SY figure.
