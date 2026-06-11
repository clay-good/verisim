# Host semantics (SPEC-6, HC0-HC3)

> The normative English description of the host world's syscall semantics, paired with the executable
> truth — the Tier-A reference host oracle ([`verisim.hostoracle.ReferenceHostOracle`](../src/verisim/hostoracle/reference.py)).
> The analogue of [`semantics.md`](semantics.md) (filesystem) and [`network-semantics.md`](network-semantics.md).
> Any disagreement between this document and the oracle is a bug in one of them, resolved by test
> ([`tests/test_host_oracle.py`](../tests/test_host_oracle.py)). Golden trajectories pin the semantics and
> are denylisted from the autoresearch engine (SPEC-4 §5).

This is the **HC0 increment 1 + HC1** subset. The structural thesis of SPEC-6 is that the host oracle is a
**composition of sub-oracles**: it owns the process/fd/credential glue and *delegates* file effects to the
v0 filesystem oracle ([`ReferenceOracle`](../src/verisim/oracle/reference.py)) verbatim — it never
reimplements filesystem semantics.

## State (the bundle)

`HostState` is a bundle of coupled subsystems (SPEC-6 §3.1):

- **`procs`** — the process table, `pid → Process(pid, ppid, state, uid, exit_code)`. `state ∈ {RUNNING,
  ZOMBIE}`. The boot state is a single running root process `pid 1` (`ppid 0`, `uid 0`).
- **`fds`** — per-process file-descriptor table, `(pid, fd) → FdEntry(path)`. An fd points *into* the FS
  subsystem (an absolute path). Pipe/socket/dev fd targets are later increments.
- **`fs`** — the v0 filesystem `State`, embedded verbatim (SPEC-2).
- **`next_pid`** — the monotonic pid allocator; canonical, so identical trajectories produce identical
  pids (no nondeterministic identifier churn, SPEC-3 DD-1).
- **`last_exit`** — the exit code of the last syscall.

Canonicalization (sorted process/fd maps; the fs in v0-canonical form) is mandatory, so divergence
measures competence, not identifier churn.

## Syscalls

Every syscall names the **acting pid** explicitly (a scheduler / "current process" is a later
increment). A syscall on a non-`RUNNING` pid fails (`exit 1`) and leaves the state unchanged.

| Syscall | Effect | Failure (`exit 1`) |
|---|---|---|
| `fork <pid>` | Create a child `Process(next_pid, ppid=pid, RUNNING, uid=parent.uid)`; `next_pid++`. Stdout is the child pid. | `pid` not RUNNING |
| `exit <pid> <code>` | `pid → ZOMBIE` with `exit_code = code`; **all of `pid`'s fds are released**. | `pid` not RUNNING; non-integer code |
| `setuid <pid> <uid>` | Set `pid`'s uid. **Root-only:** permitted iff the acting process has `uid == 0`. | `pid` not RUNNING; non-root caller (EPERM); non-integer uid |
| `open <pid> <path>` | Bind the **smallest free** fd for `pid` to `resolve("/", path)`. Stdout is the fd number. | `pid` not RUNNING |
| `write <pid> <fd> <token>` | Resolve `(pid, fd) → path`; **delegate `write <path> <token>` to the v0 FS sub-oracle**; the fs subsystem takes the sub-oracle's next state, exit code, and stdout. | `pid` not RUNNING; `fd` not open (EBADF) |
| `read <pid> <fd>` | Resolve `(pid, fd) → path`; **delegate `cat <path>` to the v0 FS sub-oracle**; stdout is the file's content. **Read-only** — the only bundle effect is `SetExit` (no `FsDelta`), so it closes the write/read round trip without mutating state. Without a per-fd offset (a later increment) it returns the whole content each time. | `pid` not RUNNING; `fd` not open (EBADF); path not a readable file (inherits the FS oracle's failure) |
| `close <pid> <fd>` | Release `(pid, fd)`. | `pid` not RUNNING; `fd` not open (EBADF) |

## Determinism contract

`O(s, a)` is a **pure function of `(s, a)`** — no clock, no RNG, no concurrency, no environment leakage
(single-threaded floor regime; recorded/chaos-seeded scheduling are later regimes, SPEC-6 §3.3). `step`
never mutates its input. The composition invariant: a `write` produces exactly the next state the v0 FS
oracle would, with the process/fd table updated by the glue.

## The bundle delta + the `apply == oracle` invariant (HC1)

`M_θ` will predict a **structured bundle delta**, not a full state — the v0 E4 result (delta prediction
beats absolute-state prediction) carried up the ladder. The host delta vocabulary
([`verisim.host.delta`](../../src/verisim/host/delta.py)) is one edit type per subsystem:
`ProcSpawn` / `ProcExit` (process table), `FdOpen` / `FdClose` (per-process fd table), `CredChange`
(credentials), `SetExit` (the syscall's host-level exit code), and — the composition — `FsDelta`, which
**wraps a v0 filesystem `Delta` verbatim** and is applied by the v0 `apply`. A `write` syscall's bundle
delta is therefore `[FsDelta(<the v0 sub-oracle's own delta>), SetExit(...)]`: the host layer never
reimplements file semantics, it embeds them.

`apply(state, delta)` is compositional and **builds a fresh state** (it never mutates its input): each
edit touches one subsystem, `ProcExit` also releases the exiting pid's fds, and `FsDelta` re-applies
through the v0 `apply`. The contract that keeps the loop model-agnostic — the analogue of v0's **M1** and
the network's **NW1** — is

```
apply(state, oracle.step(state, a).delta) == oracle.step(state, a).state
```

for every transition, by construction. The bundle delta also round-trips through serialization
(`delta_to_list` / `delta_from_list`, the embedded FS delta reusing v0's). Both are pinned by
[`tests/test_host_delta.py`](../../tests/test_host_delta.py) over a mixed trajectory.

## Metrics (HC3)

The deterministic metric core for the host world ([`verisim.hostmetrics`](../src/verisim/hostmetrics/),
pinned by [`tests/test_host_metrics.py`](../tests/test_host_metrics.py)) — no runtime deps, no GPU, the
M3/NW3 discipline. Every metric is a pure function of the bundle state/delta, is `0` (or `1.0`) iff
faithful, and — the structural point of SPEC-6 — **decomposes by subsystem** (`proc` / `fd` / `fs` /
`global`), because the per-subsystem breakdown is what the composition law (H13) needs and what a single
scalar hides (SPEC-6 §5.4, §9).

- **Composed divergence `d`** (`divergence`, §9.1): normalized symmetric difference over the *union* of
  all subsystem facts, `0.0` iff every subsystem agrees. `divergence_by_subsystem` reports the same
  metric *within* each subsystem (each normalized independently); the embedded filesystem reuses v0's
  `state_facts` verbatim, so the composition is visible right down to the metric. `next_pid` is omitted
  from the fact set — it is a derived allocator and would double-penalize a mispredicted `fork`.
- **Bits-to-correct** (`bits_to_correct`, `bits_to_correct_by_subsystem`, §5.4): the scale-free MDL gate
  over bundle deltas, `Σ_subsystem MDL(correction)`. The `fs` subsystem's correction is **delegated to
  v0's own gate** over the embedded edits, exactly as the oracle delegates the file effect. `0` iff the
  predicted delta equals truth as a multiset; `delta_exact` is the `bits_to_correct == 0` predicate.
- **Faithful horizon `H_ε`** (§9.3): reused **verbatim** from v0 (`verisim.metrics.horizon`) — its
  definition (longest faithful prefix of a per-step divergence trajectory) is world-independent.
- **Composition-faithfulness diagnostic** (`composition_law`, the headline-new metric, §9.2, H13):
  given the per-step per-subsystem faithfulness booleans, it measures composed per-step acceptance `a`
  against two candidate laws over the per-subsystem rates `a_i`. Because a step fails if *any* subsystem
  fails, `∏_i a_i ≤ a ≤ min_i a_i`; the verdict reports whether `a` sits at the **multiplicative**
  (independent failures) end, the **weakest-link** (coincident failures) end, or is **coupled**
  (anti-correlated — the honest negative where "model subsystems independently" is the wrong bet).
- **Privilege-faithfulness** (`privilege_faithfulness`, §9.4): the security-relevant metric — the
  agreement rate on the *denied/allowed* outcome over privilege-relevant transitions, because a
  defender's trust in the simulator hinges on it getting **failures** (EPERM, EBADF) right, not just
  successes.
- **Run-record schema** (`HostRunRecord`, §9): one structured record per rollout — config, seed, `ε`,
  the **composed** divergence trajectory *and* the **per-subsystem** trajectories, and the consultation
  schedule — from which `faithful_horizon`, `subsystem_horizons`, and `oracle_calls` are derived and the
  EH1 curve / H13 measurement (HC6) will be read. Round-trips through JSONL. Figures come *only* from
  these records (the SPEC-2 §7.3 discipline).

## Deferred (later HC increments)

`wait`/`kill`, `dup`/`lseek`/`read`, `chdir` + per-process cwd, pipes/signals (IPC), sockets (the SPEC-5
net sub-oracle), and the scheduler (`yield`/`advance`) and its interleaving-entropy dial. This document
grows with them.
