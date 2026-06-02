# Host semantics (SPEC-6, HC0)

> The normative English description of the host world's syscall semantics, paired with the executable
> truth — the Tier-A reference host oracle ([`verisim.hostoracle.ReferenceHostOracle`](../src/verisim/hostoracle/reference.py)).
> The analogue of [`semantics.md`](semantics.md) (filesystem) and [`network-semantics.md`](network-semantics.md).
> Any disagreement between this document and the oracle is a bug in one of them, resolved by test
> ([`tests/test_host_oracle.py`](../tests/test_host_oracle.py)). Golden trajectories pin the semantics and
> are denylisted from the autoresearch engine (SPEC-4 §5).

This is the **HC0 increment 1** subset. The structural thesis of SPEC-6 is that the host oracle is a
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
| `close <pid> <fd>` | Release `(pid, fd)`. | `pid` not RUNNING; `fd` not open (EBADF) |

## Determinism contract

`O(s, a)` is a **pure function of `(s, a)`** — no clock, no RNG, no concurrency, no environment leakage
(single-threaded floor regime; recorded/chaos-seeded scheduling are later regimes, SPEC-6 §3.3). `step`
never mutates its input. The composition invariant: a `write` produces exactly the next state the v0 FS
oracle would, with the process/fd table updated by the glue.

## Deferred (later HC increments)

`wait`/`kill`, `dup`/`lseek`/`read`, `chdir` + per-process cwd, pipes/signals (IPC), sockets (the SPEC-5
net sub-oracle), the scheduler (`yield`/`advance`) and its interleaving-entropy dial, and the
compositional bundle `Delta` + `apply == oracle` invariant (HC1). This document grows with them.
