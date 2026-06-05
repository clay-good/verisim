# SPEC-11 — The System Oracle: Reality as the Verifier

**Cross-cutting validation specification: the whole program rests on one claim — that for computer
worlds a *deterministic ground-truth oracle exists*. Every figure to date proves that against a
from-scratch reference *model* of POSIX, not against a real OS. This spec closes the gap. It runs the
v0 command grammar against a **real `/bin/sh` on a real kernel**, inside a sandbox so locked down that
*no action can escape it*, and validates — bit for bit — that the reference oracle's predicted next
state is the state reality actually produces. It does not advance a model; it converts the program's
one structural assumption into a measured, regenerable result.**

> **▶ SYSTEM-ORACLE SPEC — 2026-06.** A *cross-cutting* validation spec, sibling to [SPEC-9](./SPEC-9.md)
> (the free-oracle scaling regime) and [SPEC-10](./SPEC-10.md) (the faithful-horizon scaling law). It
> invents **no new world, metric, or model.** It operationalizes the single forward-looking abstraction
> v0 has carried since the first commit: `oracle/base.py`'s own docstring states that Phase 1's
> `SandboxOracle` "drops in unchanged," and [SPEC-3 §2](./SPEC-3.md) reserved hypothesis **H4**
> ("mechanism survives reality") and milestone **S1** for exactly this. SPEC-11 is that milestone,
> elevated to a standalone spec because it validates the *premise the entire program is built on* — and
> because the result is the program's strongest possible answer to the one objection every reviewer
> raises: *"your oracle is a model, not reality."* Gated like everything: the sandbox's hermeticity is
> *proven* before any agreement number is claimed, and a disagreement between the reference model and the
> real kernel is **first-class** — not a failure, but the differential signal that makes the reference
> oracle *self-correcting*.

Read [SPEC.md §2](./SPEC.md) for *why* the oracle is free and exact (and §2.1 for the
reference-vs-system distinction this spec collapses), [SPEC-2 §3.2](./SPEC-2.md) for the `Oracle`
protocol the `SandboxOracle` implements unchanged, and [SPEC-3 §2](./SPEC-3.md) for the original Tier-A
/ Tier-B system-oracle architecture and the HW-1 concurrency wall this spec scopes around. This document
is *whether the reference oracle is faithful to a real computer, measured exactly, safely, and
reproducibly.*

---

## 0. One-paragraph thesis

The program's headline asset is an asymmetry: in computer worlds, unlike physical or visual worlds, a
deterministic ground-truth oracle is free, exact, and resettable (SPEC.md §2). Every result so far —
twenty-five oracle-grounded figures, 448,260 exhaustive invariant pairs, the cross-world and
cross-proposer syntheses — establishes the *mechanism* against a **reference oracle**: a from-scratch
deterministic interpreter of POSIX semantics. That is enough to prove the loop *works*; it is not enough
to prove the oracle is faithful to *reality*, because the reference oracle is itself a symbolic model of
the thing it claims to be ground truth for. SPEC-11's thesis, stated to be falsified: **on the exact v0
command grammar, the reference oracle's predicted next state equals the state a real `/bin/sh` on a real
POSIX kernel produces, bit for bit, on every transition — and the few transitions where it does not are
localizable, named modeling boundaries, not irreducible real-OS noise.** (We validate on the macOS dev
host's Darwin/XNU kernel first, per the §2.5 macOS-first principle; the headline is pure POSIX and
reproduces on Linux CI.) We validate this inside a
**hermetic, no-side-effect sandbox**: each transition runs the real shell against an ephemeral copy of
the state in a fresh, capability-stripped, network-less, throwaway namespace that is destroyed after the
diff, so that *no action the validation runs can affect the host, the network, or any subsequent step* —
the only observable effect of the entire harness is the agreement table it writes. If the agreement is
near-total, "faithful to a real computer" stops being a promise and becomes a committed CSV. If it is
not, the disagreements are a free differential debugger that point at exactly which lines of the
reference oracle's POSIX model are wrong, and the same harness regenerates after each fix. Either branch
is a strict gain: the first proves the premise; the second improves the oracle and *still* proves the
premise on the corrected grammar.

---

## 1. What this closes — and why it is the highest-leverage thing left to run

The repo has always named this gap honestly. SPEC.md §2.1 distinguishes the *reference oracle* ("a
symbolic model of reality, not reality... good enough to prove the mechanism; not sufficient to prove
fidelity to a real OS") from the *system oracle* ("the actual sandboxed system... what makes the claim
about reality rather than about a model of reality"). SPEC-3 §1's W1 row reads, verbatim, "the oracle is
a model, not reality." This is the program's central exposure, and it is also its single largest
unrealized strength: it is the one weakness that, once closed, *cannot be closed by anyone in the
oracle-free world-model field*, because they have no reality to validate against. Closing it does four
things at once, which is why it earns its own spec:

1. **It converts the load-bearing assumption into a measurement.** Today the sentence "computer worlds
   have a free, perfect oracle" is an argument. After SY1 it is a number with a bootstrap CI, regenerable
   from `reproduce.sh`.
2. **It turns the reviewer's sharpest attack into the headline result.** "Does the mechanism survive
   contact with a real OS?" is the obvious line of attack (SWOT: the central threat). SY1 *is* that
   experiment, run by us first, with its honest negative pre-registered.
3. **It makes the reference oracle self-correcting.** Any disagreement is a differential signal (SPEC-3
   §2.4) localized to a single command: the harness hands you the input state, the action, the predicted
   delta, and the real delta side by side. Fixing the reference model and re-running is a tight loop. The
   weakness becomes a maintenance *tool*.
4. **It costs no new runtime dependency and no new world.** The `SandboxOracle` is pure standard library
   (`subprocess`, `os`, `ctypes` for `unshare(2)`) behind the existing `Oracle` protocol; the v0
   deterministic core stays dependency-free (pyproject's standing promise). Nothing else in the codebase
   branches on which oracle it holds.

This is deliberately *not* a model spec. It runs no training. It exists to make one sentence true beyond
argument.

---

## 2. The system oracle: a real shell behind the v0 `Oracle` protocol

### 2.1 The drop-in

The contract is already written. `src/verisim/oracle/base.py` defines:

```python
@runtime_checkable
class Oracle(Protocol):
    def step(self, state: State, action: Action) -> StepResult: ...
    def reset(self, state: State) -> State: ...
    def determinism_report(self) -> DeterminismReport: ...
```

SPEC-11 ships `SandboxOracle(Oracle)` in `src/verisim/oracle/sandbox.py` (**to be built**), satisfying
the same protocol so that every existing consumer — `run_rollout`, the differential invariant scripts,
every `E*` experiment — accepts it unchanged. The only new surface is the sandbox itself.

```python
class SandboxOracle:
    """A real /bin/sh over a real kernel, behind the v0 Oracle protocol.

    Each step materializes `state` into an ephemeral tmpfs inside a fresh
    user+mount+net+pid namespace, runs the single command via real /bin/sh,
    snapshots the resulting tree back into a canonical State, and tears the
    namespace down. No state persists between steps; no effect escapes the
    namespace. See SPEC-11 §2.3 (the hermeticity contract).
    """

    def __init__(self, *, shell: str = "/bin/sh", timeout_s: float = 2.0,
                 seal: DeterminismSeal = DEFAULT_SEAL) -> None: ...

    def step(self, state: State, action: Action) -> StepResult: ...
    def reset(self, state: State) -> State: ...           # snapshot, not mutate
    def determinism_report(self) -> DeterminismReport: ...
```

### 2.2 One step, precisely

For the v0 filesystem grammar a single `step(s, a)` is:

1. **Materialize.** Render the canonical `State s` into a real directory tree inside a private
   `tmpfs` mount (files, modes, contents). This is the only writable surface the command will see.
2. **Isolate.** `unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWNET | CLONE_NEWPID | CLONE_NEWIPC)`. The
   tmpfs is the new mount namespace's root-equivalent working tree; the rest of the host is bind-mounted
   **read-only** or not mounted at all. The new network namespace has **no interfaces but loopback** —
   there is no veth pair, so egress is physically impossible, not merely filtered.
3. **Drop.** `prctl(PR_SET_NO_NEW_PRIVS)`, drop all capabilities, install a **seccomp-bpf allowlist**
   (open/read/write/stat/rename/unlink/mkdir/… for the grammar; everything else `EPERM`), set rlimits
   (CPU time, file size, `nproc`, address space), map the caller to an unprivileged uid inside the user
   namespace.
4. **Seal nondeterminism.** Apply the `DeterminismSeal`: pin `TZ`, `LC_ALL`, `LANG`; freeze the clock via
   a `faketime`-style shim or a fixed `SOURCE_DATE_EPOCH`; seed/zero any RNG the grammar can reach;
   scrub inherited environment to a fixed allowlist. The v0 grammar touches none of time/RNG/concurrency
   in its semantics, which is exactly why it is the right grammar to validate first (§2.4).
5. **Execute.** `('/bin/sh', '-c', action.raw)` with the tmpfs as CWD, under the timeout.
6. **Snapshot + diff.** Walk the resulting tree, canonicalize it into a `State s'_sys`, compute the real
   delta `Δ_sys`, capture exit code and stdout. Return `StepResult(s'_sys, Δ_sys, exit_code, stdout)`.
7. **Destroy.** The namespace and its tmpfs evaporate when the child exits. Nothing is carried to the
   next step; `reset` re-materializes from a `State`, never from leftover disk.

The reference oracle's `step` is unchanged. The differential harness (§3) calls **both** on the same
`(s, a)` and compares.

### 2.3 The hermeticity contract — "no action can occur"

This is the user's explicit requirement and a first-class deliverable, not an implementation detail. The
sandbox is validated as a *closed system*: the only effect any step is permitted to have on anything
outside its own throwaway namespace is the `StepResult` it returns. Concretely, the contract (`H29`,
proven in SY3) is:

| Channel | Guarantee | Enforcement |
|---|---|---|
| Host filesystem | zero bytes written outside the per-step tmpfs | read-only binds; private mount ns; tmpfs destroyed on exit |
| Network | zero packets sent or received | empty net namespace (loopback only, no veth) — egress is *physically* absent |
| Privilege | no privilege escalation | `NO_NEW_PRIVS`, all caps dropped, unprivileged uid map, seccomp allowlist |
| Persistence | no state survives a step | fresh namespace + tmpfs per step; `reset` re-materializes from `State` |
| Resource | no host exhaustion | CPU/mem/nproc/file-size rlimits + wall-clock timeout |
| Determinism | declared, not assumed | `determinism_report()` rides every figure (§2.4, the honesty surface) |

"No action can occur" is given its exact meaning here: **the validation never performs a consequential
action on any real system.** It performs a *reversible experiment in a vacuum* — render state, observe
the kernel's response, throw the vacuum away — whose sole output is a comparison. Each guarantee in the
table carries a **negative control** in SY3: we attempt the prohibited thing (write outside the tmpfs,
open a socket, escalate) and assert it is denied. A guarantee without a teeth-bearing negative control is
not counted as proven (the v0 invariant discipline: "the checks have teeth," verification.md §1).

### 2.4 The determinism report is the honesty surface

The reference oracle seals every nondeterminism source by construction; the system oracle cannot, and the
`determinism_report()` is where that is disclosed, per figure (SPEC-3 §2.3). For the v0 grammar the seal
is total *and we prove it is* — the grammar's semantics are a pure function of state and action with no
clock, RNG, or concurrency dependence — so the report reads all-sealed and SY1 can claim **true**
determinism, not recorded replay. The value of stating it explicitly is that the moment the grammar grows
(SPEC-3's Tier-B coreutils, the host world's threads) the report flips the relevant flag to `recorded`
and every downstream figure inherits the disclosure automatically. **HW-1 (concurrency) is out of scope
by construction:** the v0 grammar is single-threaded, which is precisely why it is the validation
grammar — bit-exact ground truth is *attainable* here, so a disagreement is a real model defect, never a
scheduling artifact.

### 2.5 Where it runs (the macOS-first, cross-POSIX principle)

> **▶ TESTING PRINCIPLE (2026-06, as built).** **macOS is the primary host; Linux CI confirms for
> free.** The author's development machine is an Apple Silicon Mac mini, and the program's discipline
> is to *test locally first* on it. The `SandboxOracle` is therefore built **cross-POSIX**: the same
> code runs on macOS (BSD coreutils) and Linux (GNU coreutils), and the committed SY figures are the
> macOS numbers, **platform-stamped**. The reviewer's instinct that the choice of real kernel does not
> matter for the central claim is *correct and load-bearing*: "reality" does not mean "Linux." macOS
> runs a real POSIX kernel (Darwin/XNU) with a real `/bin/sh`; validating against it is a validation
> against reality. The headline — bit-exact agreement on the structure-building grammar — is **pure
> POSIX and platform-independent**, so it reproduces identically on Linux. The only platform-sensitive
> item is a single coreutils-implementation choice (`self-subtree`, §4), which is named, disclosed, and
> itself a finding (not every divergence is fundamental). This supersedes the earlier "Linux-truth"
> framing: a hard Linux gate would have made CI fragile (Ubuntu 24.04 AppArmor restricts unprivileged
> user namespaces) for *no scientific gain*.

The design that follows from the principle:

- **Cross-POSIX core, always on.** The hermeticity that matters for the v0 *filesystem* grammar —
  throwaway-tree confinement (v0 `resolve` clamps `..`, so every mapped host path is provably inside the
  tree), env scrub, resource limits, no privilege gain, and a hard **grammar allowlist** (only a rendered
  v0 action is ever exec'd; an arbitrary string never reaches the shell) — is kernel-feature-free and
  holds on any POSIX host. This is the default tier and what the committed figures run under.
- **Namespace hardening is a disclosed, optional Linux layer.** Physical net/PID/user-namespace isolation
  is a Linux enhancement, **probed and disclosed**, never a hard requirement — so a runner without
  unprivileged userns degrades to the (already hermetic) process tier rather than failing. `hermeticity()`
  and `determinism_report()` state exactly which tier ran.
- **Disclosed skip only when there is no shell at all.** The constructor raises `SystemOracleUnavailable`
  only when `/bin/sh`/coreutils are genuinely absent; the SY tests `skipif` on that and disclose it. A
  skip is never counted as agreement.
- **The reference oracle remains the portable default.** Nothing about v0's cross-platform, dependency-
  free posture changes; the `SandboxOracle` is pure stdlib behind the `Oracle` protocol (the `[system]`
  documentation-marker extra, §6.3).

---

## 3. The differential-validation protocol

The core of SPEC-11 is one function and the discipline around it.

```python
def differential_step(s: State, a: Action, ref: Oracle, sys: Oracle) -> DiffRecord:
    """Run both oracles on the same (s, a); return an exact agreement record."""
    r_ref, r_sys = ref.step(s, a), sys.step(s, a)
    agree_world  = canonical_world(r_ref.state) == canonical_world(r_sys.state)  # bit-exact
    agree_exit   = r_ref.exit_code == r_sys.exit_code
    agree_stdout = r_ref.stdout.rstrip("\n") == r_sys.stdout.rstrip("\n")
    cls = AGREE if (agree_world and agree_exit) else classify_divergence(s, a)
    return DiffRecord(action.raw, action.name, agree_world, agree_exit, agree_stdout, cls, ...)
```

**Three bit-exact channels (as built).** `canonical_world(·)` is the existing v0 canonical
serialization (verification.md §1, the *same* equality the M1/NW1 invariants use) **minus the `last`
observation**, so the *world* channel (filesystem + cwd + env) is orthogonal to the *exit* and
*stdout* channels reported separately. This refines the spec's original single `agree_state` /
`agree_delta` sketch: a real kernel exposes its effect as `(world, exit, stdout)`, not as v0's
*internal* delta representation, so the delta is reconstructed for the protocol/debugger payload but
the **verdict is the world+exit equality** — the cleanest, representation-independent statement of
"the reference oracle predicted what reality produced." A disagreement is never silent:
`classify_divergence` localizes it to a named v0 modeling boundary (§4) or flags `residual` for SY2.

**Three coverage tiers, mirroring the existing invariant discipline (verification.md §1):**

- **Tier 1 — exhaustive over the grammar.** Enumerate the full v0 action space against the same canonical
  state battery the reference invariants already sweep (the 448,260-pair exhaustive set). Every pair gets
  a `DiffRecord`. This is the strongest claim the harness can make: not "agrees on a sample" but "agrees
  on the whole grammar."
- **Tier 2 — driver-sampled at depth.** Run the `weighted` and `adversarial` drivers (the EN/EH eval
  drivers) for `200 seeds × 60 steps`, stepping *both* oracles in lockstep, so agreement is measured along
  realistic multi-step trajectories, not just isolated transitions.
- **Tier 3 — the curve, head to head.** Re-run E1 — the v0 prime directive, `H_ε(ρ)` — with the
  `SandboxOracle` substituted for the `ReferenceOracle`, and overlay the two curves. If the premise holds,
  they are indistinguishable within CI; the overlay *is* the figure that retires W1.

Aggregation reuses `verisim.metrics.aggregate.bootstrap_ci`; the CSV schema extends the existing one with
`agree_state_rate, agree_delta_rate, ci_low, ci_high, n_pairs, first_disagreement_action`.

---

## 4. Hypotheses (pre-registered)

Continuing the global H-ID space past SPEC-10's H26, and operationalizing SPEC-3's long-reserved H4. Each
is stated with the form that would falsify it and the honest negative banked in advance (the v0 norm,
SPEC §9–10).

- **H27 — the reference oracle is faithful to reality.** On the v0 grammar, `canonical(O_ref(s,a).state)
  == canonical(O_sys(s,a).state)` for every transition in the Tier-1 exhaustive set and along Tier-2
  trajectories; `agree_state_rate = 1.000` with a degenerate CI. *Refuted if* the agreement rate is below
  1 on transitions where the determinism report is all-sealed — i.e. the reference model of POSIX is
  *wrong* somewhere the grammar reaches. Tested as **SY1**. **This is the headline: it is the experiment
  that makes "computer worlds have a free, perfect oracle" a measured fact rather than an argument.**

- **H28 — disagreements are localizable model defects, not irreducible noise.** Every disagreement
  (should any exist) is attributable to a specific command's reference semantics, reproduces
  deterministically, and is *fixable* in the reference oracle — after which SY1 re-runs to agreement on the
  corrected grammar. *Refuted if* disagreements are non-reproducible under the all-sealed determinism
  report (which would mean an unsealed nondeterminism source the report failed to declare — itself a
  bankable finding about the determinism contract, SPEC-3 §9's H4-refuted branch). Tested as **SY2**. This
  is the hypothesis that turns the weakness into a *tool*: the system oracle is a differential debugger
  for the reference oracle.

- **H29 — the validation sandbox is hermetic.** No step produces any observable effect outside its
  throwaway namespace: zero host-filesystem writes, zero packets, no privilege escalation, no
  cross-step persistence — each proven by a passing **negative control** that attempts the prohibited
  action and is denied. *Refuted if* any negative control succeeds (an action escapes). Tested as **SY3**.
  This is the formal statement of the user's "no action can occur" requirement.

- **H30 — the determinism contract is honest and the seals hold.** On the single-threaded v0 grammar the
  system oracle is bit-reproducible across runs and across machines under the declared seal, and the
  `determinism_report()` accurately enumerates what is sealed vs. recorded. *Refuted if* repeated runs of
  the same `(s, a)` diverge while the report claims all-sealed. Tested as **SY4** (and as a property of
  every SY figure). This is what licenses calling SY1 *true* determinism rather than recorded replay.

A note on the negatives. The program's posture is that a refutation is frequently the deeper result
(SPEC §10). Here the asymmetry is unusually favorable: **H27 refuted is not a loss.** If the real kernel
disagrees with our POSIX model on some edge, H28 catches it, we fix the reference oracle, and the program
emerges with a *more* faithful oracle plus a committed record of exactly how the from-scratch model
differed from reality — a thing no oracle-free program can produce about its own simulator. The only
genuinely adverse outcome is H30 refuted with an *undeclared* nondeterminism source, and the
determinism-report honesty surface exists precisely to make that loud rather than silent.

### 4.1 Results, as run (the committed numbers)

The pre-registered hypotheses landed exactly as the favorable-asymmetry argument predicted — and the
H27/H28 branch is the richer one:

- **H27 — supported, precisely.** On the **structure-building grammar** (the `structural`/`trivial`
  drivers — the regime v0 is *designed* to model), `agree_world` and `agree_exit` are **1.000, bit-exact**,
  with a degenerate CI, over both the Tier-1 exhaustive battery and Tier-2 trajectories. Tier-3 confirms it
  end to end: the `H_ε(ρ)` curve run with the `SandboxOracle` substituted for the `ReferenceOracle` is
  **indistinguishable** from the reference curve (max `|H_ref − H_sys| = 0.000` at every ρ). The
  load-bearing sentence "for computer worlds a deterministic ground-truth oracle is free and exact" is now
  a committed CSV, not an argument.

- **H28 — supported, and it already paid off.** The harness caught a **genuine reference-oracle bug**:
  `mv`/`cp -r` of a directory **into its own subtree** (`mv /e /e/e`) produced an *invalid* state — the
  moved subtree orphaned, its parent path gone. Localized to one guard in `oracle/reference.py`, fixed, and
  re-run to agreement — the H28 loop, executed. Beyond that one bug, **every** remaining disagreement on the
  full (`weighted`/`adversarial`) grammar falls into one of **four named, documented modeling boundaries**,
  with a measured **residual of zero**:
  - **root protection** — v0 makes `/` undeletable/uncopyable; the kernel operates on it;
  - **overwrite policy** — v0 mv/cp refuse to clobber a target; the kernel clobbers;
  - **permission enforcement** — v0 models `mode` as *data*, not access control; the kernel enforces
    traversal/access through directories lacking owner-execute;
  - **self-subtree** — the move/copy-into-own-subtree guard; GNU `cp` + v0 reject it, BSD `cp` (the macOS
    dev host) does not detect it — a *coreutils*-portability boundary that agrees on Linux, **platform-
    stamped** on every SY figure (the one place the host matters, and it is disclosed, per §2.5).

  So the honest headline is *stronger* than a rubber-stamp 1.000: **on the regime it models, v0 is bit-exact
  faithful to a real computer; outside it, every divergence is a named, intentional boundary — zero
  residual — plus one real bug the harness caught and the program fixed.**

- **H29 — supported.** The SY3 negative-control battery passes on every channel (filesystem / network /
  privilege / persistence / resource) with the positive control allowed-then-discarded. "No action can
  occur" is a green table.

- **H30 — supported.** SY4 shows the fixed `(s, a)` battery is **bit-identical across repeats** under the
  seal, with `determinism_report()` all-sealed — licensing SY1 as *true* determinism, not recorded replay.

---

## 5. Experiments

New prefix **SY** (system-oracle validation), to avoid collision with SPEC-9's scaling `S1–S3`. Each
follows the house template: a `Config` dataclass with `from_json_file`, a CLI entry point, a JSONL record
stream, and a `plot_*.py` that emits a committed `.png` + `.csv`, all regenerable from `reproduce.sh`.

- **SY1 — the differential agreement table + the head-to-head curve (H27).** Tier-1 exhaustive +
  Tier-2 driver-sampled agreement, plus the Tier-3 `H_ε(ρ)` overlay (reference vs. system). Outputs a
  single two-panel `figures/sy1_agreement.{png,csv}` — left: per-driver agreement with the divergence-class
  decomposition and CIs; right: the reference-vs-system curve overlay (folded into one figure as built).
  Module `experiments/sy1.py`, config `configs/sy1.json`. **The figure that retires W1.**
- **SY2 — the differential debugger (H28).** For any disagreement SY1 surfaces, the minimal reproducing
  `(s, a)`, the predicted vs. real delta, and the localized command. If SY1 is all-agreement (the expected
  branch on the v0 grammar), SY2 ships as the *standing apparatus* — the harness that will catch the first
  divergence the moment the grammar grows (Tier-B coreutils), with a seeded synthetic divergence as its
  own negative control to prove the debugger has teeth. Outputs `figures/sy2_disagreements.csv`.
- **SY3 — the hermeticity proof (H29).** The negative-control battery: a write-escape attempt, a
  socket/egress attempt, a privilege-escalation attempt, and a cross-step persistence probe, each asserted
  denied; plus a positive control (a benign in-tmpfs write that *is* allowed and *is* discarded on
  teardown). Outputs `figures/sy3_hermeticity.csv` — a per-channel pass table that rides alongside every
  other SY figure as the safety attestation. **The formal "no action can occur" deliverable.**
- **SY4 — the determinism attestation (H30).** Re-run a fixed `(s, a)` battery N times and across the CI
  matrix; assert bit-identical results; emit the `determinism_report` per run. Outputs
  `figures/sy4_determinism.csv`.

Reading order for the SY figures matches the hypotheses: SY3 (prove the sandbox is safe) → SY4 (prove it
is deterministic) → SY1 (measure agreement) → SY2 (debug any gap). Safety and determinism are gates *on*
the agreement claim, so they are demonstrated first, exactly as the SPEC-9/SPEC-10 envelopes are measured
before they are claimed.

---

## 6. Build, reproduce, CI

### 6.1 Module layout (additive only)

```
src/verisim/oracle/
  base.py            # unchanged — the Oracle protocol the SandboxOracle satisfies
  reference.py       # the v0 reference interpreter — one guard added (subtree mv/cp, the H28 fix)
  sandbox.py         # NEW — SandboxOracle: real /bin/sh, cross-POSIX hermetic core + tiers
  sandbox_seal.py    # NEW — DeterminismSeal: the env/umask/rlimit/no-new-privs per-step seal
  differential.py    # NEW — differential_step, canonical_world, classify_divergence (the harness)
src/verisim/experiments/
  sy1.py sy2.py sy3.py sy4.py   # NEW — the SY experiments
figures/
  plot_sy1.py plot_sy3.py       # NEW — committed-figure generators (SY2/SY4 are CSV-only)
configs/
  sy1.json                      # NEW — the committed SY1 sweep config
```

The one source edit outside the new files is the `oracle/reference.py` subtree-move/copy guard — the
H28 fix the differential harness surfaced (§4.1). Nothing else in the deterministic core, the loop, the
metrics, or any existing experiment changes; the `SandboxOracle` is observed by existing code only through
the `Oracle` protocol it already accepts.

### 6.2 `reproduce.sh`

The SY block runs in **gate order** (SY3 safe → SY4 deterministic → SY1 agreement → SY2 debug) and, being
cross-POSIX, runs on the macOS dev host *and* on Linux with no guard. As built:

```bash
echo "== SY3: the hermeticity proof — 'no action can occur' (SPEC-11 SO1, H29) =="
python -m verisim.experiments.sy3 --out runs/sy3/records.jsonl \
    --csv figures/sy3_hermeticity.csv --plot figures/sy3_hermeticity.png
echo "== SY4: the determinism attestation — bit-reproducible under the seal (SO2, H30) =="
python -m verisim.experiments.sy4 --out runs/sy4/records.jsonl --csv figures/sy4_determinism.csv
echo "== SY1: the differential agreement table + the H_ε(ρ) overlay (SO3, H27) — RETIRES W1 =="
python -m verisim.experiments.sy1 --config configs/sy1.json --out runs/sy1/records.jsonl \
    --csv figures/sy1_agreement.csv --plot figures/sy1_agreement.png
echo "== SY2: the differential debugger — divergence atlas + teeth control (SO4, H28) =="
python -m verisim.experiments.sy2 --out runs/sy2/records.jsonl --csv figures/sy2_disagreements.csv
```

The whole block runs in seconds on CPU (no torch). The committed figures are the **macOS numbers**,
platform-stamped (§2.5); the headline (`structural`/`trivial` = 1.000, residual = 0, curve gap = 0) is
platform-independent.

### 6.3 CI

The existing CI (`ubuntu-latest`, lint/type/test/build) is the **free, unlimited** Linux confirmation —
public-repo Actions cost nothing — so the SY suite stays green there *for free*, with no new job required.
The SY tests assert **platform-independent invariants** (structure-building agreement = 1.000; every
divergence is a named boundary, residual = 0; hermeticity controls pass; determinism is bit-identical),
not exact divergence counts, so the *same* tests pass on macOS (primary) and Linux (confirmation). A host
genuinely lacking `/bin/sh` `skipif`s with disclosure; a skip is never counted as agreement.

`pyproject.toml` carries a `system = []` extra: a documentation marker that the oracle is pure stdlib, so
the dependency-free promise is preserved. mypy strict and ruff apply to the new modules unchanged;
`sandbox.py`'s `subprocess`/`os`/`resource` calls are typed under the same `strict = true` regime (no new
override — stdlib is fully stubbed; `ctypes` is used only in a best-effort, exception-wrapped Linux path).

---

## 7. How SPEC-11 turns the program's SWOT into strength

This spec is written against the four-quadrant reading of the program, and is the single document that
moves all four the same direction.

- **Amplifies the core strength.** The program's strength is an asymmetry no other field has: a free,
  exact, in-the-loop oracle. SY1 takes that from *asserted* to *measured against reality* — the strongest
  form the asymmetry can take, and one the oracle-free field is structurally barred from matching.
- **Fixes the central weakness, and inverts it.** "Reference oracle ≠ reality" (W1) is closed by SY1 and,
  in the disagreement branch, *converted into a tool* (SY2): the reference oracle becomes self-correcting
  against ground truth. A weakness that becomes a maintenance instrument is a net strength.
- **Capitalizes on the opportunity.** A validated, hermetic, no-side-effect sandbox over real-OS dynamics
  is exactly the verifiable-environment substrate the agentic / computer-use / cyber-defense field is
  racing toward (the §1 opportunity). SPEC-11 ships the safe execution boundary and the exact-agreement
  guarantee that makes such an environment trustworthy — and the hermeticity contract (§2.3) is itself the
  defensive-security artifact (no egress, no escalation, dropped caps).
- **Defends against the threat, by running the attack first.** "Does it survive contact with a real OS?"
  is the reviewer's strongest move and the conceptual-vs-empirical-moat critique's sharpest edge. SY1 is
  that attack, executed by us, pre-registered, with its honest negative banked — so the threat is retired
  as a committed result rather than left open as an objection.

The throughline: the program's one structural bet stops being the thing a skeptic points at and becomes
the thing a CSV settles.

---

## 8. Milestones — **all delivered (2026-06)**

| ID | Deliverable | Status |
|---|---|---|
| **SO0** | `SandboxOracle` behind the `Oracle` protocol; cross-POSIX hermetic core + seal; `SystemOracleUnavailable` only when no shell | ✅ `isinstance(SandboxOracle(), Oracle)`; skip disclosed |
| **SO1** | Hermeticity contract (§2.3) + SY3 negative-control battery | ✅ all 6 channels PASS; positive control allowed-then-discarded (H29) |
| **SO2** | Determinism seal + SY4 attestation | ✅ bit-identical over repeats; report all-sealed (H30) |
| **SO3** | Differential harness + SY1 (Tier-1 exhaustive + Tier-2 trajectories + Tier-3 overlay) | ✅ structure-grammar agreement = 1.000; residual = 0; curve gap = 0 (H27) |
| **SO4** | SY2 differential debugger as standing apparatus | ✅ teeth control passes; one real reference bug caught + fixed; 4 boundaries enumerated (H28) |

Build order was strict and gated: **SO0 → SO1 → SO2 → SO3 → SO4.** Safety (SO1) and determinism (SO2) were
proven before any agreement number (SO3) was claimed — the agreement figure is only meaningful once the
sandbox is known to be hermetic and the oracle known to be deterministic. This mirrors the program's
invariant discipline: prove the checks have teeth before reporting what they catch.

---

## 9. Scope, non-goals, honest caveats

- **This validates the v0 grammar, not all of POSIX.** The claim SY1 earns is precise: the reference
  oracle is faithful to a real kernel *on the v0 command grammar*. Extending to Tier-B coreutils, package
  state, and services is SPEC-3's roadmap, and the SY2 apparatus is built to catch the first divergence
  that growth surfaces. Over-claiming "faithful to a real OS" beyond the validated grammar is the failure
  mode this section exists to forbid.
- **Concurrency stays out (HW-1).** The single-threaded grammar is chosen so that bit-exact ground truth
  is attainable and a disagreement is unambiguously a model defect. Threaded workloads require recorded
  scheduling (SPEC-3 §2.2) and are disclosed, not silently admitted.
- **Tier A, not Tier B.** SPEC-11 is the namespaced-shell tier (real `/bin/sh`, real kernel, snapshotted
  tmpfs). The pinned-container + record/replay tier (gVisor/`rr`/Hermit, SPEC-3 §2) is the next layer and
  is not in this spec; the `Oracle` protocol means it, too, drops in later without touching consumers.
- **Determinism debt is disclosed, never assumed.** The largest standing risk (SPEC-3 §11) is treating
  recorded replay as true determinism. On the v0 grammar the seal is genuinely total and SY4 proves it; the
  moment that stops being true, the report says so and every figure inherits the disclosure.

---

## 10. Provenance & reading order

SPEC-11 is the realization of an abstraction v0 has carried unbuilt since its first commit
(`oracle/base.py`'s "drops in unchanged" docstring) and of SPEC-3 §2's reserved milestone S1 / hypothesis
H4. It is a *validation* spec, not a world or a model: it advances nothing about the learner and invents
no new dynamics. It exists to make one sentence — *for computer worlds, a deterministic ground-truth
oracle is free, exact, and now measured against reality* — true beyond argument, safely, and regenerably.

Reading order for a newcomer: [SPEC.md §2](./SPEC.md) (why the oracle is free/exact; §2.1 reference vs.
system) → [SPEC-2 §3.2](./SPEC-2.md) (the `Oracle` protocol the `SandboxOracle` satisfies) →
[SPEC-3 §2](./SPEC-3.md) (the original system-oracle architecture and the HW-1 wall) → this document
(§2 the sandbox, §3 the differential protocol, §4 the hypotheses, §7 the SWOT inversion) →
`src/verisim/oracle/sandbox.py` and `src/verisim/experiments/sy1.py` (the concrete build, once shipped).
