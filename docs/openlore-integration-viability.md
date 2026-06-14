# Findings: Viability of a Verisim ↔ OpenLore Unified Verification & Deployment Engine

> Status: RESEARCH / VIABILITY ASSESSMENT — no code, no specs yet. This document decides
> *whether* and *in what form* the proposed integration is buildable before any OpenSpec
> change proposal is written.
> Author: architecture review, 2026-06-14.
> Method: direct inspection of both repositories on this machine (commit `5e34f06` for
> Verisim; OpenLore `package.json` version 2.0.18). Every claim below is grounded in a named
> file or a query against the live `call-graph.db`; see the Evidence Appendix.
> Scope assumption (per request): Verisim is treated as fully built per all shipped SPECs and
> the current `src/verisim` tree, not as aspirational.

## 0. One-paragraph bottom line

The proposed integration, *as literally specified*, is **not viable** — not because the
engineering is hard, but because four of its load-bearing premises are factually false against
what is on disk, and two of its actions are destructive. The two systems share vocabulary
("graph," "node," "edge," "epistemic," "oracle") but not meaning: OpenLore is a TypeScript
static-analysis tool whose graph is a **call graph of source code**, and Verisim is a Python
research simulator whose graph is a **simulated threat-model world** feeding a learned world
model M_θ. There is, however, a **smaller, coherent, on-mission integration** hiding inside the
request — a *one-way, read-only* grounding of Verisim's world models in OpenLore's real
call-graph, plus runtime tracing attached to Verisim's *real* execution surface (the
`SandboxOracle`), plus a *detect-and-halt* trajectory monitor, plus a *human-gated* headless
loop. Section 7 sketches that viable core. Sections 2–6 show the work behind the verdict.

---

## 1. The two systems as they actually are

A faithful integration starts from what each system *is*, not from a shared metaphor.

### 1.1 OpenLore (the static substrate)

- **Language / runtime:** TypeScript / Node.js ESM (`"type": "module"`, version 2.0.18 in
  `OpenLore/package.json`). Ships as an MCP server + CLI.
- **What its graph means:** a **static call graph of source code**. The canonical artifact is
  `.openlore/analysis/call-graph.db`, a SQLite database whose real tables are
  `nodes` (functions), `edges` (caller→callee call sites), `classes`, `inheritance_edges`,
  `decisions`, `decision_edges`, `cfg_overlay`, `provenance`, `change_coupling`, and FTS
  shadow tables. (Verified by `.schema`; see Appendix.)
- **Crucial property — the DB is a *cache*, not a store of record.** `.openlore/` is
  **gitignored** (`OpenLore/.gitignore:19`). The database is regenerated from source by the
  analyzer; it is not committed, not shared, and is rebuilt/overwritten on re-analysis.
- **`FunctionNode`, `CallEdge`, `Iac` are TypeScript types, not DB tables.** They appear in
  ~59 / ~24 / ~25 source files respectively, but only `nodes`/`edges` persist to SQLite as the
  call graph; there is **no `Iac` table** in `call-graph.db`.
- **"Epistemic Lease" is real but narrow.** It lives in
  `OpenLore/src/core/services/mcp-handlers/epistemic-lease.ts` and is, verbatim from its own
  header, *"session-level architectural confidence decay for MCP agents."* It models how stale
  **an agent's understanding of the source repo** has become, from signals like time since
  `orient`, git-hash divergence, tool "cognitive load," and module-switch density. Its states
  are `fresh` / `degraded` / `stale` (with stale depths 1–3). It already carries an
  `oscillation` field — but that field is *repeated bigram transitions among MCP tool/module
  accesses*, a developer-context-freshness signal, not a planning-state signal. There is **no
  `Stale [Critical]` tier**.
- **Reach into OpenLore is via MCP tools**, not a published cross-process write API. The
  `mcp__openlore__*` surface (`get_call_graph`, `find_path`, `analyze_impact`,
  `search_code`, `get_subgraph`, `list_decisions`, …) is the supported, stable way another
  process consumes the graph.

### 1.2 Verisim (the dynamic substrate)

- **Language / runtime:** Python 3.12 (`.venv`), torch-optional. **No SQLite anywhere** in
  `src/verisim` (verified by grep).
- **What its "graphs" mean:** featurizations of *simulated worlds*.
  [netmodel/graph.py](src/verisim/netmodel/graph.py) turns a `NetworkState` + `NetAction` into
  a `NetGraph` of host/link/flow nodes for a torch GNN — the graph arm of the learned world
  model M_θ. [landmark/graph.py](src/verisim/landmark/graph.py) is the SPEC-12 landmark planner
  over *reachability space*. These describe a security/threat world (hosts, ports, files,
  privileges, replicas), not source code.
- **The real-execution surface exists and is principled.**
  [oracle/sandbox.py](src/verisim/oracle/sandbox.py) is a `SandboxOracle` that runs a **real
  `/bin/sh` over a real kernel** in a throwaway tree under a `DeterminismSeal` (SPEC-11). This
  is the *only* place in Verisim where actual processes execute and actual syscalls happen.
- **"Planning simulation" means imagined rollout.** The loop's planner rolls M_θ *in
  imagination* (`loop/speculative.py`, `loop/runner.py`; the CU19/CU20 belief-rollout and
  teacher-forced arms). These are tensor computations predicting world deltas — there is no
  process being executed during a simulated rollout.
- **Headless entry points are thin.** `auto/search.py` is the closest thing to a driver;
  `pyproject.toml` defines **no `console_scripts`**. There is no existing
  "build → simulate → trace → commit → deploy" pipeline to extend.

The single most important sentence in this document: **Verisim's graph nodes and OpenLore's
graph nodes are not the same kind of object, and neither one is a duplicate of the other.**

---

## 2. Premise audit: Unified Graph Substrate

**Requested:** eliminate Verisim's duplicate graph storage; make Verisim read/write OpenLore's
canonical SQLite store; map host/filesystem/network assets onto `FunctionNode` / `CallEdge` /
`Iac` / `decision` nodes.

**Findings:**

1. **There is no duplicate storage to eliminate.** Verisim uses no SQLite and stores no call
   graph. Its graphs are in-memory world featurizations. Premise false.
2. **The mapping is a category error.** A simulated host, a `/passwd` file asset, or a network
   flow is not a function-definition node. There is no information-preserving function from
   Verisim's typed world state to OpenLore's `nodes`/`edges` rows. Forcing one would inject
   non-code rows into the call graph, and **every OpenLore consumer** (`find_dead_code`,
   `analyze_impact`, `get_subgraph`, reachability, `select_tests`) would then reason over
   garbage. It also discards Verisim's world semantics, which is the entire research asset.
3. **The target store is a regenerable cache, not canonical.** Because `.openlore/` is
   gitignored and rebuilt on analyze, any Verisim writes are non-durable and will be clobbered.
   "Canonical SQLite storage" does not describe this file.
4. **`Iac` is not a persisted primitive**, and `decision` nodes are an OpenLore-internal ADR
   construct, not a place to record simulated assets.

**Verdict: not viable as specified.** Salvageable core (Section 7.1): Verisim *consumes*
OpenLore's call graph **read-only** (via MCP tools or a read-only DB attach) to *ground* a
world model in a real codebase's structure, and writes any derived artifacts to **its own**
store. One-way, never mutating the cache.

---

## 3. Premise audit: Runtime Trace Oracle (dynamic-to-static sync)

**Requested:** extend Verisim's oracles into a dynamic instrumentation layer using eBPF /
dtrace / native profilers to intercept code execution **during planning simulations**, capture
dispatches / indirect calls / network bindings / DB mutations, and project them into the shared
DB as `runtime_verified` / `dynamically_called` edges.

**Findings:**

1. **The tracer is pointed at the wrong layer.** "Planning simulations" are M_θ tensor
   rollouts. Attaching eBPF/dtrace there traces the Python/numpy/torch interpreter, whose
   syscalls have **no correspondence** to the *simulated* world's "network bindings" or "DB
   mutations." You would capture the profiler observing the simulator, not the world.
2. **A real surface to trace does exist — but it is not the planner.** The `SandboxOracle`
   (Section 1.2) executes real shell commands and is the honest attach point for dynamic
   tracing. Tracing *there* yields a real syscall/exec stream tied to a known v0 action.
3. **Platform reality.** eBPF is Linux-only; this is a macOS-first machine
   ([memory: macos-first-testing]); dtrace on modern macOS requires disabling SIP. A spec that
   names eBPF as the default is dead on the development host.
4. **The edge kinds do not exist** (`runtime_verified` / `dynamically_called` are absent from
   the schema), and writing them back collides with Premises §2.2–2.3 (corrupting the call
   graph, into a cache that gets rebuilt). OpenLore *does* already have a sanctioned, on-mission
   way to add non-direct edges — the `synthesized_by` / `confidence='synthesized'` provenance
   on `edges`, used by its own "synthesized dynamic-dispatch edges" change. Any dynamic edges
   belong there, written *by OpenLore's analyzer contract*, not stamped in cross-process by a
   Python simulator.

**Verdict: not viable as specified.** Salvageable core (Section 7.2): a tracing oracle that
attaches to the **real `SandboxOracle` execution**, records the observed syscall/exec/bind
stream as a Verisim-owned artifact, and (optionally) *feeds a read-only suggestion* to
OpenLore through its supported synthesized-edge path — never a direct cross-process DB write.

---

## 4. Premise audit: Epistemic Circuit Breaker & State Recovery

**Requested:** extend OpenLore's Epistemic Lease inside Verisim's planning loop; compute an
"oscillation coefficient" over state transitions to detect infinite-debug/repetitive-edit
loops; intercept decay tiers "up to `Stale [Critical]`"; on breach, hard-freeze, drop the
speculative rollout, and **programmatically run Git/filesystem primitives to roll back the
workspace to the last stable baseline.**

**Findings:**

1. **Two unrelated mechanisms are being conflated.** OpenLore's Epistemic Lease tracks
   *staleness of an agent's understanding of source code*; it is TypeScript, session-scoped,
   and inside the MCP server. Verisim's planning loop is a Python rollout over a simulated
   world. The Lease exposes no cross-process API to "extend," and its `oscillation` signal
   measures tool-access bigrams, not world-state transitions.
2. **`Stale [Critical]` does not exist.** The tiers are `fresh`/`degraded`/`stale` (depths 1–3).
   A spec must not key safety behavior off a tier name that isn't in the system.
3. **The recovery action is destructive and heuristic-triggered.** "Programmatically roll back
   the workspace directory" means autonomous `git reset --hard` / file deletion fired by a
   threshold on a coefficient. That can destroy uncommitted work, and it conflates *simulated*
   state recovery (dropping a bad M_θ rollout — cheap, internal, safe) with *real filesystem*
   recovery (irreversible).

**Decision recorded:** per the requester, the posture is **human-gated everywhere** — the
breaker may *detect and halt* and *surface a recommended rollback*, but a human confirms before
any `git` reset, file deletion, commit, or deploy.

**Verdict: partially viable, reframed.** Salvageable core (Section 7.3): an **internal
trajectory-oscillation monitor** over Verisim's *own* planning-state transitions (this is a
genuinely sound idea and is conceptually adjacent to existing drift/horizon metrics). On breach
it **freezes the loop and drops the in-memory speculative rollout** (safe, reversible) and
**emits a recommended workspace-rollback for human confirmation** — it does not execute
destructive git/fs operations autonomously. The "lease" concept is *borrowed by analogy*, not
wired into OpenLore's TS lease object.

---

## 5. Premise audit: Headless Deployment Delivery Engine

**Requested:** a fully local, network-isolated headless loop: Intent Graph → Speculative Rollout
→ Trace Side-Effects → Evaluate Invariants → Intercept Oscillation → **Commit and Deliver to
the target environment using local system git credentials.**

**Findings:**

1. **The pipeline's first stages depend on §2–§4, which are not viable as written** (no shared
   substrate, mis-aimed tracer, conflated breaker). The deploy stage therefore inherits their
   defects.
2. **Auto-commit/deploy with the user's real git credentials, unattended, is outward-facing and
   irreversible.** It also fights Verisim's existing local commit gate ([memory: commit-gate]),
   which blocks commits by design.
3. **Verisim's mission is research instrumentation**, not a CD system. An autonomous deployer is
   a large scope departure with a poor risk/reward fit for this repo.

**Verdict: not viable as an autonomous default.** Salvageable core (Section 7.4): a headless
*evaluation* pipeline (build intent → simulate → trace real oracle → evaluate invariants →
oscillation check) that ends by producing a **signed report + a prepared, unpushed
commit/patch** for **human review and confirmation**. Delivery is a human action, with the loop
preparing everything up to (not through) the irreversible step.

---

## 6. Cross-cutting engineering realities

- **Cross-language coupling through a cache.** Even the read-only path crosses TS↔Python and
  targets a gitignored, regenerated file. Verisim must treat the DB as **read-only and
  ephemeral**, re-deriving on cache rebuild, and must prefer the MCP tool surface (stable
  contract) over raw SQL (schema is OpenLore-internal and versioned by `schema_version`).
- **"Zero-allocation cross-process file locks" is not a thing you can impose on SQLite.** SQLite
  owns its own locking (rollback-journal/WAL). Two writers is exactly what we are *avoiding* by
  making Verisim read-only. If concurrent read during OpenLore writes is a concern, the answer
  is WAL-mode read or a snapshot copy, not a hand-rolled lock.
- **"Do not use mocks; write the low-level primitives fully"** amplifies risk: it asks for real
  eBPF, real auto-`git reset`, and real auto-deploy. Under the human-gated posture, the
  low-level primitives we *do* build (sandbox tracing, snapshot-before-suggest) are real but
  bounded; the irreversible ones are replaced by prepare-and-confirm.
- **Determinism contract.** Anything written must respect Verisim's torch-free determinism
  discipline (SPEC-5 §13) and the sandbox hermeticity contract (SPEC-11 §2.3); a tracer must not
  perturb oracle determinism.

---

## 7. The viable core (what a later OpenSpec proposal could specify)

Each item is one-way, on-mission, and human-gated where it touches anything irreversible. These
are *sketches to decide scope*, not specs.

### 7.1 Read-only call-graph grounding
Verisim gains an **adapter that reads** OpenLore's call graph (via `mcp__openlore__*` tools, or
a read-only/WAL attach to `call-graph.db` re-derived on cache rebuild) and exposes it as an
input feature/landmark source for grounding a world model in a *real codebase's* structure.
Writes go to a Verisim-owned artifact only. Never mutates `.openlore/`.

### 7.2 Sandbox runtime-trace oracle
A tracing wrapper around the **`SandboxOracle`** (the real-execution surface) that captures the
real exec/syscall/bind stream of each oracle step as a typed, Verisim-owned trace record.
Platform-honest (native profiler / ptrace on macOS; eBPF only where available, never as the
default). Optionally emits a *read-only suggestion* of synthesized dynamic edges through
OpenLore's existing `synthesized_by` provenance path — produced for, not written into, OpenLore.

### 7.3 Trajectory-oscillation monitor (detect-and-halt)
An internal monitor over Verisim's **own** planning-state transitions computing an oscillation
metric (repeated state bigrams / repetitive edits), sitting alongside existing drift/horizon
metrics. On breach: **freeze the loop, drop the in-memory speculative rollout** (safe), and
**emit a recommended workspace-rollback for human confirmation**. No autonomous git/fs writes.

### 7.4 Human-gated headless evaluation pipeline
A headless entry point that runs intent → simulate → trace (7.2) → evaluate invariants →
oscillation check (7.3), then **produces a report plus a prepared, unpushed commit/patch** and
**stops for human confirmation** before any commit/push/deploy. Honors the existing commit gate.

---

## 8. Recommendation

1. **Do not author the four-part integration as originally specified.** It would encode
   category errors and destructive defaults into `openspec/changes/`.
2. **If we proceed, scope to Section 7** — and even there, treat 7.1 (read-only grounding) and
   7.2 (sandbox tracing) as the genuinely novel, on-mission pieces; treat 7.3/7.4 as
   conservative, human-gated safety scaffolding.
3. **Open questions for the requester** (resolve before any spec):
   - What is the actual goal the integration serves — grounding Verisim's world models in real
     code structure (research), or building a CD/verification product (engineering)? These imply
     very different specs.
   - Is read-only consumption of OpenLore's graph sufficient, or is a *bidirectional* sync a
     hard requirement? (If bidirectional, it must go through OpenLore's analyzer/synthesized-edge
     contract, not raw DB writes — and that is an OpenLore-side change, not a Verisim one.)
   - Is there any real codebase Verisim is meant to ground *on*, or is the "host/filesystem/
     network assets" language still referring to *simulated* worlds? (The answer determines
     whether 7.1 is even applicable.)

---

## 9. Resolution (2026-06-14) — scope locked for the prototype

The requester answered the Section 8 open questions, which converts this from "is it viable" to
"build the viable core." Decisions recorded:

1. **Goal:** a **working CD prototype to prove the research** (not a research-grounding toy and
   not a production CD system) — small, real, and built to scale.
2. **Data path:** **decoupled, with contract-mediated bidirectional sync.** Verisim consumes
   OpenLore's call graph **read-only**. When Verisim finds something the static graph is wrong
   about (a dynamic edge it never saw, or an invariant violation), it emits a **structured
   payload** that OpenLore ingests **through the `synthesized_by` provenance contract** and
   integrates *on its own terms*. Verisim never writes the DB directly.
3. **Subject:** a **real local codebase**, copied from `/Users/user/Documents/development/public/`
   into a fixture, with `.git` **de-fanged** so the prototype can never commit or push to the
   original. The simulated "host/filesystem/network assets" are now grounded in this real tree.

This produces the dependency chain specified as six OpenSpec changes under
[openspec/changes/](../openspec/changes/) (see that directory's `README.md`). Posture, per the
requester, is **human-gated everywhere** anything irreversible is touched. One honest caveat
carried into the specs: the **write side requires an OpenLore-side ingestion entry point that
does not exist yet** (synthesis is currently internal-only); the Verisim specs define the
payload + producer and flag the OpenLore counterpart as a cross-repo dependency.

## Evidence Appendix (commands run, 2026-06-14)

- **OpenLore DB schema** — `sqlite3 .openlore/analysis/call-graph.db .schema`: tables are
  `nodes, edges, classes, inheritance_edges, decisions, decision_edges, cfg_overlay,
  provenance, change_coupling, file_hashes, schema_version` + `nodes_fts*`. No `Iac`,
  no `runtime_verified`/`dynamically_called`. `edges` has `kind, call_type, synthesized_by`.
- **DB is gitignored** — `OpenLore/.gitignore:19` → `.openlore/`.
- **Primitive grep in OpenLore `src`** — `FunctionNode` 59 files, `CallEdge` 24, `Iac` 25,
  `EpistemicLease` 1, `oscillation` 3, `Stale [Critical]` 1, `context decay` 0,
  `runtime_verified` 0, `dynamically_called` 0.
- **Epistemic Lease definition** — `OpenLore/src/core/services/mcp-handlers/epistemic-lease.ts`
  header: "session-level architectural confidence decay for MCP agents"; states
  `fresh|degraded|stale`, stale depths 1–3; `oscillation` = repeated tool/module bigrams.
- **Verisim has no SQLite** — grep for `sqlite`/`call-graph` in `src/verisim`: empty.
- **Verisim graph semantics** — `src/verisim/netmodel/graph.py` (NetGraph for the GNN arm of
  M_θ), `src/verisim/landmark/graph.py` (SPEC-12 landmark planner over reachability space).
- **Verisim real-execution surface** — `src/verisim/oracle/sandbox.py` (`SandboxOracle`: real
  `/bin/sh` over a real kernel, throwaway tree, `DeterminismSeal`, SPEC-11).
- **Headless entry points** — `src/verisim/auto/search.py`; `pyproject.toml` defines no
  `console_scripts`.
- **OpenLore is TS/Node** — `OpenLore/package.json` `"type": "module"`, version 2.0.18.
