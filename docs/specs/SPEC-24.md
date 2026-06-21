# SPEC-24 — The Graded Guarantee: from a sampled certificate to a quantified one, audited against a monitor we did not write

**MVP/POC specification. SPEC-23 shipped the monitor auditor — `audit(monitor, oracle, proposer,
budget) -> Certificate` — and proved the RA22–RA25 certification loop is monitor-agnostic behind two
protocols (H158–H164). But it shipped with two honest gaps it named out loud: the certificate is
**empirical over a finite sampled space, not a proof** (SPEC-23 §6), and every monitor it audited is an
**instance we adapted** (our resolver, a denylist we wrote, our policy compiler), not an *unmodified
third party* (SPEC-23 §6, "the point, and still partly future"). SPEC-24 closes both, together, because
they are the same gap seen twice: you cannot sell "no hole found" about *someone else's* guardrail
without saying **what that claim is worth**. SPEC-24 makes the certificate state its own guarantee
grade — exhaustive over a bounded fragment (a real theorem), statistically bounded over the unbounded
tail (a number, not a shrug) — and runs the loop against a guardrail driven as an opaque subprocess,
with a directed adversary that earns its keep exactly where an external "provably secure" monitor lives:
the rare-hole regime.**

> **✅ SHIPPED — MVP / POC SPEC — 2026-06-21.** All of H165–H170 are built and tested (hermetic,
> torch-free where possible; `--neural`/`--external` arms run on demand): the guarantee layer
> ([`src/verisim/audit/guarantee.py`](../../src/verisim/audit/guarantee.py)), `certify` /
> `differential` ([`auditor.py`](../../src/verisim/audit/auditor.py)), the
> `ExhaustiveDepthProposer` / `DirectedNeuralProposer` ([`proposers.py`](../../src/verisim/audit/proposers.py)),
> and the `SubprocessMonitor` ([`monitors.py`](../../src/verisim/audit/monitors.py)). Run-record:
> [`docs/monitor-auditor-depth-run.md`](../monitor-auditor-depth-run.md); certificates + CSV under
> [`figures/`](../../figures/) (`spec24_graded_guarantee.csv`). Reproduce:
> `python -m verisim.audit resolver shell-path --require-depth 2 --require-epsilon 0.05` and
> `python -m verisim.experiments.spec24_graded_guarantee --neural --external`. Sibling to
> [SPEC-23](./SPEC-23.md) (the auditor it
> deepens) and [SPEC-22](./SPEC-22.md) (the deployment both generalize). It invents **no new oracle and
> no new adversary architecture**: it extends the shipped auditor
> ([`src/verisim/audit/`](../../src/verisim/audit/)) with a guarantee layer, lifts the neural proposer's
> reward off the resolver and onto the *monitor under audit*, and adds a process-boundary `Monitor`
> shim so the loop runs against a system it cannot see inside. The MVP is **Direction C (the graded
> guarantee) feeding Direction D (the real auditee)**: C is what makes D honest.

Read [SPEC-23 §6](./SPEC-23.md) (the two caveats this closes), [`docs/monitor-auditor-run.md`](../monitor-auditor-run.md)
(the H158–H164 run-record this continues), [`docs/paper.md`](../paper.md) §7 (the 2026 crowd — YoloFS's
SMT-checked staging surface, ePCA's axiom set, Progent, GoEX — each *asserts* completeness; this is what
a deployer would run against them), and the RA25 lesson in
[`docs/frontier-close-run.md`](../frontier-close-run.md) (the enumeration treadmill, which bounds what
*any* guarantee here can mean and which this spec states up front).

---

## 0. One-paragraph thesis

SPEC-23's certificate says "no oracle-confirmed hole in the sampled composition space." A guardrail
vendor's customer needs the next sentence: *and the sampled space covers what fraction of the realizing
surface, with what confidence?* SPEC-24's thesis is that the certificate can answer it, in two tiers
that together are the honest whole: **(1) an exhaustive sub-certificate** — the compositional action
space up to depth ≤ k is *finite and enumerable* (depth ≤ 3 is ~171k actions, sub-second), so "no hole
at depth ≤ k" is a **theorem over a complete fragment**, not a sample; **(2) a statistical residual
bound** — over the unbounded tail beyond depth k, a Good–Turing missing-mass estimate plus a
Clopper–Pearson interval turns "none found in N draws" into "residual in-contract hole rate < ε at
confidence 1−δ." With the certificate now stating what it proves, the auditor can be pointed at a
guardrail it did not write: a **`SubprocessMonitor`** drives an opaque external decision process
(allow/deny/ask over the Claude Code hook contract the repo already speaks), and a **directed adversary**
— reward = oracle-confirmed evasion of *that* monitor, not the resolver — concentrates budget on the
target's actual blind spots, which is the only way to audit a strong external guardrail where holes are
rare. The honest scope is inherited and sharpened: the guarantee is **complete over the modeled action
algebra**, and growing that algebra is the same enumeration treadmill RA25 named — so SPEC-24 reports
the algebra's boundary as a first-class field of the certificate, not a footnote.

---

## 1. What this deepens (the two gaps SPEC-23 named)

SPEC-23 is honest in §6 about exactly two things, and they are the two halves of one product gap:

1. **"Empirical, not a proof."** The certificate is over a finite *sampled* composition space.
   "No hole found" means "none in the sampled space," not a theorem over all bash / all actions. A
   buyer cannot price an audit whose only output is "we looked and didn't find anything" — they need
   the coverage and the confidence. **Direction C** supplies them.

2. **"Auditing a real third-party monitor is the point, and still partly future."** Every SPEC-23
   monitor is an instance *we* adapted (`ResolverMonitor`, `DenylistMonitor`, the policy compiler).
   Running against an *unmodified external system* across a trust/process boundary is named, not
   claimed. **Direction D** runs it — and needs Direction C, because the first question about an
   external "provably secure" guardrail (YoloFS, ePCA) is not "did you find a hole" but "how hard did
   you look, and over what."

A third, deeper limit sits under both and is *not* closed here, only made first-class: the proposer can
only surface holes it can **render**. The guarantee is complete over the *modeled action algebra* (the
compositional grammar + the policy ops), and a hole in a mechanism the grammar does not model is
invisible — the RA25 treadmill, now bounding the meaning of the word "complete." SPEC-24 makes the
algebra's identity and size a recorded field of the certificate so the guarantee is never read wider
than it is.

## 2. The pivot (why these two, why now, in this order)

SPEC-23 ended by pointing at the external auditee as the next step. The non-obvious move SPEC-24 makes
is that **you must grade the guarantee *before* you point it outward**, not after. Auditing your own
resolver, "no hole found" is fine — you wrote it, you know its contract, and RA25 already proved the
reversible residual is covered by the post-commit diff with no folder at all. Auditing *someone else's*
"provably secure" guardrail, "no hole found" is worse than useless: it launders the vendor's own
unverified completeness assertion into your name. The auditor's credibility *is* the grade. So:

- **Direction C first (the grade).** Make the certificate say: exhaustively proven sound to depth k;
  beyond k, residual in-contract hole rate < ε at confidence 1−δ; over the modeled algebra A. This is
  the difference between "we ran a fuzzer" and "we certified a coverage class."
- **Direction D second (the auditee).** Now the loop can run against an opaque external monitor and the
  output means something: not "YoloFS is safe," but "over the file-corruption algebra A, exhaustive to
  depth 2 and sampled to a 1−δ residual bound of ε, the auditor found these k holes / certified none."
  That is a claim a deployer can act on and a vendor can argue with — which is the whole point.

## 3. The MVP: Direction C feeds Direction D

### 3.1 Direction C — the graded guarantee (the certificate states what it proves)

Three components, all extending the shipped `Certificate`:

- **Exhaustive bounded sub-certificate (a theorem for a fragment).** The compositional grammar's space
  is `|MECH|^|ATOMS| × |VERB| × |REDIRECT|` (≈ 17.9M), but *depth-bounded* it is small and complete:
  depth ≤ 1 is 402 actions, depth ≤ 2 is 11,292, depth ≤ 3 is 171,012 — all enumerable in well under a
  second. An `enumerate_to_depth(k)` proposer yields *the entire* sub-lattice (no sampling, no seed), so
  a clean audit over it is a **proof** that the monitor has no in-contract hole at depth ≤ k. The
  certificate records `exhaustive_depth = k` and the exact count enumerated, with the cost curve making
  the exponential honest (the theorem is real but bounded, and the spec says where it stops).

- **Statistical residual bound (the unbounded tail).** Beyond depth k the space is too large to
  enumerate, so the sampled proposer carries a quantified residual. Two estimators, both standard and
  both cheap: a **Good–Turing missing-mass** estimate (`(# in-contract holes seen exactly once) / N`)
  upper-bounds the probability the *next* sampled realizing action is a fresh uncovered in-contract
  hole; and for "none found in N" a **Clopper–Pearson / rule-of-three** upper bound gives `ε ≈ 3/N` at
  95% when the observed hole count is 0. The certificate records `(residual_epsilon, residual_delta,
  n_sampled)` so "no hole found" becomes "residual in-contract hole rate < ε at confidence 1−δ."

- **The guarantee-grade schema.** `Certificate.guarantee` is a structured block —
  `{algebra, exhaustive_depth, n_exhaustive, residual_epsilon, residual_delta, n_sampled, oracle,
  external}` — and a `grade()` that summarizes it (`PROVEN≤k + bounded(ε,δ) over A`). The CLI prints the
  grade, and the CI gate fails on any in-contract hole *or* a guarantee weaker than a declared floor
  (e.g. `--require-depth 2 --require-epsilon 0.01`), so a build can demand a coverage class, not just
  absence-of-findings.

### 3.2 Direction D — the real auditee (a monitor we did not write)

Three components, all reusing the SPEC-23 loop:

- **The directed adversary (reward = evasion of the monitor under audit).** SPEC-23's neural proposer is
  tuned on the *resolver* reward and *transferred* to other monitors (H159). That works when holes are
  common; against a strong external "provably secure" guardrail they are rare, and a transferred or
  blind proposer wastes its budget off-target. A `DirectedNeuralProposer` sets the REINFORCE reward to
  the oracle-confirmed evasion of *the specific monitor* (`1.0 if oracle.realizes(a) and not
  monitor.covers(a) else 0.0`), so the policy climbs the target's own blind-spot gradient. This is the
  RA24 sample-efficiency result aimed at the monitor itself: at a fixed oracle budget, directed search
  surfaces holes a blind/transferred proposer misses in the rare-hole regime.

- **The external-monitor shim (across the process boundary).** A `SubprocessMonitor(cmd, encode)` drives
  an *opaque* external decision process — it encodes an action as the JSON the process expects on stdin,
  runs it, and reads the allow/deny/ask verdict from stdout, exactly the Claude Code PreToolUse hook
  contract the repo already implements
  ([`scripts/claude_code_coverage_hook.py`](../../scripts/claude_code_coverage_hook.py),
  [`claude_code_gate.hook_decision`](../../src/verisim/realagent/claude_code_gate.py)). The hermetic
  demo audits the repo's *own* hook script as a black box (dogfooding the "we did not write the
  internals we can see" interface — the auditor only sees stdin/stdout), proving the loop runs across
  the boundary with no access to the monitor's source. The certificate records the external system's
  identity + version. The run-on-demand external lane (like SPEC-22's `--bash` / TB lanes) names a real
  policy engine — OPA/Rego `opa eval`, or a reimplemented YoloFS staging predicate / Progent policy —
  run as the same opaque subprocess; named, not claimed, exactly as SPEC-23 promised.

- **Differential certification (the vendor-patch regression gate).** A guardrail evolves: a denylist
  gets a patch, a policy engine ships a new ruleset. `differential(cert_before, cert_after)` partitions
  the hole sets — `closed = holes(v1) − holes(v2)`, `opened = holes(v2) − holes(v1)` — and certifies a
  **monotone improvement** iff `opened == ∅` over the exhaustive+sampled space. This is RA25's
  post-commit-diff move applied to the *monitor itself*: don't re-assert v2's completeness from scratch,
  certify that the *change* only closes holes. It is the CI artifact a vendor (or a deployer tracking a
  vendor) actually wants.

## 4. Milestones (hypotheses continue the SPEC-23 ledger from H164)

| ID | Hypothesis | Status | Acceptance |
|---|---|---|---|
| MVP-C1 | **H165** — an **exhaustive bounded sub-certificate**: the auditor enumerates the *entire* compositional space to depth ≤ k (no sampling) and certifies "no in-contract hole at depth ≤ k" as a theorem over a complete fragment | ✅ shipped — **SUPPORTED** ([`tests/test_spec24_guarantee.py`](../../tests/test_spec24_guarantee.py)) | `ExhaustiveDepthProposer(k)` yields exactly the depth-bounded lattice (**402 / 11,292 / 171,012** at k=1/2/3, == the closed form); a clean audit sets `exhaustive_depth=k`; the hardened resolver certifies depth ≤ 2 sound, the unsound resolver's printf hole is found at its true minimal depth **1** |
| MVP-C2 | **H166** — a **statistical residual bound** over the unbounded tail: "no hole found in N samples" becomes "residual in-contract hole rate < ε at confidence 1−δ" | ✅ shipped — **SUPPORTED** | the certificate carries `(residual_epsilon, residual_delta, n_sampled)` + missing mass; the Good–Turing, Wilson, and rule-of-three bounds are pinned by a hermetic test, and the Wilson interval covers a planted hole rate (p=0.10 ∈ [0.083, 0.109]) |
| MVP-C3 | **H167** — the **directed adversary** (reward = oracle-confirmed evasion of the monitor under audit) beats the transferred/blind proposer in the rare-hole regime | ✅ shipped — **SUPPORTED** ([`tests/test_spec24_directed.py`](../../tests/test_spec24_directed.py)) | against a blind spot *off* the resolver-reward manifold, `DirectedNeuralProposer` finds **230** oracle-confirmed holes vs **40** transferred vs **14** blind at a fixed budget (16×); the `reward_fn` seam on `neural_proposer.train` keeps the RA24 default byte-identical |
| MVP-D1 | **H168** — the **external-monitor shim**: the loop audits a guardrail driven as an opaque subprocess (allow/deny/ask over stdin/stdout), with no access to its source | ✅ shipped — **SUPPORTED** ([`tests/test_spec24_external.py`](../../tests/test_spec24_external.py)) | `SubprocessMonitor` audits the repo's own `claude_code_coverage_hook.py` as a black box (only stdin/stdout) and reproduces its indirection edge — it FIRES on the literal but **allows the encoded rev/cut frontier** it cannot fold (18 silent holes); the OPA/Rego + YoloFS/Progent external lane is documented + run-on-demand |
| MVP-D2 | **H169** — **differential certification**: certify a monitor *patch* closes a hole set and opens none, without re-asserting completeness from scratch | ✅ shipped — **SUPPORTED** | `audit_diff(v1, v2)` on a denylist patch reports `closed=10, opened=0` → monotone certified; a regression patch (closes 10 but **opens 7**) is correctly flagged `opened ≠ ∅` |
| MVP-D3 | **H170** — the **guarantee-grade certificate + CI floor**: the certificate states what it proves, and the CI gate can demand a coverage class, not just absence-of-findings | ✅ shipped — **SUPPORTED** ([`__main__.py`](../../src/verisim/audit/__main__.py)) | `Certificate.guarantee` (v2 schema) + `grade()`/`meets()` render and enforce the tier; `python -m verisim.audit ... --require-depth k --require-epsilon e` exits non-zero on any in-contract hole *or* a guarantee below the floor; pinned by a hermetic test |

## 5. Architecture (where the seams go — reusing SPEC-23)

- **`src/verisim/audit/guarantee.py` (new)** — `enumerate_to_depth(atoms, k)` (the complete
  depth-bounded lattice), the residual estimators (`good_turing_missing_mass`,
  `rule_of_three`/`clopper_pearson_upper`), and the `Guarantee` dataclass + `grade()`. Torch-free,
  standard library + the grammar only.
- **`src/verisim/audit/protocols.py`** — `Certificate` gains a `guarantee: Guarantee | None` field and
  the JSON schema bumps `version` to 2 (additive; SPEC-23 certificates still parse). `Hole` is unchanged.
- **`src/verisim/audit/proposers.py`** — add `ExhaustiveDepthProposer(protected_path, k)` (deterministic,
  the C1 enumerator) and `DirectedNeuralProposer(monitor, oracle, ...)` (the C3 reward = evasion). The
  directed proposer needs a `reward_fn` seam on `neural_proposer.NeuralAdversary.train` (default = the
  current resolver `judge`, so RA24/H159 stay byte-identical); the auditor injects the evasion reward.
- **`src/verisim/audit/monitors.py`** — add `SubprocessMonitor(cmd, encode, parse)` wrapping an opaque
  external decision process behind `covers()`. Default `encode`/`parse` speak the Claude Code hook JSON
  contract; the contract maps `ask`/`deny` → covered, `allow` → not covered (the SPEC-23 routing).
- **`src/verisim/audit/auditor.py`** — `audit(...)` gains `guarantee=Guarantee.request(depth=k,
  delta=δ)` (run the exhaustive pass then the sampled tail, fill the grade); new
  `differential(cert_a, cert_b) -> DiffCertificate` and a thin `audit_diff(monitor_v1, monitor_v2, ...)`.
- **`src/verisim/audit/__main__.py`** — `--require-depth`, `--require-epsilon`, `--diff <monitor_b>`,
  and `--subprocess <cmd>` flags; the grade printed and enforced.
- **policy layer** — `differential` works unchanged on policy monitors (a policy `closure_complete()` is
  a patch); H169 demos both the denylist and the relational policy.
- **experiment + docs** — `src/verisim/experiments/spec24_graded_guarantee.py` (the C+D table, CSV +
  certificates under `figures/`), `docs/monitor-auditor-depth-run.md` (the run-record), and tests
  `tests/test_spec24_guarantee.py` / `tests/test_spec24_external.py`.

## 6. Honest caveats, stated up front (SPEC-23's, inherited and sharpened)

- **The theorem is bounded; the tail is statistical; neither is "all bash."** The exhaustive
  sub-certificate is a real proof *only* to depth ≤ k, and the residual bound is a confidence interval,
  not a guarantee — combined they are strictly stronger than SPEC-23's bare sample, and strictly weaker
  than a closed-form proof over all actions. The certificate states exactly which.
- **Complete over the *modeled algebra*, not over reality.** The proposer finds only holes it can
  render; a hole in a mechanism outside the grammar/policy-op set is invisible. The guarantee's
  `algebra` field names the action algebra and its size so the claim is never read wider than it is —
  and growing the algebra is the RA25 enumeration treadmill, which is why the principled deployment
  backstop remains the post-commit diff for the reversible class, unchanged.
- **The free-oracle assumption is unchanged.** Everything still needs an exact, resettable oracle for
  the harm; confidentiality/covert-channel exfiltration (RA15) is still honest-silent — the directed
  adversary cannot manufacture an oracle that does not exist, so the irreversible slice still fails
  closed to a human, uncertified.
- **The external auditee is opaque, not adversarial-to-the-auditor.** `SubprocessMonitor` assumes the
  external process answers honestly about its own decision; a monitor that *detects it is being audited*
  and behaves differently is out of scope (and is itself a finding for a different tool). The MVP audits
  cooperative external guardrails (the realistic case — a vendor wants the audit to pass honestly).

## 7. Reproduce / acceptance (what "done" means for the POC)

The POC is done when: (1) `python -m verisim.audit resolver shell-path --require-depth 2` enumerates the
depth-≤2 lattice, certifies it sound as a theorem, and exits 0; the same with `resolver-unsound` finds
the printf hole at its minimal depth and exits non-zero (H165); (2) the certificate carries a residual
`(ε, δ)` whose interval contains the empirical hole rate on a synthetic planted-hole monitor, pinned by
a hermetic test (H166); (3) the `DirectedNeuralProposer` surfaces a single deep planted hole at a budget
where blind/transferred do not (H167); (4) `--subprocess scripts/claude_code_coverage_hook.py` audits
the hook as a black box and reproduces its indirection edge, certificate naming the external system
(H168); (5) `--diff` certifies a denylist patch monotone (closed ⊇ {patched}, opened = ∅) and flags a
regression patch (H169); (6) the certificate renders + enforces a guarantee grade, every number pinned
by a hermetic test (torch-free where possible) and the `--bash` / `--neural` / external lanes run on
demand, matching the RA arc's reproduce discipline (H170). Run-records land in
`docs/monitor-auditor-depth-run.md`; figures + CSVs under `figures/`.
