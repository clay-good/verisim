# The graded guarantee: from a sampled certificate to a quantified one, audited against a guardrail we did not write (SPEC-24, 2026-06-21)

> SPEC-23 shipped the monitor auditor and named two honest gaps: the certificate is **empirical, not
> a proof**, and every monitor it audited was **one we adapted**, not a third party. They are the same
> gap seen twice — you cannot sell "no hole found" about someone else's guardrail without saying what
> that claim is worth. SPEC-24 closes both: the certificate now **states its own guarantee grade**, and
> the loop runs against a guardrail driven as an opaque subprocess, with a directed adversary that earns
> its keep exactly where a strong external monitor lives (the rare-hole regime).

## What SPEC-24 adds

- **[`src/verisim/audit/guarantee.py`](../src/verisim/audit/guarantee.py)** — the `Guarantee` block,
  the exact `depth_bounded_count` closed form, and the residual estimators (`rule_of_three_upper`,
  `wilson_upper`/`wilson_lower`, `good_turing_missing_mass`). Torch-free, standard library only.
- **`certify(...)`** ([`auditor.py`](../src/verisim/audit/auditor.py)) — runs an exhaustive pass over
  the complete depth-≤k lattice (a theorem) plus a sampled tail (a residual bound) and fills
  `Certificate.guarantee` (the schema bumps to v2; SPEC-23 certificates still parse).
- **`ExhaustiveDepthProposer`** + **`DirectedNeuralProposer`** ([`proposers.py`](../src/verisim/audit/proposers.py))
  — the complete depth-bounded enumerator, and the adversary whose REINFORCE reward is oracle-confirmed
  evasion of the *monitor under audit* (a `reward_fn` seam added to `neural_proposer.train`, default =
  byte-identical RA24 resolver reward).
- **`SubprocessMonitor`** ([`monitors.py`](../src/verisim/audit/monitors.py)) — an opaque external
  guardrail driven across the process boundary over the Claude Code hook contract.
- **`differential(...)` / `audit_diff(...)`** — certify a monitor *patch* is a monotone improvement.
- CLI flags: `--require-depth`, `--require-epsilon` (certify + coverage floor), `--diff <monitor>`,
  `--subprocess "<cmd>"` with the `external` monitor.

## The result (hermetic; seeded; `--neural`/`--external` arms run on demand)

Run records are the graded certificates under [`figures/`](../figures/) (`spec24_graded_guarantee.csv`
+ per-run JSON).

### Direction C — the certificate states what it proves (H165/H166/H170)

The compositional space is ~17.9M overall but *depth-bounded* it is finite and complete — **402 /
11,292 / 171,012** actions at depth ≤ 1/2/3 (the enumerator matches the closed form exactly), so a
clean audit over it is a **theorem for that fragment**, not a sample:

| monitor | guarantee grade | silent holes | min hole depth |
|---|---|---|---|
| resolver (hardened) | `PROVEN≤depth2 (n=10,956) + residual<0.0030 @ conf 0.95 (n=1006)` | **0** | — (sound) |
| resolver (unsound) | same exhaustive tier; residual<0.10 | **877** | **1** (printf at depth 1) |

The CLI enforces a coverage floor: `python -m verisim.audit resolver shell-path --require-depth 2
--require-epsilon 0.05` exits 0 (theorem proven, residual under floor); the unsound resolver exits
non-zero on its in-contract hole. "No hole found" is now "proven sound to depth 2, residual in-contract
hole rate < ε at 95% over the modeled algebra" — a coverage class, not a shrug.

### Direction D — the real auditee (H167/H168/H169)

- **The directed adversary (H167).** Against a blind spot placed *off* the resolver-reward manifold (a
  conjunction of FIRES-class encodings the resolver-tuned policy actively avoids), the directed proposer
  finds **230** oracle-confirmed holes vs **40** for the SPEC-23 transferred (resolver-tuned) proposer
  and **14** for blind uniform — a 16× margin. Tuning a proposer on monitor X does not transfer to
  monitor Y's blind spots; rewarding evasion of Y directly does.
- **The external auditee (H168).** `SubprocessMonitor` audits the repo's own
  [`claude_code_coverage_hook.py`](../scripts/claude_code_coverage_hook.py) as a **black box** (only
  stdin/stdout), finding **18** silent holes — the rev/cut/printf_fmt frontier the hook *allows* because
  it ABSTAINs on the unfoldable pipe and routes the reversible action through (relying on a post-commit
  diff the standalone audit cannot see). It FIRES on the literal. This is the loop running against a
  guardrail with no access to its source; the OPA/Rego + YoloFS/Progent external lane is named and
  run-on-demand.
- **Differential certification (H169).** A denylist patch (add the `${` var-split class) is certified
  **monotone** (closed 10, opened 0); a regression patch (add `${` but drop `rm `) is correctly flagged
  **opened 7**. RA25's post-commit-diff move applied to the monitor itself — the vendor-patch CI gate.

## The honest takeaway (sharpened from SPEC-23)

The theorem is real but **bounded to depth ≤ k**; the tail is a **confidence interval**, not a
guarantee; together they are strictly stronger than SPEC-23's bare sample and strictly weaker than a
closed-form proof over all bash — and the certificate states exactly which. Most important, the
guarantee is **complete over the *modeled action algebra*** (recorded in the certificate's `algebra`
field, `compositional_shell(MECH=12,ATOMS=6,VERBS=6)`), never over reality: a hole in a mechanism the
grammar does not render is invisible, and growing the algebra is the RA25 enumeration treadmill — which
is why the post-commit diff remains the principled deployment backstop for the reversible class. The
free-oracle assumption is unchanged; confidentiality/covert-channel exfiltration (RA15) is still
honest-silent — the directed adversary cannot manufacture an oracle that does not exist.

## Reproduce

```
python -m verisim.audit resolver shell-path --require-depth 2 --require-epsilon 0.05   # certify a theorem
python -m verisim.audit denylist shell-path --diff resolver                            # patch monotone
python -m verisim.audit external shell-path --subprocess "python3 scripts/claude_code_coverage_hook.py"
python -m verisim.experiments.spec24_graded_guarantee --neural --external              # the full table
python -m pytest tests/test_spec24_guarantee.py tests/test_spec24_directed.py tests/test_spec24_external.py -q
```
