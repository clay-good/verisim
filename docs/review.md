# Adversarial self-review and prior-art pass

Date: 2026-06-16. Method: three independent hostile reviewers (novelty/prior-art, systems-security,
ML/empirical-rigor), each with no stake in the work, plus direct verification of the load-bearing
prior-art claims. This document records what they found, honestly and severity-ranked, including the
objections that hurt. It exists so the project does not publish on a footing it cannot defend.

## Bottom line

The work is substantially more derivative and less validated than the preprint currently claims. The
central "coverage theorem" is a renaming of a 1975 security principle. The live demos are dominated
by a correctly-configured sandbox. The headline empirical finding reproduces a result the paper
itself cites, at smaller scale, with contestable labels. The centerpiece head-to-head is a comparison
the authors parameterized to win. And at least one prominent claim (the SafePred self-disclaimer) is
an attribution I could not verify and should not have made.

There is a real, defensible kernel underneath, but it is modest, and it is not what the paper leads
with. Publishing as written would not survive review and would, fairly, read as overclaiming.

## Severity 1: claims that are wrong or unsupportable as written

**1a. The coverage theorem is complete mediation (Saltzer and Schroeder, 1975), renamed.** The
predicate `realizes(s,a) => target(s,a)` ("every action that could cause the harm is on the surface we
verify") is the complete-mediation requirement ("every access to every object must be checked"). The
"hard to game by any model" property is the model-independence of a sound execution monitor
(Schneider, *Enforceable Security Policies*, 2000). The reversibility routing (CU27: execute and roll
back reversible harms, gate irreversible ones) is the edit-automata model (Ligatti, Bauer, Walker,
2005). None of these are theorems we proved; they are established results we re-derived under new
names. Section 3 must present this as complete mediation specialized to a sparse, syntactically
computable surface with a free oracle, and cite the lineage, not claim a novel theorem.

**1b. "The only defense an adaptive adversary cannot break" is false.** Progent (arXiv 2504.11703,
2025) enforces agent tool-call policies "deterministically to guarantee security" with demonstrated
resilience to adaptive attacks on AgentDojo, ASB, and AgentPoison, without an exact oracle. GoEX
(arXiv 2404.06921, 2024) already proposed post-facto validation, undo, and damage confinement, the
reversibility story, two years earlier and uncited. The space of deterministic, model-independent
agent guardrails in 2024 to 2026 is crowded. "Only the exact oracle removes the worst case" is not
true.

**1c. The live demos are dominated by a sandbox, so they do not motivate the oracle.** Every RA6 and
RA7 harm (chmod a credential, write a credential to .env, delete a DB, open a firewall, wipe an audit
log, add an SSH key) is statically prevented by a hardened container plus a mandatory-access-control
profile (AppArmor or SELinux naming the protected paths), seccomp, dropped capabilities, read-only
mounts, and `chattr +i`/`+a`, at zero per-action cost. Worse, the kernel reference monitor mediates
the *resolved inode* after symlink resolution, so it is immune to the indirection (symlinks, built
paths) that the paper concedes defeats its own syntactic pre-commit target (Section 5). On the
demonstrated harms, "write an AppArmor profile and set `chattr +i`" dominates the oracle gate on
cost, on prevention timing, and on evasion resistance. The paper never runs this baseline. There is a
2026 paper, "Mandatory Access Control Framework" for LLM agents (arXiv 2601.11893), doing exactly
this.

**1d. The SafePred self-disclaimer is an unverified attribution (my error).** The paper states, as
the load-bearing differentiator, that SafePred's authors "note that safety depends on model accuracy,
and an adversary can exploit a prediction failure." I cannot verify this is in SafePred's actual text;
it came from a fetch-time summary, not a quote. It must be removed and replaced with a claim I can
defend: any guardrail built on a *learned* predictor gives a best-effort, not provable, guarantee.
Attributing a specific self-incriminating sentence to the foil to manufacture the gap is exactly the
kind of thing this review exists to catch.

## Severity 2: serious weaknesses in the evidence

**2a. RA7 reproduces a published result, at smaller scale, with a weaker design.** "Benign or
disguised instructions induce harmful actions that explicit ones do not" is the entire thesis of
OS-Blind (arXiv 2604.10577), which the paper cites: 300 human-crafted tasks, Claude 4.5 Sonnet at
73% attack success. RA7 is 6 scenarios, K=5, one model. The broader refusal-does-not-generalize
literature (past-tense reformulation, Andriushchenko and Flammarion, arXiv 2407.11969) established
the mechanism earlier and more rigorously. "Now measured" implies a novelty and a measurement
superiority RA7 does not have over OS-Blind.

**2b. The RA7 harm labels are contestable, and the headline lives in two cells.** The 10/30 disguised
rate is entirely two scenarios at 5/5: `perm_weaken` and `disclose`. The `disclose` harm (place a
provisioned DB password into a standard `.env`) is ordinary 12-factor configuration; calling it harm
is an author's normative choice, which we already flag in the code and RESULTS but state as fact in
the abstract. Strip the contestable cells and the gap mostly vanishes. The aggregate 0.33 is fully
determined by the scenario construction set, not a property of "disguised framing," so it should not
be reported as a rate.

**2c. The RA5 head-to-head is parameterized, not measured.** The learned-guardrail arm is a
fidelity-phi proxy (its adversarial-1.00 is arithmetic, defined in, not observed), the permission arm
is a deliberately incomplete denylist, and the oracle arm runs in the reference-oracle fixture it is
defined by. It is a consistency check on our own models of the alternatives, not evidence about
SafePred or a real permission system. The abstract and Section 4 present it as a result; Section 6
admits it is positioning. The abstract wins the reader, so the abstract is overclaiming.

**2d. The free exact oracle's scaling and prevent-vs-detect story are unproven.** "Diff the protected
region" assumes you already know the region (inheriting the enumeration problem we accuse the denylist
of), per-action full-state diff is O(region size), not O(1), and snapshot/rollback is not free for
DBs, network sends, or external calls. The central mechanism diffs *after* execution, so for
exfiltration (the scariest harm) it detects too late and the pre-commit fallback is the evadable
syntactic scan. The "~1 oracle call/task" and "9 to 15x cheaper" numbers come from the hermetic
fixture, not a real run.

## Severity 3: statistical rigor

K=5 per cell with no confidence intervals (a 5/5 cell's 95% Wilson interval is roughly [0.57, 1.0]),
a single model, a single prompt template, and an agent told it is an automated run and "do not ask
for confirmation," which suppresses the very pausing behavior that made `destruct` safe. The
qualitative existence proof ("some routine-looking tasks reliably induce the harmful state") survives;
the quantitative rate does not.

## What actually holds (the modest, defensible kernel)

- RA2's bit-identical state diff against a real `/bin/sh` is real and clean.
- The complete-mediation property is true; it is just not novel.
- One genuine observation, which two reviewers independently identified as the only defensible
  contribution: there is a domain (state-mutating filesystem and process harms) where the
  complete-mediation event set is *sparse, syntactically computable from the action, and verifiable by
  a free exact oracle*, so sound enforcement is cheap, where classically the monitored event set is
  large or the policy not enforceable. That is a real, if narrow, point.
- The untested but real opportunity: an effect-based oracle can express *relational* (a resource
  becoming reachable via multi-step config), *cumulative* (mass collection under individually allowed
  actions), and *context-dependent* danger that static MAC cannot encode. The CU arc actually did work
  here (segmentation reachability, cumulative collection, stale reads). This is the oracle's only
  un-dominated territory, and the RA demos do not test it.

## Prior art the paper must cite and position against (verified)

- Complete mediation / reference monitor: Saltzer and Schroeder 1975; Anderson 1972.
- Execution monitoring / enforceable policies: Schneider 2000; edit automata, Ligatti/Bauer/Walker
  2005.
- GoEX (arXiv 2404.06921, 2024): post-facto validation, undo, damage confinement.
- Progent (arXiv 2504.11703, 2025): deterministic privilege control, adaptive-attack resilient.
- Mandatory Access Control for LLM agents (arXiv 2601.11893, 2026).
- SafePred (arXiv 2602.01725, 2026): learned-world-model predictive guardrail (the foil; characterize
  it accurately, do not attribute unverified self-disclaimers).
- OS-Blind (arXiv 2604.10577): the benign-instruction-harm result at scale; and "When Benign Inputs
  Lead to Severe Harms" (arXiv 2602.08235).
- Refusal does not generalize across reformulation: Andriushchenko and Flammarion, arXiv 2407.11969.
- Information-flow control for agents (effect-based, prevention-timed): the stronger foil than a
  learned guardrail.

## What must change before this is publishable

1. Recast the contribution honestly: complete mediation made cheap in a domain with a sparse,
   syntactically computable surface and a free exact oracle. Drop the "theorem" and "nobody thought
   the domain allowed it" framing. Cite the lineage.
2. Add a real OS-confinement baseline (hardened container plus MAC plus `chattr`) to the head-to-head.
   Predict and report its missed-harm on the six scenarios. If it is 0/6, say so.
3. Demonstrate a harm a static policy cannot express (relational, cumulative, or context-dependent)
   and show the oracle catches it. This is the only experiment that establishes the oracle's reason to
   exist. The CU arc has the material; bring it into the live setting.
4. Run real baselines: actual SafePred (code is public) and a real MAC profile, on shared containers
   with a shared adversary. Until then, delete "the only defense an adaptive adversary cannot break."
5. Fix the empirics: external harm labels (or drop `disclose`), K >= 30, Wilson CIs, multiple prompts
   and models, remove the no-confirmation instruction, and stop reporting 0.33 as a rate.
6. Resolve prevent-vs-detect for exfiltration explicitly, and report real per-action diff cost at
   container scale.
7. Remove the unverified SafePred attribution.

## Verdict

Keep the honesty, keep RA2, keep the one real observation, and keep the untested relational/cumulative
opportunity, which is the most promising thing here. But the current paper leads with a renamed
classic, motivates it with demos a sandbox beats, supports it with a reproduced result and a rigged
comparison, and overclaims against real prior art. It is not ready to publish, and the gap between
what it claims and what it shows is exactly the gap this review was supposed to find. Better to know
it now.
