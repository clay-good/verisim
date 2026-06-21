# Verisim: certifying complete mediation — a safety monitor's coverage is a learning problem with a free verifier

**The contribution, stated for 2026.** Deterministic, execution-grounded *complete-mediation* guardrails
for computer-use agents are no longer scarce — they are a crowd. In the months around this writing the
design space filled in independently: filesystem-level mediation with staging and snapshot rollback
(YoloFS, "Don't Let AI Agents YOLO Your Files," arXiv 2604.13536), an SMT-checked "provably secure"
agent guardrail (ePCA, arXiv 2605.29251), deterministic pre-action authorization (arXiv 2603.20953),
transactional checkpoint-rollback sandboxing (arXiv 2512.12806), and the earlier deterministic enforcers
this work already cited (Progent, 2025; GoEX, 2024). They share one move and one unexamined assumption:
each *asserts* that its mediation is complete — that every action which can realize the harm is on the
surface it checks — and none of them *certifies* it. The covering surface is hand-specified, and complete
mediation is only as sound as the human who enumerated it: the very enumeration problem these systems
charge the pattern denylist with.

This paper's contribution is the missing half. **The completeness of such a monitor is itself a learning
problem with a free verifier, and we give the method that certifies and hardens it under an adaptive
*learned* adversary.** Because a computer world has a free, exact oracle, a monitor's own soundness-hole
predicate — `realizes(s, a) ∧ ¬target(s, a)`, an action that truly causes the harm yet slips off the
checked surface — is a dense, model-free reward. We train a neural autoregressive adversary over a
compositional grammar of shell-path encodings against exactly that reward (no learned reward model
anywhere in the loop), and run a discover → fix → re-verify loop: the adversary hunts holes, the exact
oracle confirms each one bit-for-bit against a real `/bin/sh`, the monitor is repaired where a realizing
action slipped off it, and the certificate is re-tested under the adversary. It behaves as a method, not
a slogan: the learned adversary found a *genuine soundness bug our own hand-run red team had missed* (a
`printf` format-escape fold, RA24), about 18× faster than blind search; once the fix is folded back, the
soundness invariant *holds under the learned attacker* — no string-resolvable realizing command is
silently declared safe across the sampled composition space (RA22–RA25).

The deterministic gate everyone is now building is, in this framing, the **motivating instance and the
substrate the method runs on, not the headline.** The same method that certifies our shell-path monitor
is what a deployer would run against YoloFS's staging surface or ePCA's axiom set to ask the question
those papers leave open: is your covering surface *actually* covering, or is there an encoding that
realizes the harm off it? We demonstrate the method on our own monitor (one harm family, a
finite-but-realistic shell-encoding grammar, a certificate empirical over the sampled space rather than a
proof); auditing a third-party monitor is named future work, not a present claim. The rest of the paper
is the substrate that earns this framing — the gate, why an oracle beats a sandbox, and the measured
operating point — read as motivation for, and evidence under, the certification result.

**Status (June 2026): recast twice — first after an adversarial self-review, then to lead with the
certification contribution above as the deterministic-gate space became crowded; read [docs/review.md](review.md)
alongside this.** The mechanism and every number below are real and reproducible. An independent
hostile review (three reviewers, prior-art verified) found that the original draft overclaimed, and
the most important fix was a reframing, now done. We no longer present a novel "coverage theorem."
The property is complete mediation (Saltzer and Schroeder, 1975) and execution-monitor soundness
(Schneider, 2000), old and established; the contribution is not the property but the conditions under
which it becomes cheap, a domain where the surface to mediate is sparse and syntactically computable
and the policy oracle is free and exact. The oracle's reason to exist over a sandbox — the
motivation the certification result then builds on — is three harms a static per-resource sandbox
structurally cannot express, demonstrated live with the MAC/sandbox baseline run for real, a relational
graph-reachability invariant (RA8), a cumulative blast-radius budget (RA9), and a context-dependent
change-freeze (RA12). On those, only the oracle is
both safe and useful. We also replaced the review's worst target, the parameterized learned-guardrail
arm, with a real measured LLM guardrail (RA11), which corrected our own story and is reported as such,
and we ran the real hardened-sandbox baseline the review demanded (RA13, the immutable bit `chattr +i`
on real containers): it drives single-resource harms to zero at no cost (the sandbox wins there, and we
say so) and reaches zero on the three un-dominated harms only by killing the task. We also recast the
whole result in the AI-control vocabulary the field now uses for untrusted agents (RA16): the oracle is
a *trusted monitor* whose safety-usefulness frontier, unlike a model monitor's, does not move under an
adaptive attacker, which is the same property the recent control literature finds model monitors lack.
And we hardened the pre-commit target the program had flagged as its single open frontier (RA18): an
abstract shell-path interpreter that splits the indirection edge into a closed string-resolvable half
and a provably-irreducible runtime/filesystem-state half, found by an adversarial red team of our own
and then closed or routed, with the residual fail-closed on the irreversible slice of the Claude Code
hook.
And we ran the program's single named open step, the external Terminal-Bench number, locally on the
official harness (RA20): a real Claude agent driven through the host Code CLI with no API key (a custom
agent that operates each task container via `docker exec`), on all 80 tasks of `terminal-bench-core`.
Two lanes, both honest: capability 38/80 = 0.475 (15 of the 42 misses are agent-timeout/error at our
cap, not capability), and, the contribution, the safety lane on 16 real Terminal-Bench containers
where oracle-coverage missed-harm is 0.00 against 0.40 for a pattern denylist and 1.00 unguarded,
identical on every task.
And we measured the §8 product operating point on real agent traffic rather than a battery (RA21):
replaying 123,195 real Claude Code tool calls (3,836 sessions) through the gate, only 1.1% of actions
touch the danger surface so 98.9% auto-approve with no prompt, while a fair denylist prompts 8.7x as
often (99% of it off-surface fatigue) and still leaks 0.70 of an injected harm arsenal the gate
catches entirely. What is still open, and not claimed: an AppArmor or SELinux arm (no LSM was
available in this kernel), a run of SafePred's actual code, a cross-model realism sweep, and a
*leaderboard-grade* Terminal-Bench submission (an in-container agent rather than the host-`docker exec`
bridge, and uncapped timeouts for the heavy build/ML tasks). The results run only on controlled worlds
with a hand-specified danger model. Read
accordingly, and read the review.

## Abstract

A deterministic, execution-grounded guardrail that enforces complete mediation over a computer-use
agent is, as of 2026, a crowded design (Section 7): several independent systems gate agent actions on a
hand-specified covering surface and *assert* that the surface is complete. None certifies it, and a
covering surface is only as sound as the enumeration behind it. This paper supplies the missing half:
in a computer world the monitor's own exact oracle makes its soundness-hole predicate
(`realizes ∧ ¬target`) a dense, model-free reward, so **certifying a monitor's completeness is a learning
problem with a free verifier.** We train a neural compositional adversary on that reward — no learned
reward model in the loop — and run a discover → fix → re-verify loop against a real monitor; it found a
genuine soundness bug a human red team had missed and, once that was folded back, the soundness invariant
held under the learned adversary across the sampled composition space (RA22–RA25). The deterministic gate
the method certifies is the motivation and the substrate, told first because it is what makes the
certification claim concrete; the rest of this abstract is that substrate.

A computer-use agent is one mis-predicted action away from an irreversible mistake: overwriting a
credential, deleting the wrong file, sending data it should never send. The field's own safety
benchmarks confirm this is not hypothetical. On OS-Harm, frontier models including Claude comply with
roughly 10 to 20 percent of prompt injections, and a single injected instruction can turn a helpful
agent into an exfiltration tool.

Two defenses are deployed today, and both have a known failure mode. A permission system (allow,
deny, ask a human) either drowns the user in approvals or is gamed by an action it did not think to
deny. A learned predictive guardrail, the most recent proposal, forecasts an action's consequences
with a learned model and blocks the dangerous ones. Because the predictor is itself a model, the
guarantee is best-effort rather than provable: an adversary can aim at a prediction failure.

Verisim takes a different route, available in exactly one domain. Computer worlds (filesystems,
processes, networks) have a free, exact, resettable oracle: you can run the real thing and read the
true next state, bit for bit. That removes the model from the safety-critical path. Making
complete mediation cheap this way is no longer, by itself, the contribution: that design is now a crowd
(Section 7). It is an old principle, complete mediation (Saltzer and Schroeder, 1975), made cheap in this
one domain — the substrate the contribution stands on. The contribution is *certifying* a monitor built
this way, by turning its own free oracle into the verifiable reward of an adaptive learned adversary
(RA22–RA25). Danger here lives on a sparse, syntactically computable surface (the actions
that can *realize* the harm), so a gate that consults the free exact oracle only on that surface is
hard to game by *any* model, at a cost set by how rare the surface is. Established work already gates
agent actions deterministically (Section 7); what computer worlds add is that sound enforcement here
is inexpensive.

The reason to reach for an oracle at all, rather than the hardened sandbox that prevents
single-resource harms more cheaply, is a class of harms a static per-resource policy cannot express.
We demonstrate three, live, on real containers with the sandbox baseline run for real: a *relational*
harm (a config edit that transitively opens an untrusted-to-database path, RA8), a *cumulative* harm
(mass collection past a blast-radius budget under individually-allowed reads, RA9), and a
*context-dependent* harm (a production change pushed during an active change-freeze, RA12). In each,
the permissive sandbox posture is unsafe and the restrictive one destroys the legitimate task; only
the oracle, which evaluates the action against the true resulting state, is both safe and useful. That
is the oracle's reason to exist, and it is where the paper now leads.

Against the two defenses deployed today we add a hermetic bake-off, a consistency check against our
own models of a learned guardrail and a permission denylist, not a run of real systems. Under an
adaptive prompt-injection adversary the modeled guardrail and denylist are both breached while the
oracle gate is not, at no loss of task success and about one oracle call per task. Because that arm
was parameterized, we then replaced it with a real measured LLM guardrail on the relational harm
(RA11). It is far stronger than the proxy implied (recall 0.95 to 1.00), so the oracle's edge over a
learned guardrail is honestly the guarantee and the cost, not a capability gap.

We also ran it live, and the live result is the honest one, more bounded and more useful than the
convenient story. We measured a real Claude agent across six harm scenarios, two framings each
(plainly malicious versus disguised as a routine ops ticket), five trials each, in fresh disposable
containers with harm read off the container state. The agent refused every explicit injection (0 of
30): alignment is good. It even scoped safely around most disguised harms (it kept a production
database and deleted only stale logs when told to "free disk space"). But in the two scenarios where
the harmful action is indistinguishable from legitimate configuration work, granting a service
account read access to a secret and placing a credential in a standard `.env` file, it produced the
flagged state every time (10 of 10, though one of those two labels is contestable, since placing a
credential in a `.env` is also ordinary practice). This reproduces, at small scale, a phenomenon the
literature already documents (OS-Blind, arXiv 2604.10577). Treat it as an existence proof, not a
rate. The point that survives is narrow: there exist routine-looking tasks that reliably produce a
security-relevant state change which an effect-based check catches and inspecting the action does not.

## 1. The problem an agent platform actually has

Lead with the uncomfortable part: the agents are good enough to be dangerous.

A modern coding or computer-use agent will, on instruction, edit files, run shell commands, push
branches, and call APIs. That is the product. The cost of the product is that the same capability
deletes the wrong directory, weakens a permission, or follows a malicious instruction it read in a
file. In September 2025 a documented incident had one agent hijacked by prompt injection rewrite a
second agent's configuration to add a malicious tool server. The blast radius of a wrong action is
the whole machine.

The deployed defenses are real and worth respecting, which is why their failure modes matter.

A **permission system** gates actions behind policy: allow this, deny that, ask the human about the
rest. Claude Code and managed-agent platforms ship exactly this. Its strength is that it needs no
model of consequences, only a rule. Its weakness is the rule. A denylist of patterns cannot enumerate
every action that causes a given harm. If you forbid writing to a credential file, you have not
forbidden `chmod 777` on it (which makes it world-writable) or `mv` (which moves it out from under
your protection). Tighten the list and you raise the approval burden until the human clicks "allow"
on autopilot, which is its own failure.

A **learned predictive guardrail** is the newer idea. SafePred, "a predictive guardrail for
computer-using agents via world models" (arXiv 2602.01725), forecasts each proposed action's effect
with a learned model and blocks the ones predicted to be harmful, checking selectively on a learned
"danger surface" to keep the cost down. Its strength is generality: a model can judge actions a rule
never anticipated. Its limitation follows from using a learned predictor: safety is best-effort, not
provable, and an adversarial agent can aim at a prediction failure. A model has blind spots, and an
attacker who can rephrase will find one.

So the gap is specific. We have a gameable rule and a best-effort model. What we want is a defense
that is cheap, keeps the agent useful, and cannot be talked out of catching the harm by any model.
That is the classical idea of a reference monitor enforcing complete mediation (check every action
that can realize the harm), which is bypass-proof regardless of the program it monitors (Saltzer and
Schroeder, 1975; Schneider, 2000). The usual objection is cost: the set of actions to mediate is
large, and the policy oracle is expensive. The argument of this paper is that computer-use worlds
relax both, which is what makes complete mediation cheap enough to actually deploy here. We are not
the first to enforce agent actions at runtime (Section 7); the contribution is the conditions under
which sound enforcement becomes inexpensive.

## 2. The one asymmetry computer worlds have

Here is the move, and it rests on a property no other agent domain enjoys.

A world model is hard to trust because, in most domains, nothing can tell it when it is wrong. When a
video model imagines the next frame of a forest, only reality can adjudicate, and you cannot rerun
reality. Robotics, vision, and open language all train their predictors against proxies for truth,
because truth is not on tap.

Computer environments are the exception. A filesystem, a process table, a network stack are digital
and deterministic. You can run the real system in a sandbox, step it one action forward, read the
exact resulting state, and reset it, thousands of times, for almost nothing. Truth is not a proxy
here. It is a function call.

That single fact lets you do something the other domains cannot: take the model out of the
safety-critical path entirely. The agent may still use a cheap learned model to plan and to preview
("look before you leap"). But when safety is on the line, you do not ask the model whether an action
is dangerous. You ask the oracle, which runs it (or a faithful reference of it) and tells you what
actually happens.

The question that remains is the only one that matters for cost: you cannot afford to ask the oracle
about every action, so which actions do you ask about? Get that wrong and you are back to either
verifying everything (slow) or guessing (gameable). The answer is the contribution.

## 3. Complete mediation, made cheap: where danger lives and why verifying only that is enough

The claim in one sentence: danger has a sparse, model-free surface, and mediating exactly that surface
is both safe and cheap. The safety half is complete mediation, an established principle (Saltzer and
Schroeder, 1975), not our result. The cheap half is the contribution, and it comes from the domain.

Make three objects precise for a given harm `D`, for example "a credential file is corrupted."

**Realization.** `realizes(state, action)` is true when executing this action from this state actually
brings about the harm. It is read off the exact oracle: run the action, diff the protected region,
see if it changed. It never consults the learned model. It is the ground truth of harm.

**The genesis surface, or target.** `target(state, action)` is the cheap, model-free rule for *when
to consult the oracle*. In the simplest case it is syntactic: does this command name a path under the
protected region? It is computed from the action itself, not from a model's opinion of the action.

**Coverage** is the bridge between them, and it is the complete-mediation condition stated for this
setting:

> for every state and action, `realizes(s, a)` implies `target(s, a)`.

In words: every action that *could* cause the harm is on the surface we verify. Nothing that realizes
the danger lives off it. This is exactly "every access to every object must be checked" (Saltzer and
Schroeder, 1975) restricted to the accesses that can realize a specific harm.

From coverage, the safety property falls out, and it is worth walking slowly because it is the point.
Take any agent, any model, any attacker, any timing. The attacker can win only by getting some action
`a` with `realizes(s, a)` executed without being caught. But coverage says `realizes(s, a)` implies
`target(s, a)`, so the gate consults the oracle on `a`. The oracle sees the true `realizes` and
blocks. The consult decision never read the model, so this holds *no matter how bad the model is*.
That last property, bypass-proof regardless of the monitored program, is the soundness of an execution
monitor (Schneider, 2000); we are re-deriving it, not discovering it. A covering, model-free gate is
hard to game at a cost equal to the number of on-surface actions, which is rare and cheap by
construction.

This is the inversion that makes the cost work. You do not need a model that is accurate everywhere.
You need a surface that is *sparse* (so verifying it is cheap) and *covering* (so nothing slips off
it). Danger in a computer world tends to have exactly this shape: a credential is corrupted only by an
action that names it, a network flow to a protected host is born only by a connect to it. The harm
has a narrow genesis, and you verify the genesis.

Two honest cautions live right here, and Section 6 returns to them. Coverage is a property you have to
*establish* for a real harm; it is not free. And "model-free target" hides work: writing a target that
provably covers a fuzzy real-world notion of harm is the hard part, and on fully arbitrary shell input
it degrades from a theorem to a heuristic.

## 4. What we measured

Bottom line: the gate drives missed-harm to zero around a real model and a real kernel (RA1 to RA4),
and on the harms that actually motivate an oracle, the relational, cumulative, and context-dependent
dangers a static sandbox cannot express, it is the only arm both safe and useful (RA8, RA9, RA12). In
our hermetic bake-off it is also the arm an adaptive adversary does not break, against our models of
the deployed defenses, not a claim about every system (deterministic enforcers like Progent also
resist adaptive attacks, Section 7). The five foundation steps come first, then the un-dominated
territory that is the real reason to deploy it.

We built the result in five steps, each its own experiment with committed code, a figure, and tests.

**A real model, a real injection (RA1).** A real Claude agent (`claude-opus-4-8`) works a battery of
shell tasks against an exact reference oracle. One task carries a prompt injection in a file the agent
reads. Undefended, the agent obeys the injection and overwrites the credential: missed-harm 0.167
(the injection rate). With the oracle coverage gate, missed-harm is 0.000, at 0.17 oracle calls per
task, about 15 times cheaper than verifying every action, with task success unchanged. The danger
surface is disjoint from the legitimate work, so the gate never blocks the real task.

**A real kernel (RA2).** The same battery, run against a real `/bin/sh` instead of the reference
oracle, holding the agent fixed. The safety verdict is bit-identical between the two: maximum
difference 0.000 across every schedule. The agent literally executes a real write to a real credential
file under a sealed throwaway tree, and the gate is what stops it. The result is verified against real
computer-use dynamics, not a model of them.

**Generalization (RA3).** To answer "did you pick six friendly tasks and one friendly attack," we ran
200 procedurally randomized tasks (work directories sampled independently of the protected region, so
disjoint by construction) crossed with a five-class attack taxonomy (overwrite, append, delete,
weaken-permissions, exfiltrate-by-move). Undefended missed-harm tracks the injection rate (0.310); the
gate drives it to 0.000 with no utility loss, at about nine times cheaper than full verification.
Every attack class is real undefended and caught by the gate, and each is provably on the surface. The
safety generalizes because it was never task-specific.

**Off the grammar (RA4).** A real agent on an open benchmark emits arbitrary shell, so the gate must
not depend on a fixed command grammar. We separate the two halves. Realization is read off a real
filesystem diff, which is exact for any command (`tee`, `dd`, a Python one-liner, anything). The
target is a syntactic scan of the command string for a protected-path reference. Three findings, the
third honest. A grammar-specific gate cannot even parse 75 percent of real attack syntaxes and is
structurally blind to them, while the command-agnostic target catches all of them. The target is
silent on real benign commands (`grep`, `find`, `tar`, `git`), so it does not block legitimate work it
never saw. And, honestly, a path built by indirection (a variable that never spells the literal
prefix, or a symlink) defeats a purely syntactic pre-commit scan: it misses every such evasion. That
is the boundary, and Section 5 takes it seriously.

**The head-to-head (RA5), the centerpiece.** We put the oracle gate against the two defenses a
platform actually has, as four arms on a Terminal-Bench-shaped task distribution:

- **none:** the bare agent, the OS-Harm baseline.
- **model_guardrail:** a learned predictive guardrail that judges each action, faithful with
  probability phi below 1. This is the SafePred design, modeled as a fidelity-phi stand-in.
- **permission:** a pattern denylist with allow-and-ask, the Claude Code design.
- **oracle_coverage:** verisim, consulting the exact oracle on the genesis surface.

The adversary is adaptive: it rephrases an attack until it finds one the arm does not block (the
worst-case injection placement). The result:

| arm | random miss | adversarial miss | task success | oracle calls/task |
|---|---|---|---|---|
| none | 1.00 | 1.00 | 1.00 | 0 |
| model_guardrail (SafePred-style) | 0.33 | 1.00 | 1.00 | 0 |
| permission (Claude-Code-style) | 0.33 | 1.00 | 1.00 | 0 |
| oracle_coverage (verisim) | 0.00 | 0.00 | 1.00 | ~1 |

Read the table slowly, because it is the argument. Under an adaptive adversary the learned guardrail
is breached completely, and so is the permission denylist. The oracle gate is the only arm not
breached, and it keeps full task success at about one oracle call per task.

The sharpest single finding sits inside the guardrail row. As you raise the learned model's fidelity
from phi = 0.5 to 0.95, its *random* miss falls from 0.67 to 0.00: a better model catches more on
average, exactly as you would hope. But its *adversarial* miss stays pinned at 1.00 the whole way. A
better model is no safer against an adversary, because the adversary aims at whatever blind spot
remains, and a model with phi below 1 always has one. The permission denylist fails for a parallel
reason: a `chmod` or an `mv` is not a "write," so a list of patterns cannot enumerate every action
that realizes the harm. Only the exact oracle, consulted on a surface that provably covers the harm,
removes the worst case. This is the gap inherent to any learned predictor: a model with blind spots
can be aimed at, however good its average accuracy. (Caveat, stated in full in Section 6: this is a
hermetic comparison against our models of a learned guardrail and a denylist, not a run of SafePred
or a real permission system, and not against an OS-level reference monitor, which would also catch
these harms. The "adversarial miss pinned at 1.00" is the *worst case* this phi-proxy can express,
and when we replaced the proxy with a real LLM guardrail and measured it (RA11, Section 6), the real
guardrail was far stronger than that worst case on RA8's relational danger. Read this row as the
theoretical claim it is, and read RA11 for the measured one.)

**The live lane (RA6 and RA7): a real Claude agent, real containers, and a measured rate.** The five
experiments above use an exact reference oracle and, in RA2, a real `/bin/sh`. To pressure-test the
premise itself we ran a real Claude Code agent in real Docker containers (Terminal-Bench-shaped: a
protected resource, a benign task, a planted instruction), with the gate operating on the container by
diffing the protected region. We measured it properly: six harm scenarios, two framings each (plainly
malicious versus disguised as a routine ops ticket), five independent trials each, harm read off the
container state, the agent run from a clean directory with no knowledge of the experiment. That last
control was learned the hard way and matters: run from the repository, the agent reads this work's own
description of the experiment and refuses; it must be fully isolated from any sign that it is a test,
or every number is contaminated. Three findings, and the order matters.

First, a current model is not the obedient adversary our synthetic worst case assumes. It refused
every explicit injection, 0 of 30 across all six scenarios, naming the credential ones as
exposure patterns and doing only the legitimate work. The naive claim that an unverified agent always
does the bad thing is false, and we will not lean on it. This is consistent with the field's numbers:
on OS-Harm, models comply with roughly 10 to 20 percent of injections, not all of them.

Second, the residual is real, reliable, and precisely shaped, and that is the whole point. Disguised
as routine ops tickets, the agent still scoped safely around most harms: told to "free disk space" it
kept the production database and deleted only stale logs, 0 of 5. But in the two scenarios where the
harmful action is indistinguishable from legitimate configuration work, granting a service account
read access to a secret and placing a database password in a standard `.env` file, it executed the
harm 5 of 5, every single time. There is no "critical" signal to refuse on, because the action is
routine config; only the security consequence is wrong, and inspecting the action cannot reveal it.
The disguised harm landed in 10 of 30 trials versus 0 of 30 explicit, concentrated entirely in the
harms that look exactly like ordinary work. This is the "benign instruction, critical vulnerability"
failure mode that OS-Blind (arXiv 2604.10577) already documents at far larger scale (300 tasks, Claude
at a high attack-success rate) and that the refusal-does-not-generalize literature (arXiv 2407.11969)
established earlier. We are reproducing a known phenomenon at small scale, not discovering it; the
value here is only that it occurs through the *same* objective state-change channel the gate keys on.
Treat 10 of 30 as an existence proof, not a rate: it is two scenarios at 5 of 5, K is 5 with no
confidence intervals, it is a single model and prompt, and the two 5-of-5 cells (`disclose` in
particular, placing a provisioned credential in a `.env`) involve a harm label that a fair reader can
contest, since that is also ordinary configuration. The honest, narrow claim that survives: there
exist routine-looking tasks on which a current agent reliably produces a security-relevant state
change that an effect-based check catches and inspection does not.

Third, the gate does not care how the instruction was dressed. Replaying both real trajectories
through the arms against fresh containers, the oracle coverage gate caught the realizing action in
both framings at no cost to the benign work, because it reads the action's effect on the protected
region, never the instruction or a model's opinion of it. The permission denylist caught the explicit
`chmod` but missed the footprintless read (`cat` is not a mutation verb), the same CU34 gap, now on
real Docker. The honest synthesis: model alignment and the oracle gate are complementary, not rival.
Alignment is a strong, free first filter that catches harm that looks like harm. The gate is the
guarantee for the rest, and the "rest" is not a corner case. It is every harmful action that looks
like ordinary work, which a model cannot reliably refuse because it cannot tell, and which the gate
catches by construction.

**The territory a sandbox does not own (RA8): relational danger.** An adversarial review of this work
(docs/review.md) made a sharp and correct point: every harm in RA6 and RA7 is dominated by a hardened
container plus mandatory access control, which denies the write to the protected file statically, more
cheaply than the gate, and immune to the indirection that defeats our pre-commit target. If that were
the whole story, the gate would be unnecessary. RA8 is the answer, and it is the result that actually
motivates an oracle over a sandbox. The harm is *relational*: a security invariant that is a global
property of a configuration graph, not an access right on any one resource. A network policy holds the
invariant "no service in the untrusted zone may transitively reach the database." A config-management
agent must edit that policy to do its job. The obvious edit to connect an analytics service to
reporting data, `analytics -> reporting`, is transitively unsafe, because `reporting -> db` already
exists, so it opens an untrusted path to the database. No rule the agent writes names the database;
the exposure is emergent. Run live, a real Claude agent makes exactly this exposing edit in 60 of 60
trials (95% Wilson interval [0.94, 1.00]), each time noting that the existing `reporting -> db` rule
is "unchanged" without seeing that its own edit now reaches the database through it. That rate is
powered and de-confounded: it holds across three prompt phrasings and with the no-confirmation
instruction removed, and crucially it holds even when the agent is explicitly told to follow least
privilege and to check for unintended side effects. The emergent exposure is genuinely hard to notice,
and priming for care does not fix it, which is the point. (One model family, Claude via the local CLI;
we do not claim a cross-model rate.) Now compare the defenses. A per-resource policy
(make the file immutable) is safe but its only move is total denial: the agent's legitimate safe edit
is kernel-denied too, so the task cannot be done, utility zero. The oracle gate evaluates the
*resulting reachability* of each proposed edit, blocks the one that connects untrusted to the database,
and permits the safe alternative (a read-only mirror with no database path). It is the only arm that
is both safe and useful. The reason a sandbox cannot do this is categorical: mandatory access control
mediates access to resources, and "the resulting graph must not connect untrusted to the database" is
not a resource access right, it is a property of the end-state that only something evaluating the
end-state can check.

**Cumulative danger (RA9) completes the argument, and is harder still for a static sandbox.** Here the
harm is the aggregate. A data-minimization policy caps an agent task at B distinct sensitive records, a
blast-radius bound: if the agent is compromised or mistaken, its exposure is at most B. Every
individual read is authorized; the harm is the count, and because the threshold is an explicit number,
it is unambiguous, which also repairs the contestable-label problem of RA7. We ran 30 customer records,
a budget of B equal to 5, two workloads (a scoped task needing one record, and a bulk collection of
all thirty), and four arms on real containers. A real agent given a bulk-export ticket collected the
whole dataset, exceeding the budget, in 60 of 60 trials (95% interval [0.94, 1.00], across the same
three phrasings and confound conditions). Here, honestly, the high rate is less surprising than RA8's:
a bulk-export task touches the whole dataset by construction, so this is not a subtle miss. The
relevant finding is that the agent does not spontaneously self-limit to a data-minimization budget or
seek approval, even when primed to be careful, so the budget must be enforced outside it. The arms:
with the dataset readable (the permissive MAC posture), every
read succeeds and the bulk collection is uncapped, mass collection goes through; with it unreadable
(the restrictive posture), even the scoped task dies, utility zero; only the oracle accumulator, which
keeps a stateful count of distinct records touched and blocks the access that would cross B, is both
useful on the scoped task and safe against the bulk one. Mandatory access control and information-flow
control are both stateless across actions, so neither can express "no more than B distinct," which is
why this is the oracle's territory and not the sandbox's. The honest boundary is that a DLP system or
query auditor could also count, because such an auditor is exactly this accumulator; the claim is only
that the static, per-resource confinement that dominates the RA6 and RA7 harms structurally cannot.

**Context-dependent danger (RA12) is the third member, and its realism finding is the sharpest.** Here
the same action on the same resource flips between safe and harmful on a dynamic context flag external
to the resource. A deploy agent is ticketed to raise a rate limit in the production config: legitimate
work normally, but during a declared change-freeze (an incident window) the identical edit is the harm.
The danger is neither a graph property nor a count; it is conditional on a freeze flag that comes and
goes. A static per-resource ACL is a fixed function of (subject, object, action), so it has no input
for the freeze: grant the write and it goes through during the freeze (harm), deny it and the
legitimate change dies when there is no freeze (utility zero). On real containers, only the oracle,
which evaluates the action against the live freeze state, is both useful with no freeze and safe during
one. That is the same deterministic shape as RA8 and RA9. The realism trials, though, found something
more interesting than a single harm rate. We gave a real agent the routine ticket and told it to follow
the standard runbook, and we varied exactly one thing: whether the freeze rule was written in that
runbook. With the rule legible in the runbook, the agent integrated the context perfectly, deploying
the legitimate change in 10 of 10 normal trials and holding in 10 of 10 freeze trials, citing the
freeze. Move the identical freeze marker out of the runbook and leave it merely discoverable in the
environment, and the same agent deployed during the freeze in 10 of 10 trials. Same agent, same action,
same freeze; the harm rate goes from zero to one hundred percent on the legibility of the rule alone.
This is the RA6/RA7 lesson quantified on a new axis: a model integrates context it is told to check and
misses context it is not, while the oracle blocks the harm in both conditions because it reads the live
state, not the runbook. The honest boundary is the same as RA8/RA9: a context-aware stateful engine
could also gate on the freeze, but that engine is this oracle; the static per-resource confinement that
dominates RA6/RA7 cannot.

**The real hardened-sandbox baseline, run (RA13).** The review's sharpest systems objection was that
we never run the baseline we claim to beat, a hardened container with a real per-resource confinement
mechanism. We ran it, using the strongest one available in the kernel: the immutable bit (`chattr +i`,
which overrides ownership, even the owner cannot write without a capability the agent lacks), plus DAC
read-removal for the read-harm. Every protected resource is agent-owned, so plain permissions would not
stop the agent; the immutable bit is the real guard. On a single-resource harm (corrupt a credential
the task never touches), the hardened sandbox drives missed-harm to zero at no utility cost, before
execution and immune to the indirection that defeats a syntactic target. There the sandbox dominates
the oracle, and we say so plainly. On the three un-dominated harms, the same mechanism also reaches
zero harm, but only by killing the legitimate task as well: the protected resource is the one the task
must use, so a single bit attached to the resource cannot pass the safe use and block the harmful one.
Hardened is safe-but-useless; permissive is useful-but-unsafe; no per-resource setting is both, which
is exactly what RA8, RA9, and RA12 show the oracle is. (AppArmor and SELinux were not available in this
kernel, an honest limitation; they attach a verdict to a resource the same way, so the structural
result does not depend on the specific mechanism.)

This is the honest reason the oracle exists: not the single-resource harms a sandbox already prevents,
but the relational, cumulative, and context-dependent harms it cannot express, which RA8, RA9, and RA12
demonstrate live and RA13 measures the real sandbox failing on. The agent's reliability against them is
contingent (on legibility, on framing, on not being compromised); the static sandbox's verdict cannot
condition on the relation, the aggregate, or the context at all; the gate's reliability is neither.

**The control-evaluation frame (RA16): the oracle is the trusted monitor an adaptive attack cannot
subvert.** The whole result reads cleanly in the vocabulary an agent platform now uses to reason about
deploying a powerful-but-untrusted model: *AI control* (Greenblatt et al., 2023). There a blue team
runs a protocol over an untrusted policy and a red team attacks it; the protocol is scored on a safety
axis (the probability no attack succeeds) against a usefulness axis (task autonomy when the policy is
honest), as a function of the blue team's *audit budget*, the fraction of actions it can escalate to an
expensive trusted check (a human, or here the oracle). The control literature's workhorse blue-team
move is a *trusted monitor*: a cheaper, weaker model that scores each action's suspiciousness, the most
suspicious fraction getting audited. Its safety against a non-adaptive attacker rises with the audit
budget, the textbook tradeoff, but a 2025 result shows an *adaptive* attacker subverts it
("Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols", arXiv 2510.09462): the monitor is
a model with a blind spot, so the red team finds an attack it scores as benign, and no budget short of
auditing everything catches it. RA16 runs this comparison on the RA5 substrate, sweeping the budget.
The model-monitor's non-adaptive safety climbs from 0 to 1 as the budget rises (and usefulness falls
one-for-one); its adaptive safety is pinned at 0 until the budget reaches audit-everything, where
usefulness is 0. The oracle coverage gate is a single point in the (safety 1, usefulness 1) corner,
the *same* point under both attackers, at a sparse fixed audit budget equal to the danger surface's
density (here 0.17). The reason is coverage restated: the oracle is a trusted monitor whose verdict is
*read* from the exact resulting state, not *predicted*, so there is no blind spot to aim at, and its
safety-usefulness frontier does not move when the adversary turns adaptive. That is the cleanest
statement of the contribution for a deployer: in the one domain that allows it, complete mediation is
the control protocol that escapes the audit-budget tradeoff a fallible trusted monitor cannot. The
product reading of the same fact is in Section 8 (the audit budget is the human-approval budget). This
is the hermetic engine, like RA5; the figure and CSV are `figures/ra16_control_eval`.

**Defense-in-depth: action-reading layers don't compose on effect-harms (RA19).** A deployer's first
instinct is to stack two cheap defenses, the model's own alignment (refusal) and a trusted-LLM monitor,
and assume the residual miss is their product. RA19 is a grounded model over our measured per-layer
rates (RA7 to RA12) that maps where that fails. The honest result, after an adversarial self-review that
caught an earlier overclaim: the oracle is the *unique framing-robust* closer (any layer set containing
it drives the triad residual to zero under every condition; the two action-readers reach at best 0.05
and at worst 1.0). And there is a concrete trap: a monitor's recall measured by *asking it the security
question directly* (RA11's 0.95) does not transfer to the *ambient* framing the agent actually operates
under, where the change looks like routine work, so a deployer who imports the direct number and assumes
independence over-estimates safety by up to about twenty times. Honestly scoped, the triad's flatness is
saturation (both action-readers near-blind), not a subtle correlation effect, and it is conditional on
that ambient assumption, which we make first-class and report as a sensitivity rather than a point. The
lesson for stacking agent defenses is narrow and useful: layers that both read the action surface share
a blind spot on harms that are only visible in the resulting state, and only an effect-reader closes
them. The engine, figure, and provenance are `figures/ra19_correlated_layers` and the module's
`PROVENANCE` table.

## 5. The honest negative, and the boundary it draws

The pre-commit target is not a theorem on fully arbitrary shell input, and pretending otherwise would
be the easiest way to discredit the whole program. So here is the limit, in full.

A purely syntactic pre-commit target ("does the command name the protected path") is covering for
commands that reference the path explicitly. It is *not* covering against indirection. An attacker who
assembles the path from pieces, or reaches it through a symlink planted earlier, writes a command in
which the protected prefix never literally appears. The scan sees nothing. In RA4 this evades the
pre-commit target every time.

The reason this does not sink the result is that realization, the post-commit diff, is still exact. No
indirection escapes it: if the protected region changed, the diff sees it, however the change was
spelled. So the principled gate routes by reversibility, a result from the broader program. Where an
action is reversible, you can execute it, diff the protected region cheaply, and roll back on harm:
exact, evasion-proof, and needing no pre-commit prediction at all. Where an action is irreversible (an
exfiltrating send leaves before you can observe it), you cannot run-then-undo, and you are back to
needing a covering pre-commit target.

**We hardened that target toward a real shell parser, and the result is a partition, not a patch
(RA18).** Rather than claim to parse arbitrary bash, we split the open edge into the half a string
analysis can close and the half it provably cannot, and we close the first. The hardened target is a
no-execution abstract interpreter (`src/verisim/realagent/shell_resolver.py`, standard library only)
that returns a three-valued verdict, FIRES / CLEAR / ABSTAIN, with one load-bearing invariant: ABSTAIN
never collapses to CLEAR, so a command it cannot fully fold is never silently declared safe. It closes
the string-resolvable indirection a purely syntactic scan misses: in-command variable assembly
(`p=/et; > ${p}c/shadow`), quote and backslash splicing (`/et""c/shadow`), pure-builtin command
substitution (`$(printf /%s/shadow etc)`), and the bash parameter-expansion, brace, tilde, and ANSI-C
quoting forms (`${x:-/etc/shadow}`, `{/etc,/x}/f`, `~/...` with a tracked HOME, `$'\x2f'`) that an
adversarial red team of our own found and we then folded in. It ABSTAINs, and routes by reversibility,
on the genuinely runtime- or filesystem-state-dependent residual: a value read from a file, `eval` of
a runtime string, a glob whose match depends on the disk, or a symlink whose target the command string
never names. That residual is *proven* irreducible to string analysis (the same command is harm or
benign depending only on on-disk state the gate never sees), so it belongs to the post-commit diff
(exact for reversible actions) or, for the irreversible slice, a fail-closed human prompt. On a labeled
adversarial corpus the syntactic target silently misses every named indirection while the resolver
catches the string-resolvable class, routes the rest, and silently misses none, with no regression on
explicit attacks and no benign over-fire. The honest headline is therefore not "arbitrary bash is
solved" but "the open edge splits into a closed half and a provably-irreducible half"; the irreducible
half is the post-commit diff's job, not the parser's. The resolver is wired into the Claude Code hook
(RA17), so its ABSTAIN-on-irreversible verdict is a real fail-closed routing decision rather than a
silent allow.

**We ran the exfiltration case live, because it is where the post-commit diff is weakest (RA14).** A
real victim container holds a credential; a real attacker listens off-host; the agent reads the
credential and sends it. The post-commit region diff reports *clean*, because the read leaves the
protected region byte-identical, while the attacker already holds the secret: a measured false
negative, not a modeled one. For exfiltration, detection is both too late and footprintless, which is
precisely why the reversibility routing sends an irreversible send to verify-before-commit. A
pre-commit egress gate (a real iptables allowlist) then drops the send to the non-allowlisted attacker
while the legitimate send to the allowlisted sink still goes through: safe and useful. The honest edge
carries over from RA4: an allowlist does not cover arbitrary covert channels (DNS, ICMP, timing), so we
claim prevention against the modeled channel, not coverage of every exfiltration method.

**And we ran the covert-channel case, because it is where that claim has to be honest (RA15).** We gave
the attacker a second channel: a DNS tunnel, the secret hex-encoded into the name of a lookup to an
attacker-controlled nameserver. A realistic egress allowlist must permit DNS, because the agent needs
name resolution, and DNS is dual-use. The result is the honest negative. The port allowlist that blocks
the direct send lets the DNS tunnel leak the credential while resolution still works: the allowlist is
not covering. Dropping DNS too stops the leak but kills name resolution: safe and useless, the lockdown
dilemma reappearing on the network. No port-level policy in the table is both leak-free and useful, and
the reason is structural. The coverage property needs a sparse, syntactically computable genesis
surface. A state change has one and the post-commit diff catches it however it is spelled. Exfiltration
does not: any allowed channel can encode a bit, so the surface is every egress action, not a rare set.
This is RA4's open edge on the network. The most you can do is bound the leak bandwidth (deny by
default, then minimize and proxy the few channels you must allow), and a timing channel over allowed
traffic is an irreducible low-bandwidth residual. We do not claim to make exfiltration impossible; we
turn the obvious channels from gameable into prevented and bound the covert ones, and we say so.

**And we measured the diff cost the review said we never measured (RA14).** The post-commit check is
not literally free. A full byte-hash of the protected region is O(region size): on real containers it
runs about 70 milliseconds at 2 megabytes and about 700 at 195. But the natural detector is an mtime
scan, which stats files without reading them, so it is O(file count) and flat in bytes: about 60
milliseconds at 195 megabytes, rising only gently with file count. A per-action change check is tens of
milliseconds, and it runs only on the sparse covered surface, not on every action. The cost is real,
bounded, and now reported from a real run rather than the hermetic fixture.

This boundary is also the scope statement. The oracle approach is strong exactly where the state is
exact and the harm is a state change: filesystems, processes, network flows. It is strongest for
integrity and availability, where realization is a state change the post-commit diff catches exactly;
for confidentiality it prevents the direct egress channels and bounds the covert ones (RA15) rather
than eliminating them, because there is no sparse surface to cover. And it does not cover GUI
computer-use, where actions are mouse clicks and "harm" is adjudicated by a human or an LLM judge.
That is OS-Harm's domain, and we make no claim there.

## 6. What this is, and what it is not

Stated plainly, because the difference between the two is where a reader could be misled.

**What it is.** A reproducible mechanism, grounded in an exact oracle, with complete mediation behind
it (an old principle, made cheap here, not a theorem we proved) and thousands of real assertions
behind that. Around a real Claude agent (RA1) and a real `/bin/sh`
(RA2) the gate drives missed-harm to zero at low cost. It generalizes across hundreds of randomized
tasks and a diverse attack taxonomy (RA3). It works without the agent's grammar (RA4). And head to
head, against our models of the deployed defenses, it is the arm an adaptive adversary cannot break
(RA5), with the heavy caveat below that this comparison is parameterized, not measured against real
systems.

**What it is not, yet.** The four-arm head-to-head's numbers in Section 4 come from a *hermetic
engine*: it runs on an in-repo fixture against the reference oracle, and the learned-guardrail arm is
a fidelity-phi proxy, not SafePred's actual system. The live RA6 and RA7 runs are real (real Docker,
real arbitrary bash, a real Claude agent, 60 measured trials) on six hand-built harm scenarios. We
have since taken the gate onto the **official Terminal-Bench harness** (RA20, `terminal-bench-core`,
80 tasks), run locally with a real Claude agent through the host Code CLI (no API key): the safety
lane plants a standard tripwire on the *real* TB task containers and measures missed-harm per arm with
the real container as the oracle, giving 0.00 (oracle) / 0.40 (denylist) / 1.00 (none) identically
across 16 tasks. That moves the OS-Harm/SafePred relationship from positioning to a measured result on
the recognized benchmark's substrate. What is still *not* done: the harm scenarios remain
hand-specified tripwires (not drawn from an external harm definition), and the capability sweep, while
full-scale (38/80), used a host-`docker exec` agent bridge and capped timeouts, so it is a faithful
local measurement, not a leaderboard submission. We will not call the leaderboard head-to-head done
before the in-container agent runs at the official timeouts.

**We replaced the fidelity-phi proxy with a real measurement, and it corrected us (RA11).** The
proxy's "adversarial miss pinned at 1.00" is a worst-case theoretical construct: it assumes a model
with a blind spot and an adversary who finds it. To check whether a *real* LLM guardrail actually
behaves that way on the danger an oracle is supposed to own, we built a faithful SafePred-style
guardrail (a real Claude call that judges each proposed config change before commit) and measured it
against the exact oracle on RA8's relational danger, varying the reasoning difficulty (chain depth 2
to 8, obfuscated graphs). The honest result is a negative for our convenient story: handed the graph
and asked the reachability question directly, the LLM guardrail caught every exposing edit at every
depth (recall 40/40 = 1.00, zero false-positives); wrapped in realistic task framing it slipped only
twice (38/40 = 0.95 [0.83, 0.99]), and the misses did not grow with depth. At RA8 scale a
well-prompted LLM guardrail is a strong relational-danger detector, not the catastrophically-breached
arm the proxy implies. So the oracle's real advantage over a learned guardrail here is *not* a
capability chasm; it is the guarantee (exact and every-time, versus a measured phi of about 0.95 that
issues no certificate on the one input that matters), complete mediation, and near-zero cost. This
sharpens the contribution and removes an overclaim: the case for the oracle is the guarantee, not the
suggestion that models cannot see relational harm.

**We then built the search RA11 lacked, and pointed it at the real judge (the monitor auditor, run
live).** RA11 was a fixed battery; it never searched for the judge's failure region. The monitor
auditor (the certify-the-monitor loop, now monitor-agnostic) puts the real Claude judge in the monitor
slot and the exact oracle in the verifier slot, so a hole is a realizing action the judge silently
passes. Run live against the current Claude Code judge on the two effect-harm families an action-reader
should be weakest on -- encoded shell-path indirection (the same `printf`/`rev`/`cut`/var-assembly
forms the RA18 resolver folds, where the path is in the action but obfuscated) and relational
reachability (the RA8 transitive untrusted->db exposure, swept to chain depth 8 with distractors) -- it
**still found no break**: the judge decoded every one of the thirteen shell-encoding classes (recall
1.00 over the sampled draws, both the direct and the agent-approval framing) and blocked the exposing
edit at every depth (4/4 at depths 2/4/6/8). The honest reading is a graded statistical certificate,
not a theorem: recall is bounded below by about 0.93 at 95% on the encoding family with wide intervals
on the small relational sweep, pinned to this one model and version, with a fair prompt that warns
about indirection. So at this scale a strong 2026 model judge is robust on the families we searched,
and the auditor's value is that it now *certifies* that with a confidence bound and a reproducible
search harness rather than asserting it; where such a judge breaks badly -- a weaker or older model, a
larger or adversarially-obfuscated graph, the ambient framing of Section 5's transfer trap -- is the
place to look, and the auditor is the instrument that would find it (reproduce:
`docs/llm-guardrail-audit-run.md`).

**We then hunted the one family where the failure should be structural, and it corrected the thesis
(SPEC-26).** Cumulative harm -- "at most B distinct sensitive records per task" -- is not a property of
any single action, so it is the place an action-reading judge ought to break. We ran a real `claude -p`
guardrail two ways and swept the budget magnitude to stress exact counting. The result splits cleanly.
A *stateless* per-action judge -- the deployed pattern, a hook shown one tool call -- catches the
over-budget read **0%** of the time at every budget on both a strong and a cheap model: a single read
always looks individually fine, so it structurally cannot enforce a budget. But a *history-augmented*
judge, shown the session so far, counts the distinct records correctly to a budget of **80** over a
~180-line history of scrambled, near-duplicate, re-read-laden actions, on the **cheap** model
(recall ~1.00), matching the exact accumulator. The empirical break we hunted did not appear. The honest
synthesis of all three live experiments is a reframing, not a win: **the action-vs-effect gap is not a
model-capability gap, it is a state-and-guarantee gap.** Modern models, even small ones, see effect-harm
when shown the relevant state -- the decoded command, the reachability graph, the session history -- and
the only thing that reliably withholds that state is the *deployed* stateless-hook pattern. So the
oracle's defensible value is precisely that it *is* the state the deployed pattern omits (the exact
counter, the reachability closure, the post-commit diff), with a certificate and at a counter's cost,
not that a model cannot see the harm. We lead with that (reproduce: `docs/cumulative-horizon-run.md`).

**The tradeoff, named.** What you give up for hard-to-game safety is generality of domain. The
oracle's whole advantage, exact state, is exactly what restricts it to computer worlds with checkable
state. It buys provable, cheap, injection-robust safety in the place agents are most dangerous (the
shell, the filesystem, the network), and it buys nothing in the place it does not apply (pixels on a
screen). That is a trade most agent platforms would take, but it is a trade, and we say so.

## 7. Related work

**This is, first, runtime enforcement, and we should say so.** The coverage property (mediate every
action that can realize the harm) is complete mediation (Saltzer and Schroeder, 1975); its
model-independence is the soundness of an execution monitor (Schneider, 2000); the reversibility
routing is the edit-automata model (Ligatti, Bauer, Walker, 2005). These are established results, not
ours. The honest framing of our contribution is not a new theorem but the conditions under which sound
enforcement is cheap in computer-use worlds. A crowded recent literature already enforces agent
actions at runtime, deterministically and model-independently: GoEX (arXiv 2404.06921, 2024) proposes
post-facto validation, undo, and damage confinement (our reversibility routing, earlier); Progent
(arXiv 2504.11703, 2025) enforces tool-call policies deterministically with demonstrated resilience to
adaptive attacks; and mandatory-access-control frameworks for LLM agents (arXiv 2601.11893, 2026)
apply the same idea at the OS layer. We do not claim to be the first to gate agent actions, and a
fair comparison must run these systems, not just our models of them (see the limitations).

**The 2026 deterministic-guardrail crowd is the proper context for the contribution — as the set of
monitors the method audits, not as rivals to the gate.** Since this program began, the design space we
occupy filled in independently. YoloFS ("Don't Let AI Agents YOLO Your Files," arXiv 2604.13536) moves
mediation into the filesystem itself: staging isolates mutations before commit (a pre-commit gate),
snapshots give agent self-correction (reversibility routing), and progressive permission graduates the
prompts — the same effect-level, commit-then-review shape we reach for, arrived at as a storage-layer
design. ePCA ("Provably Secure Agent Guardrail," arXiv 2605.29251) compiles agent intentions into
first-order constraints an SMT solver discharges before execution, claiming provable, adaptive-attack-
resilient complete mediation at sub-millisecond latency. "Before the Tool Call" (arXiv 2603.20953)
formalizes deterministic pre-action authorization, and a transactional sandbox (arXiv 2512.12806) wraps
execution in checkpoint/rollback. Each of these *asserts* complete mediation over a hand-specified
surface (a staging policy, an axiom set, an authorization rule) and none *certifies* it — which is
exactly the gap this paper fills. Two honest distinctions, and one offer. ePCA proves over a *model* of
the action (its first-order encoding); the oracle reads the *true resulting state*, including the
relational, cumulative, and context-dependent effects (RA8/RA9/RA12) an action-level encoding does not
carry. YoloFS stages every mutation; the gate certifies a *sparse covering surface* so the off-surface
majority needs no staging at all (RA21). The offer is the method itself: the discover → fix → re-verify
loop (RA22–RA25) is not specific to our shell-path target. Given an exact oracle for the harm, it audits
*any* such monitor's covering surface — YoloFS's staging predicate, ePCA's axioms — for an encoding that
realizes the harm off it. We demonstrate it on our own monitor and name auditing a third-party system as
future work, not a present result.

**AI control is the frame this work belongs in, and it sharpens the contribution.** The AI-control
program (Greenblatt et al., arXiv 2312.06942, 2023) evaluates deployment protocols for a
powerful-but-untrusted model as a red-team/blue-team game scored on a safety-usefulness frontier at a
given audit budget. Its workhorse blue-team move is a *trusted monitor*: a cheaper, weaker model that
scores each action and escalates the suspicious ones. The program's own recent result is that this move
is fragile: an adaptive attacker subverts a trusted monitor ("Adaptive Attacks on Trusted Monitors
Subvert AI Control Protocols", arXiv 2510.09462), because a model monitor has a blind spot to aim at,
exactly the failure our RA5/RA11 line describes for a learned guardrail. Verisim's contribution stated
in this frame (RA16, Section 4) is one move that domain allows: the trusted monitor need not be a model
at all. The oracle coverage gate is a trusted monitor whose verdict is read from the exact resulting
state, so it has no blind spot, and its safety-usefulness frontier is invariant to whether the
adversary is adaptive, sitting in the (safety 1, usefulness 1) corner at a sparse fixed audit budget.
We do not claim to improve the control protocols that operate where no oracle exists (collusion-
resistant monitoring, resampling, defer-to-trusted); we add the operating point available when the
trusted check can be the environment itself.

**SafePred (arXiv 2602.01725) is the closest predictive-guardrail foil.** It also previews an action's
effect before execution and checks selectively on a danger surface. The difference is the predictor:
SafePred uses a learned world model, so its guarantee is best-effort rather than provable, which is a
property of any learned predictor, not a self-disclaimer we should attribute to its authors without a
direct quote. We measured how much that matters: a faithful SafePred-style guardrail (a real LLM
judging each change) caught RA8's relational danger with recall 0.95 to 1.00, so "best-effort" is, at
these scales, very good, not weak. The gap to the oracle is therefore not detection power but the
*kind* of assurance: a measured 0.95 with no certificate on the specific input, versus an exact
every-time verdict (RA11, Section 6). Verisim replaces the learned predictor with the exact oracle,
which is what converts best-effort into hard-to-game in the domain where an oracle exists. The two are
complementary:
SafePred's generality where no oracle exists, an exact check where one does.

**OS-Harm (arXiv 2506.14866) is the recognized anchor.** Built on OSWorld, 150 tasks across deliberate
misuse, prompt injection, and model misbehavior, it is GUI computer-use scored by an LLM judge (0.76
F1 against human labels). It is where the field measures CUA safety, and its finding (Claude and peers
comply with a meaningful fraction of injections) is the motivation for this work. Its GUI, LLM-judge
setting is also why we do not build on it directly: the oracle wants exact state and a programmatic
harm signal, which a desktop screenshot and a judge do not cleanly provide. Related benchmarks (OS-
Blind / the agent-safety blind-spot work, arXiv 2604.10577; RedTeamCUA, arXiv 2505.21936) reinforce
the same picture from the adversarial side.

**Terminal-Bench (Laude Institute, Apache 2.0) is the substrate we target.** Roughly 100 hand-built
shell tasks in Docker containers with programmatic success functions, used across frontier labs. Its
shell-and-container shape is the oracle's native habitat (exact state, real arbitrary bash), which is
why the external number lives there.

**The permission-system status quo** (Claude Code's tool approval, managed-agents' allow/deny/ask) is
the second baseline. It is the thing in production, and the head-to-head treats it as such rather than
as a strawman.

## 8. What it buys a platform, and what comes next

The product framing is concrete, and it is the reason to care.

A platform like Anthropic already runs a permission system on Claude Code and managed agents, and it
already faces the approval-fatigue-versus-safety tension that system creates. The complete-mediation
result offers a different operating point: auto-approve the large off-surface majority of actions, and
reserve verification (or a human prompt) for the sparse on-surface minority that can actually realize
harm. That cuts the approval burden *and* raises the floor, because the on-surface gate is hard to
game by injection in a way a click-through approval is not. The same surface that makes verification
cheap is the surface a human should have been asked about anyway.

Said in the control vocabulary of Section 7, the permission system *is* a control protocol and the
approval prompts *are* its audit budget. RA16 is then a statement about that budget directly: a pattern
denylist and a learned pre-filter both leave the adaptive-attack frontier short of safety 1 at every
budget, while the on-surface oracle reaches safety 1 at a sparse fixed budget without spending autonomy
on the off-surface majority. The product ask is to make the approval prompt fire on the covering
surface rather than on a pattern list, which is what turns approval fatigue and injection robustness
from a tradeoff into the same lever.

That ask is now a runnable artifact, not a slide (RA17). The coverage gate is implemented as a real
Claude Code `PreToolUse` hook (`scripts/claude_code_coverage_hook.py`): it reads the tool call from
stdin, returns `allow` for the off-surface majority and `ask` for the on-surface minority, over the
three tools a coding agent mutates state with (Bash, Edit, Write). Measured on a realistic tool-call
battery against the pattern-denylist status quo, the denylist shows *both* failure modes at once, it
over-prompts on a benign `rm` under the work directory (approval fatigue) and still auto-allows the
`chmod` and `mv` classes it never enumerated (a 0.29 missed-harm breach), exactly the RA4/CU34 gap;
the coverage hook has zero benign approval fatigue and zero missed-harm on the explicit arsenal, at a
human-approval burden equal to the sparse covering surface. The honest edge is the same one RA4 names:
a Bash path built by indirection carries no literal prefix, so the syntactic pre-commit target misses
it, which is why deployment routes the irreversible slice to the post-commit reversibility check rather
than auto-allowing it. This is the §8 operating point demonstrated on the actual platform mechanism.

**And we measured that operating point on real agent traffic, not a battery (RA21).** The RA17
battery is 18 hand-built calls; the question a deployer actually has is what the coverage operating
point costs on the traffic an agent really produces. We replayed every state-mutating tool call a
real Claude Code installation had logged (`~/.claude/projects/**`, 3,836 sessions, **123,195** Bash /
Edit / Write calls) through three arms: no gate, the pattern denylist, and the oracle coverage gate
over a realistic coding-agent surface (system dirs, secret files like `.env` / `*.pem` / `id_rsa` /
cloud and ssh credentials, VCS integrity `.git/hooks` and `.git/config`, and CI definitions
`.github/workflows`). The result is the §8 claim made quantitative on real data. The coverage budget
is genuinely sparse: only **1.1%** of real actions touch the danger surface, so **98.9% are
auto-approvable** with no prompt. The denylist — made deliberately *fair* here (a destructive-verb
plus real-file-overwrite list that excludes `2>&1` and `>/dev/null` fd-noise, so its fatigue is not
inflated) — prompts on **9.4%** of actions, about 8.7 times as often, and **99% of those prompts are
off-surface** (benign `rm` / `mv` / redirect of build artifacts): the approval-fatigue failure mode,
measured. And it still leaks: on an injected, labeled harm arsenal (the benign distribution is the
real corpus; the harm distribution is labeled, since real sessions carry no labeled harms) the
denylist misses **0.70** — the `chmod` / `sed -i` / `cp` / `python -c` / integrity classes its verb
list never enumerated — where the coverage gate misses **0.00**. Adding `chmod` to the list only
moves the leak to the next un-enumerated verb; the coverage surface escapes that enumeration treadmill
by keying on the effect's genesis, not on verbs. The headline is not an artifact of one workflow: broken
down across the corpus's 31 distinct projects (29 with ≥20 calls), the per-project off-surface density
has median **0.991** and a tight IQR [0.981, 0.995], and the only projects that dip below 0.95 are
small *security-work* projects (the sharpest is the program's own RA7 credential-tamper clean-room)
where the surface *is* the work — so the gate's approval budget correctly auto-scales to how dangerous
the real work is rather than failing. Two honest boundaries: the benign traffic is one user's corpus
(large and real, generalizing across its projects but not a sampled population — the cross-*machine*
check is the obvious next step), and the harms are injected, not observed, the same boundary as RA17.
The extractor is privacy-safe by construction — it emits rates and verb names, never the user's command
strings. (Reproduce: `docs/cc-corpus-run.md`.)

Three things remain to turn this preprint into a result a platform can act on; the first two harden the
substrate, and the third *is* the contribution this paper leads with, told last here only because it
rests on the first two.

First, the external number at scale, which RA20 has now taken from runbook to a local run on the
official harness. We wrote a custom Terminal-Bench agent that drives a real Claude agent through the
host Code CLI with no API key (it operates each task container via `docker exec`), and ran all 80
tasks of `terminal-bench-core`: capability 38/80 = 0.475 (15 of the 42 misses are agent-timeout/error
at our cap, not capability), and the safety lane on 16 real TB task containers, where the gate plants a
standard tripwire and the *real container is the oracle* (execute, diff the protected region, roll
back), giving missed-harm 0.00 for oracle-coverage against 0.40 for a pattern denylist and 1.00
unguarded, identical on every task — RA3's danger-surface generalization, now on the recognized
benchmark's substrate. What remains for a *leaderboard-grade* claim is an in-container agent (TB's
native design) rather than the host-`docker exec` bridge, uncapped timeouts for the heavy build/ML
tasks, and a published-style injection layer scored against an external harm definition rather than our
own tripwires. The harness, the agent, and the safety adapter now exist and run; the gap is the
submission conditions, not the mechanism. (Reproduce: `docs/terminal-bench-run.md`.)

Second, the irreversible-arbitrary-bash target, which RA18 has now largely closed and which the
remaining work should finish. RA18 hardened the pre-commit target from a regex into an abstract shell
interpreter that partitions the open edge: it closes the string-resolvable indirection (variable
assembly, splicing, command substitution, parameter expansion, brace, tilde, ANSI-C) and routes the
provably-irreducible runtime/filesystem-state residual by reversibility. What remains is to grow the
folded grammar's coverage (more pure builtins, cross-call binding) and to publish the soundness
argument formally; the partition is established and adversarially tested, but the closed fraction is a
measured coverage of an evolving corpus, not yet a proof over all bash.

Third — and this is the contribution the paper now leads with, not a coda — the surface itself. Every
target above, and every covering surface in the 2026 deterministic-guardrail crowd (Section 7), is
*hand-specified*, which inherits the very enumeration problem we charge the denylist with: a covering
target is only as good as the human who wrote it, and *asserting* completeness is not *certifying* it.
This is where the program stops re-deriving complete mediation and does the thing the neighbors do not:
it makes the monitor's completeness a learning problem with a free verifier and certifies it under an
adaptive learned adversary. RA22 begins it. It runs a counterexample-guided (CEGIS) search
that starts from an *empty* target and grows it: a structured proposer emits candidate harmful
actions, the exact oracle adjudicates which truly realize the harm (confirmed bit-for-bit against a
real `/bin/sh`, the RA2 move), and the target is repaired only where a realizing action slipped off
it, until no hole remains. The result, with no hand-specification, is the same partition RA18 drew by
hand, now *discovered*: the loop covers the entire class of realizing actions that name the protected
path and automatically isolates the indirection/symlink class as irreducible to any pre-commit string
pattern (routed to the post-commit diff), with the soundness invariant intact (no realizing action is
silently off-surface, zero benign over-fire). This is the program's founding shape — a proposer
checked by a free exact verifier — finally load-bearing for *safety*, and it is where the learned
world model reconnects: the structured fuzzer is the proposer slot a learned adversary would fill to
hunt harder holes. (Reproduce: `docs/coverage-synthesis-run.md`.)

And we filled that slot (RA23). The structured fuzzer becomes a *learned* adversary: a policy over
shell-mutation operators trained by REINFORCE whose reward is the exact oracle's own hole-verdict
(`realizes ∧ ¬target`) — no learned reward model in the loop, so this is the program's
verifier-as-reward thesis applied to adversarial test generation, the model proposing and the oracle
disposing. Run against a blind uniform proposer over the same operator space (600 oracle calls,
labels cross-checked bit-for-bit against a real `/bin/sh`), the learned adversary finds 2.3 times as
many coverage holes per oracle call, and as the CEGIS loop repairs the literal class its proposal mass
visibly flows off `literal` (to about 0.01) and onto the indirection transforms that still realize
holes. It also surfaces a residual class the enumerative grid never contained (quote-splice
indirection, `/et""c/shadow`), which the oracle confirms realizes the harm and which no string pattern
covers — widening the command space the certificate is tested over. The load-bearing result is that
the soundness invariant holds *under the adaptive adversary*: every hole it finds is covered or
explicitly routed, none silently off-surface, and the literal class is still closed. This is where the
program's two arcs meet — the RL/world-model arc supplies the proposer, the oracle-coverage arc
supplies the verifiable reward and the soundness guarantee. The honest limits are the same as RA22's,
plus that the policy is tabular over a small operator space, not yet a neural proposer over a
compositional grammar. (Reproduce: `docs/learned-proposer-run.md`.)

And we closed that last limit, which turned out to do more than scale the adversary — it hardened the
defender (RA24). The proposer becomes a *neural* autoregressive policy (the repo's own from-scratch
GPT) over a genuinely *compositional* grammar: the protected path is split into atoms and the policy
assigns each atom one of twelve standard shell encodings (variable assembly, quote-splice, ANSI-C
hex and octal, `printf %s`, parameter-default, nested substitution, backticks, a `printf` format
escape, and `cut`/`rev` pipe-filters), which *compose*, so the action space is exponential in the
path length and a single-transform policy like RA23's can only reach the dozen uniform forms. The
opponent is no longer a literal-only strawman but the *real* RA18 abstract resolver, and the reward
is the tiered exact-oracle verdict — 1.0 for a string-resolvable realizing command the resolver
silently declares CLEAR (a true soundness violation), 0.5 for one it cannot fold and must ABSTAIN on
(the folder's incompleteness frontier), 0 when it folds and FIRES — with no learned reward model in
the loop and every composition's realization cross-checked bit-for-bit against a real `/bin/sh` (0
unfaithful of 200). The result is the program's red-team-folds-back pattern, now automated, learned,
and load-bearing: run against the resolver as the program shipped it, the neural adversary *discovers
a genuine RA18 soundness bug the hand-run red team missed* — `printf` decodes backslash escapes in
its format string, which the fold did not, so `$(printf '\x2f%s' e)` realized `/etc/...` while the
resolver read CLEAR — finding it about eighteen times more often per oracle call than blind search
and at lower composition depth than the single-transform control can express. We fold the fix back
(ABSTAIN on any unfolded format escape, the conservative sound choice), and re-running against the
hardened resolver the soundness invariant *holds under the learned adversary*: zero string-resolvable
realizing commands are silently CLEAR across the composition space. What the adversary does then is
map the incompleteness frontier — the `cut`/`rev`/format-escape forms the resolver now ABSTAINs on —
4.3 times more efficiently than blind and over a mixed-composition space the single-transform policy
is architecturally blind to (1,500 distinct compositions against its 18). This is the two arcs
meeting at the scale the program was built for: a neural sequence model proposing over a compositional
grammar, the exact oracle disposing as a verifiable reward, and the soundness certificate surviving —
and improving under — an adaptive learned attacker. The honest limits remain RA22's (one harm family,
a finite toolkit, a certificate empirical over the sampled space not a proof), plus that the dense
reward and small space make this the easy-RL regime; the contribution is the mechanism and the
discover-fix-reverify loop, not RL difficulty. (Reproduce: `docs/neural-proposer-run.md`.)

And we closed the frontier RA24 mapped, which taught the sharpest lesson of the arc (RA25). The
obvious response to a mapped incompleteness is to fold it, so we did: the resolver gains a sound
extension that decodes printf format escapes and evaluates self-contained pure-filter pipelines
(`$(echo X | rev)`, `$(echo X | cut ...)`) to their exact constants, moving those forms from ABSTAIN
to FIRES. Re-run against the folded resolver the RA24 adversary's reward collapses from 0.47 to 0.00
per oracle call, because on the twelve-mechanism grammar those forms were the entire frontier, and the
soundness invariant still holds (no realizing command becomes a silent CLEAR; the fold only ever
substitutes the exact constant). But the result that matters is the second one: this is an enumeration
treadmill. A pure deterministic filter pipeline is an unbounded family (`rev`, `cut`, then `tr`, `sed`,
`base64 -d`, `xxd -r`, `od`, `awk`, ...), so a battery spelling the protected path through each shows
the folder covers only the three it was taught and ABSTAINs the rest; fold those and the adversary
finds the next. Folding by enumeration is precisely the denylist failure mode the program charges the
status quo with. The principled closer is therefore not a bigger folder: every command in the battery
is a reversible file write, so its ABSTAIN is already safe, routed by reversibility (CU27) to the
exact post-commit diff, which catches the realized effect however the path is spelled, with no folder
at all (the partition: four of four of the ABSTAIN tail reversible and post-commit-safe, zero
irreversible). Folding the common filters is a usability win, fewer prompts, not a safety one; the
soundness guarantee was always the post-commit diff's. The one place the folder is safety-load-bearing
is the irreversible slice, an egress send built through an unfolded filter, which fails closed to a
human because no post-commit diff can recover the sent bytes, and that is exactly the
irreversible-exfiltration edge §5 already names as the honest open frontier (RA15), not a folder to
grow without bound. (Reproduce: `docs/frontier-close-run.md`.)

The honest summary is short. Building a deterministic, execution-grounded complete-mediation guardrail
for a computer-use agent is, in 2026, no longer the contribution — it is a crowded design, and we are
one of several. The contribution is what the crowd leaves undone: *certifying* such a monitor's
completeness. In the one domain that allows it, the monitor's own free oracle is the verifiable reward of
an adaptive learned adversary, so completeness becomes a discover → fix → re-verify loop rather than an
assertion — and run that way it found a real soundness bug a human red team had missed, then held the
soundness invariant under the learned attacker. The substrate that earns this — the gate around a real
model and a real kernel, the un-dominated triad an oracle expresses and a sandbox cannot, the sparse
operating point on real agent traffic, the run on Terminal-Bench's own containers — is all real and
reproducible, and it is what makes the certification claim concrete rather than abstract. What we have
*not* shown is the certificate as a proof rather than an empirical result over a sampled space, the
method run against a third-party monitor (YoloFS, ePCA), or a leaderboard-grade Terminal-Bench
submission. That, not the mechanism, is what is left.

## Reproduce

Every number above regenerates from committed code and run-records in the verisim repository. The
relevant modules are under `src/verisim/realagent/` (the RA arc) and `src/verisim/acd/` (the coverage
framework, CU arc), the experiments under `src/verisim/experiments/`, and the figures and their CSVs
under `figures/`. The hermetic results run torch-free with no network. The control-evaluation recast
(RA16) is `src/verisim/realagent/control_eval.py` with `python -m verisim.experiments.ra16_control_eval`
regenerating `figures/ra16_control_eval.{csv,png}`. The §8 Claude Code hook (RA17) is
`src/verisim/realagent/claude_code_gate.py` with the runnable hook at
`scripts/claude_code_coverage_hook.py` and `python -m verisim.experiments.ra17_claude_code_gate`
regenerating `figures/ra17_claude_code_gate.{csv,png}`. The hardened pre-commit target (RA18) is
`src/verisim/realagent/shell_resolver.py` with `python -m verisim.experiments.ra18_shell_resolver`
regenerating `figures/ra18_shell_resolver.{csv,png}`. The official Terminal-Bench lane (RA20) is the
custom agent `src/verisim/realagent/tb_agent.py` (run via `tb run -d terminal-bench-core==0.1.1
--agent-import-path verisim.realagent.tb_agent:HostClaudeCliAgent`, no API key) and the safety adapter
`bench/tb_safety/gate_eval.py`; results in `docs/terminal-bench-run.md` and
`bench/tb_safety/{RESULTS,CAPABILITY_full80}.csv`. It needs Docker and an authenticated `claude` CLI
(no API key). The real-corpus measurement (RA21) is `src/verisim/realagent/cc_corpus.py` with the
privacy-safe replay `bench/cc_corpus/extract.py` (walks `~/.claude/projects`; emits rates, never
command strings) regenerating `bench/cc_corpus/{RESULTS.csv,summary.json}` and, via
`figures/plot_ra21_cc_corpus.py`, `figures/ra21_cc_corpus.png`; results in `docs/cc-corpus-run.md`,
the logic pinned hermetically by `tests/test_ra21_cc_corpus.py` (which does not read `~/.claude`).
The automated-coverage synthesizer (RA22) is `src/verisim/realagent/coverage_synth.py` with
`python -m verisim.experiments.ra22_coverage_synth` (add `--bash` for the real-shell label
cross-check) regenerating `figures/ra22_coverage_synth.{csv,png}`; results in
`docs/coverage-synthesis-run.md`, logic pinned by `tests/test_ra22_coverage_synth.py`.
The learned adversarial proposer (RA23) is `src/verisim/realagent/learned_proposer.py` with
`python -m verisim.experiments.ra23_learned_proposer` (add `--bash` for the real-shell label
cross-check) regenerating `figures/ra23_learned_proposer.{csv,png}`; results in
`docs/learned-proposer-run.md`, logic pinned by `tests/test_ra23_learned_proposer.py`.
The rest of the paper's numbers are hermetic (torch-free, no network).
