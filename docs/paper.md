# Verisim: oracle-grounded, hard-to-game safety for computer-use agents

**Status (June 2026): under revision after an adversarial self-review; read [docs/review.md](review.md)
alongside this.** The mechanism and every number below are real and reproducible. But an independent
hostile review (three reviewers, prior-art verified) found that this paper, as written, overclaims:
the central "coverage theorem" is a renaming of complete mediation (Saltzer and Schroeder, 1975) and
execution-monitor soundness (Schneider, 2000); the live demos are dominated by a correctly-configured
OS sandbox (SELinux/AppArmor plus `chattr`), which we do not yet run as a baseline; the empirical
framing-dependence result reproduces a published finding (OS-Blind) at smaller scale with one
contestable harm label; the head-to-head is parameterized, not a run of real systems; and real
deterministic-enforcement prior art (GoEX 2024, Progent 2025) already does model-independent agent
gating. We have corrected the specific overclaims in this revision and added the limitations. The deeper
repositioning the review calls for is underway: Section 4 now includes RA8 and RA9, which run the
MAC/sandbox baseline for real and demonstrate the two harms a per-resource sandbox cannot express, a
relational graph-reachability invariant and a cumulative blast-radius budget. This is the honest
reason the oracle exists, and it is now the part of the paper that leads. It is still not complete: a
real SafePred run, powered empirics, context-dependent danger, and the full recast around
complete-mediation-made-cheap remain. The results run only on controlled worlds with a hand-specified
danger model. Read accordingly, and read the review.

## Abstract

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
true next state, bit for bit. That removes the model from the safety-critical path. Our central
result is a coverage theorem. Danger in such a world lives on a sparse, model-free surface (the
actions that can *realize* the harm), and a gate that consults the exact oracle on that surface is
hard to game by *any* model, at a cost set by how rare the surface is.

We measure this as a deployment bake-off against the two deployed defenses. Under an adaptive
prompt-injection adversary, a learned guardrail and a permission denylist are both breached
completely, while the oracle coverage gate is not breached at all, at no loss of task success and
about one oracle call per task. A better model does not close the gap, and a longer denylist does
not either. The exact oracle does.

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

## 3. Coverage: where danger lives, and why verifying only that is enough

The claim in one sentence: danger has a sparse, model-free surface, and verifying that surface is
both safe and cheap, provably, because of a single property called coverage.

Make three objects precise for a given harm `D`, for example "a credential file is corrupted."

**Realization.** `realizes(state, action)` is true when executing this action from this state actually
brings about the harm. It is read off the exact oracle: run the action, diff the protected region,
see if it changed. It never consults the learned model. It is the ground truth of harm.

**The genesis surface, or target.** `target(state, action)` is the cheap, model-free rule for *when
to consult the oracle*. In the simplest case it is syntactic: does this command name a path under the
protected region? It is computed from the action itself, not from a model's opinion of the action.

**Coverage** is the bridge between them, and the whole result hangs on it:

> for every state and action, `realizes(s, a)` implies `target(s, a)`.

In words: every action that *could* cause the harm is on the surface we verify. Nothing that realizes
the danger lives off it.

From coverage, the safety property falls out, and it is worth walking slowly because it is the point.
Take any agent, any model, any attacker, any timing. The attacker can win only by getting some action
`a` with `realizes(s, a)` executed without being caught. But coverage says `realizes(s, a)` implies
`target(s, a)`, so the gate consults the oracle on `a`. The oracle sees the true `realizes` and
blocks. The consult decision never read the model, so this holds *no matter how bad the model is*. A
covering, model-free gate is hard to game at a cost equal to the number of on-surface actions, which
is rare and cheap by construction.

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

Bottom line: the gate drives missed-harm to zero around a real model and a real kernel, it
generalizes to tasks and attacks it never saw, and head-to-head it is the only defense an adaptive
adversary cannot break.

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
these harms.)

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

This is the honest reason the oracle exists: not the single-resource harms a sandbox already prevents,
but the relational and cumulative harms it cannot express, which RA8 and RA9 now both demonstrate live.
Context-dependent danger is the third member of this family and remains to be shown.

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
action is reversible, you can execute it, diff the protected region for free, and roll back on harm:
exact, evasion-proof, and needing no pre-commit prediction at all. Where an action is irreversible (an
exfiltrating send leaves before you can observe it), you cannot run-then-undo, and you are back to
needing a covering pre-commit target. Hardening that target toward a real shell parser, for
irreversible actions on arbitrary bash, is the open problem the program is honest about rather than
papering over.

This boundary is also the scope statement. The oracle approach is strong exactly where the state is
exact and the harm is a state change: filesystems, processes, network flows. It does not cover GUI
computer-use, where actions are mouse clicks and "harm" is adjudicated by a human or an LLM judge.
That is OS-Harm's domain, and we make no claim there.

## 6. What this is, and what it is not

Stated plainly, because the difference between the two is where a reader could be misled.

**What it is.** A reproducible mechanism, grounded in an exact oracle, with a coverage theorem behind
it and thousands of real assertions behind that. Around a real Claude agent (RA1) and a real `/bin/sh`
(RA2) the gate drives missed-harm to zero at low cost. It generalizes across hundreds of randomized
tasks and a diverse attack taxonomy (RA3). It works without the agent's grammar (RA4). And head to
head, against our models of the deployed defenses, it is the arm an adaptive adversary cannot break
(RA5), with the heavy caveat below that this comparison is parameterized, not measured against real
systems.

**What it is not, yet.** The four-arm head-to-head's numbers in Section 4 come from a *hermetic
engine*: it runs on an in-repo fixture against the reference oracle, and the learned-guardrail arm is
a fidelity-phi proxy, not SafePred's actual system. The live RA6 and RA7 runs are real (real Docker,
real arbitrary bash, a real Claude agent, 60 measured trials), but they are a *vertical slice*: six
hand-built harm scenarios on Terminal-Bench-shaped containers, not the full official Terminal-Bench
harness across its ~100 tasks, and not OS-Harm. The harm scenarios are hand-specified (we wrote the
tripwires and framings), not drawn from or scored against an external harm definition. So the relationship to
OS-Harm and SafePred is, today, a sound *positioning* plus a real *vertical-slice demonstration*, not
a full *head-to-head on the published leaderboards*. The engine and the live harness exist to produce
that; running it at the official scale is the next step, and we will not call it done before it runs.

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

**SafePred (arXiv 2602.01725) is the closest predictive-guardrail foil.** It also previews an action's
effect before execution and checks selectively on a danger surface. The difference is the predictor:
SafePred uses a learned world model, so its guarantee is best-effort rather than provable, which is a
property of any learned predictor, not a self-disclaimer we should attribute to its authors without a
direct quote. Verisim replaces the learned predictor with the exact oracle, which is what converts
best-effort into hard-to-game in the domain where an oracle exists. The two are complementary:
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
already faces the approval-fatigue-versus-safety tension that system creates. The coverage result
offers a different operating point: auto-approve the large off-surface majority of actions, and
reserve verification (or a human prompt) for the sparse on-surface minority that can actually realize
harm. That cuts the approval burden *and* raises the floor, because the on-surface gate is hard to
game by injection in a way a click-through approval is not. The same surface that makes verification
cheap is the surface a human should have been asked about anyway.

Two steps turn this preprint into a result a platform can act on, and neither is another theorem.

First, the external number at scale. The RA6 vertical slice already runs the gate live on a real
container against a real Claude agent and shows the framing-independence result. The remaining step is
breadth: run it across the full official Terminal-Bench harness (~100 tasks) with held-out harm
tripwires and a published-style injection layer, reporting missed-harm, task success, and cost per
arm, against OS-Harm's numbers as the anchor and SafePred as the foil. That converts the vertical
slice into a leaderboard-scale comparison on shared ground.

Second, the irreversible-arbitrary-bash target. Close the RA4 boundary by hardening the pre-commit
target toward a real shell parser for the irreversible slice, where post-commit rollback is not
available. That is the one place the guarantee is currently a heuristic, and it is worth making it a
theorem.

The honest summary is short. In the one domain that allows it, the exact oracle turns a guardrail from
something an adversary can talk around into something it cannot, cheaply and without getting in the
agent's way. We have shown that around a real model and a real kernel, and against the two defenses in
production. We have not yet shown it on an external benchmark, and that is the next thing we will do,
not the thing we will claim we did.

## Reproduce

Every number above regenerates from committed code and run-records in the verisim repository. The
relevant modules are under `src/verisim/realagent/` (the RA arc) and `src/verisim/acd/` (the coverage
framework, CU arc), the experiments under `src/verisim/experiments/`, and the figures and their CSVs
under `figures/`. The hermetic results run torch-free with no network. The live Terminal-Bench lane,
which requires Docker and a model API key, is documented as a runbook in `docs/specs/SPEC-22.md`.
