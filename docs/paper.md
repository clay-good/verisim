# Verisim: oracle-grounded, hard-to-game safety for computer-use agents

**Status (June 2026):** a research preprint distilled from the verisim repository. The mechanism and
every number below are real, reproducible, and tested against an exact oracle. The honest boundary is
stated up front and again in Section 6: the results run on controlled computer worlds (a reference
filesystem and a real `/bin/sh`) with a hand-specified danger model, and the head-to-head in Section
4 is a hermetic engine against named baselines, not yet a live run on an external benchmark. That
live run is the immediate next step, not a finished result. Read accordingly.

## Abstract

A computer-use agent is one mis-predicted action away from an irreversible mistake: overwriting a
credential, deleting the wrong file, sending data it should never send. The field's own safety
benchmarks confirm this is not hypothetical. On OS-Harm, frontier models including Claude comply with
roughly 10 to 20 percent of prompt injections, and a single injected instruction can turn a helpful
agent into an exfiltration tool.

Two defenses are deployed today, and both have a known failure mode. A permission system (allow,
deny, ask a human) either drowns the user in approvals or is gamed by an action it did not think to
deny. A learned predictive guardrail, the most recent proposal, forecasts an action's consequences
with a model and blocks the dangerous ones; its own authors note that safety then depends on model
accuracy, and an adversary can exploit a prediction failure.

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
never anticipated. Its limitation is stated plainly by its own authors. Safety depends on model
accuracy, and an adversarial agent can exploit a prediction failure. A model has blind spots, and an
attacker who can rephrase will find one.

So the gap is specific. We have a gameable rule and a best-effort model. What nobody has, because
nobody thought the domain allowed it, is a defense that is cheap, keeps the agent useful, and cannot
be talked out of catching the harm.

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
removes the worst case. This is precisely the gap SafePred names as its own limitation, measured.

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
head it is the only defense an adaptive adversary cannot break (RA5).

**What it is not, yet.** The head-to-head in Section 4 is a *hermetic engine*: it runs on an in-repo
fixture against the reference oracle, and the learned-guardrail arm is a fidelity-phi proxy, not
SafePred's actual system. We do not yet have a number on OS-Harm, and we do not yet have a number on
Terminal-Bench. The danger model throughout is hand-specified (a credential region), not learned from
or scored against an external harm definition. So the relationship to OS-Harm and SafePred is, today, a
sound *positioning and prediction*, not a *demonstrated head-to-head on shared ground*. The engine
exists to produce that demonstration; the demonstration is the next step, and we will not call it done
before it runs.

**The tradeoff, named.** What you give up for hard-to-game safety is generality of domain. The
oracle's whole advantage, exact state, is exactly what restricts it to computer worlds with checkable
state. It buys provable, cheap, injection-robust safety in the place agents are most dangerous (the
shell, the filesystem, the network), and it buys nothing in the place it does not apply (pixels on a
screen). That is a trade most agent platforms would take, but it is a trade, and we say so.

## 7. Related work

**SafePred (arXiv 2602.01725) is the closest prior work and the natural foil.** It also previews an
action's effect before execution, and it also checks selectively on a danger surface. The difference
is the predictor. SafePred uses a learned world model and explicitly disclaims provable or un-gameable
guarantees: safety depends on model accuracy. Verisim replaces the learned predictor with the exact
oracle and the learned danger surface with a model-free covering target, which is what converts
"best-effort" into "hard to game." We see the two as complementary: SafePred's generality where no
oracle exists, verisim's guarantee where one does. Section 4 measures the exact gap SafePred names.

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

First, the external number. Run the four-arm head-to-head live: a real Claude agent on real
Terminal-Bench tasks in Docker, with held-out harm tripwires and a published-style injection layer,
reporting missed-harm, task success, and cost per arm, against OS-Harm's numbers as the anchor and
SafePred as the foil. That converts Section 4 from a hermetic prediction into a measured comparison on
shared ground.

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
