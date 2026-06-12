# The Verifier Is the Primitive: Where Scale Cannot Save a Computer-Use World Model

A world model is a program that predicts what a system does next. Feed it the current state and an action, and it returns the state that follows. Chain those predictions and you can plan, imagine, and train an agent without touching the real system. That is the dream behind every model-based RL paper, every "learn a simulator and act inside it" pitch, and every demo where a robot rehearses in its own head.

There is one stubborn problem, and it is the same one every time: the predictions compound. A model that is 99% accurate per step is wrong about a quarter of the time after thirty steps, and catastrophically wrong long before a real task ends. The field has a name for the wall, exposure bias, and a long shelf of partial cures. None of them remove it.

This essay is about a different move. Not "make the model better until the wall recedes," but "put a verifier in the loop and ask, precisely, where the verifier is load-bearing and where it is not." For one specific and important domain, computer use, that question has an exact answer, because that domain has something almost no other domain has: a free, perfect oracle.

The punchline, measured rather than asserted: faithfulness to ground truth is load-bearing for control exactly when the task depends on the dynamics the model gets wrong, and that boundary moves with scale in a predictable way, and it does not move all the way to zero. There is an irreducible residue of content that stays unlearnable at every scale we can reach, and on that residue a verifier is not a crutch you outgrow. It is a permanent primitive.

## The unfair advantage: computer worlds carry their own ground truth

Start with what makes computer use special. If you want to know what `rm -rf build/` does to a filesystem, you do not need a learned model. You run it. The kernel is the ground truth, it is deterministic enough to be reset and replayed, and it is sitting right there.

That is the oracle: a function that returns the exact next state for any state and action, for free, at every step. Reinforcement learning calls this a verifiable reward, and it is the engine behind the recent jump in coding and math models. A unit test either passes or it does not. A theorem checker either accepts the proof or it does not. No human in the loop, no reward model to game, just a yes or a no from reality.

Verisim, the project this essay reports on, takes that idea and points it at the world model itself rather than the policy. The oracle is not just a reward at the end of an episode. It is a step-by-step truth source you can consult mid-rollout to ask "is the model still right?" and, when it has drifted, to snap it back. The loop is propose, verify, correct: the cheap learned model proposes the next state, the oracle verifies it, and on a divergence you re-anchor to truth and pay one oracle call.

Most of the world does not get this gift. A vision model predicting the next video frame has no oracle for "what the next frame really is." A language model has no oracle for "what a helpful answer really is." Computer use, specifically the shell, filesystem, and process slice of it, is the rare place where the oracle exists and is cheap. That is the whole game. It is also the honest boundary of the claim: this is command-line computer use, not clicking around a browser. Graphical computer use has no oracle, and the project says so plainly rather than pretending otherwise.

## The wall, stated honestly

Here is the first hard result, and it is a negative.

Train a faithful world model of a small computer environment, freeze it, and let it run free, predicting step after step with no correction. Measure the faithful horizon: how many steps until it diverges from the truth beyond a tolerance. For the most structurally careful model architecture in the project, at exact tolerance, that horizon is zero. Not "short." Zero. It drifts on the first step it is asked to predict something it cannot copy.

The curve that captures this is `H_ε(ρ)`, the faithful horizon as a function of the consultation rate ρ, the fraction of steps where you let the model peek at the oracle. The hope was a knee: a regime where a little consultation buys a lot of horizon, so you pay 20% oracle calls and get 80% of the way to a perfect simulator. What the data showed instead, across the first world, was a floor and a cliff. Flat and useless until ρ approaches 1, then a sharp jump only when you are basically consulting the oracle every step. No knee. No bargain.

That could have been the end. Instead it became the method. The project treats every result, knee or no knee, as information, because the oracle makes both branches exact. A positive result tells you the bargain exists. A negative result tells you, with certainty rather than hand-waving, that in this regime faithful simulation costs oracle calls roughly one for one. Both are publishable. Both are true. The discipline has a name in the specs, and it is worth stealing: all data is good data, as long as the measurement is exact.

## The boundary: faithfulness matters exactly where the model is wrong

The floor-and-cliff negative provoked the right question. If a free-running model is useless as a simulator, is it useless for everything? Or are there tasks where its drift simply does not matter?

This is where the project's sharpest result lives, and it is a boundary, drawn twice and confirmed across worlds.

Split the dynamics of a computer environment into two kinds. Structure is the part the model learns cleanly: the process tree, which process spawned which, which file descriptors are open, the reachability graph of a network. Content is the part it drifts on: the actual bytes written to a file, the specific flows across a link, the values that depend on inputs the model never saw.

Now the claim. World-model faithfulness is load-bearing for a control task if and only if the task's optimal policy depends on content, the stuff the model gets wrong, rather than structure, the stuff it gets right.

Read that again, because it is more useful than it looks. It says you can often deploy a drifting, imperfect, free-running world model and lose nothing, as long as your task only cares about structure. A defender deciding which processes to kill does not need the oracle, because the model tracks the process tree faithfully. A defender deciding which files an attacker corrupted does need the oracle, because that is content, and the model drifts on content.

The project measured this as a predictive-defense game. An adversarial workload touches a set of objects (processes spawned, files written), and a defender, working from a model's forward predictions, protects the objects it expects to be hit. Score it on how much of the true set it caught. A faithful predictor, the oracle, catches everything. A free predictor, the learned model, catches whatever its drift leaves intact. The gap between them is the load-bearing signal. On structure tasks the gap is roughly zero. On content tasks it is large and grows with horizon. Six structural nulls, two content positives, each measured exactly against ground truth. The boundary is not a vibe. It is a number, and the number is reproducible.

## Scaling the boundary into a law

A boundary measured on one tiny model is a vignette. The obvious objection writes itself: sure, your 110-thousand-parameter toy drifts on content, but a real model would learn it, and your whole distinction evaporates at scale.

So the project scaled it. Not by training one giant model, which it cannot afford, but by sweeping a capacity ladder through the same measurement and asking not "does the boundary hold" but "where does the boundary go."

The result is a law about motion. As capacity grows, dimensions fall to the model in order of how learnable they are. A dimension that is "content the model drifts on" at small scale can become "structure the model has learned" at larger scale, and when it does, the set of tasks for which faithfulness is load-bearing shrinks. The load-bearing frontier recedes.

Three findings give the law its spine, and the third is the one that matters most.

First, the gradient is universal. At every rung of the ladder, on every world tested (host, network, distributed), the content gap exceeds the structure gap. Content is always harder than structure. That ordering never broke.

Second, the recession is not universal, and the project says so where a weaker write-up would have hidden it. On the host and network worlds, structural tasks lose their gap first, exactly as predicted. But on the distributed world, where even the partition structure is genuinely hard to learn, the structure gap persists rather than receding first. So "structural-first recession" is a property of worlds where structure happens to be trivially learnable, not a law of nature. The universal thing is the gradient, not the order of the recession. That is a refinement the data forced, and banking it honestly is what makes the rest trustworthy.

Third, and this is the headline: the frontier recedes toward zero but not to zero. There is a deepest-content task, keyed on the actual bytes written, where the content depends on input the model has no way to learn. Its accuracy plateaus below threshold at every scale the ladder reaches. The gap stays open. Verification stays load-bearing. Forever, as far as any reachable scale can tell.

This is the irreducible residue, and it is the whole argument in one observation. If every dimension eventually crossed into "learned" with enough scale, then faithfulness would be a small-model crutch and the field could scale past the verifier. The project pre-registered that branch as the deepest possible negative, the one that would reshape its own central claim, and then measured the opposite. The residue holds. There is content you cannot scale away, and exactly there, a verifier is permanent.

The honest caveat rides along: the ladder is single-GPU-scale, not frontier-scale, so the claim is about the trajectory across the measured range, with the extrapolation flagged rather than asserted. The residue might fall to a model larger than anything tested. The project measures the plateau it can reach and states the risk out loud. That is the difference between a law and a hope.

## The cheap forecast, and why it matters operationally

A second result turns the boundary from a curiosity into a tool. Measuring whether faithfulness is load-bearing for a task requires the full faithful-versus-free ablation, which is expensive. But there is a shortcut: just free-run the model once and measure how much it drifts on each dimension. That cheap drift profile forecasts the expensive load-bearing verdict with a rank correlation of +0.965, and it even forecasts the cost of buying the gap back (+0.717).

Operationally, this means a defender can look at a cheap one-shot measurement of a model and predict, without running any task, which decisions that model can be trusted to make on its own and which ones require pulling the oracle into the loop. Cheap predicts expensive. The profile is the forecast.

## Does it survive a real kernel?

Everything so far is measured against a deterministic reference oracle, a clean in-repo model of a shell. The fair objection: maybe the whole structure-versus-content story is an artifact of your tidy simulator, and a real `/bin/sh` on a real kernel, with all its messy nondeterminism, would smear the boundary.

It does not. On the grammar where a real shell and the reference oracle were proven to agree bit-for-bit, the load-bearing gap measured against the real kernel equals the gap measured against the reference, exactly, to a maximum difference of zero. The gradient holds under the real shell. The residue stays load-bearing under the real shell. The cheap drift forecasts the gap under the real shell at a rank correlation of +1.000. The law is about real computer-use dynamics, not a model of them.

This matters for cyber defense specifically, because the entire value proposition of a defensive world model is that rehearsing in the model transfers to the real system. A boundary that held only in simulation would be a parlor trick. A boundary that is provably anchor-invariant against the real kernel is an engineering fact you can build on.

## The artifact: a benchmark that knows when faithfulness mattered

A measurement nobody can reproduce is a claim, not a contribution. So the project ships the apparatus as `verisim-cue`, a verifiable computer-use environment and benchmark with three properties no other computer-use benchmark has at once.

Ground-truth next-state, for free, at every step, plus a real-kernel anchor. A spectrum of tasks ordered from pure structure to deep content, each scored by an exact faithful-versus-free predictor, so the benchmark reports not just whether a model succeeds but whether faithfulness was load-bearing for that success. And a scale axis built in, so the environment is the substrate of a law rather than a single number.

The last piece to land, and the one written most recently, answers the question any adopter asks before trusting a benchmark: does it actually discriminate? A leaderboard that ranks every model the same, or ranks them by coin flip, is worse than useless. The test, borrowed from the project's own faithfulness benchmark, is strict. Score a ladder of models of known, graded fidelity. Then check two things: that the ranking is stable when you resample the random seeds (a high Kendall rank correlation between disjoint seed splits), and, harder, that every adjacent pair of models separates by more than the noise between them. Not just "the best beats the worst," which is trivial, but "the benchmark resolves neighbors."

It does. The rank correlation is +1.000, and every adjacent tier clears twice its own noise. The benchmark stably ranks computer-use world models by faithfulness, and the ranking is carried entirely by the structure-to-content gradient: the structural tasks saturate for every model and separate no one, while content recall climbs smoothly from zero at the floor to one at the ceiling. The leaderboard and the scaling frontier turn out to be two views of the same object. One asks, for these models, which ranks highest. The other asks, across capacity, where the boundary sits. Both key on the same content drift.

## The honest ledger: strengths, limits, and what is still owed

The case for this program is that it makes a measurement no oracle-free field can make. Faithfulness-versus-budget curves with ground-truth labels, a structure-content boundary drawn exactly and twice, a scale law with a named irreducible residue, all anchored to a real kernel and packaged as a benchmark that provably discriminates. Where most world-model work can only show you a model that seems to work, this work tells you precisely where the model can be trusted and where it cannot, and proves the where against reality.

The limits are equally plain, and the project states them itself rather than waiting to be caught. The domain is command-line, not graphical, computer use, because that is the slice with an oracle. The capacity ladder is single-GPU-scale, so the scale law describes a trajectory across a reachable range and flags its extrapolation rather than claiming frontier scale. And the load-bearing results live, for now, on controlled stand-in models of graded fidelity, because the one expensive thing the program has deferred is training the real flagship model at scale. That deferral recurs across the specs as a single honest phrase, "only the trained model arm remains," and it is the program's main outstanding debt. The apparatus is proven on CPU so that the GPU run is a one-command config swap rather than a rewrite, which is the right way to owe that debt, but it is owed.

The tradeoff underneath all of it is the one worth naming. The oracle is what makes every result exact, and the oracle is also what bounds the domain. You get certainty in exchange for scope. Computer use is large enough to matter, and the oracle is rare enough to be a real moat, so the trade is a good one. But it is a trade, not a free lunch, and the project reads better for admitting it.

## Why a defender should care

Strip away the apparatus and here is what a security engineer takes home.

You can build a cheap, fast, learned simulator of a host or a network, let an autonomous defender rehearse inside it, and you do not have to make that simulator perfect. You have to make it faithful on the dimensions your defense actually keys on, and you can measure, cheaply and in advance, which dimensions those are. For the structural decisions, process containment, reachability, privilege, the free-running model is enough and the oracle is idle. For the content decisions, what was written, what flowed, what was exfiltrated, you keep the oracle in the loop, and the good news is that it is cheap to do so and provably faithful to the real kernel when you do.

And for the deepest content, the genuinely unlearnable bytes, you stop trying to scale your way out. You accept that verification is a primitive there, you budget for it, and you build the verifier in rather than betting on a bigger model that the data says is not coming. That acceptance is not defeat. It is the rare luxury of knowing exactly where the hard part is, and pointing your one expensive tool straight at it.

The wall that every world model hits is real. This work does not knock it down. It does something more useful: it draws the line on the wall where you can climb and where you cannot, proves the line against a real kernel, and hands you a cheap way to find that line for any model you are handed. In a field full of simulators that seem to work, knowing exactly where yours does not is the whole advantage.
