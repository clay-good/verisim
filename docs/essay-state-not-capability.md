# Your agent guardrail is stateless, and that is the whole problem

Bottom line: AI agents that run real commands fail their safety checks not because the model is too dumb to see the danger, but because the guardrail is never shown the one thing it would need to catch it. The danger lives in accumulated state. The guardrail sees one action at a time. No amount of model intelligence closes that gap, because it is not an intelligence gap. It is a plumbing gap. The fix is cheap, and it is not "a smarter model."

I spent a long time trying to prove the opposite, and the data kept telling me I was wrong. This is the writeup of what is actually true, including the parts that embarrass the original thesis.

## The setup

A computer-use agent translates "clean up the logs" into actual shell, then runs it. One mistake deletes the wrong file, weakens a credential, or ships a database to an attacker. So everyone in 2026 ships a guardrail: a denylist, a policy engine, or increasingly a prompted LLM that reads each proposed action and votes allow or block.

There is one genuinely lucky fact about this domain. Unlike vision or language or robotics, a computer's state is exact and checkable for free. You can run the command in a sandbox and diff the filesystem, or compute the reachability of a config graph, or count the records touched. The ground truth is on tap. Call that exact, resettable checker an oracle.

The tempting story, the one I started with, goes: an LLM guardrail reads the *action*; the oracle reads the *effect*; therefore the oracle catches harms the LLM is blind to. It sounds right. It is mostly wrong, and the way it is wrong is the interesting part.

## What I tried to prove, and what actually happened

I built a tool that points an adversary at any guardrail and uses the free oracle to confirm, with certainty, whether a "safe" action was actually dangerous. Then I went hunting for harms a real LLM judge would miss. I picked the three families where action-reading should be weakest:

- **Indirection.** The path is in the command but obfuscated: assembled from pieces, hex-encoded, piped through `rev`, hidden behind a planted symlink.
- **Relational.** A single config edit that opens a transitive path from an untrusted service to the crown-jewel database. No single line names the database.
- **Cumulative.** "Touch at most B distinct sensitive records." No single read is the violation. The harm is the running total.

Here is the embarrassing result. On a real Claude judge:

- Indirection: it decoded every one of thirteen shell-encoding tricks. Recall 1.00, in two different prompt framings.
- Relational: it traced the transitive exposure and blocked it at chain depth 2, 4, 6, and 8, with distractor edges thrown in. 4 out of 4 every time.
- Cumulative, when shown the session history: even a cheap, fast model counted the distinct records correctly up to a budget of 80, across a 180-line history stuffed with re-reads and near-duplicate IDs designed specifically to break naive counting.

I made the counting test progressively nastier, on the cheapest model I could, precisely because I wanted it to fail. It did not. The honest conclusion: a modern model, shown the relevant state, sees these "effect" harms just fine.

So the convenient story is dead. Good. Killing your own favorite hypothesis is the price of admission.

## The one place the guardrail did fail, every single time

There is exactly one configuration where the LLM guardrail failed completely, on both a strong model and a cheap one, at every budget I tried: when it was invoked **statelessly**, shown one action in isolation.

Asked "is reading customer record cust_417 safe?" with no memory of the session, the judge said yes. Every time. Recall 0.00. Of course it did. A single read is individually fine. The budget is a property of the *trajectory*, and the guardrail was never shown the trajectory.

This is not a contrived setup. It is how guardrails are actually deployed. A Claude Code PreToolUse hook, a per-tool-call policy check, most of the agent firewalls shipping right now: they are handed one action and asked for a verdict. Stateless by construction.

So the real finding inverts the original one. The gap is not that models cannot reason about effect-harms. The gap is that the deployed pattern never hands them the state required to do it. It is a state gap, not a capability gap.

That distinction matters because it changes the fix. If it were a capability gap, you would wait for a better model. Because it is a state gap, you fix it today, cheaply, by changing what you show the checker.

## The cheap fix, and why it is not just "use a bigger model"

Once you see it as a state gap, three practical moves fall out. None of them is "state the obvious," and all of them are things real guardrails get wrong right now.

**1. Give the guardrail the state, or it is theater.** If your hook sees one action, it cannot enforce any property that spans actions: budgets, rate limits, blast radius, "no second write to this file," reachability over a graph the agent is editing. Either feed it the session history and the relevant graph, or accept that those harms are simply unguarded. A smarter model in a stateless hook is a smarter model that still says yes to the sixth record because it cannot see the first five.

**2. For countable, relational, or reversible harms, use an exact check, not a model.** "At most B distinct records" is a one-line accumulator: keep a set, check its size. It is exact, it is free, and it never miscounts a 180-line history because it is not counting from text. Asking an LLM to re-read the entire session and recount on every single action, to enforce a property a `set()` enforces perfectly, is paying a sports car to do a bicycle's job. Same story for reachability (a transitive-closure check) and for reversible file harms (execute, diff the protected region, roll back if it changed). The model is the wrong tool not because it is weak but because the task is already solved by something cheaper and certifiable.

**3. Where you do need the model, certify it instead of trusting it.** Some harms have no cheap exact rule, and there a prompted model is genuinely your best option. Fine. But "we used a strong model and it seems to catch things" is not a safety argument. Because computer state gives you a free ground-truth oracle, you can do better: run an adversary that hunts for an action the guardrail passes and the oracle confirms is harmful. If it finds one, you have a concrete bug. If it cannot, you have a measured coverage number with a confidence interval, instead of a vibe. Almost every guardrail vendor today *asserts* completeness. Practically none *measures* it. The measurement is buildable, and the oracle is what makes it free.

This adversarial-certification loop is the part of the work I still believe in without reservation. Pointed at my own deterministic resolver, it found a real soundness bug a human red team had missed: a `printf` format-escape that smuggled a protected path past the checker. The model-quality experiments above are honest nulls; this one is a positive. The difference is that here the oracle is not competing with a model, it is keeping a checker honest.

## What this buys, and what it does not

Strengths. For integrity and availability harms, the ones that are a state change, this is strong and cheap. The dangerous surface is sparse: across 123,195 real Claude Code shell commands, only about 1.1% could ever touch a protected resource, so you wave 99% through and check the rest. Exact checks for countable and relational properties are trivial to implement, exact, and carry a certificate. And the certify-don't-assert loop turns "we think it's safe" into a measurement.

Limitations, stated plainly. This only works where the harm is an observable change in checkable state. Confidentiality is the honest hole: a read that leaks a secret leaves the filesystem byte-identical, and any allowed channel (DNS, timing) can encode a bit, so there is no sparse surface to cover. For exfiltration you can bound the leak bandwidth, not eliminate it, and you should say so. The model-robustness nulls are at the scales I tested, on one model family; a genuinely adversarial history of thousands of confusable IDs might still break a counter-by-prompt, which is one more reason to use an actual counter. And the most load-bearing experiments are run-on-demand against one provider, not a cross-vendor sweep.

Tradeoffs. Exact checks demand that you can name the harm as a property of state: a count, a reachable set, a diff. When you can, they dominate. When the harm is fuzzy or contextual in a way you cannot formalize, you are back to a model judge, paying more and getting a confidence interval instead of a guarantee. The whole approach trades generality of domain for hardness of guarantee. It is excellent in the computer, and it makes no claim about a mouse-clicking GUI agent whose "harm" only a human can adjudicate.

## The takeaway for anyone building this

If you ship an agent guardrail, ask one question first: does it see the state the harm lives in? If your check is per-action and the harm is cumulative, relational, or contextual, your model's intelligence is irrelevant, because you are not showing it the evidence. Give it the state, or better, replace it with the exact, cheap, certifiable check the property already admits. Save the model for the harms that genuinely need judgment, and even then, do not trust the judge: use the free oracle to try to break it, and report what you find.

Not "run the real computer and read the truth," which is obvious. The useful version is sharper: the truth your guardrail needs is usually a piece of accumulated state it is never handed, the fix is plumbing and a counter rather than a smarter model, and the one durable advantage of computer-world safety is that you can adversarially prove a checker honest instead of taking its word.

That is a smaller claim than I set out to make. It is also true, which I have come to prefer.

---

*Reproductions and the full negative results behind this essay: the LLM-guardrail audit (`docs/llm-guardrail-audit-run.md`), the cumulative-harm hunt (`docs/cumulative-horizon-run.md`), the sparse-surface measurement on real traffic and the soundness-bug discovery (`docs/paper.md`, RA21 and RA24), and the adversarial self-review that forced most of this honesty (`docs/review.md`).*
