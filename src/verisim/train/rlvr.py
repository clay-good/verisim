"""Stage-2 RLVR: train the world model against the oracle reward (SPEC-2 §5.3).

Stage 1 (``supervised.py``) teacher-forces the model on oracle deltas. Stage 2
closes the loop SPEC.md §6.3 describes: roll the model out, let the *oracle* score
each predicted transition, and reward faithful horizon. The oracle is the
verifiable reward -- no learned reward model -- which is the author's
verifier-as-reward thesis (SPEC.md §8) realized as RL training.

The reward source is :class:`verisim.rl.WorldModelEnv` (SPEC-2 §15): a teacher-forced
episode whose return *is* the faithful horizon ``H_ε``. The optimizer is plain
REINFORCE with a moving-average baseline -- the smallest policy-gradient method that
does the job; the point of v0 is the *mechanism* (train against a verifiable oracle),
not RL sophistication.

The policy emits a delta by **sampling** the grammar-constrained decode (unlike the
greedy :func:`verisim.model.decode.constrained_decode`), tracking the log-prob of the
sampled tokens so the gradient can flow. Sampling shares the exact grammar walk and
termination caps as the greedy path; it is kept separate to avoid entangling the
gradient-tracking sampler with that tested ``@torch.no_grad()`` core.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from verisim.delta.edits import Delta
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle
from verisim.rl.environment import Observation, WorldModelEnv

from ..model.grammar import DeltaGrammar
from ..model.tokenizer import encode_prompt, parse_target
from ..model.transformer import GPT
from ..model.vocab import Vocab

# The repeating grammar leaves and the token that closes each, mirrored from
# ``verisim.model.decode``: capping a run forces the closer so sampling, like the
# greedy decode, always terminates.
_CLOSE_TOKEN = {"PATH_SEG": "</p>", "CONTENT_TOK": "</c>", "STDOUT_TOK": "</o>"}


def sample_delta_with_logprob(
    model: GPT,
    prompt_ids: list[int],
    vocab: Vocab,
    grammar: DeltaGrammar,
    *,
    generator: torch.Generator | None = None,
    max_edits: int = 64,
    max_run: int = 256,
    max_new_tokens: int = 4096,
) -> tuple[Delta, str, Tensor]:
    """Sample a grammar-valid delta; return ``(delta, stdout, summed_log_prob)``.

    Identical grammar walk and termination caps as
    :func:`verisim.model.decode.constrained_decode`, but each token is *sampled*
    from the masked next-token distribution rather than argmax-ed, and the summed
    log-prob of the sampled tokens is returned as a grad-tracking scalar so a
    policy gradient can backprop through it. Forced tokens (a single allowed token)
    contribute log-prob ``0``.
    """
    device = next(model.parameters()).device
    block_size = model.config.block_size

    seq = list(prompt_ids)
    generated: list[int] = []
    stack = grammar.start()
    edits = 0
    prev_top: str | None = None
    run = 0
    log_probs: list[Tensor] = []

    while not grammar.is_accept(stack):
        if len(generated) >= max_new_tokens:
            raise RuntimeError("sample_delta_with_logprob exceeded max_new_tokens")
        top = stack[0]
        run = run + 1 if top == prev_top else 0

        window = seq[-block_size:]
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]

        allowed = grammar.allowed(stack)
        if top == "DELTA" and edits >= max_edits:
            allowed = frozenset({vocab.eos})
        elif top in _CLOSE_TOKEN and run >= max_run:
            allowed = frozenset({vocab.id(_CLOSE_TOKEN[top])})

        mask = torch.full((len(vocab),), float("-inf"), device=device)
        mask[list(allowed)] = 0.0
        log_dist = torch.log_softmax(logits + mask, dim=-1)
        token = int(torch.multinomial(log_dist.exp(), 1, generator=generator).item())
        log_probs.append(log_dist[token])

        if top == "DELTA" and token in vocab.op_ids:
            edits += 1
        stack = grammar.advance(stack, token)
        seq.append(token)
        generated.append(token)
        prev_top = top

    delta, stdout = parse_target(generated, vocab)
    summed = torch.stack(log_probs).sum() if log_probs else torch.zeros((), device=device)
    return delta, stdout, summed


@dataclass
class RLVRStats:
    """Per-optimizer-step training trace (means over the episode batch)."""

    returns: list[float]  # mean faithful horizon H_ε of the sampled episodes
    losses: list[float]  # REINFORCE loss
    baselines: list[float]  # the moving-average return baseline used


def train_rlvr(
    model: GPT,
    vocab: Vocab,
    *,
    oracle: Oracle | None = None,
    config: EnvConfig = DEFAULT_CONFIG,
    driver: str = "weighted",
    seeds: tuple[int, ...] = (0,),
    n_steps: int = 24,
    epsilon: float = 0.0,
    steps: int = 100,
    samples_per_env: int = 4,
    lr: float = 1e-3,
    baseline_decay: float = 0.9,
    max_edits: int = 64,
    max_run: int = 256,
    seed: int = 0,
) -> RLVRStats:
    """REINFORCE the model against the oracle faithful-horizon reward (Stage 2).

    Each optimizer step samples ``samples_per_env`` episodes per env seed through
    :class:`WorldModelEnv` (return = faithful horizon ``H_ε``), then takes a
    vanilla policy-gradient step: maximize ``(G - b) * sum(log π)`` where ``G`` is
    the episode return and ``b`` a moving-average baseline (variance reduction).
    Deterministic given ``seed`` (sampling and env rollout are both seeded). Returns
    the per-step :class:`RLVRStats`.
    """
    oracle = oracle or ReferenceOracle()
    grammar = DeltaGrammar(vocab)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    baseline = 0.0
    stats = RLVRStats(returns=[], losses=[], baselines=[])

    for _ in range(steps):
        episodes: list[tuple[Tensor, float]] = []  # (summed log-prob, return)
        for env_seed in seeds:
            for _ in range(samples_per_env):
                env = WorldModelEnv(
                    driver=driver,
                    seed=env_seed,
                    n_steps=n_steps,
                    epsilon=epsilon,
                    env=config,
                    oracle=oracle,
                    terminate_on_divergence=True,
                )
                observation: Observation | None = env.reset()
                step_log_probs: list[Tensor] = []
                episode_return = 0.0
                while observation is not None:
                    prompt = encode_prompt(observation.state, observation.action, vocab)
                    delta, _, log_prob = sample_delta_with_logprob(
                        model,
                        prompt,
                        vocab,
                        grammar,
                        generator=generator,
                        max_edits=max_edits,
                        max_run=max_run,
                    )
                    transition = env.step(delta)
                    step_log_probs.append(log_prob)
                    episode_return += transition.reward
                    observation = transition.observation
                summed_log_prob = (
                    torch.stack(step_log_probs).sum()
                    if step_log_probs
                    else torch.zeros((), device=next(model.parameters()).device)
                )
                episodes.append((summed_log_prob, episode_return))

        mean_return = sum(ret for _, ret in episodes) / len(episodes)
        loss = -torch.stack(
            [log_prob * (ret - baseline) for log_prob, ret in episodes]
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        baseline = baseline_decay * baseline + (1.0 - baseline_decay) * mean_return

        stats.returns.append(mean_return)
        stats.losses.append(float(loss.item()))
        stats.baselines.append(baseline)

    return stats
