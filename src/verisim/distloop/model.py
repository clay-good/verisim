"""The distributed ``Model`` interface and baseline world models (SPEC-7 §6, §8; DS5).

The tiered propose-verify-correct loop (SPEC-7 §8) is generic over *any* model that predicts a
structured ``DistDelta`` from ``(state, action)`` -- as v0's M5 loop, the network's NW5 loop,
and the host's HC5 loop are generic over their model protocols. The learned `M_θ` (DS4) will drop
into the loop unchanged via this protocol.

Two dependency-free baselines ship now (mirroring every prior world's §8 baselines), so every loop
invariant is testable with no GPU before the learned model:

  - ``DistNullModel`` -- predicts the empty delta ("nothing happens"). It drifts the moment the
    oracle changes anything; the absolute floor and the proof the task is nontrivial.
  - ``DistOracleBackedModel`` -- predicts exactly the oracle's delta: a perfect model that never
    drifts even at ``ρ = 0``, framing what the learned model is *for* (cheap rollout between
    consultations) and serving as the sanity ceiling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from verisim.dist.action import DistAction
from verisim.dist.delta import DistDelta
from verisim.dist.state import DistributedState
from verisim.distoracle.base import DistOracle


@runtime_checkable
class DistModel(Protocol):
    """Predicts the structured log/replica delta an action makes to the cluster state (§6.1)."""

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta: ...


class DistNullModel:
    """Trivial predictor: predicts no change (the empty delta). The drift floor."""

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        return []


class DistOracleBackedModel:
    """Symbolic-only: a perfect model that returns the oracle's own delta. The ceiling."""

    def __init__(self, oracle: DistOracle) -> None:
        self.oracle = oracle

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        return self.oracle.step(state, action).delta


class DistNoisyModel:
    """A tunable, drifting proposer: the oracle's delta corrupted with probability ``noise``.

    The apparatus DS6 needs before the learned ``M_θ`` (DS4) -- a model whose per-step accuracy
    AND whose *error tier* are both controllable, so the distributed `H_ε(ρ)` curve has a meaningful
    interior and the H17 tiered-oracle tradeoff can be shown honestly. The corruption ``mode``
    targets a specific tier of the tiered oracle (§5), which is the whole point of H17:

      - ``gross``  -- flip a replica write to an **out-of-vocab** value: caught by the cheapest
        (metamorphic) tier, so escalating-from-cheap *wins* oracle-dollars over always-bit-exact;
      - ``subtle`` -- corrupt an **in-flight message's** payload (which the cheap/symbolic tiers do
        not inspect for a write): caught only by **bit-exact**, so escalation *loses* -- it pays the
        cheap probes before the bit-exact it always needed.

    A learned model (DS4) replaces this with a *real* error distribution; this synthetic one exists
    to prove the loop + tiered-oracle + oracle-dollar apparatus measures the H17 tradeoff correctly.
    """

    _OUT_OF_VOCAB = "Q"  # deliberately not in any DistConfig.values

    def __init__(self, oracle: DistOracle, *, noise: float, mode: str, rng: object) -> None:
        import random as _random

        if mode not in ("gross", "subtle"):
            raise ValueError(f"unknown mode {mode!r}; choose from ('gross', 'subtle')")
        self.oracle = oracle
        self.noise = noise
        self.mode = mode
        assert isinstance(rng, _random.Random)
        self.rng = rng

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta:
        from verisim.dist.delta import MsgSend, ReplicaWrite

        delta = list(self.oracle.step(state, action).delta)
        if not delta or self.rng.random() >= self.noise:
            return delta  # accept the truth this step
        if self.mode == "gross":
            writes = [i for i, e in enumerate(delta) if isinstance(e, ReplicaWrite)]
            if writes:
                i = self.rng.choice(writes)
                w = delta[i]
                assert isinstance(w, ReplicaWrite)
                delta[i] = ReplicaWrite(w.object_id, w.node_id, w.version, self._OUT_OF_VOCAB)
                return delta
        else:  # subtle: corrupt an in-flight message's payload (bit-exact-only)
            msgs = [i for i, e in enumerate(delta) if isinstance(e, MsgSend)]
            if msgs:
                i = self.rng.choice(msgs)
                m = delta[i]
                assert isinstance(m, MsgSend)
                delta[i] = MsgSend(m.msg_id, m.src, m.dst, m.object_id, m.version,
                                   self._OUT_OF_VOCAB, m.deliver_after)
                return delta
        # fallback (no targetable edit this step): drop one edit so the model still drifts
        delta.pop(self.rng.randrange(len(delta)))
        return delta
