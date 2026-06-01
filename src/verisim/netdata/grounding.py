"""Oracle-grounded training targets and the decidable/residual partition (SPEC-8 §3, OG1).

SPEC-8's case is that the deterministic oracle belongs in the *bulk* of the cake (self-supervised
pretraining), not only the cherry (RLVR). Before any GPU touches that claim, the **deterministic
target machinery** has to ship and be property-tested -- the same NW0-NW3 discipline that put the
oracle/metric/loop core in place before the model (SPEC-5 §13). This module is that machinery
(milestone OG1). It emits, for any ``(state, action)``:

  - the **true next-state target** ``s' = O(s, a)`` and its delta (what an oracle-anchored target,
    SPEC-8 §4.1, regresses onto -- ground truth, not a learned EMA encoder);
  - the **exact divergence target** ``d(s', ŝ)`` for any prediction ``ŝ`` -- equal to
    :func:`~verisim.netmetrics.divergence.divergence` *by construction* (the latent-distance target
    of §4.1 is pinned to this external referent, which is why there is nothing for the embedding to
    collapse toward);
  - the **decidable/residual partition** of ``s'`` (SPEC-8 §3): the facts the oracle fixes given the
    *observation* (``D``, "verify, do not learn") versus the facts left genuinely uncertain (``R``,
    "learn, because no cheap oracle resolves them"). The training objective masks ``D`` and spends
    gradient on ``R`` (§4.2) -- *even nature offloads* the decidable part to the world rather than
    storing it in the genome.

The partition is tied to an **observation** -- a set of observed host ids (the NW5 host-probe
model, SPEC-5 §5.3). A next-state fact is decidable iff every host it references is observed;
clock/exit are global and always decidable. Under full observation (``observed_hosts is None``)
``D`` is all of ``s'`` and ``R`` is empty -- the degenerate fully-observed case SPEC-8 §3 names.
Pure and dependency-free, like the rest of the deterministic core -- no torch, no GPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.edits import NetDelta
from verisim.netmetrics.divergence import Fact, divergence, net_facts
from verisim.netoracle.base import NetOracle


def fact_hosts(fact: Fact) -> frozenset[str]:
    """The host ids a typed state fact references (empty for the global clock/exit facts)."""
    kind = fact[0]
    if kind in ("up", "svc"):
        return frozenset({fact[1]})  # type: ignore[arg-type]
    if kind in ("fw", "link"):
        return frozenset({fact[1], fact[2]})  # type: ignore[arg-type]
    if kind == "flow":
        return frozenset({fact[1], fact[2]})  # type: ignore[arg-type]
    return frozenset()  # "\x00clock" / "\x00exit" -- global, decidable by construction


def is_decidable(fact: Fact, observed_hosts: frozenset[str] | None) -> bool:
    """``True`` iff the oracle fixes ``fact`` given the observation (SPEC-8 §3 the ``D`` regime).

    Full observation (``observed_hosts is None``) makes every fact decidable. Otherwise a fact is
    decidable iff every host it references is observed; the global clock/exit facts reference no
    host and are always decidable.
    """
    if observed_hosts is None:
        return True
    return fact_hosts(fact) <= observed_hosts


@dataclass(frozen=True)
class OracleTargets:
    """The oracle-grounded targets for one ``(state, action)`` (SPEC-8 §4.1-4.2, OG1).

    ``decidable`` (``D``) and ``residual`` (``R``) partition ``net_facts(next_state)`` exactly:
    their union is ``s'`` and their intersection is empty, by construction.
    """

    next_state: NetworkState
    delta: NetDelta
    decidable: frozenset[Fact]
    residual: frozenset[Fact]

    def divergence_to(self, predicted: NetworkState) -> float:
        """The exact divergence target ``d(s', ŝ)`` -- equal to ``netmetrics.divergence`` (§4.1)."""
        return divergence(self.next_state, predicted)


def oracle_targets(
    state: NetworkState,
    action: NetAction,
    oracle: NetOracle,
    observed_hosts: Iterable[str] | None = None,
) -> OracleTargets:
    """Emit the oracle-grounded targets + decidable/residual partition for ``(state, action)``.

    ``observed_hosts`` is the partial view (the NW5 probe set); ``None`` means full observation.
    """
    result = oracle.step(state, action)
    observed = None if observed_hosts is None else frozenset(observed_hosts)
    facts = frozenset(net_facts(result.state))
    decidable = frozenset(f for f in facts if is_decidable(f, observed))
    residual = facts - decidable
    return OracleTargets(
        next_state=result.state,
        delta=result.delta,
        decidable=decidable,
        residual=residual,
    )


__all__ = ["OracleTargets", "fact_hosts", "is_decidable", "oracle_targets"]
