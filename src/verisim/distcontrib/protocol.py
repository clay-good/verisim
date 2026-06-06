"""The trustless verification protocol for distributed traces (SPEC-7 §16).

A contribution to the open corpus is a serialized distributed transition (or a whole trajectory of
them). The verifier holds the *trusted local oracle* and accepts the contribution iff re-running
that oracle on the contributed ``(state, action)`` reproduces the contributed result — nothing about
the contributor is trusted; the oracle is the sole adjudicator, free and deterministic, so
verification is exact rather than probabilistic (the contrast with Prime Intellect's TOPLOC
heuristic, SPEC-6 §2.9). This is the distributed analogue of the host :mod:`verisim.contrib`.

The distributed world adds the one thing the host protocol could not need: **tiered acceptance**.
Bit-exact reproduction is the host's only mode, and it is the strict tier here too — but in the
distributed world the full bit-exact next-state is sometimes *intractable* (W7: under a weak
consistency model many admissible interleavings produce different bytes, so a contributor running an
equally-valid schedule reproduces the *consistency behavior* without reproducing the *bytes*). So a
contribution names the **tier** it should be checked at, and is accepted iff the corresponding
oracle tier does not refute it (SPEC-7 §5, §16): ``bit_exact`` demands byte-for-byte reproduction
(and optionally the delta + client result), while ``metamorphic`` / ``cycle`` / ``symbolic`` accept
*any* next-state the cheap tier admits as legal under the declared model. *Accepted iff it
reproduces (or admits the declared consistency model)* — the exact wording of §16, made concrete.

Three granularities, mirroring §16's "(a) oracle-verified cluster trajectories":

  - :func:`verify_transition` — the atomic check on one ``(state, action, next_state[, delta,
    observation])`` at a named tier.
  - :func:`verify_trajectory` — a contributed trajectory: every step must reproduce/admit at its
    tier *and* the steps must chain (``next_state[i] == state[i+1]``), so a contributor cannot
    splice unrelated transitions.
  - :func:`content_address` — the SHA-256 of the contribution's canonical JSON (the §16 manifest
    content-hash), so the corpus is content-addressed and tamper-evident independent of re-exec.

A malformed contribution (unparseable state/action) is *rejected*, never raised — the verifier is
hostile-input-safe by construction, since contributions come from untrusted parties. Pure,
dependency-free, GPU-free (the deterministic core; Tier-B real-DST verification is the later §16
increment).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from verisim.dist.action import DistAction, parse_dist_action
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import delta_to_list
from verisim.dist.serialize import from_canonical, to_canonical
from verisim.dist.state import DistributedState
from verisim.distoracle.tiers import TIER_COSTS, TieredOracle


def content_address(payload: Any) -> str:
    """The SHA-256 content hash of ``payload`` over its canonical JSON (§16 manifest hash).

    Canonical = ``sort_keys`` + compact separators, so the address is stable across dict ordering
    and whitespace — two byte-identical-after-canonicalization contributions share an address.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerificationReport:
    """The verdict on a contribution: did re-execution reproduce (or admit) it at the named tier?

    ``accepted`` is the only thing a gatekeeper needs; ``reason`` and ``first_failure`` localize a
    rejection (which step diverged and on what — next-state, delta, observation, chaining, or
    tier-refutation), ``tier`` is the tier it was checked at, ``oracle_cost`` is the cumulative
    nominal tier cost paid (§5), and ``content_hash`` is the contribution's content address.
    """

    accepted: bool
    reason: str
    tier: str
    n_transitions: int
    n_reproduced: int
    content_hash: str
    oracle_cost: int = 0
    first_failure: int | None = None
    mismatches: tuple[str, ...] = field(default_factory=tuple)


def _check_transition(
    state_dict: dict[str, Any],
    action_raw: str,
    next_state_dict: dict[str, Any],
    tiered: TieredOracle,
    tier: str,
    *,
    claimed_delta: list[dict[str, Any]] | None,
    claimed_observation: dict[str, Any] | None,
) -> list[str]:
    """Re-execute/verify one transition at ``tier``; return the mismatching facets (empty == ok).

    Parse/deserialize failures are reported as a mismatch, not raised — the input is untrusted.
    For ``bit_exact`` the next-state (and any supplied delta/observation) must reproduce byte-for-
    byte; for a cheaper tier, the contributed next-state is accepted iff that tier does not refute
    it as illegal under the declared model (the W7 / §16 tiered-acceptance path).
    """
    try:
        state = from_canonical(state_dict)
    except (KeyError, TypeError, ValueError):
        return ["unparseable_state"]
    try:
        next_state = from_canonical(next_state_dict)
    except (KeyError, TypeError, ValueError):
        return ["unparseable_next_state"]
    try:
        action = parse_dist_action(action_raw)
    except (ValueError, AttributeError, TypeError):
        return ["unparseable_action"]
    try:
        verdict = tiered.check(tier, state, action, next_state)
    except Exception:  # an action the oracle rejects is a rejected contribution, not a crash
        return ["oracle_rejected_action"]

    mismatches: list[str] = []
    if verdict.refuted:
        mismatches.append(f"{tier}_refuted")
        return mismatches  # the tier rejected the next-state; cheaper facets are moot
    # the named tier admits the next-state; for the strict tier, also check the optional facets
    if tier == "bit_exact" and (claimed_delta is not None or claimed_observation is not None):
        result = tiered.reference.step(state, action)
        if claimed_delta is not None and delta_to_list(result.delta) != claimed_delta:
            mismatches.append("delta")
        if claimed_observation is not None:
            true_obs = {"status": result.status, "value": result.value}
            if claimed_observation != true_obs:
                mismatches.append("observation")
    return mismatches


def verify_transition(
    state: dict[str, Any],
    action: str,
    next_state: dict[str, Any],
    *,
    delta: list[dict[str, Any]] | None = None,
    observation: dict[str, Any] | None = None,
    tier: str = "bit_exact",
    oracle: TieredOracle | None = None,
    config: DistConfig = DEFAULT_DIST_CONFIG,
) -> VerificationReport:
    """Verify one contributed ``(state, action, next_state)`` at ``tier`` by re-execution (§16).

    ``state`` / ``next_state`` are canonical distributed dicts (:func:`to_canonical`); ``action`` is
    a raw action string. At ``bit_exact`` the next-state must reproduce byte-for-byte and any
    supplied ``delta`` (a :func:`delta_to_list` list) or ``observation`` (``{"status", "value"}``)
    is checked too; at a cheaper tier the next-state need only be *admitted* as legal (the W7 path).
    Accepted iff every checked facet passes. Content-addressed over exactly what was supplied.
    """
    if tier not in TIER_COSTS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIER_COSTS)}")
    tiered = oracle or TieredOracle(config)
    payload: dict[str, Any] = {"state": state, "action": action, "next_state": next_state,
                               "tier": tier}
    if delta is not None:
        payload["delta"] = delta
    if observation is not None:
        payload["observation"] = observation
    chash = content_address(payload)

    mismatches = _check_transition(
        state, action, next_state, tiered, tier,
        claimed_delta=delta, claimed_observation=observation,
    )
    if mismatches:
        return VerificationReport(
            accepted=False, reason="re-execution diverged: " + ", ".join(mismatches),
            tier=tier, n_transitions=1, n_reproduced=0, content_hash=chash,
            oracle_cost=TIER_COSTS[tier], first_failure=0, mismatches=tuple(mismatches),
        )
    verb = "reproduced bit-for-bit" if tier == "bit_exact" else f"admitted by the {tier} tier"
    return VerificationReport(
        accepted=True, reason=verb, tier=tier,
        n_transitions=1, n_reproduced=1, content_hash=chash, oracle_cost=TIER_COSTS[tier],
    )


def verify_trajectory(
    trajectory: dict[str, Any],
    *,
    tier: str = "bit_exact",
    oracle: TieredOracle | None = None,
    config: DistConfig = DEFAULT_DIST_CONFIG,
) -> VerificationReport:
    """Verify a contributed trajectory record (``{"steps": [...]}``, §16) at ``tier``.

    Every step must (1) reproduce/admit under re-execution and (2) *chain* — the recorded
    ``next_state`` of step ``i`` must equal the recorded ``state`` of step ``i+1``, so a contributor
    cannot stitch together individually-valid transitions that never actually followed one another.
    A per-step ``tier`` overrides the default for that step. Accepted iff all steps pass and chain;
    the content hash addresses the whole trajectory.
    """
    if tier not in TIER_COSTS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIER_COSTS)}")
    tiered = oracle or TieredOracle(config)
    chash = content_address(trajectory)
    steps = trajectory.get("steps", [])
    n = len(steps)
    if n == 0:
        return VerificationReport(
            accepted=False, reason="empty trajectory (nothing to verify)",
            tier=tier, n_transitions=0, n_reproduced=0, content_hash=chash,
        )

    reproduced = 0
    cost = 0
    for i, step in enumerate(steps):
        step_tier = step.get("tier", tier)
        if step_tier not in TIER_COSTS:
            return VerificationReport(
                accepted=False, reason=f"step {i} names unknown tier {step_tier!r}",
                tier=tier, n_transitions=n, n_reproduced=reproduced, content_hash=chash,
                oracle_cost=cost, first_failure=i, mismatches=("unknown_tier",),
            )
        if i > 0 and steps[i - 1].get("next_state") != step.get("state"):
            return VerificationReport(
                accepted=False,
                reason=f"step {i} does not chain from step {i - 1} (spliced transition)",
                tier=tier, n_transitions=n, n_reproduced=reproduced, content_hash=chash,
                oracle_cost=cost, first_failure=i, mismatches=("chaining",),
            )
        mismatches = _check_transition(
            step.get("state", {}), step.get("action", ""), step.get("next_state", {}),
            tiered, step_tier,
            claimed_delta=step.get("delta"), claimed_observation=step.get("observation"),
        )
        cost += TIER_COSTS[step_tier]
        if mismatches:
            return VerificationReport(
                accepted=False, reason=f"step {i} diverged: " + ", ".join(mismatches),
                tier=tier, n_transitions=n, n_reproduced=reproduced, content_hash=chash,
                oracle_cost=cost, first_failure=i, mismatches=tuple(mismatches),
            )
        reproduced += 1

    return VerificationReport(
        accepted=True, reason=f"all {n} steps reproduced/admitted and chained",
        tier=tier, n_transitions=n, n_reproduced=reproduced, content_hash=chash, oracle_cost=cost,
    )


def transition_record(
    state: DistributedState, action: DistAction, next_state: DistributedState
) -> dict[str, Any]:
    """A canonical contributable transition record (the wire shape :func:`verify_transition` reads).

    A convenience for producers (and tests): serialize a ``(state, action, next_state)`` triple a
    contributor would submit. The action is rendered back to its raw grammar string.
    """
    return {
        "state": to_canonical(state),
        "action": action.raw,
        "next_state": to_canonical(next_state),
    }
