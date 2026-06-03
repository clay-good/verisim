"""The trustless verification protocol: re-execute the oracle, compare bit-for-bit (SPEC-6 §16).

A contribution to the open corpus is a serialized host transition (or a whole trajectory of
them). The verifier holds the *trusted local oracle* and accepts the contribution iff re-running
that oracle on the contributed ``(state, action)`` reproduces the contributed result exactly —
the next state, the bundle delta, and the observation. Nothing about the contributor is trusted;
the oracle is the sole adjudicator, and it is free and deterministic, so verification is exact
rather than probabilistic (the contrast with TOPLOC, §2.9).

Two granularities, mirroring §16's "(a) oracle-verified host trajectories":

  - :func:`verify_transition` — the atomic check on one ``(state, action, next_state[, delta])``.
  - :func:`verify_trajectory` — a contributed trajectory (the :class:`HostTrajectory` record
    shape, HC2): every step's transition must reproduce *and* the steps must chain
    (``next_state[i] == state[i+1]``), so a contributor cannot splice unrelated transitions.

:func:`content_address` gives the corpus its integrity hash (the manifest content-hash of §16):
a contribution's identity is the SHA-256 of its canonical JSON, so the corpus is
content-addressed and tamper-evident independently of the semantic re-execution check.

A malformed contribution (unparseable state/action/delta) is *rejected*, never raised — the
verifier is hostile-input-safe by construction, since contributions come from untrusted parties.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from verisim.host.action import HostParseError, parse_host_action
from verisim.host.delta import delta_to_list
from verisim.host.state import from_canonical_host, to_canonical_host
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle


def content_address(payload: Any) -> str:
    """The SHA-256 content hash of ``payload`` over its canonical JSON (§16 manifest hash).

    Canonical = ``sort_keys`` + compact separators, so the address is stable across dict ordering
    and whitespace — two byte-identical-after-canonicalization contributions share an address.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerificationReport:
    """The verdict on a contribution: did re-execution reproduce it bit-for-bit?

    ``accepted`` is the only thing a gatekeeper needs; ``reason`` and ``first_failure`` localize a
    rejection (which step diverged and on what — next-state, delta, observation, or chaining), and
    ``content_hash`` is the contribution's content address (:func:`content_address`).
    """

    accepted: bool
    reason: str
    n_transitions: int
    n_reproduced: int
    content_hash: str
    first_failure: int | None = None
    mismatches: tuple[str, ...] = field(default_factory=tuple)


def _check_transition(
    state_dict: dict[str, Any],
    action_raw: str,
    next_state_dict: dict[str, Any],
    oracle: HostOracle,
    *,
    claimed_delta: list[dict[str, Any]] | None,
    claimed_observation: dict[str, Any] | None,
) -> list[str]:
    """Re-execute one transition and return the list of mismatching facets (empty == reproduced).

    Parse/deserialize failures are reported as a mismatch, not raised — the input is untrusted.
    """
    try:
        state = from_canonical_host(state_dict)
    except (KeyError, TypeError, ValueError):
        return ["unparseable_state"]
    try:
        action = parse_host_action(action_raw)
    except (HostParseError, AttributeError, TypeError):
        return ["unparseable_action"]
    try:
        result = oracle.step(state, action)
    except Exception:  # an action the oracle rejects is a rejected contribution, not a crash
        return ["oracle_rejected_action"]

    mismatches: list[str] = []
    if to_canonical_host(result.state) != next_state_dict:
        mismatches.append("next_state")
    if claimed_delta is not None and delta_to_list(result.delta) != claimed_delta:
        mismatches.append("delta")
    if claimed_observation is not None:
        true_obs = {"exit_code": result.exit_code, "stdout": result.stdout}
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
    oracle: HostOracle | None = None,
) -> VerificationReport:
    """Verify one contributed ``(state, action, next_state)`` by exact re-execution (§16).

    ``state`` / ``next_state`` are canonical host dicts (:func:`to_canonical_host`); ``action`` is
    a raw syscall string. If ``delta`` (a :func:`delta_to_list` list) or ``observation``
    (``{"exit_code", "stdout"}``) is supplied, those are checked too. Accepted iff *every* supplied
    facet reproduces. The contribution is content-addressed over exactly what was supplied.
    """
    oracle = oracle or ReferenceHostOracle()
    payload: dict[str, Any] = {"state": state, "action": action, "next_state": next_state}
    if delta is not None:
        payload["delta"] = delta
    if observation is not None:
        payload["observation"] = observation
    chash = content_address(payload)

    mismatches = _check_transition(
        state, action, next_state, oracle,
        claimed_delta=delta, claimed_observation=observation,
    )
    if mismatches:
        return VerificationReport(
            accepted=False,
            reason="re-execution diverged: " + ", ".join(mismatches),
            n_transitions=1, n_reproduced=0, content_hash=chash,
            first_failure=0, mismatches=tuple(mismatches),
        )
    return VerificationReport(
        accepted=True, reason="reproduced bit-for-bit",
        n_transitions=1, n_reproduced=1, content_hash=chash,
    )


def verify_trajectory(
    trajectory: dict[str, Any], *, oracle: HostOracle | None = None
) -> VerificationReport:
    """Verify a contributed trajectory record (the HC2 :class:`HostTrajectory` dict shape, §16).

    Every step must (1) reproduce under re-execution and (2) *chain* — the recorded
    ``next_state`` of step ``i`` must equal the recorded ``state`` of step ``i+1``, so a
    contributor cannot stitch together individually-valid transitions that never actually
    followed one another. Accepted iff all steps reproduce and chain. The content hash addresses
    the whole trajectory.
    """
    oracle = oracle or ReferenceHostOracle()
    chash = content_address(trajectory)
    steps = trajectory.get("steps", [])
    n = len(steps)
    if n == 0:
        return VerificationReport(
            accepted=False, reason="empty trajectory (nothing to verify)",
            n_transitions=0, n_reproduced=0, content_hash=chash,
        )

    reproduced = 0
    for i, step in enumerate(steps):
        if i > 0 and steps[i - 1].get("next_state") != step.get("state"):
            return VerificationReport(
                accepted=False,
                reason=f"step {i} does not chain from step {i - 1} (spliced transition)",
                n_transitions=n, n_reproduced=reproduced, content_hash=chash,
                first_failure=i, mismatches=("chaining",),
            )
        mismatches = _check_transition(
            step.get("state", {}), step.get("action", ""), step.get("next_state", {}), oracle,
            claimed_delta=step.get("delta"), claimed_observation=step.get("observation"),
        )
        if mismatches:
            return VerificationReport(
                accepted=False,
                reason=f"step {i} diverged: " + ", ".join(mismatches),
                n_transitions=n, n_reproduced=reproduced, content_hash=chash,
                first_failure=i, mismatches=tuple(mismatches),
            )
        reproduced += 1

    return VerificationReport(
        accepted=True, reason=f"all {n} steps reproduced and chained",
        n_transitions=n, n_reproduced=reproduced, content_hash=chash,
    )
