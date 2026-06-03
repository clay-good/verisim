"""Verified-contribution protocol tests (SPEC-6 §16, HC8): trustless by re-execution.

The thesis the oracle makes possible: a contributed host trajectory is accepted iff re-running
the deterministic oracle reproduces it bit-for-bit. These tests pin (1) genuine oracle-generated
contributions are accepted, (2) every kind of tampering — next_state, delta, observation, and
spliced/non-chaining transitions — is caught, and (3) malformed contributions from untrusted
parties are *rejected*, never raised. Plus the content-address integrity hash is stable and
order-independent.
"""

from __future__ import annotations

import copy
from typing import Any

from verisim.contrib import (
    content_address,
    verify_trajectory,
    verify_transition,
)
from verisim.host import DEFAULT_HOST_CONFIG, HostState, to_canonical_host
from verisim.hostdata import generate_host_trajectory
from verisim.hostoracle import ReferenceHostOracle


def _trajectory(seed: int = 7, n_steps: int = 24) -> dict[str, Any]:
    oracle = ReferenceHostOracle()
    traj = generate_host_trajectory(oracle, DEFAULT_HOST_CONFIG, "forky", seed, n_steps)
    return traj.to_dict()


def _first_step(seed: int = 7) -> dict[str, Any]:
    step: dict[str, Any] = _trajectory(seed)["steps"][0]
    return step


# -- genuine contributions are accepted ---------------------------------------


def test_genuine_trajectory_is_accepted() -> None:
    report = verify_trajectory(_trajectory())
    assert report.accepted
    assert report.n_reproduced == report.n_transitions > 0
    assert report.first_failure is None


def test_genuine_transition_is_accepted() -> None:
    step = _first_step()
    report = verify_transition(
        step["state"], step["action"], step["next_state"],
        delta=step["delta"], observation=step["observation"],
    )
    assert report.accepted
    assert report.mismatches == ()


def test_acceptance_holds_across_drivers_and_seeds() -> None:
    oracle = ReferenceHostOracle()
    for driver in ("uniform", "forky", "adversarial"):
        for seed in (1, 2, 3):
            traj = generate_host_trajectory(oracle, DEFAULT_HOST_CONFIG, driver, seed, 20)
            assert verify_trajectory(traj.to_dict()).accepted


# -- tampering is caught ------------------------------------------------------


def test_tampered_next_state_is_rejected() -> None:
    step = _first_step()
    forged = copy.deepcopy(step["next_state"])
    forged["last_exit"] = forged["last_exit"] + 99  # a lie the oracle will not reproduce
    report = verify_transition(step["state"], step["action"], forged, delta=step["delta"])
    assert not report.accepted
    assert "next_state" in report.mismatches


def test_tampered_delta_is_rejected() -> None:
    step = _first_step()
    forged_delta = [{"op": "SetExit", "exit_code": 123}]
    report = verify_transition(
        step["state"], step["action"], step["next_state"], delta=forged_delta
    )
    assert not report.accepted
    assert "delta" in report.mismatches


def test_tampered_observation_is_rejected() -> None:
    step = _first_step()
    forged_obs = {"exit_code": step["observation"]["exit_code"] + 1, "stdout": "lie"}
    report = verify_transition(
        step["state"], step["action"], step["next_state"], observation=forged_obs
    )
    assert not report.accepted
    assert "observation" in report.mismatches


def test_spliced_trajectory_is_rejected() -> None:
    # Two genuine trajectories; graft a real step from B into A so each transition is individually
    # valid but the chain breaks — the splice the chaining check exists to catch.
    traj = _trajectory(seed=7)
    other = _trajectory(seed=8)
    traj["steps"][3] = copy.deepcopy(other["steps"][3])
    report = verify_trajectory(traj)
    assert not report.accepted
    assert report.first_failure is not None
    assert "chaining" in report.mismatches or report.first_failure <= 3


def test_mutated_step_in_trajectory_is_localized() -> None:
    traj = _trajectory()
    traj["steps"][5]["next_state"]["next_pid"] += 1
    report = verify_trajectory(traj)
    assert not report.accepted
    assert report.first_failure == 5
    assert report.n_reproduced == 5


# -- hostile input is rejected, never raised ----------------------------------


def test_malformed_state_is_rejected_not_raised() -> None:
    report = verify_transition({"garbage": True}, "fork 1", {})
    assert not report.accepted
    assert "unparseable_state" in report.mismatches


def test_malformed_action_is_rejected_not_raised() -> None:
    step = _first_step()
    report = verify_transition(step["state"], "not_a_real_syscall 1 2 3", step["next_state"])
    assert not report.accepted
    assert report.mismatches  # rejected on parse or oracle-rejection, not crashed


def test_empty_trajectory_is_rejected() -> None:
    report = verify_trajectory({"steps": []})
    assert not report.accepted
    assert report.n_transitions == 0


# -- content addressing -------------------------------------------------------


def test_content_address_is_order_independent() -> None:
    a = {"state": {"x": 1, "y": 2}, "action": "fork 1"}
    b = {"action": "fork 1", "state": {"y": 2, "x": 1}}
    assert content_address(a) == content_address(b)


def test_content_address_changes_with_payload() -> None:
    step = _first_step()
    r1 = verify_transition(step["state"], step["action"], step["next_state"])
    r2 = verify_transition(step["state"], step["action"], step["next_state"], delta=step["delta"])
    assert r1.content_hash != r2.content_hash  # the delta is part of what was contributed


# -- the serialization inverse the protocol rests on --------------------------


def test_from_canonical_host_round_trips() -> None:
    from verisim.host import from_canonical_host

    oracle = ReferenceHostOracle()
    state = HostState.initial()
    traj = generate_host_trajectory(oracle, DEFAULT_HOST_CONFIG, "adversarial", 42, 30)
    # the boot state and every recorded state must survive a canonical round-trip
    assert to_canonical_host(from_canonical_host(to_canonical_host(state))) == to_canonical_host(
        state
    )
    for step in traj.steps:
        d = step["state"]
        assert to_canonical_host(from_canonical_host(d)) == d
