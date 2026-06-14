"""SPEC-22 CU14 / H107 -- the defended incident (torch-free core).

Validated with cheap stand-in models on the contested incident battery: a *blind* model (the real
``M_θ``'s omission profile -- foresees no flow) makes the undefended agent exfiltrate while the
structure defense catches exactly the crown-jewel connects and still completes the mission at a
fraction of the full oracle's cost; a *perfect* oracle model is safe under every defense. The
paranoid defense is safe but abandons the mission. The real trained transformer arm is the
experiment's committed run (torch-gated, not exercised in CI).
"""

from __future__ import annotations

from verisim.acd.closed_loop_net import build_episodes
from verisim.acd.incident_response import (
    IncidentConfig,
    cu14_verdict,
    run_cu14,
    run_incident,
)
from verisim.netoracle.reference import ReferenceNetworkOracle


class _OracleModel:
    """A perfect world-model stand-in: its predicted delta is the oracle's true delta."""

    def __init__(self, oracle: ReferenceNetworkOracle) -> None:
        self._oracle = oracle

    def predict_delta(self, state: object, action: object) -> object:
        return self._oracle.step(state, action).delta  # type: ignore[arg-type]


class _BlindModel:
    """The real M_θ's profile: predicts no change, so it foresees no flow (pure omission)."""

    def predict_delta(self, state: object, action: object) -> list[object]:
        return []


_SMOKE = IncidentConfig.smoke()


def test_contested_episodes_present_both_axes() -> None:
    oracle = ReferenceNetworkOracle()
    episodes = build_episodes(_SMOKE._episode_config(), oracle)
    assert episodes, "smoke battery should yield contested incidents"
    for ep in episodes:
        assert ep.n_goal_opens > 0 and ep.n_danger_opens > 0


def test_blind_undefended_breaches_structure_prevents() -> None:
    """The headline mechanism: the omitter walks the lure free; structure catches exactly it."""
    result = run_cu14(_BlindModel(), _SMOKE)
    led = {x.defense: x for x in result.ledgers}
    assert led["undefended"].breach_rate > 0.0  # the free agent exfiltrates
    assert led["structure"].breach_rate == 0.0  # verifying the jewel connects stops it
    assert led["full_oracle"].breach_rate == 0.0


def test_structure_keeps_mission_and_is_cheaper_than_oracle() -> None:
    result = run_cu14(_BlindModel(), _SMOKE)
    led = {x.defense: x for x in result.ledgers}
    # the one-sided omitter never false-aborts a benign connect: targeting costs the mission nothing
    assert led["structure"].completion_rate == led["full_oracle"].completion_rate
    assert led["structure"].completion_rate >= 0.95
    assert led["structure"].mean_calls < led["full_oracle"].mean_calls
    assert result.cost_ratio_oracle_over_structure > 1.0


def test_paranoid_is_safe_but_abandons_the_mission() -> None:
    result = run_cu14(_BlindModel(), _SMOKE)
    led = {x.defense: x for x in result.ledgers}
    assert led["paranoid"].breach_rate == 0.0  # blocking every connect opens no flow at all
    assert led["paranoid"].completion_rate < 0.5  # ... including the mission's work flows


def test_perfect_model_safe_under_every_defense() -> None:
    oracle = ReferenceNetworkOracle()
    result = run_cu14(_OracleModel(oracle), _SMOKE)
    for x in result.ledgers:
        assert x.breach_rate == 0.0  # a faithful model foresees and aborts the exfil even free


def test_only_structure_in_the_all_good_corner() -> None:
    result = run_cu14(_BlindModel(), _SMOKE)
    verdict = cu14_verdict(result)
    led = {x.defense: x for x in result.ledgers}
    assert led["undefended"].breach_rate > 0.0  # read the numeric off the typed ledger
    assert verdict["paranoid_safe_but_useless"] is True
    assert verdict["structure_in_all_good_corner"] is True
    assert verdict["structure_is_safe"] is True
    assert verdict["structure_keeps_the_mission"] is True
    assert verdict["structure_cheaper_than_oracle"] is True


def test_playback_contrasts_the_same_incident() -> None:
    result = run_cu14(_BlindModel(), _SMOKE)
    # the two playbacks are the same action sequence under two defenses
    assert len(result.playback_undefended) == len(result.playback_structure)
    assert [s.action for s in result.playback_undefended] == [
        s.action for s in result.playback_structure
    ]
    # the chosen incident must show the contrast: undefended breaches, structure does not
    assert any(s.breach for s in result.playback_undefended)
    assert not any(s.breach for s in result.playback_structure)
    # structure only ever spends a call on a connect-to-jewel step
    for s in result.playback_structure:
        if s.consulted:
            assert s.is_connect and s.dst_class == "jewel"


def test_run_incident_records_every_step() -> None:
    oracle = ReferenceNetworkOracle()
    episodes = build_episodes(_SMOKE._episode_config(), oracle)
    run = run_incident(_BlindModel(), oracle, episodes[0], _SMOKE, "structure")
    assert len(run.steps) == len(episodes[0].actions)
    assert run.n_goal == episodes[0].n_goal_opens
