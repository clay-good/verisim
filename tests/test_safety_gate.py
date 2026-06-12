"""SPEC-22 CU1 / H93 -- the agent-in-the-loop safety gate.

The contract: a guardrail labels plans safe/unsafe exactly, the oracle preview never misses a danger
(it IS the truth), a blind preview cannot catch content dangers, the ρ-grounded extremes recover
oracle/free, and -- with the trained host `M_θ` -- a free preview misses real dangers on the content
guardrail (the agent executes credential-corrupting plans) while the cheap knee drives missed-danger
to zero. Torch-free core.
"""

from __future__ import annotations

import pytest

from verisim.acd.host_integrity import make_workload, written_files
from verisim.acd.safety_gate import (
    evaluate_free_gate,
    free_gate,
    grounded_rollout_final,
    label_plans,
    no_write_to,
    oracle_gate,
    proc_stays_alive,
    rollout_final,
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle


class _BlindModel:
    """Predicts no change -- never writes, never kills (the degenerate free preview)."""

    def predict_delta(self, state: HostState, action: HostAction) -> list[object]:
        return []


def test_guardrails_carry_their_keyed_dimension():
    assert no_write_to("/passwd").keyed == "content"
    assert proc_stays_alive(2).keyed == "structure"


def test_content_guardrail_labels_match_truth():
    oracle = ReferenceHostOracle()
    g = no_write_to("/passwd")
    plans = label_plans(g, range(700, 716), 16, oracle=oracle)
    # the label is exactly the guardrail on the oracle's true final state
    from verisim.acd.host_integrity import oracle_step

    step = oracle_step(oracle)
    for p in plans:
        final = rollout_final(step, p.start, p.actions)
        assert p.true_safe == ("/passwd" not in written_files(final))
    # the battery is a genuine mix (some plans really do overwrite /passwd)
    assert any(not p.true_safe for p in plans)


def test_oracle_preview_never_misses_a_danger():
    # the oracle preview IS the ground truth, so it cannot miss or false-block
    oracle = ReferenceHostOracle()
    g = no_write_to("/passwd")
    plans = label_plans(g, range(700, 730), 16, oracle=oracle)
    out = evaluate_free_gate(oracle_gate(oracle), g, plans)
    assert out.missed_dangers == 0 and out.false_blocks == 0
    assert out.missed_danger_rate == 0.0


def test_blind_preview_misses_every_content_danger():
    # a preview that predicts no writes thinks every plan is safe -> executes every real danger
    oracle = ReferenceHostOracle()
    g = no_write_to("/passwd")
    plans = label_plans(g, range(700, 730), 16, oracle=oracle)
    out = evaluate_free_gate(free_gate(_BlindModel()), g, plans)
    assert out.n_unsafe > 0
    assert out.missed_danger_rate == pytest.approx(1.0)  # never predicts a write -> never aborts
    assert out.false_blocks == 0


def test_grounded_extremes_recover_oracle_and_free():
    oracle = ReferenceHostOracle()
    start, actions = make_workload(701, 16, oracle=oracle)
    blind = _BlindModel()
    # ρ=1 -> the preview is the true final state (full oracle), |actions| calls
    final1, calls1 = grounded_rollout_final(blind, oracle, start, actions, 1.0)
    from verisim.acd.host_integrity import oracle_step

    truth = rollout_final(oracle_step(oracle), start, actions)
    assert written_files(final1) == written_files(truth) and calls1 == len(actions)
    # ρ=0 -> free (no oracle calls)
    _, calls0 = grounded_rollout_final(blind, oracle, start, actions, 0.0)
    assert calls0 == 0


def test_safety_outcome_rates_well_formed():
    oracle = ReferenceHostOracle()
    g = no_write_to("/passwd")
    plans = label_plans(g, range(700, 712), 14, oracle=oracle)
    out = evaluate_free_gate(free_gate(_BlindModel()), g, plans)
    assert out.n_unsafe + out.n_safe == len(plans)
    assert 0.0 <= out.missed_danger_rate <= 1.0
    assert 0.0 <= out.false_block_rate <= 1.0


# --- torch-gated: the trained-M_θ deployment story ------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.cu_safety_gate import (  # noqa: E402
    CU1Config,
    knee_rho,
    run_cu1,
)
from verisim.experiments.host_flagship import (  # noqa: E402
    HostFlagshipConfig,
    train_host_flagship,
)


def test_free_preview_misses_content_dangers_and_knee_restores_safety():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    result = run_cu1(model, CU1Config.smoke())
    # the content guardrail has truly-unsafe plans and the free preview misses some of them
    assert result.content.n_unsafe > 0
    assert result.content.free.missed_dangers > 0  # the agent executed destructive plans
    # the oracle preview never misses
    assert result.content.oracle.missed_danger_rate == 0.0
    # the ρ-grounded preview at full budget recovers the oracle (missed-danger 0 at ρ=1)
    by_rho = {r: o.missed_danger_rate for r, o in result.content.knee}
    assert by_rho[1.0] == 0.0
    # grounding helps: the rate at ρ=1 is no worse than free (ρ=0)
    assert by_rho[1.0] <= by_rho[0.0]
    # the cheapest-safe-gate ρ is in the swept grid
    assert knee_rho(result.content) in {r for r, _ in result.content.knee}
