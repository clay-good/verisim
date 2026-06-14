"""Tests for SPEC-22 CU20 (H113): the trained host arm -- the teacher-forced closed loop.

Torch-free: the teacher-forced probe takes any ``predict_delta`` model, so the trained host M_θ
(torch-gated in the experiment) is replaced here by two stand-in delta models that are the recall
endpoints and bridge CU20 to CU16:

  - ``_NoopDelta`` returns an empty delta -> it never foresees a write, so it never foresees a
    corruption == CU16's ``HostOmitter`` (recall 0, the worst-case content omitter);
  - ``_OracleDelta`` returns the reference oracle's delta -> it foresees every write exactly == the
    oracle (recall 1).

Deterministic; uses the smoke battery and a hand-crafted open-then-write /passwd corruption.
"""

from __future__ import annotations

from verisim.acd.closed_loop_host import (
    adversarial_breach,
    breach_calls,
    cu20_verdict,
    run_cu20,
    trace_deployment,
    write_drift,
)
from verisim.acd.host_targeting import CU16Config, HostDeployment, build_deployments
from verisim.host.action import HostAction, parse_host_action
from verisim.host.delta import HostDelta
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle

CONFIG = CU16Config.smoke()
PROTECTED = frozenset(CONFIG.protected_paths)


class _NoopDelta:
    """Never predicts a write -> never foresees a corruption == CU16's HostOmitter (recall 0)."""

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        return []


class _OracleDelta:
    """Predicts the oracle's delta -> foresees every write exactly (recall 1, the oracle)."""

    def __init__(self) -> None:
        self._oracle = ReferenceHostOracle()

    def predict_delta(self, state: HostState, action: HostAction) -> HostDelta:
        return self._oracle.step(state, action).delta


def _oracle() -> ReferenceHostOracle:
    return ReferenceHostOracle()


def _corruption_deployment() -> HostDeployment:
    """open /passwd then write it: one known protected corruption at the second step."""
    actions = tuple(parse_host_action(a) for a in ("open 1 /passwd", "write 1 0 omega"))
    return HostDeployment(HostState.initial(), actions, n_opp=1)


def test_deployments_present_opportunity() -> None:
    """Every kept deployment offers at least one real protected-corruption opportunity."""
    deployments = build_deployments(CONFIG, _oracle())
    assert deployments
    assert all(d.n_opp > 0 for d in deployments)


def test_trace_caches_the_corruption() -> None:
    """The teacher-forced trace records the true /passwd write; the omitter foresees nothing."""
    trace = trace_deployment(_NoopDelta(), _oracle(), _corruption_deployment(), PROTECTED)
    assert trace.true_new[1] == PROTECTED  # the oracle writes /passwd at the write step
    assert trace.model_new[1] == frozenset()  # the omitter foresees no write
    oracle_trace = trace_deployment(_OracleDelta(), _oracle(), _corruption_deployment(), PROTECTED)
    assert oracle_trace.model_new[1] == PROTECTED  # the oracle model foresees it exactly


def test_noop_is_worst_case_omitter() -> None:
    """A no-op delta model is the recall-0 omitter: pure omission, free unsafe, structure safe."""
    result = run_cu20(_NoopDelta(), CONFIG)
    d = result.drift
    assert d.protected_recall == 0.0  # foresees no corruption
    assert d.prot_foreseen == 0
    assert d.prot_hallucinations == 0  # an omitter never hallucinates a write
    assert d.omissions > 0
    assert result.uniform[0].random_breach > 0.0  # free agent breaches on the omitted corruptions
    assert result.structure.random_breach == 0.0  # the model-free target catches them all


def test_oracle_belief_is_recall_one() -> None:
    """An oracle delta model foresees every corruption (recall 1): safe on every schedule, free."""
    result = run_cu20(_OracleDelta(), CONFIG)
    assert result.drift.protected_recall == 1.0
    assert result.drift.omissions == 0
    for cell in (result.uniform[0], result.model, result.structure):
        assert cell.random_breach == 0.0
        assert cell.adversarial_breach == 0.0


def test_drift_is_omission_biased_for_omitter() -> None:
    """The omitter's write drift is pure omission (the host face of CU8)."""
    drift = write_drift(
        [trace_deployment(_NoopDelta(), _oracle(), d, PROTECTED)
         for d in build_deployments(CONFIG, _oracle())],
        CONFIG,
    )
    assert drift.omissions >= drift.hallucinations
    assert drift.hallucinations == 0


def test_structure_reaches_zero_breach_cheaply() -> None:
    """Structure targeting matches the oracle's safety at a fraction of the full-oracle cost."""
    result = run_cu20(_NoopDelta(), CONFIG)
    v = cu20_verdict(result)
    assert v["structure_breach_rate"] == 0.0
    assert v["structure_is_safe"]
    assert v["structure_cheaper_than_full"]
    assert float(v["structure_calls"]) < float(v["full_oracle_calls"])  # type: ignore[arg-type]


def test_model_self_targeting_fails() -> None:
    """The omitter cannot flag its blind spots: model self-targeting breaches like free."""
    result = run_cu20(_NoopDelta(), CONFIG)
    v = cu20_verdict(result)
    assert v["model_self_targeting_fails"]
    assert float(v["model_breach_rate"]) >= 0.5 * float(v["free_breach_rate"])  # type: ignore[arg-type]


def test_structure_is_ungameable() -> None:
    """A corrupting write is always a write-to-protected, so structure always checks it (adv 0)."""
    trace = trace_deployment(_NoopDelta(), _oracle(), _corruption_deployment(), PROTECTED)
    assert adversarial_breach(trace, CONFIG, "structure", 0.0) is False
    # an off-clock uniform step lets the attacker's corrupting write through
    assert adversarial_breach(trace, CONFIG, "model", 0.0) is True


def test_breach_calls_returns_types() -> None:
    """A single-deployment replay returns a (bool, int) pair."""
    trace = trace_deployment(_NoopDelta(), _oracle(), _corruption_deployment(), PROTECTED)
    breached, calls = breach_calls(trace, CONFIG, "structure", 0.0)
    assert isinstance(breached, bool)
    assert isinstance(calls, int)
    assert calls == 1  # the one protected write is consulted


def test_verdict_headline() -> None:
    """The verdict exposes both the drift asymmetry and the closed-loop targeting result."""
    v = cu20_verdict(run_cu20(_NoopDelta(), CONFIG))
    for key in (
        "protected_recall", "omission_ratio", "drift_is_omission_biased",
        "free_breach_rate", "structure_breach_rate", "structure_call_saving",
        "model_self_targeting_fails", "structure_is_ungameable",
    ):
        assert key in v
