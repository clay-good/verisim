"""Tests for SPEC-22 CU38 (H131): the heterogeneous verifier ensemble -- dual of CU24's composite.

CU24 proved the target-side composition theorem with a perfect verifier; CU38 proves the matching
verifier-side theorem with the union target held fixed (covering). The decisive mechanics, all
torch-free over the CU34 host CIA battery:

  - a read-audit monitor catches the footprintless confidentiality danger (a disclosing read of a
    secret-bound fd) that a state-diff monitor is structurally blind to;
  - an ensemble OR-combines its members, so its faithful surface is the UNION of theirs -- it is
    faithful on a leg iff SOME member is (the composition theorem);
  - NO single cheap monitor is safe on the composite; the ensemble {state-diff, read-audit} tiles
    CIA and is exactly as safe as the perfect oracle; dropping a member re-opens exactly its leg.
"""

from __future__ import annotations

from verisim.acd.footprintless_targeting import CU34Config, provision
from verisim.acd.grounded_verifier import StateDiffVerifier
from verisim.acd.verifier_ensemble import (
    _LEGS,
    EnsembleVerifier,
    ReadAuditVerifier,
    _cell,
    cu38_verdict,
    run_cu38,
)
from verisim.host.action import parse_host_action
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle

# --------------------------------------------------------------------------------------------------
# The read-audit monitor: catches a disclosing read of a secret fd; the OR-ensemble combinator.
# --------------------------------------------------------------------------------------------------


def _state_reading_secret(secret_path: str):
    """A state where pid 1 has written content to ``secret_path`` and holds an fd open to it."""
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    state = oracle.step(state, parse_host_action(f"open 1 {secret_path}")).state
    state = oracle.step(state, parse_host_action("write 1 0 topsecret")).state
    return state


def test_read_audit_flags_disclosing_read_only() -> None:
    secret = frozenset({"/secret"})
    state = _state_reading_secret("/secret")
    read_secret = parse_host_action("read 1 0")  # discloses /secret
    audit = ReadAuditVerifier(secret)
    assert audit.verdict(state, read_secret) is True
    # a state-diff monitor watching /secret is blind to the read (it mutates no state)
    _, daemons = provision(CU34Config.smoke(), ReferenceHostOracle())
    state_diff = StateDiffVerifier(ReferenceHostOracle(), secret, daemons)
    assert state_diff.verdict(state, read_secret) is False


def test_ensemble_blocks_iff_any_member_blocks() -> None:
    secret = frozenset({"/secret"})
    state = _state_reading_secret("/secret")
    read_secret = parse_host_action("read 1 0")
    audit = ReadAuditVerifier(secret)
    _, daemons = provision(CU34Config.smoke(), ReferenceHostOracle())
    state_diff = StateDiffVerifier(ReferenceHostOracle(), secret, daemons)
    # state-diff alone misses the read; the ensemble with read-audit catches it (the OR)
    assert EnsembleVerifier((state_diff,)).verdict(state, read_secret) is False
    assert EnsembleVerifier((state_diff, audit)).verdict(state, read_secret) is True


# --------------------------------------------------------------------------------------------------
# The grid: no single monitor safe on the composite; the ensemble == the perfect oracle.
# --------------------------------------------------------------------------------------------------


def _smoke_result():
    return run_cu38(CU34Config.smoke())


def test_no_single_partial_monitor_is_safe_on_composite() -> None:
    result = _smoke_result()
    for v in ("state-diff", "structure", "read-audit"):
        assert _cell(result, v, "composite").adversarial_breach >= 0.5


def test_state_diff_leaks_exactly_the_footprintless_confidentiality_leg() -> None:
    result = _smoke_result()
    assert _cell(result, "state-diff", "integrity").adversarial_breach <= 1e-9
    assert _cell(result, "state-diff", "availability").adversarial_breach <= 1e-9
    assert _cell(result, "state-diff", "confidentiality").adversarial_breach >= 0.5


def test_ensemble_is_exactly_as_safe_as_the_perfect_oracle() -> None:
    result = _smoke_result()
    for scope in (*_LEGS, "composite"):
        assert _cell(result, "ensemble", scope).adversarial_breach <= 1e-9
    # bit-identical to the exact oracle on the composite
    ens = _cell(result, "ensemble", "composite").adversarial_breach
    exact = _cell(result, "exact oracle", "composite").adversarial_breach
    assert abs(ens - exact) <= 1e-9


def test_dropping_a_member_reopens_exactly_its_leg() -> None:
    result = _smoke_result()
    drop_read = "ensemble-no-read-audit"
    assert _cell(result, drop_read, "confidentiality").adversarial_breach >= 0.5
    assert _cell(result, drop_read, "integrity").adversarial_breach <= 1e-9
    assert _cell(result, drop_read, "availability").adversarial_breach <= 1e-9
    drop_sd = "ensemble-no-state-diff"
    assert _cell(result, drop_sd, "integrity").adversarial_breach >= 0.5
    assert _cell(result, drop_sd, "availability").adversarial_breach >= 0.5
    assert _cell(result, drop_sd, "confidentiality").adversarial_breach <= 1e-9


# --------------------------------------------------------------------------------------------------
# The composition theorem + the a-priori predictor + the full verdict.
# --------------------------------------------------------------------------------------------------


def test_composition_theorem_ensemble_faithful_iff_a_member_is() -> None:
    result = _smoke_result()
    for leg in _LEGS:
        sd = _cell(result, "state-diff", leg).faithful_on_surface
        ra = _cell(result, "read-audit", leg).faithful_on_surface
        ens = _cell(result, "ensemble", leg).faithful_on_surface
        assert ens == (sd or ra)


def test_structural_predictor_matches_observed_channels() -> None:
    result = _smoke_result()
    for c in result.cells:
        assert c.faithful_on_surface == c.structurally_faithful


def test_cu38_verdict_supports_h131() -> None:
    v = cu38_verdict(_smoke_result())
    assert v["exact_safe_composite"] is True
    assert v["no_gate_leaks_composite"] is True
    assert v["no_single_partial_safe_composite"] is True
    assert v["state_diff_leaks_only_confidentiality"] is True
    assert v["read_audit_leaks_integrity_and_availability"] is True
    assert v["ensemble_safe_every_leg"] is True
    assert v["ensemble_safe_composite"] is True
    assert v["ensemble_matches_exact_composite"] is True
    assert v["drop_read_reopens_only_confidentiality"] is True
    assert v["drop_statediff_reopens_integrity_availability"] is True
    assert v["composition_theorem"] is True
    assert v["structural_predictor_exact"] is True
    assert v["all_targets_cover"] is True
