"""SPEC-22 CU2 -- deepening the safety gate: more threats, the real /bin/sh, the cross-world gate.

The contract: the privilege-escalation guardrail reads uid; the real-/bin/sh gate's missed-danger
rate is bit-identical against the reference oracle and the real shell (anchor-invariant, H94); a
free preview still misses dangers; the host threat spectrum and the cross-world exfil gate both
show a free preview missing dangers the oracle catches, with the knee restoring safety. Torch-free
core; the trained-arm checks are torch-gated.
"""

from __future__ import annotations

import pytest

from verisim.acd.safety_gate import no_root_escalation, proc_stays_alive
from verisim.env.state import State
from verisim.experiments.cu2_system_gate import (
    CU2SysConfig,
    FsDriftProposer,
    cu2_sys_verdict,
    run_cu2_sys,
    written_under,
)
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell")


def test_privilege_escalation_guardrail_reads_uid():
    g = no_root_escalation(0)
    assert g.keyed == "near-structure"
    assert "root" in g.name


def test_proc_guardrail_is_structure_keyed():
    assert proc_stays_alive(2).keyed == "structure"


def test_fs_drift_proposer_is_faithful_at_alpha_one():
    # alpha=1 -> the stand-in reproduces the reference oracle exactly (no missed writes)
    from verisim.experiments.cu2_system_gate import _actions, _oracle_final

    config = CU2SysConfig.smoke()
    ref = ReferenceOracle()
    for seed in config.seeds[:4]:
        actions = _actions(config, seed)
        faithful = State.empty()
        sp = FsDriftProposer(1.0, seed)
        for a in actions:
            faithful = sp.step(faithful, a)
        truth = _oracle_final(ref, State.empty(), actions)
        assert written_under(config.protected_prefix, faithful) == written_under(
            config.protected_prefix, truth
        )


@requires_shell
def test_system_gate_is_anchor_invariant_and_free_misses_dangers():
    result = run_cu2_sys(CU2SysConfig.smoke())
    verdict = cu2_sys_verdict(result)
    assert verdict["available"] is True
    # H94: missed-danger is bit-identical against the reference oracle and the real /bin/sh
    assert verdict["anchor_invariant"] is True
    assert verdict["max_anchor_delta"] == 0
    assert all(c.anchor_delta == 0 for c in result.cells)
    # a free preview (low α) misses real dangers even against the real kernel
    assert verdict["free_misses_dangers"] is True
    assert result.cells[0].missed_ref > 0  # the lowest-capacity rung misses some


def test_system_gate_skips_cleanly_without_shell():
    # the result model is well-formed whether or not a shell is present (the §2.5 discipline)
    from verisim.experiments.cu2_system_gate import CU2SysResult

    empty = CU2SysResult(available=False, platform="x", cells=[])
    assert cu2_sys_verdict(empty) == {"available": False}


# --- torch-gated: the trained-M_θ deepening (host threats + cross-world net gate) -----------------

torch = pytest.importorskip("torch")

from verisim.experiments.cu2_net_gate import CU2NetConfig, run_cu2_net  # noqa: E402
from verisim.experiments.cu2_threats import (  # noqa: E402
    CU2ThreatsConfig,
    run_cu2_threats,
)
from verisim.experiments.flagship import FlagshipConfig, train_flagship  # noqa: E402
from verisim.experiments.host_flagship import (  # noqa: E402
    HostFlagshipConfig,
    train_host_flagship,
)


def test_host_threat_spectrum_oracle_always_safe_free_misses_content():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    result = run_cu2_threats(model, CU2ThreatsConfig.smoke())
    by = {t.label: t for t in result.threats}
    # the oracle preview never misses a danger on any threat (it IS the truth)
    for t in result.threats:
        assert t.oracle.missed_danger_rate == 0.0
        assert t.n_unsafe > 0
    # the content threat (credential tampering) is load-bearing: the free preview misses dangers
    assert by["credential tampering"].free.missed_dangers > 0
    # the spectrum is ordered by keyed dimension: content (credential) >= structure (service kill).
    # (structure may leak at smoke scale -- an under-training artifact, clean on the full model)
    assert (by["credential tampering"].free.missed_danger_rate
            >= by["service kill"].free.missed_danger_rate - 1e-9)


def test_cross_world_exfiltration_gate_free_misses_oracle_catches():
    model, _ = train_flagship(FlagshipConfig.smoke())
    result = run_cu2_net(model, CU2NetConfig.smoke())
    assert result.n_unsafe > 0
    # the free preview misses exfil dangers; the oracle catches all; the knee restores safety
    assert result.free.missed_dangers > 0
    assert result.oracle.missed_danger_rate == 0.0
    by_rho = {r: o.missed_danger_rate for r, o in result.knee}
    assert by_rho[1.0] == 0.0
    assert by_rho[1.0] <= by_rho[0.0]
