"""Tests for SPEC-22 CU36 (H129): the grounded verifier -- CU35's fidelity law, real verifiers.

CU35 proved the verifier-fidelity dual coverage law with an abstract hash-coin verifier; CU36
grounds it against two REAL, structurally-defined partial verifiers over the CU34 host CIA battery.
The decisive mechanics, all torch-free:

  - the state-diff verifier (CU34's after-the-fact detector, as a before-commit verifier) is
    faithful on integrity + availability (they mutate a watched resource) and an on-surface omitter
    on confidentiality (a read mutates nothing);
  - the structure verifier (the SPEC-20 structure/content boundary) is faithful only on
    availability (a kill changes the process table), blind on integrity (content) + confidentiality;
  - LOCALIZATION: a real verifier is exactly as safe as the perfect oracle where it is faithful on
    the surface, and exactly as blind as no gate wherever it is not -- decided a priori by whether
    the danger mutates a channel the verifier observes (the structural predictor).
"""

from __future__ import annotations

from verisim.acd.footprintless_targeting import (
    CU34Config,
    build_deployments,
    provision,
)
from verisim.acd.grounded_verifier import (
    NoGateVerifier,
    StateDiffVerifier,
    StructureVerifier,
    cu36_verdict,
    run_cu36,
    run_verifier_leg,
)
from verisim.host.action import parse_host_action
from verisim.hostoracle.reference import ReferenceHostOracle


def _battery():
    config = CU34Config.smoke()
    oracle = ReferenceHostOracle()
    deployments = build_deployments(config, oracle)
    _, daemons = provision(config, oracle)
    watched = frozenset({config.secret_path, config.integrity_path})
    return config, oracle, deployments, daemons, watched


# --- the verifiers' per-action verdicts are the grammar facts, not coins -------------------------


def _worker_with_open_files(config: CU34Config, oracle: ReferenceHostOracle):
    """Fork a worker that holds an open fd onto the integrity file and the secret file."""
    start, daemons = provision(config, oracle)
    s = oracle.step(start, parse_host_action("fork 1")).state
    worker = max(s.procs)
    res = oracle.step(s, parse_host_action(f"open {worker} {config.integrity_path}"))
    s, cfg_fd = res.state, int(res.stdout)
    res = oracle.step(s, parse_host_action(f"open {worker} {config.secret_path}"))
    s, sec_fd = res.state, int(res.stdout)
    return s, worker, cfg_fd, sec_fd, daemons


def test_state_diff_verifier_sees_writes_and_kills_but_not_reads():
    """The cheap monitor flags a corrupting write and a daemon kill, not a footprintless read."""
    config = CU34Config()
    oracle = ReferenceHostOracle()
    watched = frozenset({config.secret_path, config.integrity_path})
    s, worker, cfg_fd, sec_fd, daemons = _worker_with_open_files(config, oracle)
    v = StateDiffVerifier(oracle, watched, daemons)

    write = parse_host_action(f"write {worker} {cfg_fd} junk")
    read = parse_host_action(f"read {worker} {sec_fd}")
    kill = parse_host_action(f"kill {worker} {min(daemons)}")
    assert v.verdict(s, write) is True  # a write flips watched-file content presence
    assert v.verdict(s, kill) is True  # a kill ends a daemon
    assert v.verdict(s, read) is False  # a read mutates nothing -> blind (the footprintless leg)


def test_structure_verifier_sees_kills_but_not_writes_or_reads():
    """The structure monitor flags a proc-table change (kill), is blind to content (write)/read."""
    config = CU34Config()
    oracle = ReferenceHostOracle()
    v = StructureVerifier(oracle)
    s, worker, cfg_fd, sec_fd, daemons = _worker_with_open_files(config, oracle)

    write = parse_host_action(f"write {worker} {cfg_fd} junk")
    read = parse_host_action(f"read {worker} {sec_fd}")
    kill = parse_host_action(f"kill {worker} {min(daemons)}")
    assert v.verdict(s, kill) is True  # a kill changes the process table
    assert v.verdict(s, write) is False  # a write changes only file content -> blind
    assert v.verdict(s, read) is False  # a read changes nothing -> blind


# --- the per-leg safety follows the grammar-predicted fidelity -----------------------------------


def test_state_diff_safe_on_footprinted_legs_blind_on_confidentiality():
    config, oracle, deployments, daemons, watched = _battery()
    v = StateDiffVerifier(oracle, watched, daemons)
    integ = run_verifier_leg(v, "state-diff", "integrity", deployments, config, oracle)
    avail = run_verifier_leg(v, "state-diff", "availability", deployments, config, oracle)
    conf = run_verifier_leg(v, "state-diff", "confidentiality", deployments, config, oracle)
    assert integ.faithful_on_surface and integ.adversarial_breach <= 1e-9
    assert avail.faithful_on_surface and avail.adversarial_breach <= 1e-9
    assert not conf.faithful_on_surface and conf.adversarial_breach >= 0.5


def test_structure_safe_only_on_availability():
    config, oracle, deployments, _daemons, _watched = _battery()
    v = StructureVerifier(oracle)
    avail = run_verifier_leg(v, "structure", "availability", deployments, config, oracle)
    integ = run_verifier_leg(v, "structure", "integrity", deployments, config, oracle)
    conf = run_verifier_leg(v, "structure", "confidentiality", deployments, config, oracle)
    assert avail.faithful_on_surface and avail.adversarial_breach <= 1e-9
    assert not integ.faithful_on_surface and integ.adversarial_breach >= 0.5
    assert not conf.faithful_on_surface and conf.adversarial_breach >= 0.5


def test_no_gate_leaks_every_leg():
    config, oracle, deployments, _daemons, _watched = _battery()
    v = NoGateVerifier()
    for leg in ("availability", "integrity", "confidentiality"):
        res = run_verifier_leg(v, "no gate", leg, deployments, config, oracle)
        assert res.adversarial_breach >= 0.5


def test_blind_leg_is_exactly_as_blind_as_no_gate():
    """Localization's negative half: an on-surface omitter is no better than no gate at all."""
    config, oracle, deployments, _daemons, _watched = _battery()
    structure = run_verifier_leg(
        StructureVerifier(oracle), "structure", "integrity", deployments, config, oracle
    )
    none = run_verifier_leg(NoGateVerifier(), "no gate", "integrity", deployments, config, oracle)
    assert abs(structure.adversarial_breach - none.adversarial_breach) <= 1e-9


def test_structural_predictor_matches_empirical_fidelity():
    """faithful_on_surface == (the danger mutates a channel the verifier observes), every cell."""
    result = run_cu36(CU34Config.smoke())
    assert all(c.faithful_on_surface == c.structurally_faithful for c in result.cells)


def test_cu36_verdict_headline():
    v = cu36_verdict(run_cu36(CU34Config.smoke()))
    assert v["exact_safe_everywhere"] is True
    assert v["no_gate_leaks_everywhere"] is True
    assert v["localization_holds_everywhere"] is True
    assert v["structural_predictor_exact"] is True
    assert v["state_diff_globally_partial_locally_safe"] is True
    assert v["structure_globally_partial_locally_safe"] is True
    assert v["all_targets_cover"] is True
