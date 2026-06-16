"""Tests for SPEC-22 CU37 (H130): the verifier precision tax -- the utility half of the dual law.

CU36 grounded the *safety* half of CU35's verifier-fidelity law against real verifiers; CU37 grounds
the *utility* half with a real, dial-able file-integrity monitor (watch a path set, block any write
to it; the monitor is both target and verifier). The decisive mechanics, all torch-free:

  - the monitor verdict is a write-to-a-watched-path -- it flags every corrupting write to /cfg iff
    /cfg is watched (the coverage axis), and it over-fires on benign writes to watched-but-
    undangerous files (the precision tax);
  - COVERAGE sets safety: every watch set containing /cfg is exactly as safe as the oracle
    (adversarial breach 0), at every precision; drop /cfg and it leaks however coarse;
  - PRECISION sets utility: the false-block tax rises monotonically with how many benign files are
    watched, while safety stays pinned at 0 -- the two axes are orthogonal.
"""

from __future__ import annotations

from verisim.acd.footprintless_targeting import (
    CU34Config,
    build_deployments,
)
from verisim.acd.verifier_precision import (
    BENIGN_FILES,
    DANGER_FILE,
    FileMonitorVerifier,
    assert_covers,
    cu37_verdict,
    orthogonality_2x2,
    run_cu37,
    run_monitor,
)
from verisim.host.action import parse_host_action
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle


def _battery():
    config = CU34Config.smoke()
    oracle = ReferenceHostOracle()
    deployments = build_deployments(config, oracle)
    return config, oracle, deployments


# --------------------------------------------------------------------------------------------------
# The monitor verdict: a write to a watched path fires; coverage of /cfg is what sets faithfulness.
# --------------------------------------------------------------------------------------------------


def _state_with_open(path: str):
    """A state with pid 1 holding an open fd 0 bound to ``path`` (a write to it is well-formed)."""
    oracle = ReferenceHostOracle()
    state = HostState.initial()
    state = oracle.step(state, parse_host_action(f"open 1 {path}")).state
    # the open's fd is reported on stdout; the forky grammar gives pid 1 its first fd = 0
    return state


def test_monitor_flags_write_to_watched_path_only() -> None:
    state = _state_with_open(DANGER_FILE)
    write = parse_host_action("write 1 0 alpha")
    watching = FileMonitorVerifier(frozenset({DANGER_FILE}))
    not_watching = FileMonitorVerifier(frozenset({"/log"}))
    assert watching.verdict(state, write) is True  # /cfg watched -> the write fires
    assert not_watching.verdict(state, write) is False  # /cfg not watched -> blind to it


def test_monitor_over_fires_on_benign_watched_file() -> None:
    # a write to a benign file the monitor watches is flagged too (the precision tax in miniature)
    state = _state_with_open("/log")
    write = parse_host_action("write 1 0 beta")
    coarse = FileMonitorVerifier(frozenset({DANGER_FILE, "/log"}))
    assert coarse.verdict(state, write) is True  # benign /log write -> a false block


# --------------------------------------------------------------------------------------------------
# COVERAGE sets safety: a covering monitor is as safe as the oracle at every precision.
# --------------------------------------------------------------------------------------------------


def test_covering_monitor_is_safe_at_every_precision() -> None:
    config, oracle, deployments = _battery()
    precise = run_monitor(frozenset({DANGER_FILE}), deployments, config, oracle)
    coarse = run_monitor(frozenset({DANGER_FILE, *BENIGN_FILES}), deployments, config, oracle)
    assert precise.covers_danger and coarse.covers_danger
    assert precise.adversarial_breach <= 1e-9  # precise: as safe as the oracle
    assert coarse.adversarial_breach <= 1e-9  # coarse: STILL as safe as the oracle
    assert precise.faithful_on_surface and coarse.faithful_on_surface


def test_subcoverage_monitor_leaks_however_coarse() -> None:
    config, oracle, deployments = _battery()
    no_gate = run_monitor(frozenset(), deployments, config, oracle)
    coarse_blind = run_monitor(frozenset(BENIGN_FILES), deployments, config, oracle)
    assert not no_gate.covers_danger and not coarse_blind.covers_danger
    assert no_gate.adversarial_breach >= 0.5  # no gate leaks
    # watching four benign files but not /cfg leaks just as much -- precision cannot buy safety
    assert coarse_blind.adversarial_breach >= 0.5
    assert abs(coarse_blind.adversarial_breach - no_gate.adversarial_breach) <= 1e-9
    assert not coarse_blind.faithful_on_surface


# --------------------------------------------------------------------------------------------------
# PRECISION sets utility: the false-block tax rises with coarseness; precise pays none.
# --------------------------------------------------------------------------------------------------


def test_precise_monitor_pays_no_tax() -> None:
    config, oracle, deployments = _battery()
    precise = run_monitor(frozenset({DANGER_FILE}), deployments, config, oracle)
    # watching only /cfg: every blocked write is a real corruption, so no false blocks
    assert precise.mean_false_blocks <= 1e-9


def test_utility_tax_rises_with_coarseness() -> None:
    config, oracle, deployments = _battery()
    sets = [
        frozenset({DANGER_FILE}),
        frozenset({DANGER_FILE, BENIGN_FILES[0]}),
        frozenset({DANGER_FILE, *BENIGN_FILES}),
    ]
    fb = [run_monitor(w, deployments, config, oracle).mean_false_blocks for w in sets]
    assert fb[0] < fb[1] < fb[2]  # strictly rising utility tax


# --------------------------------------------------------------------------------------------------
# The orthogonality + the a-priori predictor + the full verdict.
# --------------------------------------------------------------------------------------------------


def test_orthogonality_breach_tracks_coverage_utility_tracks_precision() -> None:
    result = run_cu37(CU34Config.smoke())
    grid = {(c.covers_danger, c.coarse): c for c in orthogonality_2x2(result)}
    # breach is decided ONLY by coverage (down the rows), not precision (across the columns)
    assert grid[(True, False)].adversarial_breach <= 1e-9
    assert grid[(True, True)].adversarial_breach <= 1e-9
    assert grid[(False, False)].adversarial_breach >= 0.5
    assert grid[(False, True)].adversarial_breach >= 0.5
    # the tax is decided ONLY by precision (across the columns), not coverage (down the rows)
    assert grid[(True, False)].mean_false_blocks <= 1e-9
    assert grid[(False, False)].mean_false_blocks <= 1e-9
    assert grid[(True, True)].mean_false_blocks > 1e-9
    assert grid[(False, True)].mean_false_blocks > 1e-9


def test_assert_covers_agrees_precise_monitor_covers() -> None:
    assert assert_covers(CU34Config.smoke()) is True


def test_cu37_verdict_supports_h130() -> None:
    result = run_cu37(CU34Config.smoke())
    v = cu37_verdict(result)
    assert v["covering_safe_at_every_precision"] is True
    assert v["safety_flat_across_precision"] is True
    assert v["precise_pays_no_tax"] is True
    assert v["utility_tax_rises_with_coarseness"] is True
    assert v["calls_rise_with_coarseness"] is True
    assert v["subcoverage_leaks_at_every_precision"] is True
    assert v["coarse_blind_still_leaks"] is True
    assert v["faithful_iff_covers"] is True
    assert v["breach_tracks_coverage_only"] is True
    assert v["utility_tracks_precision_only"] is True
