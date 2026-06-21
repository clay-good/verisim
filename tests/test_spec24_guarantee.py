"""Tests for SPEC-24 Direction C (H165/H166/H170): the graded guarantee.

Hermetic and deterministic. They pin:

  - H165: the exhaustive depth-bounded enumerator yields *exactly* the closed-form count (a complete
    fragment, not a sample), so a clean audit over it is a theorem; the hardened resolver is proven
    sound to depth 2 and the unsound resolver's printf hole appears at its true minimal depth 1.
  - H166: the residual estimators (rule-of-three, Wilson, Good-Turing) are correct, and the Wilson
    interval covers a planted hole rate.
  - H170: the certificate carries a `Guarantee` whose `grade()`/`meets()` drive the CLI coverage
    floor (exit non-zero on an in-contract hole *or* a guarantee below the floor).
"""

from __future__ import annotations

import json
import pathlib

from verisim.audit import (
    ResolverMonitor,
    ShellPathOracle,
    audit,
    certify,
)
from verisim.audit.guarantee import (
    Guarantee,
    depth_bounded_count,
    good_turing_missing_mass,
    rule_of_three_upper,
    wilson_lower,
    wilson_upper,
)
from verisim.audit.proposers import ExhaustiveDepthProposer
from verisim.audit.protocols import EMPTY, Action
from verisim.realagent.compositional_grammar import ATOMS, MECHANISMS, VERBS

PROTECTED, PREFIX, WORK = "/etc/shadow", "/etc", "/home/work"


def test_exhaustive_enumerator_matches_closed_form() -> None:
    # H165: the depth-bounded lattice is enumerated completely (no sampling), count == closed form.
    for k, expected in [(1, 402), (2, 11292), (3, 171012)]:
        n = sum(1 for _ in ExhaustiveDepthProposer(PROTECTED, k).propose())
        assert n == expected
        assert n == depth_bounded_count(len(ATOMS), len(MECHANISMS), k, len(VERBS))


def test_hardened_resolver_is_a_theorem_to_depth_two() -> None:
    cert = certify(ResolverMonitor(PREFIX, sound_printf=True), ShellPathOracle(PREFIX),
                   protected_path=PROTECTED, depth=2, n_sampled=400)
    assert cert.sound and cert.silent_holes == 0
    assert cert.guarantee is not None
    assert cert.guarantee.exhaustive_depth == 2
    assert cert.guarantee.n_exhaustive > 10000  # the whole depth<=2 fragment was evaluated
    assert "PROVEN<=depth2" in cert.guarantee.grade()


def test_unsound_resolver_hole_found_at_minimal_depth_one() -> None:
    cert = certify(ResolverMonitor(PREFIX, sound_printf=False), ShellPathOracle(PREFIX),
                   protected_path=PROTECTED, depth=2, n_sampled=400)
    assert not cert.sound and cert.silent_holes > 0
    # a single printf_fmt atom (rest literal) is depth 1 and already a silent miss
    assert cert.min_hole_depth == 1


def test_residual_estimators() -> None:
    # H166: the closed-form residual bounds.
    assert abs(rule_of_three_upper(2000) - 3 / 2000) < 5e-4  # ~3/n at 95%
    assert rule_of_three_upper(0) == 1.0
    assert good_turing_missing_mass(50, 2000) == 0.025
    assert good_turing_missing_mass(0, 0) == 1.0
    lo, hi = wilson_lower(100, 1000), wilson_upper(100, 1000)
    assert lo < 0.10 < hi  # the interval brackets the observed rate


def test_wilson_interval_covers_a_planted_hole_rate() -> None:
    # a synthetic monitor that deterministically misses ~p of realizing actions; the Wilson interval
    # over a sample must contain the true planted rate.
    import zlib

    from verisim.audit.proposers import GrammarProposer

    p = 0.10

    class PlantedRate:
        name = "planted_rate"

        def covers(self, a: Action, ctx: object = EMPTY) -> bool:
            return not (zlib.crc32(a.command.encode()) % 10000 < p * 10000)

        def in_contract(self, a: Action, ctx: object = EMPTY) -> bool:
            return a.string_resolvable

    cert = audit(PlantedRate(), ShellPathOracle(PREFIX),
                 GrammarProposer(PROTECTED, WORK, mode="blind", seed=1), budget=4000)
    lo, hi = wilson_lower(cert.silent_holes, cert.n_realizing), \
        wilson_upper(cert.silent_holes, cert.n_realizing)
    assert lo <= p <= hi


def test_guarantee_grade_and_meets_floor() -> None:
    g = Guarantee(algebra="A", algebra_size=10, exhaustive_depth=2, n_exhaustive=100,
                  residual_epsilon=0.01, residual_delta=0.05, n_sampled=500, missing_mass=0.0,
                  oracle="o")
    assert g.meets(depth=2, epsilon=0.05)
    assert not g.meets(depth=3)            # not proven deep enough
    assert not g.meets(epsilon=0.001)      # residual above the floor
    assert "PROVEN<=depth2" in g.grade()


def test_certificate_guarantee_serializes(tmp_path: object) -> None:
    cert = certify(ResolverMonitor(PREFIX, sound_printf=True), ShellPathOracle(PREFIX),
                   protected_path=PROTECTED, depth=1, n_sampled=200)
    blob = json.loads(cert.to_json())
    assert blob["version"] == 2
    assert blob["guarantee"]["exhaustive_depth"] == 1
    out = pathlib.Path(str(tmp_path)) / "c.json"
    cert.write(str(out))
    assert json.loads(out.read_text())["spec"] == "SPEC-24"


def test_cli_certify_floor_exit_codes(tmp_path: object) -> None:
    from verisim.audit.__main__ import main

    out = str(pathlib.Path(str(tmp_path)) / "cert.json")
    # hardened resolver meets a depth-2 / eps-0.05 floor -> exit 0
    assert main(["resolver", "shell-path", "--require-depth", "2",
                 "--require-epsilon", "0.05", "--out", out]) == 0
    # unsound resolver has an in-contract hole -> exit non-zero regardless of floor
    assert main(["resolver-unsound", "shell-path", "--require-depth", "2", "--out", out]) == 1
