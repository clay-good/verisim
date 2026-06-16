"""SPEC-22 CU37 (H130): the verifier precision tax -- the *utility* half of the dual coverage law,
grounded against a real, dial-able verifier.

CU35 proved the dual coverage law and split its two knobs: a deployed verifier's **safety** is set
by its fidelity **on** the danger surface (``phi``), while its fidelity **off** the surface
(``psi``) is irrelevant to safety and buys only **utility** (false blocks). CU36 grounded *safety*
two real verifiers (a state-diff monitor, a structure monitor) and read their safety off the danger
grammar a priori -- but it left the *utility* knob untouched: both of CU36's verifiers happen to sit
at ``false_blocks = 0`` (they observe the state at exactly the danger's granularity, so they never
over-fire on a benign action). The skeptic's matching question therefore still stood: *is CU35's
off-surface / utility axis a property of a real verifier, or only of the synthetic ``psi`` dial?*

CU37 grounds it with the most ordinary verifier a host defender actually runs: a **file-integrity
monitor**, which watches a *set of paths* and blocks any write that touches one of them. A real
monitor is *both* the targeting surface (which actions it gates -- writes to a watched file) and the
verifier (its allow/abort verdict on that action), so the two coincide exactly as they do in
deployment. Its two structural properties are independent, and each governs one axis of the law:

  - **COVERAGE sets SAFETY.** The integrity danger is a write that corrupts the protected file
    ``/cfg``. A monitor is faithful on that surface -- and so **exactly as safe as the perfect
    oracle** (adversarial breach 0) -- iff ``/cfg`` is in its watch set, *however many other files
    it also watches*. Drop ``/cfg`` and it is blind on the surface and **as leaky as no gate at
    all**, however many files it watches (CU36's coverage cliff). This is CU16/CU21 target coverage.
  - **PRECISION sets UTILITY.** Over-watching benign files (``/log``, ``/scratch``, ``/data``, ...)
    costs nothing in safety but **over-fires**: every benign write to a watched-but-undangerous
    file is a false block (and an oracle call). So the monitor's utility cost rises monotonically
    with how *coarsely* it watches -- the **precision tax** -- while its safety stays pinned at 0.
    This is CU35's ``psi`` axis, now a real, measurable observation model rather than a hash-coin.

The two axes are **orthogonal**: across the 2x2 of {covers ``/cfg``?} x {precise / coarse}, the
breach is decided *only* by coverage and the utility cost *only* by precision. The defender payoff
is the actionable rule a security engineer can use directly: **coarsen the monitor as much as you
like -- for cheapness, or for robustness to an incomplete inventory (CU12: watch the whole directory
and catch corruption of shadow files you never enumerated) -- at ZERO safety cost; you pay only a
bounded utility tax proportional to the benign traffic in the watch set.** It is the verifier
analogue of CU3's "any model is safe by being useless," sharpened: a coarse-but-covering monitor is
safe *and* its uselessness is **bounded and priced** (false blocks proportional to coarseness), not
all-or-nothing.

Substrate: the CU34 host integrity leg verbatim (the write-to-``/cfg`` danger over the provisioned
read-injecting battery), gated by file-integrity monitors at every granularity from precise
(``{/cfg}``) to coarsest (watch every host path), run through CU35's verifier-gated runners with the
monitor serving as both target and verifier. Torch-free, deterministic; the monitor is the real
reference oracle's write surface restricted to a watched path set.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from verisim.acd.footprintless_targeting import (
    CU34Config,
    TriadDeployment,
    build_deployments,
    build_legs,
)
from verisim.acd.host_targeting import _writes_protected
from verisim.acd.unified_targeting import (
    Danger,
    Scenario,
    World,
    covers,
)
from verisim.acd.verifier_fidelity import (
    adversarial_with_verifier,
    faithful_on_surface,
    run_with_verifier,
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

# The integrity danger lives on /cfg; the other host paths are benign for this leg -- watching them
# costs utility, never safety. (Drawn from the host config's path namespace, CU34's battery.)
DANGER_FILE = "/cfg"
BENIGN_FILES: tuple[str, ...] = ("/log", "/scratch", "/data", "/passwd")


# --------------------------------------------------------------------------------------------------
# The real, dial-able verifier: a file-integrity monitor over a watched path set.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileMonitorVerifier:
    """A file-integrity monitor: block any write that touches a path in ``watched``.

    The ordinary deployed host defense (Tripwire / auditd-style). It is *both* the targeting surface
    (it gates exactly the writes to a watched file) and the verifier (its allow/abort verdict on
    that write), so target and verifier coincide as they do in a real deployment. By the host
    grammar it is faithful on the integrity surface (it flags every corrupting write to ``/cfg``)
    iff ``/cfg`` is watched -- and over-fires on every write to a watched-but-undangerous file (the
    utility tax). ``watched`` sets both axes: coverage (contains ``/cfg``?) and precision (how many
    benign files does it also contain?).
    """

    watched: frozenset[str]

    def verdict(self, state: object, action: object) -> bool:
        assert isinstance(state, HostState) and isinstance(action, HostAction)
        return _writes_protected(state, action, self.watched)


# --------------------------------------------------------------------------------------------------
# Running one monitor over the integrity battery (the monitor is both target and verifier).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorPoint:
    """One file-integrity monitor at one granularity: its safety and its utility cost."""

    watched: frozenset[str]
    granularity: int  # number of watched files (the precision dial; lower = more precise)
    covers_danger: bool  # is /cfg watched? (the coverage axis -> safety)
    faithful_on_surface: bool  # CU35's dual condition (== covers_danger, the a-priori predictor)
    adversarial_breach: float  # worst case over the attacker (0 = as safe as the oracle)
    random_breach: float
    mean_false_blocks: float  # the utility tax: benign writes to watched undangerous files
    mean_calls: float  # the monitor's firing rate (its cost also rises with coarseness)


def _scenarios(
    deployments: list[TriadDeployment], config: CU34Config, oracle: HostOracle,
) -> tuple[list[Scenario], World]:
    """The integrity leg's scenarios over the shared battery, with the monitor as the target."""
    leg = build_legs(config, oracle)["integrity"]
    world = World(advance=lambda s, a: oracle.step(s, a).state)
    danger = Danger(realizes=leg.realizes, attacks=leg.attacks)
    # the target is the monitor itself, supplied per-monitor in run_monitor; here a placeholder None
    scenarios = [
        Scenario(d.start, d.actions, danger, lambda s, a: False, None) for d in deployments
    ]
    return scenarios, world


def run_monitor(
    watched: frozenset[str], deployments: list[TriadDeployment], config: CU34Config,
    oracle: HostOracle,
) -> MonitorPoint:
    """Gate the integrity leg with a file-integrity monitor at granularity ``watched``.

    The monitor is both the schedule (consult iff it would fire = a write to a watched file) and the
    verifier (block iff it fires), exactly as a deployed file-integrity monitor behaves. Returns the
    monitor's safety (random + adversarial breach) and its utility cost (false blocks + calls).
    """
    scenarios, world = _scenarios(deployments, config, oracle)
    monitor = FileMonitorVerifier(watched)
    target = monitor.verdict  # the monitor IS the targeting surface

    runs = [run_with_verifier(world, sc, monitor, target) for sc in scenarios]
    adv = [adversarial_with_verifier(world, sc, monitor, target) for sc in scenarios]
    faithful = all(faithful_on_surface(world, sc, monitor) for sc in scenarios)
    return MonitorPoint(
        watched=watched,
        granularity=len(watched),
        covers_danger=DANGER_FILE in watched,
        faithful_on_surface=faithful,
        adversarial_breach=fmean(adv) if adv else 0.0,
        random_breach=fmean(b for b, _, _ in runs) if runs else 0.0,
        mean_false_blocks=fmean(fb for _, _, fb in runs) if runs else 0.0,
        mean_calls=fmean(c for _, c, _ in runs) if runs else 0.0,
    )


# --------------------------------------------------------------------------------------------------
# The granularity sweep + the orthogonality 2x2.
# --------------------------------------------------------------------------------------------------


def covering_watch_sets() -> list[frozenset[str]]:
    """Nested watch sets, each adding one benign file -- all COVER /cfg (safety held fixed at 0).

    From precise (``{/cfg}`` -- no utility tax) to coarsest (watch every host path -- maximal tax).
    """
    sets: list[frozenset[str]] = [frozenset({DANGER_FILE})]
    for k in range(len(BENIGN_FILES)):
        sets.append(frozenset({DANGER_FILE, *BENIGN_FILES[: k + 1]}))
    return sets


def subcoverage_watch_sets() -> list[frozenset[str]]:
    """Watch sets that MISS /cfg, at rising granularity -- the coverage cliff (all leak).

    Includes the empty set (no gate) and a *coarse-but-blind* monitor that watches four benign files
    yet still leaks, because precision cannot substitute for coverage (CU35/CU36 negative).
    """
    sets: list[frozenset[str]] = [frozenset()]
    for k in range(len(BENIGN_FILES)):
        sets.append(frozenset(BENIGN_FILES[: k + 1]))
    return sets


@dataclass(frozen=True)
class CU37Result:
    n_episodes: int
    horizon: int
    covering: list[MonitorPoint]  # nested watch sets, all cover /cfg (precision tax, safety flat)
    subcoverage: list[MonitorPoint]  # watch sets that miss /cfg (the coverage cliff, all leak)


def run_cu37(config: CU34Config | None = None) -> CU37Result:
    """Sweep file-integrity monitors over granularity, with and without coverage of /cfg."""
    config = config or CU34Config()
    oracle: HostOracle = ReferenceHostOracle()
    deployments = build_deployments(config, oracle)
    covering = [run_monitor(w, deployments, config, oracle) for w in covering_watch_sets()]
    subcoverage = [run_monitor(w, deployments, config, oracle) for w in subcoverage_watch_sets()]
    return CU37Result(
        n_episodes=len(deployments), horizon=config.horizon,
        covering=covering, subcoverage=subcoverage,
    )


# --------------------------------------------------------------------------------------------------
# The 2x2: breach is decided only by coverage, utility only by precision (the orthogonality).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrthogonalityCell:
    label: str
    covers_danger: bool
    coarse: bool
    adversarial_breach: float
    mean_false_blocks: float


def orthogonality_2x2(result: CU37Result) -> list[OrthogonalityCell]:
    """The four corners of {covers /cfg?} x {watches benign files?}, from the swept monitors.

    The two independent quantities are coverage (is /cfg watched -> safety) and how many *benign*
    files are also watched (the precision tax -> utility). The clean corners: a monitor that watches
    only the danger file (no benign, no tax) vs the coarsest one (all benign), on each coverage side
    -- the precise-blind corner is the empty watch set (no gate: misses /cfg, watches no benign
    file). Shows breach tracks coverage only and false blocks track the benign-watch count only.
    """
    cov_precise = min(result.covering, key=lambda p: p.granularity)  # {/cfg}: covers, 0 benign
    cov_coarse = max(result.covering, key=lambda p: p.granularity)  # {/cfg, all benign}
    blind_precise = min(result.subcoverage, key=lambda p: p.granularity)  # {}: no gate, 0 benign
    blind_coarse = max(result.subcoverage, key=lambda p: p.granularity)  # {all benign}: misses /cfg
    return [
        OrthogonalityCell("covers + precise", True, False,
                          cov_precise.adversarial_breach, cov_precise.mean_false_blocks),
        OrthogonalityCell("covers + coarse", True, True,
                          cov_coarse.adversarial_breach, cov_coarse.mean_false_blocks),
        OrthogonalityCell("misses + precise", False, False,
                          blind_precise.adversarial_breach, blind_precise.mean_false_blocks),
        OrthogonalityCell("misses + coarse", False, True,
                          blind_coarse.adversarial_breach, blind_coarse.mean_false_blocks),
    ]


def _strictly_increasing(values: list[float]) -> bool:
    from itertools import pairwise

    return all(b > a + 1e-9 for a, b in pairwise(values))


def _non_decreasing(values: list[float]) -> bool:
    from itertools import pairwise

    return all(b >= a - 1e-9 for a, b in pairwise(values))


def cu37_verdict(result: CU37Result) -> dict[str, object]:
    """H130: a file-integrity monitor's SAFETY is set by coverage of the danger file and its UTILITY
    by precision, and the two are orthogonal -- so a coarse-but-covering monitor is exactly as safe
    as the oracle while paying a utility tax that rises monotonically with its coarseness.
    """
    cov = result.covering
    sub = result.subcoverage
    precise = min(cov, key=lambda p: p.granularity)
    coarsest = max(cov, key=lambda p: p.granularity)
    fb_curve = [p.mean_false_blocks for p in cov]
    calls_curve = [p.mean_calls for p in cov]
    grid = orthogonality_2x2(result)

    return {
        "n_episodes": result.n_episodes,
        # COVERAGE sets safety: every covering monitor is as safe as the oracle, at any precision
        "covering_safe_at_every_precision": all(p.adversarial_breach <= 1e-9 for p in cov),
        "safety_flat_across_precision": (
            max(p.adversarial_breach for p in cov) - min(p.adversarial_breach for p in cov) <= 1e-9
        ),
        # PRECISION sets utility: the tax rises monotonically with coarseness; precise pays none
        "precise_pays_no_tax": precise.mean_false_blocks <= 1e-9,
        "utility_tax_rises_with_coarseness": _strictly_increasing(fb_curve),
        "calls_rise_with_coarseness": _non_decreasing(calls_curve),
        "coarsest_false_blocks": coarsest.mean_false_blocks,
        "precise_false_blocks": precise.mean_false_blocks,
        # COVERAGE is a cliff: every sub-coverage monitor leaks, however coarse (precision != safe)
        "subcoverage_leaks_at_every_precision": all(p.adversarial_breach >= 0.5 for p in sub),
        "coarse_blind_still_leaks": (
            max(sub, key=lambda p: p.granularity).adversarial_breach >= 0.5
        ),
        # the a-priori predictor (CU35/CU36): faithful_on_surface == covers_danger in every cell
        "faithful_iff_covers": all(
            p.faithful_on_surface == p.covers_danger for p in (*cov, *sub)
        ),
        # the orthogonality: breach tracks coverage only, false blocks track precision only
        "breach_tracks_coverage_only": (
            all(c.adversarial_breach <= 1e-9 for c in grid if c.covers_danger)
            and all(c.adversarial_breach >= 0.5 for c in grid if not c.covers_danger)
        ),
        "utility_tracks_precision_only": (
            all(c.mean_false_blocks <= 1e-9 for c in grid if not c.coarse)
            and all(c.mean_false_blocks > 1e-9 for c in grid if c.coarse)
        ),
        "covering": [
            {
                "granularity": p.granularity, "covers_danger": p.covers_danger,
                "adversarial_breach": p.adversarial_breach, "random_breach": p.random_breach,
                "mean_false_blocks": p.mean_false_blocks, "mean_calls": p.mean_calls,
            }
            for p in cov
        ],
        "subcoverage": [
            {
                "granularity": p.granularity, "covers_danger": p.covers_danger,
                "adversarial_breach": p.adversarial_breach,
                "mean_false_blocks": p.mean_false_blocks,
            }
            for p in sub
        ],
        "grid": [
            {
                "label": c.label, "covers_danger": c.covers_danger, "coarse": c.coarse,
                "adversarial_breach": c.adversarial_breach,
                "mean_false_blocks": c.mean_false_blocks,
            }
            for c in grid
        ],
    }


def assert_covers(config: CU34Config | None = None) -> bool:
    """Sanity: the unified ``covers`` predicate agrees that the precise watch set covers /cfg."""
    config = config or CU34Config()
    oracle: HostOracle = ReferenceHostOracle()
    deployments = build_deployments(config, oracle)
    scenarios, world = _scenarios(deployments, config, oracle)
    monitor = FileMonitorVerifier(frozenset({DANGER_FILE}))
    covered = [
        covers(world, Scenario(sc.start, sc.actions, sc.danger, monitor.verdict, None))
        for sc in scenarios
    ]
    return all(covered)


CSV_HEADER = (
    "series,granularity,covers_danger,faithful_on_surface,adversarial_breach,random_breach,"
    "mean_false_blocks,mean_calls,n_episodes"
)


def write_csv(result: CU37Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for series, points in (("covering", result.covering), ("subcoverage", result.subcoverage)):
        for p in points:
            rows.append(
                f"{series},{p.granularity},{int(p.covers_danger)},{int(p.faithful_on_surface)},"
                f"{p.adversarial_breach:.6f},{p.random_breach:.6f},{p.mean_false_blocks:.6f},"
                f"{p.mean_calls:.6f},{result.n_episodes}"
            )
    out.write_text("\n".join(rows) + "\n")
    return str(out)


def main() -> None:  # pragma: no cover - exercised via the experiment CLI
    result = run_cu37()
    v = cu37_verdict(result)
    print("CU37 / H130 -- the verifier precision tax (the utility half of the dual coverage law):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")
    print("  COVERING monitors (all watch /cfg -- safety flat at 0, utility tax rises):")
    print(f"    {'granularity':>11s} {'adv_breach':>11s} {'false_blocks':>13s} {'calls':>7s}")
    for p in result.covering:
        print(f"    {p.granularity:11d} {p.adversarial_breach:11.3f} "
              f"{p.mean_false_blocks:13.3f} {p.mean_calls:7.2f}")
    print("\n  SUB-COVERAGE monitors (miss /cfg -- leak however coarse):")
    for p in result.subcoverage:
        print(f"    granularity {p.granularity}: adv_breach {p.adversarial_breach:.3f}")
    print(f"\n  covering safe at every precision = {v['covering_safe_at_every_precision']}")
    print(f"  utility tax rises with coarseness = {v['utility_tax_rises_with_coarseness']}")
    print(f"  breach tracks coverage only       = {v['breach_tracks_coverage_only']}")
    print(f"  utility tracks precision only     = {v['utility_tracks_precision_only']}")


if __name__ == "__main__":  # pragma: no cover
    main()
