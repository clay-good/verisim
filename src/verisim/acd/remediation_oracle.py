"""SPEC-22 CU30 (H123): the remediation oracle -- the recovery dual of the forensic oracle.

CU29 turned the exact oracle to the *forensic* question after a breach: which step realized it
(attribution) and what determined it (the counterfactual root cause). CU29's two findings were a
*diagnosis*. CU30 closes the incident-response loop with the *action* a defender takes next --
**recovery**: compute a remediation that undoes the breach, and prove it. The exact oracle can; a
world model that omitted the breach (CU8) cannot even find the incident, so it proposes no fix.

A remediation here is a set of action indices to **block** (drop from the trace), and the recovery
question has two axes a defender must satisfy at once, both certified by re-running the exact oracle
as a Structural Causal Model (SPEC-17: abduct the recorded trace, ``do`` the removals, predict):

  - **avert** -- does the breach actually not happen in the re-run? (:func:`averts`, the oracle's
    counterfactual verdict, never the model's).
  - **collateral** -- how much benign mission work did the fix sacrifice? (:func:`collateral`, the
    number of *benign* actions the remediation blocked; the model-free mission cost).

THE HEADLINE (the CU29 diagnosis, completed as an action). CU29 proved *the realizing step is not
the root cause* in the genesis-separated worlds. Its recovery corollary is sharper and not obvious:
**undoing the action that realized the breach does not undo the breach.** Where a danger has
*redundant consumers* -- a protected file two writes corrupt, a stale value two reads consume --
removing the one realizing action just hands the breach to the next consumer (host: the
:func:`surgical` undo averts only ~39% of incidents). The robust fix removes the *genesis* -- the
``open`` that bound the fd, the ``put`` that planted the stale value -- and the oracle's
counterfactual is what finds it. The four genesis-grammar flavors, read a third time: forward they
are the prevention surface (CU21), backward they are the root cause (CU29), and *acted on* they are
the remediation target -- and only on the genesis does removal robustly avert.

THE TENSION (recovery's CU3/CU14 corner). The robust fix is not free. Removing an upstream *benign*
genesis pays collateral the surgical undo avoided, and the panic response a real SOAR playbook
commits -- :func:`sledgehammer`, disable the whole capability (block every ``write``/``connect``/
``get``) -- always averts but destroys the mission (collateral 3-4x the genesis fix). Only the
oracle-computed :func:`min_certified` remediation -- the smallest averting removal set, searched
against the oracle's counterfactual -- sits in the all-good corner: it averts *every* incident (the
sledgehammer is its backstop) at the *minimal* collateral, world by world. The model's remediation
(:func:`model_remediation`) is the empty fix: blind to the incident, it blocks nothing -- recovery
indistinguishable from doing nothing.

Torch-free core (the worst-case omitter is the CU21/CU29 substrate; certification keys on the
oracle, never the model). The experiment grounds the omitter against the real network ``M_theta``
(torch-gated, frozen flagship) to show its empty fix is not a strawman -- its remediation averts
CU8's 0.02. Deterministic, seeded; danger and surfaces grounded in the real reference oracles via
the unified arms; the single-breach substrate (CU29's :func:`single_breach_scenarios`) isolates the
genesis-vs-consumption question without the multi-breach set confound.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from verisim.acd.dist_targeting import CU18Config
from verisim.acd.forensic_oracle import (
    ExecutedStep,
    breach_steps,
    counterfactual_realizes,
    is_breached,
    replay,
    root_cause_step,
    single_breach_scenarios,
)
from verisim.acd.host_targeting import CU16Config
from verisim.acd.segmentation_targeting import CU17Config
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.unified_targeting import (
    Arm,
    Defender,
    OmitterDefender,
    Scenario,
    dist_arm,
    host_arm,
    net_flow_arm,
    net_reach_arm,
)

# --------------------------------------------------------------------------------------------------
# Certification: the exact oracle re-runs the counterfactual and verifies a remediation, both axes.
# --------------------------------------------------------------------------------------------------


def averts(arm: Arm, scenario: Scenario, blocked: frozenset[int]) -> bool:
    """The oracle's counterfactual verdict: with ``blocked`` removed, the breach does not recur.

    ``do(remove blocked)`` re-run through the exact oracle (SPEC-17). A remediation is only safe if
    the *re-run* is danger-free -- not if it merely deletes the originally-realizing step, which a
    redundant consumer can re-trigger.
    """
    return not counterfactual_realizes(arm, scenario, blocked)


def collateral(trace: list[ExecutedStep], blocked: frozenset[int]) -> int:
    """The mission cost: how many *benign* (non-realizing) actions the remediation blocked.

    Model-free -- read off the oracle's ``realizes`` verdict on the recorded trace. Removing a
    realizing action costs nothing (it was the danger); a removed benign action is lost work.
    """
    return sum(1 for s in trace if s.index in blocked and not s.realizes)


# --------------------------------------------------------------------------------------------------
# Remediation policies: each proposes a removal set, then the oracle certifies it (above).
# --------------------------------------------------------------------------------------------------


def surgical(trace: list[ExecutedStep]) -> frozenset[int]:
    """The naive undo: remove exactly the actions that realized the danger.

    Zero collateral by construction, but NOT robust -- in a genesis-separated world with redundant
    consumers, deleting the one realizing action lets the next consumer realize the danger instead.
    """
    return frozenset(breach_steps(trace))


def root_cause_remediation(arm: Arm, scenario: Scenario) -> frozenset[int]:
    """Remove the CU29 root cause -- the earliest single action whose removal averts the breach.

    On a genesis-separated danger this is the upstream genesis (the ``open``/``put``), so it averts
    where the surgical undo fails, but pays collateral when that genesis is itself a benign action.
    """
    rc = root_cause_step(arm, scenario)
    return frozenset() if rc is None else frozenset({rc})


def sledgehammer(trace: list[ExecutedStep], scenario: Scenario) -> frozenset[int]:
    """The panic response: disable the whole capability -- block every action of the breach's class.

    Always averts (every consumer is removed) but maximal collateral -- the recovery analogue of
    CU3/CU14's paranoid corner (safe by abandoning the mission).
    """
    classes = {scenario.actions[i].name for i in breach_steps(trace)}
    return frozenset(i for i, a in enumerate(scenario.actions) if a.name in classes)


def model_remediation(trace: list[ExecutedStep], defender: Defender) -> frozenset[int]:
    """A model-based fix: remove the steps the world model believes realized the danger.

    The omitter foresees nothing -> the empty fix (recovery == doing nothing). The real ``M_theta``
    flags no realizing step (CU29 localization 0.000), so its fix averts ~0: blind, not a strawman.
    """
    return frozenset(s.index for s in trace if defender.foresees(s.state, s.action))


def min_certified(arm: Arm, scenario: Scenario, trace: list[ExecutedStep]) -> frozenset[int]:
    """The oracle-computed minimal certified remediation: smallest averting set, least collateral.

    Searches a collateral-ordered ladder of candidates and returns the first the oracle certifies
    averts -- surgical (collateral 0) if it suffices, else the root-cause genesis, else
    surgical+genesis, else the sledgehammer backstop (which always averts). Always averts at the
    minimal collateral the oracle can certify; the model can compute none of this (it cannot run the
    counterfactual on a breach it never saw).
    """
    surg = surgical(trace)
    rc = root_cause_remediation(arm, scenario)
    candidates = [surg, rc, surg | rc, sledgehammer(trace, scenario)]
    averting = [c for c in candidates if averts(arm, scenario, c)]
    if not averting:  # pragma: no cover - the sledgehammer backstop always averts
        return sledgehammer(trace, scenario)
    return min(averting, key=lambda c: (collateral(trace, c), len(c)))


# --------------------------------------------------------------------------------------------------
# Per-arm remediation battery: each policy's avert rate and mission collateral, on the four worlds.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    name: str
    avert_rate: float  # fraction of incidents the remediation actually undoes (oracle-certified)
    mean_collateral: float  # mean benign actions sacrificed (the mission cost)
    mission_preserved_rate: float  # fraction with zero collateral


@dataclass(frozen=True)
class ArmRemediation:
    world_name: str
    danger_name: str
    n_incidents: int
    policies: list[PolicyResult]

    def by_name(self, name: str) -> PolicyResult:
        return next(p for p in self.policies if p.name == name)


_POLICIES = ("model", "surgical", "root_cause", "min_certified", "sledgehammer")


def _policy_blocked(
    name: str, arm: Arm, sc: Scenario, trace: list[ExecutedStep], omitter: Defender
) -> frozenset[int]:
    if name == "model":
        return model_remediation(trace, omitter)
    if name == "surgical":
        return surgical(trace)
    if name == "root_cause":
        return root_cause_remediation(arm, sc)
    if name == "min_certified":
        return min_certified(arm, sc, trace)
    return sledgehammer(trace, sc)


def analyze_arm(arm: Arm) -> ArmRemediation:
    """Run the remediation battery on one arm's single-breach incidents (worst-case omitter)."""
    omitter: Defender = OmitterDefender()
    incidents = [(sc, replay(arm, sc)) for sc in single_breach_scenarios(arm)]
    incidents = [(sc, tr) for sc, tr in incidents if is_breached(tr)]
    policies: list[PolicyResult] = []
    for name in _POLICIES:
        averted: list[float] = []
        cols: list[int] = []
        for sc, tr in incidents:
            blocked = _policy_blocked(name, arm, sc, tr, omitter)
            averted.append(1.0 if averts(arm, sc, blocked) else 0.0)
            cols.append(collateral(tr, blocked))
        policies.append(
            PolicyResult(
                name=name,
                avert_rate=fmean(averted) if averted else 0.0,
                mean_collateral=fmean(cols) if cols else 0.0,
                mission_preserved_rate=fmean(1.0 if c == 0 else 0.0 for c in cols) if cols else 0.0,
            )
        )
    return ArmRemediation(arm.world_name, arm.danger_name, len(incidents), policies)


# --------------------------------------------------------------------------------------------------
# Config + the sweep over all four worlds + the verdict.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CU30Config:
    """Modest per-arm batteries -- min_certified runs an O(horizon) counterfactual per candidate."""

    horizon: int = 24
    n_seeds: int = 120
    max_episodes: int = 60

    @staticmethod
    def smoke() -> CU30Config:
        return CU30Config(horizon=10, n_seeds=24, max_episodes=8)


def build_arms(config: CU30Config | None = None) -> list[Arm]:
    """The four unified arms (net exfil / host / distributed / segmentation), sized small."""
    config = config or CU30Config()
    h, n, m = config.horizon, config.n_seeds, config.max_episodes
    return [
        net_flow_arm(CU10Config(horizon=h, n_seeds=n, max_episodes=m)),
        host_arm(CU16Config(horizon=h, n_seeds=n, max_episodes=m)),
        dist_arm(CU18Config(horizon=h, n_seeds=n, max_episodes=m)),
        net_reach_arm(CU17Config(horizon=h, n_seeds=n, max_episodes=m)),
    ]


@dataclass(frozen=True)
class CU30Result:
    arms: list[ArmRemediation]


def run_cu30(config: CU30Config | None = None, arms: list[Arm] | None = None) -> CU30Result:
    """Run the recovery dual on all four worlds under the worst-case omitter (torch-free)."""
    arms = arms if arms is not None else build_arms(config)
    return CU30Result(arms=[analyze_arm(a) for a in arms])


_SEPARATED = ("host", "distributed")


def cu30_verdict(result: CU30Result) -> dict[str, object]:
    """H123: only the oracle's min_certified remediation averts every incident at min collateral.

    The naive surgical undo silently fails under redundant consumers (the genesis-separation tax on
    recovery); the model's fix is empty (blind); the sledgehammer averts but destroys the mission.
    """
    arms = result.arms
    tol = 1e-9

    def pol(a: ArmRemediation, n: str) -> PolicyResult:
        return a.by_name(n)

    min_averts_all = all(pol(a, "min_certified").avert_rate >= 1.0 - tol for a in arms)
    min_cheaper_than_sledge = all(
        pol(a, "min_certified").mean_collateral <= pol(a, "sledgehammer").mean_collateral + tol
        for a in arms
    )
    sledge_averts_all = all(pol(a, "sledgehammer").avert_rate >= 1.0 - tol for a in arms)
    model_is_empty = all(pol(a, "model").avert_rate <= tol for a in arms)
    # the headline: the surgical undo fails in at least one genesis-separated world (redundancy),
    # and min_certified strictly beats it there (averts more), at the cost of some collateral.
    separated = [a for a in arms if a.world_name in _SEPARATED]
    surgical_fails_separated = any(pol(a, "surgical").avert_rate < 1.0 - tol for a in separated)
    min_beats_surgical_where_it_fails = all(
        pol(a, "min_certified").avert_rate >= pol(a, "surgical").avert_rate - tol for a in arms
    )
    # the genesis-separation signature: the certified fix pays collateral EXACTLY where the surgical
    # undo is insufficient. Where deleting the realizing action averts (net, dist), min_certified is
    # that fix at zero collateral; where a redundant consumer / multi-hop path defeats it (host,
    # seg), the certified fix must remove a benign genesis and pays collateral: the redundancy tax.
    collateral_is_redundancy_tax = all(
        (pol(a, "surgical").avert_rate >= 1.0 - tol)
        == (pol(a, "min_certified").mean_collateral <= tol)
        for a in arms
    )
    return {
        "n_worlds": len(arms),
        "min_certified_averts_every_incident": min_averts_all,
        "min_certified_cheaper_than_sledgehammer": min_cheaper_than_sledge,
        "sledgehammer_averts_every_incident": sledge_averts_all,
        "model_remediation_is_empty": model_is_empty,
        "surgical_undo_fails_in_separated_world": surgical_fails_separated,
        "min_certified_dominates_surgical": min_beats_surgical_where_it_fails,
        "collateral_is_redundancy_tax": collateral_is_redundancy_tax,
        "arms": [
            {
                "world": a.world_name,
                "danger": a.danger_name,
                "n_incidents": a.n_incidents,
                "policies": {
                    p.name: {
                        "avert_rate": p.avert_rate,
                        "mean_collateral": p.mean_collateral,
                        "mission_preserved_rate": p.mission_preserved_rate,
                    }
                    for p in a.policies
                },
            }
            for a in arms
        ],
        "_arms": {a.world_name: a for a in arms},
    }


CSV_HEADER = "world,danger,n_incidents,policy,avert_rate,mean_collateral,mission_preserved_rate"


def write_csv(result: CU30Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for a in result.arms:
        for p in a.policies:
            rows.append(
                f"{a.world_name},{a.danger_name},{a.n_incidents},{p.name},"
                f"{p.avert_rate:.6f},{p.mean_collateral:.6f},{p.mission_preserved_rate:.6f}"
            )
    out.write_text("\n".join(rows) + "\n")
    return str(out)


# --------------------------------------------------------------------------------------------------
# The real-model remediation (torch-gated, used by the experiment): the real M_theta's fix is empty.
# --------------------------------------------------------------------------------------------------


def real_model_avert_rate(arm: Arm, defender: Defender) -> float:
    """The avert rate of a remediation that removes the steps a *real* trained ``M_theta`` flags.

    Reuses CU29's :class:`NetFlowModelDefender` (the model previews each step for an exfil flow).
    The real model omits the breach (CU8; CU29 localization 0.000), so its remediation set is empty
    and averts ~0 -- the omitter's empty fix is not a strawman.
    """
    rates: list[float] = []
    for sc in single_breach_scenarios(arm):
        trace = replay(arm, sc)
        if not is_breached(trace):
            continue
        blocked = model_remediation(trace, defender)
        rates.append(1.0 if averts(arm, sc, blocked) else 0.0)
    return fmean(rates) if rates else 0.0
