"""SPEC-22 CU29 (H122): the forensic oracle -- the posterior dual of the targeting arc.

The whole targeting arc (CU10-CU28) is *a priori* and *preventive*: a danger has a model-free
surface, and :func:`verisim.acd.unified_targeting.covers` predicts, before deployment, that
verifying that surface is cheap, safe, and un-gameable. CU29 turns the same exact oracle around and
asks the *a posteriori*, *forensic* question a defender faces after an incident has already
happened: **given a breached trace, which action caused it, and what was the root cause?**

This is not a re-skin of prevention. It is the dual mode, and it has two findings.

THE HEADLINE (the CU8/CU10 dual). Forensic attribution needs a ground-truth verdict on each
executed action -- "did *this* step realize the danger?" The exact oracle *is* that verdict: it
replays the trace and pinpoints the realizing step exactly (:func:`oracle_localize`). A world model
cannot. A model that drifts by *omission* (CU8: the real network ``M_theta`` omits 98% of exfil
flows, 146:1 on the danger hosts) predicts *no consequence* at the very step that breached -- so a
model-based forensic analyst (:func:`model_localize`) reports *no incident occurred*. The arc's
preventive slogan was "you can't ask the omitter where it omits"; its forensic dual is **you can't
ask the omitter where it breached.** Only the exact, free oracle can attribute an incident.

THE ROOT CAUSE (the SPEC-17 / genesis-grammar tie). The realizing step is not always the cause. A
deterministic, resettable oracle is an exact Structural Causal Model (SPEC-17), so it can answer
Pearl's third rung exactly and for free: abduct the exogenous state (the recorded trace *is* it),
intervene (``do`` -- remove an earlier action), predict (re-run). :func:`counterfactual_realizes`
does exactly this, and :func:`root_cause_step` finds the *earliest* single action whose removal
makes the mission danger-free. For a single-action danger (a ``connect`` that both opens and is the
exfil) the cause is the breach step itself. But for a *genesis-separated* danger the cause
**precedes** the breach -- the ``open`` that bound the fd a later ``write`` corrupts, the ``put``
under partition a later ``get`` reads stale, the upstream link a later action completes into a path.
The four genesis-grammar flavors of the targeting arc reappear here as four *root-cause* structures,
read backward: blaming the action that tripped the breach blames the wrong step, and the exact
oracle is what reveals the true upstream genesis.

And the two modes converge: the steps the oracle forensic flags are exactly a *covering* target's
consults on that trace (:func:`forensic_surface_is_covering`) -- so the a-posteriori "where should
we have gated" equals the a-priori covering target (CU21). A model-based forensic, flagging nothing,
would propose an empty gate and the incident would recur.

Torch-free core (the worst-case omitter is the CU16/CU21 substrate; the schedule and the attribution
key on the oracle, never the model's competence). The experiment grounds the omitter against the
real network ``M_theta`` (torch-gated) to show the blindness is not a strawman: its forensic recall
is CU8's 0.02. Deterministic, seeded; danger and surfaces grounded in the real reference oracles via
the unified arms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

from verisim.acd.dist_targeting import CU18Config
from verisim.acd.host_targeting import CU16Config
from verisim.acd.segmentation_targeting import CU17Config
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.unified_targeting import (
    Action,
    Arm,
    Defender,
    OmitterDefender,
    Scenario,
    State,
    dist_arm,
    host_arm,
    net_flow_arm,
    net_reach_arm,
)


@dataclass(frozen=True)
class ExecutedStep:
    """One step of an undefended (breached) trace, with the oracle's ground-truth verdict.

    ``state`` is the state *before* ``action``; ``realizes`` is the exact oracle's answer to "did
    this action realize the danger?" -- the forensic ground truth, read off the observed structure,
    never the model.
    """

    index: int
    state: State
    action: Action
    realizes: bool


def replay(arm: Arm, scenario: Scenario) -> list[ExecutedStep]:
    """The breached trace the analyst receives: the benign mission run with no gate at all.

    Nothing is blocked (the agent trusts an omitting model), so every action executes and the
    oracle's ``realizes`` verdict flags exactly the breach steps.
    """
    steps: list[ExecutedStep] = []
    state = scenario.start
    for i, action in enumerate(scenario.actions):
        realizes = bool(scenario.danger.realizes(state, action))
        steps.append(ExecutedStep(i, state, action, realizes))
        state = arm.world.advance(state, action)
    return steps


def breach_steps(trace: list[ExecutedStep]) -> list[int]:
    """The indices of every executed action that truly realized the danger (the oracle's truth)."""
    return [s.index for s in trace if s.realizes]


def is_breached(trace: list[ExecutedStep]) -> bool:
    return any(s.realizes for s in trace)


def first_breach_step(trace: list[ExecutedStep]) -> int | None:
    bs = breach_steps(trace)
    return bs[0] if bs else None


# --------------------------------------------------------------------------------------------------
# Forensic localization: the exact oracle attributes the breach; the world model cannot.
# --------------------------------------------------------------------------------------------------


def oracle_localize(trace: list[ExecutedStep]) -> int | None:
    """The exact oracle's forensic verdict: the first step that truly realized the danger."""
    return first_breach_step(trace)


def model_localize(trace: list[ExecutedStep], defender: Defender) -> int | None:
    """A model-based forensic analyst: the first step the world model believes realized the danger.

    Teacher-forced on the true executed states (the generous version -- no compounding). An omitting
    model foresees nothing and returns ``None``: the incident is invisible to it.
    """
    for s in trace:
        if defender.foresees(s.state, s.action):
            return s.index
    return None


def localization_accuracy(arm: Arm, defender: Defender | None = None) -> float:
    """Over breached deployments, the fraction whose first breach step the forensic pinpoints.

    ``defender=None`` is the exact oracle forensic (always exact -> 1.0); a ``defender`` is a
    model-based forensic (the omitter -> 0.0; the real ``M_theta`` -> its danger recall).
    """
    accs: list[float] = []
    for sc in arm.scenarios:
        trace = replay(arm, sc)
        truth = first_breach_step(trace)
        if truth is None:
            continue  # not a breached deployment -- no incident to attribute
        loc = oracle_localize(trace) if defender is None else model_localize(trace, defender)
        accs.append(1.0 if loc == truth else 0.0)
    return fmean(accs) if accs else 0.0


def detection_rate(arm: Arm, defender: Defender) -> float:
    """Over breached deployments, the fraction where the model flags *any* step at all.

    The weaker, even more damning question: not "did it find the right step" but "did it notice an
    incident happened." The omitter never does.
    """
    rates: list[float] = []
    for sc in arm.scenarios:
        trace = replay(arm, sc)
        if first_breach_step(trace) is None:
            continue
        rates.append(1.0 if model_localize(trace, defender) is not None else 0.0)
    return fmean(rates) if rates else 0.0


def forensic_surface_is_covering(arm: Arm) -> bool:
    """Do the steps the oracle forensic flags all lie on the (a-priori) covering target's surface?

    The two modes converge: the a-posteriori "where should we have gated" (the realizing steps) is
    exactly a covering target's consult set (CU21). True for every covering target.
    """
    for sc in arm.scenarios:
        if sc.target is None:
            return False
        for s in replay(arm, sc):
            if s.realizes and not sc.target(s.state, s.action):
                return False
    return True


# --------------------------------------------------------------------------------------------------
# The counterfactual root cause: abduct (the trace) -> do (remove an action) -> predict (re-run).
# Exact and free because the oracle is a Structural Causal Model (SPEC-17).
# --------------------------------------------------------------------------------------------------


def counterfactual_realizes(arm: Arm, scenario: Scenario, skip: frozenset[int]) -> bool:
    """``do(remove skip)``: re-run with those action indices removed; does the danger still happen?

    Pearl's third rung, executable because the oracle is an exact SCM: abduction is the recorded
    scenario (the exogenous state), the intervention is the skip, the prediction is the re-run.
    """
    state = scenario.start
    realized = False
    for i, action in enumerate(scenario.actions):
        if i in skip:
            continue
        if scenario.danger.realizes(state, action):
            realized = True
        state = arm.world.advance(state, action)
    return realized


def root_cause_step(arm: Arm, scenario: Scenario) -> int | None:
    """The earliest single action whose removal makes the whole mission danger-free -- the cause.

    For a single-action danger this is the breach step itself; for a genesis-separated danger it
    precedes the breach (the ``open``/``put``/upstream-link the breach depends on). Returns ``None``
    on a non-breached trace.
    """
    trace = replay(arm, scenario)
    breach = first_breach_step(trace)
    if breach is None:
        return None
    for j in range(breach + 1):
        if not counterfactual_realizes(arm, scenario, frozenset({j})):
            return j
    return breach  # no single upstream removal suffices -> the breach action is its own cause


def single_breach_scenarios(arm: Arm) -> list[Scenario]:
    """Deployments with exactly one realized breach -- the clean substrate for root-cause analysis.

    (A multi-breach trace needs a *set* of removals; the single-breach subset isolates the
    genesis-vs-consumption question without that confound.)
    """
    return [sc for sc in arm.scenarios if len(breach_steps(replay(arm, sc))) == 1]


# --------------------------------------------------------------------------------------------------
# The per-arm forensic summary + the sweep over all four worlds.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmForensics:
    world_name: str
    danger_name: str
    n_breached: int
    oracle_localization_accuracy: float  # the exact oracle attributes -> 1.0
    omitter_localization_accuracy: float  # the worst-case model is blind -> 0.0
    omitter_detection_rate: float  # it cannot even tell an incident happened -> 0.0
    forensic_surface_is_covering: bool  # the flagged steps are a covering target's consults
    n_single_breach: int
    mean_root_cause_lag: float  # breach_index - root_cause_index (0 if cause == breach)
    fraction_cause_precedes_breach: float  # genesis-separated share (lag > 0)


def analyze_arm(arm: Arm) -> ArmForensics:
    """Run the forensic battery on one arm under the worst-case omitter."""
    omitter: Defender = OmitterDefender()
    breached = [sc for sc in arm.scenarios if is_breached(replay(arm, sc))]
    singles = single_breach_scenarios(arm)
    lags: list[int] = []
    for sc in singles:
        breach = first_breach_step(replay(arm, sc))
        cause = root_cause_step(arm, sc)
        if breach is not None and cause is not None:
            lags.append(breach - cause)
    return ArmForensics(
        world_name=arm.world_name,
        danger_name=arm.danger_name,
        n_breached=len(breached),
        oracle_localization_accuracy=localization_accuracy(arm, None),
        omitter_localization_accuracy=localization_accuracy(arm, omitter),
        omitter_detection_rate=detection_rate(arm, omitter),
        forensic_surface_is_covering=forensic_surface_is_covering(arm),
        n_single_breach=len(singles),
        mean_root_cause_lag=fmean(lags) if lags else 0.0,
        fraction_cause_precedes_breach=(
            fmean(1.0 if x > 0 else 0.0 for x in lags) if lags else 0.0
        ),
    )


@dataclass(frozen=True)
class CU29Config:
    """Modest per-arm batteries -- the root-cause counterfactual is O(horizon^2) per deployment."""

    horizon: int = 24
    n_seeds: int = 120
    max_episodes: int = 60

    @staticmethod
    def smoke() -> CU29Config:
        return CU29Config(horizon=10, n_seeds=24, max_episodes=8)


def build_arms(config: CU29Config | None = None) -> list[Arm]:
    """The four unified arms (net exfil / host / distributed / segmentation), sized small."""
    config = config or CU29Config()
    h, n, m = config.horizon, config.n_seeds, config.max_episodes
    return [
        net_flow_arm(CU10Config(horizon=h, n_seeds=n, max_episodes=m)),
        host_arm(CU16Config(horizon=h, n_seeds=n, max_episodes=m)),
        dist_arm(CU18Config(horizon=h, n_seeds=n, max_episodes=m)),
        net_reach_arm(CU17Config(horizon=h, n_seeds=n, max_episodes=m)),
    ]


@dataclass(frozen=True)
class CU29Result:
    arms: list[ArmForensics]


def run_cu29(config: CU29Config | None = None, arms: list[Arm] | None = None) -> CU29Result:
    """Run the forensic dual on all four worlds under the worst-case omitter (torch-free)."""
    arms = arms if arms is not None else build_arms(config)
    return CU29Result(arms=[analyze_arm(a) for a in arms])


def cu29_verdict(result: CU29Result) -> dict[str, object]:
    """H122: the exact oracle attributes every breach; the omitting model is forensically blind; and
    the counterfactual root cause precedes the breach exactly in the genesis-separated worlds.
    """
    arms = result.arms
    oracle_exact = all(a.oracle_localization_accuracy >= 1.0 - 1e-9 for a in arms)
    model_blind = all(a.omitter_localization_accuracy <= 1e-9 for a in arms)
    model_undetecting = all(a.omitter_detection_rate <= 1e-9 for a in arms)
    surfaces_converge = all(a.forensic_surface_is_covering for a in arms)
    by_world = {a.world_name: a for a in arms}
    # the genesis-grammar boundary, read backward: in the worlds whose danger genesis is temporally
    # separated from its consumption (host open->write, dist put->stale-read) the earliest averting
    # intervention precedes the breach by several steps in a majority of incidents -- the incident
    # was determined long before the action that tripped it; where genesis ~ consumption (net exfil,
    # segmentation flip) the window is tight.
    any_separated = any(a.fraction_cause_precedes_breach > 0.0 for a in arms)
    separated = [
        a for a in arms if a.world_name in ("host", "distributed")
    ]
    upstream_dominates = bool(separated) and all(
        a.fraction_cause_precedes_breach > 0.5 for a in separated
    )
    return {
        "n_worlds": len(arms),
        "oracle_attributes_every_breach": oracle_exact,
        "model_is_forensically_blind": model_blind,
        "model_cannot_detect_incident": model_undetecting,
        "forensic_and_prevention_surfaces_converge": surfaces_converge,
        "upstream_cause_dominates_separated_worlds": upstream_dominates,
        "genesis_separation_reappears_as_root_cause": any_separated,
        "arms": [
            {
                "world": a.world_name,
                "n_breached": a.n_breached,
                "oracle_localization": a.oracle_localization_accuracy,
                "model_localization": a.omitter_localization_accuracy,
                "model_detection": a.omitter_detection_rate,
                "surface_covers": a.forensic_surface_is_covering,
                "n_single_breach": a.n_single_breach,
                "mean_root_cause_lag": a.mean_root_cause_lag,
                "fraction_cause_precedes_breach": a.fraction_cause_precedes_breach,
            }
            for a in arms
        ],
        "_arms": {k: v for k, v in by_world.items()},
    }


CSV_HEADER = (
    "world,danger,n_breached,oracle_localization,model_localization,model_detection,"
    "surface_covers,n_single_breach,mean_root_cause_lag,fraction_cause_precedes_breach"
)


def write_csv(result: CU29Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for a in result.arms:
        rows.append(
            f"{a.world_name},{a.danger_name},{a.n_breached},"
            f"{a.oracle_localization_accuracy:.6f},{a.omitter_localization_accuracy:.6f},"
            f"{a.omitter_detection_rate:.6f},{a.forensic_surface_is_covering},"
            f"{a.n_single_breach},{a.mean_root_cause_lag:.6f},"
            f"{a.fraction_cause_precedes_breach:.6f}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)


# --------------------------------------------------------------------------------------------------
# The real-model forensic (torch-gated, used by the experiment): even the real M_theta is blind.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NetFlowModelDefender:
    """A forensic analyst backed by a trained network ``M_theta``: does it foresee an exfil flow?

    ``foresees(s, a)`` previews ``a`` through the model and asks whether it opens a new flow to a
    crown jewel -- the model's belief that this step realized the danger. (CU8: its recall is 0.02.)
    """

    model: object
    jewels: frozenset[str]
    predict_new_flows: Callable[[object, State, Action, frozenset[str]], bool]

    def foresees(self, state: State, action: Action) -> bool:
        return self.predict_new_flows(self.model, state, action, self.jewels)
