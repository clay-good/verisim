"""SPEC-22 CU27 (H120): the reversibility boundary -- *when* to verify, not *what*.

The whole targeting arc (CU10-CU26) answers one question -- **what** to verify: a danger in an
oracle-grounded world has a model-free surface, and verifying that surface is cheap, safe, and
un-gameable (the coverage theorem, :mod:`verisim.acd.unified_targeting`). Every milestone, however,
assumes the SAME verification discipline: *verify-before-commit* -- the agent previews an action
through its world model ("look before you leap") and only then acts. CU27 opens the orthogonal axis:
**when** must that preview happen at all?

A computer-use agent does not face one verification regime. Many of its actions are **reversible**
(a local file write it journaled, a flow it can close, a config it can restore) -- the breach, if
any, lives in a state it can snapshot and roll back. Some are **irreversible** -- an exfiltration
*send*, a destructive overwrite of un-journaled state -- the harm escapes the snapshot the instant
the action executes. The two demand different disciplines:

  - **verify-AFTER-commit** (the optimistic discipline): execute the action, *observe the realized
    state* (free -- the agent is already in it; no model, no preview, no drift), and if a guardrail
    is violated, **roll back** to the journaled pre-state. For a reversible danger this is safe and
    **model-free**: the realized state is the EXACT oracle state, so faithfulness is irrelevant; and
    it is un-gameable for the same reason (an adversary cannot hide a realized breach from a free
    read of the true post-state). It costs ZERO before-commit oracle previews.

  - **verify-BEFORE-commit** (the pessimistic discipline, the whole prior arc): for an irreversible
    danger after-commit is too late -- the breach is consummated at execution, before a post-check.
    Safety then requires a preview, which depends on a faithful model (the boundary law) or an
    oracle consult (the targeting arc). This costs a before-commit oracle preview on its surface.

THE REVERSIBILITY THEOREM. A reversible danger is safe **model-free** via after-commit rollback at
zero before-commit cost; an irreversible danger has no safe after-commit policy, so its safety
requires a before-commit preview. Therefore the model-faithfulness requirement the entire CU arc
establishes -- and the verification cost it minimizes -- is **load-bearing only on the irreversible
slice** of the action space. Route by reversibility: verify-after-commit the reversible (free), and
spend the costly trustworthy preview only where the harm escapes the snapshot.

THE COST LAW (the new quantitative content). Mix the action space by its irreversible fraction
``f``. The routed policy's before-commit oracle cost scales *linearly* with ``f`` (zero on the
reversible half), while the after-commit-everywhere policy's residual breach *also* scales with
``f`` (it cannot undo the irreversible part), and the verify-everything baseline overpays a constant
full cost regardless. **The price of trustworthy world-modeling is exactly the irreversibility you
face.**

Substrate: the two network dangers the arc already grounds in the reference oracle -- exfiltration
(CU10, a *send*: irreversible, the harm leaves the boundary) and segmentation exposure (CU17, a
*posture*: reversible, the harm is a reachability flip you can re-segment). Reusing
:func:`verisim.acd.unified_targeting.net_flow_arm` / ``net_reach_arm`` keeps the danger genesis,
the attacker arsenals, and the covering targets bit-identical to CU21. Torch-free, deterministic;
the worst-case omitter is the model (the schedule keys on the oracle and reversibility, never the
model's competence).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from verisim.acd.unified_targeting import (
    Danger,
    Defender,
    OmitterDefender,
    Target,
    World,
    net_flow_arm,
    net_reach_arm,
)

State = object
Action = object

Policy = Literal["after_commit", "before_commit_free", "before_commit_oracle", "routed"]
POLICIES: tuple[Policy, ...] = (
    "after_commit", "before_commit_free", "before_commit_oracle", "routed",
)

DangerClass = Literal["reversible", "irreversible"]


@dataclass(frozen=True)
class RevScenario:
    """One deployment plus the static reversibility of its danger.

    ``reversible`` is a true property of the danger, not a label: it is whether reverting to the
    journaled pre-state undoes the breach. An exposure (a reachability posture) lives in state ->
    reversible; an exfil *send* has already left the boundary -> irreversible. ``target`` is the
    covering before-commit oracle rule (used only on the irreversible route).
    """

    world: World
    start: State
    actions: tuple[Action, ...]
    danger: Danger
    target: Target
    reversible: bool


def _step(
    sc: RevScenario, policy: Policy, defender: Defender, state: State, action: Action,
) -> tuple[State, bool, int, int]:
    """Advance one action under a policy. Returns (next_state, breached, oracle_calls, rollbacks).

    ``after_commit`` (and the reversible route of ``routed``) is OPTIMISTIC: it executes, observes
    the realized danger on the true transition (free -- no oracle preview), and rolls back iff the
    danger is reversible. The before-commit policies preview the action (model only, oracle on every
    action, or oracle on the covering target) and block before executing.
    """
    realizes = sc.danger.realizes(state, action)
    optimistic = policy == "after_commit" or (policy == "routed" and sc.reversible)
    if optimistic:
        if realizes:
            if sc.reversible:
                return state, False, 0, 1  # observed + rolled back: averted, zero oracle cost
            return sc.world.advance(state, action), True, 0, 0  # irreversible: the send left
        return sc.world.advance(state, action), False, 0, 0

    oracle_calls = 0
    if policy == "before_commit_free":
        block = defender.foresees(state, action)  # model preview, no oracle
    elif policy == "before_commit_oracle":
        oracle_calls = 1
        block = realizes  # consult the oracle on every action (treat everything as irreversible)
    else:  # routed & irreversible -> the targeted before-commit oracle (the CU10/CU21 surface)
        if sc.target(state, action):
            oracle_calls = 1
            block = realizes
        else:
            block = defender.foresees(state, action)
    breached = realizes and not block
    next_state = state if (realizes and block) else sc.world.advance(state, action)
    return next_state, breached, oracle_calls, 0


def run_policy(sc: RevScenario, policy: Policy, defender: Defender) -> tuple[bool, int, int]:
    """Run the benign mission under one policy; return (breached, oracle_calls, rollbacks)."""
    state = sc.start
    breached = False
    calls = 0
    rolls = 0
    for action in sc.actions:
        state, b, c, r = _step(sc, policy, defender, state, action)
        breached = breached or b
        calls += c
        rolls += r
    return breached, calls, rolls


def _attack_breaches(
    sc: RevScenario, policy: Policy, defender: Defender, state: State, atk: Action,
) -> bool:
    """Does an injected danger action win against this policy from here? (each atk realizes)."""
    if policy == "after_commit" or (policy == "routed" and sc.reversible):
        return not sc.reversible  # reversible -> observed + rolled back; irreversible -> breach
    if policy == "before_commit_free":
        return not defender.foresees(state, atk)
    if policy == "before_commit_oracle":
        return False  # the oracle blocks every realized danger before it commits
    # routed & irreversible: the targeted oracle covers the danger surface
    if sc.target(state, atk):
        return False
    return not defender.foresees(state, atk)


def adversarial_policy(sc: RevScenario, policy: Policy, defender: Defender) -> bool:
    """Worst-case over the attacker's choice of danger action AND timing (the CU11 probe)."""
    state = sc.start
    for action in sc.actions:
        for atk in sc.danger.attacks(state):
            if _attack_breaches(sc, policy, defender, state, atk):
                return True
        state, _b, _c, _r = _step(sc, policy, defender, state, action)
    return False


# --------------------------------------------------------------------------------------------------
# The two arms: irreversible (exfil send) + reversible (segmentation posture), from CU21 grounding.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CU27Config:
    max_episodes: int = 60  # cap per danger class (the net arms over-produce deployments)

    @staticmethod
    def smoke() -> CU27Config:
        return CU27Config(max_episodes=8)


def _rev_scenarios(config: CU27Config) -> tuple[list[RevScenario], list[RevScenario], int]:
    """Build the irreversible (exfil) and reversible (exposure) deployment pools + the horizon."""
    exfil = net_flow_arm()  # CU10: a flow to a crown jewel -- an irreversible send
    expose = net_reach_arm()  # CU17: a jewel reachable from the untrusted set -- reversible posture
    irreversible = [
        RevScenario(exfil.world, sc.start, sc.actions, sc.danger, sc.target, reversible=False)
        for sc in exfil.scenarios[: config.max_episodes]
    ]
    reversible = [
        RevScenario(expose.world, sc.start, sc.actions, sc.danger, sc.target, reversible=True)
        for sc in expose.scenarios[: config.max_episodes]
    ]
    return reversible, irreversible, exfil.horizon


@dataclass(frozen=True)
class PolicyCell:
    """One (policy, danger class): its random + adversarial breach, oracle cost, rollback cost."""

    policy: str
    danger_class: str
    random_breach: float
    adversarial_breach: float
    mean_oracle_calls: float
    mean_rollbacks: float


def _cell(scenarios: list[RevScenario], policy: Policy, danger_class: DangerClass) -> PolicyCell:
    omitter = OmitterDefender()
    runs = [run_policy(sc, policy, omitter) for sc in scenarios]
    adv = [adversarial_policy(sc, policy, omitter) for sc in scenarios]
    return PolicyCell(
        policy=policy,
        danger_class=danger_class,
        random_breach=fmean(b for b, _, _ in runs) if runs else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_oracle_calls=fmean(c for _, c, _ in runs) if runs else 0.0,
        mean_rollbacks=fmean(r for _, _, r in runs) if runs else 0.0,
    )


@dataclass(frozen=True)
class CostLawPoint:
    irreversible_fraction: float
    routed_oracle_calls: float  # before-commit oracle the routed policy must spend
    after_commit_breach: float  # residual breach if you optimistically run everything
    verify_all_oracle_calls: float  # the price of treating everything as irreversible


@dataclass(frozen=True)
class CU27Result:
    horizon: int
    n_reversible: int
    n_irreversible: int
    cells: list[PolicyCell]
    cost_law: list[CostLawPoint]

    def cell(self, policy: str, danger_class: str) -> PolicyCell:
        for c in self.cells:
            if c.policy == policy and c.danger_class == danger_class:
                return c
        raise KeyError(f"no cell for {policy}/{danger_class}")


FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_cu27(config: CU27Config | None = None) -> CU27Result:
    """Score all four policies on each danger class, then sweep the cost law over the mix."""
    config = config or CU27Config()
    reversible, irreversible, horizon = _rev_scenarios(config)

    cells: list[PolicyCell] = []
    for policy in POLICIES:
        cells.append(_cell(reversible, policy, "reversible"))
        cells.append(_cell(irreversible, policy, "irreversible"))

    # The cost law: a pool that is fraction f irreversible is a weighted average of the two classes
    # (after-commit residual breach and routed oracle cost are both zero on the reversible half).
    rev_routed = _cell_lookup(cells, "routed", "reversible")
    irr_routed = _cell_lookup(cells, "routed", "irreversible")
    rev_after = _cell_lookup(cells, "after_commit", "reversible")
    irr_after = _cell_lookup(cells, "after_commit", "irreversible")
    rev_all = _cell_lookup(cells, "before_commit_oracle", "reversible")
    irr_all = _cell_lookup(cells, "before_commit_oracle", "irreversible")
    cost_law = [
        CostLawPoint(
            irreversible_fraction=f,
            routed_oracle_calls=f * irr_routed.mean_oracle_calls
            + (1 - f) * rev_routed.mean_oracle_calls,
            after_commit_breach=f * irr_after.adversarial_breach
            + (1 - f) * rev_after.adversarial_breach,
            verify_all_oracle_calls=f * irr_all.mean_oracle_calls
            + (1 - f) * rev_all.mean_oracle_calls,
        )
        for f in FRACTIONS
    ]
    return CU27Result(
        horizon=horizon,
        n_reversible=len(reversible),
        n_irreversible=len(irreversible),
        cells=cells,
        cost_law=cost_law,
    )


def _cell_lookup(cells: list[PolicyCell], policy: str, danger_class: str) -> PolicyCell:
    for c in cells:
        if c.policy == policy and c.danger_class == danger_class:
            return c
    raise KeyError(f"no cell for {policy}/{danger_class}")


def cu27_verdict(result: CU27Result) -> dict[str, object]:
    """H120: after-commit is free + model-free + un-gameable on the reversible class and FAILS on
    irreversible; routing is safe everywhere at a before-commit cost = the irreversible slice only.
    """
    ac_rev = result.cell("after_commit", "reversible")
    ac_irr = result.cell("after_commit", "irreversible")
    rt_rev = result.cell("routed", "reversible")
    rt_irr = result.cell("routed", "irreversible")
    all_rev = result.cell("before_commit_oracle", "reversible")
    all_irr = result.cell("before_commit_oracle", "irreversible")
    free_rev = result.cell("before_commit_free", "reversible")
    free_irr = result.cell("before_commit_free", "irreversible")

    routed_oracle = 0.5 * (rt_rev.mean_oracle_calls + rt_irr.mean_oracle_calls)
    verify_all_oracle = 0.5 * (all_rev.mean_oracle_calls + all_irr.mean_oracle_calls)
    routed_safe = max(
        rt_rev.random_breach, rt_rev.adversarial_breach,
        rt_irr.random_breach, rt_irr.adversarial_breach,
    )
    law = result.cost_law
    return {
        # the reversible class: after-commit is free, model-free, and un-gameable
        "after_commit_reversible_safe": ac_rev.random_breach <= 1e-9
        and ac_rev.adversarial_breach <= 1e-9,
        "after_commit_reversible_is_free": ac_rev.mean_oracle_calls <= 1e-9,
        "after_commit_reversible_ungameable": ac_rev.adversarial_breach <= 1e-9,
        # the irreversible class: after-commit cannot undo the send -> it fails adversarially
        "after_commit_irreversible_fails": ac_irr.adversarial_breach > 0.5,
        "after_commit_irreversible_adv_breach": ac_irr.adversarial_breach,
        # routing: safe everywhere, and the costly preview is the irreversible slice only
        "routed_safe_everywhere": routed_safe <= 1e-9,
        "routed_reversible_is_free": rt_rev.mean_oracle_calls <= 1e-9,
        "routed_oracle_calls": routed_oracle,
        "verify_all_oracle_calls": verify_all_oracle,
        "routed_cheaper_than_verify_all": routed_oracle < verify_all_oracle,
        "routed_call_saving": (
            verify_all_oracle / routed_oracle if routed_oracle > 0 else float("inf")
        ),
        # the free (omitter) before-commit gate is unsafe on both classes (the boundary law)
        "before_commit_free_unsafe": free_rev.adversarial_breach > 1e-9
        or free_irr.adversarial_breach > 1e-9,
        # the cost law: the price of trustworthy preview = the irreversibility fraction
        "price_is_irreversibility": _is_increasing([p.routed_oracle_calls for p in law])
        and law[0].routed_oracle_calls <= 1e-9,
        "after_commit_breach_tracks_irreversibility": _is_increasing(
            [p.after_commit_breach for p in law]
        )
        and law[0].after_commit_breach <= 1e-9,
        "verify_all_is_flat_full": all(
            p.verify_all_oracle_calls >= result.horizon - 1e-6 for p in law
        ),
    }


def _is_increasing(xs: list[float]) -> bool:
    from itertools import pairwise

    return all(b >= a - 1e-9 for a, b in pairwise(xs)) and xs[-1] > xs[0] + 1e-9


CSV_HEADER = (
    "section,policy,danger_class,irreversible_fraction,random_breach,adversarial_breach,"
    "mean_oracle_calls,mean_rollbacks"
)


def write_csv(result: CU27Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.cells:
        rows.append(
            f"cell,{c.policy},{c.danger_class},,{c.random_breach:.6f},"
            f"{c.adversarial_breach:.6f},{c.mean_oracle_calls:.6f},{c.mean_rollbacks:.6f}"
        )
    for p in result.cost_law:
        rows.append(
            f"cost_law,,,{p.irreversible_fraction:.3f},,{p.after_commit_breach:.6f},"
            f"{p.routed_oracle_calls:.6f},{p.verify_all_oracle_calls:.6f}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
