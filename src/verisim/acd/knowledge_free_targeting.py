"""SPEC-22 CU12 (H105): knowledge-free targeting -- target the grammar, not the assets.

CU10/CU11 made targeted verification cheap and un-gameable by spending the oracle on exactly the
``connect``-to-crown-jewel actions -- but that defense rests on an assumption a real defender cannot
always meet: that the **crown-jewel inventory is complete**. Asset inventories drift, miss shadow
services, and lag the network; an adversary exfiltrates to the host you *didn't* flag. CU12 asks
what the cheap targeting costs you when the inventory is wrong, and what to do instead.

There are two distinct structural targets, and the difference is the whole result:

  - **asset-indexed** (CU10/CU11): verify ``connect`` to a host in the *known* crown-jewel set
    ``K``. Cheap, but blind to any sensitive host outside ``K`` -- an incomplete inventory degrades
    it to the *unverified* breach rate while still reporting "verified," a false sense of security.
  - **grammar-indexed**: verify *every* flow-genesis action -- every ``connect``, regardless of
    destination. In the v0 net grammar a flow is born only by ``connect`` (the only edit that opens
    a flow is ``FlowOpen``, emitted solely by ``connect``; the CU11 invariant), so verifying every
    ``connect`` catches every exfiltration to *any* host. This needs **zero asset knowledge**, is
    un-gameable against both adversarial timing *and* the adversary's choice of target, and --
    because ``connect`` is still a small fraction of the action stream -- far cheaper than verifying
    everything.

Measured against the **true** sensitive set ``T`` (which the defender only partially knows as ``K``)
on the same battery and real trained ``M_θ`` as CU10. The prediction (H105): asset-indexed targeting
degrades toward the unverified breach as ``K`` shrinks below ``T`` (and is fully gameable
adversarially -- the attacker picks a host in ``T\\K``), while grammar-indexed targeting reaches the
oracle's zero breach **inventory-independently** and still costs far less than the full oracle.
**Refuted if** asset-indexed breach does not rise as the inventory becomes incomplete, or
grammar-indexed targeting does not reach ~0 breach, or it costs about as much as the full oracle (no
saving over verifying everything). The defender principle: **when you cannot trust your asset
inventory, target the grammar, not the assets.**

Torch-free core: :func:`run_cu12` takes any object with ``predict_delta(state, action)`` -- the
trained ``M_θ`` in the experiment (torch-gated), cheap stand-ins in the tests. Danger and
reachability are grounded in the real :class:`ReferenceNetworkOracle`. Deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from verisim.acd.adversarial_targeting import reachable_exfils
from verisim.acd.closed_loop_net import _new_flows
from verisim.acd.safety_horizon import Deployment
from verisim.acd.targeted_verification import CU10Config, build_deployments
from verisim.netdelta.apply import apply
from verisim.netoracle.base import NetOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

KFSchedule = Literal["uniform", "asset", "grammar"]


@dataclass(frozen=True)
class CU12Config:
    """The knowledge-free sweep: the *true* sensitive set, inventory levels, the uniform grid."""

    true_sensitive: tuple[str, ...] = ("h0", "h4")  # T -- every host an exfil to is a breach
    inventories: tuple[tuple[str, ...], ...] = ((), ("h0",), ("h0", "h4"))  # K levels (⊆ T)
    horizon: int = 48
    n_seeds: int = 400
    max_episodes: int = 200
    rhos: tuple[float, ...] = (0.5, 1.0)  # the uniform reference points
    seed0: int = 6000
    driver: str = "weighted"

    def base(self) -> CU10Config:
        """The CU10 deployment config keyed on the *true* sensitive set (exfil to any T counts)."""
        return CU10Config(
            protected_servers=self.true_sensitive, horizon=self.horizon, n_seeds=self.n_seeds,
            max_episodes=self.max_episodes, rhos=self.rhos, seed0=self.seed0, driver=self.driver,
        )

    @staticmethod
    def smoke() -> CU12Config:
        return CU12Config(horizon=24, n_seeds=80, max_episodes=24, rhos=(0.5, 1.0))


def run_deployment_kf(
    model: object, oracle: NetOracle, deployment: Deployment, true_sensitive: frozenset[str],
    schedule: KFSchedule, known: frozenset[str], rho: float = 0.0,
) -> tuple[bool, int]:
    """Run the closed loop under one schedule; breach is an exfil to **any** true-sensitive host.

    ``uniform`` consults every ``round(1/ρ)`` steps; ``asset`` consults ``connect`` to a *known*
    crown jewel (``known``); ``grammar`` consults every ``connect``. The gate always judges danger
    against the full true-sensitive set ``true_sensitive`` -- the defender is scored on the real
    asset set, not the one it happens to know.
    """
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = deployment.start
    belief = deployment.start
    breached = False
    calls = 0
    for i, action in enumerate(deployment.actions, start=1):
        true_next = oracle.step(true, action).state
        opens_sensitive = bool(_new_flows(true, true_next, true_sensitive))

        if schedule == "uniform":
            consult = rho >= 1.0 or bool(interval and i % interval == 0)
        elif schedule == "asset":
            consult = action.name == "connect" and action.args[1] in known
        else:  # grammar: every flow-genesis action
            consult = action.name == "connect"

        if consult:
            calls += 1
            gate_blocks = opens_sensitive
            belief_next = true_next
        else:
            belief_next = apply(belief, model.predict_delta(belief, action))  # type: ignore[attr-defined]
            gate_blocks = bool(_new_flows(belief, belief_next, true_sensitive))

        if not gate_blocks:
            if opens_sensitive:
                breached = True
            true = true_next
            belief = belief_next
    return breached, calls


def adversarial_breach_kf(
    oracle: NetOracle, deployment: Deployment, true_sensitive: frozenset[str],
    schedule: KFSchedule, known: frozenset[str],
) -> bool:
    """Worst case over the attacker's timing *and* target: can one exfil evade this schedule?

    The attacker mounts a single ``connect`` to any true-sensitive host at any step where it is
    reachable. It is blocked only if the schedule consults on that action (the oracle then sees the
    real open). ``asset`` consults only ``connect`` to a *known* jewel, so an exfil to a host in
    ``T\\K`` is never consulted and breaches; ``grammar`` consults every ``connect`` and is immune.
    (No model needed: an unconsulted exfil rides the omitting model's preview -- the CU8 law.)
    """
    true = deployment.start
    for action in deployment.actions:
        for exfil in reachable_exfils(true, true_sensitive):
            # grammar consults every connect; asset consults only connects to a known jewel
            consult = exfil.args[1] in known if schedule == "asset" else True
            if not consult:
                return True
        true = oracle.step(true, action).state
    return False


@dataclass(frozen=True)
class CU12Cell:
    """One defense over the battery: its random-timing and adversarial breach, and its cost."""

    schedule: str
    label: str
    inventory_frac: float | None  # |K| / |T| for the asset target (None for uniform/grammar)
    breach_random: float
    breach_adversarial: float
    mean_calls: float


@dataclass(frozen=True)
class CU12Result:
    n_episodes: int
    horizon: int
    n_true_sensitive: int
    uniform: list[CU12Cell]
    asset: list[CU12Cell]  # one per inventory level
    grammar: CU12Cell


def _asset_cell(
    model: object, oracle: NetOracle, deployments: list[Deployment], T: frozenset[str],
    known: frozenset[str], n_T: int,
) -> CU12Cell:
    rand = [run_deployment_kf(model, oracle, d, T, "asset", known, 0.0) for d in deployments]
    adv = [adversarial_breach_kf(oracle, d, T, "asset", known) for d in deployments]
    label = "asset K={%s}" % ("+".join(sorted(known)) if known else "∅")
    return CU12Cell(
        schedule="asset", label=label, inventory_frac=len(known) / n_T if n_T else 0.0,
        breach_random=fmean(b for b, _ in rand) if rand else 0.0,
        breach_adversarial=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
    )


def run_cu12(model: object, config: CU12Config | None = None) -> CU12Result:
    """Sweep inventory completeness; compare asset-indexed, grammar-indexed, and uniform targets."""
    config = config or CU12Config()
    oracle: NetOracle = ReferenceNetworkOracle()
    base = config.base()
    deployments = build_deployments(base, oracle)
    T = frozenset(config.true_sensitive)
    n_T = len(T)

    uniform: list[CU12Cell] = []
    for rho in config.rhos:
        rand = [run_deployment_kf(model, oracle, d, T, "uniform", frozenset(), rho)
                for d in deployments]
        uniform.append(CU12Cell(
            schedule="uniform", label=f"uniform ρ={rho:g}", inventory_frac=None,
            breach_random=fmean(b for b, _ in rand) if rand else 0.0,
            breach_adversarial=fmean(b for b, _ in rand) if rand else 0.0,
            mean_calls=fmean(c for _, c in rand) if rand else 0.0,
        ))

    asset = [_asset_cell(model, oracle, deployments, T, frozenset(k), n_T)
             for k in config.inventories]

    g_rand = [run_deployment_kf(model, oracle, d, T, "grammar", frozenset(), 0.0)
              for d in deployments]
    g_adv = [adversarial_breach_kf(oracle, d, T, "grammar", frozenset()) for d in deployments]
    grammar = CU12Cell(
        schedule="grammar", label="grammar (all connects)", inventory_frac=None,
        breach_random=fmean(b for b, _ in g_rand) if g_rand else 0.0,
        breach_adversarial=fmean(g_adv) if g_adv else 0.0,
        mean_calls=fmean(c for _, c in g_rand) if g_rand else 0.0,
    )
    return CU12Result(
        n_episodes=len(deployments), horizon=config.horizon, n_true_sensitive=n_T,
        uniform=uniform, asset=asset, grammar=grammar,
    )


def cu12_verdict(result: CU12Result) -> dict[str, object]:
    """H105: asset-indexed degrades with the inventory; grammar targeting is asset-free."""
    complete = result.asset[-1]  # K == T
    incomplete = next((c for c in result.asset
                       if c.inventory_frac is not None and 0.0 < c.inventory_frac < 1.0), complete)
    grammar = result.grammar
    full = next((c for c in result.uniform if c.breach_random <= grammar.breach_random + 1e-9),
                result.uniform[-1])
    return {
        # an incomplete inventory makes the cheap target a false sense of security
        "incomplete_inventory_frac": incomplete.inventory_frac,
        "incomplete_breach_random": incomplete.breach_random,
        "incomplete_breach_adversarial": incomplete.breach_adversarial,
        "asset_target_degrades": incomplete.breach_random > complete.breach_random + 0.1,
        "asset_target_gameable": incomplete.breach_adversarial > 0.5,
        # the grammar target is inventory-independent, safe, and still cheap
        "grammar_breach_random": grammar.breach_random,
        "grammar_breach_adversarial": grammar.breach_adversarial,
        "grammar_is_safe": grammar.breach_random <= complete.breach_random + 1e-9
        and grammar.breach_adversarial <= 1e-9,
        "grammar_calls": grammar.mean_calls,
        "full_oracle_calls": full.mean_calls,
        "grammar_cheaper_than_full": grammar.mean_calls < full.mean_calls,
        "grammar_saving_vs_full": full.mean_calls / grammar.mean_calls
        if grammar.mean_calls > 0 else float("inf"),
    }


CSV_HEADER = (
    "schedule,label,inventory_frac,breach_random,breach_adversarial,mean_calls,n_episodes,horizon"
)


def write_csv(result: CU12Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in (*result.uniform, *result.asset, result.grammar):
        frac = f"{c.inventory_frac:.3f}" if c.inventory_frac is not None else ""
        rows.append(
            f"{c.schedule},{c.label},{frac},{c.breach_random:.6f},{c.breach_adversarial:.6f},"
            f"{c.mean_calls:.6f},{result.n_episodes},{result.horizon}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
