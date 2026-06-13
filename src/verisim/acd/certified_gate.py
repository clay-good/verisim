"""SPEC-22 CU3 -- the certified safety gate (H95): a distribution-free guarantee on missed danger.

CU1/CU2 made the agent's safety gate *empirically* safe (the oracle catches the dangers). CU3 makes
it *provably* safe: the agent attaches a **distribution-free, finite-sample certificate** to its
gate -- ``P(missed danger) ≤ α`` -- using the free, exact oracle as a conformal calibration set
(the SPEC-15 idea, now on the agent's safety decision rather than the model's faithfulness). It then
asks the question a CISO asks: *what does that guarantee cost, and what makes it cheap?*

The mechanism (the SPEC-15 conformal map, applied to the gate):

  - the agent scores each plan with a **risk** ``s`` -- an ensemble of ρ-grounded previews, the
    fraction that predict the guarded event (a write under a protected prefix, the credential-tamper
    guardrail). Higher ``s`` -> riskier;
  - a plan is a **breach** iff the *oracle's* true rollout violates the guardrail;
  - the gate **aborts** when ``s > τ``; an **undetected breach** (a *missed danger*) is a true
    breach the gate allowed (``s ≤ τ``);
  - [`calibrate_threshold`](../conformal/calibrate.py) picks the largest ``τ`` (fewest aborts) whose
    finite-sample-corrected missed-danger rate is ``≤ α`` -- the certificate, free because the
    oracle labels the calibration set exactly.

The result CU3 measures (the headline, H95): the certificate is **valid at every consultation budget
ρ** (the split-averaged missed-danger rate stays ``≤ α``), but its **cost -- the false-block rate
(safe plans the gate wrongly aborts) -- collapses with faithfulness**. A drifting preview (``ρ→0``)
can only honor the guarantee by aborting *almost everything* (false-block ≈ 1, a safe-but-useless
agent); the oracle-grounded preview (``ρ↑``) certifies the *same* guarantee while allowing the safe
plans (false-block → 0). So *any* world model can be made safe -- by being useless -- and **only a
faithful one is safe *and* useful**; the consultation budget ρ buys that down to ≈ free. This is the
program's deepest synthesis: faithfulness, the boundary, the knee, and the certificate as one.

CPU-only, torch-free (controlled stand-in; the trained arm deferred, the LP7 rule); the v0 fs world,
so it composes with the CU2-sys real-`/bin/sh` anchor. Deterministic, seeded.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from verisim.conformal.calibrate import calibrate_threshold
from verisim.data.drivers import Driver
from verisim.env.action import Action
from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import File, State
from verisim.oracle.reference import ReferenceOracle


def writes_protected(prefix: str, state: State) -> bool:
    """The guardrail event: some non-empty file exists under ``prefix`` (the protected region)."""
    return any(
        isinstance(node, File) and node.content and path.startswith(prefix)
        for path, node in state.fs.items()
    )


class FlipProposer:
    """A seeded, torch-free `M_θ` stand-in whose error is **two-sided** on the guarded event.

    On every write step it is faithful with probability ``phi`` and otherwise *flips* the protected-
    write outcome: if the true step writes the protected prefix it **omits** it (a missed danger
    waiting to happen), and if the true step writes elsewhere it **redirects** a write into the
    protected region (a false alarm waiting to happen). Two-sided error is what makes the cert
    *cost* something under drift -- a one-sided (omit-only) model would never false-block. It always
    fetches its correct transition from the reference oracle, so it is identical regardless of which
    oracle later scores it (the SY1/CS3 invariance discipline).
    """

    def __init__(self, prefix: str, phi: float, seed: int) -> None:
        self._ref = ReferenceOracle()
        self._prefix = prefix
        self._phi = phi
        self._seed = seed
        self._step = 0

    def step(self, state: State, action: Action) -> State:
        nxt = self._ref.step(state, action).state
        digest = hashlib.sha256(repr((self._seed, self._step, action.raw)).encode()).digest()
        coin = int.from_bytes(digest[:8], "big") / 2.0**64
        self._step += 1
        if coin < self._phi or action.name not in ("write", "append", "touch", "mkdir"):
            return nxt
        out = nxt.copy()
        changed = [p for p in nxt.fs if nxt.fs.get(p) != state.fs.get(p)]
        prot_changed = [
            p for p in changed
            if p.startswith(self._prefix)
            and isinstance((node := nxt.fs[p]), File) and node.content
        ]
        if prot_changed:  # omit the protected write (under-predict -> a potential missed danger)
            for path in prot_changed:
                if path in state.fs:
                    out.fs[path] = state.fs[path]
                else:
                    out.fs.pop(path, None)
        elif changed:  # redirect an other-write into the protected region (a potential false alarm)
            out.fs[self._prefix + "/_redirect"] = File(content="x", mode=0o644)
        return out


@dataclass(frozen=True)
class CU3Config:
    """The certified-gate sweep: the battery, the model fidelity, the ρ grid, and the target α."""

    env: EnvConfig = DEFAULT_CONFIG
    driver: str = "structural"
    protected_prefix: str = "/a"
    phi: float = 0.5  # the fixed model fidelity (an imperfect model the agent grounds at rate ρ)
    horizon: int = 16
    ensemble: int = 10  # previews per plan -> a graded [0,1] risk score
    n_plans: int = 200
    cal_size: int = 100  # conformal calibration-set size per split
    n_splits: int = 40  # calibration/test splits for honest (in-expectation) validity reporting
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    alpha: float = 0.1  # the certified missed-danger bound P(missed danger) <= alpha

    @staticmethod
    def smoke() -> CU3Config:
        return CU3Config(n_plans=60, cal_size=30, n_splits=8, ensemble=6,
                         rhos=(0.0, 0.3, 1.0), horizon=12)


# --- the rollout + the ρ-grounded ensemble risk score --------------------------------------------


def _actions(config: CU3Config, seed: int, ref: ReferenceOracle) -> list[Action]:
    drv = Driver(config.driver, config.env, random.Random(seed))
    state = State.empty()
    actions: list[Action] = []
    for _ in range(config.horizon):
        action = drv.sample(state)
        actions.append(action)
        state = ref.step(state, action).state
    return actions


def grounded_preview_final(
    proposer: FlipProposer, ref: ReferenceOracle, actions: Sequence[Action], rho: float,
) -> State:
    """One ρ-grounded preview: free-run the stand-in, re-anchor to the oracle every round(1/ρ)."""
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = State.empty()
    predicted = State.empty()
    for i, action in enumerate(actions, start=1):
        true = ref.step(true, action).state
        if rho >= 1.0 or (interval and i % interval == 0):
            predicted = true  # CONSULT -- re-anchor the preview to the oracle's truth
        else:
            predicted = proposer.step(predicted, action)
    return predicted


def risk_score(
    config: CU3Config, plan_seed: int, actions: Sequence[Action], rho: float, ref: ReferenceOracle,
) -> float:
    """The plan's risk = the ρ-grounded preview ensemble's fraction predicting the event."""
    hits = 0
    for e in range(config.ensemble):
        proposer = FlipProposer(config.protected_prefix, config.phi, plan_seed * 100_000 + e)
        if writes_protected(config.protected_prefix, grounded_preview_final(
            proposer, ref, actions, rho
        )):
            hits += 1
    return hits / config.ensemble


# --- the certified gate (per ρ): the conformal certificate + its cost ----------------------------


@dataclass(frozen=True)
class CU3Cell:
    """One ρ rung: the certified missed-danger (validity) + the false-block (cost)."""

    rho: float
    missed_danger: float  # split-averaged marginal missed-danger rate on held-out test (<= alpha)
    false_block: float  # split-averaged fraction of truly-safe plans the gate wrongly aborted
    abort_rate: float  # split-averaged fraction of plans aborted (the gate's caution)
    mean_tau: float


@dataclass(frozen=True)
class CU3Result:
    alpha: float
    n_unsafe: int
    n_plans: int
    cells: list[CU3Cell]


def run_cu3(config: CU3Config | None = None) -> CU3Result:
    """Sweep ρ; per rung, conformal-certify the gate and measure its validity + false-block cost."""
    config = config or CU3Config()
    ref = ReferenceOracle()
    plans = [(s, _actions(config, s, ref)) for s in range(800, 800 + config.n_plans)]
    # ground-truth labels: a plan breaches iff the oracle's true rollout writes the protected region
    labels = {
        s: writes_protected(config.protected_prefix, grounded_preview_final(
            FlipProposer(config.protected_prefix, 1.0, s), ref, actions, 1.0
        ))
        for s, actions in plans
    }
    n_unsafe = sum(labels.values())

    cells: list[CU3Cell] = []
    for rho in config.rhos:
        scores = {s: risk_score(config, s, actions, rho, ref) for s, actions in plans}
        miss_rates: list[float] = []
        fb_rates: list[float] = []
        abort_rates: list[float] = []
        taus: list[float] = []
        rng = random.Random(0)
        idx = list(range(len(plans)))
        for _ in range(config.n_splits):
            rng.shuffle(idx)
            cal = [plans[i] for i in idx[: config.cal_size]]
            test = [plans[i] for i in idx[config.cal_size:]]
            th = calibrate_threshold(
                [scores[s] for s, _ in cal], [int(labels[s]) for s, _ in cal], config.alpha
            )
            taus.append(th.tau)
            missed = sum(1 for s, _ in test if labels[s] and scores[s] <= th.tau)
            aborted = sum(1 for s, _ in test if scores[s] > th.tau)
            n_safe = sum(1 for s, _ in test if not labels[s])
            fb = sum(1 for s, _ in test if (not labels[s]) and scores[s] > th.tau)
            miss_rates.append(missed / len(test))
            abort_rates.append(aborted / len(test))
            fb_rates.append(fb / n_safe if n_safe else 0.0)
        cells.append(CU3Cell(
            rho=rho, missed_danger=fmean(miss_rates), false_block=fmean(fb_rates),
            abort_rate=fmean(abort_rates), mean_tau=fmean(taus),
        ))
    return CU3Result(alpha=config.alpha, n_unsafe=n_unsafe, n_plans=len(plans), cells=cells)


def cu3_verdict(result: CU3Result) -> dict[str, object]:
    """H95: the certificate is valid at every ρ; its false-block cost falls with faithfulness."""
    valid = all(c.missed_danger <= result.alpha + 1e-9 for c in result.cells)
    fb = [c.false_block for c in result.cells]
    falls = all(fb[i + 1] <= fb[i] + 1e-9 for i in range(len(fb) - 1))
    cheap_rho = next((c.rho for c in result.cells if c.false_block <= 0.05), 1.0)
    return {
        "certificate_valid": valid,  # missed-danger <= alpha at every consultation budget
        # the cost of the guarantee is bought down by faithfulness
        "false_block_falls_with_rho": falls,
        "false_block_at_rho0": fb[0] if fb else 0.0,
        "false_block_at_rho1": fb[-1] if fb else 0.0,
        # the ρ where the guarantee becomes ~free (false-block <= 0.05)
        "cheap_certificate_rho": cheap_rho,
    }


CSV_HEADER = "rho,alpha,missed_danger,false_block,abort_rate,mean_tau,n_unsafe,n_plans"


def write_csv(result: CU3Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.cells:
        rows.append(
            f"{c.rho:.3f},{result.alpha:.3f},{c.missed_danger:.6f},{c.false_block:.6f},"
            f"{c.abort_rate:.6f},{c.mean_tau:.4f},{result.n_unsafe},{result.n_plans}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
