"""FL6 -- why the real signal schedules well: ranking vs calibration (SPEC-19 §4, H77).

FL1/FL2 found that triggering consultations on the flagship's *real* decode-entropy nearly doubles
fixed-interval at equal budget -- and that the conformal trigger, not the speculative window,
carries it. That is surprising against SPEC-15 CF6, which found the *same* real signal **not
conformalizable** (it cannot certify a finite-sample coverage bound). FL6 resolves the tension with
one measurement: **ranking is not calibration.** A signal that *ranks* steps by how likely they are
to breach -- high Spearman with the true divergence -- is an excellent *scheduler* (spend
the budget on the most-uncertain steps) even if it cannot *calibrate* an absolute threshold `τ` for
which `P(undetected breach) ≤ α` holds (the conformal-coverage job CF6 measured). Timing a
consultation needs an *order*; certifying coverage needs a *level*. FL6 shows the signal supplies
the first and not the second, which is exactly why FL1's scheduling win and CF6's coverage null are
both true.

The measurement is direct: free-run the real flagship `M_θ`, record `(decode-entropy signal, true
divergence)` per step, and report (1) the Spearman rank correlation -- does the signal *order* it?
-- and (2) the *trigger precision lift* -- if you consult the top-budget steps by signal, what
fraction are real breaches, versus the base breach rate a uniform clock would catch? A signal that
ranks drift gives precision well above the base rate; an uninformative one matches it. Reuses FL1's
free-running collection ([`flagship_curve.build_calibration`](./flagship_curve.py)) and the
shipped [`landmark.geometry.spearman`](../landmark/geometry.py). CPU-local; CI runs smoke.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from verisim.landmark.geometry import spearman
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import NetworkState
from verisim.netdelta.apply import apply
from verisim.netloop import PartialNetOracle, ground_truth_rollout
from verisim.netmetrics.divergence import divergence
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .en1 import eval_actions
from .flagship import load_checkpoint
from .flagship_curve import FlagshipCurveConfig

if TYPE_CHECKING:
    from .flagship_curve import UncertainNetModel


def collect_signal_divergence(
    world_model: UncertainNetModel, config: FlagshipCurveConfig, oracle: NetOracle, net: NetConfig,
) -> list[tuple[float, float]]:
    """Free-run the real model; record `(decode-entropy signal, true divergence)` per step.

    The continuous-divergence sibling of `build_calibration` (which thresholds to a breach bit): FL6
    needs the real-valued divergence to measure how well the signal *ranks* drift, not just whether
    it crosses `ε`.
    """
    partial = PartialNetOracle(oracle)
    pairs: list[tuple[float, float]] = []
    for seed in config.cal_seeds:
        actions = eval_actions(oracle, net, config.cal_driver, seed, config.cal_steps)
        truth = ground_truth_rollout(partial, NetworkState.initial(net.hosts), actions)
        state = NetworkState.initial(net.hosts)
        for t, action in enumerate(actions):
            delta, signal = world_model.predict_delta_with_uncertainty(state, action)
            predicted = apply(state, delta)
            pairs.append((signal, divergence(truth[t + 1], predicted)))
            state = predicted  # free-run
    return pairs


def signal_diagnostic(
    pairs: list[tuple[float, float]], *, epsilon: float = 0.0, budget_frac: float = 0.2,
) -> dict[str, Any]:
    """Rank correlation + trigger-precision lift -- does the signal *order* drift (H77)?

    `spearman` is the ranking quality (signal vs divergence). `trigger_precision` consults the
    `budget_frac` highest-signal steps and reports the fraction that are real breaches; `base_rate`
    is the overall breach rate (what a uniform clock catches in expectation). `precision_lift` =
    trigger_precision − base_rate is the scheduling edge the signal buys (the FL1 mechanism).
    """
    signals = [s for s, _ in pairs]
    divs = [d for _, d in pairs]
    breaches = [1 if d > epsilon else 0 for d in divs]
    base_rate = (sum(breaches) / len(breaches)) if breaches else 0.0

    rho = spearman(signals, divs)
    # consult the top budget_frac of steps by signal; what fraction of THOSE are breaches?
    k = max(1, round(budget_frac * len(pairs)))
    top = sorted(range(len(pairs)), key=lambda i: signals[i], reverse=True)[:k]
    trigger_precision = sum(breaches[i] for i in top) / k if k else 0.0
    return {
        "n": len(pairs),
        "spearman": rho,
        "base_breach_rate": base_rate,
        "budget_frac": budget_frac,
        "trigger_precision": trigger_precision,
        "precision_lift": trigger_precision - base_rate,
        # H77: the signal RANKS drift (positive Spearman) and triggering beats the base rate, even
        # though CF6 found it non-conformalizable (cannot certify an absolute coverage level).
        "h77_supported": rho > 0.1 and trigger_precision > base_rate,
    }


def write_csv(pairs: list[tuple[float, float]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{s:.6f},{d:.6f}" for s, d in pairs]
    out.write_text("\n".join(["signal,divergence", *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="FL6 -- the real signal: ranking vs calibration.")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/fl6_signal_diag.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    config = FlagshipCurveConfig.smoke() if args.smoke else FlagshipCurveConfig()
    oracle = ReferenceNetworkOracle()
    pairs = collect_signal_divergence(ckpt.world_model, config, oracle, DEFAULT_NET_CONFIG)
    write_csv(pairs, args.out)
    diag = signal_diagnostic(pairs, epsilon=config.epsilon)
    print(f"FL6 signal diagnostic (n={diag['n']}):")
    print(f"  Spearman(signal, divergence) = {diag['spearman']:+.3f}  (does it RANK drift?)")
    print(f"  base breach rate             = {diag['base_breach_rate']:.3f}")
    print(f"  trigger precision @ρ={diag['budget_frac']:.2f}      = {diag['trigger_precision']:.3f}"
          f"  (lift {diag['precision_lift']:+.3f})")
    print(f"H77 (ranks drift, beats base): {'SUPPORTED' if diag['h77_supported'] else 'no'}")


if __name__ == "__main__":  # pragma: no cover
    main()
