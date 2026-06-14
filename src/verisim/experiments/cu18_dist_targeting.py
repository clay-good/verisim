"""Experiment CU18 -- the asynchronous danger: target the medium, not the action (H111).

The targeting arc carried one cheap result across two worlds (CU10-CU12 network, CU16 host): verify
the single action class that is the danger's *genesis* and you defend it cheaply, un-gameably. CU18
carries it to the **distributed** world -- the one world the CU arc never touched, and the one whose
defining feature is an asynchronous medium -- and finds the boundary that breaks the cheap transfer.

The danger is a **stale read**: an agent reads a sensitive key from a node whose replica is behind
the converged value (the canonical distributed hazard). Its genesis (a write that creates a newer
version elsewhere) is separated from its consumption (a stale read on another node, later) by the
medium -- so the CU10-CU16 genesis-action target does **not** transfer. Four schedules on a battery
of stale-read deployments: uniform(ρ), model self-targeting, write_target (the literal CU10-16
transfer: verify writes to sensitive keys), and medium (consult a read iff the medium shows it is
stale -- the consumption-side closure), each on the random workload (CU10) and the adversarial worst
case over the attacker's timing (CU11).

The finding (H111): write_target does not transfer (its breach stays at the free rate while it
spends calls on writes -- a false sense of security); model self-targeting fails; uniform's knee
is a mirage under adversarial timing; only the **medium** target reaches the oracle's zero breach,
cheaply and un-gameably. Target the medium condition at consumption, not the action that planted it.

The trained distributed ``M_θ`` is the deferred GPU arm (per LP7); the schedule result keys on the
oracle and the medium grammar, not the model's competence, so the experiment runs a **worst-case
medium omitter** stand-in (it never foresees staleness -- the distributed face of CU8's omission
bias). Torch-free; runs in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.dist_targeting import (
    CU18Config,
    StaleOmitter,
    cu18_verdict,
    run_cu18,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU18 -- the asynchronous danger (H111).")
    parser.add_argument("--out", type=str, default="figures/cu18_dist_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--recall", type=float, default=0.0,
        help="the omitter's staleness recall (0.0 = worst-case medium omitter, the headline).",
    )
    args = parser.parse_args()

    config = CU18Config.smoke() if args.smoke else CU18Config()
    config = CU18Config(**{**config.__dict__, "recall": args.recall})
    model = StaleOmitter(recall=config.recall)
    result = run_cu18(model, config)
    verdict = cu18_verdict(result)

    print(f"\nCU18 / H111 -- the asynchronous danger (worst-case medium omitter, recall="
          f"{config.recall:g}; {result.n_episodes} deployments, horizon {result.horizon}):")
    print(f"  {'schedule':<30} {'random':>10} {'adversarial':>12} {'calls':>8}")
    for c in (*result.uniform, result.model, result.write_target, result.medium):
        print(f"  {c.label:<30} {c.random_breach:>10.3f} {c.adversarial_breach:>12.3f} "
              f"{c.mean_calls:>8.2f}")

    print(f"\n  medium target is safe + cheap: breach {verdict['medium_breach_rate']:.3f} at "
          f"{verdict['medium_calls']:.2f} calls = {verdict['medium_call_saving']:.1f}x cheaper "
          f"than the full oracle ({verdict['full_oracle_calls']:.1f})")
    print(f"  medium target is un-gameable: {verdict['medium_is_ungameable']} "
          f"(adversarial breach {verdict['medium_adversarial_breach']:.3f})")
    print(f"  the genesis-action target does NOT transfer: "
          f"{verdict['write_target_does_not_transfer']} (write_target breach "
          f"{verdict['write_target_breach_rate']:.3f} at "
          f"{verdict['write_target_calls']:.2f} calls)")
    print(f"  uniform knee is a mirage: {verdict['uniform_is_gameable']} (ρ="
          f"{verdict['uniform_knee_rho']:g}: random {verdict['uniform_knee_random_breach']:.3f} -> "
          f"adversarial {verdict['uniform_knee_adversarial_breach']:.3f})")
    print(f"  model self-targeting fails: {verdict['model_self_targeting_fails']} "
          f"(breach {verdict['model_breach_rate']:.3f} at {verdict['model_calls']:.2f} calls)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu18 import plot_cu18

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu18(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
