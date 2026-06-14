"""Experiment CU16 -- cross-world targeting: the host danger surface is grammar-fixed too (H109).

The whole targeting arc (CU10 cheap, CU11 un-gameable, CU12 knowledge-free) was measured on the
**network** world. CU16 carries the headline result to the **host** world (credential / config
tampering) to test whether it is network-specific or a general property of an oracle-grounded world.
It scores the same three schedules -- uniform(ρ), model self-targeting, structure (verify the
``write``-to-a-protected-path actions) -- on a host deployment battery, on **both** the random
workload (CU10) and the adversarial worst case over the attacker's timing (CU11).

The finding (H109): exactly as on the network -- uniform needs the full oracle to reach zero breach
and its sub-oracle knee is a mirage under adversarial timing; model self-targeting fails (an omitter
cannot flag its own blind spots); **structure reaches zero breach at a fraction of the calls AND is
un-gameable**, because a ``/passwd`` corruption is born only by a ``write`` to an fd bound to that
path -- the host flow-genesis surface, localized through the fd table the model learns faithfully.

The host trained ``M_θ`` is the deferred GPU arm (its per-step rollout over fork-heavy workloads is
pathologically slow on the throttled local CPU -- the LP7 rule). The schedule result keys on the
oracle and the grammar, not the model's competence, so the experiment runs a **worst-case content
omitter** stand-in (faithful on structure, omits file writes -- the realistic drift direction CU8
measured and CU1 confirmed the real host ``M_θ`` exhibits). Torch-free; runs in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.host_targeting import (
    CU16Config,
    HostOmitter,
    cu16_verdict,
    run_cu16,
    write_csv,
)
from verisim.hostoracle.reference import ReferenceHostOracle


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU16 -- cross-world host targeting (H109).")
    parser.add_argument("--out", type=str, default="figures/cu16_host_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--recall", type=float, default=0.0,
        help="the omitter's write recall (0.0 = worst-case content omitter, the headline).",
    )
    args = parser.parse_args()

    config = CU16Config.smoke() if args.smoke else CU16Config()
    config = CU16Config(**{**config.__dict__, "recall": args.recall})
    model = HostOmitter(recall=config.recall, oracle=ReferenceHostOracle())
    result = run_cu16(model, config)
    verdict = cu16_verdict(result)

    print(f"\nCU16 / H109 -- cross-world host targeting (worst-case content omitter, recall="
          f"{config.recall:g}; {result.n_episodes} deployments, horizon {result.horizon}):")
    print(f"  {'schedule':<28} {'random':>10} {'adversarial':>12} {'calls':>8}")
    for c in (*result.uniform, result.model, result.structure):
        print(f"  {c.label:<28} {c.random_breach:>10.3f} {c.adversarial_breach:>12.3f} "
              f"{c.mean_calls:>8.2f}")

    print(f"\n  structure is safe + cheap: breach {verdict['structure_breach_rate']:.3f} at "
          f"{verdict['structure_calls']:.2f} calls = "
          f"{verdict['structure_call_saving']:.1f}x cheaper than the full oracle "
          f"({verdict['full_oracle_calls']:.1f})")
    print(f"  structure is un-gameable: {verdict['structure_is_ungameable']} "
          f"(adversarial breach {verdict['structure_adversarial_breach']:.3f})")
    print(f"  uniform knee is a mirage: {verdict['uniform_is_gameable']} (ρ="
          f"{verdict['uniform_knee_rho']:g}: random {verdict['uniform_knee_random_breach']:.3f} -> "
          f"adversarial {verdict['uniform_knee_adversarial_breach']:.3f})")
    print(f"  model self-targeting fails: {verdict['model_self_targeting_fails']} "
          f"(breach {verdict['model_breach_rate']:.3f} at {verdict['model_calls']:.2f} calls)")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu16 import plot_cu16

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu16(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
