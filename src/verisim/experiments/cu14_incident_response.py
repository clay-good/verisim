"""Experiment CU14 -- the defended incident: the whole stack on one scenario (SPEC-22, H107).

An autonomous incident-response agent must restore work connectivity (``h1/h2/h3``) on a compromised
segment salted with exfiltration lures (``connect`` to crown jewels ``h0/h4``). It is run under four
defenses -- undefended (trust the model), paranoid (block every connect), structure (verify the
connect-to-jewel actions, CU10/CU12), full oracle (verify every step) -- and scored on all three
axes at once: mission completed (utility), exfiltrated (safety), oracle calls (cost). One
representative incident is replayed step by step.

The finding (H107): only structure is in the all-good corner -- it completes the mission (the
one-sided model never false-aborts a benign connect, so targeting the danger surface is free of
mission cost), reaches the oracle's zero breach, and does so at a fraction of the oracle's cost. The
trained model is torch-gated; the incident core is torch-free.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.incident_response import IncidentConfig, cu14_verdict, run_cu14, write_csv


def _print_playback(title: str, steps: tuple, *, max_rows: int = 14) -> None:  # type: ignore[type-arg]
    print(f"\n  {title}")
    print(f"    {'step':>4} {'action':<22} {'oracle':<8} {'consult':<8} {'decision':<8} note")
    shown = 0
    for s in steps:
        interesting = s.is_connect and s.dst_class in ("jewel", "work")
        if not interesting and shown >= max_rows:
            continue
        if not interesting:
            continue
        truth = "EXFIL" if s.oracle_exfil else ("work" if s.oracle_work else "-")
        note = ""
        if s.breach:
            note = "<-- BREACH (walked the lure)"
        elif s.consulted and s.oracle_exfil:
            note = "<-- caught the lure"
        elif s.oracle_work and s.decision == "EXECUTE":
            note = "mission progress"
        print(f"    {s.step:>4} {s.action:<22} {truth:<8} "
              f"{'yes' if s.consulted else 'no':<8} {s.decision:<8} {note}")
        shown += 1


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU14 -- the defended incident (SPEC-22 H107).")
    parser.add_argument("--out", type=str, default="figures/cu14_incident_response.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/flagship/net-l",
        help="reuse a frozen flagship checkpoint instead of retraining (the ~11-min train).",
    )
    parser.add_argument("--retrain", action="store_true", help="train a fresh flagship M_θ.")
    args = parser.parse_args()

    if args.retrain or not Path(args.checkpoint, "model.pt").exists():
        from verisim.experiments.flagship import FlagshipConfig, train_flagship

        base = FlagshipConfig.smoke() if args.smoke else FlagshipConfig()
        print("training the network flagship M_θ (one-time)...")
        model, _ = train_flagship(base)
    else:
        from verisim.experiments.flagship import load_checkpoint

        print(f"reusing frozen flagship checkpoint {args.checkpoint} (no retrain)...")
        model = load_checkpoint(args.checkpoint).world_model

    config = IncidentConfig.smoke() if args.smoke else IncidentConfig()
    result = run_cu14(model, config)
    verdict = cu14_verdict(result)

    print(f"\nCU14 / H107 -- the defended incident on a REAL trained network M_θ "
          f"({result.n_episodes} incidents, horizon {result.horizon}):")
    print(f"  {'defense':<34} {'mission':>8} {'breach':>8} {'calls':>8}")
    for c in result.ledgers:
        print(f"  {c.label:<34} {c.completion_rate:>8.2f} {c.breach_rate:>8.2f} "
              f"{c.mean_calls:>8.2f}")

    print(f"\n  only structure is in the all-good corner: "
          f"{verdict['structure_in_all_good_corner']}")
    print(f"    safe ({verdict['structure_breach_rate']:.2f} breach, oracle-equal), "
          f"on-mission ({verdict['structure_completion']:.2f}), and "
          f"{verdict['cost_ratio_oracle_over_structure']:.1f}x cheaper than the full oracle "
          f"({verdict['structure_calls']:.2f} vs {verdict['oracle_calls']:.2f} calls)")
    print(f"  undefended exfiltrates ({verdict['undefended_breaches']:.2f}) though it completes "
          f"({verdict['undefended_completes']:.2f}); paranoid is safe but useless "
          f"({verdict['paranoid_safe_but_useless']})")

    print(f"\n  PLAYBACK -- incident #{result.playback_index} "
          f"(the same actions, undefended vs structure):")
    _print_playback("undefended -- trusts the omitting model:", result.playback_undefended)
    _print_playback("structure -- verifies the connect-to-jewel:", result.playback_structure)

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu14 import plot_cu14

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu14(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
