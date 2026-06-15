"""Experiment CU23 -- the second generative test: the framework predicts again, in a new world.

CU22 made the generative claim once (network availability). CU23 (H116) carries it to the **host**
world on a different resource -- **process liveness** (an automated containment agent terminating a
critical defensive daemon). It applies the CU21 ``unified_targeting`` engine, using ``covers`` as an
a-priori predictor: the host world's own CU16 integrity target (write-to-jewel-fd) is predicted to
leak (a termination is not a write), a syntactic terminate rule is predicted to cover (process death
has no cascade -- the OPPOSITE of CU22's syntactic, which leaked via multi-hop), and the
framework-derived process-liveness closure is predicted to be safe + cheap + un-gameable. The
worst-case-omitter run confirms every prediction. Torch-free; ~3s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.process_availability_targeting import (
    CU23Config,
    cu23_verdict,
    run_cu23,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU23 -- the second generative test (H116).")
    parser.add_argument("--out", type=str, default="figures/cu23_process_availability.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU23Config.smoke() if args.smoke else CU23Config()
    result = run_cu23(config)
    v = cu23_verdict(result)

    print("\nCU23 / H116 -- the second generative test (host process-availability, worst-case "
          "omitter):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}, "
          f"free breach {v['free_breach_rate']:.3f}\n")
    print("  candidate targets (covers = the a-priori prediction, before any deployment runs):")
    for c in result.candidates:
        print(f"    {c.label:42s}  covers={c.covers!s:5s}  "
              f"random {c.random_breach:.3f}  adversarial {c.adversarial_breach:.3f}  "
              f"{c.mean_calls:.2f} calls")
    print(f"\n  derived process-liveness closure: safe={v['liveness_is_safe']}  "
          f"un-gameable={v['liveness_is_ungameable']}  "
          f"= {v['liveness_call_saving']:.1f}x cheaper than full oracle "
          f"({v['full_oracle_calls']:.0f} calls)")
    print(f"  CU16 integrity carry-over (write-to-fd): covers={v['write_covers']}  "
          f"leaks={v['write_leaks']} (false security -- it watches the wrong resource)")
    print(f"  syntactic terminate: covers={v['syntactic_covers']} (no cascade -- OPPOSITE of "
          f"CU22's), overpays={v['syntactic_overpays_vs_liveness']} "
          f"({v['syntactic_calls']:.2f} vs {v['liveness_calls']:.2f} calls)")
    print(f"  uniform knee rho={v['uniform_knee_rho']:g}: random "
          f"{v['uniform_knee_random_breach']:.3f} -> adversarial "
          f"{v['uniform_knee_adversarial_breach']:.3f} (gameable={v['uniform_is_gameable']})")
    print(f"  model self-targeting fails={v['model_self_targeting_fails']}  "
          f"perfect model self-governs={v['oracle_self_governs']}")
    print(f"\n  GENERATIVE HEADLINE (H116): covers predicted every candidate's fate; confirmed = "
          f"{v['framework_predicts_every_candidate']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu23 import plot_cu23

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu23(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
