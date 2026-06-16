"""Experiment CU36 -- the grounded verifier: CU35's fidelity law against REAL verifiers (H129).

CU35 proved the dual coverage law -- a deployed verifier need only be faithful on the danger surface
-- but with an abstract hash-coin verifier. CU36 grounds it against two real, deployable,
structurally-defined partial verifiers (the cheap state-diff monitor of CU34 and the
structure/content boundary of SPEC-20 as a verifier) over the CU34 host CIA battery. The twist: each
verifier's fidelity profile is read off the danger grammar a priori (does it observe the channel the
danger mutates?), so its safety is decided without ever measuring its fidelity -- the state-diff
verifier is exactly as safe as the perfect oracle on integrity + availability and exactly as blind
as no gate on the footprintless confidentiality leg, while the structure verifier is exact only on
availability. Torch-free (worst-case omitter + exact host oracle); ~5s.
"""

from __future__ import annotations

import argparse

from verisim.acd.footprintless_targeting import CU34Config
from verisim.acd.grounded_verifier import cu36_verdict, run_cu36, write_csv


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU36 -- the grounded verifier (H129).")
    parser.add_argument("--out", type=str, default="figures/cu36_grounded_verifier.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU34Config.smoke() if args.smoke else CU34Config()
    result = run_cu36(config)
    v = cu36_verdict(result)

    print("\nCU36 / H129 -- the grounded verifier (CU35's fidelity law against real verifiers):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")
    print("  ADVERSARIAL BREACH per (real verifier x CIA leg) -- 0 = as safe as the oracle:")
    print(f"    {'verifier':14s} {'leg':16s} {'mutates':24s} {'faithful':>8s} "
          f"{'adv':>6s} {'==?':>14s}")
    no_gate = {c.leg: c.adversarial_breach for c in result.cells if c.verifier == "no gate"}
    for c in result.cells:
        same = ("oracle" if c.adversarial_breach <= 1e-9
                else "no-gate" if abs(c.adversarial_breach - no_gate[c.leg]) <= 1e-9 else "?")
        print(f"    {c.verifier:14s} {c.leg:16s} {c.danger_mutates:24s} "
              f"{('yes' if c.faithful_on_surface else 'NO'):>8s} {c.adversarial_breach:6.3f} "
              f"{same:>14s}")

    print(f"\n  localization holds everywhere (faithful<=>as-safe-as-oracle, else as-blind-as-"
          f"no-gate) = {v['localization_holds_everywhere']}")
    print(f"  structural predictor exact (faithful_on_surface == danger mutates an observed "
          f"channel) = {v['structural_predictor_exact']}")
    print(f"  state-diff: globally partial yet locally safe = "
          f"{v['state_diff_globally_partial_locally_safe']} "
          f"(safe integ+avail, BLIND confidentiality)")
    print(f"  structure: globally partial yet locally safe = "
          f"{v['structure_globally_partial_locally_safe']} "
          f"(safe avail, BLIND integrity+confidentiality)")

    out = write_csv(result, args.out)
    print(f"\n  wrote {out}")

    if args.plot:
        from figures.plot_cu36 import plot_cu36

        path = plot_cu36(result, args.plot)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
