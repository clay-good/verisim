"""Experiment CU31 -- the concurrent (multi-agent) safety gate (SPEC-22, H124).

Every CU result (CU1-CU30) is single-agent. CU31 carries the targeting result to a FLEET of agents
sharing one network, and the result is a sharp negative with a clean fix, both predicted a priori by
the coverage machinery (CU21) generalized to multiple principals:

  - THE NEGATIVE -- the realistic per-agent gate (the CU26 covering closure scoped to one principal,
    "no single agent should hoard B sensitive hosts") is genuinely covering against a single
    principal, but a fleet adversary spreads the collection so each agent stays under B while the
    JOINT union crosses it -- the per-agent gate never fires (adv breach 1.000, `covers`=False).
  - THE FIX -- the same closure on the JOINT accumulator, a single shared gate over the merged
    action stream, covers the joint danger un-gameably (0.000) at the cost of the boundary crossings
    (cheap), strictly below the paranoid shared grammar.
  - THE LAW -- `covers` flips at K=1->2: the per-agent gate covers iff a single principal must hold
    >= B (K=1) and leaks the moment the fleet fragments; the shared gate is invariant to K.

Torch-free; danger, accumulator, the fleet low-and-slow adversary, and every target are grounded in
the real ReferenceNetworkOracle (the worst-case-omitter substrate of CU16-CU26).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.concurrent_targeting import CU31Config, cu31_verdict, run_cu31, write_csv


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(
        description="CU31 -- the multi-agent safety gate (SPEC-22 H124).")
    parser.add_argument("--out", type=str, default="figures/cu31_concurrent_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU31Config.smoke() if args.smoke else CU31Config()
    result = run_cu31(config)
    v = cu31_verdict(result)

    print("CU31 / H124 -- the concurrent (multi-agent) safety gate:")
    print(f"  fleet K={result.n_agents}, joint budget B={result.budget}, "
          f"{result.n_episodes} deployments, horizon {result.horizon}")
    print(f"  {'target (joint danger)':<46} {'rand':>6} {'adv':>6} {'calls':>7} {'covers':>7}")
    for c in result.candidates:
        print(f"  {c.label:<46} {c.random_breach:>6.3f} {c.adversarial_breach:>6.3f} "
              f"{c.mean_calls:>7.3f} {c.covers!s:>7}")
    print(f"  {'full oracle':<46} {result.full_oracle.random_breach:>6.3f} "
          f"{result.full_oracle.adversarial_breach:>6.3f} {result.full_oracle.mean_calls:>7.3f}")
    print(f"\n  per-agent gate LEAKS the fleet: {v['per_agent_leaks_fleet']} "
          f"(adv {v['per_agent_adversarial_breach']:.3f}, covers {v['per_agent_covers_joint']})")
    print(f"  shared closure un-gameable: {v['shared_closure_is_ungameable']} "
          f"(adv {v['shared_closure_adversarial_breach']:.3f}) at "
          f"{v['shared_closure_calls']:.3f} calls "
          f"= {v['shared_closure_saving_vs_grammar']:.1f}x cheaper than grammar, "
          f"{v['shared_closure_saving_vs_full']:.1f}x than full oracle")
    print(f"  uniform knee a mirage: {v['uniform_is_gameable']}; model self-targeting fails: "
          f"{v['model_self_targeting_fails']}; "
          f"perfect model self-governs: {v['oracle_self_governs']}")
    print("  fragmentation law:")
    for k in sorted(result.by_n_agents):
        pa = result.by_n_agents[k]["per_agent"]
        sh = result.by_n_agents[k]["shared_closure"]
        print(f"    K={k}: per-agent adv={pa[0]:.3f} covers={pa[2]} | "
              f"shared adv={sh[0]:.3f} covers={sh[2]} calls={sh[1]:.3f}")
    print(f"  per-agent covers iff single principal: "
          f"{v['per_agent_covers_only_single_principal']}; "
          f"leak grows with K: {v['per_agent_leak_grows_with_fragmentation']}; "
          f"shared invariant: {v['shared_invariant_to_fragmentation']}")
    print(f"  framework predicts every candidate a priori: "
          f"{v['framework_predicts_every_candidate']}")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu31 import plot_cu31

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu31(result, v, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
