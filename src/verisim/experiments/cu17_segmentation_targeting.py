"""Experiment CU17 -- the genesis-grammar boundary: target the genesis, not an action class (H110).

The targeting arc (CU10 cheap, CU11 un-gameable, CU12 knowledge-free, CU16 cross-world) always
targeted a single, syntactically visible danger action: the ``connect`` to a crown jewel. CU17 tests
whether the targeting *principle* survives a danger with a richer genesis grammar -- network
**segmentation exposure**, where a crown jewel becoming *reachable* from an untrusted host is born
by the config grammar (``link_up``/``svc_up``/``host_up``/``fw_allow``), not ``connect``. It scores
four schedules on a battery of segmented deployments: uniform(rho), the CU10-CU16 ``connect`` target
carried over verbatim, a *syntactic* genesis-grammar target, and the *semantic* reachability-closure
target -- on both the random workload and the adversarial worst case.

The finding (H110): the cheap ``connect`` target **does not transfer** (it is blind to the config
genesis -- breach stays at the free rate, a false sense of security at a low call count); the
syntactic ``grammar`` target reaches near-zero breach but **leaks through multi-hop intermediates**
(an exposing ``host_up`` of a non-jewel relay it cannot name) and overpays; only the semantic
``closure`` target reaches the oracle's zero breach, un-gameable, at a fraction of the full-oracle
cost. The principle: target the danger's genesis grammar, computed via the world's reachability
*closure* (the SPEC-12 machinery), not pattern-matched from an action class.

The trained network/host ``M_theta`` is the deferred GPU arm (per LP7). The schedule result keys on
the oracle and the grammar, not the model's competence, so the experiment runs a **worst-case
content omitter** stand-in. Torch-free; runs in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verisim.acd.segmentation_targeting import (
    CU17Config,
    NetOmitter,
    cu17_verdict,
    run_cu17,
    write_csv,
)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CU17 -- the genesis-grammar boundary (H110).")
    parser.add_argument("--out", type=str, default="figures/cu17_segmentation_targeting.csv")
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = CU17Config.smoke() if args.smoke else CU17Config()
    model = NetOmitter()
    result = run_cu17(model, config)
    verdict = cu17_verdict(result)

    print(f"\nCU17 / H110 -- the genesis-grammar boundary (worst-case content omitter; "
          f"{result.n_episodes} segmented deployments, horizon {result.horizon}):")
    print(f"  {'schedule':<24} {'random':>10} {'adversarial':>12} {'calls':>8}")
    for c in (*result.uniform, result.connect, result.grammar, result.closure):
        print(f"  {c.label:<24} {c.random_breach:>10.3f} {c.adversarial_breach:>12.3f} "
              f"{c.mean_calls:>8.2f}")

    print(f"\n  connect target does NOT transfer: {verdict['connect_fails_to_transfer']} "
          f"(breach {verdict['connect_random_breach']:.3f} at {verdict['connect_calls']:.2f} calls "
          f"vs free {verdict['free_breach_rate']:.3f})")
    print(f"  syntactic grammar leaks adversarially: {verdict['grammar_leaks_adversarially']} "
          f"(random {verdict['grammar_random_breach']:.3f}, adversarial "
          f"{verdict['grammar_adversarial_breach']:.3f}, {verdict['grammar_calls']:.2f} calls)")
    print(f"  closure target is safe + cheap: breach {verdict['closure_random_breach']:.3f} at "
          f"{verdict['closure_calls']:.2f} calls = {verdict['closure_call_saving']:.1f}x cheaper "
          f"than the full oracle ({verdict['full_oracle_calls']:.1f})")
    print(f"  closure is un-gameable: {verdict['closure_is_ungameable']} "
          f"(adversarial breach {verdict['closure_adversarial_breach']:.3f})")

    out = write_csv(result, args.out)
    print(f"\nwrote {out}")
    try:
        from figures.plot_cu17 import plot_cu17

        plot_path = Path(args.plot) if args.plot else Path(out).with_suffix(".png")
        plot_cu17(result, plot_path)
        print(f"wrote {plot_path}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
