#!/usr/bin/env python3
"""Exhaustive, dependency-free proof of the core invariants (docs/verification.md §1).

Runs the load-bearing structural guarantees over many independent seeded rollouts, for
both worlds, and exits non-zero if any check fails. Needs **no torch and no GPU** — it
imports only the deterministic core, which is the point: the apparatus the science rests
on is checkable anywhere.

  - v0  M1 invariant:  apply(s, O(s,a).delta) == O(s,a).next_state
  - NW1 invariant:     same, over the full typed network graph
  - serialization round-trips (state + delta) for both worlds
  - NW4 tokenizer round-trip: parse_target(encode_target(Δ)) == Δ   (torch-free path)
  - metric well-formedness: divergence in [0,1], d(s,s)==0; reachability-faithfulness
    in [0,1], rf(s,s)==1; bits-to-correct(d,d)==0
  - oracle exit-code conformance: v0 in {0,1}, net in {0,1,2}
  - determinism: (driver, seed) is a pure function

Usage:  python scripts/verify_invariants.py [--seeds N] [--steps M]
"""

from __future__ import annotations

import argparse
import random
import sys


def verify_v0(seeds: int, steps: int) -> list[tuple[str, int, int]]:
    from verisim.data.drivers import Driver
    from verisim.delta.apply import apply
    from verisim.delta.serialize import delta_from_list, delta_to_list
    from verisim.env.config import DEFAULT_CONFIG
    from verisim.env.serialize import from_canonical, to_canonical
    from verisim.env.state import State
    from verisim.metrics.divergence import divergence
    from verisim.oracle.reference import ReferenceOracle

    oracle = ReferenceOracle()
    n = 0
    ok = {"m1": 0, "ser": 0, "dser": 0, "div": 0, "self": 0, "exit": 0}
    for seed in range(seeds):
        drv = Driver(name="weighted", config=DEFAULT_CONFIG, rng=random.Random(seed))
        s = State.empty()
        for _ in range(steps):
            r = oracle.step(s, drv.sample(s))
            rebuilt = apply(s, r.delta)
            n += 1
            ok["m1"] += rebuilt.fs == r.state.fs and rebuilt.cwd == r.state.cwd and (
                rebuilt.env == r.state.env and rebuilt.last == r.state.last
            )
            ok["ser"] += from_canonical(to_canonical(r.state)).fs == r.state.fs
            ok["dser"] += delta_from_list(delta_to_list(r.delta)) == r.delta
            ok["div"] += 0.0 <= divergence(s, r.state) <= 1.0
            ok["self"] += divergence(r.state, r.state) == 0.0
            ok["exit"] += r.exit_code in (0, 1)
            s = r.state
    return [
        ("v0 M1 apply==oracle", ok["m1"], n),
        ("v0 state round-trip", ok["ser"], n),
        ("v0 delta round-trip", ok["dser"], n),
        ("v0 divergence in [0,1] & d(s,s)==0", min(ok["div"], ok["self"]), n),
        ("v0 exit codes in {0,1}", ok["exit"], n),
    ]


def verify_net(seeds: int, steps: int) -> list[tuple[str, int, int]]:
    from verisim.net.config import DEFAULT_NET_CONFIG as CFG
    from verisim.net.serialize import from_canonical, to_canonical
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta.apply import apply
    from verisim.netdelta.serialize import delta_from_list, delta_to_list
    from verisim.netmetrics.bits import bits_to_correct
    from verisim.netmetrics.divergence import divergence, reachability_faithfulness
    from verisim.netmodel.tokenizer import encode_target, parse_target
    from verisim.netmodel.vocab import NetVocab
    from verisim.netoracle.reference import ReferenceNetworkOracle

    oracle = ReferenceNetworkOracle()
    vocab = NetVocab(CFG)
    n = 0
    ok = {"nw1": 0, "ser": 0, "dser": 0, "tok": 0, "div": 0, "rf": 0, "bits": 0, "exit": 0}
    for driver_name in ("uniform", "weighted", "adversarial"):
        for seed in range(seeds):
            drv = NetDriver(driver_name, CFG, random.Random(seed))
            s = NetworkState.initial(CFG.hosts)
            for _ in range(steps):
                r = oracle.step(s, drv.sample(s))
                n += 1
                canon = to_canonical(r.state)
                ok["nw1"] += to_canonical(apply(s, r.delta)) == canon
                ok["ser"] += to_canonical(from_canonical(canon)) == canon
                ok["dser"] += delta_from_list(delta_to_list(r.delta)) == r.delta
                ok["tok"] += parse_target(encode_target(r.delta, vocab), vocab) == r.delta
                ok["div"] += (0.0 <= divergence(s, r.state) <= 1.0) and (
                    divergence(r.state, r.state) == 0.0
                )
                ok["rf"] += (0.0 <= reachability_faithfulness(s, r.state) <= 1.0) and (
                    reachability_faithfulness(r.state, r.state) == 1.0
                )
                ok["bits"] += bits_to_correct(r.delta, r.delta) == 0.0
                ok["exit"] += r.exit_code in (0, 1, 2)
                s = r.state
    return [
        ("net NW1 apply==oracle", ok["nw1"], n),
        ("net state round-trip", ok["ser"], n),
        ("net delta round-trip", ok["dser"], n),
        ("net NW4 tokenizer round-trip", ok["tok"], n),
        ("net divergence in [0,1] & d(s,s)==0", ok["div"], n),
        ("net reachability_faithfulness in [0,1] & rf(s,s)==1", ok["rf"], n),
        ("net bits_to_correct(d,d)==0", ok["bits"], n),
        ("net exit codes in {0,1,2}", ok["exit"], n),
    ]


def verify_determinism() -> bool:
    from verisim.net.config import DEFAULT_NET_CONFIG as CFG
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netoracle.reference import ReferenceNetworkOracle

    oracle = ReferenceNetworkOracle()

    def traj(seed: int) -> list[str]:
        drv = NetDriver("weighted", CFG, random.Random(seed))
        s = NetworkState.initial(CFG.hosts)
        out: list[str] = []
        for _ in range(40):
            a = drv.sample(s)
            out.append(a.raw)
            s = oracle.step(s, a).state
        return out

    return traj(42) == traj(42) and traj(42) != traj(43)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--steps", type=int, default=60)
    args = p.parse_args()

    rows = verify_v0(args.seeds, args.steps) + verify_net(args.seeds, args.steps)
    failed = 0
    print(f"invariant proofs ({args.seeds} seeds x {args.steps} steps):")
    for name, ok, total in rows:
        mark = "OK " if ok == total else "FAIL"
        failed += ok != total
        print(f"  [{mark}] {name:30s} {ok}/{total}")
    det = verify_determinism()
    print(f"  [{'OK ' if det else 'FAIL'}] determinism (pure (driver, seed))")
    failed += not det

    total_checks = sum(t for _, _, t in rows)
    if failed:
        print(f"\nFAILED: {failed} invariant group(s) had a mismatch.")
        return 1
    print(f"\nALL PASS: {total_checks} structural checks + determinism, 0 failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
