"""Experiment CX0: the oracle is an exact SCM (H60, SPEC-17 §6). *The gate that licenses rung 3.*

The identification the whole spec rests on, made empirical. For each world it (a) **abducts** ``U``
from
the seed and replays ``F``, asserting two independent recoveries are bit-identical -- the
SCM-contract
check (abduction-exactness rate, expected ``1.0`` on the reference oracles; any ``< 1.0`` names
seed-incomplete nondeterminism), and (b) confirms abduction is *non-trivial* -- the rung-3
counterfactual (re-run ``F`` forward with the factual future held fixed, one action overridden)
genuinely differs from the factual rollout (the cf-differs rate), so the recovered ``U`` is
producing a
real counterfactual, not echoing the factual.

This is the build that makes rung-3 counterfactual targets *exact and free* -- a near-certain result
by
construction, not a bet. The *magnitude* of the counterfactual effect and its
hidden-state-dependence
(the do-calculus reading of H5) is the CX1 effect-size law; the *learned* counterfactual lift is the
deferred trained-arm bet (CX2-CX4, SPEC-17 §7). Pure-oracle, CPU-only, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.causal.scm import (
    Intervention,
    abduct_and_replay,
    abduction_exact,
    rung3_counterfactual,
)
from verisim.experiments.cx_common import CXWorld, all_cx_worlds


@dataclass(frozen=True)
class CX0Config:
    """A small, fast SCM-gate instance (the dependency-free core)."""

    n_steps: int = 40
    n_seeds: int = 16
    base_seed: int = 0
    intervene_at: float = 0.5  # intervene at this fraction of the rollout depth

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CX0Config:
        b = CX0Config()
        return CX0Config(
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            base_seed=d.get("base_seed", b.base_seed),
            intervene_at=d.get("intervene_at", b.intervene_at),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> CX0Config:
        return CX0Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CX0Stat:
    """One world's SCM-gate cell: abduction exactness (H60) and the cf-differs rate."""

    world: str
    abduction_exact_rate: float  # fraction of seeds whose recovery is bit-identical (expect 1.0)
    cf_differs_rate: float  # fraction whose rung-3 counterfactual differs from factual
    n: int

    def csv_row(self) -> str:
        return f"{self.world},{self.abduction_exact_rate:.6f},{self.cf_differs_rate:.6f},{self.n}"


CSV_HEADER = "world,abduction_exact_rate,cf_differs_rate,n"


def _world_stat(world: CXWorld[Any, Any], config: CX0Config) -> CX0Stat:
    exacts: list[float] = []
    differs: list[float] = []
    t = max(0, int(config.intervene_at * config.n_steps))
    for s in range(config.n_seeds):
        seed = config.base_seed + s
        exacts.append(
            1.0 if abduction_exact(world.make_actions, world.oracle_step, world.diverge, seed,
                                   config.n_steps) else 0.0
        )
        s0, actions, states = abduct_and_replay(
            world.make_actions, world.oracle_step, seed, config.n_steps
        )
        alt = world.alt_action(states[t], 9000 + seed)
        cf = rung3_counterfactual(world.oracle_step, s0, actions, Intervention(t, alt))
        differs.append(1.0 if world.diverge(states[-1], cf[-1]) > 0 else 0.0)
    return CX0Stat(world.name, fmean(exacts), fmean(differs), config.n_seeds)


def run_cx0(config: CX0Config | None = None) -> list[CX0Stat]:
    """Per world: abduction-exactness (H60) and the rate at which rung-3 produces a real
    counterfactual.
    """
    config = config or CX0Config()
    return [_world_stat(world, config) for world in all_cx_worlds()]


def _print_summary(stats: list[CX0Stat]) -> None:
    print("CX0 / H60 - the oracle is an exact SCM (abduction bit-exact; rung-3 is a real cf):")
    print(f"  {'world':>11} {'abduction-exact':>16} {'cf-differs':>11}")
    for s in stats:
        print(f"  {s.world:>11} {s.abduction_exact_rate:>16.3f} {s.cf_differs_rate:>11.3f}")
    all_exact = all(s.abduction_exact_rate >= 0.999 for s in stats)
    min_rate = min(s.abduction_exact_rate for s in stats)
    cf_lo = min(s.cf_differs_rate for s in stats)
    cf_hi = max(s.cf_differs_rate for s in stats)
    verdict = (
        f"abduction is bit-exact on every world (rate {min_rate:.2f}) - H60 supported: the oracle "
        "is an exact SCM, so rung-3 counterfactuals are exact and free (abduction is O(1) replay, "
        f"not the intractable inference of an oracle-free SCM); the rung-3 trajectory genuinely "
        f"differs from the factual (cf-differs {cf_lo:.2f}-{cf_hi:.2f})"
        if all_exact
        else "abduction is NOT bit-exact on some world - H60 names seed-incomplete nondeterminism"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[CX0Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    worlds = [s.world for s in stats]
    x = range(len(worlds))
    width = 0.38
    ax.bar([i - width / 2 for i in x], [s.abduction_exact_rate for s in stats], width,
           color="#1f77b4", label="abduction-exactness")
    ax.bar([i + width / 2 for i in x], [s.cf_differs_rate for s in stats], width,
           color="#2ca02c", label="rung-3 counterfactual differs from factual")
    ax.axhline(1.0, color="#888", ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(worlds, fontsize=9)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("CX0 / H60: the oracle is an exact SCM — abduction is bit-exact and free")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="CX0 the oracle-as-SCM gate (H60).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/cx0_scm_gate.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = CX0Config.from_json_file(args.config) if args.config else CX0Config()
    stats = run_cx0(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
