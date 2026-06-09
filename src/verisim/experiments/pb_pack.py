"""Experiment PB-pack: packaging + contamination control (SPEC-18 §6, H68 + the PB milestones).

The engineering hardening that turns the two validated claims (PB-bench, PB-transfer) into a
versioned
release. Three deliverables:

  - **the metadata** (PB milestones): emit the **Croissant** descriptor, the **datasheet**, and the
    **model-card** for the frozen battery (the standardized, machine-readable
    provenance/composition/
    limits BetterBench found almost no benchmark ships, SPEC-18 §7).
  - **the conformance suite**: assert each world's RL env honors the Gymnasium reset/step and
    the `verifiers` `load_environment` entrypoint (the Inspect-task contract is asserted in the test
    suite where `inspect_ai` is optional).
  - **the contamination control** (H68): a proposer overfit to the *public* battery (perfect on its
    seeds, blind off them) scores conspicuously worse on the **held-out** shard than an honestly-fit
    proposer of equal capacity — so the public-minus-held-out gap is a usable overfit detector. The
    committed proposers are controlled stand-ins (a per-step-accuracy drafter); a real memorizer is
    deferred with the trained arms.

CPU-only, deterministic, seeded. The metadata files are written under ``bench/`` (a committed
artifact
directory), regenerable from the manifest hash.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.bench.conformance import run_conformance
from verisim.bench.leaderboard import score_proposer
from verisim.bench.manifest import (
    BatteryManifest,
    Proposer,
    croissant_metadata,
    datasheet,
    model_card,
)
from verisim.experiments.sr2 import SR2Config, granularity
from verisim.experiments.sr_common import net_world
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class PBPackConfig:
    """A small, fast packaging + contamination-control instance (the dependency-free core)."""

    public_seeds: int = 16  # the public battery shard
    heldout_offset: int = 10_000  # the held-out shard (disjoint seeds)
    heldout_seeds: int = 16
    honest_alpha: float = 0.85  # an honestly-fit proposer: same fidelity on both shards
    memorizer_public_alpha: float = 1.0  # the memorizer: perfect on public seeds...
    memorizer_heldout_alpha: float = 0.5  # ...degraded off them (memorized labels, not dynamics)
    n_steps: int = 80
    epsilon_g: float = 1.0
    bench_dir: str = "bench"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PBPackConfig:
        b = PBPackConfig()
        return PBPackConfig(
            public_seeds=d.get("public_seeds", b.public_seeds),
            heldout_offset=d.get("heldout_offset", b.heldout_offset),
            heldout_seeds=d.get("heldout_seeds", b.heldout_seeds),
            honest_alpha=d.get("honest_alpha", b.honest_alpha),
            memorizer_public_alpha=d.get("memorizer_public_alpha", b.memorizer_public_alpha),
            memorizer_heldout_alpha=d.get("memorizer_heldout_alpha", b.memorizer_heldout_alpha),
            n_steps=d.get("n_steps", b.n_steps),
            epsilon_g=d.get("epsilon_g", b.epsilon_g),
            bench_dir=d.get("bench_dir", b.bench_dir),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> PBPackConfig:
        return PBPackConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class PBPackResult:
    """The contamination control + the conformance verdict (H68 + PB milestones)."""

    honest_gap: float  # honest proposer's public-minus-heldout faithful-fraction gap
    honest_lo: float
    honest_hi: float
    memorizer_gap: float  # memorizer's public-minus-heldout gap (the overfit signal)
    mem_lo: float
    mem_hi: float
    conformance_pass: int
    conformance_total: int


def _shard_scores(
    alpha_pub: float, alpha_held: float, config: PBPackConfig
) -> tuple[list[float], list[float]]:
    """Per-seed faithful fractions on the public and held-out shards (network world)."""
    world = net_world()
    eps = config.epsilon_g * granularity(world, SR2Config(n_steps=config.n_steps))
    public = [
        score_proposer(world, Proposer("p", "p", alpha_pub), s, config.n_steps, eps)
        for s in range(config.public_seeds)
    ]
    heldout = [
        score_proposer(world, Proposer("p", "p", alpha_held), config.heldout_offset + s,
                       config.n_steps, eps)
        for s in range(config.heldout_seeds)
    ]
    return public, heldout


def run_pb_pack(config: PBPackConfig | None = None) -> PBPackResult:
    """Contamination control (H68) + conformance suite + metadata emission (PB milestones)."""
    config = config or PBPackConfig()
    # Honest proposer: equal fidelity on both shards -> small gap.
    h_pub, h_held = _shard_scores(config.honest_alpha, config.honest_alpha, config)
    # Memorizer: perfect on public, degraded on held-out -> large gap (the overfit tell).
    m_pub, m_held = _shard_scores(
        config.memorizer_public_alpha, config.memorizer_heldout_alpha, config
    )
    honest_gaps = [p - h for p, h in zip(h_pub, h_held, strict=True)]
    mem_gaps = [p - h for p, h in zip(m_pub, m_held, strict=True)]
    h_lo, h_hi = bootstrap_ci(honest_gaps, seed=0)
    m_lo, m_hi = bootstrap_ci(mem_gaps, seed=0)
    conformance = run_conformance()
    return PBPackResult(
        honest_gap=sum(honest_gaps) / len(honest_gaps), honest_lo=h_lo, honest_hi=h_hi,
        memorizer_gap=sum(mem_gaps) / len(mem_gaps), mem_lo=m_lo, mem_hi=m_hi,
        conformance_pass=sum(1 for r in conformance if r.passed),
        conformance_total=len(conformance),
    )


def emit_metadata(config: PBPackConfig | None = None) -> dict[str, str]:
    """Write the Croissant / datasheet / model-card for the frozen battery; return the paths."""
    config = config or PBPackConfig()
    manifest = BatteryManifest(n_steps=config.n_steps, epsilon_g=config.epsilon_g)
    out_dir = Path(config.bench_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    croissant = out_dir / "croissant.json"
    croissant.write_text(json.dumps(croissant_metadata(manifest), indent=2, sort_keys=True) + "\n")
    paths["croissant"] = str(croissant)
    ds = out_dir / "datasheet.md"
    ds.write_text(datasheet(manifest))
    paths["datasheet"] = str(ds)
    mc = out_dir / "model-card.md"
    mc.write_text(model_card(manifest))
    paths["model_card"] = str(mc)
    return paths


def _print_summary(result: PBPackResult, paths: dict[str, str]) -> None:
    from verisim.bench.conformance import run_conformance

    print("PB-pack / H68 + milestones - contamination control, conformance, and metadata:")
    print("  [a real public-manifest memorizer is DEFERRED -- committed proposers are stand-ins]")
    print("  contamination control (public − held-out faithful gap):")
    res = result
    print(f"    honest:    {res.honest_gap:+.3f}  [{res.honest_lo:+.2f}, {res.honest_hi:+.2f}]")
    print(f"    memorizer: {res.memorizer_gap:+.3f}  [{res.mem_lo:+.2f}, {res.mem_hi:+.2f}]")
    print(f"  conformance: {result.conformance_pass}/{result.conformance_total} contracts pass")
    for c in run_conformance():
        flag = "PASS" if c.passed else "FAIL"
        print(f"    {flag:>4}  {c.surface:<16} {c.contract:<22} {c.detail}")
    print("  metadata written:")
    for k, v in paths.items():
        print(f"    {k}: {v}")
    margin = max(0.05, result.honest_hi - result.honest_lo)
    detector_works = result.memorizer_gap > result.honest_gap + margin
    conformance_green = result.conformance_pass == result.conformance_total
    verdict = (
        f"the public-minus-held-out gap separates the memorizer ({result.memorizer_gap:+.2f}) from "
        f"honest proposer ({result.honest_gap:+.2f}) - H68 supported (the frozen eval is "
        f"contamination-resistant: overfitting the public manifest is detectable); conformance "
        f"{'green' if conformance_green else 'RED'}, metadata emitted - the benchmark is packaged"
        if detector_works and conformance_green
        else "the overfit detector or conformance did not clear - revisit shards / contracts"
    )
    print(f"  verdict: {verdict}")


def _plot(result: PBPackResult, path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = ["honest\nproposer", "public-manifest\nmemorizer"]
    gaps = [result.honest_gap, result.memorizer_gap]
    errs = [[result.honest_gap - result.honest_lo, result.memorizer_gap - result.mem_lo],
            [result.honest_hi - result.honest_gap, result.mem_hi - result.memorizer_gap]]
    ax.bar(labels, gaps, yerr=errs, capsize=5, color=["#1f77b4", "#d62728"])
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylabel("public − held-out faithful-fraction gap")
    ax.set_title("PB-pack / H68: the public-minus-held-out gap detects overfitting")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="PB-pack contamination control + packaging (H68).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/pb_pack_contamination.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = PBPackConfig.from_json_file(args.config) if args.config else PBPackConfig()
    result = run_pb_pack(cfg)
    paths = emit_metadata(cfg)
    _print_summary(result, paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "proposer,public_minus_heldout_gap,gap_lo,gap_hi,conformance_pass,conformance_total\n"
        f"honest,{result.honest_gap:.6f},{result.honest_lo:.6f},{result.honest_hi:.6f},"
        f"{result.conformance_pass},{result.conformance_total}\n"
        f"memorizer,{result.memorizer_gap:.6f},{result.mem_lo:.6f},{result.mem_hi:.6f},"
        f"{result.conformance_pass},{result.conformance_total}\n"
    )
    print(f"wrote {out}")
    _plot(result, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
