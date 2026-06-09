"""Experiment PB-transfer-broad: the sim-to-emulation boundary across grammars (SPEC-18 §6, H66).

PB-transfer measured the transfer gap ``ΔH = H_ε^ref − H_ε^sys`` (reference oracle vs the real
`/bin/sh` [`SandboxOracle`](../../src/verisim/oracle/sandbox.py)) on the **validated**
structure-building grammar and found it **0.000** — transfer is lossless where SY1/H27 proved the
reference bit-exact. But the faithfulness claim is honest only if its *boundary* is measured too:
SPEC-18 §10 names "a non-zero ΔH outside the validated grammar" as the open arm, and SPEC-3's W1
wall ("the oracle is a model, not reality") demands knowing *where* the model stops matching truth.

PB-transfer-broad runs the identical ΔH measurement across a sweep of **grammars** of widening
scope: the validated ``structural`` (mkdir/touch/write), the ``weighted`` everyday grammar (adds
cp/mv/rm/chmod/append/rmdir), and the ``adversarial`` destructive grammar (rm -r / mv / rmdir).
A proposer fit against the reference oracle is scored against both oracles per grammar; where the
reference FS model diverges from the real shell, the reference-fit proposer is wrong against the
system oracle and ΔH grows. So ΔH(grammar) maps *where the simulator stops being faithful to a real
OS* — the first such boundary in faithful-horizon terms.

The ρ-sweep is reported per grammar too (the H67-broad question): does oracle-in-the-loop correction
shrink the gap where the model diverges? The pre-registered reading: ΔH ≈ 0 on the validated grammar
(confirming PB-transfer), rising on the broader grammars (the honest boundary). A flat-zero ΔH on
*all* grammars would be a stronger positive (the reference is faithful beyond the validated subset);
either is banked, and the SPEC-18 §10 datasheet caveat becomes a measured number, not a disclaimer.

The proposer is PB-transfer's reference-fit drifting stand-in (the trained-`M_θ` arm is deferred);
``skipif``-guarded and disclosed when no real shell is present (a skip is never a result, SPEC-11
§2.5). CPU, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.experiments.pb_transfer import (
    PBTransferConfig,
    run_pb_transfer,
    system_oracle_available,
)


@dataclass(frozen=True)
class PBTransferBroadConfig:
    """A grammar sweep of the PB-transfer ΔH measurement (the boundary map)."""

    grammars: tuple[str, ...] = ("structural", "weighted", "adversarial")
    alpha: float = 0.8
    rhos: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    n_steps: int = 24
    n_seeds: int = 8
    epsilon: float = 0.0
    base_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PBTransferBroadConfig:
        b = PBTransferBroadConfig()
        return PBTransferBroadConfig(
            grammars=tuple(d.get("grammars", b.grammars)),
            alpha=d.get("alpha", b.alpha),
            rhos=tuple(d.get("rhos", b.rhos)),
            n_steps=d.get("n_steps", b.n_steps),
            n_seeds=d.get("n_seeds", b.n_seeds),
            epsilon=d.get("epsilon", b.epsilon),
            base_seed=d.get("base_seed", b.base_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> PBTransferBroadConfig:
        return PBTransferBroadConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class BroadStat:
    """One (grammar, ρ) cell: faithful horizon vs each oracle and the transfer gap ΔH."""

    grammar: str
    rho: float
    h_ref: float
    h_sys: float
    delta_h: float
    dh_lo: float
    dh_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.grammar},{self.rho:.4f},{self.h_ref:.6f},{self.h_sys:.6f},{self.delta_h:.6f},"
            f"{self.dh_lo:.6f},{self.dh_hi:.6f},{self.n}"
        )


CSV_HEADER = "grammar,rho,h_ref,h_sys,delta_h,dh_lo,dh_hi,n"


def run_pb_transfer_broad(config: PBTransferBroadConfig | None = None) -> list[BroadStat]:
    """Per grammar: the PB-transfer ΔH(ρ) measurement; the sim-to-emulation boundary map (H66)."""
    config = config or PBTransferBroadConfig()
    if not system_oracle_available():
        return []  # skip (disclosed by the caller); never counted as a result (SPEC-11 §2.5)
    out: list[BroadStat] = []
    for grammar in config.grammars:
        stats = run_pb_transfer(
            PBTransferConfig(
                alpha=config.alpha, driver=grammar, rhos=config.rhos, n_steps=config.n_steps,
                n_seeds=config.n_seeds, epsilon=config.epsilon, base_seed=config.base_seed,
            )
        )
        for s in stats:
            out.append(
                BroadStat(grammar, s.rho, s.h_ref, s.h_sys, s.delta_h, s.dh_lo, s.dh_hi, s.n)
            )
    return out


def _verdict(stats: list[BroadStat], config: PBTransferBroadConfig) -> str:
    by_grammar_rho0 = {s.grammar: s for s in stats if s.rho == 0.0}
    validated = by_grammar_rho0.get("structural")
    if validated is None:
        return "inconclusive (no structural baseline)"
    broad = [
        by_grammar_rho0[g]
        for g in config.grammars
        if g != "structural" and g in by_grammar_rho0
    ]
    if not broad:
        return f"validated grammar ΔH(ρ=0)={validated.delta_h:+.2f}; no broad grammar to compare"
    worst = max(broad, key=lambda s: s.delta_h)
    # does oracle-in-the-loop correction shrink the gap on the worst grammar? (the H67-broad read)
    worst_curve = sorted((s for s in stats if s.grammar == worst.grammar), key=lambda s: s.rho)
    shrinks = len(worst_curve) > 1 and worst_curve[-1].delta_h < worst_curve[0].delta_h
    if worst.delta_h > 0.5:
        corr = (
            "oracle-in-the-loop correction shrinks it as ρ rises (the loop recovers faithfulness "
            "where the static model diverges)"
            if shrinks
            else "and oracle-in-the-loop correction does not close it (the divergence is in a "
            "regime the loop's consult budget cannot reach at these ρ)"
        )
        return (
            f"the sim-to-emulation gap is **grammar-dependent and now mapped**: ΔH(ρ=0) ≈ "
            f"{validated.delta_h:+.2f} on the validated structural grammar (lossless, confirming "
            f"PB-transfer) but rises to {worst.delta_h:+.2f} on the '{worst.grammar}' grammar — "
            f"the reference FS model diverges from the real /bin/sh on the ops SY1/H27 did not "
            f"validate (cp/mv/rm/chmod/rmdir). {corr}. The first quantified faithfulness boundary "
            f"(SPEC-3 W1), turning the SPEC-18 §10 datasheet caveat into a measured number."
        )
    return (
        f"transfer is essentially lossless across all grammars tested (max "
        f"ΔH(ρ=0)={worst.delta_h:+.2f} on '{worst.grammar}', validated {validated.delta_h:+.2f}): "
        f"the reference FS model is faithful to the real /bin/sh beyond the validated structure "
        f"grammar too — a stronger positive than PB-transfer's, widening the transfer claim."
    )


def _print_summary(stats: list[BroadStat], config: PBTransferBroadConfig) -> None:
    print("PB-transfer-broad / H66 — the sim-to-emulation boundary across grammars (ΔH/grammar):")
    if not stats:
        print("  [system oracle UNAVAILABLE — boundary measurement skipped, not counted (§2.5)]")
        return
    print("  [trained-M_θ arm DEFERRED — proposer is the reference-fit drifting stand-in]")
    print(f"  {'grammar':<13} {'ρ':>5} {'H_ε^ref':>8} {'H_ε^sys':>8} {'ΔH':>7} {'95% CI':>16}")
    for s in stats:
        ci = f"[{s.dh_lo:+.2f},{s.dh_hi:+.2f}]"
        print(f"  {s.grammar:<13} {s.rho:>5.2f} {s.h_ref:>8.2f} {s.h_sys:>8.2f} {s.delta_h:>+7.3f} "
              f"{ci:>16}")
    print("  verdict: " + _verdict(stats, config))


def write_csv(stats: list[BroadStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    return out


def _plot(  # pragma: no cover - plotting
    stats: list[BroadStat], path: Path, config: PBTransferBroadConfig
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grammars = [g for g in config.grammars if any(s.grammar == g for s in stats)]
    cmap = plt.get_cmap("viridis")
    colors = {g: cmap(0.1 + 0.8 * i / max(1, len(grammars) - 1)) for i, g in enumerate(grammars)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel 1: ΔH(ρ=0) per grammar — the boundary bar (validated ~0, broad grows).
    rows0 = [s for g in grammars for s in stats if s.grammar == g and s.rho == 0.0]
    x = range(len(rows0))
    ax1.bar(x, [s.delta_h for s in rows0], 0.6,
            yerr=[[max(0.0, s.delta_h - s.dh_lo) for s in rows0],
                  [max(0.0, s.dh_hi - s.delta_h) for s in rows0]], capsize=4,
            color=[colors.get(s.grammar, "#555") for s in rows0])
    ax1.axhline(0, color="#888", lw=1)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{s.grammar}\n(ΔH={s.delta_h:+.2f})" for s in rows0], fontsize=8)
    ax1.set_ylabel("transfer gap ΔH(ρ=0)  (H_ε^ref − H_ε^sys)")
    ax1.set_title("the sim-to-emulation boundary: lossless on validated, grows on broad")

    # Panel 2: ΔH(ρ) per grammar — does correction shrink the gap?
    for g in grammars:
        curve = sorted((s for s in stats if s.grammar == g), key=lambda s: s.rho)
        ax2.plot([s.rho for s in curve], [s.delta_h for s in curve], "-o",
                 color=colors.get(g, "#555"), label=g)
    ax2.axhline(0, color="#888", lw=1)
    ax2.set_xlabel("oracle budget ρ")
    ax2.set_ylabel("transfer gap ΔH")
    ax2.set_title("does oracle-in-the-loop correction close the gap?")
    ax2.legend(fontsize=8)
    fig.suptitle("PB-transfer-broad / H66: the sim-to-emulation gap, mapped across grammars")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="PB-transfer-broad: ΔH across grammars (H66).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/pb_transfer_broad.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = (
        PBTransferBroadConfig.from_json_file(args.config) if args.config
        else PBTransferBroadConfig()
    )
    stats = run_pb_transfer_broad(cfg)
    _print_summary(stats, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    if stats:
        _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"), cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
