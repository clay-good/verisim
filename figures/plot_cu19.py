"""Plot the CU19 figure (SPEC-22 H112): the trained distributed arm -- a real M_θ omits the medium.

Two panels from a :class:`~verisim.acd.closed_loop_dist.CU19Result` (reconstructed from the CSV, no
retrain), the distributed trained-arm confirmation of the targeting result:

  - **left -- the drift asymmetry (the CU8 analogue).** Rolled forward as a belief, the real learned
    M_θ's staleness errors split into omissions (true stale, belief fresh -- the missed-danger
    source) vs hallucinations (true fresh, belief stale -- the false-alarm source). The distributed
    face of CU8: the model *hides* staleness by omission, so its staleness recall is near zero and
    it cannot self-target the danger.
  - **right -- the cost/safety frontier on the real model (the CU5-net analogue).** Breach vs calls:
    uniform is the blind curve to the full oracle; model self-targeting and write_target (the
    CU10-16 genesis-action transfer) are failed points; only the medium target is the safe-and-cheap
    green star -- the loop closes on a real learned model exactly as on the CU18 stand-in.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_dist import CU19Result, StalenessDrift
from verisim.acd.dist_targeting import CU18Cell


def load_cu19_csv(path: str | Path) -> CU19Result:
    """Reconstruct a CU19Result from the experiment's CSV (so the figure needs no retrain)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    cells: dict[str, list[CU18Cell]] = {
        "uniform": [], "model": [], "write_target": [], "medium": []
    }
    n_ep = horizon = 0
    drift = StalenessDrift(0, 0, 0, 0)
    i = 1  # skip the schedule header
    while i < len(lines) and not lines[i].startswith("true_stale"):
        parts = lines[i].split(",")
        sched, label, rho = parts[0], parts[1], parts[2]
        cell = CU18Cell(
            schedule=sched, label=label, rho=float(rho) if rho else None,
            random_breach=float(parts[3]), adversarial_breach=float(parts[4]),
            mean_calls=float(parts[5]),
        )
        cells[sched].append(cell)
        n_ep, horizon = int(parts[6]), int(parts[7])
        i += 1
    if i + 1 < len(lines):  # the drift row follows the DRIFT_HEADER
        d = lines[i + 1].split(",")
        drift = StalenessDrift(int(d[0]), int(d[1]), int(d[2]), int(d[3]))
    return CU19Result(
        n_episodes=n_ep, horizon=horizon, drift=drift, uniform=cells["uniform"],
        model=cells["model"][0], write_target=cells["write_target"][0], medium=cells["medium"][0],
    )


def plot_cu19(result: CU19Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    d = result.drift
    full = result.uniform[-1]
    medium = result.medium
    write = result.write_target
    model = result.model

    # left: the drift asymmetry -- omissions vs hallucinations on the medium (the CU8 analogue).
    # Unlike the net world's omission bias (CU8), the free dist belief over-predicts staleness.
    ax_l.bar([0], [d.omissions], width=0.55, color="#d62728",
             label="omissions (true stale, belief fresh — the breach source)")
    ax_l.bar([1], [d.hallucinations], width=0.55, color="#7f7fbf",
             label="hallucinations (true fresh, belief stale — wasted calls)")
    ax_l.set_xticks([0, 1])
    ax_l.set_xticklabels(["missed-danger\nsource", "false-alarm\nsource"], fontsize=8.4)
    ax_l.set_ylabel("belief-vs-truth staleness errors over the rollout")
    top = max(d.omissions, d.hallucinations, 1)
    ax_l.text(0, d.omissions + top * 0.02, str(d.omissions), ha="center", fontsize=10,
              color="#d62728")
    ax_l.text(1, d.hallucinations + top * 0.02, str(d.hallucinations), ha="center", fontsize=10,
              color="#5a5a8a")
    ax_l.set_ylim(0, top * 1.18)
    ax_l.set_title(
        f"the real M_θ is an unreliable staleness oracle\nrecall {d.recall:.2f}, precision "
        f"{d.precision:.2f} — hallucination-biased, the opposite of the net CU8 omitter",
        fontsize=8.4,
    )
    ax_l.legend(fontsize=7.2, loc="upper left")

    # right: cost/safety frontier on the real model (the CU5-net analogue) -- mirrors CU18 left
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.random_breach for c in result.uniform]
    ax_r.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_r.plot(model.mean_calls, model.random_breach, "X", color="#d62728", ms=13,
              label="model self-targeting (fails)")
    ax_r.plot(write.mean_calls, write.random_breach, "P", color="#ff7f0e", ms=13,
              label="write-to-key = genesis action (does not transfer)")
    ax_r.plot(medium.mean_calls, medium.random_breach, "*", color="#2ca02c", ms=20,
              label="medium = stale-read closure", zorder=5)
    saving = full.mean_calls / medium.mean_calls if medium.mean_calls else float("inf")
    ax_r.annotate(
        f"medium: 0 breach\n{medium.mean_calls:.1f} calls ({saving:.0f}x cheaper)",
        xy=(medium.mean_calls, medium.random_breach), xytext=(medium.mean_calls + 8, 0.22),
        fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_r.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.random_breach), xytext=(full.mean_calls - 16, 0.16),
        fontsize=8.2, color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_r.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_r.set_ylabel("breach rate  (the safety axis)")
    ax_r.set_ylim(-0.05, 1.10)
    ax_r.set_title("only the medium target closes the loop on the real M_θ", fontsize=9.0)
    ax_r.legend(fontsize=7.6, loc="center right")

    fig.suptitle(
        "CU19 / H112: the trained distributed arm — a real learned M_θ is an unreliable staleness "
        "oracle (hallucination-biased,\nthe opposite of the net CU8 omitter), so self-targeting "
        "fails and only the model-free medium target is safe AND cheap  "
        f"({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.6, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - local plotting
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="figures/cu19_dist_trained.csv")
    parser.add_argument("--out", type=str, default="figures/cu19_dist_trained.png")
    args = parser.parse_args()
    plot_cu19(load_cu19_csv(args.csv), args.out)


if __name__ == "__main__":
    main()
