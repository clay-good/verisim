"""Plot the CU20 figure (SPEC-22 H113): the trained host arm -- a real M_θ omits corruptions.

Two panels from a :class:`~verisim.acd.closed_loop_host.CU20Result` (reconstructed from the CSV, no
retrain), the host trained-arm confirmation of the targeting result:

  - **left -- the drift asymmetry (the CU8 analogue).** The real learned host M_θ's teacher-forced
    write errors split into omissions (true wrote, model missed -- the missed-corruption source) vs
    hallucinations (model wrote, true did not -- the false-alarm source). The host face of CU8: like
    the network world (and unlike the distributed world's hallucination bias, CU19), the host model
    *hides* corruptions by omission -- its protected recall is low, so it cannot self-target the
    danger.
  - **right -- the cost/safety frontier on the real model (the CU5-net analogue).** Breach vs calls:
    uniform is the blind curve to the full oracle; model self-targeting is a failed point; only the
    model-free structure target (write-to-jewel via the observable fd table) is the safe-and-cheap
    green star -- the loop closes on a real learned model exactly as on the CU16 stand-in, but the
    real recall 0.27 (not the worst-case 0) makes the free breach 0.74, not 1.0.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_host import CU20Cell, CU20Result, WriteDrift


def load_cu20_csv(path: str | Path) -> CU20Result:
    """Reconstruct a CU20Result from the experiment's CSV (so the figure needs no retrain)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    cells: dict[str, list[CU20Cell]] = {"uniform": [], "model": [], "structure": []}
    n_ep = horizon = 0
    drift = WriteDrift(0, 0, 0, 0, 0, 0)
    i = 1  # skip the schedule header
    while i < len(lines) and not lines[i].startswith("prot_foreseen"):
        parts = lines[i].split(",")
        sched, label, rho = parts[0], parts[1], parts[2]
        cell = CU20Cell(
            schedule=sched, label=label, rho=float(rho) if rho else None,
            random_breach=float(parts[3]), adversarial_breach=float(parts[4]),
            mean_calls=float(parts[5]),
        )
        cells[sched].append(cell)
        n_ep, horizon = int(parts[6]), int(parts[7])
        i += 1
    if i + 1 < len(lines):  # the drift row follows the DRIFT_HEADER
        d = lines[i + 1].split(",")
        drift = WriteDrift(int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4]), int(d[5]))
    return CU20Result(
        n_episodes=n_ep, horizon=horizon, drift=drift, uniform=cells["uniform"],
        model=cells["model"][0], structure=cells["structure"][0],
    )


def plot_cu20(result: CU20Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    d = result.drift
    full = result.uniform[-1]
    free = result.uniform[0]
    structure = result.structure
    model = result.model

    # left: the drift asymmetry -- omissions vs hallucinations on writes (the CU8 analogue).
    # Like the net world (CU8), the host model HIDES corruptions by omission.
    ax_l.bar([0], [d.omissions], width=0.55, color="#d62728",
             label="omissions (true wrote, model missed — the breach source)")
    ax_l.bar([1], [d.hallucinations], width=0.55, color="#7f7fbf",
             label="hallucinations (model wrote, true did not — wasted calls)")
    ax_l.set_xticks([0, 1])
    ax_l.set_xticklabels(["missed-corruption\nsource", "false-alarm\nsource"], fontsize=8.4)
    ax_l.set_ylabel("teacher-forced write errors over the battery")
    top = max(d.omissions, d.hallucinations, 1)
    ax_l.text(0, d.omissions + top * 0.02, str(d.omissions), ha="center", fontsize=10,
              color="#d62728")
    ax_l.text(1, d.hallucinations + top * 0.02, str(d.hallucinations), ha="center", fontsize=10,
              color="#5a5a8a")
    ax_l.set_ylim(0, top * 1.18)
    ratio = d.omissions / d.hallucinations if d.hallucinations else float("inf")
    ax_l.set_title(
        f"the real host M_θ hides corruptions by omission\nprotected recall "
        f"{d.protected_recall:.2f}, {ratio:.0f}:1 omission — like net CU8, unlike dist CU19",
        fontsize=8.4,
    )
    ax_l.legend(fontsize=7.2, loc="upper right")

    # right: cost/safety frontier on the real model (the CU5-net analogue) -- mirrors CU16
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.random_breach for c in result.uniform]
    ax_r.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_r.plot(model.mean_calls, model.random_breach, "X", color="#d62728", ms=13,
              label="model self-targeting (fails)")
    ax_r.plot(structure.mean_calls, structure.random_breach, "*", color="#2ca02c", ms=20,
              label="structure = write-to-jewel (fd table)", zorder=5)
    saving = full.mean_calls / structure.mean_calls if structure.mean_calls else float("inf")
    ax_r.annotate(
        f"structure: 0 breach\n{structure.mean_calls:.1f} calls ({saving:.0f}x cheaper)",
        xy=(structure.mean_calls, structure.random_breach),
        xytext=(structure.mean_calls + 8, 0.22), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_r.annotate(
        f"free agent\nbreach {free.random_breach:.2f}",
        xy=(free.mean_calls, free.random_breach), xytext=(free.mean_calls + 6, 0.84),
        fontsize=8.2, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
    )
    ax_r.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.random_breach), xytext=(full.mean_calls - 16, 0.16),
        fontsize=8.2, color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_r.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_r.set_ylabel("breach rate  (the safety axis)")
    ax_r.set_ylim(-0.05, 1.10)
    ax_r.set_title("only the structure target closes the loop on the real M_θ", fontsize=9.0)
    ax_r.legend(fontsize=7.6, loc="center right")

    fig.suptitle(
        "CU20 / H113: the trained host arm — a real learned M_θ hides credential corruptions by "
        "omission (recall 0.27,\nlike the net CU8 omitter, unlike the dist CU19 hallucinator), so "
        "self-targeting fails and only the model-free structure target is safe AND cheap  "
        f"({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.4, fontweight="bold", y=0.995,
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
    parser.add_argument("--csv", type=str, default="figures/cu20_host_trained.csv")
    parser.add_argument("--out", type=str, default="figures/cu20_host_trained.png")
    args = parser.parse_args()
    plot_cu20(load_cu20_csv(args.csv), args.out)


if __name__ == "__main__":
    main()
