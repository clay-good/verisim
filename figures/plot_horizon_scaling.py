"""Plot the HS1 faithful-horizon scaling law (SPEC-10) from its committed stats.

Two panels, both read only from the reduced :class:`ScaleStat` rows (figures-from-records, SPEC-2
§7.3). Metrics are suffixed by eval regime: ``_id`` (in-distribution, the trained driver) and
``_ood`` (the harder adversarial driver).

  - **left -- free-running horizon vs capacity.** `H_free` in-distribution and out-of-distribution
    vs model params (log x), with CI bands, plus the i.i.d. independence reference `H_indep =
    p/(1-p)` (dashed). Whether `H_free` *climbs* with capacity is H26; whether the climb transfers
    to the harder ``ood`` regime is the honesty check.
  - **right -- one-step acceptance `p` vs capacity** for both regimes. The per-step accuracy that
    capacity lifts; read against the left panel it answers whether accuracy converts into horizon.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.horizon_scaling import ScaleStat


def plot_horizon_scaling(  # pragma: no cover
    stats: list[ScaleStat],
    path: str | Path,
    *,
    suptitle: str = "HS1 — the faithful-horizon scaling law (SPEC-10, H26)",
    left_title: str = "Does capacity lift the free-running horizon?",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(metric: str) -> tuple[list[int], list[float], list[float], list[float]]:
        cells = sorted((s for s in stats if s.metric == metric), key=lambda s: s.params)
        return (
            [c.params for c in cells],
            [c.mean for c in cells],
            [c.ci_lo for c in cells],
            [c.ci_hi for c in cells],
        )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

    for metric, label, color, ls in (
        ("h_free_id", "H_free  in-distribution", "#1f77b4", "-"),
        ("h_free_ood", "H_free  adversarial (ood)", "#d62728", "-"),
        ("h_indep_id", "independence p/(1−p)  [id]", "#999999", "--"),
    ):
        xs, ys, lo, hi = series(metric)
        if not xs:
            continue
        ax_l.plot(xs, ys, marker="o", ls=ls, color=color, label=label)
        if ls == "-":
            ax_l.fill_between(xs, lo, hi, alpha=0.15, color=color)
    ax_l.set_xscale("log")
    ax_l.set_xlabel("model params (≈ n_layer · n_embd²)")
    ax_l.set_ylabel("free-running faithful horizon (steps)")
    ax_l.set_title(left_title)
    ax_l.legend(loc="upper left", fontsize=8)

    for metric, label, color in (
        ("one_step_acc_id", "p  in-distribution", "#1f77b4"),
        ("one_step_acc_ood", "p  adversarial (ood)", "#d62728"),
    ):
        xs, ys, lo, hi = series(metric)
        if not xs:
            continue
        ax_r.plot(xs, ys, marker="s", color=color, label=label)
        ax_r.fill_between(xs, lo, hi, alpha=0.15, color=color)
    ax_r.set_xscale("log")
    ax_r.set_ylim(0.0, 1.02)
    ax_r.set_xlabel("model params (≈ n_layer · n_embd²)")
    ax_r.set_ylabel("one-step acceptance p")
    ax_r.set_title("Per-step accuracy vs capacity")
    ax_r.legend(loc="lower right", fontsize=8)

    fig.suptitle(suptitle)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
