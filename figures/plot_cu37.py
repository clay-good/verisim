"""Plot the CU37 figure (SPEC-22 H130): the verifier precision tax -- the utility half of the law.

Two panels from a :class:`~verisim.acd.verifier_precision.CU37Result`:

  - **left -- the precision tax.** For file-integrity monitors that all COVER the danger file /cfg,
    the utility cost (mean false blocks, bars; oracle calls, line) rises monotonically with how
    coarsely the monitor watches (the number of watched files), while the adversarial breach (the
    safety, green line on the twin axis) stays pinned at 0. Precision sets utility; it does not
    touch safety.
  - **right -- the orthogonality 2x2.** The four corners of {covers /cfg?} x {watches benign
    files?}. Cell color is the adversarial breach (green = as safe as the oracle, red = as leaky as
    no gate); the annotation also reports the false-block tax. Breach varies only DOWN the coverage
    axis; the false-block tax varies only ACROSS the precision axis -- the two are independent, so a
    coarse-but-covering monitor is exactly as safe as the oracle and pays only a bounded tax.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.verifier_precision import CU37Result, orthogonality_2x2


def plot_cu37(result: CU37Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # --- left: the precision tax (utility rises, safety flat) ------------------------------------
    cov = sorted(result.covering, key=lambda p: p.granularity)
    xs = [p.granularity for p in cov]
    fb = [p.mean_false_blocks for p in cov]
    calls = [p.mean_calls for p in cov]
    breach = [p.adversarial_breach for p in cov]

    ax_l.bar(xs, fb, width=0.6, color="#d08a3e", label="false blocks (utility tax)", zorder=2)
    ax_l.plot(xs, calls, "o-", color="#8a5a2a", lw=2, label="oracle calls (cost)", zorder=3)
    ax_l.set_xlabel("monitor coarseness  (number of watched files)")
    ax_l.set_ylabel("mean false blocks / oracle calls per deployment")
    ax_l.set_xticks(xs)
    ax_l.set_title("Precision sets utility\n(coarsen the watch set -> the tax rises)")
    ax_l.grid(True, axis="y", alpha=0.3)

    ax_s = ax_l.twinx()
    ax_s.plot(xs, breach, "s--", color="#2a7", lw=2.5, label="adversarial breach (safety)",
              zorder=4)
    ax_s.set_ylabel("adversarial breach  (0 = as safe as the oracle)", color="#1a6")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.tick_params(axis="y", labelcolor="#1a6")
    ax_s.annotate("safety flat at 0\n(every watch set covers /cfg)",
                  xy=(xs[len(xs) // 2], 0.0), xytext=(xs[0] + 0.1, 0.42),
                  color="#1a6", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="#1a6", alpha=0.7))

    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_s.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)

    # --- right: the orthogonality 2x2 ------------------------------------------------------------
    grid = {(c.covers_danger, c.coarse): c for c in orthogonality_2x2(result)}
    cmap = LinearSegmentedColormap.from_list("safe_leak", ["#2a7", "#f2e36b", "#c44"])
    # rows = coverage (top: covers, bottom: misses); cols = precision (left: precise, right: coarse)
    rows = [True, False]
    cols = [False, True]
    mat = [[grid[(r, c)].adversarial_breach for c in cols] for r in rows]
    ax_r.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax_r.set_xticks(range(2))
    ax_r.set_xticklabels(["precise\n(watch only /cfg)", "coarse\n(watch every file)"])
    ax_r.set_yticks(range(2))
    ax_r.set_yticklabels(["covers /cfg\n(safe)", "misses /cfg\n(blind)"])
    ax_r.set_title("Coverage sets safety -- orthogonal to precision\n"
                   "(breach varies down rows, tax varies across columns)")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cell = grid[(r, c)]
            txt = (f"breach\n{cell.adversarial_breach:.2f}\n"
                   f"false blocks\n{cell.mean_false_blocks:.2f}")
            ax_r.text(j, i, txt, ha="center", va="center", fontsize=10,
                      color="#222" if cell.adversarial_breach < 0.5 else "#fff", fontweight="bold")
    ax_r.set_xlabel("precision axis  ->  utility")
    ax_r.set_ylabel("coverage axis  ->  safety")

    fig.suptitle(
        "CU37 / H130 -- the verifier precision tax: a file-integrity monitor's safety is set by "
        "coverage, its utility by precision (the two are orthogonal)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":  # pragma: no cover
    from verisim.acd.verifier_precision import run_cu37

    plot_cu37(run_cu37(), "figures/cu37_verifier_precision.png")
