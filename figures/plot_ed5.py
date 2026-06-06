"""Plot the ED5 figure (SPEC-7 §12, DS8): consistency-vs-bit horizon (H19) + competitive ratio.

Two panels:

  - **left -- H19**: per error mode, the free-running bit-faithful vs consistency-faithful horizon
    (grouped bars with bootstrap-CI whiskers). The ``subtle`` (in-flight) bars show the gap --
    consistency outlasts bit, because a corrupted in-flight payload is bit-visible but
    consistency-invisible until delivery; the ``gross`` (durable-replica) bars coincide (the
    control where both break at once).
  - **right -- H18**: the competitive ratio ``H_ε(ρ)/ceiling`` vs ``ρ``, one line per prediction-
    error (noise) level. The dashed vertical marks the quarter budget ``B/4``. The fan of lines is
    the learning-augmented signal: the ratio degrades gracefully with prediction error (the perfect
    model is flat at 1.0; the useless model hugs the floor until ρ→1), and the floor→cliff shape at
    sub-linear budget is the honest no-knee negative.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed5 import ED5Result

_MODE_COLORS = {"bit": "#d62728", "cons": "#2ca02c"}


def plot_ed5(result: ED5Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_h19, ax_h18) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # --- left: H19 grouped bars (bit vs consistency horizon per mode) -----------------------------
    modes = [r["mode"] for r in result.h19]
    x = range(len(modes))
    width = 0.38
    bit_h = [r["bit_h"] for r in result.h19]
    cons_h = [r["cons_h"] for r in result.h19]
    bit_err = [[r["bit_h"] - r["bit_lo"] for r in result.h19],
               [r["bit_hi"] - r["bit_h"] for r in result.h19]]
    cons_err = [[r["cons_h"] - r["cons_lo"] for r in result.h19],
                [r["cons_hi"] - r["cons_h"] for r in result.h19]]
    ax_h19.bar([i - width / 2 for i in x], bit_h, width, yerr=bit_err, capsize=4,
               color=_MODE_COLORS["bit"], label="bit-faithful  H_ε")
    ax_h19.bar([i + width / 2 for i in x], cons_h, width, yerr=cons_err, capsize=4,
               color=_MODE_COLORS["cons"], label="consistency-faithful  H_ε")
    for i, r in enumerate(result.h19):
        if r["consistency_outlasts"]:
            ax_h19.annotate(f"gap +{r['gap']:.1f}", (i, max(bit_h[i], cons_h[i])),
                            textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=9, color="#2ca02c", fontweight="bold")
    ax_h19.set_xticks(list(x))
    ax_h19.set_xticklabels([f"{m}\nerror" for m in modes])
    ax_h19.set_ylabel("free-running faithful horizon  H_ε  (steps)")
    ax_h19.set_title("H19 — consistency-faithful outlasts bit-faithful\n"
                     "(the in-flight medium: bit-visible, consistency-invisible)", fontsize=10)
    ax_h19.legend(fontsize=8, loc="upper left")
    ax_h19.grid(True, axis="y", alpha=0.3)

    # --- right: H18 competitive ratio vs ρ, one line per prediction-error level -------------------
    noises = sorted({c["noise"] for c in result.h18})
    cmap = plt.get_cmap("viridis")
    for k, noise in enumerate(noises):
        pts = sorted((c for c in result.h18 if c["noise"] == noise), key=lambda c: c["rho"])
        xs = [c["rho"] for c in pts]
        ys = [c["ratio"] for c in pts]
        ax_h18.plot(xs, ys, marker="o", linewidth=1.8, markersize=4,
                    color=cmap(k / max(1, len(noises) - 1)),
                    label=f"error={noise:.1f}")
    v = result.h18_verdict
    ax_h18.axvline(v["quarter_rho"], color="#555", linestyle="--", linewidth=1, alpha=0.7)
    ax_h18.text(v["quarter_rho"], 0.02, "  B/4", fontsize=8, color="#555")
    grace = "confirmed" if v["degrades_gracefully"] else "not monotone"
    ax_h18.set_xlabel("consultation rate  ρ  (≡ oracle-dollars at bit-exact tier)")
    ax_h18.set_ylabel("competitive ratio  H_ε(ρ) / ceiling")
    ax_h18.set_title(f"H18 — the loop's competitive ratio\n"
                     f"graceful degradation w/ prediction error: {grace}", fontsize=10)
    ax_h18.legend(title="prediction error", fontsize=8, loc="upper left")
    ax_h18.grid(True, alpha=0.3)

    fig.suptitle("ED5 — consistency-vs-bit horizon (H19) + the learning-augmented "
                 "competitive ratio (H18), SPEC-7")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
