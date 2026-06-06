"""Plot the ED12 figure (SPEC-7 §5.4, DS3 incr 4): partial observation, the probe-faithful horizon.

Two panels:

  - **left, Panel A**: per error mode, the free-running bit / observable / consistency horizon
    (grouped bars with bootstrap-CI whiskers). The ``subtle`` (in-flight) bars show the gap -- the
    *observable* horizon outlasts the *bit* horizon because no probe, at any vantage, can read the
    in-flight replication medium; the corrupted message is invisible until ``advance`` delivers it.
    The ``gross`` (durable-replica) bars coincide (the control where the probe sees the corruption
    at once). The consistency bar is the longest -- its abstraction also forgives node placement.
  - **right, Panel B**: the crash/partition indistinguishability. From a single external vantage the
    crashed-node and partitioned-away states are byte-identical (indistinguishable rate 1.0 -- one
    probe cannot localize the fault); a paired vantage that reaches the node's side separates them
    (rate 0.0). The bar pair is the operational reason failure detection needs more than one probe.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed12 import ED12Result

_COLORS = {"bit": "#d62728", "obs": "#1f77b4", "cons": "#2ca02c"}


def plot_ed12(result: ED12Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: Panel A grouped bars (bit vs observable vs consistency horizon per mode)
    modes = [r["mode"] for r in result.horizons]
    x = range(len(modes))
    width = 0.26

    def _err(key: str) -> list[list[float]]:
        return [[r[f"{key}_h"] - r[f"{key}_lo"] for r in result.horizons],
                [r[f"{key}_hi"] - r[f"{key}_h"] for r in result.horizons]]

    ax_a.bar([i - width for i in x], [r["bit_h"] for r in result.horizons], width,
             yerr=_err("bit"), capsize=4, color=_COLORS["bit"], label="bit-faithful  H_ε")
    ax_a.bar(list(x), [r["obs_h"] for r in result.horizons], width,
             yerr=_err("obs"), capsize=4, color=_COLORS["obs"], label="observable (probe)  H_ε")
    ax_a.bar([i + width for i in x], [r["cons_h"] for r in result.horizons], width,
             yerr=_err("cons"), capsize=4, color=_COLORS["cons"], label="consistency  H_ε")
    for i, r in enumerate(result.horizons):
        if r["observable_outlasts"]:
            ax_a.annotate(f"probe gap +{r['gap']:.1f}", (i, r["cons_h"]),
                          textcoords="offset points", xytext=(0, 8), ha="center",
                          fontsize=9, color=_COLORS["obs"], fontweight="bold")
    ax_a.set_xticks(list(x))
    ax_a.set_xticklabels([f"{m}\nerror" for m in modes])
    ax_a.set_ylabel("free-running faithful horizon  H_ε  (steps)")
    ax_a.set_title("Panel A — the probe-faithful horizon outlasts the bit horizon\n"
                   "(no vantage can observe the in-flight medium)", fontsize=10)
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.grid(True, axis="y", alpha=0.3)

    # right: Panel B crash/partition indistinguishability
    labels = ["single\nexternal vantage", "paired\nvantage"]
    rates = [result.single_vantage_indistinguishable, result.paired_vantage_indistinguishable]
    colors = ["#9467bd", "#ff7f0e"]
    bars = ax_b.bar(labels, rates, color=colors, width=0.6)
    for bar, rate in zip(bars, rates, strict=True):
        ax_b.annotate(f"{rate:.2f}", (bar.get_x() + bar.get_width() / 2, rate),
                      textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
    ax_b.set_ylim(0, 1.15)
    ax_b.set_ylabel("indistinguishable rate  (crash ≡ partition)")
    ax_b.set_title(f"Panel B — crash and partition are indistinguishable\n"
                   f"from one probe, separable from two (n={result.battery_n})", fontsize=10)
    ax_b.axhline(1.0, color="#9467bd", linestyle=":", linewidth=1, alpha=0.6)
    ax_b.text(0.0, 0.5, "one probe\ncannot localize\nthe fault", fontsize=8,
              color="white", ha="center", va="center", fontweight="bold")
    ax_b.text(1.0, 0.08, "a quorum\nseparates them", fontsize=8,
              color="#ff7f0e", ha="center", va="bottom")
    ax_b.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED12 — partial observation: the probe-faithful horizon + "
                 "the crash/partition indistinguishability (SPEC-7 §5.4)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
