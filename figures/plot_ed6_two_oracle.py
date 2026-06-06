"""Plot the ED6 two-oracle figure (SPEC-7 §10.1, DS8): H12 — consistency vs bit-exact oracle.

Two panels from one :class:`~verisim.experiments.ed6_two_oracle.ED6TwoOracleResult`:

  - **left** -- the three H12 rates per error mode: **non-redundant** (~0 by construction — the
    cheap consistency oracle catches nothing the bit-exact oracle misses, so it is *redundant for
    verification*), **consistency-sufficient** (of the bit-wrong steps, how often the split-brain
    verdict is still right — the *decision payoff*), and **full-wrong** (the model's overall error
    rate). The mode split is the headline: ``subtle`` (in-flight) errors are consistency-invisible
    (sufficiency ≈ 1), ``gross`` (durable-replica) errors are consistency-visible (≈ 0, control).
  - **right** -- the **consult-fact ratio**: the consistency answer (per-object converged/split
    view) as a fraction of the full state (replicas + in-flight + partition/crash/clock). The
    decision consult is far cheaper, because the in-flight medium and partition structure inflate
    the full state but never enter the consistency view.

The H12 reading: the consistency oracle is *redundant* but a **cheaper, decision-sufficient**
consult for the question an SRE/defender actually asks — the tiered-oracle premise made concrete.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed6_two_oracle import ED6TwoOracleResult

_COLOR = {"gross": "#d62728", "subtle": "#1f77b4"}


def plot_ed6_two_oracle(result: ED6TwoOracleResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})

    rates = ("non_redundant_rate", "consistency_sufficient_rate", "full_wrong_rate")
    labels = ("non-redundant\n(H12: ≈0)", "consistency-sufficient\n(decision payoff)",
              "full-state\nwrong")
    width = 0.38
    for i, c in enumerate(result.per_mode):
        xs = [j + (i - 0.5) * width for j in range(len(rates))]
        means = [c[m] for m in rates]
        yerr = [[c[m] - c[f"{m}_lo"] for m in rates], [c[f"{m}_hi"] - c[m] for m in rates]]
        ax_l.bar(xs, means, width, yerr=yerr, capsize=4, color=_COLOR.get(c["mode"], "#999"),
                 label=f"{c['mode']} error")
    ax_l.set_xticks(range(len(rates)))
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_ylim(0, 1.08)
    ax_l.set_title("consistency second-oracle vs full bit-exact state")
    ax_l.legend(loc="upper center", fontsize=8)
    ax_l.grid(True, axis="y", alpha=0.3)

    ratio = result.per_mode[0]["consult_fact_ratio"] if result.per_mode else 0.0
    ax_r.bar([0, 1], [1.0, ratio], color=["#9bbcd6", "#2ca02c"])
    ax_r.annotate(f"{ratio:.2f}\n(~{1 / ratio:.1f}× cheaper)" if ratio else "0",
                  (1, ratio), ha="center", va="bottom", fontsize=9)
    ax_r.set_xticks([0, 1])
    ax_r.set_xticklabels(["full state\n(replicas+inflight+medium)", "consistency\nview"],
                         fontsize=8)
    ax_r.set_ylim(0, 1.1)
    ax_r.set_ylabel("consult facts (fraction of full)")
    ax_r.set_title("the decision consult is far cheaper")
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED6 / H12 — the consistency oracle is redundant but decision-sufficient and "
                 "cheaper (SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
