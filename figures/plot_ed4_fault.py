"""Plot the ED4-fault figure (SPEC-7 §10.2, DS7): H21 — fault-injected vs fault-free under fault.

Two panels from one :class:`~verisim.experiments.ed4_fault.ED4FaultResult`:

  - **left** -- free-run (ρ=0) faithful horizon vs the eval workload's fault-intensity, one curve
    per training regime. The H21 signal is the **gap opening as faults intensify**: the
    fault-injected model degrades more slowly because it has seen partitions/crashes/recoveries. A
    fault-injected model that holds horizon longer under fault confirms the DST/BUGGIFY lesson.
  - **right** -- the fairness control: each model's clean (fault-free) teacher-forced accuracy, so
    the left gap is read against matched clean quality (H21 asks for the comparison *at equal clean
    accuracy*).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed4_fault import REGIMES, ED4FaultResult

_LABEL = {"fault_free": "fault-free training", "fault_injected": "fault-injected training"}
_COLOR = {"fault_free": "#d62728", "fault_injected": "#2ca02c"}


def plot_ed4_fault(result: ED4FaultResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})

    # left: free-run H_eps vs eval fault-intensity, one curve per training regime
    for regime in REGIMES:
        curve = result.curves.get(regime, [])
        fps = [p["fault_prob"] for p in curve]
        hs = [p["h_eps"] for p in curve]
        lo = [p["ci_lo"] for p in curve]
        hi = [p["ci_hi"] for p in curve]
        ax_l.plot(fps, hs, marker="o", color=_COLOR[regime], label=_LABEL[regime])
        ax_l.fill_between(fps, lo, hi, alpha=0.15, color=_COLOR[regime])
    ax_l.set_xlabel("eval workload fault-intensity  (fault_prob)")
    ax_l.set_ylabel("free-run faithful horizon  H_ε  (ρ=0, steps)")
    ax_l.set_title("H21: does fault-injected training hold horizon under fault?")
    ax_l.legend(loc="upper left", fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # right: the clean-accuracy control (matched quality => the left gap is fault-robustness)
    regimes = list(REGIMES)
    accs = [result.clean_accuracy.get(r, 0.0) for r in regimes]
    bars = ax_r.bar(range(len(regimes)), accs, 0.6, color=[_COLOR[r] for r in regimes])
    for rect, a in zip(bars, accs, strict=True):
        ax_r.annotate(f"{a:.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                      ha="center", va="bottom", fontsize=8)
    ax_r.set_xticks(range(len(regimes)))
    ax_r.set_xticklabels([_LABEL[r].replace(" training", "") for r in regimes], fontsize=8)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("clean teacher-forced accuracy")
    ax_r.set_title("control: matched clean quality")
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED4-fault — H21: fault-injected vs fault-free training, under fault (SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
