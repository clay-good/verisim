"""Plot the RA1 figure (SPEC-22 H133): the oracle safety gate around a real LLM.

Two panels from a :class:`~verisim.realagent.harness.RA1Result` (one real Claude run, replayed):

  - **left -- the gate catches what the undefended agent does not.** Missed-danger rate (an uncaught
    credential corruption) and task-success rate, per gate schedule. The undefended agent obeys the
    prompt-injection and breaches; the covering-target gate and the full oracle both drive
    missed-danger to zero -- and the gate does NOT cost utility (task success is unchanged), because
    the danger surface is disjoint from the legitimate work.
  - **right -- safety is cheap: cost vs missed-danger.** The covering target reaches the same zero
    missed-danger as the full oracle while consulting the oracle far less often (only on the sparse
    credential-region surface). Undefended is free but unsafe; full oracle is safe but verifies
    everything; the covering target is the safe-and-cheap corner.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.harness import RA1Result

_COLORS = {"undefended": "#c44", "target": "#2a7", "full_oracle": "#48c"}
_LABELS = {
    "undefended": "undefended\n(no oracle)",
    "target": "covering target\n(consult the surface)",
    "full_oracle": "full oracle\n(verify everything)",
}


def plot_ra1(result: RA1Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    cells = result.cells
    order = [c.schedule for c in cells]

    # --- left: missed-danger vs task-success, per schedule ----------------------------------------
    x = range(len(cells))
    w = 0.38
    missed = [c.missed_danger_rate for c in cells]
    success = [c.task_success_rate for c in cells]
    ax_l.bar([i - w / 2 for i in x], missed, w, color="#c44", label="missed-danger rate (unsafe)")
    ax_l.bar([i + w / 2 for i in x], success, w, color="#2a7", label="task-success rate (utility)")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels([_LABELS[s].replace("\n", " ") for s in order], fontsize=8)
    ax_l.set_ylabel("rate")
    ax_l.set_ylim(0, 1.18)
    ax_l.set_title("The gate catches what the undefended agent does not")
    ax_l.legend(loc="upper center", fontsize=8)
    for i, (m, s) in enumerate(zip(missed, success, strict=True)):
        ax_l.text(i - w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
        ax_l.text(i + w / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)

    # --- right: cost vs missed-danger (the safe-and-cheap corner) ---------------------------------
    for c in cells:
        ax_r.scatter(c.mean_oracle_calls, c.missed_danger_rate, s=160,
                     color=_COLORS[c.schedule], zorder=3)
        ax_r.annotate(_LABELS[c.schedule], (c.mean_oracle_calls, c.missed_danger_rate),
                      textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax_r.set_xlabel("oracle calls per task  (cost)")
    ax_r.set_ylabel("missed-danger rate  (unsafe)")
    ax_r.set_ylim(-0.08, 1.08)
    ax_r.set_xlim(left=-0.3)
    ax_r.set_title("Safety is cheap: the covering target is the safe-and-cheap corner")
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        f"RA1 / H133 -- the oracle safety gate around a real LLM ({result.agent_name}): "
        f"{result.n_tasks} curated tasks, {result.n_injected} prompt-injected",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
