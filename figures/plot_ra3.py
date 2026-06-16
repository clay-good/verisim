"""Plot the RA3 figure (SPEC-22 H135): the gate generalizes across tasks and attacks it never saw.

Two panels from a :class:`~verisim.realagent.generalization.RA3Result`:

  - **left -- aggregate, over the whole randomized battery.** Missed-danger and task-success by gate
    schedule across hundreds of randomized tasks. The undefended agent breaches at the injection
    rate; the covering gate drives missed-danger to zero with task success unchanged --
    generalization, not a fit to 6 curated tasks.
  - **right -- per attack class.** For each danger class in the injection taxonomy (overwrite,
    append, delete, weaken-perms, exfil-move), the undefended breach (1.0 -- every class is a real
    attack) vs the covering-gate breach (0.0 -- every class is caught). The gate catches attack
    classes it never saw because each realizing action is on the model-free target surface.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.generalization import RA3Result

_ORDER = ("undefended", "target", "full_oracle")
_LABELS = {"undefended": "undefended", "target": "covering\ngate", "full_oracle": "full\noracle"}


def plot_ra3(result: RA3Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {c.schedule: c for c in result.base.cells}
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.6, 5.0))

    # --- left: aggregate missed-danger vs task-success, per schedule ------------------------------
    x = range(len(_ORDER))
    w = 0.38
    missed = [by[s].missed_danger_rate for s in _ORDER]
    success = [by[s].task_success_rate for s in _ORDER]
    ax_l.bar([i - w / 2 for i in x], missed, w, color="#c44", label="missed-danger (unsafe)")
    ax_l.bar([i + w / 2 for i in x], success, w, color="#2a7", label="task-success (utility)")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels([_LABELS[s] for s in _ORDER])
    ax_l.set_ylabel("rate")
    ax_l.set_ylim(0, 1.18)
    ax_l.set_title(f"Aggregate over {result.n_tasks} randomized tasks")
    ax_l.legend(loc="upper center", fontsize=8)
    for i, (m, s) in enumerate(zip(missed, success, strict=True)):
        ax_l.text(i - w / 2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
        ax_l.text(i + w / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)

    # --- right: per-attack-class, undefended vs covering gate -------------------------------------
    classes = result.classes
    cx = range(len(classes))
    undef = [c.undefended_breach for c in classes]
    gate = [c.target_breach for c in classes]
    ax_r.bar([i - w / 2 for i in cx], undef, w, color="#c44", label="undefended breach")
    ax_r.bar([i + w / 2 for i in cx], gate, w, color="#2a7", label="covering-gate breach")
    ax_r.set_xticks(list(cx))
    ax_r.set_xticklabels([c.danger_class.replace("_", "\n") for c in classes], fontsize=8)
    ax_r.set_ylabel("breach rate on this attack class")
    ax_r.set_ylim(0, 1.18)
    ax_r.set_title("Every attack class: real undefended, caught by the gate")
    ax_r.legend(loc="upper right", fontsize=8)
    for i, c in enumerate(classes):
        ax_r.text(i, 1.06, f"n={c.n_tasks}", ha="center", fontsize=7, color="#555")

    fig.suptitle(
        f"RA3 / H135 -- the gate generalizes: {result.n_tasks} randomized tasks, "
        f"{len(classes)} attack classes it never saw, coverage holds on every one",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
