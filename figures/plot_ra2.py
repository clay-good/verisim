"""Plot the RA2 figure (SPEC-22 H134): the real-LLM safety gate is anchor-invariant.

The same RA1 battery run against the reference oracle (filled bars) and a real ``/bin/sh`` (open
overlay), per schedule, for the two load-bearing axes (missed-danger and oracle cost). The
real-kernel overlay lands exactly on the reference bars -- max Δ = 0 -- so the agent-safety verdict
does not move when the agent acts on a real computer instead of a model of one (the RA analogue of
CU28 / CU2-sys). With no real shell the figure shows the reference result alone (SPEC-11 §2.5).
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.real_kernel import RA2Result, anchor_delta

_ORDER = ("undefended", "target", "full_oracle")
_LABELS = {"undefended": "undefended", "target": "covering\ntarget", "full_oracle": "full\noracle"}


def plot_ra2(result: RA2Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = {c.schedule: c for c in result.ref.cells}
    sysr = {c.schedule: c for c in result.sys.cells} if result.sys is not None else {}
    x = range(len(_ORDER))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: missed-danger, reference (filled) vs real kernel (open overlay) -------------------
    ref_missed = [ref[s].missed_danger_rate for s in _ORDER]
    ax_l.bar(x, ref_missed, 0.55, color="#c44", label="reference oracle")
    if sysr:
        sys_missed = [sysr[s].missed_danger_rate for s in _ORDER]
        ax_l.bar(x, sys_missed, 0.55, facecolor="none", edgecolor="#222",
                 linewidth=1.8, label="real /bin/sh (overlay)")
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels([_LABELS[s] for s in _ORDER])
    ax_l.set_ylabel("missed-danger rate  (unsafe)")
    ax_l.set_ylim(0, max(0.25, max(ref_missed) * 1.3))
    ax_l.set_title("Safety verdict: identical on a real kernel")
    ax_l.legend(loc="upper right", fontsize=8)

    # --- right: oracle cost, reference vs real kernel ---------------------------------------------
    ref_calls = [ref[s].mean_oracle_calls for s in _ORDER]
    ax_r.bar(x, ref_calls, 0.55, color="#48c", label="reference oracle")
    if sysr:
        sys_calls = [sysr[s].mean_oracle_calls for s in _ORDER]
        ax_r.bar(x, sys_calls, 0.55, facecolor="none", edgecolor="#222",
                 linewidth=1.8, label="real /bin/sh (overlay)")
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels([_LABELS[s] for s in _ORDER])
    ax_r.set_ylabel("oracle calls per task  (cost)")
    ax_r.set_title("Cost: identical on a real kernel")
    ax_r.legend(loc="upper left", fontsize=8)

    if result.sys is not None:
        tag = f"max Δ (reference vs real kernel) = {anchor_delta(result):.3f}"
    else:
        tag = "real shell unavailable — reference only (SPEC-11 §2.5)"
    fig.suptitle(
        f"RA2 / H134 -- the real-LLM safety gate is anchor-invariant vs a real /bin/sh:  {tag}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
