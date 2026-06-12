"""Plot the CS3 figure (SPEC-21 §6, H90): the scale law survives the system oracle.

Two panels from one :class:`~verisim.experiments.cs3_system_anchor.CS3Result`:

  - **left -- the load-bearing gap vs the capacity proxy, overlaid for both reality anchors.**
    The per-task faithful-vs-free gap as α (the capacity proxy) rises, scored against the reference
    oracle (solid) and a real ``/bin/sh`` (dashed). The two overlays are *indistinguishable* -- the
    frontier and its motion do not move when the real kernel replaces the reference oracle as the
    reality anchor (H90). The structure task (``file-integrity``) sits flat at zero (not load-
    bearing at any capacity); the deep-content task (``content-value``) recedes with capacity but
    stays above the load-bearing threshold at the top of the ladder -- the residue on a real OS.
  - **right -- the anchor delta.** ``|gap_sys - gap_ref|`` per cell: a flat line at zero, the
    bit-exact form of H90 (the reference oracle and the real shell produce the identical keyed sets
    on the validated content grammar -- SY1/H27).

The platform is stamped on the figure (the macOS-first principle, SPEC-11 §2.5): the headline
anchor-invariance is platform-independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verisim.experiments.cs3_system_anchor import CS3Result

_TASK_COLOR = {"file-integrity": "#1f77b4", "content-value": "#d62728"}


def plot_cs3(
    result: CS3Result, verdict: dict[str, Any], path: str | Path, threshold: float = 0.05
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))
    alphas = sorted({c.alpha for c in result.cells})
    tasks = ["file-integrity", "content-value"]
    by = {(c.alpha, c.task): c for c in result.cells}

    # left: the load-bearing gap vs capacity, reference (solid) vs real shell (dashed), overlaid
    for task in tasks:
        color = _TASK_COLOR[task]
        ref = [by[(a, task)].gap_ref for a in alphas]
        sysv = [by[(a, task)].gap_sys for a in alphas]
        ax_l.plot(alphas, ref, "-o", color=color, label=f"{task} -- vs reference oracle")
        ax_l.plot(alphas, sysv, "--s", color=color, mfc="none",
                  label=f"{task} -- vs real /bin/sh")
    ax_l.axhline(threshold, color="#888", lw=1, ls=":", label=f"load-bearing gap = {threshold}")
    ax_l.set_xlabel("capacity proxy α  (α→1 = larger, less-drifting model)")
    ax_l.set_ylabel("load-bearing gap  (faithful − free)")
    ax_l.set_title("the frontier and its motion are anchor-invariant (H90)")
    ax_l.legend(fontsize=7.5, loc="upper right")
    ax_l.set_ylim(bottom=-0.02)

    # right: the anchor delta per cell -- a flat zero line (the bit-exact form of H90)
    labels = [f"{c.task.split('-')[0]}\nα={c.alpha:g}" for c in result.cells]
    deltas = [c.anchor_delta for c in result.cells]
    x = range(len(result.cells))
    ax_r.bar(x, deltas, color="#2ca02c", alpha=0.85)
    ax_r.axhline(0, color="#333", lw=1)
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels, fontsize=6.5)
    ax_r.set_ylabel("|gap_sys − gap_ref|  (the anchor delta)")
    max_delta = verdict.get("max_anchor_delta", 0.0)
    ax_r.set_title(f"the real kernel agrees bit-for-bit (max Δ = {max_delta:.1e})")
    ax_r.set_ylim(-0.005, max(0.05, max_delta * 1.5 + 0.005))

    plat = result.platform or "?"
    fig.suptitle(
        "CS3 / H90: the faithfulness-for-control scale law survives the system oracle "
        f"(real /bin/sh, platform={plat})"
    )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
