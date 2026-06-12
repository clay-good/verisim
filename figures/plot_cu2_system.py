"""Plot the CU2-sys figure (SPEC-22 §5, H94): the agent safety gate against a real /bin/sh.

One panel from a :class:`~verisim.experiments.cu2_system_gate.CU2SysResult`: the missed-danger
rate vs the capacity proxy α, overlaid for both reality anchors: the reference oracle (solid) and a
real ``/bin/sh`` (open markers). The curves lie exactly on top of each other (anchor-invariant,
H94): the gate's safety verdict does not move when the real kernel replaces the reference. The
rate is materially positive at low α (a free preview misses real dangers even against the shell) and
recedes with capacity. The platform is stamped on the figure (the macOS-first principle).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verisim.experiments.cu2_system_gate import CU2SysResult


def plot_cu2_system(
    result: CU2SysResult, verdict: dict[str, Any], path: str | Path
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    alphas = [c.alpha for c in result.cells]
    ref = [c.missed_rate_ref for c in result.cells]
    sysv = [c.missed_rate_sys for c in result.cells]

    ax.plot(alphas, ref, "-o", color="#d62728", lw=2, ms=8, label="vs reference oracle")
    ax.plot(alphas, sysv, "--s", color="#1f77b4", mfc="none", ms=12, mew=2,
            label="vs real /bin/sh")
    ax.set_xlabel("capacity proxy α  (α→1 = larger, less-drifting model)")
    ax.set_ylabel("missed-danger rate\n(truly-unsafe plans the agent executed)")
    max_delta = verdict.get("max_anchor_delta", 0)
    ax.set_title(
        f"the agent's safety gate is verified against a real /bin/sh (H94)\n"
        f"missed-danger is anchor-invariant — identical vs the real kernel (max Δ = {max_delta})"
    )
    ax.set_ylim(bottom=-0.02)
    ax.legend(fontsize=10, loc="upper right")
    plat = result.platform or "?"
    ax.text(0.99, -0.14, f"platform = {plat}; anchor-invariance is platform-independent (SY1/H27)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#666")

    fig.suptitle(
        "CU2-sys / H94: a free preview misses real dangers even vs a real kernel; oracle catches",
        fontsize=10.5, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
