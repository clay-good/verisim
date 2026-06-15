"""Plot the CU28 figure (SPEC-22 §5, H121): the targeting result against a real /bin/sh.

Two panels from a :class:`~verisim.acd.realkernel_targeting.CU28Result`, each overlaying both
reality anchors -- the reference oracle (filled bars) and a real ``/bin/sh`` (open/hatched bars):

  - **left -- the un-gameability headline:** adversarial breach by schedule. Only the model-free
    *covering* target (and the full oracle) is un-gameable (0.000); the asset-indexed shortcut, the
    uniform clock, and the omitter all leak (1.000). The two anchors lie exactly on top of each
    other (bit-identical, max Δ = 0) -- the targeting verdict is about real computer-use dynamics.
  - **right -- the cost:** mean oracle calls by schedule. The covering target reaches the full
    oracle's safety at a fraction of its cost, again identical against the real kernel.

The platform is stamped on the figure (the macOS-first principle); anchor-invariance is
platform-independent (SY1/H27 proved ref ≡ sandbox bit-exact on the v0 fs grammar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verisim.acd.realkernel_targeting import CU28Result
from verisim.acd.unified_targeting import ArmResult, Cell


def _row(arm: ArmResult) -> dict[str, Cell]:
    """The schedules to display, in headline order."""
    free = arm.uniform[0]
    knee = min(
        (c for c in arm.uniform if c.rho is not None and 0.0 < c.rho < 1.0),
        key=lambda c: c.adversarial_breach, default=arm.uniform[0],
    )
    out = {
        "free\n(ρ=0)": free,
        "uniform\nknee": knee,
        "model\nself-target": arm.model,
        "asset\nshortcut": arm.shortcut if arm.shortcut is not None else free,
        "covering\ntarget": arm.target,
        "full\noracle": arm.full_oracle,
    }
    return out


def plot_cu28(
    result: CU28Result, verdict: dict[str, Any], path: str | Path
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ref_row = _row(result.ref)
    labels = list(ref_row.keys())
    x = np.arange(len(labels))
    w = 0.38
    have_sys = result.sys is not None
    sys_row = _row(result.sys) if result.sys is not None else {}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel A -- adversarial breach
    ref_adv = [ref_row[k].adversarial_breach for k in labels]
    axA.bar(x - w / 2, ref_adv, w, color="#d62728", label="vs reference oracle")
    if have_sys:
        sys_adv = [sys_row[k].adversarial_breach for k in labels]
        axA.bar(x + w / 2, sys_adv, w, color="white", edgecolor="#1f77b4", hatch="///", lw=1.5,
                label="vs real /bin/sh")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel("adversarial breach rate\n(attacker picks the tamper & its timing)")
    axA.set_ylim(0, 1.08)
    axA.set_title("only the model-free covering target is un-gameable")
    axA.legend(fontsize=9, loc="upper right")
    axA.axhline(0, color="#333", lw=0.8)

    # Panel B -- oracle cost
    ref_calls = [ref_row[k].mean_calls for k in labels]
    axB.bar(x - w / 2, ref_calls, w, color="#2ca02c", label="vs reference oracle")
    if have_sys:
        sys_calls = [sys_row[k].mean_calls for k in labels]
        axB.bar(x + w / 2, sys_calls, w, color="white", edgecolor="#555", hatch="///", lw=1.5,
                label="vs real /bin/sh")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylabel("mean oracle calls / deployment")
    saving = verdict.get("target_call_saving", float("nan"))
    axB.set_title(f"covering target reaches it at {saving:.1f}× lower cost")
    axB.legend(fontsize=9, loc="upper left")

    delta = verdict.get("max_anchor_delta", 0.0)
    plat = verdict.get("platform", "?")
    fig.suptitle(
        "CU28 / H121: the targeting result is verified against a real /bin/sh — "
        f"anchor-invariant (max Δ = {delta:.0e}); platform = {plat}",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.005,
             "the covering target is grammar-indexed (verify any write under the protected "
             "prefix); the asset shortcut watches only the known credential and leaks the "
             "unflagged path — both verdicts bit-identical vs the real kernel",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
