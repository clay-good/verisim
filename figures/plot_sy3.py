"""Plot the SY3 hermeticity attestation (SPEC-11 §2.3, §5): the per-channel pass table that
rides alongside every other SY figure as the safety attestation (H29).

A single panel: one row per guaranteed channel (filesystem / network / privilege / persistence
/ resource) plus the positive control, each marked PASS (the prohibited action was denied, or
the benign action was allowed-then-discarded). "No action can occur" made into a green table.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.sy3 import SY3Result


def plot_sy3(result: SY3Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controls = result.controls
    fig, ax = plt.subplots(figsize=(10.5, 0.7 * max(len(controls), 1) + 1.4))
    ax.axis("off")

    ax.set_title(
        f"SY3: the hermeticity contract — 'no action can occur' (H29)   [{result.platform}]",
        fontsize=11, loc="left",
    )
    y = len(controls)
    for c in controls:
        ok = c.passed
        ax.text(0.01, y, "PASS" if ok else "FAIL", color="white", fontsize=9, fontweight="bold",
                va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#2ca02c" if ok else "#d62728", ec="none"),
                transform=ax.get_yaxis_transform())
        kind = "deny" if c.kind == "negative" else "allow+discard"
        ax.text(0.07, y, f"[{c.channel}]  {c.name}   ({kind})", fontsize=9, va="center",
                transform=ax.get_yaxis_transform())
        y -= 1
    ax.set_ylim(0, len(controls) + 1)
    ax.set_xlim(0, 1)

    out = Path(path)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
