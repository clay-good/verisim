"""Plot the ED30 figure (SPEC-7 §3.1/§4, DS0 incr 23): the embedded host.

Two panels from one :class:`~verisim.experiments.ed30.ED30Result`:

  - **left — composition + per-node isolation.** A fork is isolated to its node's host (rate 1.0); a
    node serves a KV put and a host fork together (1.0); open+write materializes the file in the
    node's embedded v0 filesystem (1.0).
  - **right — the cross-layer crash linkage.** A host op on a crashed node is unavailable (1.0),
    restart restores host ops (1.0), and the host state survives the crash (1.0).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed30 import ED30Result


def plot_ed30(result: ED30Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # left: composition + isolation
    labels = ["fork isolated\nper node", "KV + host\ncoexist", "embedded FS\nopen + write"]
    vals = [
        result.fork_isolated_rate,
        1.0 if result.kv_and_host_coexist else 0.0,
        1.0 if result.embedded_fs_works else 0.0,
    ]
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    bars = ax_l.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    caps = [f"node-local ({result.n_sizes})", "put + fork", "file in host FS"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.2f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_l.set_ylim(0, 1.3)
    ax_l.set_ylabel("rate")
    ax_l.set_title("Panel A — composition + per-node isolation\n"
                   "each node runs a real SPEC-6 host, isolated from the rest",
                   fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: the crash linkage
    labels = ["crashed host\nunavailable", "restart\nrestores", "state survives\ncrash"]
    vals = [
        1.0 if result.crashed_host_unavailable else 0.0,
        1.0 if result.restart_restores else 0.0,
        1.0 if result.host_state_survives_crash else 0.0,
    ]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    bars = ax_r.bar(labels, vals, color=colors, alpha=0.85, width=0.6)
    caps = ["unavailable", "ok again", "proc table kept"]
    for bar, v, cap in zip(bars, vals, caps, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                  f"{v:.0f}\n{cap}", ha="center", va="bottom", fontsize=9)
    ax_r.set_ylim(0, 1.3)
    ax_r.set_ylabel("rate")
    ax_r.set_title("Panel B — the cross-layer crash linkage\n"
                   "a crash pauses the node's host; restart resumes it",
                   fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED30 — the embedded host: each cluster node runs a real SPEC-6 host"
                 f"  •  Tier-B agrees = {result.tier_b_agrees} ({result.tier_b_steps} steps)",
                 y=1.01, fontsize=9.0)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
