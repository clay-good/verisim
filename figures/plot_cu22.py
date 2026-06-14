"""Plot the CU22 figure (SPEC-22 H115): the generative test of the unified framework.

Two panels from a :class:`~verisim.acd.availability_targeting.CU22Result` -- the framework applied
to a danger it never saw (availability: a self-inflicted outage):

  - **left -- the worst-case cost/safety frontier.** Adversarial breach vs oracle calls. The uniform
    blind clock is flat at 1.0 until the full oracle (the knee is a mirage). The carried-over CU10
    ``connect`` and CU17 ``exposure`` targets are red X's stuck at breach 1.0 (cheap, but blind --
    ``covers`` predicted this a priori). The syntactic disconnect target is the orange diamond:
    near-safe on random workloads but adversarially leaking and overpaying. Only the
    framework-*derived* disconnect-closure is the green star in the safe-and-cheap corner.
  - **right -- prediction vs measurement.** Per candidate, the a-priori ``covers`` verdict
    (✓ covering / ✗ breaks coverage) above paired random/adversarial bars: every ``covers=False``
    target leaks, the one ``covers=True`` target is 0/0. The framework predicted each fate before a
    single deployment ran.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.availability_targeting import CU22Result


def plot_cu22(result: CU22Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.2, 4.9))
    full = result.full_oracle
    by = {c.name: c for c in result.candidates}
    connect, exposure = by["connect"], by["exposure"]
    syntactic, closure = by["syntactic"], by["closure"]

    # left: worst-case (adversarial) cost/safety frontier
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.adversarial_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_l.plot(connect.mean_calls, connect.adversarial_breach, "X", color="#d62728", ms=13,
              label="connect (CU10 carry): leaks", zorder=5)
    ax_l.plot(exposure.mean_calls, exposure.adversarial_breach, "P", color="#9467bd", ms=12,
              label="exposure (CU17 carry): leaks", zorder=5)
    ax_l.plot(syntactic.mean_calls, syntactic.adversarial_breach, "D", color="#ff7f0e", ms=10,
              label="syntactic disconnect: leaks + overpays", zorder=5)
    ax_l.plot(closure.mean_calls, closure.adversarial_breach, "*", color="#2ca02c", ms=22,
              label="disconnect-closure (DERIVED): safe + cheap", zorder=6)
    ax_l.annotate(
        "carried-over catalogue:\ncovers()=False → leaks\n(predicted a priori)",
        xy=(connect.mean_calls, connect.adversarial_breach),
        xytext=(connect.mean_calls + 7, 0.80), fontsize=8.0, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax_l.annotate(
        f"covers()=True → 0 breach\n{closure.mean_calls:.1f} calls "
        f"({full.mean_calls / closure.mean_calls:.0f}x cheaper)",
        xy=(closure.mean_calls, closure.adversarial_breach),
        xytext=(closure.mean_calls + 9, 0.20), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_l.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.adversarial_breach), xytext=(full.mean_calls - 16, 0.16),
        fontsize=8.2, color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("adversarial breach rate  (the worst-case safety axis)")
    ax_l.set_ylim(-0.05, 1.10)
    ax_l.set_title("a danger the framework never saw (availability / self-inflicted outage)",
                   fontsize=9.0)
    ax_l.legend(fontsize=7.6, loc="center right")

    # right: prediction (covers) vs measurement (random/adversarial breach)
    cands = [connect, exposure, syntactic, closure]
    labels = ["connect\n(CU10)", "exposure\n(CU17)", "syntactic\ndisconnect",
              "disconnect-\nclosure (DERIVED)"]
    x = np.arange(len(cands))
    w = 0.38
    rand = [c.random_breach for c in cands]
    adv = [c.adversarial_breach for c in cands]
    ax_r.bar(x - w / 2, rand, w, color="#7f7fbf", label="random timing")
    ax_r.bar(x + w / 2, adv, w, color="#d62728", label="adversarial (worst case)")
    for xi, c in zip(x, cands, strict=True):
        top = max(c.random_breach, c.adversarial_breach)
        pred = "covers ✓" if c.covers else "covers ✗"
        pcolor = "#2ca02c" if c.covers else "#d62728"
        ax_r.annotate(pred, xy=(xi, 1.20), ha="center", va="bottom", fontsize=8.8,
                      fontweight="bold", color=pcolor)
        ax_r.annotate(f"{c.mean_calls:.1f} calls", xy=(xi, min(top, 1.0) + 0.03), ha="center",
                      va="bottom", fontsize=7.6, color="#333")
    ax_r.axhline(0.0, color="#999", lw=0.8)
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels, fontsize=8.2)
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(-0.03, 1.34)
    ax_r.set_title("covers() predicts each fate a priori; the run confirms it", fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center left")

    fig.suptitle(
        "CU22 / H115: the unified framework is GENERATIVE — applied to a danger it never saw "
        "(availability), its coverage\ninvariant predicts the entire existing target catalogue "
        "leaks and derives the unique covering target — and the run confirms it  "
        f"({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.7, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
