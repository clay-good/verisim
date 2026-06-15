"""Plot the CU23 figure (SPEC-22 H116): the second generative test, in the host world.

Two panels from a :class:`~verisim.acd.process_availability_targeting.CU23Result` -- the unified
framework applied to a second danger it never saw (host availability: terminating a critical
defensive daemon):

  - **left -- the worst-case cost/safety frontier.** Adversarial breach vs oracle calls. The uniform
    blind clock is flat at 1.0 until the full oracle (the knee is a mirage). The carried-over CU16
    write-to-fd target -- the host world's own integrity defense -- is the red X stuck at breach 1.0
    (cheap, but blind: a termination is not a write; ``covers`` predicted this a priori). The
    syntactic terminate target is the orange diamond: safe but overpaying (it consults every benign
    process exit). Only the framework-*derived* process-liveness closure is the green star in the
    safe-and-cheap corner.
  - **right -- prediction vs measurement, with the cross-world contrast.** Per candidate, the
    a-priori ``covers`` verdict above paired random/adversarial bars: the CU16 carry-over leaks,
    both liveness-aware targets are 0/0. The annotation carries the punchline: the *syntactic*
    COVERS here (process death has no cascade) but LEAKED in CU22 (reachability is multi-hop) --
    the same candidate class, opposite fate, both called a priori by ``covers``.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.process_availability_targeting import CU23Result


def plot_cu23(result: CU23Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.2, 4.9))
    full = result.full_oracle
    by = {c.name: c for c in result.candidates}
    write, syntactic, liveness = by["write"], by["syntactic"], by["liveness"]

    # left: worst-case (adversarial) cost/safety frontier
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.adversarial_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_l.plot(write.mean_calls, write.adversarial_breach, "X", color="#d62728", ms=14,
              label="write-to-fd (CU16 carry): leaks", zorder=5)
    ax_l.plot(syntactic.mean_calls, syntactic.adversarial_breach, "D", color="#ff7f0e", ms=10,
              label="terminate-any (syntactic): safe, overpays", zorder=5)
    ax_l.plot(liveness.mean_calls, liveness.adversarial_breach, "*", color="#2ca02c", ms=22,
              label="liveness closure (DERIVED): safe + cheap", zorder=6)
    ax_l.annotate(
        "host's own integrity target:\ncovers()=False → leaks\n(watches the wrong resource)",
        xy=(write.mean_calls, write.adversarial_breach),
        xytext=(write.mean_calls + 7, 0.80), fontsize=8.0, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )
    ax_l.annotate(
        f"covers()=True → 0 breach\n{liveness.mean_calls:.1f} calls "
        f"({full.mean_calls / liveness.mean_calls:.0f}x cheaper)",
        xy=(liveness.mean_calls, liveness.adversarial_breach),
        xytext=(liveness.mean_calls + 9, 0.20), fontsize=8.2, color="#2ca02c",
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
    ax_l.set_title("a 2nd danger the framework never saw (host availability / kill a daemon)",
                   fontsize=9.0)
    ax_l.legend(fontsize=7.6, loc="center right")

    # right: prediction (covers) vs measurement (random/adversarial breach)
    cands = [write, syntactic, liveness]
    labels = ["write-to-fd\n(CU16)", "terminate-any\n(syntactic)", "liveness\nclosure (DERIVED)"]
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
    ax_r.annotate(
        "the SAME syntactic class\nLEAKED in CU22 (multi-hop\nreachability) but COVERS here\n"
        "(process death has no cascade)\n— covers() calls both a priori",
        xy=(1.0, 0.10), xytext=(0.42, 0.40), fontsize=7.4, color="#ff7f0e",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.0),
    )
    ax_r.axhline(0.0, color="#999", lw=0.8)
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels, fontsize=8.2)
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(-0.03, 1.34)
    ax_r.set_title("covers() predicts each fate a priori; the run confirms it", fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center right")

    fig.suptitle(
        "CU23 / H116: the framework is GENERATIVE a 2nd time — applied to host "
        "process-availability (a danger it never saw, in a\nnew world), covers() predicts the "
        "host's own integrity target leaks and derives the unique covering target — confirmed  "
        f"({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.7, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
