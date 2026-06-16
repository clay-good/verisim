"""Plot the CU32 figure (SPEC-22 H125): the verification-latency barrier -- the throughput cost.

Two panels from a :class:`~verisim.acd.latency_barrier.CU32Result`:

  - **left -- safe throughput vs verifier latency ``L``** (a balanced 50/50 reversible/irreversible
    mix). ``pipeline_all`` (hide all latency) is flat at throughput 1.0 but adversarially breached
    on the irreversible slice -- fast and UNSAFE (drawn dashed/red). ``barrier_all`` (stall every
    consult) is safe but decays fastest in ``L``. ``routed`` (stall the irreversible, pipeline the
    reversible) is safe everywhere and decays *slower* -- it recovers the throughput barrier wastes
    on the reversible consults. Safety costs throughput, and routing buys back the reversible slice.
  - **right -- the mix law at the reference ``L``.** Sweeping the irreversible fraction ``f``: the
    routed throughput RISES to 1.0 as ``f -> 0`` (a fully reversible world is safe for free at any
    latency), while ``pipeline_all`` keeps full throughput but pays a residual breach (twin red
    axis) that grows linearly with ``f``. The bill for safety is ``L`` x the irreversible rate.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.latency_barrier import CU32Result


def plot_cu32(result: CU32Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    # left: safe throughput vs latency L (50/50 mix)
    curve = result.latency_curve
    ls = [p.latency for p in curve]
    ax_l.plot(ls, [p.pipeline_all_throughput for p in curve], "--x", color="#d62728", lw=2.0, ms=8,
              label="pipeline_all (hide latency) -- UNSAFE on irreversible")
    ax_l.plot(ls, [p.barrier_all_throughput for p in curve], "-s", color="#888", lw=1.8, ms=7,
              label="barrier_all (stall every consult) -- safe, slowest")
    ax_l.plot(ls, [p.routed_throughput for p in curve], "-o", color="#2ca02c", lw=2.4, ms=8,
              label="routed by reversibility -- safe + fastest-safe")
    ax_l.set_xlabel("verifier latency  L  (wall-clock units per stalled consult)")
    ax_l.set_ylabel("safe throughput  (actions / wall-clock)")
    ax_l.set_ylim(0, 1.06)
    ax_l.set_title("safety costs THROUGHPUT under latency; routing buys back the\n"
                   "reversible slice barrier_all wastes (pipeline is fast but unsafe)",
                   fontsize=9.0)
    ax_l.legend(fontsize=7.8, loc="lower left")
    ref = next((p for p in curve if p.latency == result.latency), curve[-1])
    ax_l.annotate(
        f"routed {ref.routed_throughput:.2f} vs barrier {ref.barrier_all_throughput:.2f}\n"
        f"at L={result.latency} "
        f"({ref.routed_throughput / max(ref.barrier_all_throughput, 1e-9):.2f}x)",
        xy=(result.latency, ref.routed_throughput),
        xytext=(result.latency * 0.30, 0.30), fontsize=8.0, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1),
    )

    # right: the mix law at the reference L
    law = result.fraction_law
    fs = [p.irreversible_fraction for p in law]
    ax_r.plot(fs, [p.routed_throughput for p in law], "-o", color="#2ca02c", lw=2.4, ms=8,
              label="routed throughput (safe)")
    ax_r.plot(fs, [p.barrier_all_throughput for p in law], "-s", color="#888", lw=1.8, ms=7,
              label="barrier_all throughput (safe, flat-slow)")
    ax_r.plot(fs, [p.pipeline_all_throughput for p in law], "--x", color="#d62728", lw=1.8, ms=7,
              label="pipeline_all throughput (UNSAFE)")
    ax_r.set_xlabel("irreversible fraction  f  of the action space")
    ax_r.set_ylabel(f"safe throughput at L={result.latency}")
    ax_r.set_ylim(0, 1.06)
    ax_r.set_title("the mix law: safety is FREE in a reversible world (routed -> 1\n"
                   "as f -> 0); pipeline keeps speed but pays a breach growing in f", fontsize=9.0)

    ax_r2 = ax_r.twinx()
    ax_r2.plot(fs, [p.pipeline_all_breach for p in law], "-^", color="#d62728", lw=2.0, ms=8,
               label="pipeline_all residual breach")
    ax_r2.set_ylabel("pipeline_all adversarial breach", color="#d62728")
    ax_r2.tick_params(axis="y", labelcolor="#d62728")
    ax_r2.set_ylim(-0.03, 1.05)
    lines_l, labels_l = ax_r.get_legend_handles_labels()
    lines_r, labels_r = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7.4, loc="center left")

    fig.suptitle(
        "CU32 / H125: the verification-latency barrier -- the THROUGHPUT cost of safety. "
        "Pipelining (latency-hiding) re-breaches the irreversible slice;\nrouting by reversibility "
        "is the only safe-and-fast policy -- you can defer verification you can undo, you must "
        f"stall for verification you can't  (horizon {result.horizon}, ref L={result.latency})",
        fontsize=8.7, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
