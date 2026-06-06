"""Plot the ED7 Tier-B system-oracle differential figure (SPEC-7 §5.2, DS8) -- *the figure that
retires W1 for the distributed world*.

Three panels from one :class:`~verisim.experiments.ed7.ED7Result`:

  - **left -- per-driver agreement.** Bit-exact observable-cluster agreement between Tier-A (the
    analytic DES) and Tier-B (the autonomous-actor system oracle, seed-shuffled delivery) along the
    three workload drivers, with bootstrap CIs. All three sit at a flat **1.000**, including the
    fault-heavy ``adversarial`` driver where the in-flight medium is largest -- the analytic DES is
    faithful to a genuine message-passing execution.
  - **middle -- the curve overlay.** The prime-directive ``H_ε(ρ)`` run with Tier-A and again with
    Tier-B as the ground-truth/correction oracle. The two curves are *indistinguishable* (gap = 0 at
    every ρ): substituting a real distributed execution for the analytic model leaves the
    faithful-horizon curve unchanged.
  - **right -- the negative control (teeth).** A broken actor that adopts deliveries by *arrival
    order* (order-dependent) is caught by the differential as the ``delivery_order`` boundary, while
    the correct actor agrees 1.000 -- the harness detects a faithfulness break rather than
    rubber-stamping the reimplementation.

The platform + real-OS-thread attestation are stamped on the figure (the macOS-first, disclosed
principle): the headline 1.000 is platform-independent; the threaded-tier availability is disclosed.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed7 import ED7Result


def plot_ed7(result: ED7Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_m, ax_r) = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # --- left: per-driver agreement (Tier-2) ------------------------------------------------------
    drivers = [r["driver"] for r in result.tier2]
    rates = [r["rate"] for r in result.tier2]
    lo = [max(0.0, r["rate"] - r["ci_lo"]) for r in result.tier2]
    hi = [max(0.0, r["ci_hi"] - r["rate"]) for r in result.tier2]
    x = range(len(drivers))
    ax_l.bar(x, rates, color="#2ca02c", alpha=0.85, label="agree (bit-exact cluster)")
    ax_l.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=4, lw=1)
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(drivers, rotation=15, fontsize=9)
    ax_l.set_ylabel("fraction of transitions agreeing")
    ax_l.set_ylim(0, 1.04)
    ax_l.axhline(1.0, color="#2ca02c", lw=0.8, ls=":")
    ax_l.set_title("ED7: Tier-A vs Tier-B (real actors) — per-driver agreement")
    ax_l.legend(loc="lower left", fontsize=8)

    # --- middle: the head-to-head H_eps(rho) overlay (Tier-3) -------------------------------------
    rhos = [p["rho"] for p in result.tier3]
    h_ref = [p["h_ref"] for p in result.tier3]
    h_sys = [p["h_sys"] for p in result.tier3]
    ax_m.plot(rhos, h_ref, marker="o", color="#1f77b4", lw=2.4, label="Tier-A (analytic DES)")
    ax_m.plot(rhos, h_sys, marker="x", color="#d62728", lw=1.4, ls="--",
              label="Tier-B (real DST actors)")
    ax_m.set_xlabel("oracle-consultation budget  ρ")
    ax_m.set_ylabel("faithful horizon  H_ε(ρ)  (steps)")
    max_gap = max((p["gap"] for p in result.tier3), default=0.0)
    ax_m.set_title(f"ED7 Tier-3: the curve is oracle-invariant (max gap = {max_gap:.3f})")
    ax_m.legend(loc="upper left", fontsize=8)
    ax_m.grid(True, alpha=0.3)

    # --- right: the negative control (teeth) ------------------------------------------------------
    nc = result.negative_control
    correct_rate = result.tier2[-1]["rate"] if result.tier2 else 1.0  # adversarial, correct actor
    broken_rate = nc["n_agree"] / nc["n"] if nc.get("n") else 0.0
    bars = ax_r.bar(
        ["correct\nLWW actor", "broken\narrival-order actor"],
        [correct_rate, broken_rate],
        color=["#2ca02c", "#d62728"], alpha=0.85,
    )
    ax_r.set_ylabel("agreement with Tier-A (adversarial driver)")
    ax_r.set_ylim(0, 1.04)
    ax_r.axhline(1.0, color="#2ca02c", lw=0.8, ls=":")
    caught = nc.get("n_caught", 0)
    ax_r.set_title(f"ED7 negative control: the harness has teeth\n"
                   f"(caught {caught} delivery-order breaks)")
    for bar, val in zip(bars, [correct_rate, broken_rate], strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=9)

    th = result.threaded
    th_txt = (f"real-OS-thread tier: agreement {th['rate']:.3f}"
              if th.get("available") else "real-OS-thread tier: unavailable (disclosed)")
    headline = (f"observable-cluster agreement = {result.overall_agreement:.3f}  •  "
                f"residual = {result.residual_fraction:.3f}  •  {th_txt}  •  "
                f"platform = {result.platform}")
    fig.suptitle(headline, y=1.00, fontsize=9.5)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
