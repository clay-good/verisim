"""Plot the ED1 distributed prime-directive figure (SPEC-7 §0, DS6): the H_ε(ρ) curve + H17.

Two panels from one :class:`~verisim.experiments.ed1.ED1Result`:

  - **left** -- `H_ε(ρ)`: free-running → fully-consulted faithful horizon at the bit-exact tier (the
    standard prime-directive shape, comparable to v0/EN1/EH1), with a CI band.
  - **right** -- the **H17 tradeoff**: oracle-dollar *per faithful step* for each tier, grouped
    by the proposer's error class. Lower is better. For **gross** (out-of-vocab) errors the
    cheap metamorphic tier is far cheaper per faithful step than bit-exact; for **subtle**
    (in-flight) errors the cheap tiers miss the drift, so bit-exact wins — *whether the tiered
    oracle helps depends on where the model's errors fall.*
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed1 import ED1Result


def plot_ed1(result: ED1Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: H_eps(rho)
    rhos = [p["rho"] for p in result.curve]
    hs = [p["h_eps"] for p in result.curve]
    lo = [p["ci_lo"] for p in result.curve]
    hi = [p["ci_hi"] for p in result.curve]
    ax_l.plot(rhos, hs, marker="o", color="#1f77b4", label="H_ε  (bit-exact tier)")
    ax_l.fill_between(rhos, lo, hi, alpha=0.15, color="#1f77b4")
    ax_l.set_xlabel("oracle-consultation budget  ρ")
    ax_l.set_ylabel("faithful horizon  H_ε(ρ)  (steps)")
    ax_l.set_title("The distributed prime directive: H_ε(ρ)")
    ax_l.legend(loc="upper left", fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # right: H17 — oracle-dollar per faithful step, grouped by error mode, bars per tier
    modes = list(dict.fromkeys(c["mode"] for c in result.h17))
    tiers = list(dict.fromkeys(c["tier"] for c in result.h17))
    colors = {"metamorphic": "#2ca02c", "symbolic": "#ff7f0e", "bit_exact": "#d62728"}
    width = 0.8 / max(1, len(tiers))
    x = range(len(modes))
    for j, tier in enumerate(tiers):
        ys = [
            next((c["dollars_per_step"] for c in result.h17
                  if c["mode"] == m and c["tier"] == tier), 0.0)
            for m in modes
        ]
        offs = [i + (j - (len(tiers) - 1) / 2) * width for i in x]
        ax_r.bar(offs, ys, width, label=tier, color=colors.get(tier, "#888"))
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels([f"{m}-error\nproposer" for m in modes])
    ax_r.set_ylabel("oracle-$ per faithful step  (lower = better)")
    ax_r.set_title("H17: does a cheap tier buy more horizon per $?")
    ax_r.legend(title="π_w tier", fontsize=8)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED1 — the distributed H_ε(ρ) + the tiered-oracle (H17) measurement (SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
