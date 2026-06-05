"""Plot the ED1-learned figure (SPEC-7 §0, DS6): the real-model H_ε(ρ) curve + H17.

Two panels from one :class:`~verisim.experiments.ed1_learned.ED1LearnedResult`:

  - **left** -- `H_ε(ρ)` for the trained flat `M_θ` at the bit-exact tier (the standard
    prime-directive shape, comparable to v0/EN1/EH1 and to ED1's synthetic curve), with a CI band.
  - **right** -- the **H17 tradeoff for the real model**: at full consultation (ρ=1), the
    oracle-dollar *per faithful step* under each fixed tier and the cheapest-refutation `escalate`
    policy, with the faithful horizon `H` each arm reaches annotated on its bar. The constrained
    decoder removes out-of-vocab errors, so the real model's residual errors are *subtle*: a cheap
    tier that misses them shows a low `H` (it drifts even at ρ=1) and a high $/faithful-step —
    *the real model's errors are caught at the symbolic/bit-exact tiers, not the cheap one.*
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed1_learned import ED1LearnedResult


def plot_ed1_learned(result: ED1LearnedResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: H_eps(rho) for the trained model
    rhos = [p["rho"] for p in result.curve]
    hs = [p["h_eps"] for p in result.curve]
    lo = [p["ci_lo"] for p in result.curve]
    hi = [p["ci_hi"] for p in result.curve]
    ax_l.plot(rhos, hs, marker="o", color="#1f77b4", label="H_ε  (learned M_θ, bit-exact tier)")
    ax_l.fill_between(rhos, lo, hi, alpha=0.15, color="#1f77b4")
    ax_l.set_xlabel("oracle-consultation budget  ρ")
    ax_l.set_ylabel("faithful horizon  H_ε(ρ)  (steps)")
    ax_l.set_title("The distributed prime directive, learned M_θ: H_ε(ρ)")
    ax_l.legend(loc="upper left", fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # right: H17 — oracle-$ per faithful step per tier/policy arm (ρ=1), H annotated on each bar
    arms = [c["tier"] for c in result.h17]
    dps = [c["dollars_per_step"] for c in result.h17]
    horizons = [c["h_eps"] for c in result.h17]
    colors = {"metamorphic": "#2ca02c", "symbolic": "#ff7f0e",
              "bit_exact": "#d62728", "escalate": "#9467bd"}
    x = range(len(arms))
    bars = ax_r.bar(list(x), dps, 0.62, color=[colors.get(a, "#888") for a in arms])
    for rect, h in zip(bars, horizons, strict=True):
        ax_r.annotate(f"H={h:.0f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                      ha="center", va="bottom", fontsize=8)
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(arms, rotation=15, fontsize=8)
    ax_r.set_ylabel("oracle-$ per faithful step  (lower = better)")
    ax_r.set_title("H17 (real errors): which tier catches the learned model?")
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED1-learned — the real-model distributed H_ε(ρ) + the H17 measurement (SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
