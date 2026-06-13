"""CU12 figure (SPEC-22 H105): knowledge-free targeting -- target the grammar, not the assets.

Two panels from a :class:`~verisim.acd.knowledge_free_targeting.CU12Result`:

  - **left -- breach vs inventory completeness.** The asset-indexed target's breach (over the *true*
    sensitive set) rises as the defender's crown-jewel inventory shrinks below the truth -- from
    zero at a complete inventory to the unverified rate when empty -- a false sense of security. The
    grammar-indexed target (verify every connect) holds flat at zero, inventory-independent, with
    the uniform reference for scale.
  - **right -- the price of knowledge-free safety.** Mean oracle calls per deployment for each
    defense: the full oracle, the uniform reference, the grammar target, and the complete asset
    target -- the grammar target buys inventory-independent zero breach for a small multiple of the
    asset target and a large fraction *under* the full oracle.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.knowledge_free_targeting import CU12Result


def plot_cu12(result: CU12Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    # left: breach vs inventory completeness
    fracs = [c.inventory_frac for c in result.asset]
    breach = [c.breach_random for c in result.asset]
    adv = [c.breach_adversarial for c in result.asset]
    ax_l.plot(fracs, breach, "-o", color="#d62728", lw=2.0, ms=7,
              label="asset-indexed (random timing)")
    ax_l.plot(fracs, adv, "--X", color="#d62728", lw=1.6, ms=8, alpha=0.7,
              label="asset-indexed (adversarial target)")
    ax_l.axhline(result.grammar.breach_random, color="#2ca02c", lw=2.4, ls="-",
                 label="grammar-indexed (all connects)")
    uni = next((c for c in result.uniform if 0.0 < c.breach_random < 1.0), result.uniform[0])
    ax_l.axhline(uni.breach_random, color="#7f7fbf", lw=1.4, ls=":",
                 label=f"uniform {uni.label.split()[-1]} (reference)")
    ax_l.fill_between(fracs, breach, result.grammar.breach_random, color="#d62728", alpha=0.08)
    ax_l.set_xlabel("inventory completeness  |known jewels K| / |true sensitive set T|")
    ax_l.set_ylabel("breach rate over the true sensitive set")
    ax_l.set_xticks(fracs)
    ax_l.set_ylim(-0.04, 1.04)
    ax_l.set_title("asset-indexed targeting is only as good as your inventory", fontsize=9.2)
    ax_l.legend(fontsize=8.0, loc="center left")
    ax_l.annotate(
        "a flagged-but-incomplete\ninventory = false security",
        xy=(result.asset[1].inventory_frac, result.asset[1].breach_random),
        xytext=(0.50, 0.82), ha="center", fontsize=8.2, color="#d62728",
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
    )

    # right: the price of knowledge-free safety (mean calls per defense)
    full = next((c for c in result.uniform if c.label == "uniform ρ=1"), result.uniform[-1])
    uni_ref = next((c for c in result.uniform if c.label != "uniform ρ=1"), result.uniform[0])
    complete_asset = result.asset[-1]
    bars = [
        (f"full oracle\n({full.label.split()[-1]})", full.mean_calls, full.breach_random,
         "#7f7fbf"),
        (f"uniform\n{uni_ref.label.split()[-1]}", uni_ref.mean_calls, uni_ref.breach_random,
         "#7f7fbf"),
        ("grammar\n(all connects)", result.grammar.mean_calls, result.grammar.breach_random,
         "#2ca02c"),
        ("asset\n(complete K=T)", complete_asset.mean_calls, complete_asset.breach_random,
         "#9467bd"),
    ]
    xs = range(len(bars))
    ax_r.bar(xs, [b[1] for b in bars], color=[b[3] for b in bars], width=0.62)
    for x, (_, calls, br, _c) in zip(xs, bars, strict=True):
        ax_r.text(x, calls + 0.6, f"{calls:.1f} calls\nbreach {br:.2f}", ha="center", va="bottom",
                  fontsize=7.8)
    saving = full.mean_calls / max(result.grammar.mean_calls, 1e-9)
    ax_r.annotate(
        f"knowledge-free,\nzero breach,\n{saving:.0f}x under the\nfull oracle",
        xy=(2, result.grammar.mean_calls), xytext=(2.2, 0.5 * full.mean_calls),
        fontsize=8.2, color="#2ca02c", arrowprops={"arrowstyle": "->", "color": "#2ca02c"},
    )
    ax_r.set_xticks(list(xs))
    ax_r.set_xticklabels([b[0] for b in bars], fontsize=8.4)
    ax_r.set_ylabel("mean oracle calls per deployment  (the verification cost)")
    ax_r.set_ylim(0, full.mean_calls * 1.2)
    ax_r.set_title("the price of dropping the asset assumption is small", fontsize=9.2)

    fig.suptitle(
        "CU12 / H105: knowledge-free targeting — the cheap asset-indexed target is only as good as "
        "your crown-jewel inventory;\nverifying the grammar's flow-genesis surface (every connect) "
        f"needs no asset knowledge and is still cheap  ({result.n_episodes} deployments, real M_θ)",
        fontsize=9.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
