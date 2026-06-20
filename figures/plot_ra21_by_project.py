"""Plot the RA21 cross-project generalization figure (SPEC-22 H153).

Reads the committed, anonymized ``bench/cc_corpus/BY_PROJECT.csv`` (ordinal project ids, no real
names) and draws two panels:

  - **left -- the distribution.** A histogram of per-project off-surface density. The mass sits at
    ~0.98-1.0 across structurally different projects: the headline is not one-workflow-specific.
  - **right -- the dip is small security-work projects, not the bulk of traffic.** Off-surface
    density (y) vs the project's call volume (log x). High-volume ordinary projects cluster at the
    top; the low-density outliers are small projects whose work *is* credential/security work (the
    surface is the work), so the gate's budget correctly auto-scales to how dangerous the work is.

Run: ``python figures/plot_ra21_by_project.py`` (after ``bench/cc_corpus/extract.py``).
"""

from __future__ import annotations

import csv
from pathlib import Path


def _read(path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "n_calls": float(r["n_calls"]),
                "offsurface_density": float(r["offsurface_density"]),
                "coverage_prompt_rate": float(r["coverage_prompt_rate"]),
                "denylist_prompt_rate": float(r["denylist_prompt_rate"]),
            })
    return rows


def plot_by_project(rows: list[dict[str, float]], path: str | Path,
                    min_calls: int = 20) -> Path:  # pragma: no cover - plot
    import matplotlib

    matplotlib.use("Agg")
    from statistics import median

    import matplotlib.pyplot as plt

    kept = [r for r in rows if r["n_calls"] >= min_calls]
    dens = [r["offsurface_density"] for r in kept]
    med = median(dens)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: distribution of per-project off-surface density ------------------------------------
    ax_l.hist(dens, bins=[i / 20 for i in range(11, 21)], color="#2a7", edgecolor="white")
    ax_l.axvline(med, color="#c44", linestyle="--", label=f"median {med:.3f}")
    ax_l.set_xlabel("off-surface (auto-approved) density")
    ax_l.set_ylabel(f"projects (n={len(kept)}, >= {min_calls} calls)")
    ax_l.set_xlim(0.5, 1.01)
    ax_l.set_title("Off-surface density is high and tight across projects")
    ax_l.legend(fontsize=9)

    # --- right: density vs volume -- the dip is small security-work projects ----------------------
    xs = [r["n_calls"] for r in kept]
    ys = [r["offsurface_density"] for r in kept]
    ax_r.scatter(xs, ys, s=40, color="#2a7", alpha=0.8, edgecolor="#176")
    ax_r.set_xscale("log")
    ax_r.set_xlabel("project call volume (log)")
    ax_r.set_ylabel("off-surface density")
    ax_r.set_ylim(0.5, 1.01)
    ax_r.axhline(0.95, color="#888", linestyle=":", linewidth=1)
    ax_r.set_title("Low-density outliers are small security-work projects")
    low = [(x, y) for x, y in zip(xs, ys, strict=False) if y < 0.95]
    if low:
        lx, ly = min(low, key=lambda p: p[1])
        ax_r.annotate("surface = the work\n(credential/security project)",
                      xy=(lx, ly), xytext=(lx * 1.5, ly + 0.12), fontsize=8, color="#a40",
                      arrowprops={"arrowstyle": "->", "color": "#a40"})

    fig.suptitle("RA21 cross-project generalization: the §8 operating point holds across projects",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - local entrypoint
    root = Path(__file__).resolve().parent.parent
    rows = _read(root / "bench" / "cc_corpus" / "BY_PROJECT.csv")
    out = plot_by_project(rows, root / "figures" / "ra21_by_project.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
