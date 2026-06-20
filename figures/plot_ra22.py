"""Plot the RA22 figure (SPEC-22 H154): automated coverage synthesis + certification.

One panel, the certificate as a stacked picture of the realizing surface: the fraction the
synthesized target (grown from empty) now covers, vs the indirection residual it isolated and routed
to the post-commit diff -- with the soundness invariant (zero silent miss) annotated. A small
companion bar shows the per-class outcome.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.coverage_synth import CoverageCertificate


def plot_ra22(cert: CoverageCertificate, path: str | Path) -> Path:  # pragma: no cover - plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = cert.n_realizing or 1
    covered = cert.covered / n
    residual = cert.residual / n

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # --- left: the realizing surface, partitioned by the synthesized certificate -----------------
    ax_l.bar([0], [covered], color="#2a7")
    ax_l.bar([0], [residual], bottom=[covered], color="#48c")
    ax_l.set_xlim(-0.7, 0.7)
    ax_l.set_xticks([0])
    ax_l.set_xticklabels([f"realizing actions\n(n={cert.n_realizing})"])
    ax_l.set_ylabel("fraction of realizing surface")
    ax_l.set_ylim(0, 1.08)
    ax_l.set_title(f"Target synthesized from EMPTY in {cert.rounds_to_converge} CEGIS rounds")
    ax_l.text(0, covered / 2, f"{covered:.0%}\ncovered by\nsynthesized target",
              ha="center", va="center", color="white", fontsize=9)
    ax_l.text(0, covered + residual / 2,
              f"{residual:.0%}\nindirection residual\n-> post-commit diff",
              ha="center", va="center", color="white", fontsize=9)
    ax_l.text(0.5, 1.10, f"silent miss = {cert.silent_miss}  |  benign over-fire = "
              f"{cert.benign_overfire}   (soundness invariant)",
              transform=ax_l.transAxes, ha="center", fontsize=8, color="#a40")

    # --- right: per-class outcome -----------------------------------------------------------------
    classes = ["literal", *cert.residual_classes]
    outcome = [1.0] + [0.0] * len(cert.residual_classes)  # covered vs residual
    colors = ["#2a7"] + ["#48c"] * len(cert.residual_classes)
    y = range(len(classes))
    ax_r.barh(list(y), [1.0] * len(classes), color="#eee")  # full-width track
    ax_r.barh(list(y), outcome, color=colors)
    ax_r.set_yticks(list(y))
    ax_r.set_yticklabels(classes, fontsize=8)
    ax_r.set_xlim(0, 1.0)
    ax_r.set_xlabel("fraction covered by the synthesized pre-commit target")
    ax_r.set_title("Literal class covered; indirection isolated as residual")
    ax_r.invert_yaxis()

    fig.suptitle("RA22: automated coverage -- the covering target synthesized and certified, "
                 "not hand-specified", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - local entrypoint
    from verisim.realagent.coverage_synth import generate_corpus, synthesize

    _t, cert = synthesize(generate_corpus())
    root = Path(__file__).resolve().parent.parent
    out = plot_ra22(cert, root / "figures" / "ra22_coverage_synth.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
