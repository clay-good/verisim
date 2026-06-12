"""The frozen verisim-cue benchmark: manifest, hash, version, metadata (SPEC-21 §8, the artifact).

`verisim-cue` is the *artifact* half of SPEC-21: the verifiable computer-use environment (the host
shell/file/process world) + the **load-bearing-frontier benchmark** -- the one computer-use
world-model benchmark with ground-truth labels *and* a faithfulness-load-bearing verdict per task.
It is the computer-use vertical of the SPEC-18 `verisim-bench` line, hardened the same way: the task
battery is pinned into a hashable :class:`CueManifest`, and the standardized metadata almost no
benchmark ships (BetterBench's finding) is emitted -- a **Croissant** descriptor, a **datasheet**,
and a **task-card** carrying the thing that distinguishes verisim-cue: the per-task *load-bearing
verdict* (does the faithful predictor beat the free one -- is the oracle load-bearing for control on
this task at this scale).

The battery is the SPEC-21 ordered structure->content task suite
([`tasks.TASK_SUITE`](./tasks.py)); the manifest pins it with the workload regime (driver, horizon,
seeds), the capacity ladder labels, and the load-bearing threshold. The manifest hash is the
MAJOR-version identity (`verisim-cue@MAJOR.MINOR.PATCH`); a load-bearing verdict is always
attributable to a battery version. Adoption is *not* a hypothesis (SPEC-18 §9); it ships regardless.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from .tasks import TASK_SUITE

CUE_VERSION = "0.1.0"


@dataclass(frozen=True)
class CueTaskSpec:
    """One frozen task spec on the structure->content spectrum (the benchmark's unit)."""

    name: str
    keyed_dimension: str  # "procs" | "fds" | "fs" -- the host dimension it keys on
    order: int  # 0 = structure ... 3 = deep content (the spectrum position)
    budget: int


def _suite_specs() -> tuple[CueTaskSpec, ...]:
    return tuple(
        CueTaskSpec(t.name, t.keyed_dimension, t.order, t.budget) for t in TASK_SUITE
    )


@dataclass(frozen=True)
class CueManifest:
    """A frozen, hashable verifiable-computer-use benchmark battery (SPEC-21 §8)."""

    name: str = "verisim-cue"
    version: str = CUE_VERSION
    world: str = "host"  # the shell / filesystem / process slice (oracle-grounded computer use)
    driver: str = "forky"
    horizon: int = 16
    seeds: tuple[int, ...] = tuple(range(700, 724))
    ladder: tuple[str, ...] = ("xs", "s", "m", "l")  # CPU rungs; the GPU run extends to xxxl
    threshold: float = 0.05  # gap above which the oracle is load-bearing for a task
    tasks: tuple[CueTaskSpec, ...] = field(default_factory=_suite_specs)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["tasks"] = [asdict(t) for t in self.tasks]
        return d

    def manifest_hash(self) -> str:
        """A stable hash of the battery -- the MAJOR-version identity printed on every verdict."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def version_tag(self) -> str:
        return f"{self.name}@{self.version}+{self.manifest_hash()}"


def croissant_metadata(manifest: CueManifest) -> dict[str, object]:
    """An MLCommons Croissant descriptor for the frozen computer-use battery (SPEC-21 §8).

    The machine-readable, ML-ready descriptor BetterBench found almost no benchmark ships -- so the
    battery is discoverable and portable for the Inspect / Hub ecosystems. The records are
    *oracle-generated* (the host reference oracle), which the provenance states. One record set
    per task, ordered structure->content; each carries its keyed dimension + load-bearing role.
    """
    return {
        "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/"},
        "@type": "Dataset",
        "name": manifest.name,
        "version": manifest.version,
        "description": (
            "Verifiable computer-use (shell/file/process) world-model benchmark: per-task "
            "faithful-vs-free predictive-defense rollouts on the host world, labeled exactly by a "
            "deterministic reference oracle, ordered structure->content, each carrying a "
            "faithfulness-load-bearing verdict swept across the model-capacity ladder (the SPEC-21 "
            "scale law)."
        ),
        "license": "https://opensource.org/licenses/MIT",
        "creator": {"@type": "Organization", "name": "Verisim"},
        "cr:provenance": "oracle-generated (host reference oracle); no scraped data",
        "manifestHash": manifest.manifest_hash(),
        "cr:recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": t.name,
                "description": (
                    f"order-{t.order} ({'structure' if t.order == 0 else 'content'}) task keyed on "
                    f"the {t.keyed_dimension} dimension; budget {t.budget}"
                ),
                "cr:field": [
                    {"name": "state", "dataType": "Text"},
                    {"name": "action", "dataType": "Text"},
                    {"name": "next_state", "dataType": "Text"},
                    {"name": "keyed_set", "dataType": "Text"},
                    {"name": "load_bearing_gap", "dataType": "Float"},
                ],
            }
            for t in manifest.tasks
        ],
    }


def datasheet(manifest: CueManifest) -> str:
    """A Gebru-et-al. datasheet for the computer-use benchmark (SPEC-21 §8)."""
    return f"""# Datasheet — {manifest.version_tag()}

## Motivation
`verisim-cue` is the one **computer-use** world-model benchmark whose labels are *exact*: a
deterministic host oracle (the shell/file/process slice of computer use — the slice that *admits* a
ground-truth oracle, unlike GUI) supplies the true next state at every step. Each task is scored not
only on success but on whether **faithfulness was load-bearing** for it — a verdict no oracle-free
computer-use benchmark can produce. Built for defensive autonomous-cyber-defense and world-model
faithfulness research.

## Composition
- **Environment:** the SPEC-6 host world (`{manifest.world}`): processes, file descriptors, a
  filesystem, commands, under a deterministic reference oracle and a real-`/bin/sh` system anchor
  (SPEC-11).
- **Tasks:** {len(manifest.tasks)} predictive-defense tasks ordered structure→content
  ({", ".join(t.name for t in manifest.tasks)}), each a faithful-vs-free gap over a keyed-set
  extractor.
- **Workload:** the `{manifest.driver}` driver; {len(manifest.seeds)} seeds, {manifest.horizon}-step
  episodes per task.
- **Capacity ladder:** {", ".join(manifest.ladder)} (CPU rungs; the GPU run extends the top) — the
  benchmark is the substrate of a *scale law*, not a single number.
- **Records are oracle-generated**, not scraped — no PII, no copyright surface.

## Collection process
Deterministic and seeded: every record is a pure function of (task, driver, seed), regenerable from
this manifest (hash `{manifest.manifest_hash()}`). No human annotation.

## Intended use
Defensive ACD environments and computer-use world-model faithfulness research. **Not** for offensive
automation (SPEC.md §13).

## Limits
The reference oracle is a *model* of host semantics, validated bit-exact against a real shell on the
structure-building grammar (SPEC-11 H27) but not across all of POSIX. Computer use here is
shell/file/process, **not** GUI — that is the oracle-grounded slice and the point. Load-bearing
verdicts are comparable only within a fixed manifest hash and at a stated scale.
"""


def task_card(manifest: CueManifest, verdicts: dict[str, dict[str, float]] | None = None) -> str:
    """A task-card carrying the per-task *load-bearing verdict* — the thing verisim-cue adds.

    ``verdicts`` maps task name -> ``{"gap": float, "scale": float}`` (e.g. read from a committed
    CS1 frontier run at the top CPU rung); when absent the card documents the spectrum structurally
    and marks the verdict pending. A task is *load-bearing* iff its gap exceeds the manifest
    threshold — i.e. the oracle-in-the-loop is needed for control on that task at that scale.
    """
    lines = []
    for t in sorted(manifest.tasks, key=lambda s: s.order):
        band = "structure" if t.order == 0 else ("near-structure" if t.order == 1 else "content")
        if verdicts and t.name in verdicts:
            gap = verdicts[t.name]["gap"]
            verdict = "load-bearing" if gap > manifest.threshold else "not load-bearing"
            cell = f"{gap:+.2f} → **{verdict}**"
        else:
            cell = "pending (run the CS1 frontier)"
        lines.append(f"| {t.order} | {t.name} | {t.keyed_dimension} | {band} | {cell} |")
    rows = "\n".join(lines)
    note = ""
    if verdicts:
        scales = {v.get("scale", 0.0) for v in verdicts.values()}
        note = f"\n\nVerdicts measured at scale ≈ {max(scales):.0f} params (the top CPU rung)."
    return f"""# Task card — load-bearing verdicts — {manifest.version_tag()}

What distinguishes `verisim-cue` from every other computer-use benchmark: each task carries a
**faithfulness-load-bearing verdict** — does a faithful predictor (oracle rollout) beat a free one
(`M_θ` rollout) by more than the threshold ({manifest.threshold})? If yes, the oracle-in-the-loop is
load-bearing for control on that task; if no, the model already gets that dimension right. Swept
across the capacity ladder, the boundary between the two *is* the SPEC-21 load-bearing frontier.

| order | task | keyed dimension | structure↔content | load-bearing verdict |
|---|---|---|---|---|
{rows}{note}

**Reading:** the gap should rise with the task order (structure→content) and the load-bearing tasks
should be the deeper-content ones — the structure tasks the model already learns faithfully. The
SPEC-21 scale law measures how this frontier *moves* as capacity grows (it recedes structural-first,
leaving an irreducible content residue, H87/H88).
"""
