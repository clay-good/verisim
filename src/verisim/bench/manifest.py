"""The frozen faithfulness-benchmark battery: manifest, hash, versioning, metadata (SPEC-18 §6, §7).

A benchmark is only an asset if it is *frozen, versioned, and attributable*. This module pins the
battery (which worlds, drivers, seeds, horizons, and the ε policy) into a hashable
:class:`BatteryManifest` and emits the standardized metadata almost no benchmark ships
(BetterBench's
finding, SPEC-18 §7): a **Croissant** descriptor, a **datasheet**, and a **model-card**. The
manifest
hash is printed on every leaderboard row, so a score is always attributable to a battery version
(`verisim-bench@MAJOR.MINOR.PATCH`: MAJOR = manifest-hash change, MINOR = additive worlds/proposers,
PATCH = metadata only).

The reference **proposers** are a fidelity ladder: a floor (predicts no change), graded learned
tiers,
and an oracle ceiling. The committed core uses controlled stand-ins (the SPEC-13
:class:`~verisim.experiments.sr_common.StallDrafter` at increasing per-step accuracy ``α``); the
trained flat-transformer / GNN+RSSM arms are the deferred entries (``trained=True``,
``skipif``-guarded,
never scored when no checkpoint is present -- the LP7 rule). A discriminative benchmark must *stably
order* the ladder (H65); that is exactly what PB-bench measures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

BENCH_VERSION = "0.1.0"


@dataclass(frozen=True)
class Proposer:
    """One ranked entry on the leaderboard -- a world model of some fidelity (SPEC-18 §6)."""

    name: str
    tier: str  # "floor" | "learned-lo" | "learned-mid" | "learned-hi" | "ceiling"
    alpha: float  # controlled per-step fidelity (the stand-in); a trained arm replaces this
    trained: bool = False  # the deferred real arms set this; skipif when no checkpoint


# The reference fidelity ladder (SPEC-18 §6): floor -> graded learned tiers -> oracle ceiling. The
# learned tiers stand in for the flat transformer and GNN+RSSM (trained arms deferred).
REFERENCE_PROPOSERS: tuple[Proposer, ...] = (
    Proposer("null", "floor", alpha=0.0),
    Proposer("learned-lo", "learned-lo", alpha=0.65),
    Proposer("learned-mid", "learned-mid", alpha=0.80),
    Proposer("learned-hi", "learned-hi", alpha=0.92),
    Proposer("oracle-ceiling", "ceiling", alpha=1.0),
)


@dataclass(frozen=True)
class BatteryManifest:
    """A frozen, hashable benchmark battery (SPEC-18 §6, §7)."""

    name: str = "verisim-bench"
    version: str = BENCH_VERSION
    worlds: tuple[str, ...] = ("network", "host", "filesystem")
    drivers: tuple[str, ...] = ("weighted", "forky", "structural")  # one per world, in order
    seeds: tuple[int, ...] = tuple(range(16))
    n_steps: int = 80
    epsilon_g: float = 1.0  # ε = g·δ per world (SPEC-13 ratio; δ measured per world)
    proposers: tuple[Proposer, ...] = field(default_factory=lambda: REFERENCE_PROPOSERS)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["proposers"] = [asdict(p) for p in self.proposers]
        return d

    def manifest_hash(self) -> str:
        """A stable hash of the battery -- the MAJOR-version identity printed on every score."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def version_tag(self) -> str:
        return f"{self.name}@{self.version}+{self.manifest_hash()}"


def croissant_metadata(manifest: BatteryManifest) -> dict[str, object]:
    """An MLCommons Croissant descriptor for the frozen battery (SPEC-18 §7).

    The machine-readable, ML-ready dataset descriptor BetterBench found almost no benchmark ships --
    so the battery is discoverable and portable for the Inspect / Hub ecosystems. The records are
    *oracle-generated* (not scraped), which the provenance field states.
    """
    return {
        "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/"},
        "@type": "Dataset",
        "name": manifest.name,
        "version": manifest.version,
        "description": (
            "Oracle-grounded world-model faithfulness benchmark: per-step (state, action, true "
            "next-state) records and H_ε(ρ) faithful-horizon scores across the network, host, and "
            "filesystem worlds, labeled bit-exact by a deterministic reference oracle."
        ),
        "license": "https://opensource.org/licenses/MIT",
        "creator": {"@type": "Organization", "name": "Verisim"},
        "cr:provenance": "oracle-generated (deterministic reference oracle); no scraped data",
        "manifestHash": manifest.manifest_hash(),
        "cr:recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": world,
                "description": f"{world} world rollouts under the {driver} driver",
                "cr:field": [
                    {"name": "state", "dataType": "Text"},
                    {"name": "action", "dataType": "Text"},
                    {"name": "next_state", "dataType": "Text"},
                    {"name": "faithful_horizon", "dataType": "Integer"},
                ],
            }
            for world, driver in zip(manifest.worlds, manifest.drivers, strict=False)
        ],
    }


def datasheet(manifest: BatteryManifest) -> str:
    """A Gebru-et-al. datasheet for the benchmark (SPEC-18 §7) -- provenance, composition, limits.
    """
    worlds = ", ".join(manifest.worlds)
    return f"""# Datasheet — {manifest.version_tag()}

## Motivation
The one world-model faithfulness benchmark whose labels are **exact**: a deterministic oracle
supplies the true next state at every step, so `H_ε(ρ)` (faithful horizon at oracle budget ρ) is
measured against ground truth rather than eyeballed. Built for defensive autonomous-cyber-defense
and world-model faithfulness research.

## Composition
- **Worlds:** {worlds} ({len(manifest.worlds)} worlds).
- **Drivers:** {", ".join(manifest.drivers)} (one per world).
- **Seeds:** {len(manifest.seeds)} per (world, driver); rollouts of {manifest.n_steps} steps.
- **Labels:** bit-exact next-state from the reference oracle; `ε = {manifest.epsilon_g}·δ` per world
  (δ = the world's single-edit divergence granularity, SPEC-13).
- **Records are oracle-generated**, not scraped — no PII, no copyright surface.

## Collection process
Deterministic and seeded: every record is a pure function of (world, driver, seed), regenerable from
this manifest (hash `{manifest.manifest_hash()}`). No human annotation.

## Intended use
Defensive ACD environments and world-model faithfulness research. **Not** for offensive automation.

## Limits
The reference oracle is a *model* of POSIX/network/host semantics, validated bit-exact against a
real shell on the structure-building grammar (SPEC-11 H27) but not across all of POSIX. Scores are
comparable only within a fixed manifest hash.
"""


def model_card(manifest: BatteryManifest) -> str:
    """A model-card for the benchmark's reference proposers (SPEC-18 §7)."""
    rows = "\n".join(
        f"| {p.name} | {p.tier} | {p.alpha:.2f} | "
        f"{'deferred (GPU)' if p.trained else 'shipped (CPU stand-in)'} |"
        for p in manifest.proposers
    )
    return f"""# Model card — reference proposers — {manifest.version_tag()}

The fidelity ladder the benchmark must stably order (H65). The committed entries are controlled CPU
stand-ins (a per-step-accuracy `α` drafter); the trained flat-transformer and GNN+RSSM arms are
deferred (GPU, `skipif`-guarded, never scored without a checkpoint — the LP7 rule).

| proposer | tier | fidelity α | status |
|---|---|---|---|
{rows}

**Intended use:** discriminative-validity and rank-stability measurement of the benchmark.
**Caveat:** the stand-ins isolate the benchmark's *discriminative* behavior (does it order fidelity
stably?); absolute `H_ε` magnitudes for the real arms require the trained checkpoints.
"""
