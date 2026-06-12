"""verisim-cue contamination control: is the frozen eval overfit-resistant? (SPEC-21 §8, H68).

A frozen benchmark is only trustworthy if a model that has *memorized its public eval seeds* scores
conspicuously worse on a **disjoint held-out shard** -- so the public-minus-held-out gap is a usable
overfit detector. This is the SPEC-18 PB-pack H68 control ported to the computer-use vertical: it is
what lets an adopter trust a verisim-cue scorecard was earned on the dynamics, not the seeds.

Following the SPEC-18 abstraction (per-shard fidelity, the established codebase pattern), the two
models are controlled stand-ins, torch-free:

  - **the memorizer** -- faithful on the *public* shard (it memorized those workloads; the oracle
    predictor stands in), blind on the *held-out* shard (unseen workloads; the identity/no-change
    predictor stands in). Its public catch is ~1.0, its held-out catch low -> a **large** gap.
  - **the honest model** -- *equally* imperfect on both shards (the identity predictor on both; real
    trained model generalizes, so it drifts the same on seen and unseen seeds) -> a **small** gap.

The benchmark is contamination-resistant iff the memorizer gap is materially larger than the honest
model's -- i.e. the held-out shard catches the overfit the public shard hides. CPU-only, seeded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

from verisim.acd.host_integrity import make_workload, oracle_step
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle

from .pack import CueManifest
from .tasks import TASK_SUITE, keyed_defense_reward

# A host step function (the predictor's model of dynamics): (state, action) -> next state.
HostStepFn = Callable[[HostState, HostAction], HostState]


def _blind_step(state: HostState, action: HostAction) -> HostState:
    """The blind predictor: predict no change (identity). It writes nothing, so it drifts on every
    content task -- but it drifts *equally* on any seed shard, which is the honest-model control."""
    return state


def _shard_catch(
    predictor: HostStepFn, true_step: HostStepFn, manifest: CueManifest, seeds: tuple[int, ...]
) -> float:
    """Mean catch rate over the suite on a seed shard, scoring ``predictor`` against the truth."""
    workloads = [
        make_workload(s, manifest.horizon, driver=manifest.driver) for s in seeds
    ]
    task_means: list[float] = []
    for task in TASK_SUITE:
        rewards = [
            keyed_defense_reward(predictor, true_step, start, actions, task.budget, task.key_fn)
            for start, actions in workloads
        ]
        task_means.append(fmean(rewards))
    return fmean(task_means)


@dataclass(frozen=True)
class ContaminationResult:
    """The H68 overfit-detector verdict for verisim-cue."""

    honest_gap: float  # honest model's public-minus-held-out catch gap (~0: equally imperfect)
    memorizer_gap: float  # memorizer's public-minus-held-out gap (large: overfit to public)
    margin: float  # memorizer_gap - honest_gap (the separation)
    contamination_resistant: bool  # the held-out shard catches the overfit the public shard hides


def run_contamination(
    manifest: CueManifest | None = None, *, heldout_offset: int = 10_000, min_margin: float = 0.2
) -> ContaminationResult:
    """The H68 contamination control: does the held-out shard catch a public-seed memorizer?

    The public shard is the manifest's frozen seeds; the held-out shard is a disjoint range
    (``+heldout_offset``). The memorizer is faithful on public / blind on held-out; the honest model
    is blind on both. Contamination-resistant iff the memorizer's public-minus-held-out gap exceeds
    the honest model's by at least ``min_margin``.
    """
    manifest = manifest or CueManifest()
    oracle = ReferenceHostOracle()
    faithful = oracle_step(oracle)
    true_step = oracle_step(oracle)
    public = manifest.seeds
    heldout = tuple(s + heldout_offset for s in manifest.seeds)

    # honest: equally imperfect on both shards (blind everywhere) -> small gap.
    honest_gap = (
        _shard_catch(_blind_step, true_step, manifest, public)
        - _shard_catch(_blind_step, true_step, manifest, heldout)
    )
    # memorizer: faithful on public (memorized), blind on held-out (unseen) -> large gap.
    memorizer_gap = (
        _shard_catch(faithful, true_step, manifest, public)
        - _shard_catch(_blind_step, true_step, manifest, heldout)
    )
    margin = memorizer_gap - honest_gap
    return ContaminationResult(
        honest_gap=honest_gap, memorizer_gap=memorizer_gap, margin=margin,
        contamination_resistant=margin >= min_margin,
    )
