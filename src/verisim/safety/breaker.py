"""Trajectory-oscillation circuit breaker (OpenSpec ``add-trajectory-oscillation-breaker``).

The fifth link in the Verisim ↔ OpenLore prototype chain (findings doc §4): a *detect-and-halt*
safety breaker that watches the planning loop's **own** state-transition stream for the two
degenerate shapes a stuck agent falls into — flip-flopping between a few states, or re-editing the
same file in a cycle — and, on breach, **stops the loop and drops the in-memory speculative
rollout**, then surfaces a *recommended* workspace rollback for a human to confirm. It never runs
an autonomous ``git reset`` or file delete behind a heuristic.

Why this shape (and not the original "epistemic lease" idea). The findings doc (§4) showed the
borrowed mechanism — OpenLore's Epistemic Lease — is unrelated machinery (it measures a coding
agent's staleness about *source*, exposes no cross-process API, and has no ``Stale [Critical]``
tier), and that autonomously rewriting the workspace behind a confidence heuristic is destructive.
What *is* sound is the **detection** computed over Verisim's own trajectory, and a strict split
between **reversible** actions (freeze + drop the speculative rollout — internal, cheap) which fire
automatically, and **irreversible** actions (any git/filesystem mutation) which are **human-gated**.
We borrow the lease's *confidence-decay → escalating-response* concept by analogy only; we do not
wire into OpenLore's TypeScript object.

The two metrics are pure, deterministic functions of the transition stream (same stream → same
tier), so the breaker is testable at its boundaries and adds no nondeterminism to the loop:

  - **oscillation** — repeated state→state *bigrams* (the transitions themselves) divided by total
    transitions over a sliding window. A long ``A→B→A→B…`` flip-flop has only two distinct bigrams,
    so the ratio climbs toward 1; a run of distinct, progressing states scores 0.
  - **repetitive-file-modification** — the maximum number of times any single file is modified
    across the window. A loop re-touching one file trips this path even when the *states* differ.

Boundaries this module holds (the proposal's contract):

  - **Fixture only.** Snapshot and rollback operate on the Change-1
    :class:`~verisim.fixture.Fixture` working tree (under Verisim-owned scratch), never the source.
  - **Reversible automatic, irreversible gated.** Freezing the loop and dropping the speculative
    rollout happen on the trip; *no* git/filesystem mutation occurs as part of tripping. A rollback
    runs only on explicit human confirmation (:meth:`TrajectoryBreaker.confirm_and_rollback`).
  - **Snapshot-before-anything.** Known-good checkpoints are taken proactively (``auto_baseline``),
    and a confirmed rollback snapshots the *current* tree first, so no uncommitted work is lost.

The planning-state transition stream is defined here (:class:`PlanningTransition`); the Change-6
headless loop is its producer. Keeping the breaker decoupled from the loop runner lets it ship and
be verified on its own.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from verisim.fixture import DEFAULT_EXCLUDE, Fixture, tree_hash

# Decay tiers, named honestly (no fictional ``Stale [Critical]``). ``critical`` is the trip point.
TIER_OK = "ok"
TIER_DEGRADED = "degraded"
TIER_CRITICAL = "critical"

# Tier ordering, so two sub-metric tiers combine by taking the worse of the two.
_TIER_RANK = {TIER_OK: 0, TIER_DEGRADED: 1, TIER_CRITICAL: 2}

# Where proactive working-tree snapshots live: a sibling of ``repo/`` under the fixture root, so it
# is outside the repo working tree (it never enters ``tree_hash`` of the subject) and under
# Verisim-owned scratch (never the source).
SNAPSHOT_DIRNAME = ".verisim-snapshots"


class BreakerError(RuntimeError):
    """A breaker operation could not be performed safely (e.g. no known-good checkpoint exists)."""


class RollbackNotConfirmed(BreakerError):
    """A rollback was requested without explicit human confirmation; nothing was mutated."""


@dataclass(frozen=True, slots=True)
class PlanningTransition:
    """One step of the planning loop's trajectory: the state it moved *from*, the state it moved
    *to*, and the fixture files that step modified.

    ``from_state``/``to_state`` are opaque, hashable labels chosen by the loop (Change 6); the
    breaker only compares them for equality, so any stable encoding works. ``files_modified`` are
    repo-relative paths, used by the repetitive-modification metric.
    """

    from_state: str
    to_state: str
    files_modified: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    """Thresholds for the two metrics and the sliding-window size. Defaults are the spec's; every
    field is configurable, and the tests pin behavior at the boundaries.

    A flip-flop ``A→B→A→B…`` reaches ``oscillation = 0.5`` after four transitions, so
    ``critical_oscillation = 0.5`` makes a sustained two-state cycle trip ``critical``; a period-3
    cycle lands at ``≈0.4`` (``degraded``), honestly less degenerate.
    """

    window: int = 12
    degraded_oscillation: float = 0.34
    critical_oscillation: float = 0.5
    degraded_file_repeat: int = 3
    critical_file_repeat: int = 5


@dataclass(frozen=True, slots=True)
class OscillationMetric:
    """The two sub-metrics over a window, plus the bookkeeping that justifies them."""

    oscillation: float
    file_repeat: int
    worst_file: str | None
    n_transitions: int
    window: int


@dataclass(frozen=True, slots=True)
class RollbackRecommendation:
    """A *recommendation*, not an action: which known-good baseline to return to, what would
    change, and why. Surfacing this is the most the breaker does to the filesystem on its own —
    executing it requires explicit human confirmation.
    """

    target_label: str
    target_tree_hash: str
    current_tree_hash: str
    diff_preview: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A proactive known-good snapshot of the fixture working tree (content-addressed)."""

    label: str
    tree_hash: str
    snapshot_path: Path
    n_files: int


@dataclass
class SpeculativeRollout:
    """An in-memory speculative planning rollout the breaker can drop on a trip.

    Dropping is *cheap, reversible, and internal* — it discards imagined future states, not
    anything on disk — which is exactly why the breaker may do it automatically.
    """

    states: list[str] = field(default_factory=list)
    dropped: bool = False

    def drop(self) -> None:
        self.states = []
        self.dropped = True


def compute_metric(
    transitions: list[PlanningTransition], config: BreakerConfig | None = None
) -> OscillationMetric:
    """Compute the oscillation and repetitive-modification metrics over the last ``window``
    transitions. Pure: same stream (and config) → same result.

    ``oscillation = (n − distinct bigrams) / n`` over the window, where a *bigram* is a
    ``(from_state, to_state)`` pair (the transition itself). ``file_repeat`` is the maximum number
    of times any single file is modified across the window, with ``worst_file`` naming it.
    """
    cfg = config or BreakerConfig()
    window = transitions[-cfg.window :] if cfg.window > 0 else list(transitions)
    n = len(window)
    if n == 0:
        return OscillationMetric(0.0, 0, None, 0, cfg.window)

    distinct_bigrams = len({(t.from_state, t.to_state) for t in window})
    oscillation = (n - distinct_bigrams) / n

    file_counts: Counter[str] = Counter()
    for t in window:
        file_counts.update(t.files_modified)
    if file_counts:
        worst_file, file_repeat = file_counts.most_common(1)[0]
    else:
        worst_file, file_repeat = None, 0

    return OscillationMetric(oscillation, file_repeat, worst_file, n, cfg.window)


def _osc_tier(oscillation: float, cfg: BreakerConfig) -> str:
    if oscillation >= cfg.critical_oscillation:
        return TIER_CRITICAL
    if oscillation >= cfg.degraded_oscillation:
        return TIER_DEGRADED
    return TIER_OK


def _file_tier(file_repeat: int, cfg: BreakerConfig) -> str:
    if file_repeat >= cfg.critical_file_repeat:
        return TIER_CRITICAL
    if file_repeat >= cfg.degraded_file_repeat:
        return TIER_DEGRADED
    return TIER_OK


def classify(metric: OscillationMetric, config: BreakerConfig | None = None) -> str:
    """Map a metric to a tier — the worse of the oscillation tier and the file-repeat tier. Pure."""
    cfg = config or BreakerConfig()
    osc = _osc_tier(metric.oscillation, cfg)
    fil = _file_tier(metric.file_repeat, cfg)
    return osc if _TIER_RANK[osc] >= _TIER_RANK[fil] else fil


def evaluate(
    transitions: list[PlanningTransition], config: BreakerConfig | None = None
) -> tuple[OscillationMetric, str]:
    """Convenience: ``(metric, tier)`` for a whole stream. Pure (composes the pure functions)."""
    cfg = config or BreakerConfig()
    metric = compute_metric(transitions, cfg)
    return metric, classify(metric, cfg)


def _tree_map(root: Path, exclude: frozenset[str]) -> dict[str, str]:
    """``{relpath: content-token}`` of a working tree (``.git`` and ``exclude`` dirs omitted), for a
    human-readable diff preview. The token distinguishes files (bytes), symlinks (target), and empty
    dirs, so the preview catches structural change as well as content change."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude and d != ".git")
        rel_dir = Path(dirpath).relative_to(root)
        if not filenames and not dirnames and str(rel_dir) != ".":
            out[str(rel_dir) + "/"] = "<empty-dir>"
        for name in sorted(filenames):
            if name in exclude:
                continue
            full = Path(dirpath) / name
            rel = str(rel_dir / name) if str(rel_dir) != "." else name
            if full.is_symlink():
                out[rel] = "symlink:" + os.readlink(full)
            elif full.is_file():
                out[rel] = "file:" + _hash_file(full)
    return out


def _hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so large files do not load wholesale."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _diff_preview(current: dict[str, str], target: dict[str, str]) -> tuple[str, ...]:
    """Human-readable ``A/D/M`` lines describing what a rollback from ``current`` to ``target``
    would do (added-since-baseline removed, deleted-since restored, changed reverted)."""
    lines: list[str] = []
    for path in sorted(set(current) | set(target)):
        in_cur, in_tgt = path in current, path in target
        if in_cur and not in_tgt:
            lines.append(f"A {path}")  # present now, absent at baseline → rollback removes it
        elif in_tgt and not in_cur:
            lines.append(f"D {path}")  # absent now, present at baseline → rollback restores it
        elif current[path] != target[path]:
            lines.append(f"M {path}")  # differs → rollback reverts it
    return tuple(lines)


class TrajectoryBreaker:
    """A detect-and-halt breaker over one planning run's trajectory.

    The loop feeds transitions one at a time to :meth:`observe`; the breaker returns the current
    tier and, on the first ``critical`` tier, fires the safe automatic actions (freeze + drop the
    speculative rollout) and computes a :class:`RollbackRecommendation`. Nothing on disk changes
    until a human calls :meth:`confirm_and_rollback` with ``confirmed=True``.
    """

    def __init__(
        self,
        fixture: Fixture,
        config: BreakerConfig | None = None,
        *,
        rollout: SpeculativeRollout | None = None,
        exclude: frozenset[str] = DEFAULT_EXCLUDE,
        auto_baseline: bool = True,
    ) -> None:
        self.fixture = fixture
        self.config = config or BreakerConfig()
        self.rollout = rollout
        self.exclude = exclude
        self.transitions: list[PlanningTransition] = []
        self.checkpoints: list[Checkpoint] = []
        self.frozen = False
        self.tripped = False
        self.recommendation: RollbackRecommendation | None = None
        self.status: str = TIER_OK
        self._snapshot_seq = 0
        if auto_baseline:
            self.checkpoint("baseline")

    @property
    def _snapshot_root(self) -> Path:
        return self.fixture.root / SNAPSHOT_DIRNAME

    def checkpoint(self, label: str) -> Checkpoint:
        """Take a proactive known-good snapshot of the fixture working tree under scratch.

        Snapshots live outside ``repo/`` so they never enter the subject's ``tree_hash`` and are
        never committed. The label is made unique with a monotonic sequence so repeated labels (and
        the pre-rollback snapshot) never clobber an earlier checkpoint.
        """
        repo = self.fixture.repo_path
        seq = self._snapshot_seq
        self._snapshot_seq += 1
        dest = self._snapshot_root / f"{seq:04d}-{label}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            repo,
            dest,
            symlinks=True,
            ignore=lambda _d, names: {n for n in names if n in self.exclude or n == ".git"},
        )
        cp = Checkpoint(
            label=label,
            tree_hash=tree_hash(repo, self.exclude),
            snapshot_path=dest,
            n_files=len(_tree_map(dest, self.exclude)),
        )
        self.checkpoints.append(cp)
        return cp

    def observe(self, transition: PlanningTransition) -> str:
        """Record a transition, recompute the tier, and trip on the first ``critical``. Returns the
        current tier. Once frozen, further transitions are ignored (the loop is halted)."""
        if self.frozen:
            return self.status
        self.transitions.append(transition)
        metric, tier = evaluate(self.transitions, self.config)
        self.status = tier
        if tier == TIER_CRITICAL and not self.tripped:
            self._trip(metric)
        return tier

    def _trip(self, metric: OscillationMetric) -> None:
        """Fire the safe automatic actions and build a rollback recommendation. Reversible only:
        no git/filesystem mutation happens here."""
        self.tripped = True
        self.frozen = True
        if self.rollout is not None:
            self.rollout.drop()
        reason = (
            f"oscillation={metric.oscillation:.2f} (tier critical) over {metric.n_transitions} "
            f"transitions; worst-file={metric.worst_file!r} modified {metric.file_repeat}×"
        )
        self.recommendation = self._build_recommendation(reason)

    def _build_recommendation(self, reason: str) -> RollbackRecommendation | None:
        """Compute a recommendation against the most recent known-good checkpoint, or ``None`` if no
        checkpoint was ever taken (nothing safe to recommend returning to)."""
        if not self.checkpoints:
            return None
        target = self.checkpoints[-1]
        repo = self.fixture.repo_path
        current_map = _tree_map(repo, self.exclude)
        target_map = _tree_map(target.snapshot_path, self.exclude)
        return RollbackRecommendation(
            target_label=target.label,
            target_tree_hash=target.tree_hash,
            current_tree_hash=tree_hash(repo, self.exclude),
            diff_preview=_diff_preview(current_map, target_map),
            reason=reason,
        )

    def confirm_and_rollback(self, *, confirmed: bool) -> Checkpoint:
        """Execute the recommended rollback — the only path that mutates the filesystem.

        Refuses unless ``confirmed`` is explicitly ``True`` (raising :class:`RollbackNotConfirmed`
        without touching anything). On confirmation it snapshots the *current* tree first (so no
        uncommitted work is lost), then restores the target checkpoint's working tree. Operates
        only on the fixture; the original source repo is never referenced. Returns the pre-rollback
        snapshot's checkpoint.
        """
        if not confirmed:
            raise RollbackNotConfirmed(
                "rollback requires explicit human confirmation; no filesystem mutation performed"
            )
        if self.recommendation is None:
            raise BreakerError("no rollback recommendation to execute")
        target = self.checkpoints[-1]
        pre = self.checkpoint(f"pre-rollback-{target.label}")
        self._restore(target)
        return pre

    def _restore(self, checkpoint: Checkpoint) -> None:
        """Restore the fixture working tree to a checkpoint snapshot, in place.

        Removes the repo's tree entries (preserving ``.git`` and excluded dirs, which the snapshot
        never captured), then copies the snapshot's entries back, so the resulting working tree
        hashes equal to the checkpoint.
        """
        repo = self.fixture.repo_path
        for child in repo.iterdir():
            if child.name == ".git" or child.name in self.exclude:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in checkpoint.snapshot_path.iterdir():
            dest = repo / child.name
            if child.is_dir() and not child.is_symlink():
                shutil.copytree(child, dest, symlinks=True)
            else:
                shutil.copy2(child, dest, follow_symlinks=False)
