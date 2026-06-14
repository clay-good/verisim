"""Typed run model for the headless CD pipeline (OpenSpec ``add-headless-cd-pipeline``).

The sixth and final link in the Verisim ↔ OpenLore prototype chain (findings doc §9.1): the
types the headless entry point produces. Every stage of the pipeline writes one of these as a
typed artifact, and the whole run aggregates into a single :class:`RunReport` carrying a content
**seal** (a sha256 over the report's deterministic content) so a run is inspectable, re-runnable,
and tamper-evident.

The discipline mirrors the rest of the chain: nothing here reads a clock or an RNG, so a report
built from a fixed ``(fixture revision, intent, seed)`` is byte-identical across runs — the
``HeadlessOrderedPipeline`` determinism requirement, met by construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# The report artifact format version. A reader keys on this and fails closed on an unknown version
# rather than mis-parsing — the schema-guard discipline carried from the bridge/trace modules.
REPORT_SCHEMA_VERSION = "verisim-cd-run-v1"

# Per-stage outcome vocabulary. ``halted`` is reserved for the breaker gate (a *safe* stop, not an
# error); ``failed`` is an exception in the stage; ``skipped`` is a later stage blocked by an
# earlier halt/failure — the fail-safe ordering requirement made legible in the report.
STAGE_OK = "ok"
STAGE_HALTED = "halted"
STAGE_SKIPPED = "skipped"
STAGE_FAILED = "failed"

# Where a finding's evidence came from: the static call graph (Change 2) or a runtime-discovered
# edge (Change 4). Both feed invariant evaluation; the source is recorded so a human can see whether
# a violation is a static-structure fact or one only runtime reality revealed.
SOURCE_STATIC = "static"
SOURCE_RUNTIME = "runtime"


@dataclass(frozen=True, slots=True)
class ArchInvariant:
    """A declared architectural invariant: a one-directional layering rule.

    An edge (static **or** runtime-discovered) from a node whose ``file_path`` starts with
    ``forbidden_caller_prefix`` to one whose ``file_path`` starts with ``forbidden_callee_prefix``
    violates the rule. Prefix matching keeps the rule expressible against OpenLore's repo-relative
    file paths without a dependency on a richer module system.
    """

    name: str
    forbidden_caller_prefix: str
    forbidden_callee_prefix: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "forbiddenCallerPrefix": self.forbidden_caller_prefix,
            "forbiddenCalleePrefix": self.forbidden_callee_prefix,
        }


@dataclass(frozen=True, slots=True)
class Intent:
    """What a run attempts.

    ``actions`` are v0-grammar command strings — the plan's verifying steps that get simulated
    (stage 2) and traced (stage 3). ``change`` is the concrete file edit the run prepares as its
    delivery — ``(repo-relative path, content)`` writes applied to the fixture in stage 7. Both may
    be empty (a pure inspection run prepares an empty delivery). ``seed`` pins any seeded choice;
    the v0 core is deterministic, so it is recorded for traceability and future seeded policies.
    """

    goal: str
    actions: tuple[str, ...] = ()
    invariants: tuple[ArchInvariant, ...] = ()
    change: tuple[tuple[str, str], ...] = ()
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "actions": list(self.actions),
            "invariants": [inv.to_dict() for inv in self.invariants],
            "change": [list(c) for c in self.change],
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class StageResult:
    """One stage's outcome: its name, status, a deterministic summary, and the relative path of the
    typed artifact it wrote (``None`` if the stage was skipped or wrote nothing)."""

    name: str
    status: str
    summary: dict[str, Any]
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "artifact": self.artifact,
        }


@dataclass(frozen=True, slots=True)
class InvariantFinding:
    """A violated architectural invariant: which rule, which edge, and where the edge came from."""

    invariant: str
    caller_id: str
    callee_id: str
    caller_file: str
    callee_file: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "callerId": self.caller_id,
            "calleeId": self.callee_id,
            "callerFile": self.caller_file,
            "calleeFile": self.callee_file,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PreparedDelivery:
    """The stage-7 output: a *prepared* (staged), unpushed change on the fixture.

    ``committed`` is ``True`` only when a human explicitly confirmed; ``commit_sha`` is then the
    new commit on the fixture (never pushed — the fixture has no remote). ``patch_artifact`` is
    the relative path of the unified diff of the staged change, always written so the change is
    auditable before any commit.
    """

    prepared: bool
    committed: bool
    commit_sha: str | None
    patch_artifact: str | None
    files_changed: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared": self.prepared,
            "committed": self.committed,
            "commitSha": self.commit_sha,
            "patchArtifact": self.patch_artifact,
            "filesChanged": list(self.files_changed),
        }


@dataclass(frozen=True, slots=True)
class RunReport:
    """The aggregate, content-sealed record of one headless CD run.

    ``signature`` is a sha256 over the report's deterministic content (every field but the signature
    itself) — the run's tamper-evident seal and its determinism witness: two runs of a fixed
    ``(fixture revision, intent, seed)`` produce the same signature.
    """

    version: str
    goal: str
    seed: int
    fixture_source_sha: str
    fixture_repo_tree_hash: str
    stages: tuple[StageResult, ...]
    halted: bool
    halt_reason: str | None
    breaker_status: str
    rollback_recommendation: dict[str, Any] | None
    findings: tuple[InvariantFinding, ...]
    feedback_edges: int
    feedback_dropped: int
    network_isolated: bool
    prepared_delivery: PreparedDelivery | None
    signature: str = ""

    def _content(self) -> dict[str, Any]:
        """The signed body — every field except the signature, in a stable shape."""
        return {
            "version": self.version,
            "goal": self.goal,
            "seed": self.seed,
            "fixtureSourceSha": self.fixture_source_sha,
            "fixtureRepoTreeHash": self.fixture_repo_tree_hash,
            "stages": [s.to_dict() for s in self.stages],
            "halted": self.halted,
            "haltReason": self.halt_reason,
            "breakerStatus": self.breaker_status,
            "rollbackRecommendation": self.rollback_recommendation,
            "findings": [f.to_dict() for f in self.findings],
            "feedbackEdges": self.feedback_edges,
            "feedbackDropped": self.feedback_dropped,
            "networkIsolated": self.network_isolated,
            "preparedDelivery": (
                None if self.prepared_delivery is None else self.prepared_delivery.to_dict()
            ),
        }

    def compute_signature(self) -> str:
        """The sha256 of the canonical signed body — stable for a fixed run."""
        body = json.dumps(self._content(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def signed(self) -> RunReport:
        """Return a copy with :attr:`signature` set to the content seal."""
        from dataclasses import replace

        return replace(self, signature=self.compute_signature())

    def to_dict(self) -> dict[str, Any]:
        body = self._content()
        body["signature"] = self.signature
        return body

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, stable separators) — byte-stable for a fixed run."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# A tiny accumulator the orchestrator threads through the stages; not part of the artifact surface.
@dataclass
class _Accumulator:
    stages: list[StageResult] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None
