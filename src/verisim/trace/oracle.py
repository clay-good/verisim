"""``TracingOracle``: an observational decorator that records a :class:`RuntimeTrace` per step
(OpenSpec ``add-sandbox-trace-oracle``).

A drop-in wrapper over *any* :class:`~verisim.oracle.base.Oracle` (the ``SandboxOracle`` in
particular): every consumer that accepts an ``Oracle`` accepts this unchanged, and the wrapped
oracle's semantics are untouched. ``step`` calls the inner oracle, returns its
:class:`~verisim.oracle.base.StepResult` **verbatim**, and on the side records a trace of the
step's real effects via a :class:`~verisim.trace.tracer.Tracer`. Because the trace is built purely
from the action and the result — never by perturbing the execution, the env scrub, the umask, or
the resource limits — a traced step and an untraced step produce an identical canonical state (the
``DeterminismPreservedUnderTracing`` requirement, met by construction).

Tracing is additive and removable: drop the wrapper and the behavior is exactly the inner oracle's.
The tracer's own overhead is measured and bounded (``overhead_budget_s``): a step whose tracing
overruns the budget *fails loudly* (:class:`TraceBudgetExceeded`) rather than silently eating into
the sandbox's wall-clock budget.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from verisim.env.action import Action
from verisim.env.state import State
from verisim.fixture import DEFAULT_SOURCE_ROOT
from verisim.oracle.base import DeterminismReport, Oracle, StepResult

from .model import RuntimeTrace
from .tracer import Tracer, select_tracer


class TraceError(RuntimeError):
    """A trace could not be recorded or written safely."""


class TraceBudgetExceeded(TraceError):
    """Tracing overhead exceeded its budget — fail the step rather than overrun silently."""


class TracingOracle:
    """An :class:`Oracle` decorator that records one :class:`RuntimeTrace` per step.

    ``fixture_source_sha`` (e.g. ``Fixture.manifest.source_head_sha``) stamps every trace with the
    source revision it pertains to. If ``artifact_dir`` is set, each trace is also written there as
    a typed, versioned JSON file (Verisim-owned scratch only). Traces are retained in
    :attr:`traces` for in-process consumers (Change 4).
    """

    def __init__(
        self,
        inner: Oracle,
        *,
        tracer: Tracer | None = None,
        fixture_source_sha: str | None = None,
        artifact_dir: str | Path | None = None,
        overhead_budget_s: float | None = 1.0,
    ) -> None:
        self._inner = inner
        self._tracer = tracer if tracer is not None else select_tracer()
        self._fixture_source_sha = fixture_source_sha
        self._artifact_dir = None if artifact_dir is None else Path(artifact_dir)
        self._overhead_budget_s = overhead_budget_s
        self.traces: list[RuntimeTrace] = []
        self._counter = 0

    # -- the Oracle protocol (step is wrapped; the rest delegates) ------------

    def step(self, state: State, action: Action) -> StepResult:
        """Run the inner step, record its trace, and return the result **unchanged**."""
        self._tracer.begin()
        t0 = time.perf_counter()
        result = self._inner.step(state, action)
        elapsed_s = time.perf_counter() - t0

        overhead_start = time.perf_counter()
        trace = self._tracer.finish(
            action=action,
            before=state,
            result=result,
            fixture_source_sha=self._fixture_source_sha,
            elapsed_s=elapsed_s,
        )
        overhead_s = time.perf_counter() - overhead_start
        if self._overhead_budget_s is not None and overhead_s > self._overhead_budget_s:
            raise TraceBudgetExceeded(
                f"tracing overhead {overhead_s:.3f}s exceeded budget "
                f"{self._overhead_budget_s:.3f}s for action {action.name!r}"
            )

        self.traces.append(trace)
        if self._artifact_dir is not None:
            write_trace(trace, self._artifact_dir, index=self._counter)
        self._counter += 1
        return result

    def reset(self, state: State) -> State:
        return self._inner.reset(state)

    def determinism_report(self) -> DeterminismReport:
        return self._inner.determinism_report()

    def __getattr__(self, name: str) -> Any:
        """Delegate any non-overridden attribute (e.g. ``hermeticity``, ``version``) to the inner
        oracle, so the wrapper stays a faithful drop-in."""
        return getattr(self._inner, name)


def write_trace(trace: RuntimeTrace, artifact_dir: str | Path, *, index: int) -> Path:
    """Write ``trace`` as a typed, versioned JSON file under ``artifact_dir`` (scratch only).

    Refuses to write inside the source-roots allowlist (the fixture module's isolation discipline):
    traces are Verisim-owned artifacts, never written into a repo the user cares about. Returns the
    written path.
    """
    out_dir = Path(artifact_dir)
    resolved = out_dir.resolve()
    root = DEFAULT_SOURCE_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise TraceError(
            f"refusing to write traces under a source root: {resolved} (traces are scratch-only)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trace-{index:04d}-{trace.action_name}.json"
    path.write_text(trace.to_json(), encoding="utf-8")
    return path
