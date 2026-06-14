"""OpenLore analysis trigger and the supported-surface cross-check (OpenSpec
``add-openlore-graph-adapter``).

Two narrow jobs that keep the call-graph read (in :mod:`verisim.bridge.graph`) honest:

  - **Analysis trigger** (:func:`analyze_fixture`) — ensure a fixture has a *fresh,
    OpenLore-authored* ``call-graph.db`` by invoking OpenLore's own analyzer (``openlore init`` +
    ``openlore analyze``). The database is OpenLore-authored, never Verisim-authored: Verisim only
    *reads* it. If the ``openlore`` CLI is absent the trigger raises
    :class:`OpenLoreUnavailable` — a first-class, disclosed failure (the fixture module's
    ``GitUnavailable`` discipline), never a silent half-analysis.

  - **Supported-surface cross-check** — confirm the canonical SQLite read agrees with what
    OpenLore's MCP contract serves, independently of the raw-SQL read (a raw read that silently
    diverged from the supported surface would be the schema-drift hazard the guard defends against).
    The MCP surface has two faces, both verified against the latest OpenLore (2.0.18):

      * :func:`subgraph_via_mcp` — the **real call graph**. ``get_subgraph`` returns actual
        per-edge ``caller``/``callee``/``kind``/``callType`` relationships (not an aggregate). It
        reads an in-memory graph index that an OpenLore *version upgrade resets*, so the call must
        first **(re)build the index e2e** via the ``analyze_codebase`` tool in the same MCP session
        — exactly the "run it to generate" step. This is the strong parity: the real MCP edges
        equal the SQLite read's internal edges.
      * :func:`call_graph_summary_via_mcp` — the **aggregate summary**. ``get_call_graph`` returns
        counts/hubs/entry-points (a *derived* view: its ``total_edges`` is an expanded count, not
        the raw table size), used for the cheap count-level invariants.

    Neither MCP face carries the per-edge **provenance** (``confidence``/``synthesized_by``) — that
    lives only in the SQLite ``edges`` table — so the SQLite read remains the canonical loader and
    the MCP surface is a cross-check, not a replacement.
"""

from __future__ import annotations

import json
import select
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .graph import BridgeError, analysis_db_path, load_code_graph

# The OpenLore CLI executable. Resolved on PATH; absence is a disclosed failure, not a silent skip.
OPENLORE_BIN = "openlore"

# The MCP stdio handshake OpenLore speaks (protocol 2024-11-05). Kept minimal: initialize,
# notify-initialized, one tools/call, done.
_MCP_PROTOCOL_VERSION = "2024-11-05"
_MCP_TIMEOUT_S = 60.0


class OpenLoreUnavailable(BridgeError):
    """No ``openlore`` CLI is available to author or cross-check an analysis.

    The analogue of the fixture module's ``GitUnavailable``: the trigger cannot guarantee an
    OpenLore-authored database without the CLI, so it refuses rather than fabricating one.
    """


@dataclass(frozen=True, slots=True)
class CallGraphSummary:
    """The aggregate call-graph summary from OpenLore's supported MCP surface (not the full graph).

    The surface reports a *derived* view, so only some of its aggregates are clean parity invariants
    against the raw SQLite read (verified empirically):

      - ``total_nodes`` counts the codebase's **internal** functions (``is_external`` false) — it
        equals ``len(CodeGraph.internal_nodes())``.
      - ``entry_point_count`` equals the number of internal entry-point nodes.
      - ``total_edges`` is the surface's own expanded call count and does **not** equal the raw
        ``edges`` table size (the surface counts call relationships differently); it is recorded for
        traceability, not asserted. This divergence is exactly why the provenance-bearing graph must
        be read from SQLite rather than reconstructed from the summary — the schema guard's premise.
    """

    total_nodes: int
    total_edges: int
    entry_point_count: int
    hub_count: int


@dataclass(frozen=True, slots=True)
class McpEdge:
    """A single real call edge from OpenLore's ``get_subgraph`` MCP surface.

    Carries the actual call relationship (caller/callee plus their files and the call kind/type) but
    **not** the ``confidence``/``synthesized_by`` provenance — that is SQLite-only. ``get_subgraph``
    reports only internal endpoints (external call targets are excluded, matching the summary's
    internal-node count).
    """

    caller: str
    callee: str
    caller_file: str
    callee_file: str
    kind: str | None
    call_type: str | None

    def as_pair(self) -> tuple[str, str, str, str]:
        """The ``(caller_file, caller, callee_file, callee)`` identity, for set comparison with the
        SQLite read's internal edges (``CodeGraph.internal_edges_named``)."""
        return (self.caller_file, self.caller, self.callee_file, self.callee)


def openlore_available() -> bool:
    """True iff the ``openlore`` CLI is resolvable on PATH."""
    return shutil.which(OPENLORE_BIN) is not None


def _run_openlore(
    repo_path: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run ``openlore <args>`` with ``repo_path`` as the working directory.

    Raises :class:`OpenLoreUnavailable` if the CLI is absent and :class:`BridgeError` on a non-zero
    exit when ``check``.
    """
    try:
        proc = subprocess.run(
            [OPENLORE_BIN, *args],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise OpenLoreUnavailable("openlore executable not found on PATH") from exc
    if check and proc.returncode != 0:
        raise BridgeError(
            f"openlore {' '.join(args)} failed in {repo_path} (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc


def analyze_fixture(repo_path: str | Path, *, embed: bool = False) -> Path:
    """Run OpenLore's analyzer over a fixture repo and return the path to its ``call-graph.db``.

    Runs ``openlore init`` (idempotent — an existing config is fine) then ``openlore analyze
    --force`` so the database reflects the current working tree. Asserts the database was produced
    and is readable under the supported schema (via :func:`load_code_graph`, which fails closed on
    an unsupported version). ``embed=False`` (the default) builds a keyword-only index — no
    embedding service required, so the trigger is hermetic.

    Raises :class:`OpenLoreUnavailable` if the CLI is absent and :class:`BridgeError` on any
    analysis or read failure.
    """
    repo = Path(repo_path)
    if not repo.exists():
        raise BridgeError(f"fixture repo does not exist: {repo}")

    # ``init`` refuses with a non-zero exit when config already exists; that is success for us.
    init = _run_openlore(repo, "init", check=False)
    combined = (init.stdout + init.stderr).lower()
    if init.returncode != 0 and "exists" not in combined:
        raise BridgeError(f"openlore init failed in {repo}: {init.stderr.strip()}")

    analyze_args = ["analyze", "--force", "--no-embed"] if not embed else ["analyze", "--force"]
    _run_openlore(repo, *analyze_args)

    db = analysis_db_path(repo)
    if not db.exists():
        raise BridgeError(f"openlore analyze produced no database at {db}")
    # Validate it loads under the supported schema (fails closed otherwise); discard the graph.
    load_code_graph(db)
    return db


def _read_until_result(
    proc: subprocess.Popen[str], *, request_id: int, deadline_s: float
) -> dict[str, object] | None:
    """Read JSON-RPC lines from ``proc.stdout`` until the response to ``request_id`` arrives.

    ``select`` bounds each read so a silent server cannot hang the caller; returns the matching
    response message or ``None`` on timeout / EOF.
    """
    assert proc.stdout is not None
    deadline = time.monotonic() + deadline_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            return None
        line = proc.stdout.readline()
        if not line:  # EOF — server exited
            return None
        stripped = line.strip()
        if not stripped:
            continue
        try:
            msg = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and msg.get("id") == request_id and "result" in msg:
            return msg


class _McpSession:
    """A minimal stdio MCP session against ``openlore mcp``: initialize once, then call tools.

    Spawns one server process, performs the initialize handshake, and lets the caller issue several
    ``tools/call`` requests on the same connection (the index ``analyze_codebase`` builds is held by
    the server process, so the build and the query that uses it must share a session). Use as a
    context manager so the process is always reaped. ``select`` bounds every read so a silent server
    cannot hang the caller (POSIX; the prototype is macOS/Linux only).
    """

    def __init__(self, repo: Path) -> None:
        try:
            self._proc: subprocess.Popen[str] = subprocess.Popen(
                [OPENLORE_BIN, "mcp", "--no-watch-auto"],
                cwd=str(repo),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError as exc:
            raise OpenLoreUnavailable("openlore executable not found on PATH") from exc
        self._next_id = 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self._take_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "verisim-bridge", "version": "0"},
                },
            }
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def __enter__(self) -> _McpSession:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _take_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send(self, obj: dict[str, object]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Call MCP tool ``name`` and return the parsed JSON payload of its text result."""
        rid = self._take_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": rid,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        msg = _read_until_result(self._proc, request_id=rid, deadline_s=_MCP_TIMEOUT_S)
        if msg is None:
            raise BridgeError(f"openlore mcp returned no result for {name}")
        try:
            result = msg["result"]
            assert isinstance(result, dict)
            content = result["content"]
            assert isinstance(content, list)
            first = content[0]
            assert isinstance(first, dict)
            parsed = json.loads(first["text"])
        except (KeyError, IndexError, AssertionError, TypeError, json.JSONDecodeError) as exc:
            raise BridgeError(f"unexpected MCP payload for {name}: {msg}") from exc
        if not isinstance(parsed, dict):
            raise BridgeError(f"MCP payload for {name} was not a JSON object")
        return parsed

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def call_graph_summary_via_mcp(repo_path: str | Path) -> CallGraphSummary:
    """The call-graph aggregate summary via OpenLore's supported MCP surface (``get_call_graph``).

    Used to cross-check the canonical SQLite read against the supported contract on the aggregates
    the contract exposes (internal-node/entry-point counts). Raises :class:`OpenLoreUnavailable` if
    the CLI is absent and :class:`BridgeError` on a protocol error.
    """
    repo = Path(repo_path)
    with _McpSession(repo) as session:
        payload = session.call("get_call_graph", {"directory": str(repo)})

    stats_obj = payload.get("stats", {})
    stats: dict[str, object] = stats_obj if isinstance(stats_obj, dict) else {}
    entries_obj = payload.get("entryPoints", [])
    entries = entries_obj if isinstance(entries_obj, list) else []
    hubs_obj = payload.get("hubFunctions", [])
    hubs = hubs_obj if isinstance(hubs_obj, list) else []

    return CallGraphSummary(
        total_nodes=_as_int(stats.get("totalNodes")),
        total_edges=_as_int(stats.get("totalEdges")),
        entry_point_count=len(entries),
        hub_count=len(hubs),
    )


def subgraph_via_mcp(
    repo_path: str | Path,
    function_name: str,
    *,
    direction: str = "both",
    max_depth: int = 1,
    build_index: bool = True,
) -> tuple[McpEdge, ...]:
    """The **real call graph** around ``function_name`` via OpenLore's ``get_subgraph`` MCP surface.

    Returns the actual call edges (caller/callee/kind/callType) — not an aggregate. Because an
    OpenLore version upgrade resets the in-memory graph index ``get_subgraph`` reads,
    ``build_index`` (default) first (re)builds it e2e via the ``analyze_codebase`` tool **in the
    same session**; pass ``build_index=False`` only when a fresh index is known to exist. Raises
    :class:`OpenLoreUnavailable` if the CLI is absent and :class:`BridgeError` on a protocol error.
    """
    repo = Path(repo_path)
    with _McpSession(repo) as session:
        if build_index:
            session.call("analyze_codebase", {"directory": str(repo), "force": True})
        payload = session.call(
            "get_subgraph",
            {
                "directory": str(repo),
                "functionName": function_name,
                "direction": direction,
                "maxDepth": max_depth,
            },
        )

    if "error" in payload:
        raise BridgeError(f"get_subgraph failed for {function_name!r}: {payload['error']}")
    edges_obj = payload.get("edges", [])
    edges = edges_obj if isinstance(edges_obj, list) else []
    out: list[McpEdge] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        out.append(
            McpEdge(
                caller=str(e.get("caller", "")),
                callee=str(e.get("callee", "")),
                caller_file=str(e.get("callerFile", "")),
                callee_file=str(e.get("calleeFile", "")),
                kind=_opt_str(e.get("kind")),
                call_type=_opt_str(e.get("callType")),
            )
        )
    return tuple(out)


def _opt_str(value: object) -> str | None:
    """Coerce a JSON value to ``str | None`` (a missing field stays ``None``)."""
    return None if value is None else str(value)


def _as_int(value: object) -> int:
    """Coerce a JSON value to int, defaulting to 0 for missing/non-numeric (best-effort)."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
