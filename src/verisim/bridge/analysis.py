"""OpenLore analysis trigger and the supported-surface cross-check (OpenSpec
``add-openlore-graph-adapter``).

Two narrow jobs that keep the call-graph read (in :mod:`verisim.bridge.graph`) honest:

  - **Analysis trigger** (:func:`analyze_fixture`) — ensure a fixture has a *fresh,
    OpenLore-authored* ``call-graph.db`` by invoking OpenLore's own analyzer (``openlore init`` +
    ``openlore analyze``). The database is OpenLore-authored, never Verisim-authored: Verisim only
    *reads* it. If the ``openlore`` CLI is absent the trigger raises
    :class:`OpenLoreUnavailable` — a first-class, disclosed failure (the fixture module's
    ``GitUnavailable`` discipline), never a silent half-analysis.

  - **Supported-surface cross-check** (:func:`call_graph_summary_via_mcp`) — read the call graph's
    *aggregate summary* (node/edge counts, entry points, hubs) through OpenLore's **supported MCP
    contract** (``get_call_graph`` over the stdio MCP server), independently of the raw-SQL read.
    The full provenance-bearing edge set lives only in the SQLite ``edges`` table, so the SQLite
    read is canonical; this cross-check confirms that canonical read agrees with the supported
    contract on the aggregates the contract *does* expose — the honest, achievable form of the
    proposal's MCP/SQL "parity" (a raw-SQL read that silently diverged from the supported surface
    would be exactly the schema-drift hazard the schema guard defends against).
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


def _mcp_request_summary(repo: Path) -> dict[str, object]:
    """Speak the minimal MCP stdio handshake to OpenLore and return parsed ``get_call_graph`` JSON.

    Spawns ``openlore mcp`` (stdio transport), performs the initialize handshake, calls the
    ``get_call_graph`` tool for ``repo``, and parses the tool's text payload. Raises
    :class:`OpenLoreUnavailable` if the CLI is absent and :class:`BridgeError` on a protocol error.
    """
    try:
        proc = subprocess.Popen(
            [OPENLORE_BIN, "mcp"],
            cwd=str(repo),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError as exc:
        raise OpenLoreUnavailable("openlore executable not found on PATH") from exc

    def send(obj: dict[str, object]) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "verisim-bridge", "version": "0"},
                },
            }
        )
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "get_call_graph", "arguments": {"directory": str(repo)}},
            }
        )
        # Read responses line by line: the MCP server is long-lived (it does not exit after a call),
        # so we read until the id=2 result arrives rather than waiting for the process to terminate.
        # ``select`` bounds the wait so a silent server cannot hang the caller (POSIX; the prototype
        # is macOS/Linux only).
        call_result = _read_until_result(proc, request_id=2, deadline_s=_MCP_TIMEOUT_S)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if call_result is None:
        raise BridgeError("openlore mcp returned no get_call_graph result")

    try:
        result = call_result["result"]
        assert isinstance(result, dict)
        content = result["content"]
        assert isinstance(content, list)
        first = content[0]
        assert isinstance(first, dict)
        parsed = json.loads(first["text"])
    except (KeyError, IndexError, AssertionError, TypeError, json.JSONDecodeError) as exc:
        raise BridgeError(f"unexpected get_call_graph MCP payload: {call_result}") from exc
    if not isinstance(parsed, dict):
        raise BridgeError("get_call_graph payload was not a JSON object")
    return parsed


def call_graph_summary_via_mcp(repo_path: str | Path) -> CallGraphSummary:
    """The call-graph aggregate summary via OpenLore's supported MCP surface (``get_call_graph``).

    Used to cross-check the canonical SQLite read against the supported contract on the aggregates
    the contract exposes (node/edge counts, entry points). Raises :class:`OpenLoreUnavailable` if
    the CLI is absent and :class:`BridgeError` on a protocol error.
    """
    repo = Path(repo_path)
    payload = _mcp_request_summary(repo)

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
