"""Read-only OpenLore call-graph adapter (OpenSpec ``add-openlore-graph-adapter``).

The second link in the Verisim ↔ OpenLore prototype chain (findings doc §9): let Verisim read
OpenLore's static call graph for a fixture as a typed, **read-only** substrate it can ground a
world model and architectural invariants on, treating the database as the regenerable cache it
actually is. The adapter never writes OpenLore's database, fails closed on an unsupported schema,
preserves every edge's provenance verbatim, and detects when a regenerated database has gone stale.

See :mod:`verisim.bridge.graph` for the typed read model and the SQLite loader, and
:mod:`verisim.bridge.analysis` for the OpenLore-authored analysis trigger and the supported-surface
cross-check.
"""

from __future__ import annotations

from .analysis import (
    CallGraphSummary,
    OpenLoreUnavailable,
    analyze_fixture,
    call_graph_summary_via_mcp,
    openlore_available,
)
from .graph import (
    ANALYSIS_DB_RELPATH,
    MAX_SCHEMA_VERSION,
    MIN_SCHEMA_VERSION,
    BridgeError,
    CallEdge,
    CfgEntry,
    CodeClass,
    CodeGraph,
    CodeNode,
    SchemaVersionError,
    analysis_db_path,
    load_code_graph,
)

__all__ = [
    "ANALYSIS_DB_RELPATH",
    "MAX_SCHEMA_VERSION",
    "MIN_SCHEMA_VERSION",
    "BridgeError",
    "CallEdge",
    "CallGraphSummary",
    "CfgEntry",
    "CodeClass",
    "CodeGraph",
    "CodeNode",
    "OpenLoreUnavailable",
    "SchemaVersionError",
    "analysis_db_path",
    "analyze_fixture",
    "call_graph_summary_via_mcp",
    "load_code_graph",
    "openlore_available",
]
