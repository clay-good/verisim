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
    McpEdge,
    OpenLoreUnavailable,
    analyze_fixture,
    call_graph_summary_via_mcp,
    openlore_available,
    subgraph_via_mcp,
)
from .feedback import (
    EDGE_CONFIDENCE,
    EDGE_KIND,
    FEEDBACK_SCHEMA_VERSION,
    SYNTHESIZED_BY,
    CandidateEdge,
    FeedbackError,
    FeedbackPayload,
    FeedbackValidationError,
    RuntimeEvidence,
    build_feedback_payload,
    detect_candidates,
    validate_payload,
    write_feedback,
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
    "EDGE_CONFIDENCE",
    "EDGE_KIND",
    "FEEDBACK_SCHEMA_VERSION",
    "MAX_SCHEMA_VERSION",
    "MIN_SCHEMA_VERSION",
    "SYNTHESIZED_BY",
    "BridgeError",
    "CallEdge",
    "CallGraphSummary",
    "CandidateEdge",
    "CfgEntry",
    "CodeClass",
    "CodeGraph",
    "CodeNode",
    "FeedbackError",
    "FeedbackPayload",
    "FeedbackValidationError",
    "McpEdge",
    "OpenLoreUnavailable",
    "RuntimeEvidence",
    "SchemaVersionError",
    "analysis_db_path",
    "analyze_fixture",
    "build_feedback_payload",
    "call_graph_summary_via_mcp",
    "detect_candidates",
    "load_code_graph",
    "openlore_available",
    "subgraph_via_mcp",
    "validate_payload",
    "write_feedback",
]
