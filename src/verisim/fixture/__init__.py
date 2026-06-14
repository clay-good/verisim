"""De-fanged real-codebase fixture (OpenSpec ``add-defanged-codebase-fixture``).

The first link in the Verisim ↔ OpenLore prototype chain (findings doc §9): a real local
repository, copied into Verisim-owned scratch and **de-fanged by construction** so the
downstream human-gated CD loop can operate on real structure without any path back to the
original's history or remotes. Isolation and inertness are properties of how the fixture is
built — mirroring SPEC-11's hermeticity-by-construction — not of trusting later code.

See :mod:`verisim.fixture.materialize` for the contract and the selection criteria.
"""

from __future__ import annotations

from .materialize import (
    DEFAULT_EXCLUDE,
    DEFAULT_SOURCE_ROOT,
    Fixture,
    FixtureConfig,
    FixtureError,
    FixtureManifest,
    GitUnavailable,
    load_manifest,
    materialize,
    remotes,
    teardown,
    tree_hash,
    validate_source,
)

__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_SOURCE_ROOT",
    "Fixture",
    "FixtureConfig",
    "FixtureError",
    "FixtureManifest",
    "GitUnavailable",
    "load_manifest",
    "materialize",
    "remotes",
    "teardown",
    "tree_hash",
    "validate_source",
]
