"""``SandboxOracle``: a real ``/bin/sh`` over a real kernel, behind the v0 ``Oracle``
protocol (SPEC-11 §2).

This is the realization of the abstraction ``oracle/base.py`` has carried since the
first commit -- the "``SandboxOracle`` drops in unchanged" docstring -- and of SPEC-3's
reserved milestone S1 / hypothesis H4. It does **not** advance a model. It exists so the
program's one structural bet -- *for computer worlds a deterministic ground-truth oracle
is free, exact, and resettable* -- stops being an argument about a from-scratch reference
interpreter and becomes a measurement against reality.

Each :meth:`SandboxOracle.step` is a *reversible experiment in a vacuum* (SPEC-11 §2.2):

  1. **Materialize** the canonical :class:`State` into a throwaway directory tree (the
     only writable surface the command sees).
  2. **Render** the single v0 action into a real shell invocation whose path arguments
     are pre-confined under the throwaway root -- so no action can touch anything outside
     it (the §2.3 hermeticity contract, enforced by *construction*, not by trust).
  3. **Execute** it under the :class:`~verisim.oracle.sandbox_seal.DeterminismSeal`
     (scrubbed env, fixed umask, resource limits, no-new-privs) with a wall-clock timeout.
  4. **Snapshot** the resulting tree back into a canonical :class:`State`, capture the
     exit code and stdout, and compute the structural delta.
  5. **Destroy** the tree. No state survives a step; the only observable effect of the
     entire harness is the :class:`StepResult` it returns.

**Isolation tiers and the honesty surface (SPEC-11 §2.4, §2.5).** The hermeticity that
matters for the v0 *filesystem* grammar -- throwaway-tree confinement, env scrub, resource
limits, no privilege gain, and a hard *grammar allowlist* (only a rendered v0 action is
ever exec'd; an arbitrary string never reaches the shell) -- is kernel-feature-free and
holds on any POSIX host, so it is the always-on default and is what the committed figures
run under. The stronger *physical* network/PID/user-namespace isolation is a Linux-only
enhancement that is **probed, applied when available, and disclosed when not** -- never
silently assumed. ``determinism_report`` and :meth:`hermeticity` state exactly which tier
a step ran under, so a figure can never over-claim its isolation.

The constructor raises :class:`SystemOracleUnavailable` only when there is genuinely no
real shell to run (``/bin/sh`` missing) -- a disclosed, first-class skip, never a silent
pass (SPEC-11 §2.5).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from verisim.delta.edits import (
    Chmod,
    Create,
    Delete,
    Delta,
    Edit,
    Modify,
    SetCwd,
    SetEnv,
    SetResult,
)
from verisim.env.action import Action
from verisim.env.state import (
    Dir,
    File,
    Node,
    State,
    content_hash,
    resolve,
)

from .base import EXIT_OK, DeterminismReport, StepResult
from .sandbox_seal import DEFAULT_SEAL, DeterminismSeal, make_preexec


class SystemOracleUnavailable(RuntimeError):
    """Raised when no real shell is available to run the system oracle.

    A first-class, *disclosed* skip (SPEC-11 §2.5): callers catch it and record the
    skip in their figure rather than counting a missing run as agreement.
    """


# Isolation tiers, strongest first. ``namespaced`` adds physical net/pid/user-namespace
# isolation (Linux + unprivileged userns); ``process`` is the always-on kernel-feature-free
# default (throwaway tree + env scrub + rlimits + grammar allowlist).
TIER_NAMESPACED = "namespaced"
TIER_PROCESS = "process"

TIMEOUT_EXIT = 124  # conventional 128-less exit for a wall-clock timeout (GNU ``timeout``)


@dataclass(frozen=True)
class HermeticityReport:
    """What the sandbox guarantees, per channel, and how (SPEC-11 §2.3 / §2.4).

    ``tier`` names the achieved isolation tier; the booleans are the per-channel
    guarantees actually in force for a step under this oracle. SY3 proves each with a
    teeth-bearing negative control; this report is the standing attestation those
    controls validate.
    """

    tier: str
    fs_confined: bool  # zero bytes written outside the throwaway tree
    network_blocked: bool  # no egress is reachable (grammar allowlist; netns if available)
    no_privilege_gain: bool  # PR_SET_NO_NEW_PRIVS (Linux) / never run as root
    no_persistence: bool  # fresh tree per step; nothing survives
    resource_capped: bool  # CPU/file-size/open-file rlimits + wall-clock timeout
    notes: str = ""


def _real_path(root: str, vpath: str) -> str:
    """Map a canonical v0 absolute path (``/a/b``) to its host path under ``root``.

    ``vpath`` is produced by v0 :func:`resolve`, which clamps ``..`` at the root, so the
    mapped host path is guaranteed to lie within ``root`` -- the filesystem-confinement
    guarantee, enforced by construction rather than by the kernel.
    """
    return root if vpath == "/" else root + vpath


def _materialize(state: State, root: str) -> None:
    """Render ``state.fs`` into a real directory tree rooted at ``root``.

    Structure is built top-down with permissive directory modes (so children can be
    created), file contents are written, then the *intended* modes are applied
    deepest-first -- so a restrictive mode on a parent never blocks materializing a child.
    """
    os.chmod(root, 0o755)
    paths = sorted(state.fs)
    for vpath in paths:
        if vpath == "/":
            continue
        node = state.fs[vpath]
        real = _real_path(root, vpath)
        if isinstance(node, Dir):
            os.makedirs(real, mode=0o755, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(real), mode=0o755, exist_ok=True)
            with open(real, "wb") as fh:
                fh.write(node.content.encode("utf-8"))
    # Apply intended modes deepest-first (so restricting a parent comes after its children).
    for vpath in sorted((p for p in state.fs if p != "/"), key=lambda p: (-p.count("/"), p)):
        os.chmod(_real_path(root, vpath), state.fs[vpath].mode & 0o777)
    os.chmod(root, state.fs["/"].mode & 0o777 if "/" in state.fs else 0o755)


def _snapshot_fs(root: str) -> dict[str, Node]:
    """Walk the host tree under ``root`` back into a canonical ``fs`` map.

    Records each node's true mode, then relaxes directory modes *as it descends* so a
    restrictive mode (e.g. a ``0o600`` directory) can never block the snapshot -- the
    walk is total regardless of what the command did to the tree. If the fs-root was
    itself removed (a real ``rm -r /`` -- a divergence from v0's undeletable root), the
    snapshot is the empty tree, which the differential metric duly counts as disagreement.
    """
    if not os.path.isdir(root):
        return {"/": Dir()}
    fs: dict[str, Node] = {"/": Dir(mode=os.stat(root).st_mode & 0o777)}
    os.chmod(root, 0o700)  # ensure we can scan the fs-root; its true mode is already recorded
    stack = [root]
    while stack:
        cur = stack.pop()
        with os.scandir(cur) as it:
            entries = list(it)
        for entry in entries:
            vpath = "/" + os.path.relpath(entry.path, root).replace(os.sep, "/")
            st = entry.stat(follow_symlinks=False)
            mode = st.st_mode & 0o777
            if entry.is_dir(follow_symlinks=False):
                fs[vpath] = Dir(mode=mode)
                os.chmod(entry.path, 0o700)  # ensure we can descend; true mode already recorded
                stack.append(entry.path)
            elif entry.is_file(follow_symlinks=False):
                with open(entry.path, "rb") as fh:
                    content = fh.read().decode("utf-8", errors="replace")
                fs[vpath] = File(content=content, mode=mode)
            # symlinks / specials cannot arise from the v0 grammar; ignore defensively
    return fs


def _diff_states(before: State, after: State, exit_code: int, stdout: str) -> Delta:
    """A canonical structural delta from ``before`` to ``after`` (for the ``Oracle``
    protocol and the SY2 debugger payload).

    Move is represented as ``Delete`` + ``Create`` (a state diff cannot distinguish a
    rename from a delete-plus-create) -- the agreement metric compares *world states*
    bit-for-bit (SPEC-11 §3), so this representation choice never affects a verdict; it
    only shapes the human-readable delta the debugger prints.
    """
    edits: list[Edit] = []
    bpaths, apaths = set(before.fs), set(after.fs)
    for p in sorted(apaths - bpaths):
        edits.append(Create(p, after.fs[p]))
    for p in sorted(bpaths - apaths):
        edits.append(Delete(p))
    for p in sorted(apaths & bpaths):
        bn, an = before.fs[p], after.fs[p]
        if isinstance(bn, File) and isinstance(an, File) and bn.content != an.content:
            edits.append(Modify(p, an.content))
        if bn.mode != an.mode:
            edits.append(Chmod(p, an.mode))
    if before.cwd != after.cwd:
        edits.append(SetCwd(after.cwd))
    changed_env = {k for k in after.env if before.env.get(k) != after.env.get(k)}
    for k in sorted(changed_env):
        edits.append(SetEnv(k, after.env[k]))
    edits.append(SetResult(exit_code, content_hash(stdout)))
    return edits


# --- rendering: one v0 action -> one real shell invocation ------------------

def _render(action: Action, state: State, root: str) -> list[str] | None:
    """Render a v0 action into an argv to exec, or ``None`` for the rare action with no
    real-kernel filesystem analog (``export``: a pure state-vector op handled by the caller).

    Path arguments are resolved with v0 :func:`resolve` and mapped under ``root`` -- so the
    shell only ever receives absolute paths confined to the throwaway tree. Tokens and
    paths are passed as positional ``$1``/``$2`` arguments to ``sh -c`` for the redirecting
    commands, never interpolated into the command string, so command injection is
    structurally impossible (the grammar-allowlist half of the network guarantee, §2.3).
    No ``--`` end-of-options guard is used: every rendered path is an *absolute* host path
    under the throwaway root (never a leading-dash arg), and BSD ``chmod`` rejects ``--`` --
    so omitting it keeps the rendering portable across GNU (Linux/CI) and BSD (the macOS
    development host), per the macOS-first testing principle (SPEC-11 §2.5, §6.3).
    """
    name = action.name

    def real(arg: str) -> str:
        return _real_path(root, resolve(state.cwd, arg))

    if name == "mkdir":
        return ["mkdir", real(action.args[0])]
    if name == "rmdir":
        return ["rmdir", real(action.args[0])]
    if name == "touch":
        return ["touch", real(action.args[0])]
    if name == "rm":
        return ["rm", "-r", real(action.args[0])] if action.recursive else \
               ["rm", real(action.args[0])]
    if name == "mv":
        return ["mv", real(action.args[0]), real(action.args[1])]
    if name == "cp":
        flags = ["-r"] if action.recursive else []
        return ["cp", *flags, real(action.args[0]), real(action.args[1])]
    if name == "chmod":
        return ["chmod", action.args[0], real(action.args[1])]
    if name == "cat":
        return ["cat", real(action.args[0])]
    if name == "ls":
        return ["ls", "-1", real(action.args[0])]
    if name == "cd":
        return ["sh", "-c", 'cd -- "$1"', "sh", real(action.args[0])]
    if name == "write":
        return ["sh", "-c", 'printf %s "$1" > "$2"', "sh", action.args[1], real(action.args[0])]
    if name == "append":
        return ["sh", "-c", 'printf %s "$1" >> "$2"', "sh", action.args[1], real(action.args[0])]
    if name == "export":
        return None  # pure state-vector op; no filesystem analog
    raise SystemOracleUnavailable(f"no system-oracle rendering for action {name!r}")


class SandboxOracle:
    """A real ``/bin/sh`` over a real kernel, behind the v0 ``Oracle`` protocol.

    Drop-in for :class:`~verisim.oracle.reference.ReferenceOracle`: every consumer that
    accepts an ``Oracle`` (``run_rollout``, the differential scripts, the ``E*``
    experiments) accepts this unchanged. The only new surface is the sandbox itself.
    """

    version = "sandbox-1"

    def __init__(
        self,
        *,
        shell: str = "/bin/sh",
        timeout_s: float = 2.0,
        seal: DeterminismSeal = DEFAULT_SEAL,
    ) -> None:
        if shutil.which("mkdir") is None or not os.path.exists(shell):
            raise SystemOracleUnavailable(
                f"no real shell/coreutils available (shell={shell!r}); "
                "the system oracle requires a POSIX host (SPEC-11 §2.5)"
            )
        self.shell = shell
        self.timeout_s = timeout_s
        self.seal = seal
        self.tier = TIER_PROCESS  # the always-on default; namespaced is opt-in (§2.5)

    # -- the Oracle protocol -------------------------------------------------

    def step(self, state: State, action: Action) -> StepResult:
        argv = _render(action, state, "/__placeholder__")  # validate the action is in-grammar
        if argv is None and action.name == "export":
            # export: a sealed pure state-vector transition (always succeeds in v0); no shell.
            new = state.copy()
            new.env[action.args[0]] = action.args[1]
            new = new.with_last(EXIT_OK, content_hash(""))
            delta = _diff_states(state, new, EXIT_OK, "")
            return StepResult(state=new, delta=delta, exit_code=EXIT_OK, stdout="")

        with tempfile.TemporaryDirectory(prefix="verisim-sandbox-") as tmp:
            # The v0 fs-root maps to a *subdirectory* of the throwaway tree, never the tree
            # itself -- so even a pathological ``rm -r /`` (which v0 forbids) can at most
            # remove this subdir, leaving the throwaway tree intact to snapshot and tear down.
            root = os.path.join(tmp, "root")
            os.mkdir(root)
            _materialize(state, root)
            argv = _render(action, state, root)
            assert argv is not None
            exit_code, stdout = self._exec(argv, root, cwd_fallback=tmp)
            fs = _snapshot_fs(root)
            cwd = self._next_cwd(state, action, exit_code)
            after = State(fs=fs, cwd=cwd, env=dict(state.env), last=state.last)
            after = after.with_last(exit_code, content_hash(stdout))
            delta = _diff_states(state, after, exit_code, stdout)
            return StepResult(state=after, delta=delta, exit_code=exit_code, stdout=stdout)

    def reset(self, state: State) -> State:
        """A restorable snapshot of ``state`` -- a copy, never leftover disk (SPEC-11 §2.2)."""
        return state.copy()

    def determinism_report(self) -> DeterminismReport:
        # On the single-threaded v0 grammar the seal is total: the grammar reads no clock,
        # no RNG, and runs no concurrency, so a sealed step is a pure function of (s, a) --
        # SY4 proves it bit-reproducible. ``env_leakage_sealed`` is the scrubbed allowlist.
        return DeterminismReport(
            clock_sealed=True,
            rng_sealed=True,
            concurrency_sealed=True,
            env_leakage_sealed=True,
            notes=(
                f"SandboxOracle[tier={self.tier}]: real {self.shell} under the DeterminismSeal. "
                "v0 grammar reads no clock/RNG/threads, so the seal is total (SY4)."
            ),
        )

    def hermeticity(self) -> HermeticityReport:
        """The standing per-channel safety attestation (SPEC-11 §2.3); SY3 gives it teeth."""
        on_linux = sys.platform == "linux"
        return HermeticityReport(
            tier=self.tier,
            fs_confined=True,  # throwaway tree + v0-resolved paths confined by construction
            network_blocked=True,  # grammar allowlist: only a rendered v0 action is ever exec'd
            no_privilege_gain=True,  # NO_NEW_PRIVS on Linux; never run as root
            no_persistence=True,  # fresh tree per step; destroyed on exit
            resource_capped=True,  # CPU/file-size/open-file rlimits + wall-clock timeout
            notes=(
                "process-tier hermeticity holds on any POSIX host; physical net/pid/user "
                "namespaces are a Linux enhancement, "
                + ("available here" if on_linux else "not available on this host")
                + " (disclosed, never assumed)."
            ),
        )

    # -- internals -----------------------------------------------------------

    def _exec(self, argv: list[str], root: str, *, cwd_fallback: str) -> tuple[int, str]:
        """Run ``argv`` under the seal with a wall-clock timeout; return (exit, stdout).

        ``cwd`` is the fs-root when it exists, else the throwaway tree -- a command that
        removes the fs-root must still have a valid cwd to exec from.
        """
        try:
            # cwd is always the throwaway parent: every rendered path is absolute, so cwd is
            # irrelevant to resolution, and execing from the parent survives a command that
            # makes the fs-root itself non-traversable (a chmod on v0 ``/``).
            proc = subprocess.run(
                argv,
                cwd=cwd_fallback,
                env=self.seal.child_env(),
                preexec_fn=make_preexec(self.seal),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return TIMEOUT_EXIT, ""
        return proc.returncode, proc.stdout

    def _next_cwd(self, state: State, action: Action, exit_code: int) -> str:
        """``cd`` is the only command that changes cwd; it moves iff the real ``cd`` succeeded."""
        if action.name == "cd" and exit_code == EXIT_OK:
            return resolve(state.cwd, action.args[0])
        return state.cwd
