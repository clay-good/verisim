"""The differential-validation harness (SPEC-11 §3): run the reference oracle and the
system oracle on the same ``(state, action)`` and return an exact agreement record.

This is the core of SPEC-11 and the discipline around it. ``differential_step`` calls
*both* oracles on the identical transition and compares them on three bit-exact channels:

  - **world** -- the filesystem + cwd + env, compared via the existing v0 canonical
    serialization (the same equality the M1/NW1 invariants use), with the ``last``
    observation excluded so the world channel is orthogonal to the exit/stdout channels;
  - **exit** -- the process exit code;
  - **stdout** -- the command's stdout (read commands; trailing-newline normalized, the
    one disclosed normalization, since ``ls`` formats with a trailing newline the v0
    string form omits).

When the world or exit channels disagree, :func:`classify_divergence` localizes the cause
into one of the *named v0 modeling boundaries* (SPEC-11 §9 honest caveats) -- root
protection, overwrite policy, permission enforcement -- or flags it ``residual`` for the
SY2 debugger to inspect. A divergence is never silent: it is either a known, documented
boundary or a first-class finding.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisim.env.action import Action
from verisim.env.serialize import to_canonical
from verisim.env.state import Dir, State, basename, resolve

from .base import Oracle, StepResult

# Named v0 modeling boundaries -- the places v0's reference semantics *deliberately*
# differ from a raw POSIX kernel (documented in docs/semantics.md, surfaced as a
# committed catalog by SY2). Plus the two non-divergent verdicts.
# Named v0 modeling boundaries -- the places v0's reference semantics deliberately differ
# from a raw POSIX kernel (see each constant; all documented in docs/semantics.md):
#   root_protection        -- v0 makes ``/`` undeletable/uncopyable; the kernel does not
#   overwrite_policy       -- v0 mv/cp refuse to clobber a target; the kernel clobbers
#   permission_enforcement -- v0 models mode as data; the kernel enforces traversal/access
#   self_subtree           -- mv/cp a dir into its own subtree: GNU+v0 reject, BSD cp does not
AGREE = "agree"
C_ROOT = "root_protection"
C_OVERWRITE = "overwrite_policy"
C_PERMISSION = "permission_enforcement"
C_SELF_SUBTREE = "self_subtree"
RESIDUAL = "residual"  # an unexplained disagreement -- a first-class finding for SY2

BOUNDARY_CLASSES = (C_ROOT, C_OVERWRITE, C_PERMISSION, C_SELF_SUBTREE)


def canonical_world(state: State) -> str:
    """The canonical world string (fs + cwd + env), excluding the ``last`` observation.

    Identical to :func:`verisim.env.serialize.to_canonical` minus ``last``, so the world
    channel is bit-exact yet orthogonal to the exit/stdout channels compared separately.
    """
    d = to_canonical(state)
    d.pop("last", None)
    return repr(d)


@dataclass(frozen=True)
class DiffRecord:
    """An exact agreement record for one ``(state, action)`` transition (SPEC-11 §3)."""

    action_raw: str
    command: str
    agree_world: bool
    agree_exit: bool
    agree_stdout: bool
    divergence_class: str
    ref_exit: int
    sys_exit: int

    @property
    def agree(self) -> bool:
        """Full agreement on the world *and* exit channels (the headline relation)."""
        return self.agree_world and self.agree_exit


def differential_step(state: State, action: Action, ref: Oracle, sys: Oracle) -> DiffRecord:
    """Run ``ref`` and ``sys`` on the same ``(state, action)``; return the agreement record."""
    r_ref: StepResult = ref.step(state, action)
    r_sys: StepResult = sys.step(state, action)
    agree_world = canonical_world(r_ref.state) == canonical_world(r_sys.state)
    agree_exit = r_ref.exit_code == r_sys.exit_code
    agree_stdout = r_ref.stdout.rstrip("\n") == r_sys.stdout.rstrip("\n")
    cls = AGREE if (agree_world and agree_exit) else classify_divergence(state, action)
    return DiffRecord(
        action_raw=action.raw,
        command=action.name,
        agree_world=agree_world,
        agree_exit=agree_exit,
        agree_stdout=agree_stdout,
        divergence_class=cls,
        ref_exit=r_ref.exit_code,
        sys_exit=r_sys.exit_code,
    )


def _chain(vpath: str) -> list[str]:
    """All ancestor directories of ``vpath`` including the root ``/`` (parent-most first)."""
    out = ["/"]
    cur = ""
    for seg in vpath.strip("/").split("/"):
        if not seg:
            continue
        cur += "/" + seg
        out.append(cur)
    return out


def _blocked_by_mode(state: State, vpath: str, *, include_self: bool) -> bool:
    """True if traversing to ``vpath`` crosses a directory whose mode lacks owner-execute.

    The real kernel denies traversal through a non-``x`` directory; v0 ignores mode for
    access (it models mode as *data*, not access control), so such transitions diverge.
    ``include_self`` covers commands that must access the target directory itself (``cd``,
    ``ls``, ``rmdir``), not merely traverse to its parent.
    """
    chain = _chain(vpath) if include_self else _chain(vpath)[:-1] or ["/"]
    for anc in chain:
        node = state.fs.get(anc)
        if isinstance(node, Dir) and not (node.mode & 0o100):
            return True
    return False


def classify_divergence(state: State, action: Action) -> str:
    """Localize a world/exit disagreement to a named v0 modeling boundary (or ``residual``).

    A pure function of the *input* ``(state, action)`` -- it inspects only what both oracles
    saw, never their outputs, so the classification is itself reproducible and auditable.
    """
    name = action.name
    if name == "chmod":
        targets = [resolve(state.cwd, action.args[1])]
    elif name in ("mv", "cp"):
        targets = [resolve(state.cwd, action.args[0]), resolve(state.cwd, action.args[1])]
    elif name == "export":
        return RESIDUAL  # export has no kernel analog; it should never disagree
    else:
        targets = [resolve(state.cwd, action.args[0])] if action.args else []

    if any(t == "/" for t in targets):
        return C_ROOT
    needs_self = name in ("cd", "ls", "rmdir", "chmod")
    if any(_blocked_by_mode(state, t, include_self=needs_self) for t in targets):
        return C_PERMISSION
    if name in ("mv", "cp"):
        src = resolve(state.cwd, action.args[0])
        dst = resolve(state.cwd, action.args[1])
        final = resolve(dst, basename(src)) if isinstance(state.fs.get(dst), Dir) else dst
        # A directory copied/moved into its own subtree: v0 (post the SPEC-11 fix) and GNU
        # both reject it; BSD ``cp`` on macOS does not detect it -- a disclosed coreutils
        # portability boundary, agreeing on Linux/CI, diverging on the macOS dev host.
        if isinstance(state.fs.get(src), Dir) and final.startswith(src + "/"):
            return C_SELF_SUBTREE
        if final in state.fs:
            return C_OVERWRITE
    return RESIDUAL
