"""The ``DeterminismSeal``: the per-step nondeterminism seal for the system oracle
(SPEC-11 §2.4, the honesty surface).

The reference oracle seals every nondeterminism source *by construction* -- it is a
pure function of ``(state, action)``. The system oracle runs a real ``/bin/sh`` over a
real kernel and therefore *cannot* assume that; it must actively seal what it can and
**disclose** the rest. This module is that seal made explicit and inspectable:

  - the fixed, scrubbed environment a step is allowed to inherit (``TZ``, ``LC_ALL``,
    ``LANG``, ``PATH`` only -- everything else is dropped, the env-leakage seal);
  - a frozen wall clock (``SOURCE_DATE_EPOCH``) so the few clock-reading commands a
    future grammar might add are reproducible rather than live;
  - the per-step ``preexec`` hook that fixes ``umask`` (so created-file/dir modes are
    deterministic, not host-umask-dependent), applies resource limits (no host
    exhaustion), and -- on Linux -- sets ``PR_SET_NO_NEW_PRIVS`` (no privilege gain).

For the single-threaded v0 grammar the seal is *total* and SY4 proves it: the grammar's
semantics read no clock, no RNG, and run no concurrency, so a sealed step is a pure
function of ``(state, action)`` exactly as the reference is. The value of stating the
seal explicitly is forward-looking: the moment the grammar grows a clock/RNG/thread
dependence, the relevant flag flips to ``recorded`` and every figure inherits the
disclosure automatically (SPEC-11 §2.4).
"""

from __future__ import annotations

import os
import resource
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

# A frozen epoch (2026-06-05T00:00:00Z) -- the v0 grammar reads no clock, so this is a
# forward-looking seal: any future clock-reading command becomes reproducible, not live.
FROZEN_EPOCH = "1780963200"

# The only environment a sealed step inherits. Everything else is dropped, so host
# environment cannot leak into a transition (the env-leakage seal, SPEC-11 §2.3).
_BASE_ENV: dict[str, str] = {
    "PATH": "/usr/bin:/bin",
    "TZ": "UTC",
    "LC_ALL": "C",
    "LANG": "C",
    "SOURCE_DATE_EPOCH": FROZEN_EPOCH,
}

# Deterministic file/dir creation modes depend on a fixed umask. ``022`` makes a real
# ``touch`` create ``0644`` and a real ``mkdir`` create ``0755`` -- exactly the v0
# ``File``/``Dir`` defaults, so created-node modes agree by construction.
DEFAULT_UMASK = 0o022


@dataclass(frozen=True)
class DeterminismSeal:
    """The declarative seal applied to every system-oracle step.

    ``cpu_seconds`` / ``file_size_bytes`` / ``open_files`` are resource ceilings (no host
    exhaustion); ``umask`` fixes created-node modes; ``env`` is the scrubbed allowlist.
    All fields are data -- :func:`apply_seal` turns them into the actual environment and
    the actual ``preexec`` closure, so the seal a figure ran under is fully inspectable.
    """

    umask: int = DEFAULT_UMASK
    cpu_seconds: int = 4
    file_size_bytes: int = 8 * 1024 * 1024
    open_files: int = 256
    env: dict[str, str] = field(default_factory=lambda: dict(_BASE_ENV))

    def child_env(self) -> dict[str, str]:
        """The exact environment the child ``/bin/sh`` inherits -- the scrubbed allowlist."""
        return dict(self.env)


DEFAULT_SEAL = DeterminismSeal()


def _set_no_new_privs() -> None:
    """Best-effort ``prctl(PR_SET_NO_NEW_PRIVS, 1)`` on Linux (no privilege escalation).

    A no-op (disclosed via the determinism report) off Linux or if ``prctl`` is
    unavailable. Setting it means no ``execve`` in the child can ever gain privileges
    via setuid/setgid binaries -- a teeth-bearing half of the privilege guarantee
    (SPEC-11 §2.3) that costs nothing and is enforced by the kernel.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_NO_NEW_PRIVS = 38
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    except Exception:  # pragma: no cover - exercised only on Linux runners
        # Disclosed, never silent: the report's ``privilege_sealed`` reflects reality.
        pass


def make_preexec(seal: DeterminismSeal) -> Callable[[], None]:
    """Return the ``preexec_fn`` run in the forked child *before* ``exec`` of the shell.

    It (1) fixes ``umask`` for deterministic created-node modes, (2) installs CPU /
    file-size / open-file rlimits so a runaway command cannot exhaust the host, and
    (3) drops the privilege-gain bit on Linux. Runs in the child only -- it constrains
    the sandboxed shell, never the parent harness.
    """

    def _preexec() -> None:  # pragma: no cover - runs in the forked child
        os.umask(seal.umask)
        resource.setrlimit(resource.RLIMIT_CPU, (seal.cpu_seconds, seal.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (seal.file_size_bytes, seal.file_size_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (seal.open_files, seal.open_files))
        _set_no_new_privs()

    return _preexec
