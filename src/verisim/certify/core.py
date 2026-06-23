"""SPEC-28 M1 -- the Coverage Certifier core: point the audit engine at a deployed guardrail.

This is product assembly, not new mechanism. Every load-bearing piece is audited research code:
:func:`~verisim.audit.auditor.audit` (the engine), the :class:`~verisim.audit.protocols.Monitor`
adapters (``SubprocessMonitor`` drives an opaque external hook over stdin/stdout in the Claude Code
PreToolUse contract), the free oracle (``ShellPathOracle`` fast / ``ContainerDiffOracle`` proves
realization vs a real ``/bin/sh``), and the competent proposers SPEC-27 kept (``GrammarProposer``
enumerate + ``BanditProposer``; the neural one is retired). SPEC-28 wires them into one call:

  ``certify_monitor(monitor, oracle, ...) -> CertifyResult``  -- the testable core (any Monitor).
  ``certify_hook(hook_path, ...) -> CertifyResult``           -- builds the SubprocessMonitor.

A **bypass** is a silent hole from the certificate: a command that *realizes* the harm (oracle-true)
that the guardrail *allows* (off its checked surface) and that is *in the guardrail's contract* (so
it is a genuine coverage gap, not the explicitly-routed symlink/runtime residual). That is the thing
a security engineer acts on. Torch-free.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from verisim.audit.auditor import audit
from verisim.audit.bandit import BanditProposer
from verisim.audit.monitors import DenylistMonitor, SubprocessMonitor
from verisim.audit.oracles import ContainerDiffOracle, ShellPathOracle
from verisim.audit.proposers import GrammarProposer
from verisim.audit.protocols import Action, Certificate, Hole, Monitor, Oracle, Proposer

DEFAULT_PROTECTED_PATH = "/etc/shadow"
DEFAULT_WORKDIR = "/home/work"


class _ChainProposer:
    """Yield from several proposers in order; ``audit`` dedups, so one clean certificate covers the
    union. Used for ``proposer='both'`` (systematic enumerate + adaptive bandit)."""

    def __init__(self, proposers: list[Proposer]) -> None:
        self._proposers = proposers
        self.name = "+".join(p.name for p in proposers)

    def propose(self, budget: int) -> Iterator[Action]:
        for p in self._proposers:
            yield from p.propose(budget)


def _build_proposer(kind: str, monitor: Monitor, oracle: Oracle, *, protected_path: str,
                    prefix: str, work: str, seed: int) -> Proposer:
    enumerate_arm = GrammarProposer(protected_path, work, mode="enumerate")
    if kind == "enumerate":
        return enumerate_arm
    bandit_arm = BanditProposer(monitor, oracle, protected_path, prefix, work, seed=seed)
    if kind == "bandit":
        return bandit_arm
    if kind == "both":
        return _ChainProposer([enumerate_arm, bandit_arm])
    raise ValueError(f"unknown proposer {kind!r} (expected enumerate|bandit|both)")


@dataclass
class CertifyResult:
    """The audit verdict on one guardrail: the certificate, the actionable bypasses, a verdict."""

    certificate: Certificate
    bypasses: list[Hole]          # silent holes: realizing, allowed, in-contract == coverage gaps
    monitor: str
    oracle: str
    proposer: str
    protected_path: str
    budget: int = 0

    @property
    def sound(self) -> bool:
        """True iff no in-contract bypass was found over the sampled space (a clean certificate)."""
        return not self.bypasses

    @property
    def n_realizing(self) -> int:
        return self.certificate.n_realizing

    @property
    def covered(self) -> int:
        return self.certificate.covered

    @property
    def coverage_pct(self) -> float:
        """Fraction of realizing actions the guardrail covered (over the sampled space)."""
        n = self.certificate.n_realizing
        return 100.0 * self.certificate.covered / n if n else 100.0

    @property
    def routed_residual(self) -> int:
        """Out-of-contract realizing actions the guardrail routes (symlink/runtime), not gaps."""
        return self.certificate.residual_post_commit + self.certificate.residual_human

    def verdict_line(self) -> str:
        if self.sound:
            return (f"SOUND over the sampled space: 0 in-contract bypasses across "
                    f"{self.certificate.n_realizing} realizing actions "
                    f"({self.routed_residual} routed residual).")
        return (f"LEAKS: {len(self.bypasses)} in-contract bypass(es) the guardrail allows but that "
                f"realize harm — coverage {self.coverage_pct:.1f}% over "
                f"{self.certificate.n_realizing} realizing actions.")


def certify_monitor(
    monitor: Monitor,
    oracle: Oracle,
    *,
    protected_path: str = DEFAULT_PROTECTED_PATH,
    prefix: str = "/etc",
    work: str = DEFAULT_WORKDIR,
    proposer: str = "enumerate",
    budget: int = 512,
    seed: int = 0,
) -> CertifyResult:
    """Audit any :class:`Monitor` (a guardrail) and return its coverage verdict. The testable core:
    pass a fast in-process monitor in tests, or the ``SubprocessMonitor`` from ``certify_hook``."""
    prop = _build_proposer(proposer, monitor, oracle, protected_path=protected_path,
                           prefix=prefix, work=work, seed=seed)
    cert = audit(monitor, oracle, prop, budget)
    bypasses = [h for h in cert.holes if h.silent]
    return CertifyResult(certificate=cert, bypasses=bypasses, monitor=monitor.name,
                         oracle=oracle.name, proposer=prop.name, protected_path=protected_path,
                         budget=budget)


def certify_hook(
    hook_path: str,
    *,
    protected_path: str = DEFAULT_PROTECTED_PATH,
    proposer: str = "enumerate",
    budget: int = 512,
    oracle: str = "syntactic",
    seed: int = 0,
    work: str = DEFAULT_WORKDIR,
    extra_args: Iterable[str] = (),
) -> CertifyResult:
    """Audit a deployed PreToolUse hook (a Python script speaking the Claude Code contract).

    ``oracle='syntactic'`` (default) uses the fast ``ShellPathOracle``; ``oracle='real'`` uses
    ``ContainerDiffOracle`` to *prove* each realizing command against a real ``/bin/sh``, retarget
    the audit to the sandbox's protected path so hook and oracle agree on the protected region.
    """
    if oracle == "real":
        orc: Oracle = ContainerDiffOracle()
        cdo = orc
        assert isinstance(cdo, ContainerDiffOracle)
        prefix = cdo.prefix
        work = cdo.work
        protected_path = f"{prefix}/shadow"
    else:
        prefix = os.path.dirname(protected_path) or "/etc"
        orc = ShellPathOracle(prefix)

    # full env (SubprocessMonitor REPLACES env, so we must carry PATH/PYTHONPATH) + the protected
    # region the hook reads from VERISIM_PROTECTED_PREFIX, so it guards what the oracle judges.
    env = {**os.environ, "VERISIM_PROTECTED_PREFIX": prefix}
    cmd = [sys.executable, hook_path, *extra_args]
    monitor = SubprocessMonitor(cmd, cwd=work, env=env, name=f"hook:{os.path.basename(hook_path)}")
    return certify_monitor(monitor, orc, protected_path=protected_path, prefix=prefix, work=work,
                           proposer=proposer, budget=budget, seed=seed)


def certify_denylist(
    patterns: Iterable[str],
    *,
    protected_path: str = DEFAULT_PROTECTED_PATH,
    proposer: str = "enumerate",
    budget: int = 512,
    seed: int = 0,
    work: str = DEFAULT_WORKDIR,
) -> CertifyResult:
    """Audit a pattern denylist directly (no hook script): a guardrail that blocks a command iff it
    contains one of ``patterns``. The fast second target type -- point it at the deny rules you ship
    and see which harm-realizing encodings sail past them."""
    prefix = os.path.dirname(protected_path) or "/etc"
    pats = tuple(patterns)
    monitor = DenylistMonitor(pats, name=f"denylist[{len(pats)} patterns]")
    return certify_monitor(monitor, ShellPathOracle(prefix), protected_path=protected_path,
                           prefix=prefix, work=work, proposer=proposer, budget=budget, seed=seed)
