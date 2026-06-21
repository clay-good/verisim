"""SPEC-23 (H158/H159) -- the Monitor adapters: today's hardcoded predicates, lifted behind the
:class:`~verisim.audit.protocols.Monitor` protocol with the target passed in, not baked.

These are the SPEC-23 §3.1 table-3 instances. None changes the behavior of the shipped RA module it
wraps -- each *imports* the proven predicate and adapts it to ``covers(action, ctx)``:

  - :class:`SyntacticPathMonitor` -- RA4's :func:`command_targets_protected` (the literal token
    scan). Growable: the auditor synthesizes its covering prefix from empty (RA22's CEGIS). Its
    contract is *literal naming* only, so string-resolvable indirection is its honest residual.
  - :class:`ResolverMonitor` -- RA18's :func:`abstract_targets_protected` (abstract interpreter).
    ``covers`` is True for FIRES *and* ABSTAIN (caught, or routed by reversibility CU27); only a
    CLEAR is uncovered. Its contract is every *string-resolvable* form (its whole purpose is to fold
    them), so a CLEAR on a string-resolvable realizing command is a true silent miss -- exactly the
    RA24 printf-format-escape class, reproduced through the interface.
  - :class:`DenylistMonitor` -- a pattern denylist (the Claude Code status-quo, RA17). A monitor the
    proposer was not tuned against; the audit surfaces the encodings that realize the harm off its
    patterns (H159).

Torch-free, standard library only.
"""

from __future__ import annotations

from verisim.realagent.command_agnostic import command_targets_protected
from verisim.realagent.shell_resolver import abstract_targets_protected

from .protocols import EMPTY, Action, Context, default_in_contract


class SyntacticPathMonitor:
    """RA4's command-agnostic literal target, lifted. Growable (the auditor grows its prefix set
    from empty, reproducing RA22's synthesis); its contract is literal naming, so indirection and
    symlink are its routed residual, not silent holes."""

    def __init__(self, prefix: str | None = None, name: str = "syntactic_path") -> None:
        #: the synthesized covering surface (a set of protected path-prefixes). Empty == not yet
        #: synthesized; the auditor grows it via :meth:`repair` (CEGIS). A non-None ``prefix`` seeds
        #: it (the already-deployed RA4 target).
        self.prefixes: list[str] = [] if prefix is None else [prefix]
        self.name = name

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool:
        return any(command_targets_protected(action.command, p) for p in self.prefixes)

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool:
        """In contract iff the command names a protected path *literally* -- the only feature a
        string-prefix rule can generalize from. Indirection (no literal token) is out of contract:
        the honest residual RA22 routes to the post-commit diff, not a silent hole."""
        return any(command_targets_protected(action.command, p) for p in self._candidate_prefixes())

    def repair(self, action: Action, ctx: Context = EMPTY) -> bool:
        """CEGIS repair: if the command names a protected path literally, generalize it into a
        prefix rule. Returns True iff the surface grew to cover this action (RA22 monotone grow)."""
        for p in self._candidate_prefixes():
            if command_targets_protected(action.command, p):
                if p not in self.prefixes:
                    self.prefixes.append(p)
                return True
        return False

    def _candidate_prefixes(self) -> tuple[str, ...]:
        """The prefixes a literal command could be generalized to. We synthesize toward the known
        protected roots the corpus targets; the seed (if any) plus the conventional roots."""
        return tuple(dict.fromkeys([*self.prefixes, "/etc"]))


class ResolverMonitor:
    """RA18's abstract shell-path resolver, lifted. ``covers`` == the resolver does not declare the
    command CLEAR (FIRES caught, ABSTAIN routed by reversibility). Its contract is every
    string-resolvable form, so a CLEAR on a string-resolvable realizing command is a silent miss."""

    def __init__(self, prefix: str, *, sound_printf: bool = True, fold_filters: bool = False,
                 name: str = "resolver") -> None:
        self.prefix = prefix
        self.sound_printf = sound_printf
        self.fold_filters = fold_filters
        self.name = name

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool:
        verdict = abstract_targets_protected(
            action.command, self.prefix, self.sound_printf, self.fold_filters
        )
        return verdict != "CLEAR"  # FIRES (caught) or ABSTAIN (routed) -- both are "I would check"

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool:
        """The resolver's job is to fold every string-resolvable form, so any such form is in its
        contract; the symlink residual (``string_resolvable=False``) is not (no string analysis can
        decide it)."""
        return action.string_resolvable


class DenylistMonitor:
    """A pattern denylist (the Claude Code status-quo permission list, RA17). ``covers`` == the
    command matches a listed pattern. A monitor the proposer was not tuned against; uses the default
    contract (a string-resolvable miss is a finding), so the audit surfaces real holes (H159)."""

    def __init__(self, patterns: tuple[str, ...], name: str = "denylist") -> None:
        self.patterns = patterns
        self.name = name

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool:
        return any(p in action.command for p in self.patterns)

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool:
        return default_in_contract(action)
