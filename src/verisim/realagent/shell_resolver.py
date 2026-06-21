"""SPEC-22 RA18 (H150) -- the abstract shell-path resolver: the open edge, split and closed on the
half that can be, with the residual proven irreducible to string analysis.

RA4 (command_agnostic.py) is honest that its pre-commit ``target`` is a regex token-scan: it fires
on any command that names the protected path *literally*, and it misses *indirection* -- a path
built from pieces, assembled in a subshell, or reached through a planted symlink -- because the
literal prefix never appears in the command string. The paper calls hardening this the single open
frontier (§5, §8). RA18 attacks it the only honest way: it **partitions** the open edge into a half
that string analysis can close and a half that provably cannot, closes the first, and routes the
second.

The mechanism is a **no-execution abstract interpreter** over a single Bash command string
(:func:`abstract_targets_protected`), built only on the standard library (``shlex`` + a constant
environment fold). It returns a **3-valued verdict**:

  - ``FIRES``   -- after folding, a word provably names the protected prefix (caught pre-commit).
  - ``CLEAR``   -- every expansion resolved to a constant and none named the prefix (provably safe
                   *with respect to the command string*).
  - ``ABSTAIN`` -- an expansion could not be resolved without filesystem or runtime state (a
    variable bound elsewhere, a non-folded ``$(...)``, ``eval``, ``$RANDOM``, arithmetic, ...).

**The soundness invariant, by construction: ABSTAIN never collapses to CLEAR.** Every unresolved
expansion is replaced by a sentinel and forces ABSTAIN, so a command that *could* name the prefix at
runtime is never silently declared CLEAR. CLEAR requires that *all* words folded to constants. This
is the no-silent-breach property the whole RA arc depends on, now preserved across indirection.

What the resolver closes (string-resolvable indirection): constant variable assembly within one
command (``p=/et; echo x > ${p}c/shadow``), quote/backslash splicing (``/et""c/shadow`` folds to
``/etc/shadow`` under ``shlex``, which the regex does NOT do), and pure-literal command substitution
(``$(printf /%s/shadow etc)``) over a whitelisted side-effect-free builtin set (printf/echo/
basename/dirname).

What it provably cannot close, and why that is not a defeat:

  - **Symlink indirection** (``echo x > /home/work/link/shadow`` with ``link -> /etc`` planted on
    disk). The literal ``/etc`` is *nowhere* in the command string; the same string is harm-or-
    benign depending solely on on-disk link state the gate never sees. No pure-string analysis can
    decide it (one-line impossibility argument). The resolver returns CLEAR *on the string*
    (correct: the string names ``/home/work/...``); the danger is caught by the exact post-commit
    fs-diff for reversible actions (CU27), which sees the realized effect however the path was
    spelled.
  - **Runtime-valued expansion** (``eval $x``, ``$RANDOM``, ``$(cat /tmp/target)``, cross-call
    bindings). These cannot be folded without execution; they are ABSTAIN, and the routing layer
    sends them to the post-commit diff (reversible) or a human prompt (irreversible) -- never a
    silent miss.

So the honest headline is not "RA4 is solved." It is: *the open edge splits into a closed,
string-resolvable half and a provably-irreducible state/runtime-dependent half; the first is closed
pre-commit by abstract interpretation, the second is routed to the post-commit oracle by
reversibility (CU27).* The reversibility routing this enables is wired into the Claude Code hook
(:mod:`verisim.realagent.claude_code_gate`), making the artifact finally match the paper's promise.

Hermetic: standard library only, no command execution, deterministic, sub-millisecond. Torch-free.
"""

from __future__ import annotations

import codecs
import os
import re
import shlex
from typing import Literal

from verisim.realagent.command_agnostic import command_targets_protected

Verdict = Literal["FIRES", "CLEAR", "ABSTAIN"]
PROTECTED_PREFIX = "/etc"

#: A sentinel standing for "an expansion that could not be resolved without runtime/FS state". It
#: can never match the protected prefix, and its presence in a folded statement forces ABSTAIN.
_UNRES = "\x00UNRES\x00"

#: Statement separators (we keep redirects like ``>`` inside a statement --
#: command_targets_protected handles those -- and split on the control operators that start a fresh
#: command).
_STMT_SPLIT = re.compile(r"\|\||&&|[;\n|&]")

#: The whitelisted pure, side-effect-free, deterministic builtins we will constant-fold inside
#: $(...).
_PURE_BUILTINS = {"printf", "echo", "basename", "dirname"}

_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_ARITH = re.compile(r"\$\(\([^)]*\)\)")
_PROCSUB = re.compile(r"[<>]\([^)]*\)")
_SUBST = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")  # non-nested command substitution / backticks


def _printf_fold(fmt: str, args: list[str], sound_printf: bool = True) -> str | None:
    """Emulate a SAFE subset of printf: literal text plus ``%s`` substitution only. Any other format
    directive (``%d``, widths, ...) or an argument-count mismatch returns None (conservative).

    Soundness (RA24/H156): ``printf`` decodes C backslash escapes in the FORMAT (``\\x2f`` -> ``/``,
    ``\\057`` octal), so a format carrying one expands to bytes this fold does not produce. Folding
    it as literal text yields a *wrong constant* that can drop the protected prefix -- a silent miss
    the learned adversary found. The sound fix is to refuse (ABSTAIN) on any format with a backslash
    escape rather than fold it wrong. ``sound_printf=False`` restores the pre-RA24 behavior so the
    discovery is reproducible."""
    if sound_printf and "\\" in fmt:
        return None  # an undecoded printf format escape -- refuse rather than fold wrong
    if "%%" in fmt:
        return None  # escaped percent -- out of our safe subset
    parts = fmt.split("%s")
    specifiers = len(parts) - 1
    if specifiers == 0:
        return fmt if not args else None  # no %s but args given -> not our subset
    if any("%" in p for p in parts):
        return None  # a non-%s directive remains
    if len(args) != specifiers:
        return None  # printf cycles/pads; we refuse rather than guess
    out = parts[0]
    for arg, tail in zip(args, parts[1:], strict=True):  # lengths checked equal above
        out += arg + tail
    return out


def _fold_builtin(body: str, env: dict[str, str], sound_printf: bool = True) -> str | None:
    """Constant-fold one command-substitution body if it is a whitelisted pure builtin with constant
    args; otherwise None (which becomes the ABSTAIN sentinel)."""
    expanded = _expand_params(_decode_ansi_c(body), env)
    if _UNRES in expanded:
        return None
    try:
        toks = shlex.split(expanded, posix=True)
    except ValueError:
        return None
    if not toks or toks[0] not in _PURE_BUILTINS:
        return None
    name, rest = toks[0], toks[1:]
    if name == "printf" and rest:
        return _printf_fold(rest[0], rest[1:], sound_printf)
    if name == "echo":
        return " ".join(a for a in rest if a != "-n")
    if name == "basename" and len(rest) == 1:
        return os.path.basename(rest[0])
    if name == "dirname" and len(rest) == 1:
        return os.path.dirname(rest[0])
    return None


_ANSI_C = re.compile(r"\$'((?:\\.|[^'\\])*)'")  # ANSI-C quoting $'...' (shlex does NOT decode this)
_PARAM = re.compile(r"\$\{([^{}]*)\}")  # innermost ${...} (no nested braces)
_DOLLAR_NAME = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*|[0-9]+)")
_BRACE = re.compile(r"\{([^{}]*,[^{}]*)\}")  # a brace LIST {a,b,...}
_GLOB = re.compile(r"[*?]|\[[^\]]+\]")  # path globbing -- FS-state-dependent, so ABSTAIN


def _decode_ansi_c(text: str) -> str:
    """Decode ``$'...'`` ANSI-C quotes (``\\x2f`` -> ``/``, ``\\057`` octal, ``\\n`` ...). bash
    decodes these; Python's ``shlex`` does not, so an undecoded ``$'\\x2f'`` is a real evasion -- we
    decode it here. A body that will not decode becomes the sentinel."""
    def repl(m: re.Match[str]) -> str:
        try:
            return codecs.decode(m.group(1).encode("utf-8", "backslashreplace"), "unicode_escape")
        except Exception:
            return _UNRES

    return _ANSI_C.sub(repl, text)


def _const(name: str, env: dict[str, str]) -> str | None:
    """The value of ``name`` if it is bound to a known constant, else None (unknown/runtime)."""
    v = env.get(name)
    return v if (v is not None and _UNRES not in v) else None


def _expand_one_param(body: str, env: dict[str, str]) -> str:
    """Expand one ``${...}`` body (no nested ``${}``) under the constant environment. Returns the
    sentinel for any operand that is not a known constant -- so an unresolved parameter never folds
    to a wrong constant. Covers the bash forms a path can be smuggled through: default/alternate
    (``:-`` ``-`` ``:+`` ``+`` ``:=``), indirect (``!``), substring (``:off:len``), strip
    (``#`` ``##`` ``%`` ``%%``), replace (``/`` ``//``), and case (``^`` ``^^`` ``,`` ``,,``)."""
    if body.startswith("!"):  # indirect ${!name}
        nm = body[1:]
        target = _const(nm, env) if re.fullmatch(r"[A-Za-z_]\w*", nm) else None
        inner = _const(target, env) if target is not None else None
        return inner if inner is not None else _UNRES
    m = re.match(r"^([A-Za-z_]\w*|[0-9]+)(.*)$", body, re.DOTALL)
    if not m:
        return _UNRES
    name, op = m.group(1), m.group(2)
    val = _const(name, env)  # str if a known constant, else None
    if op == "":
        return val if val is not None else _UNRES
    for sig in (":-", "-"):  # use default when unset (the smuggled path is usually the default)
        if op.startswith(sig):
            if val is not None and (val != "" or sig == "-"):
                return val
            return _expand_params(op[len(sig):], env)
    if op.startswith(":="):
        return val if (val is not None and val != "") else _expand_params(op[2:], env)
    for sig in (":+", "+"):  # use alternate when set
        if op.startswith(sig):
            if val is not None and (val != "" or sig == "+"):
                return _expand_params(op[len(sig):], env)
            return ""
    if op.startswith((":?", "?")):
        return val if val is not None else _UNRES
    if val is None:
        return _UNRES  # all remaining forms need a constant base value (mypy narrows val to str)
    mo = re.match(r"^:(-?\d+)(?::(-?\d+))?$", op)  # substring ${name:off[:len]}
    if mo:
        off = int(mo.group(1))
        s = val[off:] if off >= 0 else val[len(val) + off:]
        if mo.group(2) is not None:
            ln = int(mo.group(2))
            s = s[:ln] if ln >= 0 else s[:len(s) + ln]
        return s
    for sig in ("##", "#", "%%", "%"):  # strip prefix/suffix (literal only)
        if op.startswith(sig):
            pat = op[len(sig):]
            if any(c in pat for c in "*?["):
                return _UNRES  # glob strip is FS-pattern-dependent -> conservative
            if sig in ("#", "##") and val.startswith(pat):
                return val[len(pat):]
            if sig in ("%", "%%") and pat and val.endswith(pat):
                return val[: -len(pat)]
            return val
    mr = re.match(r"^(//?)(.+)$", op)  # replace ${name/pat/repl} or //
    if mr and "/" in mr.group(2):
        pat, _, rep = mr.group(2).partition("/")
        if any(c in pat for c in "*?["):
            return _UNRES
        rep = _expand_params(rep, env)
        return val.replace(pat, rep) if mr.group(1) == "//" else val.replace(pat, rep, 1)
    if op in ("^", "^^"):
        return val.upper() if op == "^^" else (val[:1].upper() + val[1:])
    if op in (",", ",,"):
        return val.lower() if op == ",," else (val[:1].lower() + val[1:])
    return _UNRES


def _expand_params(text: str, env: dict[str, str]) -> str:
    """Expand ``${...}`` (full grammar) and ``$name`` under the constant env; any residual ``$``
    that is not a form we model becomes the sentinel (the soundness backstop)."""
    for _ in range(8):  # innermost-out, bounded
        m = _PARAM.search(text)
        if not m:
            break
        text = text[: m.start()] + _expand_one_param(m.group(1), env) + text[m.end():]

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        return env[name] if (name in env and _UNRES not in env[name]) else _UNRES

    text = _DOLLAR_NAME.sub(repl, text)
    if "$" in text:  # an unmodeled $-construct remains -> unresolved (never silently CLEAR)
        text = text.replace("$", _UNRES)
    return text


def _expand_braces(token: str, _depth: int = 0) -> list[str]:
    """Brace-list expansion ``{a,b}`` -> [a, b] (bash expands these before word splitting)."""
    if _depth > 6:
        return [token]
    m = _BRACE.search(token)
    if not m:
        return [token]
    pre, post = token[: m.start()], token[m.end():]
    out: list[str] = []
    for opt in m.group(1).split(","):
        out.extend(_expand_braces(pre + opt + post, _depth + 1))
        if len(out) > 64:
            break
    return out


def _expand_tilde(token: str, env: dict[str, str]) -> str:
    """Expand a leading ``~`` / ``~/`` from a tracked constant ``HOME``; ``~user`` or unknown
    HOME is the sentinel (state-dependent)."""
    if token == "~" or token.startswith("~/"):
        home = env.get("HOME")
        return (home + token[1:]) if (home and _UNRES not in home) else (_UNRES + token[1:])
    if token.startswith("~"):
        return _UNRES + token
    return token


def _fold_substitutions(text: str, env: dict[str, str], sound_printf: bool = True) -> str:
    """Fold ``$(...)``/backticks (pure builtins) to constants; everything else becomes the
    sentinel. Arithmetic and process substitution are conservatively the sentinel. Operates at the
    STRING level (before tokenizing) so a substitution is never shattered across shlex tokens."""
    text = _ARITH.sub(_UNRES, text)
    text = _PROCSUB.sub(_UNRES, text)

    def repl(m: re.Match[str]) -> str:
        body = m.group(1) if m.group(1) is not None else m.group(2)
        folded = _fold_builtin(body, env, sound_printf)
        return folded if folded is not None else _UNRES

    for _ in range(4):  # fold left-to-right, a few passes for non-overlapping nesting
        new = _SUBST.sub(repl, text)
        if new == text:
            break
        text = new
    if "$(" in text or "`" in text:  # an unfoldable substitution remains -> unresolved
        text = text.replace("$(", _UNRES).replace("`", _UNRES)
    return text


def _fold_statement(stmt: str, env: dict[str, str], sound_printf: bool = True) -> str:
    """Fold one statement to a constant string (sentinels for unresolved parts), updating ``env``
    for standalone assignments. Returns the folded string to scan for the protected prefix.

    Order matters: fold ``$(...)`` and expand ``$VAR`` at the STRING level first (so substitutions
    are not split by the tokenizer), THEN shlex to fold quote/backslash splicing and peel
    assignments.
    """
    stmt = stmt.strip()
    if not stmt:
        return ""

    # string-level expansion BEFORE tokenizing (a construct is never shattered across shlex tokens):
    # ANSI-C quotes, then command substitution / arithmetic / process substitution, then parameters.
    folded = _decode_ansi_c(stmt)
    folded = _fold_substitutions(folded, env, sound_printf)
    folded = _expand_params(folded, env)
    try:
        toks = shlex.split(folded, posix=True, comments=False)
    except ValueError:
        return _UNRES  # cannot tokenize safely -> ABSTAIN

    # peel leading assignments (NAME=VALUE ...). A statement that is ONLY assignments persists into
    # env; an assignment PREFIX on a command does not persist (bash scopes it to that command).
    i = 0
    assigned: dict[str, str] = {}
    while i < len(toks):
        word = toks[i]
        if word == "export" and i + 1 < len(toks):
            i += 1
            word = toks[i]
        m = _ASSIGN.match(word)
        if not m:
            break
        assigned[m.group(1)] = m.group(2)
        i += 1

    rest = toks[i:]
    if not rest:
        env.update(assigned)  # standalone assignment statement persists for later statements
        return ""

    # word-level expansion bash does after tokenizing: tilde, then brace lists. Backstop: any token
    # with a residual brace-list / leading tilde we could not expand becomes the sentinel (ABSTAIN).
    words: list[str] = []
    for w in rest:
        w = _expand_tilde(w, env)
        for bw in _expand_braces(w):
            # backstop: a residual brace-list, leading tilde, or glob metachar is FS/runtime-state-
            # dependent (the realized path is not fixed by the string) -> ABSTAIN, not silent CLEAR.
            if _BRACE.search(bw) or bw.startswith("~") or _GLOB.search(bw):
                bw = bw + _UNRES
            words.append(bw)
    joined = " ".join(words)
    if "eval" in rest:  # eval runs a runtime-built string: keep the literal scan but force ABSTAIN
        joined = joined + " " + _UNRES
    return joined


def _strip_sentinels(text: str) -> str:
    return text.replace(_UNRES, " ")


def abstract_targets_protected(command: str, prefix: str = PROTECTED_PREFIX,
                               sound_printf: bool = True) -> Verdict:
    """The abstract shell-path resolver: FIRES / CLEAR / ABSTAIN for whether ``command`` reaches
    under ``prefix`` after constant folding. ABSTAIN never collapses to CLEAR (sound by
    construction).

    ``sound_printf`` (default True) refuses to fold a ``printf`` format that carries a backslash
    escape (RA24/H156 hardening); pass False to reproduce the pre-RA24 silent miss the learned
    adversary discovered.
    """
    env: dict[str, str] = {}
    fired = False
    abstain = False
    for stmt in _STMT_SPLIT.split(command):
        folded = _fold_statement(stmt, env, sound_printf)
        if _UNRES in folded:
            abstain = True
        # scan the resolved part for a literal/indirect-literal protected reference (reuse RA4's
        # proven token splitter, which handles redirects, `=`, and quote delimiters)
        if command_targets_protected(_strip_sentinels(folded), prefix):
            fired = True
    if fired:
        return "FIRES"
    return "ABSTAIN" if abstain else "CLEAR"


# --- reversibility routing (CU27): turn the 3-valued verdict into a gate decision ----------------
#
# CU27's reversibility theorem: a reversible action can be verified AFTER commit (execute, diff the
# protected region, roll back on harm) -- exact and model-free; an irreversible action (a network
# send) must be verified BEFORE commit. So an ABSTAIN routes by reversibility: reversible -> allow
# and let the post-commit diff catch any realized effect (including symlink indirection);
# irreversible -> fail closed to a human prompt, because it cannot be undone and we cannot prove it
# off-surface.

#: Tokens that begin an off-host send (irreversible: the data leaves before any post-check).
_EGRESS = ("curl", "wget", "nc", "ncat", "netcat", "ssh", "scp", "sftp", "rsync", "telnet",
           "ftp", "mail", "sendmail", "ssh-copy-id")


def is_irreversible(command: str) -> bool:
    """Conservative reversibility classifier (CU27): a command is irreversible if it begins off-host
    send, OR if its effect cannot be bounded at all (``eval`` of a runtime string -- which could
    itself be a send). Filesystem mutations are reversible (snapshot + post-commit diff +
    rollback); a send, or an unbounded ``eval``, is not. Unparseable commands fail closed (treated
    irreversible).
    """
    try:
        for seg in _STMT_SPLIT.split(command):
            toks = shlex.split(seg, posix=True)
            for t in toks:
                base = os.path.basename(t)
                if base in _EGRESS or base == "eval":
                    return True
    except ValueError:
        return True  # cannot parse -> assume irreversible (fail closed)
    return False
