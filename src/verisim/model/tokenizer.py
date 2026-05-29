"""Serialize states, actions, and deltas to/from the closed token vocabulary.

The model maps ``<bos> serialize(state) serialize(action) <gen>`` -> ``serialize(Δ)
<eos>`` (SPEC-2 §5.2). This module is the encoder and the inverse parser.

Representation decisions (resolving SPEC-2 §17.1, recorded in docs/semantics.md):
  - **Paths** are absolute in v0 (the drivers only ever emit absolute paths), so a
    path is ``<p>`` + one name token per segment + ``</p>`` and reconstructs as
    ``"/" + "/".join(segments)``.
  - **File content** is a string formed by concatenating the *prefix-free* content
    vocabulary, so it decomposes uniquely by greedy longest-match into content
    tokens (``<c> ... </c>``).
  - **stdout** is predicted as tokens (``<o> ... </o>``) and the parser
    reconstructs ``SetResult`` as ``SetResult(exit, content_hash(stdout))``. cat
    stdout (content words) and ls stdout (single-char names + ``<nl>``) use
    disjoint token languages, so reconstruction is unambiguous. The model *input*
    serializes ``last.exit_code`` only and omits ``stdout_hash`` (a hash is not
    tokenizable, and no command reads prior stdout).
"""

from __future__ import annotations

from verisim.delta.edits import (
    Chmod,
    Create,
    Delete,
    Delta,
    Edit,
    Modify,
    Move,
    SetCwd,
    SetEnv,
    SetResult,
)
from verisim.env.action import Action
from verisim.env.state import Dir, File, Node, State, content_hash

from .vocab import Vocab


class TokenizeError(ValueError):
    """Raised when a string or token sequence is not encodable/parseable in v0."""


# --- leaf helpers -----------------------------------------------------------


def greedy_decompose(content: str, vocab: Vocab) -> list[int]:
    """Decompose a content string into content-token ids by greedy longest-match.

    Unique because the content vocabulary is prefix-free (SPEC-2 §17.1). Raises if
    a suffix matches no content token.
    """
    ids: list[int] = []
    i = 0
    while i < len(content):
        for token in vocab.content_tokens_by_length:
            if content.startswith(token, i):
                ids.append(vocab.content_to_id[token])
                i += len(token)
                break
        else:
            raise TokenizeError(f"content {content!r} not decomposable at offset {i}")
    return ids


def _stdout_token_ids(stdout: str, vocab: Vocab) -> list[int]:
    """Tokenize stdout. cat output decomposes via the content vocab; otherwise it
    is ls output: ``\\n``-joined single-char names."""
    if stdout == "":
        return []
    try:
        return greedy_decompose(stdout, vocab)
    except TokenizeError:
        pass
    ids: list[int] = []
    parts = stdout.split("\n")
    for k, part in enumerate(parts):
        if k:
            ids.append(vocab.id("<nl>"))
        if part:
            if part not in vocab.name_to_id:
                raise TokenizeError(f"stdout part {part!r} is not a known name")
            ids.append(vocab.name_to_id[part])
    return ids


def _path_ids(path: str, vocab: Vocab) -> list[int]:
    ids = [vocab.id("<p>")]
    for seg in path.split("/"):
        if not seg:
            continue
        if seg not in vocab.name_to_id:
            raise TokenizeError(f"path segment {seg!r} is not a known name")
        ids.append(vocab.name_to_id[seg])
    ids.append(vocab.id("</p>"))
    return ids


def _content_ids(content: str, vocab: Vocab) -> list[int]:
    return [vocab.id("<c>"), *greedy_decompose(content, vocab), vocab.id("</c>")]


def _stdout_ids(stdout: str, vocab: Vocab) -> list[int]:
    return [vocab.id("<o>"), *_stdout_token_ids(stdout, vocab), vocab.id("</o>")]


def _node_ids(node: Node, vocab: Vocab) -> list[int]:
    if isinstance(node, File):
        return [vocab.id("<file>"), *_content_ids(node.content, vocab), vocab.mode_to_id[node.mode]]
    return [vocab.id("<dir>"), vocab.mode_to_id[node.mode]]


# --- encoders ---------------------------------------------------------------


def encode_state(state: State, vocab: Vocab) -> list[int]:
    ids = [vocab.id("<state>")]
    for path in sorted(state.fs):
        ids += _path_ids(path, vocab)
        ids += _node_ids(state.fs[path], vocab)
    ids.append(vocab.id("<cwd>"))
    ids += _path_ids(state.cwd, vocab)
    ids.append(vocab.id("<env>"))
    for key in sorted(state.env):
        ids.append(vocab.envkey_to_id[key])
        ids.append(vocab.content_to_id[state.env[key]])
    ids.append(vocab.id("<last>"))
    ids.append(vocab.exit_to_id[state.last.exit_code])
    return ids


def encode_action(action: Action, vocab: Vocab) -> list[int]:
    ids = [vocab.id("<action>"), vocab.command_token_id(action.name, action.recursive)]
    name = action.name
    if name in {"mkdir", "rmdir", "touch", "cd", "cat", "ls", "rm"}:
        ids += _path_ids(action.args[0], vocab)
    elif name in {"mv", "cp"}:
        ids += _path_ids(action.args[0], vocab)
        ids += _path_ids(action.args[1], vocab)
    elif name in {"write", "append"}:
        ids += _path_ids(action.args[0], vocab)
        ids.append(vocab.content_to_id[action.args[1]])
    elif name == "chmod":
        ids.append(vocab.mode_to_id[int(action.args[0], 8)])
        ids += _path_ids(action.args[1], vocab)
    elif name == "export":
        ids.append(vocab.envkey_to_id[action.args[0]])
        ids.append(vocab.content_to_id[action.args[1]])
    else:  # pragma: no cover - parser already rejects unknown commands
        raise TokenizeError(f"cannot encode action {name!r}")
    return ids


def _edit_ids(edit: Edit, stdout: str, vocab: Vocab) -> list[int]:
    if isinstance(edit, Create):
        return [vocab.id("<create>"), *_path_ids(edit.path, vocab), *_node_ids(edit.node, vocab)]
    if isinstance(edit, Delete):
        return [vocab.id("<delete>"), *_path_ids(edit.path, vocab)]
    if isinstance(edit, Modify):
        return [
            vocab.id("<modify>"),
            *_path_ids(edit.path, vocab),
            *_content_ids(edit.content, vocab),
        ]
    if isinstance(edit, Move):
        return [vocab.id("<move>"), *_path_ids(edit.src, vocab), *_path_ids(edit.dst, vocab)]
    if isinstance(edit, Chmod):
        return [vocab.id("<chmod>"), *_path_ids(edit.path, vocab), vocab.mode_to_id[edit.mode]]
    if isinstance(edit, SetCwd):
        return [vocab.id("<setcwd>"), *_path_ids(edit.path, vocab)]
    if isinstance(edit, SetEnv):
        return [vocab.id("<setenv>"), vocab.envkey_to_id[edit.key], vocab.content_to_id[edit.token]]
    return [vocab.id("<setresult>"), vocab.exit_to_id[edit.exit_code], *_stdout_ids(stdout, vocab)]


def encode_target(delta: Delta, stdout: str, vocab: Vocab) -> list[int]:
    """Encode a delta (with the SetResult's stdout string) as the model target."""
    ids: list[int] = []
    for edit in delta:
        ids += _edit_ids(edit, stdout, vocab)
    ids.append(vocab.eos)
    return ids


def encode_prompt(state: State, action: Action, vocab: Vocab) -> list[int]:
    """The model input: ``<bos> state action <gen>``."""
    return [vocab.bos, *encode_state(state, vocab), *encode_action(action, vocab), vocab.gen]


# --- parser (inverse of encode_target) --------------------------------------


class _Cursor:
    def __init__(self, ids: list[int], vocab: Vocab) -> None:
        self.ids = ids
        self.vocab = vocab
        self.i = 0

    def peek(self) -> int:
        if self.i >= len(self.ids):
            raise TokenizeError("unexpected end of token stream")
        return self.ids[self.i]

    def take(self) -> int:
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, token: str) -> None:
        got = self.take()
        if got != self.vocab.id(token):
            raise TokenizeError(f"expected {token!r}, got {self.vocab.token(got)!r}")


def _parse_path(cur: _Cursor) -> str:
    cur.expect("<p>")
    segs: list[str] = []
    end = cur.vocab.id("</p>")
    while cur.peek() != end:
        tok = cur.take()
        if tok not in cur.vocab.id_to_name:
            raise TokenizeError(f"expected name in path, got {cur.vocab.token(tok)!r}")
        segs.append(cur.vocab.id_to_name[tok])
    cur.take()  # </p>
    return "/" + "/".join(segs)


def _parse_content(cur: _Cursor) -> str:
    cur.expect("<c>")
    words: list[str] = []
    end = cur.vocab.id("</c>")
    while cur.peek() != end:
        tok = cur.take()
        if tok not in cur.vocab.id_to_content:
            raise TokenizeError(f"expected content token, got {cur.vocab.token(tok)!r}")
        words.append(cur.vocab.id_to_content[tok])
    cur.take()  # </c>
    return "".join(words)


def _parse_stdout(cur: _Cursor) -> str:
    cur.expect("<o>")
    out: list[str] = []
    end = cur.vocab.id("</o>")
    nl = cur.vocab.id("<nl>")
    while cur.peek() != end:
        tok = cur.take()
        if tok == nl:
            out.append("\n")
        elif tok in cur.vocab.id_to_content:
            out.append(cur.vocab.id_to_content[tok])
        elif tok in cur.vocab.id_to_name:
            out.append(cur.vocab.id_to_name[tok])
        else:
            raise TokenizeError(f"expected stdout token, got {cur.vocab.token(tok)!r}")
    cur.take()  # </o>
    return "".join(out)


def _parse_mode(cur: _Cursor) -> int:
    tok = cur.take()
    if tok not in cur.vocab.id_to_mode:
        raise TokenizeError(f"expected mode token, got {cur.vocab.token(tok)!r}")
    return cur.vocab.id_to_mode[tok]


def _parse_node(cur: _Cursor) -> Node:
    tag = cur.take()
    if tag == cur.vocab.id("<file>"):
        content = _parse_content(cur)
        return File(content=content, mode=_parse_mode(cur))
    if tag == cur.vocab.id("<dir>"):
        return Dir(mode=_parse_mode(cur))
    raise TokenizeError(f"expected <file> or <dir>, got {cur.vocab.token(tag)!r}")


def parse_target(ids: list[int], vocab: Vocab) -> tuple[Delta, str]:
    """Parse a target token sequence into ``(delta, stdout)``; raise if malformed.

    The returned delta's ``SetResult`` carries ``content_hash(stdout)``, matching
    the oracle's own representation.
    """
    cur = _Cursor(ids, vocab)
    delta: Delta = []
    stdout = ""
    while cur.peek() != vocab.eos:
        op = cur.take()
        if op == vocab.id("<create>"):
            path = _parse_path(cur)
            delta.append(Create(path, _parse_node(cur)))
        elif op == vocab.id("<delete>"):
            delta.append(Delete(_parse_path(cur)))
        elif op == vocab.id("<modify>"):
            path = _parse_path(cur)
            delta.append(Modify(path, _parse_content(cur)))
        elif op == vocab.id("<move>"):
            src = _parse_path(cur)
            delta.append(Move(src, _parse_path(cur)))
        elif op == vocab.id("<chmod>"):
            path = _parse_path(cur)
            delta.append(Chmod(path, _parse_mode(cur)))
        elif op == vocab.id("<setcwd>"):
            delta.append(SetCwd(_parse_path(cur)))
        elif op == vocab.id("<setenv>"):
            key_tok = cur.take()
            val_tok = cur.take()
            if key_tok not in vocab.id_to_envkey or val_tok not in vocab.id_to_content:
                raise TokenizeError("malformed setenv")
            delta.append(SetEnv(vocab.id_to_envkey[key_tok], vocab.id_to_content[val_tok]))
        elif op == vocab.id("<setresult>"):
            exit_tok = cur.take()
            if exit_tok not in vocab.id_to_exit:
                raise TokenizeError("malformed setresult exit code")
            stdout = _parse_stdout(cur)
            delta.append(SetResult(vocab.id_to_exit[exit_tok], content_hash(stdout)))
        else:
            raise TokenizeError(f"expected an edit op, got {vocab.token(op)!r}")
    return delta, stdout


__all__ = [
    "TokenizeError",
    "encode_action",
    "encode_prompt",
    "encode_state",
    "encode_target",
    "greedy_decompose",
    "parse_target",
]
