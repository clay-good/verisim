"""The closed token vocabulary for the world model (SPEC-2 §5.2).

v0 fixes finite content/mode/env/name vocabularies (SPEC-2 §2.2), so the entire
serialization DSL -- states, actions, and deltas -- maps to a small *closed* set
of tokens built deterministically from an :class:`~verisim.env.config.EnvConfig`.
A from-scratch tiny transformer (M4) trains over exactly these ids.

Token families:
  - specials: ``<pad> <bos> <eos>``;
  - section markers: ``<state> <action> <gen> <cwd> <env> <last>``;
  - structure markers: path ``<p>../p>``, content ``<c>../c>``, stdout
    ``<o>../o>``, ``<nl>``, node types ``<file> <dir>``;
  - delta ops: ``<create> <delete> <modify> <move> <chmod> <setcwd> <setenv>
    <setresult>``;
  - command tokens (one per §2.2 command), exit codes, and the four leaf
    vocabularies (path-name segments, content tokens, modes, env keys).
"""

from __future__ import annotations

from verisim.env.config import EnvConfig

_SPECIALS = ("<pad>", "<bos>", "<eos>")
_MARKERS = (
    "<state>", "<action>", "<gen>", "<cwd>", "<env>", "<last>",
    "<p>", "</p>", "<c>", "</c>", "<o>", "</o>", "<nl>", "<file>", "<dir>",
)
_OPS = (
    "<create>", "<delete>", "<modify>", "<move>",
    "<chmod>", "<setcwd>", "<setenv>", "<setresult>",
)
_COMMANDS = (
    "mkdir", "rmdir", "touch", "rm", "rm-r", "mv", "cp", "cp-r",
    "write", "append", "chmod", "cd", "cat", "ls", "export",
)
_EXITS = (0, 1)

# Maps a parsed Action (name, recursive) to its command token suffix.
_COMMAND_TOKEN = {
    ("mkdir", False): "mkdir", ("rmdir", False): "rmdir", ("touch", False): "touch",
    ("rm", False): "rm", ("rm", True): "rm-r", ("mv", False): "mv",
    ("cp", False): "cp", ("cp", True): "cp-r", ("write", False): "write",
    ("append", False): "append", ("chmod", False): "chmod", ("cd", False): "cd",
    ("cat", False): "cat", ("ls", False): "ls", ("export", False): "export",
}


class Vocab:
    """A closed, deterministic token<->id mapping for one ``EnvConfig``."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        tokens: list[str] = []
        tokens += _SPECIALS
        tokens += _MARKERS
        tokens += _OPS
        tokens += [f"<cmd:{c}>" for c in _COMMANDS]
        tokens += [f"<exit:{e}>" for e in _EXITS]
        tokens += [f"<name:{n}>" for n in config.name_pool]
        tokens += [f"<content:{c}>" for c in config.content_tokens]
        tokens += [f"<mode:{m:o}>" for m in config.modes]
        tokens += [f"<env:{k}>" for k in config.env_keys]

        self._tokens = tuple(tokens)
        self._id = {tok: i for i, tok in enumerate(self._tokens)}

        # Leaf lookup tables (token id <-> domain value).
        self.name_to_id = {n: self._id[f"<name:{n}>"] for n in config.name_pool}
        self.id_to_name = {v: k for k, v in self.name_to_id.items()}
        self.content_to_id = {c: self._id[f"<content:{c}>"] for c in config.content_tokens}
        self.id_to_content = {v: k for k, v in self.content_to_id.items()}
        self.mode_to_id = {m: self._id[f"<mode:{m:o}>"] for m in config.modes}
        self.id_to_mode = {v: k for k, v in self.mode_to_id.items()}
        self.envkey_to_id = {k: self._id[f"<env:{k}>"] for k in config.env_keys}
        self.id_to_envkey = {v: k for k, v in self.envkey_to_id.items()}
        self.exit_to_id = {e: self._id[f"<exit:{e}>"] for e in _EXITS}
        self.id_to_exit = {v: k for k, v in self.exit_to_id.items()}

        # Content tokens sorted longest-first for greedy prefix-free decomposition.
        self.content_tokens_by_length = tuple(
            sorted(config.content_tokens, key=len, reverse=True)
        )

    # -- core mapping --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tokens)

    def id(self, token: str) -> int:
        return self._id[token]

    def token(self, token_id: int) -> str:
        return self._tokens[token_id]

    def command_token_id(self, name: str, recursive: bool) -> int:
        return self._id[f"<cmd:{_COMMAND_TOKEN[(name, recursive)]}>"]

    # -- frequently used ids (named for readability) -------------------------

    @property
    def pad(self) -> int:
        return self._id["<pad>"]

    @property
    def bos(self) -> int:
        return self._id["<bos>"]

    @property
    def eos(self) -> int:
        return self._id["<eos>"]

    @property
    def gen(self) -> int:
        return self._id["<gen>"]

    # -- token-class sets (for the grammar / constrained decoder) ------------

    @property
    def op_ids(self) -> frozenset[int]:
        return frozenset(self._id[op] for op in _OPS)

    @property
    def name_ids(self) -> frozenset[int]:
        return frozenset(self.name_to_id.values())

    @property
    def content_ids(self) -> frozenset[int]:
        return frozenset(self.content_to_id.values())

    @property
    def mode_ids(self) -> frozenset[int]:
        return frozenset(self.mode_to_id.values())

    @property
    def exit_ids(self) -> frozenset[int]:
        return frozenset(self.exit_to_id.values())

    @property
    def envkey_ids(self) -> frozenset[int]:
        return frozenset(self.envkey_to_id.values())
