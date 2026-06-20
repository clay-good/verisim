#!/usr/bin/env python3
"""A runnable Claude Code PreToolUse hook: the oracle coverage gate as the permission decision.

This is the thin stdin/stdout entrypoint for SPEC-22 RA17 (H149); the logic is
:func:`verisim.realagent.claude_code_gate.hook_decision`. Claude Code writes the call to stdin as
JSON (`tool_name`, `tool_input`, `cwd`, ...); this prints a `hookSpecificOutput` whose
`permissionDecision` is `allow` for the off-surface majority (auto-approve, no prompt) and `ask` for
the sparse on-surface minority that can realize harm (escalate to the human = spend an approval
prompt on exactly the covering surface). See the paper §8 product reading.

Register it in `.claude/settings.json` (the protected region is passed via VERISIM_PROTECTED_PREFIX,
default `/etc`)::

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash|Edit|Write",
            "hooks": [
              {"type": "command", "command": "python3 scripts/claude_code_coverage_hook.py"}
            ]
          }
        ]
      }
    }

Reads one JSON event from stdin, prints one JSON decision to stdout, exits 0. On any error it prints
`{}` (proceed through the normal permission flow) rather than failing the agent's turn.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    try:
        from verisim.realagent.claude_code_gate import PROTECTED_PREFIX, hook_decision

        event = json.load(sys.stdin)
        prefix = os.environ.get("VERISIM_PROTECTED_PREFIX", PROTECTED_PREFIX)
        decision = hook_decision(event, prefix=prefix)
    except Exception:
        decision = {}  # fail open to the normal permission flow; never break the agent's turn
    json.dump(decision, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
