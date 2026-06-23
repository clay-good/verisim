#!/usr/bin/env python3
"""A deliberately WEAK example Claude Code PreToolUse hook -- a literal-substring denylist.

For SPEC-28 demos and tests: this is the kind of hand-rolled guardrail a team ships in an afternoon,
and exactly the kind the Coverage Certifier should catch. It blocks a command only if the protected
path appears *literally*, so every indirection encoding (assembled paths, printf/hex escapes,
pipe filters) sails through. ``verisim-certify audit --hook scripts/examples/weak_denylist_hook.py``
should report those bypasses. Pure stdlib, no imports beyond json/os/sys (fast to spawn).
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    try:
        event = json.load(sys.stdin)
        prefix = os.environ.get("VERISIM_PROTECTED_PREFIX", "/etc")
        protected = f"{prefix}/shadow"
        command = str((event.get("tool_input") or {}).get("command", ""))
        # the naive rail: block only on a literal mention of the protected path.
        blocked = protected in command
        decision = {"hookSpecificOutput": {"permissionDecision": "ask"}} if blocked else {}
    except Exception:
        decision = {}  # fail open, like a real hook
    json.dump(decision, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
