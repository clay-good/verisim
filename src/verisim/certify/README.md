# verisim-certify — prove your agent guardrail's completeness, don't assert it

Point it at the guardrail you already shipped — a Claude Code `PreToolUse` hook or a denylist — and it
returns **the concrete commands that realize harm off your checked surface**, each oracle-confirmed,
plus a **coverage certificate** with a confidence interval. It is the measurement between the
red-teamers (who *find* prompt attacks) and the guardrails (who *assert* completeness): it tells you,
with proof, whether your tool-call rail is actually complete.

Part of [Verisim](../../../README.md); the vision is [SPEC-28](../../../docs/specs/SPEC-28.md).

## Quick start

```bash
# audit a deployed PreToolUse hook (a script that speaks the Claude Code contract)
python -m verisim.certify audit --hook ./my_hook.py --report cert.md

# audit a denylist directly, no hook script needed
python -m verisim.certify audit --denylist "/etc/shadow,rm -rf /etc" --report cert.md

# prove each bypass against a real /bin/sh in a sandbox (the strongest evidence)
python -m verisim.certify audit --hook ./my_hook.py --oracle real

# use both proposers (systematic + adaptive); write the machine-readable certificate
python -m verisim.certify audit --hook ./my_hook.py --proposer both --out cert.json
```

The CLI exits **nonzero if it finds a bypass**, so it drops straight into CI as a coverage gate.

## What a "bypass" is

A command that (1) the oracle confirms **realizes the harm** (it changes the protected file), that (2)
your guardrail **allows** (off its checked surface), and that (3) is **in your guardrail's contract**
(a genuine gap, not the symlink/runtime residual a guardrail explicitly defers). That is the thing you
act on.

## Bring your own guardrail

**A hook.** Any script that reads a PreToolUse event as JSON on stdin and prints a decision on stdout:

```
stdin :  {"tool_name": "Bash", "tool_input": {"command": "<cmd>"}, "cwd": "<dir>"}
stdout:  {"hookSpecificOutput": {"permissionDecision": "ask"}}      # ask/deny == covered
         {}                                                         # allow == NOT covered
```

Point `--hook` at it. The certifier passes the protected region in `VERISIM_PROTECTED_PREFIX`.

**A custom guardrail (not a hook).** Implement the one-method `Monitor` protocol and call the core
directly — no subprocess:

```python
from verisim.certify import certify_monitor
from verisim.audit.oracles import ShellPathOracle

class MyGuardrail:
    name = "my-guardrail"
    def covers(self, action, ctx=None):
        return my_policy_would_block(action.command)   # True == I would block/ask

res = certify_monitor(MyGuardrail(), ShellPathOracle("/etc"), protected_path="/etc/shadow")
print(res.verdict_line())
for h in res.bypasses:
    print(h.klass, h.command)
```

## What it certifies (and what it does not)

- **Certifies:** integrity / state-change harm on a protected path, over a sampled shell-encoding
  algebra, with a stated residual bound. Strong and cheap, because the dangerous surface is sparse.
- **Does not certify:** confidentiality / exfiltration — there is no sparse surface to cover, so it is
  **out of scope by design**. The certificate says so. The result is empirical over the sampled space,
  not a proof over all of bash; read the certificate's upper bound as the honest residual.

## Output

A terminal summary, a machine-readable certificate (`--out cert.json`), and a human-readable Markdown
report (`--report cert.md`) with coverage %, a Wilson confidence interval, the bypasses by class, and
an [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
mapping (ASI02 Tool Misuse, ASI03 Privilege Abuse, ASI05 Unexpected Code Execution) for a compliance
file.
