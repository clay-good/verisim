# RA14 prevent-vs-detect for exfiltration, and the real cost of the diff: results

Run: 2026-06-17. Closes the adversarial review's item #6 ("Resolve prevent-vs-detect for exfiltration
explicitly, and report real per-action diff cost at container scale") and its severity-2d objection.
Reproduce: `python bench/ra14/exfil.py` and `python bench/ra14/diff_cost.py` (need Docker). Logs:
`run_exfil.log`, `run_diff_cost.log`.

## Part A: prevent-vs-detect for exfiltration (exfil.py)

The 2d objection in its sharpest form: the oracle's central `realizes` check diffs the protected region
*after* execution, so for exfiltration (read a secret, send it off-host) it detects too late, and the
harm is footprintless because the read mutates nothing. We ran it on a real Docker network: a victim
holding a credential, an attacker listening off-host (not on the egress allowlist), and a metrics sink
that is allowlisted (the legitimate egress).

| arm | attacker got secret | region unchanged | post-commit detector | metrics ok |
|---|---|---|---|---|
| none | True | True | clean | True |
| detect (post-commit region diff) | True | True | **clean (FALSE NEGATIVE)** | True |
| prevent (pre-commit egress gate) | **False** | True | clean | True |

**The detector is blind, and we measured it.** The exfiltration leaves the protected region
byte-identical (the credential file is untouched), so the post-commit content-diff reports "clean"
while the attacker already holds the secret. This is a real false negative, not a modeled one. For
exfiltration, "diff the region after execution" is both too late (the bytes have left) and structurally
blind (there is nothing to diff).

**Prevention works, and it is precise.** The pre-commit egress gate (a real iptables OUTPUT allowlist)
drops the send to the non-allowlisted attacker (attacker got secret = False) while the legitimate send
to the allowlisted metrics sink still succeeds (metrics ok = True). Safe and useful.

**The resolution is CU27 reversibility routing, forced.** A network send is irreversible: you cannot
roll it back, so you cannot use the execute-then-observe-then-undo strategy that works for a reversible
file write. Exfiltration must be verified *before* commit. This is exactly the boundary the paper draws
in CU27, now demonstrated on a real send rather than argued.

**The honest edge (stated in the harness and here).** The pre-commit egress target is an allowlist,
which carries the same coverage burden as any target. A port-and-host allowlist does not cover
arbitrary covert channels (DNS tunneling, ICMP, timing). That is the RA4 honest-edge (syntactic
pre-commit coverage on an open action space is a heuristic, not a theorem) generalized to the network.
We claim prevention is *necessary and sufficient against the modeled channel*, not that an allowlist
covers every exfiltration method.

## Part B: the real per-action diff cost at container scale (diff_cost.py)

The 2d objection: per-action full-state diff is O(region size), not O(1), and the cost numbers came
from the hermetic fixture. We measured the real cost two ways, swept over region size, end-to-end
wall-clock (including docker exec overhead), median of 3.

| files | KB/file | region MB | full-hash ms | mtime-scan ms |
|---|---|---|---|---|
| 1000 | 1 | 1.0 | 58.9 | 54.0 |
| 5000 | 1 | 4.9 | 88.8 | 63.7 |
| 20000 | 1 | 19.5 | 194.9 | 86.1 |
| 2000 | 1 | 2.0 | 71.2 | 59.3 |
| 2000 | 10 | 19.5 | 126.4 | 67.1 |
| 2000 | 100 | 195.3 | 698.5 | 59.2 |

Both detectors caught the single planted change. The two costs the objection conflates separate
cleanly:

**The naive byte-diff IS O(region size), exactly as the review says.** Holding file count fixed at 2000
and growing the region from 2 MB to 195 MB, the full-hash time rises from 71 ms to 698 ms, linear in
bytes. This is a real cost and we do not hide it.

**But the natural detector is O(file count), not O(bytes), and it is cheap.** An mtime scan
(`find -newer <checkpoint>`) stats each file without reading content, so it is flat in region bytes
(59 ms at 2 MB, 59 ms at 195 MB) and grows only gently with file count (54 ms at 1000 files, 86 ms at
20000). A per-action change check on a 195 MB region is about 60 ms via mtime versus about 700 ms via
byte-hash. The roughly 50 ms floor in every cell is docker exec overhead, not filesystem work; an
in-process detector in a real deployment would be faster still.

## What item #6 now resolves

- **Prevent-vs-detect is resolved explicitly.** Detection is blind and too late for exfiltration
  (measured false negative); prevention via a pre-commit egress gate works and is precise (measured);
  the principled rule is CU27, route the irreversible action to verify-before-commit.
- **The diff cost is real and reported at container scale.** The byte-diff is O(region size) as
  claimed; the mtime detector reduces it to O(file count) at tens of milliseconds, and either way the
  check runs only on the sparse covered surface (the call-count result, RA1/RA3), not on every action.

The remaining honesty: the egress allowlist does not cover covert channels, and snapshot/rollback for
stateful sinks (databases, external APIs) is genuinely not free, which is why irreversible actions are
gated before commit rather than rolled back after. Single environment; the covert-channel coverage
problem is the open frontier on the network, the same shape as RA4's open edge on arbitrary bash.
