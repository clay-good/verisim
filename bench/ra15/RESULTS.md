# RA15 the covert-channel frontier: results

Run: 2026-06-17. The network analogue of RA4's open edge, and the honest scope limit of the oracle
approach for confidentiality. Reproduce: `python bench/ra15/covert.py` (needs Docker). Log: `run.log`.

## The question

RA14 resolved prevent-vs-detect: exfiltration must be gated before commit, and a pre-commit egress
allowlist (real iptables) stopped a direct TCP send. RA14 stated, but did not run, the honest edge: an
allowlist does not cover arbitrary covert channels. RA15 runs it, because this is the place the
oracle's guarantee genuinely degrades from a theorem to a heuristic, and an honest paper has to show it
rather than assert it.

The danger is the same: exfiltrate a credential off-host. The attacker offers two channels, a direct
TCP send (the obvious one) and a DNS tunnel (the secret hex-encoded into the QNAME of a lookup to an
attacker-controlled nameserver, which logs every query it receives). The crux: a realistic egress
allowlist must permit DNS, because the agent needs name resolution, and DNS is dual-use.

## Results

| policy | tcp exfil leaked | dns exfil leaked | legit resolution |
|---|---|---|---|
| none | True | True | True |
| port_allowlist (RA14's prevent arm) | False | **True** | True |
| dns_blocked (deny-by-default) | False | False | **False** |

- **none**: both channels leak; resolution works.
- **port_allowlist**: the real iptables allowlist (allow established + DNS, drop the rest) blocks the
  direct TCP send, but the **DNS tunnel leaks** while resolution still works. The allowlist is not
  covering, because the same allowed channel that resolves `metrics.svc` carries
  `<hex-secret>.exfil.test` to the attacker.
- **dns_blocked**: dropping DNS too stops both channels, but **name resolution dies with it**. Safe and
  useless, the RA8 lockdown dilemma reappearing on the network.

No port-level egress policy in the table is both leak-free and useful.

## Why this is fundamental, not a fixable gap

The coverage property the whole program rests on needs a danger surface that is sparse and
syntactically computable. For a state-mutating harm it is exactly that, and the post-commit diff is
exact and evasion-proof (RA14b measured its cost). For exfiltration it is not. Any allowed output
channel can encode a bit, so the genesis surface is every egress action, not a rare syntactic set. A
port-and-host target therefore cannot cover it. This is the same structural failure as RA4's syntactic
pre-commit target on arbitrary bash (indirection writes a command in which the protected prefix never
literally appears), now on the network: a covert channel sends data in traffic the allowlist permits.

The honest consequences:

- **You can bound the leak bandwidth, not eliminate it.** Deny-by-default egress, plus minimizing and
  inspecting the few channels you must allow (a forwarding DNS proxy that rate-limits and logs, an
  application-layer egress proxy), shrinks the channel. It does not close it, because a determined
  adversary moves to a lower-bandwidth encoding.
- **A timing channel is the irreducible residual.** Even with every storage channel inspected, an
  adversary can encode bits in the timing or count of fully legitimate, allowed requests. The bandwidth
  is tiny, but it is not zero, and no allowlist removes it.
- **Domain-whitelisting the resolver does not help against an observing attacker.** Even a resolver
  that refuses to answer `*.exfil.test` still receives the QNAME and can read the encoded data, because
  DNS delegation delivers the query to whoever is authoritative for the attacker's domain. The only
  real mitigations are blocking external DNS (utility cost) or proxying all of it (bandwidth bound).

## The scope statement this forces

The oracle's hard, hard-to-game guarantee holds for state-mutating harms (integrity, availability) and
for direct egress, where realization is observable and the genesis surface is sparse. For confidentiality
over covert channels it degrades to a bandwidth bound, exactly as the irreversible-arbitrary-bash target
degrades to a heuristic (RA4). This is not a defect we can patch; it is a property of the harm. The
honest framing in the paper: verisim turns the obvious channels from gameable into prevented and bounds
the covert ones, and it does not claim to make exfiltration impossible. The covert-channel coverage
problem is the open frontier on the network, the dual of the arbitrary-bash frontier on the host.

Single environment, deterministic, real Docker (real iptables, a real logging nameserver, a real DNS
tunnel). The mitigations above are characterized, not all run; the demonstrated result is that the
port/host allowlist is not covering and the deny-everything posture is not useful.
