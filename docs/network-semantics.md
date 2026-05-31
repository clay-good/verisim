# Network semantics (SPEC-5 v0)

> The normative English description of the SPEC-5 v0 network world, paired with the executable
> truth in [`src/verisim/netoracle/reference.py`](../src/verisim/netoracle/reference.py). Any
> disagreement between this document and the oracle is a bug in one of them, resolved by the
> golden trajectories ([`tests/test_net_goldens.py`](../tests/test_net_goldens.py)). This is the
> network analogue of [`docs/semantics.md`](./semantics.md).

This is the *smallest network world with the hard property* (compounding state, long-range
dependencies, combinatorial reachability) that still has a free, deterministic oracle — chosen
per SPEC-5 §17.1 to be "small enough to be correct and replayable, large enough to make the
science bite." TCP retransmit/congestion, packet-level timing, and routing tables are
deliberately **out of v0** (deferred enrichments); v0 models *connectivity and reachability*.

## State

A `NetworkState` is a typed graph plus two scalars:

- **hosts** — a fixed set of host ids, each with `up` (bool), `services` (a set of listening
  ports), and `fw_deny` (a set of source host ids whose *incoming* traffic is blocked).
- **links** — undirected host–host edges (the topology), each present (up) or absent.
- **flows** — established connections, each a `(src, dst, port)` triple.
- **clock** — a virtual integer clock advanced by `advance`.
- **last_exit** — the exit code of the most recent action (the observation).

The state is canonicalized (sorted hosts/services/links/flows) so equal states serialize
identically. The **reachability matrix** `R[(src, dst, port)]` is *derived* (not stored): the
operationally meaningful quantity (SPEC-5 §9.2).

## Reachability (the derived truth)

`src` can reach service `(dst, port)` iff **all** hold:

1. `src` and `dst` are up;
2. `dst` is listening on `port` (`port ∈ dst.services`);
3. `src` is **not** in `dst.fw_deny`;
4. there is a path from `src` to `dst` over **up** links between **up** hosts (BFS).

Loopback (`src == dst`) is reachable iff the host is up and listening on the port (no link
required). Firewall rules apply to *incoming* traffic at the destination, keyed by source host.

## Actions and their effects

The exit-code convention: **0** = valid (whether or not state changed); **1** = a well-defined
runtime failure; **2** = invalid arguments (unknown host, self-link). Every transition records
`last_exit`. Config ops are *idempotent* — applying one whose effect already holds is a
no-op success (exit 0, no state change).

| Action | Effect | Exit |
|---|---|---|
| `host_up <h>` / `host_down <h>` | set `h.up` | 0; 2 if `h` unknown |
| `link_up <a> <b>` / `link_down <a> <b>` | add / remove the undirected link `(a,b)` | 0; 2 if a host is unknown or `a==b` |
| `svc_up <h> <port>` / `svc_down <h> <port>` | add / remove `port` from `h.services` | 0; 2 if `h` unknown |
| `fw_deny <h> <src>` / `fw_allow <h> <src>` | add / remove `src` from `h.fw_deny` | 0; 2 if a host is unknown |
| `connect <src> <dst> <port>` | if reachable, establish flow `(src,dst,port)` | 0 if reachable (or already open); 1 if unreachable; 2 if a host is unknown |
| `close <src> <dst> <port>` | remove the flow if present | 0 if removed; 1 if no such flow |
| `advance` | clock += 1, then **drop every flow that is no longer reachable** | 0 |

## The one asynchronous dynamic: lazy flow drop on `advance`

Topology/firewall/service changes affect *reachability* immediately, but **established flows are
not dropped at the moment the path breaks** — they are dropped lazily, at the next `advance`,
when the oracle re-validates every flow against current reachability. This is the minimal
*temporally-extended* effect SPEC-5 §11 (wall **W5**) calls for: a `link_down` at step 5 silently
breaks a flow that is only observed as `CLOSED` at the next `advance`. It is the seed of the
delayed-consequence dynamics (route convergence, retransmit timers) that later phases enrich.

## Determinism

`step(state, action)` is a pure function of `(state, action)`: no clock, no RNG, no environment
leakage (all driver randomness lives in `netdata/drivers.py`, seeded). The oracle derives
`next_state` by applying its own delta (`apply(state, delta) == next_state`, the NW1 invariant),
so prediction and truth share one application semantics. A trajectory is a deterministic function
of `(config, driver, seed)` and regenerates identically from its manifest (SPEC-2 §12 discipline).
