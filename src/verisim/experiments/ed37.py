"""ED37 — the CRDT OR-Set: add-wins, re-addable, convergent (the canonical interesting CRDT).

The DS0-increment-30 experiment for SPEC-7 §3.2 — `sadd`/`srem`/`smembers`, a *state-based*
observed-remove set. A replicated set is the CRDT a naive impl gets wrong: an element-level
2P-Set (a grow-only add-set + a grow-only remove-set) is **remove-wins** (a remove beats a
concurrent add) and can **never re-add** a removed element. The OR-Set fixes both with a **unique
dot**: `sadd n key elem` tags the element with `(owner=n, seq)` — a per-(key, owner) monotone
sequence — and stores it in `n`'s observed add-set; `srem n key elem` tombstones only the dots `n`
has *observed*; `smembers` is the elements with a non-tombstoned dot. The join is **set union** of
both halves (commutative, associative, idempotent). Both panels are measured dependency-free and
confirmed Tier-A ≡ Tier-B:

  - **Panel A — the OR-Set's defining wins.** Across a cluster-size sweep, `sadd`-ing `k` distinct
    elements then `smembers` reads back all `k` (**1.0**). A removed element is **re-addable** —
    `srem` then `sadd` makes a fresh dot, so the element returns (**1.0**), where a 2P-Set cannot.
    **Add wins** over a concurrent remove — an element present cluster-wide is re-added at `n0` (a
    fresh dot) while `n3` removes it (tombstoning only the dot it saw); after `heal`+`gossip` the
    element **survives** (**1.0**), where a 2P-Set would drop it. And `sadd` is **always available**
    — a partitioned-alone node still adds (**1.0**), the AP property.

  - **Panel B — convergence (the CRDT union join over both halves).** From a diverged state (one
    side adds a new element, the other removes an old one), the union join converges **every** node
    to the same member set: a `gossip` chain spreads it epidemically (**1.0**), and `anti_entropy`
    on each node reaches the same set (**1.0**). The join is **idempotent** — a second `gossip`
    leaves the set unchanged (**1.0**).

`sadd`/`srem` are purely node-local (no replication, no in-flight message) and the merge is a
coordinator-level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces
every transition bit-for-bit (§5.2). Two omitted-when-empty `orset_adds`/`orset_tombs` maps + the
`ORSetAdd`/`ORSetTomb` edits; the ops are purely additive — no prior golden/hash/tokenization
changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED37Config:
    name: str = "ed37-orset"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    key: str = "s"
    k: int = 3  # number of distinct elements added for Panel A

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED37Config:
        b = ED37Config()
        return ED37Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            k=int(d.get("k", b.k)),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED37Config:
        return ED37Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED37Result:
    #: Panel A: sadd k distinct elems then smembers reads back all k (cluster-size sweep).
    adds_read_rate: float = 0.0
    n_sizes: int = 0
    k: int = 0
    #: Panel A: a removed element is re-addable (a 2P-Set cannot).
    re_addable: bool = False
    #: Panel A: a concurrent add survives a concurrent remove (add-wins).
    add_wins: bool = False
    #: Panel A: a sadd on a partitioned-alone node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel B: a gossip chain after heal converges every node to the same member set.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node to the same member set.
    anti_entropy_converges: bool = False
    #: Panel B: the union join is idempotent — a second gossip leaves the set unchanged.
    idempotent: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, read back all k)


def _config(n: int, key: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed37-{n}n", nodes=nodes, objects=(key,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _members(value: str) -> set[str]:
    """Parse a ``smembers`` value (``{a,b}`` / ``{}``) back into a set."""
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


def run_ed37(cfg: ED37Config | None = None) -> ED37Result:
    cfg = cfg or ED37Config()
    result = ED37Result(k=cfg.k)
    tier_b_agree = True
    tier_b_steps = 0
    elems = [f"e{i}" for i in range(cfg.k)]

    def step_both(
        ref: ReferenceDistOracle, sysb: SystemDistOracle,
        sa: DistributedState, sb: DistributedState, cmd: str,
    ) -> tuple[DistributedState, DistributedState, str, str]:
        nonlocal tier_b_agree, tier_b_steps
        action = parse_dist_action(cmd)
        ra, rb = ref.step(sa, action), sysb.step(sb, action)
        if cluster_view(ra.state) != cluster_view(rb.state) or (ra.status, ra.value) != (
            rb.status, rb.value
        ):
            tier_b_agree = False
        tier_b_steps += 1
        return ra.state, rb.state, ra.status, ra.value

    # --- Panel A: read-back, re-addable, add-wins, always-available -------------------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for e in elems:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"sadd n0 {cfg.key} {e}")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"smembers n0 {cfg.key}")
        ok = _members(v) == set(elems)
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.adds_read_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # re-addable: srem then sadd brings the element back (a 2P-Set cannot).
    config = _config(3, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"sadd n0 {cfg.key} x")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"srem n0 {cfg.key} x")
    sa, sb, _, v_gone = step_both(ref, sysb, sa, sb, f"smembers n0 {cfg.key}")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"sadd n0 {cfg.key} x")
    sa, sb, _, v_back = step_both(ref, sysb, sa, sb, f"smembers n0 {cfg.key}")
    result.re_addable = "x" not in _members(v_gone) and "x" in _members(v_back)

    # add-wins: x present cluster-wide; partition; n0 re-adds (fresh dot) while n3 removes the dot
    # it observed. After heal+gossip the element survives — a 2P-Set would drop it (remove-wins).
    config = _config(5, cfg.key)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"sadd n0 {cfg.key} x")
    for nd in ("n1", "n2", "n3", "n4"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")  # all observe x's dot
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, s_add, _ = step_both(ref, sysb, sa, sb, f"sadd n0 {cfg.key} x")  # concurrent: fresh dot
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"srem n3 {cfg.key} x")  # concurrent: seen dot
    result.always_available = s_add == "ok"  # the add succeeded (and so did the partitioned remove)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    sa, sb, _, v0 = step_both(ref, sysb, sa, sb, f"smembers n0 {cfg.key}")
    sa, sb, _, v3 = step_both(ref, sysb, sa, sb, f"smembers n3 {cfg.key}")
    result.add_wins = "x" in _members(v0) and "x" in _members(v3)  # add survived the remove

    # --- Panel B: convergence (the union join over both halves) -----------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, cfg.key)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        # pre: x present cluster-wide; then partition, n0 adds a new elem, n3 removes x.
        for cmd in [f"sadd n0 {cfg.key} x", "anti_entropy n1", "anti_entropy n2", "anti_entropy n3",
                    "anti_entropy n4", "partition n0 n1 n2 | n3 n4", f"sadd n0 {cfg.key} a",
                    f"srem n3 {cfg.key} x", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, cfg.key).nodes
    target = {"a"}  # x removed on one side, a added on the other → union join converges to {a}
    # gossip chain: a forward sweep then a backward pass converges every node on the line.
    ref, sysb, sa, sb = setup_diverged()
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    sets = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"smembers {nd} {cfg.key}")
        sets.append(_members(v))
    result.gossip_converges = all(m == target for m in sets)
    # a second gossip pass is idempotent — the set does not change.
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    sa, sb, _, v_again = step_both(ref, sysb, sa, sb, f"smembers n0 {cfg.key}")
    result.idempotent = _members(v_again) == target

    # anti_entropy on each node also converges every node to the same set.
    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae_sets = []
    for nd in nodes:
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"smembers {nd} {cfg.key}")
        ae_sets.append(_members(v))
    result.anti_entropy_converges = all(m == target for m in ae_sets)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED37Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"orset,reads_back_k,{result.adds_read_rate:.4f},"
                 f"k={result.k}_over_{result.n_sizes}_clusters")
    lines.append(f"orset,re_addable,"
                 f"{1.0 if result.re_addable else 0.0:.4f},srem_then_sadd_returns")
    lines.append(f"orset,add_wins,"
                 f"{1.0 if result.add_wins else 0.0:.4f},concurrent_add_survives_remove")
    lines.append(f"orset,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_sadd_acked_AP")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},epidemic_all_nodes_same_set")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_same_set")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED37 (the CRDT OR-Set: add-wins, re-addable, convergent)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed37.csv")
    parser.add_argument("--plot", type=str, default="figures/ed37.png")
    args = parser.parse_args()
    cfg = ED37Config.from_json_file(args.config) if args.config else ED37Config()
    result = run_ed37(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — the OR-Set's defining wins:")
    print(f"    reads back k={result.k}: {result.adds_read_rate:.2f} "
          f"| re-addable: {result.re_addable} "
          f"| add-wins: {result.add_wins} "
          f"| always available (AP): {result.always_available}")
    print("  Panel B — convergence (the CRDT union join over both halves):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed37 import plot_ed37

        plot_ed37(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
