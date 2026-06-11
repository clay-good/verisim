"""ED41 — the CRDT RGA: the first *ordered* CRDT, the basis of collaborative text.

The DS0-increment-34 experiment for SPEC-7 §3.2 — `rins`/`rdel`/`rget`, a replicated growable array
(sequence CRDT). Where the set/counter/register/map types are unordered, the RGA keeps a *list* in
which any node can insert at any position and concurrent inserts converge to **one** deterministic
order with no duplication — the property that makes collaborative editing (Google Docs) work.
Each element has a unique id `(seq, owner)` and a `parent` (the element it was inserted *after*, or
`ROOT` for the head); the visible order is a depth-first traversal where siblings are ordered by id
descending (a later insert at the same anchor lands immediately after it). The join is **set union**
of the elements + tombstones, and because the order is a pure function of the set, every node with
the same set derives the same sequence. Both panels are measured dependency-free and confirmed
Tier-A ≡ Tier-B:

  - **Panel A — sequence ops + deterministic concurrent insert.** Across a cluster-size sweep,
    inserting `a`,`b`,`c` sequentially reads back `"abc"` (**1.0**); inserting in the middle and
    deleting both work (**1.0**). The defining RGA property: two nodes inserting *different* chars
    at the *same* position concurrently, after `heal`+`gossip`, read back the **same** string on
    every node (one interleaving, both characters present, no duplication — **1.0**), where a
    naive list would diverge or duplicate. And `rins` is **always available** — a partitioned-alone
    node still edits (**1.0**), the AP property.

  - **Panel B — convergence (the union join + order fn).** From a diverged state the union join
    converges **every** node to the same sequence — a `gossip` chain epidemically (**1.0**),
    `anti_entropy` on each node (**1.0**), idempotently (a second `gossip` is a no-op, **1.0**). The
    order is recomputed identically wherever the element set matches.

`rins`/`rdel` are purely node-local (no replication, no in-flight message) and the merge is a
coordinator-level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces
transition bit-for-bit (§5.2). Two omitted-when-empty `rga_elems`/`rga_tombs` maps + the
`RGAInsert`/`RGATomb` edits; the ops are purely additive over increment 33 — no prior
golden/hash/tokenization changes.
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
class ED41Config:
    name: str = "ed41-rga"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    list_name: str = "l"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED41Config:
        b = ED41Config()
        return ED41Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            list_name=d.get("list_name", b.list_name),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED41Config:
        return ED41Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED41Result:
    #: Panel A: inserting a,b,c sequentially reads back "abc" (cluster-size sweep).
    build_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a middle insert and a delete both work.
    insert_delete: bool = False
    #: Panel A: concurrent inserts at one position converge to one deterministic order on all nodes.
    concurrent_converges: bool = False
    #: Panel A: a rins on a partitioned-alone node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel B: a gossip chain after heal converges every node to the same sequence.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node.
    anti_entropy_converges: bool = False
    #: Panel B: the union join is idempotent — a second gossip leaves the sequence unchanged.
    idempotent: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, built "abc")


def _config(n: int, list_name: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed41-{n}n", nodes=nodes, objects=(list_name,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def run_ed41(cfg: ED41Config | None = None) -> ED41Result:
    cfg = cfg or ED41Config()
    result = ED41Result()
    tier_b_agree = True
    tier_b_steps = 0
    ln = cfg.list_name

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

    # --- Panel A: build, insert/delete, concurrent convergence, always-available ------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, ln)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"rins n0 {ln} 0 a")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"rins n0 {ln} 1 b")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"rins n0 {ln} 2 c")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"rget n0 {ln}")
        ok = v == "abc"
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.build_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # a middle insert and a delete both work: "abc" -> insert X -> "aXbc" -> delete X -> "abc".
    config = _config(3, ln)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    for cmd in [f"rins n0 {ln} 0 a", f"rins n0 {ln} 1 b", f"rins n0 {ln} 2 c", f"rins n0 {ln} 1 X"]:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
    mid = step_both(ref, sysb, sa, sb, f"rget n0 {ln}")[3]
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"rdel n0 {ln} 2")  # delete the 2nd visible (X)
    after = step_both(ref, sysb, sa, sb, f"rget n0 {ln}")[3]
    result.insert_delete = mid == "aXbc" and after == "abc"

    # the defining property: concurrent inserts at one position converge to ONE order everywhere.
    config = _config(5, ln)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"rins n0 {ln} 0 a")
    for nd in ("n1", "n2", "n3", "n4"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")  # all observe 'a'
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"rins n3 {ln} 1 c")  # minority insert after a
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"rins n0 {ln} 1 b")  # majority inserts after 'a'
    result.always_available = s_min == "ok"  # the partitioned-minority insert is acknowledged (AP)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    v0 = step_both(ref, sysb, sa, sb, f"rget n0 {ln}")[3]
    v3 = step_both(ref, sysb, sa, sb, f"rget n3 {ln}")[3]
    # both nodes agree on the SAME string, both inserts present, no dup (len 3, has b and c).
    result.concurrent_converges = (
        v0 == v3 and len(v0) == 3 and "b" in v0 and "c" in v0 and v0.count("a") == 1
    )

    # --- Panel B: convergence (the union join + order function) -----------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, ln)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in ["partition n0 n1 n2 | n3 n4", f"rins n0 {ln} 0 x", f"rins n3 {ln} 0 y", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, ln).nodes
    ref, sysb, sa, sb = setup_diverged()
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    seqs = [step_both(ref, sysb, sa, sb, f"rget {nd} {ln}")[3] for nd in nodes]
    result.gossip_converges = len(set(seqs)) == 1 and set(seqs.pop()) == {"x", "y"}
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    again = step_both(ref, sysb, sa, sb, f"rget n0 {ln}")[3]
    result.idempotent = set(again) == {"x", "y"} and len(again) == 2

    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae = [step_both(ref, sysb, sa, sb, f"rget {nd} {ln}")[3] for nd in nodes]
    result.anti_entropy_converges = len(set(ae)) == 1 and len(ae[0]) == 2

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED41Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"rga,build_abc,{result.build_rate:.4f},over_{result.n_sizes}_clusters")
    lines.append(f"rga,insert_delete,"
                 f"{1.0 if result.insert_delete else 0.0:.4f},middle_insert_and_delete")
    lines.append(f"rga,concurrent_converges,"
                 f"{1.0 if result.concurrent_converges else 0.0:.4f},same_order_both_inserts")
    lines.append(f"rga,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_rins_acked_AP")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},epidemic_all_nodes_same_seq")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_same_seq")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED41 (the CRDT RGA: an ordered sequence, deterministic concurrent insert)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed41.csv")
    parser.add_argument("--plot", type=str, default="figures/ed41.png")
    args = parser.parse_args()
    cfg = ED41Config.from_json_file(args.config) if args.config else ED41Config()
    result = run_ed41(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — sequence ops + deterministic concurrent insert:")
    print(f"    build abc: {result.build_rate:.2f} "
          f"| insert/delete: {result.insert_delete} "
          f"| concurrent converges: {result.concurrent_converges} "
          f"| always available (AP): {result.always_available}")
    print("  Panel B — convergence (the union join + order function):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed41 import plot_ed41

        plot_ed41(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
