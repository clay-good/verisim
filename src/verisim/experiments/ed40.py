"""ED40 — the CRDT OR-Map: a CRDT *of* CRDTs, the compositional capstone of the family.

The DS0-increment-33 experiment for SPEC-7 §3.2 — `mput`/`mget`/`mdel`/`mkeys`, an add-wins
observed-remove map. It is the capstone because it is a CRDT **composed of** two earlier CRDTs:
the **OR-Set** (ED37) governs *field presence* (which fields the map has, add-wins + observed-remove
over field names), and the **LWW-register** (ED39) governs each field's *value*. `mput n map field
val` adds a fresh presence dot for `field` *and* LWW-writes `val`; `mdel` observed-removes it;
`mget` reads a present field's value; `mkeys` enumerates the present fields — the map capability the
flat KV and registers lack. The join is the OR-Set union of the presence halves plus the LWW max of
each field's value (sharing the Lamport clock). Both panels are dependency-free, confirmed
Tier-A ≡ Tier-B:

  - **Panel A — map operations + the two composed semantics.** Across a cluster-size sweep, `mput`
    then `mget`/`mkeys` reads the field and value back (**1.0**); a `mdel` removes the field (absent
    from `mkeys`, **1.0**). The composition shows in two concurrency outcomes: a concurrent value
    update to the same field resolves by **LWW** (one winner, **1.0**), while a concurrent `mput`
    survives a concurrent `mdel` — **add-wins field presence** (the field stays, **1.0**), where a
    naive map would lose the update. And `mput` is **always available** (**1.0**), the AP property.

  - **Panel B — convergence (the composed join).** From a diverged state the OR-Map converges
    **every** node to the same fields *and* per-field values — `gossip` epidemically (**1.0**),
    `anti_entropy` on each node (**1.0**), idempotently (a second `gossip` is a no-op, **1.0**). The
    two halves converge independently and compose: field presence by set-union, field value by LWW.

`mput`/`mdel` are purely node-local (no replication, no in-flight message) and the merge is a
coordinator-level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces
each transition bit-for-bit (§5.2). Three omitted-when-empty `ormap_*` maps
+ the `ORMapField`/`ORMapTomb`/`ORMapVal` edits (sharing the Lamport clock); the ops are purely
additive over increment 32 — no prior golden/hash/tokenization changes.
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
class ED40Config:
    name: str = "ed40-ormap"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    mapname: str = "m"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED40Config:
        b = ED40Config()
        return ED40Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            mapname=d.get("mapname", b.mapname),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED40Config:
        return ED40Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED40Result:
    #: Panel A: mput then mget/mkeys reads the field+value back (cluster-size sweep).
    basic_read_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a mdel removes the field (absent from mkeys).
    delete_removes_field: bool = False
    #: Panel A: a concurrent value update to a field resolves by LWW (one winner).
    value_resolves_lww: bool = False
    #: Panel A: a concurrent mput survives a concurrent mdel (add-wins field presence).
    add_wins_presence: bool = False
    #: Panel A: a mput on a partitioned-alone node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel B: a gossip chain after heal converges every node to the same fields + values.
    gossip_converges: bool = False
    #: Panel B: anti_entropy on each node after heal converges every node.
    anti_entropy_converges: bool = False
    #: Panel B: the composed join is idempotent — a second gossip leaves the map unchanged.
    idempotent: bool = False
    #: Tier-B reproduces every transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, bool]] = field(default_factory=list)  # (n, basic read ok)


def _config(n: int, mapname: str) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(name=f"ed40-{n}n", nodes=nodes, objects=(mapname,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _keys(value: str) -> set[str]:
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


def run_ed40(cfg: ED40Config | None = None) -> ED40Result:
    cfg = cfg or ED40Config()
    result = ED40Result()
    tier_b_agree = True
    tier_b_steps = 0
    m = cfg.mapname

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

    # --- Panel A: basic ops, delete, value-LWW, add-wins, always-available ------------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, m)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} name alice")
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} age 30")
        sa, sb, _, v = step_both(ref, sysb, sa, sb, f"mget n0 {m} name")
        sa, sb, _, ks = step_both(ref, sysb, sa, sb, f"mkeys n0 {m}")
        ok = v == "alice" and _keys(ks) == {"name", "age"}
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.basic_read_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # mdel removes a field (absent from mkeys and mget).
    config = _config(3, m)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} a 1")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} b 2")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mdel n0 {m} a")
    sa, sb, _, ks = step_both(ref, sysb, sa, sb, f"mkeys n0 {m}")
    sa, sb, _, va = step_both(ref, sysb, sa, sb, f"mget n0 {m} a")
    result.delete_removes_field = _keys(ks) == {"b"} and va == ""

    # concurrent VALUE update to the same field resolves by LWW (n3 > n0 at equal ts).
    config = _config(5, m)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} k a")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n3 {m} k b")  # concurrent value
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    vk0 = step_both(ref, sysb, sa, sb, f"mget n0 {m} k")[3]
    vk3 = step_both(ref, sysb, sa, sb, f"mget n3 {m} k")[3]
    result.value_resolves_lww = vk0 == vk3 == "b"  # owner n3 wins the LWW tie

    # add-wins: field present cluster-wide; partition; n0 mput (fresh dot) vs n3 mdel — survives.
    config = _config(5, m)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} k v0")
    for nd in ("n1", "n2", "n3", "n4"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")  # all observe k's dot
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"mdel n3 {m} k")  # minority removes
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"mput n0 {m} k v1")  # majority re-writes
    result.always_available = s_min == "ok"  # the partitioned-minority op is acknowledged (AP)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    ks0 = step_both(ref, sysb, sa, sb, f"mkeys n0 {m}")[3]
    ks3 = step_both(ref, sysb, sa, sb, f"mkeys n3 {m}")[3]
    result.add_wins_presence = "k" in _keys(ks0) and "k" in _keys(ks3)  # field survived the mdel

    # --- Panel B: convergence (the composed join) -------------------------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, m)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        # one side adds field x; the other adds field y and sets a value — both must converge.
        for cmd in ["partition n0 n1 n2 | n3 n4", f"mput n0 {m} x 1", f"mput n3 {m} y 2", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, m).nodes
    target_keys = {"x", "y"}
    ref, sysb, sa, sb = setup_diverged()
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    converged = []
    for nd in nodes:
        ks = step_both(ref, sysb, sa, sb, f"mkeys {nd} {m}")[3]
        vx = step_both(ref, sysb, sa, sb, f"mget {nd} {m} x")[3]
        vy = step_both(ref, sysb, sa, sb, f"mget {nd} {m} y")[3]
        converged.append((_keys(ks), vx, vy))
    result.gossip_converges = all(c == (target_keys, "1", "2") for c in converged)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    ks_again = step_both(ref, sysb, sa, sb, f"mkeys n0 {m}")[3]
    result.idempotent = _keys(ks_again) == target_keys

    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae = []
    for nd in nodes:
        ks = step_both(ref, sysb, sa, sb, f"mkeys {nd} {m}")[3]
        ae.append(_keys(ks))
    result.anti_entropy_converges = all(k == target_keys for k in ae)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED40Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"ormap,basic_read,{result.basic_read_rate:.4f},over_{result.n_sizes}_clusters")
    lines.append(f"ormap,delete_removes,"
                 f"{1.0 if result.delete_removes_field else 0.0:.4f},mdel_field_absent")
    lines.append(f"ormap,value_lww,"
                 f"{1.0 if result.value_resolves_lww else 0.0:.4f},concurrent_value_one_winner")
    lines.append(f"ormap,add_wins,"
                 f"{1.0 if result.add_wins_presence else 0.0:.4f},mput_survives_concurrent_mdel")
    lines.append(f"ormap,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_op_acked_AP")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},all_nodes_same_fields_values")
    lines.append(f"converge,anti_entropy,"
                 f"{1.0 if result.anti_entropy_converges else 0.0:.4f},all_nodes_same_fields")
    lines.append(f"converge,idempotent,"
                 f"{1.0 if result.idempotent else 0.0:.4f},second_gossip_unchanged")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED40 (the CRDT OR-Map: a CRDT of CRDTs, add-wins + LWW, convergent)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed40.csv")
    parser.add_argument("--plot", type=str, default="figures/ed40.png")
    args = parser.parse_args()
    cfg = ED40Config.from_json_file(args.config) if args.config else ED40Config()
    result = run_ed40(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — map operations + the two composed semantics:")
    print(f"    basic read: {result.basic_read_rate:.2f} "
          f"| delete removes: {result.delete_removes_field} "
          f"| value LWW: {result.value_resolves_lww} "
          f"| add-wins presence: {result.add_wins_presence} "
          f"| always available (AP): {result.always_available}")
    print("  Panel B — convergence (the composed join):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed40 import plot_ed40

        plot_ed40(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
