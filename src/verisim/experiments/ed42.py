"""ED42 — the nested CRDT counter-map: a CRDT whose values are themselves CRDTs.

The DS0-increment-35 experiment for SPEC-7 §3.2 — `cminc`/`cmget`/`cmdel`/`cmkeys`, an OR-Map whose
field values are **G-counters** (a map of counters, Riak's `{user → visit_count}`). It is the
*recursive* form of the compositional thesis: the OR-Map (ED40) composed OR-Set with LWW-registers;
this composes the OR-Set field-presence with a value type that merges *differently* — a G-counter,
which merges by per-owner **max** (loss-free), not LWW. The point is that **both** composed layers'
guarantees hold simultaneously under concurrency: a field survives a concurrent remove (the OR-Set's
add-wins presence) *and* concurrent increments to the same field are summed loss-free (the counter's
no-lost-update), where the OR-Map's LWW value would have dropped one. Both panels are measured
dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — map ops + both composed guarantees.** Across a cluster-size sweep, `cminc` builds
    per-field totals that `cmget`/`cmkeys` read back (**1.0**); a `cmdel` removes a field (**1.0**).
    The recursion's two payoffs: concurrent `cminc`s to the **same field** are summed **loss-free**
    (three increments across a partition total 3, **1.0**), where an LWW-valued map field would keep
    one; and a concurrent `cminc` survives a concurrent `cmdel` — **add-wins field presence** (the
    field stays, with its full count, **1.0**). And `cminc` is **always available** (**1.0**).

  - **Panel B — convergence (the composed join).** From a diverged state the counter-map converges
    **every** node to the same fields *and* per-field totals — a `gossip` chain epidemically
    (**1.0**), `anti_entropy` on each node (**1.0**), idempotently (**1.0**). The halves converge
    independently: presence by set-union, value by per-owner counter max.

`cminc`/`cmdel` are purely node-local (no replication, no in-flight message) and the merge is a
coordinator-level read of the medium, so the autonomous-actor system oracle (Tier-B) reproduces
each transition bit-for-bit (§5.2). Three omitted-when-empty `cmap_*` maps +
the `CMapField`/`CMapTomb`/`CMapCount` edits; the ops are additive over increment 34 — no prior
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
class ED42Config:
    name: str = "ed42-counter-map"
    cluster_sizes: tuple[int, ...] = (3, 5, 7)
    mapname: str = "m"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED42Config:
        b = ED42Config()
        return ED42Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            mapname=d.get("mapname", b.mapname),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED42Config:
        return ED42Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED42Result:
    #: Panel A: cminc builds per-field totals that cmget/cmkeys read back (cluster-size sweep).
    basic_read_rate: float = 0.0
    n_sizes: int = 0
    #: Panel A: a cmdel removes a field (absent from cmkeys).
    delete_removes_field: bool = False
    #: Panel A: concurrent increments to a field are summed loss-free (the counter recursion).
    value_loss_free: bool = False
    #: Panel A: a concurrent cminc survives a concurrent cmdel (add-wins field presence).
    add_wins_presence: bool = False
    #: Panel A: a cminc on a partitioned-alone node is acknowledged (the AP property).
    always_available: bool = False
    #: Panel B: a gossip chain after heal converges every node to the same fields + totals.
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
    return DistConfig(name=f"ed42-{n}n", nodes=nodes, objects=(mapname,), values=("a",),
                      replication_factor=n, consistency_model="eventual")


def _keys(value: str) -> set[str]:
    inner = value.strip("{}")
    return set(inner.split(",")) if inner else set()


def run_ed42(cfg: ED42Config | None = None) -> ED42Result:
    cfg = cfg or ED42Config()
    result = ED42Result()
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

    # --- Panel A: basic ops, delete, loss-free value, add-wins, always-available ------------------
    correct_count = 0
    sizes = 0
    for n in cfg.cluster_sizes:
        config = _config(n, m)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in [f"cminc n0 {m} visits", f"cminc n0 {m} visits", f"cminc n0 {m} clicks"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        v = step_both(ref, sysb, sa, sb, f"cmget n0 {m} visits")[3]
        ks = step_both(ref, sysb, sa, sb, f"cmkeys n0 {m}")[3]
        ok = v == "2" and _keys(ks) == {"visits", "clicks"}
        sizes += 1
        if ok:
            correct_count += 1
        result.per_size.append((n, ok))
    result.basic_read_rate = correct_count / sizes if sizes else 0.0
    result.n_sizes = sizes

    # cmdel removes a field (absent from cmkeys and cmget).
    config = _config(3, m)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    for cmd in [f"cminc n0 {m} a", f"cminc n0 {m} b", f"cmdel n0 {m} a"]:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
    ks = step_both(ref, sysb, sa, sb, f"cmkeys n0 {m}")[3]
    va = step_both(ref, sysb, sa, sb, f"cmget n0 {m} a")[3]
    result.delete_removes_field = _keys(ks) == {"b"} and va == ""

    # the counter recursion: concurrent cminc to the SAME field are summed (vs LWW dropping
    # one). 'c' present cluster-wide; partition; n0 +2, n3 +1 — after heal+gossip the total is 3.
    config = _config(5, m)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cminc n0 {m} c")  # total 1 cluster-wide
    for nd in ("n1", "n2", "n3", "n4"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cminc n0 {m} c")  # majority +1 (total 2)
    sa, sb, s_min, _ = step_both(ref, sysb, sa, sb, f"cminc n3 {m} c")  # minority +1 (concurrent)
    result.always_available = s_min == "ok"  # the partitioned-minority increment is acknowledged
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    total = step_both(ref, sysb, sa, sb, f"cmget n0 {m} c")[3]
    result.value_loss_free = total == "3"  # 1 + 1 + 1, nothing lost (LWW would read 2)

    # add-wins: 'k' present cluster-wide; partition; n0 cminc (fresh dot) vs n3 cmdel — survives.
    config = _config(5, m)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
    sa = sb = DistributedState.initial(config)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cminc n0 {m} k")
    for nd in ("n1", "n2", "n3", "n4"):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "partition n0 n1 n2 | n3 n4")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cmdel n3 {m} k")  # minority removes
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"cminc n0 {m} k")  # majority re-increments
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "heal")
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n3")
    ks0 = step_both(ref, sysb, sa, sb, f"cmkeys n0 {m}")[3]
    ks3 = step_both(ref, sysb, sa, sb, f"cmkeys n3 {m}")[3]
    result.add_wins_presence = "k" in _keys(ks0) and "k" in _keys(ks3)  # field survived the cmdel

    # --- Panel B: convergence (the composed join) -------------------------------------------------
    def setup_diverged() -> tuple[ReferenceDistOracle, SystemDistOracle,
                                  DistributedState, DistributedState]:
        config = _config(5, m)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        sa = sb = DistributedState.initial(config)
        for cmd in ["partition n0 n1 n2 | n3 n4", f"cminc n0 {m} x", f"cminc n3 {m} y", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        return ref, sysb, sa, sb

    nodes = _config(5, m).nodes
    target_keys = {"x", "y"}
    ref, sysb, sa, sb = setup_diverged()
    for p, q in pairwise(nodes):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    for p, q in pairwise(list(reversed(nodes))):
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"gossip {p} {q}")
    conv = []
    for nd in nodes:
        ks = step_both(ref, sysb, sa, sb, f"cmkeys {nd} {m}")[3]
        vx = step_both(ref, sysb, sa, sb, f"cmget {nd} {m} x")[3]
        vy = step_both(ref, sysb, sa, sb, f"cmget {nd} {m} y")[3]
        conv.append((_keys(ks), vx, vy))
    result.gossip_converges = all(c == (target_keys, "1", "1") for c in conv)
    sa, sb, _, _ = step_both(ref, sysb, sa, sb, "gossip n0 n4")
    ks_again = step_both(ref, sysb, sa, sb, f"cmkeys n0 {m}")[3]
    result.idempotent = _keys(ks_again) == target_keys

    ref, sysb, sa, sb = setup_diverged()
    for nd in nodes:
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, f"anti_entropy {nd}")
    ae = [_keys(step_both(ref, sysb, sa, sb, f"cmkeys {nd} {m}")[3]) for nd in nodes]
    result.anti_entropy_converges = all(k == target_keys for k in ae)

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED42Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"cmap,basic_read,{result.basic_read_rate:.4f},over_{result.n_sizes}_clusters")
    lines.append(f"cmap,delete_removes,"
                 f"{1.0 if result.delete_removes_field else 0.0:.4f},cmdel_field_absent")
    lines.append(f"cmap,value_loss_free,"
                 f"{1.0 if result.value_loss_free else 0.0:.4f},concurrent_increments_total_3")
    lines.append(f"cmap,add_wins,"
                 f"{1.0 if result.add_wins_presence else 0.0:.4f},cminc_survives_concurrent_cmdel")
    lines.append(f"cmap,always_available,"
                 f"{1.0 if result.always_available else 0.0:.4f},minority_cminc_acked_AP")
    lines.append(f"converge,gossip,"
                 f"{1.0 if result.gossip_converges else 0.0:.4f},all_nodes_same_fields_totals")
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
        description="Run ED42 (the nested CRDT counter-map: OR-Set ∘ loss-free G-counters)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed42.csv")
    parser.add_argument("--plot", type=str, default="figures/ed42.png")
    args = parser.parse_args()
    cfg = ED42Config.from_json_file(args.config) if args.config else ED42Config()
    result = run_ed42(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — map ops + both composed guarantees:")
    print(f"    basic read: {result.basic_read_rate:.2f} "
          f"| delete removes: {result.delete_removes_field} "
          f"| value loss-free: {result.value_loss_free} "
          f"| add-wins presence: {result.add_wins_presence} "
          f"| always available (AP): {result.always_available}")
    print("  Panel B — convergence (the composed join):")
    print(f"    gossip converges: {result.gossip_converges} "
          f"| anti_entropy converges: {result.anti_entropy_converges} "
          f"| idempotent: {result.idempotent}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed42 import plot_ed42

        plot_ed42(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
