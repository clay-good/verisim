"""ED23 — leader election with terms: no split-brain, and the fence plain quorum lacks.

The DS0-increment-16 experiment for SPEC-7 §3.2 (the consensus family) — the Raft-subset
`elect`/`propose` core. `elect node` makes `node` the leader **iff its partition side holds a
strict majority of the live nodes**, bumping a monotone `term`; `propose node key val` is a
**leader-fenced** majority write. Two safety properties plain `quorum` writes lack, measured
dependency-free and confirmed Tier-A ≡ Tier-B:

  - **Panel A — no split-brain leadership.** Across a sweep of clusters split into a minority | a
    majority, the **minority side can never elect** (rejected `no_quorum`, rate **1.0**) while the
    majority side always can (`elected`, rate **1.0**) — so at most one leader exists across the
    cluster, structurally. The even-split edge (a `2 | 2` in a 4-node cluster) is the sharpest case:
    *neither* side is a strict majority, so **neither can elect** — the cluster has no leader rather
    than two, the CAP-availability price consensus pays to never fork.

  - **Panel B — term-fencing / leader completeness (the property `quorum` lacks).** A leader is
    partitioned into the minority; the majority side elects a new leader (a higher term). After the
    partition **heals**, the deposed leader's `propose` is **rejected** (`not_leader`, fenced rate
    **1.0**), because the global leader already moved on — whereas a plain `put` by that same stale
    coordinator **still commits** (rate **1.0**), the stale write the fence exists to stop. The new
    leader commits (rate **1.0**). This is the Raft leader-completeness guarantee: an old leader
    cannot commit after a new term, *even once the network is whole again* — which a leaderless
    `quorum` write, available to any coordinator that can reach a majority, cannot provide.

Both `elect` and `propose` are coordinator-level consensus decisions (the quorum is read from the
medium, not an actor's local view), so the autonomous-actor system oracle (Tier-B) computes
byte-identical leader/term/replica deltas and reproduces the fencing bit-for-bit (the W1 retirement,
§5.2). The consensus metadata is omitted from the canonical form until the first election, so the
`elect`/`propose` action family is purely additive — no prior golden, hash, or tokenization changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.dist.action import parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.differential import cluster_view
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.distoracle.system import SystemDistOracle


@dataclass(frozen=True)
class ED23Config:
    name: str = "ed23-consensus"
    #: cluster sizes for the split-brain sweep (odd → a clean minority|majority; 4 → the even edge).
    cluster_sizes: tuple[int, ...] = (3, 4, 5, 7)
    key: str = "x"
    v_old: str = "b"  # the value the deposed leader tries to (over)write after heal
    v_new: str = "c"  # the value the legitimate new leader commits

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED23Config:
        b = ED23Config()
        return ED23Config(
            name=d.get("name", b.name),
            cluster_sizes=tuple(d.get("cluster_sizes", b.cluster_sizes)),
            key=d.get("key", b.key),
            v_old=d.get("v_old", b.v_old),
            v_new=d.get("v_new", b.v_new),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED23Config:
        return ED23Config.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ED23Result:
    #: Panel A: fraction of minority-side elections correctly rejected (split-brain blocked).
    minority_elect_blocked_rate: float = 0.0
    #: Panel A: fraction of majority-side elections that correctly succeed.
    majority_elect_rate: float = 0.0
    #: Panel A: even-split clusters where *no* side can elect (a 2|2 leaves the cluster leaderless).
    even_split_leaderless_rate: float = 0.0
    n_split_scenarios: int = 0
    n_even_scenarios: int = 0
    #: Panel B: the deposed leader's propose is fenced after heal (the property quorum lacks).
    deposed_propose_fenced: bool = False
    #: Panel B: a plain put by that same stale coordinator still commits (what the fence prevents).
    unfenced_put_commits: bool = False
    #: Panel B: the legitimate new leader commits.
    new_leader_commits: bool = False
    #: Tier-B reproduces every elect/propose/fence transition bit-for-bit.
    tier_b_agrees: bool = True
    tier_b_steps: int = 0
    per_size: list[tuple[int, str, str]] = field(default_factory=list)  # (n, minority, majority)


def _config(n: int, key: str, vals: tuple[str, ...]) -> DistConfig:
    nodes = tuple(f"n{i}" for i in range(n))
    return DistConfig(
        name=f"ed23-{n}n", nodes=nodes, objects=(key,),
        values=vals, replication_factor=n,
    )


def run_ed23(cfg: ED23Config | None = None) -> ED23Result:
    cfg = cfg or ED23Config()
    result = ED23Result()
    vals = ("a", cfg.v_old, cfg.v_new, "d")
    tier_b_agree = True
    tier_b_steps = 0

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

    # --- Panel A: no split-brain — only a strict-majority side can elect --------------------------
    minority_blocked = 0
    minority_total = 0
    majority_elected = 0
    majority_total = 0
    even_leaderless = 0
    even_total = 0
    for n in cfg.cluster_sizes:
        config = _config(n, cfg.key, vals)
        ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)
        nodes = config.nodes
        half = n // 2
        minority = nodes[:half]          # size floor(n/2): a strict minority for odd n
        majority = nodes[half:]          # size ceil(n/2)
        part_cmd = f"partition {' '.join(minority)} | {' '.join(majority)}"
        is_even = (n % 2 == 0)
        sa = sb = DistributedState.initial(config)
        sa, sb, _, _ = step_both(ref, sysb, sa, sb, part_cmd)
        # try to elect on the smaller side, then the larger side
        _, _, s_small, _ = step_both(ref, sysb, sa, sb, f"elect {minority[0]}")
        _, _, s_big, _ = step_both(ref, sysb, sa, sb, f"elect {majority[0]}")
        result.per_size.append((n, minority[0], majority[0]))
        if is_even:
            # a 2|2 (etc.): neither side is a strict majority, so neither elects — leaderless, not
            # split: both must be no_quorum.
            even_total += 1
            if s_small == "no_quorum" and s_big == "no_quorum":
                even_leaderless += 1
        else:
            minority_total += 1
            majority_total += 1
            if s_small == "no_quorum":
                minority_blocked += 1
            if s_big == "elected":
                majority_elected += 1
    result.minority_elect_blocked_rate = (
        minority_blocked / minority_total if minority_total else 0.0
    )
    result.majority_elect_rate = majority_elected / majority_total if majority_total else 0.0
    result.even_split_leaderless_rate = even_leaderless / even_total if even_total else 0.0
    result.n_split_scenarios = minority_total
    result.n_even_scenarios = even_total

    # --- Panel B: term-fencing — the deposed leader cannot commit after heal ----------------------
    # A 3-node cluster: n0 leads, is partitioned into the minority, the majority {n1,n2} elects n1
    # (higher term). After heal, n0's propose is fenced; a plain put by n0 would still write.
    config = _config(3, cfg.key, vals)
    ref, sysb = ReferenceDistOracle(config), SystemDistOracle(config)

    def fence_scenario(deposed_cmd: str) -> str:
        """Run the depose-then-heal scenario; return the deposed-coordinator op's status."""
        sa = sb = DistributedState.initial(config)
        for cmd in ["elect n0", "partition n0 | n1 n2", "elect n1", "heal"]:
            sa, sb, _, _ = step_both(ref, sysb, sa, sb, cmd)
        sa, sb, status, _ = step_both(ref, sysb, sa, sb, deposed_cmd)
        return status

    result.deposed_propose_fenced = (
        fence_scenario(f"propose n0 {cfg.key} {cfg.v_old}") == "not_leader"
    )
    # The control: without the consensus fence, a plain put by the stale ex-leader still commits —
    # exactly the stale write term-fencing exists to prevent.
    result.unfenced_put_commits = fence_scenario(f"put n0 {cfg.key} {cfg.v_old}") == "ok"
    # The legitimate new leader commits.
    result.new_leader_commits = fence_scenario(f"propose n1 {cfg.key} {cfg.v_new}") == "ok"

    result.tier_b_agrees = tier_b_agree
    result.tier_b_steps = tier_b_steps
    return result


CSV_HEADER = "panel,metric,value,detail"


def write_csv(result: ED23Result, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_HEADER]
    lines.append(f"split_brain,minority_elect_blocked,{result.minority_elect_blocked_rate:.4f},"
                 f"over_{result.n_split_scenarios}_odd_clusters")
    lines.append(f"split_brain,majority_elect,{result.majority_elect_rate:.4f},"
                 f"over_{result.n_split_scenarios}_odd_clusters")
    lines.append(f"split_brain,even_split_leaderless,{result.even_split_leaderless_rate:.4f},"
                 f"over_{result.n_even_scenarios}_even_clusters")
    lines.append(f"fencing,deposed_propose_fenced,"
                 f"{1.0 if result.deposed_propose_fenced else 0.0:.4f},not_leader_after_heal")
    lines.append(f"fencing,unfenced_put_commits,"
                 f"{1.0 if result.unfenced_put_commits else 0.0:.4f},stale_write_the_fence_stops")
    lines.append(f"fencing,new_leader_commits,"
                 f"{1.0 if result.new_leader_commits else 0.0:.4f},legitimate_leader_ok")
    lines.append(f"tier_b,all,{1.0 if result.tier_b_agrees else 0.0:.4f},"
                 f"steps={result.tier_b_steps}")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ED23 (leader election with terms: no split-brain + the quorum-less fence)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed23.csv")
    parser.add_argument("--plot", type=str, default="figures/ed23.png")
    args = parser.parse_args()
    cfg = ED23Config.from_json_file(args.config) if args.config else ED23Config()
    result = run_ed23(cfg)
    path = write_csv(result, args.out)
    print(f"wrote {path}")
    print("  Panel A — no split-brain leadership (only a strict-majority side elects):")
    print(f"    minority elect blocked: {result.minority_elect_blocked_rate:.2f} "
          f"| majority elect: {result.majority_elect_rate:.2f} "
          f"| even-split leaderless: {result.even_split_leaderless_rate:.2f}")
    print("  Panel B — term-fencing (the property a leaderless quorum write lacks):")
    print(f"    deposed leader propose fenced after heal: {result.deposed_propose_fenced}")
    print(f"    (control) plain put by the same stale node still commits: "
          f"{result.unfenced_put_commits}")
    print(f"    legitimate new leader commits: {result.new_leader_commits}")
    print(f"  Tier-B reproduces every transition bit-for-bit: "
          f"{result.tier_b_agrees} over {result.tier_b_steps} steps")
    try:
        from figures.plot_ed23 import plot_ed23

        plot_ed23(result, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
