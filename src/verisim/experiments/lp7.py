"""Experiment LP7: the LLM-at-the-leaves boundary -- traversal belongs to search (H37, SPEC-12 §6).

The practitioner's founding insight and the NLGraph / Talk-like-a-Graph result (§2.2): an LLM's
graph-reasoning accuracy degrades with traversal depth and per-node degree, and a positive-answer
bias makes it assert paths that do not exist. SPEC-12's design confines the LLM to the *leaves*
(NL intent -> one hop's action; exploitation at a target) and does the *traversal* by graph search
over the verified graph, server-side. H37 makes that a measured boundary rather than an assertion.

LP7 has a hard external dependency (a frozen LLM proposer) that the rest of SPEC-12 does not, so it
is split, exactly as the spec pre-registers (§10, §6 LP7):

  - **the dependency-free core (this module's committed result).** Over the verified landmark graph,
    as true path length and node degree grow, compare two *traversal strategies that need no LLM*:
    the **graph search** the planner uses (:func:`~verisim.landmark.plan.shortest_landmark_path`,
    exact and complete -- every path it returns is real, LP2's zero-false-edges) against a **myopic
    greedy traverser** -- the structural class a hop-by-hop walk (an LLM deciding the next node from
    local information, no backtracking) falls into. The greedy traverser is *not* an LLM and is
    never presented as one; it is the deterministic stand-in that isolates the *structural* reason
    hop-by-hop traversal fails on a branching graph -- local choices cannot recover the route.
    The committed figure shows search stays exact and degree-invariant while the myopic strategy's
    path-validity and optimality decay with depth and degree (the NLGraph *shape*, derived
    mechanistically), which is the correctness argument for delegating traversal to search.
  - **the LLM-traverser arm (deferred, skipif-guarded).** The real H37 number -- an actual LLM
    walking the raw graph vs an LLM fed only the planner's leaf subgoals -- needs a frozen model
    behind the proposer seam. It is wired (:func:`llm_traverse_available`) and **never counted when
    absent**, the §9 disclosure discipline (as LP1's scatter was deferred).

Torch-free and LLM-free for the committed core: the graph, the search, and the greedy traverser are
pure Python (the NW0-NW3 discipline, SPEC-5 §13), so the result reproduces on CPU with no model.
Reduced over (path-length / degree) buckets with bootstrap CIs. CPU, deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.landmark.build import sample_landmarks
from verisim.landmark.graph import LandmarkGraph, ReachSig
from verisim.landmark.plan import shortest_landmark_path
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.config import NetConfig, scaled_net_config
from verisim.netoracle import ReferenceNetworkOracle


def llm_traverse_available() -> bool:
    """Whether the deferred LLM-traverser arm can run (a frozen LLM proposer is configured).

    Always ``False`` in the dependency-free CI/primary-host run -- the LP7 LLM arm is the one
    SPEC-12 component needing an external model, and is never counted when absent (§9). Wiring a
    real proposer behind the ``NetModel`` seam flips this on; until then the committed result is the
    search-vs-myopic-traverser core below.
    """
    return False


@dataclass(frozen=True)
class LP7Config:
    """A small, fast LLM-at-the-leaves-boundary (H37) instance (the dependency-free core)."""

    n_hosts: int = 6
    n_ports: int = 3
    build_driver: str = "weighted"
    build_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    build_steps: int = 64
    max_path_length: int = 6
    pairs_per_bucket: int = 40
    greedy_step_cap: int = 64
    pair_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP7Config:
        b = LP7Config()
        return LP7Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            build_driver=d.get("build_driver", b.build_driver),
            build_seeds=tuple(d.get("build_seeds", b.build_seeds)),
            build_steps=d.get("build_steps", b.build_steps),
            max_path_length=d.get("max_path_length", b.max_path_length),
            pairs_per_bucket=d.get("pairs_per_bucket", b.pairs_per_bucket),
            greedy_step_cap=d.get("greedy_step_cap", b.greedy_step_cap),
            pair_seed=d.get("pair_seed", b.pair_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP7Config:
        return LP7Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class LP7Stat:
    """One (strategy, axis, bucket) cell: validity + optimality, mean + bootstrap CI."""

    strategy: str  # "search" | "greedy"
    axis: str  # "path_length" | "degree"
    bucket: float
    validity: float  # fraction of pairs the strategy reaches the goal at all (a real path)
    val_lo: float
    val_hi: float
    optimality: float  # fraction reached by a shortest path
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.strategy},{self.axis},{self.bucket:.6f},{self.validity:.6f},"
            f"{self.val_lo:.6f},{self.val_hi:.6f},{self.optimality:.6f},{self.n}"
        )


CSV_HEADER = "strategy,axis,bucket,validity,val_lo,val_hi,optimality,n"


def _bfs_distances(graph: LandmarkGraph, src: int) -> dict[int, int]:
    """Shortest verified-hop distance from ``src`` to every reachable landmark (BFS)."""
    dist = {src: 0}
    queue: deque[int] = deque([src])
    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _greedy_traverse(
    graph: LandmarkGraph, src: int, dst: int, dst_sig: ReachSig, cap: int
) -> list[int] | None:
    """A myopic hop-by-hop walk: step to the neighbor whose signature is closest to the goal's.

    No backtracking, no global search -- the structural class of an LLM walking the graph node by
    node (§2.2). At each node it picks the unvisited neighbor minimizing the symmetric-difference
    size to ``dst_sig`` (ties by id), which is the locally-greedy "am I getting closer?" heuristic.
    Returns the path if it reaches ``dst`` within ``cap`` steps, else ``None`` (a dead end -- the
    failure search never has).
    """
    sigs = graph.signatures
    path = [src]
    visited = {src}
    cur = src
    for _ in range(cap):
        if cur == dst:
            return path
        candidates = [v for v in graph.neighbors(cur) if v not in visited]
        if not candidates:
            return None
        cur = min(candidates, key=lambda v: (len(sigs[v] ^ dst_sig), v))
        visited.add(cur)
        path.append(cur)
    return path if cur == dst else None


def _build_graph(config: LP7Config) -> LandmarkGraph:
    """Sample a verified landmark graph (every edge oracle-true) for the traversal study."""
    oracle = ReferenceNetworkOracle()
    net: NetConfig = scaled_net_config(config.n_hosts, config.n_ports)
    sample = sample_landmarks(
        oracle, net, driver=config.build_driver, seeds=config.build_seeds,
        n_steps=config.build_steps,
    )
    return LandmarkGraph(
        nodes=sample.nodes, signatures=sample.signatures, edges=sample.oracle_edges()
    )


@dataclass(frozen=True)
class _Pair:
    src: int
    dst: int
    true_len: int
    max_degree: int  # the hardest branching on the optimal route (the NLGraph degree axis)


def _eval_pairs(graph: LandmarkGraph, config: LP7Config) -> list[_Pair]:
    """Sample (src, dst) pairs bucketed by true verified-hop distance, up to ``max_path_length``."""
    rng = random.Random(config.pair_seed)
    nodes = list(range(graph.num_nodes))
    by_len: dict[int, list[_Pair]] = {}
    rng.shuffle(nodes)
    for src in nodes:
        dist = _bfs_distances(graph, src)
        for dst, d in dist.items():
            if 1 <= d <= config.max_path_length:
                path = shortest_landmark_path(graph, src, dst)
                assert path is not None
                max_deg = max(len(graph.neighbors(u)) for u in path)
                by_len.setdefault(d, []).append(_Pair(src, dst, d, max_deg))
    pairs: list[_Pair] = []
    for group in by_len.values():
        rng.shuffle(group)
        pairs.extend(group[: config.pairs_per_bucket])
    return pairs


def run_lp7(config: LP7Config | None = None) -> list[LP7Stat]:
    """Build the verified graph; measure search vs myopic-traverser by depth and degree (H37)."""
    config = config or LP7Config()
    graph = _build_graph(config)
    pairs = _eval_pairs(graph, config)

    # Per pair: did each strategy reach the goal (validity), and by a shortest path (optimality)?
    records: list[tuple[_Pair, str, float, float]] = []
    for p in pairs:
        dst_sig = graph.signatures[p.dst]
        search = shortest_landmark_path(graph, p.src, p.dst)  # exact + complete
        s_valid = float(search is not None)
        s_opt = float(search is not None and len(search) - 1 == p.true_len)
        greedy = _greedy_traverse(graph, p.src, p.dst, dst_sig, config.greedy_step_cap)
        g_valid = float(greedy is not None)
        g_opt = float(greedy is not None and len(greedy) - 1 == p.true_len)
        records.append((p, "search", s_valid, s_opt))
        records.append((p, "greedy", g_valid, g_opt))

    def _by_len(p: _Pair) -> int:
        return p.true_len

    def _by_deg(p: _Pair) -> int:
        return p.max_degree

    stats: list[LP7Stat] = []
    for axis, key in (("path_length", _by_len), ("degree", _by_deg)):
        buckets = sorted({key(p) for p, *_ in records})
        for strategy in ("search", "greedy"):
            for b in buckets:
                cell = [(v, o) for p, s, v, o in records if s == strategy and key(p) == b]
                if not cell:
                    continue
                valid = [v for v, _ in cell]
                lo, hi = bootstrap_ci(valid, seed=0)
                stats.append(
                    LP7Stat(strategy, axis, float(b), fmean(valid), lo, hi,
                            fmean([o for _, o in cell]), len(cell))
                )
    return stats


def _print_summary(stats: list[LP7Stat]) -> None:
    print("LP7 / H37 - the LLM-at-the-leaves boundary (search vs myopic walk):")
    if not llm_traverse_available():
        print("  [LLM-traverser arm DEFERRED -- no LLM proposer configured; not counted (§9)]")
    for axis in ("path_length", "degree"):
        print(f"  -- by {axis} --")
        print(f"  {'strategy':<8} {'bucket':>7} {'validity':>9} {'95% CI':>18} {'optimality':>11}")
        for s in (s for s in stats if s.axis == axis):
            print(
                f"  {s.strategy:<8} {s.bucket:>7.0f} {s.validity:>9.3f} "
                f"{f'[{s.val_lo:.3f}, {s.val_hi:.3f}]':>18} {s.optimality:>11.3f}"
            )
    by = {(s.strategy, s.axis, s.bucket): s for s in stats}
    # The boundary is supported if, on *either* axis, search stays exact while the myopic walk
    # decays at the far bucket (NLGraph degrades with depth OR per-node degree).
    gaps: dict[str, tuple[float, float, float, float]] = {}
    for axis in ("path_length", "degree"):
        bs = sorted({s.bucket for s in stats if s.axis == axis})
        far, near = max(bs), min(bs)
        s_far = by[("search", axis, far)].validity
        g_far = by[("greedy", axis, far)].validity
        g_near = by[("greedy", axis, near)].validity
        gaps[axis] = (far, s_far, g_near, g_far)
        print(f"  {axis}: search@{far:.0f}={s_far:.2f}, myopic {g_near:.2f}->{g_far:.2f} "
              f"(gap {s_far - g_far:+.2f})")
    best_axis = max(gaps, key=lambda a: gaps[a][1] - gaps[a][3])
    far, s_far, g_near, g_far = gaps[best_axis]
    verdict = (
        f"search stays exact while the myopic walk decays with {best_axis} "
        f"({g_near:.2f}->{g_far:.2f} vs search {s_far:.2f}) - H37 core supported "
        "(traversal belongs to search); LLM arm deferred"
        if s_far > g_far else
        "the myopic walk matches search on this graph - H37 core inconclusive (graph too shallow); "
        "LLM arm deferred"
    )
    print(f"  verdict: {verdict}")


def _plot(stats: list[LP7Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {"search": "#16a", "greedy": "#c33"}
    labels = {"search": "graph search (planner-fed leaf)", "greedy": "myopic walk (LLM-walk class)"}
    for ax, axis, xlabel in (
        (ax1, "path_length", "true path length (hops)"),
        (ax2, "degree", "max node degree on route"),
    ):
        for strategy in ("search", "greedy"):
            cells = sorted((s for s in stats if s.axis == axis and s.strategy == strategy),
                           key=lambda s: s.bucket)
            xs = [s.bucket for s in cells]
            ys = [s.validity for s in cells]
            lo = [s.val_lo for s in cells]
            hi = [s.val_hi for s in cells]
            ax.plot(xs, ys, "-o", color=colors[strategy], label=labels[strategy])
            ax.fill_between(xs, lo, hi, color=colors[strategy], alpha=0.12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("path-validity (reaches the goal)")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8)
    ax1.set_title("search is depth-invariant; the myopic walk decays")
    ax2.set_title("...and degree-invariant; the walk decays with branching")
    fig.suptitle("LP7 / H37: traversal belongs to search (NLGraph shape; LLM arm deferred)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP7 LLM-at-the-leaves boundary core (H37).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp7_traversal.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP7Config.from_json_file(args.config) if args.config else LP7Config()
    stats = run_lp7(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
