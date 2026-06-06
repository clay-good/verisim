"""Distributed-world config and curriculum dials (SPEC-7 §3.4, DS0 increment 1).

The finite vocabulary the workload draws from -- the cluster's **nodes** and the replicated
**key-value objects** with their **value tokens** -- plus the curriculum difficulty dials (SPEC-7
§3.4): replication factor, the declared consistency model, and the fault axes (fault intensity,
partition entropy) that H20/H21 will sweep. The analogue of v0's
:class:`~verisim.env.config.EnvConfig`, SPEC-5's ``NetConfig``, and SPEC-6's
:class:`~verisim.host.config.HostConfig` -- the argument space is fixed and small, so divergence
measures *competence over the protocol*, not arbitrary-byte churn.

DS0 increment 1 ships the **replicated KV under partition** core: ``put``/``get``/``cas`` over a
fully-replicated keyspace with MVCC versions, an asynchronous message layer, and the fault/time
medium (``partition``/``heal``/``crash``/``restart``/``advance``). The Raft-subset consensus group,
the transaction/lock table, and the embedded SPEC-6 hosts / SPEC-5 network are later DS0/DS1
increments (``docs/distributed-semantics.md``). No runtime dependencies, no GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# Declared consistency models, ordered strong -> weak (weaker is *harder* to predict because more
# histories are legal, SPEC-7 §3.4). DS0 increment 1 implements the asynchronous-replication
# dynamics directly; the model is the label the consistency-cycle tier (a later DS increment) checks
# a history against. ``eventual`` is the increment-1 default: writes return locally and replicas
# converge only when replication messages are delivered by ``advance``.
CONSISTENCY_MODELS: tuple[str, ...] = (
    "linearizable",
    "serializable",
    "snapshot",
    "causal",
    "eventual",
)

# Transaction isolation levels (DS0 increment 3, SPEC-7 §3.2). ``serializable`` is OCC backward
# validation of the **read-set** (a committed txn's read versions must be unchanged) — it prevents
# write skew; ``snapshot`` validates only **write-write** conflicts (the write-set versions,
# first-committer-wins) — it permits write skew, the classic SI anomaly the experiment exhibits.
TXN_ISOLATION_LEVELS: tuple[str, ...] = ("serializable", "snapshot")


@dataclass(frozen=True)
class DistConfig:
    """The finite cluster vocabulary + curriculum dials. Embedded (with its hash) in manifests."""

    name: str = "dist-v0"
    nodes: tuple[str, ...] = ("n0", "n1", "n2")
    objects: tuple[str, ...] = ("x", "y")  # the replicated keyspace
    values: tuple[str, ...] = ("a", "b", "c", "d")  # value tokens a put/cas may write
    replication_factor: int = 3  # DS0 incr 1: full replication (every node holds every object)
    consistency_model: str = "eventual"
    default_value: str = "nil"  # the boot value of every replica (version 0)
    txn_isolation: str = "serializable"  # DS0 incr 3: serializable (read-set OCC) | snapshot (SI)

    def __post_init__(self) -> None:
        if self.consistency_model not in CONSISTENCY_MODELS:
            raise ValueError(
                f"unknown consistency_model {self.consistency_model!r}; "
                f"choose from {list(CONSISTENCY_MODELS)}"
            )
        if self.txn_isolation not in TXN_ISOLATION_LEVELS:
            raise ValueError(
                f"unknown txn_isolation {self.txn_isolation!r}; "
                f"choose from {list(TXN_ISOLATION_LEVELS)}"
            )
        if not 1 <= self.replication_factor <= len(self.nodes):
            raise ValueError(
                f"replication_factor {self.replication_factor} must be in "
                f"[1, {len(self.nodes)}] (n_nodes)"
            )
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("node ids must be unique")

    def replicas_of(self, object_id: str) -> tuple[str, ...]:
        """The nodes holding a replica of ``object_id`` -- the first ``replication_factor`` nodes.

        DS0 increment 1 places every object on the same prefix of nodes (deterministic, simple);
        sharding/placement policies are a later increment.
        """
        return self.nodes[: self.replication_factor]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "nodes": list(self.nodes),
            "objects": list(self.objects),
            "values": list(self.values),
            "replication_factor": self.replication_factor,
            "consistency_model": self.consistency_model,
            "default_value": self.default_value,
            "txn_isolation": self.txn_isolation,
        }

    def config_hash(self) -> str:
        """Stable hash identifying this config; embedded in dataset manifests (SPEC-7 §3.1)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


DEFAULT_DIST_CONFIG = DistConfig()


def scaled_dist_config(
    n_nodes: int,
    *,
    n_objects: int = 2,
    replication_factor: int | None = None,
    consistency_model: str = "eventual",
    txn_isolation: str = "serializable",
    name: str | None = None,
) -> DistConfig:
    """A :class:`DistConfig` of ``n_nodes`` nodes and ``n_objects`` objects (SPEC-7 §3.4).

    The world-size axis for the distributed world (the analogue of SPEC-5's ``scaled_net_config``):
    every node-count uses canonical ``n{i}`` ids and ``o{j}`` object ids, so a bigger cluster is a
    pure learner-compute choice. ``replication_factor`` defaults to ``min(3, n_nodes)``.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    if n_objects < 1:
        raise ValueError(f"n_objects must be >= 1, got {n_objects}")
    nodes = tuple(f"n{i}" for i in range(n_nodes))
    objects = tuple(f"o{j}" for j in range(n_objects))
    rf = replication_factor if replication_factor is not None else min(3, n_nodes)
    return DistConfig(
        name=name or f"dist-{n_nodes}n{n_objects}o",
        nodes=nodes,
        objects=objects,
        replication_factor=rf,
        consistency_model=consistency_model,
        txn_isolation=txn_isolation,
    )
