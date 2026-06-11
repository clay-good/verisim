"""The SPEC-20 autonomous-cyber-defense (ACD) usefulness layer.

Where every prior spec studied the world model as the *object*, SPEC-20 uses it as the *environment*
for a separate learned policy -- the field's gold-standard test of a world model (train a policy
inside it, transfer to reality). This package builds the defensive containment task on the SPEC-5
network world and the three plug-swappable backends (`E_oracle` / `E_grounded` / `E_free`) that
isolate what oracle-grounding buys for downstream transfer (SPEC-20 §3).

UA0 ships the env + backends; UA1/UA2 add the learn-in-imagination training and the grounding
ablation (H73/H74). It invents no new world or oracle: the dynamics are the SPEC-5 reachability
graph, the compromise spread is computed on top via the shipped reachability functions, and the
backends reuse the reference oracle and the trained `M_θ`.
"""

from .containment import (
    Backend,
    ContainmentConfig,
    ContainmentEnv,
    DefenderAction,
    FreeBackend,
    GroundedBackend,
    OracleBackend,
    containment_fraction,
    legal_actions,
    seed_topology,
)

__all__ = [
    "Backend",
    "ContainmentConfig",
    "ContainmentEnv",
    "DefenderAction",
    "FreeBackend",
    "GroundedBackend",
    "OracleBackend",
    "containment_fraction",
    "legal_actions",
    "seed_topology",
]
