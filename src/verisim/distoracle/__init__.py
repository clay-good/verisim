"""The distributed oracle (SPEC-7 §5). DS0 increment 1 ships Tier-A (the reference DES)."""

from verisim.distoracle.base import DistOracle, DistStepResult
from verisim.distoracle.reference import ReferenceDistOracle

__all__ = ["DistOracle", "DistStepResult", "ReferenceDistOracle"]
