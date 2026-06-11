"""HFL0 host-flagship train/freeze/load/gate tests (SPEC-20 §7 host fork).

Mirrors the FL0 contract on the host world: train one flat host M_θ, freeze it, reload it bit-exact,
and gate on reload-determinism + the HS2 host plausibility band. The smoke model is trivial; the
real `l` host checkpoint comes from the local run. CI proves the lifecycle.
"""

from __future__ import annotations

import pytest

from verisim.experiments.host_flagship import HostFlagshipConfig

torch = pytest.importorskip("torch")

from verisim.experiments.host_flagship import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    train_host_flagship,
    verify_checkpoint,
)


def test_smoke_config_is_tiny():
    cfg = HostFlagshipConfig.smoke()
    assert cfg.scale.params < HostFlagshipConfig().scale.params
    assert cfg.num_threads == 1  # bit-deterministic reload


def test_train_host_flagship_metrics_well_formed():
    _, metrics = train_host_flagship(HostFlagshipConfig.smoke())
    for regime in ("id", "ood"):
        assert 0.0 <= metrics[f"one_step_acc_{regime}"] <= 1.0
        assert metrics[f"h_free_{regime}"] >= 0.0
    assert metrics["n_train"] > 0


def test_freeze_reload_roundtrip_and_determinism(tmp_path):
    cfg = HostFlagshipConfig.smoke()
    world_model, metrics = train_host_flagship(cfg)
    directory = save_checkpoint(world_model, cfg, metrics, tmp_path / "host")
    ckpt = load_checkpoint(directory)
    assert ckpt.manifest["spec"] == "SPEC-20 HFL0 (host flagship)"
    assert ckpt.metrics["h_free_id"] == metrics["h_free_id"]
    assert ckpt.manifest["arch"]["n_embd"] == cfg.scale.n_embd

    # reloaded model predicts an identical delta to the original on a held-out (state, action)
    import random

    from verisim.host.config import DEFAULT_HOST_CONFIG
    from verisim.host.state import HostState
    from verisim.hostdata import HostDriver

    drv = HostDriver(name="forky", config=DEFAULT_HOST_CONFIG, rng=random.Random(7))
    state = HostState.initial()
    action = drv.sample(state)
    assert world_model.predict_delta(state, action) == ckpt.world_model.predict_delta(state, action)


def test_verify_checkpoint_gate(tmp_path, monkeypatch):
    cfg = HostFlagshipConfig.smoke()
    world_model, metrics = train_host_flagship(cfg)
    directory = save_checkpoint(world_model, cfg, metrics, tmp_path / "host")
    verdict = verify_checkpoint(directory, config=cfg)
    assert verdict["reload_deterministic"]  # bit-exact reload regardless of band
    # an impossible band -> band fails -> ok fails
    import verisim.experiments.host_flagship as hf

    monkeypatch.setattr(hf, "HOST_FRONTIER_H_FREE_ID_BAND", (1e6, 2e6))
    assert not verify_checkpoint(directory, config=cfg)["ok"]
