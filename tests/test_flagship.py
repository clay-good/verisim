"""FL0 flagship train/freeze/load/gate tests (SPEC-19 §5, milestone FL0).

The flagship is a *single frozen checkpoint* every downstream SPEC-19 experiment runs against, so
the contract under test is the lifecycle, not a sweep:

  - ``train_flagship`` produces well-formed id+ood metrics (``p`` a probability, ``H_free`` ≥ 0);
  - ``save_checkpoint`` then ``load_checkpoint`` round-trips the model AND the manifest;
  - a reloaded checkpoint reproduces its frozen ``H_free`` **bit-for-bit** (the gate's reload-
    determinism check -- what "frozen" means), and predicts identical deltas to the original;
  - ``verify_checkpoint`` passes on a freshly-frozen smoke checkpoint and FAILS when the band is
    impossible (a corrupted/under-trained checkpoint cannot masquerade as the flagship).

The smoke instance trains a trivial model in seconds; the real ``l@9.6k`` frontier checkpoint comes
from the local run (``python -m verisim.experiments.flagship --config configs/flagship.json``), not
CI -- the SPEC-9 envelope discipline, matching the rest of SPEC-10.
"""

from __future__ import annotations

import pytest

from verisim.experiments.flagship import FlagshipConfig

torch = pytest.importorskip("torch")

from verisim.experiments.flagship import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    train_flagship,
    verify_checkpoint,
)


def test_smoke_config_is_tiny_but_real():
    cfg = FlagshipConfig.smoke()
    assert cfg.scale.params < FlagshipConfig().scale.params  # genuinely smaller than the frontier
    assert cfg.num_threads == 1  # bit-deterministic so the reload gate holds


def test_train_flagship_metrics_well_formed():
    _, metrics = train_flagship(FlagshipConfig.smoke())
    for regime in ("id", "ood"):
        assert 0.0 <= metrics[f"one_step_acc_{regime}"] <= 1.0
        assert metrics[f"h_free_{regime}"] >= 0.0
        assert metrics[f"horizon_efficiency_{regime}"] >= 0.0
    assert metrics["n_train"] > 0


def test_freeze_reload_roundtrip_and_determinism(tmp_path):
    cfg = FlagshipConfig.smoke()
    world_model, metrics = train_flagship(cfg)
    directory = save_checkpoint(world_model, cfg, metrics, tmp_path / "ckpt")

    ckpt = load_checkpoint(directory)
    # manifest round-trips
    assert ckpt.manifest["spec"] == "SPEC-19 FL0"
    assert ckpt.metrics["h_free_id"] == metrics["h_free_id"]
    assert ckpt.manifest["arch"]["n_embd"] == cfg.scale.n_embd

    # the reloaded model predicts an identical delta to the original on a held-out (state, action)
    import random

    from verisim.net.config import DEFAULT_NET_CONFIG
    from verisim.net.state import NetworkState
    from verisim.netdata import NetDriver

    net = DEFAULT_NET_CONFIG
    drv = NetDriver(name="weighted", config=net, rng=random.Random(7))
    state = NetworkState.initial(net.hosts)
    action = drv.sample(state)
    assert world_model.predict_delta(state, action) == ckpt.world_model.predict_delta(state, action)


def test_verify_checkpoint_passes_on_fresh_freeze(tmp_path):
    cfg = FlagshipConfig.smoke()
    world_model, metrics = train_flagship(cfg)
    directory = save_checkpoint(world_model, cfg, metrics, tmp_path / "ckpt")
    verdict = verify_checkpoint(directory, config=cfg)
    assert verdict["reload_deterministic"], verdict  # the load is bit-exact regardless of band
    assert verdict["reeval_h_free_id"] == pytest.approx(verdict["frozen_h_free_id"], abs=1e-6)


def test_verify_checkpoint_band_rejects_impossible_horizon(tmp_path, monkeypatch):
    cfg = FlagshipConfig.smoke()
    world_model, metrics = train_flagship(cfg)
    directory = save_checkpoint(world_model, cfg, metrics, tmp_path / "ckpt")
    import verisim.experiments.flagship as fl

    # a band the smoke model cannot satisfy -> band_ok False -> ok False, even if reload is exact
    monkeypatch.setattr(fl, "FRONTIER_H_FREE_ID_BAND", (1e6, 2e6))
    verdict = verify_checkpoint(directory, config=cfg)
    assert verdict["reload_deterministic"]
    assert not verdict["in_frontier_band"]
    assert not verdict["ok"]
