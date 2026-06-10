"""ED30 — the embedded host: each cluster node runs a real SPEC-6 host (DS0 increment 23).

The smoke instance of the DS0-increment-23 apparatus (dependency-free, GPU-free): a tiny check that
the two findings have the right shape — a `host` syscall runs on its node's own host (per-node
isolated), coexists with the KV subsystem, and reaches the embedded v0 filesystem (Panel A); and
host ops respect the node's up/down status, with the host state surviving a crash/restart (Panel B)
— with Tier-B reproducing every transition bit-for-bit. The committed figure comes from local run.
"""

from __future__ import annotations

from verisim.experiments.ed30 import ED30Config, ED30Result, run_ed30, write_csv


def test_panel_a_composition_and_isolation():
    result = run_ed30(ED30Config())
    assert result.fork_isolated_rate == 1.0  # a fork is isolated to its node's host
    assert result.kv_and_host_coexist is True  # a node serves a KV put and a host fork together
    assert result.embedded_fs_works is True  # open + write reaches the node's embedded v0 FS


def test_panel_b_crash_linkage():
    result = run_ed30(ED30Config())
    assert result.crashed_host_unavailable is True  # host op on a crashed node is unavailable
    assert result.restart_restores is True  # host ops work again after restart
    assert result.host_state_survives_crash is True  # the process table persists across the crash


def test_tier_b_reproduces_every_transition():
    result = run_ed30(ED30Config())
    assert result.tier_b_agrees is True
    assert result.tier_b_steps > 0


def test_write_csv(tmp_path):
    result = run_ed30(ED30Config())
    out = write_csv(result, tmp_path / "ed30.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("compose,") for line in lines)
    assert any(line.startswith("crash,") for line in lines)


def test_config_round_trips():
    cfg = ED30Config.from_dict({"cluster_sizes": [3], "path": "/g", "token": "b"})
    assert cfg.cluster_sizes == (3,) and cfg.path == "/g"
    assert isinstance(run_ed30(cfg), ED30Result)
