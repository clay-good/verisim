"""verisim-cue packaging + conformance tests (SPEC-21 §8, deliverable #2).

The contract: the manifest is frozen + hashable + versioned, the Croissant/datasheet/task-card emit
well-formed content (including the per-task load-bearing verdict from a frontier CSV), the
conformance suite holds (ground-truth labels exact, ordered spectrum, recognized dimensions), and
emit writes the three files. All torch-free -- the ground-truth check uses only the oracle.
"""

from __future__ import annotations

import json

import pytest

from verisim.cue.conformance import (
    all_passed,
    check_ground_truth_labels,
    check_ordered_spectrum,
    run_conformance,
)
from verisim.cue.pack import (
    CueManifest,
    croissant_metadata,
    datasheet,
    task_card,
)
from verisim.cue.scorecard import (
    TaskScore,
    model_card,
    reference_scores_from_csv,
    scorecard_headline,
)
from verisim.experiments.cue_pack import emit, load_verdicts


def test_manifest_hash_stable_and_versioned():
    m = CueManifest()
    assert m.manifest_hash() == CueManifest().manifest_hash()  # deterministic
    assert m.version_tag().startswith("verisim-cue@0.1.0+")
    # changing the battery changes the hash (the MAJOR identity)
    m2 = CueManifest(horizon=99)
    assert m2.manifest_hash() != m.manifest_hash()
    # the suite is pinned, ordered structure->content
    assert [t.order for t in m.tasks] == [0, 1, 2, 3]


def test_croissant_is_wellformed():
    cr = croissant_metadata(CueManifest())
    assert cr["@type"] == "Dataset" and cr["name"] == "verisim-cue"
    assert cr["manifestHash"] == CueManifest().manifest_hash()
    record_sets = cr["cr:recordSet"]
    assert isinstance(record_sets, list) and len(record_sets) == 4  # one per task
    # it is JSON-serializable (the emitted artifact)
    assert json.loads(json.dumps(cr))["name"] == "verisim-cue"


def test_datasheet_mentions_the_tasks_and_caveats():
    ds = datasheet(CueManifest())
    for name in ("process-control", "fd-control", "file-integrity", "content-value"):
        assert name in ds
    assert "not** for offensive" in ds.lower() or "**Not** for offensive" in ds
    assert "GUI" in ds  # the honest scope caveat


def test_task_card_renders_verdicts():
    verdicts = {
        "process-control": {"gap": 0.03, "scale": 110592.0},
        "fd-control": {"gap": 0.16, "scale": 110592.0},
        "file-integrity": {"gap": 0.56, "scale": 110592.0},
        "content-value": {"gap": 0.84, "scale": 110592.0},
    }
    card = task_card(CueManifest(), verdicts)
    assert "not load-bearing" in card  # process-control (gap 0.03 < threshold)
    assert "load-bearing" in card  # the content tasks
    assert "+0.84" in card  # the content-value verdict
    # without verdicts the card marks pending
    assert "pending" in task_card(CueManifest(), None)


def test_load_verdicts_picks_top_rung(tmp_path):
    csv = tmp_path / "frontier.csv"
    csv.write_text(
        "label,params,task,order,keyed_dimension,faithful,free,gap,keyed_drift,knee_rho\n"
        "xs,1024,content-value,3,fs,1.0,0.2,0.80,0.9,0.3\n"
        "l,110592,content-value,3,fs,1.0,0.16,0.84,0.9,0.5\n"
    )
    v = load_verdicts(csv)
    assert v["content-value"]["gap"] == 0.84  # the largest-params (top rung) row
    assert v["content-value"]["scale"] == 110592.0
    assert load_verdicts(tmp_path / "missing.csv") == {}  # absent -> empty


def test_conformance_holds_ground_truth_labels():
    results = run_conformance()
    assert all_passed(results)
    # the defining property: the faithful predictor is exact (== 1.0)
    gt = check_ground_truth_labels(CueManifest())
    assert gt.passed and "1.0000" in gt.detail
    assert check_ordered_spectrum(CueManifest()).passed


def test_emit_writes_four_artifacts(tmp_path):
    paths = emit(CueManifest(), tmp_path, frontier_csv="does-not-exist.csv")
    for key in ("croissant", "datasheet", "task_card", "model_card"):
        assert paths[key].exists() and paths[key].read_text()
    # the croissant file is valid JSON
    assert json.loads(paths["croissant"].read_text())["name"] == "verisim-cue"


def _frontier_csv(tmp_path):
    csv = tmp_path / "frontier.csv"
    csv.write_text(
        "label,params,task,order,keyed_dimension,faithful,free,gap,keyed_drift,knee_rho\n"
        "xs,1024,process-control,0,procs,1.0,0.84,0.16,0.3,0.3\n"
        "xs,1024,content-value,3,fs,1.0,0.19,0.81,0.9,0.3\n"
        "l,110592,process-control,0,procs,1.0,0.97,0.03,0.2,-1.0\n"
        "l,110592,content-value,3,fs,1.0,0.16,0.84,0.9,0.5\n"
    )
    return csv


def test_reference_scores_per_rung(tmp_path):
    ref = reference_scores_from_csv(_frontier_csv(tmp_path), threshold=0.05)
    assert set(ref) == {"xs", "l"}
    # each rung's scorecard is ordered by task; the load-bearing flag is gap > threshold
    l_scores = {s.task: s for s in ref["l"]}
    assert l_scores["process-control"].catch_rate == 0.97
    assert not l_scores["process-control"].load_bearing  # gap 0.03 < 0.05
    assert l_scores["content-value"].load_bearing  # gap 0.84 > 0.05
    assert reference_scores_from_csv(tmp_path / "missing.csv") == {}


def test_model_card_renders_schema_and_reference(tmp_path):
    # with a frontier CSV -> shows the reference scorecards (the per-rung models)
    card = model_card(CueManifest(), frontier_csv=_frontier_csv(tmp_path))
    assert "Scorecard schema" in card and "Reference scorecards" in card
    assert "`l`" in card and "process 0.97" in card  # a reference rung's catch rate
    # without a CSV -> just the schema (no reference section)
    bare = model_card(CueManifest(), frontier_csv=None)
    assert "Scorecard schema" in bare and "Reference scorecards" not in bare


def test_scorecard_headline_logic():
    scores = [
        TaskScore("process-control", 0, 0.97, 1.0, 0.03, False),
        TaskScore("fd-control", 1, 0.84, 1.0, 0.16, True),
        TaskScore("content-value", 3, 0.16, 1.0, 0.84, True),
    ]
    h = scorecard_headline(scores)
    assert h["n_load_bearing"] == 2
    assert h["load_bearing_tasks"] == ["fd-control", "content-value"]
    assert h["structure_clean"]  # the order-0 task is not load-bearing (the model gets structure)


# --- torch-gated: scoring a real (smoke) model through the suite ----------------------------------

torch = pytest.importorskip("torch")

from verisim.cue.scorecard import score_model  # noqa: E402
from verisim.cue.tasks import TaskGapConfig  # noqa: E402
from verisim.experiments.host_flagship import HostFlagshipConfig, train_host_flagship  # noqa: E402


def test_score_model_through_the_suite():
    model, _ = train_host_flagship(HostFlagshipConfig.smoke())
    manifest = CueManifest()
    config = TaskGapConfig(horizon=10, workload_seeds=tuple(range(700, 706)))
    scores = score_model(model, manifest, config)
    assert [s.order for s in scores] == [0, 1, 2, 3]
    for s in scores:
        assert 0.0 <= s.catch_rate <= 1.0 and s.faithful_ceiling == pytest.approx(1.0)
        assert s.load_bearing == (s.gap > manifest.threshold)
    # the headline summarizes the load-bearing footprint
    h = scorecard_headline(scores, manifest)
    assert 0.0 <= h["mean_catch_rate"] <= 1.0 and isinstance(h["n_load_bearing"], int)
