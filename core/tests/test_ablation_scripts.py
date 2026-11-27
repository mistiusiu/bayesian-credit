"""scripts/ablation: the study definition stays consistent, compare catches drift, clean keeps results."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from bayesian_credit.config import SegmentationConfig
from bayesian_credit.data.schema import PRE_DECISION_COLUMNS
from bayesian_credit.data.segmentation import DIMENSIONS

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ablation"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"ablation_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


study = _load("study")
clean = _load("clean")


@pytest.mark.parametrize("scale", ["pilot", "full"])
def test_no_subset_uses_a_variable_it_removed(scale):
    p = study.plan(scale)
    assert list(p["subsets"]) == study.SUBSET_KEYS
    for s in study.SUBSETS:
        req = p["subsets"][s["key"]]
        excluded = set(req["dataset"]["exclude_variables"])
        seg = req["fit"]["segmentation"]
        assert req["cold"]["segmentation"] == seg  # cold runs are segmented like the warm fit
        for d in seg["dimensions"] + seg["refinement_order"]:
            assert DIMENSIONS[d][0] not in excluded, (s["key"], d)
        assert set(seg["refinement_order"]) <= set(SegmentationConfig().refinement_order)
        assert "sub_grade" not in seg["refinement_order"] and seg["max_refinements"] == 2
        for mode in ("warm", "cold"):
            feats = req[mode]["psi"]["features"]
            assert feats and set(feats) <= set(PRE_DECISION_COLUMNS) and not set(feats) & excluded
            assert req[mode]["policies"] == study.POLICIES and req[mode]["feedback_mode"] == "hybrid"
            assert req[mode]["start"] == study.TRAIN_END == req["fit"]["train_end"]
    assert [len(study.psi_features(s)) for s in study.SUBSETS] == [8, 18, 20, 22]


def test_pilot_and_full_differ_only_in_scale():
    a, b = study.plan("pilot"), study.plan("full")
    for key in study.SUBSET_KEYS:
        for part in ("dataset", "fit", "warm", "cold"):
            x, y = dict(a["subsets"][key][part]), dict(b["subsets"][key][part])
            for d in (x, y):
                d.pop("name"), d.pop("max_rows", None), d.pop("max_loans", None), d.pop("checkpoint_every", None)
            assert x == y, (key, part)


def _results(reward: float) -> dict:
    row = {"subset": "thin_file", "mode": "warm", "policy": "thompson", "n_decisions": 10, "cum_reward": reward,
           "cum_regret": 5.0, "cum_realized": 1.0, "decline_rate": 0.1}
    return {"fits": {"thin_file": {"auc": 0.65, "brier": 0.13, "ece": 0.03}},
            "runs": {"thin_file:warm": {"trip_epochs": ["2014-04"]}}, "policies": [row]}


def test_compare_flags_any_difference(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(_results(100.0)))
    (tmp_path / "b.json").write_text(json.dumps(_results(100.0)))
    (tmp_path / "c.json").write_text(json.dumps(_results(100.5)))
    assert study.compare(tmp_path / "a.json", tmp_path / "b.json", 1e-9) == 0
    assert study.compare(tmp_path / "a.json", tmp_path / "c.json", 1e-9) == 1
    assert "cum_reward" in capsys.readouterr().out
    assert study.compare(tmp_path / "a.json", tmp_path / "c.json", 0.01) == 0


def _run_dir(root: Path, sim: int, status: str) -> Path:
    run = root / "artifacts" / "simulations" / str(sim) / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "epoch=2015-01.pkl").write_bytes(b"x" * 100)
    (run / "progress.json").write_text(json.dumps({"status": status}))
    for keep in ("timeseries.parquet", "events.jsonl", "sentinel.parquet", "report.json"):
        (run / keep).write_text("keep")
    (run / "decisions").mkdir()
    (run / "decisions" / "epoch=2015-01.parquet").write_text("keep")
    return run


def test_clean_removes_finished_checkpoints_and_caches_only(tmp_path):
    done = _run_dir(tmp_path, 1, "completed")
    live = _run_dir(tmp_path, 2, "running")
    failed = _run_dir(tmp_path, 3, "failed")
    fit = tmp_path / "artifacts" / "fits" / "1"
    fit.mkdir(parents=True)
    (fit / "two_stage.json").write_text("keep")
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "Loan_status_2007-2020Q3.canonical.v3.parquet").write_bytes(b"cache")
    (proc / "Loan_status_2007-2020Q3.canonical.v3.json").write_text("{}")
    results = tmp_path / "runs" / "ablation-pilot"
    results.mkdir(parents=True)
    (results / "results.json").write_text("{}")

    assert clean.main(["--data-root", str(tmp_path)]) == 0  # dry run: nothing goes
    assert (done / "checkpoints").is_dir()
    # a simulation is still running, so the loader cache stays even with --yes
    assert clean.main(["--data-root", str(tmp_path), "--yes"]) == 0
    assert not (done / "checkpoints").exists()
    assert (live / "checkpoints").is_dir() and (failed / "checkpoints").is_dir()
    assert (proc / "Loan_status_2007-2020Q3.canonical.v3.parquet").is_file()
    for kept in (done / "timeseries.parquet", done / "events.jsonl", done / "decisions" / "epoch=2015-01.parquet",
                 fit / "two_stage.json", results / "results.json"):
        assert kept.is_file(), kept

    (live / "progress.json").write_text(json.dumps({"status": "completed"}))
    assert clean.main(["--data-root", str(tmp_path), "--yes", "--unfinished"]) == 0
    assert not (live / "checkpoints").exists() and not (failed / "checkpoints").exists()
    assert not list(proc.glob("*.canonical.*"))
