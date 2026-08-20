from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "tune_draw_thresholds.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("tune_draw_thresholds", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        predictions="",
        output_dir=str(tmp_path / "draw_threshold_tuning"),
        tune_each_league=False,
        models_root=str(tmp_path),
        prediction_filename="test_predictions.parquet",
        min_draw_p="0.30",
        max_draw_gap="0.10",
        max_side_gap="0.06",
        min_balanced_draw_p="0.28",
        min_draw_pick_p="0.22",
        policy_version="draw_policy_test",
        leagues="",
        export_config_dir=str(tmp_path / "configs" / "draw_policies"),
    )


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prob_home": [0.36, 0.70, 0.25],
            "prob_draw": [0.34, 0.20, 0.45],
            "prob_away": [0.30, 0.10, 0.30],
            "result_target": [1, 0, 1],
        }
    )


def test_tune_prediction_file_writes_outputs(tmp_path: Path) -> None:
    module = load_script_module()
    predictions = tmp_path / "test_predictions.parquet"
    sample_predictions().to_parquet(predictions, index=False)

    summary = module.tune_prediction_file(predictions, tmp_path / "draw_threshold_tuning", tiny_args(tmp_path))

    assert summary["active_policy"]["version"] == "draw_policy_test"
    assert (tmp_path / "draw_threshold_tuning" / "active_draw_policy.json").exists()
    assert (tmp_path / "draw_threshold_tuning" / "draw_threshold_grid.csv").exists()


def test_tune_each_league_writes_batch_summary(tmp_path: Path) -> None:
    module = load_script_module()
    for league in ["eng1_soccer_nn", "ger1_soccer_nn"]:
        model_dir = tmp_path / league
        model_dir.mkdir()
        sample_predictions().to_parquet(model_dir / "test_predictions.parquet", index=False)
    args = tiny_args(tmp_path)
    args.tune_each_league = True
    args.models_root = str(tmp_path)

    summary = module.tune_each_league(args)

    assert len(summary["completed"]) == 2
    assert not summary["failures"]
    assert (tmp_path / "eng1_soccer_nn" / "draw_threshold_tuning" / "active_draw_policy.json").exists()
    assert (tmp_path / "configs" / "draw_policies" / "eng1_draw_policy.json").exists()
    assert (tmp_path / "draw_threshold_batch_summary.json").exists()


def test_tune_each_league_can_filter_specific_leagues(tmp_path: Path) -> None:
    module = load_script_module()
    for league in ["aut1_soccer_nn", "ger2_soccer_nn", "por1_soccer_nn"]:
        model_dir = tmp_path / league
        model_dir.mkdir()
        sample_predictions().to_parquet(model_dir / "test_predictions.parquet", index=False)
    args = tiny_args(tmp_path)
    args.tune_each_league = True
    args.models_root = str(tmp_path)
    args.leagues = "ger2,por1"

    summary = module.tune_each_league(args)

    assert [row["competition_id"] for row in summary["completed"]] == ["ger.2", "por.1"]
    assert not (tmp_path / "aut1_soccer_nn" / "draw_threshold_tuning" / "active_draw_policy.json").exists()
    assert (tmp_path / "configs" / "draw_policies" / "ger2_draw_policy.json").exists()
