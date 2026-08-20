#!/usr/bin/env python3
"""Tune simple draw-risk decision thresholds on held-out soccer predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


LEAGUE_ALIASES = {
    "aut1": "aut.1",
    "den1": "den.1",
    "eng1": "eng.1",
    "epl": "eng.1",
    "eng2": "eng.2",
    "championship": "eng.2",
    "eng3": "eng.3",
    "esp1": "esp.1",
    "laliga": "esp.1",
    "esp2": "esp.2",
    "fra1": "fra.1",
    "fra2": "fra.2",
    "ger1": "ger.1",
    "bundesliga": "ger.1",
    "ger2": "ger.2",
    "bundesliga2": "ger.2",
    "ita1": "ita.1",
    "seriea": "ita.1",
    "ita2": "ita.2",
    "ned1": "ned.1",
    "eredivisie": "ned.1",
    "por1": "por.1",
    "sco1": "sco.1",
}


try:
    from betboard_soccer_extension.modeling.draw_policy import DrawPolicy, apply_draw_policy, evaluate_draw_policy
except ModuleNotFoundError:
    if "__file__" in globals():
        candidate = Path(__file__).resolve().parents[2] / "src"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
    from betboard_soccer_extension.modeling.draw_policy import DrawPolicy, apply_draw_policy, evaluate_draw_policy  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune draw-risk thresholds from held-out prediction results.")
    parser.add_argument("--predictions", default="", help="Path to test_predictions.parquet or CSV.")
    parser.add_argument("--output-dir", default="outputs/eng1_soccer_nn/draw_threshold_tuning")
    parser.add_argument(
        "--tune-each-league",
        action="store_true",
        help="Tune every model folder under --models-root that contains test_predictions.parquet.",
    )
    parser.add_argument("--models-root", default="outputs/soccer_nn_by_league")
    parser.add_argument("--prediction-filename", default="test_predictions.parquet")
    parser.add_argument(
        "--leagues",
        default="",
        help="Optional comma-separated league filter for --tune-each-league, e.g. eng1,ger2,aut1.",
    )
    parser.add_argument(
        "--export-config-dir",
        default="configs/draw_policies",
        help="Optional folder for mirrored per-league draw policy configs. Use empty string to disable.",
    )
    parser.add_argument("--min-draw-p", default="0.20,0.22,0.24,0.26,0.28,0.30,0.32")
    parser.add_argument("--max-draw-gap", default="0.02,0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--max-side-gap", default="0.06,0.08,0.10,0.12,0.15,0.18")
    parser.add_argument("--min-balanced-draw-p", default="0.20,0.22,0.24,0.26,0.28")
    parser.add_argument("--min-draw-pick-p", default="0.22,0.24,0.26,0.28,0.30")
    parser.add_argument("--policy-version", default="draw_policy_tuned_v1")
    return parser.parse_args()


def grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_league(value: str) -> str:
    key = value.strip().lower().replace("-", "").replace("_", "").replace(".", "")
    return LEAGUE_ALIASES.get(key, value.strip().lower())


def league_slug(competition_id: str) -> str:
    return competition_id.lower().replace(".", "").replace("_", "").replace("-", "")


def competition_from_model_dir(model_dir: Path) -> str:
    name = model_dir.name
    if name.endswith("_soccer_nn"):
        name = name.removesuffix("_soccer_nn")
    normalized = normalize_league(name)
    if "." in normalized:
        return normalized
    if len(name) > 3 and name[-1].isdigit():
        return f"{name[:-1]}.{name[-1]}"
    return normalized


def selected_leagues(args: argparse.Namespace) -> set[str]:
    return {normalize_league(league) for league in csv_list(getattr(args, "leagues", ""))}


def read_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def apply_policy(
    df: pd.DataFrame,
    min_draw_p: float,
    max_draw_gap: float,
    max_side_gap: float,
    min_balanced_draw_p: float,
    min_draw_pick_p: float,
) -> pd.DataFrame:
    policy = DrawPolicy(
        version="grid_candidate",
        min_draw_p=min_draw_p,
        max_draw_gap=max_draw_gap,
        max_side_gap=max_side_gap,
        min_balanced_draw_p=min_balanced_draw_p,
        min_draw_pick_p=min_draw_pick_p,
    )
    return apply_draw_policy(df, policy)


def tune(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for min_draw_p in grid(args.min_draw_p):
        for max_draw_gap in grid(args.max_draw_gap):
            for max_side_gap in grid(args.max_side_gap):
                for min_balanced_draw_p in grid(args.min_balanced_draw_p):
                    for min_draw_pick_p in grid(args.min_draw_pick_p):
                        policy = apply_policy(
                            df,
                            min_draw_p=min_draw_p,
                            max_draw_gap=max_draw_gap,
                            max_side_gap=max_side_gap,
                            min_balanced_draw_p=min_balanced_draw_p,
                            min_draw_pick_p=min_draw_pick_p,
                        )
                        metrics = evaluate_draw_policy(policy)
                        rows.append(
                            {
                                "min_draw_p": min_draw_p,
                                "max_draw_gap": max_draw_gap,
                                "max_side_gap": max_side_gap,
                                "min_balanced_draw_p": min_balanced_draw_p,
                                "min_draw_pick_p": min_draw_pick_p,
                                **metrics,
                            }
                        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["policy_accuracy", "draw_pick_recall", "draw_risk_recall", "draw_pick_precision"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def tune_prediction_file(
    predictions_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    competition_id: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_predictions(str(predictions_path))
    required = {"prob_home", "prob_draw", "prob_away", "result_target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Prediction file {predictions_path} is missing required columns: {missing}")

    results = tune(df, args)
    raw = apply_policy(df, 1.0, -1.0, -1.0, 1.0, 1.0)
    raw_metrics = evaluate_draw_policy(raw)
    best = results.iloc[0].to_dict()
    balanced = results[
        (results["policy_accuracy"] >= raw_metrics["raw_accuracy"] - 0.02)
        & (results["draw_pick_recall"].fillna(0) > 0)
    ].head(20)

    results_path = output_dir / "draw_threshold_grid.csv"
    best_path = output_dir / "best_draw_thresholds.json"
    active_policy_path = output_dir / "active_draw_policy.json"
    balanced_path = output_dir / "balanced_draw_threshold_candidates.csv"
    results.to_csv(results_path, index=False)
    balanced.to_csv(balanced_path, index=False)
    active_policy = DrawPolicy(
        version=f"{args.policy_version}_{league_slug(competition_id)}" if competition_id else args.policy_version,
        min_draw_p=float(best["min_draw_p"]),
        max_draw_gap=float(best["max_draw_gap"]),
        max_side_gap=float(best["max_side_gap"]),
        min_balanced_draw_p=float(best["min_balanced_draw_p"]),
        min_draw_pick_p=float(best["min_draw_pick_p"]),
        source=str(predictions_path),
        metrics={key: best[key] for key in best if key not in {"min_draw_p", "max_draw_gap", "max_side_gap", "min_balanced_draw_p", "min_draw_pick_p"}},
    )
    active_policy_path.write_text(json.dumps(active_policy.to_dict(), indent=2, default=str) + "\n")
    exported_config_path = None
    export_config_dir = getattr(args, "export_config_dir", "")
    if competition_id and export_config_dir:
        config_dir = Path(export_config_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        exported_config_path = config_dir / f"{league_slug(competition_id)}_draw_policy.json"
        exported_config_path.write_text(json.dumps(active_policy.to_dict(), indent=2, default=str) + "\n")
    summary = {
        "predictions": str(predictions_path),
        "competition_id": competition_id,
        "raw_metrics": raw_metrics,
        "best_by_accuracy": best,
        "active_policy": active_policy.to_dict(),
        "outputs": {
            "grid": str(results_path),
            "balanced_candidates": str(balanced_path),
            "best": str(best_path),
            "active_policy": str(active_policy_path),
            "exported_config": None if exported_config_path is None else str(exported_config_path),
        },
    }
    best_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def tune_each_league(args: argparse.Namespace) -> dict[str, Any]:
    models_root = Path(args.models_root)
    prediction_paths = sorted(models_root.glob(f"*/{args.prediction_filename}"))
    if not prediction_paths:
        raise RuntimeError(f"No {args.prediction_filename} files found under {models_root}.")

    completed = []
    failures = []
    requested_leagues = selected_leagues(args)
    for predictions_path in prediction_paths:
        model_dir = predictions_path.parent
        competition_id = competition_from_model_dir(model_dir)
        if requested_leagues and competition_id not in requested_leagues:
            continue
        output_dir = model_dir / "draw_threshold_tuning"
        try:
            summary = tune_prediction_file(predictions_path, output_dir, args, competition_id=competition_id)
            completed.append(
                {
                    "competition_id": competition_id,
                    "model_dir": str(model_dir),
                    "predictions": str(predictions_path),
                    "output_dir": str(output_dir),
                    "raw_accuracy": summary["raw_metrics"]["raw_accuracy"],
                    "policy_accuracy": summary["best_by_accuracy"]["policy_accuracy"],
                    "draw_pick_recall": summary["best_by_accuracy"]["draw_pick_recall"],
                    "active_policy": summary["outputs"]["active_policy"],
                    "exported_config": summary["outputs"]["exported_config"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch tuning going across partial folders.
            failures.append({"model_dir": str(model_dir), "predictions": str(predictions_path), "error": str(exc)})

    batch_summary = {
        "models_root": str(models_root),
        "prediction_filename": args.prediction_filename,
        "leagues": sorted(requested_leagues) if requested_leagues else ["all"],
        "export_config_dir": getattr(args, "export_config_dir", ""),
        "completed": completed,
        "failures": failures,
    }
    batch_path = models_root / "draw_threshold_batch_summary.json"
    batch_path.write_text(json.dumps(batch_summary, indent=2, default=str) + "\n")
    print(json.dumps(batch_summary, indent=2, default=str))
    return batch_summary


def main() -> int:
    args = parse_args()
    if args.tune_each_league:
        tune_each_league(args)
        return 0
    if not args.predictions:
        raise SystemExit("--predictions is required unless --tune-each-league is set.")
    summary = tune_prediction_file(Path(args.predictions), Path(args.output_dir), args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
