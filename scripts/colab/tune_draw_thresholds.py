#!/usr/bin/env python3
"""Tune simple draw-risk decision thresholds on held-out soccer predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_TO_TARGET = {"home": 0, "draw": 1, "away": 2}
TARGET_TO_LABEL = {value: key for key, value in LABEL_TO_TARGET.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune draw-risk thresholds from held-out prediction results.")
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.parquet or CSV.")
    parser.add_argument("--output-dir", default="outputs/eng1_soccer_nn/draw_threshold_tuning")
    parser.add_argument("--min-draw-p", default="0.20,0.22,0.24,0.26,0.28,0.30,0.32")
    parser.add_argument("--max-draw-gap", default="0.02,0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--max-side-gap", default="0.06,0.08,0.10,0.12,0.15,0.18")
    parser.add_argument("--min-balanced-draw-p", default="0.20,0.22,0.24,0.26,0.28")
    parser.add_argument("--min-draw-pick-p", default="0.22,0.24,0.26,0.28,0.30")
    return parser.parse_args()


def grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def read_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def base_pick(df: pd.DataFrame) -> pd.Series:
    labels = np.array(["home", "draw", "away"])
    return pd.Series(labels[np.argmax(df[["prob_home", "prob_draw", "prob_away"]].to_numpy(), axis=1)], index=df.index)


def apply_policy(
    df: pd.DataFrame,
    min_draw_p: float,
    max_draw_gap: float,
    max_side_gap: float,
    min_balanced_draw_p: float,
    min_draw_pick_p: float,
) -> pd.DataFrame:
    out = df.copy()
    out["raw_model_pick"] = base_pick(out)
    out["top_side_prob"] = out[["prob_home", "prob_away"]].max(axis=1)
    out["draw_gap"] = out["top_side_prob"] - out["prob_draw"]
    out["home_away_gap"] = (out["prob_home"] - out["prob_away"]).abs()
    out["draw_risk"] = (
        (out["prob_draw"] >= min_draw_p)
        | (out["draw_gap"] <= max_draw_gap)
        | ((out["prob_draw"] >= min_balanced_draw_p) & (out["home_away_gap"] <= max_side_gap))
    )
    out["recommended_pick"] = out["raw_model_pick"]
    out.loc[out["draw_risk"] & (out["prob_draw"] >= min_draw_pick_p), "recommended_pick"] = "draw"
    return out


def evaluate(policy_df: pd.DataFrame) -> dict[str, Any]:
    actual = pd.to_numeric(policy_df["result_target"], errors="coerce").astype("Int64")
    keep = actual.isin([0, 1, 2])
    df = policy_df[keep].copy()
    actual = actual[keep].astype(int)
    raw_target = df["raw_model_pick"].map(LABEL_TO_TARGET)
    rec_target = df["recommended_pick"].map(LABEL_TO_TARGET)
    draw_actual = actual == LABEL_TO_TARGET["draw"]
    draw_recommended = df["recommended_pick"] == "draw"
    draw_risk = df["draw_risk"].astype(bool)

    draw_pick_correct = int((draw_recommended & draw_actual).sum())
    draw_pick_count = int(draw_recommended.sum())
    actual_draw_count = int(draw_actual.sum())
    return {
        "rows": int(len(df)),
        "actual_draw_rate": float(draw_actual.mean()),
        "raw_accuracy": float((raw_target == actual).mean()),
        "policy_accuracy": float((rec_target == actual).mean()),
        "accuracy_delta": float((rec_target == actual).mean() - (raw_target == actual).mean()),
        "raw_draw_pick_rate": float((df["raw_model_pick"] == "draw").mean()),
        "policy_draw_pick_rate": float(draw_recommended.mean()),
        "draw_risk_rate": float(draw_risk.mean()),
        "draw_pick_precision": None if draw_pick_count == 0 else float(draw_pick_correct / draw_pick_count),
        "draw_pick_recall": None if actual_draw_count == 0 else float(draw_pick_correct / actual_draw_count),
        "draw_risk_recall": None if actual_draw_count == 0 else float((draw_risk & draw_actual).sum() / actual_draw_count),
        "draw_risk_precision": None if int(draw_risk.sum()) == 0 else float((draw_risk & draw_actual).sum() / int(draw_risk.sum())),
        "draw_pick_count": draw_pick_count,
        "actual_draw_count": actual_draw_count,
    }


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
                        metrics = evaluate(policy)
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


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_predictions(args.predictions)
    required = {"prob_home", "prob_draw", "prob_away", "result_target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Prediction file is missing required columns: {missing}")

    results = tune(df, args)
    raw = apply_policy(df, 1.0, -1.0, -1.0, 1.0, 1.0)
    raw_metrics = evaluate(raw)
    best = results.iloc[0].to_dict()
    balanced = results[
        (results["policy_accuracy"] >= raw_metrics["raw_accuracy"] - 0.02)
        & (results["draw_pick_recall"].fillna(0) > 0)
    ].head(20)

    results_path = output_dir / "draw_threshold_grid.csv"
    best_path = output_dir / "best_draw_thresholds.json"
    balanced_path = output_dir / "balanced_draw_threshold_candidates.csv"
    results.to_csv(results_path, index=False)
    balanced.to_csv(balanced_path, index=False)
    summary = {
        "predictions": args.predictions,
        "raw_metrics": raw_metrics,
        "best_by_accuracy": best,
        "outputs": {
            "grid": str(results_path),
            "balanced_candidates": str(balanced_path),
            "best": str(best_path),
        },
    }
    best_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
