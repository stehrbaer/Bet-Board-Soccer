from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_TO_TARGET = {"home": 0, "draw": 1, "away": 2}
TARGET_TO_LABEL = {value: key for key, value in LABEL_TO_TARGET.items()}
PROBABILITY_COLUMNS = ["prob_home", "prob_draw", "prob_away"]


@dataclass(frozen=True)
class DrawPolicy:
    version: str
    min_draw_p: float
    max_draw_gap: float
    max_side_gap: float
    min_balanced_draw_p: float
    min_draw_pick_p: float
    source: str = ""
    metrics: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DrawPolicy":
        return cls(
            version=str(value.get("version", "draw_policy_unversioned")),
            min_draw_p=float(value["min_draw_p"]),
            max_draw_gap=float(value["max_draw_gap"]),
            max_side_gap=float(value["max_side_gap"]),
            min_balanced_draw_p=float(value["min_balanced_draw_p"]),
            min_draw_pick_p=float(value["min_draw_pick_p"]),
            source=str(value.get("source", "")),
            metrics=value.get("metrics"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_draw_policy(path: str | Path) -> DrawPolicy:
    return DrawPolicy.from_mapping(json.loads(Path(path).read_text()))


def save_draw_policy(policy: DrawPolicy, path: str | Path) -> None:
    Path(path).write_text(json.dumps(policy.to_dict(), indent=2, default=str) + "\n")


def raw_model_pick(df: pd.DataFrame) -> pd.Series:
    labels = np.array(["home", "draw", "away"])
    return pd.Series(labels[np.argmax(df[PROBABILITY_COLUMNS].to_numpy(), axis=1)], index=df.index)


def apply_draw_policy(df: pd.DataFrame, policy: DrawPolicy) -> pd.DataFrame:
    missing = [column for column in PROBABILITY_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"Prediction dataframe is missing probability columns: {missing}")
    out = df.copy()
    out["raw_model_pick"] = raw_model_pick(out)
    out["top_side_prob"] = out[["prob_home", "prob_away"]].max(axis=1)
    out["draw_gap"] = out["top_side_prob"] - out["prob_draw"]
    out["home_away_gap"] = (out["prob_home"] - out["prob_away"]).abs()
    out["draw_risk"] = (
        (out["prob_draw"] >= policy.min_draw_p)
        | (out["draw_gap"] <= policy.max_draw_gap)
        | ((out["prob_draw"] >= policy.min_balanced_draw_p) & (out["home_away_gap"] <= policy.max_side_gap))
    )
    out["recommended_pick"] = out["raw_model_pick"]
    out.loc[out["draw_risk"] & (out["prob_draw"] >= policy.min_draw_pick_p), "recommended_pick"] = "draw"
    out["draw_policy_version"] = policy.version
    return out


def evaluate_draw_policy(policy_df: pd.DataFrame) -> dict[str, Any]:
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
