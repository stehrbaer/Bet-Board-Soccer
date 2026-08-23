from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


TARGET_CLASSES = [0, 1, 2]  # home, draw, away
TARGET_LABELS = {0: "home", 1: "draw", 2: "away"}

LEAKAGE_PATTERNS = [
    "score",
    "home_win",
    "result_3way",
    "result_target",
    "goal_differential",
    "total_score",
    "total_goals",
    "reg_",
    "actual_",
    "postgame",
    "settled",
    "final",
    "home_goals",
    "away_goals",
    "odds",
    "moneyline",
    "implied_prob",
    "market",
    "b365",
    "bookmaker",
]

IDENTIFIER_COLUMNS = {
    "match_id",
    "event_id",
    "event_date",
    "kickoff_utc",
    "competition_id",
    "season",
    "season_year",
    "season_type",
    "home_team_id",
    "away_team_id",
    "feature_cutoff_utc",
    "feature_build_timestamp",
    "feature_schema_version",
    "team_data_completeness",
    "player_data_completeness",
    "lineup_data_status",
    "weather_data_status",
}


@dataclass
class PreparedFold:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler


def one_hot_target(values: Iterable[int]) -> np.ndarray:
    raw = np.asarray(list(values), dtype=int)
    out = np.zeros((len(raw), len(TARGET_CLASSES)), dtype=float)
    for idx, cls in enumerate(TARGET_CLASSES):
        out[:, idx] = raw == cls
    return out


def choose_numeric_feature_columns(df: pd.DataFrame, max_features: int) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS)
    for col in df.columns:
        low = col.lower()
        if any(pattern in low for pattern in LEAKAGE_PATTERNS):
            excluded.add(col)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded]
    if not numeric_cols:
        return []
    variances = df[numeric_cols].var(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
    return variances.sort_values(ascending=False).head(max_features).index.tolist()


def make_walkforward_folds(
    df: pd.DataFrame,
    n_folds: int,
    val_size: int,
    min_train_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_rows = len(df)
    max_start = n_rows - val_size
    if max_start <= min_train_size:
        raise ValueError(
            f"Not enough rows for walk-forward folds: rows={n_rows}, "
            f"min_train_size={min_train_size}, val_size={val_size}"
        )
    starts = np.linspace(min_train_size, max_start, num=n_folds, dtype=int)
    return [(np.arange(0, start), np.arange(start, start + val_size)) for start in starts]


def prepare_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> PreparedFold:
    x_train_raw = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable_cols = x_train_raw.columns[x_train_raw.notna().any(axis=0)].tolist()
    if not usable_cols:
        raise ValueError("No feature columns have observed values in the training fold.")
    x_train_raw = x_train_raw[usable_cols]
    x_val_raw = val_df[usable_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw))
    x_val = scaler.transform(imputer.transform(x_val_raw))
    y_train = one_hot_target(train_df["result_target"].astype(int).tolist())
    y_val = one_hot_target(val_df["result_target"].astype(int).tolist())
    return PreparedFold(
        x_train=np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0),
        y_train=y_train,
        x_val=np.nan_to_num(x_val, nan=0.0, posinf=0.0, neginf=0.0),
        y_val=y_val,
        feature_names=usable_cols,
        imputer=imputer,
        scaler=scaler,
    )
