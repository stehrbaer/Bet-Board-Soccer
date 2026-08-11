#!/usr/bin/env python3
"""Colab-ready neural three-way soccer training script.

Target contract:
    0 = home win
    1 = draw
    2 = away win

Example Colab run:
    !pip install pandas pyarrow s3fs scikit-learn tensorflow optuna joblib
    !python train_soccer_three_way_nn_optuna.py \
      --input s3://betboard-ml-artifacts/soccer-prediction-data/gold/prematch_model_input \
      --competitions eng.1,esp.1,ger.1,ita.1,fra.1 \
      --train-through-season 2024 \
      --test-season 2025 \
      --n-trials 50 \
      --epochs 150
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


LOGGER = logging.getLogger("soccer_nn")

LEAGUE_ALIASES = {
    "aut1": "aut.1",
    "aut_1": "aut.1",
    "austria": "aut.1",
    "den1": "den.1",
    "den_1": "den.1",
    "denmark": "den.1",
    "eng1": "eng.1",
    "eng_1": "eng.1",
    "epl": "eng.1",
    "premierleague": "eng.1",
    "eng2": "eng.2",
    "eng_2": "eng.2",
    "championship": "eng.2",
    "eng3": "eng.3",
    "eng_3": "eng.3",
    "leagueone": "eng.3",
    "esp1": "esp.1",
    "esp_1": "esp.1",
    "laliga": "esp.1",
    "esp2": "esp.2",
    "esp_2": "esp.2",
    "laliga2": "esp.2",
    "ger1": "ger.1",
    "ger_1": "ger.1",
    "bundesliga": "ger.1",
    "ger2": "ger.2",
    "ger_2": "ger.2",
    "bundesliga2": "ger.2",
    "ita1": "ita.1",
    "ita_1": "ita.1",
    "seriea": "ita.1",
    "ita2": "ita.2",
    "ita_2": "ita.2",
    "serieb": "ita.2",
    "fra1": "fra.1",
    "fra_1": "fra.1",
    "ligue1": "fra.1",
    "fra2": "fra.2",
    "fra_2": "fra.2",
    "ligue2": "fra.2",
    "ned1": "ned.1",
    "ned_1": "ned.1",
    "eredivisie": "ned.1",
    "por1": "por.1",
    "por_1": "por.1",
    "portugal": "por.1",
    "sco1": "sco.1",
    "sco_1": "sco.1",
    "scotland": "sco.1",
    "uclqual": "uefa.champions.qual",
    "ucl_qual": "uefa.champions.qual",
    "championsqual": "uefa.champions.qual",
    "champions_qual": "uefa.champions.qual",
    "championsleaguequal": "uefa.champions.qual",
    "uefa_champions_qual": "uefa.champions.qual",
}

CORE_LEAGUES = ["eng.1", "eng.2", "esp.1", "ger.1", "ita.1", "fra.1"]
DOMESTIC_LEAGUE_PATTERN = re.compile(r"^[a-z]{3}\.\d+$")

REQUIRED_MODEL_COLUMNS = [
    "match_id",
    "competition_id",
    "season",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "result_target",
]


try:
    from betboard_soccer_extension.modeling.nn_preprocessing import (
        TARGET_LABELS,
        choose_numeric_feature_columns,
        make_walkforward_folds,
        prepare_fold,
    )
except ModuleNotFoundError:
    # Allows copying this file into Colab without installing the package, as long
    # as the repo root or src directory is present beside the script.
    if "__file__" in globals():
        candidates = [Path(__file__).resolve().parents[2] / "src"]
    else:
        candidates = [Path.cwd() / "src", Path.cwd()]
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    from betboard_soccer_extension.modeling.nn_preprocessing import (  # type: ignore[no-redef]
        TARGET_LABELS,
        choose_numeric_feature_columns,
        make_walkforward_folds,
        prepare_fold,
    )


@dataclass
class TrainingConfig:
    input: str
    output_dir: str
    competitions: list[str]
    seasons: list[str]
    train_through_season: str
    test_season: str
    max_features: int
    n_folds: int
    val_size: int
    min_train_size: int
    epochs: int
    n_trials: int
    batch_size_default: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Optuna-tuned neural soccer 1X2 model.")
    parser.add_argument("--input", required=True, help="Gold dataset root or local parquet file.")
    parser.add_argument("--output-dir", default="outputs/soccer_three_way_nn")
    parser.add_argument(
        "--league",
        default="all",
        help="League selector such as eng1, eng_1, epl, esp1, ger1, ita1, fra1, or all.",
    )
    parser.add_argument("--competitions", default="", help="Comma-separated competition ids, e.g. eng.1,esp.1")
    parser.add_argument("--seasons", default="", help="Comma-separated seasons to load. Empty loads all under input.")
    parser.add_argument("--train-through-season", default="2024")
    parser.add_argument("--test-season", default="2025")
    parser.add_argument("--max-features", type=int, default=800)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--min-train-size", type=int, default=600)
    parser.add_argument(
        "--no-auto-fold-size",
        action="store_true",
        help="Disable automatic walk-forward fold downscaling for smaller leagues.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-each-league",
        action="store_true",
        help="Train one separate model per selected league and write each under output-dir/<league_slug>_soccer_nn.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --train-each-league mode, skip league folders that already contain model, preprocessing, predictions, and summary artifacts.",
    )
    parser.add_argument(
        "--full-scope-start-season",
        default="2021",
        help="When using --train-each-league --league all, only include competitions with every season from this value through --test-season.",
    )
    parser.add_argument(
        "--core-leagues-only",
        action="store_true",
        help="With --train-each-league --league all, limit discovery to the original core league set.",
    )
    parser.add_argument(
        "--next-games",
        type=int,
        default=5,
        help="Write the first N held-out test predictions by kickoff time for each training run. Use 0 to disable.",
    )
    parser.add_argument(
        "--duckdb-preprocess",
        action="store_true",
        help="Use DuckDB to scan/filter parquet before loading into pandas. Recommended for global models.",
    )
    parser.add_argument("--smoke", action="store_true", help="Override settings for a fast local/Colab smoke run.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def add_file_logging(output_dir: Path) -> Path:
    log_path = output_dir / "training.log"
    for handler in list(LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            LOGGER.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    LOGGER.addHandler(file_handler)
    return log_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_league(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(".", "_").replace(" ", "")
    if normalized == "all":
        return "all"
    return LEAGUE_ALIASES.get(normalized, normalized)


def league_slug(competition_id: str) -> str:
    return competition_id.lower().replace(".", "").replace("_", "").replace("-", "")


def resolve_competitions(args: argparse.Namespace) -> list[str]:
    competitions = csv_list(args.competitions)
    if competitions:
        resolved = [normalize_league(comp) for comp in competitions]
        if resolved == ["all"]:
            return []
        if "all" in resolved:
            raise SystemExit("--competitions all cannot be combined with specific competitions.")
        return resolved
    leagues = [normalize_league(league) for league in csv_list(args.league)]
    if not leagues or leagues == ["all"]:
        return []
    if "all" in leagues:
        raise SystemExit("--league all cannot be combined with specific leagues.")
    return leagues


def default_partition_seasons(train_through_season: str, test_season: str) -> str:
    start = 2021
    try:
        end = max(int(train_through_season), int(test_season))
    except ValueError:
        return ""
    return ",".join(str(season) for season in range(start, end + 1))


def s3_storage_options() -> dict[str, Any] | None:
    key = os.getenv("DO_SPACES_KEY")
    secret = os.getenv("DO_SPACES_SECRET")
    if not key or not secret:
        return None
    return {
        "key": key,
        "secret": secret,
        "client_kwargs": {
            "endpoint_url": os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com"),
            "region_name": os.getenv("DO_SPACES_REGION", "fra1"),
        },
    }


def s3_filesystem():
    storage_options = s3_storage_options()
    if storage_options is None:
        raise SystemExit(
            "DigitalOcean Spaces credentials are missing. In Colab set:\n"
            '  os.environ["DO_SPACES_KEY"] = "your_access_key"\n'
            '  os.environ["DO_SPACES_SECRET"] = "your_secret_key"\n'
            'Optional endpoint defaults to https://fra1.digitaloceanspaces.com.'
        )
    try:
        import s3fs
    except ModuleNotFoundError as exc:
        raise SystemExit("s3fs is missing. In Colab run: !pip install -r requirements-colab.txt") from exc
    return s3fs.S3FileSystem(**storage_options)


def s3_key(path: str) -> str:
    return path.removeprefix("s3://")


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_list(values: list[str]) -> str:
    return ", ".join(sql_literal(value) for value in values)


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def configure_duckdb_s3(con) -> None:
    key = os.getenv("DO_SPACES_KEY")
    secret = os.getenv("DO_SPACES_SECRET")
    if not key or not secret:
        raise SystemExit(
            "DigitalOcean Spaces credentials are missing. In Colab set DO_SPACES_KEY and DO_SPACES_SECRET "
            "before using --duckdb-preprocess against s3:// input."
        )
    endpoint = os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com")
    endpoint = endpoint.removeprefix("https://").removeprefix("http://")
    region = os.getenv("DO_SPACES_REGION", "fra1")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(f"SET s3_access_key_id={sql_literal(key)}")
    con.execute(f"SET s3_secret_access_key={sql_literal(secret)}")
    con.execute(f"SET s3_endpoint={sql_literal(endpoint)}")
    con.execute(f"SET s3_region={sql_literal(region)}")
    con.execute("SET s3_url_style='path'")
    con.execute("SET s3_use_ssl=true")


def duckdb_type_is_numeric(type_name: str) -> bool:
    normalized = type_name.upper()
    return any(
        token in normalized
        for token in [
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "REAL",
        ]
    )


def duckdb_read_expr(paths: list[str]) -> str:
    path_list = "[" + sql_list(paths) + "]"
    return f"read_parquet({path_list}, union_by_name=true, hive_partitioning=false)"


def duckdb_schema(con, paths: list[str]) -> list[tuple[str, str]]:
    query = f"DESCRIBE SELECT * FROM {duckdb_read_expr(paths)}"
    rows = con.execute(query).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def partition_paths(root: str, competitions: list[str], seasons: list[str]) -> list[str]:
    if not competitions or not seasons:
        return [root]
    return [f"{root.rstrip('/')}/competition={comp}/season={season}/part-000.parquet" for comp in competitions for season in seasons]


def available_competition_seasons(root: str) -> dict[str, list[str]]:
    """Return available gold competition seasons for local or S3 partition roots."""
    root = root.rstrip("/")
    if root.startswith("s3://"):
        filesystem = s3_filesystem()
        root_key = s3_key(root)
        competitions = []
        for key in filesystem.ls(root_key, detail=False):
            name = key.rstrip("/").rsplit("/", 1)[-1]
            if name.startswith("competition="):
                competitions.append(name.removeprefix("competition="))
        result: dict[str, list[str]] = {}
        for competition in sorted(competitions):
            season_prefix = f"{root_key}/competition={competition}"
            seasons = []
            for key in filesystem.ls(season_prefix, detail=False):
                name = key.rstrip("/").rsplit("/", 1)[-1]
                if name.startswith("season="):
                    seasons.append(name.removeprefix("season="))
            result[competition] = sorted(seasons, key=lambda item: int(item) if item.isdigit() else item)
        return result

    root_path = Path(root)
    result = {}
    for competition_dir in sorted(root_path.glob("competition=*")):
        if not competition_dir.is_dir():
            continue
        competition = competition_dir.name.removeprefix("competition=")
        seasons = [
            season_dir.name.removeprefix("season=")
            for season_dir in sorted(competition_dir.glob("season=*"))
            if season_dir.is_dir()
        ]
        result[competition] = sorted(seasons, key=lambda item: int(item) if item.isdigit() else item)
    return result


def required_scope_seasons(start_season: str, test_season: str) -> list[str]:
    try:
        start = int(start_season)
        end = int(test_season)
    except ValueError:
        return []
    if start > end:
        raise SystemExit("--full-scope-start-season cannot be after --test-season.")
    return [str(season) for season in range(start, end + 1)]


def discover_full_scope_competitions(args: argparse.Namespace) -> tuple[list[str], dict[str, list[str]]]:
    available = available_competition_seasons(args.input)
    required = required_scope_seasons(args.full_scope_start_season, args.test_season)
    selected = []
    for competition, seasons in available.items():
        if args.core_leagues_only and competition not in CORE_LEAGUES:
            continue
        if not DOMESTIC_LEAGUE_PATTERN.match(competition):
            continue
        if all(season in seasons for season in required):
            selected.append(competition)
    return selected, available


def is_global_default_scope(args: argparse.Namespace) -> bool:
    return (
        not args.train_each_league
        and not csv_list(args.competitions)
        and [normalize_league(league) for league in csv_list(args.league)] in ([], ["all"])
    )


def read_parquet_path(path: str, filesystem=None) -> pd.DataFrame:
    if filesystem is None:
        return pd.read_parquet(path)
    with filesystem.open(s3_key(path), "rb") as handle:
        return pd.read_parquet(handle)


def load_gold_dataset_duckdb(root: str, competitions: list[str], seasons: list[str], output_dir: Path | None = None) -> pd.DataFrame:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise SystemExit("duckdb is missing. In Colab run: !pip install -r requirements-colab.txt") from exc

    filesystem = s3_filesystem() if root.startswith("s3://") else None
    paths = partition_paths(root, competitions, seasons)
    if filesystem is not None and (not competitions or not seasons) and not root.endswith(".parquet"):
        root_key = s3_key(root).rstrip("/")
        parquet_keys = sorted(filesystem.glob(f"{root_key}/**/*.parquet"))
        paths = [f"s3://{path}" for path in parquet_keys]
    if not paths:
        raise RuntimeError("No parquet paths selected for DuckDB preprocessing.")

    con = duckdb.connect(database=":memory:")
    if root.startswith("s3://"):
        configure_duckdb_s3(con)

    started = time.monotonic()
    LOGGER.info(
        "duckdb preprocessing started parquet_files=%s input=%s competitions=%s seasons=%s",
        len(paths),
        root,
        competitions or ["all"],
        seasons or ["all"],
    )
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "duckdb_schema",
                "parquet_files": len(paths),
                "input": root,
                "competitions": competitions or ["all"],
                "seasons": seasons or ["all"],
            },
        )

    schema = duckdb_schema(con, paths)
    schema_map = {name: type_name for name, type_name in schema}
    missing_required = [column for column in REQUIRED_MODEL_COLUMNS if column not in schema_map]
    if missing_required:
        raise RuntimeError(f"Input dataset is missing required columns: {missing_required}")

    numeric_columns = [
        name
        for name, type_name in schema
        if duckdb_type_is_numeric(type_name) and name not in REQUIRED_MODEL_COLUMNS
    ]
    keep_columns = list(dict.fromkeys(REQUIRED_MODEL_COLUMNS + numeric_columns))
    select_exprs = []
    for column in keep_columns:
        ident = sql_identifier(column)
        if column == "season":
            select_exprs.append(f"CAST({ident} AS VARCHAR) AS {ident}")
        elif column == "result_target":
            select_exprs.append(f"TRY_CAST({ident} AS INTEGER) AS {ident}")
        elif column == "kickoff_utc":
            select_exprs.append(f"TRY_CAST({ident} AS TIMESTAMPTZ) AS {ident}")
        else:
            select_exprs.append(ident)

    predicates = [
        "kickoff_utc IS NOT NULL",
        "TRY_CAST(result_target AS INTEGER) IN (0, 1, 2)",
    ]
    if competitions:
        predicates.append(f"CAST(competition_id AS VARCHAR) IN ({sql_list(competitions)})")
    if seasons:
        predicates.append(f"CAST(season AS VARCHAR) IN ({sql_list(seasons)})")

    query = f"""
        SELECT {", ".join(select_exprs)}
        FROM {duckdb_read_expr(paths)}
        WHERE {" AND ".join(predicates)}
        ORDER BY kickoff_utc, match_id
    """
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "duckdb_query",
                "parquet_files": len(paths),
                "schema_columns": len(schema),
                "kept_columns": len(keep_columns),
                "numeric_columns": len(numeric_columns),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    df = con.execute(query).fetchdf()
    con.close()
    LOGGER.info(
        "duckdb preprocessing complete rows=%s columns=%s elapsed_seconds=%.1f",
        len(df),
        len(df.columns),
        time.monotonic() - started,
    )
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "loaded",
                "engine": "duckdb",
                "rows": len(df),
                "columns": len(df.columns),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    if df.empty:
        raise RuntimeError("DuckDB preprocessing returned no input rows.")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df["season"] = df["season"].astype(str)
    df["result_target"] = pd.to_numeric(df["result_target"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def load_gold_dataset(root: str, competitions: list[str], seasons: list[str], output_dir: Path | None = None) -> pd.DataFrame:
    filesystem = s3_filesystem() if root.startswith("s3://") else None
    paths = partition_paths(root, competitions, seasons)
    if filesystem is not None and not competitions and not root.endswith(".parquet"):
        root_key = s3_key(root).rstrip("/")
        if seasons:
            parquet_keys = []
            for season in seasons:
                parquet_keys.extend(sorted(filesystem.glob(f"{root_key}/competition=*/season={season}/*.parquet")))
        else:
            parquet_keys = sorted(filesystem.glob(f"{root_key}/**/*.parquet"))
        paths = [f"s3://{path}" for path in parquet_keys]
    LOGGER.info(
        "loading parquet files count=%s input=%s competitions=%s seasons=%s",
        len(paths),
        root,
        competitions or ["all"],
        seasons or ["all"],
    )
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "reading_parquet",
                "parquet_files": len(paths),
                "input": root,
                "competitions": competitions or ["all"],
                "seasons": seasons or ["all"],
            },
        )
    frames: list[pd.DataFrame] = []
    started = time.monotonic()
    for idx, path in enumerate(paths, start=1):
        try:
            LOGGER.info("reading parquet %s/%s %s", idx, len(paths), path)
            frame = read_parquet_path(path, filesystem)
            if not frame.empty:
                frame = frame.dropna(axis=1, how="all")
            frames.append(frame)
            if output_dir is not None and (idx == len(paths) or idx % 10 == 0):
                write_json(
                    output_dir / "load_status.json",
                    {
                        "stage": "reading_parquet",
                        "parquet_files": len(paths),
                        "files_read": idx,
                        "non_empty_frames": len([item for item in frames if not item.empty]),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    },
                )
        except FileNotFoundError:
            LOGGER.warning("missing partition: %s", path)
    if not frames:
        raise RuntimeError("No input rows loaded.")
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("Only empty input parquet files were loaded.")
    LOGGER.info("concatenating dataframes count=%s", len(frames))
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "concatenating",
                "dataframes": len(frames),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    df = pd.concat(frames, ignore_index=True)
    LOGGER.info("concat complete rows=%s columns=%s", len(df), len(df.columns))
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "concat_complete",
                "rows": len(df),
                "columns": len(df.columns),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    LOGGER.info("cleaning loaded dataframe")
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "cleaning",
                "rows": len(df),
                "columns": len(df.columns),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    missing_required = [column for column in REQUIRED_MODEL_COLUMNS if column not in df.columns]
    if missing_required:
        raise RuntimeError(f"Input dataset is missing required columns: {missing_required}")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df["season"] = df["season"].astype(str)
    df["result_target"] = pd.to_numeric(df["result_target"], errors="coerce").astype("Int64")
    valid_rows = df["kickoff_utc"].notna() & df["result_target"].isin([0, 1, 2])
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_columns = list(dict.fromkeys(REQUIRED_MODEL_COLUMNS + numeric_columns))
    LOGGER.info(
        "pruning dataframe columns original=%s kept=%s valid_rows=%s",
        len(df.columns),
        len(keep_columns),
        int(valid_rows.sum()),
    )
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "pruning_columns",
                "rows": len(df),
                "valid_rows": int(valid_rows.sum()),
                "original_columns": len(df.columns),
                "kept_columns": len(keep_columns),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    df = df.loc[valid_rows, keep_columns].copy()
    LOGGER.info("sorting loaded dataframe rows=%s columns=%s", len(df), len(df.columns))
    df = df.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    LOGGER.info("loaded rows=%s elapsed_seconds=%.1f", len(df), time.monotonic() - started)
    if output_dir is not None:
        write_json(
            output_dir / "load_status.json",
            {
                "stage": "loaded",
                "rows": len(df),
                "columns": len(df.columns),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
        )
    return df


def require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is missing. In Colab run:\n"
            "  !pip install tensorflow optuna joblib s3fs pyarrow scikit-learn\n"
        ) from exc
    return tf, Sequential, Input, Dense, Dropout, LeakyReLU, l2, Adam, EarlyStopping


def build_model(tf_pack, input_dim: int, params: dict[str, Any]):
    _, Sequential, Input, Dense, Dropout, LeakyReLU, l2, Adam, _ = tf_pack
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(params["n_units_1"], kernel_regularizer=l2(params["l2_reg"])),
            LeakyReLU(negative_slope=0.01),
            Dropout(params["dropout_1"]),
            Dense(params["n_units_2"], kernel_regularizer=l2(params["l2_reg"])),
            LeakyReLU(negative_slope=0.01),
            Dropout(params["dropout_2"]),
            Dense(params["n_units_3"], kernel_regularizer=l2(params["l2_reg"])),
            LeakyReLU(negative_slope=0.01),
            Dropout(params["dropout_3"]),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="categorical_crossentropy")
    return model


def multiclass_brier(y_true_onehot: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean(np.sum((p_pred - y_true_onehot) ** 2, axis=1)))


def class_brier_scores(y_true_cls: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for cls, label in TARGET_LABELS.items():
        out[f"brier_{label}"] = float(brier_score_loss((y_true_cls == cls).astype(int), p_pred[:, cls]))
    return out


def evaluate_probs(y_true_onehot: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)
    p_pred = p_pred / p_pred.sum(axis=1, keepdims=True)
    y_true_cls = np.argmax(y_true_onehot, axis=1)
    y_pred_cls = np.argmax(p_pred, axis=1)
    metrics = {
        "log_loss": float(log_loss(y_true_cls, p_pred, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "multiclass_brier": multiclass_brier(y_true_onehot, p_pred),
        "mean_prob_home": float(p_pred[:, 0].mean()),
        "mean_prob_draw": float(p_pred[:, 1].mean()),
        "mean_prob_away": float(p_pred[:, 2].mean()),
        "actual_home_rate": float((y_true_cls == 0).mean()),
        "actual_draw_rate": float((y_true_cls == 1).mean()),
        "actual_away_rate": float((y_true_cls == 2).mean()),
    }
    metrics.update(class_brier_scores(y_true_cls, p_pred))
    return metrics


def dataset_profile(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": len(feature_cols),
        "season_counts": df["season"].astype(str).value_counts().sort_index().to_dict(),
        "competition_counts": df["competition_id"].astype(str).value_counts().sort_index().to_dict(),
        "target_counts": df["result_target"].astype(str).value_counts().sort_index().to_dict(),
        "train_date_min": train_df["kickoff_utc"].min(),
        "train_date_max": train_df["kickoff_utc"].max(),
        "test_date_min": test_df["kickoff_utc"].min(),
        "test_date_max": test_df["kickoff_utc"].max(),
    }


def text_slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ranked_label(row: pd.Series) -> str:
    scores = {
        "home": row["prob_home"],
        "draw": row["prob_draw"],
        "away": row["prob_away"],
    }
    return max(scores, key=scores.get)


def export_next_games_predictions(predictions: pd.DataFrame, output_dir: Path, count: int) -> dict[str, Any] | None:
    if count <= 0:
        return None
    if predictions.empty:
        return None
    out = predictions.copy()
    out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], errors="coerce", utc=True)
    out = out.dropna(subset=["kickoff_utc"]).sort_values(["kickoff_utc", "match_id"]).head(count).copy()
    out["prediction"] = out.apply(ranked_label, axis=1)
    if "predicted_label" not in out.columns and "predicted_target" in out.columns:
        out["predicted_label"] = out["predicted_target"].map(TARGET_LABELS)
    home_names = out["home_team_name"] if "home_team_name" in out.columns else out["home_team_id"]
    away_names = out["away_team_name"] if "away_team_name" in out.columns else out["away_team_id"]
    out["matchup_number"] = range(1, len(out) + 1)
    out["matchup_key"] = [
        f"{idx}_{text_slug(home)}_{text_slug(away)}"
        for idx, home, away in zip(out["matchup_number"], home_names, away_names)
    ]

    display_cols = [
        "matchup_number",
        "matchup_key",
        "kickoff_utc",
        "competition_id",
        "season",
        "match_id",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "prob_home",
        "prob_draw",
        "prob_away",
        "prediction",
        "predicted_label",
        "result_target",
    ]
    display_cols = [column for column in display_cols if column in out.columns]
    out = out[display_cols]

    csv_path = output_dir / f"next_{count}_games_predictions.csv"
    json_path = output_dir / f"next_{count}_games_summary.json"
    out.to_csv(csv_path, index=False)
    summary = {
        "rows": len(out),
        "source": "held_out_test_predictions",
        "output_csv": str(csv_path),
        "kickoff_min": None if out.empty else str(out["kickoff_utc"].min()),
        "kickoff_max": None if out.empty else str(out["kickoff_utc"].max()),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return {"csv": str(csv_path), "summary": str(json_path), "rows": len(out)}


def resolve_fold_settings(train_rows: int, args: argparse.Namespace) -> tuple[int, int, int, dict[str, Any]]:
    requested = {
        "n_folds": args.n_folds,
        "val_size": args.val_size,
        "min_train_size": args.min_train_size,
    }
    if args.no_auto_fold_size or train_rows > args.min_train_size + args.val_size:
        return args.n_folds, args.val_size, args.min_train_size, {"requested": requested, "adjusted": False}

    if train_rows < 240:
        raise ValueError(
            f"Not enough rows for walk-forward folds even with auto sizing: rows={train_rows}. "
            "Use more seasons or exclude this league."
        )

    val_size = min(args.val_size, max(50, train_rows // 5))
    min_train_size = min(args.min_train_size, max(120, train_rows - (2 * val_size)))
    if train_rows - val_size <= min_train_size:
        min_train_size = max(120, train_rows - val_size - 1)
    if train_rows - val_size <= min_train_size:
        raise ValueError(
            f"Not enough rows for walk-forward folds after auto sizing: rows={train_rows}, "
            f"min_train_size={min_train_size}, val_size={val_size}"
        )
    n_folds = min(args.n_folds, max(1, train_rows - min_train_size - val_size + 1))
    metadata = {
        "requested": requested,
        "adjusted": True,
        "reason": "train_rows did not exceed requested min_train_size + val_size",
    }
    return n_folds, val_size, min_train_size, metadata


def save_trial_progress(output_dir: Path):
    def callback(study: optuna.Study, trial: optuna.Trial) -> None:
        trials_path = output_dir / "optuna_trials.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)
        payload = {
            "completed_trials": len([item for item in study.trials if item.state.is_finished()]),
            "last_trial": trial.number,
            "last_trial_state": str(trial.state),
        }
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        if best_trial is not None:
            payload.update(
                {
                    "best_trial": best_trial.number,
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                }
            )
        write_json(output_dir / "best_trial_so_far.json", payload)
        LOGGER.info("saved trial progress to %s", trials_path)

    return callback


def trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_units_1": trial.suggest_categorical("n_units_1", [64, 128, 256, 384]),
        "n_units_2": trial.suggest_categorical("n_units_2", [32, 64, 128, 192]),
        "n_units_3": trial.suggest_categorical("n_units_3", [16, 32, 64, 96]),
        "dropout_1": trial.suggest_float("dropout_1", 0.05, 0.35),
        "dropout_2": trial.suggest_float("dropout_2", 0.05, 0.35),
        "dropout_3": trial.suggest_float("dropout_3", 0.00, 0.25),
        "l2_reg": trial.suggest_float("l2_reg", 1e-7, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def objective_factory(df: pd.DataFrame, folds, feature_cols: list[str], tf_pack, args: argparse.Namespace):
    tf, *_, EarlyStopping = tf_pack

    def objective(trial: optuna.Trial) -> float:
        LOGGER.info("trial %s started", trial.number)
        tf.keras.backend.clear_session()
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
        params = trial_params(trial)
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            fold = prepare_fold(train_df, val_df, feature_cols)
            model = build_model(tf_pack, fold.x_train.shape[1], params)
            early = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
            LOGGER.info(
                "trial %s fold %s/%s train_rows=%s val_rows=%s batch_size=%s",
                trial.number,
                fold_idx + 1,
                len(folds),
                len(train_df),
                len(val_df),
                params["batch_size"],
            )
            model.fit(
                fold.x_train,
                fold.y_train,
                validation_data=(fold.x_val, fold.y_val),
                epochs=args.epochs,
                batch_size=params["batch_size"],
                callbacks=[early],
                verbose=0,
            )
            probs = model.predict(fold.x_val, verbose=0)
            metrics = evaluate_probs(fold.y_val, probs)
            LOGGER.info(
                "trial %s fold %s metrics log_loss=%.5f accuracy=%.4f brier=%.5f",
                trial.number,
                fold_idx + 1,
                metrics["log_loss"],
                metrics["accuracy"],
                metrics["multiclass_brier"],
            )
            trial.set_user_attr(f"fold_{fold_idx}", metrics)
            scores.append(-metrics["log_loss"])
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def train_final_model(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], params, tf_pack, args):
    _, *_, EarlyStopping = tf_pack
    LOGGER.info("final training started train_rows=%s test_rows=%s features=%s", len(train_df), len(test_df), len(feature_cols))
    fold = prepare_fold(train_df, test_df, feature_cols)
    model = build_model(tf_pack, fold.x_train.shape[1], params)
    early = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    model.fit(
        fold.x_train,
        fold.y_train,
        validation_data=(fold.x_val, fold.y_val),
        epochs=args.epochs,
        batch_size=params["batch_size"],
        callbacks=[early],
        verbose=1,
    )
    probs = model.predict(fold.x_val, verbose=0)
    metrics = evaluate_probs(fold.y_val, probs)
    LOGGER.info(
        "final test metrics log_loss=%.5f accuracy=%.4f brier=%.5f",
        metrics["log_loss"],
        metrics["accuracy"],
        metrics["multiclass_brier"],
    )
    prediction_columns = [
        "match_id",
        "competition_id",
        "season",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "result_target",
    ]
    prediction_frame = test_df[[column for column in prediction_columns if column in test_df.columns]].copy()
    prediction_frame["prob_home"] = probs[:, 0]
    prediction_frame["prob_draw"] = probs[:, 1]
    prediction_frame["prob_away"] = probs[:, 2]
    prediction_frame["predicted_target"] = np.argmax(probs, axis=1)
    prediction_frame["predicted_label"] = prediction_frame["predicted_target"].map(TARGET_LABELS)
    return model, fold, metrics, prediction_frame


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.n_trials = min(args.n_trials, 3)
    args.epochs = min(args.epochs, 10)
    args.max_features = min(args.max_features, 100)
    args.n_folds = min(args.n_folds, 2)
    args.val_size = min(args.val_size, 80)
    args.min_train_size = min(args.min_train_size, 160)
    if args.league == "all" and not args.competitions and not args.seasons:
        args.league = "eng1"
        args.seasons = "2021,2022,2023,2024,2025"


def run_training(args: argparse.Namespace, competitions: list[str], seasons: list[str], output_dir: Path, league_label: str | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = add_file_logging(output_dir)
    available_scope: dict[str, list[str]] = {}
    if is_global_default_scope(args):
        competitions, available_scope = discover_full_scope_competitions(args)
        if not competitions:
            raise SystemExit("No full-scope competitions found for the requested global model settings.")
        if not seasons:
            seasons = required_scope_seasons(args.full_scope_start_season, args.test_season)
            args.seasons = ",".join(seasons)
        LOGGER.info("defaulting global model to full-scope regular leagues competitions=%s seasons=%s", competitions, seasons)
    else:
        competitions = resolve_competitions(args)
    if competitions and not seasons:
        args.seasons = default_partition_seasons(args.train_through_season, args.test_season)
        seasons = csv_list(args.seasons)
        LOGGER.info("defaulting seasons for league run to %s", seasons)
    if competitions and not seasons:
        raise SystemExit("--competitions requires --seasons so the script can target partition files.")

    LOGGER.info(
        "run config league=%s competitions=%s seasons=%s train_through=%s test=%s trials=%s epochs=%s smoke=%s",
        args.league,
        competitions or ["all"],
        seasons or ["all"],
        args.train_through_season,
        args.test_season,
        args.n_trials,
        args.epochs,
        args.smoke,
    )
    write_json(
        output_dir / "run_config.json",
        {
            "input": args.input,
            "output_dir": str(output_dir),
            "league": league_label or args.league,
            "competitions": competitions or ["all"],
            "seasons": seasons or ["all"],
            "available_competition_seasons": available_scope,
            "train_through_season": args.train_through_season,
            "test_season": args.test_season,
            "max_features": args.max_features,
            "n_folds": args.n_folds,
            "val_size": args.val_size,
            "min_train_size": args.min_train_size,
            "no_auto_fold_size": args.no_auto_fold_size,
            "epochs": args.epochs,
            "n_trials": args.n_trials,
            "patience": args.patience,
            "seed": args.seed,
            "smoke": args.smoke,
            "next_games": args.next_games,
            "duckdb_preprocess": args.duckdb_preprocess,
            "log_file": str(log_path),
        },
    )
    tf_pack = require_tensorflow()
    tf_pack[0].random.set_seed(args.seed)
    np.random.seed(args.seed)

    load_engine = "duckdb" if args.duckdb_preprocess else "pandas"
    df = (
        load_gold_dataset_duckdb(args.input, competitions, seasons, output_dir)
        if args.duckdb_preprocess
        else load_gold_dataset(args.input, competitions, seasons, output_dir)
    )
    write_json(output_dir / "load_complete.json", {"rows": len(df), "columns": len(df.columns)})
    LOGGER.info("building train/test split")
    train_all = df[df["season"].astype(str) <= str(args.train_through_season)].copy()
    test = df[df["season"].astype(str) == str(args.test_season)].copy()
    if train_all.empty or test.empty:
        raise RuntimeError(f"Bad train/test split: train_rows={len(train_all)} test_rows={len(test)}")
    write_json(
        output_dir / "split_profile.json",
        {
            "train_rows": len(train_all),
            "test_rows": len(test),
            "train_through_season": args.train_through_season,
            "test_season": args.test_season,
        },
    )

    LOGGER.info("selecting numeric feature columns max_features=%s total_columns=%s", args.max_features, len(train_all.columns))
    feature_started = time.monotonic()
    feature_cols = choose_numeric_feature_columns(train_all, args.max_features)
    LOGGER.info("feature selection complete features=%s elapsed_seconds=%.1f", len(feature_cols), time.monotonic() - feature_started)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns selected.")
    write_json(
        output_dir / "feature_selection_status.json",
        {
            "selected_features": len(feature_cols),
            "max_features": args.max_features,
            "elapsed_seconds": round(time.monotonic() - feature_started, 1),
        },
    )
    LOGGER.info("building walk-forward folds")
    n_folds, val_size, min_train_size, fold_settings = resolve_fold_settings(len(train_all), args)
    if fold_settings["adjusted"]:
        LOGGER.info(
            "auto-adjusted fold settings train_rows=%s n_folds=%s val_size=%s min_train_size=%s requested=%s",
            len(train_all),
            n_folds,
            val_size,
            min_train_size,
            fold_settings["requested"],
        )
    folds = make_walkforward_folds(train_all, n_folds, val_size, min_train_size)
    LOGGER.info("selected features=%s folds=%s", len(feature_cols), len(folds))
    print(json.dumps({"rows": len(df), "train_rows": len(train_all), "test_rows": len(test), "features": len(feature_cols)}, indent=2))
    features_path = output_dir / "feature_columns.json"
    features_path.write_text(json.dumps(feature_cols, indent=2) + "\n")
    write_json(output_dir / "dataset_profile.json", dataset_profile(df, train_all, test, feature_cols))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(2, min(5, args.n_trials // 4)), n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    LOGGER.info("optuna optimization started trials=%s folds=%s", args.n_trials, len(folds))
    study.optimize(
        objective_factory(train_all, folds, feature_cols, tf_pack, args),
        n_trials=args.n_trials,
        callbacks=[save_trial_progress(output_dir)],
    )
    LOGGER.info("optuna optimization finished best_value=%.5f best_params=%s", study.best_value, study.best_params)

    model, fold, test_metrics, predictions = train_final_model(
        df=df,
        train_df=train_all,
        test_df=test,
        feature_cols=feature_cols,
        params=study.best_params,
        tf_pack=tf_pack,
        args=args,
    )

    model_path = output_dir / "soccer_three_way_nn.keras"
    preprocessing_path = output_dir / "preprocessing.joblib"
    predictions_path = output_dir / "test_predictions.parquet"
    trials_path = output_dir / "optuna_trials.csv"
    summary_path = output_dir / "summary.json"

    model.save(model_path)
    joblib.dump({"imputer": fold.imputer, "scaler": fold.scaler, "feature_names": fold.feature_names}, preprocessing_path)
    predictions.to_parquet(predictions_path, index=False)
    study.trials_dataframe().to_csv(trials_path, index=False)
    features_path.write_text(json.dumps(fold.feature_names, indent=2) + "\n")
    next_games_artifact = export_next_games_predictions(predictions, output_dir, args.next_games)
    config = TrainingConfig(
        input=args.input,
        output_dir=str(output_dir),
        competitions=competitions,
        seasons=seasons,
        train_through_season=str(args.train_through_season),
        test_season=str(args.test_season),
        max_features=args.max_features,
        n_folds=args.n_folds,
        val_size=args.val_size,
        min_train_size=args.min_train_size,
        epochs=args.epochs,
        n_trials=args.n_trials,
        batch_size_default=0,
        seed=args.seed,
    )
    summary = {
        "config": asdict(config),
        "load_engine": load_engine,
        "fold_settings": {
            **fold_settings,
            "effective": {
                "n_folds": n_folds,
                "val_size": val_size,
                "min_train_size": min_train_size,
            },
        },
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_attrs": study.best_trial.user_attrs,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "preprocessing": str(preprocessing_path),
            "predictions": str(predictions_path),
            "trials": str(trials_path),
            "features": str(features_path),
            "next_games": next_games_artifact,
        },
        "candidate_features": len(feature_cols),
        "trained_features": len(fold.feature_names),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def run_each_league(args: argparse.Namespace) -> int:
    base_output_dir = Path(args.output_dir)
    explicit_competitions = resolve_competitions(args)
    if explicit_competitions:
        competitions = explicit_competitions
        available: dict[str, list[str]] = {}
    else:
        competitions, available = discover_full_scope_competitions(args)
    if not competitions:
        raise SystemExit("No full-scope competitions found for the requested settings.")

    seasons = csv_list(args.seasons)
    if not seasons:
        seasons = required_scope_seasons(args.full_scope_start_season, args.test_season)
        args.seasons = ",".join(seasons)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        base_output_dir / "batch_plan.json",
        {
            "input": args.input,
            "competitions": competitions,
            "seasons": seasons,
            "available_competition_seasons": available,
            "output_layout": "output_dir/<league_slug>_soccer_nn",
        },
    )
    LOGGER.info("batch league training plan competitions=%s seasons=%s output_dir=%s", competitions, seasons, base_output_dir)
    summaries = []
    failures = []
    for idx, competition in enumerate(competitions):
        league_args = argparse.Namespace(**vars(args))
        league_args.league = competition
        league_args.competitions = competition
        league_args.seasons = ",".join(seasons)
        output_dir = base_output_dir / f"{league_slug(competition)}_soccer_nn"
        LOGGER.info("batch league started competition=%s output_dir=%s", competition, output_dir)
        required_artifacts = [
            output_dir / "soccer_three_way_nn.keras",
            output_dir / "preprocessing.joblib",
            output_dir / "test_predictions.parquet",
            output_dir / "summary.json",
        ]
        if args.skip_existing and all(path.exists() for path in required_artifacts):
            LOGGER.info("batch league skipped existing artifacts competition=%s output_dir=%s", competition, output_dir)
            summaries.append({"competition": competition, "output_dir": str(output_dir), "skipped": True})
            write_json(
                base_output_dir / "batch_summary.json",
                {
                    "completed": summaries,
                    "failures": failures,
                    "remaining": competitions[idx + 1 :],
                },
            )
            continue
        try:
            summary = run_training(league_args, [competition], seasons, output_dir, league_label=league_slug(competition))
            summaries.append(
                {
                    "competition": competition,
                    "output_dir": str(output_dir),
                    "test_metrics": summary["test_metrics"],
                    "fold_settings": summary["fold_settings"],
                    "artifacts": summary["artifacts"],
                    "next_games": summary["artifacts"].get("next_games"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep long batch training moving across small/problem leagues.
            LOGGER.exception("batch league failed competition=%s output_dir=%s", competition, output_dir)
            failures.append({"competition": competition, "output_dir": str(output_dir), "error": str(exc)})
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "failed_summary.json", failures[-1])
        write_json(
            base_output_dir / "batch_summary.json",
            {
                "completed": summaries,
                "failures": failures,
                "remaining": competitions[idx + 1 :],
            },
        )
    write_json(base_output_dir / "batch_summary.json", {"completed": summaries, "failures": failures, "remaining": []})
    return 0


def main() -> int:
    setup_logging()
    args = parse_args()
    apply_smoke_overrides(args)
    if args.train_each_league:
        return run_each_league(args)

    competitions = resolve_competitions(args)
    seasons = csv_list(args.seasons)
    run_training(args, competitions, seasons, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
