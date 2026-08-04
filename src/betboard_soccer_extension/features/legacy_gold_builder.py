from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from betboard_soccer_extension.storage.checksums import sha256_file
from betboard_soccer_extension.storage.manifests import BuildManifest, ObjectManifestEntry
from betboard_soccer_extension.storage.object_paths import ObjectPathBuilder
from betboard_soccer_extension.storage.spaces_client import SpacesClient


LEGACY_RESULT_TO_GEMINI_TARGET = {
    2: 0,  # home win
    1: 1,  # draw
    0: 2,  # away win
}


@dataclass(frozen=True)
class GoldBuildResult:
    run_id: str
    partitions: int
    rows: int
    objects: list[ObjectManifestEntry]
    manifest_key: str


def _parse_training_key(key: str) -> tuple[str, str] | None:
    parts = key.split("/")
    if len(parts) != 6:
        return None
    if parts[:3] != ["soccer", "training_data", "historical"]:
        return None
    if parts[5] != "training.parquet":
        return None
    return parts[3], parts[4]


def _normalize_league_id(value: str) -> str:
    return value.replace("_", ".")


def _with_contract_columns(df: pd.DataFrame, league_slug: str, season: str, schema_version: str) -> pd.DataFrame:
    result = df.copy()
    if "event_date" in result.columns:
        result["kickoff_utc"] = pd.to_datetime(result["event_date"], errors="coerce", utc=True)
    else:
        result["kickoff_utc"] = pd.NaT

    competition_id = _normalize_league_id(league_slug)
    result["match_id"] = result.get("event_id", pd.Series(index=result.index, dtype="string")).astype("string")
    result["competition_id"] = result.get("league_id", competition_id).fillna(competition_id).astype("string")
    result["season"] = result.get("season_year", season).fillna(season).astype("string")

    result["home_team_id"] = result.get("home_team_id", pd.Series(index=result.index, dtype="string")).astype("string")
    result["away_team_id"] = result.get("away_team_id", pd.Series(index=result.index, dtype="string")).astype("string")
    result["home_goals"] = pd.to_numeric(result.get("home_score"), errors="coerce").astype("Int64")
    result["away_goals"] = pd.to_numeric(result.get("away_score"), errors="coerce").astype("Int64")

    legacy_target = pd.to_numeric(result.get("result_3way"), errors="coerce")
    result["result_target"] = legacy_target.map(LEGACY_RESULT_TO_GEMINI_TARGET).astype("Int64")

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result["feature_cutoff_utc"] = result["kickoff_utc"]
    result["feature_build_timestamp"] = now
    result["feature_schema_version"] = schema_version
    result["team_data_completeness"] = "legacy_training"
    result["player_data_completeness"] = "not_available"
    result["lineup_data_status"] = "not_available"
    result["weather_data_status"] = "not_available"

    contract_cols = [
        "match_id",
        "competition_id",
        "season",
        "kickoff_utc",
        "home_team_id",
        "away_team_id",
        "home_goals",
        "away_goals",
        "result_target",
        "feature_cutoff_utc",
        "feature_build_timestamp",
        "feature_schema_version",
        "team_data_completeness",
        "player_data_completeness",
        "lineup_data_status",
        "weather_data_status",
    ]
    remaining = [col for col in result.columns if col not in contract_cols]
    return result[contract_cols + remaining]


def build_gold_from_legacy_historical(
    client: SpacesClient,
    path_builder: ObjectPathBuilder,
    run_id: str,
    schema_version: str = "soccer_gold_v0.1",
    leagues: set[str] | None = None,
    seasons: set[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
    skip_existing: bool = True,
) -> GoldBuildResult:
    legacy_prefix = path_builder.legacy_historical_training_prefix()
    training_keys: list[tuple[str, str, str]] = []
    for key in client.list_keys(legacy_prefix):
        parsed = _parse_training_key(key)
        if not parsed:
            continue
        league_slug, season = parsed
        if leagues and league_slug not in leagues and _normalize_league_id(league_slug) not in leagues:
            continue
        if seasons and season not in seasons:
            continue
        training_keys.append((league_slug, season, key))

    training_keys.sort()
    if limit is not None:
        training_keys = training_keys[:limit]

    existing_gold = set()
    if skip_existing:
        for key in client.list_keys(path_builder.key("gold", "prematch_model_input")):
            if key.endswith(".parquet"):
                existing_gold.add(key)

    jobs: list[tuple[str, str, str, str]] = []
    for league_slug, season, source_key in training_keys:
        competition_id = _normalize_league_id(league_slug)
        out_key = path_builder.gold_prematch_model_input_key(competition_id, season)
        if skip_existing and out_key in existing_gold:
            continue
        jobs.append((league_slug, season, source_key, out_key))

    def process_one(job: tuple[str, str, str, str]) -> int:
        league_slug, season, source_key, out_key = job
        worker_client = SpacesClient(client.config)
        with tempfile.TemporaryDirectory(prefix="betboard_gold_legacy_part_") as tmp:
            tmpdir = Path(tmp)
            local_in = tmpdir / league_slug / season / "source_training.parquet"
            local_out = tmpdir / league_slug / season / "part-000.parquet"
            worker_client.download_file(source_key, local_in)
            df = pd.read_parquet(local_in)
            gold = _with_contract_columns(df, league_slug, season, schema_version)
            local_out.parent.mkdir(parents=True, exist_ok=True)
            gold.to_parquet(local_out, index=False)
            worker_client.upload_file(local_out, out_key, content_type="application/octet-stream")
            return len(gold)

    rows = 0
    if jobs:
        max_workers = max(1, workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, job) for job in jobs]
            for future in as_completed(futures):
                rows += future.result()

    objects: list[ObjectManifestEntry] = []
    gold_prefix = path_builder.key("gold", "prematch_model_input")
    for item in client.list_objects(gold_prefix):
        key = str(item["Key"])
        if key.endswith(".parquet"):
            objects.append(
                ObjectManifestEntry(
                    key=key,
                    size_bytes=int(item.get("Size", 0)),
                    sha256=None,
                    content_type="application/octet-stream",
                )
            )
    objects.sort(key=lambda item: item.key)

    manifest = BuildManifest.create(
        run_id=run_id,
        manifest_type="feature_builds",
        source=legacy_prefix,
        schema_version=schema_version,
        objects=objects,
        attrs={
            "builder": "legacy_historical_training_to_gold",
            "target_mapping": {"home_win": 0, "draw": 1, "away_win": 2},
            "legacy_result_3way_mapping": {"2": "home_win", "1": "draw", "0": "away_win"},
            "partitions": len(objects),
            "rows_processed_this_run": rows,
            "objects_processed_this_run": len(jobs),
            "skip_existing": skip_existing,
        },
    )
    with tempfile.TemporaryDirectory(prefix="betboard_gold_manifest_") as tmp:
        manifest_path = Path(tmp) / "manifest.json"
        manifest.write(manifest_path)
        manifest_key = path_builder.manifest_key("feature_builds", run_id)
        client.upload_file(manifest_path, manifest_key, content_type="application/json")

    return GoldBuildResult(
        run_id=run_id,
        partitions=len(objects),
        rows=rows,
        objects=objects,
        manifest_key=manifest_key,
    )
