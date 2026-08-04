from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath


VALID_ZONES = {"raw", "bronze", "silver", "gold", "manifests", "knowledge_graph"}


def _clean_part(value: str) -> str:
    cleaned = str(value).strip().strip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise ValueError(f"Invalid path component: {value!r}")
    return cleaned


@dataclass(frozen=True)
class ObjectPathBuilder:
    root_prefix: str = "soccer-prediction-data"

    def key(self, zone: str, *parts: str) -> str:
        if zone not in VALID_ZONES:
            raise ValueError(f"Unknown storage zone: {zone}")
        components = [_clean_part(self.root_prefix), zone]
        components.extend(_clean_part(part) for part in parts if str(part).strip())
        return str(PurePosixPath(*components))

    def raw_key(
        self,
        source: str,
        endpoint: str,
        extracted_at: datetime,
        filename: str,
        **partitions: str,
    ) -> str:
        if extracted_at.tzinfo is None:
            raise ValueError("extracted_at must be timezone-aware")
        extracted_at = extracted_at.astimezone(timezone.utc)
        date_part = extracted_at.strftime("extracted_date=%Y-%m-%d")
        time_part = extracted_at.strftime("%Y%m%dT%H%M%SZ")
        partition_parts = [f"{_clean_part(k)}={_clean_part(v)}" for k, v in sorted(partitions.items())]
        return self.key("raw", source, f"endpoint={endpoint}", *partition_parts, date_part, f"{time_part}_{filename}")

    def gold_prematch_model_input_key(self, competition_id: str, season: str, filename: str = "part-000.parquet") -> str:
        return self.key(
            "gold",
            "prematch_model_input",
            f"competition={competition_id}",
            f"season={season}",
            filename,
        )

    def manifest_key(self, manifest_type: str, run_id: str, filename: str = "manifest.json") -> str:
        return self.key("manifests", manifest_type, f"run_id={run_id}", filename)

    def legacy_historical_training_prefix(self) -> str:
        return "soccer/training_data/historical"

