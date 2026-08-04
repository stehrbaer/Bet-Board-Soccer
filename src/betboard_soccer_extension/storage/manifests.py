from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ObjectManifestEntry:
    key: str
    size_bytes: int
    sha256: str | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class BuildManifest:
    run_id: str
    manifest_type: str
    created_at: str
    source: str
    schema_version: str
    objects: list[ObjectManifestEntry]
    attrs: dict[str, Any]

    @classmethod
    def create(
        cls,
        run_id: str,
        manifest_type: str,
        source: str,
        schema_version: str,
        objects: list[ObjectManifestEntry],
        attrs: dict[str, Any] | None = None,
    ) -> "BuildManifest":
        return cls(
            run_id=run_id,
            manifest_type=manifest_type,
            created_at=utc_now_iso(),
            source=source,
            schema_version=schema_version,
            objects=objects,
            attrs=attrs or {},
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")

