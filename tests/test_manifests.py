from __future__ import annotations

import json

from betboard_soccer_extension.storage.manifests import BuildManifest, ObjectManifestEntry


def test_manifest_serializes_objects() -> None:
    manifest = BuildManifest.create(
        run_id="run-1",
        manifest_type="feature_builds",
        source="test",
        schema_version="v1",
        objects=[ObjectManifestEntry(key="gold/a.parquet", size_bytes=10, sha256="abc")],
    )
    payload = json.loads(manifest.to_json())
    assert payload["run_id"] == "run-1"
    assert payload["objects"][0]["key"] == "gold/a.parquet"

