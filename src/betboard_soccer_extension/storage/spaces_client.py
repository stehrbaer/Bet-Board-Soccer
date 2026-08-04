from __future__ import annotations

from pathlib import Path
from typing import Iterable

import boto3

from betboard_soccer_extension.storage.config import SpacesConfig


class SpacesClient:
    def __init__(self, config: SpacesConfig | None = None):
        self.config = config or SpacesConfig.from_env()
        session_kwargs = {}
        if self.config.has_credentials:
            session_kwargs["aws_access_key_id"] = self.config.access_key_id
            session_kwargs["aws_secret_access_key"] = self.config.secret_access_key
            session = boto3.session.Session()
        elif self.config.profile_name:
            session = boto3.session.Session(profile_name=self.config.profile_name)
        else:
            session = boto3.session.Session()
        self._client = session.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            region_name=self.config.region_name,
            **session_kwargs,
        )

    def upload_file(self, local_path: Path, key: str, content_type: str | None = None) -> None:
        extra_args = {"ContentType": content_type} if content_type else None
        kwargs = {"ExtraArgs": extra_args} if extra_args else {}
        self._client.upload_file(str(local_path), self.config.bucket, key, **kwargs)

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.config.bucket, key, str(local_path))

    def list_keys(self, prefix: str) -> Iterable[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield str(item["Key"])

    def list_objects(self, prefix: str) -> Iterable[dict]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            yield from page.get("Contents", [])

    def object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.config.bucket, Key=key)
            return True
        except Exception:
            return False
