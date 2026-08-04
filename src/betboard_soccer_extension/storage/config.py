from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SpacesConfig:
    bucket: str
    endpoint_url: str
    region_name: str
    root_prefix: str
    access_key_id: str | None = None
    secret_access_key: str | None = None
    profile_name: str | None = None

    @classmethod
    def from_env(cls) -> "SpacesConfig":
        return cls(
            bucket=os.getenv("DO_SPACES_BUCKET", "betboard-ml-artifacts"),
            endpoint_url=os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com"),
            region_name=os.getenv("DO_SPACES_REGION", "fra1"),
            root_prefix=os.getenv("DO_SPACES_ROOT_PREFIX", "soccer-prediction-data").strip("/"),
            access_key_id=os.getenv("DO_SPACES_KEY"),
            secret_access_key=os.getenv("DO_SPACES_SECRET"),
            profile_name=os.getenv("DO_SPACES_PROFILE") or os.getenv("AWS_PROFILE"),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key)
