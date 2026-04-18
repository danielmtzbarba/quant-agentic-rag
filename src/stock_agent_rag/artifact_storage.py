from __future__ import annotations

from typing import Protocol

from .config import Settings, get_settings
from .schemas import StoredObject


class ObjectStorageProvider(Protocol):
    def put_text(
        self,
        *,
        bucket: str,
        object_key: str,
        content: str,
        content_type: str,
    ) -> StoredObject:
        ...


class LocalArtifactStorage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def put_text(
        self,
        *,
        bucket: str,
        object_key: str,
        content: str,
        content_type: str,
    ) -> StoredObject:
        base_dir = self.settings.thesis_artifact_base_dir / bucket
        target = base_dir / object_key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return StoredObject(
            storage_provider="local",
            bucket=bucket,
            object_key=object_key,
            content_type=content_type,
            markdown_path=str(target),
        )


class S3ArtifactStorage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - depends on runtime install state
            raise RuntimeError(
                "boto3 is required for THESIS_STORAGE_PROVIDER=s3. "
                "Run `uv sync` to install dependencies."
            ) from exc

        session = boto3.session.Session(
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            aws_session_token=self.settings.aws_session_token,
            region_name=self.settings.aws_region,
        )
        client_kwargs: dict[str, object] = {}
        if self.settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = self.settings.s3_endpoint_url
        if self.settings.s3_force_path_style:
            client_kwargs["config"] = Config(s3={"addressing_style": "path"})
        self._client = session.client("s3", **client_kwargs)

    def put_text(
        self,
        *,
        bucket: str,
        object_key: str,
        content: str,
        content_type: str,
    ) -> StoredObject:
        response = self._client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )
        etag = response.get("ETag")
        if isinstance(etag, str):
            etag = etag.strip('"')
        return StoredObject(
            storage_provider="s3",
            bucket=bucket,
            object_key=object_key,
            content_type=content_type,
            etag=etag if isinstance(etag, str) else None,
        )
