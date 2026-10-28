"""Where datasets, held-out sets and finished adapters live: a local folder, or an S3-compatible bucket.

Keys look like "datasets/<id>/train.jsonl". Customer data is stored here and sent to Tinker; nowhere else.
"""

import shutil
from pathlib import Path

from . import config


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"bad storage key {key!r}")
        return path

    def put(self, key: str, src: Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    def get(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), dest)
        return dest

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def local_path(self, key: str) -> Path | None:
        """For streaming a download straight off disk."""
        return self._path(key)

    def presigned_get(self, key: str, ttl_seconds: int, filename: str) -> str | None:
        return None  # served by our own /v1/downloads route


class S3Storage:
    def __init__(self, s: config.Settings):
        import boto3

        self.bucket = s.s3_bucket
        self.client = boto3.client("s3", endpoint_url=s.s3_endpoint or None, region_name=s.s3_region,
                                   aws_access_key_id=s.s3_access_key_id, aws_secret_access_key=s.s3_secret_access_key)

    def put(self, key: str, src: Path) -> None:
        self.client.upload_file(str(src), self.bucket, key)

    def get(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))
        return dest

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def size(self, key: str) -> int:
        return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])

    def local_path(self, key: str) -> Path | None:
        return None

    def presigned_get(self, key: str, ttl_seconds: int, filename: str) -> str | None:
        return self.client.generate_presigned_url(
            "get_object", ExpiresIn=ttl_seconds,
            Params={"Bucket": self.bucket, "Key": key,
                    "ResponseContentDisposition": f'attachment; filename="{filename}"'})


_store = None


def store():
    global _store
    if _store is None:
        s = config.get()
        _store = S3Storage(s) if s.storage == "s3" else LocalStorage(s.data_dir / "storage")
    return _store


def reset() -> None:
    global _store
    _store = None
