"""Copy run folders to and from S3, so a run measured on an AWS instance can be viewed anywhere.

    innards sync up   deltanet-study s3://my-bucket/innards      # before the instance stops
    innards sync down s3://my-bucket/innards/deltanet-study     # on your laptop, then `innards view`

Files are compared by size and MD5 (the ETag of a single-part upload), so repeated syncs only move
what changed. Needs ``boto3`` (``pip install "innards[aws]"``); credentials come from the standard
AWS chain (environment, profile, instance role).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from innards.store import RunStore, valid_name

__all__ = ["SyncResult", "parse_s3_url", "sync_down", "sync_up"]


@dataclass
class SyncResult:
    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{len(self.copied)} copied, {len(self.skipped)} unchanged"


def parse_s3_url(url: str) -> tuple[str, str]:
    """``s3://bucket/some/prefix`` → ``("bucket", "some/prefix")``."""
    if not url.startswith("s3://"):
        raise ValueError(f"Not an S3 URL: {url!r} (expected s3://bucket/prefix)")
    bucket, _, prefix = url[5:].partition("/")
    if not bucket:
        raise ValueError(f"S3 URL needs a bucket: {url!r}")
    return bucket, prefix.strip("/")


def _client(client: Any) -> Any:
    if client is not None:
        return client
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('Syncing to S3 needs boto3: pip install "innards[aws]"') from exc
    return boto3.client("s3")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_etag(s3: Any, bucket: str, key: str) -> tuple[str, int] | None:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # botocore ClientError without importing botocore
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    return str(head.get("ETag", "")).strip('"'), int(head.get("ContentLength", -1))


def sync_up(store: RunStore, run: str, url: str, *, client: Any = None) -> SyncResult:
    """Upload ``run``'s folder to ``url/<run>/`` (only files that changed)."""
    bucket, prefix = parse_s3_url(url)
    s3 = _client(client)
    base = store.run_dir(run)
    if not base.is_dir():
        raise FileNotFoundError(f"No run named {run!r} in {store.root}")
    result = SyncResult()
    for path in store.files(run):
        relative = path.relative_to(base).as_posix()
        key = "/".join(p for p in (prefix, run, relative) if p)
        remote = _remote_etag(s3, bucket, key)
        if remote is not None and remote == (_md5(path), path.stat().st_size):
            result.skipped.append(key)
            continue
        content_type = "application/x-ndjson" if path.suffix == ".jsonl" else "application/json"
        s3.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
        result.copied.append(key)
    return result


def sync_down(url: str, root: str | Path, *, client: Any = None) -> SyncResult:
    """Download runs under ``url`` into ``root``.

    ``url`` names one run (``s3://bucket/prefix/<run>``, it holds ``run.json``) or a prefix of runs.
    """
    bucket, prefix = parse_s3_url(url)
    s3 = _client(client)
    target = Path(root).expanduser()
    listing = f"{prefix}/" if prefix else ""
    objects = [
        obj
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=listing)
        for obj in page.get("Contents", [])
    ]
    is_run = bool(prefix) and any(str(obj["Key"]) == f"{prefix}/run.json" for obj in objects)
    # s3://bucket/prefix/<run> keeps the run's own folder name; a prefix of runs maps onto the root
    base = (prefix.rsplit("/", 1)[0] if "/" in prefix else "") if is_run else prefix
    result = SyncResult()
    for obj in objects:
        key = str(obj["Key"])
        relative = key[len(base) + 1 :] if base else key
        parts = relative.split("/")
        if len(parts) < 2 or not valid_name(parts[0]):
            continue
        if any(part in {"", ".", ".."} or part.startswith(".") for part in parts):
            continue
        dest = target.joinpath(*parts)
        etag = str(obj.get("ETag", "")).strip('"')
        if dest.is_file() and dest.stat().st_size == int(obj.get("Size", -1)) and _md5(dest) == etag:
            result.skipped.append(key)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        result.copied.append(key)
    return result
