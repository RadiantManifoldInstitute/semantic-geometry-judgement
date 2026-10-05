from __future__ import annotations

import hashlib
import json
import math
import pathlib
import unicodedata
from typing import Any, Iterable


def normalized(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, tuple):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {normalized(str(key)): normalized(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NONFINITE_CANONICAL_VALUE")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            normalized(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL_ROW_NOT_OBJECT:{path}:{ordinal}")
        rows.append(row)
    return rows


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_bytes(row))


def tree_manifest(root: pathlib.Path) -> list[dict[str, Any]]:
    return tree_manifest_rows(root)


def tree_manifest_rows(root: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(root)).encode()):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def tree_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_bytes(b"".join(canonical_bytes(row) for row in rows))
