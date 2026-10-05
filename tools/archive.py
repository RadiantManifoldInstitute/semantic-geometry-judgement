#!/usr/bin/env python3
"""Deterministic exact-manifest archive; never publishes or overwrites."""
import argparse
import gzip
import io
import json
from pathlib import Path
import tarfile

from mechanical import boundary, digest, load, nonsymlink, pinned, records, safe, verify


def archive_bytes(root):
    verified = verify(root)
    manifest_raw = safe(root, "MANIFEST.json").read_bytes()
    boundary("MANIFEST.json", manifest_raw)
    if digest(manifest_raw) != verified["manifest_sha256"]:
        raise ValueError("manifest changed after verification")
    rows = records(load(manifest_raw))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted(set(rows) | {"MANIFEST.json"}):
                row = rows.get(name)
                raw = manifest_raw if row is None else pinned(safe(root, name), row["sha256"], row["bytes"])
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(raw), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def write_archive(root, output):
    root, output = nonsymlink(root).resolve(), nonsymlink(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("archive must be outside candidate")
    raw = archive_bytes(root)
    if output.exists():
        if not output.is_file() or output.read_bytes() != raw:
            raise ValueError("archive conflict")
        created = False
    else:
        with output.open("xb") as stream:
            stream.write(raw)
        created = True
    return {"archive_sha256": digest(raw), "bytes": len(raw), "created": created,
            "status": "CANDIDATE_ARCHIVE_NOT_RELEASE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_archive(args.candidate, args.output), sort_keys=True))
