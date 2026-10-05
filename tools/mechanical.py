#!/usr/bin/env python3
"""Offline integrity and saved-field checks, not scientific recomputation."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

PRIVATE_PATH = re.compile(r"(?i)(^|/)(private[^/]*|.*ledger.*|\.git|\.aws|\.env[^/]*|credentials|READINESS\.json|PREPARATION\.md)(/|$)")
# Escape detection literals so the scanner source is not itself a coordinate.
PRIVATE_BYTES = re.compile(rb"\x2fUsers/|\x2fhome/|arn\x3aaws:|s3\x3a//|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?i:aws_secret_access_key|aws_session_token)\s*[=:]\s*['\"]?\S+|(?i:authorization:\s*bearer)\s+\S+")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def load(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = child
        return value
    def reject(value):
        raise ValueError("uninspected JSON numeric representation: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_float=reject,
                      parse_constant=reject)


def member_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("invalid member name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("nonrelative member")
    if ":" in name or PRIVATE_PATH.search(name):
        raise ValueError("private or nonportable member: " + name)
    return name


def nonsymlink(path):
    path = Path(path).absolute()
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError("symlink rejected: " + str(part))
    return path


def safe(root, name):
    return nonsymlink(nonsymlink(root) / member_name(name))


def pinned(path, sha, size=None):
    path = nonsymlink(path)
    if not path.is_file():
        raise ValueError("missing/nonregular file: " + str(path))
    raw = path.read_bytes()
    if digest(raw) != sha or (size is not None and len(raw) != size):
        raise ValueError("hash/size drift: " + str(path))
    return raw


def boundary(name, raw):
    member_name(name)
    if PRIVATE_BYTES.search(raw):
        raise ValueError("private coordinate/credential marker: " + name)


def records(manifest):
    rows = manifest["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty/nonlist manifest")
    found = {}
    for row in rows:
        name = member_name(row["path"])
        if name in found or name == "MANIFEST.json":
            raise ValueError("duplicate/self manifest member")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("invalid length")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("invalid SHA-256")
        found[name] = row
    names = set(found)
    if any(str(parent) in names for name in names for parent in PurePosixPath(name).parents):
        raise ValueError("file/ancestor collision")
    return found


def pointer(value, reference):
    if not isinstance(reference, str) or not reference.startswith("/"):
        raise ValueError("explicit JSON pointer required")
    for token in reference[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise ValueError("invalid pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError("invalid array pointer")
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise ValueError("pointer descends into scalar")
    return value


def fields(value, spec):
    # Only the observed frozen schemas' type/required/const/enum/property subset.
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int}
    if "type" in spec and type(value) is not types[spec["type"]]:
        raise ValueError("selected schema type mismatch")
    if "const" in spec and (type(value) is not type(spec["const"]) or value != spec["const"]):
        raise ValueError("selected schema constant mismatch")
    if "enum" in spec and value not in spec["enum"]:
        raise ValueError("selected schema enum mismatch")
    if isinstance(value, dict):
        if not set(spec.get("required", [])) <= set(value):
            raise ValueError("required result field missing")
        properties = spec.get("properties", {})
        if spec.get("additionalProperties") is False and not set(value) <= set(properties):
            raise ValueError("unexpected selected-schema field")
        for key, child in properties.items():
            if key in value:
                fields(value[key], child)


def verify(root, exact_tree=True):
    root = nonsymlink(root)
    manifest_raw = safe(root, "MANIFEST.json").read_bytes()
    boundary("MANIFEST.json", manifest_raw)
    manifest = load(manifest_raw)
    if manifest.get("schema") != "companion-release-manifest-v0.1":
        raise ValueError("unsupported release manifest")
    rows = records(manifest)
    required = {"README.md", "LICENSE", "NOTICE", "CITATION.cff", "REPRODUCIBILITY.json",
                "CLAIM-FIELD-MAP.json", "tools/mechanical.py", "tools/archive.py", "THIRD-PARTY-REFERENCES.md"}
    if not required <= set(rows):
        raise ValueError("required release surface missing")
    for overlay_name in ("B4-MANIFEST.json", "B4-PROTOCOL-MANIFEST.json"):
        if overlay_name not in rows:
            continue
        overlay = load(safe(root, overlay_name).read_bytes())
        for item in overlay["files"]:
            name = member_name(item["path"])
            if name not in rows or item["bytes"] != rows[name]["bytes"] or item["sha256"] != rows[name]["sha256"]:
                raise ValueError("root B4 manifest/member binding mismatch")
    for name, row in rows.items():
        boundary(name, pinned(safe(root, name), row["sha256"], row["bytes"]))
    if exact_tree:
        actual = set()
        for path in root.rglob("*"):
            nonsymlink(path)
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise ValueError("nonregular candidate member")
        if actual != set(rows) | {"MANIFEST.json"}:
            raise ValueError("unlisted or missing candidate files")
    claim_map = load(safe(root, "CLAIM-FIELD-MAP.json").read_bytes())
    if claim_map.get("schema") != "saved-result-claim-field-map-v0.1":
        raise ValueError("claim-map schema mismatch")
    directories = dict(zip("CDEF", ("01-scalar-and-norms", "02-component-measures",
                                   "03-two-stage-policy-checks", "04-routing-relation-disposition")))
    expected_results = {f"results/{study}/scientific-result.json" for study in directories}
    claims = claim_map["claims"]
    if not claims or {c["result_file"] for c in claims} != expected_results:
        raise ValueError("claim result coverage mismatch")
    checked = 0
    for study, directory in directories.items():
        name = f"results/{study}/scientific-result.json"
        body = load(safe(root, name).read_bytes())
        schema = load(safe(root, f"studies/{directory}/schemas/scientific-result.schema.json").read_bytes())
        fields(body, schema)
        if body["study"] != "SYSTEM-K-K-F1" + study or body["replica"] != "replica-a" or body["scientific_result"] is not True:
            raise ValueError("saved result identity mismatch")
        for claim in claims:
            if claim["result_file"] != name:
                continue
            if claim["study"] != body["study"] or claim["status"] != "SAVED_FIELD_TRACE_ONLY":
                raise ValueError("claim identity/status mismatch")
            value = pointer(body, claim["json_pointer"])
            if digest(encoded(value)) != claim["value_sha256"]:
                raise ValueError("claim value hash mismatch")
            stored = claim["stored_value"]
            if stored is not None and (type(stored) is not type(value) or stored != value):
                raise ValueError("stored claim scalar mismatch")
            checked += 1
    return {"status": "MECHANICAL_CHECK_ONLY_NOT_REPLICATION", "files": len(rows) + 1,
            "bytes": sum(r["bytes"] for r in rows.values()) + len(manifest_raw),
            "manifest_sha256": digest(manifest_raw), "claim_fields_checked": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    print(json.dumps(verify(parser.parse_args().candidate), sort_keys=True))
