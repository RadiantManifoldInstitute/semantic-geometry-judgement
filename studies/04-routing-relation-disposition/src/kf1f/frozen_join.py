from __future__ import annotations
import pathlib
from typing import Any
from .canonical import read_jsonl, sha256_bytes, canonical_bytes

import numpy as np
COUNTS = {"blinded_pairs": 322560, "actions": 5760, "calibration_pairs": 129024, "validation_pairs": 64512, "sealed_pairs": 129024, "encoders": 2, "relation_arms": 7}

def _verify_sealed(row: dict[str, Any]) -> None:
    expected = row.get("payload_sha256")
    payload = {key: value for key, value in row.items() if key != "payload_sha256"}
    if not isinstance(expected, str) or sha256_bytes(canonical_bytes(payload)) != expected:
        raise RuntimeError("SEALED_RECORD_SELF_HASH_MISMATCH")

def _join_split(blinded: list[dict[str, Any]], records: pathlib.Path, representations: list[dict[str, float]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, float]], np.ndarray]:
    expected = {"calibration": COUNTS["calibration_pairs"], "validation": COUNTS["validation_pairs"], "sealed": COUNTS["sealed_pairs"]}[split]
    joins = read_jsonl(records / f"{split}-join.jsonl"); gold = read_jsonl(records / f"{split}-gold.jsonl")
    if len(joins) != expected or len(gold) != expected:
        raise RuntimeError(f"{split.upper()}_JOIN_OR_GOLD_CENSUS_MISMATCH")
    for row in joins + gold:
        _verify_sealed(row)
    gold_by_row = {row["row_id"]: row for row in gold}; join_by_blind = {row["blind_pair_id"]: row for row in joins}
    rows = []; derived = []; indices = []
    for index, blind in enumerate(blinded):
        join = join_by_blind.get(blind["blind_pair_id"])
        if not join:
            continue
        answer = gold_by_row.get(join["row_id"])
        if not answer or answer["payload_sha256"] != join["gold_payload_sha256"]:
            raise RuntimeError(f"{split.upper()}_GOLD_POINTER_MISMATCH")
        rows.append(blind | {key: value for key, value in answer.items() if key != "payload_sha256"})
        derived.append(representations[index]); indices.append(index)
    if len(rows) != expected:
        raise RuntimeError(f"{split.upper()}_JOINED_CENSUS_MISMATCH")
    return rows, derived, np.asarray(indices, dtype=np.int64)
