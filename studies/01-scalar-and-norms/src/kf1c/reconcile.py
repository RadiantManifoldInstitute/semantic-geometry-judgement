from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

from .canonical import canonical_bytes


TIMING_KEYS = frozenset(
    {
        "latency_summary",
        "E10_LATENCY",
        "latency",
        "latency_consequence",
        "utc",
        "start_utc",
        "end_utc",
        "duration_seconds",
        "resource_usage",
        "job_id",
        "job_name",
        "replica",
        "prejoin_seal",
    }
)

LATENCY_CONSEQUENCE_RANK = {
    "HOT_LATENCY_GATE_PASS": 0,
    "BOUNDED_NONINTERACTIVE_ONLY": 1,
    "INTENDED_FAST_USE_INFEASIBLE_OBSERVED_ENVELOPE": 2,
}


def substantive(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: substantive(child) for key, child in value.items() if key not in TIMING_KEYS}
    if isinstance(value, list):
        return [substantive(child) for child in value]
    return value


def reconcile(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_body = canonical_bytes(substantive(left))
    right_body = canonical_bytes(substantive(right))
    if left_body != right_body:
        raise RuntimeError("REPLICA_SUBSTANTIVE_DISAGREEMENT_NO_RESULT")
    if left.get("complete_census") != right.get("complete_census"):
        raise RuntimeError("REPLICA_CENSUS_DISAGREEMENT_NO_RESULT")
    latency_by_replica = {
        "replica-a": left["terminal_classification"]["latency_consequence"],
        "replica-b": right["terminal_classification"]["latency_consequence"],
    }
    if any(value not in LATENCY_CONSEQUENCE_RANK for value in latency_by_replica.values()):
        raise RuntimeError("UNKNOWN_REPLICA_LATENCY_CONSEQUENCE")
    terminal_classification = dict(left["terminal_classification"])
    terminal_classification["latency_consequence"] = max(
        latency_by_replica.values(), key=LATENCY_CONSEQUENCE_RANK.__getitem__
    )
    terminal_classification["replica_latency_consequences"] = latency_by_replica
    scientific_result = left.get("scientific_result") is True and right.get("scientific_result") is True
    return {
        "schema_version": "K_F1C_REPLICA_RECONCILIATION_V0_1",
        "status": "PASS_EXACT_SUBSTANTIVE_AGREEMENT",
        "scientific_result": scientific_result,
        "substantive_sha256": hashlib.sha256(left_body).hexdigest(),
        "left_result_sha256": hashlib.sha256(canonical_bytes(left)).hexdigest(),
        "right_result_sha256": hashlib.sha256(canonical_bytes(right)).hexdigest(),
        "timing_excluded_from_byte_agreement": sorted(TIMING_KEYS),
        "terminal_classification": terminal_classification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=pathlib.Path)
    parser.add_argument("right", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    receipt = reconcile(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    args.output.write_bytes(canonical_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
