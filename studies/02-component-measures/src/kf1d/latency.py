from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from .canonical import canonical_bytes
from .representations import derive


def _clock_ns() -> int:
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return time.clock_gettime_ns(clock)


def _timed(call):
    start = _clock_ns()
    value = call()
    return value, _clock_ns() - start


def _groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    if len(rows) != 2808:
        raise RuntimeError("LATENCY_BLINDED_CENSUS_MISMATCH")
    order = np.random.default_rng(41384011).permutation(len(rows))[:288]
    return [list(map(int, order[start : start + 3])) for start in range(0, 288, 3)]


def measure(
    slot: str,
    encoder: Any,
    blinded: list[dict[str, Any]],
    reference_vectors: np.ndarray,
    *,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure the direct three-geometry path without opening labels."""
    for ordinal in range(20):
        encoder.encode([f"nonclaim warmup {ordinal:02d}"], role="action", batch_size=1)
        encoder.encode([f"nonclaim warmup {ordinal:02d}"], role="reference", batch_size=1)
    records = []
    for cache_state in ("HOT", "COLD"):
        for repetition in range(repetitions):
            for order, indices in enumerate(_groups(blinded)):
                start = _clock_ns()
                _, t0 = _timed(lambda: canonical_bytes({"rows": [blinded[i]["blind_surface_id"] for i in indices]}))
                actions, t1 = _timed(lambda: encoder.encode([blinded[i]["action_text"] for i in indices], role="action", batch_size=1))
                if cache_state == "HOT":
                    references, t2 = _timed(lambda: reference_vectors[indices].copy())
                    reference_stage = "REFERENCE_FETCH_HOT"
                else:
                    references, t2 = _timed(lambda: encoder.encode([blinded[i]["reference_text"] for i in indices], role="reference", batch_size=1))
                    reference_stage = "REFERENCE_ENCODE_COLD"
                geometries, t3 = _timed(lambda: [derive(actions[j], references[j]) for j in range(3)])
                output, t4 = _timed(lambda: canonical_bytes({"geometries": geometries}))
                total = _clock_ns() - start
                components = {"INPUT": t0, "ACTION_ENCODE": t1, reference_stage: t2, "THREE_GEOMETRIES": t3, "SERIALIZE": t4}
                records.append({
                    "schema_version": "K_F1D_LATENCY_RECORD_V0_1",
                    "encoder_slot": slot,
                    "cache_state": cache_state,
                    "repetition": repetition,
                    "order": order,
                    "blind_transaction_id": hashlib.sha256("|".join(blinded[i]["blind_surface_id"] for i in indices).encode()).hexdigest(),
                    "components_ns": components,
                    "independent_total_ns": total,
                    "residual_ns": total - sum(components.values()),
                    "serialized_bytes": len(output),
                    "answer_key_opened": False,
                })
    summary: dict[str, Any] = {"records": len(records), "warmup_transactions": 20}
    for cache_state in ("HOT", "COLD"):
        values = np.asarray([row["independent_total_ns"] / 1_000_000 for row in records if row["cache_state"] == cache_state])
        summary[cache_state] = {
            "transactions": len(values),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "maximum_ms": float(np.max(values)),
        }
    summary["hot_workable"] = summary["HOT"]["p95_ms"] <= 250 and summary["HOT"]["p99_ms"] <= 500
    summary["hot_kill"] = summary["HOT"]["p95_ms"] > 500 or summary["HOT"]["p99_ms"] > 1000
    return summary, records
