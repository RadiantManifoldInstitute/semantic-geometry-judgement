from __future__ import annotations

import collections
import hashlib
import time
from typing import Any

import numpy as np

from .canonical import canonical_bytes
from .representations import derive_pair


def _clock_ns() -> int:
    return time.clock_gettime_ns(getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC))


def _timed(call):
    start = _clock_ns(); value = call(); return value, _clock_ns() - start


def _groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["blind_action_id"]].append(index)
    if len(grouped) != 5760 or any(len(indices) not in {8, 32, 128} for indices in grouped.values()):
        raise RuntimeError("LATENCY_ACTION_OR_POLICY_CENSUS_MISMATCH")
    keys = sorted(grouped)
    selected = np.random.default_rng(17021).choice(len(keys), size=24, replace=False)
    return [grouped[keys[int(index)]] for index in selected]


def measure(
    slot: str,
    encoder: Any,
    blinded: list[dict[str, Any]],
    reference_set: tuple[list[str], np.ndarray],
    *,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure one action against its complete variable-size policy set without labels."""
    reference_texts, reference_vectors = reference_set
    reference_index = {text: index for index, text in enumerate(reference_texts)}
    for ordinal in range(10):
        encoder.encode([f"nonclaim warmup {ordinal:02d}"], role="action", batch_size=1)
        encoder.encode([f"nonclaim warmup {ordinal:02d}"], role="reference", batch_size=1)
    records = []
    groups = _groups(blinded)
    for cache_state in ("HOT", "COLD"):
        for repetition in range(repetitions):
            for order, indices in enumerate(groups):
                start = _clock_ns()
                _, t0 = _timed(lambda: canonical_bytes({"action": blinded[indices[0]]["blind_action_id"], "policies": [blinded[index]["blind_policy_id"] for index in indices]}))
                actions, t1 = _timed(lambda: encoder.encode([blinded[indices[0]]["action_text"]], role="action", batch_size=1))
                fields = ("policy_text", "aligned_reference_text", "opposed_reference_text")
                if cache_state == "HOT":
                    references, t2 = _timed(lambda: np.stack([reference_vectors[reference_index[blinded[index][field]]] for index in indices for field in fields]))
                    reference_stage = "REFERENCE_FETCH_HOT"
                else:
                    references, t2 = _timed(lambda: encoder.encode([blinded[index][field] for index in indices for field in fields], role="reference", batch_size=32))
                    reference_stage = "REFERENCE_ENCODE_COLD"
                geometries, t3 = _timed(lambda: [derive_pair(actions[0], references[offset], references[offset + 1], references[offset + 2]) for offset in range(0, len(references), 3)])
                output, t4 = _timed(lambda: canonical_bytes({"policy_geometries": geometries}))
                total = _clock_ns() - start
                components = {"INPUT": t0, "ACTION_ENCODE": t1, reference_stage: t2, "POLICY_GEOMETRIES": t3, "SERIALIZE": t4}
                records.append({
                    "schema_version": "K_F1F_LATENCY_RECORD_V0_1", "encoder_slot": slot, "cache_state": cache_state, "repetition": repetition, "order": order,
                    "blind_transaction_id": hashlib.sha256(blinded[indices[0]]["blind_action_id"].encode()).hexdigest(), "policies": len(indices),
                    "components_ns": components, "independent_total_ns": total, "residual_ns": total - sum(components.values()), "serialized_bytes": len(output), "answer_key_opened": False,
                })
    summary: dict[str, Any] = {"records": len(records), "warmup_transactions": 10, "transactions_per_cache_state_per_repetition": 24}
    for cache_state in ("HOT", "COLD"):
        values = np.asarray([row["independent_total_ns"] / 1_000_000 for row in records if row["cache_state"] == cache_state])
        summary[cache_state] = {"transactions": len(values), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)), "maximum_ms": float(np.max(values))}
    summary["hot_workable"] = summary["HOT"]["p95_ms"] <= 250 and summary["HOT"]["p99_ms"] <= 500
    summary["hot_kill"] = summary["HOT"]["p95_ms"] > 500 or summary["HOT"]["p99_ms"] > 1000
    return summary, records
