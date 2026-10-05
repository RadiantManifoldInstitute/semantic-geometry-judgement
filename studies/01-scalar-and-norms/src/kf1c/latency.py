from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from typing import Any

import numpy as np

from .canonical import canonical_bytes
from .representations import M1_FIELDS, derive, matrix


def clock_ns() -> int:
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    return time.clock_gettime_ns(clock)


def characterize_clock() -> dict[str, Any]:
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    name = "CLOCK_MONOTONIC_RAW" if hasattr(time, "CLOCK_MONOTONIC_RAW") else "CLOCK_MONOTONIC"
    values = [time.clock_gettime_ns(clock) for _ in range(10000)]
    deltas = np.diff(np.asarray(values, dtype=np.int64))
    if np.any(deltas < 0):
        raise RuntimeError("LATENCY_CLOCK_NONMONOTONIC")
    positive = deltas[deltas > 0]
    if not len(positive):
        raise RuntimeError("LATENCY_CLOCK_NO_POSITIVE_DELTA")
    resolution = int(round(time.clock_getres(clock) * 1_000_000_000))
    return {
        "clock_identity": name,
        "reported_resolution_ns": resolution,
        "minimum_positive_delta_ns": int(np.min(positive)),
        "median_positive_delta_ns": int(np.median(positive)),
        "zero_delta_count": int(np.sum(deltas == 0)),
        "effective_resolution_R_ns": max(resolution, int(np.min(positive))),
        "calls": 10000,
    }


def _timed(call):
    start = clock_ns()
    value = call()
    return value, clock_ns() - start


def _transaction_groups(blinded: list[dict[str, Any]]) -> list[list[int]]:
    # Frozen materializer order is scenario -> level -> Q slot -> surface.  The
    # holdout is the final eight families.  This derives no answer-key field.
    if len(blinded) != 2160:
        raise RuntimeError("LATENCY_BLINDED_CENSUS_MISMATCH")
    groups = []
    rows_per_family = 108
    rows_per_scenario = 36
    for family in range(12, 20):
        for scenario in range(3):
            origin = family * rows_per_family + scenario * rows_per_scenario
            for slot in range(4):
                groups.append([origin + level * 12 + slot * 3 for level in range(3)])
    if len(groups) != 96 or len({i for group in groups for i in group}) != 288:
        raise RuntimeError("LATENCY_TRANSACTION_CENSUS_MISMATCH")
    return groups


def measure(
    slot: str,
    encoder: Any,
    blinded: list[dict[str, Any]],
    action_vectors: np.ndarray,
    reference_vectors: np.ndarray,
    m1_reader: Any,
    *,
    repetitions: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clock = characterize_clock()
    for ordinal in range(20):
        text = f"nonclaim warmup transaction {ordinal:02d}"
        encoder.encode([text], role="action", batch_size=1)
        encoder.encode([text], role="reference", batch_size=1)

    groups = _transaction_groups(blinded)
    order = np.random.default_rng(41383001).permutation(len(groups))
    records: list[dict[str, Any]] = []
    for cache_state in ("HOT", "COLD"):
        for repetition in range(repetitions):
            for order_ordinal, group_ordinal in enumerate(order):
                indices = groups[int(group_ordinal)]
                transaction_id = hashlib.sha256(
                    "|".join(blinded[i]["blind_pair_id"] for i in indices).encode()
                ).hexdigest()
                outer_start = clock_ns()
                (_, t00) = _timed(
                    lambda: canonical_bytes(
                        {
                            "transaction_id": transaction_id,
                            "rows": [
                                {
                                    "blind_pair_id": blinded[i]["blind_pair_id"],
                                    "action_text_sha256": blinded[i]["action_text_sha256"],
                                    "reference_text_sha256": blinded[i]["reference_text_sha256"],
                                }
                                for i in indices
                            ],
                        }
                    )
                )
                actions, t01 = _timed(
                    lambda: encoder.encode([blinded[i]["action_text"] for i in indices], role="action", batch_size=1)
                )
                if cache_state == "HOT":
                    references, t_ref = _timed(lambda: reference_vectors[indices].copy())
                    reference_field = "T02_REFERENCE_FETCH_HOT"
                else:
                    references, t_ref = _timed(
                        lambda: encoder.encode([blinded[i]["reference_text"] for i in indices], role="reference", batch_size=1)
                    )
                    reference_field = "T03_REFERENCE_ENCODE_COLD"
                representations, t04 = _timed(
                    lambda: [derive(actions[j], references[j]) for j in range(3)]
                )
                probabilities, t05 = _timed(lambda: m1_reader.probabilities(matrix(representations, M1_FIELDS)))
                interpretations, t06 = _timed(
                    lambda: {
                        "predictions": [int(v) for v in np.argmax(probabilities, axis=1)],
                        "cross_level_disagreement": len(set(int(v) for v in np.argmax(probabilities, axis=1))) > 1,
                    }
                )
                serialized, t07 = _timed(lambda: canonical_bytes(interpretations))
                outer_finish = clock_ns()
                components = {
                    "T00_INPUT_VALIDATE_SERIALIZE": t00,
                    "T01_ACTION_ENCODE": t01,
                    reference_field: t_ref,
                    "T04_VECTOR_COMPARISON": t04,
                    "T05_PROFILE_READOUT": t05,
                    "T06_CROSS_LEVEL_INTERPRETATION": t06,
                    "T07_OUTPUT_SERIALIZE": t07,
                }
                total = outer_finish - outer_start
                component_sum = sum(components.values())
                record = {
                    "schema_version": "K_F1C_LATENCY_RECORD_V0_1",
                    "encoder_slot": slot,
                    "cache_state": cache_state,
                    "repetition": repetition,
                    "measured_order": order_ordinal,
                    "blind_transaction_id": transaction_id,
                    "blind_surface_ids": [blinded[i]["blind_surface_id"] for i in indices],
                    "family_ids": [blinded[i]["family_id"] for i in indices],
                    "mission_levels": [blinded[i]["level"] for i in indices],
                    "gold_cells": [blinded[i]["gold_cell"] for i in indices],
                    "components_ns": components,
                    "independent_total_ns": total,
                    "component_sum_ns": component_sum,
                    "residual_ns": total - component_sum,
                    "residual_fraction": (total - component_sum) / total,
                    "serialized_bytes": len(serialized),
                    "warmup": False,
                    "timeout": False,
                    "nonfinite": False,
                    "negative_timer": any(v < 0 for v in [total, *components.values()]),
                }
                if record["negative_timer"]:
                    raise RuntimeError("NEGATIVE_LATENCY_TIMER")
                if record["residual_ns"] < -3 * clock["effective_resolution_R_ns"]:
                    raise RuntimeError("LATENCY_RESIDUAL_BELOW_NEGATIVE_3R")
                records.append(record)

    summary: dict[str, Any] = {"clock": clock, "warmup_transactions": 20, "records": len(records)}
    for cache_state in ("HOT", "COLD"):
        subset = [row for row in records if row["cache_state"] == cache_state]
        totals = np.asarray([row["independent_total_ns"] / 1_000_000 for row in subset])
        semantic = np.asarray(
            [
                (
                    row["components_ns"]["T04_VECTOR_COMPARISON"]
                    + row["components_ns"]["T05_PROFILE_READOUT"]
                    + row["components_ns"]["T06_CROSS_LEVEL_INTERPRETATION"]
                )
                / 1_000_000
                for row in subset
            ]
        )
        summary[cache_state] = {
            "transactions": len(subset),
            "T_TOTAL_ms": {
                "p50": float(np.percentile(totals, 50)),
                "p95": float(np.percentile(totals, 95)),
                "p99": float(np.percentile(totals, 99)),
                "maximum": float(np.max(totals)),
            },
            "T04_plus_T05_plus_T06_ms": {
                "p50": float(np.percentile(semantic, 50)),
                "p95": float(np.percentile(semantic, 95)),
                "p99": float(np.percentile(semantic, 99)),
                "maximum": float(np.max(semantic)),
            },
            "by_family": {
                family: {
                    "transactions": len(family_rows),
                    "p50_ms": float(np.percentile([row["independent_total_ns"] / 1_000_000 for row in family_rows], 50)),
                    "p95_ms": float(np.percentile([row["independent_total_ns"] / 1_000_000 for row in family_rows], 95)),
                    "p99_ms": float(np.percentile([row["independent_total_ns"] / 1_000_000 for row in family_rows], 99)),
                    "maximum_ms": float(np.max([row["independent_total_ns"] / 1_000_000 for row in family_rows])),
                }
                for family in sorted({row["family_ids"][0] for row in subset})
                for family_rows in [[row for row in subset if row["family_ids"][0] == family]]
            },
        }
        residuals = np.asarray([row["residual_ns"] for row in subset], dtype=np.float64)
        residual_fractions = np.asarray([row["residual_fraction"] for row in subset], dtype=np.float64)
        summary[cache_state]["decomposition"] = {
            "median_residual_fraction": float(np.median(residual_fractions)),
            "p95_residual_ns": float(np.percentile(residuals, 95)),
            "complete": bool(
                np.median(residual_fractions) <= 0.10
                and np.percentile(residuals, 95) <= max(5_000_000.0, 0.20 * np.percentile(totals, 95) * 1_000_000)
            ),
        }
    summary["hot_workable"] = bool(
        summary["HOT"]["T_TOTAL_ms"]["p95"] <= 250
        and summary["HOT"]["T_TOTAL_ms"]["p99"] <= 500
        and summary["HOT"]["T04_plus_T05_plus_T06_ms"]["p95"] <= 25
    )
    summary["hot_kill"] = bool(
        summary["HOT"]["T_TOTAL_ms"]["p95"] > 500
        or summary["HOT"]["T_TOTAL_ms"]["p99"] > 1000
    )
    return summary, records
