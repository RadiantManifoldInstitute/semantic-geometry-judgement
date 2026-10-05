from __future__ import annotations

import math
from typing import Any

import numpy as np


MEASURES = ("COSINE", "EUCLIDEAN", "DOT")


def derive(action: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Derive the three pre-specified geometric constructs without collapsing them."""
    if action.ndim != 1 or reference.ndim != 1 or action.shape != reference.shape or action.size == 0:
        raise ValueError("VECTOR_DIMENSION_INVALID")
    x = action.astype(np.float64, copy=False)
    y = reference.astype(np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("VECTOR_NONFINITE")
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("VECTOR_ZERO_NORM")

    dot = float(np.dot(x, y))
    cosine = max(-1.0, min(1.0, dot / (x_norm * y_norm)))
    raw_euclidean = float(np.linalg.norm(x - y))
    normalized_euclidean = float(np.linalg.norm(x / x_norm - y / y_norm))
    cosine_equivalent = math.sqrt(max(0.0, 2.0 - 2.0 * cosine))
    identity_residual = abs(normalized_euclidean - cosine_equivalent)
    values = {
        "cosine": cosine,
        "raw_euclidean": raw_euclidean,
        "normalized_euclidean_control": normalized_euclidean,
        "raw_dot_product": dot,
        "action_l2_norm": x_norm,
        "reference_l2_norm": y_norm,
        "norm_product": x_norm * y_norm,
        "normalized_euclidean_cosine_identity_residual": identity_residual,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("REPRESENTATION_NONFINITE")
    if identity_residual > 1.0e-10:
        raise ValueError("NORMALIZED_EUCLIDEAN_COSINE_IDENTITY_FAILURE")
    return values


def measure_value(row: dict[str, Any], measure: str) -> float:
    field = {
        "COSINE": "cosine",
        "EUCLIDEAN": "raw_euclidean",
        "DOT": "raw_dot_product",
    }.get(measure)
    if field is None:
        raise ValueError(f"UNKNOWN_MEASURE:{measure}")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"NONFINITE_MEASURE:{measure}")
    return value
