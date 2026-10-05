from __future__ import annotations

import math
from typing import Any

import numpy as np


FEATURES = ("cosine", "euclidean_distance", "signed_dot_product")


def derive(action: np.ndarray, policy: np.ndarray) -> dict[str, float]:
    if action.ndim != 1 or policy.ndim != 1 or action.shape != policy.shape or action.size == 0:
        raise ValueError("VECTOR_DIMENSION_INVALID")
    x = action.astype(np.float64, copy=False)
    y = policy.astype(np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("VECTOR_NONFINITE")
    x_norm = float(np.linalg.norm(x)); y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("VECTOR_ZERO_NORM")
    dot = float(np.dot(x, y))
    cosine = max(-1.0, min(1.0, dot / (x_norm * y_norm)))
    euclidean = float(np.linalg.norm(x - y))
    ux, uy = x / x_norm, y / y_norm
    unit_dot = float(np.dot(ux, uy))
    unit_euclidean = float(np.linalg.norm(ux - uy))
    expected_unit_euclidean = math.sqrt(max(0.0, 2.0 - 2.0 * unit_dot))
    collapse_residual = abs(unit_euclidean - expected_unit_euclidean)
    values = {
        "cosine": cosine,
        "euclidean_distance": euclidean,
        "signed_dot_product": dot,
        "action_l2": x_norm,
        "policy_l2": y_norm,
        "unit_cosine": unit_dot,
        "unit_euclidean_distance": unit_euclidean,
        "unit_signed_dot_product": unit_dot,
        "unit_collapse_residual": collapse_residual,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("MEASUREMENT_NONFINITE")
    if collapse_residual > 1.0e-10 or abs(unit_dot - cosine) > 1.0e-10:
        raise ValueError("UNIT_VECTOR_COLLAPSE_CONTROL_FAILURE")
    return values


def derive_pair(
    action: np.ndarray,
    policy: np.ndarray,
    aligned_reference: np.ndarray,
    opposed_reference: np.ndarray,
) -> dict[str, float]:
    values = derive(action, policy)
    aligned = derive(action, aligned_reference)
    opposed = derive(action, opposed_reference)
    values.update({
        "aligned_cosine": aligned["cosine"],
        "opposed_cosine": opposed["cosine"],
        "legacy_direction_margin": aligned["cosine"] - opposed["cosine"],
    })
    return values


def vector(row: dict[str, Any], features: tuple[str, ...]) -> list[float]:
    values = [float(row[name]) for name in features]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("NONFINITE_FEATURE")
    return values
