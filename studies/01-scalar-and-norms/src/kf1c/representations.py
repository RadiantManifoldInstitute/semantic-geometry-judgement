from __future__ import annotations

import math
from typing import Any

import numpy as np


S0_FIELDS = ("normalized_RDVI_scalar",)
M1_FIELDS = ("normalized_RDVI_scalar", "action_l2_norm", "reference_l2_norm")
F1_FIELDS = (
    "normalized_RDVI_scalar",
    "action_l2_norm",
    "reference_l2_norm",
    "reconstructed_cosine",
    "raw_dot_product",
    "raw_squared_euclidean",
    "norm_product",
    "raw_directional_dot_divergence",
)


def derive(action: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if action.ndim != 1 or reference.ndim != 1 or action.shape != reference.shape or action.size == 0:
        raise ValueError("VECTOR_DIMENSION_INVALID")
    x = action.astype(np.float64, copy=False)
    y = reference.astype(np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("VECTOR_NONFINITE")
    action_norm = float(np.linalg.norm(x))
    reference_norm = float(np.linalg.norm(y))
    if action_norm == 0.0 or reference_norm == 0.0:
        raise ValueError("VECTOR_ZERO_NORM")
    direct_dot = float(np.dot(x, y))
    direct_cosine = direct_dot / (action_norm * reference_norm)
    direct_cosine = max(-1.0, min(1.0, direct_cosine))
    scalar = (1.0 - direct_cosine) / 2.0
    reconstructed_cosine = 1.0 - 2.0 * scalar
    reconstructed_dot = action_norm * reference_norm * reconstructed_cosine
    direct_squared_euclidean = float(np.sum((x - y) ** 2))
    reconstructed_squared_euclidean = (
        action_norm * action_norm
        + reference_norm * reference_norm
        - 2.0 * reconstructed_dot
    )
    norm_product = action_norm * reference_norm
    unit_dot_divergence = 2.0 * scalar
    unit_squared_euclidean = 4.0 * scalar
    normalized_cosine_deviation = 2.0 * scalar
    values = {
        "normalized_RDVI_scalar": scalar,
        "action_l2_norm": action_norm,
        "reference_l2_norm": reference_norm,
        "direct_cosine": direct_cosine,
        "reconstructed_cosine": reconstructed_cosine,
        "direct_raw_dot_product": direct_dot,
        "raw_dot_product": reconstructed_dot,
        "direct_raw_squared_euclidean": direct_squared_euclidean,
        "raw_squared_euclidean": reconstructed_squared_euclidean,
        "norm_product": norm_product,
        "raw_directional_dot_divergence": 1.0 - reconstructed_dot,
        "unit_dot_divergence": unit_dot_divergence,
        "unit_squared_euclidean": unit_squared_euclidean,
        "normalized_cosine_deviation": normalized_cosine_deviation,
        "cosine_identity_absolute_residual": abs(direct_cosine - reconstructed_cosine),
        "raw_dot_absolute_residual": abs(direct_dot - reconstructed_dot),
        "raw_squared_euclidean_absolute_residual": abs(direct_squared_euclidean - reconstructed_squared_euclidean),
        "normalized_unit_dot_residual": abs(unit_dot_divergence - 2.0 * scalar),
        "normalized_unit_squared_euclidean_residual": abs(unit_squared_euclidean - 4.0 * scalar),
        "normalized_cosine_deviation_residual": abs(normalized_cosine_deviation - 2.0 * scalar),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("REPRESENTATION_NONFINITE")
    raw_dot_tolerance = 1.0e-8 + 1.0e-6 * abs(direct_dot)
    raw_euclidean_tolerance = 1.0e-8 + 1.0e-6 * abs(direct_squared_euclidean)
    if values["cosine_identity_absolute_residual"] > 1.0e-12:
        raise ValueError("COSINE_IDENTITY_FAILURE")
    if values["raw_dot_absolute_residual"] > raw_dot_tolerance:
        raise ValueError("RAW_DOT_RECONSTRUCTION_FAILURE")
    if values["raw_squared_euclidean_absolute_residual"] > raw_euclidean_tolerance:
        raise ValueError("RAW_EUCLIDEAN_RECONSTRUCTION_FAILURE")
    if max(
        values["normalized_unit_dot_residual"],
        values["normalized_unit_squared_euclidean_residual"],
        values["normalized_cosine_deviation_residual"],
    ) > 1.0e-12:
        raise ValueError("NORMALIZED_IDENTITY_FAILURE")
    return values


def matrix(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> np.ndarray:
    result = np.asarray([[row[field] for field in fields] for row in rows], dtype=np.float64)
    if result.ndim != 2 or result.shape != (len(rows), len(fields)) or not np.isfinite(result).all():
        raise ValueError("REPRESENTATION_MATRIX_INVALID")
    return result
