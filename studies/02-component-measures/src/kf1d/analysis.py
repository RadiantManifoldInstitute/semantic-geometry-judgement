from __future__ import annotations

import collections
from typing import Any, Iterable

import numpy as np

from .representations import MEASURES, measure_value


ARMS: dict[str, tuple[str, ...]] = {
    "COSINE_ONLY": ("COSINE",),
    "EUCLIDEAN_ONLY": ("EUCLIDEAN",),
    "DOT_ONLY": ("DOT",),
    "COSINE_PLUS_EUCLIDEAN": ("COSINE", "EUCLIDEAN"),
    "COSINE_PLUS_DOT": ("COSINE", "DOT"),
    "EUCLIDEAN_PLUS_DOT": ("EUCLIDEAN", "DOT"),
    "ALL_THREE": ("COSINE", "EUCLIDEAN", "DOT"),
}

PANEL_PRIORITY = {
    "DIRECTION": ("COSINE", "DOT", "EUCLIDEAN"),
    "PROXIMITY": ("EUCLIDEAN", "COSINE", "DOT"),
    "MAGNITUDE": ("DOT", "EUCLIDEAN", "COSINE"),
}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("EMPTY_METRIC_DENOMINATOR")
    return float(np.mean(values))


def _rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("RANK_CORRELATION_INPUT_INVALID")

    def ranks(values: list[float]) -> np.ndarray:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = np.zeros(len(values), dtype=np.float64)
        cursor = 0
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and values[order[end]] == values[order[cursor]]:
                end += 1
            result[order[cursor:end]] = (cursor + 1 + end) / 2.0
            cursor = end
        return result

    a, b = ranks(left), ranks(right)
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _family_summary(values: dict[str, float]) -> dict[str, Any]:
    if len(values) != 8:
        raise ValueError(f"EXACT_EIGHT_SEALED_FAMILIES_REQUIRED:{len(values)}")
    ordered = [values[key] for key in sorted(values)]
    leave_one_out = {
        family: _mean(value for key, value in values.items() if key != family)
        for family in sorted(values)
    }
    random = np.random.default_rng(41384009)
    indices = random.integers(0, len(ordered), size=(10000, len(ordered)))
    bootstrap = np.mean(np.asarray(ordered, dtype=np.float64)[indices], axis=1)
    return {
        "family_values": {key: values[key] for key in sorted(values)},
        "equal_weight_mean": _mean(ordered),
        "median": float(np.median(ordered)),
        "range": [float(np.min(ordered)), float(np.max(ordered))],
        "leave_one_family_out_equal_weight_means": leave_one_out,
        "descriptive_family_bootstrap": {
            "seed": 41384009,
            "replicates": 10000,
            "percentile_95_interval": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
            "hard_gate_use": False,
        },
    }


def _group(rows: list[dict[str, Any]], panel: str) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if row["panel"] == panel:
            grouped[(row["group_id"], row["surface_class"])].append(index)
    return dict(grouped)


def _opposition_prediction(value: float, threshold: float, measure: str) -> bool:
    return value >= threshold if measure == "EUCLIDEAN" else value <= threshold


def _threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        raise ValueError("CALIBRATION_VALUE_CENSUS_ZERO")
    scale = max(1.0, max(abs(value) for value in unique))
    epsilon = np.finfo(np.float64).eps * scale * 8.0
    return [unique[0] - epsilon, *[(left + right) / 2.0 for left, right in zip(unique, unique[1:])], unique[-1] + epsilon]


def calibrate_thresholds(rows: list[dict[str, Any]], representations: list[dict[str, float]]) -> dict[str, Any]:
    """Freeze direction thresholds from calibration rows only."""
    if len(rows) != len(representations) or not rows:
        raise ValueError("CALIBRATION_CENSUS_MISMATCH")
    if {row["split"] for row in rows} != {"CALIBRATION"}:
        raise ValueError("NONCALIBRATION_ROW_AT_THRESHOLD_FREEZE")
    direction = [index for index, row in enumerate(rows) if row["panel"] == "DIRECTION"]
    if len(direction) != 4 * 2 * 3 * 2 * 3:
        raise ValueError(f"CALIBRATION_DIRECTION_CENSUS_MISMATCH:{len(direction)}")
    result: dict[str, Any] = {
        "schema_version": "K_F1D_THRESHOLD_FREEZE_V0_1",
        "source_split": "CALIBRATION_ONLY",
        "false_alarm_budget": 0.10,
        "selection": {},
    }
    for measure in MEASURES:
        values = [measure_value(representations[index], measure) for index in direction]
        candidates = []
        for threshold in _threshold_candidates(values):
            opposed = [index for index in direction if rows[index]["relation_polarity"] == "OPPOSED"]
            aligned = [index for index in direction if rows[index]["relation_polarity"] == "ALIGNED"]
            recall = _mean(_opposition_prediction(measure_value(representations[index], measure), threshold, measure) for index in opposed)
            false_alarm = _mean(_opposition_prediction(measure_value(representations[index], measure), threshold, measure) for index in aligned)
            candidates.append({"threshold": threshold, "opposition_recall": recall, "aligned_false_alarm_rate": false_alarm})
        feasible = [row for row in candidates if row["aligned_false_alarm_rate"] <= 0.10]
        pool = feasible or candidates
        selected = sorted(
            pool,
            key=lambda row: (
                -row["opposition_recall"],
                row["aligned_false_alarm_rate"],
                -row["threshold"] if measure == "EUCLIDEAN" else row["threshold"],
            ),
        )[0]
        result["selection"][measure] = {
            **selected,
            "budget_feasible": bool(feasible),
            "candidate_count": len(candidates),
            "threshold_curve": candidates,
            "orientation": "OPPOSED_IF_AT_OR_ABOVE" if measure == "EUCLIDEAN" else "OPPOSED_IF_AT_OR_BELOW",
        }
    return result


def _direction(rows: list[dict[str, Any]], representations: list[dict[str, float]], measure: str, threshold: float) -> dict[str, Any]:
    family_pairs: dict[str, list[float]] = collections.defaultdict(list)
    opposed_predictions, aligned_predictions, margins = [], [], []
    for indices in _group(rows, "DIRECTION").values():
        if len(indices) != 2 or {rows[i]["relation_polarity"] for i in indices} != {"ALIGNED", "OPPOSED"}:
            raise ValueError("DIRECTION_GROUP_INVALID")
        aligned = next(i for i in indices if rows[i]["relation_polarity"] == "ALIGNED")
        opposed = next(i for i in indices if rows[i]["relation_polarity"] == "OPPOSED")
        aligned_value = measure_value(representations[aligned], measure)
        opposed_value = measure_value(representations[opposed], measure)
        correct = opposed_value > aligned_value if measure == "EUCLIDEAN" else opposed_value < aligned_value
        family_pairs[rows[aligned]["family_id"]].append(float(correct))
        opposed_predictions.append(float(_opposition_prediction(opposed_value, threshold, measure)))
        aligned_predictions.append(float(_opposition_prediction(aligned_value, threshold, measure)))
        margins.append((opposed_value - aligned_value) if measure == "EUCLIDEAN" else (aligned_value - opposed_value))
    family = {key: _mean(value) for key, value in sorted(family_pairs.items())}
    return {
        "pair_ordering_accuracy": _mean(value for values in family_pairs.values() for value in values),
        "opposition_recall": _mean(opposed_predictions),
        "aligned_false_alarm_rate": _mean(aligned_predictions),
        "median_correct_orientation_margin": float(np.median(margins)),
        "family_support": _family_summary(family),
        "groups": sum(map(len, family_pairs.values())),
    }


def _proximity(rows: list[dict[str, Any]], representations: list[dict[str, float]], measure: str) -> dict[str, Any]:
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    wrong_top, margins = [], []
    for indices in _group(rows, "PROXIMITY").values():
        if len(indices) != 3 or {rows[i]["applicability_class"] for i in indices} != {"APPLICABLE", "ADJACENT", "WRONG_MISSION"}:
            raise ValueError("PROXIMITY_GROUP_INVALID")
        values = {rows[i]["applicability_class"]: measure_value(representations[i], measure) for i in indices}
        ordered = sorted(values, key=lambda key: (values[key] if measure == "EUCLIDEAN" else -values[key], key))
        success = float(ordered[0] == "APPLICABLE" and len({values[key] for key in values}) == 3)
        family = rows[indices[0]]["family_id"]
        by_family[family].append(success)
        wrong_top.append(float(ordered[0] == "WRONG_MISSION"))
        alternatives = [values["ADJACENT"], values["WRONG_MISSION"]]
        margins.append(min(alternatives) - values["APPLICABLE"] if measure == "EUCLIDEAN" else values["APPLICABLE"] - max(alternatives))
    family = {key: _mean(value) for key, value in sorted(by_family.items())}
    return {
        "applicability_recall_at_review_budget_1_of_3": _mean(value for values in by_family.values() for value in values),
        "applicable_beats_both_rate": _mean(value for values in by_family.values() for value in values),
        "wrong_mission_top_rate": _mean(wrong_top),
        "median_applicability_margin": float(np.median(margins)),
        "family_support": _family_summary(family),
        "groups": sum(map(len, by_family.values())),
    }


def _strength(value: float, measure: str, polarity: str) -> float:
    if measure == "EUCLIDEAN":
        return -value if polarity == "ALIGNED" else value
    return value if polarity == "ALIGNED" else -value


def _magnitude(rows: list[dict[str, Any]], representations: list[dict[str, float]], measure: str) -> dict[str, Any]:
    family_rho: dict[str, list[float]] = collections.defaultdict(list)
    adjacent, reversals = [], []
    for indices in _group(rows, "MAGNITUDE").values():
        if len(indices) != 4 or {rows[i]["magnitude_level"] for i in indices} != {1, 2, 3, 4}:
            raise ValueError("MAGNITUDE_GROUP_INVALID")
        ordered = sorted(indices, key=lambda index: rows[index]["magnitude_level"])
        polarity = rows[ordered[0]]["relation_polarity"]
        scores = [_strength(measure_value(representations[index], measure), measure, polarity) for index in ordered]
        rho = _rank_correlation([1.0, 2.0, 3.0, 4.0], scores)
        family_rho[rows[ordered[0]]["family_id"]].append(rho)
        adjacent.extend(float(right > left) for left, right in zip(scores, scores[1:]))
        reversals.extend(float(right < left) for left, right in zip(scores, scores[1:]))
    family = {key: _mean(value) for key, value in sorted(family_rho.items())}
    return {
        "mean_spearman_magnitude_ordering": _mean(value for values in family_rho.values() for value in values),
        "adjacent_ordering_accuracy": _mean(adjacent),
        "adjacent_reversal_rate": _mean(reversals),
        "family_support": _family_summary(family),
        "groups": sum(map(len, family_rho.values())),
    }


def _normalized_control(representations: list[dict[str, float]]) -> dict[str, Any]:
    residuals = [row["normalized_euclidean_cosine_identity_residual"] for row in representations]
    return {
        "identity": "NORMALIZED_EUCLIDEAN_EQUALS_SQRT_2_MINUS_2_COSINE",
        "maximum_absolute_residual": max(residuals),
        "pass": max(residuals) <= 1.0e-10,
        "independent_arm": False,
    }


def _control_diagnostics(rows: list[dict[str, Any]], representations: list[dict[str, float]]) -> dict[str, Any]:
    length_differences = []
    for panel in ("DIRECTION", "MAGNITUDE"):
        for indices in _group(rows, panel).values():
            lengths = [rows[index]["action_token_count"] for index in indices]
            length_differences.append(max(lengths) - min(lengths))
    action_norms = np.asarray([row["action_l2_norm"] for row in representations])
    reference_norms = np.asarray([row["reference_l2_norm"] for row in representations])
    failures = []
    if max(length_differences) != 0:
        failures.append("TEXT_LENGTH_MATCH_FAILURE")
    return {
        "text_length_matched": max(length_differences) == 0,
        "maximum_within_group_action_token_difference": max(length_differences),
        "encoder_specific_raw_scale": {
            "action_norm_min_median_max": [float(np.min(action_norms)), float(np.median(action_norms)), float(np.max(action_norms))],
            "reference_norm_min_median_max": [float(np.min(reference_norms)), float(np.median(reference_norms)), float(np.max(reference_norms))],
        },
        "control_failures": failures,
    }


def _panel_result(
    panel: str,
    rows: list[dict[str, Any]],
    representations: list[dict[str, float]],
    measure: str,
    threshold: float,
) -> dict[str, Any]:
    if panel == "DIRECTION":
        return _direction(rows, representations, measure, threshold)
    if panel == "PROXIMITY":
        return _proximity(rows, representations, measure)
    if panel == "MAGNITUDE":
        return _magnitude(rows, representations, measure)
    raise ValueError(f"UNKNOWN_PANEL:{panel}")


def _subset(
    rows: list[dict[str, Any]],
    representations: list[dict[str, float]],
    field: str,
    value: str,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    indices = [index for index, row in enumerate(rows) if row[field] == value]
    return [rows[index] for index in indices], [representations[index] for index in indices]


def _disposition(panel: str, metrics: dict[str, Any]) -> str:
    family_values = list(metrics["family_support"]["family_values"].values())
    if panel == "DIRECTION":
        if metrics["pair_ordering_accuracy"] <= 0.40 or metrics["aligned_false_alarm_rate"] >= 0.50:
            return "MISLEADING"
        if metrics["pair_ordering_accuracy"] >= 0.65 and metrics["opposition_recall"] >= 0.60 and metrics["aligned_false_alarm_rate"] <= 0.15 and sum(value >= 0.60 for value in family_values) >= 6:
            return "WORKABLE"
        positive = metrics["pair_ordering_accuracy"] > 0.50
    elif panel == "PROXIMITY":
        if metrics["applicability_recall_at_review_budget_1_of_3"] <= 0.40:
            return "MISLEADING"
        if metrics["applicability_recall_at_review_budget_1_of_3"] >= 0.65 and metrics["wrong_mission_top_rate"] <= 0.20 and sum(value >= 0.60 for value in family_values) >= 6:
            return "WORKABLE"
        positive = metrics["applicability_recall_at_review_budget_1_of_3"] > (1.0 / 3.0)
    elif panel == "MAGNITUDE":
        if metrics["mean_spearman_magnitude_ordering"] <= -0.25 or metrics["adjacent_reversal_rate"] >= 0.50:
            return "MISLEADING"
        if metrics["mean_spearman_magnitude_ordering"] >= 0.50 and metrics["adjacent_ordering_accuracy"] >= 0.60 and metrics["adjacent_reversal_rate"] <= 0.20 and sum(value >= 0.30 for value in family_values) >= 6:
            return "WORKABLE"
        positive = metrics["mean_spearman_magnitude_ordering"] > 0.0
    else:
        raise ValueError(f"UNKNOWN_PANEL:{panel}")
    return "NARROW" if positive else "NULL"


def _select(arm: tuple[str, ...], panel: str) -> str:
    return next(measure for measure in PANEL_PRIORITY[panel] if measure in arm)


def _pareto(arms: dict[str, Any]) -> dict[str, Any]:
    full = arms["ALL_THREE"]["profile"]
    comparisons = {}
    all_pass = True
    for name, arm in arms.items():
        if name == "ALL_THREE":
            continue
        candidate = arm["profile"]
        deltas = {
            "applicability_recall": full["applicability_recall"] - candidate["applicability_recall"],
            "opposition_recall": full["opposition_recall"] - candidate["opposition_recall"],
            "aligned_false_alarm_improvement": candidate["aligned_false_alarm_rate"] - full["aligned_false_alarm_rate"],
            "magnitude_spearman": full["magnitude_spearman"] - candidate["magnitude_spearman"],
        }
        no_degradation = all(value >= -0.03 for value in deltas.values())
        material_improvement = any(value >= 0.05 for value in deltas.values())
        family_deltas = {
            family: {
                coordinate: full["family_profile"][family][coordinate] - candidate["family_profile"][family][coordinate]
                for coordinate in ("direction_pair_ordering", "applicability_recall", "magnitude_spearman")
            }
            for family in sorted(full["family_profile"])
        }
        no_family_kill = all(value >= -0.10 for row in family_deltas.values() for value in row.values())
        family_support = sum(any(value >= 0.05 for value in row.values()) and all(value >= -0.10 for value in row.values()) for row in family_deltas.values())
        passed = no_degradation and material_improvement and no_family_kill and family_support >= 6
        comparisons[name] = {
            "deltas_full_minus_comparator_or_improvement": deltas,
            "no_material_degradation": no_degradation,
            "material_improvement": material_improvement,
            "family_deltas": family_deltas,
            "no_family_degradation_beyond_0.10": no_family_kill,
            "families_supporting_material_improvement": family_support,
            "dominates": passed,
        }
        all_pass &= passed
    return {
        "rule": "FULL_PROFILE_WINS_ONLY_IF_IT_DOMINATES_EACH_OTHER_ARM_WITH_AT_LEAST_ONE_0.05_IMPROVEMENT_AND_NO_GREATER_THAN_0.03_DEGRADATION",
        "comparisons": comparisons,
        "full_profile_wins": all_pass,
    }


def analyze_encoder(rows: list[dict[str, Any]], representations: list[dict[str, float]], threshold_freeze: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != len(representations) or not rows:
        raise ValueError("SEALED_ANALYSIS_CENSUS_MISMATCH")
    if {row["split"] for row in rows} != {"SEALED"}:
        raise ValueError("NONSEALED_ROW_AT_FINAL_ANALYSIS")
    if threshold_freeze.get("source_split") != "CALIBRATION_ONLY":
        raise ValueError("THRESHOLD_NOT_CALIBRATION_FROZEN")
    direct = {}
    for measure in MEASURES:
        threshold = float(threshold_freeze["selection"][measure]["threshold"])
        panels = {panel: _panel_result(panel, rows, representations, measure, threshold) for panel in PANEL_PRIORITY}
        by_level = {}
        for level in ("L1_CONCRETE", "L2_OPERATIONAL", "L3_PURPOSE"):
            subset_rows, subset_rep = _subset(rows, representations, "level", level)
            by_level[level] = {panel: _panel_result(panel, subset_rows, subset_rep, measure, threshold) for panel in PANEL_PRIORITY}
        by_surface = {}
        for surface in ("CANONICAL", "BOUNDED_PARAPHRASE", "CLAUSE_ORDER"):
            subset_rows, subset_rep = _subset(rows, representations, "surface_class", surface)
            by_surface[surface] = {panel: _panel_result(panel, subset_rows, subset_rep, measure, threshold) for panel in PANEL_PRIORITY}
        direct[measure] = {
            "panels": panels,
            "by_mission_level": by_level,
            "by_surface": by_surface,
            "dispositions": {panel: _disposition(panel, result) for panel, result in panels.items()},
        }
    arms = {}
    for name, included in ARMS.items():
        selected = {panel: _select(included, panel) for panel in PANEL_PRIORITY}
        direction = direct[selected["DIRECTION"]]["panels"]["DIRECTION"]
        proximity = direct[selected["PROXIMITY"]]["panels"]["PROXIMITY"]
        magnitude = direct[selected["MAGNITUDE"]]["panels"]["MAGNITUDE"]
        arms[name] = {
            "included_measures": list(included),
            "panel_measure_selection": selected,
            "profile": {
                "applicability_recall": proximity["applicability_recall_at_review_budget_1_of_3"],
                "opposition_recall": direction["opposition_recall"],
                "aligned_false_alarm_rate": direction["aligned_false_alarm_rate"],
                "magnitude_spearman": magnitude["mean_spearman_magnitude_ordering"],
                "family_profile": {
                    family: {
                        "direction_pair_ordering": direction["family_support"]["family_values"][family],
                        "applicability_recall": proximity["family_support"]["family_values"][family],
                        "magnitude_spearman": magnitude["family_support"]["family_values"][family],
                    }
                    for family in sorted(direction["family_support"]["family_values"])
                },
            },
        }
    return {
        "threshold_freeze": threshold_freeze,
        "direct_measure_results": direct,
        "seven_arm_ablation": arms,
        "full_profile_pareto": _pareto(arms),
        "normalized_euclidean_control": _normalized_control(representations),
        "control_diagnostics": _control_diagnostics(rows, representations),
        "unfavorable_dispositions_permitted": ["NULL", "MISLEADING", "INVALID"],
    }


def terminal_classification(per_encoder: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    if set(per_encoder) != {"E01", "E02", "E03"}:
        raise ValueError("EXACT_THREE_ENCODERS_REQUIRED")

    def combine(values: list[str]) -> str:
        if values.count("MISLEADING") >= 2:
            return "MISLEADING"
        if values.count("WORKABLE") >= 2 and "MISLEADING" not in values:
            return "WORKABLE"
        if "WORKABLE" in values or "NARROW" in values:
            return "NARROW"
        return "NULL"

    component_specs = {
        "COSINE_DIRECTION": ("COSINE", "DIRECTION"),
        "EUCLIDEAN_PROXIMITY": ("EUCLIDEAN", "PROXIMITY"),
        "DOT_MAGNITUDE": ("DOT", "MAGNITUDE"),
    }
    components = {
        name: {"per_encoder": {slot: per_encoder[slot]["direct_measure_results"][measure]["dispositions"][panel] for slot in sorted(per_encoder)}}
        for name, (measure, panel) in component_specs.items()
    }
    for value in components.values():
        value["cross_encoder_disposition"] = combine(list(value["per_encoder"].values()))
    profile = {slot: bool(per_encoder[slot]["full_profile_pareto"]["full_profile_wins"]) for slot in sorted(per_encoder)}
    return {
        "single_aggregate_scientific_verdict": None,
        "components": components,
        "FULL_PROFILE_PARETO": {"per_encoder": profile, "cross_encoder_disposition": "WORKABLE" if sum(profile.values()) >= 2 else "NULL"},
        "latency": latency,
        "claim_boundary": "GEOMETRIC_BEHAVIOR_ON_THE_FROZEN_SYNTHETIC_WORLD_NOT_REAL_WORLD_MISSION_CORRECTNESS",
    }
