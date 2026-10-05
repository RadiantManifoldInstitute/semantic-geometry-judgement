from __future__ import annotations

import collections
import math
import re
from typing import Any, Iterable

import numpy as np


ARMS = (
    "SERIAL_EUCLIDEAN_THEN_COSINE",
    "COSINE_ONLY",
    "EUCLIDEAN_ONLY",
    "REVERSED_COSINE_THEN_EUCLIDEAN",
    "CONVENTIONAL_TOP_K_RETRIEVAL",
    "DETERMINISTIC_KEYWORD_OR_RULE_ROUTING_BASELINE",
    "FROZEN_FULL_RDVI_PROFILE",
    "EXHAUSTIVE_POLICY_EVALUATION_ORACLE",
)

TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
SURFACES = (
    "CANONICAL", "BOUNDED_PARAPHRASE", "EXPLICIT_NEGATION", "TOPICAL_OPPOSITION",
    "WRONG_MISSION_DECOY", "LEXICALLY_SIMILAR_NONGOVERNING", "UNUSUAL_MISSION_SERVING",
    "CLAUSE_ORDER", "HARMLESS_WORDING",
)
CARDINALITIES = ("ZERO", "ONE", "MULTIPLE_COMPATIBLE", "MULTIPLE_CONFLICTING")
ABSTRACTIONS = ("L1_CONCRETE", "L2_OPERATIONAL", "L3_PURPOSE")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else 0.0


def _groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["blind_action_id"]].append(index)
    result = []
    for key in sorted(grouped):
        indices = grouped[key]
        if len(indices) != 8:
            raise ValueError(f"EXACT_EIGHT_POLICIES_PER_ACTION_REQUIRED:{key}:{len(indices)}")
        if len({rows[index]["expected_disposition"] for index in indices}) != 1:
            raise ValueError("ACTION_ORACLE_DISPOSITION_DISAGREEMENT")
        result.append(indices)
    return result


def _direction(rep: dict[str, float], margin: float) -> str:
    value = float(rep["direction_margin"])
    if value > margin:
        return "ALIGNED"
    if value < -margin:
        return "OPPOSED"
    return "AMBIGUOUS"


def _semantic_disposition(relations: list[str]) -> str:
    if not relations:
        return "ACT"
    observed = set(relations)
    if "AMBIGUOUS" in observed or {"ALIGNED", "OPPOSED"}.issubset(observed):
        return "ESCALATE"
    if observed == {"OPPOSED"}:
        return "HOLD"
    return "ACT"


def _oracle_disposition(rows: list[dict[str, Any]], indices: list[int]) -> str:
    relations = [rows[index]["relation"] for index in indices if rows[index]["applicable_bit"]]
    return _semantic_disposition(relations)


def _tokens(text: str) -> list[str]:
    return [value.lower() for value in TOKEN.findall(text)]


def _bm25_scores(rows: list[dict[str, Any]], indices: list[int]) -> dict[int, float]:
    documents = [_tokens(rows[index]["policy_text"]) for index in indices]
    query = collections.Counter(_tokens(rows[indices[0]]["action_text"]))
    lengths = [len(document) for document in documents]
    average = _mean(lengths)
    document_frequency = {term: sum(term in document for document in documents) for term in query}
    scores: dict[int, float] = {}
    for index, document in zip(indices, documents):
        counts = collections.Counter(document)
        score = 0.0
        for term, query_count in query.items():
            frequency = counts[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denom = frequency + 1.2 * (1.0 - 0.75 + 0.75 * len(document) / max(average, 1.0))
            score += inverse * frequency * 2.2 / denom * query_count
        scores[index] = score
    return scores


def _select(
    arm: str,
    rows: list[dict[str, Any]],
    representations: list[dict[str, float]],
    indices: list[int],
    threshold: float,
    budget: int,
    margin: float,
) -> tuple[list[int], int]:
    policy_key = lambda index: rows[index]["blind_policy_id"]
    euclidean = sorted(indices, key=lambda index: (representations[index]["centroid_raw_euclidean"], policy_key(index)))
    cosine = sorted(indices, key=lambda index: (-representations[index]["centroid_cosine"], policy_key(index)))
    dot = sorted(indices, key=lambda index: (-representations[index]["centroid_raw_dot_product"], policy_key(index)))
    if arm in {"SERIAL_EUCLIDEAN_THEN_COSINE", "EUCLIDEAN_ONLY"}:
        eligible = [index for index in euclidean if representations[index]["centroid_raw_euclidean"] <= threshold]
        return eligible[:budget], len(indices)
    if arm == "COSINE_ONLY":
        return cosine[:budget], len(indices)
    if arm == "REVERSED_COSINE_THEN_EUCLIDEAN":
        eligible = [index for index in indices if abs(representations[index]["direction_margin"]) > margin]
        eligible.sort(key=lambda index: (representations[index]["centroid_raw_euclidean"], policy_key(index)))
        return eligible[:budget], len(indices)
    if arm == "CONVENTIONAL_TOP_K_RETRIEVAL":
        scores = _bm25_scores(rows, indices)
        return sorted(indices, key=lambda index: (-scores[index], policy_key(index)))[:budget], len(indices)
    if arm == "DETERMINISTIC_KEYWORD_OR_RULE_ROUTING_BASELINE":
        action_terms = set(_tokens(rows[indices[0]]["action_text"]))
        scored = [(len(action_terms.intersection(rows[index]["policy_route_terms"])), index) for index in indices]
        eligible = [(score, index) for score, index in scored if score > 0]
        return [index for _score, index in sorted(eligible, key=lambda item: (-item[0], policy_key(item[1])))[:budget]], len(indices)
    if arm == "FROZEN_FULL_RDVI_PROFILE":
        ranks = {"e": {index: rank for rank, index in enumerate(euclidean)}, "c": {index: rank for rank, index in enumerate(cosine)}, "d": {index: rank for rank, index in enumerate(dot)}}
        ordered = sorted(indices, key=lambda index: (min(ranks[key][index] for key in ranks), ranks["e"][index], ranks["c"][index], ranks["d"][index], policy_key(index)))
        return ordered[:budget], len(indices)
    if arm == "EXHAUSTIVE_POLICY_EVALUATION_ORACLE":
        return list(indices), 0
    raise ValueError(f"UNKNOWN_ARM:{arm}")


def _evaluate_cases(
    rows: list[dict[str, Any]],
    representations: list[dict[str, float]],
    arm: str,
    threshold: float,
    budget: int,
    margin: float,
) -> list[dict[str, Any]]:
    outcomes = []
    for indices in _groups(rows):
        selected, semantic_scored = _select(arm, rows, representations, indices, threshold, budget, margin)
        if arm == "EXHAUSTIVE_POLICY_EVALUATION_ORACLE":
            predicted = _oracle_disposition(rows, indices)
            predicted_relations = [rows[index]["relation"] for index in selected if rows[index]["applicable_bit"]]
        elif arm == "EUCLIDEAN_ONLY":
            predicted = "ESCALATE" if selected else "ACT"
            predicted_relations = ["AMBIGUOUS"] * len(selected)
        else:
            predicted_relations = [_direction(representations[index], margin) for index in selected]
            predicted = _semantic_disposition(predicted_relations)
        applicable = {index for index in indices if rows[index]["applicable_bit"]}
        blocking = {index for index in indices if rows[index]["blocking_bit"]}
        selected_set = set(selected)
        head = rows[indices[0]]
        outcomes.append({
            "blind_action_id": head["blind_action_id"], "family_id": head["family_id"], "cardinality": head["cardinality"],
            "abstraction": head["abstraction"], "surface_class": head["surface_class"], "expected": head["expected_disposition"], "predicted": predicted,
            "applicable": len(applicable), "applicable_recovered": len(applicable & selected_set), "blocking": len(blocking), "blocking_missed": len(blocking - selected_set),
            "selected": len(selected), "irrelevant_selected": len(selected_set - applicable), "semantic_policies_scored": semantic_scored,
            "selected_blind_policy_ids": [rows[index]["blind_policy_id"] for index in selected], "predicted_relations": predicted_relations,
        })
    return outcomes


def _metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    if not outcomes:
        return {"actions": 0}
    applicable = sum(row["applicable"] for row in outcomes); recovered = sum(row["applicable_recovered"] for row in outcomes)
    blocking = sum(row["blocking"] for row in outcomes); missed = sum(row["blocking_missed"] for row in outcomes)
    selected = sum(row["selected"] for row in outcomes); irrelevant = sum(row["irrelevant_selected"] for row in outcomes)
    non_act = [row for row in outcomes if row["expected"] != "ACT"]
    non_hold = [row for row in outcomes if row["expected"] != "HOLD"]
    conflicts = [row for row in outcomes if row["cardinality"] == "MULTIPLE_CONFLICTING"]
    average_selected = _mean(row["selected"] for row in outcomes)
    return {
        "actions": len(outcomes),
        "governing_policy_recall": recovered / applicable if applicable else 1.0,
        "blocking_policy_miss_rate": missed / blocking if blocking else 0.0,
        "final_disposition_accuracy": _mean(row["predicted"] == row["expected"] for row in outcomes),
        "false_act_rate": _mean(row["predicted"] == "ACT" for row in non_act),
        "false_hold_rate": _mean(row["predicted"] == "HOLD" for row in non_hold),
        "escalation_rate": _mean(row["predicted"] == "ESCALATE" for row in outcomes),
        "conflict_escalation_recall": _mean(row["predicted"] == "ESCALATE" for row in conflicts),
        "policies_evaluated_per_action": average_selected,
        "semantic_policies_scored_per_action": _mean(row["semantic_policies_scored"] for row in outcomes),
        "policy_evaluation_reduction_vs_exhaustive": 1.0 - average_selected / 8.0,
        "irrelevant_review_rate": irrelevant / selected if selected else 0.0,
        "governing_policies": applicable, "governing_policies_recovered": recovered, "blocking_policies": blocking, "blocking_policies_missed": missed,
    }


def _by(outcomes: list[dict[str, Any]], field: str, values: Iterable[str]) -> dict[str, Any]:
    return {value: _metrics([row for row in outcomes if row[field] == value]) for value in values}


def _threshold_candidates(representations: list[dict[str, float]]) -> list[float]:
    values = np.asarray(sorted({float(row["centroid_raw_euclidean"]) for row in representations}), dtype=np.float64)
    if not len(values):
        raise ValueError("NO_EUCLIDEAN_CALIBRATION_VALUES")
    positions = np.unique(np.linspace(0, len(values) - 1, 33).round().astype(int))
    candidates = [float(values[index]) for index in positions]
    candidates.append(float(np.nextafter(values[-1], np.inf)))
    return sorted(set(candidates))


def calibrate_thresholds(rows: list[dict[str, Any]], representations: list[dict[str, float]]) -> dict[str, Any]:
    if len(rows) != 3456 or len(rows) != len(representations) or {row["split"] for row in rows} != {"CALIBRATION"}:
        raise ValueError("CALIBRATION_CENSUS_OR_SPLIT_MISMATCH")
    curve = []
    for budget in (1, 2, 3, 4):
        for threshold in _threshold_candidates(representations):
            for margin in (0.0, 0.01, 0.02, 0.05, 0.10):
                metrics = _metrics(_evaluate_cases(rows, representations, ARMS[0], threshold, budget, margin))
                safe = metrics["governing_policy_recall"] >= 0.95 and metrics["blocking_policy_miss_rate"] == 0.0 and metrics["false_act_rate"] == 0.0
                curve.append({"budget": budget, "euclidean_threshold": threshold, "direction_margin": margin, "meets_safety_targets": safe, **metrics})
    feasible = [row for row in curve if row["meets_safety_targets"]]
    if feasible:
        selected = sorted(feasible, key=lambda row: (row["budget"], -row["final_disposition_accuracy"], -row["governing_policy_recall"], row["euclidean_threshold"], row["direction_margin"]))[0]
        status = "CALIBRATION_TARGETS_MET"
    else:
        selected = sorted(curve, key=lambda row: (row["false_act_rate"], row["blocking_policy_miss_rate"], -row["governing_policy_recall"], -row["final_disposition_accuracy"], row["budget"], row["euclidean_threshold"], row["direction_margin"]))[0]
        status = "CALIBRATION_TARGETS_NOT_MET_FAIL_CLOSED_SELECTION"
    return {
        "schema_version": "K_F1E_THRESHOLD_FREEZE_V0_1", "source_split": "CALIBRATION_ONLY", "selection_status": status,
        "selected": selected, "candidate_count": len(curve), "candidate_curve": curve,
        "tie_order": ["MEET_SAFETY", "MIN_POLICY_BUDGET", "MAX_DISPOSITION_ACCURACY", "MAX_GOVERNING_RECALL", "MIN_THRESHOLD", "MIN_MARGIN"],
    }


def _family_support(by_family: dict[str, Any]) -> dict[str, Any]:
    support = {
        family: metrics["governing_policy_recall"] >= 0.95 and metrics["blocking_policy_miss_rate"] == 0.0 and metrics["false_act_rate"] == 0.0 and metrics["final_disposition_accuracy"] >= 0.95 and metrics["policy_evaluation_reduction_vs_exhaustive"] >= 0.50
        for family, metrics in by_family.items()
    }
    return {"per_family": support, "supporting_families": sum(support.values()), "required": 7}


def _serial_disposition(metrics: dict[str, Any], by_surface: dict[str, Any], support: dict[str, Any]) -> str:
    passes = (
        metrics["governing_policy_recall"] >= 0.95 and metrics["blocking_policy_miss_rate"] == 0.0 and
        metrics["final_disposition_accuracy"] >= 0.95 and metrics["false_act_rate"] == 0.0 and
        metrics["policy_evaluation_reduction_vs_exhaustive"] >= 0.50 and metrics["conflict_escalation_recall"] >= 0.95 and
        all(row["governing_policy_recall"] >= 0.90 for row in by_surface.values()) and support["supporting_families"] >= 7
    )
    if passes:
        return "WORKABLE"
    if metrics["false_act_rate"] >= 0.10 or metrics["blocking_policy_miss_rate"] >= 0.25:
        return "HARMFUL"
    if metrics["false_act_rate"] == 0.0 and metrics["blocking_policy_miss_rate"] <= 0.05 and metrics["final_disposition_accuracy"] >= 0.80:
        return "NARROW"
    return "NULL"


def analyze_encoder(rows: list[dict[str, Any]], representations: list[dict[str, float]], threshold_freeze: dict[str, Any]) -> dict[str, Any]:
    if len(rows) != 6912 or len(rows) != len(representations) or {row["split"] for row in rows} != {"SEALED"}:
        raise ValueError("SEALED_ANALYSIS_CENSUS_OR_SPLIT_MISMATCH")
    selected = threshold_freeze["selected"]
    threshold = float(selected["euclidean_threshold"]); budget = int(selected["budget"]); margin = float(selected["direction_margin"])
    arms: dict[str, Any] = {}
    for arm in ARMS:
        outcomes = _evaluate_cases(rows, representations, arm, threshold, budget, margin)
        overall = _metrics(outcomes)
        by_family = _by(outcomes, "family_id", sorted({row["family_id"] for row in outcomes}))
        by_surface = _by(outcomes, "surface_class", SURFACES)
        arms[arm] = {
            "overall": overall, "by_family": by_family, "by_cardinality": _by(outcomes, "cardinality", CARDINALITIES),
            "by_abstraction": _by(outcomes, "abstraction", ABSTRACTIONS), "by_surface": by_surface,
            "family_support": _family_support(by_family),
            "case_outcomes": outcomes,
        }
    serial = arms[ARMS[0]]
    curves = {}
    for candidate_budget in range(1, 9):
        outcomes = _evaluate_cases(rows, representations, ARMS[0], threshold, candidate_budget, margin)
        curves[str(candidate_budget)] = _metrics(outcomes)
    residuals = [abs(float(row["centroid_normalized_euclidean_cosine_identity_residual"])) for row in representations]
    return {
        "threshold_freeze": threshold_freeze, "arms": arms,
        "serial_budget_curve_descriptive_only": curves,
        "serial_disposition": _serial_disposition(serial["overall"], serial["by_surface"], serial["family_support"]),
        "controls": {"normalized_euclidean_identity_max_abs_residual": max(residuals), "normalized_euclidean_independent_arm": False, "nonfinite_scores": 0},
        "claim_boundary": "CLOSED_SYNTHETIC_POLICY_WORLD_APPLICABILITY_AND_DISPOSITION_ONLY_NOT_EXTERNAL_POLICY_CORRECTNESS",
        "unfavorable_outcomes_permitted": ["NULL", "HARMFUL", "INVALID", "INFEASIBLE"],
    }


def terminal_classification(per_encoder: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    if set(per_encoder) != {"E01", "E02", "E03"}:
        raise ValueError("EXACT_THREE_ENCODERS_REQUIRED")
    dispositions = {slot: per_encoder[slot]["serial_disposition"] for slot in sorted(per_encoder)}
    values = list(dispositions.values())
    if values.count("WORKABLE") >= 2 and "HARMFUL" not in values:
        cross = "WORKABLE"
    elif values.count("HARMFUL") >= 2:
        cross = "HARMFUL"
    elif "WORKABLE" in values or "NARROW" in values:
        cross = "NARROW"
    else:
        cross = "NULL"
    return {
        "primary_serial_path": {"per_encoder": dispositions, "cross_encoder_disposition": cross},
        "latency": latency,
        "claim_boundary": "SERIAL_SEMANTIC_ROUTING_USEFULNESS_IN_THE_FROZEN_MACHINE_VERIFIABLE_POLICY_WORLD_ONLY",
    }
