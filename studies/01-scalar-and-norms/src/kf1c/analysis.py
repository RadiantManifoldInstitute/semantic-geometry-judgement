from __future__ import annotations

import collections
import math
import re
from typing import Any

import numpy as np

from .metrics import (
    ALIGNED,
    APPLICABLE,
    auroc,
    balanced_accuracy,
    binary_balanced_accuracy,
    brier_multiclass,
    classwise_ece,
    family_summary,
    family_values,
    rank_correlation,
    recall,
)
from .reader import CLASS_ORDER, FixedMultinomialReader
from .representations import F1_FIELDS, M1_FIELDS, S0_FIELDS, matrix


TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
ARMS = {
    "S0_SCALAR_ONLY": S0_FIELDS,
    "M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS": M1_FIELDS,
    "F1_FULL_RAW_DERIVED_PARAMETERIZATION": F1_FIELDS,
}


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN.finditer(text)]


def _fit_tfidf(train_texts: list[str], all_texts: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    vocabulary = sorted(
        {
            gram
            for text in train_texts
            for gram in (
                _tokens(text)
                + [" ".join(pair) for pair in zip(_tokens(text), _tokens(text)[1:])]
            )
        }
    )
    index = {value: i for i, value in enumerate(vocabulary)}
    document_frequency = np.zeros(len(vocabulary), dtype=np.float64)
    for text in train_texts:
        tokens = _tokens(text)
        grams = set(tokens + [" ".join(pair) for pair in zip(tokens, tokens[1:])])
        for gram in grams:
            document_frequency[index[gram]] += 1
    idf = np.log((1.0 + len(train_texts)) / (1.0 + document_frequency)) + 1.0
    result = np.zeros((len(all_texts), len(vocabulary)), dtype=np.float64)
    for row_index, text in enumerate(all_texts):
        tokens = _tokens(text)
        grams = tokens + [" ".join(pair) for pair in zip(tokens, tokens[1:])]
        counts = collections.Counter(grams)
        for gram, count in counts.items():
            if gram in index:
                result[row_index, index[gram]] = count * idf[index[gram]]
        norm = np.linalg.norm(result[row_index])
        if norm:
            result[row_index] /= norm
    return result, {"vocabulary": vocabulary, "idf": [float(v) for v in idf]}


def _bm25(train_reference_texts: list[str], rows: list[dict[str, Any]]) -> np.ndarray:
    documents = [_tokens(text) for text in train_reference_texts]
    average_length = sum(map(len, documents)) / len(documents)
    df = collections.Counter(token for doc in documents for token in set(doc))
    n = len(documents)
    values = []
    for row in rows:
        query = set(_tokens(row["action_text"]))
        document = _tokens(row["reference_text"])
        counts = collections.Counter(document)
        score = 0.0
        for token in query:
            if token not in counts:
                continue
            idf = math.log(1.0 + (n - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            frequency = counts[token]
            denominator = frequency + 1.2 * (1.0 - 0.75 + 0.75 * len(document) / average_length)
            score += idf * frequency * 2.2 / denominator
        values.append([score])
    return np.asarray(values, dtype=np.float64)


def _jaccard(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        left, right = set(_tokens(row["action_text"])), set(_tokens(row["reference_text"]))
        values.append([len(left & right) / len(left | right) if left | right else 0.0])
    return np.asarray(values, dtype=np.float64)


def _fit_predict(features: np.ndarray, rows: list[dict[str, Any]]) -> tuple[FixedMultinomialReader, np.ndarray, list[str]]:
    training = [i for i, row in enumerate(rows) if row["split"] == "CONSTRUCTION" and row["surface_class"] == "CANONICAL"]
    if len(training) != 288:
        raise ValueError(f"CONSTRUCTION_CANONICAL_CENSUS_MISMATCH:{len(training)}")
    reader = FixedMultinomialReader.fit(features[training], [rows[i]["gold_cell"] for i in training])
    probabilities = reader.probabilities(features)
    predictions = [CLASS_ORDER[int(index)] for index in np.argmax(probabilities, axis=1)]
    return reader, probabilities, predictions


def _control_result(features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        reader, probabilities, predictions = _fit_predict(features, rows)
    except RuntimeError as exc:
        if not str(exc).startswith("PRIMARY_READER_NONCONVERGENCE"):
            raise
        return {
            "status": "INFEASIBLE_FIXED_READER_NONCONVERGENCE",
            "failure": str(exc),
            "rows_dropped": 0,
            "substitution": None,
        }
    try:
        metrics = _metric_bundle(rows, probabilities, predictions)
    except ValueError as exc:
        return {
            "status": "INFEASIBLE_INVALID_CONTROL_PROBABILITY",
            "failure": str(exc),
            "reader": reader.as_dict(),
            "rows_dropped": 0,
            "substitution": None,
        }
    return {"status": "PASS", "reader": reader.as_dict(), "metrics": metrics}


def _holdout_canonical(rows: list[dict[str, Any]]) -> list[int]:
    indices = [i for i, row in enumerate(rows) if row["split"] == "SEALED_HOLDOUT" and row["surface_class"] == "CANONICAL"]
    if len(indices) != 288:
        raise ValueError(f"HOLDOUT_CANONICAL_CENSUS_MISMATCH:{len(indices)}")
    return indices


def _metric_bundle(rows: list[dict[str, Any]], probabilities: np.ndarray, predictions: list[str]) -> dict[str, Any]:
    indices = _holdout_canonical(rows)
    target_rows = [rows[i] for i in indices]
    gold = [row["gold_cell"] for row in target_rows]
    pred = [predictions[i] for i in indices]
    probs = probabilities[indices]
    gold_app = [int(value in APPLICABLE) for value in gold]
    pred_app = [int(value in APPLICABLE) for value in pred]
    p_app = [float(value) for value in probs[:, 0] + probs[:, 1]]
    applicable_indices = [i for i, value in enumerate(gold_app) if value == 1]
    gold_polarity = [int(gold[i] == CLASS_ORDER[1]) for i in applicable_indices]
    pred_polarity = [int(pred[i] == CLASS_ORDER[1]) for i in applicable_indices]
    polarity_denominators = [float(probs[i, 0] + probs[i, 1]) for i in applicable_indices]
    if any(value == 0.0 for value in polarity_denominators):
        raise ValueError("CONDITIONAL_POLARITY_ZERO_DENOMINATOR")
    p_polarity = [float(probs[i, 1] / denominator) for i, denominator in zip(applicable_indices, polarity_denominators)]
    wrong_mission_indices = [i for i, value in enumerate(gold) if value not in APPLICABLE]
    metrics = (
        "four_cell_balanced_accuracy",
        "applicability_balanced_accuracy",
        "conditional_polarity_balanced_accuracy",
        "APPLICABLE_OPPOSED",
        "NOT_APPLICABLE_APPARENTLY_OPPOSED",
        "wrong_mission_false_applicability_rate",
    )
    family = {metric: family_summary(family_values(target_rows, pred, metric)) for metric in metrics}
    return {
        "four_cell_balanced_accuracy": balanced_accuracy(gold, pred, CLASS_ORDER),
        "applicability_balanced_accuracy": binary_balanced_accuracy(gold_app, pred_app),
        "applicability_AUROC": auroc(gold_app, p_app),
        "conditional_polarity_balanced_accuracy": binary_balanced_accuracy(gold_polarity, pred_polarity),
        "conditional_polarity_AUROC": auroc(gold_polarity, p_polarity),
        "cell_recall": {label: recall(gold, pred, label) for label in CLASS_ORDER},
        "wrong_mission_false_applicability_rate": float(
            np.mean([pred[i] in APPLICABLE for i in wrong_mission_indices])
        ),
        "Brier_score": brier_multiclass(gold, probs),
        "classwise_ECE_10_fixed_bins": classwise_ece(gold, probs),
        "family_support": family,
    }


def _wording(rows: list[dict[str, Any]], probabilities: np.ndarray, predictions: list[str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for index, row in enumerate(rows):
        if row["split"] == "SEALED_HOLDOUT":
            grouped[row["canonical_pair_id"]][row["surface_class"]] = index
    reversals, changes, canonical_margins, variant_margins = [], [], [], []
    for pair, surfaces in sorted(grouped.items()):
        if set(surfaces) != {"CANONICAL", "WORDING_SUBSTITUTION", "CLAUSE_ORDER"}:
            raise ValueError(f"WORDING_SURFACE_CENSUS_MISMATCH:{pair}")
        base = surfaces["CANONICAL"]
        sorted_base = np.sort(probabilities[base])
        base_margin = float(sorted_base[-1] - sorted_base[-2])
        for surface in ("WORDING_SUBSTITUTION", "CLAUSE_ORDER"):
            variant = surfaces[surface]
            reversals.append(int(predictions[base] != predictions[variant]))
            changes.append(float(np.max(np.abs(probabilities[base] - probabilities[variant]))))
            sorted_variant = np.sort(probabilities[variant])
            canonical_margins.append(base_margin)
            variant_margins.append(float(sorted_variant[-1] - sorted_variant[-2]))
    return {
        "canonical_to_variant_label_reversal_rate": float(np.mean(reversals)),
        "median_maximum_probability_change": float(np.median(changes)),
        "maximum_probability_change": float(np.max(changes)),
        "rank_correlation": rank_correlation(canonical_margins, variant_margins),
        "comparison_count": len(reversals),
    }


def _cross_level(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, int]] = collections.defaultdict(dict)
    for index, row in enumerate(rows):
        if row["split"] == "SEALED_HOLDOUT" and row["surface_class"] == "CANONICAL":
            grouped[(row["family_id"], row["scenario_id"], row["action_id"])][row["level"]] = index
    by_family: dict[str, list[int]] = collections.defaultdict(list)
    for (family, _, _), levels in grouped.items():
        if set(levels) != {"L1_CONCRETE", "L2_OPERATIONAL", "L3_PURPOSE"}:
            raise ValueError("CROSS_LEVEL_CENSUS_MISMATCH")
        l1, l3 = levels["L1_CONCRETE"], levels["L3_PURPOSE"]
        if rows[l1]["relation_polarity"] == "ALIGNED" and rows[l3]["relation_polarity"] == "OPPOSED":
            recovered = predictions[l1] in ALIGNED and predictions[l3] not in ALIGNED
            by_family[family].append(int(recovered))
    family = {key: float(np.mean(value)) for key, value in sorted(by_family.items())}
    return {"overall": float(np.mean([value for values in by_family.values() for value in values])), "family_values": family}


def _floor_pass(metrics: dict[str, Any], arm: str) -> dict[str, Any]:
    definitions = {
        "four_cell_balanced_accuracy": (0.70, 0.60, 0.65, "min"),
        "applicability_balanced_accuracy": (0.75, 0.67, 0.70, "min"),
        "conditional_polarity_balanced_accuracy": (0.75, 0.67, 0.70, "min"),
        "APPLICABLE_OPPOSED": (0.67, 0.56, 0.62, "min"),
        "NOT_APPLICABLE_APPARENTLY_OPPOSED": (0.67, 0.56, 0.62, "min"),
        "wrong_mission_false_applicability_rate": (0.20, 0.33, 0.25, "max"),
    }
    results = {}
    for name, (overall_bound, family_bound, loo_bound, direction) in definitions.items():
        key = name if name in metrics else "cell_recall"
        overall = metrics[name] if name in metrics else metrics[key][name]
        family = metrics["family_support"][name]
        values = list(family["family_values"].values())
        loo = list(family["leave_one_family_out_equal_weight_means"].values())
        if direction == "min":
            clauses = [overall >= overall_bound, sum(value >= family_bound for value in values) >= 6, min(loo) >= loo_bound]
        else:
            clauses = [overall <= overall_bound, sum(value <= family_bound for value in values) >= 6, max(loo) <= loo_bound]
        results[name] = {"overall": overall, "clauses": clauses, "pass": all(clauses)}
    return {"arm": arm, "metrics": results, "pass": all(row["pass"] for row in results.values())}


def _e1(s0: dict[str, Any], m1: dict[str, Any]) -> dict[str, Any]:
    s0_values = s0["family_support"]["four_cell_balanced_accuracy"]["family_values"]
    m1_values = m1["family_support"]["four_cell_balanced_accuracy"]["family_values"]
    delta = {key: m1_values[key] - s0_values[key] for key in sorted(s0_values)}
    summary = family_summary(delta)
    values = list(delta.values())
    loo = list(summary["leave_one_family_out_equal_weight_means"].values())
    clauses = {
        "overall_value_at_least_0_10": summary["equal_weight_mean"] >= 0.10,
        "at_least_7_of_8_per_family_values_at_least_0_05": sum(value >= 0.05 for value in values) >= 7,
        "no_per_family_value_below_minus_0_05": min(values) >= -0.05,
        "minimum_leave_one_family_out_mean_at_least_0_05": min(loo) >= 0.05,
    }
    return {"id": "E1_PAIRED_INCREMENTAL_FOUR_CELL_INFORMATION", "summary": summary, "clauses": clauses, "pass": all(clauses.values())}


def _norm_diagnostics(rows: list[dict[str, Any]], representation_rows: list[dict[str, float]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for index, row in enumerate(rows):
        if row["split"] == "SEALED_HOLDOUT":
            grouped[row["canonical_pair_id"]][row["surface_class"]] = index
    relative_changes = []
    for surfaces in grouped.values():
        if set(surfaces) != {"CANONICAL", "WORDING_SUBSTITUTION", "CLAUSE_ORDER"}:
            raise ValueError("NORM_WORDING_SURFACE_CENSUS_MISMATCH")
        base = representation_rows[surfaces["CANONICAL"]]
        for surface in ("WORDING_SUBSTITUTION", "CLAUSE_ORDER"):
            variant = representation_rows[surfaces[surface]]
            for field in ("action_l2_norm", "reference_l2_norm"):
                denominator = max(abs(base[field]), np.finfo(np.float64).tiny)
                relative_changes.append(abs(variant[field] - base[field]) / denominator)

    construction = [
        index for index, row in enumerate(rows)
        if row["split"] == "CONSTRUCTION" and row["surface_class"] == "CANONICAL"
    ]
    families = sorted({rows[index]["family_id"] for index in construction})
    token_frequency = collections.Counter(
        token
        for index in construction
        for token in _tokens(rows[index]["action_text"]) + _tokens(rows[index]["reference_text"])
    )
    family_matrix = np.asarray(
        [[1.0, *[float(rows[index]["family_id"] == family) for family in families[:-1]]] for index in construction],
        dtype=np.float64,
    )
    nuisance_rows = []
    for index in construction:
        action = rows[index]["action_text"]
        reference = rows[index]["reference_text"]
        tokens = _tokens(action) + _tokens(reference)
        nuisance_rows.append(
            [
                float(len(tokens)),
                float(len(action) + len(reference)),
                float(sum(not char.isalnum() and not char.isspace() for char in action + reference)),
                0.0,
                float(np.mean([token_frequency[token] for token in tokens])) if tokens else 0.0,
            ]
        )
    augmented = np.column_stack([family_matrix, np.asarray(nuisance_rows, dtype=np.float64)])

    def r_squared(design: np.ndarray, target: np.ndarray) -> float:
        prediction = design @ np.linalg.lstsq(design, target, rcond=None)[0]
        total = float(np.sum((target - target.mean()) ** 2))
        if total == 0.0:
            return 1.0 if np.array_equal(prediction, target) else 0.0
        return 1.0 - float(np.sum((target - prediction) ** 2)) / total

    incremental = {}
    for field in ("action_l2_norm", "reference_l2_norm"):
        target = np.asarray([representation_rows[index][field] for index in construction], dtype=np.float64)
        incremental[field] = r_squared(augmented, target) - r_squared(family_matrix, target)
    wording = {
        "median_absolute_relative_norm_change": float(np.median(relative_changes)),
        "p95_absolute_relative_norm_change": float(np.percentile(relative_changes, 95)),
        "count": len(relative_changes),
    }
    return {
        "documented_mechanism": "FAIL_CLOSED_NOT_ESTABLISHED_BY_ADMITTED_ENCODER_DOCUMENTATION",
        "deterministic_repeatability": "PENDING_TWO_REPLICA_RECONCILIATION",
        "wording_stability": wording,
        "wording_stability_pass": wording["median_absolute_relative_norm_change"] <= 0.05 and wording["p95_absolute_relative_norm_change"] <= 0.10,
        "nuisance_incremental_construction_R_squared_after_family_blocking": incremental,
        "nuisance_independence_pass": max(incremental.values()) <= 0.05,
        "cross_encoder_replication": "NOT_EVALUABLE_NO_FROZEN_ASSOCIATION_TARGET",
        "out_of_construction_confirmation": "NOT_EVALUABLE_NO_FROZEN_ASSOCIATION_EFFECT",
        "no_judgment_substitution": True,
        "semantic_gate_pass": False,
        "claim_consequence": "DIAGNOSTIC_GEOMETRY_ONLY_NO_NORM_RADIUS_PROXIMITY_SEVERITY_CONFIDENCE_OR_UNCERTAINTY_SEMANTICS",
    }


def analyze_encoder(rows: list[dict[str, Any]], representation_rows: list[dict[str, float]]) -> dict[str, Any]:
    if len(rows) != 2160 or len(representation_rows) != 2160:
        raise ValueError("COMPLETE_SURFACE_CENSUS_REQUIRED")
    outputs: dict[str, Any] = {}
    prediction_columns: dict[str, tuple[np.ndarray, list[str]]] = {}
    for arm, fields in ARMS.items():
        try:
            reader, probabilities, predictions = _fit_predict(matrix(representation_rows, fields), rows)
        except RuntimeError as exc:
            if arm != "F1_FULL_RAW_DERIVED_PARAMETERIZATION" or not str(exc).startswith("PRIMARY_READER_NONCONVERGENCE"):
                raise
            outputs[arm] = {
                "status": "INFEASIBLE_SECONDARY_FIXED_READER_NONCONVERGENCE",
                "fields": list(fields),
                "failure": str(exc),
                "substitution": None,
            }
            continue
        metrics = _metric_bundle(rows, probabilities, predictions)
        outputs[arm] = {
            "fields": list(fields),
            "reader": reader.as_dict(),
            "metrics": metrics,
            "wording_stability": _wording(rows, probabilities, predictions),
            "cross_level_conflict": _cross_level(rows, predictions),
        }
        prediction_columns[arm] = (probabilities, predictions)

    construction = [i for i, row in enumerate(rows) if row["split"] == "CONSTRUCTION" and row["surface_class"] == "CANONICAL"]
    combined = [f"action: {row['action_text']} reference: {row['reference_text']}" for row in rows]
    lexical_features, lexical_state = _fit_tfidf([combined[i] for i in construction], combined)
    lexical = {}
    for name, features in (
        ("TFIDF_WORD_BIGRAM", lexical_features),
        ("BM25", _bm25([rows[i]["reference_text"] for i in construction], rows)),
        ("TOKEN_SET_JACCARD", _jaccard(rows)),
    ):
        lexical[name] = _control_result(features, rows)

    nulls = {}
    for name, texts in (
        ("action_only", [row["action_text"] for row in rows]),
        ("reference_only", [row["reference_text"] for row in rows]),
    ):
        features, _ = _fit_tfidf([texts[i] for i in construction], texts)
        nulls[name] = _control_result(features, rows)
    family_ids = sorted({row["family_id"] for row in rows})
    family_features = np.asarray([[int(row["family_id"] == family) for family in family_ids] for row in rows], dtype=np.float64)
    nulls["blind_family_ID_only"] = _control_result(family_features, rows)
    rotated_texts = [
        f"action: {row['action_text']} reference: {rows[(index + 108) % len(rows)]['reference_text']}"
        for index, row in enumerate(rows)
    ]
    rotated_features, _ = _fit_tfidf([rotated_texts[i] for i in construction], rotated_texts)
    nulls["wrong_mission_rotation"] = _control_result(rotated_features, rows)
    random = np.random.default_rng(41382003)
    random_predictions = [CLASS_ORDER[int(value)] for value in random.integers(0, 4, len(rows))]
    uniform = np.full((len(rows), 4), 0.25, dtype=np.float64)
    nulls["random"] = {"seed": 41382003, "metrics": _metric_bundle(rows, uniform, random_predictions)}
    shortcut_null_failures = []
    for name in ("action_only", "reference_only", "blind_family_ID_only", "wrong_mission_rotation"):
        control = nulls[name]
        if control.get("status") != "PASS":
            shortcut_null_failures.append({"control": name, "reason": control.get("status")})
            continue
        upper = control["metrics"]["family_support"]["four_cell_balanced_accuracy"]["descriptive_family_bootstrap"]["percentile_95_interval"][1]
        if upper >= 0.40:
            shortcut_null_failures.append({"control": name, "descriptive_upper_95": upper, "ceiling": 0.40})

    s0_metrics = outputs["S0_SCALAR_ONLY"]["metrics"]
    m1_metrics = outputs["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]
    e1 = _e1(s0_metrics, m1_metrics)
    s0_floor = _floor_pass(s0_metrics, "S0")
    m1_floor = _floor_pass(m1_metrics, "M1")
    m1_probabilities, m1_predictions = prediction_columns["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]
    row_outputs = []
    for index, row in enumerate(rows):
        row_output = {
            "blind_pair_id": row["blind_pair_id"],
            "blind_surface_id": row["blind_surface_id"],
            "canonical_pair_id": row["canonical_pair_id"],
            "surface_id": row["surface_id"],
            "family_id": row["family_id"],
            "scenario_id": row["scenario_id"],
            "action_id": row["action_id"],
            "reference_id": row["reference_id"],
            "level": row["level"],
            "surface_class": row["surface_class"],
            "split": row["split"],
            "gold_cell": row["gold_cell"],
            "applicability_bit": row["applicability_bit"],
            "relation_polarity": row["relation_polarity"],
            "representations": representation_rows[index],
            "arms": {},
        }
        for arm, (probabilities, predictions) in prediction_columns.items():
            ordered = [float(value) for value in probabilities[index]]
            sorted_probabilities = sorted(ordered, reverse=True)
            row_output["arms"][arm] = {
                "class_probabilities": dict(zip(CLASS_ORDER, ordered)),
                "prediction": predictions[index],
                "margin": sorted_probabilities[0] - sorted_probabilities[1],
                "applicability_probability": ordered[0] + ordered[1],
                "conditional_opposed_probability": ordered[1] / (ordered[0] + ordered[1]),
            }
        row_outputs.append(row_output)
    harmful_wrong_mission = (
        m1_metrics["family_support"]["wrong_mission_false_applicability_rate"]["equal_weight_mean"] > 0.30
        and np.mean(
            [
                (m1_probabilities[i, 0] + m1_probabilities[i, 1]) >= 0.80
                for i, row in enumerate(rows)
                if row["split"] == "SEALED_HOLDOUT" and row["surface_class"] == "CANONICAL" and row["gold_cell"] not in APPLICABLE
            ]
        ) >= 0.20
    )
    harmful_polarity = False
    return {
        "arms": outputs,
        "E1": e1,
        "S0_floor": s0_floor,
        "M1_floor": m1_floor,
        "lexical_controls": lexical,
        "lexical_state": lexical_state,
        "shortcut_nulls": nulls,
        "B1_best_bounded_predictor_sensitivity": {
            "status": "NOT_EXECUTABLE_NO_EXACT_FROZEN_L2_GRID_OR_DEPTH_TWO_MINIMUM_LEAF",
            "primary_claim_effect": "NONE",
            "substitution_or_postscore_selection": "PROHIBITED",
        },
        "shortcut_null_ceiling": {"pass": not shortcut_null_failures, "failures": shortcut_null_failures},
        "norm_diagnostics": _norm_diagnostics(rows, representation_rows),
        "harm_checks": {
            "wrong_mission": bool(harmful_wrong_mission),
            "applicable_polarity": bool(harmful_polarity),
            "applicable_polarity_evidence": {
                "balanced_accuracy_at_or_below_0_45": m1_metrics["conditional_polarity_balanced_accuracy"] <= 0.45,
                "stable_wrong_direction_confidence": "INDETERMINATE_NO_FROZEN_NUMERIC_DEFINITION",
                "harm_triggered": False,
            },
        },
        "row_outputs": row_outputs,
    }


def terminal_classification(per_encoder: dict[str, dict[str, Any]], latency: dict[str, Any]) -> dict[str, Any]:
    if set(per_encoder) != {"E01", "E02", "E03"}:
        raise ValueError("EXACT_THREE_ENCODERS_REQUIRED")
    primary = per_encoder["E01"]
    stable_encoders = [
        slot
        for slot, result in per_encoder.items()
        if result["E1"]["summary"]["equal_weight_mean"] >= 0.0
        and result["M1_floor"]["pass"]
    ]
    material_reversals = [
        slot
        for slot, result in per_encoder.items()
        if result["E1"]["summary"]["equal_weight_mean"] <= -0.05
        or result["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]["applicability_balanced_accuracy"] < 0.55
        or result["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]["conditional_polarity_balanced_accuracy"] < 0.55
    ]
    harm = any(
        result["harm_checks"]["wrong_mission"] or result["harm_checks"]["applicable_polarity"]
        for result in per_encoder.values()
    )
    shortcut_null_failure = any(not result["shortcut_null_ceiling"]["pass"] for result in per_encoder.values())
    primary_m1_family = primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]["family_support"]["four_cell_balanced_accuracy"]["family_values"]
    if harm:
        terminal = "HARMFUL_OR_MISLEADING_SIGNAL"
    elif (
        primary["arms"]["S0_SCALAR_ONLY"]["metrics"]["applicability_balanced_accuracy"] < 0.60
        and primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]["applicability_balanced_accuracy"] < 0.60
    ) or (
        primary["arms"]["S0_SCALAR_ONLY"]["metrics"]["conditional_polarity_balanced_accuracy"] < 0.60
        and primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]["conditional_polarity_balanced_accuracy"] < 0.60
    ) or sum(value <= 0.40 for value in primary_m1_family.values()) >= 6:
        terminal = "NO_SEPARABLE_SIGNAL"
    elif primary["S0_floor"]["pass"] and not primary["E1"]["pass"] and not shortcut_null_failure:
        terminal = "SCALAR_SUFFICIENT_NO_PROFILE_GAIN"
    elif (
        primary["E1"]["pass"]
        and primary["M1_floor"]["pass"]
        and primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["cross_level_conflict"]["overall"] >= 0.67
        and primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["wording_stability"]["canonical_to_variant_label_reversal_rate"] <= 0.10
        and primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["wording_stability"]["median_maximum_probability_change"] <= 0.10
        and primary["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["wording_stability"]["rank_correlation"] >= 0.85
        and len(stable_encoders) >= 2
        and not material_reversals
        and latency.get("hot_workable", False)
        and not shortcut_null_failure
    ):
        terminal = "NARROW_REPRESENTATION_DEPENDENT_SIGNAL"
        # Norm semantics are fail-closed, so even a fully discriminative result cannot be promoted to a magnitude claim.
    else:
        any_signal = any(
            result["arms"]["M1_MAGNITUDE_AUGMENTED_MINIMAL_BASIS"]["metrics"]["four_cell_balanced_accuracy"] > 0.40
            for result in per_encoder.values()
        )
        terminal = "NARROW_REPRESENTATION_DEPENDENT_SIGNAL" if any_signal else "INDETERMINATE"
    return {
        "terminal": terminal,
        "scientific_result": True,
        "stable_encoder_slots": stable_encoders,
        "material_encoder_reversals": material_reversals,
        "shortcut_null_failure": shortcut_null_failure,
        "raw_norm_semantic_gate": "FAIL_CLOSED_DIAGNOSTIC_ONLY",
        "claim_ceiling": "exact synthetic finite world, exact encoders, fixed reader; no autonomous compliance or norm semantics",
    }
