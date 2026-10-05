from __future__ import annotations

import collections
import hashlib
import itertools
import math
import re
from typing import Any, Iterable

import numpy as np


RELATIONS = ("ALIGNED", "OPPOSED", "AMBIGUOUS", "NON_GOVERNING")
RELATION_INDEX = {value: index for index, value in enumerate(RELATIONS)}
FEATURE_ARMS: dict[str, tuple[str, ...]] = {
    "COSINE_ONLY": ("cosine",),
    "EUCLIDEAN_ONLY": ("euclidean_distance",),
    "SIGNED_DOT_PRODUCT_ONLY": ("signed_dot_product",),
    "COSINE_PLUS_EUCLIDEAN": ("cosine", "euclidean_distance"),
    "COSINE_PLUS_DOT_PRODUCT": ("cosine", "signed_dot_product"),
    "EUCLIDEAN_PLUS_DOT_PRODUCT": ("euclidean_distance", "signed_dot_product"),
    "COSINE_PLUS_EUCLIDEAN_PLUS_DOT_PRODUCT": ("cosine", "euclidean_distance", "signed_dot_product"),
}
SINGLETON_ARMS = ("COSINE_ONLY", "EUCLIDEAN_ONLY", "SIGNED_DOT_PRODUCT_ONLY")
THRESHOLD_ARMS = (
    "COSINE_THRESHOLD_ROUTING", "EUCLIDEAN_THRESHOLD_ROUTING",
    "SIGNED_DOT_THRESHOLD_ROUTING", "LEXICAL_OVERLAP_ROUTING",
)
TOP_K_GRID = (1, 2, 4, 8, 16, 32, 64)
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(np.mean(rows)) if rows else 0.0


def _percentile(values: Iterable[float], q: float) -> float:
    rows = list(values)
    return float(np.percentile(np.asarray(rows, dtype=np.float64), q)) if rows else 0.0


def _groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["blind_action_id"]].append(index)
    return [grouped[key] for key in sorted(grouped)]


def lexical_overlap(row: dict[str, Any]) -> float:
    action = set(token.lower() for token in TOKEN.findall(row["action_text"]))
    anchors = set(token.lower() for token in row["policy_route_terms"])
    return len(action & anchors) / len(anchors) if anchors else 0.0


def _candidate_mask(
    rows: list[dict[str, Any]], representations: list[dict[str, float]], arm: str, parameter: Any
) -> np.ndarray:
    if len(rows) != len(representations):
        raise ValueError("ROW_REPRESENTATION_CENSUS_MISMATCH")
    mask = np.zeros(len(rows), dtype=np.bool_)
    if arm == "COSINE_THRESHOLD_ROUTING":
        return np.asarray([float(rep["cosine"]) >= float(parameter) for rep in representations], dtype=np.bool_)
    if arm == "EUCLIDEAN_THRESHOLD_ROUTING":
        return np.asarray([float(rep["euclidean_distance"]) <= float(parameter) for rep in representations], dtype=np.bool_)
    if arm == "SIGNED_DOT_THRESHOLD_ROUTING":
        return np.asarray([float(rep["signed_dot_product"]) >= float(parameter) for rep in representations], dtype=np.bool_)
    if arm == "LEXICAL_OVERLAP_ROUTING":
        return np.asarray([lexical_overlap(row) >= float(parameter) for row in rows], dtype=np.bool_)
    if arm == "DETERMINISTIC_KEYWORD_ROUTING_BASELINE":
        return np.asarray([lexical_overlap(row) > 0.0 for row in rows], dtype=np.bool_)
    if arm == "EXHAUSTIVE_ORACLE_EVALUATION":
        return np.ones(len(rows), dtype=np.bool_)
    if arm == "CONVENTIONAL_TOP_K":
        k = int(parameter)
        for indices in _groups(rows):
            ordered = sorted(indices, key=lambda i: (-float(representations[i]["cosine"]), rows[i]["blind_policy_id"]))
            mask[ordered[: min(k, len(ordered))]] = True
        return mask
    if arm == "COMBINED_SEMANTIC_ROUTING":
        for source in THRESHOLD_ARMS:
            mask |= _candidate_mask(rows, representations, source, parameter[source])
        return mask
    raise ValueError(f"UNKNOWN_ROUTER_ARM:{arm}")


def router_predictions(rows: list[dict[str, Any]], representations: list[dict[str, float]], freeze: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {
        arm: _candidate_mask(rows, representations, arm, freeze["selected_parameters"][arm])
        for arm in THRESHOLD_ARMS
    }
    result["COMBINED_SEMANTIC_ROUTING"] = _candidate_mask(rows, representations, "COMBINED_SEMANTIC_ROUTING", freeze["selected_parameters"])
    result["DETERMINISTIC_KEYWORD_ROUTING_BASELINE"] = _candidate_mask(rows, representations, "DETERMINISTIC_KEYWORD_ROUTING_BASELINE", None)
    result["EXHAUSTIVE_ORACLE_EVALUATION"] = np.ones(len(rows), dtype=np.bool_)
    for k in TOP_K_GRID:
        result[f"CONVENTIONAL_TOP_K_K{k}"] = _candidate_mask(rows, representations, "CONVENTIONAL_TOP_K", k)
    result["CONVENTIONAL_TOP_K"] = result[f"CONVENTIONAL_TOP_K_K{freeze['selected_top_k']}"]
    return result


def _routing_metrics(rows: list[dict[str, Any]], mask: np.ndarray) -> dict[str, Any]:
    outcomes = []
    for indices in _groups(rows):
        selected = {index for index in indices if bool(mask[index])}
        governing = {index for index in indices if int(rows[index]["governing"])}
        blocking = {index for index in indices if int(rows[index]["blocking"])}
        head = rows[indices[0]]
        outcomes.append({
            "family_id": head["family_id"], "policy_set_size": head["policy_set_size"],
            "cardinality": head["cardinality"], "perturbation_kind": head["perturbation_kind"],
            "base_action_id": head["base_action_id"], "blind_action_id": head["blind_action_id"],
            "policy_count": len(indices), "selected": len(selected), "governing": len(governing),
            "recovered": len(governing & selected), "blocking": len(blocking),
            "blocking_missed": len(blocking - selected), "irrelevant": len(selected - governing),
            "selected_policy_ids": tuple(sorted(rows[index]["blind_policy_id"] for index in selected)),
        })
    governing = sum(row["governing"] for row in outcomes); recovered = sum(row["recovered"] for row in outcomes)
    blocking = sum(row["blocking"] for row in outcomes); missed = sum(row["blocking_missed"] for row in outcomes)
    selected = [row["selected"] for row in outcomes]; irrelevant = sum(row["irrelevant"] for row in outcomes)
    reductions = [1.0 - row["selected"] / row["policy_count"] for row in outcomes]
    by_base = {(row["family_id"], row["base_action_id"]): row for row in outcomes if row["perturbation_kind"] == "BASE"}
    stability = [
        row["selected_policy_ids"] == by_base[(row["family_id"], row["base_action_id"])]["selected_policy_ids"]
        for row in outcomes if row["perturbation_kind"] != "BASE" and (row["family_id"], row["base_action_id"]) in by_base
    ]
    return {
        "actions": len(outcomes), "governing_policies": governing, "governing_policies_recovered": recovered,
        "governing_policy_recall": recovered / governing if governing else 1.0,
        "blocking_policies": blocking, "blocking_policies_missed": missed,
        "blocking_policy_miss_rate": missed / blocking if blocking else 0.0,
        "candidate_set_size_mean": _mean(selected), "candidate_set_size_median": _percentile(selected, 50),
        "candidate_set_size_p95": _percentile(selected, 95), "policy_evaluation_reduction_median": _percentile(reductions, 50),
        "false_retrieval_burden": irrelevant / sum(selected) if sum(selected) else 0.0,
        "perturbation_exact_candidate_stability": _mean(stability),
        "case_rows": outcomes,
    }


def _without_cases(value: dict[str, Any]) -> dict[str, Any]:
    return {key: row for key, row in value.items() if key != "case_rows"}


def _threshold_grid(arm: str, representations: list[dict[str, float]]) -> list[float]:
    if arm == "COSINE_THRESHOLD_ROUTING":
        return [0.50, 0.60, 0.70, 0.80, 0.90]
    if arm == "LEXICAL_OVERLAP_ROUTING":
        return [0.25, 0.50, 0.75, 1.00]
    if arm == "EUCLIDEAN_THRESHOLD_ROUTING":
        values = np.asarray([row["euclidean_distance"] for row in representations], dtype=np.float64)
        return sorted(set(float(value) for value in np.quantile(values, [0.10, 0.20, 0.30, 0.40, 0.50])))
    if arm == "SIGNED_DOT_THRESHOLD_ROUTING":
        values = np.asarray([row["signed_dot_product"] for row in representations], dtype=np.float64)
        return sorted(set(float(value) for value in np.quantile(values, [0.50, 0.60, 0.70, 0.80, 0.90])))
    raise ValueError(arm)


def calibrate_routers(rows: list[dict[str, Any]], representations: list[dict[str, float]]) -> dict[str, Any]:
    if {row["split"] for row in rows} != {"CALIBRATION"}:
        raise ValueError("ROUTER_CALIBRATION_SPLIT_MISMATCH")
    selected: dict[str, float] = {}; curves: dict[str, list[dict[str, Any]]] = {}
    for arm in THRESHOLD_ARMS:
        curve = []
        for parameter in _threshold_grid(arm, representations):
            metrics = _routing_metrics(rows, _candidate_mask(rows, representations, arm, parameter))
            curve.append({"parameter": parameter, "feasible": metrics["governing_policy_recall"] >= 0.98 and metrics["blocking_policy_miss_rate"] <= 0.01, **_without_cases(metrics)})
        feasible = [row for row in curve if row["feasible"]]
        pool = feasible or curve
        choice = sorted(pool, key=lambda row: (not row["feasible"], row["candidate_set_size_median"], -row["governing_policy_recall"], row["blocking_policy_miss_rate"], row["parameter"]))[0]
        selected[arm] = float(choice["parameter"]); curves[arm] = curve
    combined = _routing_metrics(rows, _candidate_mask(rows, representations, "COMBINED_SEMANTIC_ROUTING", selected))
    top_curves = []
    for k in TOP_K_GRID:
        metrics = _routing_metrics(rows, _candidate_mask(rows, representations, "CONVENTIONAL_TOP_K", k))
        top_curves.append({"k": k, "feasible": metrics["governing_policy_recall"] >= 0.98 and metrics["blocking_policy_miss_rate"] <= 0.01, **_without_cases(metrics)})
    feasible_top = [row for row in top_curves if row["feasible"]]
    selected_top = int(sorted(feasible_top or top_curves, key=lambda row: (not row["feasible"], row["candidate_set_size_median"], -row["governing_policy_recall"], row["k"]))[0]["k"])
    candidate_metrics = {
        arm: _routing_metrics(rows, _candidate_mask(rows, representations, arm, selected[arm])) for arm in THRESHOLD_ARMS
    }
    candidate_metrics["COMBINED_SEMANTIC_ROUTING"] = combined
    feasible_semantic = [
        (arm, value) for arm, value in candidate_metrics.items()
        if value["governing_policy_recall"] >= 0.98 and value["blocking_policy_miss_rate"] <= 0.01
    ]
    pool = feasible_semantic or list(candidate_metrics.items())
    best_router = sorted(pool, key=lambda item: (item[1]["candidate_set_size_median"], -item[1]["governing_policy_recall"], item[1]["blocking_policy_miss_rate"], item[0]))[0][0]
    return {
        "schema_version": "K_F1F_ROUTER_FREEZE_V0_1", "selection_split": "CALIBRATION_ONLY",
        "selected_parameters": selected, "curves": curves, "top_k_curve": top_curves,
        "selected_top_k": selected_top, "best_semantic_router": best_router,
        "combined_calibration": _without_cases(combined),
    }


def relation_sample_mask(rows: list[dict[str, Any]], seed: int = 80412026) -> np.ndarray:
    selected_base: set[tuple[str, str]] = set()
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row["perturbation_kind"] == "BASE" and not int(row["governing"]):
            grouped[row["family_id"]].append(row)
        if row["perturbation_kind"] == "BASE" and int(row["governing"]):
            selected_base.add((row["base_action_id"], row["policy_id"]))
    for family in sorted(grouped):
        ordered = sorted(grouped[family], key=lambda row: (hashlib.sha256(f"{seed}|{row['row_id']}".encode()).hexdigest(), row["row_id"]))
        selected_base.update((row["base_action_id"], row["policy_id"]) for row in ordered[:10])
    mask = np.asarray([(row["base_action_id"], row["policy_id"]) in selected_base for row in rows], dtype=np.bool_)
    expected = 320 * len({row["family_id"] for row in rows})
    if int(mask.sum()) != expected:
        raise ValueError(f"RELATION_SAMPLE_CENSUS_MISMATCH:{int(mask.sum())}:{expected}")
    return mask


def _macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for label in range(len(RELATIONS)):
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return _mean(scores)


def _brier(truth: np.ndarray, probabilities: np.ndarray) -> float:
    target = np.eye(len(RELATIONS), dtype=np.float64)[truth]
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def fit_logistic(X: np.ndarray, truth: np.ndarray, C: float) -> dict[str, Any]:
    mean = X.mean(axis=0); scale = X.std(axis=0); scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale
    W = np.zeros((Z.shape[1], len(RELATIONS)), dtype=np.float64)
    counts = np.bincount(truth, minlength=len(RELATIONS)).astype(np.float64)
    b = np.log((counts + 1.0) / (counts.sum() + len(RELATIONS)))
    target = np.eye(len(RELATIONS), dtype=np.float64)[truth]
    converged = False; gradient_norm = math.inf; iterations = 0
    for iteration in range(2500):
        logits = Z @ W + b; logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
        residual = probabilities - target
        grad_w = Z.T @ residual / len(Z) + W / (C * len(Z))
        grad_b = residual.mean(axis=0)
        gradient_norm = float(math.sqrt(float(np.sum(grad_w * grad_w) + np.sum(grad_b * grad_b))))
        iterations = iteration + 1
        if gradient_norm < 1e-9:
            converged = True; break
        step = 0.05 / math.sqrt(1.0 + iteration / 250.0)
        W -= step * grad_w; b -= step * grad_b
    return {
        "features": None, "C": C, "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": W.tolist(), "intercept": b.tolist(), "iterations": iterations,
        "gradient_norm": gradient_norm, "converged": converged,
        "optimizer": "deterministic_full_batch_gradient_descent_v0.1",
    }


def predict_logistic(X: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64); scale = np.asarray(model["scale"], dtype=np.float64)
    W = np.asarray(model["weights"], dtype=np.float64); b = np.asarray(model["intercept"], dtype=np.float64)
    logits = ((X - mean) / scale) @ W + b; logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("MODEL_PROBABILITY_NONFINITE")
    return probabilities


def calibrate_relation_models(rows: list[dict[str, Any]], representations: list[dict[str, float]]) -> dict[str, Any]:
    if {row["split"] for row in rows} != {"CALIBRATION"}:
        raise ValueError("MODEL_CALIBRATION_SPLIT_MISMATCH")
    sample = relation_sample_mask(rows); truth = np.asarray([RELATION_INDEX[row["relation"]] for row, use in zip(rows, sample) if use], dtype=np.int64)
    arms = {}
    for arm, features in FEATURE_ARMS.items():
        X = np.asarray([[rep[name] for name in features] for rep, use in zip(representations, sample) if use], dtype=np.float64)
        candidates = []
        for C in (0.1, 1.0, 10.0):
            model = fit_logistic(X, truth, C); model["features"] = list(features)
            probabilities = predict_logistic(X, model); pred = probabilities.argmax(axis=1)
            candidates.append({"model": model, "calibration_macro_F1": _macro_f1(truth, pred), "calibration_Brier": _brier(truth, probabilities)})
        selected = sorted(candidates, key=lambda row: (-row["calibration_macro_F1"], row["calibration_Brier"], row["model"]["C"]))[0]
        arms[arm] = {"selected": selected, "candidate_summaries": [{key: value for key, value in row.items() if key != "model"} | {"C": row["model"]["C"]} for row in candidates]}
    best_singleton = sorted(SINGLETON_ARMS, key=lambda arm: (-arms[arm]["selected"]["calibration_macro_F1"], arms[arm]["selected"]["calibration_Brier"], arm))[0]
    return {
        "schema_version": "K_F1F_RELATION_MODEL_FREEZE_V0_1", "selection_split": "CALIBRATION_ONLY",
        "arms": arms, "best_singleton": best_singleton, "full_feature_arm": "COSINE_PLUS_EUCLIDEAN_PLUS_DOT_PRODUCT",
        "abstain_threshold": 0.70,
    }


def relation_predictions(representations: list[dict[str, float]], freeze: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for arm, features in FEATURE_ARMS.items():
        X = np.asarray([[rep[name] for name in features] for rep in representations], dtype=np.float64)
        result[arm] = predict_logistic(X, freeze["arms"][arm]["selected"]["model"]).astype("<f4")
    return result


def aggregate_relation_predictions(predictions: list[tuple[int, float]]) -> str:
    governing = [(RELATIONS[label], confidence) for label, confidence in predictions if RELATIONS[label] != "NON_GOVERNING"]
    if not governing:
        return "ESCALATE"
    # The accepted fixed precedence makes ambiguity/low confidence fail safe over
    # either ACT or HOLD.  Only a fully resolved opposed row may yield HOLD.
    if any(label == "AMBIGUOUS" or confidence < 0.70 for label, confidence in governing):
        return "ESCALATE"
    if any(label == "OPPOSED" for label, _confidence in governing):
        return "HOLD"
    return "ACT"


def aggregate_accepted_kf1e_predictions(predictions: list[tuple[int, float]]) -> str:
    """Exact accepted K-F1E action mapping, retained only for its comparator."""
    governing = [RELATIONS[label] for label, _confidence in predictions if RELATIONS[label] != "NON_GOVERNING"]
    if not governing:
        return "ACT"
    observed = set(governing)
    if "AMBIGUOUS" in observed or {"ALIGNED", "OPPOSED"}.issubset(observed):
        return "ESCALATE"
    if observed == {"OPPOSED"}:
        return "HOLD"
    return "ACT"


def _confusion(truth: list[str], pred: list[str], labels: tuple[str, ...]) -> dict[str, int]:
    return {f"{left}_to_{right}": sum(a == left and b == right for a, b in zip(truth, pred)) for left in labels for right in labels}


def _calibration_metrics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    confidence = probabilities.max(axis=1); predicted = probabilities.argmax(axis=1); correct = predicted == truth
    bins = []
    ece = 0.0
    for ordinal in range(10):
        low, high = ordinal / 10.0, (ordinal + 1) / 10.0
        mask = (confidence >= low) & ((confidence < high) if ordinal < 9 else (confidence <= high))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(truth) * abs(float(accuracy) - float(mean_confidence))
        bins.append({"low": low, "high": high, "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return {"ECE": ece, "Brier": _brier(truth, probabilities), "reliability_bins": bins, "abstention_rate": float(np.mean(confidence < 0.70))}


def exact_sign_flip(effects: list[float]) -> dict[str, Any]:
    if len(effects) != 12:
        raise ValueError(f"EXACT_TWELVE_FAMILY_EFFECTS_REQUIRED:{len(effects)}")
    observed = abs(_mean(effects)); extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=12):
        statistic = abs(_mean(value * sign for value, sign in zip(effects, signs)))
        extreme += int(statistic >= observed - 1e-15)
    return {"family_effects": effects, "mean_difference": _mean(effects), "two_sided_p": extreme / 4096.0, "permutations": 4096}


def cluster_interval(values: dict[str, float], seed_offset: int = 0) -> dict[str, float]:
    keys = sorted(values)
    rng = np.random.default_rng(17023 + seed_offset)
    samples = [float(np.mean([values[keys[index]] for index in rng.integers(0, len(keys), len(keys))])) for _ in range(2000)]
    return {"lower_95": _percentile(samples, 2.5), "upper_95": _percentile(samples, 97.5)}


def _relation_metrics(rows: list[dict[str, Any]], probabilities: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    indices = np.flatnonzero(mask); truth = np.asarray([RELATION_INDEX[rows[index]["relation"]] for index in indices], dtype=np.int64)
    selected = probabilities[indices]; pred = selected.argmax(axis=1)
    truth_text = [RELATIONS[index] for index in truth]; pred_text = [RELATIONS[index] for index in pred]
    return {
        "rows": len(indices), "macro_F1_relation": _macro_f1(truth, pred),
        "confusion": _confusion(truth_text, pred_text, RELATIONS),
        "calibration": _calibration_metrics(truth, selected),
    }


def analyze_relation(rows: list[dict[str, Any]], predictions: dict[str, np.ndarray], freeze: dict[str, Any]) -> dict[str, Any]:
    sample = relation_sample_mask(rows); overall = {}; by_family = {}; action_overall = {}; action_by_family = {}
    families = sorted({row["family_id"] for row in rows})
    for arm in FEATURE_ARMS:
        overall[arm] = _relation_metrics(rows, predictions[arm], sample)
        action_overall[arm] = _action_metrics(rows, sample, predictions[arm])
        by_family[arm] = {}
        action_by_family[arm] = {}
        for family in families:
            family_mask = sample & np.asarray([row["family_id"] == family for row in rows], dtype=np.bool_)
            by_family[arm][family] = _relation_metrics(rows, predictions[arm], family_mask)["macro_F1_relation"]
            family_rows_mask = np.asarray([row["family_id"] == family for row in rows], dtype=np.bool_)
            family_rows = [row for row, use in zip(rows, family_rows_mask) if use]
            action_by_family[arm][family] = _without_cases(
                _action_metrics(family_rows, sample[family_rows_mask], predictions[arm][family_rows_mask])
            )
    full = freeze["full_feature_arm"]; best = freeze["best_singleton"]
    effects = [by_family[full][family] - by_family[best][family] for family in families]
    inference = exact_sign_flip(effects)
    inference["descriptive_cluster_bootstrap"] = cluster_interval({family: effect for family, effect in zip(families, effects)})
    comparisons = {}
    raw_p = []
    for arm in FEATURE_ARMS:
        if arm == full:
            continue
        test = exact_sign_flip([by_family[full][family] - by_family[arm][family] for family in families])
        comparisons[arm] = test; raw_p.append((arm, test["two_sided_p"]))
    running = 0.0
    for rank, (arm, p_value) in enumerate(sorted(raw_p, key=lambda row: (row[1], row[0]))):
        adjusted = min(1.0, p_value * (len(raw_p) - rank)); running = max(running, adjusted)
        comparisons[arm]["Holm_adjusted_p"] = running
    inference["Holm_adjusted_p"] = comparisons[best]["Holm_adjusted_p"]
    full_false_act = {family: value["false_ACT_rate"] for family, value in action_by_family[full].items()}
    full_blocking_miss = {family: value["blocking_policy_miss_rate"] for family, value in action_by_family[full].items()}
    safety = {
        "false_ACT_cluster_upper_95": cluster_interval(full_false_act, 400)["upper_95"],
        "blocking_miss_cluster_upper_95": cluster_interval(full_blocking_miss, 500)["upper_95"],
    }
    terminal = "B_INCREMENTAL_MEASUREMENT_UTILITY" if (
        inference["mean_difference"] >= 0.03
        and inference["Holm_adjusted_p"] < 0.05
        and inference["descriptive_cluster_bootstrap"]["lower_95"] > 0.0
        and safety["false_ACT_cluster_upper_95"] <= 0.02
        and safety["blocking_miss_cluster_upper_95"] <= 0.02
    ) else "B_NO_INCREMENTAL_MEASUREMENT_UTILITY"
    breakdowns: dict[str, Any] = {}
    for field in ("policy_set_size", "cardinality", "perturbation_kind", "relation"):
        breakdowns[field] = {}
        for value in sorted({row[field] for row in rows}):
            field_mask = sample & np.asarray([row[field] == value for row in rows], dtype=np.bool_)
            breakdowns[field][value] = {arm: _relation_metrics(rows, predictions[arm], field_mask) for arm in FEATURE_ARMS}
    return {
        "pair_level_overall": overall,
        "pair_level_by_family_macro_F1": by_family,
        "action_level_overall": action_overall,
        "action_level_by_family": action_by_family,
        "breakdowns": breakdowns,
        "full_vs_best_singleton": inference,
        "full_vs_other_arms": comparisons,
        "safety": safety,
        "terminal": terminal,
    }


def analyze_router(rows: list[dict[str, Any]], representations: list[dict[str, float]], masks: dict[str, np.ndarray], freeze: dict[str, Any]) -> dict[str, Any]:
    arms = {arm: _without_cases(_routing_metrics(rows, mask)) for arm, mask in masks.items()}
    by_family = {}
    for arm, mask in masks.items():
        by_family[arm] = {}
        for family in sorted({row["family_id"] for row in rows}):
            family_mask = np.asarray([row["family_id"] == family for row in rows], dtype=np.bool_)
            subset_rows = [row for row, use in zip(rows, family_mask) if use]
            by_family[arm][family] = _without_cases(_routing_metrics(subset_rows, mask[family_mask]))
    budget_match = {}
    for arm in THRESHOLD_ARMS:
        source_cases = _routing_metrics(rows, masks[arm])["case_rows"]
        budget_match[arm] = {}
        for size in ("SMALL", "MEDIUM", "LARGE"):
            counts = [row["selected"] for row in source_cases if row["policy_set_size"] == size]
            k = max(1, int(math.floor(_percentile(counts, 50) + 0.5)))
            size_mask = np.asarray([row["policy_set_size"] == size for row in rows], dtype=np.bool_)
            top_mask = _candidate_mask(rows, representations, "CONVENTIONAL_TOP_K", k)
            top_metrics = _without_cases(_routing_metrics([row for row, use in zip(rows, size_mask) if use], top_mask[size_mask]))
            threshold_metrics = _without_cases(_routing_metrics([row for row, use in zip(rows, size_mask) if use], masks[arm][size_mask]))
            budget_match[arm][size] = {"threshold_median_candidate_count": _percentile(counts, 50), "deterministic_rounded_k": k, "threshold": threshold_metrics, "top_k_exact_budget": top_metrics}
    best = freeze["best_semantic_router"]; primary = arms[best]
    terminal = "A_ROUTING_UTILITY" if primary["governing_policy_recall"] >= 0.98 and primary["blocking_policy_miss_rate"] <= 0.01 and primary["policy_evaluation_reduction_median"] >= 0.20 else "A_ROUTING_NO_UTILITY"
    breakdowns: dict[str, Any] = {}
    for field in ("policy_set_size", "cardinality", "perturbation_kind"):
        breakdowns[field] = {}
        for value in sorted({row[field] for row in rows}):
            field_mask = np.asarray([row[field] == value for row in rows], dtype=np.bool_)
            subset = [row for row, use in zip(rows, field_mask) if use]
            breakdowns[field][value] = {arm: _without_cases(_routing_metrics(subset, mask[field_mask])) for arm, mask in masks.items()}
    return {"arms": arms, "by_family": by_family, "breakdowns": breakdowns, "budget_matched_top_k": budget_match, "best_semantic_router": best, "terminal": terminal}


def _action_metrics(
    rows: list[dict[str, Any]], mask: np.ndarray, probabilities: np.ndarray,
    legacy: bool = False, margin: float = 0.0, truth_oracle: bool = False,
) -> dict[str, Any]:
    outcomes = []
    for indices in _groups(rows):
        head = rows[indices[0]]
        selected = [index for index in indices if bool(mask[index])]
        if legacy:
            relation_predictions_for_action = []
            for index in selected:
                value = float(probabilities[index, 0])
                label = RELATION_INDEX["ALIGNED"] if value > margin else RELATION_INDEX["OPPOSED"] if value < -margin else RELATION_INDEX["AMBIGUOUS"]
                relation_predictions_for_action.append((label, 1.0))
        else:
            relation_predictions_for_action = [(int(probabilities[index].argmax()), float(probabilities[index].max())) for index in selected]
        predicted = head["expected_disposition"] if truth_oracle else (
            aggregate_accepted_kf1e_predictions(relation_predictions_for_action)
            if legacy else aggregate_relation_predictions(relation_predictions_for_action)
        )
        governing = {index for index in indices if int(rows[index]["governing"])}; blocking = {index for index in indices if int(rows[index]["blocking"])}
        selected_set = set(selected)
        outcomes.append({
            "blind_action_id": head["blind_action_id"], "family_id": head["family_id"],
            "expected": head["expected_disposition"], "predicted": predicted,
            "policy_count": len(indices), "selected": len(selected), "governing": len(governing),
            "recovered": len(governing & selected_set), "blocking": len(blocking), "blocking_missed": len(blocking - selected_set),
            "perturbation_kind": head["perturbation_kind"], "policy_set_size": head["policy_set_size"], "cardinality": head["cardinality"],
            "not_estimable": len(governing) == 0,
        })
    truth = [row["expected"] for row in outcomes]; pred = [row["predicted"] for row in outcomes]
    non_act = [row for row in outcomes if row["expected"] != "ACT"]
    non_hold = [row for row in outcomes if row["expected"] != "HOLD"]
    governing = sum(row["governing"] for row in outcomes); blocking = sum(row["blocking"] for row in outcomes)
    selected = [row["selected"] for row in outcomes]; reductions = [1.0 - row["selected"] / row["policy_count"] for row in outcomes]
    return {
        "actions": len(outcomes), "final_disposition_accuracy": _mean(a == b for a, b in zip(truth, pred)),
        "confusion": _confusion(truth, pred, ("ACT", "HOLD", "ESCALATE")),
        "false_ACT_rate": _mean(row["predicted"] == "ACT" for row in non_act),
        "false_HOLD_rate": _mean(row["predicted"] == "HOLD" for row in non_hold),
        "abstention_or_escalation_rate": _mean(row["predicted"] == "ESCALATE" for row in outcomes),
        "governing_policy_recall": sum(row["recovered"] for row in outcomes) / governing if governing else 1.0,
        "blocking_policy_miss_rate": sum(row["blocking_missed"] for row in outcomes) / blocking if blocking else 0.0,
        "policies_evaluated_per_action_median": _percentile(selected, 50),
        "policy_evaluation_reduction_median": _percentile(reductions, 50), "case_rows": outcomes,
    }


def _accepted_kf1e_threshold_candidates(representations: list[dict[str, float]]) -> list[float]:
    values = np.asarray(sorted({float(row["euclidean_distance"]) for row in representations}), dtype=np.float64)
    if not len(values):
        raise ValueError("NO_KF1E_EUCLIDEAN_CALIBRATION_VALUES")
    positions = np.unique(np.linspace(0, len(values) - 1, 33).round().astype(int))
    return sorted(set([float(values[index]) for index in positions] + [float(np.nextafter(values[-1], np.inf))]))


def _accepted_kf1e_mask(
    rows: list[dict[str, Any]], representations: list[dict[str, float]], threshold: float, budget: int,
) -> np.ndarray:
    mask = np.zeros(len(rows), dtype=np.bool_)
    for indices in _groups(rows):
        ordered = sorted(indices, key=lambda index: (float(representations[index]["euclidean_distance"]), rows[index]["blind_policy_id"]))
        eligible = [index for index in ordered if float(representations[index]["euclidean_distance"]) <= threshold]
        mask[eligible[:budget]] = True
    return mask


def calibrate_accepted_kf1e_serial(rows: list[dict[str, Any]], representations: list[dict[str, float]]) -> dict[str, Any]:
    if {row["split"] for row in rows} != {"CALIBRATION"}:
        raise ValueError("KF1E_BASELINE_CALIBRATION_SPLIT_MISMATCH")
    direction = np.zeros((len(rows), len(RELATIONS)), dtype=np.float64)
    direction[:, 0] = [row["legacy_direction_margin"] for row in representations]
    curve = []
    for budget in (1, 2, 3, 4):
        for threshold in _accepted_kf1e_threshold_candidates(representations):
            mask = _accepted_kf1e_mask(rows, representations, threshold, budget)
            for margin in (0.0, 0.01, 0.02, 0.05, 0.10):
                metrics = _without_cases(_action_metrics(rows, mask, direction, legacy=True, margin=margin))
                feasible = (
                    metrics["governing_policy_recall"] >= 0.95
                    and metrics["blocking_policy_miss_rate"] == 0.0
                    and metrics["false_ACT_rate"] == 0.0
                )
                curve.append({
                    "budget": budget,
                    "euclidean_threshold": threshold,
                    "direction_margin": margin,
                    "meets_accepted_K_F1E_safety_targets": feasible,
                    **metrics,
                })
    feasible = [row for row in curve if row["meets_accepted_K_F1E_safety_targets"]]
    if feasible:
        selected = sorted(feasible, key=lambda row: (
            row["budget"], -row["final_disposition_accuracy"], -row["governing_policy_recall"],
            row["euclidean_threshold"], row["direction_margin"],
        ))[0]
        status = "CALIBRATION_TARGETS_MET"
    else:
        selected = sorted(curve, key=lambda row: (
            row["false_ACT_rate"], row["blocking_policy_miss_rate"], -row["governing_policy_recall"],
            -row["final_disposition_accuracy"], row["budget"], row["euclidean_threshold"], row["direction_margin"],
        ))[0]
        status = "CALIBRATION_TARGETS_NOT_MET_FAIL_CLOSED_SELECTION"
    return {
        "schema_version": "K_F1F_ACCEPTED_K_F1E_SERIAL_FREEZE_V0_1",
        "source_split": "CALIBRATION_ONLY",
        "selection_status": status,
        "selected": selected,
        "candidate_count": len(curve),
        "candidate_curve": curve,
        "exact_predecessor_semantics": {
            "budgets": [1, 2, 3, 4],
            "direction_margins": [0.0, 0.01, 0.02, 0.05, 0.10],
            "zero_route": "ACT",
            "aligned_plus_opposed": "ESCALATE",
        },
    }


def analyze_end_to_end(
    rows: list[dict[str, Any]], representations: list[dict[str, float]], masks: dict[str, np.ndarray],
    relation: dict[str, np.ndarray], router_freeze: dict[str, Any], relation_freeze: dict[str, Any], legacy_freeze: dict[str, Any],
) -> dict[str, Any]:
    full = relation_freeze["full_feature_arm"]; best_router = router_freeze["best_semantic_router"]
    comparators: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {
        "COMPOSED_CANDIDATE": (masks[best_router], relation[full], False),
        "EXHAUSTIVE_SEMANTIC_NO_ROUTER": (masks["EXHAUSTIVE_ORACLE_EVALUATION"], relation[full], False),
        "TOP_K_PLUS_SAME_RELATION_MODEL": (masks["CONVENTIONAL_TOP_K"], relation[full], False),
        "COSINE_ISOLATED": (masks["COSINE_THRESHOLD_ROUTING"], relation["COSINE_ONLY"], False),
        "EUCLIDEAN_ISOLATED": (masks["EUCLIDEAN_THRESHOLD_ROUTING"], relation["EUCLIDEAN_ONLY"], False),
        "SIGNED_DOT_ISOLATED": (masks["SIGNED_DOT_THRESHOLD_ROUTING"], relation["SIGNED_DOT_PRODUCT_ONLY"], False),
    }
    legacy_probs = np.zeros((len(rows), 4), dtype=np.float64); legacy_probs[:, 0] = [row["legacy_direction_margin"] for row in representations]
    legacy_selected = legacy_freeze["selected"]
    legacy_mask = _accepted_kf1e_mask(
        rows, representations, float(legacy_selected["euclidean_threshold"]), int(legacy_selected["budget"])
    )
    legacy_margin = float(legacy_selected["direction_margin"])
    comparators["ACCEPTED_K_F1E_SERIAL_BASELINE"] = (legacy_mask, legacy_probs, True)
    overall = {}; by_family = {}; action_outputs = {}
    for name, (mask, probabilities, legacy) in comparators.items():
        metrics = _action_metrics(rows, mask, probabilities, legacy=legacy, margin=legacy_margin)
        action_outputs[name] = metrics["case_rows"]
        overall[name] = _without_cases(metrics); by_family[name] = {}
        for family in sorted({row["family_id"] for row in rows}):
            family_mask = np.asarray([row["family_id"] == family for row in rows], dtype=np.bool_)
            sub_rows = [row for row, use in zip(rows, family_mask) if use]
            by_family[name][family] = _without_cases(_action_metrics(sub_rows, mask[family_mask], probabilities[family_mask], legacy=legacy, margin=legacy_margin))
    oracle_probabilities = np.eye(len(RELATIONS), dtype=np.float64)[
        np.asarray([RELATION_INDEX[row["relation"]] for row in rows], dtype=np.int64)
    ]
    oracle = _action_metrics(rows, np.ones(len(rows), dtype=np.bool_), oracle_probabilities, truth_oracle=True)
    if oracle["final_disposition_accuracy"] != 1.0 or oracle["false_ACT_rate"] != 0.0 or oracle["false_HOLD_rate"] != 0.0:
        raise ValueError("DETERMINISTIC_ORACLE_CEILING_MISMATCH")
    overall["EXHAUSTIVE_DETERMINISTIC_ORACLE"] = _without_cases(oracle)
    action_outputs["EXHAUSTIVE_DETERMINISTIC_ORACLE"] = oracle["case_rows"]
    families = sorted(by_family["COMPOSED_CANDIDATE"])
    direct_effects = [by_family["COMPOSED_CANDIDATE"][family]["final_disposition_accuracy"] - by_family["EXHAUSTIVE_SEMANTIC_NO_ROUTER"][family]["final_disposition_accuracy"] for family in families]
    direct_inference = exact_sign_flip(direct_effects); direct_inference["descriptive_cluster_bootstrap"] = cluster_interval({family: effect for family, effect in zip(families, direct_effects)}, 100)
    serial_effects = [by_family["COMPOSED_CANDIDATE"][family]["final_disposition_accuracy"] - by_family["ACCEPTED_K_F1E_SERIAL_BASELINE"][family]["final_disposition_accuracy"] for family in families]
    serial_inference = exact_sign_flip(serial_effects); serial_inference["descriptive_cluster_bootstrap"] = cluster_interval({family: effect for family, effect in zip(families, serial_effects)}, 101)
    composed = overall["COMPOSED_CANDIDATE"]
    family_false_act = {family: value["false_ACT_rate"] for family, value in by_family["COMPOSED_CANDIDATE"].items()}
    family_block = {family: value["blocking_policy_miss_rate"] for family, value in by_family["COMPOSED_CANDIDATE"].items()}
    safety = {"false_ACT_cluster_upper_95": cluster_interval(family_false_act, 200)["upper_95"], "blocking_miss_cluster_upper_95": cluster_interval(family_block, 300)["upper_95"]}
    terminal = "C_END_TO_END_UTILITY" if composed["final_disposition_accuracy"] >= 0.70 and direct_inference["mean_difference"] >= 0.05 and direct_inference["two_sided_p"] < 0.05 and direct_inference["descriptive_cluster_bootstrap"]["lower_95"] > 0.0 and safety["false_ACT_cluster_upper_95"] <= 0.02 and safety["blocking_miss_cluster_upper_95"] <= 0.02 and composed["policy_evaluation_reduction_median"] >= 0.20 else "C_NO_END_TO_END_UTILITY"
    breakdowns: dict[str, Any] = {}
    for field in ("policy_set_size", "cardinality", "perturbation_kind"):
        breakdowns[field] = {}
        for value in sorted({row[field] for row in rows}):
            field_mask = np.asarray([row[field] == value for row in rows], dtype=np.bool_)
            subset = [row for row, use in zip(rows, field_mask) if use]
            breakdowns[field][value] = {
                name: _without_cases(_action_metrics(subset, mask[field_mask], probabilities[field_mask], legacy=legacy, margin=legacy_margin))
                for name, (mask, probabilities, legacy) in comparators.items()
            }
    return {
        "overall": overall,
        "action_outputs": action_outputs,
        "by_family": by_family,
        "breakdowns": breakdowns,
        "composed_vs_direct_semantic_baseline": direct_inference,
        "composed_vs_accepted_K_F1E_serial_baseline": serial_inference,
        "safety": safety,
        "accepted_K_F1E_serial_freeze": legacy_freeze,
        "terminal": terminal,
    }


def describe_validation(
    rows: list[dict[str, Any]], representations: list[dict[str, float]], masks: dict[str, np.ndarray],
    relation: dict[str, np.ndarray], router_freeze: dict[str, Any], relation_freeze: dict[str, Any], legacy_freeze: dict[str, Any],
) -> dict[str, Any]:
    """Emit the frozen validation checks without selection or inferential reuse."""
    sample = relation_sample_mask(rows)
    routing = {arm: _without_cases(_routing_metrics(rows, mask)) for arm, mask in masks.items()}
    relation_pair = {arm: _relation_metrics(rows, probabilities, sample) for arm, probabilities in relation.items()}
    relation_action = {arm: _without_cases(_action_metrics(rows, sample, probabilities)) for arm, probabilities in relation.items()}
    full = relation_freeze["full_feature_arm"]
    best_router = router_freeze["best_semantic_router"]
    legacy_probabilities = np.zeros((len(rows), len(RELATIONS)), dtype=np.float64)
    legacy_probabilities[:, 0] = [row["legacy_direction_margin"] for row in representations]
    legacy_selected = legacy_freeze["selected"]
    legacy_mask = _accepted_kf1e_mask(
        rows, representations, float(legacy_selected["euclidean_threshold"]), int(legacy_selected["budget"])
    )
    legacy_margin = float(legacy_selected["direction_margin"])
    end_to_end = {
        "COMPOSED_CANDIDATE": _without_cases(_action_metrics(rows, masks[best_router], relation[full])),
        "DIRECT_SEMANTIC_DISPOSITION_BASELINE": _without_cases(_action_metrics(rows, masks["EXHAUSTIVE_ORACLE_EVALUATION"], relation[full])),
        "ACCEPTED_K_F1E_SERIAL_BASELINE": _without_cases(_action_metrics(rows, legacy_mask, legacy_probabilities, legacy=True, margin=legacy_margin)),
    }
    return {
        "selection_use": "DESCRIPTIVE_ONLY_NO_MODEL_OR_THRESHOLD_CHANGE",
        "pairs": len(rows),
        "actions": len({row["blind_action_id"] for row in rows}),
        "relation_sample_rows": int(sample.sum()),
        "routing": routing,
        "relation_pair_level": relation_pair,
        "relation_action_level": relation_action,
        "end_to_end": end_to_end,
    }


def conclusion_matrix(A: dict[str, Any], B: dict[str, Any], C: dict[str, Any]) -> dict[str, Any]:
    terminals = {"A_terminal": A["terminal"], "B_terminal": B["terminal"], "C_terminal": C["terminal"]}
    if B["safety"]["false_ACT_cluster_upper_95"] > 0.02 or B["safety"]["blocking_miss_cluster_upper_95"] > 0.02 or C["safety"]["false_ACT_cluster_upper_95"] > 0.02 or C["safety"]["blocking_miss_cluster_upper_95"] > 0.02:
        overall = "SAFETY_FAIL"
    elif A["terminal"] != "A_ROUTING_UTILITY":
        overall = "A_ROUTING_NO_UTILITY"
    elif B["terminal"] != "B_INCREMENTAL_MEASUREMENT_UTILITY":
        overall = "B_NO_INCREMENTAL_MEASUREMENT_UTILITY"
    elif C["terminal"] != "C_END_TO_END_UTILITY":
        overall = "C_NO_END_TO_END_UTILITY"
    else:
        overall = "UTILITY_SUPPORTED_FOR_FUNDING_SCREEN"
    return terminals | {"overall_funding_screen_disposition": overall, "claim_ceiling": "RIMS-2026-008_SYSTEM_K_FUNDING_SCREEN_ONLY"}
