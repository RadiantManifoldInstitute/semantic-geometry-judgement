from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .reader import CLASS_ORDER


APPLICABLE = frozenset(CLASS_ORDER[:2])
ALIGNED = frozenset((CLASS_ORDER[0], CLASS_ORDER[2]))


def recall(gold: list[str], predicted: list[str], label: str) -> float:
    indices = [index for index, value in enumerate(gold) if value == label]
    if not indices:
        raise ValueError(f"RECALL_LABEL_ABSENT:{label}")
    return sum(predicted[index] == label for index in indices) / len(indices)


def balanced_accuracy(gold: list[str], predicted: list[str], labels: Iterable[str]) -> float:
    labels = tuple(labels)
    if len(gold) != len(predicted) or not gold:
        raise ValueError("BALANCED_ACCURACY_INPUT_INVALID")
    return float(np.mean([recall(gold, predicted, label) for label in labels]))


def binary_balanced_accuracy(gold: list[int], predicted: list[int]) -> float:
    return balanced_accuracy([str(v) for v in gold], [str(v) for v in predicted], ("0", "1"))


def auroc(gold: list[int], probability: list[float]) -> float:
    if len(gold) != len(probability) or set(gold) != {0, 1}:
        raise ValueError("AUROC_INPUT_INVALID")
    if any(not math.isfinite(value) for value in probability):
        raise ValueError("AUROC_NONFINITE")
    order = sorted(range(len(gold)), key=lambda i: (probability[i], i))
    ranks = [0.0] * len(gold)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and probability[order[end]] == probability[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    positives = [i for i, value in enumerate(gold) if value == 1]
    negatives = len(gold) - len(positives)
    return (sum(ranks[i] for i in positives) - len(positives) * (len(positives) + 1) / 2) / (len(positives) * negatives)


def brier_multiclass(gold: list[str], probabilities: np.ndarray) -> float:
    expected = np.zeros_like(probabilities, dtype=np.float64)
    for index, label in enumerate(gold):
        expected[index, CLASS_ORDER.index(label)] = 1.0
    return float(np.mean(np.sum((probabilities - expected) ** 2, axis=1)))


def classwise_ece(gold: list[str], probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for class_index, label in enumerate(CLASS_ORDER):
        class_rows = []
        total = 0.0
        for bin_index in range(bins):
            lower = bin_index / bins
            upper = (bin_index + 1) / bins
            indices = [
                i
                for i, value in enumerate(probabilities[:, class_index])
                if value >= lower and (value < upper or (bin_index == bins - 1 and value <= upper))
            ]
            if indices:
                confidence = float(np.mean(probabilities[indices, class_index]))
                observed = float(np.mean([gold[i] == label for i in indices]))
                total += len(indices) / len(gold) * abs(confidence - observed)
            else:
                confidence = observed = None
            class_rows.append(
                {
                    "bin": bin_index,
                    "lower_inclusive": lower,
                    "upper": upper,
                    "count": len(indices),
                    "mean_probability": confidence,
                    "observed_frequency": observed,
                }
            )
        rows[label] = {"ece": total, "reliability_table": class_rows}
    return rows


def rank_correlation(left: list[float], right: list[float]) -> float:
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


def family_values(
    rows: list[dict[str, Any]],
    predicted: list[str],
    metric: str,
) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["family_id"]].append(index)
    values: dict[str, float] = {}
    for family, indices in sorted(groups.items()):
        subset = [rows[i] for i in indices]
        pred = [predicted[i] for i in indices]
        gold = [row["gold_cell"] for row in subset]
        if metric == "four_cell_balanced_accuracy":
            value = balanced_accuracy(gold, pred, CLASS_ORDER)
        elif metric == "applicability_balanced_accuracy":
            value = binary_balanced_accuracy(
                [int(value in APPLICABLE) for value in gold],
                [int(value in APPLICABLE) for value in pred],
            )
        elif metric == "conditional_polarity_balanced_accuracy":
            applicable_indices = [i for i, value in enumerate(gold) if value in APPLICABLE]
            value = binary_balanced_accuracy(
                [int(gold[i] == CLASS_ORDER[1]) for i in applicable_indices],
                [int(pred[i] == CLASS_ORDER[1]) for i in applicable_indices],
            )
        elif metric in CLASS_ORDER:
            value = recall(gold, pred, metric)
        elif metric == "wrong_mission_false_applicability_rate":
            target = [i for i, row in enumerate(subset) if row.get("wrong_mission", False)]
            if not target:
                target = [i for i, value in enumerate(gold) if value not in APPLICABLE]
            value = sum(pred[i] in APPLICABLE for i in target) / len(target)
        else:
            raise ValueError(f"UNKNOWN_FAMILY_METRIC:{metric}")
        values[family] = float(value)
    return values


def family_summary(values: dict[str, float], seed: int = 41382001, replicates: int = 10000) -> dict[str, Any]:
    if len(values) != 8:
        raise ValueError(f"EXACT_EIGHT_HOLDOUT_FAMILIES_REQUIRED:{len(values)}")
    ordered = [values[key] for key in sorted(values)]
    leave_one_out = {
        family: float(np.mean([value for key, value in values.items() if key != family]))
        for family in sorted(values)
    }
    random = np.random.default_rng(seed)
    indices = random.integers(0, len(ordered), size=(replicates, len(ordered)))
    bootstrap = np.mean(np.asarray(ordered, dtype=np.float64)[indices], axis=1)
    return {
        "family_values": {key: values[key] for key in sorted(values)},
        "sorted_values": sorted(ordered),
        "equal_weight_mean": float(np.mean(ordered)),
        "median": float(np.median(ordered)),
        "range": [float(np.min(ordered)), float(np.max(ordered))],
        "leave_one_family_out_equal_weight_means": leave_one_out,
        "descriptive_family_bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "percentile_95_interval": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
            "hard_gate_use": False,
        },
    }
