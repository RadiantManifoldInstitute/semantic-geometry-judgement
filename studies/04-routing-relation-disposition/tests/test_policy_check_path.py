from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kf1f.analysis import (  # noqa: E402
    RELATION_INDEX, _action_metrics, _candidate_mask, aggregate_accepted_kf1e_predictions,
    aggregate_relation_predictions, exact_sign_flip, fit_logistic,
    predict_logistic, relation_sample_mask,
)
from kf1f.representations import derive  # noqa: E402


def materializer_module():
    path = ROOT / "materializer/path-a/construct.py"
    spec = importlib.util.spec_from_file_location("kf1f_construct", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class MeasurementSpeciesTests(unittest.TestCase):
    def test_raw_species_preserve_magnitude_and_unit_control_collapses(self) -> None:
        row = derive(np.asarray([2.0, 0.0]), np.asarray([1.0, 1.0]))
        self.assertAlmostEqual(row["cosine"], 2 ** -0.5)
        self.assertNotAlmostEqual(row["euclidean_distance"], row["unit_euclidean_distance"])
        self.assertNotAlmostEqual(row["signed_dot_product"], row["unit_signed_dot_product"])
        self.assertLessEqual(row["unit_collapse_residual"], 1e-10)

    def test_zero_vector_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "VECTOR_ZERO_NORM"):
            derive(np.zeros(2), np.ones(2))


class DenominatorTests(unittest.TestCase):
    def test_binding_denominators_reconcile(self) -> None:
        contract = json.loads((ROOT / "spec/generation-contract.json").read_text())
        self.assertEqual(8 * 8 + 8 * 32 + 8 * 128, 1344)
        self.assertEqual(30 * 1344, contract["counts"]["base_pairs"])
        self.assertEqual(30 * 24 * 8, contract["counts"]["actions"])
        self.assertEqual(30 * 1344 * 8, contract["counts"]["pairs"])
        self.assertEqual(sum(contract["base_per_family"]["governing_relation_pairs"].values()), 30)
        self.assertEqual(sum(contract["base_per_family"]["action_dispositions"].values()), 24)
        self.assertEqual(contract["base_per_family"]["governing_pairs_by_cardinality"], {"ZERO": 0, "ONE": 6, "MULTIPLE_COMPATIBLE": 12, "MULTIPLE_CONFLICTING": 12})

    def test_exact_relation_pattern(self) -> None:
        module = materializer_module()
        relations = []
        dispositions = []
        for size in range(3):
            for cardinality in module.CARDINALITIES:
                pattern = module.relation_pattern(size, cardinality)
                for variant in range(2):
                    relations.extend(pattern)
                    dispositions.append(module.disposition(pattern, cardinality == "ZERO" and variant == 0))
        self.assertEqual({name: relations.count(name) for name in ("ALIGNED", "OPPOSED", "AMBIGUOUS")}, {"ALIGNED": 12, "OPPOSED": 10, "AMBIGUOUS": 8})
        self.assertEqual({name: dispositions.count(name) for name in ("ACT", "HOLD", "ESCALATE")}, {"ACT": 7, "HOLD": 8, "ESCALATE": 9})

    def test_explicit_negation_changes_the_action_surface_and_oracle(self) -> None:
        module = materializer_module()
        blueprint = json.loads((ROOT / "spec/family-blueprint.json").read_text())
        family = blueprint["families"][0]
        policies = module.policy_rows(family, blueprint["abstraction_phrases"])
        base = module.action_text(family, policies[:1], ["ALIGNED"], False)
        negated = "Explicitly negated form: " + module.action_text(family, policies[:1], module.flipped(["ALIGNED"]), False)
        self.assertIn(family["aligned"], base)
        self.assertIn(family["opposed"], negated)
        self.assertNotEqual(base, negated)


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"blind_action_id": "a", "blind_policy_id": f"p{i}", "action_text": "protect pressure boundary", "policy_route_terms": ["pressure", "boundary"]}
            for i in range(4)
        ]
        self.representations = [
            {"cosine": value, "euclidean_distance": 1.0 - value, "signed_dot_product": value * 2.0}
            for value in (0.9, 0.8, 0.4, 0.1)
        ]

    def test_isolated_thresholds_and_top_k(self) -> None:
        self.assertEqual(_candidate_mask(self.rows, self.representations, "COSINE_THRESHOLD_ROUTING", 0.5).tolist(), [True, True, False, False])
        self.assertEqual(_candidate_mask(self.rows, self.representations, "EUCLIDEAN_THRESHOLD_ROUTING", 0.3).tolist(), [True, True, False, False])
        self.assertEqual(int(_candidate_mask(self.rows, self.representations, "CONVENTIONAL_TOP_K", 1).sum()), 1)

    def test_combined_is_union_not_replacement(self) -> None:
        parameters = {"COSINE_THRESHOLD_ROUTING": 0.95, "EUCLIDEAN_THRESHOLD_ROUTING": 0.3, "SIGNED_DOT_THRESHOLD_ROUTING": 1.9, "LEXICAL_OVERLAP_ROUTING": 1.0}
        combined = _candidate_mask(self.rows, self.representations, "COMBINED_SEMANTIC_ROUTING", parameters)
        self.assertEqual(combined.tolist(), [True, True, True, True])


class RelationModelTests(unittest.TestCase):
    def test_multinomial_model_is_deterministic(self) -> None:
        X = np.asarray([[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0], [0.0], [0.1]], dtype=np.float64)
        truth = np.asarray([0, 0, 0, 1, 1, 1, 2, 3], dtype=np.int64)
        left = fit_logistic(X, truth, 1.0); right = fit_logistic(X, truth, 1.0)
        self.assertEqual(left, right)
        probabilities = predict_logistic(X, left)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(len(X)))

    def test_action_aggregation_precedence(self) -> None:
        self.assertEqual(aggregate_relation_predictions([]), "ESCALATE")
        self.assertEqual(aggregate_relation_predictions([(0, 0.99)]), "ACT")
        self.assertEqual(aggregate_relation_predictions([(0, 0.99), (1, 0.99)]), "HOLD")
        self.assertEqual(aggregate_relation_predictions([(1, 0.99), (2, 0.99)]), "ESCALATE")
        self.assertEqual(aggregate_relation_predictions([(1, 0.60)]), "ESCALATE")
        self.assertEqual(aggregate_relation_predictions([(0, 0.60)]), "ESCALATE")
        self.assertEqual(aggregate_relation_predictions([(3, 0.99)]), "ESCALATE")

    def test_accepted_kf1e_comparator_keeps_its_original_mapping(self) -> None:
        self.assertEqual(aggregate_accepted_kf1e_predictions([]), "ACT")
        self.assertEqual(aggregate_accepted_kf1e_predictions([(0, 1.0), (1, 1.0)]), "ESCALATE")
        self.assertEqual(aggregate_accepted_kf1e_predictions([(1, 1.0)]), "HOLD")

    def test_exact_sign_flip_has_4096_assignments(self) -> None:
        result = exact_sign_flip([0.05] * 12)
        self.assertEqual(result["permutations"], 4096)
        self.assertLessEqual(result["two_sided_p"], 0.001)

    def test_truth_one_hot_is_an_exact_action_oracle(self) -> None:
        rows = [
            {"blind_action_id": "a1", "family_id": "F19", "expected_disposition": "HOLD", "governing": 1, "blocking": 1, "relation": "OPPOSED", "perturbation_kind": "BASE", "policy_set_size": "SMALL", "cardinality": "ONE"},
            {"blind_action_id": "a1", "family_id": "F19", "expected_disposition": "HOLD", "governing": 0, "blocking": 0, "relation": "NON_GOVERNING", "perturbation_kind": "BASE", "policy_set_size": "SMALL", "cardinality": "ONE"},
            {"blind_action_id": "a2", "family_id": "F19", "expected_disposition": "ESCALATE", "governing": 0, "blocking": 0, "relation": "NON_GOVERNING", "perturbation_kind": "BASE", "policy_set_size": "SMALL", "cardinality": "ZERO"},
            {"blind_action_id": "a3", "family_id": "F19", "expected_disposition": "ACT", "governing": 0, "blocking": 0, "relation": "NON_GOVERNING", "perturbation_kind": "BASE", "policy_set_size": "SMALL", "cardinality": "ZERO"},
        ]
        truth = np.asarray([RELATION_INDEX[row["relation"]] for row in rows], dtype=np.int64)
        probabilities = np.eye(4, dtype=np.float64)[truth]
        metrics = _action_metrics(rows, np.ones(len(rows), dtype=np.bool_), probabilities, truth_oracle=True)
        self.assertEqual(metrics["final_disposition_accuracy"], 1.0)
        self.assertEqual(metrics["false_ACT_rate"], 0.0)


class OracleIsolationTests(unittest.TestCase):
    def test_relation_sample_is_exact_320_per_family(self) -> None:
        rows = []
        for action in range(24):
            for policy in range(8):
                governing = policy == 0 or (policy == 1 and action < 6)
                for perturbation in ["BASE"] + [f"P{i}" for i in range(7)]:
                    rows.append({
                        "family_id": "F19", "row_id": f"a{action}|p{policy}|{perturbation}",
                        "base_action_id": f"a{action}", "policy_id": f"p{policy}",
                        "perturbation_kind": perturbation, "governing": int(governing),
                    })
        mask = relation_sample_mask(rows)
        self.assertEqual(int(mask.sum()), 320)

    def test_no_unrelated_study_residue(self) -> None:
        for path in ROOT.rglob("*"):
            if path.is_file() and path.name != "SOURCE-MANIFEST.json":
                text = path.read_text(errors="ignore")
                self.assertNotIn("DBRR" + "-002", text)
                self.assertNotIn("cutoff" + "_day", text)


if __name__ == "__main__":
    unittest.main()
