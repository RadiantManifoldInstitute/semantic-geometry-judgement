#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable


TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
SIZES = (("SMALL", 8, "CONCRETE_RULE"), ("MEDIUM", 32, "OPERATIONAL_OBJECTIVE"), ("LARGE", 128, "HIGHER_PURPOSE"))
CARDINALITIES = ("ZERO", "ONE", "MULTIPLE_COMPATIBLE", "MULTIPLE_CONFLICTING")
PERTURBATIONS = (
    "BOUNDED_PARAPHRASE", "EXPLICIT_NEGATION", "LEXICAL_DECOY", "WRONG_MISSION_DECOY",
    "CLAUSE_ORDER", "HARMLESS_WORDING", "UNUSUAL_MISSION_SERVING_ACTION",
)


def canonical(value: Any) -> bytes:
    def normalized(item: Any) -> Any:
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return [normalized(child) for child in item]
        if isinstance(item, dict):
            return {str(key): normalized(child) for key, child in item.items()}
        return item
    return (json.dumps(normalized(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    return value | {"payload_sha256": digest(value)}


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical(row))


def policy_rows(family: dict[str, Any], phrases: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for ordinal in range(1, 129):
        abstraction = SIZES[(ordinal - 1) % len(SIZES)][2]
        if ordinal == 1:
            obj = family["object"]
            objective = family["objective"]
        elif ordinal == 2:
            obj = f"secondary {family['object']}"
            objective = f"secondary {family['objective']}"
        else:
            obj = f"ancillary control {ordinal:03d}"
            objective = f"bounded operating objective {ordinal:03d}"
        policy_id = f"{family['family_id']}-P{ordinal:03d}"
        rows.append({
            "schema_version": "K_F1F_POLICY_V0_1", "policy_id": policy_id,
            "family_id": family["family_id"], "split": family["split"], "ordinal": ordinal,
            "policy_template_id": f"{family['family_id']}-POLICY-{abstraction}",
            "abstraction": abstraction, "object": obj, "objective": objective,
            "text": f"Within {family['domain']}, this {phrases[abstraction]} requires the {family['actor']} to {family['aligned']} the {obj} so that {objective} is protected.",
            "aligned_reference_text": f"The {family['actor']} will {family['aligned']} the {obj} to protect {objective} in {family['domain']}.",
            "opposed_reference_text": f"The {family['actor']} will {family['opposed']} the {obj} despite {objective} in {family['domain']}.",
            "route_terms": sorted(set(TOKEN.findall(f"{family['domain']} {obj} {objective}".lower()))),
        })
    return rows


def relation_pattern(size_index: int, cardinality: str) -> list[str]:
    if cardinality == "ZERO":
        return []
    if cardinality == "ONE":
        return [["ALIGNED"], ["OPPOSED"], ["AMBIGUOUS"]][size_index]
    if cardinality == "MULTIPLE_COMPATIBLE":
        label = ("ALIGNED", "OPPOSED", "AMBIGUOUS")[size_index]
        return [label, label]
    if cardinality == "MULTIPLE_CONFLICTING":
        return ["ALIGNED", "OPPOSED"] if size_index < 2 else ["ALIGNED", "AMBIGUOUS"]
    raise ValueError(cardinality)


def disposition(relations: list[str], unconstrained: bool) -> str:
    if not relations:
        return "ACT" if unconstrained else "ESCALATE"
    if "OPPOSED" in relations:
        return "HOLD"
    if "AMBIGUOUS" in relations:
        return "ESCALATE"
    return "ACT"


def action_text(family: dict[str, Any], policies: list[dict[str, Any]], relations: list[str], unconstrained: bool) -> str:
    if not relations:
        status = "outside the declared control set" if unconstrained else "near an unresolved operating boundary"
        return f"The {family['actor']} plans a routine action {status} in {family['domain']} involving {family['adjacent']}."
    clauses = []
    verbs = {"ALIGNED": family["aligned"], "OPPOSED": family["opposed"], "AMBIGUOUS": family["ambiguous"]}
    for policy, relation in zip(policies, relations):
        clauses.append(f"{verbs[relation]} the {policy['object']} with respect to {policy['objective']}")
    return f"In {family['domain']}, the {family['actor']} plans to " + " and ".join(clauses) + "."


def perturb(text: str, kind: str, family: dict[str, Any]) -> str:
    if kind == "BOUNDED_PARAPHRASE":
        return "Restated without changing purpose: " + text
    if kind == "EXPLICIT_NEGATION":
        return "Invert the declared operating choice: do not execute the following as stated. " + text
    if kind == "LEXICAL_DECOY":
        return text + f" Administrative notes repeat {family['object']} but concern filing only."
    if kind == "WRONG_MISSION_DECOY":
        return text + " A historical costume exhibition is separately updating its caption inventory."
    if kind == "CLAUSE_ORDER":
        return "After considering the protective purpose, execute this plan: " + text
    if kind == "HARMLESS_WORDING":
        return "Please record courteously that " + text[0].lower() + text[1:]
    if kind == "UNUSUAL_MISSION_SERVING_ACTION":
        return "Using an uncommon but mission-serving procedure, " + text[0].lower() + text[1:]
    raise ValueError(kind)


def flipped(relations: list[str]) -> list[str]:
    table = {"ALIGNED": "OPPOSED", "OPPOSED": "ALIGNED", "AMBIGUOUS": "AMBIGUOUS"}
    return [table[value] for value in relations]


def construct(blueprint: dict[str, Any], contract: dict[str, Any]) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    if len(blueprint["families"]) != 30 or len({row["family_id"] for row in blueprint["families"]}) != 30:
        raise ValueError("EXACT_THIRTY_UNIQUE_FAMILIES_REQUIRED")
    salt = hashlib.sha256(b"SYSTEM-K-K-F1F|80412026|JOIN|V0.1\n").hexdigest()
    output: dict[str, list[dict[str, Any]] | dict[str, Any]] = {
        key: [] for key in ("policies", "actions", "pairs", "blinded", "policy_oracle", "action_oracle", "proofs", "calibration_gold", "validation_gold", "sealed_gold", "calibration_join", "validation_join", "sealed_join")
    }
    per_family_base: dict[str, Counter[str]] = {}
    for family in blueprint["families"]:
        policies = policy_rows(family, blueprint["abstraction_phrases"])
        output["policies"].extend(policies)  # type: ignore[union-attr]
        for policy in policies:
            output["policy_oracle"].append(sealed({
                "schema_version": "K_F1F_EXECUTABLE_POLICY_ORACLE_V0_1", "policy_id": policy["policy_id"],
                "family_id": family["family_id"], "predicate": "GENERATED_GOVERNING_MEMBERSHIP_TABLE_LOOKUP",
                "output_domain": ["ALIGNED", "OPPOSED", "AMBIGUOUS", "NON_GOVERNING"],
            }))  # type: ignore[union-attr]
        base_counts: Counter[str] = Counter()
        for size_index, (size_name, policy_count, abstraction) in enumerate(SIZES):
            active_policies = policies[:policy_count]
            for cardinality in CARDINALITIES:
                relations = relation_pattern(size_index, cardinality)
                governing = active_policies[:len(relations)]
                for variant in range(2):
                    base_action_id = f"{family['family_id']}-A-{size_name}-{cardinality}-V{variant + 1}"
                    unconstrained = cardinality == "ZERO" and variant == 0
                    base_text = action_text(family, governing, relations, unconstrained)
                    versions = [("BASE", base_text, relations, unconstrained)]
                    for kind in PERTURBATIONS:
                        changed = kind == "EXPLICIT_NEGATION"
                        effective_relations = flipped(relations) if changed else relations
                        effective_unconstrained = (not unconstrained) if changed and cardinality == "ZERO" else unconstrained
                        changed_text = (
                            "Explicitly negated form: " + action_text(family, governing, effective_relations, effective_unconstrained)
                            if changed else perturb(base_text, kind, family)
                        )
                        versions.append((kind, changed_text, effective_relations, effective_unconstrained))
                    for kind, text, effective_relations, effective_unconstrained in versions:
                        action_id = base_action_id if kind == "BASE" else f"{base_action_id}-{kind}"
                        action_disposition = disposition(effective_relations, effective_unconstrained)
                        action = {
                            "schema_version": "K_F1F_ACTION_V0_1", "action_id": action_id, "base_action_id": base_action_id,
                            "family_id": family["family_id"], "split": family["split"], "text": text,
                            "action_template_id": f"{family['family_id']}-{size_name}-{cardinality}-V{variant + 1}",
                            "mission_context": family["domain"], "policy_set_size": size_name, "policy_count": policy_count,
                            "cardinality": cardinality, "abstraction": abstraction, "variant": variant + 1,
                            "perturbation_kind": kind, "is_perturbed": kind != "BASE", "unconstrained": effective_unconstrained,
                            "governing_policy_ids": [row["policy_id"] for row in governing],
                            "relations_by_policy": {row["policy_id"]: relation for row, relation in zip(governing, effective_relations)},
                            "expected_disposition": action_disposition,
                        }
                        output["actions"].append(action)  # type: ignore[union-attr]
                        output["action_oracle"].append(sealed({key: value for key, value in action.items() if key != "text"}))  # type: ignore[union-attr]
                        blind_action_id = hashlib.sha256(f"{salt}|{action_id}".encode()).hexdigest()
                        for policy in active_policies:
                            policy_id = policy["policy_id"]
                            relation = action["relations_by_policy"].get(policy_id, "NON_GOVERNING")
                            governing_bit = int(relation != "NON_GOVERNING")
                            pair_id = f"{action_id}|{policy_id}"
                            pair = {
                                "schema_version": "K_F1F_PAIR_IR_V0_1", "row_id": pair_id, "action_id": action_id,
                                "base_action_id": base_action_id, "policy_id": policy_id, "family_id": family["family_id"],
                                "split": family["split"], "policy_set_size": size_name, "policy_count": policy_count,
                                "cardinality": cardinality, "abstraction": abstraction, "variant": variant + 1,
                                "perturbation_kind": kind, "is_perturbed": kind != "BASE", "governing": governing_bit,
                                "relation": relation, "blocking": int(relation == "OPPOSED"), "expected_disposition": action_disposition,
                            }
                            output["pairs"].append(pair)  # type: ignore[union-attr]
                            blind_pair_id = hashlib.sha256(f"{salt}|{pair_id}".encode()).hexdigest()
                            blind_policy_id = hashlib.sha256(f"{salt}|{policy_id}".encode()).hexdigest()
                            blind = {
                                "schema_version": "K_F1F_BLINDED_INPUT_V0_1", "blind_pair_id": blind_pair_id,
                                "blind_action_id": blind_action_id, "blind_policy_id": blind_policy_id,
                                "action_text": text, "policy_text": policy["text"],
                                "aligned_reference_text": policy["aligned_reference_text"], "opposed_reference_text": policy["opposed_reference_text"],
                                "policy_route_terms": policy["route_terms"],
                            }
                            output["blinded"].append(blind)  # type: ignore[union-attr]
                            proof = sealed({
                                "schema_version": "K_F1F_PERTURBATION_PROOF_V0_1", "row_id": pair_id,
                                "perturbation_id": action_id, "base_id": base_action_id, "deterministic_seed": 17022,
                                "base_action_id": base_action_id, "perturbation_kind": kind,
                                "semantics_preserved": kind != "EXPLICIT_NEGATION",
                                "action_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "pair_ir_sha256": digest(pair),
                            })
                            output["proofs"].append(proof)  # type: ignore[union-attr]
                            gold = sealed(pair | {"schema_version": "K_F1F_PAIR_GOLD_V0_1", "perturbation_proof_payload_sha256": proof["payload_sha256"]})
                            join = sealed({
                                "schema_version": "K_F1F_JOIN_MAP_V0_1", "blind_pair_id": blind_pair_id,
                                "blind_action_id": blind_action_id, "blind_policy_id": blind_policy_id,
                                "row_id": pair_id, "gold_payload_sha256": gold["payload_sha256"], "join_salt_sha256": salt,
                            })
                            target = family["split"].lower()
                            output[f"{target}_gold"].append(gold)  # type: ignore[index,union-attr]
                            output[f"{target}_join"].append(join)  # type: ignore[index,union-attr]
                            if kind == "BASE":
                                base_counts[f"pairs_{size_name}"] += 1
                                base_counts[f"cardinality_{cardinality}_actions"] += int(policy is active_policies[0])
                                base_counts[f"relation_{relation}"] += 1
                                base_counts[f"governing_cardinality_{cardinality}"] += governing_bit
                        if kind == "BASE":
                            base_counts[f"actions_{size_name}"] += 1
                            base_counts[f"disposition_{action_disposition}"] += 1
        per_family_base[family["family_id"]] = base_counts

    counts = contract["counts"]
    # Canonical record order is family, then action, then policy.  This occurs
    # before any write so both reproductions consume the identical primitive IR.
    output["policies"].sort(key=lambda row: (row["family_id"], row["policy_id"]))  # type: ignore[union-attr]
    output["actions"].sort(key=lambda row: (row["family_id"], row["action_id"]))  # type: ignore[union-attr]
    output["pairs"].sort(key=lambda row: (row["family_id"], row["action_id"], row["policy_id"]))  # type: ignore[union-attr]
    # The independent Ruby reproduction consumes the canonical primitive bytes,
    # whereas this path originally derived proofs from pre-serialization Python
    # objects.  Canonical-roundtrip the shared primitive IR, then reproduce every
    # pair proof, gold row, and join from that exact IR.  This keeps the two
    # implementations independent while making their byte-agreement boundary
    # explicit rather than relying on incidental in-memory string state.
    for name in ("policies", "actions", "pairs", "blinded"):
        output[name] = [json.loads(canonical(row)) for row in output[name]]  # type: ignore[union-attr]
    blind_by_id = {row["blind_pair_id"]: row for row in output["blinded"]}  # type: ignore[union-attr]
    output["blinded"] = [
        blind_by_id[hashlib.sha256(f"{salt}|{row['row_id']}".encode()).hexdigest()]
        for row in output["pairs"]  # type: ignore[union-attr]
    ]
    action_by_id = {row["action_id"]: row for row in output["actions"]}  # type: ignore[union-attr]
    output["proofs"] = []
    for name in ("calibration_gold", "validation_gold", "sealed_gold", "calibration_join", "validation_join", "sealed_join"):
        output[name] = []
    for pair in output["pairs"]:  # type: ignore[union-attr]
        action = action_by_id[pair["action_id"]]
        normalized_text = unicodedata.normalize("NFC", action["text"])
        proof = sealed({
            "schema_version": "K_F1F_PERTURBATION_PROOF_V0_1", "row_id": pair["row_id"],
            "perturbation_id": action["action_id"], "base_id": pair["base_action_id"], "deterministic_seed": 17022,
            "base_action_id": pair["base_action_id"], "perturbation_kind": pair["perturbation_kind"],
            "semantics_preserved": pair["perturbation_kind"] != "EXPLICIT_NEGATION",
            "action_text_sha256": hashlib.sha256(normalized_text.encode()).hexdigest(),
            "pair_ir_sha256": digest(pair),
        })
        output["proofs"].append(proof)  # type: ignore[union-attr]
        pair_gold = sealed(pair | {"schema_version": "K_F1F_PAIR_GOLD_V0_1", "perturbation_proof_payload_sha256": proof["payload_sha256"]})
        blind_pair_id = hashlib.sha256(f"{salt}|{pair['row_id']}".encode()).hexdigest()
        join = sealed({
            "schema_version": "K_F1F_JOIN_MAP_V0_1", "blind_pair_id": blind_pair_id,
            "blind_action_id": hashlib.sha256(f"{salt}|{pair['action_id']}".encode()).hexdigest(),
            "blind_policy_id": hashlib.sha256(f"{salt}|{pair['policy_id']}".encode()).hexdigest(),
            "row_id": pair["row_id"], "gold_payload_sha256": pair_gold["payload_sha256"], "join_salt_sha256": salt,
        })
        target = pair["split"].lower()
        output[f"{target}_gold"].append(pair_gold)  # type: ignore[union-attr]
        output[f"{target}_join"].append(join)  # type: ignore[union-attr]
    output["policy_oracle"].sort(key=lambda row: (row["family_id"], row["policy_id"]))  # type: ignore[union-attr]
    output["action_oracle"].sort(key=lambda row: (row["family_id"], row["action_id"]))  # type: ignore[union-attr]
    output["proofs"].sort(key=lambda row: row["row_id"])  # type: ignore[union-attr]
    for name in ("calibration_gold", "validation_gold", "sealed_gold"):
        output[name].sort(key=lambda row: (row["family_id"], row["action_id"], row["policy_id"]))  # type: ignore[union-attr]
    for name in ("calibration_join", "validation_join", "sealed_join"):
        output[name].sort(key=lambda row: row["row_id"])  # type: ignore[union-attr]
    observed = {
        "actions": len(output["actions"]), "pairs": len(output["pairs"]),
        "calibration_actions": sum(row["split"] == "CALIBRATION" for row in output["actions"]),
        "validation_actions": sum(row["split"] == "VALIDATION" for row in output["actions"]),
        "sealed_actions": sum(row["split"] == "SEALED" for row in output["actions"]),
        "calibration_pairs": len(output["calibration_gold"]), "validation_pairs": len(output["validation_gold"]), "sealed_pairs": len(output["sealed_gold"]),
        "base_actions": sum(row["perturbation_kind"] == "BASE" for row in output["actions"]),
        "base_pairs": sum(row["perturbation_kind"] == "BASE" for row in output["pairs"]),
    }
    if observed != counts:
        raise ValueError(f"CENSUS_MISMATCH:{observed}")
    expected = contract["base_per_family"]
    for family_id, values in per_family_base.items():
        required = {
            "actions_SMALL": 8, "actions_MEDIUM": 8, "actions_LARGE": 8,
            "relation_ALIGNED": expected["governing_relation_pairs"]["ALIGNED"],
            "relation_OPPOSED": expected["governing_relation_pairs"]["OPPOSED"],
            "relation_AMBIGUOUS": expected["governing_relation_pairs"]["AMBIGUOUS"],
            "relation_NON_GOVERNING": expected["non_governing_pairs_total"],
            "disposition_ACT": expected["action_dispositions"]["ACT"],
            "disposition_HOLD": expected["action_dispositions"]["HOLD"],
            "disposition_ESCALATE": expected["action_dispositions"]["ESCALATE"],
            "cardinality_ZERO_actions": expected["cardinality_actions"]["ZERO"],
            "cardinality_ONE_actions": expected["cardinality_actions"]["ONE"],
            "cardinality_MULTIPLE_COMPATIBLE_actions": expected["cardinality_actions"]["MULTIPLE_COMPATIBLE"],
            "cardinality_MULTIPLE_CONFLICTING_actions": expected["cardinality_actions"]["MULTIPLE_CONFLICTING"],
            "governing_cardinality_ZERO": expected["governing_pairs_by_cardinality"]["ZERO"],
            "governing_cardinality_ONE": expected["governing_pairs_by_cardinality"]["ONE"],
            "governing_cardinality_MULTIPLE_COMPATIBLE": expected["governing_pairs_by_cardinality"]["MULTIPLE_COMPATIBLE"],
            "governing_cardinality_MULTIPLE_CONFLICTING": expected["governing_pairs_by_cardinality"]["MULTIPLE_CONFLICTING"],
        }
        if any(values[key] != count for key, count in required.items()):
            raise ValueError(f"FAMILY_DENOMINATOR_MISMATCH:{family_id}:{dict(values)}")
    output["denominators"] = {
        "schema_version": "K_F1F_DENOMINATOR_MANIFEST_V0_1", "counts": observed,
        "per_family_base": {family_id: dict(sorted(values.items())) for family_id, values in sorted(per_family_base.items())},
        "base_expected": expected, "perturbation_kinds": list(PERTURBATIONS),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", required=True); parser.add_argument("--contract", required=True)
    parser.add_argument("--primitive-dir", required=True); parser.add_argument("--record-dir", required=True)
    args = parser.parse_args()
    blueprint = json.loads(pathlib.Path(args.blueprint).read_text()); contract = json.loads(pathlib.Path(args.contract).read_text())
    rows = construct(blueprint, contract); primitive = pathlib.Path(args.primitive_dir); records = pathlib.Path(args.record_dir)
    for name, filename in {"policies":"policy-catalog.jsonl", "actions":"action-cases.jsonl", "pairs":"pair-ir.jsonl", "blinded":"blinded-scoring-input.jsonl"}.items():
        write_jsonl(primitive / filename, rows[name])  # type: ignore[arg-type]
    mapping = {
        "policy_oracle":"policy-oracle.jsonl", "action_oracle":"action-oracle.jsonl", "proofs":"perturbation-proofs.jsonl",
        "calibration_gold":"calibration-gold.jsonl", "validation_gold":"validation-gold.jsonl", "sealed_gold":"sealed-gold.jsonl",
        "calibration_join":"calibration-join.jsonl", "validation_join":"validation-join.jsonl", "sealed_join":"sealed-join.jsonl",
    }
    for name, filename in mapping.items():
        write_jsonl(records / filename, rows[name])  # type: ignore[arg-type]
    (records / "denominator-manifest.json").write_bytes(canonical(rows["denominators"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
