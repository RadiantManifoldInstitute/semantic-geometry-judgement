#!/usr/bin/python3
"""Imperative Path A constructor for the closed K-F1C synthetic world.

This module is source-only in the launch-admission packet. It must not be run
until the later hash-bound AWS materialization order admits its exact bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import unicodedata
from typing import Any, Iterable


LEVELS = ("L1_CONCRETE", "L2_OPERATIONAL", "L3_PURPOSE")
SURFACES = ("CANONICAL", "WORDING_SUBSTITUTION", "CLAUSE_ORDER")
APPLICABLE_SLOTS = {
    "L1_CONCRETE": (0, 1),
    "L2_OPERATIONAL": (0, 3),
    "L3_PURPOSE": (0, 2),
}


def canonical(value: Any) -> bytes:
    def nfc(item: Any) -> Any:
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list):
            return [nfc(v) for v in item]
        if isinstance(item, dict):
            return {nfc(k): nfc(v) for k, v in item.items()}
        return item

    return json.dumps(
        nfc(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sealed_record(fields: dict[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in fields:
        raise ValueError("payload_sha256 supplied by caller")
    payload = canonical(fields) + b"\n"
    result = dict(fields)
    result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical(row) + b"\n")


def cell_bits(cell: str) -> tuple[int, str]:
    return {
        "APPLICABLE_ALIGNED": (1, "ALIGNED"),
        "APPLICABLE_OPPOSED": (1, "OPPOSED"),
        "NOT_APPLICABLE_NONOPPOSED": (0, "ALIGNED"),
        "NOT_APPLICABLE_APPARENTLY_OPPOSED": (0, "OPPOSED"),
    }[cell]


def render(root: str, operator: str, level: str, slot: int, surface: str) -> tuple[str, str, dict[str, Any]]:
    actor = f"{root}-actor-{slot + 1}"
    obj = f"{root}-token-{slot + 1}"
    scope = f"{root}-{level.lower()}"
    action_forms = {
        "CANONICAL": f"{actor} {operator} {obj} within {scope}",
        "WORDING_SUBSTITUTION": f"within {scope} the {obj} is {operator} by {actor}",
        "CLAUSE_ORDER": f"{obj} within {scope}; {actor} {operator} it",
    }
    reference_forms = {
        "CANONICAL": f"within {scope} the required relation is {operator} for {obj}",
        "WORDING_SUBSTITUTION": f"the {obj} is to be {operator} inside {scope}",
        "CLAUSE_ORDER": f"required for {obj}: {operator} within {scope}",
    }
    action_text = action_forms[surface]
    reference_text = reference_forms[surface]
    spans = {
        "action_operator": [action_text.encode().find(operator.encode()), action_text.encode().find(operator.encode()) + len(operator.encode())],
        "reference_operator": [reference_text.encode().find(operator.encode()), reference_text.encode().find(operator.encode()) + len(operator.encode())],
    }
    return action_text, reference_text, spans


def construct(blueprint: dict[str, Any], contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    families = blueprint["families"]
    if len(families) != 20:
        raise ValueError("exactly twenty families required")
    schedule = contract["cell_schedule"]
    out: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (
            "mission_families", "scenarios", "actions", "references", "canonical_pairs",
            "rendered_surfaces", "blinded", "ancestry", "transitions", "canonical_gold",
            "surface_proofs", "sealed_gold", "sealed_join"
        )
    }
    join_salt = hashlib.sha256(b"SYSTEM-K-K-F1C|41381008|JOIN|V0.1\n").hexdigest()
    scenario_ordinal = action_ordinal = reference_ordinal = pair_ordinal = surface_ordinal = 0

    for family in families:
        out["mission_families"].append(family)
        lex = blueprint["split_lexical_operators"][family["split"]]
        for local_scenario in range(1, 4):
            scenario_ordinal += 1
            scenario_id = f"SCN_{scenario_ordinal:04d}"
            actor_ids = [f"ACT_{(scenario_ordinal - 1) * 4 + i + 1:04d}" for i in range(4)]
            object_ids = [f"OBJ_{(scenario_ordinal - 1) * 4 + i + 1:04d}" for i in range(4)]
            pre_atom = f"ATM_{(scenario_ordinal - 1) * 4 + 1:04d}"
            effect_atoms = {level: f"ATM_{(scenario_ordinal - 1) * 4 + i + 2:04d}" for i, level in enumerate(LEVELS)}
            scopes = {
                level: {
                    "scope_id": f"SCP_{(scenario_ordinal - 1) * 3 + i + 1:04d}",
                    "member_actor_ids": [actor_ids[s] for s in APPLICABLE_SLOTS[level]],
                    "member_object_ids": [object_ids[s] for s in APPLICABLE_SLOTS[level]],
                }
                for i, level in enumerate(LEVELS)
            }
            scenario = {
                "scenario_id": scenario_id,
                "family_id": family["family_id"],
                "split": family["split"],
                "local_ordinal": local_scenario,
                "actor_ids": actor_ids,
                "object_ids": object_ids,
                "pre_state_atom_ids": [pre_atom, *effect_atoms.values()],
                "scopes": scopes,
            }
            out["scenarios"].append(scenario)

            refs: dict[str, dict[str, Any]] = {}
            for level_index, level in enumerate(LEVELS):
                reference_ordinal += 1
                ref = {
                    "reference_id": f"REF_{reference_ordinal:04d}",
                    "scenario_id": scenario_id,
                    "family_id": family["family_id"],
                    "level": level,
                    "actor_type_constraint": "INITIATOR",
                    "object_type_constraint": "TOKEN",
                    "scope_id": scopes[level]["scope_id"],
                    "scope_member_actor_ids": scopes[level]["member_actor_ids"],
                    "scope_member_object_ids": scopes[level]["member_object_ids"],
                    "applicability_precondition_atom_ids": [pre_atom],
                    "required_effect_atom_ids": [effect_atoms[level]],
                    "prohibited_effect_atom_ids": [],
                    "parent_reference_id_or_null": None,
                }
                refs[level] = ref
                out["references"].append(ref)
            refs["L1_CONCRETE"]["parent_reference_id_or_null"] = refs["L2_OPERATIONAL"]["reference_id"]
            refs["L2_OPERATIONAL"]["parent_reference_id_or_null"] = refs["L3_PURPOSE"]["reference_id"]

            ancestry_for_level: dict[str, dict[str, Any]] = {}
            for child_level, parent_level, rule in (
                ("L1_CONCRETE", "L2_OPERATIONAL", "P01_L1_TO_L2"),
                ("L2_OPERATIONAL", "L3_PURPOSE", "P02_L2_TO_L3"),
            ):
                record = sealed_record({
                    "child_reference_id": refs[child_level]["reference_id"],
                    "parent_reference_id": refs[parent_level]["reference_id"],
                    "child_level": child_level,
                    "parent_level": parent_level,
                    "projection_rule_id": rule,
                    "child_atom_id": effect_atoms[child_level],
                    "parent_atom_id_or_null": effect_atoms[parent_level],
                    "disposition": "RETAINED_GENERALIZED_SCOPE",
                    "predicate_sign_before": "ASSERT",
                    "predicate_sign_after_or_null": "ASSERT",
                    "scope_mapping": f"{scopes[child_level]['scope_id']}->{scopes[parent_level]['scope_id']}",
                })
                out["ancestry"].append(record)
                ancestry_for_level[child_level] = record
                ancestry_for_level[parent_level] = record

            actions: list[dict[str, Any]] = []
            transition_by_action: dict[str, dict[str, Any]] = {}
            for slot in range(4):
                action_ordinal += 1
                action_id = f"ACN_{action_ordinal:04d}"
                asserted: list[str] = []
                retracted: list[str] = []
                for level in LEVELS:
                    _, polarity = cell_bits(schedule[level][slot])
                    (asserted if polarity == "ALIGNED" else retracted).append(effect_atoms[level])
                action = {
                    "action_id": action_id,
                    "scenario_id": scenario_id,
                    "family_id": family["family_id"],
                    "slot": f"Q{slot}",
                    "actor_id": actor_ids[slot],
                    "actor_type": "INITIATOR",
                    "object_id": object_ids[slot],
                    "object_type": "TOKEN",
                    "precondition_atom_ids": [pre_atom],
                    "pre_state_atom_ids": scenario["pre_state_atom_ids"],
                    "asserted_atom_ids": sorted(asserted),
                    "retracted_atom_ids": sorted(retracted),
                }
                actions.append(action)
                out["actions"].append(action)
                transition = sealed_record({
                    "transition_id": f"TRN_{action_ordinal:04d}",
                    "action_id": action_id,
                    "pre_state_sha256": hashlib.sha256(canonical(action["pre_state_atom_ids"]) + b"\n").hexdigest(),
                    "asserted_atom_ids": action["asserted_atom_ids"],
                    "retracted_atom_ids": action["retracted_atom_ids"],
                    "post_state_sha256": hashlib.sha256(canonical(sorted((set(action["pre_state_atom_ids"]) | set(asserted)) - set(retracted))) + b"\n").hexdigest(),
                    "entailed_required_atom_ids": action["asserted_atom_ids"],
                    "entailed_prohibited_atom_ids": [],
                    "contradicted_required_atom_ids": action["retracted_atom_ids"],
                })
                out["transitions"].append(transition)
                transition_by_action[action_id] = transition

            for level in LEVELS:
                ref = refs[level]
                for slot, action in enumerate(actions):
                    pair_ordinal += 1
                    pair_id = f"PAIR_{pair_ordinal:04d}"
                    actor_match = int(action["actor_type"] == ref["actor_type_constraint"])
                    object_match = int(action["object_type"] == ref["object_type_constraint"])
                    scope_match = int(action["actor_id"] in ref["scope_member_actor_ids"] and action["object_id"] in ref["scope_member_object_ids"])
                    preconditions = int(set(ref["applicability_precondition_atom_ids"]).issubset(action["pre_state_atom_ids"]))
                    applicable = actor_match * object_match * scope_match * preconditions
                    required_entailed = int(set(ref["required_effect_atom_ids"]).issubset(action["asserted_atom_ids"]))
                    prohibited_entailed = 0
                    required_contradicted = int(bool(set(ref["required_effect_atom_ids"]) & set(action["retracted_atom_ids"])))
                    polarity = "OPPOSED" if required_contradicted or prohibited_entailed else "ALIGNED"
                    cell = {
                        (1, "ALIGNED"): "APPLICABLE_ALIGNED",
                        (1, "OPPOSED"): "APPLICABLE_OPPOSED",
                        (0, "ALIGNED"): "NOT_APPLICABLE_NONOPPOSED",
                        (0, "OPPOSED"): "NOT_APPLICABLE_APPARENTLY_OPPOSED",
                    }[(applicable, polarity)]
                    if cell != schedule[level][slot]:
                        raise ValueError(f"schedule mismatch {scenario_id} {level} Q{slot}")
                    canonical_gold = sealed_record({
                        "schema_version": "K_F1C_CANONICAL_GOLD_V0_2",
                        "canonical_pair_id": pair_id,
                        "family_id": family["family_id"],
                        "scenario_id": scenario_id,
                        "action_id": action["action_id"],
                        "reference_id": ref["reference_id"],
                        "level": level,
                        "applicability_bit": applicable,
                        "actor_match_bit": actor_match,
                        "object_match_bit": object_match,
                        "scope_match_bit": scope_match,
                        "preconditions_satisfied_bit": preconditions,
                        "required_effects_entailed_bit": required_entailed,
                        "prohibited_effects_entailed_bit": prohibited_entailed,
                        "required_effects_contradicted_bit": required_contradicted,
                        "relation_polarity": polarity,
                        "gold_cell": cell,
                        "ancestry_proof_payload_sha256": ancestry_for_level[level]["payload_sha256"],
                        "transition_proof_payload_sha256": transition_by_action[action["action_id"]]["payload_sha256"],
                    })
                    out["canonical_gold"].append(canonical_gold)
                    out["canonical_pairs"].append({"canonical_pair_id": pair_id, "action_id": action["action_id"], "reference_id": ref["reference_id"], "level": level})
                    operator = lex["aligned"] if polarity == "ALIGNED" else lex["opposed"]
                    level_pattern = "|".join(cell_bits(schedule[l][slot])[1] for l in LEVELS)
                    for surface_class in SURFACES:
                        surface_ordinal += 1
                        surface_id = f"SRF_{surface_ordinal:04d}"
                        action_text, reference_text, spans = render(family["content_root"], operator, level, slot, surface_class)
                        surface = {"surface_id": surface_id, "canonical_pair_id": pair_id, "surface_class": surface_class, "action_text": action_text, "reference_text": reference_text, "atom_span_map": spans}
                        out["rendered_surfaces"].append(surface)
                        surface_proof = sealed_record({
                            "surface_id": surface_id,
                            "canonical_pair_id": pair_id,
                            "surface_class": surface_class,
                            "canonical_gold_payload_sha256": canonical_gold["payload_sha256"],
                            "rendered_text_sha256": hashlib.sha256((action_text + "\n" + reference_text + "\n").encode()).hexdigest(),
                            "atom_span_map_sha256": hashlib.sha256(canonical(spans) + b"\n").hexdigest(),
                            "semantics_preserved_bit": 1,
                        })
                        out["surface_proofs"].append(surface_proof)
                        sealed_gold = sealed_record({
                            "schema_version": "K_F1C_SEALED_GOLD_V0_2",
                            "canonical_pair_id": pair_id,
                            "surface_id": surface_id,
                            "family_id": family["family_id"],
                            "scenario_id": scenario_id,
                            "action_id": action["action_id"],
                            "reference_id": ref["reference_id"],
                            "level": level,
                            "split": family["split"],
                            "applicability_bit": applicable,
                            "actor_match_bit": actor_match,
                            "object_match_bit": object_match,
                            "scope_match_bit": scope_match,
                            "preconditions_satisfied_bit": preconditions,
                            "required_effects_entailed_bit": required_entailed,
                            "prohibited_effects_entailed_bit": prohibited_entailed,
                            "required_effects_contradicted_bit": required_contradicted,
                            "relation_polarity": polarity,
                            "gold_cell": cell,
                            "level_conflict_pattern": level_pattern,
                            "canonical_gold_payload_sha256": canonical_gold["payload_sha256"],
                            "ancestry_proof_payload_sha256": ancestry_for_level[level]["payload_sha256"],
                            "transition_proof_payload_sha256": transition_by_action[action["action_id"]]["payload_sha256"],
                            "surface_inheritance_proof_payload_sha256": surface_proof["payload_sha256"],
                        })
                        out["sealed_gold"].append(sealed_gold)
                        blind_pair_id = hashlib.sha256(f"{join_salt}|{pair_id}".encode()).hexdigest()
                        blind_surface_id = hashlib.sha256(f"{join_salt}|{surface_id}".encode()).hexdigest()
                        out["blinded"].append({
                            "schema_version": "K_F1C_BLINDED_INPUT_V0_1",
                            "blind_pair_id": blind_pair_id,
                            "blind_family_id": hashlib.sha256(f"{join_salt}|{family['family_id']}".encode()).hexdigest(),
                            "blind_level_id": hashlib.sha256(f"{join_salt}|{level}".encode()).hexdigest(),
                            "blind_surface_id": blind_surface_id,
                            "action_text": action_text,
                            "reference_text": reference_text,
                            "action_text_sha256": hashlib.sha256(action_text.encode()).hexdigest(),
                            "reference_text_sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
                        })
                        out["sealed_join"].append(sealed_record({
                            "schema_version": "K_F1C_SEALED_JOIN_MAP_V0_2",
                            "blind_pair_id": blind_pair_id,
                            "blind_surface_id": blind_surface_id,
                            "canonical_pair_id": pair_id,
                            "surface_id": surface_id,
                            "canonical_gold_payload_sha256": canonical_gold["payload_sha256"],
                            "sealed_gold_payload_sha256": sealed_gold["payload_sha256"],
                            "join_salt_version_sha256": join_salt,
                        }))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--primitive-dir", required=True)
    parser.add_argument("--record-dir", required=True)
    args = parser.parse_args()
    blueprint = json.loads(pathlib.Path(args.blueprint).read_text(encoding="utf-8"))
    contract = json.loads(pathlib.Path(args.contract).read_text(encoding="utf-8"))
    rows = construct(blueprint, contract)
    primitive = pathlib.Path(args.primitive_dir)
    records = pathlib.Path(args.record_dir)
    primitive_names = {
        "mission_families": "mission-families.jsonl",
        "scenarios": "scenarios.jsonl",
        "actions": "action-ir.jsonl",
        "references": "reference-ir.jsonl",
        "canonical_pairs": "canonical-pairs.jsonl",
        "rendered_surfaces": "rendered-surfaces.jsonl",
        "blinded": "blinded-scoring-input.jsonl",
    }
    for name, filename in primitive_names.items():
        write_jsonl(primitive / filename, rows[name])
    record_names = {
        "ancestry": "ancestry-proof.jsonl",
        "transitions": "transition-proof.jsonl",
        "canonical_gold": "canonical-gold.jsonl",
        "surface_proofs": "surface-inheritance-proof.jsonl",
        "sealed_gold": "sealed-gold.jsonl",
        "sealed_join": "sealed-join.jsonl",
    }
    for name, filename in record_names.items():
        write_jsonl(records / filename, rows[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
